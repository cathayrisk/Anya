# -*- coding: utf-8 -*-
"""LLM attempt 級 telemetry：每一次模型呼叫一筆記錄，零行為改動，只觀測不干預。

為什麼需要（2026-09-03 討論結論 §二.1）：
- 同一個「卡住 ~340s」現象有兩個候選機制（SDK 在 headers 前重試逾時 vs app 階梯在最後一格反覆重打），
  沒有 attempt 級時間線分不出來。
- 「四模型備援鏈同時耗盡」到底是四個獨立配額還是共用 project 配額，只有 429 的 QuotaFailure
  維度（quotaId / quotaMetric / quotaDimensions）能回答；錯誤訊息文字回答不了。
- 工具步驟行在回合結束後從 DOM 移除，事後無法稽核——需要一份不依賴 UI 的紀錄。

輸出去哪：
1. stdout 一行 JSON（`TELEMETRY {...}` 前綴）→ Streamlit Cloud「Manage app → Logs」直接可看。
2. 呼叫端自己保存的 list（本回合 `rt["telemetry"]`）→ `?dev=1` 的 expander 可看，也方便 Playwright 驗證。

版本標記：每筆帶 langchain-google-genai / google-genai 實際安裝版本與 Home.py 內容 hash，
避免「行號證據」在升級後失效而不自知。

這個模組刻意不 import streamlit：純函式，tests/test_telemetry.py 用假例外與假 iterator 離線測。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import uuid
from typing import Any, Iterable, Iterator

# ── 版本標記（import 時算一次；缺套件不擋頁）───────────────────────────────────

def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def _file_md5(path: str | None) -> str | None:
    if not path:
        return None
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:10]
    except Exception:
        return None


def versions(app_file: str | None = None) -> dict[str, Any]:
    """一次性版本快照。app_file 給 Home.py 路徑 → 附內容 hash（部署端與本機是否同一份）。"""
    return {
        "langchain_google_genai": _pkg_version("langchain-google-genai"),
        "google_genai": _pkg_version("google-genai"),
        "app_md5": _file_md5(app_file),
        "python": ".".join(map(str, sys.version_info[:3])),
    }


# ── 429 結構化解析：沿 __cause__ 鏈找帶 .details 的 APIError ─────────────────────

def _find_api_error(exc: BaseException | None, max_depth: int = 6) -> BaseException | None:
    """langchain-google-genai 4.4.0 把 ClientError 重包成 GoogleRateLimitError(msg) from e，
    新例外只剩訊息字串；原始 google-genai APIError（有 .code/.status/.details）在 __cause__。
    往下最多走 6 層，找到第一個有 dict 型 .details 的就回傳。"""
    cur = exc
    for _ in range(max_depth):
        if cur is None:
            return None
        if isinstance(getattr(cur, "details", None), dict):
            return cur
        cur = cur.__cause__ or cur.__context__
    return None


def parse_quota_failure(exc: BaseException | None) -> dict[str, Any]:
    """從 429 例外抽 google.rpc.QuotaFailure 與 RetryInfo。拿不到就回空 dict，不拋。

    Google 錯誤 JSON 形狀：
      {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "...",
                 "details": [
                   {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [{"quotaMetric": "...", "quotaId": "...", "quotaDimensions": {...}}]},
                   {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "26s"}]}}
    quotaDimensions 裡有沒有 "model" 鍵，就是「per-model 還是 per-project 配額」的答案。"""
    api = _find_api_error(exc)
    if api is None:
        return {}
    body = api.details if isinstance(api.details, dict) else {}
    err = body.get("error", body) if isinstance(body, dict) else {}
    out: dict[str, Any] = {}
    if isinstance(err, dict):
        if "code" in err:
            out["http_status"] = err.get("code")
        if "status" in err:
            out["status"] = err.get("status")
        for d in err.get("details") or []:
            if not isinstance(d, dict):
                continue
            t = str(d.get("@type", ""))
            if t.endswith("QuotaFailure"):
                out["quota_violations"] = [
                    {k: v.get(k) for k in ("quotaMetric", "quotaId", "quotaDimensions", "quotaValue") if k in v}
                    for v in (d.get("violations") or []) if isinstance(v, dict)
                ]
            elif t.endswith("RetryInfo"):
                out["retry_delay"] = d.get("retryDelay")
    # 沒有 details 也至少把 code/status 帶回來
    out.setdefault("http_status", getattr(api, "code", None))
    out.setdefault("status", getattr(api, "status", None))
    return out


# ── attempt 記錄 ────────────────────────────────────────────────────────────────

def new_attempt(*, turn_id: str, purpose: str, model: str, tier: int, attempt_n: int,
                mode: str | None = None) -> dict[str, Any]:
    """一次模型呼叫開始時建立。時間全用 time.time()（epoch 秒），事後用差值。"""
    return {
        "turn_id": turn_id, "purpose": purpose, "mode": mode, "model": model, "tier": tier,
        "attempt_n": attempt_n, "t_start": time.time(),
        "t_first_chunk": None, "t_last_chunk": None, "n_chunks": 0,
        "outcome": None, "exc_type": None, "exc_msg": None,
        "finish_reason": None, "http_status": None,
    }


def timed_stream(it: Iterable[Any], rec: dict[str, Any]) -> Iterator[Any]:
    """包住 llm.stream(...) 的 iterator：記首 chunk、末 chunk 時間與 chunk 數。不改任何 chunk。
    注意：app 層看不到「headers 到達」那一刻（SDK 拿到 headers 才把 iterator 交出來），
    所以 t_first_chunk 同時代表「連線＋headers＋第一段 body」。"""
    for c in it:
        now = time.time()
        if rec.get("t_first_chunk") is None:
            rec["t_first_chunk"] = now
        rec["t_last_chunk"] = now
        rec["n_chunks"] = rec.get("n_chunks", 0) + 1
        yield c


def finish_ok(rec: dict[str, Any], result: Any = None) -> dict[str, Any]:
    """成功結束：補 finish_reason（langchain 聚合 chunk 的 response_metadata）與耗時。"""
    rec["outcome"] = "ok"
    meta = getattr(result, "response_metadata", None)
    if isinstance(meta, dict):
        rec["finish_reason"] = meta.get("finish_reason")
    return _close(rec)


def finish_exc(rec: dict[str, Any], exc: BaseException, *, is_quota: bool = False,
               is_stuck: bool = False, retriable: bool = False) -> dict[str, Any]:
    """例外結束：型別、訊息（截 300 字）、app 的分類旗標、以及 429 的結構化維度。"""
    rec["outcome"] = "exc"
    rec["exc_type"] = type(exc).__name__
    rec["exc_msg"] = str(exc)[:300]
    rec["is_quota"] = bool(is_quota)
    rec["is_stuck"] = bool(is_stuck)
    rec["retriable"] = bool(retriable)
    rec.update({k: v for k, v in parse_quota_failure(exc).items() if v is not None})
    return _close(rec)


def _close(rec: dict[str, Any]) -> dict[str, Any]:
    t_end = time.time()
    rec["t_end"] = t_end
    rec["elapsed_s"] = round(t_end - rec["t_start"], 3)
    if rec.get("t_first_chunk"):
        rec["ttfb_s"] = round(rec["t_first_chunk"] - rec["t_start"], 3)  # 首 chunk 延遲
    return rec


# ── 輸出 ────────────────────────────────────────────────────────────────────────

_PREFIX = "TELEMETRY "


def emit(rec: dict[str, Any], sink: list | None = None, *, stream=None) -> str:
    """一行 JSON 到 stdout（Cloud logs）＋ 可選的 list sink（回合內檢視）。回傳那一行（測試用）。"""
    line = _PREFIX + json.dumps(rec, ensure_ascii=False, default=str)
    try:
        print(line, file=stream or sys.stdout, flush=True)
    except Exception:
        pass  # 日誌永遠不能弄掛主流程
    if sink is not None:
        sink.append(rec)
    return line


def parse_line(line: str) -> dict[str, Any] | None:
    """讀回 stdout 那一行（給離線分析 / 測試）。非 telemetry 行回 None。"""
    if not line.startswith(_PREFIX):
        return None
    try:
        return json.loads(line[len(_PREFIX):])
    except Exception:
        return None


def new_turn_id() -> str:
    return uuid.uuid4().hex[:8]
