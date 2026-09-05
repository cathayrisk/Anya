# -*- coding: utf-8 -*-
"""每回合的檢索憑證帳本（純函式，無 streamlit 依賴）。

## 為什麼需要
一小時測試 T5：問「最近台灣有地震嗎」→ 正確升級 General、**CWA 工具 0 呼叫**，
回答卻宣稱查了氣象署並給出半年前的地震資料。要在系統層擋住這種事，得先能回答
「這回合到底檢索了什麼、成功了沒、什麼時候」——而現況答不出來：

    gm_ds_web_search_log  = {run_id, query, sources[:8]}
    gm_ds_doc_search_log  = {run_id, query, k, hits[:6]}
    CWA 三支                完全沒有 log；例外只轉成字串回給模型，系統這側不留痕跡

沒有 timestamp、沒有 status、沒有 scope，所以**現在不可能寫出正確的 banner**
（「已於 09:05 查詢 CWA 地震資料」vs「查詢失敗」vs「查了但沒資料」根本分不出來）。

## 三個欄位（採納 OAI deep_think 的最小可用版）
- `scope`：受控 enum。**覆蓋是逐 scope 判斷的**——`web_search` 成功不能宣稱地震已查證。
- `status`：`ok` / `empty` / `error`。**`empty` 必須與 `error` 分開**：
  「官方回了空集合」和「API 掛了」對使用者是完全不同的兩件事，措辭不能一樣。
- `completed_at`：UTC ISO。banner 要寫「已於 X 查詢」就得有它。

`authority`（official / internal / open_web）由 scope 推導，不佔第四欄。

## 這個模組不做的事
不產生 banner 文字、不決定要不要 fail-closed——那是呼叫端的政策。
這裡只負責「如實記錄發生過什麼」，保持純函式好測。
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Iterable

# ── scope：受控詞彙 ──────────────────────────────────────────────────────────
SCOPE_EARTHQUAKE = "earthquake_latest"
SCOPE_TYPHOON = "typhoon_active"
SCOPE_WEATHER = "weather_current"
SCOPE_WEB = "web_unclassified"          # 網路搜尋只能證明「搜尋過」，不等於「已查證」
SCOPE_DOC = "doc_unclassified"          # 內部文件檢索，**不是**即時網路查證
SCOPE_PAGE = "page_fetch"

ALL_SCOPES = frozenset({SCOPE_EARTHQUAKE, SCOPE_TYPHOON, SCOPE_WEATHER,
                        SCOPE_WEB, SCOPE_DOC, SCOPE_PAGE})

STATUS_OK = "ok"
STATUS_EMPTY = "empty"
STATUS_ERROR = "error"
ALL_STATUS = frozenset({STATUS_OK, STATUS_EMPTY, STATUS_ERROR})

# 權威等級由 scope 推導——CWA 是官方 API，網路搜尋不是
AUTHORITY = {
    SCOPE_EARTHQUAKE: "official",
    SCOPE_TYPHOON: "official",
    SCOPE_WEATHER: "official",
    SCOPE_DOC: "internal",
    SCOPE_WEB: "open_web",
    SCOPE_PAGE: "open_web",
}


def _now_iso(now: _dt.datetime | None = None) -> str:
    now = now or _dt.datetime.now(_dt.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=_dt.timezone.utc)
    return now.astimezone(_dt.timezone.utc).isoformat(timespec="seconds")


def make_event(*, tool: str, scope: str, status: str,
               run_id: str | None = None, detail: Any = None,
               now: _dt.datetime | None = None) -> dict:
    """建立一筆憑證事件。scope／status 不在受控詞彙裡就直接報錯——
    這種東西一旦允許自由字串，半年後就會有五種寫法，覆蓋判斷跟著失效。"""
    if scope not in ALL_SCOPES:
        raise ValueError(f"未知 scope：{scope}")
    if status not in ALL_STATUS:
        raise ValueError(f"未知 status：{status}")
    return {"tool": tool, "scope": scope, "status": status,
            "authority": AUTHORITY[scope], "completed_at": _now_iso(now),
            "run_id": run_id, "detail": detail}


def coverage(events: Iterable[dict], scope: str) -> dict:
    """這個 scope 這回合被覆蓋了嗎？回傳最能代表現況的那一筆。

    優先序 ok > empty > error：同一 scope 若重試過並成功，就以成功為準。
    完全沒有事件 → `{"covered": False, "status": None}`，呼叫端據此決定
    是不是要 fail closed 或標示「未查證」。
    """
    best = None
    rank = {STATUS_OK: 3, STATUS_EMPTY: 2, STATUS_ERROR: 1}
    for e in events or []:
        if e.get("scope") != scope:
            continue
        if best is None or rank.get(e.get("status"), 0) > rank.get(best.get("status"), 0):
            best = e
    if best is None:
        return {"covered": False, "status": None, "event": None}
    return {"covered": best["status"] == STATUS_OK, "status": best["status"], "event": best}


def summarize(events: Iterable[dict]) -> dict[str, str]:
    """{scope: status}，給 `?dev=1` 與日後的 banner 用。"""
    out: dict[str, str] = {}
    rank = {STATUS_OK: 3, STATUS_EMPTY: 2, STATUS_ERROR: 1}
    for e in events or []:
        sc, st = e.get("scope"), e.get("status")
        if sc in ALL_SCOPES and rank.get(st, 0) > rank.get(out.get(sc), 0):
            out[sc] = st
    return out
