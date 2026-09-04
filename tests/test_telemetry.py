# -*- coding: utf-8 -*-
"""utils/telemetry.py 離線測試：假例外、假 iterator、假 stdout。不 import streamlit、不打 API。

跑法（專案根目錄）：python -m pytest tests/test_telemetry.py -v
"""
from __future__ import annotations

import io
import json
import pathlib
import sys
import time

import pytest

ROOT = next(p for p in [pathlib.Path(__file__).resolve().parent, *pathlib.Path(__file__).resolve().parents]
            if (p / "Home.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import telemetry as T  # noqa: E402


# ── 假物件：模仿 google-genai APIError 與 langchain 4.4.0 的重包 ──────────────────

class FakeAPIError(Exception):
    """有 .code/.status/.details 的底層例外（google.genai.errors.APIError 的形狀）。"""
    def __init__(self, code, status, details):
        self.code, self.status, self.details = code, status, details
        super().__init__(f"{code} {status}. {details}")


class GoogleRateLimitError(Exception):
    """langchain 4.4.0 重包後的形狀：只有訊息，沒有 code/details。"""


QUOTA_429 = {
    "error": {
        "code": 429, "status": "RESOURCE_EXHAUSTED",
        "message": "You exceeded your current quota.",
        "details": [
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
             "violations": [{"quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                             "quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
                             "quotaDimensions": {"model": "gemma-4-31b-it", "location": "global"},
                             "quotaValue": "15"}]},
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "26s"},
        ],
    }
}


def _rewrapped_429():
    inner = FakeAPIError(429, "RESOURCE_EXHAUSTED", QUOTA_429)
    try:
        raise GoogleRateLimitError("Error calling model 'gemma-4-31b-it' (RESOURCE_EXHAUSTED): 429 ...") from inner
    except GoogleRateLimitError as e:
        return e


# ── parse_quota_failure ───────────────────────────────────────────────────────

def test_parse_quota_walks_cause_chain_and_extracts_dimensions():
    out = T.parse_quota_failure(_rewrapped_429())
    assert out["http_status"] == 429 and out["status"] == "RESOURCE_EXHAUSTED"
    assert out["retry_delay"] == "26s"
    v = out["quota_violations"][0]
    assert v["quotaId"] == "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
    # 這個鍵就是「per-model 還是 per-project 配額」的答案
    assert v["quotaDimensions"]["model"] == "gemma-4-31b-it"


def test_parse_quota_direct_api_error_no_wrap():
    out = T.parse_quota_failure(FakeAPIError(429, "RESOURCE_EXHAUSTED", QUOTA_429))
    assert out["quota_violations"][0]["quotaValue"] == "15"


def test_parse_quota_on_plain_exception_is_empty_not_crash():
    assert T.parse_quota_failure(RuntimeError("boom")) == {}
    assert T.parse_quota_failure(None) == {}


def test_parse_quota_details_without_error_wrapper():
    e = FakeAPIError(503, "UNAVAILABLE", {"code": 503, "status": "UNAVAILABLE", "message": "x"})
    out = T.parse_quota_failure(e)
    assert out["http_status"] == 503 and "quota_violations" not in out


# ── attempt 生命週期 ──────────────────────────────────────────────────────────

def test_timed_stream_records_first_last_and_count_without_altering_chunks():
    rec = T.new_attempt(turn_id="t1", purpose="general", model="m", tier=0, attempt_n=0)
    src = ["a", "b", "c"]
    out = list(T.timed_stream(iter(src), rec))
    assert out == src
    assert rec["n_chunks"] == 3
    assert rec["t_first_chunk"] is not None and rec["t_last_chunk"] >= rec["t_first_chunk"]


def test_timed_stream_empty_iterator_leaves_no_chunk_times():
    rec = T.new_attempt(turn_id="t1", purpose="fast", model="m", tier=0, attempt_n=0)
    assert list(T.timed_stream(iter([]), rec)) == []
    assert rec["n_chunks"] == 0 and rec["t_first_chunk"] is None


def test_finish_ok_reads_finish_reason_and_computes_ttfb():
    class Result:
        response_metadata = {"finish_reason": "STOP"}
    rec = T.new_attempt(turn_id="t1", purpose="fast", model="m", tier=1, attempt_n=2)
    rec["t_first_chunk"] = rec["t_start"] + 0.25
    rec["t_last_chunk"] = rec["t_start"] + 0.5
    T.finish_ok(rec, Result())
    assert rec["outcome"] == "ok" and rec["finish_reason"] == "STOP"
    assert abs(rec["ttfb_s"] - 0.25) < 0.01 and rec["elapsed_s"] >= 0


def test_finish_exc_carries_classification_and_quota_fields():
    rec = T.new_attempt(turn_id="t1", purpose="general", model="gemma-4-31b-it", tier=0, attempt_n=0)
    T.finish_exc(rec, _rewrapped_429(), is_quota=True, is_stuck=False, retriable=True)
    assert rec["outcome"] == "exc" and rec["exc_type"] == "GoogleRateLimitError"
    assert rec["is_quota"] is True and rec["retriable"] is True
    assert rec["http_status"] == 429 and rec["quota_violations"][0]["quotaDimensions"]["model"] == "gemma-4-31b-it"


def test_finish_exc_truncates_message():
    rec = T.new_attempt(turn_id="t1", purpose="fast", model="m", tier=0, attempt_n=0)
    T.finish_exc(rec, RuntimeError("x" * 1000))
    assert len(rec["exc_msg"]) == 300


# ── emit / parse_line 往返 ────────────────────────────────────────────────────

def test_emit_writes_one_json_line_and_appends_to_sink():
    rec = T.new_attempt(turn_id="t1", purpose="fast", model="m", tier=0, attempt_n=0)
    T.finish_ok(rec)
    buf, sink = io.StringIO(), []
    line = T.emit(rec, sink=sink, stream=buf)
    assert buf.getvalue() == line + "\n"
    assert line.startswith("TELEMETRY {") and "\n" not in line
    assert sink == [rec]
    back = T.parse_line(line)
    assert back["turn_id"] == "t1" and back["outcome"] == "ok"


def test_emit_survives_broken_stream():
    class Broken(io.StringIO):
        def write(self, s):
            raise OSError("closed")
    rec = T.new_attempt(turn_id="t1", purpose="fast", model="m", tier=0, attempt_n=0)
    T.finish_ok(rec)
    assert T.emit(rec, stream=Broken()).startswith("TELEMETRY ")  # 不拋


def test_parse_line_rejects_non_telemetry_lines():
    assert T.parse_line("hello") is None
    assert T.parse_line("TELEMETRY not-json") is None


def test_emit_handles_non_serializable_via_default_str():
    rec = T.new_attempt(turn_id="t1", purpose="fast", model="m", tier=0, attempt_n=0)
    rec["weird"] = object()
    T.finish_ok(rec)
    assert T.parse_line(T.emit(rec, stream=io.StringIO())) is not None


# ── 版本標記 ─────────────────────────────────────────────────────────────────

def test_versions_has_expected_keys_and_app_md5(tmp_path):
    f = tmp_path / "Home.py"
    f.write_text("print(1)\n", encoding="utf-8")
    v = T.versions(app_file=str(f))
    assert set(v) == {"langchain_google_genai", "google_genai", "app_md5", "python"}
    assert v["app_md5"] and len(v["app_md5"]) == 10
    assert T.versions(app_file="/no/such/file")["app_md5"] is None


def test_new_turn_id_is_short_hex():
    t = T.new_turn_id()
    assert len(t) == 8 and int(t, 16) >= 0


# ── finish_reason：線上實測為 NULL 的兩個修法 ───────────────────────────────────

class _Msg:
    """模仿 langchain AIMessage/AIMessageChunk：finish_reason 在 response_metadata。"""
    def __init__(self, fr=None):
        self.response_metadata = {"finish_reason": fr} if fr else {}


def test_finish_ok_unwraps_fast_path_tuple():
    """Fast 的 _consume 回傳 (message, escalate_flag)——線上 finish_reason 為 NULL 的根因。"""
    rec = T.new_attempt(turn_id="t1", purpose="fast", model="m", tier=0, attempt_n=0)
    T.finish_ok(rec, (_Msg("STOP"), False))
    assert rec["finish_reason"] == "STOP"


def test_finish_ok_plain_message_still_works():
    rec = T.new_attempt(turn_id="t1", purpose="general", model="m", tier=0, attempt_n=0)
    T.finish_ok(rec, _Msg("MAX_TOKENS"))
    assert rec["finish_reason"] == "MAX_TOKENS"


def test_timed_stream_captures_finish_reason_from_last_chunk():
    """finish_reason 只在最後一個 chunk 出現；逐 chunk 覆寫後留下的是最後一個。"""
    rec = T.new_attempt(turn_id="t1", purpose="fast", model="m", tier=0, attempt_n=0)
    chunks = [_Msg(), _Msg(), _Msg("STOP")]
    list(T.timed_stream(iter(chunks), rec))
    assert rec["finish_reason"] == "STOP"
    # finish_ok 不覆寫 timed_stream 已記到的值（即使 result 是不帶 metadata 的 tuple）
    T.finish_ok(rec, (object(), True))
    assert rec["finish_reason"] == "STOP"


def test_new_attempt_has_no_mode_field():
    rec = T.new_attempt(turn_id="t1", purpose="fast", model="m", tier=0, attempt_n=0)
    assert "mode" not in rec
    assert rec["purpose"] == "fast"


# ── 接線檢查：模組寫了但 Home.py 沒接等於沒改 ─────────────────────────────────

def test_home_is_wired():
    src = (ROOT / "Home.py").read_text(encoding="utf-8")
    assert "from utils import telemetry as TELE" in src
    assert "TELE.new_attempt(" in src and "TELE.finish_ok(" in src and "TELE.finish_exc(" in src
    # 四個 LLM 串流消費點都要包：run_fast_turn_streaming._consume（Fast 主路徑）、
    # run_general_turn._consume_round（General 工具迴圈）、run_general_turn 的第二個 stream、
    # _run_subagent._consume（深研 persona）。漏包的那條首/末 chunk 時間會靜默缺失。
    assert src.count("TELE.timed_stream(") == 4, "四條串流迴圈都要包（漏一條就少一條時間線）"
    import re as _re
    unwrapped = [m.group(0) for m in _re.finditer(r"for c in (?!TELE\.timed_stream\()[^\n]*\.stream\(", src)]
    assert not unwrapped, f"仍有未包的串流迴圈：{unwrapped}"
    # turn_id 必須在 Fast/General 共同分派點初始化（且只一次），不能在 run_general_turn 裡：
    # _gm_rt 跨回合存活，放錯位置 Fast 回合會繼承上一個 General 回合的 turn_id 並 append 進舊 list。
    init = '_tele_rt["turn_id"] = TELE.new_turn_id()'
    assert src.count(init) == 1, "turn_id 初始化應恰好一處"
    assert src.count("TELE.new_turn_id()") == 1
    i_init = src.index(init)
    i_dispatch = src.index('with st.chat_message("assistant", avatar=')
    i_general = src.index("def run_general_turn(")
    i_main = src.index("if prompt or retry_payload or pending_prompt:")   # 主區塊：run_general_turn 之後、分派所在
    assert i_init < i_dispatch, "初始化要在 Fast/General 分派（chat_message）之前"
    assert i_general < i_main < i_init, "初始化必須在主區塊內（run_general_turn 之外、chat_message 之前）"
