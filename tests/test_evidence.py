# -*- coding: utf-8 -*-
"""檢索憑證帳本（utils/evidence.py，2026-09-05 第 2 步）。

背景：一小時測試 T5 問「最近台灣有地震嗎」→ CWA 工具 0 呼叫卻宣稱查了氣象署。
要在系統層擋住，得先能回答「這回合檢索了什麼、成功沒、什麼時候」——
而現況（web/doc log 缺 timestamp/status/scope、CWA 完全沒 log）答不出來。

跑法：python -m pytest tests/test_evidence.py -v
"""
from __future__ import annotations

import datetime as dt
import pathlib
import sys

import pytest

ROOT = next(p for p in [pathlib.Path(__file__).resolve().parent, *pathlib.Path(__file__).resolve().parents]
            if (p / "Home.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import evidence as EV  # noqa: E402

SRC = (ROOT / "Home.py").read_text(encoding="utf-8")
T0 = dt.datetime(2026, 9, 5, 3, 5, 0, tzinfo=dt.timezone.utc)


def test_event_has_the_three_fields_plus_derived_authority():
    e = EV.make_event(tool="get_earthquake_info", scope=EV.SCOPE_EARTHQUAKE,
                      status=EV.STATUS_OK, now=T0)
    assert e["scope"] == "earthquake_latest"
    assert e["status"] == "ok"
    assert e["completed_at"] == "2026-09-05T03:05:00+00:00"
    assert e["authority"] == "official", "authority 由 scope 推導，不佔第四欄"


def test_authority_separates_official_from_web_and_internal():
    """web_search 成功不能被當成「已查證」，doc_search 也不是即時網路查證。"""
    assert EV.AUTHORITY[EV.SCOPE_EARTHQUAKE] == "official"
    assert EV.AUTHORITY[EV.SCOPE_WEB] == "open_web"
    assert EV.AUTHORITY[EV.SCOPE_DOC] == "internal"


@pytest.mark.parametrize("bad", [("nope", EV.STATUS_OK), (EV.SCOPE_WEB, "fine")])
def test_rejects_uncontrolled_vocabulary(bad):
    """允許自由字串的話，半年後會有五種寫法，覆蓋判斷跟著失效。"""
    with pytest.raises(ValueError):
        EV.make_event(tool="t", scope=bad[0], status=bad[1])


# ── coverage：逐 scope 判斷，不是全域布林 ────────────────────────────────────
def test_coverage_is_per_scope_not_global():
    """T5 的核心教訓：web_search 成功不代表地震 scope 被覆蓋。"""
    events = [EV.make_event(tool="web_search", scope=EV.SCOPE_WEB, status=EV.STATUS_OK)]
    assert EV.coverage(events, EV.SCOPE_WEB)["covered"] is True
    assert EV.coverage(events, EV.SCOPE_EARTHQUAKE)["covered"] is False


def test_no_event_means_not_covered():
    c = EV.coverage([], EV.SCOPE_EARTHQUAKE)
    assert c["covered"] is False and c["status"] is None


def test_empty_and_error_are_distinct_and_neither_is_covered():
    """「官方回了空集合」與「API 掛了」對使用者是兩件事，措辭不能一樣。"""
    empty = [EV.make_event(tool="t", scope=EV.SCOPE_EARTHQUAKE, status=EV.STATUS_EMPTY)]
    err = [EV.make_event(tool="t", scope=EV.SCOPE_EARTHQUAKE, status=EV.STATUS_ERROR)]
    assert EV.coverage(empty, EV.SCOPE_EARTHQUAKE)["status"] == "empty"
    assert EV.coverage(err, EV.SCOPE_EARTHQUAKE)["status"] == "error"
    assert not EV.coverage(empty, EV.SCOPE_EARTHQUAKE)["covered"]
    assert not EV.coverage(err, EV.SCOPE_EARTHQUAKE)["covered"]


def test_success_wins_over_earlier_failure():
    """同一 scope 重試後成功 → 以成功為準（否則一次暫時性失敗會永久標成未查證）。"""
    events = [EV.make_event(tool="t", scope=EV.SCOPE_TYPHOON, status=EV.STATUS_ERROR),
              EV.make_event(tool="t", scope=EV.SCOPE_TYPHOON, status=EV.STATUS_OK)]
    assert EV.coverage(events, EV.SCOPE_TYPHOON)["covered"] is True
    assert EV.summarize(events) == {"typhoon_active": "ok"}


# ── Home.py 接線 ────────────────────────────────────────────────────────────
def test_cwa_tools_log_both_success_and_failure():
    """⚠️ 例外也必須記——原本 CWA 的 except 只把錯誤字串回給模型，系統這側不留痕跡，
    事後就分不出「查了沒資料」「API 掛了」「根本沒查」。"""
    for tool, scope in (("get_earthquake_info", "SCOPE_EARTHQUAKE"),
                        ("get_typhoon_info", "SCOPE_TYPHOON")):
        body = SRC[SRC.index(f"def {tool}("):]
        body = body[: body.index("\n@tool")]
        assert body.count("_log_evidence(") == 2, f"{tool} 應同時記成功與失敗"
        assert f"EV.{scope}" in body
        assert "EV.STATUS_ERROR" in body, f"{tool} 的 except 沒記 error event"


def test_earthquake_empty_is_not_ok():
    """found=False 是「CWA 沒回傳事件」，**不是**「近期沒有地震」——記 empty 不記 ok。"""
    body = SRC[SRC.index("def get_earthquake_info("):]
    body = body[: body.index("\n@tool")]
    assert 'EV.STATUS_OK if output.get("found") else EV.STATUS_EMPTY' in body


def test_typhoon_no_warning_is_ok_not_empty():
    """颱風語意較好：has_active_taiwan_warning 是明確布林，
    「目前無對台生效警報」是官方的有效答案。"""
    body = SRC[SRC.index("def get_typhoon_info("):]
    body = body[: body.index("\n@tool")]
    i = body.index("_log_evidence(tool=\"get_typhoon_info\", scope=EV.SCOPE_TYPHOON, status=EV.STATUS_OK")
    assert i > 0


def test_web_and_doc_also_logged():
    assert 'scope=EV.SCOPE_WEB' in SRC and 'scope=EV.SCOPE_DOC' in SRC


def test_ledger_is_reset_every_turn():
    """跨回合累積會讓「這回合查過地震嗎」永遠答 yes——那正是要防的事。"""
    assert 'st.session_state["gm_evidence_log"] = []' in SRC
    i = SRC.index('st.session_state["gm_evidence_log"] = []')
    around = SRC[i - 700:i]
    assert 'TELE.new_turn_id()' in around, "應與 turn_id 一起重置（兩條路徑都經過的分派點）"


def test_logging_helper_never_raises():
    """憑證記錄失敗不該連帶弄壞回答。"""
    body = SRC[SRC.index("def _log_evidence("):]
    body = body[: body.index("\ndef ")]
    assert "except Exception:" in body and "pass" in body


def test_dev_panel_shows_ledger_on_every_answering_path():
    """2026-09-05 第 5 步後變成三條：Fast、General，以及純即時災防題的程式渲染。
    任何一條看不到帳本，那條路徑上的「查了沒」就無從查核。"""
    assert SRC.count("🔧 [dev] evidence ledger") == 3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
