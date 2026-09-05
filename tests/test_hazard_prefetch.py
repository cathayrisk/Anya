# -*- coding: utf-8 -*-
"""災防 controller prefetch（utils/hazard_prefetch.py，2026-09-05 第 4 步）。

這一步要保證的是「檢索不再由模型決定」。所以測試分兩半：
純函式的措辭與狀態處理，以及 Home.py 的接線位置——**接錯位置等於沒接**
（例如接在 General 分支裡，Fast 那條就完全沒受保護，而 Fast 根本沒有工具）。

跑法：python -m pytest tests/test_hazard_prefetch.py -v
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
from utils import hazard_prefetch as PF  # noqa: E402

SRC = (ROOT / "Home.py").read_text(encoding="utf-8")
T0 = dt.datetime(2026, 9, 5, 12, 34, tzinfo=dt.timezone(dt.timedelta(hours=8)))


# ── 可代呼叫的範圍 ──────────────────────────────────────────────────────────
def test_only_parameter_free_scopes_are_prefetchable():
    """天氣**刻意**不在內：get_weather 需要地點，使用者沒指名時只能餵預設地點，
    等於拿台北的天氣回答高雄的問題——那是換一種方式編造，比不查更難察覺。"""
    assert set(PF.PREFETCHABLE) == {EV.SCOPE_EARTHQUAKE, EV.SCOPE_TYPHOON}
    assert EV.SCOPE_WEATHER not in PF.PREFETCHABLE


def test_prefetch_scopes_filters_dedupes_and_keeps_order():
    got = PF.prefetch_scopes([EV.SCOPE_TYPHOON, EV.SCOPE_WEATHER,
                              EV.SCOPE_EARTHQUAKE, EV.SCOPE_TYPHOON])
    assert got == (EV.SCOPE_TYPHOON, EV.SCOPE_EARTHQUAKE)


def test_scopes_are_evidence_module_vocabulary():
    for sc in list(PF.PREFETCHABLE) + list(PF.SCOPE_LABELS):
        assert sc in EV.ALL_SCOPES, sc


# ── 呼叫層 ──────────────────────────────────────────────────────────────────
def test_failure_becomes_an_error_result_and_never_raises():
    """查詢失敗不該連帶弄壞整個回合。"""
    def boom():
        raise TimeoutError("cwa down")
    r = PF.run_prefetch([EV.SCOPE_EARTHQUAKE], {EV.SCOPE_EARTHQUAKE: boom})
    assert len(r) == 1
    assert r[0]["status"] == EV.STATUS_ERROR
    assert r[0]["payload"] is None
    assert "TimeoutError" in r[0]["error"]


def test_empty_status_is_preserved_not_flattened_to_ok():
    """地震 found=False 必須保持 empty——ok/empty 混為一談就等於編造。"""
    r = PF.run_prefetch([EV.SCOPE_EARTHQUAKE],
                        {EV.SCOPE_EARTHQUAKE: lambda: ('{"found": false}', EV.STATUS_EMPTY)})
    assert r[0]["status"] == EV.STATUS_EMPTY


def test_unknown_scope_is_skipped_silently():
    assert PF.run_prefetch([EV.SCOPE_WEATHER], {}) == []


def test_payload_is_capped_for_the_16k_tpm_budget():
    """颱風季 tracked_cyclones 會長出多組預報點，而 gemma 的 input TPM 只有 16K。"""
    big = "x" * (PF.MAX_PAYLOAD_CHARS + 5000)
    r = PF.run_prefetch([EV.SCOPE_TYPHOON], {EV.SCOPE_TYPHOON: lambda: (big, EV.STATUS_OK)})
    assert len(r[0]["payload"]) < len(big)
    assert "已截斷" in r[0]["payload"]


# ── context block 的措辭 ────────────────────────────────────────────────────
def test_no_results_means_no_block():
    assert PF.build_context_block([]) == ""


def test_block_says_where_the_data_came_from_and_when():
    """模型分不清 context 裡的東西從哪來；不講明就可能被當成使用者的說法忽略掉。"""
    r = PF.run_prefetch([EV.SCOPE_TYPHOON], {EV.SCOPE_TYPHOON: lambda: ("{}", EV.STATUS_OK)})
    b = PF.build_context_block(r, now=T0)
    assert "中央氣象署" in b
    assert "2026-09-05 12:34" in b
    assert "不是你的記憶" in b


def test_empty_must_not_be_worded_as_no_recent_earthquake():
    """⚠️ 這是整個檔案最重要的一條。「氣象署沒回傳事件」講成「近期沒有地震」，
    就是拿資料缺漏冒充事實——而且聽起來完全像正常回答，最難被發現。"""
    r = PF.run_prefetch([EV.SCOPE_EARTHQUAKE],
                        {EV.SCOPE_EARTHQUAKE: lambda: ('{"found": false}', EV.STATUS_EMPTY)})
    b = PF.build_context_block(r, now=T0)
    assert "沒有回傳可顯示的事件" in b
    assert "不等於" in b
    assert "不可以" in b and "近期沒有發生" in b


def test_error_item_is_labelled_and_carries_no_payload():
    def boom():
        raise ConnectionError("nope")
    b = PF.build_context_block(PF.run_prefetch([EV.SCOPE_TYPHOON], {EV.SCOPE_TYPHOON: boom}), now=T0)
    assert "查詢失敗" in b
    assert "不要用既有知識填補" in b


def test_block_tells_the_model_the_tools_are_already_done():
    """軟偏好，不是閘門——目的只是避免白白重打一次 CWA。"""
    r = PF.run_prefetch([EV.SCOPE_EARTHQUAKE, EV.SCOPE_TYPHOON], {
        EV.SCOPE_EARTHQUAKE: lambda: ("{}", EV.STATUS_OK),
        EV.SCOPE_TYPHOON: lambda: ("{}", EV.STATUS_OK)})
    b = PF.build_context_block(r, now=T0)
    assert "get_earthquake_info" in b and "get_typhoon_info" in b


# ── Home.py 接線 ────────────────────────────────────────────────────────────
def _prefetch_region() -> str:
    i = SRC.index("災防 controller prefetch（第 4 步）")
    return SRC[i: SRC.index("# ────────────────────────── Fast ──────────────────────────", i)]


def test_prefetch_runs_before_both_paths_not_inside_general():
    """接在 General 分支裡等於沒接：Fast **完全沒有工具**，是最需要被餵資料的一邊。
    例如「剛剛震央在哪」——HAZARD_HINT_RE 沒收「震央」，會留在 Fast，但分類器抓得到。"""
    i_pf = SRC.index("災防 controller prefetch（第 4 步）")
    assert i_pf < SRC.index('if mode == "fast":')
    # 用呼叫點而不是函式定義（def 在檔案前段，比對它沒有意義）
    assert i_pf < SRC.index("fast_resp, fast_escalate = run_fast_turn_streaming(lc_msgs")


def test_gated_on_should_prefetch_not_on_the_raw_state():
    """LIVE 與 UNCERTAIN 都要查（錯誤成本不對稱），所以閘門是 should_prefetch
    這個屬性，不是 state == live。"""
    region = _prefetch_region()
    assert "_hz.should_prefetch" in region
    assert 'state == "explicit_live"' not in region


def test_every_result_is_logged_including_failures():
    """③一定留憑證。失敗也要記，否則事後分不出「查了失敗」與「根本沒查」。"""
    region = _prefetch_region()
    assert "for _r in _pf_results:" in region
    assert "_log_evidence(" in region
    assert 'tool="prefetch/"' in region, "要分得出程式代查與模型自己呼叫"
    assert 'scope=_r["scope"]' in region and 'status=_r["status"]' in region


def test_injected_before_the_current_question_not_appended_after():
    """直接 append 會造成連續兩則 user，Gemma 對角色交替敏感；而且放在提問之前
    才是「最新鮮的背景」，本檔的滾動摘要與技能建議都用同一個形狀。"""
    region = _prefetch_region()
    assert "lc_msgs[:-1]" in region and "lc_msgs[-1:]" in region
    assert "lc_msgs.append(" not in region


def test_no_threads_in_the_prefetch_path():
    """utils/cwa_weather 會讀 st.secrets；沒有 ScriptRunContext 的執行緒裡碰 st.*
    在本專案已經咬過好幾次（st.write 還會靜默 no-op）。省 0.5 秒不值得。"""
    region = _prefetch_region()
    for bad in ("ThreadPoolExecutor", "threading.Thread", "asyncio"):
        assert bad not in region


def test_prefetch_failure_cannot_break_the_turn():
    region = _prefetch_region()
    assert region.count("except Exception:") >= 3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
