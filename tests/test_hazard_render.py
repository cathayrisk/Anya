# -*- coding: utf-8 -*-
"""純即時災防題的確定性渲染（utils/hazard_render.py，2026-09-05 第 5 步）。

這一步把最危險的一類問題整個移出模型的能力範圍。所以測試的重點有兩個：
渲染出來的字**逐項對得起 payload**，以及閘門**寧可放過也不要誤抓**——
誤判成純即時題會把混合題的另一半整個吃掉，而使用者不會察覺少了什麼。

跑法：python -m pytest tests/test_hazard_render.py -v
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
from utils import hazard_render as HR  # noqa: E402
from utils.hazard_intent import classify_hazard_intent as C  # noqa: E402

SRC = (ROOT / "Home.py").read_text(encoding="utf-8")
T0 = dt.datetime(2026, 9, 5, 13, 24, tzinfo=dt.timezone(dt.timedelta(hours=8)))

# 2026-09-05 從線上實際抓到的 payload（欄位長相以真實資料為準，不是想像的）
EQ = {"found": True, "earthquake_no": 115059, "origin_time": "2026-09-04T17:11:00+08:00",
      "location": "臺東縣政府東南東方  44.9  公里 (位於臺灣東南部海域)",
      "magnitude": 4.5, "depth_km": 13.7,
      "report_image_uri": "https://scweb.cwa.gov.tw/webdata/OLDEQ/x.png",
      "shaking_areas": [{"county": "臺東縣", "intensity": "4級"},
                        {"county": "花蓮縣", "intensity": "1級"},
                        {"county": "臺東縣", "intensity": "4級"},   # CWA 實測會回重複
                        {"county": "花蓮縣", "intensity": "1級"}]}
TY = {"has_active_taiwan_warning": False, "last_bulletin_headline": "解除颱風警報",
      "last_bulletin_time": "2026-08-28T14:30:00+08:00",
      "tracked_cyclones": [{"name": "KROVANH", "cwa_name": "科羅旺",
                            "latest_position": {"time": "2026-09-05T08:00:00+08:00",
                                                "lat": "27.5", "lon": "127.0",
                                                "max_wind_mps": "20", "moving_direction": "SSW",
                                                "moving_speed_kmh": "15"}}]}
OK_EQ = {"scope": EV.SCOPE_EARTHQUAKE, "status": EV.STATUS_OK, "error": None}
OK_TY = {"scope": EV.SCOPE_TYPHOON, "status": EV.STATUS_OK, "error": None}


def _render(results, payloads):
    return HR.render(results, payloads=payloads, now=T0)


# ── 閘門：該渲染的 ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("q", [
    "最近台灣有地震嗎？有沒有颱風要來？",   # T5 原句
    "剛剛震央在哪？",
    "現在有沒有颱風警報？",
    "颱風要來了嗎",
    "最新的地震規模多少",
])
def test_pure_live_questions_are_rendered(q):
    ok, why = HR.is_pure_live(q, C(q))
    assert ok, why


# ── 閘門：不該渲染的（每一條都有不同的擋下理由）──────────────────────────────
@pytest.mark.parametrize("q,expect", [
    ("最近地震這麼多，地震規模是怎麼定義的？", "知識詞"),
    ("地震有沒有分級？", "clear_knowledge"),
    ("地震", "uncertain"),
    ("台南剛剛有地震嗎？", "縣市名"),
    ("最近有地震嗎？順便幫我算 2330 本益比", "接續詞"),
    ("剛剛地震對台積電晶圓廠有影響嗎？", "殘留"),
    ("現在有海嘯警報嗎？", "沒有可查的 scope"),
    ("幫我查一下今天台北天氣，然後說明什麼是核心通膨", "無法代呼叫"),
])
def test_impure_questions_fall_back_to_the_model(q, expect):
    ok, why = HR.is_pure_live(q, C(q))
    assert not ok, f"不該渲染卻通過了：{q}"
    assert expect in why, why


def test_the_gate_leans_toward_letting_the_model_answer():
    """錯誤成本不對稱：交回模型只是慢一點（資料已由第 4 步放進 context），
    誤判成純即時題則會把混合題的另一半整個吃掉。"""
    ok, why = HR.is_pure_live("剛剛地震對台積電晶圓廠有影響嗎？", C("剛剛地震對台積電晶圓廠有影響嗎？"))
    assert not ok and "模板涵蓋不到" in why


def test_county_name_blocks_rendering_because_data_is_nationwide():
    """地震 payload 是「全國最新一筆顯著有感地震」，程式沒有能力只答台南那部分。"""
    ok, why = HR.is_pure_live("台南剛剛有地震嗎？", C("台南剛剛有地震嗎？"))
    assert not ok and "全國最新一筆" in why


# ── 渲染內容 ────────────────────────────────────────────────────────────────
def test_earthquake_fields_are_reproduced_verbatim():
    t = _render([OK_EQ], {EV.SCOPE_EARTHQUAKE: EQ})
    assert "2026-09-04 17:11" in t
    assert "芮氏 4.5" in t and "13.7 公里" in t
    assert "臺東縣政府東南東方 44.9 公里" in t, "CWA 的連續空白要壓成單一空白"


def test_duplicate_shaking_areas_are_deduped():
    """CWA 實測會回重複的縣市；照抄會變成「臺東縣 4級、花蓮縣 1級、臺東縣 4級…」。"""
    t = _render([OK_EQ], {EV.SCOPE_EARTHQUAKE: EQ})
    assert t.count("臺東縣 4級") == 1


def test_compass_direction_is_translated_by_table_not_by_the_model():
    """實測模型把 SSW 寫成「向西南西南西方向移動」。查表不會亂寫。"""
    t = _render([OK_TY], {EV.SCOPE_TYPHOON: TY})
    assert "向西南偏南移動" in t
    assert "西南西南西" not in t


def test_cyclone_name_pairs_chinese_with_english():
    t = _render([OK_TY], {EV.SCOPE_TYPHOON: TY})
    assert "科羅旺（KROVANH）" in t


def test_no_active_warning_is_stated_as_such_with_the_last_bulletin():
    t = _render([OK_TY], {EV.SCOPE_TYPHOON: TY})
    assert "沒有**對台生效的颱風警報" in t
    assert "解除颱風警報" in t and "2026-08-28 14:30" in t


def test_active_warning_shows_areas_and_description():
    ty = dict(TY, has_active_taiwan_warning=True, last_bulletin_headline="海上颱風警報",
              affected_areas=["巴士海峽", "臺灣東南部海面"],
              description=[{"title": "警戒區域", "value": "臺東縣\n屏東縣"}])
    t = _render([OK_TY], {EV.SCOPE_TYPHOON: ty})
    assert "目前有對台生效的颱風警報" in t
    assert "巴士海峽" in t and "警戒區域" in t


# ── 狀態措辭 ────────────────────────────────────────────────────────────────
def test_empty_is_never_worded_as_nothing_happened():
    """⚠️ 與 hazard_prefetch 同一條紅線：資料取不到 ≠ 沒發生。"""
    t = _render([{"scope": EV.SCOPE_EARTHQUAKE, "status": EV.STATUS_EMPTY, "error": None}], {})
    assert "資料暫時取不到" in t and "不等於近期沒有發生地震" in t


def test_error_says_the_lookup_failed_not_that_all_is_calm():
    t = _render([{"scope": EV.SCOPE_TYPHOON, "status": EV.STATUS_ERROR,
                  "error": "TimeoutError: cwa down"}], {})
    assert "查詢失敗" in t and "TimeoutError" in t
    assert "沒有颱風" not in t


def test_unexpected_payload_shape_falls_back_instead_of_half_a_template():
    """欄位長相變了就整段退回讓模型作答，不要吐半截模板。"""
    class Boom(dict):
        def get(self, *a, **k):
            raise RuntimeError("schema changed")
    # 必須放東西進去：空 dict 會被 `or {}` 換掉，測不到例外路徑
    assert _render([OK_EQ], {EV.SCOPE_EARTHQUAKE: Boom(found=True)}) == ""


def test_output_says_it_was_not_written_by_the_model():
    t = _render([OK_EQ], {EV.SCOPE_EARTHQUAKE: EQ})
    assert "中央氣象署" in t and "未經模型改寫" in t
    assert "2026-09-05 13:24" in t


def test_nothing_renderable_returns_empty_string():
    assert _render([], {}) == ""
    assert _render([{"scope": EV.SCOPE_WEB, "status": EV.STATUS_OK}], {}) == ""


# ── Home.py 接線 ────────────────────────────────────────────────────────────
def test_deterministic_branch_runs_before_both_model_paths():
    i = SRC.index("純即時災防題：程式直接渲染（第 5 步）")
    assert i < SRC.index('if mode == "fast":')
    assert i < SRC.index("fast_resp, fast_escalate = run_fast_turn_streaming(lc_msgs")


def test_branch_stops_the_turn_so_no_model_call_happens():
    """整個重點就是不呼叫模型。少了 st.stop() 會變成渲染完又跑一次 LLM。"""
    i = SRC.index("純即時災防題：程式直接渲染（第 5 步）")
    region = SRC[i: SRC.index("# ────────────────────────── Fast ──────────────────────────", i)]
    assert "st.stop()" in region


def test_badge_does_not_claim_a_model_answered():
    """沿用 Fast／General 徽章會謊報是哪顆模型答的——這回合根本沒有模型參與。"""
    from docstore.badges import badges_markdown
    assert "氣象署直答" in badges_markdown(mode="cwa", db_used=False, web_used=False)
    assert '_MODE_AVATAR' in SRC and '"cwa": "🌐"' in SRC


def test_deterministic_text_skips_finalize_response():
    """finalize_response 是用來剝除**模型**的不實查證宣稱的；這段字是程式產生的，
    沒有東西需要剝，硬過一層只會多一個誤傷面。"""
    i = SRC.index("純即時災防題：程式直接渲染（第 5 步）")
    region = SRC[i: SRC.index("# ────────────────────────── Fast ──────────────────────────", i)]
    code = "\n".join(ln for ln in region.splitlines() if not ln.lstrip().startswith("#"))
    assert "finalize_response(" not in code


def test_purity_decided_before_the_avatar_is_chosen():
    """avatar 在 st.chat_message 就定了；判定晚於它的話，本回合會顯示 ⚡／💬，
    但歷史重播卻是 🌐，同一則訊息兩個樣子。"""
    assert SRC.index("_hz_pure, _hz_pure_why") < SRC.index('with st.chat_message("assistant", avatar=')


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
