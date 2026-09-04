# tests/test_prediction_market.py
# -*- coding: utf-8 -*-
"""
pages/預測市場.py 的純邏輯測試（完全離線）。

為什麼要這樣載入：本機到 *.polymarket.com 的 DNS 被導向非官方 IP，無法對實際 API 驗證；
而 Streamlit 頁面直接 import 會執行到 UI 段（st.stop() 會炸）。
因此這裡把原始碼切在 `st.set_page_config` 之前，只 exec 純邏輯的上半段，
並注入一個把 st.cache_* 裝飾器變成 pass-through 的假 streamlit。
"""

from __future__ import annotations

import pathlib
import sys
import types

import pandas as pd
import pytest

PAGE = pathlib.Path(__file__).resolve().parents[1] / "pages" / "預測市場.py"
UI_MARKER = "st.set_page_config("


def _fake_streamlit() -> types.ModuleType:
    mod = types.ModuleType("streamlit")

    def _passthrough_decorator(*d_args, **d_kwargs):
        # 同時支援 @st.cache_data 與 @st.cache_data(ttl=60) 兩種寫法
        if len(d_args) == 1 and callable(d_args[0]) and not d_kwargs:
            return d_args[0]

        def wrap(fn):
            return fn

        return wrap

    mod.cache_data = _passthrough_decorator
    mod.cache_resource = _passthrough_decorator
    return mod


@pytest.fixture(scope="module")
def pm():
    """把頁面的純邏輯段落 exec 進一個獨立命名空間。"""
    src = PAGE.read_text(encoding="utf-8")
    assert UI_MARKER in src, "頁面結構改了：找不到 UI 段起點，請更新 UI_MARKER"
    logic_src = src[: src.index(UI_MARKER)]

    saved = sys.modules.get("streamlit")
    sys.modules["streamlit"] = _fake_streamlit()
    try:
        ns: dict = {"__name__": "pm_logic"}
        exec(compile(logic_src, str(PAGE), "exec"), ns)
    finally:
        if saved is not None:
            sys.modules["streamlit"] = saved
        else:
            sys.modules.pop("streamlit", None)
    return types.SimpleNamespace(**ns)


# -----------------------
# as_list / to_float
# -----------------------
def test_as_list_handles_gamma_json_strings(pm):
    assert pm.as_list('["Yes", "No"]') == ["Yes", "No"]
    assert pm.as_list(["Yes", "No"]) == ["Yes", "No"]
    assert pm.as_list("[Yes, No]") == ["Yes", "No"]      # 非合法 JSON 的退路
    assert pm.as_list(None) == []
    assert pm.as_list("[]") == []
    assert pm.as_list(123) == []


def test_to_float(pm):
    assert pm.to_float("0.68") == pytest.approx(0.68)
    assert pm.to_float(None) is None
    assert pm.to_float("") is None
    assert pm.to_float("abc") is None
    assert pm.to_float(float("nan")) is None


# -----------------------
# pick_price：官方規則（價差 > 0.10 才退回 last）
# -----------------------
def test_pick_price_uses_mid_when_spread_is_narrow(pm):
    px, src, spread = pm.pick_price({"bestBid": 0.60, "bestAsk": 0.64, "lastTradePrice": 0.90})
    assert px == pytest.approx(0.62)
    assert src == "mid"
    assert spread == pytest.approx(0.04)


def test_pick_price_falls_back_to_last_when_spread_is_wide(pm):
    px, src, spread = pm.pick_price({"bestBid": 0.30, "bestAsk": 0.70, "lastTradePrice": 0.55})
    assert px == pytest.approx(0.55)
    assert src.startswith("last")
    assert spread == pytest.approx(0.40)


def test_pick_price_boundary_is_inclusive_at_010(pm):
    # 剛好 0.10 仍算窄價差，要用中價
    px, src, _ = pm.pick_price({"bestBid": 0.45, "bestAsk": 0.55, "lastTradePrice": 0.99})
    assert px == pytest.approx(0.50)
    assert src == "mid"


def test_pick_price_without_orderbook(pm):
    px, src, spread = pm.pick_price({"lastTradePrice": 0.41})
    assert px == pytest.approx(0.41)
    assert src.startswith("last")
    assert spread is None

    px, src, _ = pm.pick_price({}, fallback=0.33)
    assert px == pytest.approx(0.33)
    assert src == "outcomePrices"

    px, src, _ = pm.pick_price({})
    assert px is None and src == "n/a"


# -----------------------
# event_outcomes：overround 正規化
# -----------------------
def _binary_event():
    return {
        "id": "e1", "slug": "fed-cut-september", "title": "Fed cuts in September?",
        "volume24hr": 1_250_000, "liquidity": 80_000, "endDate": "2026-09-17T00:00:00Z",
        "markets": [{
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0.62", "0.40"]',      # 加總 1.02 = overround
            "clobTokenIds": '["tokA", "tokB"]',
            "bestBid": "0.61", "bestAsk": "0.63", "lastTradePrice": "0.58",
        }],
    }


def _multi_event():
    def mk(title, yes, bid, ask, tok):
        return {
            "groupItemTitle": title,
            "outcomes": '["Yes", "No"]',
            "outcomePrices": f'["{yes}", "{1 - float(yes):.2f}"]',
            "clobTokenIds": f'["{tok}", "{tok}_no"]',
            "bestBid": str(bid), "bestAsk": str(ask), "lastTradePrice": str(yes),
        }
    return {
        "id": "e2", "slug": "who-wins", "title": "Who wins?",
        "volume24hr": 900_000, "liquidity": 50_000, "endDate": "2026-11-03T00:00:00Z",
        "markets": [
            mk("Alice", "0.50", 0.49, 0.51, "tA"),
            mk("Bob", "0.35", 0.34, 0.36, "tB"),
            mk("Carol", "0.23", 0.22, 0.24, "tC"),
        ],
    }


def test_binary_event_uses_mid_for_yes_and_normalizes(pm):
    odf, total, normalized = pm.event_outcomes(_binary_event())
    assert list(odf["outcome"]) == ["Yes", "No"]
    # Yes 走中價 (0.61+0.63)/2 = 0.62，不是 lastTradePrice 0.58
    assert odf.loc[0, "raw"] == pytest.approx(0.62)
    assert odf.loc[0, "來源"] == "mid"
    assert total == pytest.approx(1.02)
    assert normalized is True
    assert odf["機率_%"].sum() == pytest.approx(100.0, abs=0.2)


def test_multi_outcome_event_normalizes_to_100(pm):
    odf, total, normalized = pm.event_outcomes(_multi_event())
    assert list(odf["outcome"]) == ["Alice", "Bob", "Carol"]
    assert total == pytest.approx(0.50 + 0.35 + 0.23)     # 1.08 的 overround
    assert normalized is True
    assert odf["機率_%"].sum() == pytest.approx(100.0, abs=0.2)
    # 正規化後一定比原始價低（抽水被剝掉）
    assert (odf["機率_%"] <= odf["原始_%"] + 1e-9).all()
    assert list(odf["token_id"]) == ["tA", "tB", "tC"]


def test_event_without_markets_is_empty(pm):
    odf, total, normalized = pm.event_outcomes({"id": "x", "markets": []})
    assert odf.empty and total is None and normalized is False


def test_events_to_frame_picks_leading_outcome(pm):
    df = pm.events_to_frame([_binary_event(), _multi_event()])
    assert len(df) == 2
    row = df[df["event_id"] == "e2"].iloc[0]
    assert row["主要結果"] == "Alice"      # 多結果事件：取機率最高者
    assert bool(row["二元"]) is False
    assert row["結果數"] == 3
    assert row["endDate_台北"].isoformat() == "2026-11-03"   # UTC 00:00 → 台北同日 08:00


# -----------------------
# build_series / series_stats
# -----------------------
def test_build_series_converts_to_taipei_and_percent(pm):
    hist = [{"t": 1767225600, "p": 0.2}, {"t": 1767229200, "p": 0.8}]
    s = pm.build_series(hist)
    assert list(s["prob_%"]) == pytest.approx([20.0, 80.0])
    assert str(s["timestamp"].dt.tz) == "Asia/Taipei"


def test_build_series_tolerates_garbage(pm):
    assert pm.build_series([]).empty
    assert pm.build_series([{"nope": 1}]).empty


def test_series_stats_separates_drift_from_volatility(pm):
    """20 → 80 → 20：淨變化是 0，但這是全場最劇烈的市場。舊版只看淨變化會把它排到榜尾。"""
    s = pd.DataFrame({"prob_%": [20.0, 80.0, 20.0]})
    stats = pm.series_stats(s)
    assert stats["淨變化_pp"] == pytest.approx(0.0)
    assert stats["全距_pp"] == pytest.approx(60.0)
    assert stats["波動度_pp"] == pytest.approx(28.28, abs=0.01)
    assert stats["現值_%"] == pytest.approx(20.0)
    assert stats["樣本數"] == 3


def test_series_stats_needs_two_points(pm):
    assert pm.series_stats(pd.DataFrame({"prob_%": [50.0]})) is None
    assert pm.series_stats(pd.DataFrame()) is None


# -----------------------
# 設定與格式化
# -----------------------
def test_categories_exclude_sports_and_culture(pm):
    all_tags = {t for _, tags in pm.CATEGORIES for t in tags}
    assert 100639 not in all_tags, "Sports 不該出現在財經向頁面"
    assert 596 not in all_tags, "Culture 不該出現在財經向頁面"
    assert {120, 100265}.issubset(all_tags), "Finance 與 Geopolitics 是主軸"
    assert pm.CATEGORIES[0][0] == "財經綜合"


def test_range_map_month_not_minute(pm):
    assert pm.RANGE_MAP["1M"] == "1m"     # 官方語意就是「近一個月」
    assert pm.RANGE_MAP["ALL"] == "max"


def test_compact_number_uses_chinese_units(pm):
    """與表格欄位的 format="compact"（Streamlit 在地化成「萬」）保持同一套單位制。"""
    assert pm.compact_number(5413482.27) == "541萬"      # >=100 不留小數
    assert pm.compact_number(852439.31) == "85.2萬"      # <100 留一位
    assert pm.compact_number(2464508.59) == "246萬"      # 與表格顯示一致
    assert pm.compact_number(999) == "999"
    assert pm.compact_number(-2_500_000_000) == "-25.0億"
    assert pm.compact_number(None) == "N/A"
    assert pm.compact_number("x") == "N/A"


# -----------------------
# 正規化防呆（實測回報：有事件的價格加總只有 0.091）
# -----------------------
def _non_exhaustive_event():
    """幾個彼此獨立的問題被綁成同一個 event——加總 0.09，不是互斥窮盡的一組。"""
    def mk(title, yes, tok):
        return {
            "groupItemTitle": title,
            "outcomes": '["Yes", "No"]',
            "outcomePrices": f'["{yes}", "{1 - float(yes):.2f}"]',
            "clobTokenIds": f'["{tok}", "{tok}_no"]',
            "bestBid": str(round(float(yes) - 0.005, 3)),
            "bestAsk": str(round(float(yes) + 0.005, 3)),
            "lastTradePrice": str(yes),
        }
    return {
        "id": "e3", "slug": "bundle", "title": "獨立問題包",
        "volume24hr": 10_000, "liquidity": 5_000, "endDate": "2026-12-31T00:00:00Z",
        "markets": [mk("X 會發生嗎", "0.05", "tX"),
                    mk("Y 會發生嗎", "0.02", "tY"),
                    mk("Z 會發生嗎", "0.02", "tZ")],
    }


def test_does_not_normalize_when_sum_is_far_below_one(pm):
    """
    這是最重要的一條：加總 0.09 時若硬做正規化，5% 會被放大成 55%。
    那個數字看起來完全正常，卻是憑空捏造的——寧可原樣顯示。
    """
    odf, total, normalized = pm.event_outcomes(_non_exhaustive_event())
    assert total == pytest.approx(0.09, abs=0.01)
    assert normalized is False
    assert odf["機率_%"].tolist() == odf["原始_%"].tolist()
    assert odf["機率_%"].max() == pytest.approx(5.0, abs=0.1)      # 不是 55%
    assert odf["機率_%"].sum() == pytest.approx(9.0, abs=0.2)


def test_normalize_band_boundaries(pm):
    lo, hi = pm.NORMALIZE_BAND
    assert lo < 1.0 < hi
    assert lo >= 0.85 and hi <= 1.5, "區間放太寬就失去防呆意義"


def test_limit_cap_matches_measured_reality(pm):
    """實測 /events 上限是 100（傳 500/1000 都靜默回 100），不是坊間說的 500。"""
    assert pm.LIMIT_CAP == 100


def test_slim_event_drops_bulk_but_keeps_essentials(pm):
    fat = {
        "id": "e9", "title": "t", "volume24hr": 1, "endDate": "2026-01-01T00:00:00Z",
        "description": "x" * 50_000, "image": "https://…", "resolutionSource": "y" * 5_000,
        "markets": [{
            "outcomes": '["Yes","No"]', "outcomePrices": '["0.5","0.5"]',
            "clobTokenIds": '["a","b"]', "bestBid": "0.49", "bestAsk": "0.51",
            "spread": "0.02", "description": "z" * 50_000, "umaResolutionStatuses": "…",
        }],
    }
    slim = pm.slim_event(fat)
    assert "description" not in slim and "image" not in slim
    assert "description" not in slim["markets"][0]
    assert slim["id"] == "e9" and slim["markets"][0]["bestAsk"] == "0.51"
    # 解析結果不受影響
    odf, total, normalized = pm.event_outcomes(slim)
    assert not odf.empty and normalized is True


def test_spread_falls_back_to_gamma_field_when_no_bid(pm):
    """實測 bestBid 只有 90% 覆蓋率、bestAsk 與 spread 都是 100%。"""
    px, src, spread = pm.pick_price({"bestAsk": "0.04", "lastTradePrice": "0.03", "spread": "0.02"})
    assert px == pytest.approx(0.03)
    assert src.startswith("last")
    assert spread == pytest.approx(0.02)


# -----------------------
# Playwright 實測後補上的回歸測試
# -----------------------
def test_spread_is_shared_by_both_outcomes_of_one_market(pm):
    """
    Yes 和 No 共用同一本掛單簿，價差是簿子的屬性。
    修正前只有 index 0 有價差，導致「領先結果是 No」的事件（也就是所有機率 < 50%
    的市場）在列表上價差顯示 None——實測 6 個事件裡有 2 個中招。
    """
    odf, _, _ = pm.event_outcomes(_binary_event())
    assert odf["價差"].notna().all()
    assert odf.loc[0, "價差"] == pytest.approx(odf.loc[1, "價差"])


def test_low_probability_binary_event_shows_yes_not_no(pm):
    """
    3% 的尾部風險事件必須顯示 3%（Yes），不是 97%（No）。
    顯示 No 會讓語意反轉，而且列表裡會混雜 Yes 機率與 No 機率無法比較。
    順便確認價差在事件層有值。
    """
    ev = {
        "id": "tail", "slug": "tail", "title": "Tail risk?",
        "volume24hr": 100, "liquidity": 5000, "endDate": "2026-12-01T00:00:00Z",
        "markets": [{
            "outcomes": '["Yes", "No"]', "outcomePrices": '["0.03", "0.97"]',
            "clobTokenIds": '["y", "n"]', "bestBid": "0.025", "bestAsk": "0.035",
            "lastTradePrice": "0.03", "spread": "0.01",
        }],
    }
    row = pm.events_to_frame([ev]).iloc[0]
    assert row["主要結果"] == "Yes"                       # 不是 No
    assert row["機率_%"] == pytest.approx(3.0, abs=0.2)   # 不是 97
    assert row["價差"] is not None and row["價差"] == pytest.approx(0.01, abs=0.001)


def test_primary_row_binary_takes_yes_even_when_no_is_higher(pm):
    odf = pd.DataFrame({
        "outcome": ["Yes", "No"],
        "機率_%": [3.0, 97.0],
        "token_id": ["y", "n"],
    })
    assert pm.primary_row(odf, binary=True)["outcome"] == "Yes"
    assert pm.primary_row(odf, binary=False)["outcome"] == "No"
    assert pm.primary_row(pd.DataFrame(), binary=True) is None


def test_is_binary_event(pm):
    assert pm.is_binary_event(_binary_event()) is True
    assert pm.is_binary_event(_multi_event()) is False
    assert pm.is_binary_event({"markets": []}) is False


def test_spread_has_no_float_noise(pm):
    """顯示層會 round，但 CSV 匯出與無障礙層拿到的是原始值，所以要在來源就修掉。"""
    _, _, spread = pm.pick_price({"bestBid": 0.45, "bestAsk": 0.55, "lastTradePrice": 0.5})
    assert spread == 0.1                       # 不是 0.10000000000000003
    assert len(str(spread).split(".")[-1]) <= 4


# -----------------------
# negRisk 實測（2026-08-30 Cloud）：不能當判準，維持價格加總
# -----------------------
def test_negrisk_true_does_not_force_normalization(pm):
    """
    實測有 8 個事件 negRisk=True 但加總落在區間外。
    若讓 negRisk 覆寫價格加總，這 8 個會被錯誤正規化——正是要防的事。
    """
    ev = {
        "id": "n1", "title": "階梯市場", "negRisk": True,
        "volume24hr": 1, "liquidity": 9999, "endDate": "2026-12-01T00:00:00Z",
        "markets": [
            {"outcomes": '["Yes","No"]', "outcomePrices": '["0.02","0.98"]',
             "clobTokenIds": '["a","b"]', "groupItemTitle": "8/31 前", "spread": "0.01"},
            {"outcomes": '["Yes","No"]', "outcomePrices": '["0.03","0.97"]',
             "clobTokenIds": '["c","d"]', "groupItemTitle": "9/30 前", "spread": "0.01"},
        ],
    }
    odf, total, normalized = pm.event_outcomes(ev)
    assert total == pytest.approx(0.05)
    assert normalized is False                      # negRisk=True 也不能推翻價格加總
    assert odf["機率_%"].tolist() == odf["原始_%"].tolist()


def test_negrisk_false_does_not_block_normalization(pm):
    """反向：實測有 6 個事件 negRisk=False 但加總接近 1，這些仍應正規化。"""
    ev = {
        "id": "n2", "title": "互斥但沒標 negRisk", "negRisk": False,
        "volume24hr": 1, "liquidity": 9999, "endDate": "2026-12-01T00:00:00Z",
        "markets": [
            {"outcomes": '["Yes","No"]', "outcomePrices": '["0.60","0.40"]',
             "clobTokenIds": '["a","b"]', "groupItemTitle": "甲", "spread": "0.01"},
            {"outcomes": '["Yes","No"]', "outcomePrices": '["0.45","0.55"]',
             "clobTokenIds": '["c","d"]', "groupItemTitle": "乙", "spread": "0.01"},
        ],
    }
    odf, total, normalized = pm.event_outcomes(ev)
    assert total == pytest.approx(1.05)
    assert normalized is True
    assert odf["機率_%"].sum() == pytest.approx(100.0, abs=0.2)


def test_missing_prices_are_distinguishable_from_non_exhaustive(pm):
    """
    outcomePrices 覆蓋率實測只有 90%。加總偏低可能只是資料不全，
    介面要能分辨——靠 raw 欄位的缺值數判斷。
    """
    ev = {
        "id": "n3", "title": "缺價格", "negRisk": True,
        "volume24hr": 1, "liquidity": 9999, "endDate": "2026-12-01T00:00:00Z",
        "markets": [
            {"outcomes": '["Yes","No"]', "outcomePrices": '["0.55","0.45"]',
             "clobTokenIds": '["a","b"]', "groupItemTitle": "甲", "spread": "0.01"},
            {"outcomes": '["Yes","No"]', "clobTokenIds": '["c","d"]',
             "groupItemTitle": "乙"},                       # 完全沒有價格
        ],
    }
    odf, total, normalized = pm.event_outcomes(ev)
    assert normalized is False
    assert int(odf["raw"].notna().sum()) == 1 and len(odf) == 2   # 介面據此改口徑


# -----------------------
# 診斷頁倒出欄位清單後才啟用的三個欄位
# -----------------------
def test_month_change_pp_unit_and_fuse(pm):
    """
    主頁假設 oneMonthPriceChange 是 0–1 的價格差值（探針 8 會對帳確認）。
    那道 1.5 的保險絲是防「萬一 API 給的已經是百分點」被誤放大 100 倍。
    """
    assert pm.month_change_pp(0.08) == pytest.approx(8.0)
    assert pm.month_change_pp(-0.21) == pytest.approx(-21.0)
    assert pm.month_change_pp(0.002) == pytest.approx(0.2)
    assert pm.month_change_pp(25) == pytest.approx(25.0)     # 已是百分點 → 不再 ×100
    assert pm.month_change_pp(None) is None
    assert pm.month_change_pp("") is None


def test_uma_flag_shapes(pm):
    """狀態是有序序列（proposed → disputed → proposed），保留順序才看得出過程。"""
    assert pm.uma_flag({"umaResolutionStatuses": '["proposed"]'}) == "proposed"
    assert pm.uma_flag({"umaResolutionStatuses": ["proposed", "disputed"]}) == "proposed → disputed"
    assert pm.uma_flag({"umaResolutionStatuses": []}) is None
    assert pm.uma_flag({"umaResolutionStatuses": "[]"}) is None
    assert pm.uma_flag({"umaResolutionStatuses": None}) is None
    assert pm.uma_flag({}) is None


def test_new_fields_reach_the_event_frame(pm):
    ev = {
        "id": "nf", "slug": "nf", "title": "新欄位", "negRisk": False,
        "volume24hr": 1, "liquidity": 9999, "endDate": "2026-12-01T00:00:00Z",
        "markets": [{
            "outcomes": '["Yes","No"]', "outcomePrices": '["0.30","0.70"]',
            "clobTokenIds": '["y","n"]', "bestBid": "0.29", "bestAsk": "0.31",
            "spread": "0.02", "acceptingOrders": False,
            "umaResolutionStatuses": '["proposed"]', "oneMonthPriceChange": -0.12,
        }],
    }
    row = pm.events_to_frame([ev]).iloc[0]
    assert row["月變化_pp"] == pytest.approx(-12.0)
    assert bool(row["可交易"]) is False
    assert row["解析風險"] == "proposed"


def test_slim_event_keeps_the_new_fields(pm):
    """新欄位若沒進 MARKET_KEEP，會在瘦身時被砍掉，整條路徑靜默失效。"""
    for key in ("acceptingOrders", "umaResolutionStatuses", "oneMonthPriceChange"):
        assert key in pm.MARKET_KEEP, f"{key} 不在 MARKET_KEEP 裡"
    slim = pm.slim_event({
        "id": "s", "title": "t", "markets": [{
            "outcomes": '["Yes","No"]', "outcomePrices": '["0.5","0.5"]',
            "clobTokenIds": '["a","b"]', "acceptingOrders": True,
            "umaResolutionStatuses": '["proposed"]', "oneMonthPriceChange": 0.05,
            "description": "x" * 10_000,
        }],
    })
    m = slim["markets"][0]
    assert m["acceptingOrders"] is True and m["oneMonthPriceChange"] == 0.05
    assert "description" not in m


# -----------------------
# 探針 8 實測（2026-08-30 Cloud）：手續費 65%、uma 有 proposed / disputed 兩級
# -----------------------
def test_uma_level_grading(pm):
    """disputed 是真正的解析風險；proposed 只是正常流程。"""
    assert pm.uma_level({"umaResolutionStatuses": '["proposed"]'}) == "proposed"
    assert pm.uma_level({"umaResolutionStatuses": '["proposed", "disputed"]'}) == "disputed"
    # 實測最常見的爭議形態：提案 → 被挑戰 → 再提案，仍算 disputed
    assert pm.uma_level({"umaResolutionStatuses": '["proposed","disputed","proposed"]'}) == "disputed"
    assert pm.uma_level({"umaResolutionStatuses": "[]"}) is None
    assert pm.uma_level({}) is None


def test_worst_uma_level_escalates(pm):
    """事件裡只要有任一結果被挑戰，整個事件就標 disputed。"""
    odf = pd.DataFrame({"解析等級": ["proposed", None, "disputed"]})
    assert pm.worst_uma_level(odf) == "disputed"
    assert pm.worst_uma_level(pd.DataFrame({"解析等級": ["proposed", None]})) == "proposed"
    assert pm.worst_uma_level(pd.DataFrame({"解析等級": [None, None]})) is None
    assert pm.worst_uma_level(pd.DataFrame()) is None


def test_fee_info_uses_schedule_rate_not_taker_base_fee(pm):
    """
    實測：所有收費 market 的 takerBaseFee 都是固定 1000，但 feeSchedule.rate
    有 0.04 / 0.05 兩種——所以 rate 才是實際費率，takerBaseFee 不是。
    """
    m = {"feesEnabled": True, "feeType": "finance_prices_fees", "takerBaseFee": 1000,
         "feeSchedule": {"exponent": 1, "rate": 0.04, "takerOnly": True}}
    out = pm.fee_info(m)
    assert "4.0%" in out and "僅 taker" in out and "finance_prices_fees" in out
    assert "1000" not in out                      # takerBaseFee 不該外流到介面

    # feeSchedule 也可能是 JSON 字串
    m2 = dict(m, feeSchedule='{"exponent":1,"rate":0.05,"takerOnly":true}')
    assert "5.0%" in pm.fee_info(m2)

    assert pm.fee_info({"feesEnabled": False}) is None
    assert pm.fee_info({}) is None
    # 有收費但拿不到 rate → 仍要說有費用，不能靜默當成免費
    assert pm.fee_info({"feesEnabled": True, "feeType": "tech_fees"}) is not None


def test_fee_fields_survive_slimming(pm):
    for key in ("feesEnabled", "feeType", "feeSchedule"):
        assert key in pm.MARKET_KEEP, f"{key} 不在 MARKET_KEEP 裡，手續費揭露會整條失效"
