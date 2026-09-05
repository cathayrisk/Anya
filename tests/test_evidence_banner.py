# -*- coding: utf-8 -*-
"""由憑證帳本生成的查證標示（utils/evidence_banner.py，2026-09-05 第 6 步）。

banner 是使用者唯一看得到的「這回合到底查了什麼」。它說錯話比不說更糟——
第 4 步之後 `if not web_happened:` 就開始對已取得官方資料的回合說
「內容來自模型既有知識」，那是主動誤導。

跑法：python -m pytest tests/test_evidence_banner.py -v
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
from utils.evidence_banner import build_banner as B  # noqa: E402

SRC = (ROOT / "Home.py").read_text(encoding="utf-8")
T0 = dt.datetime(2026, 9, 5, 5, 42, tzinfo=dt.timezone.utc)   # 台灣時間 13:42


def e(tool, scope, status, now=T0):
    return EV.make_event(tool=tool, scope=scope, status=status, now=now)


# ── 什麼都沒查 ──────────────────────────────────────────────────────────────
def test_nothing_retrieved_says_so_plainly():
    """這是最需要標示的情況，不能因為「沒東西可講」就不顯示。"""
    t = B([])
    assert t and "未經任何查證" in t and "模型既有知識" in t


# ── 官方來源 ────────────────────────────────────────────────────────────────
def test_official_lookup_names_what_was_checked_and_when():
    """只說「已查證」沒有用——使用者要知道查了哪一項、什麼時候。"""
    t = B([e("prefetch/get_earthquake_info", EV.SCOPE_EARTHQUAKE, EV.STATUS_OK),
           e("prefetch/get_typhoon_info", EV.SCOPE_TYPHOON, EV.STATUS_OK)])
    assert "中央氣象署" in t and "地震" in t and "颱風" in t
    assert "13:42" in t, "completed_at 是 UTC，顯示要換成台灣時間"
    assert "未經任何查證" not in t, "第 4 步之後這句話對這種回合就是假的"


def test_official_data_still_warns_the_rest_is_model_knowledge():
    """查了地震不代表整篇都查過——混合題最容易被誤解成全篇有據。"""
    t = B([e("t", EV.SCOPE_EARTHQUAKE, EV.STATUS_OK)])
    assert "其餘內容仍來自模型既有知識" in t


def test_empty_official_is_never_worded_as_nothing_happened():
    """⚠️ 全案通用的紅線：資料取不到 ≠ 沒發生。"""
    t = B([e("t", EV.SCOPE_EARTHQUAKE, EV.STATUS_EMPTY)])
    assert "沒有回傳資料" in t and "不等於沒發生" in t
    assert "目前沒有地震" not in t and "沒有地震" not in t


def test_failed_official_lookup_is_flagged_with_a_warning_icon():
    t = B([e("t", EV.SCOPE_TYPHOON, EV.STATUS_ERROR)])
    assert t.startswith("⚠️") and "查詢失敗" in t


# ── 網路搜尋：成功 ≠ 已查證 ─────────────────────────────────────────────────
def test_web_success_is_never_called_verification():
    """authority 分 official／open_web 就是為了這件事：搜到東西只證明搜尋跑過。"""
    t = B([e("web_search", EV.SCOPE_WEB, EV.STATUS_OK)])
    assert "做過網路搜尋" in t
    assert "不等於已查證" in t
    assert "已查證" not in t.replace("不等於已查證", "")


def test_web_with_zero_sources_is_distinguished_from_success():
    t = B([e("web_search", EV.SCOPE_WEB, EV.STATUS_EMPTY)])
    assert "沒有取得來源" in t


def test_web_only_does_not_get_the_official_icon():
    """🌐 保留給官方來源；網搜用 🔎，視覺上就分得出權威等級。"""
    assert B([e("web_search", EV.SCOPE_WEB, EV.STATUS_OK)]).startswith("🔎")
    assert B([e("t", EV.SCOPE_EARTHQUAKE, EV.STATUS_OK)]).startswith("🌐")


# ── 組合 ────────────────────────────────────────────────────────────────────
def test_multiple_sources_are_all_listed():
    t = B([e("get_weather", EV.SCOPE_WEATHER, EV.STATUS_OK),
           e("web_search", EV.SCOPE_WEB, EV.STATUS_OK),
           e("fetch_webpage", EV.SCOPE_PAGE, EV.STATUS_OK),
           e("doc_search", EV.SCOPE_DOC, EV.STATUS_OK)])
    for frag in ("天氣", "網路搜尋", "網頁", "文件"):
        assert frag in t


def test_any_failure_wins_the_icon_even_when_something_succeeded():
    """一半成功一半失敗時，使用者最需要知道的是失敗那半。"""
    t = B([e("t", EV.SCOPE_EARTHQUAKE, EV.STATUS_OK),
           e("web_search", EV.SCOPE_WEB, EV.STATUS_ERROR)])
    assert t.startswith("⚠️") and "網路搜尋失敗" in t and "地震" in t


def test_retry_success_is_not_reported_as_a_failure():
    """同一 scope 失敗後重試成功 → EV.summarize 取 ok，banner 不該還喊失敗。"""
    t = B([e("web_search", EV.SCOPE_WEB, EV.STATUS_ERROR),
           e("web_search", EV.SCOPE_WEB, EV.STATUS_OK)])
    assert "網路搜尋失敗" not in t and "做過網路搜尋" in t


# ── Home.py 接線 ────────────────────────────────────────────────────────────
def test_banner_shown_on_both_model_paths():
    """General 過去**完全沒有** banner——09-05 把宏達電寫成緯創那次，
    格式專業、語氣肯定、零標示。"""
    assert "EVBANNER.build_banner" in SRC
    assert SRC.count("EVBANNER.build_banner") == 2, "Fast 與 General 都要有"


def test_the_old_web_only_condition_is_gone():
    """`if not web_happened:` 在第 4 步之後會對已取得官方資料的回合說假話。"""
    assert "本回覆未經網路查證，內容來自模型既有知識" not in SRC


def test_fast_grounding_is_logged_or_the_banner_would_lie():
    """Fast 的查證走 Gemini grounding，不是 web_search 工具，帳本原本看不到它。
    漏記這一筆，有 grounding 的回合會被 banner 說成「未經任何查證」。"""
    assert 'tool="grounding"' in SRC and "scope=EV.SCOPE_WEB" in SRC


def test_fetch_webpage_now_logs_both_outcomes():
    body = SRC[SRC.index("def fetch_webpage("):]
    body = body[: body.index("\n@tool")]
    assert body.count("_log_evidence(") == 2
    assert "EV.SCOPE_PAGE" in body and "EV.STATUS_ERROR" in body


def test_web_search_failure_is_logged():
    body = SRC[SRC.index("def web_search("):]
    body = body[: body.index("\n@tool")]
    assert body.count("_log_evidence(") == 2, "成功與失敗都要記"


def test_banner_survives_into_history_not_just_the_live_turn():
    """⚠️ 2026-09-05 測試抓到的漏洞：banner 原本只用 st.caption() 當回合畫，沒存進歷史。
    44 則訊息實測下來，**歷史訊息的 banner 全部是 0 個**——誠實標示只在送出後
    到下一次 rerun 之間看得到，捲回去或重新整理就消失。badge 有存所以留得住，
    對比之下更容易讓人誤以為 banner 也在。"""
    assert SRC.count('"banner":') == 2, "Fast 與 General 兩條路徑都要存進歷史"
    assert 'st.caption(msg["banner"])' in SRC, "歷史重播要畫出來"
    # 存與畫的順序：畫的地方在歷史迴圈裡，必須在兩個 append 之前（腳本由上往下跑）
    assert SRC.index('st.caption(msg["banner"])') < SRC.index('"banner": _fast_banner')


def test_cwa_path_deliberately_has_no_banner():
    """🌐 氣象署直答本來就有更強的「由程式直接取自中央氣象署、未經模型改寫」footer，
    再加一句 banner 只是重複。"""
    i = SRC.index('"mode": "cwa"')
    assert '"banner"' not in SRC[i - 200: i + 300]


def test_weather_not_found_is_not_logged_as_success():
    """get_weather 的 status=not_found／outside_taiwan 代表**沒拿到任何天氣資料**，
    一律記 ok 的話 banner 會宣稱「已查詢氣象署天氣」——那是說謊。"""
    body = SRC[SRC.index("def get_weather("):]
    body = body[: body.index("\n@tool")]
    assert 'EV.STATUS_OK if output.get("status") == "ok" else EV.STATUS_EMPTY' in body
    assert "EV.STATUS_ERROR" in body, "except 也要記"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
