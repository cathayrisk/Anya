# -*- coding: utf-8 -*-
"""PREMIUM 輪替池（utils/premium_pool.py）＋ Home.py 接線。

跑法：python -m pytest tests/test_premium_pool.py -v
"""
from __future__ import annotations

import datetime as dt
import pathlib
import re
import sys

import pytest

ROOT = next(p for p in [pathlib.Path(__file__).resolve().parent, *pathlib.Path(__file__).resolve().parents]
            if (p / "Home.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.premium_pool import PremiumPool, pacific_day, quota_scope  # noqa: E402

TW = dt.timezone(dt.timedelta(hours=8))
POOL = ("a", "b", "c")


def test_rotation_starts_after_last_success():
    st = {}
    p = PremiumPool(POOL, st)
    assert p.candidates() == ["a", "b", "c"]
    p.advance("a")
    assert p.candidates() == ["b", "c", "a"]
    p.advance("c")
    assert p.candidates() == ["a", "b", "c"]


def test_exhausted_resets_at_taiwan_1500():
    """RPD 太平洋午夜重置 = 台灣 15:00（PDT 期間）。09-04 14:59 用盡 → 15:01 就回來。"""
    st = {}
    p = PremiumPool(POOL, st)
    t_before = dt.datetime(2026, 9, 4, 14, 59, tzinfo=TW)
    t_after = dt.datetime(2026, 9, 4, 15, 1, tzinfo=TW)
    assert pacific_day(t_before) != pacific_day(t_after)
    p.mark_exhausted("b", t_before)
    assert p.candidates(t_before) == ["a", "c"]
    assert p.exhausted_today(t_before) == ["b"]
    assert p.candidates(t_after) == ["a", "b", "c"]


def test_dead_is_permanent_and_pool_can_be_empty():
    st = {}
    p = PremiumPool(POOL, st)
    p.mark_dead("a")
    p.mark_dead("a")  # 冪等
    assert st["dead"] == ["a"]
    now = dt.datetime(2026, 9, 4, 10, 0, tzinfo=TW)
    p.mark_exhausted("b", now)
    p.mark_exhausted("c", now)
    assert p.candidates(now) == []          # 呼叫端此時退 31b
    assert p.candidates(now + dt.timedelta(days=1)) == ["b", "c"]


def test_state_survives_reconstruction():
    """state 是 session_state 裡的 dict，每次呼叫都會重新包一次 PremiumPool——不能靠物件屬性。"""
    st = {}
    PremiumPool(POOL, st).advance("b")
    assert PremiumPool(POOL, st).candidates() == ["c", "a", "b"]


def test_quota_scope():
    day = {"quota_violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}
    minute = {"quota_violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]}
    assert quota_scope(day) == "day"
    assert quota_scope(minute) == "minute"
    assert quota_scope({}) is None
    assert quota_scope(None) is None


# ── Home.py 接線 ──────────────────────────────────────────────────────────────
SRC = (ROOT / "Home.py").read_text(encoding="utf-8")


def test_premium_pool_has_all_five_rpd20_flash():
    m = re.search(r"^PREMIUM_POOL[^\n]*=\s*\(([^)]*)\)", SRC, re.M | re.S)
    assert m, "找不到 PREMIUM_POOL"
    ids = set(re.findall(r'"([^"]+)"', m.group(1)))
    assert "PREMIUM_MODEL" in m.group(1)
    for x in ("gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"):
        assert x in ids, f"PREMIUM_POOL 少了 {x}"
    assert not any("lite" in x or "gemma" in x for x in ids), "池裡不該有 lite／gemma（它們不是 RPD-20）"


def test_devils_advocate_uses_pool():
    body = SRC[SRC.index("def _run_devils_advocate"):]
    body = body[: body.index("\ndef ", 10)]
    assert "PremiumPool(PREMIUM_POOL" in body
    assert "pool.candidates()" in body and "pool.advance(model)" in body
    assert "pool.mark_exhausted(model)" in body and "pool.mark_dead(model)" in body
    assert "get_premium_llm(model)" in body
    assert 'purpose="premium"' in body, "premium 呼叫沒進 telemetry，?dev=1 看不到輪替"
    assert "from utils.premium_pool import PremiumPool, quota_scope" in SRC


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
