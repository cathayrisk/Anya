# -*- coding: utf-8 -*-
"""災防意圖三態分類（utils/hazard_intent.py，2026-09-05 第 3 步）。

案例大多取自一小時測試與 30 分鐘測試的**真實提問**，不是想像出來的句子——
T5「最近台灣有地震嗎？有沒有颱風要來？」正是這一步的起因。

跑法：python -m pytest tests/test_hazard_intent.py -v
"""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = next(p for p in [pathlib.Path(__file__).resolve().parent, *pathlib.Path(__file__).resolve().parents]
            if (p / "Home.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import hazard_intent as HI  # noqa: E402

SRC = (ROOT / "Home.py").read_text(encoding="utf-8")


# ── 起因案例 ────────────────────────────────────────────────────────────────
def test_the_t5_question_that_started_this():
    """一小時測試 T5：升級成功但工具 0 呼叫，憑記憶編出半年前的地震。"""
    it = HI.classify_hazard_intent("最近台灣有地震嗎？有沒有颱風要來？")
    assert it.state == HI.STATE_LIVE
    assert set(it.scopes) == {"earthquake_latest", "typhoon_active"}
    assert it.should_prefetch is True


# ── live ────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("q", [
    "現在有沒有颱風警報？",
    "今天台北天氣如何？",
    "剛剛是不是地震了",
    "幫我查一下今天台北天氣，然後說明什麼是核心通膨",
    "最新的地震規模多少",
    "颱風要來了嗎",
    "氣象署有發布豪雨特報嗎",
    "為什麼最近地震這麼多？",          # 同時有知識詞，但 live 優先
])
def test_live_questions(q):
    it = HI.classify_hazard_intent(q)
    assert it.state == HI.STATE_LIVE, it.signals
    assert it.should_prefetch is True


def test_live_beats_knowledge_when_both_present():
    """同一句裡兩種訊號都有時，live 優先——回答這種題需要真實資料做底。"""
    it = HI.classify_hazard_intent("最近地震這麼多，地震規模是怎麼定義的？")
    assert "最近" in it.signals["time"]
    assert it.signals["knowledge"], "知識詞確實有命中，但不該改變結論"
    assert it.state == HI.STATE_LIVE


# ── knowledge ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("q", [
    "地震有沒有分級？",                 # ⚠️ 裸的「有沒有」不算 live
    "地震規模是怎麼定義的？",
    "颱風是怎麼形成的",
    "芮氏規模跟震度的差別是什麼",
    "颱風的分級制度",
    "地震來的時候要怎麼避難",
])
def test_knowledge_questions_are_not_prefetched(q):
    it = HI.classify_hazard_intent(q)
    assert it.state == HI.STATE_KNOWLEDGE, it.signals
    assert it.should_prefetch is False


def test_the_regression_that_a_loose_terminator_caused():
    """原本第二式收裸的「沒」，「地震有沒有分級」→ 地震＋「有」＋「沒」在 4 字內 → 誤判 live。
    OAI 在 round-2 明確警告過這個 case。"""
    assert HI.LIVE_QUESTION_RE.search("地震有沒有分級？") is None
    assert HI.LIVE_QUESTION_RE.search("有沒有地震") is not None
    assert HI.LIVE_QUESTION_RE.search("颱風要來了嗎") is not None


# ── none / uncertain ────────────────────────────────────────────────────────
@pytest.mark.parametrize("q", [
    "幫我算 2330 的本益比",
    "什麼是核心通膨",
    "用 VBA 寫一個樞紐分析",
])
def test_non_hazard_questions_are_none(q):
    it = HI.classify_hazard_intent(q)
    assert it.state == HI.STATE_NONE
    assert it.scopes == () and it.should_prefetch is False


def test_uncertain_leans_toward_checking():
    """有災害詞、但既沒即時訊號也沒知識訊號 → 偏向去查。
    錯誤成本不對稱：多打一次本地 API vs 憑記憶編造即時災害資訊。"""
    it = HI.classify_hazard_intent("地震")
    assert it.state == HI.STATE_UNCERTAIN
    assert it.should_prefetch is True, "UNCERTAIN 存在的意義就是「不確定時偏向去查」"


# ── 誠實邊界 ────────────────────────────────────────────────────────────────
def test_tsunami_is_reported_as_uncovered_not_silently_dropped():
    """海嘯沒有任何 CWA 工具。標在 uncovered 讓呼叫端知道「屬於災防但無資料源」，
    而不是回 NONE 假裝這題與災防無關。"""
    it = HI.classify_hazard_intent("現在有海嘯警報嗎？")
    assert "海嘯" in it.uncovered
    assert it.scopes == (), "沒有可查的 scope"
    assert it.should_prefetch is False, "無工具可查時不該宣稱要 prefetch"
    assert it.state != HI.STATE_NONE


def test_scopes_match_the_evidence_module_vocabulary():
    """scope 字串若與 utils/evidence.py 對不上，第 4、5 步的覆蓋判斷會靜默失效。"""
    from utils import evidence as EV
    for sc in HI.ENTITY_SCOPES:
        assert sc in EV.ALL_SCOPES, sc


# ── Home.py 接線 ────────────────────────────────────────────────────────────
def test_wired_and_still_logged_every_turn():
    assert "classify_hazard_intent" in SRC, "應已接進 Home.py"
    assert '"gm_hazard_intent"' in SRC
    assert "[hazard_intent]" in SRC, "stdout shadow log 是唯一能看真實流量分佈的管道"


def test_classification_drives_prefetch_only_not_routing_or_model_choice():
    """第 3 步曾用一條測試禁止任何行為分支（先量再改）；第 4 步**刻意**解除了那個
    限制，讓 should_prefetch 驅動 controller prefetch。但範圍僅止於此——
    分類器不可以拿去改路由或選模型，那會把一個為「要不要查」設計的判準，
    悄悄變成決定配額怎麼花的東西。"""
    i = SRC.index('_hz = classify_hazard_intent')
    tail = SRC[i:]
    for forbidden in ('_hz.state', 'escalate_reason = "hazard_intent"'):
        # _hz.state 只該出現在 shadow log 的字典／print 裡，不該用來分支
        pass
    assert 'if _hz.state ==' not in SRC, "分類器不可用來改路由"
    assert 'mode = "general" if _hz' not in SRC
    assert "_hz.should_prefetch" in SRC, "第 4 步應以 should_prefetch 為閘門"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
