# -*- coding: utf-8 -*-
"""Token 預算門檻（D-1 歷史門檻 + D-3 全文預算，2026-09-04）。

背景：AI Studio dashboard 實測 gemma-4-31b 只有 16,000 input token/分（全表最緊），
而 tools/measure-doc-budget.py 量到一個「讀文件」回合是 5 輪 LLM 呼叫、歷史每輪重送：
  門檻 6,000 → 一回合合計 56,519 tok（TPM 的 3.5 倍）
  門檻 2,500 → 39,019 tok（2.4 倍）
歷史佔總成本 53%，是唯一「改一個常數就能減半」的項目。

這組測試守的是「數字不要被無意間改回去」與「門檻與保留量的關係仍成立」。

跑法：python -m pytest tests/test_token_budget.py -v
"""
from __future__ import annotations

import math
import pathlib
import re
import sys

import pytest

ROOT = next(p for p in [pathlib.Path(__file__).resolve().parent, *pathlib.Path(__file__).resolve().parents]
            if (p / "Home.py").exists())
SRC = (ROOT / "Home.py").read_text(encoding="utf-8")

TPM_GEMMA = 16_000     # dashboard 實測上限
SYS_TOKENS = 3_329     # 常駐 system prompt（tools/measure-doc-budget.py 量到）


def _const(name: str) -> int:
    m = re.search(rf"^{name}\s*=\s*([\d_]+)", SRC, re.M)
    assert m, f"找不到 {name}"
    return int(m.group(1).replace("_", ""))


def _est_tokens(n_chars: int) -> int:
    """複製 docstore.estimate_tokens_from_chars（不 import，避免拉進 faiss 等重相依）。"""
    return max(1, math.ceil(n_chars / 3.6))


def test_trigger_lowered_to_2500():
    assert _const("HISTORY_SUMMARY_TRIGGER_TOKENS") == 2_500


def test_trigger_leaves_room_within_one_call():
    """單輪 = system + 歷史。門檻必須讓單輪明顯低於 TPM，否則一次呼叫就爆窗。"""
    per_call = SYS_TOKENS + _const("HISTORY_SUMMARY_TRIGGER_TOKENS")
    assert per_call < TPM_GEMMA * 0.5, per_call


def test_recent_block_fits_under_trigger_for_normal_answers():
    """保留的近期原文若本身就超過門檻，摘要完仍在門檻上 → 每回合都重跑摘要。
    一般回答長度（≤1,500 字）下必須低於門檻，這是 keep 值與門檻的相容性條件。"""
    keep = _const("HISTORY_KEEP_RECENT_USER_TURNS")
    trigger = _const("HISTORY_SUMMARY_TRIGGER_TOKENS")
    for answer_chars in (800, 1_500):
        recent_tokens = _est_tokens(keep * (120 + answer_chars))   # 120 字/則使用者提問
        assert recent_tokens <= trigger, (answer_chars, recent_tokens, trigger)


def test_summarizer_runs_on_chore_chain_not_main_brain():
    """摘要器必須走雜活鏈：它若佔用 gemma 的 16K 分鐘窗，等於用主腦額度做壓縮，本末倒置。"""
    body = SRC[SRC.index("def _maybe_summarized_history"):]
    body = body[: body.index("\n# ")]
    assert "get_chore_llm()" in body
    assert 'purpose="chore"' in body


def test_summary_failure_falls_back_to_raw_history():
    """摘要是 token 優化，不是正確性依賴——失敗必須退回原始歷史，不能讓對話中斷。"""
    body = SRC[SRC.index("def _maybe_summarized_history"):]
    body = body[: body.index("\n# ")]
    assert "except Exception:" in body
    assert body.rstrip().endswith('return "", hist  # 摘要失敗：退回原始 trim 行為')


# ── D-3：fulltext 預算改由 TPM 決定 ────────────────────────────────────────────
from utils.token_budget import (  # noqa: E402
    fulltext_budget, tpm_limit_for, FULLTEXT_RESEND_ROUNDS, MIN_FULLTEXT_BUDGET,
)

REAL_DOC_MEDIAN = 2_379   # tools/measure-doc-budget.py 量到的實際 PDF（中位）
REAL_DOC_MAX = 2_912      # 同上（最大）


def test_gemma_tpm_is_the_binding_limit():
    assert tpm_limit_for("gemma-4-31b-it") == 16_000
    assert tpm_limit_for("gemma-4-26b-a4b-it") == 16_000
    # 未知／flash 家族一律寬估，否則會把 lite 的預算誤砍到 gemma 的水位
    assert tpm_limit_for("gemini-3.5-flash-lite") == 250_000
    assert tpm_limit_for("") == 250_000


def test_budget_fits_within_one_call_on_gemma():
    """核心：取得全文那一輪（base + 全文）必須低於 gemma 的 16K，否則單次呼叫就爆窗。"""
    for hist in (0, 1_500, 2_500):
        base = SYS_TOKENS + hist
        b = fulltext_budget("gemma-4-31b-it", base, context_budget=190_000)
        assert base + b <= TPM_GEMMA, (hist, base, b)


def test_budget_amortised_over_resend_rounds():
    """工具結果每輪重送，所以預算要攤到重送輪數上，不是單輪塞得下就好。"""
    base = SYS_TOKENS + 2_500
    b = fulltext_budget("gemma-4-31b-it", base, context_budget=190_000)
    assert b == (TPM_GEMMA - base) // FULLTEXT_RESEND_ROUNDS


def test_real_documents_still_returned_whole():
    """收緊預算不能把日常文件切碎——實測 PDF（最大 2,912 tok）仍要能整份回傳。"""
    b = fulltext_budget("gemma-4-31b-it", SYS_TOKENS + 2_500, context_budget=190_000)
    assert b >= REAL_DOC_MAX > REAL_DOC_MEDIAN


def test_lite_keeps_generous_budget():
    """降級到 flash-lite（250K TPM）之後預算不該被 gemma 的窄窗拖累。"""
    b = fulltext_budget("gemini-3.5-flash-lite", SYS_TOKENS + 2_500, context_budget=190_000)
    assert b == 60_000


def test_context_budget_still_caps():
    """context 快滿時仍以 context 為準（兩個上限取小者）。"""
    assert fulltext_budget("gemini-3.5-flash-lite", 1_000, context_budget=5_000) == 5_000


def test_returns_zero_when_no_room():
    """擠不出最低額度就回 0，呼叫端據此改走 doc_search。"""
    assert fulltext_budget("gemma-4-31b-it", 15_000, context_budget=190_000) == 0
    assert fulltext_budget("gemma-4-31b-it", 99_999, context_budget=190_000) == 0
    assert fulltext_budget("gemini-3.5-flash-lite", 1_000, context_budget=MIN_FULLTEXT_BUDGET - 1) == 0


def test_home_wires_tpm_budget_and_handles_zero():
    assert "from utils.token_budget import fulltext_budget" in SRC
    assert "rt[\"doc_fulltext_budget_hint\"] = fulltext_budget(" in SRC
    # 0 代表「額度不足」，不是「沒設定」——用 or 會放行 20,000，正是 D-3 要修的爆窗
    assert 'doc_fulltext_budget_hint") or 20000' not in SRC
    body = SRC[SRC.index("def doc_get_fulltext"):]
    body = body[: body.index("\n@tool")]
    assert "budget_exhausted" in body and "if budget_hint <= 0:" in body


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
