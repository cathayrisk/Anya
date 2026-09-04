# -*- coding: utf-8 -*-
"""依「每分鐘 input token（TPM）」算文件全文預算（純函式，無 streamlit 依賴）。

2026-09-04 AI Studio dashboard 實測：gemma-4-31b／26b 的免費層上限是 **16,000 input token/分**，
是全表最緊的數字；flash 家族則是 250,000。而 Home.py 原本的 doc_fulltext_budget_hint 是用
「256K context 還剩多少」算的，給到 60,000——是 gemma 整個分鐘窗的近四倍，單位完全錯配。

更關鍵的是 tools/measure-doc-budget.py 量到的機制：工具結果 append 進 msgs 之後
**每一輪都會重送**。所以一份 F tokens 的全文，實際成本是 F × 它還會被重送的輪數，
不是 F。預算必須攤到那些輪數上，否則「單輪剛好塞得下」仍會讓整分鐘爆掉。
"""
from __future__ import annotations

# 免費層 input TPM（dashboard 實測；未列出的模型一律當 flash 家族的 250K）
MODEL_TPM_LIMITS: dict[str, int] = {
    "gemma-4-31b-it": 16_000,
    "gemma-4-26b-a4b-it": 16_000,
}
DEFAULT_TPM = 250_000          # flash / flash-lite 家族
FULLTEXT_RESEND_ROUNDS = 3     # 取得當輪 + 之後至少 think 與作答各一次
MIN_FULLTEXT_BUDGET = 2_000    # 低於此值全文工具沒有意義（doc_search 比較適合）
MAX_FULLTEXT_BUDGET = 60_000   # 不論多寬都不要一次塞爆 context


def tpm_limit_for(model: str) -> int:
    """該模型每分鐘的 input token 上限。未知模型保守不了——寬估反而安全，
    因為窄估會把 flash-lite 的全文預算誤砍到 gemma 的水位。"""
    return MODEL_TPM_LIMITS.get(model or "", DEFAULT_TPM)


def fulltext_budget(model: str, base_tokens: int, *, context_budget: int | None = None) -> int:
    """算 doc_get_fulltext 的 token 上限。

    model         本回合實際在跑的模型（降級後要用備援那顆，不是主力）
    base_tokens   每一輪都會重送的固定量：system prompt + 對話歷史
    context_budget  context 容量算出來的上限（沿用舊邏輯），取兩者小者

    回傳 0 表示連最低額度都擠不出來 → 呼叫端應退回 doc_search，不要讀全文。
    """
    headroom = tpm_limit_for(model) - max(0, int(base_tokens))
    if headroom <= 0:
        return 0
    budget = headroom // FULLTEXT_RESEND_ROUNDS
    if context_budget is not None:
        budget = min(budget, max(0, int(context_budget)))
    budget = min(budget, MAX_FULLTEXT_BUDGET)
    return budget if budget >= MIN_FULLTEXT_BUDGET else 0
