"""
finance_critic_rules.py — 金融領域批判分析規則擴充

提供三條金融專用驗證規則，作為 critic_pipeline.py 的 addendum。
設計為純字串常數 + 純函式，不依賴任何外部 API。
"""
from __future__ import annotations

# ── 金融領域額外驗證規則（接在通用五條規則後面）────────────────────────────
FINANCE_CRITIC_ADDENDUM = """\

---
以下為金融領域專用補充驗證規則，適用於涉及估值、財務報表、賣方研究、總體經濟分析的報告：

規則 6【估值框架一致性】：估值結論必須標明使用的框架與其假設
  → 觸發詞：P/E、本益比、EV/EBITDA、DCF、目標價、折現率、WACC、可比公司
  → 缺失條件 A：使用估值倍數但未說明基準（歷史均值？同業均值？哪些同業？）
  → 缺失條件 B：使用 DCF 但未揭露終端增長率、折現率（WACC）來源或敏感性
  → 缺失條件 C：使用前向/後向倍數（forward/trailing）但未說明時間基準
  → 缺口類型：估值框架缺口

規則 7【利益衝突揭露】：來源為賣方研究時，必須標注潛在利益衝突
  → 觸發詞：投資銀行、券商報告、買入評級、目標價上調、承銷、IPO、併購顧問
  → 缺失條件 A：未揭露來源機構是否與受評公司有承銷關係
  → 缺失條件 B：評級上調時間點與近期交易（IPO、增資、債券發行）高度重合但未說明
  → 缺失條件 C：分析師薪酬與業務收入掛鉤可能性未被提及
  → 缺口類型：利益衝突揭露缺口

規則 8【會計政策一致性】：跨期或跨公司財務數據比較必須說明會計政策基礎
  → 觸發詞：財務報表、損益表、資產負債表、EBITDA、Non-GAAP、調整後、同比、年增率
  → 缺失條件 A：比較多年財務數據但未說明是否調整過會計準則變更（IFRS 16、ASC 842 等）
  → 缺失條件 B：使用 Non-GAAP 指標但未提供與 GAAP 的調整說明
  → 缺失條件 C：跨公司比較但兩家公司使用不同的收益確認政策
  → 缺口類型：會計政策不一致缺口

---
金融規則適用時，輸出格式與原始五條規則相同。
每個缺口必須指向具體論點或段落，不能只說「整體缺乏...」。
若以上三條金融規則全數通過，說「金融補充規則通過」並在原始評分基礎上維持分數（不單獨加分）。
"""

# ── 觸發詞清單（金融領域高信心詞彙）────────────────────────────────────────
_FINANCE_KEYWORDS = frozenset([
    # 估值相關
    "p/e", "pe ratio", "本益比", "ev/ebitda", "ebitda", "dcf", "wacc",
    "目標價", "price target", "折現率", "終端增長率", "terminal growth",
    "可比公司", "comparable", "倍數", "multiple",
    # 財務報表相關
    "財務報表", "損益表", "資產負債表", "現金流量表", "income statement",
    "balance sheet", "cash flow", "non-gaap", "調整後獲利", "adjusted ebitda",
    "同比增長", "年增率", "year-over-year", "yoy",
    "revenue recognition", "收益確認", "ifrs 16", "asc 842",
    # 賣方研究相關
    "投資銀行", "券商報告", "賣方研究", "sell-side", "buy-side",
    "買入評級", "持有評級", "賣出評級", "rating", "analyst report",
    "承銷", "ipo", "增資", "債券發行", "investment banking",
    # 總經相關
    "殖利率曲線", "yield curve", "央行", "central bank", "利率決策",
    "升息", "降息", "rate hike", "rate cut", "basis points", "bps",
    "通膨", "inflation", "cpi", "gdp", "pce",
    "景氣循環", "business cycle", "衰退", "recession",
    "量化寬鬆", "quantitative easing", "qe", "taper",
    "fomc", "ecb", "boj", "pboc",
])

# 高信心觸發（只需出現 1 個即觸發）
_HIGH_CONFIDENCE_KEYWORDS = frozenset([
    "dcf", "wacc", "本益比", "p/e", "ev/ebitda", "non-gaap",
    "fomc", "ecb", "boj", "pboc", "yield curve", "殖利率曲線",
    "ifrs 16", "asc 842", "basis points", "bps",
])


def is_finance_context(text: str) -> bool:
    """
    判斷文字是否屬於金融領域，決定是否套用金融補充規則。

    策略：
    - 出現任一「高信心詞彙」→ 直接觸發
    - 出現 2 個以上「一般金融詞彙」→ 觸發
    - 其餘 → 不觸發（避免一般市場情緒分析被誤判）
    """
    lowered = text.lower()

    # 高信心：任一出現即觸發
    for kw in _HIGH_CONFIDENCE_KEYWORDS:
        if kw in lowered:
            return True

    # 一般金融詞彙：需至少 2 個才觸發
    hit_count = sum(1 for kw in _FINANCE_KEYWORDS if kw in lowered)
    return hit_count >= 2
