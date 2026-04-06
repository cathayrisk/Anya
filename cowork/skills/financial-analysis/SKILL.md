---
name: financial-analysis
description: >
  Use this workflow for tasks involving financial statements, equity research reports,
  sell-side analyst notes, earnings transcripts, company filings (10-K / 20-F / annual reports),
  valuation analysis, or M&A advisory materials.
  Triggered by: P/E ratio, DCF, EPS, revenue, EBITDA, gross margin, free cash flow,
  financial statements, earnings, 財報, 損益表, 資產負債表, 現金流量表,
  估值, 本益比, EV/EBITDA, 券商報告, 賣方研究, 研究報告, 投資建議, 目標價,
  承銷, IPO, 增資, 10-K, 20-F, 年報, 季報
---

# Financial Analysis Workflow

## Purpose

This skill ensures analytical outputs on **financial documents** include rigorous
domain-specific validation, beyond the general critical-analysis four-lens framework.

It applies **on top of** `critical-analysis/SKILL.md`. Run both together.

---

## When to Activate

Activate when the source document is any of the following:

| Type | Examples |
|------|---------|
| **Company filing** | 10-K, 20-F, Annual Report, 10-Q, Proxy Statement |
| **Earnings material** | Earnings call transcript, earnings release, IR presentation |
| **Sell-side research** | Analyst initiation report, rating change note, price target revision |
| **Valuation analysis** | DCF model, sum-of-parts, precedent transaction analysis |
| **M&A / advisory** | Fairness opinion, investment committee memo |

---

## Step 1: Source Classification

Before any analysis, use `think` to classify the source:

```
think("""
來源類型：(公司申報 / 法說會逐字稿 / 賣方研究 / 買方報告 / 投行顧問報告)
偏差假設：
  - 公司申報 → 管理層傾向正面呈現，注意 Non-GAAP 調整動機
  - 賣方研究 → 承銷業務利益衝突風險，評級歷史是否一致
  - 買方報告 → 通常持倉偏見，但方法論較嚴謹
  - 法說會逐字稿 → 前瞻性聲明有 Safe Harbor 保護，措辭可能刻意模糊
知道這個偏差後，哪些數字和論點需要額外謹慎？
""")
```

---

## Step 2: Financial Statement Integrity Check

When financial data is referenced, run **three** targeted `docstore_search` queries:

```
docstore_search("收益確認政策 / revenue recognition / 會計政策變更")
docstore_search("Non-GAAP 調整說明 / adjusted metrics / reconciliation")
docstore_search("關聯方交易 / 表外負債 / 租賃負債 / off-balance sheet")
```

**Cross-check integrity:**
- [ ] GAAP 與 Non-GAAP 差異是否說明清楚？
- [ ] 多年比較數字是否跨越了會計準則切換點（IFRS 16、ASC 842）？
- [ ] EBITDA 計算是否包含異常項目（重組費用、股份薪酬）？

---

## Step 3: Valuation Framework Audit (Rule 5)

When the analysis includes a valuation conclusion, verify the following:

### DCF Analysis
```
think("""
折現率（WACC）如何得出？無風險利率基準是哪個？股權風險溢酬來源？
終端增長率假設是否合理？（通常應 ≤ 長期名目 GDP 增速）
敏感性分析是否涵蓋 WACC ±1%、終端增長率 ±0.5% 的情景？
""")
```

### Comparable Company / Precedent Transaction
```
think("""
可比公司是如何選定的？是否有排除偏差（cherry-picking 高倍數同業）？
使用 forward 還是 trailing 倍數？以哪個時點的估計值為基準？
交易溢酬是否已扣除（control premium adjustment）？
""")
```

**Report flag format (valuation assumptions):**

```markdown
> **⚠️ 估值假設揭露**：本分析使用 [估值方法]。
> 關鍵假設：折現率 [X%]、終端增長率 [Y%]、可比公司 [說明選取標準]。
> 敏感性：WACC 每上升 1%，估值約下降 [Z%]（若未揭露則標注：未提供敏感性分析）。
```

---

## Step 4: Conflict of Interest Flag (Rule 6)

When source is sell-side or investment bank material:

```
think("""
發行機構是否同時承銷此公司的股票或債券？
評級歷史：過去 12 個月評級是否曾因業務關係而偏向正面？
分析師薪酬是否與投資銀行業務收入連動？（通常在報告免責聲明中揭露）
""")
```

**Report flag format (conflict of interest):**

```markdown
:orange[**⚠️ 利益衝突注意**：[機構名稱] 同時擔任 [公司名稱] 的 [承銷商/財務顧問]。
此評級應搭配獨立研究或買方觀點交叉驗證。]
```

If no conflict found, note explicitly: `本報告未揭露明顯利益衝突（惟仍建議查閱完整免責聲明）`

---

## Step 5: Apply General Critical Lenses

Also run all four lenses from `critical-analysis/SKILL.md`:

- Lens 1（批判性視角）: Challenge the investment thesis — what would make it wrong?
- Lens 2（條件性結論）: Every Buy/Sell recommendation must state conditions (e.g., "成立前提：利率不高於 X%")
- Lens 3（來源方法論）: Flag any proprietary screens or models used in stock selection
- Lens 4（反向解讀）: For any price target or EPS estimate, provide the bear-case interpretation

---

## Step 6: Output Format

Use this enhanced structure for financial analysis:

```markdown
# [報告標題]

## 摘要
[2-3 句核心發現，含主要估值假設與利益衝突提示（若有）]

## 來源評估
- **來源類型**：[公司申報 / 賣方研究 / ...]
- **潛在偏差**：[說明]
- **利益衝突**：[有/無，說明]

## 財務數據品質
[Non-GAAP 調整說明、會計政策一致性、跨期比較合理性]

## 估值假設摘要
[估值方法 + 關鍵假設 + 敏感性（若有）]

## 主要發現
[正文，引用來源 [1][2]]

## 批判性分析
[四維度結果 + 金融補充規則（Rule 5/6/7）缺口]

## 指標雙向解讀
[EPS 預估多空解讀、本益比高低兩種解讀、目標價情景]

## 結論（條件性）
[投資建議（若有）+ 成立前提 + 改變情境]

### 參考來源
[1] 來源：頁碼/段落
```

---

## Step 7: Pre-Output Checklist

```
think("""
估值框架 ✓/✗：使用的方法論與關鍵假設已說明？
利益衝突 ✓/✗：賣方來源的潛在偏差已標注？
會計一致性 ✓/✗：Non-GAAP 調整、跨期比較已說明？
條件性結論 ✓/✗：所有投資建議都附上成立前提？
""")
```

Only proceed to output when all four are ✓.
