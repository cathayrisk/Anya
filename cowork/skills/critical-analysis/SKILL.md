---
name: critical-analysis
description: >
  Use this workflow when the task requires critical evaluation of documents,
  transcripts, reports, or data. Triggered by: analyzing strategy docs, market
  reports, research findings, data with indicators/scores, or uploaded materials
  for critical review. Keywords: "分析", "批判", "評估", "逐字稿", "市場分析",
  "策略分析", "報告", "指標", "預測", "情緒", "analyze", "evaluate", "critique".
---

# Critical Analysis Workflow

## Purpose

This skill ensures analytical outputs include four quality dimensions:

1. **批判性視角** — challenge assumptions, surface alternative explanations
2. **條件性結論** — every conclusion must state its preconditions
3. **來源方法論審查** — question proprietary models and undisclosed frameworks
4. **反向解讀** — dual-direction interpretation of indicators and predictions

Apply this skill **on top of** the standard research-and-report workflow.
It does not replace information gathering — it enriches the synthesis step.

---

## When to Activate

Activate when the task involves **any** of the following:

- Analyzing uploaded documents (transcripts, strategy docs, reports, research papers)
- Evaluating data containing scores, indices, sentiment indicators, or predictions
- Sources that reference named frameworks, models, or proprietary methodologies
- Tasks phrased as: 分析、評估、批判性分析、策略分析、市場研究、逐字稿

---

## Step 1: Map the Four Dimensions

Before gathering information, use `think` to identify which dimensions apply:

```
think("""
批判性視角：主要論點是什麼？有哪些反例或替代解釋？有什麼被刻意忽略了？
條件性結論：結論成立的前提是什麼？前提改變時結論如何轉變？
來源方法論：來源使用了哪些模型/框架/指數？計算方式是否公開？
反向解讀：有哪些數據屬於情緒/意見/預測類？需要多空雙向解讀嗎？
""")
```

---

## Step 2: Gather with Critical Queries

When using `docstore_search`, add methodology-focused queries beyond the standard ones:

```
docstore_search("方法論 / 數據來源 / 計算方式 / 框架假設")
docstore_search("限制 / 前提假設 / 適用條件 / 樣本範圍")
docstore_search("反向論點 / 風險情境 / 壓力測試 / 不利因素")
```

When delegating to `research-agent`, include verification requests:
- Who published this data? Any conflict of interest or funding bias?
- What is the sample size, time period, and geographic scope?
- Has this model/framework been independently validated?

---

## Step 3: Apply the Four Lenses

### Lens 1: 批判性視角 (Critical Perspective)

For **each** major claim or finding:

- Identify at least one counter-argument or alternative explanation
- Ask: "What evidence would make this conclusion wrong?"
- Note: What relevant factors does the source **not mention**?

Report format:

```markdown
> **⚠️ 批判性注記**：[論點] 的前提是 [假設]。
> 另一種解讀是 [替代解釋]。
> 未提及的因素包括：[遺漏點]。
```

---

### Lens 2: 條件性結論 (Conditional Conclusions)

Every conclusion or recommendation **must** state the conditions under which it holds.

Required format for all conclusions:

```markdown
**結論**：[核心結論]

:small[前提條件：本結論成立於 [條件 A] 及 [條件 B] 的情況下。
若 [條件 A] 改變（例如：[具體情境]），結論可能轉為 [相反或修正的方向]。]
```

Before finalizing conclusions, verify:
- [ ] 結論是否依賴尚未驗證的假設？
- [ ] 結論是否有時效性限制？（若有，標注有效期）
- [ ] 結論是否因地區 / 產業 / 規模不同而有差異？

---

### Lens 2.5: 假設記錄（Assumption Logging）

在寫出每個結論之前，用 `think` 把「沒有明說但被當作真的事」挖出來：

- 「數據代表 X 群體」的假設
- 「這個指標的高/低有普遍公認基準」的假設
- 「時間範圍與當前情境相關」的假設
- 「來源機構沒有利益衝突」的假設

在報告中明確標注：

```markdown
> **假設記錄**：本段落假設「[X]」。
> 此假設若不成立（例如：[具體情境]），結論將[如何變化]。
```

---

### Lens 3: 來源方法論審查 (Source Methodology Review)

When a source references data, scores, or indices, check for:

- Named proprietary model or index → disclose in report
- Undisclosed calculation method → flag explicitly
- Survey or poll data → note sample size and demographics

**Report flag format (methodology known):**

```markdown
:orange[**⚠️ 方法論說明**：此數據來自 [機構] 的 [框架/指數名稱]。
計算方式：[公開說明摘要]。獨立驗證：[有 / 無 / 部分]。
使用時應注意：[具體限制]。]
```

**Report flag format (methodology undisclosed):**

```markdown
:orange[**⚠️ 方法論未公開**：[來源] 未說明此 [指標/評分] 的計算方式。
讀者應將此數據視為參考性質，而非可獨立驗證的事實。]
```

---

### Lens 4: 反向解讀 (Counter-Interpretation)

Applies to: sentiment indices, emotion scores, survey results, prediction models, market forecasts.

**Rule**: Whenever a metric can be interpreted in two directions, always provide both.

Report format:

```markdown
**[指標名稱]**：[數值/結果]

- :green[**多方解讀**]：若視為正面信號，意味著 [解讀 A]，支持 [行動/結論 A]
- :red[**空方解讀**]：若視為警示信號，意味著 [解讀 B]，提示 [風險/結論 B]
- **判讀建議**：需搭配 [其他指標/背景條件] 才能確定方向
```

Common patterns requiring dual interpretation:

| 指標類型 | 多方解讀 | 空方解讀 |
|---------|---------|---------|
| 高情緒樂觀指數 | 市場信心強，動能持續 | 過熱警示，可能反轉 |
| 低波動率 | 市場穩定，風險可控 | 自滿情緒，尾部風險累積 |
| 高搜尋熱度 | 大眾關注，資金流入可期 | 散戶追高，接近峰值 |
| 機構加碼 | 聰明錢進場，趨勢確立 | 法人已充分持倉，上行空間有限 |
| 強烈正向措辭 | 執行團隊有信心 | 可能迴避負面資訊 |

---

## Step 4: Structure the Critical Report

Use this enhanced structure for analytical tasks:

```markdown
# [報告標題]

## 摘要
[2-3 句核心發現，含主要前提條件]

## 主要發現
[正文段落，引用來源 [1][2]]

## 批判性分析

### 主要論點的前提與限制
[Lens 1：反問與替代解釋]
[Lens 2：每個結論的條件性說明]

### 來源方法論說明
[Lens 3：所有數據的方法論旗標，未公開者明確標注]

## 指標雙向解讀
[Lens 4：情緒/預測類指標的多空解讀表格]

## 結論（條件性）
[完整結論，含前提條件與改變情境]

### 參考來源
[1] 來源標題：URL
[2] 來源標題：URL
```

---

## Step 5: Self-Check Before Output

Before finalizing, run this checklist with `think`:

```
think("""
批判性視角 ✓/✗：每個主要論點有反問或替代解釋嗎？
條件性結論 ✓/✗：所有結論都標注了成立前提嗎？
方法論審查 ✓/✗：所有數據來源的框架都說明或旗標了嗎？
反向解讀 ✓/✗：情緒/預測類指標都有多空雙向解讀嗎？
""")
```

Only proceed to output when all four are ✓.
If any is ✗, go back to Step 3 and address the gap before writing.
