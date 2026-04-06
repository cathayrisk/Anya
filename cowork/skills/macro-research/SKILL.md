---
name: macro-research
description: >
  Use this workflow for tasks involving macroeconomic reports, central bank policy documents,
  economic indicator analysis, yield curve interpretation, monetary policy assessment,
  or macroeconomic forecasting.
  Triggered by: CPI, GDP, inflation, unemployment, yield curve, central bank, monetary policy,
  rate hike, rate cut, basis points, FOMC, ECB, BOJ, PBOC, recession, stagflation,
  通膨, 央行, 利率決策, 殖利率曲線, 景氣循環, 貨幣政策, 升息, 降息, 量化寬鬆,
  前瞻指引, 就業市場, 消費者物價, 核心通膨, 實質利率
---

# Macro-Research Workflow

## Purpose

This skill ensures analytical outputs on **macroeconomic documents** include
domain-specific validation: data vintage awareness, regime context,
yield curve interpretation, and policy transmission lag.

It applies **on top of** `critical-analysis/SKILL.md`. Run both together.

---

## When to Activate

Activate when the source is any of the following:

| Type | Examples |
|------|---------|
| **Central bank communication** | FOMC minutes, ECB meeting accounts, BOJ policy statement, Beige Book |
| **Official statistics** | CPI release, GDP advance estimate, NFP report, PCE deflator |
| **Macroeconomic forecast** | IMF WEO, World Bank outlook, investment bank macro research |
| **Bond/yield curve analysis** | 2s10s inversion commentary, term premium analysis, EM sovereign spreads |
| **Monetary/fiscal policy analysis** | QE tapering, rate path projections, dot plot interpretation |

---

## Step 1: Data Vintage and Revision Risk

Before analyzing any economic indicator, use `think` to assess data reliability:

```
think("""
此數據是初估值（advance）、修正值（revised）還是終值（final）？
歷史修正幅度如何？（例如：GDP 初估值平均修正 ±0.5%，就業數字常大幅修正）
是否接近報告截止日？資料可能反映的是上個月/上個季的情況。
季節性調整方式為何？X-13ARIMA-SEATS 還是其他方式？
""")
```

**Report flag format (data vintage):**

```markdown
:orange[**⚠️ 數據版本說明**：此 [指標名稱] 為 [初估/修正/終值]（[發布日期]）。
歷史平均修正幅度 ±[X]%。結論建立在可能被修正的數字上，建議等待修正值確認。]
```

---

## Step 2: Macroeconomic Regime Identification

Every macro analysis must identify the operating regime:

```
think("""
目前處於哪個總經環境？
  □ 通膨環境（CPI > 央行目標，實質利率為負或偏低）
  □ 緊縮環境（升息週期，流動性收縮）
  □ 停滯性通膨（高通膨 + 低成長）
  □ 通縮壓力（需求疲弱，價格下行）
  □ 擴張環境（低通膨 + 強成長，Goldilocks）

此分析的結論在哪個環境下成立？如果環境切換，結論如何改變？
""")
```

**Required conditional conclusion format:**

```markdown
**結論**：[核心觀點]

:small[前提條件：本結論成立於 [通膨/緊縮/停滯等環境] 之下。
若 [Fed 提前降息 / CPI 超預期上行 / 信用市場收緊]，結論可能轉為 [相反方向]。]
```

---

## Step 3: Yield Curve Interpretation (Dual-Direction, Rule 4)

When yield curve data appears, always provide dual interpretation:

| 曲線形態 | 多方解讀 | 空方解讀 |
|---------|---------|---------|
| 殖利率曲線倒掛（2s10s < 0） | 短期市場認為未來降息，長端被壓低 | 歷史上每次衰退前均出現，預警衰退風險 |
| 曲線陡化（steepening） | 經濟復甦預期增強，長端通膨補償回升 | 若短端下行驅動：恐慌式降息預期 |
| 曲線平坦化（flattening） | 央行有效控制通膨，實質利率上升 | 經濟活動放緩，未來盈利風險上升 |
| 期限溢酬（term premium）上升 | 長端供給壓力（財政赤字），需求消化正常 | 投資人要求更高補償，風險情緒惡化 |

**Report flag format (yield curve):**

```markdown
**[殖利率指標]**：[數值/走勢]

- :green[**正面解讀**]：[解讀 A，對應哪個情境]
- :red[**負面解讀**]：[解讀 B，對應哪個風險]
- **判讀前提**：此解讀成立於 [實質利率 / 通膨預期 / 信貸狀況] 條件下
```

---

## Step 4: Central Bank Communication Analysis

When analyzing FOMC minutes, ECB accounts, or any central bank statement:

```
think("""
前瞻指引強度：具體承諾（committed）/ 條件式（data-dependent）/ 選項式（open options）？
點陣圖（dot plot）與市場定價的差距：鷹派還是鴿派偏差？
措辭演變：與上一份聲明相比，哪些關鍵詞加入/刪除？（如 "sustained" → "some time" → "appropriate"）
政策不確定性水平：是否有委員持異議？聲明是否留有政策反轉空間？
""")
```

**Required output for policy analysis:**

```markdown
### 政策立場評估

**前瞻指引強度**：[強烈承諾 / 條件式 / 選項式]
**措辭變化重點**：與前次聲明比較，[加入/刪除] 了 "[關鍵詞]"，暗示 [解讀]
**市場定價 vs 官方指引**：市場定價 [X 次升/降息]，官方點陣圖暗示 [Y 次]，差距 [說明]

:small[注意：央行前瞻指引在高不確定性環境中具條件性，
若 [通膨/就業/金融穩定] 數據出現重大偏差，政策路徑可能急劇改變。]
```

---

## Step 5: Policy Transmission Lag (Rule 2)

Any conclusion about the impact of monetary policy must acknowledge transmission lags:

```
think("""
貨幣政策傳導管道：
  - 利率管道：影響企業借貸成本（通常 6-12 個月顯現）
  - 匯率管道：影響進出口，傳遞至通膨（通常 3-6 個月）
  - 資產價格管道：影響財富效果與信用條件（即時至 12 個月）
  - 預期管道：影響通膨預期（即時，但難以量化）

這份分析的結論需要多長的傳導時間才能驗證？
在傳導完成之前，指標可能呈現反方向的滯後效果嗎？
""")
```

**Required conditional statement for policy impact:**

```markdown
:small[政策傳導假設：本結論假設貨幣政策透過 [利率/匯率/資產價格] 管道
在 [6-18 個月] 內充分傳導。若傳導速度因 [信貸條件收緊/匯率對沖] 而延遲，
實際效果顯現的時間點可能推遲至 [X 季後]。]
```

---

## Step 6: Output Format

```markdown
# [報告標題]

## 摘要
[2-3 句，含當前總經環境定性 + 核心觀點 + 主要不確定性]

## 數據品質評估
- **數據版本**：[初估/修正/終值，發布日期]
- **修正風險**：[說明歷史修正慣例]
- **適用範圍**：[此數據涵蓋的地區/期間/樣本]

## 總經環境定位
[目前所處的景氣/通膨環境，說明成立條件]

## 主要指標分析
[正文，每個指標均附雙向解讀]

## 政策傳導路徑
[央行立場評估 + 傳導時間預估 + 條件性聲明]

## 殖利率曲線解讀（若適用）
[曲線形態 + 多空雙向解讀表]

## 批判性分析
[四維度缺口 + 環境切換風險 + 數據修正風險]

## 結論（條件性）
[核心結論 + 成立環境 + 推翻條件]

### 參考來源
[1] 來源名稱：發布機構，日期
```

---

## Step 7: Pre-Output Checklist

```
think("""
數據版本 ✓/✗：初估/修正/終值已說明？修正風險已提示？
環境定位 ✓/✗：所處總經環境已定性，切換情境已說明？
殖利率曲線 ✓/✗：多空雙向解讀已提供？
政策傳導 ✓/✗：傳導時間與不確定性已標注？
條件性結論 ✓/✗：所有預測/建議都附上成立前提？
""")
```

Only proceed to output when all five are ✓.
