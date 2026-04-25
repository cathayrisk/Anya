"""
critic_pipeline.py — 五維度批判分析管道

純 Python 模組，不依賴 Streamlit。
由 Home.py 的 critique_analysis / check_source_framework 工具呼叫。
使用 Responses API（gpt-5.5）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# 金融領域補充規則（安全 import，缺少時靜默降級）
try:
    from cowork.finance_critic_rules import FINANCE_CRITIC_ADDENDUM, is_finance_context
    _HAS_FINANCE_RULES = True
except ImportError:
    FINANCE_CRITIC_ADDENDUM = ""
    _HAS_FINANCE_RULES = False
    def is_finance_context(text: str) -> bool:  # noqa: E302
        return False

# ── 五維度批判 Prompt ──────────────────────────────────────────────────────────
_CRITIC_PROMPT = """\
你是一位分析型報告的缺口驗證代理人（Gap Verification Agent）。
任務：用結構化規則檢查「這份報告的結論能不能被相信」，而不是評判「報告寫得好不好」。

## 驗證規則（五條）

規則 1【批判性視角】：每個主要論點必須有反向論證
  → 缺失條件：只說 X 為真，沒有說明什麼情況下 X 不成立
  → 缺口類型：批判性視角缺口

規則 2【條件性結論】：每個結論必須有明確前提
  → 缺失條件：結論沒有成立條件，或用「因此」「所以」跳過了假設
  → 缺口類型：條件性結論缺口

規則 3【方法論透明度】：每個數據來源必須說明方法論狀態
  → 缺失條件 A：引用指數/評分/預測，但未說明計算方式是否公開
  → 缺失條件 B：來源為「內部模型」「專有框架」「第三方評估」但未說明其驗證依據
  → 追問方向：此框架是否公開可驗證？是否有獨立第三方審計？
  → 缺口類型：方法論透明度缺口

規則 4【反向解讀】：情緒/預測類指標必須有雙向解讀
  → 觸發詞：情緒指數、市場信心、預測模型、評分、看多、看空
  → 缺失條件：只有單一方向解讀
  → 缺口類型：反向解讀缺口

規則 5【論證邏輯有效性】：每個結論的推理鏈必須能從前提合法推導
  → 缺失條件 A：前提到結論存在邏輯跳躍（中間步驟省略，例如 A→C 但沒說明 B）
  → 缺失條件 B：使用無效推理形式（肯定後件、否定前件、循環論證、虛假兩難、滑坡謬誤、訴諸權威/人身、以偏概全、相關當因果）
  → 缺失條件 C：報告內部自相矛盾（前段陳述 X、後段陳述非 X；或同一指標前後給出不一致的數字、方向、結論）
  → 追問方向：這個推理是否經得起反例？前後段的主張是否互相支援還是互相打架？
  → 缺口類型：論證邏輯缺口
  → 影響等級提示：規則 5 的缺口通常為**高影響**（邏輯不成立 → 結論方向可能反轉）

## 輸出格式（嚴格遵守）

### 偵測到的缺口
1. [缺口類型]：[具體描述，指出是哪個論點/段落]
   → 影響：[若缺口不補，結論可信度如何受影響]

### 記錄的隱含假設
1. 假設「[X]」等同於「[Y]」（位置：[對應段落摘要]）

### 讀者應追問的問題
1. [具體問題，對應特定缺口]

### 整體評分：X/10
評分邏輯：0 個缺口 → 8–10｜1–2 個中影響缺口 → 6–7｜3+ 個或任一高影響缺口 → 1–5
高影響 = 缺口會使結論方向反轉；中影響 = 降低信心但方向不變

重要：「整體評分：X/10」格式不可更改，X 必須是 1–10 整數。

## 規則
- 每個缺口必須指到具體論點，不能只說「整體缺乏...」
- 若所有規則通過，說「所有驗證規則通過」並給 8–10 分
- 語言：正體中文（台灣用語）
"""

# ── 來源方法論審查 Prompt ────────────────────────────────────────────────────
_SOURCE_FRAMEWORK_PROMPT = """\
你是一位方法論審查專員。
任務：評估使用者描述的資料來源或框架，判斷其方法論透明度，並提出應追問的問題。

請從以下三個角度分析：
1. **框架性質**：此來源是否使用了專有模型、未公開框架、或第三方黑箱指數？
2. **可驗證性**：計算方式是否公開？是否有獨立第三方審計或學術驗證？
3. **使用建議**：引用此來源時，讀者應如何標注其限制？

輸出格式：
**框架性質**：[說明]
**可驗證性**：[說明，標注：公開可驗證 / 部分公開 / 無法驗證]
**應追問的問題**：
1. [問題一]
2. [問題二]
**引用建議**：[引用時應加注的限制說明]

語言：正體中文（台灣用語）
"""


# ── 資料結構 ──────────────────────────────────────────────────────────────────
@dataclass
class CriticResult:
    score: int = 10
    raw_output: str = ""
    passed: bool = True


# ── 核心函式 ──────────────────────────────────────────────────────────────────
def run_critic_pipeline(report_text: str, model: str = "gpt-5.5") -> CriticResult:
    """五維度批判分析，回傳結構化結果。使用 Responses API。"""
    from openai import OpenAI
    client = OpenAI()  # key 由 os.environ["OPENAI_API_KEY"] 提供

    response = client.responses.create(
        model=model,
        instructions=_CRITIC_PROMPT,
        input=report_text,
        reasoning={"effort": "medium"},
        text={"verbosity": "low"},
    )
    raw = response.output_text or ""
    score = _parse_score(raw)
    passed = score >= 8 or bool(re.search(r"所有驗證規則通過\s*$", raw, re.MULTILINE))
    return CriticResult(score=score, raw_output=raw, passed=passed)


def run_finance_critic_pipeline(report_text: str, model: str = "gpt-5.5") -> CriticResult:
    """五維度批判分析 + 金融領域三條補充規則（規則 6/7/8）。
    適用於涉及估值、財務報表、賣方研究、總經指標的報告。"""
    from openai import OpenAI
    client = OpenAI()

    combined_instructions = _CRITIC_PROMPT + FINANCE_CRITIC_ADDENDUM
    response = client.responses.create(
        model=model,
        instructions=combined_instructions,
        input=report_text,
        reasoning={"effort": "medium"},
        text={"verbosity": "low"},
    )
    raw = response.output_text or ""
    score = _parse_score(raw)
    passed = score >= 8 or bool(re.search(r"所有驗證規則通過\s*$", raw, re.MULTILINE))
    return CriticResult(score=score, raw_output=raw, passed=passed)


def check_source_framework(source_description: str, model: str = "gpt-5.5") -> str:
    """針對單一來源或框架的方法論透明度審查，回傳評估文字。"""
    from openai import OpenAI
    client = OpenAI()

    response = client.responses.create(
        model=model,
        instructions=_SOURCE_FRAMEWORK_PROMPT,
        input=source_description,
        reasoning={"effort": "low"},
        text={"verbosity": "low"},
    )
    return response.output_text or ""


def format_critic_output(result: CriticResult) -> str:
    """格式化批判結果。通過（score ≥ 8）回傳空字串，避免污染輸出。"""
    if result.passed:
        return ""
    return result.raw_output


# ── 內部工具 ──────────────────────────────────────────────────────────────────
def _parse_score(text: str) -> int:
    """從 LLM 輸出中解析「整體評分：X/10」，找不到時預設 5。"""
    m = re.search(r"整體評分：(\d+)/10", text)
    if m:
        return max(1, min(10, int(m.group(1))))
    return 5
