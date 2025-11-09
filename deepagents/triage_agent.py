# Anya/deepagents/triage_agent.py
from __future__ import annotations

from agents import Agent
from .models import TriageDecision

TRIAGE_PROMPT = """
你是一位任務分流專家。請根據使用者的目標，產生結構化的分類與策略建議：
- category：從 ["research","code","data","other"] 中選一個最貼切的大類
- complexity：從 ["low","medium","high"] 中評估任務複雜度
- approach：根據複雜度建議規劃策略 ["cot","subgoals","self_critique"]
- recommended_tools：列出建議的工具名稱（例如 web_search, code_run, data_transform）
- notes：必要補充（例如資料取得風險、需要人審的地方）
輸出請使用結構化（TriageDecision）。
"""

triage_agent = Agent(
    name="TriageAgent",
    instructions=TRIAGE_PROMPT,
    model="gpt-4.1",
    output_type=TriageDecision,
)
