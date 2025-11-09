# Anya/deepagents/research_agent.py
from __future__ import annotations

from agents import Agent, ModelSettings, function_tool

RESEARCH_PROMPT = """
你是研究助理。收到研究子題與參數後：
- 呼叫 web_search 或相應工具取得至少 2 個獨立來源
- 彙整成 200-300 字摘要，附上來源列表（標題+URL）
- 僅輸出重點，避免冗長
"""

@function_tool
def web_search(query: str) -> str:
    """Search the web and return a short summary (placeholder)."""
    # 注意：此為範例占位。實務請整合真正的搜尋工具。
    return f"[MOCK] Results for: {query}\n- Source A: https://example.com/a\n- Source B: https://example.com/b"

research_agent = Agent(
    name="ResearchAgent",
    instructions=RESEARCH_PROMPT,
    model="gpt-4.1",
    tools=[web_search],
    model_settings=ModelSettings(tool_choice="required"),
)
