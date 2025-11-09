# Anya/deepagents/data_agent.py
from __future__ import annotations

from typing import Any, Dict
from agents import Agent, ModelSettings, function_tool

DATA_PROMPT = """
你是資料處理助理。依指示對資料進行轉換/清洗/摘要，並確認輸出符合指定 schema/格式。
"""

@function_tool
def data_transform(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Perform a mock data transform and return structured output."""
    return {"status": "ok", "rows": len(payload.get("rows", [])), "preview": payload.get("rows", [])[:2]}

data_agent = Agent(
    name="DataAgent",
    instructions=DATA_PROMPT,
    model="gpt-4.1",
    tools=[data_transform],
    model_settings=ModelSettings(tool_choice="auto"),
)
