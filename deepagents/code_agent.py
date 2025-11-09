# Anya/deepagents/code_agent.py
from __future__ import annotations

from agents import Agent

CODE_PROMPT = """
你是程式助理。對於描述與參數：
- 產出可執行的程式碼片段（或推理步驟）
- 若有 tests 需求，說明如何執行並預期結果
- 回傳簡短結論與重點說明
"""

code_agent = Agent(
    name="CodeAgent",
    instructions=CODE_PROMPT,
    model="gpt-5-codex",
)
