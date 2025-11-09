# Anya/deepagents/writer_agent.py
from __future__ import annotations

from agents import Agent

WRITER_PROMPT = """
你是彙整與寫作助理。請根據所有已完成步驟的輸出：
- 產生最終答案（清楚、有結構）
- 附上引用/來源列表與關鍵依據
- 註記已知限制與後續建議
"""

writer_agent = Agent(
    name="WriterAgent",
    instructions=WRITER_PROMPT,
    model="gpt-4.1",
)
