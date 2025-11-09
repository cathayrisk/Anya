# Anya/deepagents/planner_agent.py
from __future__ import annotations

from agents import Agent
from .models import Plan

PLANNER_PROMPT = """
你是一位規劃助理，請基於使用者目標與分流結果，產生一份 A+ 版任務計畫（Plan）：
- 以清單 steps 形式輸出，維持簡單順序，不需要完整 DAG
- 允許標註 is_parallel 與 parallel_group：同一 parallel_group 的步驟可以併發執行
- 每個步驟若需要工具，填 requires_tool=true、tool_name 與 parameters
- 為研究步驟提供可機器檢查的 acceptance_criteria（例如 {"type":"research","min_sources":2,"require_consistency":true}）
- 為程式步驟提供驗收（例如 {"type":"code","tests":["test_a","test_b"]}）
- 為資料處理步驟提供驗收（例如 {"type":"data","schema":{"fields":[...]}}）
- metadata 中填入 goal、估計時間/成本（大略即可）
- 步驟數量 6-10 步為宜；蒐集/查詢型步驟可併行，整合/推論/寫作步驟維持序列
輸出請使用 Plan（steps, metadata）。
"""

planner_agent = Agent(
    name="PlannerAgent",
    instructions=PLANNER_PROMPT,
    model="o3-mini",
    output_type=Plan,
)
