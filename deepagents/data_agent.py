# Anya/deepagents/data_agent.py
from __future__ import annotations

import json
from typing import Literal
from agents import Agent, ModelSettings, function_tool

DATA_PROMPT = """
你是資料處理助理。依指示對資料進行轉換/清洗/摘要，並確認輸出符合指定 schema/格式。
"""

@function_tool
def data_transform(rows_json: str, op: Literal["preview", "count"] = "preview") -> str:
    """
    Perform a simple data transform.
    - rows_json: JSON 字串（例如 [{"id":1,"v":10}, ...]）
    - op: "preview" | "count"
    回傳 JSON 字串：{"status":"ok","rows":N,"preview":[...]}
    """
    try:
        rows = json.loads(rows_json) if rows_json else []
        if not isinstance(rows, list):
            return json.dumps({"status": "err", "error": "rows_json must be a JSON array"})
    except Exception as e:
        return json.dumps({"status": "err", "error": f"invalid JSON: {e}"})

    if op == "count":
        return json.dumps({"status": "ok", "rows": len(rows), "preview": []}, ensure_ascii=False)

    # default: preview
    return json.dumps({"status": "ok", "rows": len(rows), "preview": rows[:2]}, ensure_ascii=False)

data_agent = Agent(
    name="DataAgent",
    instructions=DATA_PROMPT,
    model="gpt-4.1",
    tools=[data_transform],
    model_settings=ModelSettings(tool_choice="auto"),
)
