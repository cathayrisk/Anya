# pages/10_Planning_helper.py
import streamlit as st
import asyncio
import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError

# ===== 初始化狀態 =====
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_KEY"]

# ===== 相依：基礎 agents 套件（提供 Agent/Runner/WebSearchTool）=====
try:
    from agents import (
        WebSearchTool, Agent, ModelSettings,
        TResponseInputItem, Runner, RunConfig, trace
    )
except Exception as e:
    st.set_page_config(page_title="Initiative Planner")
    st.error(":red[無法載入基礎 agents 套件]，請確認環境有提供 Agent/Runner/WebSearchTool。")
    st.stop()

try:
    from openai.types.shared.reasoning import Reasoning
    HAS_REASONING = True
except Exception:
    HAS_REASONING = False
    class Reasoning:  # 型別占位
        def __init__(self, *args, **kwargs): ...

# ===== 頁面設定 =====
st.set_page_config(page_title="Initiative Planner", page_icon=":material_rocket:")
st.title(":material_rocket: Initiative Planner")
st.caption("用聊天方式規劃專案目標、時程、資源。這版沒有 sidebar，介面更清爽。")

# ===== Chat 狀態 =====
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "嗨嗨～請描述你的專案目標、目標時程（日期或期間），以及手上資源（人數/預算/工具），安妮亞來幫你規劃！🥜"}
    ]

# 主畫面開關（非 sidebar）
STORE_DEFAULT = True
store_opt = st.toggle("儲存對話到模型（store）", value=STORE_DEFAULT, help="若擔心隱私可關閉。")

# ===== Helper：把歷史訊息組成 transcript =====
def transcript_from_messages(msgs: List[Dict[str, str]]) -> str:
    lines = []
    for m in msgs:
        who = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{who}: {m['content']}")
    return "\n".join(lines)

# ====== Schema ======
class TriageSchema(BaseModel):
    has_all_details: bool = Field(description="Whether all required fields are present")
    initiative_goal: Optional[str] = Field(default=None, description="User-provided goal")
    target_timeframe: Optional[str] = Field(default=None, description="User-provided date or period")
    current_resources: Optional[str] = Field(default=None, description="User-provided resources")

# ====== 建立工具與 Agents（全寫在本檔案） ======
def build_agents(store_flag: bool):
    web_search_preview = WebSearchTool(
        search_context_size="medium",
        user_location={"type": "approximate"}
    )

    triage_instructions = """You are an assistant that gathers the key details needed to create a business initiative plan.

Look through the conversation to extract the following:
1. Initiative goal (what the team or organization aims to achieve)
2. Target completion date or timeframe
3. Available resources or current capacity (e.g., headcount, budget, or tool access)

If all three details are present anywhere in the conversation, return:
{
  "has_all_details": true,
  "initiative_goal": "<user-provided goal>",
  "target_timeframe": "<user-provided date or period>",
  "current_resources": "<user-provided resources>"
}
If one or more are missing, return:
{
  "has_all_details": false,
  "initiative_goal": "<goal if known or null>",
  "target_timeframe": "<timeframe if known or null>",
  "current_resources": "<resources if known or null>"
}"""

    triage_settings = ModelSettings(store=store_flag)
    if HAS_REASONING:
        triage_settings.reasoning = Reasoning(effort="minimal", summary="auto")

    triage = Agent(
        name="Triage",
        instructions=triage_instructions,
        model="gpt-5",
        output_type=TriageSchema,
        model_settings=triage_settings
    )

    launch_helper = Agent(
        name="Launch helper",
        instructions="""Come up with a tailored plan to help the user run a new business initiative.
Consider all the details they've provided and offer a succinct, bullet point list for how to run the initiative.

Use the web search tool to get additional context and synthesize a succinct answer that clearly explains how to run the project,
identifying unique opportunities, highlighting risks and laying out mitigations that make sense.""",
        model="gpt-4.1-mini",
        tools=[web_search_preview],
        model_settings=ModelSettings(
            temperature=1,
            top_p=1,
            max_tokens=2048,
            store=store_flag
        )
    )

    get_data = Agent(
        name="Get data",
        instructions="""Collect the missing data from the user.

Look through the conversation to extract the following:
1. Initiative goal (what the team or organization aims to achieve)
2. Target completion date or timeframe
3. Available resources or current capacity (e.g., headcount, budget, or tool access)

Ask concise, direct questions to obtain whatever is missing.""",
        model="gpt-5",
        model_settings=ModelSettings(
            store=store_flag,
            reasoning=Reasoning(effort="minimal", summary="auto") if HAS_REASONING else None
        )
    )

    return triage, launch_helper, get_data

# ====== Workflow 主流程（內嵌） ======
class WorkflowInput(BaseModel):
    input_as_text: str

async def run_workflow(workflow_input: WorkflowInput, store_flag: bool) -> Dict[str, Any]:
    triage, launch_helper, get_data = build_agents(store_flag)

    with trace("Agent builder workflow"):
        workflow = workflow_input.model_dump()
        conversation_history: List[TResponseInputItem] = [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": workflow["input_as_text"]}]
            }
        ]

        # 1) TRIAGE
        try:
            triage_result_temp = await Runner.run(
                triage,
                input=[*conversation_history],
                run_config=RunConfig(trace_metadata={"__trace_source__": "agent-builder"})
            )
        except Exception as e:
            return {"ok": False, "stage": "triage", "error": f"Triage agent failed: {e}"}

        conversation_history.extend([item.to_input_item() for item in triage_result_temp.new_items])

        # 解析與驗證
        try:
            triage_output: TriageSchema = triage_result_temp.final_output
            triage_parsed = triage_output.model_dump()
        except ValidationError as ve:
            return {"ok": False, "stage": "triage-parse", "error": f"Triage output validation failed: {ve}"}

        result_blob: Dict[str, Any] = {
            "ok": True,
            "triage": {
                "output_text": triage_output.model_dump_json(),
                "output_parsed": triage_parsed
            }
        }

        # 2) 分支
        if triage_parsed.get("has_all_details", False):
            try:
                launch_helper_result_temp = await Runner.run(
                    launch_helper,
                    input=[*conversation_history],
                    run_config=RunConfig(trace_metadata={"__trace_source__": "agent-builder"})
                )
                conversation_history.extend([item.to_input_item() for item in launch_helper_result_temp.new_items])
                result_blob.update({
                    "result_type": "launch_helper",
                    "result": {"output_text": launch_helper_result_temp.final_output_as(str)}
                })
            except Exception as e:
                result_blob.update({"ok": False, "stage": "launch_helper", "error": f"Launch helper failed: {e}"})
        else:
            try:
                get_data_result_temp = await Runner.run(
                    get_data,
                    input=[*conversation_history],
                    run_config=RunConfig(trace_metadata={"__trace_source__": "agent-builder"})
                )
                conversation_history.extend([item.to_input_item() for item in get_data_result_temp.new_items])
                result_blob.update({
                    "result_type": "get_data",
                    "result": {"output_text": get_data_result_temp.final_output_as(str)}
                })
            except Exception as e:
                result_blob.update({"ok": False, "stage": "get_data", "error": f"Get data failed: {e}"})

        result_blob["history"] = conversation_history
        return result_blob

# 同步包裝
def run_workflow_sync(text: str, store_flag: bool) -> Dict[str, Any]:
    return asyncio.run(run_workflow(WorkflowInput(input_as_text=text), store_flag))

# ====== 顯示歷史訊息 ======
for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="🧑‍💻" if m["role"] == "user" else "🧠"):
        st.markdown(m["content"])

# ====== Chat Input ======
prompt = st.chat_input(
    "請輸入：專案目標 / 預計完成時間 / 可用資源（人數、預算、工具）",
    max_chars=2000,
    key="chat_input_main",
    width="stretch"
)

if prompt:
    # 使用者訊息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # 整段 transcript 餵給 workflow
    full_text = transcript_from_messages(st.session_state.messages)

    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("安妮亞努力規劃中…(滴答滴答)"):
            out = run_workflow_sync(full_text, store_opt)

        # Triage 摘要（主畫面用 expander 顯示）
        tri = (out.get("triage") or {}).get("output_parsed") or {}
        with st.expander("Triage 摘要（點我展開）", expanded=False):
            st.markdown(f"- has_all_details: {tri.get('has_all_details')}")
            st.markdown(f"- initiative_goal: {tri.get('initiative_goal')}")
            st.markdown(f"- target_timeframe: {tri.get('target_timeframe')}")
            st.markdown(f"- current_resources: {tri.get('current_resources')}")

        # 主回覆
        if not out.get("ok", True):
            reply = f":red[流程失敗於 {out.get('stage','unknown')}]：{out.get('error','(未知錯誤)')}"
        else:
            rtype = out.get("result_type")
            if rtype == "launch_helper":
                reply = out["result"]["output_text"]
            elif rtype == "get_data":
                reply = out["result"]["output_text"]
            else:
                reply = "我已收到資訊，但還需要更多細節才能產出完整方案～可以再補：目標、時間、資源嗎？"

        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
