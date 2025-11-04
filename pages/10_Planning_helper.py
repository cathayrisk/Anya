# pages/10_Planning_helper.py
from pathlib import Path
import sys

# 讓 Python 可以從專案根目錄匯入（…/anya）
ROOT = Path(__file__).resolve().parents[1]  # parents[1] = /mount/src/anya
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 先試小寫的 workflows.py
try:
    from agents.workflows import run_workflow, WorkflowInput
except ModuleNotFoundError:
    # 若你的檔名其實是 Workflow.py（大寫W）
    try:
        from agents.Workflow import run_workflow, WorkflowInput
    except ModuleNotFoundError as e:
        # 顯示提示，方便你在UI看到
        import streamlit as st, pkgutil
        st.error(":red[無法載入 agents.workflows / agents.Workflow]")
        st.caption(f"sys.path 前3項：{sys.path[:3]}")
        st.caption(f"agents 可見性：{bool(pkgutil.find_loader('agents'))}")
        raise e

# app.py
import asyncio
import streamlit as st

# 你前面貼的程式：請確保 run_workflow 與 WorkflowInput 可被匯入
from Agents.workflows import run_workflow, WorkflowInput
from typing import List, Dict, Any

# ===== 若 run_workflow 是 async，包成 sync 呼叫 =====
def run_workflow_sync(text: str) -> Dict[str, Any]:
    from Agents.workflows import run_workflow, WorkflowInput  # TODO: 修改成你的實際路徑
    return asyncio.run(run_workflow(WorkflowInput(input_as_text=text)))

# ===== 小工具：把歷史訊息變成單一字串給 triage 看 =====
def transcript_from_messages(msgs: List[Dict[str, str]]) -> str:
    lines = []
    for m in msgs:
        speaker = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {m['content']}")
    return "\n".join(lines)

# ===== 初始化狀態 =====
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_KEY"]

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "嗨嗨～請描述你的專案目標、時間範圍與目前資源，安妮亞來幫你規劃！🥜"}
    ]

st.set_page_config(page_title="Initiative Planner • Streamlit", page_icon=":material_rocket:")

st.title(":material_rocket: Initiative Planner")
st.caption("用 chat 方式規劃你的專案目標、時程、資源。Powered by Streamlit chat elements.")

# ===== 顯示歷史訊息 =====
for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="🧑‍💻" if m["role"] == "user" else "🧠"):
        st.markdown(m["content"])

# ===== Chat input（可先用最簡單：不收檔案）=====
prompt = st.chat_input(
    "請輸入：專案目標 / 預計完成時間 / 可用資源（人數、預算、工具）",
    max_chars=2000,
    key="chat_input_main",
    width="stretch"
)

if prompt:
    # 1) 顯示使用者訊息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # 2) 呼叫你的 workflow（用整段對話當輸入，讓 triage 能理解上下文）
    full_text = transcript_from_messages(st.session_state.messages)
    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("安妮亞努力規劃中…(滴答滴答)"):
            try:
                out = run_workflow_sync(full_text)
            except Exception as e:
                reply = f":red[抱歉，後端流程發生錯誤]：{e}"
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
            else:
                # 3) 根據分支結果輸出
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

                # 顯示助理訊息並存入歷史
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
