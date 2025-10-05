import streamlit as st
from PIL import Image
import base64
import io
from datetime import datetime
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
import inspect
from typing import Callable, TypeVar
import time

# ==== Streamlit 基本設定、state ====
st.set_page_config(page_title="Anya", layout="wide", page_icon="🥜", initial_sidebar_state="collapsed")

if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="嗨嗨～安妮亞來了！👋 有什麼想問安妮亞的嗎？")]
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gpt-4.1"
if "current_model" not in st.session_state:
    st.session_state.current_model = None
if "llm" not in st.session_state:
    st.session_state.llm = None

# ==== OpenAI 物件 ====
client = OpenAI(api_key=st.secrets["OPENAI_KEY"])

# ==== 前處理工具：統一圖片格式 & base64 ====
def process_upload_file(file):
    file.seek(0)
    img_bytes = file.read()
    if not img_bytes or len(img_bytes) < 32:
        return None
    try:
        img = Image.open(io.BytesIO(img_bytes))
        fmt = img.format.lower()
        mime = f"image/{fmt}"
        if fmt not in ["png","jpeg","jpg","webp","gif"]:
            return None
        b64 = base64.b64encode(img_bytes).decode()
        return {"bytes": img_bytes, "file_name": file.name, "fmt": fmt, "mime": mime, "b64": b64}
    except Exception:
        return None

# ==== OCR工具範例，可複製一份再寫其他多圖tool ====
@tool
def image_ocr_tool(image_bytes: bytes, file_name: str = "uploaded_file.png") -> str:
    """AI OCR圖片識別工具，輸入圖片bytes與檔名，回傳圖中文字結果。"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        fmt = img.format.lower()
        mime = f"image/{fmt}"
    except Exception as e:
        return f"解析圖片失敗：{e}"

    img_url = f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": "You are an OCR-like data extraction tool that extracts text from images."},
                {"role": "user", "content": [
                    {"type": "input_text", "text":
                        "Please extract all visible text from the image, including any small print or footnotes. "
                        "Ensure no text is omitted, and provide a verbatim transcription of the document. "
                        "Format your answer in Markdown (no code block or triple backticks). "
                        "Do not add any explanations or commentary."
                    },
                    {"type": "input_image", "image_url": img_url, "detail": "high"}
                ]}
            ],
            timeout=45
        )
        return f"---\nfile_name: {file_name}\n---\n{response.output_text.strip()}\n"
    except Exception as e:
        return f"OCR失敗，請檢查API/圖片格式：{e}"

# ==== 你其他的工具，例如ddgs_search、wiki_tool都可以這樣註冊 ====
@tool
def echo_tool(text:str) -> str:
    return f"你輸入的內容是：{text}"

tools = [image_ocr_tool, echo_tool] # <-- 你需要的所有@tool都可以加進來

# ==== LangGraph Agent 設定 ====
ANYA_SYSTEM_PROMPT = """你是安妮亞，請根據用戶給的純文字和/或圖片提出適當的回答，可以自動判斷需呼叫OCR或進行其他回答。"""

llm = st.session_state.llm or ChatOpenAI(
    model=st.session_state.selected_model,
    openai_api_key=st.secrets["OPENAI_KEY"],
    temperature=0.0,
    streaming=True,
)
llm_with_tools = llm.bind_tools(tools)

def call_model(state: MessagesState):
    messages = state["messages"]
    sys_msg = SystemMessage(content=ANYA_SYSTEM_PROMPT)
    response = llm_with_tools.invoke([sys_msg] + messages)
    return {"messages": messages + [response]}

tool_node = ToolNode(tools)
def call_tools(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

workflow = StateGraph(MessagesState)
workflow.add_node("LLM", call_model)
workflow.add_edge(START, "LLM")
workflow.add_node("tools", tool_node)
workflow.add_conditional_edges("LLM", call_tools)
workflow.add_edge("tools", "LLM")
agent = workflow.compile()

# ==== 美美地顯示歷史 ====
for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        # 處理content型態，有多圖的話也一樣順
        if isinstance(msg.content, str):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg.content, list):
            with st.chat_message("user"):
                for block in msg.content:
                    if block.get("type") == "text":
                        st.write(block["text"])
                    elif block.get("type") == "image_url":
                        info = block["image_url"]
                        st.image(info["url"], caption=info.get("file_name", ""), width=220)

# ==== 輸入區：文字輸入 + 支援多圖輸入 ====
user_prompt = st.chat_input(
    "wakuwaku！安妮亞可以幫你看圖說故事嚕！",
    accept_file="multiple",
    file_type=["jpg", "jpeg", "png"]
)

if user_prompt:
    # 1. 組 content_blocks
    content_blocks = []
    user_text = user_prompt.text.strip() if user_prompt.text else ""
    if user_text:
        content_blocks.append({"type": "text", "text": user_text})

    images_for_history = []
    if hasattr(user_prompt, "files"):
        for f in user_prompt.files:
            asset = process_upload_file(f)
            if asset:
                dataurl = f"data:{asset['mime']};base64,{asset['b64']}"
                content_blocks.append({"type": "image_url", "image_url": {
                    "url": dataurl, "file_name": asset["file_name"]
                }})
                images_for_history.append((asset["file_name"], asset["bytes"])) # 方便顯示縮圖
            else:
                st.warning(f"{getattr(f,'name','檔案')} 格式不支援或內容異常～")

    # 2. append到messages
    if content_blocks:
        st.session_state.messages.append(HumanMessage(content=content_blocks))
        # UI顯示
        with st.chat_message("user"):
            for block in content_blocks:
                if block.get("type") == "text":
                    st.write(block["text"])
                elif block.get("type") == "image_url":
                    info = block["image_url"]
                    st.image(info["url"], caption=info.get("file_name", ""), width=220)

    # 3. murmur & agent運作
    all_text = []
    for msg in st.session_state.messages:
        if hasattr(msg, "content"):
            if isinstance(msg.content, str):
                all_text.append(msg.content)
            elif isinstance(msg.content, list):
                for part in msg.content:
                    if part.get("type") == "text":
                        all_text.append(part["text"])
    all_text = "\n".join(all_text)

    status_prompt = f"""你是安妮亞，請根據聊天紀錄自言自語一句可愛 murmur（15字內）。{all_text}"""
    status_response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": status_prompt}]
    )
    status_label = status_response.choices[0].message.content.strip()

    with st.chat_message("assistant"):
        status = st.status(status_label)
        # 如果你有 get_streamlit_cb 可以加進agent回呼（這裡可略過）
        response = agent.invoke({"messages": st.session_state.messages})
        ai_msg = response["messages"][-1]
        st.session_state.messages.append(ai_msg)
        status.update(label="安妮亞回答完畢！🎉", state="complete")
