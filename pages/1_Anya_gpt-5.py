import streamlit as st
import base64
import time
from io import BytesIO
from PIL import Image
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.callbacks.base import BaseCallbackHandler

# === 1. 設定 Streamlit 頁面 ===
st.set_page_config(page_title="Anya Multimodal Agent", page_icon="🥜", layout="wide")

# === 2. Session State ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_ai" not in st.session_state:
    st.session_state.pending_ai = False
if "pending_content" not in st.session_state:
    st.session_state.pending_content = None

# === 3. 定義 DuckDuckGo 搜尋 Tool ===
@tool
def ddgs_search(query: str) -> str:
    """DuckDuckGo 搜尋（同時查詢網頁與新聞，回傳 markdown 條列格式並附來源）。"""
    try:
        from duckduckgo_search import DDGS
        ddgs = DDGS()
        web_results = ddgs.text(query, region="wt-wt", safesearch="moderate", max_results=5)
        news_results = ddgs.news(query, region="wt-wt", safesearch="moderate", max_results=5)
        all_results = []
        if isinstance(web_results, list):
            all_results.extend(web_results)
        if isinstance(news_results, list):
            all_results.extend(news_results)
        docs = []
        sources = []
        for item in all_results:
            title = item.get("title", "無標題")
            link = item.get("href", "") or item.get("link", "") or item.get("url", "")
            snippet = item.get("body", "") or item.get("snippet", "")
            docs.append(f"- [{title}]({link})\n  > {snippet}")
            if link:
                sources.append(link)
        if not docs:
            return "No results found."
        markdown_content = "\n".join(docs)
        source_block = "\n\n## 來源\n" + "\n".join(sources)
        return markdown_content + source_block
    except Exception as e:
        return f"Error from DuckDuckGo: {e}"

tools = [ddgs_search]

# === 4. 定義系統提示與 LLM模型 ===
ANYA_SYSTEM_PROMPT = """你是安妮亞（Anya Forger），《間諜家家酒》的天真可愛小女孩，回覆風格要熱情、可愛並用正體中文，適時加上emoji。"""

llm = ChatOpenAI(
    model="gpt-5",
    temperature=0,
    openai_api_key=st.secrets["OPENAI_KEY"],
).bind_tools(tools)

# === 5. 打字機特效 Callback ===
def get_streamlit_cb(parent_container, status=None):
    class StreamHandler(BaseCallbackHandler):
        def __init__(self, container, status=None):
            self.container = container
            self.status = status
            self.token_placeholder = self.container.empty()
            self.tokens = []
            self.cursor_symbol = "｜"
            self.cursor_visible = True

        @property
        def text(self):
            return ''.join(self.tokens)

        def on_llm_new_token(self, token: str, **kwargs):
            self.tokens.append(token)
            # 切換 cursor
            self.cursor_visible = not self.cursor_visible
            cursor = self.cursor_symbol if self.cursor_visible else ""
            show_text = ''.join(self.tokens) + cursor
            self.token_placeholder.markdown(show_text)
            time.sleep(0.025)

        def on_llm_end(self, response, **kwargs):
            self.token_placeholder.markdown(self.text, unsafe_allow_html=True)

        def on_tool_start(self, serialized, input_str, **kwargs):
            if self.status:
                tool_name = serialized.get("name", "")
                # 工具即時 spinner 狀態提示
                self.status.update(label=f"安妮亞正在搜尋網路...🔍", state="running") if "ddgs_search" in tool_name \
                    else self.status.update(label="安妮亞正在用工具...", state="running")

        def on_tool_end(self, output, **kwargs):
            if self.status:
                self.status.update(label="工具查詢完成！✨", state="complete")

    return StreamHandler(parent_container, status)

# === 6. 聊天歷史呈現 ===
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        with st.chat_message("user"):
            if msg.get("text"):
                st.markdown(msg["text"])
            if msg.get("images"):
                for fn, imgbytes in msg["images"]:
                    st.image(Image.open(BytesIO(imgbytes)), caption=fn, width=220)
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            if msg.get("text"):
                st.markdown(msg["text"])

# === 7. 等待 AI 回覆時（處理所有歷史！）===
if st.session_state.pending_ai and st.session_state.pending_content:
    messages_blocks = []
    for item in st.session_state.chat_history:
        blocks = []
        if item.get("text"):
            blocks.append({"type": "text", "text": item["text"]})
        if item.get("images"):
            for _, imgbytes in item["images"]:
                b64 = base64.b64encode(imgbytes).decode()
                blocks.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        if blocks:
            messages_blocks.append({"role": item["role"], "content": blocks})
    # 加這輪 user 輸入
    messages_blocks.append({"role": "user", "content": st.session_state.pending_content})

    # spinner/狀態條callback
    with st.chat_message("assistant"):
        status = st.status("安妮亞馬上回覆你！", expanded=False)
        with st.spinner("Wait for it...", show_time=True):
            
            cb = get_streamlit_cb(st.container(), status=status)

            # 系統role
            sys_msg = {"role": "system", "content": ANYA_SYSTEM_PROMPT}
            response = llm.invoke([sys_msg] + messages_blocks, config={"callbacks": [cb]})

            ai_text = response.content if hasattr(response, "content") else str(response)
            # 儲存進歷史
            st.session_state.chat_history.append({
                "role": "assistant",
                "text": ai_text,
                "images": []
            })
            st.session_state.pending_ai = False
            st.session_state.pending_content = None
            status.update(label="安妮亞回答完畢！🥜", state="complete")
            st.rerun()

# === 8. 使用者輸入框，支援文字+多圖 ===
prompt = st.chat_input("wakuwaku！你要跟安妮亞分享什麼嗎？", accept_file="multiple", file_type=["jpg", "jpeg", "png"])
if prompt:
    user_text = prompt.text.strip() if prompt.text else ""
    images_for_history = []
    content_blocks = []

    if user_text:
        content_blocks.append({"type": "text", "text": user_text})
    for f in prompt.files:
        imgbytes = f.getbuffer()
        mime = f.type
        b64 = base64.b64encode(imgbytes).decode()
        content_blocks.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"}
        })
        images_for_history.append((f.name, imgbytes))

    # 立刻 append 入歷史（UI立刻見）
    st.session_state.chat_history.append({
        "role": "user",
        "text": user_text,
        "images": images_for_history
    })
    st.session_state.pending_ai = True
    st.session_state.pending_content = content_blocks
    st.rerun()
