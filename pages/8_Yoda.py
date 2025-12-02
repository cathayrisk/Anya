import streamlit as st
import asyncio
import os
import time
import uuid

# 匯入你之前那支多 Agent + Kerykeion 的檔案
from yoda.companion_fortune_agent_yoda_kerykeion import chat_once

# 從 Streamlit secrets 讀取 API key
# 在 .streamlit/secrets.toml 裡面放：
# OPENAI_KEY = "sk-xxxx"
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_KEY"]


# ==============================
# 小工具：同步執行 async 函式
# ==============================
def run_async(coro):
    """在非 async 環境下執行協程。"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if loop.is_running():
        # 如果之後你改成 st.experimental_async，可另外處理
        return coro
    return loop.run_until_complete(coro)


# ==============================
# 打字動畫效果（沿用你的範例）
# ==============================
def emoji_token_stream(full_text, emoji="🌸", cursor_symbol=" "):
    placeholder = st.empty()
    tokens = []
    cursor_visible = True

    for idx, token in enumerate(full_text):
        tokens.append(token)
        cursor_visible = not cursor_visible
        cursor = cursor_symbol if cursor_visible else " "
        safe_text = ''.join(tokens[:-1])
        # 1. 先用 emoji 顯示新字
        placeholder.markdown(safe_text + emoji + cursor)
        time.sleep(0.03)
        # 2. 再換成正常字
        placeholder.markdown(''.join(tokens) + cursor)
        time.sleep(0.01)
    # 最後顯示完整內容（不顯示游標）
    placeholder.markdown(''.join(tokens))


# ==============================
# Streamlit 頁面設定
# ==============================
st.set_page_config(
    page_title="尤達陪伴占星聊天",
    layout="wide",
    page_icon="🧙‍♂️",
)

st.title("🧙‍♂️ 尤達陪伴占星聊天")
st.write(
    "這是一個會用星座、命盤幫你更了解自己，又用尤達大師風格溫柔陪你聊天的 AI 夥伴。\n\n"
    "可以跟他聊心情、壓力、關係，也可以分享你的生日，讓他用命盤多認識你一點。"
)

# ==============================
# Session 狀態初始化
# ==============================

# 每個瀏覽器 session 一個固定 user_id，方便後端記憶你的資料
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

# 對話歷史
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "嗯，來到這裡，你是。\n\n想聊什麼，今天？心情，壓力，或是你的星星命盤，說說看吧。",
            "avatar": "🧙‍♂️",
        }
    ]

# ==============================
# 顯示歷史訊息
# ==============================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=msg.get("avatar")):
        st.markdown(msg["content"])

# ==============================
# 輸入框
# ==============================
user_input = st.chat_input("想跟尤達說什麼？可以聊心情、生活、或告訴他你的生日與出生地。")

if user_input:
    # 顯示使用者訊息
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "avatar": "🧑",
    })
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    # AI 回覆
    with st.chat_message("assistant", avatar="🧙‍♂️"):
        with st.spinner("尤達正在思考你的星星與心情..."):
            # 呼叫我們之前實作的 chat_once（多 Agent + 命盤 + 尤達人格）
            reply_text = run_async(chat_once(st.session_state.user_id, user_input))
            # 打字動畫
            emoji_token_stream(reply_text, emoji="🌟")

        # 存入歷史
        st.session_state.messages.append({
            "role": "assistant",
            "content": reply_text,
            "avatar": "🧙‍♂️",
        })
