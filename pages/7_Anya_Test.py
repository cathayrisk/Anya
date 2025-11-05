import streamlit as st
import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI

# === 0. Trimming 參數（可調） ===
TRIM_LAST_N_USER_TURNS = 15        # 建議先收斂一點，更省 token
MAX_STREAM_TIMEOUT_SEC = 60

# === 1. 設定 Streamlit 頁面 ===
st.set_page_config(page_title="Anya Multimodal Agent", page_icon="🥜", layout="wide")

# === 1.1 快取：縮圖 & data URL ===
@st.cache_data(show_spinner=False, max_entries=256)
def make_thumb(imgbytes: bytes, max_w=220) -> bytes:
    im = Image.open(BytesIO(imgbytes))
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    im.thumbnail((max_w, max_w))
    out = BytesIO()
    im.save(out, format="JPEG", quality=80, optimize=True)
    return out.getvalue()

def _detect_mime_from_bytes(img_bytes: bytes) -> str:
    try:
        im = Image.open(BytesIO(img_bytes))
        fmt = (im.format or "").upper()
        if fmt == "PNG":  return "image/png"
        if fmt in ("JPG", "JPEG"): return "image/jpeg"
        if fmt == "WEBP": return "image/webp"
        if fmt == "GIF":  return "image/gif"
    except Exception:
        pass
    return "application/octet-stream"

@st.cache_data(show_spinner=False, max_entries=256)
def bytes_to_data_url(imgbytes: bytes) -> str:
    mime = _detect_mime_from_bytes(imgbytes)
    b64 = base64.b64encode(imgbytes).decode()
    return f"data:{mime};base64,{b64}"

# === 2. Session State ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{
        "role": "assistant",
        "text": "嗨嗨～安妮亞大升級了！👋 有什麼想問安妮亞的嗎？",
        "images": []  # [(name, thumb_bytes, orig_bytes)]
    }]
if "pending_ai" not in st.session_state:
    st.session_state.pending_ai = False
if "pending_content" not in st.session_state:
    st.session_state.pending_content = None

# === 3. OpenAI client ===
client = OpenAI(api_key=st.secrets["OPENAI_KEY"])

# === 4. 系統提示 ===
ANYA_SYSTEM_PROMPT = """
你是安妮亞（Anya Forger），來自《SPY×FAMILY 間諜家家酒》的小女孩。請用正體中文、台灣用語，語氣可愛簡單、活潑有禮；適時加入emoji；必要時條列整理重點；若被要求翻譯則改為正式逐句翻譯。
"""

# === 5. 將 chat_history 修剪成「最近 N 個使用者回合」並轉成 Responses API input ===
def build_trimmed_input_messages(pending_user_content_blocks):
    hist = st.session_state.chat_history
    if not hist:
        return [{"role": "user", "content": pending_user_content_blocks}]

    # 1) 找到最近 N 個「使用者回合」起點
    user_count = 0
    start_idx = 0
    for i in range(len(hist) - 1, -1, -1):
        if hist[i].get("role") == "user":
            user_count += 1
            if user_count == TRIM_LAST_N_USER_TURNS:
                start_idx = i
                break
    selected = hist[start_idx:]

    # 2) 轉 Responses messages：僅保留文字歷史，且只讓「最後一輪使用者回合」帶圖片
    messages = []
    last_user_idx = max([i for i, m in enumerate(selected) if m.get("role") == "user"], default=-1)
    for i, msg in enumerate(selected):
        role = msg.get("role")
        if role == "user":
            blocks = []
            if msg.get("text"):
                blocks.append({"type": "input_text", "text": msg["text"]})
            # 僅最後一輪使用者回合帶圖，降低 payload
            if i == last_user_idx and msg.get("images"):
                for _fn, _thumb, orig in msg["images"]:
                    data_url = bytes_to_data_url(orig)
                    blocks.append({"type": "input_image", "image_url": data_url})
            if blocks:
                messages.append({"role": "user", "content": blocks})
        elif role == "assistant":
            if msg.get("text"):
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": msg["text"]}]
                })

    # 3) 加上「這一輪」使用者輸入
    messages.append({"role": "user", "content": pending_user_content_blocks})
    return messages

# === 6. Responses 串流 → 純文字產生器（給 st.write_stream） ===
def responses_text_stream(client, *, model, messages, tools=None, tool_choice="none",
                          instructions=None, timeout=MAX_STREAM_TIMEOUT_SEC):
    # 使用官方 stream context，逐事件拿 delta
    with client.responses.stream(
        model=model,
        input=messages,
        tools=tools or [],
        tool_choice=tool_choice,
        instructions=instructions,
        truncation="auto",
        parallel_tool_calls=True,
        reasoning={"effort": "medium"},
        text={"verbosity": "medium"},
        timeout=timeout,
    ) as stream:
        for event in stream:
            et = getattr(event, "type", "")
            if et == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    yield delta
            elif et == "response.error":
                err = getattr(event, "error", "")
                yield f"\n[發生錯誤] {err}\n"

# === 7. 側邊控制（可選） ===
st.sidebar.markdown("### 偏好設定")
allow_web = st.sidebar.toggle("允許網路搜尋（可能稍慢）", value=False)
tool_choice = "auto" if allow_web else "none"
tools = [{"type": "web_search"}] if allow_web else []

# === 8. 顯示歷史（縮圖顯示，省記憶體） ===
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        if msg.get("text"):
            st.markdown(msg["text"])
        if msg.get("images"):
            for fn, thumb, _orig in msg["images"]:
                st.image(thumb, caption=fn, width=220)

# === 9. 回覆階段（真正串流輸出） ===
if st.session_state.pending_ai and st.session_state.pending_content:
    with st.chat_message("assistant"):
        status = st.status("思考中…✨", expanded=False)
        try:
            status.update(label="思考中…✨", state="running")
            trimmed_messages = build_trimmed_input_messages(st.session_state.pending_content)

            # 串流到畫面；write_stream 會回傳完整文字
            ai_text = st.write_stream(
                responses_text_stream(
                    client,
                    model="gpt-5",
                    messages=trimmed_messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    instructions=ANYA_SYSTEM_PROMPT,
                    timeout=MAX_STREAM_TIMEOUT_SEC,
                )
            )
            if not ai_text:
                ai_text = "安妮亞找不到答案～（抱歉啦！）"
            status.update(label="完成！🎉", state="complete")
        except Exception as e:
            ai_text = f"API 發生錯誤：{e}"
            status.update(label="出現小狀況了…請再試一次🛠️", state="error")

        # 寫回歷史 & 收尾
        st.session_state.chat_history.append({
            "role": "assistant",
            "text": ai_text,
            "images": []
        })
        st.session_state.pending_ai = False
        st.session_state.pending_content = None
        st.rerun()

# === 10. 使用者輸入 ===
prompt = st.chat_input(
    "wakuwaku！安妮亞可以幫你看圖說故事嚕！",
    accept_file="multiple",
    file_type=["jpg", "jpeg", "png"]
)

if prompt:
    user_text = prompt.text.strip() if getattr(prompt, "text", None) else ""
    images_for_history = []
    content_blocks = []

    if user_text:
        content_blocks.append({"type": "input_text", "text": user_text})

    files = getattr(prompt, "files", []) or []
    for f in files:
        imgbytes = f.getvalue()
        thumb = make_thumb(imgbytes)
        images_for_history.append((f.name, thumb, imgbytes))
        # 當回合送模型才需要 data_url，這裡先不轉；由 build_trimmed_input_messages 處理

    # 寫入歷史（顯示用）
    st.session_state.chat_history.append({
        "role": "user",
        "text": user_text,
        "images": images_for_history
    })

    # 設定這一輪要送給模型的內容（含圖片）
    for _fn, _thumb, orig in images_for_history:
        data_url = bytes_to_data_url(orig)
        content_blocks.append({"type": "input_image", "image_url": data_url})

    st.session_state.pending_ai = True
    st.session_state.pending_content = content_blocks
    st.rerun()
