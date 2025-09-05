import os
import base64
from io import BytesIO
from dotenv import load_dotenv
from PIL import Image
import streamlit as st
from openai import OpenAI

def filter_valid_messages():
    msgs = []
    for m in st.session_state.get('messages', []):
        if isinstance(m, dict) and 'role' in m and 'content' in m:
            msgs.append(m)
        else:
            print("發現壞掉的訊息：", m)
    st.session_state['messages'] = msgs

filter_valid_messages()
if "messages" not in st.session_state or len(st.session_state.messages) == 0:  # 若空就補一條助理
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?", "image": None}
    ]

st.set_page_config(page_title='OpenAI Banana Image Chatbot', layout="wide")
avatars = {
    "assistant": "🤖",
    "user": "👤"
}

# 防呆：全部聊天訊息轉 dict
def convert_all_messages_to_dict():
    if "messages" in st.session_state:
        msgs = []
        for m in st.session_state.messages:
            if isinstance(m, dict):
                msgs.append(m)
            elif hasattr(m, '__dict__'):
                msgs.append(m.__dict__)
        st.session_state.messages = msgs
convert_all_messages_to_dict()

# 初始化 session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?", "image": None}
    ]

with st.popover("⚙️ 功能選單"):  # 你可以改成自己喜歡的emoji＆文字
    uploaded_file = st.file_uploader("上傳圖片", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_bytes = Image.open(uploaded_file)
        st.session_state["image"] = image_bytes
        st.image(image_bytes, caption="Uploaded Image", use_container_width=True)

for message in st.session_state.messages:
    if not (isinstance(message, dict) and "role" in message and "content" in message):
        continue
    with st.chat_message(message["role"], avatar=avatars[message["role"]]):
        st.write(message["content"])
        if message["role"] == "assistant" and message.get("image"):
            st.image(message["image"])
            
def pil_to_image_upload(img, fmt="PNG"):
    buf = BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return ("uploaded.png", buf, f"image/{fmt.lower()}")

def edit_img(prompt, image=None):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_KEY"]
    client = OpenAI(api_key=api_key)
    if image:
        image_data = pil_to_image_upload(image)
    else:
        return None
    result = client.images.edit(
        model="gpt-image-1",
        image=image_data,
        prompt=prompt,
        input_fidelity="high",
        quality="high",
        output_format="jpeg"
    )
    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_bytes))

if prompt := st.chat_input():
    user_msg = {"role": "user", "content": prompt, "image": None}
    st.session_state.messages.append(dict(user_msg))
    with st.chat_message("user", avatar=avatars["user"]):
        st.write(prompt)

    with st.chat_message("assistant", avatar=avatars["assistant"]):
        with st.spinner("Thinking..."):
            uploaded_img = st.session_state.get("image", None)
            if uploaded_img:
                result_img = edit_img(prompt, uploaded_img)
                response_text = "這是根據你的描述產生的新圖片～"
            else:
                result_img = None
                response_text = "請先上傳圖片才能編輯喔！"
            st.write(response_text)
            if result_img:
                st.image(result_img)
    assistant_msg = {
        "role": "assistant",
        "content": response_text,
        "image": result_img if uploaded_img else None
    }
    st.session_state.messages.append(dict(assistant_msg))
