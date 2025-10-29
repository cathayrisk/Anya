import streamlit as st
import os
import mimetypes
from openai import OpenAI
import base64
import openai


def file_to_data_url(file):
    # 讀取圖片內容
    file_bytes = file.read()
    # 判斷副檔名
    ext = file.name.split(".")[-1].lower()
    mime = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"
    # 轉 base64
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"
    
# ====== 參數設定 ======
OPENAI_API_KEY = st.secrets["OPENAI_KEY"]
VECTOR_STORE_NAME = "my_knowledge_base"
ALLOWED_FILE_TYPES = ["txt", "pdf", "jpg", "jpeg", "png", "docx", "pptx", "md"]

# ====== 初始化 OpenAI 物件與 Vector Store ======
@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=st.secrets["OPENAI_KEY"])

@st.cache_resource
def get_vector_store(_client):
    vector_store = client.vector_stores.create(name=VECTOR_STORE_NAME)
    return vector_store.id

client = get_openai_client()
vector_store_id = get_vector_store(client)

# ====== 檔案上傳到 Vector Store ======
def upload_file_to_vector_store(file, client, vector_store_id):
    # 先存到本地暫存
    temp_path = f"temp_{file.name}"
    with open(temp_path, "wb") as f:
        f.write(file.read())
    # 上傳到 OpenAI
    file_resp = client.files.create(file=open(temp_path, "rb"), purpose="assistants")
    client.vector_stores.files.create(
        vector_store_id=vector_store_id,
        file_id=file_resp.id
    )
    os.remove(temp_path)
    return file_resp.id

# ====== 多模態查詢 ======
def multimodal_query(client, vector_store_id, user_text=None, image_file=None):
    input_content = []
    if user_text:
        input_content.append({"type": "input_text", "text": user_text})
    if image_file:
        data_url = file_to_data_url(image_file)
        input_content.append({"type": "input_image", "image_url": data_url})

    params = {
        "model": "gpt-4o",
        "input": [{"role": "user", "content": input_content}],
        "tools": [{
            "type": "file_search",
            "vector_store_ids": [vector_store_id]
        }]
    }
    try:
        response = client.responses.create(**params)
        return response
    except openai.BadRequestError as e:
        st.error(f"API BadRequestError: {e}")
        return None
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

# ====== Citation 格式化 ======
def format_response(response):
    # 先檢查 output 是否存在且長度足夠
    if not hasattr(response, "output") or not response.output or len(response.output) < 2:
        return ":red[⚠️ Assistant 沒有回應，請檢查輸入格式或API狀態！]"
    output = response.output[1]
    if not hasattr(output, "content") or not output.content or len(output.content) == 0:
        return ":red[⚠️ Assistant 沒有產生內容，請檢查輸入格式或API狀態！]"
    text = output.content[0].text
    citations = output.content[0].annotations if hasattr(output.content[0], "annotations") else []
    for i, cite in enumerate(citations):
        if hasattr(cite, "filename"):
            text += f"\n[來源{i+1}: {cite.filename}]"
    return text

# ====== Streamlit UI ======
st.title("🦸‍♀️ 多模態 Responses API Chatbot by 安妮亞")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat UI
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input(
    "請輸入問題，或上傳檔案一起問我吧！",
    accept_file="multiple",
    file_type=ALLOWED_FILE_TYPES,
    key="chat_input"
)

if user_input:
    # 1. 處理檔案上傳
    uploaded_files = user_input.files if hasattr(user_input, "files") else []
    user_text = user_input.text if hasattr(user_input, "text") else user_input

    # 2. 檔案全部上傳到 Vector Store
    for file in uploaded_files:
        upload_file_to_vector_store(file, client, vector_store_id)

    # 3. 多模態查詢（只丟第一張圖片給 LLM，其他都進知識庫）
    image_file = None
    for file in uploaded_files:
        mime, _ = mimetypes.guess_type(file.name)
        if mime and mime.startswith("image/"):
            image_file = file
            break

    # 4. 呼叫 Responses API
    response = multimodal_query(client, vector_store_id, user_text, image_file)

    # 5. 顯示對話
    st.session_state.chat_history.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    answer = format_response(response)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
