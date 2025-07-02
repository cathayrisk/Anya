import streamlit as st
from supabase import create_client, Client
import traceback

# 你的Supabase專案資訊
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
BUCKET = "matlabvar"  # 你的bucket名稱

# 初始化Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

st.title("Supabase Storage 檔案上傳工具 🥜")

uploaded_file = st.file_uploader("請選擇要上傳的檔案", type=None, key="file_uploader_1")

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name
    content_type = uploaded_file.type or "application/octet-stream"

    try:
        with st.spinner("上傳中..."):
            res = supabase.storage.from_(BUCKET).upload(file_name, file_bytes, {"content-type": content_type})

        # 檢查 UploadResponse 物件
        if hasattr(res, "error") and res.error:
            st.error(f"上傳失敗：{res.error}")
        else:
            st.success("上傳成功！")
    except Exception as e:
        st.error(f"Exception: {e}")
        st.text(traceback.format_exc())
