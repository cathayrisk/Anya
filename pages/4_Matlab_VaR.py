import streamlit as st
from supabase import create_client, Client

# 你的Supabase專案資訊
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
BUCKET = "matlabvar"  # 你的bucket名稱

# 初始化Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

st.title("Supabase Storage 檔案上傳工具 🥜")

uploaded_file = st.file_uploader("請選擇要上傳的檔案", type=None)

if uploaded_file is not None:
    # 取得檔案內容
    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name

    # 上傳到Supabase Storage
    with st.spinner("上傳中..."):
        res = supabase.storage.from_(BUCKET).upload(file_name, file_bytes, {"content-type": uploaded_file.type})
    if res.get("error"):
        st.error(f"上傳失敗：{res['error']['message']}")
    else:
        st.success("上傳成功！")
        # 取得公開網址
        public_url = supabase.storage.from_(BUCKET).get_public_url(file_name)
        st.markdown(f"**檔案網址：** [{public_url['data']['publicUrl']}]({public_url['data']['publicUrl']})")
        st.image(public_url['data']['publicUrl'], caption="預覽", use_column_width=True)
