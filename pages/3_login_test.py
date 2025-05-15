import streamlit as st
from supabase import create_client, Client

# 初始化 Supabase 連線
@st.cache_resource
def init_connection():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_connection()

# 取得所有帳號
@st.cache_data(ttl=60)
def get_usernames():
    users = supabase.table("users").select("username").execute().data
    return [u["username"] for u in users]

username_list = get_usernames()

# 用 st.form 包住整個登入流程
with st.form("login_form"):
    selected_username = st.pills("請選擇登入帳號", username_list, selection_mode="single")
    password = st.text_input("請輸入密碼", type="password")
    submit = st.form_submit_button("登入")

if submit:
    if selected_username and password:
        # 查詢該帳號的密碼（明文）
        user = supabase.table("users").select("password").eq("username", selected_username).single().execute().data
        if user:
            if user["password"] == password:
                st.session_state["authenticated"] = True
                st.session_state["username"] = selected_username
                st.success(f"🎉 歡迎 {selected_username}！")
            else:
                st.error("密碼錯誤，請再試一次")
        else:
            st.error("帳號不存在")
    else:
        st.warning("請選擇帳號並輸入密碼")

# 登入狀態顯示
if st.session_state.get("authenticated"):
    st.write(f"✅ 登入成功，歡迎 {st.session_state['username']}！")
    # 這裡可以放登入後的主畫面內容
