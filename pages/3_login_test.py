import streamlit as st
from supabase import create_client, Client
import argon2

# 連接 Supabase
url = st.secrets["SUPABASE_URL"]
key = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(url, key)

# 取得所有 user name
usernames = supabase.table("users").select("username").execute().data
username_list = [u["username"] for u in usernames]

# 1. 先顯示 st.pills 讓使用者選帳號
selected_username = st.pills("請選擇登入帳號", username_list, selection_mode="single")

# 2. 選擇帳號後才顯示密碼欄位
if selected_username:
    password = st.text_input("請輸入密碼", type="password")
    if st.button("登入"):
        # 查詢該帳號的密碼 hash
        user = supabase.table("users").select("password").eq("username", selected_username).single().execute().data
        if user:
            hashed_password = user["password"]
            ph = argon2.PasswordHasher()
            try:
                ph.verify(hashed_password, password)
                st.session_state["authenticated"] = True
                st.session_state["username"] = selected_username
                st.success(f"歡迎 {selected_username}")
            except argon2.exceptions.VerifyMismatchError:
                st.error("密碼錯誤，請再試一次")
        else:
            st.error("帳號不存在")
else:
    st.info("請先選擇登入帳號")

# 3. 登入狀態顯示
if st.session_state.get("authenticated"):
    st.write(f"🎉 登入成功，歡迎 {st.session_state['username']}！")
