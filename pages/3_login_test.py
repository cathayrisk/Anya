import streamlit as st
from supabase import create_client, Client
from datetime import datetime

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

with st.sidebar:
    # 1. 用戶登入區
    if not st.session_state.get("authenticated"):
        selected_username = st.pills("選擇用戶", username_list, selection_mode="single", key="user_selector")
        if selected_username:
            user_row = supabase.table("users").select("user_id, password").eq("username", selected_username).single().execute().data
            password = st.text_input("請輸入密碼", type="password", key="pw_input")
            if st.button("登入", key="login_btn"):
                if user_row and user_row["password"] == password:
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = selected_username
                    st.session_state["user_id"] = user_row["user_id"]
                    st.success(f"🎉 歡迎 {selected_username}！")
                    st.rerun()
                else:
                    st.error("密碼錯誤，請再試一次")
        else:
            st.info("請先選擇登入帳號")
    else:
        # 2. 聊天主題管理區（用 expander 收納）
        with st.expander("聊天主題管理", expanded=True):
            threads = supabase.table("threads").select("*").eq("user_id", st.session_state["user_id"]).order("created_at").execute().data
            thread_titles = [t["title"] for t in threads]
            thread_ids = [t["thread_id"] for t in threads]
            if thread_titles:
                selected_idx = st.radio("選擇聊天主題", range(len(thread_titles)), format_func=lambda i: thread_titles[i], key="thread_selector")
                selected_thread_id = thread_ids[selected_idx]
                selected_thread_title = thread_titles[selected_idx]
                st.session_state["thread_id"] = selected_thread_id
                st.session_state["thread"] = selected_thread_title
            else:
                st.session_state["thread_id"] = None
                st.session_state["thread"] = None
                st.info("尚無主題，請新增。")
            # 新增主題
            new_title = st.text_input("新主題名稱", key="new_thread_title")
            if st.button("建立", key="create_thread_btn") and new_title:
                supabase.table("threads").insert({
                    "user_id": st.session_state["user_id"],
                    "title": new_title,
                    "created_at": datetime.now().isoformat()
                }).execute()
                st.success("已建立新主題！請重新選擇。")
                st.rerun()
            # 刪除主題
            if st.session_state.get("thread_id"):
                if st.button("🗑️ 刪除本主題", key="delete_thread_btn"):
                    supabase.table("threads").delete().eq("thread_id", st.session_state["thread_id"]).eq("user_id", st.session_state["user_id"]).execute()
                    st.success("已刪除主題！")
                    st.session_state["thread_id"] = None
                    st.session_state["thread"] = None
                    st.rerun()

# 主畫面提示
if not st.session_state.get("authenticated"):
    st.info("請在側邊欄登入用戶")
elif not st.session_state.get("thread_id"):
    st.info("請在側邊欄選擇聊天主題")
else:
    st.success(f"已登入：{st.session_state['username']}，目前主題：{st.session_state['thread']}")
    # 這裡可以放主題內容或聊天UI
