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

# sidebar popover：用戶登入＋主題管理
with st.sidebar:
    with st.popover("用戶/主題管理", icon=":material/manage_accounts:"):
        # --- 用戶登入 ---
        st.markdown("#### 選擇用戶")
        user = st.segmented_control("選擇用戶", username_list, key="user_selector")
        user_row = supabase.table("users").select("*").eq("username", user).execute().data
        if user_row:
            user_id = user_row[0]["user_id"]
            if st.session_state.get("user_id") != user_id:
                st.session_state.user_id = user_id
                st.session_state.username = user
                st.session_state.authenticated = False
                st.session_state.thread_id = None
                st.rerun()
            # 密碼驗證
            if not st.session_state.get("authenticated"):
                password = st.text_input("請輸入密碼", type="password", key="pw_input")
                if st.button("登入", key="login_btn"):
                    if user_row[0]["password"] == password:
                        st.session_state.authenticated = True
                        st.success(f"🎉 歡迎 {user}！")
                        st.rerun()
                    else:
                        st.error("密碼錯誤，請再試一次")
        else:
            st.session_state.user_id = None
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.thread_id = None

        # --- 主題管理 ---
        if st.session_state.get("authenticated"):
            st.markdown("#### 聊天主題管理")
            threads = supabase.table("threads").select("*").eq("user_id", st.session_state.user_id).order("created_at").execute().data
            thread_titles = [t["title"] for t in threads]
            if threads:
                thread_idx = st.radio("選擇聊天主題", thread_titles, key="thread_selector")
                thread_id = threads[thread_titles.index(thread_idx)]["thread_id"]
                if st.session_state.get("thread_id") != thread_id:
                    st.session_state.thread_id = thread_id
                    st.rerun()
                st.success(f"目前主題：{thread_idx}")
            else:
                st.session_state.thread_id = None
                st.info("尚無主題，請新增。")
            with st.expander("➕ 新增主題"):
                new_title = st.text_input("新主題名稱", key="new_thread_title")
                if st.button("建立", key="create_thread_btn") and new_title:
                    supabase.table("threads").insert({
                        "user_id": st.session_state.user_id,
                        "title": new_title,
                        "created_at": datetime.now().isoformat()
                    }).execute()
                    st.success("已建立新主題！請重新選擇。")
                    st.rerun()
            if st.session_state.get("thread_id"):
                if st.button("🗑️ 刪除本主題", key="delete_thread_btn"):
                    supabase.table("threads").delete().eq("thread_id", st.session_state.thread_id).execute()
                    st.success("已刪除主題！")
                    st.session_state.thread_id = None
                    st.rerun()
        else:
            st.info("請先登入用戶。")

# 主畫面完全乾淨，僅根據狀態顯示提示
if not st.session_state.get("authenticated"):
    st.info("請在側邊欄登入用戶")
elif not st.session_state.get("thread_id"):
    st.info("請在側邊欄選擇聊天主題")
else:
    st.success(f"已登入：{st.session_state['username']}，目前主題：{thread_idx}")
    # 這裡可以放主題內容或聊天UI
