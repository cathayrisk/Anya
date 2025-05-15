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
    st.header("用戶登入")
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
        st.session_state["authenticated"] = False
        st.session_state["username"] = None
        st.session_state["user_id"] = None

    # 聊天主題管理
    if st.session_state.get("authenticated"):
        st.markdown("---")
        with st.expander("💬 聊天主題管理", expanded=True):
            threads = supabase.table("threads").select("*").eq("user_id", st.session_state["user_id"]).order("created_at").execute().data
            thread_titles = [t["title"] for t in threads]
            thread_ids = [t["thread_id"] for t in threads]
            selected_idx = None

            if thread_titles:
                for i, (title, tid) in enumerate(zip(thread_titles, thread_ids)):
                    cols = st.columns([8, 1])
                    with cols[0]:
                        if st.session_state.get("thread_id") == tid:
                            st.markdown(f"**:blue[{title}]**")
                        else:
                            if st.button(title, key=f"select_thread_{tid}"):
                                st.session_state["thread_id"] = tid
                                st.session_state["thread"] = title
                                st.rerun()
                    with cols[1]:
                        if st.button("🗑️", key=f"delete_thread_{tid}"):
                            supabase.table("threads").delete().eq("thread_id", tid).eq("user_id", st.session_state["user_id"]).execute()
                            st.success("已刪除主題！")
                            if st.session_state.get("thread_id") == tid:
                                st.session_state["thread_id"] = None
                                st.session_state["thread"] = None
                            st.rerun()
            else:
                st.info("尚無主題，請新增。")

            # 新增主題區
            st.markdown("------")
            cols = st.columns([7, 2])
            with cols[0]:
                new_title = st.text_input("新主題名稱", key="new_thread_title", label_visibility="collapsed", placeholder="輸入主題名稱")
            with cols[1]:
                if st.button("➕", key="create_thread_btn") and new_title:
                    supabase.table("threads").insert({
                        "user_id": st.session_state["user_id"],
                        "title": new_title,
                        "created_at": datetime.now().isoformat()
                    }).execute()
                    st.success("已建立新主題！")
                    st.rerun()

# 主畫面提示
if not st.session_state.get("authenticated"):
    st.info("請先登入用戶")
elif not st.session_state.get("thread_id"):
    st.info("請在側邊欄選擇聊天主題")
else:
    st.success(f"已登入：{st.session_state['username']}，目前主題：{st.session_state['thread']}")
    # 這裡可以放主題內容或聊天UI
