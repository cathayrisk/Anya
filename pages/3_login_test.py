import streamlit as st
from st_login_form import login_form

client = login_form()

if st.session_state["authenticated"]:
    if st.session_state["username"]:
        st.success(f"歡迎 {st.session_state['username']}")
        # 這裡可以放登入後的主畫面
    else:
        st.success("歡迎訪客")
        # 這裡可以放訪客畫面
else:
    st.error("尚未登入")
