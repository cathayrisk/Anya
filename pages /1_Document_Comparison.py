
import os
import streamlit as st
import tempfile
import fitz  # pymupdf
from difflib import SequenceMatcher, unified_diff, HtmlDiff
import difflib
import streamlit.components.v1 as components

# 設置 OpenAI API Key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_KEY"]

# 設置頁面配置
st.set_page_config(page_title="Document_Comparison", layout="wide", page_icon=":mage:")

# 計算相似度的函數
def calculate_similarity(text1, text2):
    seq_match = SequenceMatcher(None, text1, text2)
    return seq_match.ratio()

# 計算差異的函數
def calculate_differences(text1, text2):
    diff = unified_diff(text1.splitlines(), text2.splitlines(), lineterm='')
    return '\n'.join(list(diff))

# 生成 HTML 差異報告的函數
def generate_html_diff(text1, text2):
    d = HtmlDiff()
    return d.make_file(text1.splitlines(), text2.splitlines())

# 使用 st.file_uploader 來支持多文件上傳
uploaded_files = st.file_uploader("Upload your PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files and len(uploaded_files) == 2:
    documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf_file:
            temp_pdf_file.write(uploaded_file.read())
            temp_pdf_path = temp_pdf_file.name

        # 使用 pymupdf 來讀取 PDF 文件
        doc = fitz.open(temp_pdf_path)
        doc_text = ""
        for page_number, page in enumerate(doc):
            text = page.get_text("text")
            doc_text += text
        documents.append(doc_text)

    st.success(f"已載入 {len(documents)} 個文件。")

    # 比較兩個文件
    if len(documents) == 2:
        # 計算相似度
        similarity_score = calculate_similarity(documents[0], documents[1])
        st.write(f"文件相似度分數: {similarity_score}")

        # 使用 st.expander 收納差異顯示
        with st.expander("查看文件差異", expanded=False):
            differences = calculate_differences(documents[0], documents[1])
            st.write(differences)

        # 生成並顯示 HTML 差異報告
        html_diff = generate_html_diff(documents[0], documents[1])


        # 確保 html_diff 是有效的 HTML 字符串
        #if html_diff:
        #    try:
        #        st.html(html_diff)  # 使用 st.html 來顯示 HTML 差異報告
        #    except Exception as e:
        #        st.error(f"顯示 HTML 差異報告時出現錯誤: {e}")
        #else:
        #    st.error("生成的 HTML 差異報告無效。")

        # 確保 html_diff 是有效的 HTML 字符串
        if html_diff:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as temp_html_file:
                    temp_html_file.write(html_diff.encode('utf-8'))
                    temp_html_path = temp_html_file.name

                # 提供下載鏈接
                with open(temp_html_path, 'rb') as file:
                    st.download_button(
                        label="下載並查看 HTML 差異報告",
                        data=file,
                        file_name="diff_report.html",
                        mime="text/html"
                    )
                #with st.container():
                #    components.html(html_diff, width=1800, height=800, scrolling=True)
                    #st.html(html_diff)  # 使用 st.html 來顯示 HTML 差異報告
            except Exception as e:
                st.error(f"顯示 HTML 差異報告時出現錯誤: {e}")
        else:
            st.error("生成的 HTML 差異報告無效。")
