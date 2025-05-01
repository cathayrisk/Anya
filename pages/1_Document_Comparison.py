import streamlit as st
import tempfile
import fitz  # pymupdf
import difflib
import pandas as pd
import io
import re

# AI摘要用
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage

st.set_page_config(page_title="🔍安妮亞來找碴🔎", layout="wide")
st.title("文件差異比對工具")

# 1. 快取 PDF 文字抽取
@st.cache_data(show_spinner="正在抽取 PDF 文字...")
def extract_pdf_text(pdf_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf_file:
        temp_pdf_file.write(pdf_bytes)
        temp_pdf_path = temp_pdf_file.name
    doc = fitz.open(temp_pdf_path)
    doc_text = ""
    for page_number, page in enumerate(doc):
        text = page.get_text("text")
        doc_text += f"\n--- Page {page_number+1} ---\n{text}"
    doc.close()
    return doc_text

# 2. 比對邏輯
def extract_diff_dataframe_v2(text1, text2):
    lines1 = text1.splitlines()
    lines2 = text2.splitlines()
    sm = difflib.SequenceMatcher(None, lines1, lines2)
    diff_rows = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'replace':
            maxlen = max(i2 - i1, j2 - j1)
            for k in range(maxlen):
                l1 = lines1[i1 + k] if i1 + k < i2 else ""
                l2 = lines2[j1 + k] if j1 + k < j2 else ""
                diff_rows.append({
                    "差異類型": "修改",
                    "文件1內容": l1,
                    "文件2內容": l2
                })
        elif tag == 'delete':
            for l1 in lines1[i1:i2]:
                diff_rows.append({
                    "差異類型": "刪除",
                    "文件1內容": l1,
                    "文件2內容": ""
                })
        elif tag == 'insert':
            for l2 in lines2[j1:j2]:
                diff_rows.append({
                    "差異類型": "新增",
                    "文件1內容": "",
                    "文件2內容": l2
                })
        # 'equal' 不顯示
    return pd.DataFrame(diff_rows)

# 3. 下載報告（只保留 Excel）
def download_report(df):
    excel_buffer = io.BytesIO()
    df.to_excel(excel_buffer, index=False)
    st.download_button(
        "下載 Excel 報告",
        excel_buffer.getvalue(),
        file_name="diff_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

def get_diff_brief(old, new):
    # 只顯示不同的部分
    seqm = difflib.SequenceMatcher(None, old, new)
    diff = []
    for opcode, a0, a1, b0, b1 in seqm.get_opcodes():
        if opcode == 'replace':
            diff.append(f"「{old[a0:a1]}」→「{new[b0:b1]}」")
        elif opcode == 'delete':
            diff.append(f"刪除「{old[a0:a1]}」")
        elif opcode == 'insert':
            diff.append(f"新增「{new[b0:b1]}」")
    return "；".join(diff) if diff else "細微變動"

def generate_diff_summary_brief_with_lineno_and_context(df, doc1_text, doc2_text):
    lines1 = doc1_text.splitlines()
    lines2 = doc2_text.splitlines()
    summary = []
    for idx, row in df.iterrows():
        l1 = row['文件1內容']
        l2 = row['文件2內容']
        line_no = -1
        context = ""
        if row['差異類型'] in ["修改", "刪除"]:
            try:
                line_no = lines1.index(l1) + 1
                context = l1.strip()
            except ValueError:
                pass
        elif row['差異類型'] == "新增":
            try:
                line_no = lines2.index(l2) + 1
                context = l2.strip()
            except ValueError:
                pass

        prefix = f"第{line_no}行：" if line_no > 0 else ""
        if row['差異類型'] == "修改":
            diff_brief = get_diff_brief(l1, l2)
            summary.append(f"{prefix}「{context}」{diff_brief}")
        elif row['差異類型'] == "新增":
            summary.append(f"{prefix}新增內容：「{context}」")
        elif row['差異類型'] == "刪除":
            summary.append(f"{prefix}刪除內容：「{context}」")
    # 用 <br> 強制換行
    return "<br>".join(summary)

# 5. AI摘要（LangChain）
def ai_summarize_diff(df):
    prompt = (
        "請根據下列差異表格，直接條列出每一筆文字內容的差異（例如：第X行：A→B、刪除、或新增），"
        "不用解釋意義，也不用總結，只要明確列出差異內容即可：\n"
        + df.to_string(index=False)
    )
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        openai_api_key=st.secrets["OPENAI_KEY"],
        temperature=0.0
    )
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    return response.content

# 6. UI
with st.expander("上傳文件1（基準檔）與文件2（比較檔）", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        file1 = st.file_uploader("請上傳文件1（基準檔）", type=["pdf"], key="file1")
        if file1:
            st.success(f"已上傳：{file1.name}")
            st.session_state['file1_bytes'] = file1.getvalue()
    with col2:
        file2 = st.file_uploader("請上傳文件2（比較檔）", type=["pdf"], key="file2")
        if file2:
            st.success(f"已上傳：{file2.name}")
            st.session_state['file2_bytes'] = file2.getvalue()

if file1 and file2:
    st.markdown("---")
    st.subheader("比對差異分析")
    with st.spinner("正在抽取 PDF 文字..."):
        doc1_text = extract_pdf_text(file1.getvalue())
        doc2_text = extract_pdf_text(file2.getvalue())

    if st.button("開始比對並顯示所有差異"):
        with st.spinner("正在比對..."):
            df = extract_diff_dataframe_v2(doc1_text, doc2_text)
            df = df[~((df['文件1內容'] == "") & (df['文件2內容'] == ""))]
            df = df.reset_index(drop=True)
            st.session_state['diff_df'] = df  # 存進 session_state
            st.session_state['doc1_text'] = doc1_text
            st.session_state['doc2_text'] = doc2_text
            st.session_state['has_compared'] = True
            st.write(f"本次比對共發現 {len(df)} 處差異。")

        if st.session_state.get('has_compared', False):
            df = st.session_state['diff_df']
            doc1_text = st.session_state['doc1_text']
            doc2_text = st.session_state['doc2_text']
            
            tab1, tab2 = st.tabs(["比對差異表格", "自動摘要（AI/人工）"])

            with tab1:
                st.markdown("#### 人工規則摘要")
                summary = generate_diff_summary_brief_with_lineno_and_context(df, doc1_text, doc2_text)
                if summary:
                    st.markdown(summary, unsafe_allow_html=True)
                else:
                    st.info("無明顯差異可摘要。")
                    
                if len(df) == 0:
                    st.info("兩份文件沒有明顯差異。")
                else:
                    st.text("文件差異")
                    st.dataframe(df, hide_index=True)
                    download_report(df)

            with tab2:
                st.markdown("#### AI自動摘要")
                with st.spinner("AI 正在摘要..."):
                    try:
                        ai_summary = ai_summarize_diff(df)
                        st.success(ai_summary)
                    except Exception as e:
                        st.error(f"AI 摘要失敗：{e}")

else:
    st.info("請分別上傳文件1與文件2")
