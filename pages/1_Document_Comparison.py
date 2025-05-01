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

# 4. 人工規則摘要
def find_section(line, doc_text):
    # 根據你的文件格式，找出該行屬於哪個主題
    # 這裡假設主題格式為 •【主題】
    sections = re.findall(r"•\s*【(.+?)】", doc_text)
    for section in sections:
        if section in line:
            return section
    # 若找不到，回傳空字串
    return ""

def generate_diff_summary(df, doc1_text, doc2_text):
    summary = []
    for idx, row in df.iterrows():
        # 嘗試找出主題
        section = find_section(row['文件1內容'], doc1_text) or find_section(row['文件2內容'], doc2_text)
        if not section:
            section = "未知主題"
        # 產生摘要句
        if row['差異類型'] == "修改":
            summary.append(f"在「{section}」部分，內容由「{row['文件1內容']}」修改為「{row['文件2內容']}」。")
        elif row['差異類型'] == "新增":
            summary.append(f"在「{section}」部分，新增內容：「{row['文件2內容']}」。")
        elif row['差異類型'] == "刪除":
            summary.append(f"在「{section}」部分，刪除內容：「{row['文件1內容']}」。")
    return "\n".join(summary)

# 5. AI摘要（LangChain）
def ai_summarize_diff(df):
    prompt = (
        "請根據下列差異表格，總結兩份文件的主要不同點，"
        "用條列式中文簡明說明，重點放在內容意義的變化：\n"
        + df.to_string(index=False)
    )
    llm = ChatOpenAI(
        model="gpt-4-1106-preview",
        openai_api_key=st.secrets["OPENAI_KEY"],
        temperature=0.0,
        streaming=False
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
    with col2:
        file2 = st.file_uploader("請上傳文件2（比較檔）", type=["pdf"], key="file2")
        if file2:
            st.success(f"已上傳：{file2.name}")

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
            st.write(f"本次比對共發現 {len(df)} 處差異。")

            tab1, tab2 = st.tabs(["比對差異表格", "自動摘要（AI/人工）"])

            with tab1:
                if len(df) == 0:
                    st.info("兩份文件沒有明顯差異。")
                else:
                    st.dataframe(df, hide_index=True)
                    download_report(df)

            with tab2:
                st.markdown("#### 人工規則摘要")
                summary = generate_diff_summary(df, doc1_text, doc2_text)
                if summary:
                    st.info(summary)
                else:
                    st.info("無明顯差異可摘要。")

                st.markdown("#### AI自動摘要（LangChain）")
                with st.spinner("AI 正在摘要..."):
                    try:
                        ai_summary = ai_summarize_diff(df, openai_api_key)
                        st.success(ai_summary)
                    except Exception as e:
                        st.error(f"AI 摘要失敗：{e}")

else:
    st.info("請分別上傳文件1與文件2")
