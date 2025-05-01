import streamlit as st
import tempfile
import fitz  # pymupdf
import difflib
import pandas as pd
import io

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

# 4. UI
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
    st.subheader("比對差異表格")
    with st.spinner("正在抽取 PDF 文字..."):
        doc1_text = extract_pdf_text(file1.getvalue())
        doc2_text = extract_pdf_text(file2.getvalue())

    if st.button("開始比對並顯示所有差異"):
        with st.spinner("正在比對..."):
            df = extract_diff_dataframe_v2(doc1_text, doc2_text)
            df = df[~((df['文件1內容'] == "") & (df['文件2內容'] == ""))]
            df = df.reset_index(drop=True)
            st.write(f"本次比對共發現 {len(df)} 處差異。")
            if len(df) == 0:
                st.info("兩份文件沒有明顯差異。")
            else:
                st.dataframe(df, hide_index=True)
                download_report(df)
else:
    st.info("請分別上傳文件1與文件2")
