import streamlit as st
import tempfile
import fitz  # pip install pymupdf
from docx import Document  # pip install python-docx
import difflib
import pandas as pd
import io

# ========== 1. 文字抽取 ==========
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

@st.cache_data(show_spinner="正在抽取 DOCX 文字...")
def extract_docx_text(docx_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_docx_file:
        temp_docx_file.write(docx_bytes)
        temp_docx_path = temp_docx_file.name
    doc = Document(temp_docx_path)
    doc_text = ""
    for para in doc.paragraphs:
        doc_text += para.text + "\n"
    return doc_text

@st.cache_data(show_spinner="正在抽取 TXT/MD 文字...")
def extract_txt_md_text(file_bytes):
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("big5", errors="ignore")

def get_text_from_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return extract_pdf_text(uploaded_file.getvalue())
    elif name.endswith(".docx"):
        return extract_docx_text(uploaded_file.getvalue())
    elif name.endswith(".txt") or name.endswith(".md"):
        return extract_txt_md_text(uploaded_file.getvalue())
    else:
        raise ValueError("不支援的檔案格式")

# ========== 2. 差異比對（含進度條） ==========
def extract_diff_dataframe_with_progress(text1, text2):
    lines1 = [line for line in text1.splitlines() if line.strip()]
    lines2 = [line for line in text2.splitlines() if line.strip()]
    sm = difflib.SequenceMatcher(None, lines1, lines2)
    diff_rows = []
    opcodes = sm.get_opcodes()
    progress_bar = st.progress(0, text="比對進度")
    total = len(opcodes)
    for idx, (tag, i1, i2, j1, j2) in enumerate(opcodes):
        if tag == 'replace':
            maxlen = max(i2 - i1, j2 - j1)
            for k in range(maxlen):
                l1 = lines1[i1 + k] if i1 + k < i2 else ""
                l2 = lines2[j1 + k] if j1 + k < j2 else ""
                diff_rows.append({
                    "行號1": i1 + k + 1 if i1 + k < i2 else "",
                    "行號2": j1 + k + 1 if j1 + k < j2 else "",
                    "差異類型": "修改",
                    "基準文件內容": l1,
                    "比較文件內容": l2
                })
        elif tag == 'delete':
            for k, l1 in enumerate(lines1[i1:i2]):
                diff_rows.append({
                    "行號1": i1 + k + 1,
                    "行號2": "",
                    "差異類型": "刪除",
                    "基準文件內容": l1,
                    "比較文件內容": ""
                })
        elif tag == 'insert':
            for k, l2 in enumerate(lines2[j1:j2]):
                diff_rows.append({
                    "行號1": "",
                    "行號2": j1 + k + 1,
                    "差異類型": "新增",
                    "基準文件內容": "",
                    "比較文件內容": l2
                })
        progress_bar.progress((idx + 1) / total, text=f"比對進度：{idx+1}/{total}")
    progress_bar.empty()
    return pd.DataFrame(diff_rows)

# ========== 3. 人工規則摘要 ==========
def get_diff_brief(old, new):
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

def generate_diff_summary_brief_with_lineno_and_context(df):
    summary = []
    for idx, row in df.iterrows():
        l1 = row['基準文件內容'].strip()
        l2 = row['比較文件內容'].strip()
        line_no = row['行號1'] if row['行號1'] else row['行號2']
        context = l1 if l1 else l2
        prefix = f"第{line_no}行：" if line_no else ""
        if row['差異類型'] == "修改":
            diff_brief = get_diff_brief(l1, l2)
            summary.append(f"{prefix}「{context}」{diff_brief}")
        elif row['差異類型'] == "新增":
            summary.append(f"{prefix}新增內容：「{context}」")
        elif row['差異類型'] == "刪除":
            summary.append(f"{prefix}刪除內容：「{context}」")
    return "<br>".join(summary)

# ========== 4. 下載報告 ==========
def download_report(df):
    excel_buffer = io.BytesIO()
    df.to_excel(excel_buffer, index=False)
    st.download_button(
        "下載 Excel 報告",
        excel_buffer.getvalue(),
        file_name="diff_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ========== 5. 原文高亮 ==========
def highlight_diffs_in_text(text, diff_lines, color="#fff2ac"):
    lines = [line for line in text.splitlines() if line.strip()]
    highlighted = []
    for idx, line in enumerate(lines, 1):
        if idx in diff_lines:
            highlighted.append(f"<span style='background-color:{color}'>{line}</span>")
        else:
            highlighted.append(line)
    return "<br>".join(highlighted)

# ========== 6. UI ==========
st.set_page_config(page_title="🔍文件差異比對工具", layout="wide")
st.title("文件差異比對工具")

with st.expander("上傳文件1（基準檔）與文件2（比較檔）", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        file1 = st.file_uploader("請上傳文件1（基準檔）", type=["pdf", "docx", "txt", "md"], key="file1")
        if file1:
            st.success(f"已上傳：{file1.name}")
            st.session_state['file1_bytes'] = file1.getvalue()
            st.session_state['file1_name'] = file1.name
    with col2:
        file2 = st.file_uploader("請上傳文件2（比較檔）", type=["pdf", "docx", "txt", "md"], key="file2")
        if file2:
            st.success(f"已上傳：{file2.name}")
            st.session_state['file2_bytes'] = file2.getvalue()
            st.session_state['file2_name'] = file2.name

if st.session_state.get('file1_bytes') and st.session_state.get('file2_bytes'):
    try:
        with st.spinner("正在抽取文件文字..."):
            doc1_text = get_text_from_file(
                type("FakeUpload", (), {"name": st.session_state['file1_name'], "getvalue": lambda: st.session_state['file1_bytes']})()
            )
            doc2_text = get_text_from_file(
                type("FakeUpload", (), {"name": st.session_state['file2_name'], "getvalue": lambda: st.session_state['file2_bytes']})()
            )
    except Exception as e:
        st.error(f"檔案解析失敗：{e}")
        st.stop()

    if st.button("開始比對並顯示所有差異"):
        with st.spinner("正在比對..."):
            try:
                df = extract_diff_dataframe_with_progress(doc1_text, doc2_text)
                df = df.reset_index(drop=True)
                st.session_state['diff_df'] = df
                st.session_state['doc1_text'] = doc1_text
                st.session_state['doc2_text'] = doc2_text
                st.session_state['has_compared'] = True
            except Exception as e:
                st.error(f"比對過程發生錯誤：{e}")
                st.stop()

if st.session_state.get('has_compared', False):
    df = st.session_state['diff_df']
    doc1_text = st.session_state['doc1_text']
    doc2_text = st.session_state['doc2_text']

    # ========== 頁面上方：人工摘要 ==========
    st.subheader("🔎 差異摘要")
    summary = generate_diff_summary_brief_with_lineno_and_context(df)
    st.markdown(summary, unsafe_allow_html=True)

    # ========== 中間：差異表格 ==========
    st.subheader("📋 差異明細表格")
    search = st.text_input("搜尋差異內容（可輸入關鍵字）")
    filter_type = st.selectbox("篩選差異類型", ["全部", "修改", "新增", "刪除"])
    df_show = df.copy()
    if search:
        df_show = df_show[df_show.apply(lambda row: search in row['基準文件內容'] or search in row['比較文件內容'], axis=1)]
    if filter_type != "全部":
        df_show = df_show[df_show['差異類型'] == filter_type]
    st.dataframe(df_show, hide_index=True)
    download_report(df_show)

    # ========== 下方：原文高亮 ==========
    st.subheader("📝 原文高亮顯示")
    tab_a, tab_b = st.tabs(["基準文件", "比較文件"])
    import pandas as pd
    diff_lines1 = set(pd.to_numeric(df['行號1'], errors='coerce').dropna().astype(int))
    diff_lines2 = set(pd.to_numeric(df['行號2'], errors='coerce').dropna().astype(int))
    with tab_a:
        st.markdown(highlight_diffs_in_text(doc1_text, diff_lines1, color="#ffcccc"), unsafe_allow_html=True)
    with tab_b:
        st.markdown(highlight_diffs_in_text(doc2_text, diff_lines2, color="#ccffcc"), unsafe_allow_html=True)
else:
    st.info("請先上傳文件並執行比對")
