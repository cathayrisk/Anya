import streamlit as st
import tempfile
import fitz  # pymupdf
import difflib
import pandas as pd
import html

st.set_page_config(page_title="安妮亞來找碴", layout="wide")
st.title("文件差異比對工具（高亮、上下文、分頁）")

def extract_pdf_text(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf_file:
        temp_pdf_file.write(uploaded_file.read())
        temp_pdf_path = temp_pdf_file.name
    doc = fitz.open(temp_pdf_path)
    doc_text = ""
    for page_number, page in enumerate(doc):
        text = page.get_text("text")
        doc_text += f"\n--- Page {page_number+1} ---\n{text}"
    doc.close()
    return doc_text

def highlight_diff(a, b):
    """將a, b兩行的差異部分用<mark>高亮"""
    sm = difflib.SequenceMatcher(None, a, b)
    a_out, b_out = "", ""
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        a_part = html.escape(a[i1:i2])
        b_part = html.escape(b[j1:j2])
        if tag == "equal":
            a_out += a_part
            b_out += b_part
        elif tag == "replace":
            a_out += f"<mark>{a_part}</mark>"
            b_out += f"<mark>{b_part}</mark>"
        elif tag == "delete":
            a_out += f"<mark>{a_part}</mark>"
        elif tag == "insert":
            b_out += f"<mark>{b_part}</mark>"
    return a_out, b_out

def extract_diff_dataframe_v3(text1, text2, context_lines=1):
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
                hl1, hl2 = highlight_diff(l1, l2)
                # 上下文
                ctx_start = max(i1 + k - context_lines, 0)
                ctx_end = min(i1 + k + context_lines + 1, len(lines1))
                context = "<br>".join(html.escape(line) for line in lines1[ctx_start:ctx_end])
                diff_rows.append({
                    "上下文": context,
                    "差異類型": "修改",
                    "文件1內容": hl1,
                    "文件2內容": hl2
                })
        elif tag == 'delete':
            for idx, l1 in enumerate(lines1[i1:i2]):
                hl1, _ = highlight_diff(l1, "")
                ctx_start = max(i1 + idx - context_lines, 0)
                ctx_end = min(i1 + idx + context_lines + 1, len(lines1))
                context = "<br>".join(html.escape(line) for line in lines1[ctx_start:ctx_end])
                diff_rows.append({
                    "上下文": context,
                    "差異類型": "刪除",
                    "文件1內容": hl1,
                    "文件2內容": ""
                })
        elif tag == 'insert':
            for idx, l2 in enumerate(lines2[j1:j2]):
                _, hl2 = highlight_diff("", l2)
                # 用 lines1 的前後行作為上下文
                ctx_start = max(i1 - context_lines, 0)
                ctx_end = min(i1 + context_lines + 1, len(lines1))
                context = "<br>".join(html.escape(line) for line in lines1[ctx_start:ctx_end]) if lines1 else ""
                diff_rows.append({
                    "上下文": context,
                    "差異類型": "新增",
                    "文件1內容": "",
                    "文件2內容": hl2
                })
        # 'equal' 不顯示
    df = pd.DataFrame(diff_rows)
    # 刪除兩欄都為空的列
    df = df[~((df['文件1內容'] == "") & (df['文件2內容'] == ""))]
    df = df.reset_index(drop=True)
    return df

# 上傳區塊
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

# 比對與顯示
if file1 and file2:
    st.markdown("---")
    st.subheader("比對差異表格（高亮、上下文、分頁）")
    with st.spinner("正在抽取 PDF 文字..."):
        doc1_text = extract_pdf_text(file1)
        doc2_text = extract_pdf_text(file2)

    if st.button("開始比對並顯示所有差異"):
        with st.spinner("正在比對..."):
            df = extract_diff_dataframe_v3(doc1_text, doc2_text, context_lines=1)
            st.write(f"本次比對共發現 {len(df)} 處差異。")
            if len(df) == 0:
                st.info("兩份文件沒有明顯差異。")
            else:
                # 分頁
                page_size = 20
                total_pages = (len(df) - 1) // page_size + 1
                page = st.number_input("選擇頁數", min_value=1, max_value=total_pages, value=1)
                start = (page - 1) * page_size
                end = start + page_size
                show_df = df.iloc[start:end]
                # 用 HTML 表格高亮顯示
                def row_to_html(row):
                    return f"""
                    <tr>
                        <td style='white-space:pre-wrap'>{row['上下文']}</td>
                        <td style='white-space:pre-wrap'>{row['差異類型']}</td>
                        <td style='white-space:pre-wrap'>{row['文件1內容']}</td>
                        <td style='white-space:pre-wrap'>{row['文件2內容']}</td>
                    </tr>
                    """
                html_table = """
                <table border="1" style="border-collapse:collapse;">
                    <tr>
                        <th>上下文</th>
                        <th>差異類型</th>
                        <th>文件1內容</th>
                        <th>文件2內容</th>
                    </tr>
                """
                for _, row in show_df.iterrows():
                    html_table += row_to_html(row)
                html_table += "</table>"
                st.markdown(html_table, unsafe_allow_html=True)
else:
    st.info("請分別上傳文件1與文件2")
