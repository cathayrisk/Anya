import streamlit as st
import tempfile
import os
from langchain.prompts import ChatPromptTemplate
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_community.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain_community.document_loaders.powerpoint import UnstructuredPowerPointLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.parsers import LLMImageBlobParser
from langchain.chat_models import ChatOpenAI
from PIL import Image
from io import BytesIO
import base64
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

st.set_page_config(page_title="超萬用檔案轉 Markdown 工具", page_icon="🦾", layout="wide")
st.title("🦾 超萬用檔案轉 Markdown 工具")
st.write("上傳 PDF、Word、PPT、Excel、TXT，安妮亞幫你全部轉成 Markdown！🥜")

extract_images = st.checkbox("所有 PDF 檔案都要抽取圖片（用 LLM 分析）", value=True)

uploaded_files = st.file_uploader(
    "請選擇檔案（支援 PDF, Word, PPT, Excel, TXT）",
    type=["pdf", "docx", "doc", "pptx", "xlsx", "xls", "txt", "jpg", "jpeg", "png"],
    accept_multiple_files=True
)

convert_btn = st.button("🚀 開始轉換")

custom_prompt = """
# Role and Objective
You are an assistant tasked with summarizing images for retrieval.
1. These summaries will be embedded and used to retrieve the raw image. Give a concise summary of the image that is well optimized for retrieval.
2. Extract all the text from the image. Do not exclude any content from the page.
3. If the image contains tables, charts, or diagrams, describe their structure and content clearly, and use Markdown table syntax for tables.
4. If the image contains handwriting, signatures, stamps, or barcodes, please identify and transcribe them as accurately as possible, and clearly indicate their presence.
5. If any text or content is unclear or cannot be recognized, mark it as [無法辨識].
6. For complex images, describe the content by regions (e.g., top-left, center, bottom-right) if helpful.
7. Please first provide a one-sentence summary of the image’s main topic or purpose, then list all extracted text and key visual elements.
8. If possible, briefly infer the possible use case or context of the image.
9. Do not add any explanations, commentary, or extra formatting.
10. Format your answer in Markdown, but do not include any Markdown code block delimiters (no triple backticks).
11. **If the source is in Chinese, please respond in Traditional Chinese, not Simplified Chinese.**

# Output Format
1. 圖片主題摘要：
2. 圖片用途推測：
3. 圖片中所有可見文字（逐條列出）：
4. 表格內容（如有，請用 Markdown 表格）：
5. 其他重要細節（如手寫、印章、條碼等）：

# Example
## 範例1：表格圖片
1. 圖片主題摘要：這是一張年度銷售數據表格。
2. 圖片用途推測：用於公司年度報告。
3. 圖片中所有可見文字：
   - 年份
   - 銷售額
   - 2021
   - 100萬
   - 2022
   - 120萬
4. 表格內容：
   | 年份 | 銷售額 |
   |------|--------|
   | 2021 | 100萬  |
   | 2022 | 120萬  |
5. 其他重要細節：無

## 範例2：有手寫簽名的表單
1. 圖片主題摘要：這是一份員工資料表單。
2. 圖片用途推測：用於人事資料登記。
3. 圖片中所有可見文字：
   - 姓名：王小明
   - 部門：行銷部
   - 員工編號：A12345
   - 簽名：王小明（手寫）
4. 表格內容：無
5. 其他重要細節：有手寫簽名

# Final Reminder
**若辨識的文字是中文，請務必使用正體中文，嚴禁使用簡體字。**
"""

max_tokens = 2048

if uploaded_files and convert_btn:
    for uploaded_file in uploaded_files:
        st.markdown(f"---\n### 處理檔案：:blue[{uploaded_file.name}]")
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()

        # 圖片檔案直接用 LLM 處理（用 ChatPromptTemplate）
        if file_ext in [".jpg", ".jpeg", ".png"]:
            image = Image.open(uploaded_file)
            st.image(image, caption="上傳的圖片", use_container_width=True)

            # 轉成 base64 並包成 OpenAI 支援的 data URL
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            img_url = f"data:image/png;base64,{img_str}"

            # 用 ChatPromptTemplate 組 prompt
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are an OCR-like data extraction tool that extracts text from images."),
                    (
                        "user",
                        [
                            {
                                "type": "image_url",
                                "image_url": {"url": img_url},
                            },
                            {
                                "type": "text",
                                "text": 
                                    "Please extract all visible text from the image, including any small print or footnotes. "
                                    "Ensure no text is omitted, and provide a verbatim transcription of the document. "
                                    "Format your answer in Markdown, but do not include any Markdown code block delimiters (no triple backticks). "
                                    "Do not add any explanations or commentary."
                            }
                        ],
                    ),
                ]
            )

            # 產生 messages
            messages = prompt.format_messages()

            # 呼叫 LLM
            llm = ChatOpenAI(
                openai_api_key=st.secrets["OPENAI_KEY"],
                model="gpt-4.1-mini",  # 請用 vision 版本
                max_tokens=1024
            )
            response = llm.invoke(messages)

            full_markdown = f"---\nfile_name: {uploaded_file.name}\n---\n"
            full_markdown += response.content + "\n\n"

        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # 其他檔案維持原本 loader 流程
            if file_ext == ".pdf":
                loader_kwargs = {}
                if extract_images:
                    loader_kwargs["extract_images"] = True
                    loader_kwargs["images_parser"] = LLMImageBlobParser(
                        model=ChatOpenAI(
                            openai_api_key=st.secrets["OPENAI_KEY"],
                            model="gpt-4.1-mini",
                            max_tokens=max_tokens
                        ),
                        prompt=custom_prompt
                    )
                loader = PyMuPDF4LLMLoader(tmp_path, **loader_kwargs)
            elif file_ext in [".docx", ".doc"]:
                loader = UnstructuredWordDocumentLoader(tmp_path, mode="single")
            elif file_ext in [".pptx"]:
                loader = UnstructuredPowerPointLoader(tmp_path, mode="single")
            elif file_ext in [".xlsx", ".xls"]:
                loader = UnstructuredExcelLoader(tmp_path, mode="single")
            elif file_ext == ".txt":
                loader = TextLoader(tmp_path)
            else:
                st.error(f"不支援的檔案格式：{file_ext}")
                continue

            docs = loader.load()
            full_markdown = ""
            for i, doc in enumerate(docs):
                meta = doc.metadata
                file_name = meta.get('file_name', uploaded_file.name)
                page_number = meta.get('page_number', i+1)
                full_markdown += f"---\nfile_name: {file_name}\npage_number: {page_number}\n---\n"
                full_markdown += doc.page_content.strip() + "\n\n"

        st.success(f"轉換完成！")
        st.markdown(full_markdown)
        st.download_button(
            label=f"下載 Markdown（{uploaded_file.name}）",
            data=full_markdown.encode("utf-8"),
            file_name=f"{uploaded_file.name}.md",
            mime="text/markdown"
        )
