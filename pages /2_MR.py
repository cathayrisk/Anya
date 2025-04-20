import streamlit as st
import os
import tempfile
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langchain_core.documents import Document
from langchain.chains.combine_documents.reduce import (
    acollapse_docs,
    split_list_of_docs,
)
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from typing import List, TypedDict, Literal, Annotated
import operator
import asyncio
import pymupdf4llm
import faiss
import time
from uuid import uuid4

# 初始化 OpenAI 客戶端
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_KEY"]

# 初始化 LangChain 的 ChatOpenAI 模型，設置超時
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, streaming=True)

# 設置網頁標題和圖標
st.set_page_config(page_title="Generate_Summary", layout="wide", page_icon="📝")

# 使用 expander 控制文件上傳部分
with st.expander("Upload your PDF files", expanded=True):
    uploaded_files = st.file_uploader("", type=["pdf"], accept_multiple_files=True, label_visibility="hidden")

# 分割文件
text_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        "\u200b",  # 零寬空格
        "\uff0c",  # 全角逗號
        "\u3001",  # 句讀點
        "\uff0e",  # 全角句號
        "\u3002",  # 句號
        ""
    ],
    chunk_size=500,  # 根據需要調整
    chunk_overlap=20  # 根據需要調整
)

token_max = 100000

# 定義狀態類型
class SummaryState(TypedDict):
    content: str

class OverallState(TypedDict):
    contents: List[str]
    summaries: Annotated[List[str], operator.add]  # 使用 Annotated 來允許多個值
    collapsed_summaries: List[Document]
    final_summary: str

# 生成摘要
async def generate_summary(state: SummaryState):
    try:
        #st.write(f"Generating summary for content: {state['content'][:100]}...")  # 打印前100個字符
        response = await map_chain.ainvoke(state["content"])
        if not response:  # 檢查返回的摘要是否為空
            st.write("Warning: Empty summary generated for content.")
        return {"summaries": [response]}
    except Exception as e:
        st.write(f"Error generating summary: {e}")
        return {"summaries": []}

# 映射摘要
def map_summaries(state: OverallState):
    return [Send("generate_summary", {"content": content}) for content in state["contents"]]

# 計算文件長度
def length_function(documents: List[Document]) -> int:
    return sum(llm.get_num_tokens(doc.page_content) for doc in documents)

# 收集摘要
def collect_summaries(state: OverallState):
    return {"collapsed_summaries": [Document(summary) for summary in state["summaries"]]}

# 生成最終摘要
async def generate_final_summary(state: OverallState):
    if not state['summaries']:
        return {"final_summary": "當前沒有提供具體的主題摘要，因此無法進行整合和詳細總結。"}

    # 準備輸入給 reduce_chain
    collapsed_summaries = [Document(page_content=summary) for summary in state['summaries']]
    state['collapsed_summaries'] = collapsed_summaries

    #st.write(f"Generating final summary for {len(collapsed_summaries)} summaries.")

    try:
        response = await reduce_chain.ainvoke(collapsed_summaries)
        return {"final_summary": response}
    except Exception as e:
        st.write(f"Error generating final summary: {e}")
        return {"final_summary": "無法生成最終摘要，請檢查輸入內容。"}
    
# 摺疊摘要
async def collapse_summaries(state: OverallState):
    doc_lists = split_list_of_docs(state["collapsed_summaries"], length_function, token_max)
    results = []
    for doc_list in doc_lists:
        results.append(await acollapse_docs(doc_list, reduce_chain.ainvoke))
    return {"collapsed_summaries": results}

# 判斷是否需要摺疊
def should_collapse(state: OverallState) -> Literal["collapse_summaries", "generate_final_summary"]:
    num_tokens = length_function(state["collapsed_summaries"])
    if num_tokens > token_max:
        return "collapse_summaries"
    else:
        return "generate_final_summary"

# 建立狀態圖
graph = StateGraph(OverallState)
graph.add_node("generate_summary", generate_summary)
graph.add_node("collect_summaries", collect_summaries)
graph.add_node("generate_final_summary", generate_final_summary)
graph.add_node("collapse_summaries", collapse_summaries)

# 添加邊
graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
graph.add_edge("generate_summary", "collect_summaries")
graph.add_conditional_edges("collect_summaries", should_collapse)
graph.add_conditional_edges("collapse_summaries", should_collapse)
graph.add_edge("generate_final_summary", END)

# 編譯應用程式
app = graph.compile()

# 定義提示模板
map_template = """
Please read the following content and identify the main themes and key insights especially in risk related. Focus on extracting the unique insights and specific details that are central to the document's purpose.
- Use bullet points to list key insights and specific details for each theme.
- Include specific data, examples, and references to support each theme, ensuring a rich level of detail.
- Explain the cause-and-effect relationships for each key point, highlighting how different factors interact.
- Pay attention to the tone and style to ensure consistency with the conversational nature of the recording.
- Identify and highlight any important action items or decisions, and explain the rationale behind them.
- Summarize the key points that the document most wants to convey.
Please respond in Traditional Chinese, not Simplified Chinese.
Content: {context}
"""

reduce_template = """
The following are summaries of key themes extracted from the document:
{docs}
Please consolidate these into broader categories, focusing on grouping related themes under major categories. For each major category, provide a detailed summary using bullet points to highlight key insights, specific details, and explain the cause-and-effect relationships.
- Ensure all important details and examples are included to support each category's summary, enhancing the richness of the content.
- Maintain consistency in tone and style, reflecting the conversational nature of the recording.
- Ensure identification and emphasis on any important action items or decisions, and provide explanations for these recommendations.
- Summarize the key points that the document most wants to convey, without assuming additional strategic implications unless explicitly mentioned.
Please respond in Traditional Chinese, not Simplified Chinese.
"""

map_prompt = ChatPromptTemplate([("human", map_template)])
reduce_prompt = ChatPromptTemplate([("human", reduce_template)])

map_chain = map_prompt | llm | StrOutputParser()
reduce_chain = reduce_prompt | llm | StrOutputParser()

# 定義運行應用程式的異步函數
async def run_app(split_docs):
    contents = [doc.page_content for doc in split_docs]
    #st.write(f"Processing {len(contents)} chunks.")
    tasks = [generate_summary({"content": content}) for content in contents]
    summaries = await asyncio.gather(*tasks)

    # 檢查 summaries 是否包含有效的摘要
    valid_summaries = [summary['summaries'][0] for summary in summaries if summary['summaries']]
    if not valid_summaries:
        st.write("No valid summaries generated.")
        return "當前沒有提供具體的主題摘要，因此無法進行整合和詳細總結。"

    #st.write(f"Valid summaries: {valid_summaries}")
    overall_state = OverallState(contents=contents, summaries=valid_summaries, collapsed_summaries=[], final_summary="")
    final_summary = await generate_final_summary(overall_state)
    return final_summary['final_summary']

# 處理上傳的文件
if uploaded_files:
    # 按鈕來觸發摘要生成
    if st.button("生成摘要"):
        try:
            all_transcriptions = []
            for uploaded_file in uploaded_files:
                # Save each uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf_file:
                    temp_pdf_file.write(uploaded_file.read())
                    temp_pdf_path = temp_pdf_file.name

                # 提取 PDF 內容為 Markdown
                md_text_list = pymupdf4llm.to_markdown(
                    temp_pdf_path,
                    write_images=False,  # 不提取圖像
                    page_chunks=True  # 按頁面分塊輸出
                )

                # 將提取的文本合併
                full_transcription = "\n".join(page.get('text', '') for page in md_text_list if isinstance(page, dict))
                all_transcriptions.append(full_transcription)

            # 合併所有文件的轉錄文本
            combined_transcription = "\n".join(all_transcriptions)

            # 顯示 formatted_transcription
            tab1, tab2 = st.tabs(["重點摘要", "原始內容"])
            with tab1:
                # 異步計算 summarize_transcription 並在 Tab2 中顯示 spinner
                async def calculate_summary():
                    # 分割轉錄文本並包裝成 Document 對象
                    start_time = time.time()  # 開始時間
                    split_docs = [Document(page_content=content) for content in text_splitter.split_text(combined_transcription)]
                    summarize_transcription = await run_app(split_docs)
                    end_time = time.time()  # 結束時間
                    st.write(f"calculate_summary took {end_time - start_time:.2f} seconds")
                    st.session_state['summarize_transcription'] = summarize_transcription
                    return summarize_transcription

                with st.spinner('Generating summary...'):
                    summarize_transcription = asyncio.run(calculate_summary())
                    st.markdown(summarize_transcription)

            with tab2:
                with st.container():
                    st.markdown(combined_transcription)

        except Exception as e:
            st.error(f"An error occurred: {e}")

else:
    st.info("Please upload PDF files to start the analysis.")
