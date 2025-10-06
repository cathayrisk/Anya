import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph
from typing import List, Annotated, Literal, Sequence, TypedDict
from langgraph.graph import END, StateGraph, START
import asyncio
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from tavily import TavilyClient
import nltk
for pkg in ['punkt', 'averaged_perceptron_tagger']:
    try:
        path = f'tokenizers/{pkg}' if pkg == 'punkt' else f'taggers/{pkg}'
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg)


class RouteQuery(BaseModel):
    """將使用者的問題導向最相關的資料來源。"""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="依據使用者的問題，導向至vectorstore或web_search。",
    )


class GradeDocuments(BaseModel):
    """用於檢查所取得文件相關性的二元（「yes」或「no」）評分。"""

    binary_score: str = Field(
        description="判斷文件和問題有沒有相關，用「yes」或「no」表示。"
    )


class GradeHallucinations(BaseModel):
    """用於表示生成回答中是否存在幻覺的二元評分。"""

    binary_score: str = Field(
        description="判斷回答是否根據事實，用「yes」或「no」表示。"
    )

class GradeAnswer(BaseModel):
    """用於評估回答是否有回應問題的二元評分。"""

    binary_score: str = Field(
        description="判斷回答是否有回應問題，用「yes」或「no」表示"
    )

class GraphState(TypedDict):
    """
    用來表示GraphState的類別。

    属性:
        question: 問題
        generation: LLM生成內容
        documents: 文件列表
    """

    question: str
    generation: str
    documents: List[str]

import tempfile
from langchain_community.document_loaders import (
    PyMuPDFLoader,  # PDF
    UnstructuredWordDocumentLoader,  # Word
    UnstructuredPowerPointLoader,    # PPT
    UnstructuredExcelLoader,         # Excel
    TextLoader                       # TXT
)

# ----------- 新增：全域 vectorstore 初始化 -----------
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None

uploaded_files = st.file_uploader(
    "上傳知識文件（PDF, Word, PPT, Excel, TXT）",
    type=["pdf", "docx", "doc", "pptx", "xlsx", "xls", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        # Loader 判斷
        if file_ext == ".pdf":
            loader = PyMuPDFLoader(tmp_path)
        elif file_ext in [".docx", ".doc"]:
            loader = UnstructuredWordDocumentLoader(tmp_path, mode="single")
        elif file_ext == ".pptx":
            loader = UnstructuredPowerPointLoader(tmp_path, mode="single")
        elif file_ext in [".xlsx", ".xls"]:
            loader = UnstructuredExcelLoader(tmp_path, mode="single")
        elif file_ext == ".txt":
            loader = TextLoader(tmp_path)
        else:
            st.warning(f"不支援的檔案格式：{uploaded_file.name}")
            continue
        docs = loader.load()
        if not docs:   # ← 檢查文件內容
            st.warning(f"檔案 {uploaded_file.name} 沒有任何可讀內容，請換個檔案！")
            continue  # 跳過這個檔案

        # 分段
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
        splits = splitter.split_documents(docs)

        # 嵌入
        embd = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=st.secrets["OPENAI_KEY"])

        if st.session_state["vectorstore"] is None:
            st.session_state["vectorstore"] = FAISS.from_documents(splits, embd)
        else:
            st.session_state["vectorstore"].add_documents(splits)

        st.success(f"檔案 {uploaded_file.name} 已嵌入知識庫！")

async def route_question(state):
    st.session_state.status.update(label=f"**---ROUTE QUESTION---**", state="running", expanded=True)
    st.session_state.log += "---ROUTE QUESTION---" + "\n\n"
    llm = ChatOpenAI(openai_api_key=st.secrets["OPENAI_KEY"], model="gpt-4.1-mini", temperature=0)
    structured_llm_router = llm.with_structured_output(RouteQuery)

    system = """你是將使用者的問題導向vectorstore或web_search的專家。
    如果使用者能從資料庫中取得資訊請使用vectorstore；其他情況則請使用網路搜尋"""
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    question_router = route_prompt | structured_llm_router

    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "web_search":
        st.session_state.log += "---ROUTE QUESTION TO WEB SEARCH---" + "\n\n"
        st.session_state.placeholder.markdown("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    elif source.datasource == "vectorstore":
        st.session_state.placeholder.markdown("ROUTE QUESTION TO RAG")
        st.session_state.log += "ROUTE QUESTION TO RAG" + "\n\n"
        return "vectorstore"


async def retrieve(state):
    st.session_state.status.update(label=f"**---RETRIEVE---**", state="running", expanded=True)
    st.session_state.placeholder.markdown(f"RETRIEVING…\n\nKEY WORD:{state['question']}")
    st.session_state.log += f"RETRIEVING…\n\nKEY WORD:{state['question']}" + "\n\n"
    embd = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=st.secrets["OPENAI_KEY"])
    question = state["question"]

    # 🟢 新增：優先查詢本地 vectorstore
    if st.session_state.get("vectorstore", None) is not None:
        retriever = st.session_state["vectorstore"].as_retriever()
        documents = retriever.invoke(question)
        st.session_state.placeholder.markdown("RETRIEVE FROM USER UPLOADED VECTORSTORE SUCCESS!!")
        return {"documents": documents, "question": question}

async def web_search(state):
    st.session_state.status.update(label=f"**---WEB SEARCH---**", state="running", expanded=True)
    st.session_state.placeholder.markdown(f"WEB SEARCH…\n\nKEY WORD:{state['question']}")
    st.session_state.log += f"WEB SEARCH…\n\nKEY WORD:{state['question']}" + "\n\n"
    question = state["question"]

    # 使用 TavilyClient 初始化
    client = TavilyClient(api_key=st.secrets["Tavily_key"])

    # 執行搜尋，並使用一些可選參數
    response = client.search(
        question,
        search_depth="advanced",
        max_results=5,
    )
    # 假設 response["results"] 是一個列表，每個元素有 "content" 欄位
    web_results = "\n".join([item["content"] for item in response["results"]])
    web_results = Document(page_content=web_results)

    return {"documents": web_results, "question": question}

async def grade_documents(state):
    st.session_state.number_trial += 1
    llm = ChatOpenAI(openai_api_key=st.secrets["OPENAI_KEY"],model="gpt-4.1-mini", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    system = """你是負責評估所取得文件與使用者問題相關性的評分者。
如果文件中包含與使用者問題相關的關鍵字或意義，請評為有相關性。
目的是排除明顯錯誤的取得結果，不需要進行嚴格的測試。
請用二元評分「yes」或「no」來表示文件是否與問題相關。"""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader
    st.session_state.status.update(label=f"**---CHECK DOCUMENT RELEVANCE TO QUESTION---**", state="running", expanded=False)
    st.session_state.log += "**---CHECK DOCUMENT RELEVANCE TO QUESTION---**" + "\n\n"
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    i = 0
    for d in documents:
        if st.session_state.number_trial <= 2:
            file_name = d.metadata["source"]
            file_name = os.path.basename(file_name.replace("\\","/"))
            i += 1
            score = retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                st.session_state.status.update(label=f"**---GRADE: DOCUMENT RELEVANT---**", state="running", expanded=True)
                st.session_state.placeholder.markdown(f"DOC {i}/{len(documents)} : **RELEVANT**\n\n")
                st.session_state.log += "---GRADE: DOCUMENT RELEVANT---" + "\n\n"
                st.session_state.log += f"doc {i}/{len(documents)} : RELEVANT\n\n"
                filtered_docs.append(d)
            else:
                st.session_state.status.update(label=f"**---GRADE: DOCUMENT NOT RELEVANT---**", state="error", expanded=True)
                st.session_state.placeholder.markdown(f"DOC {i}/{len(documents)} : **NOT RELEVANT**\n\n")
                st.session_state.log += "---GRADE: DOCUMENT NOT RELEVANT---" + "\n\n"
                st.session_state.log += f"DOC {i}/{len(documents)} : NOT RELEVANT\n\n"
        else:

            filtered_docs.append(d)

    if not st.session_state.number_trial <= 2:
        st.session_state.status.update(label=f"**---NO NEED TO CHECK---**", state="running", expanded=True)
        st.session_state.placeholder.markdown("QUERY TRANSFORMATION HAS BEEN COMPLETED")
        st.session_state.log += "QUERY TRANSFORMATION HAS BEEN COMPLETED" + "\n\n"

    return {"documents": filtered_docs, "question": question}

async def generate(state):
    st.session_state.status.update(label=f"**---GENERATE---**", state="running", expanded=False)
    st.session_state.log += "---GENERATE---" + "\n\n"
    prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """請參考使用者提供的背景資訊來回答問題。"""),
                ("human", """Question: {question} 
Context: {context}"""),
            ]
        )
        
    llm = ChatOpenAI(openai_api_key=st.secrets["OPENAI_KEY"], model_name="gpt-4.1-mini", temperature=0)

    rag_chain = prompt | llm | StrOutputParser()
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


async def transform_query(state):
    st.session_state.status.update(label=f"**---TRANSFORM QUERY---**", state="running", expanded=True)
    st.session_state.placeholder.empty()
    st.session_state.log += "---TRANSFORM QUERY---" + "\n\n"
    llm = ChatOpenAI(openai_api_key=st.secrets["OPENAI_KEY"], model="gpt-4.1-mini", temperature=0)

    system = """你是一位將輸入問題轉換成更適合vectorstore搜尋的優化版本的問題重寫者。
請閱讀問題，推論提問者的意圖或含義，並產生更適合vectorstore搜尋的問題。"""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )

    question_rewriter = re_write_prompt | llm | StrOutputParser()
    question = state["question"]
    documents = state["documents"]
    better_question = question_rewriter.invoke({"question": question})
    st.session_state.log += f"better_question : {better_question}\n\n"
    st.session_state.placeholder.markdown(f"better_question : {better_question}")
    return {"documents": documents, "question": better_question}


async def decide_to_generate(state):
    filtered_documents = state["documents"]
    if not filtered_documents:
        st.session_state.status.update(label=f"**---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---**", state="error", expanded=False)
        st.session_state.log += "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---" + "\n\n"
        return "transform_query"                                     
    else:
        st.session_state.status.update(label=f"**---DECISION: GENERATE---**", state="running", expanded=False)
        st.session_state.log += "---DECISION: GENERATE---" + "\n\n"
        return "generate"

async def grade_generation_v_documents_and_question(state):
    st.session_state.number_trial += 1
    st.session_state.status.update(label=f"**---CHECK HALLUCINATIONS---**", state="running", expanded=False)
    st.session_state.log += "---CHECK HALLUCINATIONS---" + "\n\n"
    llm = ChatOpenAI(openai_api_key=st.secrets["OPENAI_KEY"], model="gpt-4.1-mini", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)

    system = """你是負責評估大型語言模型（LLM）生成內容是否根據／受到所取得事實集合支持的評分者。
請給予二元評分「yes」或「no」。「yes」表示回答是根據事實集合產生或受到其支持。"""
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )
    llm = ChatOpenAI(openai_api_key=st.secrets["OPENAI_KEY"],model="gpt-4.1-mini", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeAnswer)

    system = """你是負責評估回答是否有回應／解決問題的評分者。
請給予二元評分「yes」或「no」。「yes」表示回答有解決問題。"""
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )

    answer_grader = answer_prompt | structured_llm_grader
    hallucination_grader = hallucination_prompt | structured_llm_grader
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score
    if st.session_state.number_trial <= 3:
        if grade == "yes":
            st.session_state.placeholder.markdown("DECISION: ANSWER IS BASED ON A SET OF FACTS")
            st.session_state.log += "---DECISION: ANSWER IS BASED ON A SET OF FACTS---" + "\n\n"
            st.session_state.log += "---GRADE GENERATION vs QUESTION---" + "\n\n"
            score = answer_grader.invoke({"question": question, "generation": generation})
            grade = score.binary_score
            st.session_state.status.update(label=f"**---GRADE GENERATION vs QUESTION---**", state="running", expanded=True)
            if grade == "yes":
                st.session_state.status.update(label=f"**---DECISION: GENERATION ADDRESSES QUESTION---**", state="running", expanded=True)
                with st.session_state.placeholder:
                    st.markdown("**USEFUL!!**")
                    st.markdown(f"question : {question}")
                    st.markdown(f"generation : {generation}")                   
                    st.session_state.log += "---DECISION: GENERATION ADDRESSES QUESTION---" + "\n\n"
                    st.session_state.log += f"USEFUL!!\n\n"
                    st.session_state.log += f"question:{question}\n\n"
                    st.session_state.log += f"generation:{generation}\n\n"
                return "useful"
            else:
                st.session_state.number_trial -= 1
                st.session_state.status.update(label=f"**---DECISION: GENERATION DOES NOT ADDRESS QUESTION---**", state="error", expanded=True)
                with st.session_state.placeholder:
                    st.markdown("**NOT USEFUL**")
                    st.markdown(f"question:{question}")
                    st.markdown(f"generation:{generation}")
                    st.session_state.log += "---DECISION: GENERATION DOES NOT ADDRESS QUESTION---" + "\n\n"
                    st.session_state.log += f"NOT USEFUL\n\n"
                    st.session_state.log += f"question:{question}\n\n"
                    st.session_state.log += f"generation:{generation}\n\n"
                return "not useful"
        else:
            st.session_state.status.update(label=f"**---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---**", state="error", expanded=True)
            with st.session_state.placeholder:
                st.markdown("not grounded")
                st.markdown(f"question:{question}")
                st.markdown(f"generation:{generation}")
                st.session_state.log += "---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---" + "\n\n"
                st.session_state.log += f"not grounded\n\n"
                st.session_state.log += f"question:{question}\n\n"
                st.session_state.log += f"generation:{generation}\n\n"
            return "not supported"
    else:
        st.session_state.status.update(label=f"**---NO NEED TO CHECK---**", state="running", expanded=False)
        st.session_state.placeholder.markdown("TRIAL LIMIT EXCEEDED")
        st.session_state.log += "---NO NEED TO CHECK---" + "\n\n"
        st.session_state.log += "TRIAL LIMIT EXCEEDED" + "\n\n"
        return "useful"

async def run_workflow(inputs):
    st.session_state.number_trial = 0
    with st.status(label="**GO!!**", expanded=True,state="running") as st.session_state.status:
        st.session_state.placeholder = st.empty()
        value = await st.session_state.workflow.ainvoke(inputs)

    st.session_state.placeholder.empty()
    st.session_state.message_placeholder = st.empty()
    st.session_state.status.update(label="**FINISH!!**", state="complete", expanded=False)
    #st.session_state.message_placeholder.markdown(value["generation"])
    #with st.popover("log"):
    #    st.markdown(st.session_state.log)
    return value   # <== 加這行！！

def st_rag_langgraph():

    if 'log' not in st.session_state:
        st.session_state.log = ""

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    if 'status_container' not in st.session_state:
        st.session_state.status_container = st.empty()

    if not hasattr(st.session_state, "workflow"):

        workflow = StateGraph(GraphState)

        workflow.add_node("web_search", web_search)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("grade_documents", grade_documents)
        workflow.add_node("generate", generate)
        workflow.add_node("transform_query", transform_query)

        workflow.add_conditional_edges(
            START,
            route_question,
            {
                "vectorstore": "retrieve",
                "web_search": "web_search",
            },
        )
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_conditional_edges(
            "generate",
            grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "transform_query",
            },
        )

        app = workflow.compile()
        app = app.with_config(recursion_limit=10,run_name="Agent",tags=["Agent"])
        app.name = "Agent"
        st.session_state.workflow = app


    st.title("Adaptive RAG by LangGraph")

    # 顯示全部歷史
    for entry in st.session_state['chat_history']:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])

    if prompt := st.chat_input("請輸入問題"):
        st.session_state.log = ""
        # 新留言存進歷史
        st.session_state['chat_history'].append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # 建立對話上下文串
        conversation = []
        for turn in st.session_state['chat_history']:
            conversation.append({"role": turn["role"], "content": turn["content"]})

        inputs = {"question": prompt, "chat_history": conversation}
        value = asyncio.run(run_workflow(inputs))   # <<<< 拿 value["generation"]

        # <==== 這邊是關鍵，把AI回答加入chat_history
        st.session_state['chat_history'].append({"role": "assistant", "content": value["generation"]})

        # 也建議當下直接展示
        with st.chat_message("assistant"):
            st.markdown(value["generation"])

if __name__ == "__main__":
    st_rag_langgraph()
