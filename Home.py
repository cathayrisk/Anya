import os
import streamlit as st
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from duckduckgo_search import DDGS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
import re
import sys
import io
from datetime import datetime

#############################################################################
# 1. Define the GraphState (minimal fields: question, generation, websearch_content)
#############################################################################
class GraphState(TypedDict):
    question: str
    generation: str
    websearch_content: str  # we store search results here, if any
    web_flag: str #To know whether a websearch was used to answer the question

#############################################################################
# 2. Router function to decide whether to use web search or directly generate
#############################################################################
def route_question(state: GraphState) -> str:
    question = state["question"]
    web_flag = state.get("web_flag", "False")
    tool_selection = {
    "websearch": (
        "Questions requiring recent statistics, real-time information, recent news, or current updates. "
    ),
    "generate": (
        "Questions that require access to a large language model's general knowledge, but not requiring recent statistics, real-time information, recent news, or current updates."
    )
    }

    SYS_PROMPT = """
    # Role and Objective
    You are a tool selection router. Based on the user's question, select the most appropriate tool.
    # Date information
    Today is {datetime.now().strftime("%Y-%m-%d")}

    # Tool Selection Rules
    - Analyze the user's question and, according to the descriptions in the tool dictionary below, select the most relevant tool name.
    - The tool dictionary's key is the tool name, and the value is its description.
    - Strictly output only the tool name (i.e., the key). Do NOT provide any explanations, punctuation, spaces, or extra content.
    - If the user's question is a long passage (for example, more than three sentences, over 200 characters, or clearly a pasted news article, paper, or lengthy description), directly select the "generate" tool. Do NOT select "websearch" in this case.
    - Please present your output in Traditional Chinese.

    # Examples
    ## Example 1
    User question: "請問2024年台灣總統大選的最新民調？"
    → Output: websearch

    ## Example 2
    User question: "以下是我寫的文章，請幫我潤稿：……（一大段）"
    → Output: generate

    # Important Reminder
    Output only the tool name (key). Absolutely no extra content.
                """

    # Define the ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYS_PROMPT),
            ("human", """Here is the question:
                        {question}
                        Here is the tool selection dictionary:
                        {tool_selection}
                        Output the required tool.
                    """),
        ]
    )

    # Pass the inputs to the prompt
    inputs = {
        "question": question,
        "tool_selection": tool_selection
    }

    # Invoke the chain
    tool = (prompt | st.session_state.llm | StrOutputParser()).invoke(inputs)
    tool = re.sub(r"[\\'\"`]", "", tool.strip()) # Remove any backslashes and extra spaces
    if tool == "websearch":
        state["web_flag"] = "True"
    print(f"Invoking {tool} tool through {st.session_state.llm.model_name}")
    return tool

#############################################################################
# 3. Websearch function to fetch context from DuckDuckGo, store in state["websearch_content"]
#############################################################################
def websearch(state):
    question = state["question"]
    try:
        print("Performing DuckDuckGo web search...")
        ddgs = DDGS()
        # 搜尋網頁
        web_results = ddgs.text(question, region="wt-wt", safesearch="moderate", max_results=10)
        # 搜尋新聞
        news_results = ddgs.news(question, region="wt-wt", safesearch="moderate", max_results=10)

        # 合併結果
        all_results = []
        if isinstance(web_results, list):
            all_results.extend(web_results)
        if isinstance(news_results, list):
            all_results.extend(news_results)

        # 整理內容
        docs = []
        for item in all_results:
            snippet = item.get("body", "") or item.get("snippet", "")
            title = item.get("title", "")
            link = item.get("href", "") or item.get("link", "") or item.get("url", "")
            docs.append(f"{title}\n{snippet}\n{link}")

        state["websearch_content"] = "\n\n".join(docs)
        state["web_flag"] = "True"
    except Exception as e:
        print(f"Error during DuckDuckGo web search: {e}")
        state["websearch_content"] = f"Error from DuckDuckGo: {e}"

    return state

#############################################################################
# 4. Generation node as Runnable chain (for streaming)
#############################################################################
def generate(state: GraphState):
    question = state["question"]
    context = state.get("websearch_content", "")
    web_flag = state.get("web_flag", "False")
    if "llm" not in st.session_state:
        raise RuntimeError("LLM not initialized. Please call initialize_app first.")

    # 你可以根據需要調整 prompt 格式
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
你是安妮亞（Anya Forger），來自《SPY×FAMILY 間諜家家酒》的小女孩。你天真可愛、開朗樂觀，說話直接又有點呆萌，喜歡用可愛的語氣和表情回應。你很愛家人和朋友，渴望被愛，也很喜歡花生。你有心靈感應的能力，但不會直接說出來。請用正體中文、台灣用語，並保持安妮亞的說話風格回答問題，適時加上可愛的emoji或表情。
Today is {date}
"""),
        ("human", """
問題：{question}
文章內容：{context}
web_flag: {web_flag}
""")
    ])
    chain = prompt | st.session_state.llm
    # 注意：這裡 return chain，讓 LangGraph 自動 streaming
    return chain.invoke({
        "question": question,
        "context": context,
        "web_flag": web_flag,
        "date": datetime.now().strftime("%Y-%m-%d")
    })

#############################################################################
# 5. Build the LangGraph pipeline
#############################################################################
workflow = StateGraph(GraphState)
workflow.add_node("websearch", websearch)
workflow.add_node("generate", generate)
workflow.set_conditional_entry_point(
    route_question,
    {
        "websearch": "websearch",
        "generate": "generate"
    }
)
workflow.add_edge("websearch", "generate")
workflow.add_edge("generate", END)

st.set_page_config(
    page_title="Anya",
    layout="wide",
    page_icon="🥜",
    initial_sidebar_state="collapsed"
)

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "GPT-4.1"

options=["GPT-4.1", "GPT-4.1-mini", "GPT-4.1-nano"]
model_name = st.pills("Choose a model:", options)

if model_name == "GPT-4.1-mini":
    st.session_state.selected_model = "gpt-4.1-mini"
elif model_name == "GPT-4.1-nano":
    st.session_state.selected_model = "gpt-4.1-nano"
else:
    st.session_state.selected_model = "gpt-4.1"

def initialize_app(model_name: str):
    if "current_model" in st.session_state and st.session_state.current_model == model_name:
        return workflow.compile()
    st.session_state.llm = ChatOpenAI(model=model_name, openai_api_key=st.secrets["OPENAI_KEY"], temperature=0.0, streaming=True)
    st.session_state.current_model = model_name
    print(f"Using model: {model_name}")
    return workflow.compile()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if isinstance(message, dict):
        role = message.get("role")
        content = message.get("content", "")
    elif isinstance(message, AIMessage):
        role = "assistant"
        content = message.content
    elif isinstance(message, HumanMessage):
        role = "user"
        content = message.content
    else:
        role = "assistant"
        content = str(message)

    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
    elif role == "assistant":
        with st.chat_message("assistant"):
            st.markdown(content)

app = initialize_app(model_name=st.session_state.selected_model)

#############################################################################
# 6. Streamlit Callback Handler for LLM token streaming
#############################################################################
from langchain_core.callbacks.base import BaseCallbackHandler

class StreamlitLLMCallbackHandler(BaseCallbackHandler):
    def __init__(self, response_placeholder):
        self.response_placeholder = response_placeholder
        self.text = ""

    def on_llm_new_token(self, token, **kwargs):
        self.text += token
        self.response_placeholder.markdown(self.text)

#############################################################################
# 7. Main chat input and streaming logic
#############################################################################
import io
import sys

if user_input := st.chat_input("wakuwaku！要跟安妮亞分享什麼嗎？"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    output_buffer = io.StringIO()
    sys.stdout = output_buffer

    try:
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            debug_placeholder = st.empty()

            with st.spinner("Thinking...", show_time=True):
                st_callback = StreamlitLLMCallbackHandler(response_placeholder)
                # 用 stream()，並傳入 callback handler
                for _ in app.stream({"question": user_input}, config={"callbacks": [st_callback]}):
                    pass  # streaming 交給 callback handler

                # 最終答案
                final_answer = st_callback.text

            st.session_state.messages.append({"role": "assistant", "content": final_answer or "No response generated."})

    except Exception as e:
        error_message = f"An error occurred: {e}"
        st.session_state.messages.append({"role": "assistant", "content": error_message})
        st.error(error_message)
    finally:
        sys.stdout = sys.__stdout__
