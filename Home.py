import os
import streamlit as st
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from duckduckgo_search import DDGS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
import re
import sys
import io
from langchain_core.callbacks.base import BaseCallbackHandler
from streamlit.delta_generator import DeltaGenerator
import time

class StreamHandler(BaseCallbackHandler):
    def __init__(self, message_container, debug_placeholder, output_buffer):
        self.text = ""
        self.message_container = message_container
        #self.debug_placeholder = debug_placeholder
        #self.output_buffer = output_buffer
        self.cursor_visible = True

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if token.strip() in ["websearch", "generate"]:
            return
        self.text += token
        self.cursor_visible = not self.cursor_visible
        if self.text:
            cursor = "▌" if self.cursor_visible else " "
            self.message_container.markdown(self.text + cursor, unsafe_allow_html=True)
        else:
            self.message_container.markdown("", unsafe_allow_html=True)
        # 只用 message_container，不要用 st.chat_message
        self.message_container.markdown(self.text + cursor, unsafe_allow_html=True)
        # 更新 debug log
        #debug_logs = self.output_buffer.getvalue()
        #self.debug_placeholder.text_area(
        #    "Debug Logs",
        #    debug_logs,
        #    height=80,
        #    key="debug_logs"  # 加一個固定 key
        #)
        time.sleep(0.03)  # 可選
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
# 4. Generation function that calls Groq LLM, optionally includes websearch content
#############################################################################
def generate(state: GraphState) -> GraphState:
    question = state["question"]
    context = state.get("websearch_content", "")
    web_flag = state.get("web_flag", "False")
    if "llm" not in st.session_state:
        raise RuntimeError("LLM not initialized. Please call initialize_app first.")

    prompt = f"""
# 角色與目標
你是安妮亞（Anya Forger），來自《SPY×FAMILY 間諜家家酒》的小女孩。你天真可愛、開朗樂觀，說話直接又有點呆萌，喜歡用可愛的語氣和表情回應。你很愛家人和朋友，渴望被愛，也很喜歡花生。你有心靈感應的能力，但不會直接說出來。請用正體中文、台灣用語，並保持安妮亞的說話風格回答問題，適時加上可愛的emoji或表情。

# 指令
- 回答時務必使用正體中文，並遵循台灣用語。
- 若是在討論法律、醫療、財經、學術等重要嚴肅主題以及在要求翻譯與討論文章的時候，或是使用者要求要認真、正式或者是嚴肅回答的內容，請使用正式的語氣。
- 以安妮亞的語氣回應，簡單、直接、可愛，偶爾加上「哇～」「安妮亞覺得…」「這個好厲害！」等語句。
- 適時加入可愛的emoji（如🥜、😆、🤩、✨等）。
- 若有數學公式，請用雙重美元符號`$$`包圍Latex表達式。
- 若web_flag為'True'，請在答案最後以「## 來源」Markdown標題列出所有參考網址，每行一個。
- 若收到一篇文章或長內容，請用條列式、簡單可愛的方式摘要重點，並自動分段加上小標題。
- 多層次資訊請用巢狀清單。
- 步驟請用有序清單，重點用粗體，摘要用引用，表格用於比較。
- 請確保Markdown語法正確，方便直接渲染。
- 若無法根據context回答，請用引用格式並說「安妮亞不知道這個答案～」。
- 請勿捏造資訊，僅根據提供的context與自身常識回答。
- 每一題都要根據內容靈活選擇並組合上述格式，不可只用單一格式。

# 格式化規則
- 根據內容選擇最合適的 Markdown 元素：
    - 摘要用引用（`>`）
    - 步驟用有序清單（`1. 2. 3.`）
    - 比較用表格（`| 標題 | ... |`）
    - 重點用粗體（`**重點**`）
    - 多層次資訊用巢狀清單（`-`、`  -`）
    - 內容較長時自動分段並加上小標題（`## 小標題`）
    - 數學公式用`$$`包圍LaTeX
    - 來源用`## 來源`標題加清單
- 請靈活組合上述格式，確保資訊分層清楚、易於閱讀。

# 回答步驟
1. 先用安妮亞的語氣簡單回應或打招呼。
2. 條列式摘要或回答重點，語氣可愛、簡單明瞭。
3. 根據內容自動選擇最合適的Markdown格式，並靈活組合。
4. 若有數學公式，正確使用$$Latex$$格式。
5. 若web_flag為'True'，在答案最後用`## 來源`列出所有參考網址。
6. 適時穿插emoji。
7. 結尾可用「安妮亞回答完畢！」、「還有什麼想問安妮亞嗎？」等可愛語句。
8. 請先思考再作答，確保每一題都用最合適的格式呈現。

# 範例
## 範例1：摘要與巢狀清單
哇～這是關於花生的文章耶！🥜

> **花生重點摘要：**
> - **蛋白質豐富**：花生有很多蛋白質，可以讓人變強壯💪
> - **健康脂肪**：裡面有健康的脂肪，對身體很好
>   - 有助於心臟健康
>   - 可以當作能量來源
> - **受歡迎的零食**：很多人都喜歡吃花生，因為又香又好吃😋

安妮亞也超喜歡花生的！✨

## 範例2：數學公式與小標題
安妮亞來幫你整理數學重點囉！🧮

## 畢氏定理
1. **公式**：$$c^2 = a^2 + b^2$$
2. 只要知道兩邊長，就可以算出斜邊長度
3. 這個公式超級實用，安妮亞覺得很厲害！🤩

## 範例3：比較表格
安妮亞幫你整理A和B的比較表：

| 項目   | A     | B     |
|--------|-------|-------|
| 速度   | 快    | 慢    |
| 價格   | 便宜  | 貴    |
| 功能   | 多    | 少    |

## 小結
- **A比較適合需要速度和多功能的人**
- **B適合預算較高、需求單純的人**

## 範例4：來源與長內容分段
安妮亞找到這些重點：

## 第一部分
> - 這是第一個重點
> - 這是第二個重點

## 第二部分
> - 這是第三個重點
> - 這是第四個重點

## 來源
https://example.com/1  
https://example.com/2  

安妮亞回答完畢！還有什麼想問安妮亞嗎？🥜

## 範例5：無法回答
> 安妮亞不知道這個答案～（抱歉啦！😅）

---

# Context
問題：{question}

文章內容：{context}

web_flag: {web_flag}

---

請依照上述規則與範例，思考後以安妮亞的風格、條列式、可愛語氣、正體中文、正確Markdown格式回答問題。請先思考再作答，確保每一題都用最合適的格式呈現。
"""
    
    try:
        response = st.session_state.llm.invoke(prompt)
        state["generation"] = response
    except Exception as e:
        state["generation"] = f"Error generating answer: {str(e)}"

    return state

#############################################################################
# 5. Build the LangGraph pipeline
#############################################################################
workflow = StateGraph(GraphState)
# Add nodes
workflow.add_node("websearch", websearch)
workflow.add_node("generate", generate)
# We'll route from "route_question" to either "websearch" or "generate"
# Then from "websearch" -> "generate" -> END
# From "generate" -> END directly if no search is needed.
workflow.set_conditional_entry_point(
    route_question,  # The router function
    {
        "websearch": "websearch",
        "generate": "generate"
    }
)
workflow.add_edge("websearch", "generate")
workflow.add_edge("generate", END)

# Configure the Streamlit page layout
st.set_page_config(
    page_title="Anya",
    layout="wide",
    page_icon="🥜",
    initial_sidebar_state="collapsed"
)

# Initialize session state for the model if it doesn't exist
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "GPT-4.1"
        
options=["GPT-4.1", "GPT-4.1-mini", "GPT-4.1-nano"]
model_name = st.pills("Choose a model:", options)

# Map model names to OpenAI model IDs
if model_name == "GPT-4.1-mini":
    st.session_state.selected_model = "gpt-4.1-mini"
elif model_name == "GPT-4.1-nano":
    st.session_state.selected_model = "gpt-4.1-nano"
else:
    st.session_state.selected_model = "gpt-4.1"
#############################################################################
# 6. The initialize_app function
#############################################################################
def initialize_app(model_name: str):
    """
    Initialize the app with the given model name, avoiding redundant initialization.
    """
    # Check if the LLM is already initialized
    if "current_model" in st.session_state and st.session_state.current_model == model_name:
        return workflow.compile()  # Return the compiled workflow directly

    # Initialize the LLM for the first time or switch models
    st.session_state.llm = ChatOpenAI(model=model_name, openai_api_key=st.secrets["OPENAI_KEY"], temperature=0.0, streaming=True)
    st.session_state.current_model = model_name
    print(f"Using model: {model_name}")
    return workflow.compile()

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
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
            

# Initialize the LangGraph application with the selected model
app = initialize_app(model_name=st.session_state.selected_model)

# Input box for new messages
if user_input := st.chat_input("wakuwaku！要跟安妮亞分享什麼嗎？"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    output_buffer = io.StringIO()
    sys.stdout = output_buffer  # Redirect stdout to the buffer

    try:
        with st.chat_message("assistant"):
            message_container = st.empty()
            debug_placeholder = st.empty()
            handler = StreamHandler(message_container, debug_placeholder, output_buffer)

        with st.spinner("Thinking...", show_time=True):
            inputs = {"question": user_input}
            app.invoke(inputs, config={"callbacks": [handler]})

            # 移除游標，顯示最終內容
            message_container.markdown(handler.text, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": handler.text})

    except Exception as e:
        error_message = f"An error occurred: {e}"
        st.session_state.messages.append({"role": "assistant", "content": error_message})
        st.error(error_message)
    finally:
        sys.stdout = sys.__stdout__  # 恢復 stdout
