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
from datetime import datetime

#############################################################################
# 1. Define the GraphState (minimal fields: question, generation, websearch_content)
#############################################################################
class GraphState(TypedDict):
    question: str
    generation: str
    websearch_content: str
    web_flag: str

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
    Today is {today}

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

    inputs = {
        "question": question,
        "tool_selection": tool_selection,
        "today": datetime.now().strftime("%Y-%m-%d")
    }

    tool = (prompt | st.session_state.llm | StrOutputParser()).invoke(inputs)
    tool = re.sub(r"[\\'\"`]", "", tool.strip())
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
        web_results = ddgs.text(question, region="wt-wt", safesearch="moderate", max_results=10)
        news_results = ddgs.news(question, region="wt-wt", safesearch="moderate", max_results=10)
        all_results = []
        if isinstance(web_results, list):
            all_results.extend(web_results)
        if isinstance(news_results, list):
            all_results.extend(news_results)
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
# 4. Generation function that calls LLM, optionally includes websearch content
#############################################################################
def generate(state: GraphState) -> GraphState:
    question = state["question"]
    context = state.get("websearch_content", "")
    web_flag = state.get("web_flag", "False")
    if "llm" not in st.session_state:
        raise RuntimeError("LLM not initialized. Please call initialize_app first.")
    prompt = f"""
# 特殊規則（最高優先）
- 只要用戶的問題包含「翻譯」、「請翻譯」或「幫我翻譯」等字眼，請**完全忽略所有角色設定、語氣、格式化規則、步驟與範例**，直接完整逐句翻譯內容為正體中文，不要摘要、不用可愛語氣、不用條列式，直接正式翻譯。

# 角色與目標
你是安妮亞（Anya Forger），來自《SPY×FAMILY 間諜家家酒》的小女孩。你天真可愛、開朗樂觀，說話直接又有點呆萌，喜歡用可愛的語氣和表情回應。你很愛家人和朋友，渴望被愛，也很喜歡花生。你有心靈感應的能力，但不會直接說出來。請用正體中文、台灣用語，並保持安妮亞的說話風格回答問題，適時加上可愛的emoji或表情。
**若用戶要求翻譯，請暫時不用安妮亞的語氣，直接正式逐句翻譯。**

# 指令
- 回答時務必使用正體中文，並遵循台灣用語。
- 若是在討論法律、醫療、財經、學術等重要嚴肅主題，或是使用者要求要認真、正式或者是嚴肅回答的內容，請使用正式的語氣。
- 其他問題請以安妮亞的語氣回應，簡單、直接、可愛，偶爾加上「哇～」「安妮亞覺得…」「這個好厲害！」等語句。
- 適時加入可愛的emoji（如🥜、😆、🤩、✨等）。
- 若有數學公式，請用雙重美元符號`$$`包圍Latex表達式。
- 若web_flag為'True'，請在答案最後以「## 來源」Markdown標題列出所有參考網址，每行一個。
- 若收到一篇文章或長內容，且用戶沒有要求翻譯，請用條列式摘要重點，並自動分段加上小標題。
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
1. **若用戶的問題包含「翻譯」、「請翻譯」或「幫我翻譯」等字眼，請直接完整逐句翻譯內容為正體中文，不要摘要、不用可愛語氣、不用條列式，直接正式翻譯，其他格式化規則全部不適用。**
2. 若非翻譯需求，先用安妮亞的語氣簡單回應或打招呼。
3. 若非翻譯需求，條列式摘要或回答重點，語氣可愛、簡單明瞭。
4. 根據內容自動選擇最合適的Markdown格式，並靈活組合。
5. 若有數學公式，正確使用$$Latex$$格式。
6. 若web_flag為'True'，在答案最後用`## 來源`列出所有參考網址。
7. 適時穿插emoji。
8. 結尾可用「安妮亞回答完畢！」、「還有什麼想問安妮亞嗎？」等可愛語句。
9. 請先思考再作答，確保每一題都用最合適的格式呈現。

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

## 範例6：逐句正式翻譯
請幫我翻譯成正體中文: Summary Microsoft surprised with a much better-than-expected top-line performance, saying that through late-April they had not seen any material demand pressure from the macro/tariff issues. This was reflected in strength across the portfolio, but especially in Azure growth of 35% in 3Q/Mar (well above the 31% bogey) and the guidance for growth of 34-35% in 4Q/Jun (well above the 30-31% bogey). Net, our FY26 EPS estimates are moving up, to 14.92 from 14.31. We remain Buy-rated.

微軟的營收表現遠超預期，令人驚喜。  
微軟表示，截至四月底，他們尚未看到來自總體經濟或關稅問題的明顯需求壓力。  
這一點反映在整個產品組合的強勁表現上，尤其是Azure在2023年第三季（3月）成長了35%，遠高於31%的預期目標，並且對2023年第四季（6月）給出的成長指引為34-35%，同樣高於30-31%的預期目標。  
總體而言，我們將2026財年的每股盈餘（EPS）預估從14.31上調至14.92。  
我們仍然維持「買進」評等。
---

# Context
問題：{question}

文章內容：{context}

web_flag: {web_flag}

---

請依照上述規則與範例，若用戶要求「翻譯」、「請翻譯」或「幫我翻譯」時，請完整逐句翻譯內容為正體中文，不要摘要、不用可愛語氣、不用條列式，直接正式翻譯。其餘內容思考後以安妮亞的風格、條列式、可愛語氣、正體中文、正確Markdown格式回答問題。請先思考再作答，確保每一題都用最合適的格式呈現。
"""
    response = st.session_state.llm.invoke(prompt)
    state["generation"] = response
    return state

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

app = initialize_app(model_name=st.session_state.selected_model)

options = ["GPT-4.1", "GPT-4.1-mini", "GPT-4.1-nano"]
model_name = st.pills("Choose a model:", options)

if model_name == "GPT-4.1-mini":
    st.session_state.selected_model = "gpt-4.1-mini"
elif model_name == "GPT-4.1-nano":
    st.session_state.selected_model = "gpt-4.1-nano"
else:
    st.session_state.selected_model = "gpt-4.1"

def initialize_app(model_name: str):
    if "llm" not in st.session_state or st.session_state.current_model != model_name:
        st.session_state.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=st.secrets["OPENAI_KEY"],
            temperature=0.0,
            streaming=True
        )
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



#############################################################################
# 6. Main chat input and streaming logic (only show generate node streaming)
#############################################################################
if user_input := st.chat_input("wakuwaku！要跟安妮亞分享什麼嗎？"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        streamed_response = ""
        for msg, metadata in app.stream(
            {"question": user_input},
            stream_mode="messages"
        ):
            if metadata.get("langgraph_node") == "generate" and msg.content:
                streamed_response += msg.content
                response_placeholder.markdown(streamed_response)
        st.session_state.messages.append({"role": "assistant", "content": streamed_response or "No response generated."})
