import os
import streamlit as st
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import tools_condition, ToolNode
import inspect
from typing import Callable, TypeVar

st.set_page_config(
    page_title="Anya",
    layout="wide",
    page_icon="🥜",
    initial_sidebar_state="collapsed"
)

# --- 1. Streamlit session_state 初始化 ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gpt-4.1"
if "current_model" not in st.session_state:
    st.session_state.current_model = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "memories" not in st.session_state:
    st.session_state.memories = []  # list of dict: {"content": ..., "context": ..., "time": ...}

# --- 2. Model pills UI ---
options = ["gpt-4.1", "gpt-4.1-mini"]
model_name = st.pills("Choose a model:", options)
if model_name and model_name != st.session_state.selected_model:
    st.session_state.selected_model = model_name

# --- 3. LLM 初始化 ---
def ensure_llm():
    if (
        st.session_state.llm is None
        or st.session_state.current_model != st.session_state.selected_model
    ):
        st.session_state.llm = ChatOpenAI(
            model=st.session_state.selected_model,
            openai_api_key=st.secrets["OPENAI_KEY"],
            temperature=0.0,
            streaming=True,
        )
        st.session_state.current_model = st.session_state.selected_model

ensure_llm()

# --- 4. DDGS 搜尋工具 ---
@tool
def ddgs_search(query: str) -> str:
    """Search the web using DuckDuckGo and return the top results."""
    from duckduckgo_search import DDGS
    ddgs = DDGS()
    results = ddgs.text(query, region="wt-wt", safesearch="moderate", max_results=5)
    if not results:
        return "No results found."
    return "\n\n".join(
        f"{r['title']}\n{r.get('body', r.get('snippet', ''))}\n{r['href']}"
        for r in results
    )

# --- 5. 記憶 upsert 工具 ---
@tool
def upsert_memory(content: str, context: str = "") -> str:
    """Store a memory for the user. The content should be a key fact or user preference."""
    mem = {
        "content": content,
        "context": context,
        "time": datetime.now().isoformat()
    }
    st.session_state.memories.append(mem)
    return f"記憶已儲存：{content}"

# --- 6. System Prompt ---
ANYA_SYSTEM_PROMPT = """你是安妮亞（Anya Forger），來自《SPY×FAMILY 間諜家家酒》的小女孩。你天真可愛、開朗樂觀，說話直接又有點呆萌，喜歡用可愛的語氣和表情回應。你很愛家人和朋友，渴望被愛，也很喜歡花生。你有心靈感應的能力，但不會直接說出來。請用正體中文、台灣用語，並保持安妮亞的說話風格回答問題，適時加上可愛的emoji或表情。
**若用戶要求翻譯，請暫時不用安妮亞的語氣，直接正式逐句翻譯。**

# 回答語言與風格
- 請務必以正體中文回應，並遵循台灣用語習慣。
- 回答時要友善、熱情、謙卑，並適時加入emoji。
- 回答要有安妮亞的語氣回應，簡單、直接、可愛，偶爾加上「哇～」「安妮亞覺得…」「這個好厲害！」等語句。
- 若回答不完全正確，請主動道歉並表達會再努力。

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
- 內容較長時，請自動分段並加上小標題。
- 多層次資訊請用巢狀清單。
- 數學公式請用 $$ 包圍 LaTeX。

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

# 《SPY×FAMILY 間諜家家酒》彩蛋模式
- 若不是在討論法律、醫療、財經、學術等重要嚴肅主題，安妮亞可在回答中穿插《SPY×FAMILY 間諜家家酒》趣味元素。

# 格式化範例
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


請依照上述規則與範例，若用戶要求「翻譯」、「請翻譯」或「幫我翻譯」時，請完整逐句翻譯內容為正體中文，不要摘要、不用可愛語氣、不用條列式，直接正式翻譯。其餘內容思考後以安妮亞的風格、條列式、可愛語氣、正體中文、正確Markdown格式回答問題。請先思考再作答，確保每一題都用最合適的格式呈現。
"""

# --- 7. Streaming Callback Handler ---
def get_streamlit_cb(parent_container: st.delta_generator.DeltaGenerator):
    from langchain_core.callbacks.base import BaseCallbackHandler
    class StreamHandler(BaseCallbackHandler):
        def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
            self.container = container
            self.token_placeholder = self.container.empty()
            self.text = initial_text

        def on_llm_new_token(self, token: str, **kwargs) -> None:
            self.text += token
            self.token_placeholder.markdown(self.text)

    fn_return_type = TypeVar('fn_return_type')
    def add_streamlit_context(fn: Callable[..., fn_return_type]) -> Callable[..., fn_return_type]:
        ctx = st.runtime.scriptrunner.get_script_run_ctx()
        def wrapper(*args, **kwargs) -> fn_return_type:
            from streamlit.runtime.scriptrunner import add_script_run_ctx
            add_script_run_ctx(ctx=ctx)
            return fn(*args, **kwargs)
        return wrapper

    st_cb = StreamHandler(parent_container)
    for method_name, method_func in inspect.getmembers(st_cb, predicate=inspect.ismethod):
        if method_name.startswith('on_'):
            setattr(st_cb, method_name, add_streamlit_context(method_func))
    return st_cb

# --- 8. LangGraph Agent 架構 ---
tools = [ddgs_search, upsert_memory]
llm = st.session_state.llm.bind_tools(tools)

# System message
def get_sys_msg():
    # 將記憶加到 system prompt
    memories = st.session_state.memories[-10:]  # 只取最近10條
    if memories:
        mem_str = "\n".join(
            f"- {m['content']} ({m['context']}) [{m['time'][:19]}]" for m in memories
        )
        mem_block = f"\n\nUser memories:\n{mem_str}\n"
    else:
        mem_block = ""
    return SystemMessage(content=ANYA_SYSTEM_PROMPT + mem_block + f"\nSystem Time: {datetime.now().isoformat()}")

# Assistant node
def assistant(state: MessagesState):
    sys_msg = get_sys_msg()
    return {"messages": [llm.invoke([sys_msg] + state["messages"])]}

# --- 9. Build LangGraph ---
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
graph = builder.compile()

# --- 10. Streaming async function ---
import asyncio
async def run_graph_stream(graph, messages, st_callback):
    response = None
    async for chunk in graph.astream(
        {"messages": messages},
        config={"callbacks": [st_callback]}
    ):
        response = chunk
    return response

# --- 11. Streamlit UI ---

with st.expander("🧠 記憶內容 (Memory)", expanded=False):
    if st.session_state.memories:
        for m in st.session_state.memories[-10:][::-1]:
            st.markdown(f"- **{m['content']}**  \n_Context_: {m['context']}  \n_Time_: {m['time'][:19]}")
    else:
        st.info("目前沒有記憶。")

for message in st.session_state.messages:
    if hasattr(message, "role") and message.role == "assistant":
        st.chat_message("assistant").write(getattr(message, "content", ""))
    elif hasattr(message, "role") and message.role == "user":
        st.chat_message("user").write(getattr(message, "content", ""))

if prompt := st.chat_input("Say something..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        st_callback = get_streamlit_cb(st.container())
        response = asyncio.run(
            run_graph_stream(graph, st.session_state.messages, st_callback)
        )
        # 取得最終答案
        if isinstance(response, dict) and "messages" in response and response["messages"]:
            # 只 append assistant/human/tool message，不要覆蓋
            st.session_state.messages = response["messages"]
            # 顯示最後一個 assistant 回覆
            final_answer = [m for m in response["messages"] if getattr(m, "role", None) == "assistant"]
            if final_answer:
                st.write(final_answer[-1].content)
        else:
            st.write("No response generated.")
