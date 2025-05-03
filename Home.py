import os
import streamlit as st
from typing_extensions import TypedDict, Annotated
from dataclasses import dataclass, field, fields
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, AnyMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END, add_messages
from datetime import datetime
import re
import inspect
from typing import Callable, TypeVar, Any, Dict
import asyncio

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
        
options=["GPT-4.1", "GPT-4.1-mini", "GPT-4.1-nano"]
model_name = st.pills("Choose a model:", options)

# Map model names to OpenAI model IDs
if model_name == "GPT-4.1-mini":
    st.session_state.selected_model = "gpt-4.1-mini"
elif model_name == "GPT-4.1-nano":
    st.session_state.selected_model = "gpt-4.1-nano"
else:
    st.session_state.selected_model = "gpt-4.1"

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

# --- 6. State 定義 ---
@dataclass(kw_only=True)
class State:
    messages: Annotated[list[AnyMessage], add_messages]

# --- 7. Configuration 定義 ---
@dataclass(kw_only=True)
class Configuration:
    user_id: str = "default"
    model: str = field(default="gpt-4.1")
    system_prompt: str = "You are a helpful assistant. Use tools if needed."

    @classmethod
    def from_runnable_config(cls, config=None):
        configurable = config.get("configurable", {}) if config else {}
        values = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})

# --- 8. Streaming Callback Handler ---
def get_streamlit_cb(parent_container: st.delta_generator.DeltaGenerator):
    class StreamHandler:
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

# --- 9. Graph node functions ---
async def call_model(state: State, config: RunnableConfig) -> dict:
    ensure_llm()
    configurable = Configuration.from_runnable_config(config)
    # 將記憶加到 system prompt
    memories = st.session_state.memories[-10:]  # 只取最近10條
    if memories:
        mem_str = "\n".join(
            f"- {m['content']} ({m['context']}) [{m['time'][:19]}]" for m in memories
        )
        mem_block = f"\n\nUser memories:\n{mem_str}\n"
    else:
        mem_block = ""
    sys = configurable.system_prompt + mem_block + f"\nSystem Time: {datetime.now().isoformat()}"
    llm = st.session_state.llm.bind_tools([ddgs_search, upsert_memory])
    msg = await llm.ainvoke(
        [{"role": "system", "content": sys}, *state.messages],
        {"configurable": {"model": configurable.model}}
    )
    return {"messages": [msg]}

async def store_memory(state: State, config: RunnableConfig):
    # 處理 LLM tool call upsert_memory
    last_msg = state.messages[-1]
    tool_calls = getattr(last_msg, "tool_calls", [])
    results = []
    for tc in tool_calls:
        if tc["name"] == "upsert_memory":
            args = tc.get("args", {})
            # 直接呼叫 upsert_memory tool
            result = upsert_memory(**args)
            results.append({
                "role": "tool",
                "content": result,
                "tool_call_id": tc["id"],
            })
    await asyncio.sleep(0.01)
    return {"messages": results}

def route_message(state: State):
    msg = state.messages[-1]
    if getattr(msg, "tool_calls", None):
        return "store_memory"
    return END

# --- 10. Build LangGraph ---
builder = StateGraph(State, config_schema=Configuration)
builder.add_node(call_model)
builder.add_edge("__start__", "call_model")
builder.add_node(store_memory)
builder.add_conditional_edges("call_model", route_message, ["store_memory", END])
builder.add_edge("store_memory", "call_model")
graph = builder.compile()

# --- 11. Streamlit UI ---
st.title("DDGS Agent Demo (with Streaming & Memory)")

with st.expander("🧠 記憶內容 (Memory)", expanded=False):
    if st.session_state.memories:
        for m in st.session_state.memories[-10:][::-1]:
            st.markdown(f"- **{m['content']}**  \n_Context_: {m['context']}  \n_Time_: {m['time'][:19]}")
    else:
        st.info("目前沒有記憶。")

for message in st.session_state.messages:
    if isinstance(message, AIMessage):
        st.chat_message("assistant").write(message.content)
    elif isinstance(message, HumanMessage):
        st.chat_message("user").write(message.content)

if prompt := st.chat_input("Say something..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        st_callback = get_streamlit_cb(st.container())
        # streaming with callback handler
        response = asyncio.run(
            graph.invoke(
                {"messages": st.session_state.messages},
                config={"callbacks": [st_callback]}
            )
        )
        # 取得最終答案
        if isinstance(response, dict) and "messages" in response and response["messages"]:
            final_answer = response["messages"][-1].content
        else:
            final_answer = "No response generated."
        st.session_state.messages.append(AIMessage(content=final_answer))
