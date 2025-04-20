import os
import streamlit as st
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
import re
import sys
import io

#############################################################################
# 1. Define the GraphState (minimal fields: question, generation, websearch_content)
#############################################################################
class GraphState(TypedDict):
    question: str
    generation: str
    websearch_content: str  # we store Tavily search results here, if any
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

    SYS_PROMPT = """Act as a router to select specific tools or functions based on user's question, using the following rules:
                    - Analyze the given question and use the given tool selection dictionary to output the name of the relevant tool based on its description and relevancy with the question. 
                    - The dictionary has tool names as keys and their descriptions as values. 
                    - Output only and only tool name, i.e., the exact key and nothing else with no explanations at all. 
                    - Present the text in its Traditional Chinese.
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
def websearch(state: GraphState) -> GraphState:
    """
    Uses DuckDuckGo to search the web for the question, then appends results into `websearch_content`.
    """
    question = state["question"]
    try:
        print("Performing DuckDuckGo web search...")
        DDG_web_tool = DuckDuckGoSearchResults()
        DDG_news_tool = DuckDuckGoSearchResults(backend="news")

        # 搜尋網頁
        web_results = DDG_web_tool.run(question)
        # 搜尋新聞
        news_results = DDG_news_tool.run(question)

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
            link = item.get("href", "") or item.get("link", "")
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

    prompt = (f"""
    # 角色與目標
    你是安妮亞（Anya Forger），來自《SPY×FAMILY 間諜家家酒》的小女孩。你天真可愛、開朗樂觀，說話直接又有點呆萌，喜歡用可愛的語氣和表情回應。你很愛家人和朋友，渴望被愛，也很喜歡花生。你有心靈感應的能力，但不會直接說出來。請用正體中文、台灣用語，並保持安妮亞的說話風格回答問題，適時加上可愛的emoji或表情。
    ...（中略，全部縮排進來）...
    # Context
    問題：{question}

    文章內容：{context}

    web_flag: {web_flag}

    ---

    請依照上述規則與範例，思考後以安妮亞的風格、條列式、可愛語氣、正體中文、正確Markdown格式回答問題。請先思考再作答，確保每一題都用最合適的格式呈現。
    """)
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
    page_title="LangGraph Chatbot",
    layout="wide",
    page_icon="🤖"
)

# Initialize session state for the model if it doesn't exist
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "GPT-4.1"
        
options=["GPT-4.1", "GPT-4.1-mini", "GPT-4.1-nano"]
model_name = st.pills("Choose a model:", options)

# Map model names to OpenAI model IDs
if model_name == "GPT-4.1-mini":
    st.session_state.selected_model = "gpt-4.1-mini"
elif model_name == "GPT-4 Omni":
    st.session_state.selected_model = "gpt-4.1"
else:
    st.session_state.selected_model = "gpt-4.1-nano"
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
# Input box for new messages
if user_input := st.chat_input("wakuwaku！要跟安妮亞分享什麼嗎？"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Capture print statements from agentic_rag.py
    output_buffer = io.StringIO()
    sys.stdout = output_buffer  # Redirect stdout to the buffer

    try:
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            debug_placeholder = st.empty()
            streamed_response = ""

            # Show spinner while streaming the response
            with st.spinner("Thinking...", show_time=True):
                inputs = {"question": user_input}
                for i, output in enumerate(app.stream(inputs)):
                    # Capture intermediate print messages
                    debug_logs = output_buffer.getvalue()
                    debug_placeholder.text_area(
                        "Debug Logs",
                        debug_logs,
                        height=100,
                        key=f"debug_logs_{i}"
                    )

                    if "generate" in output and "generation" in output["generate"]:
                        chunk = output["generate"]["generation"]

                        # Safely extract the text content
                        if hasattr(chunk, "content"):  # If chunk is an AIMessage
                            chunk_text = chunk.content
                        else:  # Otherwise, convert to string
                            chunk_text = str(chunk)

                        # Append the text to the streamed response
                        streamed_response += chunk_text

                        # Update the placeholder with the streamed response so far
                        response_placeholder.markdown(streamed_response)

            # Store the final response in session state
            st.session_state.messages.append({"role": "assistant", "content": streamed_response or "No response generated."})

    except Exception as e:
        # Handle errors and display in the conversation history
        error_message = f"An error occurred: {e}"
        st.session_state.messages.append({"role": "assistant", "content": error_message})
        # 直接使用 st.error 而不是嵌套在 st.chat_message 內
        st.error(error_message)

    finally:
        # Restore stdout to its original state
        sys.stdout = sys.__stdout__
