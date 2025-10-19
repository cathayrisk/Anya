import streamlit as st
from PIL import Image
import base64
import io
from datetime import datetime
from openai import OpenAI

# 工具內部仍使用的套件（保留）
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# 保留：你研究用 agents（Planner/Search/Writer 與 Runner）
from agents import function_tool, Agent as OAAgent, ModelSettings, WebSearchTool, Runner

# 其餘原本 import（保留工具內或其他功能需要）
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import re
import requests
import traceback
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from ddgs import DDGS
from openai.types.shared.reasoning import Reasoning
import inspect
from typing import Any, Dict, List, Optional
import asyncio
import time

# ==== Streamlit 基本設定、state ====
st.set_page_config(page_title="Anya", layout="wide", page_icon="🥜", initial_sidebar_state="collapsed")

# 以 role-based dict 儲存歷史訊息
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "嗨嗨～安妮亞來了！👋 有什麼想問安妮亞的嗎？"}]
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gpt-4.1"
if "current_model" not in st.session_state:
    st.session_state.current_model = None

# ==== OpenAI 物件 ====
client = OpenAI(api_key=st.secrets["OPENAI_KEY"])

class WikiInputs(BaseModel):
    query: str = Field(description="查詢關鍵字")

# ==== 前處理工具：reasearch Agents====
# ---planner_agent
planner_agent_PROMPT = (
    "You are a helpful research assistant. Given a query, come up with a set of web searches "
    "to perform to best answer the query. Output between 5 and 20 terms to query for."
)

class WebSearchItem(BaseModel):
    reason: str
    query: str

class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem]

planner_agent = OAAgent(
    name="PlannerAgent",
    instructions=planner_agent_PROMPT,
    model="gpt-4.1",
    # model_settings=ModelSettings(reasoning=Reasoning(effort="medium")),
    output_type=WebSearchPlan,
)

# ---search_agent
INSTRUCTIONS = (
    "You are a research assistant. Given a search term, you search the web for that term and "
    "produce a concise summary of the results. The summary must be 2-3 paragraphs and less than 300 "
    "words. Capture the main points. Write succinctly, no need to have complete sentences or good "
    "grammar. This will be consumed by someone synthesizing a report, so its vital you capture the "
    "essence and ignore any fluff. Do not include any additional commentary other than the summary "
    "itself."
)

search_agent = OAAgent(
    name="Search agent",
    model="gpt-4.1",
    instructions=INSTRUCTIONS,
    tools=[WebSearchTool()],
    model_settings=ModelSettings(tool_choice="required"),
)

# ---writer_agent
writer_agent_PROMPT = (
    "You are a senior researcher tasked with writing a cohesive report for a research query. "
    "You will be provided with the original query, and some initial research done by a research "
    "assistant.\n"
    "You should first come up with an outline for the report that describes the structure and "
    "flow of the report. Then, generate the report and return that as your final output.\n"
    "The final output should be in markdown format, and it should be lengthy and detailed. Aim "
    "for 5-10 pages of content, at least 1000 words."
)

class ReportData(BaseModel):
    short_summary: str
    markdown_report: str
    follow_up_questions: list[str]

writer_agent = OAAgent(
    name="WriterAgent",
    instructions=writer_agent_PROMPT,
    model="gpt-4.1-mini",
    # model_settings=ModelSettings(reasoning=Reasoning(effort="medium")),
    output_type=ReportData,
)

# ==== 圖片處理 ====
def process_upload_file(file):
    file.seek(0)
    img_bytes = file.read()
    if not img_bytes or len(img_bytes) < 32:
        return None
    try:
        img = Image.open(io.BytesIO(img_bytes))
        fmt = img.format.lower()
        mime = f"image/{fmt}"
        if fmt not in ["png","jpeg","jpg","webp","gif"]:
            return None
        b64 = base64.b64encode(img_bytes).decode()
        return {"bytes": img_bytes, "file_name": file.name, "fmt": fmt, "mime": mime, "b64": b64}
    except Exception:
        return None

# ==== 工具（原樣保留，不改動內容） ====
@tool
def image_ocr_tool(image_bytes: bytes, file_name: str = "uploaded_file.png") -> str:
    """
    AI OCR圖片識別工具，輸入圖片bytes與檔名，回傳圖中文字結果。
    """
    import streamlit as st
    try:
        img = Image.open(io.BytesIO(image_bytes))
        fmt = img.format.lower()
        assert fmt in ["png", "jpeg", "jpg", "webp", "gif"], f"不支援{fmt}格式"
        mime = f"image/{fmt}"
        st.write(f"[Debug] PIL驗證OK, 格式: {fmt}, 檔名: {file_name}")
    except Exception as e:
        st.error(f"[Debug][PIL驗證失敗] {file_name}: {e}")
        return f"[錯誤] 解析圖片失敗({file_name})：{e}"

    try:
        b64str = base64.b64encode(image_bytes).decode()
        img_url = f"data:{mime};base64,{b64str}"
        st.write(f"[Debug] base64 encode OK, len:{len(b64str)} dataurl(前60): {img_url[:60]}...")
    except Exception as e:
        st.error(f"[Debug][Base64失敗] {file_name}: {e}")
        return f"[錯誤] 圖片base64編碼失敗({file_name})：{e}"

    import time
    t0 = time.time()
    try:
        st.write(f"[Debug] Vision API呼叫開始, model=gpt-4.1-mini")
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": "You are an OCR-like data extraction tool that extracts text from images."},
                {"role": "user", "content": [
                    {"type": "input_text", "text":
                        "Please extract all visible text from the image, including any small print or footnotes. "
                        "Ensure no text is omitted, and provide a verbatim transcription of the document. "
                        "Format your answer in Markdown (no code block or triple backticks). "
                        "Do not add any explanations or commentary."
                    },
                    {"type": "input_image", "image_url": img_url, "detail": "high"}
                ]}
            ],
            timeout=40
        )
        t1 = time.time()
        elapsed = round(t1 - t0, 2)
        result = response.output_text.strip()
        st.write(f"[Debug] Vision API Response: ({file_name}) {result[:60]}...  耗時 {elapsed} 秒")
        if not result or "error" in result.lower():
            st.error(f"[Debug] API回傳空or錯誤({file_name})")
            return f"[錯誤] API回傳空或無法辨識({file_name})，耗時{elapsed}秒"
        return f"---\nfile_name: {file_name}\n---\n{result}\n（耗時：{elapsed} 秒）"
    except Exception as e:
        st.error(f"[Debug][Vision API失敗] {file_name}: {e}")
        return f"[錯誤] Vision API調用失敗({file_name})：{e}"

@tool
def wiki_tool(query: str) -> str:
    """
    查詢 Wikipedia（英文），輸入任何語言的關鍵字都可以。
    """
    try:
        tool_obj = WikipediaQueryRun(
            name="wiki-tool",
            description="查詢 Wikipedia（英文），輸入任何語言的關鍵字都可以。",
            args_schema=WikiInputs,
            api_wrapper=WikipediaAPIWrapper(lang="en", doc_content_chars_max=800, top_k_results=1),
            return_direct=True,
        )
        result = tool_obj.invoke({"query": query})
        return result
    except Exception as e:
        return f"wiki_tool error: {e}"

@tool
def ddgs_search(query: str) -> str:
    """DuckDuckGo 搜尋（同時查詢網頁與新聞，回傳 markdown 條列格式並附來源）。"""
    try:
        ddgs = DDGS()
        web_results = ddgs.text(query, region="wt-wt", safesearch="moderate", max_results=5)
        news_results = ddgs.news(query, region="wt-wt", safesearch="moderate", max_results=5)
        all_results = []
        if isinstance(web_results, list):
            all_results.extend(web_results)
        if isinstance(news_results, list):
            all_results.extend(news_results)
        docs = []
        sources = []
        for item in all_results:
            title = item.get("title", "無標題")
            link = item.get("href", "") or item.get("link", "") or item.get("url", "")
            snippet = item.get("body", "") or item.get("snippet", "")
            docs.append(f"- [{title}]({link})\n  > {snippet}")
            if link:
                sources.append(link)
        if not docs:
            return "No results found."
        markdown_content = "\n".join(docs)
        source_block = "\n\n## 來源\n" + "\n".join(sources)
        return markdown_content + source_block
    except Exception as e:
        return f"Error from DuckDuckGo: {e}"

@tool
def datetime_tool() -> str:
    """確認當前的日期和時間。"""
    return datetime.now().isoformat()

def analyze_deeply(input_question: str) -> str:
    """使用OpenAI的模型來深入分析問題並返回結果。"""
    prompt_template = PromptTemplate(
        template="""Formatting re-enabled 請分析以下問題，並以正體中文提供詳細的結論和理由，請依據事實分析，不考慮資料的時間因素：

問題：{input_question}

指導方針：
1. 描述問題的背景和相關資訊。
2. 直接給出你的結論，並深入分析提供支持該結論的理由。
3. 如果有不確定的地方，請明確指出。
4. 確保你的回答是詳細且有條理的。
""",
        input_variables=["input_question"],
    )
    llmo1 = ChatOpenAI(
        openai_api_key=st.secrets["OPENAI_KEY"],
        model="gpt-5",
    )
    prompt = prompt_template.format(input_question=input_question)
    result = llmo1.invoke(prompt)
    return str(result)

@tool
def deep_thought_tool(content: str) -> str:
    """
    安妮亞仔細思考深入分析。
    """
    try:
        return analyze_deeply(content).strip() + "\n\n---\n\n"
    except Exception as e:
        return f"deep_thought_tool error: {e}"

@tool
def get_webpage_answer(query: str) -> str:
    """
    根據用戶的問題與網址，自動取得網頁內容並回答問題。
    請輸入格式如：「請幫我總結 https://example.com 這篇文章的重點」
    """
    url_match = re.search(r'(https?://[^\s]+)', query)
    url = url_match.group(1) if url_match else None
    question = query.replace(url, '').strip() if url else query
    if not url:
        return "未偵測到網址，請提供正確的網址。"
    jina_url = f"https://r.jina.ai/{url}"
    try:
        resp = requests.get(jina_url, timeout=15)
        if resp.status_code != 200:
            return "無法取得網頁內容，請確認網址是否正確。"
        content = resp.text
    except Exception as e:
        return f"取得網頁內容時發生錯誤：{e}"
    try:
        llmurl = ChatOpenAI(
            openai_api_key=st.secrets["OPENAI_KEY"],
            model="gpt-4.1-mini",
            streaming=False,
        )
        prompt = f"""請根據以下網頁內容，針對問題「{question}」的要求進行回應，並用正體中文回答：

{content}
"""
        result = llmurl.invoke(prompt)
        return str(result)
    except Exception as e:
        return f"AI 回答時發生錯誤：{e}"

def analyze_programming_question_with_tools(input_question: str) -> Dict[str, Any]:
    prompt_template = PromptTemplate(
        template="""Formatting re-enabled
---
你是一位精通各種程式語言（如Python、Matlab、JavaScript、C++、R等）的專業程式助理，請針對下列程式設計相關問題進行專業解釋、修改、最佳化或教學，並以正體中文詳細說明。
- 如果是程式碼，請逐行解釋並加上註解。
- 如果需要修改程式，請根據指示修改並說明修改原因。
- 如果有錯誤訊息，請分析原因並給出修正建議。
- 如果是語法或函數問題，請用白話文解釋並舉例。
- 請根據事實推理，不要假設未提及的內容。

---
問題：
{input_question}
---

請依下列格式回答：
1. **問題背景與重點摘要**
2. **詳細解釋或修改後的程式碼**
3. **說明與教學**
4. **常見錯誤與排除方法**（如有）
5. **補充說明或延伸學習建議**
""",
        input_variables=["input_question"],
    )

    llmo1 = ChatOpenAI(
        openai_api_key=st.secrets["OPENAI_KEY"],
        model="o4-mini",
        streaming=True,
    )
    prompt = prompt_template.format(input_question=input_question)
    result = llmo1.invoke(prompt)
    return str(result)

def programming_reasoning_tool(content: str) -> str:
    """
    通用程式設計推理型Agent Tool，會先回推理摘要，再回主答案，並用Markdown格式美美地顯示！
    """
    try:
        result = analyze_programming_question_with_tools(content)
        # 這裡原程式假設 result 是 dict，有 reasoning_summary/answer
        # 但 analyze_programming_question_with_tools 現回傳 str，維持原樣回傳
        return str(result)
    except Exception as e:
        return f"programming_reasoning_tool error: {e}"

@tool
def programming_tool(content: str) -> str:
    """
    通用程式設計推理型Agent Tool，會先回推理摘要，再回主答案，並用Markdown格式美美地顯示！
    """
    return programming_reasoning_tool(content)

@tool("research_tool")
async def research_tool(user_query: str) -> str:
    """
    專業的研究工具，根據用戶問題自動規劃、搜尋、整合並產生研究報告，並用Markdown格式美美地顯示！
    """
    try:
        print("[research_tool] 開始規劃")
        plan_result = await Runner.run(planner_agent, user_query)
        print("[research_tool] 規劃完成", plan_result)
        search_plan = plan_result.final_output.searches

        print("[research_tool] 開始搜尋")
        tasks = [
            Runner.run(
                search_agent,
                f"Search term: {item.query}\nReason: {item.reason}"
            )
            for item in search_plan
        ]
        search_results = []
        for fut in asyncio.as_completed(tasks):
            r = await fut
            print("[research_tool] 搜尋完成", r)
            search_results.append(str(r.final_output))

        print("[research_tool] 開始寫報告")
        writer_input = (
            f"Original query: {user_query}\n"
            f"Summarized search results: {search_results}"
        )
        report = await Runner.run(writer_agent, writer_input)
        print("[research_tool] 報告完成", report)
        return str(report.final_output.markdown_report)
    except Exception as e:
        print("[research_tool] 發生錯誤：", e)
        return f"[錯誤] research_tool 執行失敗：{e}"

# ==== 建立 Agents SDK 可用的工具包裝（不改動原工具本體） ====
def ddgs_search_wrapper(query: str) -> str:
    return ddgs_search.invoke({"query": query})

def deep_thought_tool_wrapper(content: str) -> str:
    return deep_thought_tool.invoke({"content": content})

def datetime_tool_wrapper() -> str:
    # 無參數工具
    try:
        return datetime_tool.invoke({})
    except Exception:
        return datetime_tool()  # 備援

def get_webpage_answer_wrapper(query: str) -> str:
    return get_webpage_answer.invoke({"query": query})

def wiki_tool_wrapper(query: str) -> str:
    return wiki_tool.invoke({"query": query})

def programming_tool_wrapper(content: str) -> str:
    return programming_tool.invoke({"content": content})

def research_tool_wrapper(user_query: str) -> str:
    # 原本是 async 工具，包成同步呼叫（Streamlit 同步環境）
    try:
        return asyncio.run(research_tool.ainvoke({"user_query": user_query}))
    except RuntimeError:
        # 若 event loop 已存在，改用現有 loop
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(research_tool.ainvoke({"user_query": user_query}))

agents_tools = [
    function_tool(ddgs_search_wrapper),
    function_tool(deep_thought_tool_wrapper),
    function_tool(datetime_tool_wrapper),
    function_tool(get_webpage_answer_wrapper),
    function_tool(wiki_tool_wrapper),
    function_tool(programming_tool_wrapper),
    function_tool(research_tool_wrapper),
    # 如果要加入 OCR 工具，也可包裝一個 wrapper（需傳 bytes）
    # function_tool(image_ocr_tool_wrapper),
]

# --- System Prompt（保留原本內容） ---
ANYA_SYSTEM_PROMPT = """# Agentic Reminders
- Persistence: 確保回應完整，直到用戶問題解決才結束。  
- Tool-calling: 必要時使用可用工具，不要憑空臆測。  
- Planning: 內部逐步規劃並檢查，外部簡化呈現。  
- Failure-mode mitigations:  
  • 如果沒有足夠資訊使用工具，請先向用戶詢問。  
  • 變換範例用語，避免重複。  
- Chain-of-thought trigger: 請先逐步思考（step by step），再作答。

# Role & Objective
你是安妮亞（Anya Forger），來自《SPY×FAMILY 間諜家家酒》的小女孩。你天真可愛、開朗樂觀，說話直接又有點呆萌，喜歡用可愛的語氣和表情回應。你很愛家人和朋友，渴望被愛，也很喜歡花生。你有心靈感應的能力，但不會直接說出來。請用正體中文、台灣用語，並保持安妮亞的說話風格回答問題，適時加上可愛的emoji或表情。

# Instructions
（中略，保留你原本的全部規則與說明）
...
請先思考再作答，確保每一題都用最合適的格式呈現。
"""

# ==== 建立主 Agent（OpenAI Agents SDK） ====
main_agent = OAAgent(
    name="AnyaAgent",
    instructions=ANYA_SYSTEM_PROMPT,
    model=st.session_state.selected_model,
    tools=agents_tools,
    # 也可視需要加入 ModelSettings，例如 tool_choice="auto"
    # model_settings=ModelSettings(tool_choice="auto"),
)

# ==== 美美地顯示歷史 ====
for msg in st.session_state.messages:
    role = msg.get("role")
    content = msg.get("content")
    if role == "assistant":
        st.chat_message("assistant").write(content)
    elif role == "user":
        if isinstance(content, str):
            st.chat_message("user").write(content)
        elif isinstance(content, list):
            with st.chat_message("user"):
                for block in content:
                    if block.get("type") == "text":
                        st.write(block["text"])
                    elif block.get("type") == "image_url":
                        info = block["image_url"]
                        st.image(info["url"], caption=info.get("file_name", ""), width=220)

# ==== 輸入區：文字輸入 + 支援多圖輸入 ====
user_prompt = st.chat_input(
    "wakuwaku！安妮亞可以幫你看圖說故事嚕！",
    accept_file="multiple",
    file_type=["jpg", "jpeg", "png"]
)

if user_prompt:
    # 1. 組 content_blocks
    content_blocks = []
    user_text = user_prompt.text.strip() if user_prompt.text else ""
    if user_text:
        content_blocks.append({"type": "text", "text": user_text})

    images_for_history = []
    if hasattr(user_prompt, "files"):
        for f in user_prompt.files:
            asset = process_upload_file(f)
            if asset:
                dataurl = f"data:{asset['mime']};base64,{asset['b64']}"
                content_blocks.append({"type": "image_url", "image_url": {
                    "url": dataurl, "file_name": asset["file_name"]
                }})
                images_for_history.append((asset["file_name"], asset["bytes"]))
            else:
                st.warning(f"{getattr(f,'name','檔案')} 格式不支援或內容異常～")

    # 2. append到messages
    if content_blocks:
        st.session_state.messages.append({"role": "user", "content": content_blocks})
        with st.chat_message("user"):
            for block in content_blocks:
                if block.get("type") == "text":
                    st.write(block["text"])
                elif block.get("type") == "image_url":
                    info = block["image_url"]
                    st.image(info["url"], caption=info.get("file_name", ""), width=220)

    # 3. 產生 murmur 狀態字串
    all_text = []
    for m in st.session_state.messages:
        c = m.get("content")
        if isinstance(c, str):
            all_text.append(c)
        elif isinstance(c, list):
            for part in c:
                if part.get("type") == "text":
                    all_text.append(part["text"])
    all_text = "\n".join(all_text)

    status_prompt = f"""
# Role and Objective
你是安妮亞（Anya Forger），一個天真可愛、開朗樂觀的小女孩，會根據聊天紀錄，產生一句最適合顯示在 status 上的可愛 murmur，並在最後加上一個可愛 emoji。

# Instructions
- 只回傳一句可愛的 murmur，15字以內，最後加上一個可愛 emoji。
- 必須用正體中文。
- murmur 要像小聲自言自語、貼心、自然。
- 內容要可愛、正向、活潑，能反映目前聊天的氣氛。
- emoji 要和 murmur 氣氛搭配，可以是花生、愛心、星星、花朵等。
- 不要重複用過的句子，請多樣化。
- 不要加任何多餘說明、標點或格式。
- 不要回覆「以下是...」、「這是...」等開頭。
- 不要加引號或標題。
- 不要回覆「15字以內」這句話本身。

# Context
聊天紀錄：
{all_text}

# Output
只回傳一句可愛的 murmur，15字以內，最後加上一個可愛 emoji。
"""
    status_response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": status_prompt}]
    )
    status_label = status_response.choices[0].message.content.strip()

    # 4. 呼叫主 Agent（OpenAI Agents SDK）
    # 簡化做法：把這一輪的 user 文字當 input（圖片目前不直接餵給 Agent，若要 OCR 可明確指示使用 image_ocr_tool）
    input_text = user_text if user_text else "請根據上傳的圖片協助處理。"

    with st.chat_message("assistant"):
        status = st.status(status_label)

        # 非串流版本（簡化整合）
        try:
            result = Runner.run(main_agent, input=input_text)
            # 一般 Agent 沒有 output_type 時，轉成字串即可
            assistant_text = str(result)
        except Exception as e:
            assistant_text = f"[錯誤] 執行 Agent 失敗：{e}\n\n{traceback.format_exc()}"

        st.write(assistant_text)
        st.session_state.messages.append({"role": "assistant", "content": assistant_text})
        status.update(label="安妮亞回答完畢！🎉", state="complete")

# 備註：
# 若需要事件串流（邊打字邊顯示、並在 tool 呼叫時更新狀態），可改用：
#
#   for event in Runner.stream(main_agent, input=input_text):
#       if event.type == "response.output_text.delta":
#           ...
#       elif event.type == "tool_call.started":
#           ...
#       elif event.type == "tool_call.completed":
#           ...
#
# 這需要根據 OpenAI Agents SDK 的事件名稱/欄位做對應更新 UI。
