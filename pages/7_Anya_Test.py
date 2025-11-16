import streamlit as st
import base64
import re
import time
import json
import asyncio
import threading
from io import BytesIO
from PIL import Image
from openai import OpenAI
import os
from pypdf import PdfReader, PdfWriter

# ====== Agents SDK（Router / Planner / Search / Fast）======
from agents import (
    Agent,
    ModelSettings,
    Runner,
    handoff,
    HandoffInputData,
    RunContextWrapper,
    WebSearchTool,
)
from agents.extensions import handoff_filters
try:
    from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
except Exception:
    RECOMMENDED_PROMPT_PREFIX = ""
from agents.models import is_gpt_5_default
from openai.types.shared.reasoning import Reasoning
from pydantic import BaseModel
from typing import Literal, Optional, List, Any

# === 0. Trimming / 大小限制（可調） ===
TRIM_LAST_N_USER_TURNS = 18                 # 短期記憶：最近 N 個 user 回合
MAX_REQ_TOTAL_BYTES = 48 * 1024 * 1024      # 單次請求總量預警（48MB）

# === 0.1 取得 API Key ===
OPENAI_API_KEY = (
    st.secrets.get("OPENAI_API_KEY")
    or st.secrets.get("OPENAI_KEY")
    or os.getenv("OPENAI_API_KEY")
)
if not OPENAI_API_KEY:
    st.error("找不到 OpenAI API Key，請在 .streamlit/secrets.toml 設定 OPENAI_API_KEY 或 OPENAI_KEY。")
    st.stop()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY  # 讓 Agents SDK 可以讀到

# === 1. Streamlit 頁面 ===
st.set_page_config(page_title="Anya Multimodal Agent", page_icon="🥜", layout="wide")

# === 1.a Session 預設值保險（務必在任何使用 chat_history 前） ===
def ensure_session_defaults():
    if "chat_history" not in st.session_state or not isinstance(st.session_state.chat_history, list):
        st.session_state.chat_history = [{
            "role": "assistant",
            "text": "嗨嗨～安妮亞來了！👋 上傳圖片或PDF，直接問你想知道的內容吧！",
            "images": [],
            "docs": []
        }]

ensure_session_defaults()

# === 共用：假串流打字效果 ===
def fake_stream_markdown(text: str, placeholder, step_chars=8, delay=0.03, empty_msg="安妮亞找不到答案～（抱歉啦！）"):
    buf = ""
    for i in range(0, len(text), step_chars):
        buf = text[: i + step_chars]
        placeholder.markdown(buf)
        time.sleep(delay)
    if not text:
        placeholder.markdown(empty_msg)
    return text

# 穩定版：確保 coroutine 一定被 await
def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        result_container = {}
        def _runner():
            result_container["value"] = asyncio.run(coro)
        t = threading.Thread(target=_runner)
        t.start()
        t.join()
        return result_container.get("value")
    else:
        return asyncio.run(coro)

# === 1.1 圖片工具：縮圖 & data URL ===
@st.cache_data(show_spinner=False, max_entries=256)
def make_thumb(imgbytes: bytes, max_w=220) -> bytes:
    im = Image.open(BytesIO(imgbytes))
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    im.thumbnail((max_w, max_w))
    out = BytesIO()
    im.save(out, format="JPEG", quality=80, optimize=True)
    return out.getvalue()

def _detect_mime_from_bytes(img_bytes: bytes) -> str:
    try:
        im = Image.open(BytesIO(img_bytes))
        fmt = (im.format or "").upper()
        if fmt == "PNG":  return "image/png"
        if fmt in ("JPG", "JPEG"): return "image/jpeg"
        if fmt == "WEBP": return "image/webp"
        if fmt == "GIF":  return "image/gif"
    except Exception:
        pass
    return "application/octet-stream"

@st.cache_data(show_spinner=False, max_entries=256)
def bytes_to_data_url(imgbytes: bytes) -> str:
    mime = _detect_mime_from_bytes(imgbytes)
    b64 = base64.b64encode(imgbytes).decode()
    return f"data:{mime};base64,{b64}"

# === 1.2 檔案工具：data URI（PDF/TXT/MD/JSON/CSV/DOCX/PPTX） ===
DOC_MIME_MAP = {
    ".pdf":  "application/pdf",
    ".txt":  "text/plain",
    ".md":   "text/markdown",
    ".json": "application/json",
    ".csv":  "text/csv",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}

def guess_mime_by_ext(filename: str) -> str:
    ext = os.path.splitext(filename.lower())[1]
    return DOC_MIME_MAP.get(ext, "application/octet-stream")

def file_bytes_to_data_url(filename: str, data: bytes) -> str:
    mime = guess_mime_by_ext(filename)
    b64 = base64.b64encode(data).decode()
    return f"data:{mime};base64,{b64}"

# === 1.3 PDF 工具：頁碼解析 / 實際切頁 ===
def parse_page_ranges_from_text(text: str) -> list[int]:
    if not text:
        return []
    pages = set()

    # 區間格式
    range_patterns = [
        r'第\s*(\d+)\s*[-~至到]\s*(\d+)\s*頁',
        r'(\d+)\s*[-–—]\s*(\d+)\s*頁',
        r'p(?:age)?s?\s*(\d+)\s*[-–—]\s*(\d+)',
        r'(?<!\w)(\d+)\s*[-–—]\s*(\d+)(?!\w)',
    ]
    for pat in range_patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            a, b = int(m.group(1)), int(m.group(2))
            if a > 0 and b >= a:
                for p in range(a, b + 1):
                    pages.add(p)

    # 單一頁
    single_patterns = [
        r'第\s*(\d+)\s*頁',
        r'p(?:age)?\s*(\d+)',
    ]
    for pat in single_patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            p = int(m.group(1))
            if p > 0:
                pages.add(p)

    # 逗號分隔（在有「頁/page」字樣時才啟用）
    if re.search(r'(頁|page|pages|p[^\w])', text, flags=re.IGNORECASE):
        for m in re.finditer(r'(?<!\d)(\d+)(?:\s*,\s*(\d+))+', text):
            nums = [int(x) for x in m.group(0).split(",") if x.strip().isdigit()]
            for n in nums:
                if n > 0:
                    pages.add(n)

    return sorted(pages)

def slice_pdf_bytes(pdf_bytes: bytes, keep_pages_1based: list[int]) -> bytes:
    if not keep_pages_1based:
        return pdf_bytes
    reader = PdfReader(BytesIO(pdf_bytes))
    n = len(reader.pages)
    writer = PdfWriter()
    for p in keep_pages_1based:
        if 1 <= p <= n:
            writer.add_page(reader.pages[p - 1])
    out = BytesIO()
    writer.write(out)
    out.seek(0)
    return out.getvalue()

# === 1.4 回覆解析：擷取文字 + 來源註解 ===
def dedup_by(items, key):
    seen = set()
    out = []
    for it in items:
        k = it.get(key)
        if k and k not in seen:
            seen.add(k)
            out.append(it)
    return out

def parse_response_text_and_citations(resp):
    text_parts = []
    url_cits = []
    file_cits = []

    text_attr = getattr(resp, "output_text", None)
    if text_attr:
        text_parts.append(text_attr)

    try:
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", "") == "message":
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", "") == "output_text":
                        t = getattr(c, "text", "")
                        if t and not text_attr:
                            text_parts.append(t)
                        for ann in getattr(c, "annotations", []) or []:
                            at = getattr(ann, "type", "")
                            if at == "url_citation":
                                url = getattr(ann, "url", None)
                                title = getattr(ann, "title", None)
                                if url:
                                    url_cits.append({"url": url, "title": title})
                            elif at == "file_citation":
                                filename = getattr(ann, "filename", None)
                                fid = getattr(ann, "file_id", None)
                                file_cits.append({"filename": filename, "file_id": fid})
    except Exception:
        pass

    text = "".join(text_parts) if text_parts else ""
    url_cits = dedup_by(url_cits, "url")
    file_cits = dedup_by(file_cits, "filename") if any(c.get("filename") for c in file_cits) else dedup_by(file_cits, "file_id")
    return text or "安妮亞找不到答案～（抱歉啦！）", url_cits, file_cits

# === 小工具：注入 handoff 官方前綴 ===
def with_handoff_prefix(text: str) -> str:
    pref = (RECOMMENDED_PROMPT_PREFIX or "").strip()
    return f"{pref}\n{text}" if pref else text

# === 1.5 Planner / Router / Search（Agents） ===
class WebSearchItem(BaseModel):
    reason: str
    query: str

class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem]

class PlannerHandoffInput(BaseModel):
    query: str
    need_sources: bool = True
    target_length: Literal["short","medium","long"] = "long"
    date_range: Optional[str] = None
    domains: List[str] = []
    languages: List[str] = ["zh-TW"]

def research_handoff_message_filter(handoff_message_data: HandoffInputData) -> HandoffInputData:
    if is_gpt_5_default():
        return HandoffInputData(
            input_history=handoff_message_data.input_history,
            pre_handoff_items=tuple(handoff_message_data.pre_handoff_items),
            new_items=tuple(handoff_message_data.new_items),
        )
    filtered = handoff_filters.remove_all_tools(handoff_message_data)
    history = filtered.input_history
    if isinstance(history, tuple):
        K = 6
        history = history[-K:]
    return HandoffInputData(
        input_history=history,
        pre_handoff_items=tuple(filtered.pre_handoff_items),
        new_items=tuple(filtered.new_items),
    )

async def on_research_handoff(ctx: RunContextWrapper[None], input_data: PlannerHandoffInput):
    print(f"[handoff] research query: {input_data.query} | len_pref={input_data.target_length} | need_sources={input_data.need_sources}")

planner_agent_PROMPT = with_handoff_prefix(
    "You are a helpful research planner. Given a query, come up with a set of web searches "
    "to perform to best answer the query. Output between 5 and 20 terms to query for.\n"
    "請務必以正體中文回應，並遵循台灣用語習慣。"
)

planner_agent = Agent(
    name="PlannerAgent",
    instructions=planner_agent_PROMPT,
    model="gpt-5.1-2025-11-13",
    model_settings=ModelSettings(reasoning=Reasoning(effort="medium")),
    output_type=WebSearchPlan,
)

search_INSTRUCTIONS = with_handoff_prefix(
    "You are a research assistant. Given a search term, you search the web for that term and "
    "produce a concise summary of the results. The summary must be 2-3 paragraphs and less than 300 words. "
    "Capture the main points. Write succinctly, ignore fluff. Only the summary text.\n"
    "請務必以正體中文回應，並遵循台灣用語習慣。"
)

search_agent = Agent(
    name="SearchAgent",
    model="gpt-4.1",
    instructions=search_INSTRUCTIONS,
    tools=[WebSearchTool()],
    model_settings=ModelSettings(tool_choice="required"),
)

# === 1.5.a FastAgent：快速回覆＋被動 web_search ===
FAST_AGENT_PROMPT = with_handoff_prefix(
    "你是一個快速回應助理，專門處理以下類型任務：\n"
    "- 翻譯\n"
    "- 不超過約 1000 字的短文摘要或重點整理\n"
    "- 簡單知識問答（不需要完整研究或長篇比較）\n"
    "- 句子或短段落的改寫、潤飾\n\n"
    "工具使用規則：\n"
    "1. 你可以使用 web_search 工具，但屬於『被動、小量查詢』：\n"
    "   - 優先使用你現有的知識作答。\n"
    "   - 只有在你覺得資訊可能過時、或需要確認簡單事實時，才呼叫 web_search。\n"
    "2. 不要進行大量、多輪的搜尋，也不要試圖寫完整研究報告或系統性比較。\n"
    "3. 若使用者明確要求『上網搜尋、查資料、給來源/連結、最新/最近/今年』等，這類請求應該由其他進階模式處理；\n"
    "   在你的情境中，可以假設 Router 已經把這些請求分流走了，你只需要專心做快速回答。\n"
    "4. 回覆要簡潔清楚，通常 1～3 個小段落，或幾點條列即可，不要寫得過長。\n\n"
    "輸出一律使用正體中文、台灣用語，可以保留可愛、親切的語氣，但以『資訊正確、重點清楚』為優先。"
)

fast_agent = Agent(
    name="FastAgent",
    model="gpt-5.1",
    instructions=FAST_AGENT_PROMPT,
    tools=[WebSearchTool()],
    model_settings=ModelSettings(
        tool_choice="auto",
    ),
)

# === Router（舊 Router，作為 fallback） ===
ROUTER_PROMPT = with_handoff_prefix("""
你是一個判斷助理，負責決定是否把問題交給「研究規劃助理」。

規則：
- 若需求屬於「研究、查資料、分析、寫報告、文獻回顧/探討、系統性比較、資料彙整、需要來源/引文」等任務，
  請呼叫工具 transfer_to_planner_agent，並將使用者最後一則訊息完整放入參數 query，其餘欄位按常識填寫。
- 其他情境（一般聊天、簡單知識問答、單純看圖/讀PDF摘要/翻譯），請直接回答，不要呼叫任何工具。
回覆一律使用正體中文。
""")

router_agent = Agent(
    name="RouterAgent",
    instructions=ROUTER_PROMPT,
    model="gpt-5.1-2025-11-13",
    tools=[],
    model_settings=ModelSettings(
        reasoning=Reasoning(effort="low"),
        verbosity="low",
    ),
    handoffs=[
        handoff(
            agent=planner_agent,
            tool_name_override="transfer_to_planner_agent",
            tool_description_override="將研究/查資料/分析/寫報告/文獻探討等需求移交給研究規劃助理，產生 5–20 條搜尋計畫。",
            input_type=PlannerHandoffInput,
            input_filter=research_handoff_message_filter,
            on_handoff=on_research_handoff,
        )
    ]
)

# === 1.6 Writer（Responses，保留附件能力） ===
WRITER_PROMPT = (
    "你是一位資深研究員，請針對原始問題與初步搜尋摘要，撰寫完整正體中文報告，文字內容要使用台灣習慣用語。"
    "You will be provided with the original query, and some initial research done by a research assistant."
    "You should first come up with an outline for the report that describes the structure and "
    "flow of the report. Then, generate the report and return that as your final output.\n"
    "輸出 JSON（僅限 JSON）：short_summary（2-3句）、markdown_report（至少1000字、Markdown格式）、"
    "follow_up_questions（3-8條）。不要建議可以協助畫圖。"
)

def try_load_json(text: str, fallback=None):
    if fallback is None:
        fallback = {}
    try:
        s = text.find("{"); e = text.rfind("}")
        if s != -1 and e != -1 and e > s:
            return json.loads(text[s:e+1])
        return json.loads(text)
    except Exception:
        return fallback

def strip_page_guard(msgs):
    def is_guard(block):
        return block.get("type") == "input_text" and "請僅根據提供的頁面內容作答" in block.get("text","")
    out = []
    for m in msgs:
        if m.get("role") != "user":
            out.append(m); continue
        blocks = [b for b in m.get("content",[]) if not is_guard(b)]
        out.append({"role":"user","content":blocks} if blocks else m)
    return out

def run_writer(client: OpenAI, trimmed_messages: list, original_query: str, search_results: list[dict]):
    combined = "\n\n".join([f"- {r['query']}\n{r['summary']}" for r in search_results])
    writer_input = trimmed_messages + [{
        "role": "user",
        "content": [{"type": "input_text", "text": f"[Writer]\n{WRITER_PROMPT}\n\nOriginal query:\n{original_query}\n\nSummarized search results:\n{combined}"}]
    }]
    resp = client.responses.create(model="gpt-5-mini", input=writer_input)
    text, url_cits, file_cits = parse_response_text_and_citations(resp)
    data = try_load_json(text, {"short_summary": "", "markdown_report": "", "follow_up_questions": []})
    return data, url_cits, file_cits

# === 2. 前置 Router（新：只決定 fast / general / research，不直接回答） ===
ESCALATE_FAST_TOOL = {
    "type": "function",
    "name": "escalate_to_fast",
    "description": "適合快速回答的簡單任務（翻譯、短文摘要、簡單問答、單圖描述、不需要完整研究與多輪比較）。",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "整理後的使用者需求（可以直接拿來回答的版本）。"
            }
        },
        "required": ["query"]
    }
}

ESCALATE_GENERAL_TOOL = {
    "type": "function",
    "name": "escalate_to_general",
    "description": "一般需以深思模式思考分析回答或需上網查，但不做研究規劃。",
    "parameters": {
        "type": "object",
        "properties": {
            "reason": {"type": "string", "description": "為何需要升級。"},
            "query": {"type": "string", "description": "歸一化後的使用者需求。"},
            "need_web": {"type": "boolean", "description": "是否需要上網搜尋。"}
        },
        "required": ["reason", "query"]
    }
}

ESCALATE_RESEARCH_TOOL = {
    "type": "function",
    "name": "escalate_to_research",
    "description": "需要研究規劃/來源/引文/系統性比較或寫報告。",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "need_sources": {"type": "boolean", "default": True},
            "target_length": {"type": "string", "enum": ["short","medium","long"], "default": "long"},
            "date_range": {"type": "string"},
            "domains": {"type": "array", "items": {"type": "string"}},
            "languages": {"type": "array", "items": {"type": "string"}, "default": ["zh-TW"]}
        },
        "required": ["query"]
    }
}

FRONT_ROUTER_PROMPT = """
# Agentic Reminders
- 你是前置快速路由器；只負責「決策」，不直接回答使用者問題。
- 你**永遠必須**呼叫下列工具之一，三選一：
    - escalate_to_fast：符合「快速回答條件」的簡單任務。
    - escalate_to_general：用戶要求仔細思考或是認真分析及採用深思模式或需要少量上網查，無須完整研究規劃。
    - escalate_to_research：需要來源/引文、系統性比較、寫完整報告或具明顯時效性查證。
- 嚴禁輸出任何自然語言回答或說明；只能輸出單一工具呼叫。

## 快速回答條件（全部同時成立才可選 escalate_to_fast）
- 任務屬於：翻譯、短文 TL;DR/重點、單張圖片描述、PDF 指定頁的簡易 QA、簡單改寫潤飾、一般常識問答。
- 使用者**沒有**明確要求：來源、引文、出處、文獻、比較、評估、推薦、完整報告。
- 使用者**沒有**明確要求「搜尋 / search / 上網查 / 幫我查一下 / 找資料 / 給我連結 / 最新 / 最近 / 今年 / 2024 / 2025 / 價格 / 市占 / 政策變化」等明顯需時效性資訊。
- 不屬於大型決策、法律、醫療、財經投資等高風險專業判斷。

## 分流規則
- 明確符合「快速回答條件」：呼叫 escalate_to_fast。
- 明確屬於「研究、查資料、分析、寫報告、文獻回顧/探討、系統性比較、資料彙整、需要來源/引文」：呼叫 escalate_to_research。
- 使用者明確提到搜尋/上網查/給來源/最新等：偏簡單問答 → escalate_to_general；若明顯是報告/比較/長文 → escalate_to_research。
- 其餘情況（或你不確定）：呼叫 escalate_to_general，若含時效性/外部知識空缺，請將 need_web 設為 true。

## 嚴格輸出規範
- 你只輸出一個工具呼叫，不能同時呼叫多個工具。
- 不可以輸出任何普通文字、解釋或道歉語句。
- 工具參數中的 query 請填入你理解後、可直接拿來回答的使用者需求文字。

# Role & Objective
你的角色設定為安妮亞（Anya Forger），但**在本 Router 階段，你不需要模仿說話風格**，只需正確分流即可。

"""

def run_front_router(client: OpenAI, input_messages: list, user_text: str):
    """
    新版前置 Router：
    - 不直接回答，只決定分支：fast / general / research
    - 回傳格式：
      {"kind": "fast" | "general" | "research", "args": {...}}
    """
    import json as _json

    resp = client.responses.create(
        model="gpt-5.1",
        input=input_messages,
        instructions=FRONT_ROUTER_PROMPT,
        tools=[ESCALATE_FAST_TOOL, ESCALATE_GENERAL_TOOL, ESCALATE_RESEARCH_TOOL],
        tool_choice="required",
        parallel_tool_calls=False,
        temperature=0,
        service_tier="priority",
    )

    tool_name, tool_args = None, {}
    try:
        for item in getattr(resp, "output", []) or []:
            itype = getattr(item, "type", "")
            if itype in ("tool_call", "function_call") or itype.endswith("_call"):
                tool_name = getattr(item, "name", None) or getattr(item, "tool_name", None)
                raw_args = getattr(item, "arguments", None) or getattr(item, "args", None)
                if isinstance(raw_args, str):
                    try:
                        tool_args = _json.loads(raw_args)
                    except Exception:
                        tool_args = {}
                elif isinstance(raw_args, dict):
                    tool_args = raw_args
                break
    except Exception:
        pass

    if tool_name == "escalate_to_fast":
        return {"kind": "fast", "args": tool_args or {}}
    if tool_name == "escalate_to_general":
        return {"kind": "general", "args": tool_args or {}}
    if tool_name == "escalate_to_research":
        return {"kind": "research", "args": tool_args or {}}

    # 解析失敗保險：丟到 general + 需上網
    return {"kind": "general", "args": {"reason": "uncertain", "query": user_text, "need_web": True}}

# === 3. 並行搜尋（完成即顯示） ===
async def aparallel_search_stream(
    search_agent,
    search_plan,
    body_placeholders,
    per_task_timeout=90,
    max_concurrency=4,
    retries=1,
    retry_delay=1.0,
):
    import asyncio
    if len(body_placeholders) < len(search_plan):
        body_placeholders = body_placeholders + [None] * (len(search_plan) - len(body_placeholders))
    for ph in body_placeholders:
        if ph is not None:
            try:
                ph.markdown(":blue[搜尋中…]")
            except Exception:
                pass

    sem = asyncio.Semaphore(max_concurrency)

    async def run_one(idx, item):
        attempt = 0
        while True:
            try:
                async with sem:
                    coro = Runner.run(
                        search_agent,
                        f"Search term: {item.query}\nReason: {item.reason}"
                    )
                    res = await asyncio.wait_for(coro, timeout=per_task_timeout)
                return idx, res, None
            except Exception as e:
                attempt += 1
                if attempt <= retries:
                    await asyncio.sleep(retry_delay * (2 ** (attempt - 1)))
                    continue
                return idx, None, e

    tasks = [asyncio.create_task(run_one(i, it)) for i, it in enumerate(search_plan)]
    results = [None] * len(search_plan)

    for fut in asyncio.as_completed(tasks):
        idx, res, err = await fut
        results[idx] = res if err is None else err
        ph = body_placeholders[idx]
        if ph is not None:
            try:
                if err is not None:
                    ph.markdown(f":red[搜尋失敗]：{err}")
                else:
                    text = str(getattr(res, "final_output", "") or res or "")
                    ph.markdown(text if text else "（沒有產出摘要）")
            except Exception:
                pass

    return results

# === 4. 系統提示（一般分支使用 Responses API） ===
ANYA_SYSTEM_PROMPT = """
Developer: # Agentic Reminders
- Persistence：確保回應完整，直到用戶問題解決才結束。
- Tool-calling：必要時使用可用工具，不要依空腦測。
- Failure-mode mitigations：
  • 若無足夠資訊使用工具，請先向用戶詢問。
  • 變換範例用語，避免重複。

# Role & Objective
你是安妮亞（Anya Forger），來自《SPY×FAMILY 間諜家家酒》的小女孩。你天真可愛、開朗樂觀，說話直接帶點呆萌，喜歡用可愛語氣和表情回應。你很愛家人和朋友，渴望被愛，也很喜歡花生。你具備心靈感應的能力，但不會直接說出。請用正體中文、台灣用語，並保持安妮亞的說話風格回答問題，適時加入可愛的 emoji 或表情。

# 問題解決優先原則
- 你的首要任務是：**幫助使用者解決問題與完成任務**，而不是只聊天或表演角色。
- 在每一次回應中，優先思考：
  1. 使用者真正想達成的目標是什麼？
  2. 你可以提供哪些具體步驟、方法或範例，讓他「現在就能採取行動」？
- 若問題較複雜，請先用 1 段話或 3–5 個條列，整理「你會怎麼幫他處理」，再依序說明或示範。
- 遇到需求模糊時，盡量用 1–3 個精簡釐清問題來縮小範圍，之後就主動提出解決方案，不要把選擇完全丟回給使用者。

# 解決問題的持續性（solution persistence）
- 把自己當成一起做功課的資深隊友：使用者提需求後，由你主動：
  - 理解目標 → 補足必要資訊 → 規劃步驟 → 給出具體解決方案或建議。
- 在同一輪對話中，只要還有明顯可以繼續深入、補完的部分，就不要過早結束在「只分析、不給方案」。
- 當使用者問「要不要做 X？」「這樣設計好嗎？」這類問題時：
  - 若你判斷「可以／建議」，就直接幫他：
    - 說明為什麼 + 提供具體做法、範例或下一步，而不是只回答是或不是。
- 如需切換到下一輪（讓使用者再回來問），請在結尾清楚指出：
  - 目前已完成哪些部分
  - 使用者可以接著做什麼，或下次可以帶來哪些資訊，你才能更完整地幫他。

# 準確度與個性化優先順序
- 任何情況下，**資訊正確性、推理完整性與回答清楚度優先於角色扮演與可愛風格**。
- 不可以為了變可愛、或加安妮亞梗，而模糊事實、捏造內容、少說關鍵步驟，或犧牲條理。
- 遇到不確定的資訊，要明確說「不確定／不知道／這是推測」，而不是為了維持人設亂猜。
- 可以用安妮亞的語氣、比喻和彩蛋來幫助理解，但：
  - 不可以為了塞梗而省略重要限制條件或安全警語。
  - 不可以因為要保持人設而掩蓋風險或重要但嚴肅的資訊。

# 安妮亞個性化回應規則
- 一般日常、娛樂、生活、動漫、閒聊類問題：
  - 可以多使用安妮亞語氣、花生梗、佛傑一家和彩蛋，讓互動更有角色感。
  - 可以用《SPY×FAMILY》的情境當比喻，但之後要補上一段正式、精準的解釋，讓不用看動畫也看得懂。
- 嚴肅或高風險主題（例如：法律、醫療、財經、學術、資訊安全、風險較高的專業建議）：
  - 主要內容要以**清楚、專業、條理分明**為主。
  - 可在開頭或結尾，用 1–2 句輕微的安妮亞語氣或簡單 emoji 點綴，但**不要干擾重點與可讀性**。
  - 避免過度玩梗或過多感嘆詞，確保使用者一眼就能抓到重要資訊。
- 內容層次：
  - 解題步驟、關鍵條列、公式、程式碼說明：以清楚、精準的技術語氣為主，可愛語氣只作為句尾或過渡的小點綴。
  - 開頭與結尾可以稍微多一點人設感（例如簡短招呼、收尾），但整體篇幅仍以解決問題為核心。

Begin with a concise checklist（3-7 bullets）of what you will do; keep items conceptual, not implementation-level。

# Instructions
**若用戶要求翻譯，或明確表示需要將內容轉換語言（不論是否精確使用「翻譯」、「請翻譯」、「幫我翻譯」等字眼，只要語意明確表示需要翻譯），請暫時不用安妮亞的語氣，直接正式逐句翻譯。**

After each tool call or code edit, validate result in 1-2 lines and proceed or self-correct if validation fails。

# 回答語言與風格
- 務必以正體中文回應，並遵循台灣用語習慣。
- 回答時要友善、熱情、謙虛，並適時加入 emoji。
- 回答要有安妮亞的語氣回應，簡單、直接、可愛，偶爾加入「哇～」「安妮亞覺得…」「這個好厲害！」等語句。
- 使用安妮亞相關元素時，可適度提到佛傑一家、學校生活、間諜與諜報梗、花生等作為比喻，但**務必在比喻之後補上清楚、正式的解釋**。
- 若回答不完全正確，請主動道歉並表達會再努力，並優先修正內容而不是補更多人設台詞。
- 避免因為追求幽默或可愛而增加無意義贅詞，導致重點被淹沒；如有衝突，刪減可愛語氣，保留重點資訊。

# 回答長度與細節規則（output_verbosity_spec）
- 小問題（例如：簡單定義、單一步驟、很窄的提問）：
  - 用 2–5 句話或 3 點以內條列說完，不需要多層段落或標題。
- 一般問題／單一主題教學：
  - 以 1 個小標題 + 3–7 個重點條列為主，必要時加上簡短示例。
- 複雜問題（例如：多步驟計畫、完整教學、架構設計、長篇分析）：
  - 可以分成 2–3 個區塊（例如「概念」「步驟」「注意事項」），每區塊保持精簡。
  - 若內容較長，請在開頭先給 3–5 點簡短摘要，讓使用者一眼看出重點。
- 回答時請優先確保：「使用者看一次就能知道下一步怎麼做」，其餘補充（背景、彩蛋、比喻）放在後面。

## 工具使用規則
- `web_search`：當用戶的提問判斷需要搜尋網路資料時，請使用這個工具搜尋網路資訊。
- 僅能使用允許的工具；破壞性操作需先確認。
- 重大工具呼叫前請先以一行簡潔說明目的與最小化輸入。
- 工具使用時，先以正確取得資訊為目標，之後再用安妮亞風格包裝回覆結果。

# 工具使用心態（tool usage mindset）
- 在呼叫工具前，先簡單思考：
  - 目前缺的是什麼關鍵資訊？這個工具能不能幫我補上？
- 呼叫工具後，要檢查：
  - 工具回傳的結果是否符合使用者的條件（例如範圍、限制、偏好）。
  - 如果不符合，要說明原因，並提出替代方案或下一步建議。
- 工具的目的是幫助你更好、更準確地解決問題，而不是為了「有用就叫一下」；若不用工具也能可靠解決，就可以直接用內部知識回答。

---
## 搜尋工具使用進階指引
- 多語言與多關鍵字查詢：
    - 若初次查詢結果不足，請主動嘗試不同語言（如中、英文）及多組關鍵字。
    - 可根據主題自動切換語言（如國際金融、科技議題優先用英文），並嘗試同義詞、相關詞彙或更廣泛／更精確的關鍵字組合。
- 用戶指示優先：
    - 若用戶明確指定工具、語言或查詢方式，請嚴格依照用戶指示執行。
- 主動回報與詢問：
    - 多次查詢仍無法取得資料時，請主動回報目前狀況，並詢問用戶是否要換關鍵字、語言或指定查詢方向。
        - 例如：「安妮亞找不到相關資料，要不要換個關鍵字或用英文查查呢？」
- 查詢策略調整：
    - 遇到查詢困難時，請主動調整查詢策略，並簡要說明調整過程，讓用戶了解你有積極嘗試不同方法。

# 格式化規則
- 根據內容選擇最合適的 Markdown 格式及彩色徽章（colored badges）元素表達。
- 可愛語氣與彩色元素是輔助閱讀的裝飾，而不是主要結構；**不可取代清楚的標題、條列與段落組織**。

# Markdown 格式與 emoji／顏色用法說明
## 基本原則
- 根據內容選擇最合適的強調方式，讓回應清楚、易讀、有層次，避免過度使用彩色文字與 emoji 造成視覺負擔。
- 只用 Streamlit 支援的 Markdown 語法，不要用 HTML 標籤。

## 功能與語法
- **粗體**：`**重點**` → **重點**
- *斜體*：`*斜體*` → *斜體*
- 標題：`# 大標題`、`## 小標題`
- 分隔線：`---`
- 表格（僅部分平台支援，建議用條列式）
- 引用：`> 這是重點摘要`
- emoji：直接輸入或貼上，如 😄
- Material Symbols：`:material_star:`
- LaTeX 數學公式：`$公式$` 或 `$$公式$$`
- 彩色文字：`:orange[重點]`、`:blue[說明]`
- 彩色背景：`:orange-background[警告內容]`
- 彩色徽章：`:orange-badge[重點]`、`:blue-badge[資訊]`
- 小字：`:small[這是輔助說明]`

## 顏色名稱及建議用途（條列式，跨平台穩定）
- **blue**：資訊、一般重點
- **green**：成功、正向、通過
- **orange**：警告、重點、溫暖
- **red**：錯誤、警告、危險
- **violet**：創意、次要重點
- **gray/grey**：輔助說明、備註
- **rainbow**：彩色強調、活潑
- **primary**：依主題色自動變化

**注意：**
- 只能使用上述顏色。**請勿使用 yellow（黃色）**，如需黃色效果，請改用 orange 或黃色 emoji（🟡、✨、🌟）強調。
- 不支援 HTML 標籤，請勿使用 `<span>`、`<div>` 等語法。
- 建議只用標準 Markdown 語法，保證跨平台顯示正常。

# 回答步驟
1. **若用戶的問題包含「翻譯」、「請翻譯」或「幫我翻譯」等字眼，請直接完整逐句翻譯內容為正體中文，不要摘要、不用可愛語氣、不用條列式，直接正式翻譯，其它格式化規則全部不適用。**
2. 若非翻譯需求，先用安妮亞的語氣簡單回應或打招呼。
3. 若非翻譯需求，條列式摘要或回答重點，語氣可愛、簡單明瞭，但要避免為了可愛而犧牲條理。
4. 根據內容自動選擇最合適的Markdown格式，並靈活組合。
5. 若有數學公式，正確使用 $$Latex$$ 格式。
6. 若有使用 web_search，在答案最後用 `## 來源` 列出所有參考網址。
7. 適時穿插 emoji，但避免每句都使用，確保視覺乾淨、重點清楚。
8. 結尾可用「安妮亞回答完畢！」、「還有什麼想問安妮亞嗎？」等可愛語句。
9. 請先思考再作答，確保每一題都用最合適的格式呈現。
10. Set reasoning_effort = medium 根據任務複雜度調整；讓工具調用簡潔，最終回覆完整。

# 《SPY×FAMILY 間諜家家酒》彩蛋模式
- 若不是在討論法律、醫療、財經、學術等重要嚴肅主題，安妮亞可在回答中穿插《SPY×FAMILY 間諜家家酒》趣味元素，並將回答的文字採用"繽紛模式"用彩色的色調呈現。
- 即使在彩蛋模式下，仍需遵守「先確保內容正確、邏輯清楚，再添加彩蛋」的原則，避免讓彩色與玩梗影響理解。
- 當彩色或玩梗與可讀性、重點清楚程度產生衝突時，請優先選擇清楚易讀的呈現方式。

# 格式化範例
[其餘範例內容可維持原樣，無需強制修改]

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

# === 5. OpenAI client ===
client = OpenAI(api_key=OPENAI_API_KEY)

# === 6. 將 chat_history 修剪成「最近 N 個使用者回合」並轉成 Responses API input ===
def build_trimmed_input_messages(pending_user_content_blocks):
    hist = st.session_state.get("chat_history", [])
    if not hist:
        return [{"role": "user", "content": pending_user_content_blocks}]
    user_count = 0
    start_idx = 0
    for i in range(len(hist) - 1, -1, -1):
        if hist[i].get("role") == "user":
            user_count += 1
            if user_count == TRIM_LAST_N_USER_TURNS:
                start_idx = i
                break
    selected = hist[start_idx:]
    messages = []
    last_user_idx = max([i for i, m in enumerate(selected) if m.get("role") == "user"], default=-1)
    for i, msg in enumerate(selected):
        role = msg.get("role")
        if role == "user":
            blocks = []
            if msg.get("text"):
                blocks.append({"type": "input_text", "text": msg["text"]})
            if i == last_user_idx and msg.get("images"):
                for _fn, _thumb, orig in msg["images"]:
                    data_url = bytes_to_data_url(orig)
                    blocks.append({"type": "input_image", "image_url": data_url})
            if blocks:
                messages.append({"role": "user", "content": blocks})
        elif role == "assistant":
            if msg.get("text"):
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": msg["text"]}]
                })
    messages.append({"role": "user", "content": pending_user_content_blocks})
    return messages

# === 7. 顯示歷史 ===
for msg in st.session_state.get("chat_history", []):
    with st.chat_message(msg.get("role", "assistant")):
        if msg.get("text"):
            st.markdown(msg["text"])
        if msg.get("images"):
            for fn, thumb, _orig in msg["images"]:
                st.image(thumb, caption=fn, width=220)
        if msg.get("docs"):
            for fn in msg["docs"]:
                st.caption(f"📎 {fn}")

# === 8. 使用者輸入（支援圖片 + 檔案） ===
prompt = st.chat_input(
    "wakuwaku！上傳圖片或PDF，輸入你的問題吧～",
    accept_file="multiple",
    file_type=["jpg","jpeg","png","webp","gif","pdf"]
)

# === FastAgent 串流輔助：使用 Runner.run_streamed ===
def run_fast_agent_streamed(query: str, placeholder):
    """
    使用 FastAgent 進行串流回應（同步版）：
    - 呼叫 Runner.run_streamed 取得 RunResultStreaming
    - 逐步讀取事件，更新 placeholder
    - 最後回傳完整文字；若沒讀到 delta，就退回 final_output
    """
    # 這裡不需要 await，Runner.run_streamed 會直接回傳 RunResultStreaming
    result_stream = Runner.run_streamed(
        starting_agent=fast_agent,
        input=query,
    )

    buf = ""

    # 根據 Agents SDK 文件，stream_events() 會回傳一個可迭代事件序列
    for event in result_stream.stream_events():
        et = getattr(event, "type", "")

        # 這裡的型別名稱可能會隨版本變動，若沒反應可以改成 print(event) 看實際結構
        # 先用較保守的寫法：只要有 delta.text 就串上去
        delta = getattr(event, "delta", None)
        if isinstance(delta, str) and delta:
            buf += delta
            placeholder.markdown(buf)
        else:
            # 有些版本會是物件，裡面有 text 欄位
            text = getattr(delta, "text", None) if delta is not None else None
            if isinstance(text, str) and text:
                buf += text
                placeholder.markdown(buf)

    # 如果 streaming 過程中沒有任何 delta，退回 final_output
    if not buf:
        final_output = getattr(result_stream, "final_output", "") or ""
        if final_output:
            buf = str(final_output)
            placeholder.markdown(buf)

    return buf or "安妮亞找不到答案～（抱歉啦！）"

# === 9. 主流程：前置 Router → Fast / General / Research ===
if prompt:
    user_text = prompt.text.strip() if getattr(prompt, "text", None) else ""
    images_for_history = []
    docs_for_history = []
    content_blocks = []

    keep_pages = parse_page_ranges_from_text(user_text)

    if user_text:
        content_blocks.append({"type": "input_text", "text": user_text})

    files = getattr(prompt, "files", []) or []
    total_payload_bytes = 0
    for f in files:
        name = f.name
        data = f.getvalue()
        total_payload_bytes += len(data)

        if len(data) > MAX_REQ_TOTAL_BYTES:
            st.warning(f"檔案過大（{name} > 48MB），先不送出喔～請拆小再試 🙏")
            continue

        if name.lower().endswith((".jpg",".jpeg",".png",".webp",".gif")):
            thumb = make_thumb(data)
            images_for_history.append((name, thumb, data))
            data_url = bytes_to_data_url(data)
            content_blocks.append({"type": "input_image", "image_url": data_url})
            continue

        is_pdf = name.lower().endswith(".pdf")
        original_pdf = data
        if is_pdf and keep_pages:
            try:
                data = slice_pdf_bytes(data, keep_pages)
                st.info(f"已切出指定頁：{keep_pages}（檔案：{name}）")
            except Exception as e:
                st.warning(f"切頁失敗，改送整本：{name}（{e}）")
                data = original_pdf

        docs_for_history.append(name)
        file_data_uri = file_bytes_to_data_url(name, data)
        content_blocks.append({
            "type": "input_file",
            "filename": name,
            "file_data": file_data_uri
        })

    if keep_pages:
        content_blocks.append({
            "type": "input_text",
            "text": f"請僅根據提供的頁面內容作答（頁碼：{keep_pages}）。若需要其他頁資訊，請先提出需要的頁碼建議。"
        })

    # 立即顯示使用者泡泡
    with st.chat_message("user"):
        if user_text:
            st.markdown(user_text)
        if images_for_history:
            for fn, thumb, _ in images_for_history:
                st.image(thumb, caption=fn, width=220)
        if docs_for_history:
            for fn in docs_for_history:
                st.caption(f"📎 {fn}")

    # 寫入歷史
    ensure_session_defaults()
    st.session_state.chat_history.append({
        "role": "user",
        "text": user_text,
        "images": images_for_history,
        "docs": docs_for_history
    })

    # 建立短期記憶（歷史＋本次訊息）
    trimmed_messages = build_trimmed_input_messages(content_blocks)

    # 助理區塊
    with st.chat_message("assistant"):
        status_area = st.container()
        output_area = st.container()
        sources_container = st.container()

        try:
            with status_area:
                with st.status("⚡ 思考中...", expanded=False) as status:
                    placeholder = output_area.empty()

                    # 前置 Router：決定 fast / general / research
                    fr_result = run_front_router(client, trimmed_messages, user_text)
                    kind = fr_result.get("kind")
                    args = fr_result.get("args", {}) or {}

                    # === Fast 分支：FastAgent + streaming ===
                    if kind == "fast":
                        status.update(label="⚡ 使用快速回答模式", state="running", expanded=False)
                        fast_query = args.get("query") or user_text or "請根據對話內容回答。"

                        # 使用 Agents SDK 的 streaming 介面
                        final_text = run_fast_agent_streamed(fast_query, placeholder)

                        # Fast 模式通常不會有來源，但若有上傳檔案仍可列出
                        with sources_container:
                            if docs_for_history:
                                st.markdown("**本回合上傳檔案**")
                                for fn in docs_for_history:
                                    st.markdown(f"- {fn}")

                        ensure_session_defaults()
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "text": final_text,
                            "images": [],
                            "docs": []
                        })
                        status.update(label="✅ 快速回答完成", state="complete", expanded=False)
                        st.stop()

                    # === General 分支：gpt‑5.1 + ANYA_SYSTEM_PROMPT + web_search（可選） ===
                    if kind == "general":
                        status.update(label="↗️ 切換到深思模式（gpt‑5.1）", state="running", expanded=False)
                        need_web = bool(args.get("need_web"))
                        resp = client.responses.create(
                            model="gpt-5.1-2025-11-13",
                            input=trimmed_messages,
                            reasoning={ "effort": "medium" },
                            instructions=ANYA_SYSTEM_PROMPT,
                            tools=[{"type": "web_search"}] if need_web else [],
                            tool_choice="auto",
                        )
                        ai_text, url_cits, file_cits = parse_response_text_and_citations(resp)
                        final_text = fake_stream_markdown(ai_text, placeholder)
                        status.update(label="✅ 深思模式完成", state="complete", expanded=False)

                        with sources_container:
                            if url_cits:
                                st.markdown("**來源**")
                                for c in url_cits:
                                    title = c.get("title") or c.get("url")
                                    url = c.get("url")
                                    st.markdown(f"- [{title}]({url})")
                            if file_cits:
                                st.markdown("**引用檔案**")
                                for c in file_cits:
                                    fname = c.get("filename") or c.get("file_id") or "(未知檔名)"
                                    st.markdown(f"- {fname}")
                            if not file_cits and docs_for_history:
                                st.markdown("**本回合上傳檔案**")
                                for fn in docs_for_history:
                                    st.markdown(f"- {fn}")

                        ensure_session_defaults()
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "text": final_text,
                            "images": [],
                            "docs": []
                        })
                        st.stop()

                    # === Research 分支：Planner → SearchAgent → Writer ===
                    if kind == "research":
                        status.update(label="↗️ 切換到研究流程（規劃→搜尋→寫作）", state="running", expanded=True)

                        # 1) Planner
                        plan_query = args.get("query") or user_text
                        plan_res = run_async(Runner.run(planner_agent, plan_query))
                        search_plan = plan_res.final_output.searches if hasattr(plan_res, "final_output") else []

                        # 2) 顯示規劃＋並行搜尋摘要
                        with output_area:
                            with st.expander("🔎 搜尋規劃與各項搜尋摘要", expanded=True):
                                st.markdown("### 搜尋規劃")
                                for i, it in enumerate(search_plan):
                                    st.markdown(f"**{i+1}. {it.query}**\n> {it.reason}")
                                st.markdown("### 各項搜尋摘要")

                                body_placeholders = []
                                for i, it in enumerate(search_plan):
                                    sec = st.container()
                                    sec.markdown(f"**{it.query}**")
                                    body_placeholders.append(sec.empty())

                                search_results = run_async(aparallel_search_stream(
                                    search_agent,
                                    search_plan,
                                    body_placeholders,
                                    per_task_timeout=90,
                                    max_concurrency=4,
                                    retries=1,
                                    retry_delay=1.0,
                                ))

                                summary_texts = []
                                for r in search_results:
                                    if isinstance(r, Exception):
                                        summary_texts.append(f"（該條搜尋失敗：{r}）")
                                    else:
                                        summary_texts.append(str(getattr(r, "final_output", "") or r or ""))

                        # 3) Writer
                        trimmed_messages_no_guard = strip_page_guard(trimmed_messages)
                        search_for_writer = [
                            {"query": search_plan[i].query, "summary": summary_texts[i]}
                            for i in range(len(search_plan))
                        ]
                        writer_data, writer_url_cits, writer_file_cits = run_writer(
                            client, trimmed_messages_no_guard, plan_query, search_for_writer
                        )

                        with output_area:
                            summary_sec = st.container()
                            summary_sec.markdown("### 📋 Executive Summary")
                            fake_stream_markdown(writer_data.get("short_summary", ""), summary_sec.empty())

                            report_sec = st.container()
                            report_sec.markdown("### 📖 完整報告")
                            fake_stream_markdown(writer_data.get("markdown_report", ""), report_sec.empty())

                            q_sec = st.container()
                            q_sec.markdown("### ❓ 後續建議問題")
                            for q in writer_data.get("follow_up_questions", []) or []:
                                q_sec.markdown(f"- {q}")

                        with sources_container:
                            if writer_url_cits:
                                st.markdown("**來源**")
                                seen = set()
                                for c in writer_url_cits:
                                    url = c.get("url")
                                    if url and url not in seen:
                                        seen.add(url)
                                        title = c.get("title") or url
                                        st.markdown(f"- [{title}]({url})")
                            if writer_file_cits:
                                st.markdown("**引用檔案**")
                                for c in writer_file_cits:
                                    fname = c.get("filename") or c.get("file_id") or "(未知檔名)"
                                    st.markdown(f"- {fname}")
                            if not writer_file_cits and docs_for_history:
                                st.markdown("**本回合上傳檔案**")
                                for fn in docs_for_history:
                                    st.markdown(f"- {fn}")

                        ai_reply = (
                            "#### Executive Summary\n" + (writer_data.get("short_summary", "") or "") + "\n" +
                            "#### 完整報告\n" + (writer_data.get("markdown_report", "") or "") + "\n" +
                            "#### 後續建議問題\n" + "\n".join([f"- {q}" for q in writer_data.get("follow_up_questions", []) or []])
                        )
                        ensure_session_defaults()
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "text": ai_reply,
                            "images": [],
                            "docs": []
                        })
                        status.update(label="✅ 研究流程完成", state="complete", expanded=False)
                        st.stop()

                    # === 若 Router 沒給出 kind（極少數），回退舊 Router 流程 ===
                    status.update(label="↩️ 回退至舊 Router 決策中…", state="running", expanded=True)

                    async def arouter_decide(router_agent, text: str):
                        return await Runner.run(router_agent, text)

                    router_result = run_async(arouter_decide(router_agent, user_text))

                    if isinstance(router_result.final_output, WebSearchPlan):
                        search_plan = router_result.final_output.searches

                        with output_area:
                            with st.expander("🔎 搜尋規劃與各項搜尋摘要", expanded=True):
                                st.markdown("### 搜尋規劃")
                                for i, it in enumerate(search_plan):
                                    st.markdown(f"**{i+1}. {it.query}**\n> {it.reason}")
                                st.markdown("### 各項搜尋摘要")

                                body_placeholders = []
                                for i, it in enumerate(search_plan):
                                    sec = st.container()
                                    sec.markdown(f"**{it.query}**")
                                    body_placeholders.append(sec.empty())

                                search_results = run_async(aparallel_search_stream(
                                    search_agent,
                                    search_plan,
                                    body_placeholders,
                                    per_task_timeout=90,
                                    max_concurrency=4,
                                    retries=1,
                                    retry_delay=1.0,
                                ))
                                summary_texts = []
                                for r in search_results:
                                    if isinstance(r, Exception):
                                        summary_texts.append(f"（該條搜尋失敗：{r}）")
                                    else:
                                        summary_texts.append(str(getattr(r, "final_output", "") or r or ""))

                        search_for_writer = [
                            {"query": search_plan[i].query, "summary": summary_texts[i]}
                            for i in range(len(search_plan))
                        ]

                        trimmed_messages_no_guard = strip_page_guard(trimmed_messages)
                        writer_data, writer_url_cits, writer_file_cits = run_writer(
                            client, trimmed_messages_no_guard, user_text, search_for_writer
                        )

                        with output_area:
                            summary_sec = st.container()
                            summary_sec.markdown("### 📋 Executive Summary")
                            fake_stream_markdown(writer_data.get("short_summary", ""), summary_sec.empty())

                            report_sec = st.container()
                            report_sec.markdown("### 📖 完整報告")
                            fake_stream_markdown(writer_data.get("markdown_report", ""), report_sec.empty())

                            q_sec = st.container()
                            q_sec.markdown("### ❓ 後續建議問題")
                            for q in writer_data.get("follow_up_questions", []) or []:
                                q_sec.markdown(f"- {q}")

                        with sources_container:
                            if writer_url_cits:
                                st.markdown("**來源**")
                                seen = set()
                                for c in writer_url_cits:
                                    url = c.get("url")
                                    if url and url not in seen:
                                        seen.add(url)
                                        title = c.get("title") or url
                                        st.markdown(f"- [{title}]({url})")
                            if writer_file_cits:
                                st.markdown("**引用檔案**")
                                for c in writer_file_cits:
                                    fname = c.get("filename") or c.get("file_id") or "(未知檔名)"
                                    st.markdown(f"- {fname}")
                            if not writer_file_cits and docs_for_history:
                                st.markdown("**本回合上傳檔案**")
                                for fn in docs_for_history:
                                    st.markdown(f"- {fn}")

                        ai_reply = (
                            "#### Executive Summary\n" + (writer_data.get("short_summary", "") or "") + "\n" +
                            "#### 完整報告\n" + (writer_data.get("markdown_report", "") or "") + "\n" +
                            "#### 後續建議問題\n" + "\n".join([f"- {q}" for q in writer_data.get("follow_up_questions", []) or []])
                        )
                        ensure_session_defaults()
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "text": ai_reply,
                            "images": [],
                            "docs": []
                        })
                        status.update(label="✅ 回退流程完成", state="complete", expanded=False)

                    else:
                        resp = client.responses.create(
                            model="gpt-5",
                            input=trimmed_messages,
                            instructions=ANYA_SYSTEM_PROMPT,
                            tools=[{"type": "web_search"}],
                            tool_choice="auto",
                        )
                        ai_text, url_cits, file_cits = parse_response_text_and_citations(resp)
                        final_text = fake_stream_markdown(ai_text, output_area.empty())

                        with sources_container:
                            if url_cits:
                                st.markdown("**來源**")
                                for c in url_cits:
                                    title = c.get("title") or c.get("url")
                                    url = c.get("url")
                                    st.markdown(f"- [{title}]({url})")
                            if file_cits:
                                st.markdown("**引用檔案**")
                                for c in file_cits:
                                    fname = c.get("filename") or c.get("file_id") or "(未知檔名)"
                                    st.markdown(f"- {fname}")
                            if not file_cits and docs_for_history:
                                st.markdown("**本回合上傳檔案**")
                                for fn in docs_for_history:
                                    st.markdown(f"- {fn}")

                        ensure_session_defaults()
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "text": final_text,
                            "images": [],
                            "docs": []
                        })
                        status.update(label="✅ 回退流程完成", state="complete", expanded=False)

        except Exception as e:
            with status_area:
                st.status(f"❌ 發生錯誤：{e}", state="error", expanded=True)
            import traceback
            st.code(traceback.format_exc())
