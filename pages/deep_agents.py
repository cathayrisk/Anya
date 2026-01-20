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
from openai.types.responses import ResponseTextDeltaEvent
import os
from pypdf import PdfReader, PdfWriter
from datetime import datetime

import math
import hashlib
from dataclasses import dataclass
from typing import Literal, Optional, List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
import atexit

# ====== DocRAG deps（FAISS + LangChain BM25）======
import numpy as np
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# Optional OCR render for PDF
HAS_PYMUPDF = False
fitz = None
try:
    import fitz  # pymupdf
    HAS_PYMUPDF = True
except Exception:
    HAS_PYMUPDF = False


# ============================================================
# 0. Trimming / 大小限制（可調）
# ============================================================
TRIM_LAST_N_USER_TURNS = 18
MAX_REQ_TOTAL_BYTES = 48 * 1024 * 1024

# DocRAG knobs (default)
DOC_EMBED_MODEL = "text-embedding-3-small"
DOC_MODEL_PLANNER = "gpt-4.1-mini"   # planner 便宜快
DOC_MODEL_EVIDENCE = "gpt-5.2"
DOC_MODEL_WRITER = "gpt-5.2"
DOC_MODEL_OCR = "gpt-5.2"

DOC_CHUNK_SIZE = 900
DOC_CHUNK_OVERLAP = 150
DOC_EMBED_BATCH = 256
OCR_MAX_WORKERS = 2

# 檢索去 boilerplate 參數
MIN_CONTENT_CHARS = 120
MAX_EMAILS_IN_CHUNK = 2
MAX_DUP_CHAR_RUN = 14
MIN_UNIQUE_PAGES_AFTER_FILTER = 3


# ============================================================
# 0.1 取得 API Key
# ============================================================
OPENAI_API_KEY = (
    st.secrets.get("OPENAI_API_KEY")
    or st.secrets.get("OPENAI_KEY")
    or os.getenv("OPENAI_API_KEY")
)
if not OPENAI_API_KEY:
    st.error("找不到 OpenAI API Key，請在 .streamlit/secrets.toml 設定 OPENAI_API_KEY 或 OPENAI_KEY。")
    st.stop()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# ============================================================
# 1. Streamlit 頁面
# ============================================================
st.set_page_config(page_title="Anya Multimodal Agent + DocRAG(FAISS/BM25)", page_icon="🥜", layout="wide")
st.title("Anya Multimodal Agent + DocRAG（FAISS + BM25 + Multi-Query）")


# ============================================================
# 1.a Session 預設值
# ============================================================
def build_today_line() -> str:
    now = datetime.now()
    day = now.strftime("%d").lstrip("0")
    return f"Today's date is {now.strftime('%a %b')} {day}, {now.strftime('%Y')}."

def build_today_system_message():
    return {"role": "system", "content": [{"type": "input_text", "text": build_today_line()}]}

def ensure_session_defaults():
    if "chat_history" not in st.session_state or not isinstance(st.session_state.chat_history, list):
        st.session_state.chat_history = [{
            "role": "assistant",
            "text": "嗨嗨～安妮亞來了！用上方 popover 上傳文件後，直接問問題就可以～",
            "images": [],
            "docs": []
        }]

def ensure_doc_state():
    st.session_state.setdefault("doc_files", {})          # sig -> info {name, bytes, ext, ...}
    st.session_state.setdefault("doc_store", None)        # FaissBM25Store
    st.session_state.setdefault("doc_processed", set())   # sig set
    st.session_state.setdefault("doc_mq_n", 5)
    st.session_state.setdefault("doc_per_query_k", 10)
    st.session_state.setdefault("doc_fused_k", 10)
    st.session_state.setdefault("doc_filter_boilerplate", True)

ensure_session_defaults()
ensure_doc_state()


# ============================================================
# 共用：假串流打字效果
# ============================================================
def fake_stream_markdown(text: str, placeholder, step_chars=8, delay=0.02, empty_msg="安妮亞找不到答案～（抱歉啦！）"):
    buf = ""
    for i in range(0, len(text), step_chars):
        buf = text[: i + step_chars]
        placeholder.markdown(buf)
        time.sleep(delay)
    if not text:
        placeholder.markdown(empty_msg)
    return text

class AsyncLoopRunner:
    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def stop(self):
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except Exception:
            pass
        try:
            self._thread.join(timeout=2)
        except Exception:
            pass
        try:
            self._loop.close()
        except Exception:
            pass

@st.cache_resource(show_spinner=False)
def get_async_runner() -> AsyncLoopRunner:
    runner = AsyncLoopRunner()
    atexit.register(runner.stop)
    return runner

def run_async(coro):
    try:
        asyncio.get_running_loop()
        loop_running = True
    except RuntimeError:
        loop_running = False

    if not loop_running:
        return asyncio.run(coro)

    result_container = {"value": None, "error": None}

    def _runner():
        try:
            result_container["value"] = asyncio.run(coro)
        except Exception as e:
            result_container["error"] = e

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join()
    if result_container["error"] is not None:
        raise result_container["error"]
    return result_container["value"]


# ============================================================
# 1.1 圖片工具：縮圖 & data URL
# ============================================================
@st.cache_data(show_spinner=False, max_entries=256)
def make_thumb(imgbytes: bytes, max_w=220) -> bytes:
    im = Image.open(BytesIO(imgbytes))
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    im.thumbnail((max_w, max_w))
    out = BytesIO()
    out.seek(0)
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


# ============================================================
# 1.2 檔案工具：data URI
# ============================================================
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


# ============================================================
# 1.3 PDF：頁碼解析 / 切頁
# ============================================================
def parse_page_ranges_from_text(text: str) -> list[int]:
    if not text:
        return []
    text_wo_urls = re.sub(r"https?://\S+", " ", text)
    has_page_hint = bool(re.search(r"(頁|page|pages|第\s*\d+\s*頁)", text_wo_urls, flags=re.IGNORECASE))
    if not has_page_hint:
        return []
    pages = set()

    range_patterns = [
        r"第\s*(\d+)\s*[-~至到]\s*(\d+)\s*頁",
        r"(\d+)\s*[-–—]\s*(\d+)\s*頁",
        r"p(?:age)?s?\s*(\d+)\s*[-–—]\s*(\d+)",
    ]
    for pat in range_patterns:
        for m in re.finditer(pat, text_wo_urls, flags=re.IGNORECASE):
            a, b = int(m.group(1)), int(m.group(2))
            if a > 0 and b >= a:
                for p in range(a, b + 1):
                    pages.add(p)

    single_patterns = [r"第\s*(\d+)\s*頁", r"p(?:age)?\s*(\d+)"]
    for pat in single_patterns:
        for m in re.finditer(pat, text_wo_urls, flags=re.IGNORECASE):
            p = int(m.group(1))
            if p > 0:
                pages.add(p)

    if re.search(r"(頁|page|pages)", text_wo_urls, flags=re.IGNORECASE):
        for m in re.finditer(r"(?<!\d)(\d+)(?:\s*,\s*(\d+))+", text_wo_urls):
            nums = [int(x) for x in m.group(0).split(",") if x.strip().isdigit()]
            for n in nums:
                if n > 0:
                    pages.add(n)

    pages = {p for p in pages if 1 <= p <= 500}
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


# ============================================================
# 1.4 回覆解析：擷取文字 + citations
# ============================================================
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


# ============================================================
# Agents（保留結構：fast/general/research）
# ============================================================
def with_handoff_prefix(text: str) -> str:
    pref = (RECOMMENDED_PROMPT_PREFIX or "").strip()
    return f"{pref}\n{text}" if pref else text

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
        history = history[-6:]
    return HandoffInputData(input_history=history, pre_handoff_items=tuple(filtered.pre_handoff_items), new_items=tuple(filtered.new_items))

async def on_research_handoff(ctx: RunContextWrapper[None], input_data: PlannerHandoffInput):
    print(f"[handoff] research query: {input_data.query}")

planner_agent = Agent(
    name="PlannerAgent",
    instructions=with_handoff_prefix("你是研究規劃助理，請產生 5-20 條 web 搜尋 query（含 reason），用正體中文。"),
    model="gpt-5.2",
    model_settings=ModelSettings(reasoning=Reasoning(effort="medium")),
    output_type=WebSearchPlan,
)

search_agent = Agent(
    name="SearchAgent",
    model="gpt-5.2",
    instructions=with_handoff_prefix("你是研究助理，針對 Search term 產出精簡摘要（正體中文）。"),
    tools=[WebSearchTool()],
)

fast_agent = Agent(
    name="FastAgent",
    model="gpt-5.2",
    instructions=with_handoff_prefix("你是安妮亞風格快速助理，用正體中文回答，先重點後細節。"),
    tools=[WebSearchTool()],
    model_settings=ModelSettings(temperature=0, verbosity="low", tool_choice="auto"),
)


# ============================================================
# Front Router（fast/general/research）
# ============================================================
ESCALATE_FAST_TOOL = {"type": "function", "name": "escalate_to_fast", "description": "快速回答", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}
ESCALATE_GENERAL_TOOL = {"type": "function", "name": "escalate_to_general", "description": "一般深思回答（可選 web_search）", "parameters": {"type": "object", "properties": {"reason": {"type": "string"}, "query": {"type": "string"}, "need_web": {"type": "boolean"}}, "required": ["reason", "query"]}}
ESCALATE_RESEARCH_TOOL = {"type": "function", "name": "escalate_to_research", "description": "研究流程", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}
FRONT_ROUTER_PROMPT = "你是前置路由器（只決策，不回答）。永遠必須呼叫：escalate_to_fast / escalate_to_general / escalate_to_research。只輸出工具呼叫。"

def run_front_router(client: OpenAI, input_messages: list, user_text: str, runtime_messages: Optional[list] = None):
    import json as _json
    router_input = []
    if runtime_messages:
        router_input.extend(runtime_messages)
    router_input.extend(input_messages)

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=router_input,
        instructions=FRONT_ROUTER_PROMPT,
        tools=[ESCALATE_FAST_TOOL, ESCALATE_GENERAL_TOOL, ESCALATE_RESEARCH_TOOL],
        tool_choice="required",
        parallel_tool_calls=False,
        temperature=0,
        service_tier="priority",
    )

    tool_name, tool_args = None, {}
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

    if tool_name == "escalate_to_fast":
        return {"kind": "fast", "args": tool_args or {}}
    if tool_name == "escalate_to_general":
        return {"kind": "general", "args": tool_args or {}}
    if tool_name == "escalate_to_research":
        return {"kind": "research", "args": tool_args or {}}
    return {"kind": "general", "args": {"reason": "uncertain", "query": user_text, "need_web": True}}


# ============================================================
# OpenAI client
# ============================================================
client = OpenAI(api_key=OPENAI_API_KEY)


# ============================================================
# History -> Responses input
# ============================================================
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
                    blocks.append({"type": "input_image", "image_url": bytes_to_data_url(orig)})
            if blocks:
                messages.append({"role": "user", "content": blocks})
        elif role == "assistant":
            if msg.get("text"):
                messages.append({"role": "assistant", "content": [{"type": "output_text", "text": msg["text"]}]})
    messages.append({"role": "user", "content": pending_user_content_blocks})
    return messages


# ============================================================
# DocRAG core
# ============================================================
def sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def extract_pdf_text_pages_pypdf(pdf_bytes: bytes) -> list[Tuple[int, str]]:
    reader = PdfReader(BytesIO(pdf_bytes))
    out: list[Tuple[int, str]] = []
    for i, p in enumerate(reader.pages):
        try:
            t = p.extract_text() or ""
        except Exception:
            t = ""
        out.append((i + 1, norm_space(t)))
    return out

def analyze_pdf_text_quality(pdf_pages: list[Tuple[int, str]], min_chars_per_page: int = 40):
    if not pdf_pages:
        return 0, 0, 1.0, 0, 0.0
    lens = [len(t) for _, t in pdf_pages]
    blank = sum(1 for L in lens if L <= min_chars_per_page)
    total = max(1, len(lens))
    blank_ratio = blank / total
    text_pages = total - blank
    text_pages_ratio = text_pages / total
    return sum(lens), blank, blank_ratio, text_pages, text_pages_ratio

def should_suggest_ocr(pages: Optional[int], extracted_chars: int, blank_ratio: Optional[float]) -> bool:
    if pages is None or pages <= 0:
        return True
    if blank_ratio is not None and blank_ratio >= 0.6:
        return True
    avg = extracted_chars / max(1, pages)
    return avg < 120

@st.cache_data(show_spinner=False, max_entries=128)
def cached_pdf_quality(sig: str, pdf_bytes: bytes):
    pages = extract_pdf_text_pages_pypdf(pdf_bytes)
    extracted_chars, blank_pages, blank_ratio, text_pages, text_pages_ratio = analyze_pdf_text_quality(pages)
    return {
        "pages": len(pages),
        "extracted_chars": extracted_chars,
        "blank_pages": blank_pages,
        "blank_ratio": blank_ratio,
        "text_pages": text_pages,
        "text_pages_ratio": text_pages_ratio,
    }

def _img_bytes_to_data_url(img_bytes: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(img_bytes).decode()}"

def ocr_image_bytes(client: OpenAI, image_bytes: bytes, mime: str) -> str:
    resp = client.responses.create(
        model=DOC_MODEL_OCR,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "請擷取圖片中所有可見文字（含小字/註腳）。只輸出文字，不要評論。"},
                {"type": "input_image", "image_url": _img_bytes_to_data_url(image_bytes, mime)},
            ],
        }],
        truncation="auto",
    )
    return norm_space(resp.output_text or "")

def ocr_pdf_pages_parallel(client: OpenAI, pdf_bytes: bytes, dpi: int = 180) -> list[Tuple[int, str]]:
    if not HAS_PYMUPDF:
        raise RuntimeError("未安裝 pymupdf（fitz），無法做 PDF OCR。請 pip install pymupdf")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    def render_page(i: int) -> Tuple[int, bytes]:
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return i + 1, pix.tobytes("png")

    page_imgs = [render_page(i) for i in range(doc.page_count)]
    results: Dict[int, str] = {}

    def _one(page_no: int, img_bytes: bytes):
        try:
            results[page_no] = ocr_image_bytes(client, img_bytes, "image/png")
        except Exception:
            results[page_no] = ""

    with ThreadPoolExecutor(max_workers=OCR_MAX_WORKERS) as ex:
        futs = [ex.submit(_one, pno, b) for pno, b in page_imgs]
        for fut in as_completed(futs):
            _ = fut

    return [(pno, results.get(pno, "")) for pno, _b in page_imgs]

@st.cache_resource(show_spinner=False)
def get_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=DOC_CHUNK_SIZE,
        chunk_overlap=DOC_CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", "；", ";", "，", ",", " ", ""],
    )

def chunk_text(text: str) -> list[str]:
    text = norm_space(text)
    if not text:
        return []
    splitter = get_splitter()
    docs = splitter.create_documents([text])
    out = []
    for d in docs:
        t = norm_space(d.page_content)
        if t:
            out.append(t)
    return out

def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(model=DOC_EMBED_MODEL, input=texts, encoding_format="float")
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vecs / norms

def bm25_preprocess_zh_en(text: str) -> list[str]:
    t = (text or "").lower()
    return re.findall(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*|[\u4e00-\u9fff]", t)

def rrf_scores(rank_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for rl in rank_lists:
        for rank, cid in enumerate(rl, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return scores

@dataclass
class Chunk:
    chunk_id: str
    title: str
    page: Optional[int]
    text: str

class FaissBM25Store:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []
        self.bm25: Optional[BM25Retriever] = None

    def rebuild_bm25(self):
        if not self.chunks:
            self.bm25 = None
            return
        docs = [
            Document(
                page_content=c.text,
                metadata={"chunk_id": c.chunk_id, "title": c.title, "page": c.page if c.page is not None else "-"},
            )
            for c in self.chunks
        ]
        self.bm25 = BM25Retriever.from_documents(docs, k=24, preprocess_func=bm25_preprocess_zh_en)

    def add(self, vecs: np.ndarray, new_chunks: list[Chunk]):
        self.index.add(vecs.astype(np.float32))
        self.chunks.extend(new_chunks)
        self.rebuild_bm25()

    def search_semantic(self, qvec: np.ndarray, k: int = 10) -> list[Tuple[float, Chunk]]:
        if self.index.ntotal == 0:
            return []
        scores, idx = self.index.search(qvec.astype(np.float32), k)
        out = []
        for s, i in zip(scores[0], idx[0]):
            if i < 0 or i >= len(self.chunks):
                continue
            out.append((float(s), self.chunks[i]))
        return out

    def search_bm25(self, query: str, k: int = 16) -> list[Chunk]:
        if not self.bm25:
            return []
        self.bm25.k = max(1, int(k))
        docs = self.bm25.invoke(query)
        cid_to_chunk = {c.chunk_id: c for c in self.chunks}
        out = []
        for d in docs or []:
            cid = (d.metadata or {}).get("chunk_id")
            if cid and cid in cid_to_chunk:
                out.append(cid_to_chunk[cid])
        return out

    def search_hybrid(self, query: str, qvec: np.ndarray, k: int = 10) -> list[Tuple[float, Chunk]]:
        sem_hits = self.search_semantic(qvec, k=max(10, k))
        bm_chunks = self.search_bm25(query, k=max(16, k * 2))
        sem_rank = [ch.chunk_id for _, ch in sem_hits]
        bm_rank = [ch.chunk_id for ch in bm_chunks]
        fused = rrf_scores([sem_rank, bm_rank], k=60)

        cid_to_chunk: dict[str, Chunk] = {}
        for _, ch in sem_hits:
            cid_to_chunk[ch.chunk_id] = ch
        for ch in bm_chunks:
            cid_to_chunk.setdefault(ch.chunk_id, ch)

        items = list(cid_to_chunk.items())
        items.sort(key=lambda kv: fused.get(kv[0], 0.0), reverse=True)

        out: list[Tuple[float, Chunk]] = []
        for cid, ch in items[:k]:
            out.append((float(fused.get(cid, 0.0)), ch))
        return out

def render_chunks_for_model(chunks: list[Chunk], max_chars_each: int = 900) -> str:
    parts = []
    for c in chunks:
        head = f"[{c.title} p{c.page if c.page is not None else '-'}]"
        parts.append(head + "\n" + (c.text or "")[:max_chars_each])
    return "\n\n".join(parts)

# ---- 重要：boilerplate 偵測（避免免責聲明洗版）
_EMAIL_RE = re.compile(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}", re.IGNORECASE)
_BOILER_KEYWORDS = [
    "intended exclusively for",
    "disclaimer",
    "required disclosures",
    "analyst certification",
    "conflicts of interest",
    "not been approved",
    "municipal advisor",
    "authorised and regulated",
    "financial market supervisory",
    "cathayholdings.com.tw",
    "ubs.com/disclosures",
]

def _has_long_dup_run(s: str, n: int = MAX_DUP_CHAR_RUN) -> bool:
    if not s:
        return False
    run = 1
    prev = s[0]
    for ch in s[1:]:
        if ch == prev:
            run += 1
            if run >= n:
                return True
        else:
            run = 1
            prev = ch
    return False

def is_boilerplate_chunk(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < MIN_CONTENT_CHARS:
        return True

    emails = _EMAIL_RE.findall(t)
    if len(emails) >= MAX_EMAILS_IN_CHUNK:
        return True

    tl = t.lower()
    hit_kw = sum(1 for kw in _BOILER_KEYWORDS if kw in tl)
    if hit_kw >= 2:
        return True

    if _has_long_dup_run(t):
        return True

    # 很多報告的免責聲明會有大量國家/監理機關連續列舉
    if tl.count("authorised") + tl.count("regulated") + tl.count("distribute") >= 6:
        return True

    return False


DOC_PLANNER_PROMPT = """
你是「文件檢索 Planner」（像 websearch planner）。
你的目標：不是泛泛說「解釋報告」，而是產生能撈到『正文』的 queries。

只輸出 JSON：
{"queries":[{"query":"...","reason":"..."}, ...]}

規則：
- 產生 4~6 條。
- query 要「內容導向」：例如 executive summary / key takeaways / introduction / conclusion / methodology / figure / table / valuation / risks / main findings。
- 可加入中英混合關鍵字。
- 避免 query 只寫「explain report」「disclaimer」「intended exclusively」這種容易撈到免責聲明的。
- reason <= 20字。
""".strip()

DOC_EVIDENCE_PROMPT = """
你是研究助理。你會收到：使用者問題 + 文件摘錄（每段前有 [報告名稱 pN]）。
你必須只輸出『證據筆記』。

輸出格式固定：
### EVIDENCE
- 最多 10 點；每點一句、可核對；句尾必須保留引用 token（例如 [報告名稱 p2]）
### COVERAGE
- 2–5 點：覆蓋了什麼 / 缺什麼（要具體）
""".strip()

DOC_WRITER_PROMPT = """
你是寫作整理者。你會收到：使用者問題 + EVIDENCE。
規則：
- 只能用 EVIDENCE 的事實，不可腦補。
- 引用文件內容的句子，句尾要有 [報告名稱 pN]。
- 若不足以回答：寫「資料不足」，並說明『目前命中內容偏免責聲明/抽字失敗的可能原因』，再列 <=3 個可行動的下一步（例如開 OCR、指定頁碼、重新索引）。

輸出格式：
## 直接回答
- 3–8 點（句尾引用；若無法就寫資料不足）
## 你可以怎麼做（若資料不足）
- 1–3 點（可行動）
""".strip()

def doc_plan_queries(client: OpenAI, question: str, n: int) -> list[dict]:
    n = max(4, min(6, int(n)))
    resp = client.responses.create(
        model=DOC_MODEL_PLANNER,
        input=[
            {"role": "system", "content": DOC_PLANNER_PROMPT},
            {"role": "user", "content": f"問題：{question}\n請產生 {n} 條 query。"},
        ],
        truncation="auto",
    )
    data = None
    try:
        data = json.loads(resp.output_text or "")
    except Exception:
        data = None
    if not isinstance(data, dict):
        data = {}

    items = data.get("queries") if isinstance(data.get("queries"), list) else []
    out: list[dict] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        q = norm_space(it.get("query", ""))
        r = norm_space(it.get("reason", ""))
        if q:
            out.append({"query": q, "reason": r or "補召回"})

    # 保障：一定會有幾個「正文導向」fallback query
    hard_fallback = [
        {"query": "executive summary OR key takeaways OR conclusion", "reason": "抓摘要結論"},
        {"query": "introduction OR overview OR background", "reason": "抓前言概覽"},
        {"query": "figure OR table OR chart OR data", "reason": "抓圖表數據"},
    ]
    for it in hard_fallback:
        if len(out) >= n:
            break
        if all(x["query"].lower() != it["query"].lower() for x in out):
            out.append(it)

    # 也保留原始問題（但放最後，避免太泛）
    base = norm_space(question)
    if base and all(x["query"] != base for x in out) and len(out) < n:
        out.append({"query": base, "reason": "原始問題"})

    return out[:n]

def _filter_hits_boilerplate(hits: list[Tuple[float, Chunk]]) -> list[Tuple[float, Chunk]]:
    if not st.session_state.get("doc_filter_boilerplate", True):
        return hits
    out = []
    for s, ch in hits:
        if is_boilerplate_chunk(ch.text):
            continue
        out.append((s, ch))
    return out

def _unique_pages(hits: list[Tuple[float, Chunk]]) -> set:
    pages = set()
    for _s, ch in hits:
        pages.add((ch.title, ch.page))
    return pages

def doc_multi_query_fusion(
    client: OpenAI,
    store: FaissBM25Store,
    question: str,
    *,
    n_queries: int,
    per_query_k: int,
    fused_k: int,
) -> Tuple[list[dict], dict[str, list[Tuple[float, Chunk]]], list[Tuple[float, Chunk]]]:
    plan = doc_plan_queries(client, question, n=n_queries)

    per_query_hits: dict[str, list[Tuple[float, Chunk]]] = {}
    rank_lists: list[list[str]] = []
    cid_to_chunk: dict[str, Chunk] = {}

    for it in plan:
        q = it["query"]
        qvec = embed_texts(client, [q])
        hits = store.search_hybrid(q, qvec, k=per_query_k)
        hits = _filter_hits_boilerplate(hits)
        per_query_hits[q] = hits
        rank_lists.append([ch.chunk_id for _s, ch in hits])
        for _s, ch in hits:
            cid_to_chunk.setdefault(ch.chunk_id, ch)

    fused = rrf_scores(rank_lists, k=60)
    items = list(cid_to_chunk.items())
    items.sort(key=lambda kv: fused.get(kv[0], 0.0), reverse=True)
    fused_hits = [(float(fused.get(cid, 0.0)), ch) for cid, ch in items[:fused_k]]

    # 第二輪：如果過濾後仍然太少（通常正文抽不到/全被免責聲明洗版）
    if len(_unique_pages(fused_hits)) < MIN_UNIQUE_PAGES_AFTER_FILTER:
        extra_queries = [
            "abstract OR summary OR key takeaways",
            "conclusion OR recommendation OR risks",
            "methodology OR assumptions OR framework",
            "private REITs OR infrastructure REIT OR toll road OR data centre",
        ]
        extra_rank_lists: list[list[str]] = []
        extra_cid_to_chunk: dict[str, Chunk] = dict(cid_to_chunk)

        for q in extra_queries:
            qvec = embed_texts(client, [q])
            hits2 = store.search_hybrid(q, qvec, k=per_query_k)
            hits2 = _filter_hits_boilerplate(hits2)
            # 也記錄到 per_query_hits 讓 UI 看得到
            per_query_hits[f"[2nd] {q}"] = hits2
            extra_rank_lists.append([ch.chunk_id for _s, ch in hits2])
            for _s, ch in hits2:
                extra_cid_to_chunk.setdefault(ch.chunk_id, ch)

        fused2 = rrf_scores(rank_lists + extra_rank_lists, k=60)
        items2 = list(extra_cid_to_chunk.items())
        items2.sort(key=lambda kv: fused2.get(kv[0], 0.0), reverse=True)
        fused_hits = [(float(fused2.get(cid, 0.0)), ch) for cid, ch in items2[:fused_k]]

    return plan, per_query_hits, fused_hits

def doc_evidence_then_write(client: OpenAI, question: str, fused_hits: list[Tuple[float, Chunk]]) -> Tuple[str, str]:
    chunks = [ch for _s, ch in fused_hits]
    ctx = render_chunks_for_model(chunks, max_chars_each=900)

    evidence = client.responses.create(
        model=DOC_MODEL_EVIDENCE,
        input=[
            {"role": "system", "content": DOC_EVIDENCE_PROMPT},
            {"role": "user", "content": f"問題：{question}\n\n文件摘錄：\n{ctx}\n"},
        ],
        truncation="auto",
    ).output_text or ""

    answer = client.responses.create(
        model=DOC_MODEL_WRITER,
        input=[
            {"role": "system", "content": "你是安妮亞風格可靠助理，用正體中文（台灣用語）。"},
            {"role": "user", "content": f"{DOC_WRITER_PROMPT}\n\n問題：{question}\n\n=== EVIDENCE ===\n{evidence.strip()}\n"},
        ],
        truncation="auto",
    ).output_text or ""

    return (answer or "").strip(), (evidence or "").strip()

def doc_answer_insufficient(answer_text: str, evidence_text: str) -> bool:
    if "資料不足" in (answer_text or ""):
        return True
    n_bullets = len(re.findall(r"^\s*-\s+", evidence_text or "", flags=re.M))
    return n_bullets < 2

def doc_has_index() -> bool:
    store = st.session_state.get("doc_store")
    try:
        return bool(store and store.index and store.index.ntotal > 0)
    except Exception:
        return False

def doc_build_index_incremental(client: OpenAI):
    store: Optional[FaissBM25Store] = st.session_state.get("doc_store")
    processed: set = set(st.session_state.get("doc_processed") or set())
    files_map: dict = st.session_state.get("doc_files") or {}

    if store is None:
        dim = embed_texts(client, ["dim_probe"]).shape[1]
        store = FaissBM25Store(dim)
        st.session_state["doc_store"] = store

    new_chunks: list[Chunk] = []
    new_texts: list[str] = []

    for sig, info in files_map.items():
        if sig in processed:
            continue

        name = info["name"]
        data = info["bytes"]
        ext = info.get("ext") or os.path.splitext(name)[1].lower()
        use_ocr = bool(info.get("use_ocr", False))

        title = os.path.splitext(name)[0]
        report_id = sig[:10]
        pages: list[Tuple[Optional[int], str]] = []

        if ext == ".pdf":
            if use_ocr and not HAS_PYMUPDF:
                use_ocr = False
                info["ocr_error"] = "need_pymupdf"
            if use_ocr:
                pdf_pages = ocr_pdf_pages_parallel(client, data, dpi=180)
            else:
                pdf_pages = extract_pdf_text_pages_pypdf(data)
            pages = [(pno, txt) for pno, txt in pdf_pages]

        elif ext in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
            mime = "image/png"
            if ext in (".jpg", ".jpeg"):
                mime = "image/jpeg"
            txt = ocr_image_bytes(client, data, mime=mime)
            pages = [(None, txt)]
        else:
            pages = [(None, "")]

        for page_no, page_text in pages:
            if not page_text:
                continue
            chunks = chunk_text(page_text)
            for i, ch in enumerate(chunks):
                cid = f"{report_id}_p{page_no if page_no else 'na'}_c{i}"
                new_chunks.append(Chunk(chunk_id=cid, title=title, page=page_no if isinstance(page_no, int) else None, text=ch))
                new_texts.append(ch)

        processed.add(sig)

    if new_texts:
        vecs_list = []
        for i in range(0, len(new_texts), DOC_EMBED_BATCH):
            vecs_list.append(embed_texts(client, new_texts[i:i+DOC_EMBED_BATCH]))
        vecs = np.vstack(vecs_list)
        store.add(vecs, new_chunks)

    st.session_state["doc_processed"] = processed


# ============================================================
# UI helpers
# ============================================================
def _badge(label: str, color: str) -> str:
    safe = label.replace("[", "(").replace("]", ")")
    return f":{color}-badge[{safe}]"

def render_run_badges(mode: str, diff: str, db_calls: int, web_calls: int, enable_web: bool):
    parts = [
        _badge(f"Mode:{mode}", "gray"),
        _badge(f"Diff:{diff}", "blue"),
        _badge(f"DB:{db_calls}", "green" if db_calls else "gray"),
        _badge(f"Web:{web_calls}" if enable_web else "Web:off", "violet" if enable_web else "gray"),
    ]
    st.markdown(" ".join(parts))

def render_doc_debug(plan: list[dict], per_query_hits: dict, fused_hits: list[Tuple[float, Chunk]]):
    with st.expander("🧭 Doc Planner（queries + reasons）", expanded=False):
        for i, it in enumerate(plan, start=1):
            st.markdown(f"- **{i}. {it['query']}**  \n  :small[{it.get('reason','')}]")

    with st.expander("🔎 每條 query 命中（Top5）", expanded=False):
        # 只顯示前幾條 query，避免 UI 太長
        keys = list(per_query_hits.keys())[:12]
        for q in keys:
            st.markdown(f"#### {q}")
            hits = (per_query_hits.get(q) or [])[:5]
            if not hits:
                st.markdown(":small[（無命中）]")
                continue
            for s, ch in hits:
                snippet = (ch.text or "").replace("\n", " ")
                snippet = snippet[:260] + ("…" if len(snippet) > 260 else "")
                st.markdown(f"- **[{ch.title} p{ch.page if ch.page is not None else '-'}]** rrf={s:.4f}：{snippet}")

    with st.expander("🧩 融合後命中（RRF Top10）", expanded=False):
        for s, ch in (fused_hits or [])[:10]:
            snippet = (ch.text or "").replace("\n", " ")
            snippet = snippet[:300] + ("…" if len(snippet) > 300 else "")
            st.markdown(f"- **[{ch.title} p{ch.page if ch.page is not None else '-'}]** rrf={s:.4f}：{snippet}")


# ============================================================
# Popover：文件管理（不使用 sidebar）
# ============================================================
with st.popover("📦 文件管理（上傳/索引/OCR）"):
    store = st.session_state.get("doc_store")
    chunks_n = 0
    try:
        chunks_n = int(store.index.ntotal) if store else 0
    except Exception:
        chunks_n = 0

    if chunks_n > 0:
        st.success(f"已索引 chunks：{chunks_n}")
    else:
        st.info("尚未建立索引（上傳文件後按「建立/更新索引」）。")

    if HAS_PYMUPDF:
        st.caption(":green[OCR 可用（pymupdf 已安裝）]")
    else:
        st.caption(":orange[OCR 不可用（建議安裝 pymupdf 才能對掃描PDF做OCR）]")

    st.session_state.doc_filter_boilerplate = st.checkbox(
        "過濾免責/收件者/聯絡資訊（建議開）",
        value=bool(st.session_state.get("doc_filter_boilerplate", True)),
    )

    st.markdown("#### 🔧 multi‑query 參數")
    st.session_state.doc_mq_n = st.slider("multi-query 數量", 4, 6, int(st.session_state.doc_mq_n))
    st.session_state.doc_per_query_k = st.slider("每條 query 取回段落", 8, 16, int(st.session_state.doc_per_query_k))
    st.session_state.doc_fused_k = st.slider("融合後取回段落", 8, 16, int(st.session_state.doc_fused_k))

    st.markdown("---")
    st.markdown("#### 📤 上傳文件（可多選）")
    uploaded_docs = st.file_uploader(
        "支援 PDF / 圖片",
        type=["pdf", "png", "jpg", "jpeg", "webp", "gif"],
        accept_multiple_files=True,
    )

    if uploaded_docs:
        for f in uploaded_docs:
            name = f.name
            data = f.getvalue()
            if len(data) > MAX_REQ_TOTAL_BYTES:
                st.warning(f"檔案過大（{name} > 48MB），先不收進索引 🙏")
                continue
            ext = os.path.splitext(name)[1].lower()
            sig = sha1_bytes(data)

            if sig not in st.session_state.doc_files:
                info = {"name": name, "bytes": data, "ext": ext}

                if ext == ".pdf":
                    q = cached_pdf_quality(sig, data)
                    pages = q["pages"]
                    extracted_chars = q["extracted_chars"]
                    blank_ratio = q["blank_ratio"]
                    likely_scanned = should_suggest_ocr(pages, extracted_chars, blank_ratio)
                    info.update({
                        "pages": pages,
                        "extracted_chars": extracted_chars,
                        "blank_ratio": blank_ratio,
                        "likely_scanned": likely_scanned,
                        "use_ocr": bool(likely_scanned),  # 預設：疑似掃描就開
                    })

                st.session_state.doc_files[sig] = info

    files_map = st.session_state.get("doc_files") or {}
    if files_map:
        st.markdown("---")
        st.markdown("#### 📄 文件清單（最近 12 份）")
        for sig, info in list(files_map.items())[-12:]:
            name = info.get("name", "")
            ext = info.get("ext", "")
            if ext == ".pdf":
                blank_ratio = info.get("blank_ratio", None)
                extracted_chars = int(info.get("extracted_chars", 0) or 0)
                pages = int(info.get("pages", 0) or 0)
                likely = bool(info.get("likely_scanned", False))
                line = f"- {name}"
                if likely:
                    line += "  :orange[（可能掃描件，建議OCR）]"
                if blank_ratio is not None:
                    avg = extracted_chars / max(1, pages)
                    line += f"  :small[(blank_ratio={float(blank_ratio):.2f}, avg≈{avg:.0f}/page)]"
                st.markdown(line)
                info["use_ocr"] = st.checkbox("OCR 這份 PDF", value=bool(info.get("use_ocr", False)), key=f"ocr_{sig}")
            else:
                st.markdown(f"- {name}")

    colA, colB = st.columns(2)
    if colA.button("🚀 建立/更新索引", use_container_width=True):
        with st.status("DocRAG 建索引中…", expanded=False) as s:
            doc_build_index_incremental(client)
            s.update(label="DocRAG 索引完成", state="complete", expanded=False)
        st.rerun()

    if colB.button("🧹 清空索引", use_container_width=True):
        st.session_state.doc_store = None
        st.session_state.doc_processed = set()
        st.session_state.doc_files = {}
        st.rerun()


# ============================================================
# 7. 顯示歷史
# ============================================================
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


# ============================================================
# 8. 使用者輸入（支援圖片 + 檔案）
# ============================================================
prompt = st.chat_input(
    "wakuwaku！用 popover 上傳文件（或這裡直接附檔），然後輸入你的問題吧～",
    accept_file="multiple",
    file_type=["jpg", "jpeg", "png", "webp", "gif", "pdf"],
)

async def fast_agent_stream(query: str, placeholder) -> str:
    buf = ""
    result = Runner.run_streamed(fast_agent, input=query)
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            delta = event.data.delta or ""
            if not delta:
                continue
            buf += delta
            placeholder.markdown(buf)
    return buf or "安妮亞找不到答案～（抱歉啦！）"


# ============================================================
# 9. 主流程：Doc-first（有索引就先文件）→ 不足才走原始 router
# ============================================================
if prompt is not None:
    user_text = (prompt.text or "").strip()

    images_for_history = []
    docs_for_history = []
    content_blocks = []

    keep_pages = parse_page_ranges_from_text(user_text)

    files = getattr(prompt, "files", []) or []
    has_pdf_upload = False

    # ---- 收集本回合附件（同時：送給原始流程 + 加入 DocRAG file pool）
    for f in files:
        name = f.name
        data = f.getvalue()

        if len(data) > MAX_REQ_TOTAL_BYTES:
            st.warning(f"檔案過大（{name} > 48MB），先不送出喔～請拆小再試 🙏")
            continue

        ext = os.path.splitext(name)[1].lower()

        if ext in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
            thumb = make_thumb(data)
            images_for_history.append((name, thumb, data))
            content_blocks.append({"type": "input_image", "image_url": bytes_to_data_url(data)})

            sig = sha1_bytes(data)
            if sig not in st.session_state.doc_files:
                st.session_state.doc_files[sig] = {"name": name, "bytes": data, "ext": ext}
            continue

        if ext == ".pdf":
            has_pdf_upload = True

            original_pdf = data
            if keep_pages:
                try:
                    data = slice_pdf_bytes(data, keep_pages)
                    st.info(f"已切出指定頁：{keep_pages}（檔案：{name}）")
                except Exception as e:
                    st.warning(f"切頁失敗，改送整本：{name}（{e}）")
                    data = original_pdf

            docs_for_history.append(name)
            content_blocks.append({"type": "input_file", "filename": name, "file_data": file_bytes_to_data_url(name, data)})

            sig = sha1_bytes(data)
            if sig not in st.session_state.doc_files:
                info = {"name": name, "bytes": data, "ext": ".pdf"}
                q = cached_pdf_quality(sig, data)
                pages = q["pages"]
                extracted_chars = q["extracted_chars"]
                blank_ratio = q["blank_ratio"]
                likely_scanned = should_suggest_ocr(pages, extracted_chars, blank_ratio)
                info.update({
                    "pages": pages,
                    "extracted_chars": extracted_chars,
                    "blank_ratio": blank_ratio,
                    "likely_scanned": likely_scanned,
                    "use_ocr": bool(likely_scanned),
                })
                st.session_state.doc_files[sig] = info

    # ✅ 若本回合沒 PDF，頁碼就不套用（避免誤判）
    if keep_pages and not has_pdf_upload:
        keep_pages = []

    if keep_pages and has_pdf_upload:
        content_blocks.append({
            "type": "input_text",
            "text": f"請僅根據提供的頁面內容作答（頁碼：{keep_pages}）。若需要其他頁資訊，請先提出需要的頁碼建議。"
        })

    # ---- 顯示 user bubble
    with st.chat_message("user"):
        if user_text:
            st.markdown(user_text)
        if images_for_history:
            for fn, thumb, _ in images_for_history:
                st.image(thumb, caption=fn, width=220)
        if docs_for_history:
            for fn in docs_for_history:
                st.caption(f"📎 {fn}")

    # ---- 寫入歷史
    st.session_state.chat_history.append({"role": "user", "text": user_text, "images": images_for_history, "docs": docs_for_history})

    trimmed_messages = build_trimmed_input_messages(content_blocks)
    today_system_msg = build_today_system_message()

    with st.chat_message("assistant"):
        status_area = st.container()
        output_area = st.container()
        sources_container = st.container()

        with status_area:
            with st.status("⚡ 思考中...", expanded=False) as status:
                placeholder = output_area.empty()

                # =========================================================
                # ✅ Doc-first：有索引就先文件；不足也「不要回退亂講沒有檔案」
                # =========================================================
                if st.session_state.get("doc_files"):
                    status.update(label="📚 文件模式：更新索引中…", state="running", expanded=False)
                    doc_build_index_incremental(client)

                if doc_has_index():
                    status.update(label="📚 文件模式：Planner → multi-query → 檢索 → 整理", state="running", expanded=False)

                    store: FaissBM25Store = st.session_state.doc_store
                    n_queries = int(st.session_state.get("doc_mq_n", 5))
                    per_k = int(st.session_state.get("doc_per_query_k", 10))
                    fused_k = int(st.session_state.get("doc_fused_k", 10))

                    plan, per_query_hits, fused_hits = doc_multi_query_fusion(
                        client, store, user_text,
                        n_queries=n_queries, per_query_k=per_k, fused_k=fused_k
                    )

                    render_run_badges(mode="doc", diff="doc", db_calls=len(plan), web_calls=0, enable_web=False)
                    render_doc_debug(plan, per_query_hits, fused_hits)

                    answer_text, evidence_text = doc_evidence_then_write(client, user_text, fused_hits)

                    with st.expander("🧾 EVIDENCE（節錄）", expanded=False):
                        st.markdown((evidence_text or "")[:1600] if evidence_text else "（無）")

                    final_text = fake_stream_markdown(answer_text, placeholder)
                    st.session_state.chat_history.append({"role": "assistant", "text": final_text, "images": [], "docs": []})
                    status.update(label="✅ 文件模式完成", state="complete", expanded=False)

                    # 不足時：仍然回傳「資料不足 + 行動建議」，不要再跑原始流程去講「沒選檔案」
                    st.stop()

                # =========================================================
                # 沒索引：走原始 router（fast/general/research）
                # =========================================================
                fr_result = run_front_router(client, trimmed_messages, user_text, runtime_messages=[today_system_msg])
                kind = fr_result.get("kind")
                args = fr_result.get("args", {}) or {}

                has_image_or_file = any(b.get("type") in ("input_image", "input_file") for b in content_blocks)
                if has_image_or_file and kind == "fast":
                    kind = "general"
                    args = {"reason": "contains_image_or_file", "query": user_text or args.get("query") or "", "need_web": False}

                if kind == "fast":
                    status.update(label="⚡ 使用快速回答模式", state="running", expanded=False)
                    final_text = run_async(fast_agent_stream(user_text or "請回答使用者問題。", placeholder))
                    st.session_state.chat_history.append({"role": "assistant", "text": final_text, "images": [], "docs": []})
                    status.update(label="✅ 快速回答完成", state="complete", expanded=False)
                    st.stop()

                # 你原本 general/research 流程太長，這裡先略（你可以接回你自己的完整版本）
                status.update(label="（目前：無索引 fallback 未完整接回）", state="complete", expanded=False)
                placeholder.markdown("目前沒有索引，請先用 popover 上傳文件並建立索引，或把你的 general/research 原始流程接回來。")
                st.stop()
