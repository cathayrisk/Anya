# pages/deep_agents.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import io
import uuid
import math
import time
import json
import base64
import hashlib
import threading
import ast
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import numpy as np
import pandas as pd
import faiss
from pypdf import PdfReader

from openai import OpenAI

try:
    import fitz  # pymupdf
    HAS_PYMUPDF = True
except Exception:
    HAS_PYMUPDF = False


# =========================
# Streamlit config（只呼叫一次）
# =========================
st.set_page_config(page_title="研究報告助手（DeepAgent + Badges）", layout="wide")
st.title("研究報告助手（DeepAgent + Badges）")

# ✅ Markdown 顯示微調：字體/行距/標題大小/列表間距
st.markdown(
    """
<style>
/* 讓文章可讀性更好（你覺得內容 OK，但呈現需要調整） */
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1200px; }
.stMarkdown { line-height: 1.7; }
.stMarkdown h1 { font-size: 1.8rem; margin: 0.8rem 0 0.7rem; }
.stMarkdown h2 { font-size: 1.35rem; margin: 0.9rem 0 0.55rem; }
.stMarkdown h3 { font-size: 1.15rem; margin: 0.8rem 0 0.45rem; }
.stMarkdown p { margin: 0.35rem 0 0.6rem; }
.stMarkdown ul, .stMarkdown ol { margin: 0.35rem 0 0.75rem 1.2rem; }
.stMarkdown li { margin: 0.15rem 0; }
.stMarkdown blockquote { padding: 0.4rem 0.8rem; border-left: 0.25rem solid rgba(60, 60, 60, 0.25); }
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# DeepAgents / LangChain imports（可診斷版）
# =========================
HAS_DEEPAGENTS = False
DEEPAGENTS_IMPORT_ERRORS: list[str] = []

create_deep_agent = None
init_chat_model = None
ChatOpenAI = None

try:
    from deepagents import create_deep_agent as _create_deep_agent
    create_deep_agent = _create_deep_agent
except Exception as e:
    DEEPAGENTS_IMPORT_ERRORS.append(f"deepagents import failed: {repr(e)}")

try:
    from langchain.chat_models import init_chat_model as _init_chat_model
    init_chat_model = _init_chat_model
except Exception as e:
    DEEPAGENTS_IMPORT_ERRORS.append(f"langchain.chat_models.init_chat_model import failed: {repr(e)}")

try:
    from langchain_openai import ChatOpenAI as _ChatOpenAI
    ChatOpenAI = _ChatOpenAI
except Exception as e:
    DEEPAGENTS_IMPORT_ERRORS.append(f"langchain_openai.ChatOpenAI import failed: {repr(e)}")

HAS_DEEPAGENTS = (create_deep_agent is not None) and ((init_chat_model is not None) or (ChatOpenAI is not None))


def _require_deepagents():
    if HAS_DEEPAGENTS:
        return
    st.error("DeepAgent 依賴載入失敗（不一定是沒安裝，可能是版本/依賴不相容）。")
    if DEEPAGENTS_IMPORT_ERRORS:
        st.markdown("### 依賴錯誤細節（請把這段貼給我，我就能精準指你該裝哪個版本）")
        for msg in DEEPAGENTS_IMPORT_ERRORS:
            st.code(msg)
    else:
        st.info("（沒有捕捉到錯誤細節，請確認 deep_agents.py 是否已整檔覆蓋為最新版）")
    st.stop()


def _make_langchain_llm(model_name: str, temperature: float = 0.0, reasoning_effort: Optional[str] = None):
    """
    回傳 LangChain 的 chat model instance：
    - 優先 init_chat_model
    - fallback ChatOpenAI

    依你需求：
    - 推理需求高：才加 reasoning={"effort":"medium"}
    - 其餘：不設定 reasoning
    """
    if init_chat_model is not None:
        if model_name.startswith("openai:"):
            return init_chat_model(model=model_name, temperature=temperature)
        return init_chat_model(model=f"openai:{model_name}", temperature=temperature)

    if ChatOpenAI is not None:
        if model_name.startswith("openai:"):
            model_name = model_name.split("openai:", 1)[1]
        kwargs = dict(
            model=model_name,
            temperature=temperature,
            use_responses_api=True,
            max_completion_tokens=None,
        )
        if reasoning_effort in ("low", "medium", "high"):
            kwargs["reasoning"] = {"effort": reasoning_effort}
        return ChatOpenAI(**kwargs)

    raise RuntimeError("No LangChain LLM factory available.")


# =========================
# 固定模型設定（依你要求：都用 gpt-5.2）
# =========================
EMBEDDING_MODEL = "text-embedding-3-small"

MODEL_MAIN = "gpt-5.2"
MODEL_GRADER = "gpt-5.2"
MODEL_WEB = "gpt-5.2"

REASONING_EFFORT = "medium"  # ✅ 推理需求高才使用


# =========================
# 效能參數
# =========================
EMBED_BATCH_SIZE = 256
OCR_MAX_WORKERS = 2

CORPUS_DEFAULT_MAX_CHUNKS = 24
CORPUS_PER_REPORT_QUOTA = 6

# DeepAgent budgets
DA_MAX_DOC_SEARCH_CALLS = 14
DA_MAX_WEB_SEARCH_CALLS = 4
DA_MAX_REWRITE_ROUNDS = 2
DA_MAX_CLAIMS = 10

CHUNK_ID_LEAK_PAT = re.compile(r"(chunk_id\s*=\s*|_p(?:na|\d+)_c\d+)", re.IGNORECASE)

# ✅ 重要：你的內部 evidence 檔名不應出現在引用 badge
EVIDENCE_PATH_IN_CIT_RE = re.compile(r"\[(?:/)?evidence/[^ \]]+?\s+p(\d+|-)\s*\]", re.IGNORECASE)


# =========================
# 小工具
# =========================
def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def estimate_tokens_from_chars(n_chars: int) -> int:
    if n_chars <= 0:
        return 0
    return max(1, int(math.ceil(n_chars / 3.6)))


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    text = norm_space(text)
    if not text:
        return []
    out = []
    i = 0
    while i < len(text):
        j = min(len(text), i + chunk_size)
        out.append(text[i:j])
        if j == len(text):
            break
        i = max(0, j - overlap)
    return out


def sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def truncate_filename(name: str, max_len: int = 44) -> str:
    if len(name) <= max_len:
        return name
    base, ext = os.path.splitext(name)
    keep = max(10, max_len - len(ext) - 1)
    return f"{base[:keep]}…{ext}"


def _dedup_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


# =========================
# OpenAI client + wrappers
# =========================
def get_openai_api_key() -> str:
    if "OPENAI_KEY" in st.secrets and st.secrets["OPENAI_KEY"]:
        return st.secrets["OPENAI_KEY"]
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    if os.environ.get("OPENAI_KEY"):
        return os.environ["OPENAI_KEY"]
    raise RuntimeError("Missing OpenAI API key. Set st.secrets['OPENAI_KEY'] or env OPENAI_API_KEY.")


def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        encoding_format="float",
    )
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vecs / norms


def _to_messages(system: str, user: Any) -> list[Dict[str, Any]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_gpt(
    client: OpenAI,
    *,
    model: str,
    system: str,
    user: Any,
    reasoning_effort: Optional[str] = None,
    tools: Optional[list] = None,
    include_sources: bool = False,
) -> Tuple[str, Optional[list[Dict[str, Any]]]]:
    messages = _to_messages(system, user)
    resp = client.responses.create(
        model=model,
        input=messages,
        tools=tools,
        tool_choice="auto" if tools else "none",
        parallel_tool_calls=True if tools else None,
        reasoning={"effort": reasoning_effort} if reasoning_effort in ("low", "medium", "high") else None,
        include=["web_search_call.action.sources"] if (tools and include_sources) else None,
        truncation="auto",
    )
    out_text = resp.output_text
    sources = None
    if tools and include_sources:
        try:
            sources_list = []
            if hasattr(resp, "output") and resp.output:
                for item in resp.output:
                    d = item if isinstance(item, dict) else getattr(item, "__dict__", {})
                    if isinstance(d, dict) and d.get("type") == "web_search_call":
                        action = d.get("action", {}) or {}
                        if isinstance(action, dict) and action.get("sources"):
                            sources_list.extend(action["sources"])
            sources = sources_list if sources_list else None
        except Exception:
            sources = None
    return out_text, sources


# =========================
# 檔案 / OCR
# =========================
@dataclass
class FileRow:
    file_id: str
    file_sig: str
    name: str
    ext: str
    bytes_len: int
    pages: Optional[int]
    extracted_chars: int
    token_est: int

    text_pages: Optional[int]
    text_pages_ratio: Optional[float]

    blank_pages: Optional[int]
    blank_ratio: Optional[float]
    likely_scanned: bool
    use_ocr: bool


def extract_pdf_text_pages_pypdf(pdf_bytes: bytes) -> list[Tuple[int, str]]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    out = []
    for i, p in enumerate(reader.pages):
        try:
            t = p.extract_text() or ""
        except Exception:
            t = ""
        out.append((i + 1, norm_space(t)))
    return out


def extract_pdf_text_pages_pymupdf(pdf_bytes: bytes) -> list[Tuple[int, str]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    out = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        t = page.get_text("text") or ""
        out.append((i + 1, norm_space(t)))
    return out


def extract_pdf_text_pages(pdf_bytes: bytes) -> list[Tuple[int, str]]:
    if HAS_PYMUPDF:
        try:
            return extract_pdf_text_pages_pymupdf(pdf_bytes)
        except Exception:
            return extract_pdf_text_pages_pypdf(pdf_bytes)
    return extract_pdf_text_pages_pypdf(pdf_bytes)


def analyze_pdf_text_quality(
    pdf_pages: list[Tuple[int, str]],
    min_chars_per_page: int = 40,
) -> Tuple[int, int, float, int, float]:
    if not pdf_pages:
        return 0, 0, 1.0, 0, 0.0
    lens = [len(t) for _, t in pdf_pages]
    blank = sum(1 for L in lens if L <= min_chars_per_page)
    total_pages = max(1, len(lens))
    blank_ratio = blank / total_pages
    text_pages = total_pages - blank
    text_pages_ratio = text_pages / total_pages
    return sum(lens), blank, blank_ratio, text_pages, text_pages_ratio


def should_suggest_ocr(ext: str, pages: Optional[int], extracted_chars: int, blank_ratio: Optional[float]) -> bool:
    if ext != ".pdf":
        return False
    if pages is None or pages <= 0:
        return True
    if blank_ratio is not None and blank_ratio >= 0.6:
        return True
    avg = extracted_chars / max(1, pages)
    return avg < 120


def _img_bytes_to_data_url(img_bytes: bytes, mime: str = "image/png") -> str:
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def ocr_image_bytes(client: OpenAI, image_bytes: bytes, mime: str = "image/png") -> str:
    system = "你是一個OCR工具。只輸出可見文字與表格內容（若有表格用 Markdown 表格）。中文請用繁體中文。不要加評論。"
    user_content = [
        {"type": "input_text", "text": "請擷取圖片中所有可見文字（包含小字/註腳）。若無法辨識請標記[無法辨識]。"},
        {"type": "input_image", "image_url": _img_bytes_to_data_url(image_bytes, mime=mime)},
    ]
    text, _ = call_gpt(client, model=MODEL_GRADER, system=system, user=user_content, reasoning_effort=None)
    return text


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

    with ThreadPoolExecutor(max_workers=OCR_MAX_WORKERS) as ex:
        futs = {ex.submit(ocr_image_bytes, client, img_bytes, "image/png"): page_no for page_no, img_bytes in page_imgs}
        for fut in as_completed(futs):
            page_no = futs[fut]
            try:
                results[page_no] = norm_space(fut.result())
            except Exception:
                results[page_no] = ""

    return [(p, results.get(p, "")) for p, _ in page_imgs]


# =========================
# FAISS store
# =========================
@dataclass
class Chunk:
    chunk_id: str
    report_id: str
    title: str
    page: Optional[int]
    text: str


class FaissStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []

    def add(self, vecs: np.ndarray, chunks: list[Chunk]) -> None:
        self.index.add(vecs)
        self.chunks.extend(chunks)

    def search(self, qvec: np.ndarray, k: int = 8) -> list[Tuple[float, Chunk]]:
        if self.index.ntotal == 0:
            return []
        scores, idx = self.index.search(qvec.astype(np.float32), k)
        out = []
        for s, i in zip(scores[0], idx[0]):
            if i < 0 or i >= len(self.chunks):
                continue
            out.append((float(s), self.chunks[i]))
        return out


# =========================
# badges / citations / file-to-text
# =========================
CIT_RE = re.compile(r"\[[^\]]+?\s+p(\d+|-)\s*\]")
BULLET_RE = re.compile(r"^\s*(?:[-•*]|\d+\.)\s+")
CIT_PARSE_RE = re.compile(r"\[([^\]]+?)\s+p(\d+|-)\s*\]")


def file_to_text(file_obj: Any) -> str:
    """
    ✅ 修你最關鍵的點：從 dict(content=[...]) 取出真正 markdown 文字
    """
    if file_obj is None:
        return ""

    if isinstance(file_obj, dict):
        if "data" in file_obj:
            return file_to_text(file_obj.get("data"))
        if "content" in file_obj:
            return file_to_text(file_obj.get("content"))
        for k in ("text", "answer", "final", "output", "message"):
            if k in file_obj:
                return file_to_text(file_obj.get(k))
        try:
            return json.dumps(file_obj, ensure_ascii=False, indent=2)
        except Exception:
            return str(file_obj)

    if isinstance(file_obj, (bytes, bytearray)):
        return file_obj.decode("utf-8", errors="ignore")

    if isinstance(file_obj, str):
        return file_obj

    if isinstance(file_obj, (list, tuple)):
        parts: list[str] = []
        for x in file_obj:
            t = file_to_text(x).strip()
            if t:
                parts.append(t)
        return "\n".join(parts)

    return str(file_obj)


def get_files_text(files: Optional[dict], key: str) -> str:
    if not isinstance(files, dict) or key not in files:
        return ""
    return file_to_text(files.get(key)).strip()


def _parse_citations(cits: list[str]) -> list[Dict[str, str]]:
    parsed = []
    for c in cits:
        m = CIT_PARSE_RE.search(c)
        if m:
            parsed.append({"title": m.group(1).strip(), "page": m.group(2).strip()})
    return parsed


def _badge_directive(label: str, color: str) -> str:
    safe = label.replace("[", "(").replace("]", ")")
    return f":{color}-badge[{safe}]"


def _dedup_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _try_parse_json_or_py_literal(text: str) -> Optional[Any]:
    t = (text or "").strip()
    if not t:
        return None
    if t.startswith("{") or t.startswith("["):
        try:
            return json.loads(t)
        except Exception:
            pass
    if t.startswith("{") and t.endswith("}"):
        try:
            return ast.literal_eval(t)
        except Exception:
            return None
    return None


def _extract_main_text_from_payload(payload: Any) -> Optional[str]:
    if isinstance(payload, dict):
        for k in ("content", "answer", "final", "output", "text", "message"):
            if k not in payload:
                continue
            v = payload.get(k)
            if isinstance(v, str) and v.strip():
                return v
            if isinstance(v, (list, tuple)):
                joined = file_to_text(v).strip()
                if joined:
                    return joined

        msgs = payload.get("messages")
        if isinstance(msgs, list) and msgs:
            last = msgs[-1]
            if isinstance(last, dict):
                c = last.get("content")
                if isinstance(c, (str, list, tuple, dict)):
                    out = file_to_text(c).strip()
                    if out:
                        return out
            out = file_to_text(last).strip()
            return out or None

        return None

    if isinstance(payload, list):
        out = file_to_text(payload).strip()
        return out or None

    return None


def _extract_citation_strings(text: str) -> list[str]:
    if not text:
        return []
    return [m.group(0) for m in re.finditer(r"\[[^\]]+?\s+p(\d+|-)\s*\]", text)]


def _strip_citations_from_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s*\[[^\]]+?\s+p(\d+|-)\s*\]\s*", " ", text).strip()


def _group_citations_for_badges(cits: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for c in cits:
        m = CIT_PARSE_RE.search(c)
        if not m:
            continue
        title = m.group(1).strip()
        page = m.group(2).strip()
        grouped.setdefault(title, []).append(page)

    for title, pages in grouped.items():
        pages = _dedup_keep_order(pages)

        def _key(p: str):
            if p.isdigit():
                return (0, int(p))
            if p == "-":
                return (1, 10**9)
            return (2, p)

        grouped[title] = sorted(pages, key=_key)

    return grouped


def render_markdown_answer_with_source_badges(answer_text: str, badge_color: str = "green"):
    """
    - 正文不顯示引用
    - 來源用 badge 顯示「資料檔名 + 頁碼」
    - ✅ 自動忽略 /evidence/*.md 這種內部路徑引用（你說不該顯示）
    """
    raw = (answer_text or "").strip()

    if raw and CHUNK_ID_LEAK_PAT.search(raw):
        raw = CHUNK_ID_LEAK_PAT.sub("", raw)

    payload = _try_parse_json_or_py_literal(raw)
    if payload is not None:
        extracted = _extract_main_text_from_payload(payload)
        if extracted is not None:
            raw = extracted.strip()

    cits = _dedup_keep_order(_extract_citation_strings(raw))
    # ✅ 濾掉 evidence 路徑引用
    cits = [c for c in cits if not EVIDENCE_PATH_IN_CIT_RE.search(c)]

    clean = _strip_citations_from_text(raw)
    st.markdown(clean if clean else "（無內容）")

    if not cits:
        return

    grouped = _group_citations_for_badges(cits)

    doc_badges: list[str] = []
    web_badges: list[str] = []

    def _pages_str(pages: list[str], max_keep: int = 6) -> str:
        pages = pages or ["-"]
        if len(pages) <= max_keep:
            return "p" + ",".join(pages)
        kept = pages[:max_keep]
        return "p" + ",".join(kept) + "…"

    for title in sorted(grouped.keys()):
        pages = grouped[title]
        pages_part = _pages_str(pages, max_keep=6)

        is_web = title.strip().lower().startswith("websearch:")
        label = f"{title} {pages_part}"

        if is_web:
            web_badges.append(_badge_directive(label, "violet"))
        else:
            doc_badges.append(_badge_directive(label, badge_color))

    badges_line = []
    if doc_badges:
        badges_line.append(" ".join(doc_badges))
    if web_badges:
        badges_line.append(" ".join(web_badges))

    if badges_line:
        st.markdown(" ".join(badges_line))


def render_bullets_inline_badges(md_bullets: str, badge_color: str = "green"):
    lines = [l.rstrip() for l in (md_bullets or "").splitlines() if l.strip()]
    for line in lines:
        if not BULLET_RE.match(line):
            continue
        full_cits = [m.group(0) for m in re.finditer(r"\[[^\]]+?\s+p(\d+|-)\s*\]", line)]
        full_cits = [c for c in full_cits if not EVIDENCE_PATH_IN_CIT_RE.search(c)]
        clean = re.sub(r"\[[^\]]+?\s+p(\d+|-)\s*\]", "", line).strip()
        parsed = _parse_citations(full_cits)
        badges = [_badge_directive(f"{it['title']} p{it['page']}", badge_color) for it in parsed]
        st.markdown(clean + (" " + " ".join(badges) if badges else ""))


def bullets_all_have_citations(md: str) -> bool:
    lines = (md or "").splitlines()
    if not any(BULLET_RE.match(l) for l in lines):
        return False
    for line in lines:
        if BULLET_RE.match(line):
            # ✅ 一樣忽略 /evidence/*.md 這種不算有效引用
            cits = [m.group(0) for m in re.finditer(r"\[[^\]]+?\s+p(\d+|-)\s*\]", line)]
            cits = [c for c in cits if not EVIDENCE_PATH_IN_CIT_RE.search(c)]
            if not cits:
                return False
    return True


# =========================
# Debug panel（一定要先定義，避免 NameError）
# =========================
def render_debug_panel(files: Optional[dict]):
    """
    ✅ Tabs 全包在這裡，並且只會在主程式的 Debug expander 內呼叫
    """
    if not files or not isinstance(files, dict):
        st.write("（沒有 files）")
        return

    all_keys = sorted([k for k in files.keys() if isinstance(k, str)])
    evidence_keys = [k for k in all_keys if k.startswith("/evidence/")]

    draft = get_files_text(files, "/draft.md") if "/draft.md" in files else ""
    review = get_files_text(files, "/review.md") if "/review.md" in files else ""
    todos = get_files_text(files, "/workspace/todos.json") if "/workspace/todos.json" in files else ""

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["總覽", "todos.json", "draft.md", "review.md", "evidence"])

    with tab1:
        st.write(f"files keys：{len(all_keys)}")
        st.write(f"evidence：{len(evidence_keys)}")
        st.code("\n".join(all_keys[:500]), language="text")

    with tab2:
        if todos:
            st.code(todos[:20000], language="json")
        else:
            st.write("（沒有 /workspace/todos.json）")

    with tab3:
        if draft:
            st.code(draft[:20000], language="markdown")
        else:
            st.write("（沒有 /draft.md）")

    with tab4:
        if review:
            st.code(review[:20000], language="markdown")
        else:
            st.write("（沒有 /review.md）")

    with tab5:
        if evidence_keys:
            # 顯示 evidence 檔名清單（不把內容全塞爆）
            st.code("\n".join(evidence_keys[:800]), language="text")
        else:
            st.write("（沒有 /evidence/ 檔案）")


# =========================
# Indexing（增量：OCR + embeddings）
# =========================
def build_indices_incremental_no_kg(
    client: OpenAI,
    file_rows: list[FileRow],
    file_bytes_map: Dict[str, bytes],
    store: Optional[FaissStore],
    processed_keys: set,
    chunk_size: int = 900,
    overlap: int = 150,
) -> Tuple[FaissStore, Dict[str, Any], set]:
    if store is None:
        dim = embed_texts(client, ["dim_probe"]).shape[1]
        store = FaissStore(dim)

    stats = {"new_reports": 0, "new_chunks": 0}
    new_chunks: list[Chunk] = []
    new_texts: list[str] = []

    to_process = []
    for r in file_rows:
        key = (r.file_sig, bool(r.use_ocr))
        if key not in processed_keys:
            to_process.append(r)

    for row in to_process:
        data = file_bytes_map[row.file_id]
        report_id = row.file_id
        title = os.path.splitext(row.name)[0]  # ✅ 來源顯示用「檔名（不含副檔名）」很合理
        stats["new_reports"] += 1

        if row.ext == ".pdf":
            pages = ocr_pdf_pages_parallel(client, data) if row.use_ocr else extract_pdf_text_pages(data)
        elif row.ext == ".txt":
            pages = [(None, norm_space(data.decode("utf-8", errors="ignore")))]
        elif row.ext in (".png", ".jpg", ".jpeg"):
            mime = "image/jpeg" if row.ext in (".jpg", ".jpeg") else "image/png"
            txt = norm_space(ocr_image_bytes(client, data, mime=mime))
            pages = [(None, txt)]
        else:
            pages = [(None, "")]

        for page_no, page_text in pages:
            if not page_text:
                continue
            for i, ch in enumerate(chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)):
                cid = f"{report_id}_p{page_no if page_no else 'na'}_c{i}"
                new_chunks.append(Chunk(cid, report_id, title, page_no if isinstance(page_no, int) else None, ch))
                new_texts.append(ch)

        processed_keys.add((row.file_sig, bool(row.use_ocr)))

    if new_texts:
        vecs_list = []
        for i in range(0, len(new_texts), EMBED_BATCH_SIZE):
            vecs_list.append(embed_texts(client, new_texts[i:i + EMBED_BATCH_SIZE]))
        vecs = np.vstack(vecs_list)
        store.add(vecs, new_chunks)

    stats["new_chunks"] = len(new_chunks)
    return store, stats, processed_keys


# =========================
# 預設輸出（三份 bullets + citations）
# =========================
def _split_default_bundle(text: str) -> Dict[str, str]:
    t = (text or "").strip()
    pattern = re.compile(
        r"###\s*SUMMARY\s*(.*?)###\s*CLAIMS\s*(.*?)###\s*CHAIN\s*(.*)$",
        re.IGNORECASE | re.DOTALL,
    )
    m = pattern.search(t)
    if not m:
        return {"summary": "", "claims": "", "chain": ""}
    return {"summary": m.group(1).strip(), "claims": m.group(2).strip(), "chain": m.group(3).strip()}


def pick_corpus_chunks_for_default(all_chunks: list[Chunk]) -> list[Chunk]:
    by_title: Dict[str, list[Chunk]] = {}
    for c in all_chunks:
        by_title.setdefault(c.title, []).append(c)

    kw = re.compile(
        r"(outlook|risk|implication|forecast|scenario|inflation|rate|credit|spread|cap rate|vacancy|supply|demand|rental|office|retail|residential|logistics|hotel|reits)",
        re.I,
    )

    def score(c: Chunk) -> float:
        s = 0.0
        if kw.search(c.text or ""):
            s += 6.0
        if c.page is not None:
            s += max(0.0, 2.0 - min(2.0, float(c.page) / 6.0))
        s += min(2.0, len(c.text) / 1400.0)
        return s

    chosen: list[Chunk] = []
    for title, chunks in sorted(by_title.items(), key=lambda x: x[0]):
        by_page: Dict[int, list[Chunk]] = {}
        for c in chunks:
            p = c.page if c.page is not None else 0
            by_page.setdefault(p, []).append(c)

        page_best = []
        for p, cs in by_page.items():
            cs = sorted(cs, key=score, reverse=True)
            page_best.append(cs[0])

        page_best = sorted(page_best, key=score, reverse=True)
        chosen.extend(page_best[:CORPUS_PER_REPORT_QUOTA])

    chosen = sorted(chosen, key=score, reverse=True)[:CORPUS_DEFAULT_MAX_CHUNKS]
    return chosen


def render_chunks_for_model(chunks: list[Chunk], max_chars_each: int = 900) -> str:
    parts = []
    for c in chunks:
        head = f"[{c.title} p{c.page if c.page is not None else '-'}]"
        parts.append(head + "\n" + c.text[:max_chars_each])
    return "\n\n".join(parts)


def generate_default_outputs_bundle(client: OpenAI, title: str, ctx: str, max_retries: int = 2) -> Dict[str, str]:
    system = (
        "你是嚴謹的研究助理，只能根據我提供的資料回答，不可腦補。\n"
        "硬性規則：\n"
        "1) 你必須輸出三個區塊，且順序/標題固定：### SUMMARY、### CLAIMS、### CHAIN。\n"
        "2) 每個區塊都必須是純 bullet（每行以 - 開頭），不要段落。\n"
        "3) 每個 bullet 句尾必須附引用，格式固定：[報告名稱 p頁]\n"
        "4) 引用中的『報告名稱』必須是資料片段方括號內的那個名稱。\n"
        "5) 不可使用 /evidence/*.md 當作報告名稱。\n"
    )
    user = (
        f"請針對《{title}》一次輸出三份內容（融合多份報告）：\n"
        f"- SUMMARY：8~14 bullets\n"
        f"- CLAIMS：8~14 bullets\n"
        f"- CHAIN：6~12 bullets\n\n"
        f"資料：\n{ctx}\n"
    )

    last = ""
    for _ in range(max_retries + 1):
        out, _ = call_gpt(client, model=MODEL_MAIN, system=system, user=user, reasoning_effort=REASONING_EFFORT)
        parts = _split_default_bundle(out)
        ok = bullets_all_have_citations(parts["summary"]) and bullets_all_have_citations(parts["claims"]) and bullets_all_have_citations(parts["chain"])
        if ok:
            return parts
        last = out
        user += "\n\n【強制修正】整份重寫：三區塊皆為純 bullet，且每個 bullet 句尾都有 [報告名稱 p頁]；不得出現 /evidence/*.md。"
    return _split_default_bundle(last)


# =========================
# DeepAgent
# =========================
def ensure_deep_agent(client: OpenAI, store: FaissStore, enable_web: bool):
    _require_deepagents()
    from langchain_core.tools import BaseTool, StructuredTool

    st.session_state.setdefault("deep_agent", None)
    st.session_state.setdefault("deep_agent_web_flag", None)
    st.session_state.setdefault("da_usage", {"doc_search_calls": 0, "web_search_calls": 0})

    if (st.session_state.deep_agent is not None) and (st.session_state.deep_agent_web_flag == bool(enable_web)):
        return st.session_state.deep_agent

    lock = threading.Lock()
    usage = {"doc_search_calls": 0, "web_search_calls": 0}
    st.session_state["da_usage"] = usage

    def _inc(name: str, limit: int) -> bool:
        with lock:
            usage[name] += 1
            st.session_state["da_usage"] = usage
            return usage[name] <= limit

    def _get_usage_fn() -> str:
        with lock:
            return json.dumps(usage, ensure_ascii=False)

    def _doc_list_fn() -> str:
        by_title: Dict[str, int] = {}
        for c in store.chunks:
            by_title[c.title] = by_title.get(c.title, 0) + 1
        lines = [f"- {t} (chunks={n})" for t, n in sorted(by_title.items(), key=lambda x: x[0])]
        return "\n".join(lines) if lines else "（目前沒有任何已索引文件）"

    def _doc_search_fn(query: str, k: int = 8) -> str:
        if not _inc("doc_search_calls", DA_MAX_DOC_SEARCH_CALLS):
            return json.dumps({"hits": [], "error": f"Budget exceeded: doc_search_calls > {DA_MAX_DOC_SEARCH_CALLS}"}, ensure_ascii=False)

        q = (query or "").strip()
        if not q:
            return json.dumps({"hits": []}, ensure_ascii=False)

        qvec = embed_texts(client, [q])
        hits = store.search(qvec, k=max(1, min(12, int(k))))

        payload = {"hits": []}
        for score, ch in hits:
            payload["hits"].append({
                "title": ch.title,  # ✅ 這個 title 就是你要顯示在 badge 的「資料檔名」
                "page": str(ch.page) if ch.page is not None else "-",
                "chunk_id": ch.chunk_id,  # internal only
                "text": (ch.text or "")[:1200],
                "score": float(score),
            })
        return json.dumps(payload, ensure_ascii=False)

    def _doc_get_chunk_fn(chunk_id: str, max_chars: int = 2600) -> str:
        cid = (chunk_id or "").strip()
        if not cid:
            return ""
        for c in store.chunks:
            if c.chunk_id == cid:
                return (c.text or "")[:max_chars]
        return ""

    def _mk_tool(fn, name: str, description: str) -> BaseTool:
        return StructuredTool.from_function(fn, name=name, description=description)

    tool_get_usage = _mk_tool(_get_usage_fn, "get_usage", "Get current tool usage counters as JSON (budget/debug).")
    tool_doc_list = _mk_tool(_doc_list_fn, "doc_list", "List indexed documents and chunk counts.")
    tool_doc_search = _mk_tool(_doc_search_fn, "doc_search", "Semantic search over indexed chunks. Returns JSON hits with title/page/chunk_id/text.")
    tool_doc_get_chunk = _mk_tool(_doc_get_chunk_fn, "doc_get_chunk", "Fetch full text for a given chunk_id for close reading. Returns text only.")

    tools: list[BaseTool] = [tool_get_usage, tool_doc_list, tool_doc_search, tool_doc_get_chunk]

    tool_web_search_summary: Optional[BaseTool] = None
    if enable_web:
        def _web_search_summary_fn(query: str) -> str:
            if not _inc("web_search_calls", DA_MAX_WEB_SEARCH_CALLS):
                return f"[WebSearch:{(query or '')[:30]} p-]\nBudget exceeded: web_search_calls > {DA_MAX_WEB_SEARCH_CALLS}"

            q = (query or "").strip()
            if not q:
                return "[WebSearch:empty p-]\n（query 為空）"

            system = (
                "你是研究助理。用繁體中文（台灣用語）整理 web_search 結果成 2-4 段摘要，保留日期/名詞。"
                "若矛盾要指出。最後用 Sources: 列出 title + url。"
            )
            user = f"Search term: {q}"
            text, sources = call_gpt(
                client,
                model=MODEL_WEB,
                system=system,
                user=user,
                reasoning_effort=None,
                tools=[{"type": "web_search"}],
                include_sources=True,
            )
            src_lines = []
            for s in (sources or [])[:8]:
                if isinstance(s, dict):
                    t = s.get("title") or s.get("source") or "source"
                    u = s.get("url") or ""
                    if u:
                        src_lines.append(f"- {t} {u}".strip())

            out_text = (text or "").strip()
            if src_lines:
                out_text = (out_text + "\n\nSources:\n" + "\n".join(src_lines)).strip()

            return f"[WebSearch:{q[:30]} p-]\n" + out_text[:2400]

        tool_web_search_summary = _mk_tool(_web_search_summary_fn, "web_search_summary", "Run web_search and return a short Traditional Chinese summary with sources.")
        tools.append(tool_web_search_summary)

    # prompts
    retriever_prompt = f"""
你是文件檢索專家（只允許使用 doc_list/doc_search/doc_get_chunk/get_usage）。

facet 子任務格式：
facet_slug: <英文小寫_底線>
facet_goal: <這個面向要回答什麼>
hints: <可能的關鍵字/指標/名詞（可空）>

硬規則：
- 你要寫入 /evidence/doc_<facet_slug>.md
- evidence 內容只能包含：
  1) 引用標頭：[報告名稱 p頁]（不得包含 /evidence/ 路徑；不得用 doc_*.md 當報告名稱）
  2) 原文片段（可截斷）
  3) 一行說明「這段支持什麼」
- 你可以用 doc_search 拿到 chunk_id，然後用 doc_get_chunk(chunk_id=...) 精讀，
  但 chunk_id 絕對不能寫進 evidence。

若遇到 Budget exceeded：立刻停止並在 evidence 明講「證據不足」。
最後回覆 orchestrator：≤150 字摘要（找到什麼 + 最大缺口）
"""

    writer_prompt = f"""
你是寫作/整理專家（用 read_file/glob/grep/write_file/edit_file/ls）。
你必須整合 /evidence/ 底下所有檔案（doc_*.md 與可選 web_*.md）。

任務類型判斷：
- 完整報告/章節 → REPORT
- 單題回答 → QA
- 整理知識脈絡 → KNOWLEDGE
- 使用者貼草稿要查核/除錯 → VERIFY_DRAFT（最多 {DA_MAX_CLAIMS} 條主張）

引用規則（嚴格）：
- QA：純 bullet（每行 -），每個 bullet 句尾必有引用 [報告名稱 p頁] 或 [WebSearch:* p-]
- REPORT/KNOWLEDGE/VERIFY：Markdown；每個非標題段落至少 1 個引用
- enable_web=false：不得出現 WebSearch
- draft 絕對不能出現 chunk_id
- 報告名稱不得是 /evidence/*.md

把結果寫到 /draft.md
"""

    verifier_prompt = f"""
你是審稿查核專家（用 read_file/edit_file/grep）。
任務：檢查 /draft.md 是否符合引用覆蓋，並做『最少改動』修正。

規則：
- QA：每個 bullet 句尾必有 [.. p..]
- 其他：每個非標題段落至少 1 個引用 [.. p..]
- enable_web=false：不得出現 WebSearch
- 若 /draft.md 出現 chunk_id 痕跡（chunk_id= 或 _p*_c*），必須移除。
- 引用標頭不得使用 /evidence/*.md

最多修正 {DA_MAX_REWRITE_ROUNDS} 輪：
- 每輪：read /draft.md → edit_file 修正 → write /review.md 記錄
"""

    subagents = [
        {
            "name": "retriever",
            "description": "從上傳文件向量庫找證據，寫 /evidence/doc_*.md（不含 chunk_id）",
            "system_prompt": retriever_prompt,
            "tools": [tool_get_usage, tool_doc_list, tool_doc_search, tool_doc_get_chunk],
            "model": f"openai:{MODEL_MAIN}",
        },
        {
            "name": "writer",
            "description": "整合 /evidence/ → 產生 /draft.md",
            "system_prompt": writer_prompt,
            "tools": [],
            "model": f"openai:{MODEL_MAIN}",
        },
        {
            "name": "verifier",
            "description": "檢查引用覆蓋並修稿 /draft.md，寫 /review.md",
            "system_prompt": verifier_prompt,
            "tools": [],
            "model": f"openai:{MODEL_MAIN}",
        },
    ]

    if enable_web:
        web_prompt = """
你是網路搜尋專家（只允許 web_search_summary/get_usage；不允許 doc_*）。
facet 子任務格式同 retriever。

硬規則：
- 寫入 /evidence/web_<facet_slug>.md
- 每段要保留引用標頭 [WebSearch:... p-]
- 禁止捏造來源；若 Budget exceeded 就停止並寫明缺口
"""
        subagents.insert(
            1,
            {
                "name": "web-researcher",
                "description": "用 OpenAI 內建 web_search 做少量高品質搜尋，寫 /evidence/web_*.md",
                "system_prompt": web_prompt,
                "tools": [tool_web_search_summary, tool_get_usage] if tool_web_search_summary else [tool_get_usage],
                "model": f"openai:{MODEL_MAIN}",
            },
        )

    # ✅ 這裡改最重要：不要用「write_todos」這種不存在的詞；
    #    明確要求用 write_file 寫 /workspace/todos.json
    orchestrator_prompt = f"""
你是 Deep Doc Orchestrator（文件優先；enable_web={str(enable_web).lower()}）。

固定流程（必做）：
0) 立刻用 write_file 建立 /workspace/todos.json（JSON array of strings；5~9 步），並在第一步說明 enable_web 與目標。
1) write_file /evidence/README.md 記錄本次需求與 enable_web
2) 拆 2–4 個 facets（面向，不是章節）
3) 平行派工：
   - 每個 facet 至少派 1 個 retriever
   - enable_web=true 且需要外部背景時，對同 facet 再派 1 個 web-researcher
4) 叫 writer 產生 /draft.md
5) 叫 verifier 修稿（最多 {DA_MAX_REWRITE_ROUNDS} 輪）
6) read_file /draft.md 作為最終回答

引用與隱私規則：
- /evidence 與 /draft 絕對不能出現 chunk_id
- 引用只能用 [報告名稱 p頁] 或 [WebSearch:* p-]
- 報告名稱不得使用 /evidence/*.md 當作來源名稱
"""

    llm = _make_langchain_llm(model_name=f"openai:{MODEL_MAIN}", temperature=0.0, reasoning_effort=REASONING_EFFORT)

    agent = create_deep_agent(
        model=llm,
        tools=tools,
        system_prompt=orchestrator_prompt,
        subagents=subagents,
        debug=False,
        name="deep-doc-agent",
    ).with_config({"recursion_limit": 90})

    st.session_state.deep_agent = agent
    st.session_state.deep_agent_web_flag = bool(enable_web)
    return agent


# =========================
# DeepAgent run（status 不展開 + planning 後顯示 todos）
# =========================
def deep_agent_run_with_live_status(agent, user_text: str) -> Tuple[str, Optional[dict]]:
    final_state = None
    todos_preview_written = False

    def set_phase(s, phase: str):
        mapping = {
            "start": ("DeepAgent：啟動中…", "running"),
            "plan": ("DeepAgent：規劃中…", "running"),
            "evidence": ("DeepAgent：蒐證中…", "running"),
            "draft": ("DeepAgent：寫作中…", "running"),
            "review": ("DeepAgent：審稿/補引用中…", "running"),
            "done": ("DeepAgent：完成", "complete"),
            "error": ("DeepAgent：發生錯誤", "error"),
        }
        label, state = mapping.get(phase, ("DeepAgent：執行中…", "running"))
        s.update(label=label, state=state, expanded=False)

    with st.status("DeepAgent：啟動中…", expanded=False) as s:
        set_phase(s, "start")
        set_phase(s, "plan")

        try:
            for state in agent.stream(
                {"messages": [{"role": "user", "content": user_text}]},
                stream_mode="values",
            ):
                final_state = state
                files = state.get("files") or {}
                file_keys = set(files.keys()) if isinstance(files, dict) else set()

                # ✅ 規劃後如果 todos.json 出現，立刻顯示預覽在 status 裡
                if (not todos_preview_written) and isinstance(files, dict) and "/workspace/todos.json" in files:
                    todos_txt = get_files_text(files, "/workspace/todos.json")
                    if todos_txt:
                        s.write("### 本次 Todo（規劃結果預覽）")
                        s.code(todos_txt[:4000], language="json")
                        todos_preview_written = True

                if any(k.startswith("/evidence/") for k in file_keys):
                    set_phase(s, "evidence")
                if "/draft.md" in file_keys:
                    set_phase(s, "draft")
                if "/review.md" in file_keys:
                    set_phase(s, "review")

        except Exception as e:
            msg = str(e)
            if "Budget exceeded" in msg:
                set_phase(s, "evidence")
                s.update(label="DeepAgent：已達工具預算上限（停止加搜證）", state="running", expanded=False)
            else:
                try:
                    final_state = agent.invoke({"messages": [{"role": "user", "content": user_text}]})
                except Exception:
                    set_phase(s, "error")
                    raise

        files = (final_state or {}).get("files") or {}

        # ✅ 最終答案：一律取 /draft.md 的 content
        final_text = get_files_text(files, "/draft.md")

        if not final_text:
            msgs = (final_state or {}).get("messages") or []
            if msgs:
                last = msgs[-1]
                content = getattr(last, "content", None)
                final_text = (file_to_text(content) or file_to_text(last)).strip()

        if final_text and CHUNK_ID_LEAK_PAT.search(final_text):
            final_text = CHUNK_ID_LEAK_PAT.sub("", final_text)

        set_phase(s, "done")

    return final_text or "（DeepAgent 沒有產出內容）", files if isinstance(files, dict) and files else None


# =========================
# need_todo 判斷
# =========================
def decide_need_todo(client: OpenAI, question: str) -> Tuple[bool, str]:
    system = (
        "你是路由器。請判斷這個問題是否需要做『Todo + 文件/網路檢索』。\n"
        "規則：\n"
        "- 需要：涉及具體事實、數據、政策、版本、引用、研究、比較、出處；或需要引用上傳文件。\n"
        "- 不需要：純意見/寫作/改寫/腦力激盪/不要求引用的泛泛解釋。\n"
        "請輸出 JSON：{\"need_todo\": true/false, \"reason\": \"...\"}（只輸出 JSON）"
    )
    out, _ = call_gpt(
        client,
        model=MODEL_MAIN,
        system=system,
        user=question,
        reasoning_effort=REASONING_EFFORT,
        tools=None,
        include_sources=False,
    )
    data = _try_parse_json_or_py_literal(out) or {}
    need = bool(data.get("need_todo", False))
    reason = str(data.get("reason", "")).strip() or "（未提供原因）"
    return need, reason


def render_run_badges(*, mode: str, need_todo: bool, reason: str, usage: dict, enable_web: bool, todo_file_present: Optional[bool] = None):
    badges: List[str] = []
    badges.append(_badge_directive(f"Mode:{mode}", "gray"))

    if need_todo:
        badges.append(_badge_directive("Todo:需要", "blue"))
    else:
        badges.append(_badge_directive("Todo:不需要", "blue"))
        short_reason = reason if len(reason) <= 40 else reason[:40] + "…"
        badges.append(_badge_directive(f"理由:{short_reason}", "gray"))

    if todo_file_present is True:
        badges.append(_badge_directive("Todos.json:有", "blue"))
    elif todo_file_present is False and need_todo:
        badges.append(_badge_directive("Todos.json:無(流程異常)", "orange"))

    doc_calls = int((usage or {}).get("doc_search_calls", 0) or 0)
    web_calls = int((usage or {}).get("web_search_calls", 0) or 0)

    if doc_calls > 0:
        badges.append(_badge_directive(f"DB:used({doc_calls})", "green"))
    else:
        badges.append(_badge_directive("DB:unused", "gray"))

    if enable_web:
        if web_calls > 0:
            badges.append(_badge_directive(f"Web:used({web_calls})", "violet"))
        else:
            badges.append(_badge_directive("Web:unused", "gray"))
    else:
        badges.append(_badge_directive("Web:disabled", "gray"))

    st.markdown(" ".join(badges))


# =========================
# Session init
# =========================
OPENAI_API_KEY = get_openai_api_key()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ.setdefault("OPENAI_KEY", OPENAI_API_KEY)

client = get_client(OPENAI_API_KEY)

st.session_state.setdefault("file_rows", [])
st.session_state.setdefault("file_bytes", {})
st.session_state.setdefault("store", None)
st.session_state.setdefault("processed_keys", set())
st.session_state.setdefault("default_outputs", None)
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("enable_web_search_agent", False)


# =========================
# File table helpers
# =========================
def file_rows_to_df(rows: list[FileRow]) -> pd.DataFrame:
    recs = []
    for r in rows:
        if r.ext == ".pdf":
            pages_str = "-" if r.pages is None else str(r.pages)
            text_pages_str = "-" if r.text_pages is None else str(r.text_pages)
            text_ratio_str = "-" if r.text_pages_ratio is None else f"{r.text_pages_ratio:.0%}"
        else:
            pages_str = "-" if r.pages is None else str(r.pages)
            text_pages_str = "-"
            text_ratio_str = "-"

        if r.ext == ".pdf" and r.likely_scanned:
            suggest = "建議 OCR"
        elif r.ext in (".png", ".jpg", ".jpeg"):
            suggest = "必 OCR"
        else:
            suggest = ""

        recs.append({
            "_file_id": r.file_id,
            "使用OCR": bool(r.use_ocr),
            "檔名": truncate_filename(r.name, 52),
            "格式": r.ext.replace(".", ""),
            "頁數": pages_str,
            "文字頁": text_pages_str,
            "文字%": text_ratio_str,
            "token估算": int(r.token_est),
            "建議": suggest,
        })
    return pd.DataFrame(recs)


def sync_df_to_file_rows(df: pd.DataFrame, rows: list[FileRow]) -> None:
    id_to_row_idx = {r.file_id: i for i, r in enumerate(rows)}
    for _, rec in df.iterrows():
        fid = rec.get("_file_id")
        if fid not in id_to_row_idx:
            continue
        i = id_to_row_idx[fid]

        ext = rows[i].ext
        if ext in (".png", ".jpg", ".jpeg"):
            rows[i].use_ocr = True
        elif ext == ".txt":
            rows[i].use_ocr = False
        else:
            rows[i].use_ocr = bool(rec.get("使用OCR", rows[i].use_ocr))


# =========================
# Popover：文件管理 + DeepAgent 設定
# =========================
with st.popover("📦 文件管理（上傳 / OCR / 建索引 / DeepAgent設定）", use_container_width=True):
    st.caption("支援 PDF/TXT/PNG/JPG。PDF 若文字抽取偏少會建議 OCR（逐檔可勾選）。")
    st.caption("✅ 不上傳文件也能聊天；只有你需要引用文件時才需要建立索引。")

    st.session_state.enable_web_search_agent = st.checkbox(
        "啟用網路搜尋（會增加成本）",
        value=bool(st.session_state.enable_web_search_agent),
    )

    uploaded = st.file_uploader(
        "上傳文件",
        type=["pdf", "txt", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    if uploaded:
        existing = {(r.name, r.bytes_len) for r in st.session_state.file_rows}
        for f in uploaded:
            data = f.read()
            if (f.name, len(data)) in existing:
                continue

            ext = os.path.splitext(f.name)[1].lower()
            fid = str(uuid.uuid4())[:10]
            sig = sha1_bytes(data)
            st.session_state.file_bytes[fid] = data

            pages = None
            extracted_chars = 0
            blank_pages = None
            blank_ratio = None
            text_pages = None
            text_pages_ratio = None

            if ext == ".pdf":
                pdf_pages = extract_pdf_text_pages(data)
                pages = len(pdf_pages)
                extracted_chars, blank_pages, blank_ratio, text_pages, text_pages_ratio = analyze_pdf_text_quality(pdf_pages)
            elif ext == ".txt":
                extracted_chars = len(norm_space(data.decode("utf-8", errors="ignore")))

            token_est = estimate_tokens_from_chars(extracted_chars)
            likely_scanned = should_suggest_ocr(ext, pages, extracted_chars, blank_ratio)

            if ext in (".png", ".jpg", ".jpeg"):
                use_ocr = True
            elif ext == ".txt":
                use_ocr = False
            else:
                use_ocr = bool(likely_scanned)

            st.session_state.file_rows.append(
                FileRow(
                    file_id=fid,
                    file_sig=sig,
                    name=f.name,
                    ext=ext,
                    bytes_len=len(data),
                    pages=pages,
                    extracted_chars=extracted_chars,
                    token_est=token_est,
                    text_pages=text_pages,
                    text_pages_ratio=text_pages_ratio,
                    blank_pages=blank_pages,
                    blank_ratio=blank_ratio,
                    likely_scanned=likely_scanned,
                    use_ocr=use_ocr,
                )
            )

    st.markdown("### 文件清單（可逐檔勾選 OCR）")

    if not st.session_state.file_rows:
        st.info("尚未上傳文件。")
    else:
        df = file_rows_to_df(st.session_state.file_rows)

        edited = st.data_editor(
            df.drop(columns=["_file_id"]),
            key="file_table_editor",
            use_container_width=True,
            hide_index=True,
            disabled=["檔名", "格式", "頁數", "文字頁", "文字%", "token估算", "建議"],
            column_config={
                "使用OCR": st.column_config.CheckboxColumn("使用OCR", help="逐檔選擇是否啟用 OCR（PDF 可選；圖檔固定OCR；TXT固定不OCR）"),
            },
        )

        df_for_sync = df.copy()
        df_for_sync["使用OCR"] = edited["使用OCR"].values
        sync_df_to_file_rows(df_for_sync, st.session_state.file_rows)

        st.divider()
        col1, col2, col3 = st.columns([1, 1, 1])
        build_btn = col1.button("🚀 建立索引", type="primary", use_container_width=True)
        default_btn = col2.button("🧾 產生預設輸出", use_container_width=True)
        clear_btn = col3.button("🧹 清空全部", use_container_width=True)

        if clear_btn:
            st.session_state.file_rows = []
            st.session_state.file_bytes = {}
            st.session_state.store = None
            st.session_state.processed_keys = set()
            st.session_state.default_outputs = None
            st.session_state.chat_history = []
            st.session_state.deep_agent = None
            st.session_state.deep_agent_web_flag = None
            st.session_state.da_usage = {"doc_search_calls": 0, "web_search_calls": 0}
            st.rerun()

        if build_btn:
            need_ocr = any(r.ext == ".pdf" and r.use_ocr for r in st.session_state.file_rows)
            if need_ocr and not HAS_PYMUPDF:
                st.error("你有勾選 PDF OCR，但環境未安裝 pymupdf。請先 pip install pymupdf。")
                st.stop()

            with st.status("建索引中（OCR + embeddings）...", expanded=True) as s:
                t0 = time.perf_counter()
                store, stats, processed_keys = build_indices_incremental_no_kg(
                    client,
                    st.session_state.file_rows,
                    st.session_state.file_bytes,
                    st.session_state.store,
                    st.session_state.processed_keys,
                )
                st.session_state.store = store
                st.session_state.processed_keys = processed_keys
                s.write(f"新增報告數：{stats['new_reports']}")
                s.write(f"新增 chunks：{stats['new_chunks']}")
                s.write(f"耗時：{time.perf_counter() - t0:.2f}s")
                s.update(state="complete")

            st.session_state.deep_agent = None
            st.session_state.deep_agent_web_flag = None
            st.rerun()

        if default_btn:
            if st.session_state.store is None or st.session_state.store.index.ntotal == 0:
                st.warning("尚未建立索引或沒有 chunks，請先按「建立索引」。")
            else:
                with st.status("產生預設輸出（摘要/主張/推論鏈）...", expanded=True) as s2:
                    chosen = pick_corpus_chunks_for_default(st.session_state.store.chunks)
                    ctx = render_chunks_for_model(chosen)
                    bundle = generate_default_outputs_bundle(client, "整體融合（全部上傳報告）", ctx, max_retries=2)
                    st.session_state.default_outputs = bundle
                    s2.update(state="complete")

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "kind": "default",
                    "title": "整體融合（全部上傳報告）",
                    **(st.session_state.default_outputs or {}),
                })
                st.rerun()


# =========================
# 主畫面：狀態 + Chat
# =========================
has_index = (
    st.session_state.store is not None
    and getattr(st.session_state.store, "index", None) is not None
    and st.session_state.store.index.ntotal > 0
)

if has_index:
    st.success(f"已建立索引：檔案數={len(st.session_state.file_rows)} / chunks={len(st.session_state.store.chunks)}")
    st.caption("引用 badge 顯示『資料檔名 + 頁碼』；chunk_id 只在系統內部用來精讀與校對。")
else:
    st.info("目前沒有索引：你仍可直接聊天（純 LLM）。若需要引用文件，再去「文件管理」建立索引。")

st.divider()
st.subheader("Chat（DeepAgent + Badges + Todo decision）")

for msg in st.session_state.chat_history:
    with st.chat_message(msg.get("role", "assistant")):
        if msg.get("kind") == "default":
            st.markdown(f"## 預設輸出：{msg.get('title','')}")
            st.markdown("### 1) 報告摘要")
            render_bullets_inline_badges(msg.get("summary", ""), badge_color="green")
            st.markdown("### 2) 核心主張")
            render_bullets_inline_badges(msg.get("claims", ""), badge_color="violet")
            st.markdown("### 3) 推論鏈")
            render_bullets_inline_badges(msg.get("chain", ""), badge_color="orange")
        else:
            meta = msg.get("meta", {}) or {}
            render_run_badges(
                mode=meta.get("mode", "unknown"),
                need_todo=bool(meta.get("need_todo", False)),
                reason=str(meta.get("reason", "") or ""),
                usage=meta.get("usage", {}) or {},
                enable_web=bool(meta.get("enable_web", False)),
                todo_file_present=meta.get("todo_file_present", None),
            )
            render_markdown_answer_with_source_badges(msg.get("content", ""), badge_color="green")

prompt = st.chat_input("請輸入問題（也可貼草稿要我查核/除錯）。")
if prompt:
    st.session_state.chat_history.append({"role": "user", "kind": "text", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        enable_web = bool(st.session_state.enable_web_search_agent)
        need_todo, reason = decide_need_todo(client, prompt)

        if (not has_index) or (not need_todo):
            system = "你是助理。用繁體中文（台灣用語）回答，結構清楚。"
            answer_text, _ = call_gpt(
                client,
                model=MODEL_MAIN,
                system=system,
                user=prompt,
                reasoning_effort=None,
                tools=None,
            )
            meta = {
                "mode": "direct",
                "need_todo": bool(need_todo),
                "reason": reason,
                "usage": {"doc_search_calls": 0, "web_search_calls": 0},
                "enable_web": enable_web,
                "todo_file_present": None,
            }
            render_run_badges(
                mode=meta["mode"],
                need_todo=bool(need_todo),
                reason=reason,
                usage=meta["usage"],
                enable_web=enable_web,
                todo_file_present=None,
            )
            render_markdown_answer_with_source_badges(answer_text, badge_color="green")
            st.session_state.chat_history.append({"role": "assistant", "kind": "text", "content": answer_text, "meta": meta})
            st.stop()

        agent = ensure_deep_agent(client=client, store=st.session_state.store, enable_web=enable_web)
        answer_text, files = deep_agent_run_with_live_status(agent, prompt)

        todo_file_present = isinstance(files, dict) and ("/workspace/todos.json" in files)

        meta = {
            "mode": "deepagent",
            "need_todo": True,
            "reason": reason,
            "usage": dict(st.session_state.get("da_usage", {"doc_search_calls": 0, "web_search_calls": 0})),
            "enable_web": enable_web,
            "todo_file_present": bool(todo_file_present),
        }

        render_run_badges(
            mode=meta["mode"],
            need_todo=True,
            reason=reason,
            usage=meta["usage"],
            enable_web=enable_web,
            todo_file_present=meta["todo_file_present"],
        )
        render_markdown_answer_with_source_badges(answer_text, badge_color="green")

        # ✅ Debug：Tabs 全包在 Debug expander 內
        with st.expander("Debug", expanded=False):
            # 額外：在 Debug 最上方先顯示 todos 預覽（你想要「點開可看到 todos」）
            todos_txt = get_files_text(files, "/workspace/todos.json") if isinstance(files, dict) else ""
            if todos_txt:
                st.markdown("### 本次 Todo（完整）")
                st.code(todos_txt[:20000], language="json")
                st.divider()

            render_debug_panel(files)

    st.session_state.chat_history.append({"role": "assistant", "kind": "text", "content": answer_text, "meta": meta})
