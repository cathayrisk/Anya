# app.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
import io
import uuid
import math
import time
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import numpy as np
import faiss
import networkx as nx
from pypdf import PdfReader
from pydantic import BaseModel, Field, ValidationError

from openai import OpenAI
import langextract as lx

try:
    import fitz  # pymupdf
    HAS_PYMUPDF = True
except Exception:
    HAS_PYMUPDF = False


# =========================
# Streamlit config（只呼叫一次）
# =========================
st.set_page_config(page_title="研究報告助手（Web只做背景）", layout="wide")
st.title("研究報告助手（Web只做背景）")


# =========================
# 固定模型設定（依你需求）
# =========================
EMBEDDING_MODEL = "text-embedding-3-small"

MODEL_PLANNER = "gpt-5.2"
MODEL_GENERATE = "gpt-5.2"
MODEL_TRANSFORM = "gpt-5.2"

MODEL_GRADER = "gpt-4.1-mini"
MODEL_OCR = "gpt-4.1-mini"
MODEL_LANGEXTRACT = "gpt-4.1-mini"

# =========================
# 效能參數
# =========================
EMBED_BATCH_SIZE = 256
OCR_MAX_WORKERS = 2

LX_MAX_WORKERS_QUERY = 4
LX_MAX_CHUNKS_PER_QUERY = 8

CORPUS_DEFAULT_MAX_CHUNKS = 24
CORPUS_PER_REPORT_QUOTA = 6  # 讓預設輸出更分散頁碼

# web_search 觸發（預設 OFF；切到 AUTO 才會用）
MIN_RELEVANT_FOR_NO_WEB = 3
MIN_COVERAGE_RATIO = 0.45
MAX_WEB_SEARCHES = 4


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

def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:10]

def truncate_filename(name: str, max_len: int = 30) -> str:
    if len(name) <= max_len:
        return name
    base, ext = os.path.splitext(name)
    keep = max(10, max_len - len(ext) - 1)
    return f"{base[:keep]}…{ext}"

def is_web_citation_in_line(line: str) -> bool:
    return "WebSearch:" in line


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

def call_gpt52_reasoning(
    client: OpenAI,
    *,
    system: str,
    user: Any,
    effort: str = "medium",
    enable_web_search: bool = False,
    include_sources: bool = False,
) -> Tuple[str, Optional[list[Dict[str, Any]]]]:
    messages = _to_messages(system, user)
    resp = client.responses.create(
        model="gpt-5.2",
        input=messages,
        tools=[{"type": "web_search"}] if enable_web_search else None,
        tool_choice="auto" if enable_web_search else "none",
        parallel_tool_calls=True if enable_web_search else None,
        reasoning={"effort": effort},
        text={"verbosity": "medium"},
        include=[
            "web_search_call.action.sources",
            "message.input_image.image_url",
        ] if (enable_web_search and include_sources) else None,
        truncation="auto",
    )

    out_text = resp.output_text
    sources = None

    if enable_web_search and include_sources:
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

def call_gpt52_transform_effort_none(client: OpenAI, *, system: str, user: Any) -> str:
    messages = _to_messages(system, user)
    resp = client.responses.create(
        model="gpt-5.2",
        input=messages,
        reasoning={"effort": "none"},
        text={"verbosity": "medium"},
        truncation="auto",
    )
    return resp.output_text

def call_gpt41mini(client: OpenAI, *, system: str, user: Any) -> str:
    messages = _to_messages(system, user)
    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=messages,
        text={"verbosity": "medium"},
        truncation="auto",
    )
    return resp.output_text

def call_yesno_grader(client: OpenAI, *, system: str, user: str) -> str:
    out = call_gpt41mini(
        client,
        system=system + "\n\n只回覆 'yes' 或 'no'（小寫），不要加任何其他文字。",
        user=user,
    ).strip().lower()
    if "yes" in out and "no" in out:
        return "yes" if out.find("yes") < out.find("no") else "no"
    if "yes" in out:
        return "yes"
    if "no" in out:
        return "no"
    return "no"

def call_json_planner(client: OpenAI, *, system: str, user: str) -> str:
    text, _ = call_gpt52_reasoning(
        client,
        system=system + "\n\n你必須輸出「純 JSON」，不要用 Markdown code block，也不要加任何額外文字。",
        user=user,
        effort="medium",
        enable_web_search=False,
        include_sources=False,
    )
    return text.strip()


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

    # ✅ 新增：文字頁統計小工具
    text_pages: Optional[int]          # 有抽到文字的頁數
    text_pages_ratio: Optional[float]  # 文字頁比例（0~1）

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
    """
    回傳：
      total_chars, blank_pages, blank_ratio, text_pages, text_pages_ratio
    """
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

def ocr_image_bytes(client: OpenAI, image_bytes: bytes) -> str:
    system = "你是一個OCR工具。只輸出可見文字與表格內容（若有表格用 Markdown 表格）。中文請用繁體中文。不要加評論。"
    user_content = [
        {"type": "input_text", "text": "請擷取圖片中所有可見文字（包含小字/註腳）。若無法辨識請標記[無法辨識]。"},
        {"type": "input_image", "image_bytes": image_bytes},
    ]
    return call_gpt41mini(client, system=system, user=user_content)

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
        futs = {ex.submit(ocr_image_bytes, client, img_bytes): page_no for page_no, img_bytes in page_imgs}
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
# 引用顯示（略：沿用你現有）
# =========================
CIT_RE = re.compile(r"\[[^\]]+?\s+p(\d+|-)\s*\|\s*[A-Za-z0-9_\-]+\]")
BULLET_RE = re.compile(r"^\s*(?:[-•*]|\d+\.)\s+")
CIT_PARSE_RE = re.compile(r"\[([^\]]+?)\s+p(\d+|-)\s*\|\s*([A-Za-z0-9_\-]+)\]")

def _parse_citations(cits: list[str]) -> list[Dict[str, str]]:
    parsed = []
    for c in cits:
        m = CIT_PARSE_RE.search(c)
        if m:
            parsed.append({"title": m.group(1).strip(), "page": m.group(2).strip(), "chunk_id": m.group(3).strip()})
    return parsed

def _badge_directive(label: str, color: str) -> str:
    safe = label.replace("[", "(").replace("]", ")")
    return f":{color}-badge[{safe}]"

def render_bullets_inline_badges(md_bullets: str, badge_color: str = "green"):
    lines = [l.rstrip() for l in (md_bullets or "").splitlines() if l.strip()]
    for line in lines:
        if not BULLET_RE.match(line):
            continue
        full_cits = [m.group(0) for m in re.finditer(r"\[[^\]]+?\s+p(\d+|-)\s*\|\s*[A-Za-z0-9_\-]+\]", line)]
        clean = re.sub(r"\[[^\]]+?\s+p(\d+|-)\s*\|\s*[A-Za-z0-9_\-]+\]", "", line).strip()
        parsed = _parse_citations(full_cits)
        badges = [_badge_directive(f"{it['title']} p{it['page']} · {it['chunk_id']}", badge_color) for it in parsed]
        st.markdown(clean + (" " + " ".join(badges) if badges else ""))

def bullets_all_have_citations(md: str) -> bool:
    lines = (md or "").splitlines()
    if not any(BULLET_RE.match(l) for l in lines):
        return False
    for line in lines:
        if BULLET_RE.match(line) and not CIT_RE.search(line):
            return False
    return True


# =========================
# Planner（Pydantic 修正保持不變）
# =========================
class RetrievalQueryItem(BaseModel):
    reason: str = Field(...)
    query: str = Field(...)

class RetrievalPlan(BaseModel):
    needs_kg: bool = Field(...)
    queries: list[RetrievalQueryItem] = Field(...)

RetrievalQueryItem.model_rebuild()
RetrievalPlan.model_rebuild()

def plan_retrieval_queries(client: OpenAI, question: str) -> RetrievalPlan:
    system = """
你是 Planner。目標：把使用者問題拆成 5~12 條向量檢索 queries（每條要有 reason），以最大化覆蓋率。
輸出純 JSON（RetrievalPlan）。
"""
    user = f"使用者問題：{question}\n\n請輸出 RetrievalPlan JSON。"
    raw = call_json_planner(client, system=system, user=user)
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    raw = m.group(0) if m else raw
    try:
        plan = RetrievalPlan.model_validate_json(raw)
    except ValidationError:
        raw2 = call_json_planner(client, system=system + "\n⚠️ 只輸出可解析純 JSON。", user=user)
        m2 = re.search(r"\{.*\}", raw2, flags=re.DOTALL)
        raw2 = m2.group(0) if m2 else raw2
        plan = RetrievalPlan.model_validate_json(raw2)
    plan.queries = [q for q in plan.queries if q.query.strip()]
    if not plan.queries:
        plan.queries = [RetrievalQueryItem(reason="fallback", query=question)]
    return plan


def retrieve_by_plan(
    client: OpenAI,
    store: FaissStore,
    plan: RetrievalPlan,
    *,
    top_k_per_query: int = 4,
    max_total: int = 18,
) -> Tuple[list[Dict[str, Any]], Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    hit_queries = set()
    misses = []

    for item in plan.queries:
        qvec = embed_texts(client, [item.query])
        hits = store.search(qvec, k=top_k_per_query)

        if hits:
            hit_queries.add(item.query)
            for score, ch in hits:
                cur = by_id.get(ch.chunk_id)
                if (cur is None) or (score > cur["score"]):
                    by_id[ch.chunk_id] = {"chunk": ch, "score": float(score), "via_query": item.query, "via_reason": item.reason}
        else:
            misses.append({"query": item.query, "reason": item.reason})

    items = sorted(by_id.values(), key=lambda x: x["score"], reverse=True)[:max_total]
    total = max(1, len(plan.queries))
    hit_ratio = len(hit_queries) / total
    return items, {"total_queries": len(plan.queries), "hit_queries": len(hit_queries), "hit_ratio": hit_ratio, "misses": misses}


# =========================
# 建索引（不含 LangExtract）—略（同你現有）
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
        title = os.path.splitext(row.name)[0]
        stats["new_reports"] += 1

        if row.ext == ".pdf":
            if row.use_ocr:
                pages = ocr_pdf_pages_parallel(client, data)
            else:
                pages = extract_pdf_text_pages(data)
        elif row.ext == ".txt":
            pages = [(None, norm_space(data.decode("utf-8", errors="ignore")))]
        elif row.ext in (".png", ".jpg", ".jpeg"):
            txt = norm_space(ocr_image_bytes(client, data))
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
            vecs_list.append(embed_texts(client, new_texts[i:i+EMBED_BATCH_SIZE]))
        vecs = np.vstack(vecs_list)
        store.add(vecs, new_chunks)

    stats["new_chunks"] = len(new_chunks)
    return store, stats, processed_keys


# =========================
# Session init
# =========================
OPENAI_API_KEY = get_openai_api_key()
client = get_client(OPENAI_API_KEY)
api_key = OPENAI_API_KEY

if "file_rows" not in st.session_state:
    st.session_state.file_rows = []
if "file_bytes" not in st.session_state:
    st.session_state.file_bytes = {}
if "store" not in st.session_state:
    st.session_state.store = None
if "processed_keys" not in st.session_state:
    st.session_state.processed_keys = set()


# =========================
# Popover：文件管理（✅ 加上文字頁小工具顯示）
# =========================
with st.popover("📦 文件管理（上傳 / OCR / 建索引 / 設定）", width="content"):
    st.caption("支援 PDF/TXT/PNG/JPG。PDF 若文字抽取偏少會建議 OCR（逐檔可勾選）。")

    kg_mode = st.radio("KG 模式", options=["AUTO", "OFF", "FORCE"], index=0, horizontal=True, key="kg_mode")
    web_mode = st.radio("Web search（只做背景）", options=["OFF", "AUTO"], index=0, horizontal=True, key="web_mode")

    up = st.file_uploader("上傳文件", type=["pdf", "txt", "png", "jpg", "jpeg"], accept_multiple_files=True, key="uploader")
    if up:
        existing = {(r.name, r.bytes_len) for r in st.session_state.file_rows}
        for f in up:
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
            use_ocr = True if ext in (".png", ".jpg", ".jpeg") else bool(likely_scanned)

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

    if st.session_state.file_rows:
        st.markdown("### 文件清單（OCR / 檔名 / 頁(文X) / tok / 建議）")
        header = st.columns([1, 6, 1, 1, 1])
        header[0].markdown("**OCR**")
        header[1].markdown("**檔名**")
        header[2].markdown("**頁**")
        header[3].markdown("**tok**")
        header[4].markdown("**建議**")

        for idx, r in enumerate(st.session_state.file_rows):
            cols = st.columns([1, 6, 1, 1, 1])

            if r.ext in (".png", ".jpg", ".jpeg"):
                st.session_state.file_rows[idx].use_ocr = True
                cols[0].checkbox(" ", value=True, key=f"ocr_{idx}", disabled=True)
            elif r.ext == ".txt":
                st.session_state.file_rows[idx].use_ocr = False
                cols[0].checkbox(" ", value=False, key=f"ocr_{idx}", disabled=True)
            else:
                st.session_state.file_rows[idx].use_ocr = cols[0].checkbox(" ", value=bool(r.use_ocr), key=f"ocr_{idx}")

            short = truncate_filename(r.name, 34)

            # ✅ tooltip 加入「文字頁比例」資訊
            tip = [f"原檔名：{r.name}"]
            if r.ext == ".pdf":
                if r.pages is not None and r.text_pages is not None and r.text_pages_ratio is not None:
                    tip.append(f"文字頁：{r.text_pages}/{r.pages}（{r.text_pages_ratio:.0%}）")
                if r.blank_pages is not None and r.blank_ratio is not None:
                    tip.append(f"空白頁（<=40 chars）：{r.blank_pages}/{r.pages}（{r.blank_ratio:.0%}）")
                tip.append(f"抽取字數：{r.extracted_chars}")
            tip_str = "\n".join(tip)

            with cols[1]:
                name_cols = st.columns([12, 1])
                name_cols[0].markdown(short)
                name_cols[1].badge(" ", icon=":material/info:", color="gray", width="content", help=tip_str)

            # ✅ 頁欄位顯示：總頁（文X）
            if r.pages is None:
                pages_str = "-"
            else:
                if r.ext == ".pdf" and r.text_pages is not None:
                    pages_str = f"{r.pages}（文{r.text_pages}）"
                else:
                    pages_str = str(r.pages)

            cols[2].markdown(pages_str)
            cols[3].markdown(str(r.token_est))

            with cols[4]:
                if r.likely_scanned and r.ext == ".pdf":
                    st.badge("建議 OCR", icon=":material/warning:", color="orange", width="content")
                elif r.ext in (".png", ".jpg", ".jpeg"):
                    st.badge("必 OCR", icon=":material/image:", color="orange", width="content")
                else:
                    st.markdown("")

        st.divider()
        b1, b2 = st.columns([1, 1])
        build_btn = b1.button("🚀 建立索引", type="primary", width="stretch")
        clear_btn = b2.button("🧹 清空全部", width="stretch")

        if clear_btn:
            st.session_state.file_rows = []
            st.session_state.file_bytes = {}
            st.session_state.store = None
            st.session_state.processed_keys = set()
            st.rerun()

        if build_btn:
            need_ocr = any(r.ext == ".pdf" and r.use_ocr for r in st.session_state.file_rows)
            if need_ocr and not HAS_PYMUPDF:
                st.error("你有勾選 PDF OCR，但環境未安裝 pymupdf。請先 pip install pymupdf。")
                st.stop()

            with st.status("建索引中（增量：OCR + embeddings）...", expanded=True) as s:
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


# =========================
# 主畫面（簡化：只顯示索引狀態）
# =========================
if st.session_state.store is None:
    st.info("尚未建立索引。請先在 popover 建索引。")
    st.stop()

st.success(f"已建立索引：檔案數={len(st.session_state.file_rows)} / chunks={len(st.session_state.store.chunks)}")
