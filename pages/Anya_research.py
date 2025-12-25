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
import pandas as pd
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
st.set_page_config(page_title="研究報告助手（Workflow UI + Badges）", layout="wide")
st.title("研究報告助手（Workflow UI + Badges）")


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
CORPUS_PER_REPORT_QUOTA = 6

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

def truncate_filename(name: str, max_len: int = 44) -> str:
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

    # 文字頁小工具
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
# 引用顯示（badge directive 方式）
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

def render_text_with_badges(md_text: str, badge_color: str = "gray"):
    cits = [m.group(0) for m in re.finditer(r"\[[^\]]+?\s+p(\d+|-)\s*\|\s*[A-Za-z0-9_\-]+\]", md_text or "")]
    clean = re.sub(r"\[[^\]]+?\s+p(\d+|-)\s*\|\s*[A-Za-z0-9_\-]+\]", "", md_text or "").strip()
    st.markdown(clean if clean else "（無內容）")
    parsed = _parse_citations(sorted(set(cits)))
    if parsed:
        badges = [_badge_directive(f"{it['title']} p{it['page']} · {it['chunk_id']}", badge_color) for it in parsed]
        st.markdown("來源：" + " ".join(badges))

def bullets_all_have_citations(md: str) -> bool:
    lines = (md or "").splitlines()
    if not any(BULLET_RE.match(l) for l in lines):
        return False
    for line in lines:
        if BULLET_RE.match(line) and not CIT_RE.search(line):
            return False
    return True


# =========================
# Planner（Pydantic v2 修正）
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
- 覆蓋：中國/內地/香港/上海，不動產類型（住宅/商辦/零售/物流倉儲/酒店/數據中心/工業/長租等）、時間（2024-2026 / 報告年份）、指標（租金/空置率/供給/需求/cap rate/利率/政策/信用/REITs）
- 跨類型排序/傳導/彙總/跨文件串鏈 → needs_kg=true
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


# =========================
# Multi-query retrieval + coverage
# =========================
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
                    by_id[ch.chunk_id] = {
                        "chunk": ch,
                        "score": float(score),
                        "via_query": item.query,
                        "via_reason": item.reason,
                    }
        else:
            misses.append({"query": item.query, "reason": item.reason})

    items = sorted(by_id.values(), key=lambda x: x["score"], reverse=True)[:max_total]
    total = max(1, len(plan.queries))
    hit_ratio = len(hit_queries) / total
    return items, {"total_queries": len(plan.queries), "hit_queries": len(hit_queries), "hit_ratio": hit_ratio, "misses": misses}


# =========================
# WebSearch agent（只做背景）
# =========================
WEBSEARCH_AGENT_INSTRUCTIONS = (
    "You are a research assistant. Given a search term, you search the web for that term and "
    "produce a concise background summary. The summary must be 2-3 paragraphs and less than 300 words. "
    "Capture main points. Ignore fluff. Output ONLY the summary."
)

def web_search_agent(client: OpenAI, search_term: str) -> Dict[str, Any]:
    summary, sources = call_gpt52_reasoning(
        client,
        system=WEBSEARCH_AGENT_INSTRUCTIONS,
        user=f"Search term: {search_term}",
        effort="medium",
        enable_web_search=True,
        include_sources=True,
    )
    summary = norm_space(summary)
    cid = f"web_{sha1_text(search_term + summary)}"
    return {"title": f"WebSearch:{truncate_filename(search_term, 26)}", "chunk_id": cid, "text": summary, "sources": sources or [], "search_term": search_term}


# =========================
# Grading / Transform
# =========================
def grade_documents(client: OpenAI, question: str, doc_text: str) -> str:
    system = "你是負責評估所取得文件與使用者問題相關性的評分者。不需嚴格測試。"
    user = f"Retrieved:\n{doc_text[:2200]}\n\nQuestion:\n{question}"
    return call_yesno_grader(client, system=system, user=user)

def grade_hallucinations(client: OpenAI, documents: str, generation: str) -> str:
    system = "你是評估生成內容是否受到 Context 支持的評分者。"
    user = f"Facts:\n{documents[:9000]}\n\nAnswer:\n{generation[:4500]}"
    return call_yesno_grader(client, system=system, user=user)

def grade_answer_adaptive(client: OpenAI, question: str, generation: str) -> str:
    system = "你是評估回答是否回應問題的評分者。若資料不足但有清楚交代缺口＋提供支持部分，也算 yes。"
    user = f"Question:\n{question}\n\nAnswer:\n{generation}"
    return call_yesno_grader(client, system=system, user=user)

def transform_query(client: OpenAI, question: str) -> str:
    system = "把問題改寫成更適合向量檢索的版本（補地區/資產類型/指標/時間）。只輸出一行。"
    return call_gpt52_transform_effort_none(client, system=system, user=question).strip()


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
# 預設輸出（一次三份）
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

def render_chunks_with_ids(chunks: list[Chunk], max_chars_each: int = 900) -> str:
    parts = []
    for c in chunks:
        head = f"[{c.title} p{c.page if c.page else '-'} | {c.chunk_id}]"
        parts.append(head + "\n" + c.text[:max_chars_each])
    return "\n\n".join(parts)

def generate_default_outputs_bundle(client: OpenAI, title: str, ctx: str, max_retries: int = 2) -> Dict[str, str]:
    system = (
        "你是嚴謹的研究助理，只能根據我提供的資料回答，不可腦補。\n"
        "硬性規則：\n"
        "1) 你必須輸出三個區塊，且順序/標題固定：### SUMMARY、### CLAIMS、### CHAIN。\n"
        "2) 每個區塊都必須是純 bullet（每行以 - 開頭），不要段落。\n"
        "3) 每個 bullet 句尾必須附引用，格式固定：[報告名稱 p頁 | chunk_id]\n"
        "4) 引用中的『報告名稱』必須是資料片段方括號內的那個名稱。\n"
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
        out, _ = call_gpt52_reasoning(client, system=system, user=user, effort="medium")
        parts = _split_default_bundle(out)
        ok = bullets_all_have_citations(parts["summary"]) and bullets_all_have_citations(parts["claims"]) and bullets_all_have_citations(parts["chain"])
        if ok:
            return parts
        last = out
        user += "\n\n【強制修正】整份重寫：三區塊皆為純 bullet，且每個 bullet 句尾都有 [報告名稱 p頁 | chunk_id]。"
    return _split_default_bundle(last)


# =========================
# Generate（web 只做背景）
# =========================
def wants_ranking(question: str) -> bool:
    q = norm_space(question)
    return any(k in q for k in ["排序", "排名", "看好", "看壞", "從好到壞", "從壞到好", "優先順序"])

def generate_bullets_guard(client: OpenAI, question: str, context: str, max_retries: int = 2) -> str:
    system = (
        "你是嚴謹的研究助理。\n"
        "硬性規則：\n"
        "1) 只能根據 Context 回答，不可腦補。\n"
        "2) 只能輸出純 bullet（每行以 - 開頭），不要段落。\n"
        "3) 每個 bullet 句尾必須有引用：[報告名稱 p頁 | chunk_id]\n"
        "4) 若資料不足以對某些類型排序，必須在 bullet 中明確說明缺口（仍要引用）。\n"
        "5) 【以指定資料為主】排序/看好看壞/排名結論不得引用 WebSearch:*；Web 只能做背景。\n"
        "6) Context 中標記 WEB_ONLY_BACKGROUND 的段落只能作背景引用。\n"
    )
    user = f"Context:\n{context}\n\nQuestion:\n{question}\n\n請用條列回答。"
    last = ""
    for _ in range(max_retries + 1):
        out, _ = call_gpt52_reasoning(client, system=system, user=user, effort="medium")
        if bullets_all_have_citations(out):
            if wants_ranking(question):
                bad = False
                for line in out.splitlines():
                    if BULLET_RE.match(line) and any(k in line for k in ["排序", "看好", "看壞", "優先", "排名", "由好到壞", "由壞到好", " > "]):
                        if is_web_citation_in_line(line):
                            bad = True
                            break
                if bad:
                    last = out
                    user += "\n\n【強制修正】重寫：排序/看好看壞/排名 bullet 不得引用 WebSearch:*；只能引用上傳報告來源。"
                    continue
            return out
        last = out
        user += "\n\n【強制修正】重寫：每個 bullet 句尾都要有 [報告名稱 p頁 | chunk_id]。"
    return last

def build_context_from_chunks(items: list[Dict[str, Any]], top_k: int = 10) -> str:
    items = sorted(items, key=lambda x: x["score"], reverse=True)[:top_k]
    parts = []
    for it in items:
        ch: Chunk = it["chunk"]
        parts.append(f"[{ch.title} p{ch.page if ch.page is not None else '-'} | {ch.chunk_id}]\n{ch.text}")
    return "\n\n".join(parts) if parts else "（找不到任何相關內容）"

def build_context_from_web_items(web_items: list[Dict[str, Any]]) -> str:
    parts = []
    for w in web_items:
        parts.append("WEB_ONLY_BACKGROUND")
        parts.append(f"[{w['title']} p- | {w['chunk_id']}]\n{w['text']}")
        if w.get("sources"):
            src_lines = []
            for s in w["sources"][:6]:
                if isinstance(s, dict):
                    t = s.get("title") or s.get("source") or "source"
                    u = s.get("url") or ""
                    src_lines.append(f"- {t} {u}".strip())
            if src_lines:
                parts.append("Sources:\n" + "\n".join(src_lines))
    return "\n\n".join(parts)


# =========================
# Workflow（簡化但含你要的：PLAN/RETRIEVE/GRADE/WEB/GENERATE/CHECK）
# =========================
def run_workflow(
    client: OpenAI,
    store: FaissStore,
    question: str,
    *,
    web_mode: str,
) -> Dict[str, Any]:
    # PLAN
    plan = plan_retrieval_queries(client, question)
    st.markdown("### PLAN")
    st.dataframe([{"query": it.query, "reason": it.reason} for it in plan.queries], width="stretch", hide_index=True)

    # RETRIEVE
    retrieved, coverage = retrieve_by_plan(client, store, plan, top_k_per_query=4, max_total=18)
    st.markdown("### RETRIEVE")
    st.dataframe(
        [{
            "score": round(float(it["score"]), 4),
            "報告": it["chunk"].title,
            "頁": it["chunk"].page if it["chunk"].page is not None else "-",
            "chunk_id": it["chunk"].chunk_id,
            "matched_query": it["via_query"],
            "preview": (it["chunk"].text[:120] + "…") if len(it["chunk"].text) > 120 else it["chunk"].text,
        } for it in retrieved],
        width="stretch",
        hide_index=True,
    )
    with st.expander("Coverage details"):
        st.write(coverage)

    # GRADE_DOCS
    relevant = []
    graded_rows = []
    for it in retrieved:
        ch: Chunk = it["chunk"]
        verdict = grade_documents(client, question, ch.text)
        graded_rows.append({"grade": verdict, "報告": ch.title, "頁": ch.page if ch.page is not None else "-", "chunk_id": ch.chunk_id})
        if verdict == "yes":
            relevant.append(it)
    st.markdown("### GRADE_DOCS（yes/no）")
    st.dataframe(graded_rows, width="stretch", hide_index=True)

    # WEB_SEARCH（background-only；預設 OFF）
    web_items = []
    if web_mode == "AUTO":
        hit_ratio = coverage.get("hit_ratio", 1.0)
        trigger = None
        if hit_ratio < MIN_COVERAGE_RATIO:
            trigger = f"coverage hit_ratio={hit_ratio:.2f} < {MIN_COVERAGE_RATIO}"
        elif len(relevant) < MIN_RELEVANT_FOR_NO_WEB:
            trigger = f"relevant={len(relevant)} < {MIN_RELEVANT_FOR_NO_WEB}"

        if trigger:
            st.markdown("### WEB_SEARCH（background-only）")
            st.info(trigger)

            miss_terms = [m["query"] for m in coverage.get("misses", [])[:MAX_WEB_SEARCHES]]
            if len(miss_terms) < MAX_WEB_SEARCHES:
                for it in plan.queries:
                    if it.query not in miss_terms:
                        miss_terms.append(it.query)
                    if len(miss_terms) >= MAX_WEB_SEARCHES:
                        break

            web_rows = []
            for term in miss_terms[:MAX_WEB_SEARCHES]:
                w = web_search_agent(client, term)
                web_items.append(w)
                web_rows.append({"search_term": term, "chunk_id": w["chunk_id"], "sources": len(w.get("sources") or [])})
            st.dataframe(web_rows, width="stretch", hide_index=True)
            with st.expander("Web sources"):
                for w in web_items:
                    st.markdown(f"**{w['search_term']}** → `{w['chunk_id']}`")
                    for s in (w.get("sources") or [])[:10]:
                        st.write(s)

    if not relevant and not web_items:
        return {"answer": "資料不足：檢索不到足夠相關內容。建議改問法或上傳更多報告。", "context": ""}

    # GENERATE
    ctx_parts = []
    if relevant:
        ctx_parts.append(build_context_from_chunks(relevant, top_k=10))
    if web_items:
        ctx_parts.append(build_context_from_web_items(web_items))
    context = "\n\n".join([p for p in ctx_parts if p.strip()])

    st.markdown("### GENERATE")
    ans = generate_bullets_guard(client, question, context, max_retries=2)
    render_bullets_inline_badges(ans, badge_color="green")

    # CHECK
    st.markdown("### CHECK")
    hall = grade_hallucinations(client, context, ans)
    ok = grade_answer_adaptive(client, question, ans)
    st.write({"hallucination": hall, "answer_ok": ok})

    return {"answer": ans, "context": context}


# =========================
# Session init
# =========================
OPENAI_API_KEY = get_openai_api_key()
client = get_client(OPENAI_API_KEY)

if "file_rows" not in st.session_state:
    st.session_state.file_rows = []
if "file_bytes" not in st.session_state:
    st.session_state.file_bytes = {}
if "store" not in st.session_state:
    st.session_state.store = None
if "processed_keys" not in st.session_state:
    st.session_state.processed_keys = set()
if "default_outputs" not in st.session_state:
    st.session_state.default_outputs = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# =========================
# File table helpers（pandas / data_editor）
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
            "_file_id": r.file_id,               # 保留給同步用（不顯示）
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
    # 以 file_id 對齊回寫 OCR 欄位
    id_to_row_idx = {r.file_id: i for i, r in enumerate(rows)}
    for _, rec in df.iterrows():
        fid = rec.get("_file_id")
        if fid not in id_to_row_idx:
            continue
        i = id_to_row_idx[fid]

        ext = rows[i].ext
        # 強制規則：圖檔一定 OCR；txt 一定不 OCR
        if ext in (".png", ".jpg", ".jpeg"):
            rows[i].use_ocr = True
        elif ext == ".txt":
            rows[i].use_ocr = False
        else:
            rows[i].use_ocr = bool(rec.get("使用OCR", rows[i].use_ocr))


# =========================
# Popover：文件管理（pandas + data_editor）
# =========================
with st.popover("📦 文件管理（上傳 / OCR / 建索引）", width="content"):
    st.caption("支援 PDF/TXT/PNG/JPG。PDF 若文字抽取偏少會建議 OCR（逐檔可勾選）。")

    web_mode = st.radio(
        "Web search",
        options=["OFF", "AUTO"],
        index=0,
        horizontal=True,
        help="OFF：完全不使用網路；AUTO：檢索 coverage 不足或 relevant 太少才補背景",
        key="web_mode",
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

            # default OCR decision
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
            df.drop(columns=["_file_id"]),  # 不顯示 file_id
            key="file_table_editor",
            width="stretch",
            hide_index=True,
            disabled=["檔名", "格式", "頁數", "文字頁", "文字%", "token估算", "建議"],  # 只允許改「使用OCR」
            column_config={
                "使用OCR": st.column_config.CheckboxColumn("使用OCR", help="逐檔選擇是否啟用 OCR（PDF 可選；圖檔固定OCR；TXT固定不OCR）"),
                "token估算": st.column_config.NumberColumn("token估算", help="粗估 token，用於快速判斷抽取量是否偏少"),
                "文字頁": st.column_config.TextColumn("文字頁", help="PDF 中抽到足夠文字的頁數（<=40字視為空白頁）"),
                "文字%": st.column_config.TextColumn("文字%", help="文字頁/總頁 的比例（越低越可能是掃描圖）"),
                "建議": st.column_config.TextColumn("建議", help="依抽取量推測是否建議 OCR"),
            },
        )

        # 把 editor 的 OCR 選擇回寫到 session（用 df + file_id 對齊）
        # 注意：data_editor 回傳 df 沒有 _file_id，所以我們用原 df 的順序回寫
        df_for_sync = df.copy()
        df_for_sync["使用OCR"] = edited["使用OCR"].values
        sync_df_to_file_rows(df_for_sync, st.session_state.file_rows)

        st.divider()
        col1, col2 = st.columns([1, 1])
        build_btn = col1.button("🚀 建立索引 + 預設輸出", type="primary", width="stretch")
        clear_btn = col2.button("🧹 清空全部", width="stretch")

        if clear_btn:
            st.session_state.file_rows = []
            st.session_state.file_bytes = {}
            st.session_state.store = None
            st.session_state.processed_keys = set()
            st.session_state.default_outputs = None
            st.session_state.chat_history = []
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

            # 預設輸出（一次三份）→ push 到 chat
            with st.status("產生預設輸出（摘要/主張/推論鏈）...", expanded=True) as s2:
                chosen = pick_corpus_chunks_for_default(st.session_state.store.chunks)
                ctx = render_chunks_with_ids(chosen)
                bundle = generate_default_outputs_bundle(client, "整體融合（全部上傳報告）", ctx, max_retries=2)
                st.session_state.default_outputs = bundle
                s2.update(state="complete")

            st.session_state.chat_history = []
            st.session_state.chat_history.append({
                "role": "assistant",
                "kind": "default",
                "title": "整體融合（全部上傳報告）",
                **st.session_state.default_outputs,
            })
            st.rerun()


# =========================
# 主畫面：狀態 + Chat
# =========================
if st.session_state.store is None:
    st.info("尚未建立索引。請先在 popover 上傳並建立索引。")
    st.stop()

st.success(f"已建立索引：檔案數={len(st.session_state.file_rows)} / chunks={len(st.session_state.store.chunks)}")

st.divider()
st.subheader("Chat（Workflow UI + Badges）")

# 顯示 chat history
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
            st.markdown(msg.get("content", ""))

# 使用者提問
prompt = st.chat_input("請輸入問題（例如：中國/香港不動產概況、各類資產看好/看壞排序與原因…）")
if prompt:
    st.session_state.chat_history.append({"role": "user", "kind": "text", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Workflow：PLAN → RETRIEVE → GRADE_DOCS → (AUTO) WEB_SEARCH(background-only) → GENERATE → CHECK", expanded=True) as status:
            result = run_workflow(
                client=client,
                store=st.session_state.store,
                question=prompt,
                web_mode=st.session_state.get("web_mode", "OFF"),
            )
            status.update(state="complete", expanded=False)

        st.markdown("## 最終回答")
        render_bullets_inline_badges(result.get("answer", ""), badge_color="green")

        with st.expander("Debug（context 節錄）"):
            st.text((result.get("context") or "")[:12000])

    st.session_state.chat_history.append({"role": "assistant", "kind": "text", "content": result.get("answer", "")})
