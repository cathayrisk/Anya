# app.py
# -*- coding: utf-8 -*-
"""
研究報告助手（Adaptive RAG + Planner multi-query + Lazy LangExtract KG + optional web_search agent）

模型路由（依你指定）：
- Planner / Generate / 融合預設輸出：gpt-5.2 + reasoning.effort=medium
- TRANSFORM：gpt-5.2 + reasoning.effort=none
- Grading / OCR / LangExtract：gpt-4.1-mini

web_search：
- popover 可選：關閉 / 自動（僅當檢索不足才啟用）
- ✅ 預設為 OFF（依你要求）
- ✅ 觸發改為 coverage-based（不只看 relevant 數量）
- ✅ 若啟用 web_search，顯示 sources（並注入 debug / context）

依賴：
streamlit, openai, langextract[openai], numpy, faiss-cpu, networkx, pypdf, pydantic
（建議）pymupdf：PDF 快速抽字與 OCR rasterize
"""

from __future__ import annotations

import os
import re
import io
import uuid
import math
import time
import json
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
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
CORPUS_PER_REPORT_QUOTA = 4

# ===== web_search 觸發參數（你要的兩個小調整）=====
# relevant chunks 太少也會觸發（第二順位）
MIN_RELEVANT_FOR_NO_WEB = 3
# coverage 缺口達到這個比例才觸發（第一順位）
MIN_COVERAGE_RATIO = 0.45  # 例如 planner 8條 query，至少命中 4條才算及格
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

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
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


# =========================
# OpenAI client + LLM wrappers（依你要求：gpt-5.2 要帶 reasoning）
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

def embed_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        encoding_format="float",
    )
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vecs / norms

def _to_messages(system: str, user: Any) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

def call_gpt52_reasoning(
    client: OpenAI,
    *,
    system: str,
    user: Any,
    effort: str = "medium",
    enable_web_search: bool = False,
    include_sources: bool = False,
) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
    """
    回傳 (output_text, sources)
    - sources 只有在 enable_web_search + include_sources 才嘗試抓
    """
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

    # ✅ 小調整(2)：把 sources 解析出來（盡量容錯）
    if enable_web_search and include_sources:
        try:
            # 有些 SDK 會把 include 的資料塞在 resp.output / resp 內部結構中
            # 我們用最保守的方式：遍歷 output items 找 web_search_call
            sources_list = []
            if hasattr(resp, "output") and resp.output:
                for item in resp.output:
                    # item 可能是 dict 或物件
                    d = item if isinstance(item, dict) else getattr(item, "__dict__", {})
                    # 常見形式：{"type":"web_search_call","action":{"sources":[...]}}
                    if isinstance(d, dict) and d.get("type") == "web_search_call":
                        action = d.get("action", {}) or {}
                        if isinstance(action, dict) and action.get("sources"):
                            sources_list.extend(action["sources"])
            sources = sources_list if sources_list else None
        except Exception:
            sources = None

    return out_text, sources

def call_gpt52_transform_effort_none(
    client: OpenAI,
    *,
    system: str,
    user: Any,
) -> str:
    messages = _to_messages(system, user)
    resp = client.responses.create(
        model="gpt-5.2",
        input=messages,
        reasoning={"effort": "none"},
        text={"verbosity": "medium"},
        truncation="auto",
    )
    return resp.output_text

def call_gpt41mini(
    client: OpenAI,
    *,
    system: str,
    user: Any,
) -> str:
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
    blank_pages: Optional[int]
    blank_ratio: Optional[float]
    likely_scanned: bool
    use_ocr: bool

def extract_pdf_text_pages_pypdf(pdf_bytes: bytes) -> List[Tuple[int, str]]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    out = []
    for i, p in enumerate(reader.pages):
        try:
            t = p.extract_text() or ""
        except Exception:
            t = ""
        out.append((i + 1, norm_space(t)))
    return out

def extract_pdf_text_pages_pymupdf(pdf_bytes: bytes) -> List[Tuple[int, str]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    out = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        t = page.get_text("text") or ""
        out.append((i + 1, norm_space(t)))
    return out

def extract_pdf_text_pages(pdf_bytes: bytes) -> List[Tuple[int, str]]:
    if HAS_PYMUPDF:
        try:
            return extract_pdf_text_pages_pymupdf(pdf_bytes)
        except Exception:
            return extract_pdf_text_pages_pypdf(pdf_bytes)
    return extract_pdf_text_pages_pypdf(pdf_bytes)

def analyze_pdf_text_quality(pdf_pages: List[Tuple[int, str]], min_chars_per_page: int = 40) -> Tuple[int, int, float]:
    if not pdf_pages:
        return 0, 0, 1.0
    lens = [len(t) for _, t in pdf_pages]
    blank = sum(1 for L in lens if L <= min_chars_per_page)
    ratio = blank / max(1, len(lens))
    return sum(lens), blank, ratio

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

def ocr_pdf_pages_parallel(client: OpenAI, pdf_bytes: bytes, dpi: int = 180) -> List[Tuple[int, str]]:
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
        self.chunks: List[Chunk] = []

    def add(self, vecs: np.ndarray, chunks: List[Chunk]) -> None:
        self.index.add(vecs)
        self.chunks.extend(chunks)

    def search(self, qvec: np.ndarray, k: int = 8) -> List[Tuple[float, Chunk]]:
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
# 引用檢查 + inline badge 呈現
# =========================
CIT_RE = re.compile(r"\[[^\]]+?\s+p(\d+|-)\s*\|\s*[A-Za-z0-9_\-]+\]")
BULLET_RE = re.compile(r"^\s*(?:[-•*]|\d+\.)\s+")

def bullets_all_have_citations(md: str) -> bool:
    lines = (md or "").splitlines()
    if not any(BULLET_RE.match(l) for l in lines):
        return False
    for line in lines:
        if BULLET_RE.match(line) and not CIT_RE.search(line):
            return False
    return True

def paragraphs_all_have_citations(md: str) -> bool:
    paras = [p.strip() for p in re.split(r"\n\s*\n", md or "") if p.strip()]
    if not paras:
        return False
    for p in paras:
        if not CIT_RE.search(p):
            return False
    return True

CIT_PARSE_RE = re.compile(r"\[([^\]]+?)\s+p(\d+|-)\s*\|\s*([A-Za-z0-9_\-]+)\]")

def _parse_citations(cits: List[str]) -> List[Dict[str, str]]:
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


# =========================
# Planner（gpt-5.2 reasoning=medium）
# =========================
class RetrievalQueryItem(BaseModel):
    reason: str = Field(..., description="為什麼這個 query 對回答問題很重要")
    query: str = Field(..., description="要拿去向量檢索的查詢字串")

class RetrievalPlan(BaseModel):
    needs_kg: bool = Field(..., description="是否需要 KG/關係推理來回答")
    queries: List[RetrievalQueryItem] = Field(..., description="5~12 條查詢")

def plan_retrieval_queries(client: OpenAI, question: str) -> RetrievalPlan:
    system = """
你是 Planner。目標：把使用者問題拆成一組「向量檢索 queries」，以最大化找回證據片段的覆蓋率。
請遵守：
- 輸出 5~12 條 queries
- 每條 query 必須附 reason（說明為何重要）
- queries 需涵蓋：地區（中國/內地/香港）、資產類型（住宅/商辦/零售/物流倉儲/酒店/數據中心/工業/長租等）、時間（例如 2024-2026 或報告年份）、關鍵指標（租金/空置率/供給/需求/cap rate/利率/政策/信用/REITs）
- 若問題涉及「跨類型排序、因果/傳導、交集/彙總、規則/限制、跨多篇串鏈」needs_kg=true
輸出純 JSON，符合 RetrievalPlan schema。
"""
    user = f"使用者問題：{question}\n\n請輸出 RetrievalPlan JSON。"
    raw = call_json_planner(client, system=system, user=user)

    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    raw = m.group(0) if m else raw

    try:
        plan = RetrievalPlan.model_validate_json(raw)
    except ValidationError:
        raw2 = call_json_planner(client, system=system + "\n⚠️ 你剛剛輸出不合法，請只輸出可解析的純 JSON。", user=user)
        m2 = re.search(r"\{.*\}", raw2, flags=re.DOTALL)
        raw2 = m2.group(0) if m2 else raw2
        plan = RetrievalPlan.model_validate_json(raw2)

    plan.queries = [q for q in plan.queries if q.query.strip()]
    if not plan.queries:
        plan.queries = [RetrievalQueryItem(reason="fallback", query=question)]
    return plan


# =========================
# Multi-query retrieval + coverage 計算（小調整 1）
# =========================
def retrieve_by_plan(
    client: OpenAI,
    store: FaissStore,
    plan: RetrievalPlan,
    *,
    top_k_per_query: int = 4,
    max_total: int = 18,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    回傳 (items, coverage_report)
    items: [{chunk, score, via_query, via_reason}]
    coverage_report:
      - total_queries
      - hit_queries（至少命中一個 chunk 的 query）
      - hit_ratio
      - misses: [{query, reason}]
    """
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

    coverage = {
        "total_queries": len(plan.queries),
        "hit_queries": len(hit_queries),
        "hit_ratio": hit_ratio,
        "misses": misses,
    }
    return items, coverage


# =========================
# WebSearch agent（gpt-5.2 + web_search tool）+ sources（小調整 2）
# =========================
WEBSEARCH_AGENT_INSTRUCTIONS = (
    "You are a research assistant. Given a search term, you search the web for that term and "
    "produce a concise summary of the results. The summary must be 2-3 paragraphs and less than 300 "
    "words. Capture the main points. Ignore fluff. Output ONLY the summary."
)

def web_search_agent(client: OpenAI, search_term: str) -> Dict[str, Any]:
    """
    回傳 dict：
    {
      "title": "WebSearch:<term>",
      "chunk_id": "web_xxx",
      "text": "...summary...",
      "sources": [...]
    }
    """
    system = WEBSEARCH_AGENT_INSTRUCTIONS
    user = f"Search term: {search_term}"
    summary, sources = call_gpt52_reasoning(
        client,
        system=system,
        user=user,
        effort="medium",
        enable_web_search=True,
        include_sources=True,
    )
    summary = norm_space(summary)
    cid = f"web_{sha1_text(search_term + summary)}"
    return {
        "title": f"WebSearch:{truncate_filename(search_term, 26)}",
        "chunk_id": cid,
        "text": summary,
        "sources": sources or [],
        "search_term": search_term,
    }


# =========================
# LangExtract（Lazy KG；用 gpt-4.1-mini）
# =========================
class KnowledgeGraph:
    def __init__(self):
        self.g = nx.MultiDiGraph()

    def add_edge(self, s: str, r: str, o: str, prov: Dict[str, Any], attrs: Optional[Dict[str, Any]] = None):
        s = norm_space(s)
        o = norm_space(o)
        r = norm_space(r).upper().replace(" ", "_")
        if not s or not o or not r:
            return
        if s not in self.g:
            self.g.add_node(s, label=s)
        if o not in self.g:
            self.g.add_node(o, label=o)
        self.g.add_edge(s, o, key=str(uuid.uuid4()), relation=r, prov=prov, attrs=attrs or {})

    def find_nodes_in_query(self, query: str, max_n: int = 2) -> List[str]:
        q = norm_space(query)
        hits = []
        for n in self.g.nodes():
            if len(n) >= 4 and n in q:
                hits.append(n)
        return hits[:max_n]

def lx_prompt() -> str:
    return (
        "Extract structured information from macro/finance/climate-risk/sustainable-finance reports.\n"
        "Rules:\n"
        "1) Use exact text spans for extraction_text. Do NOT paraphrase.\n"
        "2) Extract only relation and claim.\n"
        "3) relation.attributes must include: {subject, relation, object}. Optional: {time, qualifier}.\n"
        "4) Only extract relations explicitly supported by text; if unsure, skip.\n"
    )

def lx_examples() -> List[lx.data.ExampleData]:
    t1 = "Mainland China retail rental growth expected at 3-5% YoY in 2025-26E, while HK retail rental to be largely flat."
    ex1 = lx.data.ExampleData(
        text=t1,
        extractions=[
            lx.data.Extraction(
                extraction_class="relation",
                extraction_text="retail rental growth expected at 3-5% YoY in 2025-26E",
                attributes={"subject": "Mainland China retail rental", "relation": "INCREASES", "object": "3-5% YoY", "time": "2025-26E"},
            ),
        ],
    )
    return [ex1]

def run_langextract(text: str, api_key: str) -> lx.data.AnnotatedDocument:
    return lx.extract(
        text_or_documents=text,
        prompt_description=lx_prompt(),
        examples=lx_examples(),
        model_id=MODEL_LANGEXTRACT,
        api_key=api_key,
        extraction_passes=1,
        max_char_buffer=1200,
        max_workers=8,
        fence_output=True,
        use_schema_constraints=False,
    )

def lazy_langextract_kg_for_chunks(
    api_key: str,
    chunks: List[Chunk],
    lx_cache: Dict[str, Any],
) -> Tuple[KnowledgeGraph, int]:
    kg = KnowledgeGraph()
    new_edges = 0

    def extract_one(ch: Chunk) -> Tuple[str, List[Dict[str, Any]]]:
        if ch.chunk_id in lx_cache:
            return ch.chunk_id, lx_cache[ch.chunk_id]

        ann = run_langextract(ch.text, api_key=api_key)
        extracted = []
        for e in ann.extractions:
            cls = getattr(e, "extraction_class", "")
            attrs = getattr(e, "attributes", {}) or {}
            etext = getattr(e, "extraction_text", "") or ""

            prov = {"title": ch.title, "page": ch.page if ch.page is not None else "-", "chunk_id": ch.chunk_id}

            if cls == "relation":
                extracted.append({"s": attrs.get("subject", ""), "r": attrs.get("relation", ""), "o": attrs.get("object", ""), "prov": prov, "attrs": attrs})
            elif cls == "claim":
                extracted.append({"s": ch.title, "r": "MENTIONS", "o": f"CLAIM: {norm_space(etext)}", "prov": prov, "attrs": attrs})

        lx_cache[ch.chunk_id] = extracted
        return ch.chunk_id, extracted

    with ThreadPoolExecutor(max_workers=LX_MAX_WORKERS_QUERY) as ex:
        futs = {ex.submit(extract_one, ch): ch.chunk_id for ch in chunks}
        for fut in as_completed(futs):
            try:
                _cid, extracted = fut.result()
            except Exception:
                extracted = []
            for item in extracted:
                before = kg.g.number_of_edges()
                kg.add_edge(item.get("s", ""), item.get("r", ""), item.get("o", ""), item.get("prov", {}), attrs=item.get("attrs"))
                after = kg.g.number_of_edges()
                new_edges += max(0, after - before)

    return kg, new_edges

def build_kg_hint_lines(question: str, kg: KnowledgeGraph, max_lines: int = 16) -> str:
    starts = kg.find_nodes_in_query(question, max_n=2)
    if not starts:
        return ""
    lines = []
    for start in starts:
        for u, v, k, data in nx.edge_bfs(kg.g, start):
            rel = data.get("relation")
            prov = data.get("prov") or {}
            title = prov.get("title", "Unknown")
            page = prov.get("page", "-")
            cid = prov.get("chunk_id", "unknown")
            lines.append(f"- {u} --[{rel}]--> {v} [{title} p{page} | {cid}]")
            if len(lines) >= max_lines:
                break
        if len(lines) >= max_lines:
            break
    return "【KG 線索】\n" + "\n".join(lines) if lines else ""


# =========================
# 建索引（不含 LangExtract）
# =========================
def build_indices_incremental_no_kg(
    client: OpenAI,
    file_rows: List[FileRow],
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
    new_chunks: List[Chunk] = []
    new_texts: List[str] = []

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
# 融合預設輸出（gpt-5.2 reasoning=medium）
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

def pick_corpus_chunks_for_default(all_chunks: List[Chunk]) -> List[Chunk]:
    by_title: Dict[str, List[Chunk]] = {}
    for c in all_chunks:
        by_title.setdefault(c.title, []).append(c)

    kw = re.compile(r"(outlook|risk|implication|forecast|scenario|inflation|rate|credit|spread|cap rate|vacancy|supply|demand|rental|office|retail|residential|logistics|hotel)", re.I)

    def score(c: Chunk) -> float:
        s = 0.0
        if c.page is not None:
            s += max(0.0, 8.0 - min(8.0, float(c.page)))
        if kw.search(c.text or ""):
            s += 6.0
        s += min(2.0, len(c.text) / 1400.0)
        return s

    chosen: List[Chunk] = []
    for title, chunks in sorted(by_title.items(), key=lambda x: x[0]):
        chunks = sorted(chunks, key=score, reverse=True)
        chosen.extend(chunks[:CORPUS_PER_REPORT_QUOTA])

    chosen = sorted(chosen, key=score, reverse=True)[:CORPUS_DEFAULT_MAX_CHUNKS]
    return chosen

def render_chunks_with_ids(chunks: List[Chunk], max_chars_each: int = 900) -> str:
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
        f"- SUMMARY：8~14 bullets（結論/預測/假設/風險/限制/市場含意）\n"
        f"- CLAIMS：8~14 bullets（可驗證主張）\n"
        f"- CHAIN：6~12 bullets（傳導：驅動→中介→結論→風險）\n\n"
        f"資料（多份報告混合）：\n{ctx}\n\n"
        f"請輸出：\n### SUMMARY\n...\n### CLAIMS\n...\n### CHAIN\n...\n"
    )

    last = ""
    for _ in range(max_retries + 1):
        out, _ = call_gpt52_reasoning(client, system=system, user=user, effort="medium", enable_web_search=False)
        parts = _split_default_bundle(out)
        ok = bullets_all_have_citations(parts["summary"]) and bullets_all_have_citations(parts["claims"]) and bullets_all_have_citations(parts["chain"])
        if ok:
            return parts
        last = out
        user += "\n\n【強制修正】整份重寫：三區塊皆為純 bullet，且每個 bullet 句尾都有 [報告名稱 p頁 | chunk_id]。"

    return _split_default_bundle(last)


# =========================
# grading（gpt-4.1-mini）
# =========================
def grade_documents(client: OpenAI, question: str, doc_text: str) -> str:
    system = """你是負責評估所取得文件與使用者問題相關性的評分者。
如果文件中包含與使用者問題相關的關鍵字或意義，請評為有相關性。
目的是排除明顯錯誤的取得結果，不需要進行嚴格的測試。"""
    user = f"Retrieved document:\n{doc_text[:2200]}\n\nUser question:\n{question}"
    return call_yesno_grader(client, system=system, user=user)

def grade_hallucinations(client: OpenAI, documents: str, generation: str) -> str:
    system = """你是負責評估生成內容是否根據／受到所取得事實集合支持的評分者。
「yes」表示回答是根據事實集合產生或受到其支持。"""
    user = f"Set of facts:\n{documents[:9000]}\n\nLLM generation:\n{generation[:4500]}"
    return call_yesno_grader(client, system=system, user=user)

def grade_answer_adaptive(client: OpenAI, question: str, generation: str) -> str:
    system = """你是負責評估回答是否有回應／解決問題的評分者。
「yes」表示回答有解決問題。
注意：如果資料不足，回答只要能清楚交代缺口、並提供可被文件支持的部分答案，也應判 yes。"""
    user = f"User question:\n{question}\n\nLLM generation:\n{generation}"
    return call_yesno_grader(client, system=system, user=user)


# =========================
# TRANSFORM（gpt-5.2 effort=none）
# =========================
def transform_query(client: OpenAI, question: str) -> str:
    system = """你是一位將輸入問題轉換成更適合向量檢索的優化版本的問題重寫者。
請閱讀問題，推論提問者意圖，產生更適合檢索的問題（補地區/資產類型/指標/時間）。只輸出一行。"""
    return call_gpt52_transform_effort_none(client, system=system, user=question).strip()


# =========================
# GENERATE（gpt-5.2 reasoning=medium；嚴格引用）
# =========================
def wants_ranking(question: str) -> bool:
    q = norm_space(question)
    return any(k in q for k in ["排序", "排名", "看好", "看壞", "從好到壞", "從壞到好", "優先順序"])

def want_bullets(question: str) -> bool:
    q = norm_space(question)
    return bool(re.search(r"(列出|有哪些|所有|清單|彙總|摘要|總結)", q)) or wants_ranking(q)

def generate_bullets_guard(client: OpenAI, question: str, context: str, max_retries: int = 2) -> str:
    system = (
        "你是嚴謹的研究助理。\n"
        "硬性規則：\n"
        "1) 只能根據 Context 回答，不可腦補。\n"
        "2) 只能輸出純 bullet（每行以 - 開頭），不要段落。\n"
        "3) 每個 bullet 句尾必須有引用，格式固定：[報告名稱 p頁 | chunk_id]\n"
        "4) 若資料不足以對某些類型排序，必須在 bullet 中明確說明缺口（仍要引用）。\n"
    )
    user = f"Context:\n{context}\n\nQuestion:\n{question}\n\n請用條列回答（含排序時請列出看好→看壞並說明理由）。"
    last = ""
    for _ in range(max_retries + 1):
        out, _ = call_gpt52_reasoning(client, system=system, user=user, effort="medium", enable_web_search=False)
        if bullets_all_have_citations(out):
            return out
        last = out
        user += "\n\n【強制修正】重寫：每個 bullet 句尾都要有 [報告名稱 p頁 | chunk_id]。"
    return last

def generate_paragraph_guard(client: OpenAI, question: str, context: str, max_retries: int = 2) -> str:
    system = (
        "你是嚴謹的研究助理。\n"
        "硬性規則：\n"
        "1) 只能根據 Context 回答，不可腦補。\n"
        "2) 請用 2~4 段回答。\n"
        "3) 每段至少 1 個引用，格式固定：[報告名稱 p頁 | chunk_id]\n"
    )
    user = (
        "請回答問題，依序：結論→依據（引用）→推論/解釋。\n"
        "每段至少1個引用。\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}"
    )
    last = ""
    for _ in range(max_retries + 1):
        out, _ = call_gpt52_reasoning(client, system=system, user=user, effort="medium", enable_web_search=False)
        if paragraphs_all_have_citations(out):
            return out
        last = out
        user += "\n\n【強制修正】重寫：每段至少 1 個 [報告名稱 p頁 | chunk_id]。"
    return last

def build_context_from_chunks(items: List[Dict[str, Any]], top_k: int = 10) -> str:
    items = sorted(items, key=lambda x: x["score"], reverse=True)[:top_k]
    parts = []
    for it in items:
        ch: Chunk = it["chunk"]
        parts.append(f"[{ch.title} p{ch.page if ch.page is not None else '-'} | {ch.chunk_id}]\n{ch.text}")
    return "\n\n".join(parts) if parts else "（找不到任何相關內容）"

def build_context_from_web_items(web_items: List[Dict[str, Any]]) -> str:
    parts = []
    for w in web_items:
        parts.append(f"[{w['title']} p- | {w['chunk_id']}]\n{w['text']}")
        # 把 sources 也塞進 context（但不強迫模型引用 sources，仍以 chunk_id 為引用單位）
        if w.get("sources"):
            src_lines = []
            for s in w["sources"][:6]:
                # s 可能是 dict: {title,url,source...}
                if isinstance(s, dict):
                    t = s.get("title") or s.get("source") or "source"
                    u = s.get("url") or ""
                    src_lines.append(f"- {t} {u}".strip())
            if src_lines:
                parts.append("Sources:\n" + "\n".join(src_lines))
    return "\n\n".join(parts) if parts else ""


# =========================
# Adaptive RAG Workflow（含 UI，web_search 預設 OFF）
# =========================
def _step_table(step_state: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"Step": k, "Status": v.get("status"), "Seconds": v.get("seconds"), "Note": v.get("note")} for k, v in step_state.items()]

def run_adaptive_rag(
    client: OpenAI,
    api_key: str,
    store: FaissStore,
    question: str,
    *,
    kg_mode: str,          # OFF / AUTO / FORCE
    web_mode: str,         # OFF / AUTO
    top_k_per_query: int = 4,
    max_total_chunks: int = 18,
    max_query_rewrites: int = 2,
    max_generate_retries: int = 2,
) -> Dict[str, Any]:
    query_history = [question]
    logs: List[str] = []

    step_state = {
        "PLAN": {"status": "PENDING", "seconds": None, "note": ""},
        "RETRIEVE": {"status": "PENDING", "seconds": None, "note": ""},
        "COVERAGE": {"status": "PENDING", "seconds": None, "note": ""},
        "GRADE_DOCS": {"status": "PENDING", "seconds": None, "note": ""},
        "WEB_SEARCH": {"status": "SKIP", "seconds": None, "note": "disabled/not needed"},
        "KG_EXTRACT": {"status": "SKIP", "seconds": None, "note": "not triggered"},
        "GENERATE": {"status": "PENDING", "seconds": None, "note": ""},
        "GRADE_HALLU": {"status": "PENDING", "seconds": None, "note": ""},
        "GRADE_ANSWER": {"status": "PENDING", "seconds": None, "note": ""},
        "TRANSFORM": {"status": "PENDING", "seconds": None, "note": ""},
    }

    step_summary_ph = st.empty()
    query_hist_ph = st.empty()

    def update_step_summary():
        step_summary_ph.markdown("#### Step Summary（✅/❌ + 耗時）")
        step_summary_ph.dataframe(_step_table(step_state), width="stretch", hide_index=True)

    def set_step(name: str, status: str, seconds: Optional[float] = None, note: str = ""):
        step_state[name]["status"] = status
        step_state[name]["seconds"] = (round(seconds, 3) if seconds is not None else None)
        step_state[name]["note"] = note
        update_step_summary()

    update_step_summary()
    query_hist_ph.markdown("#### Query history")
    query_hist_ph.code("\n".join([f"{i}. {q}" for i, q in enumerate(query_history)]))

    q = question

    for rewrite_round in range(max_query_rewrites + 1):
        # PLAN
        t0 = time.perf_counter()
        plan = plan_retrieval_queries(client, q)
        set_step("PLAN", "✅ OK", time.perf_counter() - t0, note=f"queries={len(plan.queries)}, needs_kg={plan.needs_kg}")

        st.markdown(f"### PLAN（round {rewrite_round}）")
        st.dataframe(
            [{"query": it.query, "reason": it.reason} for it in plan.queries],
            width="stretch",
            hide_index=True,
        )

        # RETRIEVE + coverage report
        t1 = time.perf_counter()
        retrieved, coverage = retrieve_by_plan(
            client=client,
            store=store,
            plan=plan,
            top_k_per_query=top_k_per_query,
            max_total=max_total_chunks,
        )
        set_step("RETRIEVE", "✅ OK", time.perf_counter() - t1, note=f"total={len(retrieved)}")

        hit_ratio = coverage["hit_ratio"]
        miss_count = len(coverage["misses"])
        set_step("COVERAGE", "✅ OK", None, note=f"hit_ratio={hit_ratio:.2f}, misses={miss_count}")

        st.markdown("### RETRIEVE（multi-query merged）")
        st.dataframe(
            [{
                "score": round(float(it["score"]), 4),
                "報告": it["chunk"].title,
                "頁": it["chunk"].page if it["chunk"].page is not None else "-",
                "chunk_id": it["chunk"].chunk_id,
                "matched_query": it["via_query"],
                "why": it["via_reason"],
                "preview": (it["chunk"].text[:140] + "…") if len(it["chunk"].text) > 140 else it["chunk"].text,
            } for it in retrieved],
            width="stretch",
            hide_index=True,
        )

        with st.expander("Coverage details（哪些 planner query 沒命中）"):
            st.write({
                "total_queries": coverage["total_queries"],
                "hit_queries": coverage["hit_queries"],
                "hit_ratio": round(hit_ratio, 3),
                "misses": coverage["misses"][:20],
            })

        # GRADE_DOCS
        t2 = time.perf_counter()
        relevant: List[Dict[str, Any]] = []
        graded_rows = []
        prog = st.progress(0, text="grading docs…")
        for i, it in enumerate(retrieved):
            ch: Chunk = it["chunk"]
            verdict = grade_documents(client, q, ch.text)
            graded_rows.append({
                "grade": verdict,
                "score": round(float(it["score"]), 4),
                "報告": ch.title,
                "頁": ch.page if ch.page is not None else "-",
                "chunk_id": ch.chunk_id,
                "matched_query": it["via_query"],
                "preview": (ch.text[:140] + "…") if len(ch.text) > 140 else ch.text,
            })
            if verdict == "yes":
                relevant.append(it)
            prog.progress((i + 1) / max(1, len(retrieved)), text=f"grading docs… {i+1}/{len(retrieved)}")

        set_step("GRADE_DOCS", "✅ OK", time.perf_counter() - t2, note=f"relevant={len(relevant)}/{len(retrieved)}")
        st.markdown("### GRADE_DOCS（yes/no）")
        st.dataframe(graded_rows, width="stretch", hide_index=True)

        # ✅ 小調整(1)：coverage-based 觸發 web_search（但 web_mode 預設 OFF）
        web_items: List[Dict[str, Any]] = []
        web_trigger_reason = None
        if web_mode == "AUTO":
            # 先看 coverage
            if hit_ratio < MIN_COVERAGE_RATIO:
                web_trigger_reason = f"coverage hit_ratio={hit_ratio:.2f} < {MIN_COVERAGE_RATIO}"
            # 再看 relevant 數量（第二順位）
            elif len(relevant) < MIN_RELEVANT_FOR_NO_WEB:
                web_trigger_reason = f"relevant={len(relevant)} < {MIN_RELEVANT_FOR_NO_WEB}"

        if web_trigger_reason:
            tw = time.perf_counter()
            set_step("WEB_SEARCH", "▶️ RUN", None, note=web_trigger_reason)
            st.markdown("### WEB_SEARCH（triggered: retrieval insufficient）")
            st.info(web_trigger_reason)

            # 用 misses 優先當 search term（因為這些是資料庫「沒命中」的面向）
            miss_terms = [m["query"] for m in coverage["misses"][:MAX_WEB_SEARCHES]]
            # 若 misses 不夠，再補 planner query 前幾條
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
                web_rows.append({
                    "search_term": term,
                    "chunk_id": w["chunk_id"],
                    "summary_preview": (w["text"][:160] + "…") if len(w["text"]) > 160 else w["text"],
                    "sources_count": len(w.get("sources") or []),
                })

            st.dataframe(web_rows, width="stretch", hide_index=True)

            # ✅ 小調整(2)：顯示 sources
            with st.expander("Web sources（每個 search_term 的 sources）"):
                for w in web_items:
                    st.markdown(f"**{w['search_term']}**  →  `{w['chunk_id']}`")
                    if not w.get("sources"):
                        st.write("（無 sources）")
                        continue
                    for s in w["sources"][:10]:
                        if isinstance(s, dict):
                            st.write({
                                "title": s.get("title"),
                                "url": s.get("url"),
                                "source": s.get("source"),
                            })
                        else:
                            st.write(s)

            set_step("WEB_SEARCH", "✅ OK", time.perf_counter() - tw, note=f"added={len(web_items)} web summaries")
        else:
            set_step("WEB_SEARCH", "SKIP", None, note="web_search OFF or no trigger")

        # 若完全沒有 relevant 且 web 也沒跑 → 走 TRANSFORM
        if not relevant and not web_items:
            if rewrite_round < max_query_rewrites:
                ttx = time.perf_counter()
                new_q = transform_query(client, q)
                query_history.append(new_q)
                query_hist_ph.code("\n".join([f"{i}. {qq}" for i, qq in enumerate(query_history)]))
                set_step("TRANSFORM", "✅ OK", time.perf_counter() - ttx, note="no relevant docs -> rewrite")
                q = new_q
                continue
            else:
                set_step("TRANSFORM", "❌ SKIP", None, note="rewrite limit reached")
                return {
                    "final_answer": "資料不足：檢索不到足夠相關內容。你可以換個問法或上傳更多報告。",
                    "query_history": query_history,
                    "logs": logs,
                    "context": "",
                    "render_mode": "text",
                }

        # KG_EXTRACT（由 planner.needs_kg + KG mode 決策）
        use_kg = False
        if kg_mode == "FORCE":
            use_kg = True
        elif kg_mode == "AUTO":
            use_kg = bool(plan.needs_kg)
        else:
            use_kg = False

        kg_hints = ""
        if use_kg and relevant:
            tkg = time.perf_counter()
            rel_sorted = sorted(relevant, key=lambda x: x["score"], reverse=True)
            target_chunks = [it["chunk"] for it in rel_sorted[:LX_MAX_CHUNKS_PER_QUERY]]
            kg, new_edges = lazy_langextract_kg_for_chunks(api_key, target_chunks, st.session_state.lx_cache)
            kg_hints = build_kg_hint_lines(q, kg, max_lines=16)
            set_step("KG_EXTRACT", "✅ OK", time.perf_counter() - tkg, note=f"chunks={len(target_chunks)}, edges≈{new_edges}")
            st.markdown("### KG_EXTRACT（Lazy LangExtract）")
            st.text(kg_hints[:4000] if kg_hints else "（KG 線索不足：query 無法匹配到起點節點）")
        else:
            set_step("KG_EXTRACT", "SKIP", None, note="not triggered or no relevant docs")

        # 組 Context：KG hints + relevant chunks + web summaries（若有）
        ctx_parts = []
        if kg_hints.strip():
            ctx_parts.append(kg_hints)
        if relevant:
            ctx_parts.append(build_context_from_chunks(relevant, top_k=10))
        if web_items:
            ctx_parts.append(build_context_from_web_items(web_items))
        context = "\n\n".join([p for p in ctx_parts if p.strip()])

        # GENERATE + grading loop
        for gen_round in range(max_generate_retries + 1):
            tg = time.perf_counter()
            if want_bullets(q):
                ans = generate_bullets_guard(client, q, context, max_retries=1)
            else:
                ans = generate_paragraph_guard(client, q, context, max_retries=1)
            set_step("GENERATE", "✅ OK", time.perf_counter() - tg, note=f"round={gen_round}")

            st.markdown(f"### GENERATE（round {gen_round}）")
            if want_bullets(q):
                render_bullets_inline_badges(ans, badge_color="green")
            else:
                render_text_with_badges(ans, badge_color="gray")

            th = time.perf_counter()
            hall = grade_hallucinations(client, context, ans)
            set_step("GRADE_HALLU", "✅ OK", time.perf_counter() - th, note=f"{hall}")

            ta = time.perf_counter()
            good = grade_answer_adaptive(client, q, ans)
            set_step("GRADE_ANSWER", "✅ OK", time.perf_counter() - ta, note=f"{good}")

            st.markdown("### CHECK（GRADE_HALLU / GRADE_ANSWER）")
            st.write({"hallucination": hall, "answer_ok": good})

            logs.append(f"[CHECK] gen_round={gen_round} hall={hall} answer_ok={good}")

            if hall == "yes" and good == "yes":
                return {
                    "final_answer": ans,
                    "query_history": query_history,
                    "logs": logs,
                    "context": context,
                    "render_mode": ("bullets" if want_bullets(q) else "text"),
                }

            if hall == "no":
                continue

            break

        # TRANSFORM（not useful）
        if rewrite_round < max_query_rewrites:
            ttx = time.perf_counter()
            new_q = transform_query(client, q)
            query_history.append(new_q)
            query_hist_ph.code("\n".join([f"{i}. {qq}" for i, qq in enumerate(query_history)]))
            set_step("TRANSFORM", "✅ OK", time.perf_counter() - ttx, note="not useful -> rewrite")
            q = new_q
            continue

        set_step("TRANSFORM", "❌ SKIP", None, note="rewrite limit reached")
        return {
            "final_answer": "資料不足：已多次嘗試仍無法產生可被證據支持且回應問題的答案。建議換問法或增加資料。",
            "query_history": query_history,
            "logs": logs,
            "context": context,
            "render_mode": "text",
        }

    return {
        "final_answer": "資料不足：工作流未能完成。",
        "query_history": query_history,
        "logs": logs,
        "context": "",
        "render_mode": "text",
    }


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="研究報告助手", layout="wide")

OPENAI_API_KEY = get_openai_api_key()
client = get_client(OPENAI_API_KEY)
api_key = OPENAI_API_KEY

# Session State
if "file_rows" not in st.session_state:
    st.session_state.file_rows: List[FileRow] = []
if "file_bytes" not in st.session_state:
    st.session_state.file_bytes: Dict[str, bytes] = {}
if "store" not in st.session_state:
    st.session_state.store: Optional[FaissStore] = None
if "processed_keys" not in st.session_state:
    st.session_state.processed_keys = set()
if "default_outputs_cache" not in st.session_state:
    st.session_state.default_outputs_cache = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[Dict[str, Any]] = []
if "lx_cache" not in st.session_state:
    st.session_state.lx_cache = {}

def push_corpus_default_outputs_to_chat(bundle: Dict[str, str]):
    st.session_state.chat_history.append({
        "role": "assistant",
        "kind": "text",
        "content": "我已把所有上傳報告融合成一份「整體預設輸出」（摘要/核心主張/推論鏈），每個 bullet 都含報告名稱來源。",
    })
    st.session_state.chat_history.append({
        "role": "assistant",
        "kind": "default_outputs_corpus",
        "title": "整體融合（全部上傳報告）",
        "summary": bundle["summary"],
        "claims": bundle["claims"],
        "chain": bundle["chain"],
    })

def render_chat_message(msg: Dict[str, Any]):
    role = msg.get("role", "assistant")
    with st.chat_message(role):
        kind = msg.get("kind", "text")
        if kind == "default_outputs_corpus":
            st.markdown(f"## 預設輸出：{msg['title']}")
            st.markdown("### 1) 整體摘要（融合多份報告）")
            render_bullets_inline_badges(msg["summary"], badge_color="green")
            st.markdown("### 2) 核心主張（融合多份報告）")
            render_bullets_inline_badges(msg["claims"], badge_color="violet")
            st.markdown("### 3) 推論鏈 / 傳導機制（融合多份報告）")
            render_bullets_inline_badges(msg["chain"], badge_color="orange")
        else:
            st.markdown(msg.get("content", ""))

# 文件管理 popover（web_search 預設 OFF）
with st.popover("📦 文件管理（上傳 / OCR / 建索引 / 設定）", width="content"):
    st.caption("支援 PDF/TXT/PNG/JPG。PDF 若文字抽取偏少會建議 OCR（逐檔可勾選）。")

    kg_mode = st.radio(
        "KG 模式（由 Planner needs_kg + 此選項共同決定）",
        options=["AUTO", "OFF", "FORCE"],
        index=0,
        horizontal=True,
        key="kg_mode",
    )

    web_mode = st.radio(
        "Web search（僅當檢索不足才啟用）",
        options=["OFF", "AUTO"],
        index=0,  # ✅ 依你要求：預設 OFF
        horizontal=True,
        help="AUTO：coverage 不足或 relevant 太少時，用 WebSearch agent 補足 Context；OFF：完全不使用網路",
        key="web_mode",
    )

    up = st.file_uploader(
        "上傳文件",
        type=["pdf", "txt", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="uploader",
    )

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

            if ext == ".pdf":
                pdf_pages = extract_pdf_text_pages(data)
                pages = len(pdf_pages)
                extracted_chars, blank_pages, blank_ratio = analyze_pdf_text_quality(pdf_pages)
            elif ext == ".txt":
                text = norm_space(data.decode("utf-8", errors="ignore"))
                extracted_chars = len(text)
            elif ext in (".png", ".jpg", ".jpeg"):
                extracted_chars = 0

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
                    blank_pages=blank_pages,
                    blank_ratio=blank_ratio,
                    likely_scanned=likely_scanned,
                    use_ocr=use_ocr,
                )
            )

    st.markdown("### 文件清單（精簡：OCR / 檔名 / 頁 / tok / 建議）")

    if not st.session_state.file_rows:
        st.info("尚未上傳文件。")
    else:
        header_cols = st.columns([1, 6, 1, 1, 1])
        header_cols[0].markdown("**OCR**")
        header_cols[1].markdown("**檔名**")
        header_cols[2].markdown("**頁**")
        header_cols[3].markdown("**tok**")
        header_cols[4].markdown("**建議**")

        for idx, r in enumerate(st.session_state.file_rows):
            cols = st.columns([1, 6, 1, 1, 1])

            if r.ext in (".png", ".jpg", ".jpeg"):
                st.session_state.file_rows[idx].use_ocr = True
                cols[0].checkbox(" ", value=True, key=f"ocr_{idx}", disabled=True)
            elif r.ext == ".txt":
                st.session_state.file_rows[idx].use_ocr = False
                cols[0].checkbox(" ", value=False, key=f"ocr_{idx}", disabled=True)
            else:
                val = cols[0].checkbox(" ", value=bool(r.use_ocr), key=f"ocr_{idx}")
                st.session_state.file_rows[idx].use_ocr = bool(val)

            short = truncate_filename(r.name, 34)
            with cols[1]:
                name_cols = st.columns([12, 1])
                name_cols[0].markdown(short)
                name_cols[1].badge(" ", icon=":material/info:", color="gray", width="content", help=r.name)

            cols[2].markdown(str(r.pages if r.pages is not None else "-"))
            cols[3].markdown(str(r.token_est))

            with cols[4]:
                if r.likely_scanned and r.ext == ".pdf":
                    st.badge("建議 OCR", icon=":material/warning:", color="orange", width="content")
                elif r.ext in (".png", ".jpg", ".jpeg"):
                    st.badge("必 OCR", icon=":material/image:", color="orange", width="content")
                else:
                    st.markdown("")

        st.divider()
        b1, b2, b3 = st.columns([1, 1, 1])
        build_btn = b1.button("🚀 建立索引 + 融合預設輸出", type="primary", width="stretch")
        clear_btn = b2.button("🧹 清空全部", width="stretch")
        clear_lx_cache_btn = b3.button("🧽 清 LangExtract 快取", width="stretch")

        if clear_btn:
            st.session_state.file_rows = []
            st.session_state.file_bytes = {}
            st.session_state.store = None
            st.session_state.processed_keys = set()
            st.session_state.default_outputs_cache = {}
            st.session_state.chat_history = []
            st.session_state.lx_cache = {}
            st.rerun()

        if clear_lx_cache_btn:
            st.session_state.lx_cache = {}
            st.success("已清除 LangExtract 快取。")

        if build_btn:
            need_ocr = any(r.ext == ".pdf" and r.use_ocr for r in st.session_state.file_rows)
            if need_ocr and not HAS_PYMUPDF:
                st.error("你有勾選 PDF OCR，但環境未安裝 pymupdf。請先 pip install pymupdf，再重試。")
                st.stop()

            with st.status("建索引中（增量：OCR + embeddings；不做 LangExtract）...", expanded=True) as s:
                t0 = time.perf_counter()
                store, stats, processed_keys = build_indices_incremental_no_kg(
                    client=client,
                    file_rows=st.session_state.file_rows,
                    file_bytes_map=st.session_state.file_bytes,
                    store=st.session_state.store,
                    processed_keys=st.session_state.processed_keys,
                )
                st.session_state.store = store
                st.session_state.processed_keys = processed_keys
                s.write(f"新增報告數：{stats['new_reports']}")
                s.write(f"新增 chunks：{stats['new_chunks']}")
                s.write(f"耗時：{time.perf_counter() - t0:.2f}s")
                s.update(state="complete")

            corpus_key = str(sorted(list(st.session_state.processed_keys)))
            old_key = st.session_state.default_outputs_cache.get("_corpus_key")

            with st.status("產生融合預設輸出（一次生成三份；gpt-5.2 reasoning=medium）...", expanded=True) as s2:
                if old_key == corpus_key and st.session_state.default_outputs_cache.get("bundle"):
                    s2.write("融合預設輸出未變更（沿用快取）。")
                    s2.update(state="complete")
                else:
                    chosen = pick_corpus_chunks_for_default(st.session_state.store.chunks)
                    ctx = render_chunks_with_ids(chosen)
                    bundle = generate_default_outputs_bundle(client, "整體融合（全部上傳報告）", ctx, max_retries=2)
                    st.session_state.default_outputs_cache["_corpus_key"] = corpus_key
                    st.session_state.default_outputs_cache["bundle"] = bundle
                    s2.update(state="complete")

            st.session_state.chat_history = []
            push_corpus_default_outputs_to_chat(st.session_state.default_outputs_cache["bundle"])
            st.rerun()


# popover 外：狀態
if st.session_state.store is None:
    st.info("尚未建立索引。請點「📦 文件管理（上傳 / OCR / 建索引 / 設定）」開始。")
else:
    st.success(
        f"已建立索引：檔案數={len(st.session_state.file_rows)} / chunks={len(st.session_state.store.chunks)} / "
        f"LangExtract快取chunks={len(st.session_state.lx_cache)}"
    )

st.divider()

# Chat 主畫面
st.subheader("Chat（Coverage-based WebSearch，預設關）")
for msg in st.session_state.chat_history:
    render_chat_message(msg)

if st.session_state.store is None:
    st.stop()

prompt = st.chat_input("輸入問題：前景/排序/理由/傳導機制/跨報告比較…")
if prompt:
    st.session_state.chat_history.append({"role": "user", "kind": "text", "content": prompt})
    render_chat_message(st.session_state.chat_history[-1])

    with st.chat_message("assistant"):
        with st.status("Workflow：PLAN → RETRIEVE → COVERAGE → GRADE_DOCS → (AUTO) WEB_SEARCH → (可選) KG_EXTRACT → GENERATE → GRADE(HALLU) → GRADE(ANSWER)", expanded=True) as status:
            result = run_adaptive_rag(
                client=client,
                api_key=api_key,
                store=st.session_state.store,
                question=prompt,
                kg_mode=st.session_state.get("kg_mode", "AUTO"),
                web_mode=st.session_state.get("web_mode", "OFF"),  # ✅ 預設 OFF
                top_k_per_query=4,
                max_total_chunks=18,
                max_query_rewrites=2,
                max_generate_retries=2,
            )
            status.update(state="complete", expanded=False)

        st.markdown("## 最終回答")
        if result.get("render_mode") == "bullets":
            render_bullets_inline_badges(result["final_answer"], badge_color="green")
        else:
            render_text_with_badges(result["final_answer"], badge_color="gray")

        with st.expander("查看 debug（query history / logs / context）"):
            st.markdown("### Query history")
            st.code("\n".join([f"{i}. {q}" for i, q in enumerate(result.get("query_history", []))]))
            st.markdown("### Logs")
            st.text("\n".join(result.get("logs", [])))
            st.markdown("### Context（節錄）")
            st.text((result.get("context", "") or "")[:12000])

    st.session_state.chat_history.append({"role": "assistant", "kind": "text", "content": result["final_answer"]})
