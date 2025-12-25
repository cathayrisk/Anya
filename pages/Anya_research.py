# app.py
# -*- coding: utf-8 -*-
"""
研究報告助手（FAISS + OpenAI embeddings + LangExtract KG + Chat + Workflow UI）
領域：總經 / 金融 / 財務 / 氣候風險 / 永續金融

重點：
- 文件管理（上傳/清單/OCR/建索引/清空）全部在 popover（width="content"）
- 文件清單：使用OCR 最前；檔名截斷；完整檔名用 st.badge(help=...) tooltip；不顯示 file_id
- 建索引加速：
  1) PDF 文字抽取優先 PyMuPDF（若有）
  2) LangExtract 逐頁並行（ThreadPoolExecutor）
  3) OCR 逐頁小並行（避免 rate limit）
  4) embeddings 大批次（減少 round-trip）
  5) 增量索引（只處理新檔或 OCR 設定變更）
- 預設輸出加速：
  - 只對「新/變更」報告產出
  - 一次 LLM 產三份（SUMMARY/CLAIMS/CHAIN），並嚴格檢查每個 bullet 都要有引用
- Chat：Workflow UI（RETRIEVE/GRADE/TRANSFORM/GENERATE/CHECK）+ ✅/❌ + 秒數 + relevant chunk 展開全文
- 呈現：引用用 st.badge 顯示（文字本體更乾淨），但內部仍用引用字串做嚴格檢查

環境：
- OPENAI_API_KEY 或 st.secrets["OPENAI_KEY"] 擇一
"""

from __future__ import annotations

import os
import re
import io
import uuid
import math
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import numpy as np
import faiss
import networkx as nx
from pypdf import PdfReader

from openai import OpenAI
import langextract as lx

try:
    import fitz  # pymupdf
    HAS_PYMUPDF = True
except Exception:
    HAS_PYMUPDF = False


# =========================
# 固定模型設定（依你需求：不讓使用者輸入）
# =========================
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-5.2"
LX_MODEL = "gpt-5.2"
OCR_MODEL = "gpt-4.1-mini"  # 若你確認 gpt-5.2 支援 vision，可改成 gpt-5.2


# =========================
# 效能參數（可調）
# =========================
EMBED_BATCH_SIZE = 256
LX_MAX_WORKERS = 6
OCR_MAX_WORKERS = 2


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

def truncate_filename(name: str, max_len: int = 30) -> str:
    if len(name) <= max_len:
        return name
    base, ext = os.path.splitext(name)
    keep = max(10, max_len - len(ext) - 1)
    return f"{base[:keep]}…{ext}"


# =========================
# OpenAI helpers
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

def gen_text(client: OpenAI, system: str, user: str, model: str = LLM_MODEL) -> str:
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.output_text if getattr(resp, "output_text", None) else str(resp)

def gen_yesno(client: OpenAI, system: str, user: str) -> str:
    out = gen_text(
        client,
        system=system + "\n\n只回覆 'yes' 或 'no'（小寫），不要加任何其他文字。",
        user=user,
        model=LLM_MODEL,
    ).strip().lower()
    if "yes" in out and "no" in out:
        return "yes" if out.find("yes") < out.find("no") else "no"
    if "yes" in out:
        return "yes"
    if "no" in out:
        return "no"
    return "no"


# =========================
# 檔案 / OCR
# =========================
@dataclass
class FileRow:
    file_id: str          # 內部用
    file_sig: str         # sha1(bytes)
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

def ocr_image_bytes_with_openai(client: OpenAI, image_bytes: bytes, model: str = OCR_MODEL) -> str:
    system = "你是一個OCR工具。只輸出可見文字與表格內容（若有表格用 Markdown 表格）。中文請用繁體中文。不要加評論。"
    user_content = [
        {"type": "input_text", "text": "請擷取圖片中所有可見文字（包含小字/註腳）。若無法辨識請標記[無法辨識]。"},
        {"type": "input_image", "image_bytes": image_bytes},
    ]
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
    )
    return resp.output_text if getattr(resp, "output_text", None) else str(resp)

def ocr_pdf_pages_with_openai_parallel(client: OpenAI, pdf_bytes: bytes, dpi: int = 180) -> List[Tuple[int, str]]:
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
        futs = {ex.submit(ocr_image_bytes_with_openai, client, img_bytes): page_no for page_no, img_bytes in page_imgs}
        for fut in as_completed(futs):
            page_no = futs[fut]
            try:
                results[page_no] = norm_space(fut.result())
            except Exception:
                results[page_no] = ""

    return [(p, results.get(p, "")) for p, _ in page_imgs]


# =========================
# FAISS
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
# KG
# =========================
@dataclass
class Prov:
    report_id: str
    title: str
    page: Optional[int]
    char_start: Optional[int]
    char_end: Optional[int]
    snippet: str

ALLOWED_RELATIONS = {
    "CAUSES", "DRIVES", "AFFECTS", "INCREASES", "DECREASES", "CORRELATES_WITH",
    "ANNOUNCES", "TIGHTENS", "EASES",
    "ASSUMES_SCENARIO", "HAS_RISK", "HAS_METRIC", "TARGETS",
    "IS_A", "LOCATED_IN", "HAS_TIME", "HAS_SOURCE",
    "MENTIONS",
}

def norm_rel(r: str) -> str:
    r = norm_space(r).upper().replace(" ", "_")
    r = re.sub(r"[^A-Z0-9_]+", "", r)
    mapping = {
        "IMPACTS": "AFFECTS",
        "IMPACT": "AFFECTS",
        "INCREASE": "INCREASES",
        "DECREASE": "DECREASES",
        "CORRELATES": "CORRELATES_WITH",
        "CORRELATION": "CORRELATES_WITH",
        "SCENARIO": "ASSUMES_SCENARIO",
    }
    return mapping.get(r, r)

class KnowledgeGraph:
    def __init__(self):
        self.g = nx.MultiDiGraph()

    def add_edge(self, s: str, r: str, o: str, prov: Prov, attrs: Optional[Dict[str, Any]] = None):
        s = norm_space(s)
        o = norm_space(o)
        r = norm_rel(r)
        if not s or not o or not r:
            return
        if r not in ALLOWED_RELATIONS:
            return
        if s not in self.g:
            self.g.add_node(s, label=s)
        if o not in self.g:
            self.g.add_node(o, label=o)
        self.g.add_edge(s, o, key=str(uuid.uuid4()), relation=r, prov=asdict(prov), attrs=attrs or {})

    def find_nodes_in_query(self, query: str, max_n: int = 2) -> List[str]:
        q = norm_space(query)
        hits = []
        for n in self.g.nodes():
            if len(n) >= 4 and n in q:
                hits.append(n)
        return hits[:max_n]

    def bfs_context(self, start: str, max_edges: int = 18) -> List[Dict[str, Any]]:
        if start not in self.g:
            return []
        out = []
        for u, v, k, data in nx.edge_bfs(self.g, start):
            out.append({"u": u, "v": v, "rel": data.get("relation"), "prov": data.get("prov")})
            if len(out) >= max_edges:
                break
        return out


# =========================
# LangExtract
# =========================
def lx_prompt() -> str:
    return (
        "Extract structured information from macro/finance/climate-risk/sustainable-finance reports.\n"
        "Rules:\n"
        "1) Use exact text spans for extraction_text. Do NOT paraphrase.\n"
        "2) Extract only two classes: claim, relation.\n"
        "3) claim.attributes may include: theme, stance, confidence, time, implication.\n"
        "4) relation.attributes must include: {subject, relation, object}. Optional: {time, polarity, qualifier}.\n"
        "5) Only extract relations explicitly supported by text; if unsure, skip.\n"
    )

def lx_examples() -> List[lx.data.ExampleData]:
    t1 = (
        "We expect US CPI inflation to decelerate in 2025Q2 as energy prices fall. "
        "The Fed is likely to keep policy restrictive through mid-2025."
    )
    ex1 = lx.data.ExampleData(
        text=t1,
        extractions=[
            lx.data.Extraction(
                extraction_class="claim",
                extraction_text="US CPI inflation to decelerate in 2025Q2",
                attributes={"theme": "inflation_outlook", "confidence": "medium"},
            ),
            lx.data.Extraction(
                extraction_class="relation",
                extraction_text="as energy prices fall",
                attributes={"subject": "energy prices", "relation": "DECREASES", "object": "US CPI inflation", "time": "2025Q2"},
            ),
        ],
    )
    return [ex1]

def run_langextract(text: str, api_key: str) -> lx.data.AnnotatedDocument:
    return lx.extract(
        text_or_documents=text,
        prompt_description=lx_prompt(),
        examples=lx_examples(),
        model_id=LX_MODEL,
        api_key=api_key,
        extraction_passes=2,
        max_char_buffer=1200,
        max_workers=8,
        fence_output=True,
        use_schema_constraints=False,
    )


# =========================
# 引用檢查 + badge 呈現
# =========================
CIT_RE = re.compile(r"\[[^\]]+\|\s*[^\]]+\]")  # [title pX | chunk_id]
BULLET_RE = re.compile(r"^\s*(?:[-•*]|\d+\.)\s+")
CIT_PARSE_RE = re.compile(r"\[([^\]]+?)\s+p(\d+|-)\s*\|\s*([A-Za-z0-9_\-]+)\]")

def bullets_all_have_citations(md: str) -> Tuple[bool, List[str]]:
    bad_lines = []
    lines = (md or "").splitlines()
    has_bullet = any(BULLET_RE.match(l) for l in lines)
    for line in lines:
        if BULLET_RE.match(line) and not CIT_RE.search(line):
            bad_lines.append(line)
    if not has_bullet:
        return False, ["（沒有產出任何 bullet 條列）"]
    return (len(bad_lines) == 0), bad_lines

def paragraphs_all_have_citations(md: str) -> Tuple[bool, List[str]]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", md or "") if p.strip()]
    bad = []
    if not paras:
        return False, ["（沒有輸出任何段落）"]
    for p in paras:
        if not CIT_RE.search(p):
            bad.append(p[:120])
    return (len(bad) == 0), bad

def generate_with_bullet_citation_guard(client: OpenAI, user: str, max_retries: int = 2) -> str:
    system = (
        "你是嚴謹的研究助理。\n"
        "規則：只能根據資料回答，不可腦補；輸出純 bullet（每行 - 開頭）；每個 bullet 句尾必須有引用 [報告 p頁 | chunk_id]。\n"
    )
    last = ""
    for _ in range(max_retries + 1):
        out = gen_text(client, system, user, model=LLM_MODEL)
        ok, _ = bullets_all_have_citations(out)
        if ok:
            return out
        last = out
        user += "\n\n【強制修正】重寫：每個 bullet 句尾都要有 [報告 p頁 | chunk_id]。"
    return last

def generate_with_paragraph_citation_guard(client: OpenAI, user: str, max_retries: int = 2) -> str:
    system = (
        "你是嚴謹的研究助理。\n"
        "規則：只能根據 Context 回答，不可腦補；2~4 段；每段至少 1 個引用 [報告 p頁 | chunk_id]。\n"
    )
    last = ""
    for _ in range(max_retries + 1):
        out = gen_text(client, system, user, model=LLM_MODEL)
        ok, _ = paragraphs_all_have_citations(out)
        if ok:
            return out
        last = out
        user += "\n\n【強制修正】重寫：每段至少 1 個 [報告 p頁 | chunk_id]。"
    return last

def _parse_citations(cits: List[str]) -> List[Dict[str, str]]:
    parsed = []
    for c in cits:
        m = CIT_PARSE_RE.search(c)
        if not m:
            parsed.append({"title": "來源", "page": "-", "chunk_id": c.strip("[]")})
        else:
            parsed.append({"title": m.group(1).strip(), "page": m.group(2).strip(), "chunk_id": m.group(3).strip()})
    return parsed

def _render_badges(parsed: List[Dict[str, str]], color: str = "blue", icon: Optional[str] = ":material_bookmark:"):
    if not parsed:
        return
    per_row = 4
    for i in range(0, len(parsed), per_row):
        row = parsed[i:i+per_row]
        cols = st.columns(len(row))
        for col, item in zip(cols, row):
            title = item["title"]
            title_short = title if len(title) <= 18 else (title[:18] + "…")
            page = item["page"]
            chunk_id = item["chunk_id"]
            label = f"{title_short} p{page} · {chunk_id}"
            with col:
                st.badge(label, icon=icon, color=color, help=f"{title} p{page} | {chunk_id}")

def render_bullets_with_badges(md_bullets: str, badge_color: str = "blue"):
    lines = [l.rstrip() for l in (md_bullets or "").splitlines() if l.strip()]
    for line in lines:
        if not BULLET_RE.match(line):
            continue
        cits = CIT_RE.findall(line)
        clean = CIT_RE.sub("", line).strip()
        st.markdown(clean)
        _render_badges(_parse_citations(cits), color=badge_color)

def render_text_with_badges(md_text: str, badge_color: str = "gray"):
    cits = CIT_RE.findall(md_text or "")
    clean = CIT_RE.sub("", md_text or "").strip()
    st.markdown(clean if clean else "（無內容）")
    parsed = _parse_citations(sorted(set(cits)))
    if parsed:
        st.markdown("**來源：**")
        _render_badges(parsed, color=badge_color, icon=":material_source:")


# =========================
# ✅ 一次生成三份預設輸出（SUMMARY/CLAIMS/CHAIN）
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

def generate_default_outputs_bundle_with_guard(client: OpenAI, title: str, ctx: str, max_retries: int = 2) -> Dict[str, str]:
    system = (
        "你是嚴謹的研究助理，只能根據我提供的資料回答，不可腦補。\n"
        "硬性規則：\n"
        "1) 你必須輸出三個區塊，且順序/標題固定：### SUMMARY、### CLAIMS、### CHAIN。\n"
        "2) 每個區塊都必須是純 bullet（每行以 - 開頭），不要段落。\n"
        "3) 每個 bullet 句尾必須附引用，格式固定：[報告 p頁 | chunk_id]\n"
    )

    base_user = (
        f"請針對報告《{title}》一次輸出三份內容：\n"
        f"- SUMMARY：8~14 bullets（結論/預測/假設/風險/限制/市場含意）\n"
        f"- CLAIMS：8~14 bullets（可驗證主張）\n"
        f"- CHAIN：6~12 bullets（傳導：驅動→中介→結論→風險）\n\n"
        f"資料：\n{ctx}\n\n"
        f"現在請輸出：\n"
        f"### SUMMARY\n"
        f"(bullets...)\n"
        f"### CLAIMS\n"
        f"(bullets...)\n"
        f"### CHAIN\n"
        f"(bullets...)\n"
    )

    user = base_user
    for _ in range(max_retries + 1):
        out = gen_text(client, system, user, model=LLM_MODEL)
        parts = _split_default_bundle(out)

        ok_s, _ = bullets_all_have_citations(parts.get("summary", ""))
        ok_c, _ = bullets_all_have_citations(parts.get("claims", ""))
        ok_h, _ = bullets_all_have_citations(parts.get("chain", ""))

        if ok_s and ok_c and ok_h:
            return {"summary": parts["summary"], "claims": parts["claims"], "chain": parts["chain"]}

        user = base_user + "\n\n【強制修正】整份重寫：三區塊皆為純 bullet，且每個 bullet 句尾都有 [報告 p頁 | chunk_id]。"

    return {"summary": "", "claims": "", "chain": ""}


# =========================
# 索引（增量 + 并行 + 批次）
# =========================
def build_indices_incremental(
    client: OpenAI,
    api_key: str,
    file_rows: List[FileRow],
    file_bytes_map: Dict[str, bytes],
    store: Optional[FaissStore],
    kg: KnowledgeGraph,
    processed_keys: set,
    chunk_size: int = 900,
    overlap: int = 150,
) -> Tuple[FaissStore, KnowledgeGraph, Dict[str, Any], set, List[str]]:
    if store is None:
        dim = embed_texts(client, ["dim_probe"]).shape[1]
        store = FaissStore(dim)

    stats = {"new_reports": 0, "new_chunks": 0, "kg_nodes": 0, "kg_edges": 0}
    new_titles: List[str] = []

    new_chunks: List[Chunk] = []
    new_texts: List[str] = []

    # to_process: (sha1, use_ocr) 沒處理過才跑
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
        new_titles.append(title)

        # pages
        if row.ext == ".pdf":
            if row.use_ocr:
                pages = ocr_pdf_pages_with_openai_parallel(client, data)
            else:
                pages = extract_pdf_text_pages(data)
        elif row.ext == ".txt":
            pages = [(None, norm_space(data.decode("utf-8", errors="ignore")))]
        elif row.ext in (".png", ".jpg", ".jpeg"):
            txt = norm_space(ocr_image_bytes_with_openai(client, data))
            pages = [(None, txt)]
        else:
            pages = [(None, "")]

        # chunks
        for page_no, page_text in pages:
            if not page_text:
                continue
            for i, ch in enumerate(chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)):
                cid = f"{report_id}_p{page_no if page_no else 'na'}_c{i}"
                new_chunks.append(Chunk(cid, report_id, title, page_no if isinstance(page_no, int) else None, ch))
                new_texts.append(ch)

        # LangExtract：逐頁并行
        def lx_one_page(page_no: Optional[int], page_text: str):
            if not page_text:
                return []
            ann = run_langextract(page_text, api_key=api_key)
            out_edges = []
            for e in ann.extractions:
                cls = getattr(e, "extraction_class", "")
                etext = getattr(e, "extraction_text", "")
                attrs = getattr(e, "attributes", {}) or {}
                cstart = getattr(e, "char_start", None)
                cend = getattr(e, "char_end", None)

                snippet = page_text[:220]
                if cstart is not None and cend is not None and 0 <= cstart < len(page_text):
                    snippet = page_text[max(0, cstart - 80): min(len(page_text), cend + 80)]

                prov = Prov(
                    report_id=report_id,
                    title=title,
                    page=page_no if isinstance(page_no, int) else None,
                    char_start=cstart,
                    char_end=cend,
                    snippet=snippet,
                )

                if cls == "relation":
                    s = attrs.get("subject", "")
                    r = attrs.get("relation", "")
                    o = attrs.get("object", "")
                    out_edges.append((s, r, o, prov, attrs))
                elif cls == "claim":
                    claim_node = f"CLAIM: {norm_space(etext)}"
                    out_edges.append((title, "MENTIONS", claim_node, prov, attrs))
            return out_edges

        with ThreadPoolExecutor(max_workers=LX_MAX_WORKERS) as ex:
            futs = {ex.submit(lx_one_page, pno, ptxt): pno for pno, ptxt in pages}
            for fut in as_completed(futs):
                try:
                    edges = fut.result()
                except Exception:
                    edges = []
                for s, r, o, prov, attrs in edges:
                    kg.add_edge(s=s, r=r, o=o, prov=prov, attrs=attrs)

        processed_keys.add((row.file_sig, bool(row.use_ocr)))

    # embeddings：大批次
    if new_texts:
        vecs_list = []
        for i in range(0, len(new_texts), EMBED_BATCH_SIZE):
            vecs_list.append(embed_texts(client, new_texts[i:i+EMBED_BATCH_SIZE]))
        vecs = np.vstack(vecs_list)
        store.add(vecs, new_chunks)

    stats["new_chunks"] = len(new_chunks)
    stats["kg_nodes"] = kg.g.number_of_nodes()
    stats["kg_edges"] = kg.g.number_of_edges()
    return store, kg, stats, processed_keys, sorted(set(new_titles))


# =========================
# 預設輸出：挑選 chunk context
# =========================
def pick_chunks_for_report(all_chunks: List[Chunk], title: str, max_n: int = 12) -> List[Chunk]:
    kw = re.compile(r"(conclusion|outlook|risk|implication|forecast|scenario|inflation|rate|credit|spread|emission|transition|physical)", re.I)

    def score(c: Chunk) -> float:
        s = 0.0
        if c.page is not None:
            s += max(0.0, 8.0 - min(8.0, float(c.page)))
        if kw.search(c.text or ""):
            s += 6.0
        s += min(2.0, len(c.text) / 1200.0)
        return s

    cands = [c for c in all_chunks if c.title == title]
    cands = sorted(cands, key=score, reverse=True)
    return cands[:max_n]

def render_chunks_with_ids(chunks: List[Chunk], max_chars_each: int = 900) -> str:
    parts = []
    for c in chunks:
        head = f"[{c.title} p{c.page if c.page else '-'} | {c.chunk_id}]"
        parts.append(head + "\n" + c.text[:max_chars_each])
    return "\n\n".join(parts)


# =========================
# Chat workflow（UI）
# =========================
def want_bullets(question: str) -> bool:
    return bool(re.search(r"(列出|有哪些|所有|清單|彙總|摘要|總結)", question))

def grade_doc_relevance(client: OpenAI, question: str, doc_text: str) -> str:
    system = "你是文件相關性評分者。片段能支持回答=>yes，否則=>no。"
    user = f"Question:\n{question}\n\nDocument:\n{doc_text[:2200]}"
    return gen_yesno(client, system, user)

def rewrite_question(client: OpenAI, question: str) -> str:
    system = "你是檢索 query 改寫者。保留原意，補上可檢索關鍵字。輸出一行改寫後問題。"
    return gen_text(client, system, question, model=LLM_MODEL).strip()

def grade_hallucination(client: OpenAI, context: str, answer: str) -> str:
    system = "判斷回答是否被 context 支持。支持=>yes，否則=>no。"
    user = f"Context:\n{context[:9000]}\n\nAnswer:\n{answer[:4500]}"
    return gen_yesno(client, system, user)

def grade_answer(client: OpenAI, question: str, answer: str) -> str:
    system = "判斷回答是否回應問題。是=>yes，否=>no。"
    user = f"Question:\n{question}\n\nAnswer:\n{answer[:4500]}"
    return gen_yesno(client, system, user)

def build_retrieval_packages(client: OpenAI, store: FaissStore, question: str, top_k: int = 10) -> List[Dict[str, Any]]:
    qvec = embed_texts(client, [question])
    hits = store.search(qvec, k=top_k)
    return [{"chunk": ch, "score": score} for score, ch in hits]

def build_context_from_chunks(items: List[Dict[str, Any]], top_k: int = 8) -> str:
    items = sorted(items, key=lambda x: x["score"], reverse=True)[:top_k]
    parts = []
    for it in items:
        ch: Chunk = it["chunk"]
        parts.append(f"[{ch.title} p{ch.page if ch.page else '-'} | {ch.chunk_id}]\n{ch.text}")
    return "\n\n".join(parts) if parts else "（找不到任何相關內容）"

def generate_answer_from_context(client: OpenAI, question: str, context: str) -> str:
    if want_bullets(question):
        user = f"Context:\n{context}\n\nQuestion:\n{question}"
        return generate_with_bullet_citation_guard(client, user, max_retries=1)
    user = (
        "請回答問題，依序：結論→依據（引用）→推論/解釋。\n"
        "每段至少1個引用 [報告 p頁 | chunk_id]。\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}"
    )
    return generate_with_paragraph_citation_guard(client, user, max_retries=1)

def _step_table(step_state: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for name, info in step_state.items():
        rows.append({
            "Step": name,
            "Status": info.get("status", "PENDING"),
            "Seconds": info.get("seconds", None),
            "Note": info.get("note", ""),
        })
    return rows

def run_chat_workflow_with_ui(
    client: OpenAI,
    store: FaissStore,
    question: str,
    *,
    max_query_rewrites: int = 2,
    max_generate_retries: int = 2,
    top_k: int = 10,
) -> Dict[str, Any]:
    query_history: List[str] = [question]
    logs: List[str] = []
    final_context = ""
    final_answer = ""

    step_state = {
        "RETRIEVE": {"status": "PENDING", "seconds": None, "note": ""},
        "GRADE": {"status": "PENDING", "seconds": None, "note": ""},
        "TRANSFORM": {"status": "PENDING", "seconds": None, "note": ""},
        "GENERATE": {"status": "PENDING", "seconds": None, "note": ""},
        "CHECK": {"status": "PENDING", "seconds": None, "note": ""},
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
        # RETRIEVE
        t0 = time.perf_counter()
        retrieved = build_retrieval_packages(client, store, q, top_k=top_k)
        set_step("RETRIEVE", "✅ OK", time.perf_counter() - t0, note=f"top_k={top_k}, got={len(retrieved)}")

        st.markdown(f"### RETRIEVE（round {rewrite_round}）")
        st.dataframe(
            [{
                "score": round(float(it["score"]), 4),
                "報告": it["chunk"].title,
                "頁": it["chunk"].page if it["chunk"].page is not None else "-",
                "chunk_id": it["chunk"].chunk_id,
                "preview": (it["chunk"].text[:140] + "…") if len(it["chunk"].text) > 140 else it["chunk"].text,
            } for it in retrieved],
            width="stretch",
            hide_index=True,
        )

        # GRADE
        t1 = time.perf_counter()
        relevant: List[Dict[str, Any]] = []
        graded_rows = []
        prog = st.progress(0, text="grading…")
        for i, it in enumerate(retrieved):
            ch: Chunk = it["chunk"]
            verdict = grade_doc_relevance(client, q, ch.text)
            graded_rows.append({
                "grade": verdict,
                "score": round(float(it["score"]), 4),
                "報告": ch.title,
                "頁": ch.page if ch.page is not None else "-",
                "chunk_id": ch.chunk_id,
                "preview": (ch.text[:140] + "…") if len(ch.text) > 140 else ch.text,
            })
            if verdict == "yes":
                relevant.append(it)
            prog.progress((i + 1) / max(1, len(retrieved)), text=f"grading… {i+1}/{len(retrieved)}")

        set_step("GRADE", "✅ OK", time.perf_counter() - t1, note=f"relevant={len(relevant)}/{len(retrieved)}")

        st.markdown("### GRADE（doc relevance yes/no）")
        st.dataframe(graded_rows, width="stretch", hide_index=True)

        st.markdown("### Relevant Chunks（YES）")
        if not relevant:
            st.info("這一輪沒有找到相關 chunks（全部被判定 no）。")
        else:
            rel_sorted = sorted(relevant, key=lambda x: x["score"], reverse=True)[:top_k]
            st.dataframe(
                [{
                    "score": round(float(it["score"]), 4),
                    "報告": it["chunk"].title,
                    "頁": it["chunk"].page if it["chunk"].page is not None else "-",
                    "chunk_id": it["chunk"].chunk_id,
                    "preview": (it["chunk"].text[:180] + "…") if len(it["chunk"].text) > 180 else it["chunk"].text,
                } for it in rel_sorted],
                width="stretch",
                hide_index=True,
            )
            st.markdown("#### 展開看全文")
            for it in rel_sorted:
                ch = it["chunk"]
                with st.expander(f"{ch.title} p{ch.page if ch.page else '-'} | {ch.chunk_id} | score={it['score']:.3f}"):
                    st.text(ch.text)

        # TRANSFORM：沒 relevant 才改寫
        if not relevant:
            if rewrite_round < max_query_rewrites:
                t2 = time.perf_counter()
                st.markdown("### TRANSFORM（rewrite query）")
                new_q = rewrite_question(client, q)
                query_history.append(new_q)
                query_hist_ph.code("\n".join([f"{i}. {qq}" for i, qq in enumerate(query_history)]))
                set_step("TRANSFORM", "✅ OK", time.perf_counter() - t2, note="rewrite applied (no relevant docs)")
                q = new_q
                continue
            else:
                set_step("TRANSFORM", "❌ SKIP", None, note="rewrite limit reached")
                return {
                    "final_answer": "資料不足：檢索不到足夠相關內容。你可以換個問法或上傳更多報告。",
                    "query_history": query_history,
                    "context": "",
                    "logs": logs,
                    "render_mode": "text",
                }

        # GENERATE + CHECK
        context = build_context_from_chunks(relevant, top_k=8)
        final_context = context

        for gen_round in range(max_generate_retries + 1):
            t3 = time.perf_counter()
            st.markdown(f"### GENERATE（round {gen_round}）")
            ans = generate_answer_from_context(client, q, context)
            set_step("GENERATE", "✅ OK", time.perf_counter() - t3, note=f"gen_round={gen_round}")

            st.markdown("#### Draft answer（呈現用 badge）")
            if want_bullets(q):
                render_bullets_with_badges(ans, badge_color="blue")
            else:
                render_text_with_badges(ans, badge_color="gray")

            t4 = time.perf_counter()
            st.markdown("### CHECK（hallucination / answer）")
            hall = grade_hallucination(client, context, ans)
            good = grade_answer(client, q, ans)
            set_step("CHECK", "✅ OK", time.perf_counter() - t4, note=f"hall={hall}, answer_ok={good}")
            logs.append(f"[CHECK] gen_round={gen_round} hall={hall} answer_ok={good}")

            st.write({"hallucination": hall, "answer_ok": good})

            if hall == "yes" and good == "yes":
                final_answer = ans
                return {
                    "final_answer": final_answer,
                    "query_history": query_history,
                    "context": final_context,
                    "logs": logs,
                    "render_mode": ("bullets" if want_bullets(q) else "text"),
                }

            if hall == "no":
                continue
            if good == "no":
                break

        # 生成不理想 → TRANSFORM
        if rewrite_round < max_query_rewrites:
            t2 = time.perf_counter()
            st.markdown("### TRANSFORM（rewrite query）")
            new_q = rewrite_question(client, q)
            query_history.append(new_q)
            query_hist_ph.code("\n".join([f"{i}. {qq}" for i, qq in enumerate(query_history)]))
            set_step("TRANSFORM", "✅ OK", time.perf_counter() - t2, note="rewrite applied (generation not ok)")
            q = new_q
            continue

        set_step("TRANSFORM", "❌ SKIP", None, note="rewrite limit reached")
        return {
            "final_answer": "資料不足：已多次嘗試仍無法產生可被證據支持且回應問題的答案。建議換問法或增加資料。",
            "query_history": query_history,
            "context": final_context,
            "logs": logs,
            "render_mode": "text",
        }

    return {
        "final_answer": "資料不足：工作流未能完成。",
        "query_history": query_history,
        "context": final_context,
        "logs": logs,
        "render_mode": "text",
    }


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="研究報告助手（最終完整版）", layout="wide")
st.title("研究報告助手（最終完整版）")

OPENAI_API_KEY = get_openai_api_key()
client = get_client(OPENAI_API_KEY)
api_key = OPENAI_API_KEY  # LangExtract 用同一把 key

# Session State
if "file_rows" not in st.session_state:
    st.session_state.file_rows: List[FileRow] = []
if "file_bytes" not in st.session_state:
    st.session_state.file_bytes: Dict[str, bytes] = {}
if "store" not in st.session_state:
    st.session_state.store: Optional[FaissStore] = None
if "kg" not in st.session_state:
    st.session_state.kg = KnowledgeGraph()
if "processed_keys" not in st.session_state:
    st.session_state.processed_keys = set()  # {(sha1, use_ocr)}
if "default_outputs_cache" not in st.session_state:
    st.session_state.default_outputs_cache = {}  # {title: {report_key, summary, claims, chain}}
if "report_key_by_title" not in st.session_state:
    st.session_state.report_key_by_title = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[Dict[str, Any]] = []

def push_default_outputs_to_chat(default_outputs: Dict[str, Dict[str, str]]):
    st.session_state.chat_history.append({
        "role": "assistant",
        "kind": "text",
        "content": "我先把上傳報告的預設輸出整理好囉（摘要/核心主張/推論鏈；每個 bullet 都有引用）。你接下來可以直接問問題。",
    })
    for title, out in default_outputs.items():
        st.session_state.chat_history.append({
            "role": "assistant",
            "kind": "default_outputs",
            "title": title,
            "summary": out["summary"],
            "claims": out["claims"],
            "chain": out["chain"],
        })

def render_chat_message(msg: Dict[str, Any]):
    role = msg.get("role", "assistant")
    with st.chat_message(role):
        kind = msg.get("kind", "text")
        if kind == "default_outputs":
            st.markdown(f"## 預設輸出：{msg['title']}")
            st.markdown("### 1) 報告摘要")
            render_bullets_with_badges(msg["summary"], badge_color="green")
            st.markdown("### 2) 核心主張")
            render_bullets_with_badges(msg["claims"], badge_color="violet")
            st.markdown("### 3) 推論鏈 / 傳導機制")
            render_bullets_with_badges(msg["chain"], badge_color="orange")
        else:
            st.markdown(msg.get("content", ""))


# =========================
# ✅ 文件管理：全部包在 popover（寬度用 content 比較美）
# =========================
with st.popover("📦 文件管理（上傳 / OCR / 建索引）", width="content"):
    st.caption("支援 PDF/TXT/PNG/JPG。PDF 若文字抽取偏少會建議 OCR（逐檔可勾選）。")

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

    st.markdown("### 文件清單（重點欄位置前）")

    if not st.session_state.file_rows:
        st.info("尚未上傳文件。")
    else:
        # 自畫表格：窄 popover 也好看
        header_cols = st.columns([1, 4, 1, 1, 1, 1])
        header_cols[0].markdown("**OCR**")
        header_cols[1].markdown("**檔名**")
        header_cols[2].markdown("**頁**")
        header_cols[3].markdown("**字數**")
        header_cols[4].markdown("**tok**")
        header_cols[5].markdown("**建議**")

        for idx, r in enumerate(st.session_state.file_rows):
            cols = st.columns([1, 4, 1, 1, 1, 1])

            # OCR checkbox（圖片固定 True；txt 固定 False）
            if r.ext in (".png", ".jpg", ".jpeg"):
                st.session_state.file_rows[idx].use_ocr = True
                cols[0].checkbox(" ", value=True, key=f"ocr_{idx}", disabled=True)
            elif r.ext == ".txt":
                st.session_state.file_rows[idx].use_ocr = False
                cols[0].checkbox(" ", value=False, key=f"ocr_{idx}", disabled=True)
            else:
                val = cols[0].checkbox(" ", value=bool(r.use_ocr), key=f"ocr_{idx}")
                st.session_state.file_rows[idx].use_ocr = bool(val)

            # 檔名截斷 + badge tooltip 顯示完整檔名
            short = truncate_filename(r.name, 30)
            with cols[1]:
                st.markdown(short)
                st.badge("Full name", icon=":material/info:", color="gray", help=r.name)

            cols[2].markdown(str(r.pages if r.pages is not None else "-"))
            cols[3].markdown(str(r.extracted_chars))
            cols[4].markdown(str(r.token_est))
            cols[5].markdown("OCR" if r.likely_scanned else "")

        st.divider()
        b1, b2 = st.columns([1, 1])
        build_btn = b1.button("🚀 建立索引 + 預設輸出", type="primary", width="stretch")
        clear_btn = b2.button("🧹 清空全部", width="stretch")

        if clear_btn:
            st.session_state.file_rows = []
            st.session_state.file_bytes = {}
            st.session_state.store = None
            st.session_state.kg = KnowledgeGraph()
            st.session_state.processed_keys = set()
            st.session_state.default_outputs_cache = {}
            st.session_state.report_key_by_title = {}
            st.session_state.chat_history = []
            st.rerun()

        if build_btn:
            need_ocr = any(r.ext == ".pdf" and r.use_ocr for r in st.session_state.file_rows)
            if need_ocr and not HAS_PYMUPDF:
                st.error("你有勾選 PDF OCR，但環境未安裝 pymupdf。請先 pip install pymupdf，再重試。")
                st.stop()

            # 1) 增量索引
            with st.status("建索引中（增量 + 并行 + 大批次 embeddings）...", expanded=True) as s:
                t0 = time.perf_counter()
                store, kg, stats, processed_keys, new_titles = build_indices_incremental(
                    client=client,
                    api_key=api_key,
                    file_rows=st.session_state.file_rows,
                    file_bytes_map=st.session_state.file_bytes,
                    store=st.session_state.store,
                    kg=st.session_state.kg,
                    processed_keys=st.session_state.processed_keys,
                )
                st.session_state.store = store
                st.session_state.kg = kg
                st.session_state.processed_keys = processed_keys
                s.write(f"新增報告數：{stats['new_reports']}")
                s.write(f"新增 chunks：{stats['new_chunks']}")
                s.write(f"KG nodes={stats['kg_nodes']} edges={stats['kg_edges']}")
                s.write(f"耗時：{time.perf_counter() - t0:.2f}s")
                s.update(state="complete")

            # 2) 預設輸出：只跑新/變更，且一次生成三份
            # report_key = f"{file_sig}:{use_ocr}"
            titles_now = []
            title_to_key = {}
            for r in st.session_state.file_rows:
                title = os.path.splitext(r.name)[0]
                titles_now.append(title)
                title_to_key[title] = f"{r.file_sig}:{int(bool(r.use_ocr))}"

            to_regen_titles = []
            for title in sorted(set(titles_now)):
                rk = title_to_key.get(title)
                old_rk = st.session_state.report_key_by_title.get(title)
                if (title not in st.session_state.default_outputs_cache) or (old_rk != rk):
                    to_regen_titles.append(title)

            st.session_state.report_key_by_title.update(title_to_key)

            with st.status("產生預設輸出（只跑新/變更；一次生成三份）...", expanded=True) as s2:
                if not to_regen_titles:
                    s2.write("沒有偵測到新報告或變更（沿用快取）。")
                    s2.update(state="complete")
                else:
                    s2.write(f"需要重新產生：{len(to_regen_titles)} 份報告")
                    for i, title in enumerate(to_regen_titles, start=1):
                        s2.write(f"[{i}/{len(to_regen_titles)}] 產生：{title}")
                        reps = pick_chunks_for_report(st.session_state.store.chunks, title, max_n=12)
                        ctx = render_chunks_with_ids(reps)
                        bundle = generate_default_outputs_bundle_with_guard(client, title, ctx, max_retries=2)
                        rk = title_to_key[title]
                        st.session_state.default_outputs_cache[title] = {
                            "report_key": rk,
                            "summary": bundle["summary"],
                            "claims": bundle["claims"],
                            "chain": bundle["chain"],
                        }
                    s2.update(state="complete")

            # 3) 組合目前存在的 titles 的輸出（快取）
            default_outputs = {}
            for title in sorted(set(titles_now)):
                cached = st.session_state.default_outputs_cache.get(title)
                if cached:
                    default_outputs[title] = {
                        "summary": cached["summary"],
                        "claims": cached["claims"],
                        "chain": cached["chain"],
                    }

            st.session_state.chat_history = []
            push_default_outputs_to_chat(default_outputs)
            st.rerun()


# popover 外：狀態
if st.session_state.store is None:
    st.info("尚未建立索引。請點「📦 文件管理（上傳 / OCR / 建索引）」開始。")
else:
    st.success(
        f"已建立索引：檔案數={len(st.session_state.file_rows)} / chunks={len(st.session_state.store.chunks)} / "
        f"KG nodes={st.session_state.kg.g.number_of_nodes()} edges={st.session_state.kg.g.number_of_edges()}"
    )

st.divider()

# Chat 主畫面
st.subheader("Chat（Workflow + 引用 badges）")

for msg in st.session_state.chat_history:
    render_chat_message(msg)

if st.session_state.store is None:
    st.stop()

prompt = st.chat_input("輸入問題：理解含意/為何這樣陳述/傳導機制/重組新報告/列出所有…")
if prompt:
    st.session_state.chat_history.append({"role": "user", "kind": "text", "content": prompt})
    render_chat_message(st.session_state.chat_history[-1])

    with st.chat_message("assistant"):
        with st.status("Workflow：RETRIEVE → GRADE → TRANSFORM → GENERATE → CHECK", expanded=True) as status:
            result = run_chat_workflow_with_ui(
                client=client,
                store=st.session_state.store,
                question=prompt,
                max_query_rewrites=2,
                max_generate_retries=2,
                top_k=10,
            )
            status.update(state="complete", expanded=False)

        st.markdown("## 最終回答")
        if result.get("render_mode") == "bullets":
            render_bullets_with_badges(result["final_answer"], badge_color="blue")
        else:
            render_text_with_badges(result["final_answer"], badge_color="gray")

        with st.expander("查看 debug（query history / logs / context）"):
            st.markdown("### Query history")
            st.code("\n".join([f"{i}. {q}" for i, q in enumerate(result.get("query_history", []))]))
            st.markdown("### Logs")
            st.text("\n".join(result.get("logs", [])))
            st.markdown("### Context（節錄）")
            st.text((result.get("context", "") or "")[:12000])

    # 存回 history（這裡存純文字即可；重畫時也能 badge）
    st.session_state.chat_history.append({"role": "assistant", "kind": "text", "content": result["final_answer"]})
