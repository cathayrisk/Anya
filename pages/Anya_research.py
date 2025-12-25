# app.py
# -*- coding: utf-8 -*-
"""
研究報告助手（FAISS + OpenAI embeddings + LangExtract KG + Chat + Workflow UI）
領域：總經 / 金融 / 財務 / 氣候風險 / 永續金融

你要求的重點：
- UI 不用 tabs
- st.popover 上傳 + 表格顯示（頁數/字數/token/空白頁比例/建議OCR/使用OCR）
- OCR：逐檔勾選（掃描 PDF 會自動建議）
- 向量：FAISS + text-embedding-3-small（固定）
- LLM：gpt-5.2（固定）
- LangExtract：gpt-5.2（固定）
- 預設輸出：摘要/核心主張/推論鏈（每個 bullet 必須引用 [報告 p頁 | chunk_id]）
- Chat：grading（yes/no）+ 自動重試；UI 顯示 RETRIEVE / GRADE / TRANSFORM / GENERATE + 中間產物漂亮呈現
- UI 強化：每一步顯示 ✅/❌ + 耗時（秒）；relevant chunks 有 expander 看全文

環境變數：
- OPENAI_API_KEY 必填

依賴：
streamlit, openai, langextract[openai], pypdf, numpy, faiss-cpu, networkx
OCR 額外：pymupdf
"""

from __future__ import annotations

import os
import re
import io
import uuid
import math
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

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
# 小工具：字數/token 估算、文字清理
# =========================
def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def estimate_tokens_from_chars(n_chars: int) -> int:
    # 粗估：每 token 約 3.6 chars（中英混合折衷）
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


# =========================
# OpenAI helpers
# =========================
def get_client() -> OpenAI:
    return OpenAI()

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
    if getattr(resp, "output_text", None):
        return resp.output_text
    return str(resp)

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
# 檔案讀取 / 掃描偵測 / OCR
# =========================
@dataclass
class FileRow:
    file_id: str
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


def extract_pdf_text_pages(pdf_bytes: bytes) -> List[Tuple[int, str]]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for i, p in enumerate(reader.pages):
        try:
            t = p.extract_text() or ""
        except Exception:
            t = ""
        pages.append((i + 1, norm_space(t)))
    return pages

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
    if avg < 120:
        return True
    return False

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

def ocr_pdf_pages_with_openai(client: OpenAI, pdf_bytes: bytes, dpi: int = 180) -> List[Tuple[int, str]]:
    if not HAS_PYMUPDF:
        raise RuntimeError("未安裝 pymupdf（fitz），無法做 PDF OCR。請 pip install pymupdf")

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    out: List[Tuple[int, str]] = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for i in range(doc.page_count):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        text = norm_space(ocr_image_bytes_with_openai(client, img_bytes))
        out.append((i + 1, text))
    return out


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

    def search(self, qvec: np.ndarray, k: int = 6) -> List[Tuple[float, Chunk]]:
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
# KG（NetworkX MultiDiGraph）：保留 provenance
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

        self.g.add_edge(
            s, o, key=str(uuid.uuid4()),
            relation=r,
            prov=asdict(prov),
            attrs=attrs or {},
        )

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
# LangExtract：只抽 claim / relation
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
            lx.data.Extraction(
                extraction_class="relation",
                extraction_text="keep policy restrictive through mid-2025",
                attributes={"subject": "The Fed", "relation": "TIGHTENS", "object": "policy stance", "time": "mid-2025"},
            ),
        ],
    )
    t2 = (
        "Under the NGFS Net Zero 2050 scenario, transition risk increases for the energy sector, "
        "while physical risk remains elevated in coastal real estate."
    )
    ex2 = lx.data.ExampleData(
        text=t2,
        extractions=[
            lx.data.Extraction(
                extraction_class="relation",
                extraction_text="Under the NGFS Net Zero 2050 scenario",
                attributes={"subject": "Report", "relation": "ASSUMES_SCENARIO", "object": "NGFS Net Zero 2050"},
            ),
            lx.data.Extraction(
                extraction_class="relation",
                extraction_text="transition risk increases for the energy sector",
                attributes={"subject": "transition risk", "relation": "AFFECTS", "object": "energy sector", "polarity": "increase"},
            ),
            lx.data.Extraction(
                extraction_class="relation",
                extraction_text="physical risk remains elevated in coastal real estate",
                attributes={"subject": "physical risk", "relation": "AFFECTS", "object": "coastal real estate", "polarity": "high"},
            ),
        ],
    )
    return [ex1, ex2]

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
# 引用檢查
# =========================
CIT_RE = re.compile(r"\[[^\]]+\|\s*[^\]]+\]")  # [報告 p頁 | chunk_id]
BULLET_RE = re.compile(r"^\s*(?:[-•*]|\d+\.)\s+")

def bullets_all_have_citations(md: str) -> Tuple[bool, List[str]]:
    bad_lines = []
    lines = (md or "").splitlines()
    has_bullet = any(BULLET_RE.match(l) for l in lines)
    for line in lines:
        if BULLET_RE.match(line):
            if not CIT_RE.search(line):
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
        "硬性規則：\n"
        "1) 只能根據我提供的資料回答，不可腦補。\n"
        "2) 你必須輸出「純 bullet 條列」（每行以 - 開頭）。不要輸出段落。\n"
        "3) 每一個 bullet 的句尾都必須附引用，格式固定：[報告 p頁 | chunk_id]\n"
        "4) 若你無法替某 bullet 找到引用，請不要寫那個 bullet，改寫或刪掉。\n"
    )

    last = ""
    for _ in range(max_retries + 1):
        out = gen_text(client, system, user, model=LLM_MODEL)
        ok, _bad = bullets_all_have_citations(out)
        if ok:
            return out
        last = out
        user = user + "\n\n【強制修正】請重新輸出純 bullet，並保證每個 bullet 句尾都有 [報告 p頁 | chunk_id]。"
    return last

def generate_with_paragraph_citation_guard(client: OpenAI, user: str, max_retries: int = 2) -> str:
    system = (
        "你是嚴謹的研究助理。\n"
        "硬性規則：\n"
        "1) 只能根據我提供的 Context 回答，不可腦補。\n"
        "2) 請用 2~4 段回答。\n"
        "3) 每一段至少要有 1 個引用，格式固定：[報告 p頁 | chunk_id]\n"
        "4) 若做不到引用，請刪掉那段並改寫。\n"
    )
    last = ""
    for _ in range(max_retries + 1):
        out = gen_text(client, system, user, model=LLM_MODEL)
        ok, _bad = paragraphs_all_have_citations(out)
        if ok:
            return out
        last = out
        user = user + "\n\n【強制修正】上一版有段落缺引用。請確保每段至少一個 [報告 p頁 | chunk_id]。"
    return last


# =========================
# 建索引：FAISS + KG
# =========================
def build_indices(
    client: OpenAI,
    api_key: str,
    file_rows: List[FileRow],
    file_bytes_map: Dict[str, bytes],
    chunk_size: int = 900,
    overlap: int = 150,
) -> Tuple[FaissStore, KnowledgeGraph, Dict[str, Any]]:
    dim = embed_texts(client, ["dim_probe"]).shape[1]
    store = FaissStore(dim)
    kg = KnowledgeGraph()

    stats = {"reports": 0, "chunks": 0, "kg_nodes": 0, "kg_edges": 0}

    all_chunks: List[Chunk] = []
    all_texts: List[str] = []

    for row in file_rows:
        data = file_bytes_map[row.file_id]
        report_id = row.file_id
        title = os.path.splitext(row.name)[0]
        stats["reports"] += 1

        pages: List[Tuple[Optional[int], str]] = []
        if row.ext == ".pdf":
            if row.use_ocr:
                pages = [(p, t) for p, t in ocr_pdf_pages_with_openai(client, data)]
            else:
                pages = [(p, t) for p, t in extract_pdf_text_pages(data)]
        elif row.ext == ".txt":
            pages = [(None, norm_space(data.decode("utf-8", errors="ignore")))]
        elif row.ext in (".png", ".jpg", ".jpeg"):
            text = norm_space(ocr_image_bytes_with_openai(client, data))
            pages = [(None, text)]
        else:
            pages = [(None, "")]

        # chunks
        for page_no, page_text in pages:
            if not page_text:
                continue
            for i, ch in enumerate(chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)):
                cid = f"{report_id}_p{page_no if page_no else 'na'}_c{i}"
                all_chunks.append(
                    Chunk(
                        chunk_id=cid,
                        report_id=report_id,
                        title=title,
                        page=page_no if isinstance(page_no, int) else None,
                        text=ch,
                    )
                )
                all_texts.append(ch)

        # LangExtract -> KG（page 級）
        for page_no, page_text in pages:
            if not page_text:
                continue
            ann = run_langextract(page_text, api_key=api_key)

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
                    kg.add_edge(s=s, r=r, o=o, prov=prov, attrs=attrs)
                elif cls == "claim":
                    claim_node = f"CLAIM: {norm_space(etext)}"
                    kg.add_edge(s=title, r="MENTIONS", o=claim_node, prov=prov, attrs=attrs)

    # embed chunks
    if all_texts:
        vecs_list = []
        bs = 64
        for i in range(0, len(all_texts), bs):
            vecs_list.append(embed_texts(client, all_texts[i:i+bs]))
        vecs = np.vstack(vecs_list)
        store.add(vecs, all_chunks)

    stats["chunks"] = len(store.chunks)
    stats["kg_nodes"] = kg.g.number_of_nodes()
    stats["kg_edges"] = kg.g.number_of_edges()
    return store, kg, stats


# =========================
# 預設輸出（摘要/核心主張/推論鏈）→ 推送到 Chat
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

def make_default_outputs_for_report(client: OpenAI, all_chunks: List[Chunk], title: str) -> Dict[str, str]:
    reps = pick_chunks_for_report(all_chunks, title, max_n=12)
    ctx = render_chunks_with_ids(reps)

    summary_user = (
        f"請針對報告《{title}》輸出「摘要」：\n"
        f"- 請輸出 8~14 個 bullet\n"
        f"- 每個 bullet 都要包含一個具體資訊點（結論/預測/假設/風險/情境/限制/市場含意）\n"
        f"- 每個 bullet 句尾必須有引用 [報告 p頁 | chunk_id]\n\n"
        f"資料：\n{ctx}"
    )
    claims_user = (
        f"請針對報告《{title}》輸出「核心主張」：\n"
        f"- 請輸出 8~14 個 bullet\n"
        f"- 每個 bullet 是一條可驗證主張（含條件/情境/期間更好）\n"
        f"- 每個 bullet 句尾必須有引用 [報告 p頁 | chunk_id]\n\n"
        f"資料：\n{ctx}"
    )
    chain_user = (
        f"請針對報告《{title}》輸出「推論鏈/傳導機制」：\n"
        f"- 請輸出 6~12 個 bullet\n"
        f"- 格式示例：驅動因子 → 中介變數 → 結論/市場含意 → 風險/不確定性\n"
        f"- 每個 bullet 句尾必須有引用 [報告 p頁 | chunk_id]\n\n"
        f"資料：\n{ctx}"
    )

    summary = generate_with_bullet_citation_guard(client, summary_user, max_retries=2)
    claims = generate_with_bullet_citation_guard(client, claims_user, max_retries=2)
    chain = generate_with_bullet_citation_guard(client, chain_user, max_retries=2)
    return {"summary": summary, "claims": claims, "chain": chain}

def push_default_outputs_to_chat(default_outputs: Dict[str, Dict[str, str]]):
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": "我先把上傳報告的「預設輸出」整理好囉（摘要/核心主張/推論鏈；每個 bullet 都有引用）。你接下來可以直接在下方問問題。",
    })
    for title, out in default_outputs.items():
        md = (
            f"## 預設輸出：{title}\n\n"
            f"### 1) 報告摘要\n{out['summary']}\n\n"
            f"### 2) 核心主張\n{out['claims']}\n\n"
            f"### 3) 推論鏈 / 傳導機制\n{out['chain']}\n"
        )
        st.session_state.chat_history.append({"role": "assistant", "content": md})


# =========================
# Chat Workflow（UI：✅/❌ + 耗時 + 產物漂亮呈現）
# =========================
def want_bullets(question: str) -> bool:
    return bool(re.search(r"(列出|有哪些|所有|清單|彙總|摘要|總結)", question))

def grade_doc_relevance(client: OpenAI, question: str, doc_text: str) -> str:
    system = (
        "你是負責評估文件片段與使用者問題是否相關的評分者。"
        "若片段含有可用來回答問題的關鍵事實或推論線索，回 yes；否則回 no。"
        "不需要嚴格，只要排除明顯不相關。"
    )
    user = f"Question:\n{question}\n\nDocument:\n{doc_text[:2200]}"
    return gen_yesno(client, system, user)

def rewrite_question(client: OpenAI, question: str) -> str:
    system = (
        "你是將使用者問題改寫成更適合向量檢索的查詢語句的專家。"
        "請保留原意，補上可檢索的關鍵字（例如：通膨、利率、殖利率曲線、信用利差、NGFS、transition risk、physical risk、WACI…）。"
        "輸出一行改寫後的問題即可。"
    )
    return gen_text(client, system, question, model=LLM_MODEL).strip()

def grade_hallucination(client: OpenAI, context: str, answer: str) -> str:
    system = (
        "你是負責判斷回答是否有被 Context 支持的評分者。"
        "若回答的關鍵主張都能在 Context 找到支持（包含引用片段），回 yes；若有編造或超出 Context，回 no。"
    )
    user = f"Context:\n{context[:9000]}\n\nAnswer:\n{answer[:4500]}"
    return gen_yesno(client, system, user)

def grade_answer(client: OpenAI, question: str, answer: str) -> str:
    system = (
        "你是負責判斷回答是否真正回應使用者問題的評分者。"
        "若回答有直接回覆問題、且結構清楚，回 yes；否則回 no。"
    )
    user = f"Question:\n{question}\n\nAnswer:\n{answer[:4500]}"
    return gen_yesno(client, system, user)

def build_retrieval_packages(
    client: OpenAI,
    store: FaissStore,
    kg: KnowledgeGraph,
    question: str,
    top_k: int = 10,
) -> Tuple[List[Dict[str, Any]], str]:
    qvec = embed_texts(client, [question])
    hits = store.search(qvec, k=top_k)
    retrieved = [{"chunk": ch, "score": score} for score, ch in hits]

    vec_parts = []
    for score, ch in hits:
        vec_parts.append(f"[{ch.title} p{ch.page if ch.page else '-'} | {ch.chunk_id} | score={score:.3f}]\n{ch.text}")

    kg_parts = []
    starts = kg.find_nodes_in_query(question, max_n=2)
    for s in starts:
        edges = kg.bfs_context(s, max_edges=18)
        for e in edges:
            prov = e.get("prov") or {}
            src = f"{prov.get('title','')} p{prov.get('page') if prov.get('page') else '-'}"
            kg_parts.append(
                f"- {e['u']} --[{e['rel']}]--> {e['v']} 〔來源：{src}〕\n"
                f"  snippet: {str(prov.get('snippet',''))[:180]}"
            )

    parts = []
    if kg_parts:
        parts.append("【KG 線索】\n" + "\n".join(kg_parts[:24]))
    if vec_parts:
        parts.append("【檢索片段】\n" + "\n\n".join(vec_parts))

    context = "\n\n".join(parts) if parts else "（找不到任何相關內容）"
    return retrieved, context

def generate_answer_from_context(client: OpenAI, question: str, context: str) -> str:
    if want_bullets(question):
        user = f"Context:\n{context}\n\nQuestion:\n{question}"
        return generate_with_bullet_citation_guard(client, user, max_retries=1)
    user = (
        "請回答問題，並務必依照：結論 → 依據（引用）→ 推論/解釋（若為意義/為何/機制）。\n\n"
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
    kg: KnowledgeGraph,
    question: str,
    *,
    max_query_rewrites: int = 2,
    max_generate_retries: int = 2,
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    UI 內顯示：
    - Step Summary（✅/❌ + 秒）
    - Query history
    - Retrieved / Graded table
    - Relevant chunks expander（全文）
    - Draft answer
    - Hallucination/Answer grade
    """
    query_history: List[str] = [question]
    logs: List[str] = []
    final_context = ""
    final_answer = ""

    # 即時 step summary
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
        step_summary_ph.dataframe(_step_table(step_state), use_container_width=True, hide_index=True)

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
        # ---------- RETRIEVE ----------
        t0 = time.perf_counter()
        try:
            retrieved, raw_context = build_retrieval_packages(client, store, kg, q, top_k=top_k)
            set_step("RETRIEVE", "✅ OK", time.perf_counter() - t0, note=f"top_k={top_k}, got={len(retrieved)}")
        except Exception as e:
            set_step("RETRIEVE", "❌ FAIL", time.perf_counter() - t0, note=str(e))
            return {
                "final_answer": "檢索階段失敗，請查看 debug。",
                "query_history": query_history,
                "context": "",
                "logs": logs + [f"[ERROR] RETRIEVE: {e}"],
            }

        # Pretty retrieved table
        st.markdown(f"### RETRIEVE（round {rewrite_round}）")
        retrieved_rows = []
        for it in retrieved:
            ch: Chunk = it["chunk"]
            retrieved_rows.append({
                "score": round(float(it["score"]), 4),
                "報告": ch.title,
                "頁": ch.page if ch.page is not None else "-",
                "chunk_id": ch.chunk_id,
                "preview": (ch.text[:140] + "…") if len(ch.text) > 140 else ch.text,
            })
        st.dataframe(retrieved_rows, use_container_width=True, hide_index=True)

        # ---------- GRADE ----------
        t1 = time.perf_counter()
        relevant: List[Dict[str, Any]] = []
        graded_rows = []

        st.markdown("### GRADE（doc relevance yes/no）")
        prog = st.progress(0, text="grading…")

        try:
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
        except Exception as e:
            set_step("GRADE", "❌ FAIL", time.perf_counter() - t1, note=str(e))
            return {
                "final_answer": "文件評分階段失敗，請查看 debug。",
                "query_history": query_history,
                "context": raw_context,
                "logs": logs + [f"[ERROR] GRADE: {e}"],
            }

        st.dataframe(graded_rows, use_container_width=True, hide_index=True)

        # Relevant chunks expander（全文）
        st.markdown("### Relevant Chunks（YES）")
        if not relevant:
            st.info("這一輪沒有找到相關 chunks（全部被判定 no）。")
        else:
            rel_sorted = sorted(relevant, key=lambda x: x["score"], reverse=True)[:top_k]
            rel_rows = []
            for it in rel_sorted:
                ch = it["chunk"]
                rel_rows.append({
                    "score": round(float(it["score"]), 4),
                    "報告": ch.title,
                    "頁": ch.page if ch.page is not None else "-",
                    "chunk_id": ch.chunk_id,
                    "preview": (ch.text[:180] + "…") if len(ch.text) > 180 else ch.text,
                })
            st.dataframe(rel_rows, use_container_width=True, hide_index=True)

            st.markdown("#### 展開看全文")
            for it in rel_sorted:
                ch = it["chunk"]
                with st.expander(f"{ch.title} p{ch.page if ch.page else '-'} | {ch.chunk_id} | score={it['score']:.3f}"):
                    st.text(ch.text)

        # 若沒有 relevant：TRANSFORM
        if not relevant:
            if rewrite_round < max_query_rewrites:
                t2 = time.perf_counter()
                st.markdown("### TRANSFORM（rewrite query）")
                new_q = rewrite_question(client, q)
                query_history.append(new_q)
                query_hist_ph.code("\n".join([f"{i}. {qq}" for i, qq in enumerate(query_history)]))
                set_step("TRANSFORM", "✅ OK", time.perf_counter() - t2, note="rewrite applied")
                q = new_q
                continue
            else:
                set_step("TRANSFORM", "❌ SKIP", None, note="rewrite limit reached")
                final_answer = "資料不足：檢索不到足夠相關內容。你可以換個問法或上傳更多報告。"
                final_context = raw_context
                return {"final_answer": final_answer, "query_history": query_history, "context": final_context, "logs": logs}

        # build context from relevant
        rel_sorted = sorted(relevant, key=lambda x: x["score"], reverse=True)[:min(top_k, len(relevant))]
        vec_parts = []
        for it in rel_sorted:
            ch = it["chunk"]
            vec_parts.append(f"[{ch.title} p{ch.page if ch.page else '-'} | {ch.chunk_id} | score={it['score']:.3f}]\n{ch.text}")

        kg_part = ""
        if "【KG 線索】" in raw_context:
            kg_part = raw_context.split("【檢索片段】")[0].strip()

        context = "\n\n".join([p for p in [kg_part, "【檢索片段】\n" + "\n\n".join(vec_parts)] if p.strip()])
        final_context = context

        # ---------- GENERATE + CHECK loop ----------
        for gen_round in range(max_generate_retries + 1):
            t3 = time.perf_counter()
            st.markdown(f"### GENERATE（round {gen_round}）")
            ans = generate_answer_from_context(client, q, context)
            set_step("GENERATE", "✅ OK", time.perf_counter() - t3, note=f"gen_round={gen_round}")

            st.markdown("#### Draft answer")
            st.markdown(ans)

            t4 = time.perf_counter()
            st.markdown("### CHECK（hallucination / answer）")
            hall = grade_hallucination(client, context, ans)
            good = grade_answer(client, q, ans)
            set_step("CHECK", "✅ OK", time.perf_counter() - t4, note=f"hall={hall}, answer_ok={good}")
            logs.append(f"[CHECK] gen_round={gen_round} hall={hall} answer_ok={good}")

            st.write({"hallucination": hall, "answer_ok": good})

            if hall == "yes" and good == "yes":
                final_answer = ans
                return {"final_answer": final_answer, "query_history": query_history, "context": final_context, "logs": logs}

            # hallucination fail -> regenerate (same query)
            if hall == "no":
                continue

            # answer fail -> break to transform
            if good == "no":
                break

        # 生成不理想 → TRANSFORM
        if rewrite_round < max_query_rewrites:
            t2 = time.perf_counter()
            st.markdown("### TRANSFORM（rewrite query）")
            new_q = rewrite_question(client, q)
            query_history.append(new_q)
            query_hist_ph.code("\n".join([f"{i}. {qq}" for i, qq in enumerate(query_history)]))
            set_step("TRANSFORM", "✅ OK", time.perf_counter() - t2, note="rewrite applied")
            q = new_q
            continue

        set_step("TRANSFORM", "❌ SKIP", None, note="rewrite limit reached")
        final_answer = "資料不足：已多次嘗試仍無法產生可被證據支持且回應問題的答案。建議換問法或增加資料。"
        return {"final_answer": final_answer, "query_history": query_history, "context": final_context, "logs": logs}

    final_answer = "資料不足：工作流未能完成。"
    return {"final_answer": final_answer, "query_history": query_history, "context": final_context, "logs": logs}


# =========================
# Streamlit UI（不使用 tabs）
# =========================
st.set_page_config(page_title="研究報告助手（Workflow UI）", layout="wide")
st.title("研究報告助手（FAISS + LangExtract + Chat + Workflow UI）")

api_key = os.environ.get("OPENAI_API_KEY", "").strip()
if not api_key:
    st.error("請先設定環境變數 OPENAI_API_KEY。")
    st.stop()

client = get_client()

# Session State
if "file_rows" not in st.session_state:
    st.session_state.file_rows: List[FileRow] = []
if "file_bytes" not in st.session_state:
    st.session_state.file_bytes: Dict[str, bytes] = {}

if "store" not in st.session_state:
    st.session_state.store: Optional[FaissStore] = None
if "kg" not in st.session_state:
    st.session_state.kg = KnowledgeGraph()
if "default_outputs" not in st.session_state:
    st.session_state.default_outputs: Dict[str, Dict[str, str]] = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[Dict[str, str]] = []


# ===== 上傳 popover =====
with st.popover("📤 上傳文件", use_container_width=True):
    st.caption("支援 PDF/TXT/PNG/JPG。PDF 若抽到文字偏少會自動建議 OCR（逐檔可勾選）。")
    up = st.file_uploader(
        "選擇檔案",
        type=["pdf", "txt", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
    if up:
        existing_keys = {(r.name, r.bytes_len) for r in st.session_state.file_rows}
        for f in up:
            data = f.read()
            key = (f.name, len(data))
            if key in existing_keys:
                continue

            ext = os.path.splitext(f.name)[1].lower()
            fid = str(uuid.uuid4())[:10]
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

# ===== 檔案表格 + OCR 勾選 =====
st.subheader("已上傳文件")
if not st.session_state.file_rows:
    st.info("還沒有上傳文件。點「📤 上傳文件」開始。")
else:
    table_data = []
    for r in st.session_state.file_rows:
        note = ""
        if r.ext == ".pdf" and r.likely_scanned:
            note = "文字抽取偏少，可能是掃描PDF，建議 OCR"
        elif r.ext in (".png", ".jpg", ".jpeg"):
            note = "圖片檔：一定會 OCR"
        elif r.ext == ".txt":
            note = "文字檔：不需要 OCR"

        table_data.append({
            "file_id": r.file_id,
            "檔名": r.name,
            "格式": r.ext,
            "頁數": r.pages if r.pages is not None else "-",
            "抽到字數(全文)": r.extracted_chars,
            "token估算(粗估)": r.token_est,
            "空白頁/頁數": f"{r.blank_pages}/{r.pages}" if r.blank_pages is not None and r.pages else "-",
            "空白頁比例": f"{r.blank_ratio:.2f}" if r.blank_ratio is not None else "-",
            "建議OCR": r.likely_scanned,
            "使用OCR": r.use_ocr,
            "備註": note,
        })

    disabled_cols = ["file_id", "檔名", "格式", "頁數", "抽到字數(全文)", "token估算(粗估)", "空白頁/頁數", "空白頁比例", "建議OCR", "備註"]
    edited = st.data_editor(
        table_data,
        use_container_width=True,
        hide_index=True,
        disabled=disabled_cols,
        column_config={
            "使用OCR": st.column_config.CheckboxColumn("使用OCR", help="PDF 字數太少時建議勾選 OCR（會更慢且花費較高）"),
        },
    )

    use_ocr_map = {row["file_id"]: bool(row["使用OCR"]) for row in edited}
    for i, r in enumerate(st.session_state.file_rows):
        if r.ext in (".png", ".jpg", ".jpeg"):
            st.session_state.file_rows[i].use_ocr = True
        elif r.ext == ".txt":
            st.session_state.file_rows[i].use_ocr = False
        else:
            st.session_state.file_rows[i].use_ocr = use_ocr_map.get(r.file_id, r.use_ocr)

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        build_btn = st.button("🚀 建立索引 + 預設輸出", type="primary", use_container_width=True)
    with c2:
        clear_btn = st.button("🧹 清空", use_container_width=True)
    with c3:
        st.caption("會建立：FAISS + LangExtract KG + 預設輸出（摘要/核心主張/推論鏈）→ 推送到 Chat。")

    if clear_btn:
        st.session_state.file_rows = []
        st.session_state.file_bytes = {}
        st.session_state.store = None
        st.session_state.kg = KnowledgeGraph()
        st.session_state.default_outputs = {}
        st.session_state.chat_history = []
        st.rerun()

    if build_btn:
        need_ocr = any(r.ext == ".pdf" and r.use_ocr for r in st.session_state.file_rows)
        if need_ocr and not HAS_PYMUPDF:
            st.error("你有勾選 PDF OCR，但環境未安裝 pymupdf。請先 pip install pymupdf，再重試。")
            st.stop()

        with st.status("建索引中（向量 + KG）...", expanded=True) as s1:
            store, kg, stats = build_indices(
                client=client,
                api_key=api_key,
                file_rows=st.session_state.file_rows,
                file_bytes_map=st.session_state.file_bytes,
            )
            st.session_state.store = store
            st.session_state.kg = kg
            s1.update(label=f"完成索引：chunks={stats['chunks']} / KG nodes={stats['kg_nodes']} edges={stats['kg_edges']}", state="complete")

        titles = sorted({c.title for c in st.session_state.store.chunks})
        with st.status("產生預設輸出（摘要/核心主張/推論鏈；每個 bullet 必須引用）...", expanded=True) as s2:
            default_outputs = {}
            for title in titles:
                default_outputs[title] = make_default_outputs_for_report(client, st.session_state.store.chunks, title)
            st.session_state.default_outputs = default_outputs
            s2.update(label="預設輸出完成", state="complete")

        st.session_state.chat_history = []
        push_default_outputs_to_chat(st.session_state.default_outputs)

st.divider()

# ===== Chat 主畫面（唯一）=====
st.subheader("Chat（Workflow：RETRIEVE / GRADE / TRANSFORM / GENERATE；含 ✅/❌ + 耗時 + 展開全文）")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.store is None:
    st.info("請先上傳文件並建立索引。")
else:
    prompt = st.chat_input("輸入問題：理解含意/為何這樣陳述/傳導機制/重組新報告/列出所有…")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.status("Workflow running…", expanded=True) as status:
                result = run_chat_workflow_with_ui(
                    client=client,
                    store=st.session_state.store,
                    kg=st.session_state.kg,
                    question=prompt,
                    max_query_rewrites=2,
                    max_generate_retries=2,
                    top_k=10,
                )
                status.update(label="Workflow done", state="complete", expanded=False)

            st.markdown("## 最終回答")
            st.markdown(result["final_answer"])

            with st.expander("查看 debug（query history / logs / context）"):
                st.markdown("### Query history")
                st.code("\n".join([f"{i}. {q}" for i, q in enumerate(result.get("query_history", []))]))
                st.markdown("### Logs")
                st.text("\n".join(result.get("logs", [])))
                st.markdown("### Context（節錄）")
                st.text((result.get("context", "") or "")[:12000])

        st.session_state.chat_history.append({"role": "assistant", "content": result["final_answer"]})
