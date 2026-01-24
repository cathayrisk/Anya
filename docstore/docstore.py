# docstore.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import io
import os
import re
import math
import uuid
import base64
import hashlib
import tempfile
from dataclasses import dataclass
from typing import Any, Optional, Dict, List, Tuple

import numpy as np
from pypdf import PdfReader
from openai import OpenAI

try:
    import faiss  # type: ignore
except Exception as e:
    raise RuntimeError("docstore.py 需要 faiss（建議安裝 faiss-cpu）。") from e


# =========================
# Optional deps (Splitter / BM25 / FlashRank / Office loaders / PyMuPDF)
# =========================
HAS_SPLITTER = False
RecursiveCharacterTextSplitter = None
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter as _R
    RecursiveCharacterTextSplitter = _R
    HAS_SPLITTER = True
except Exception:
    HAS_SPLITTER = False

HAS_BM25 = False
BM25Retriever = None
LCDocument = None
try:
    from langchain_community.retrievers import BM25Retriever as _BM25
    from langchain_core.documents import Document as _Doc
    BM25Retriever = _BM25
    LCDocument = _Doc
    HAS_BM25 = True
except Exception:
    HAS_BM25 = False

HAS_FLASHRANK = False
FlashrankRerank = None
try:
    from langchain_community.document_compressors import FlashrankRerank as _FR
    FlashrankRerank = _FR
    HAS_FLASHRANK = True
except Exception:
    HAS_FLASHRANK = False

HAS_PYMUPDF = False
fitz = None
try:
    import fitz as _fitz  # pymupdf
    fitz = _fitz
    HAS_PYMUPDF = True
except Exception:
    HAS_PYMUPDF = False

HAS_UNSTRUCTURED_LOADERS = False
UnstructuredWordDocumentLoader = None
UnstructuredPowerPointLoader = None
UnstructuredExcelLoader = None
TextLoader = None
try:
    from langchain_community.document_loaders.word_document import UnstructuredWordDocumentLoader as _UW
    from langchain_community.document_loaders.powerpoint import UnstructuredPowerPointLoader as _UP
    from langchain_community.document_loaders.excel import UnstructuredExcelLoader as _UE
    from langchain_community.document_loaders import TextLoader as _TL
    UnstructuredWordDocumentLoader = _UW
    UnstructuredPowerPointLoader = _UP
    UnstructuredExcelLoader = _UE
    TextLoader = _TL
    HAS_UNSTRUCTURED_LOADERS = True
except Exception:
    HAS_UNSTRUCTURED_LOADERS = False


# =========================
# Config
# =========================
EMBEDDING_MODEL = os.getenv("DOCSTORE_EMBEDDING_MODEL", "text-embedding-3-small")
EMBED_BATCH_SIZE = int(os.getenv("DOCSTORE_EMBED_BATCH_SIZE", "256"))

DEFAULT_CHUNK_SIZE = int(os.getenv("DOCSTORE_CHUNK_SIZE", "900"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DOCSTORE_CHUNK_OVERLAP", "150"))

# OCR (only if user enables it)
OCR_MODEL = os.getenv("DOCSTORE_OCR_MODEL", "gpt-5.2")
OCR_MAX_WORKERS = int(os.getenv("DOCSTORE_OCR_MAX_WORKERS", "2"))
OCR_PDF_DPI = int(os.getenv("DOCSTORE_OCR_PDF_DPI", "180"))


# =========================
# Utils
# =========================
def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def estimate_tokens_from_chars(n_chars: int) -> int:
    if n_chars <= 0:
        return 0
    return max(1, int(math.ceil(n_chars / 3.6)))


def detect_image_mime_by_ext(ext: str) -> str:
    ext = (ext or "").lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    return "application/octet-stream"


def img_bytes_to_data_url(img_bytes: bytes, mime: str) -> str:
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    text = norm_space(text)
    if not text:
        return []
    if HAS_SPLITTER and RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(chunk_size),
            chunk_overlap=int(overlap),
            add_start_index=True,
            separators=[
                "\n\n", "\n",
                "。", "！", "？",
                ".", "!", "?",
                "；", ";",
                "，", ",",
                " ", "",
            ],
        )
        docs = splitter.create_documents([text])
        out = []
        for d in docs:
            t = norm_space(getattr(d, "page_content", "") or "")
            if t:
                out.append(t)
        return out

    # fallback splitter
    out = []
    i = 0
    while i < len(text):
        j = min(len(text), i + int(chunk_size))
        out.append(text[i:j])
        if j == len(text):
            break
        i = max(0, j - int(overlap))
    return out


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


def should_suggest_ocr_pdf(pages: Optional[int], extracted_chars: int, blank_ratio: Optional[float]) -> bool:
    if pages is None or pages <= 0:
        return True
    if blank_ratio is not None and blank_ratio >= 0.6:
        return True
    avg = extracted_chars / max(1, pages)
    return avg < 120


# =========================
# Extractors
# =========================
def extract_pdf_text_pages(pdf_bytes: bytes) -> list[Tuple[int, str]]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    out: list[Tuple[int, str]] = []
    for i, p in enumerate(reader.pages):
        try:
            t = p.extract_text() or ""
        except Exception:
            t = ""
        out.append((i + 1, norm_space(t)))
    return out


def _write_temp_file(data: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        return tmp.name


def extract_office_text_blocks(filename: str, ext: str, data: bytes) -> list[Tuple[Optional[int], str]]:
    """
    回傳 [(block_no, text)]；block_no 當作 pN 的 N（位置序）
    先求穩：mode="single"
    """
    if not HAS_UNSTRUCTURED_LOADERS:
        return []

    tmp_path = _write_temp_file(data, suffix=ext)
    try:
        if ext in (".doc", ".docx"):
            loader = UnstructuredWordDocumentLoader(tmp_path, mode="single")
        elif ext == ".pptx":
            loader = UnstructuredPowerPointLoader(tmp_path, mode="single")
        elif ext in (".xls", ".xlsx"):
            loader = UnstructuredExcelLoader(tmp_path, mode="single")
        elif ext == ".txt":
            loader = TextLoader(tmp_path)
        else:
            return []

        docs = loader.load()
        full = "\n\n".join([(d.page_content or "").strip() for d in (docs or []) if (d.page_content or "").strip()])
        full = norm_space(full)
        if not full:
            return []
        return [(1, full)]
    except Exception:
        return []
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def ocr_image_bytes(client: OpenAI, image_bytes: bytes, mime: str) -> str:
    system_msg = {
        "role": "system",
        "content": [{"type": "input_text", "text": "你是 OCR。只輸出可見文字/表格（表格用 Markdown 表格），不要評論。"}],
    }
    user_msg = {
        "role": "user",
        "content": [
            {"type": "input_text", "text": "請擷取圖片中所有可見文字（含小字/註腳）。"},
            {"type": "input_image", "image_url": img_bytes_to_data_url(image_bytes, mime)},
        ],
    }
    resp = client.responses.create(model=OCR_MODEL, input=[system_msg, user_msg])
    return norm_space(getattr(resp, "output_text", "") or "")


def ocr_pdf_pages_parallel(client: OpenAI, pdf_bytes: bytes, dpi: int = OCR_PDF_DPI) -> list[Tuple[int, str]]:
    if not HAS_PYMUPDF or fitz is None:
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

    from concurrent.futures import ThreadPoolExecutor, as_completed
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
# Store data structures
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
    text_pages: Optional[int]
    text_pages_ratio: Optional[float]
    likely_scanned: bool
    use_ocr: bool


@dataclass
class Chunk:
    chunk_id: str
    title: str
    page: Optional[int]
    text: str


@dataclass
class FullDoc:
    title: str
    ext: str
    pages: list[Tuple[Optional[int], str]]  # (page_no, text)


class DocStore:
    """
    自管混合檢索：
    - dense: faiss.IndexFlatIP（cosine via normalized vec）
    - sparse: BM25Retriever（optional）
    - rerank: FlashrankRerank（optional；只在 difficulty=hard）
    """
    def __init__(self, dim: int):
        self.dim = int(dim)
        self.index = faiss.IndexFlatIP(self.dim)
        self.chunks: list[Chunk] = []
        self.docs: dict[str, FullDoc] = {}  # title -> FullDoc
        self._bm25 = None
        self._flashrank = None

    def _rebuild_bm25(self) -> None:
        if not (HAS_BM25 and BM25Retriever is not None and LCDocument is not None):
            self._bm25 = None
            return
        if not self.chunks:
            self._bm25 = None
            return

        docs = [
            LCDocument(
                page_content=(c.text or ""),
                metadata={"chunk_id": c.chunk_id, "title": c.title, "page": c.page if c.page is not None else "-"},
            )
            for c in self.chunks
        ]
        self._bm25 = BM25Retriever.from_documents(docs, k=24)

    def add_full_doc(self, full_doc: FullDoc) -> None:
        self.docs[full_doc.title] = full_doc

    def add_chunks(self, vecs: np.ndarray, chunks: list[Chunk]) -> None:
        if vecs.size == 0 or not chunks:
            return
        if vecs.shape[1] != self.dim:
            raise ValueError(f"Embedding dim mismatch: got {vecs.shape[1]}, expect {self.dim}")
        self.index.add(vecs.astype(np.float32))
        self.chunks.extend(chunks)
        self._rebuild_bm25()

    def search_dense(self, qvec: np.ndarray, k: int) -> list[Tuple[float, Chunk]]:
        if self.index.ntotal == 0:
            return []
        scores, idx = self.index.search(qvec.astype(np.float32), int(k))
        out: list[Tuple[float, Chunk]] = []
        for s, i in zip(scores[0], idx[0]):
            if i < 0 or i >= len(self.chunks):
                continue
            out.append((float(s), self.chunks[i]))
        return out

    def search_bm25(self, query: str, k: int) -> list[Chunk]:
        if not self._bm25:
            return []
        try:
            self._bm25.k = max(1, int(k))
        except Exception:
            pass
        docs = self._bm25.invoke(query)
        cid_to_chunk = {c.chunk_id: c for c in self.chunks}
        out: list[Chunk] = []
        for d in (docs or []):
            cid = (d.metadata or {}).get("chunk_id")
            if cid and cid in cid_to_chunk:
                out.append(cid_to_chunk[cid])
        return out

    def _rerank_flashrank(self, query: str, candidates: list[Chunk], top_k: int) -> list[Tuple[float, Chunk]]:
        if not (HAS_FLASHRANK and FlashrankRerank is not None and LCDocument is not None):
            return [(0.0, c) for c in candidates[:top_k]]

        try:
            if self._flashrank is None:
                self._flashrank = FlashrankRerank()

            docs = [
                LCDocument(page_content=(c.text or "")[:2400], metadata={"chunk_id": c.chunk_id})
                for c in candidates
            ]
            reranked_docs = self._flashrank.compress_documents(docs, query)

            cid_to_chunk = {c.chunk_id: c for c in candidates}
            out: list[Tuple[float, Chunk]] = []
            for rank, d in enumerate(reranked_docs or []):
                cid = (d.metadata or {}).get("chunk_id")
                if not cid or cid not in cid_to_chunk:
                    continue
                score = (d.metadata or {}).get("relevance_score")
                if isinstance(score, (int, float)):
                    out.append((float(score), cid_to_chunk[cid]))
                else:
                    out.append((float(top_k - rank), cid_to_chunk[cid]))
            return out[:top_k] if out else [(0.0, c) for c in candidates[:top_k]]
        except Exception:
            return [(0.0, c) for c in candidates[:top_k]]

    def search_hybrid(self, query: str, qvec: np.ndarray, k: int = 8, *, difficulty: str = "medium") -> list[Tuple[float, Chunk]]:
        k = max(1, int(k))
        sem_hits = self.search_dense(qvec, k=max(12, k))
        bm_chunks = self.search_bm25(query, k=max(16, k * 2))

        # RRF fusion
        def rrf_scores(rank_lists: list[list[str]], K: int = 60) -> dict[str, float]:
            scores: dict[str, float] = {}
            for rl in rank_lists:
                for rank, cid in enumerate(rl, start=1):
                    scores[cid] = scores.get(cid, 0.0) + 1.0 / (K + rank)
            return scores

        sem_rank = [ch.chunk_id for _, ch in sem_hits]
        bm_rank = [ch.chunk_id for ch in bm_chunks]
        fused = rrf_scores([sem_rank, bm_rank], K=60)

        cid_to_chunk: dict[str, Chunk] = {}
        for _, ch in sem_hits:
            cid_to_chunk[ch.chunk_id] = ch
        for ch in bm_chunks:
            cid_to_chunk.setdefault(ch.chunk_id, ch)

        items = list(cid_to_chunk.items())
        items.sort(key=lambda kv: fused.get(kv[0], 0.0), reverse=True)

        do_rerank = (difficulty or "medium").lower().strip() == "hard"
        if not do_rerank:
            return [(float(fused.get(cid, 0.0)), ch) for cid, ch in items[:k]]

        candidates = [ch for _, ch in items[: max(30, k)]]
        return self._rerank_flashrank(query, candidates, top_k=k)


# =========================
# Embeddings
# =========================
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


def get_or_create_store(client: OpenAI, store: Optional[DocStore]) -> DocStore:
    if store is not None:
        return store
    dim = embed_texts(client, ["dim_probe"]).shape[1]
    return DocStore(dim)


# =========================
# Indexing pipeline (incremental)
# =========================
def build_file_row_from_bytes(*, filename: str, data: bytes) -> FileRow:
    ext = os.path.splitext(filename)[1].lower()
    fid = str(uuid.uuid4())[:10]
    sig = sha1_bytes(data)

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
    likely_scanned = bool(should_suggest_ocr_pdf(pages, extracted_chars, blank_ratio)) if ext == ".pdf" else False

    if ext in (".png", ".jpg", ".jpeg"):
        use_ocr = True
    elif ext == ".pdf":
        use_ocr = bool(likely_scanned)
    else:
        use_ocr = False

    return FileRow(
        file_id=fid,
        file_sig=sig,
        name=filename,
        ext=ext,
        bytes_len=len(data),
        pages=pages,
        extracted_chars=extracted_chars,
        token_est=token_est,
        blank_pages=blank_pages,
        blank_ratio=blank_ratio,
        text_pages=text_pages,
        text_pages_ratio=text_pages_ratio,
        likely_scanned=likely_scanned,
        use_ocr=use_ocr,
    )


def build_indices_incremental(
    client: OpenAI,
    *,
    file_rows: list[FileRow],
    file_bytes_map: Dict[str, bytes],
    store: Optional[DocStore],
    processed_keys: set,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Tuple[DocStore, Dict[str, Any], set]:
    store = get_or_create_store(client, store)

    stats = {"new_reports": 0, "new_chunks": 0, "errors": []}
    new_chunks: list[Chunk] = []
    new_texts: list[str] = []

    for row in file_rows:
        key = (row.file_sig, bool(row.use_ocr))
        if key in processed_keys:
            continue

        data = file_bytes_map.get(row.file_id)
        if not data:
            continue

        title = os.path.splitext(row.name)[0]
        ext = row.ext.lower()

        pages: list[Tuple[Optional[int], str]] = []
        try:
            if ext == ".pdf":
                if row.use_ocr:
                    pdf_pages = ocr_pdf_pages_parallel(client, data)
                    pages = [(pno, txt) for pno, txt in pdf_pages]
                else:
                    pdf_pages = extract_pdf_text_pages(data)
                    pages = [(pno, txt) for pno, txt in pdf_pages]
            elif ext == ".txt":
                pages = [(None, norm_space(data.decode("utf-8", errors="ignore")))]
            elif ext in (".png", ".jpg", ".jpeg"):
                mime = detect_image_mime_by_ext(ext)
                txt = ocr_image_bytes(client, data, mime=mime)
                pages = [(None, txt)]
            elif ext in (".doc", ".docx", ".pptx", ".xls", ".xlsx"):
                pages = extract_office_text_blocks(row.name, ext, data)
            else:
                pages = [(None, "")]
        except Exception as e:
            stats["errors"].append(f"{row.name}: {repr(e)}")
            pages = [(None, "")]

        store.add_full_doc(FullDoc(title=title, ext=ext, pages=pages))

        for page_no, page_text in pages:
            if not page_text:
                continue
            chunks = chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
            for i, ch in enumerate(chunks):
                cid = f"{row.file_id}_p{page_no if page_no else 'na'}_c{i}"
                new_chunks.append(
                    Chunk(
                        chunk_id=cid,
                        title=title,
                        page=page_no if isinstance(page_no, int) else None,
                        text=ch,
                    )
                )
                new_texts.append(ch)

        stats["new_reports"] += 1
        processed_keys.add(key)

    if new_texts:
        vecs_list: list[np.ndarray] = []
        for i in range(0, len(new_texts), EMBED_BATCH_SIZE):
            vecs_list.append(embed_texts(client, new_texts[i:i + EMBED_BATCH_SIZE]))
        vecs = np.vstack(vecs_list) if vecs_list else np.zeros((0, store.dim), dtype=np.float32)
        store.add_chunks(vecs, new_chunks)

    stats["new_chunks"] = len(new_chunks)
    return store, stats, processed_keys


# =========================
# Tool payload builders
# =========================
def doc_list_payload(file_rows: list[FileRow], store: Optional[DocStore]) -> dict:
    by_title: Dict[str, int] = {}
    if store is not None:
        for c in store.chunks:
            by_title[c.title] = by_title.get(c.title, 0) + 1

    items = []
    for r in file_rows:
        title = os.path.splitext(r.name)[0]
        items.append(
            {
                "title": title,
                "ext": r.ext,
                "size_bytes": r.bytes_len,
                "pages": r.pages,
                "token_est": r.token_est,
                "likely_scanned": bool(r.likely_scanned),
                "use_ocr": bool(r.use_ocr),
                "chunks": int(by_title.get(title, 0)),
            }
        )

    return {
        "count": len(items),
        "items": sorted(items, key=lambda x: (x["title"].lower(), x["ext"])),
        "capabilities": {
            "bm25": bool(HAS_BM25),
            "flashrank": bool(HAS_FLASHRANK),
            "unstructured_loaders": bool(HAS_UNSTRUCTURED_LOADERS),
            "pymupdf": bool(HAS_PYMUPDF),
        },
    }


def _minmax_norm(values: list[float]) -> list[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if math.isclose(vmin, vmax):
        return [0.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]

def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def doc_search_payload(client, store, query: str, k: int = 8, difficulty: str = "medium") -> dict[str, Any]:
    """
    回傳 hits 內含：
    - score (final_score): 融合後分數（越大越好）
    - dense_dist: FAISS 距離（L2 越小越好；IP/ cosine 則可能是 -score 或 None，依你的 index）
    - dense_sim: 轉成 similarity（越大越好）
    - bm25_score: BM25 原始分數（越大越好）
    """
    if store is None or getattr(store, "index", None) is None or getattr(store.index, "ntotal", 0) <= 0:
        return {"ok": True, "query": query, "hits": [], "note": "empty_index"}

    query = (query or "").strip()
    if not query:
        return {"ok": True, "query": query, "hits": [], "note": "empty_query"}

    # ===== 1) Dense：取向量 =====
    # 你 docstore 建索引時應該有用 embeddings；這裡假設你用 OpenAI embedding
    # 若你有包成 store.embed_query(...)，就改成那個
    emb = client.embeddings.create(
        model=getattr(store, "embedding_model", "text-embedding-3-large"),
        input=query,
    )
    qvec = emb.data[0].embedding  # list[float]

    # FAISS 需要 float32 array（若你 docstore.py 已處理過，就用你自己的）
    import numpy as np
    q = np.array([qvec], dtype="float32")

    # ===== 2) Dense search =====
    # 注意：IndexFlatL2 -> 回來的是「距離」（越小越好）
    #       IndexFlatIP -> 回來的是「內積」（越大越好）
    D, I = store.index.search(q, max(k * 4, 20))
    dense_ids = I[0].tolist()
    dense_raw = D[0].tolist()

    # 建出 dense candidates: id -> (dist_or_ip, sim)
    dense_by_id: dict[int, dict[str, float]] = {}
    for idx, raw in zip(dense_ids, dense_raw):
        if idx < 0:
            continue

        # 嘗試辨識 metric：L2(距離) vs IP(相似度)
        # faiss index 有些沒有 metric_type，保守用「如果 raw >= 0 且值偏大」仍可能是 L2
        metric = getattr(getattr(store, "faiss_metric", None), "name", None)  # 你若有存可用
        metric_type = getattr(store.index, "metric_type", None)  # 有些 index 有

        dense_dist = None
        dense_sim = None

        if metric_type is not None:
            import faiss
            if metric_type == faiss.METRIC_L2:
                dense_dist = float(raw)
                dense_sim = 1.0 / (1.0 + dense_dist)
            elif metric_type == faiss.METRIC_INNER_PRODUCT:
                dense_sim = float(raw)  # 越大越好
            else:
                # fallback
                dense_dist = float(raw)
                dense_sim = 1.0 / (1.0 + dense_dist)
        else:
            # fallback：把它當 L2 distance（保守）
            dense_dist = float(raw)
            dense_sim = 1.0 / (1.0 + dense_dist)

        dense_by_id[idx] = {"dense_dist": dense_dist, "dense_sim": dense_sim}

    # ===== 3) BM25 =====
    # 你如果用 rank_bm25：通常會有 store.bm25.get_scores(tokens)
    # 這裡假設你有 store.bm25 和 store.tokenize()
    bm25_by_id: dict[int, float] = {}
    if getattr(store, "bm25", None) is not None and getattr(store, "tokenize", None) is not None:
        toks = store.tokenize(query)
        bm25_scores = store.bm25.get_scores(toks)  # numpy array
        # 取 top 4k 做候選
        topn = max(k * 4, 20)
        bm25_top_ids = np.argsort(-bm25_scores)[:topn].tolist()
        for idx in bm25_top_ids:
            bm25_by_id[int(idx)] = float(bm25_scores[idx])

    # ===== 4) union candidates =====
    cand_ids = list({*dense_by_id.keys(), *bm25_by_id.keys()})
    if not cand_ids:
        return {"ok": True, "query": query, "hits": [], "note": "no_candidates"}

    # ===== 5) normalize + fuse =====
    dense_sims = [_safe_float(dense_by_id.get(i, {}).get("dense_sim"), 0.0) or 0.0 for i in cand_ids]
    bm25_scores = [_safe_float(bm25_by_id.get(i), 0.0) or 0.0 for i in cand_ids]

    dense_norm = _minmax_norm(dense_sims)
    bm25_norm = _minmax_norm(bm25_scores)

    alpha = 0.6  # dense 權重；你可調 0.5~0.7
    fused = [alpha * d + (1 - alpha) * b for d, b in zip(dense_norm, bm25_norm)]

    # ===== 6) 排序、取 top-k =====
    order = sorted(range(len(cand_ids)), key=lambda j: fused[j], reverse=True)[:k]

    hits = []
    for j in order:
        idx = cand_ids[j]
        chunk = store.chunks[idx]  # 你 docstore 若不是 list，用你自己的取法

        title = chunk.get("title") or chunk.get("source") or "Document"
        page = chunk.get("page")
        text = chunk.get("text") or chunk.get("content") or chunk.get("page_content") or ""

        snippet = (text[:420] + "…") if len(text) > 420 else text
        citation_token = f"[{title} p{page if page is not None else '-'}]"

        hits.append({
            "title": title,
            "page": page if page is not None else "-",
            "snippet": snippet,
            "citation_token": citation_token,

            # === 分數（debug + UI）===
            "score": float(fused[j]),          # 最終排序分（0..1）
            "final_score": float(fused[j]),
            "dense_sim": _safe_float(dense_by_id.get(idx, {}).get("dense_sim")),
            "dense_dist": _safe_float(dense_by_id.get(idx, {}).get("dense_dist")),
            "bm25_score": _safe_float(bm25_by_id.get(idx)),
        })

    return {"ok": True, "query": query, "hits": hits}


def doc_get_fulltext_payload(
    store: Optional[DocStore],
    title: str,
    *,
    token_budget: int,
    safety_prefix: str = "",
) -> dict:
    """
    取得全文（含位置標記），依 token_budget 估算後截斷。
    注意：token_budget 是「估算用」；主程式會再做 cap，避免爆 128k。
    """
    t = (title or "").strip()
    if not t:
        return {"error": "missing_title"}

    if store is None:
        return {"error": "no_store"}

    doc = store.docs.get(t)
    if not doc:
        return {"error": "not_found", "title": t}

    parts = []
    if safety_prefix:
        parts.append(safety_prefix.strip())

    for page_no, txt in (doc.pages or []):
        if not txt:
            continue
        token = f"[{doc.title} p{page_no if page_no is not None else '-'}]"
        parts.append(token + "\n" + (txt or ""))

    full = "\n\n".join(parts).strip()
    if not full:
        return {"title": doc.title, "truncated": False, "text": ""}

    max_tokens = max(1, int(token_budget))
    max_chars = int(max_tokens * 3.6 * 0.92)  # 保守一點
    truncated = False
    if len(full) > max_chars:
        full = full[:max_chars] + "\n\n[內容已截斷]"
        truncated = True

    return {
        "title": doc.title,
        "token_budget": max_tokens,
        "estimated_tokens": estimate_tokens_from_chars(len(full)),
        "truncated": truncated,
        "text": full,
    }
