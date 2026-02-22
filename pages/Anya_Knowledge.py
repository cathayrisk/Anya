# pages/78_Anya_Knowledge.py
# -*- coding: utf-8 -*-
"""
安妮亞知識庫 — 文件上傳與向量化儲存

功能：
  1. 上傳多種格式文件（PDF、DOCX、PPTX、XLSX、TXT、PNG、JPG 等）
  2. 自動萃取文字（PDF/圖片含 OCR）並切割 chunk
  3. 批次向量化後持久儲存至 Supabase knowledge_chunks 表
  4. 支援命名知識空間，供其他頁面（如 77_Anya_Test.py）取用

Supabase 需在 SQL Editor 執行：
─────────────────────────────────────────
  create extension if not exists vector;

  create table if not exists knowledge_chunks (
    id          bigserial primary key,
    namespace   text not null,
    filename    text,
    chunk_index int,
    content     text not null,
    embedding   vector(1536),
    created_at  timestamptz default now()
  );

  create index if not exists knowledge_chunks_ns_idx
    on knowledge_chunks (namespace);

  create index if not exists knowledge_chunks_emb_idx
    on knowledge_chunks using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);

  create or replace function match_knowledge_chunks(
    query_embedding  vector(1536),
    match_threshold  float,
    match_count      int,
    namespace_filter text
  )
  returns table (
    id bigint, namespace text, filename text,
    chunk_index int, content text, similarity float
  )
  language sql stable as $$
    select id, namespace, filename, chunk_index, content,
           1 - (embedding <=> query_embedding) as similarity
    from knowledge_chunks
    where namespace = namespace_filter
      and 1 - (embedding <=> query_embedding) > match_threshold
    order by embedding <=> query_embedding
    limit match_count;
  $$;
─────────────────────────────────────────
"""

# ── 標準函式庫（不會失敗）──────────────────────────────────────────────────────
import base64
import io
import os
import re
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ── Streamlit 一定要是第一個被 import 的外部套件 ───────────────────────────────
import streamlit as st

# ─── 頁面設定（必須是第一個 Streamlit 呼叫）───────────────────────────────────
st.set_page_config(
    page_title="安妮亞知識庫",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── 標題（立即渲染，確保頁面不空白）─────────────────────────────────────────
st.title("📚 安妮亞知識庫")
st.caption(
    "支援 PDF · DOCX · PPTX · XLSX · TXT · PNG · JPG — "
    "PDF / 圖片含 OCR · 向量化後持久儲存 · 供 Anya 問答取用"
)

# ─── 第三方套件（任何一個失敗都顯示錯誤，而非空白頁）────────────────────────
try:
    import pandas as pd
    from openai import OpenAI
    from pypdf import PdfReader
    from supabase import create_client, Client
    from langchain_openai import OpenAIEmbeddings
except ImportError as _import_err:
    st.error(
        f"**缺少必要套件，頁面無法載入。**\n\n"
        f"錯誤：`{_import_err}`\n\n"
        "請確認 `requirements.txt` 已安裝並重啟 Streamlit。"
    )
    st.stop()

# ── Optional deps ──────────────────────────────────────────────────────────────
HAS_PYMUPDF = False
_fitz = None
try:
    import fitz as _fitz_mod  # type: ignore
    _fitz = _fitz_mod
    HAS_PYMUPDF = True
except Exception:
    pass

HAS_SPLITTER = False
_RecursiveTextSplitter = None
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter as _R
    _RecursiveTextSplitter = _R
    HAS_SPLITTER = True
except Exception:
    pass

HAS_UNSTRUCTURED_LOADERS = False
_UWordLoader = _UPPTLoader = _UExcelLoader = None
try:
    from langchain_community.document_loaders.word_document import UnstructuredWordDocumentLoader as _UW
    from langchain_community.document_loaders.powerpoint import UnstructuredPowerPointLoader as _UP
    from langchain_community.document_loaders.excel import UnstructuredExcelLoader as _UE
    _UWordLoader, _UPPTLoader, _UExcelLoader = _UW, _UP, _UE
    HAS_UNSTRUCTURED_LOADERS = True
except Exception:
    pass

EMBED_BATCH_SIZE = 256
_CHUNK_SIZE = 900
_CHUNK_OVERLAP = 150

# ─── API Keys & Clients ────────────────────────────────────────────────────────
try:
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_KEY")
    if not OPENAI_API_KEY:
        st.error("找不到 OpenAI API Key，請在 .streamlit/secrets.toml 設定 OPENAI_API_KEY 或 OPENAI_KEY。")
        st.stop()
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    embeddings_model = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model="text-embedding-3-small",
    )
except Exception as _init_err:
    st.error(f"初始化失敗：{_init_err}")
    st.stop()

SUPPORTED_TYPES = ["pdf", "docx", "doc", "pptx", "xlsx", "xls", "txt", "png", "jpg", "jpeg"]

# ─── Session State ─────────────────────────────────────────────────────────────
_defaults: dict = {
    "kg_namespace": "",
    "kg_processed_files": set(),      # file_key = "filename::namespace"
    "kg_file_namespaces": {},         # {filename: namespace}
    "kg_file_ocr": {},                # {filename: bool} PDF OCR 偏好
    "kg_file_tags": {},               # {filename: tag} 分類標籤
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def get_namespace() -> str:
    return st.session_state.kg_namespace.strip()


# ── 文字工具 ───────────────────────────────────────────────────────────────────

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def detect_image_mime_by_ext(ext: str) -> str:
    ext = (ext or "").lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    return "application/octet-stream"


def chunk_text(text: str) -> List[str]:
    text = norm_space(text)
    if not text:
        return []
    if HAS_SPLITTER and _RecursiveTextSplitter is not None:
        splitter = _RecursiveTextSplitter(
            chunk_size=_CHUNK_SIZE,
            chunk_overlap=_CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", "；", ";", "，", ",", " ", ""],
        )
        docs = splitter.create_documents([text])
        return [norm_space(d.page_content) for d in docs if norm_space(d.page_content)]
    out, i = [], 0
    while i < len(text):
        j = min(len(text), i + _CHUNK_SIZE)
        out.append(text[i:j])
        if j == len(text):
            break
        i = max(0, j - _CHUNK_OVERLAP)
    return out


def extract_pdf_text_pages(pdf_bytes: bytes) -> List[Tuple[int, str]]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    out: List[Tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        out.append((i + 1, norm_space(t)))
    return out


def analyze_pdf_text_quality(pdf_pages: List[Tuple[int, str]]) -> Tuple[int, int, float, int, float]:
    if not pdf_pages:
        return 0, 0, 1.0, 0, 0.0
    lens = [len(t) for _, t in pdf_pages]
    blank = sum(1 for L in lens if L <= 40)
    total = max(1, len(lens))
    blank_ratio = blank / total
    text_pages = total - blank
    return sum(lens), blank, blank_ratio, text_pages, text_pages / total


def should_suggest_ocr_pdf(pages: int, extracted_chars: int, blank_ratio: float) -> bool:
    if pages <= 0:
        return True
    if blank_ratio >= 0.6:
        return True
    return (extracted_chars / max(1, pages)) < 120


def extract_office_text_blocks(filename: str, ext: str, data: bytes) -> List[Tuple[Optional[int], str]]:
    if not HAS_UNSTRUCTURED_LOADERS:
        return []
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        if ext in (".doc", ".docx"):
            loader = _UWordLoader(tmp_path, mode="single")
        elif ext == ".pptx":
            loader = _UPPTLoader(tmp_path, mode="single")
        elif ext in (".xls", ".xlsx"):
            loader = _UExcelLoader(tmp_path, mode="single")
        else:
            return []
        docs = loader.load()
        full = norm_space("\n\n".join(
            (d.page_content or "").strip() for d in (docs or []) if (d.page_content or "").strip()
        ))
        return [(1, full)] if full else []
    except Exception:
        return []
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ─── OCR 函數 ─────────────────────────────────────────────────────────────────

OCR_MODEL = "gpt-4o-mini"


def ocr_image_gpt4o(image_bytes: bytes, mime: str) -> str:
    b64 = base64.b64encode(image_bytes).decode()
    try:
        resp = openai_client.chat.completions.create(
            model=OCR_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "請擷取圖片中所有可見文字（含小字、表格、註腳）。"
                            "表格用 Markdown 格式輸出。只輸出文字，不要評論或解釋。"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                ],
            }],
            max_tokens=4096,
            temperature=0,
        )
        return norm_space(resp.choices[0].message.content or "")
    except Exception as e:
        st.toast(f"OCR 失敗：{e}", icon="⚠️")
        return ""


def ocr_pdf_gpt4o(pdf_bytes: bytes) -> List[Tuple[int, str]]:
    if not HAS_PYMUPDF or _fitz is None:
        st.warning("⚠️ 未安裝 pymupdf，無法對 PDF 做 OCR。請 `pip install pymupdf`")
        return []
    doc = _fitz.open(stream=pdf_bytes, filetype="pdf")
    mat = _fitz.Matrix(180 / 72, 180 / 72)
    results: List[Tuple[int, str]] = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        text = ocr_image_gpt4o(img_bytes, "image/png")
        results.append((i + 1, text))
    return results


# ─── 文字萃取分派 ─────────────────────────────────────────────────────────────

def extract_text_from_file(
    file_bytes: bytes,
    filename: str,
    use_ocr: bool = False,
) -> Tuple[List[Tuple[Optional[int], str]], bool]:
    ext = os.path.splitext(filename)[1].lower()
    ocr_used = False

    if ext == ".pdf":
        if use_ocr:
            pages = ocr_pdf_gpt4o(file_bytes)
            ocr_used = True
        else:
            raw = extract_pdf_text_pages(file_bytes)
            pages = [(pno, txt) for pno, txt in raw]
            total_chars, _, blank_ratio, _, _ = analyze_pdf_text_quality(raw)
            if should_suggest_ocr_pdf(len(raw), total_chars, blank_ratio):
                st.info(
                    f"💡 **{filename}** 偵測為掃描版 PDF（擷取字元少），"
                    "建議勾選右上角「強制 PDF OCR」後重新處理。"
                )

    elif ext in (".png", ".jpg", ".jpeg"):
        mime = detect_image_mime_by_ext(ext)
        text = ocr_image_gpt4o(file_bytes, mime)
        pages = [(None, text)]
        ocr_used = True

    elif ext in (".doc", ".docx", ".pptx", ".xls", ".xlsx"):
        if HAS_UNSTRUCTURED_LOADERS:
            blocks = extract_office_text_blocks(filename, ext, file_bytes)
            pages = [(blk[0], blk[1]) for blk in blocks]
            total_chars = sum(len(t) for _, t in pages)
            if total_chars < 100:
                st.warning(
                    f"⚠️ **{filename}** 萃取文字過少（{total_chars} 字元）。"
                    "若文件以圖片為主，建議將圖片另存為 PNG/JPG 再上傳透過 OCR 擷取。"
                )
        else:
            st.warning(
                f"⚠️ 未安裝 unstructured，無法處理 **{filename}**。"
                "請 `pip install 'unstructured[all-docs]'`"
            )
            pages = []

    elif ext == ".txt":
        text = norm_space(file_bytes.decode("utf-8", errors="replace"))
        pages = [(None, text)]

    else:
        st.warning(f"⚠️ 不支援的格式：{ext}")
        pages = []

    return pages, ocr_used


# ─── Supabase 操作 ────────────────────────────────────────────────────────────

def batch_embed_and_save(
    chunks: List[str],
    filename: str,
    namespace: str,
    chunk_offset: int = 0,
    tag: str = "未分類",
) -> int:
    if not chunks:
        return 0
    saved = 0
    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i : i + EMBED_BATCH_SIZE]
        try:
            embs = embeddings_model.embed_documents(batch)
        except Exception as e:
            st.toast(f"Embedding 失敗（批次 {i}）：{e}", icon="⚠️")
            continue
        rows = [
            {
                "namespace": namespace,
                "filename": filename,
                "chunk_index": chunk_offset + i + j,
                "content": text,
                "embedding": emb,
                "tag": tag,
                "created_at": datetime.now().isoformat(),
            }
            for j, (text, emb) in enumerate(zip(batch, embs))
        ]
        try:
            supabase.table("knowledge_chunks").insert(rows).execute()
            saved += len(rows)
        except Exception as e:
            st.toast(f"Supabase 儲存失敗：{e}", icon="⚠️")
    return saved


def load_namespace_summary(namespace: str) -> List[dict]:
    try:
        data = (
            supabase.table("knowledge_chunks")
            .select("filename")
            .eq("namespace", namespace)
            .execute()
            .data
        )
        counts: dict[str, int] = {}
        for row in data or []:
            fn = row.get("filename") or "unknown"
            counts[fn] = counts.get(fn, 0) + 1
        return [{"filename": fn, "chunks": cnt} for fn, cnt in sorted(counts.items())]
    except Exception:
        return []


def load_all_namespace_summary() -> "pd.DataFrame":
    """撈所有 namespace 的彙總（namespace、tag、檔案數、chunk 數、上傳時間）。
    只 select namespace + filename + tag + created_at，避免傳輸 embedding 大欄位。
    加 limit=10000 避免 Supabase 免費版單次回傳列數限制造成誤差。
    """
    empty = pd.DataFrame(columns=["namespace", "tag", "檔案數", "chunk 數", "上傳時間"])
    try:
        data = (
            supabase.table("knowledge_chunks")
            .select("namespace, filename, tag, created_at")
            .limit(10000)
            .execute()
            .data
        )
    except Exception:
        return empty
    if not data:
        return empty
    df = pd.DataFrame(data)
    df["filename"] = df["filename"].fillna("unknown")
    df["tag"] = df["tag"].fillna("未分類")
    df["created_at"] = pd.to_datetime(df.get("created_at"), errors="coerce", utc=True)
    tag_per_ns = df.groupby("namespace")["tag"].first().reset_index()
    summary = (
        df.groupby("namespace", as_index=False)
        .agg(
            **{
                "檔案數": ("filename", "nunique"),
                "chunk 數": ("filename", "size"),
                "上傳時間": ("created_at", "max"),
            }
        )
        .sort_values("namespace")
        .reset_index(drop=True)
    )
    summary = summary.merge(tag_per_ns, on="namespace", how="left")
    return summary[["namespace", "tag", "檔案數", "chunk 數", "上傳時間"]]


def load_all_files_map() -> "Dict[str, List[dict]]":
    """一次撈出所有 namespace 的檔案列表（避免 N+1 查詢）。
    回傳：{namespace: [{filename, chunks}]}
    """
    try:
        data = (
            supabase.table("knowledge_chunks")
            .select("namespace, filename")
            .limit(10000)
            .execute()
            .data
        )
    except Exception:
        return {}
    if not data:
        return {}
    df = pd.DataFrame(data)
    df["filename"] = df["filename"].fillna("unknown")
    result: Dict[str, List[dict]] = {}
    for ns, grp in df.groupby("namespace"):
        counts = grp["filename"].value_counts()
        result[str(ns)] = [
            {"filename": str(fn), "chunks": int(cnt)}
            for fn, cnt in counts.items()
        ]
    return result


def delete_file_chunks(filename: str, namespace: str) -> None:
    try:
        supabase.table("knowledge_chunks").delete() \
            .eq("namespace", namespace) \
            .eq("filename", filename) \
            .execute()
    except Exception as e:
        st.toast(f"刪除失敗：{e}", icon="⚠️")


def delete_namespace_chunks(namespace: str) -> None:
    """刪除整個知識空間的所有資料。"""
    try:
        supabase.table("knowledge_chunks").delete().eq("namespace", namespace).execute()
    except Exception as e:
        st.toast(f"刪除失敗：{e}", icon="⚠️")


def update_namespace_tag(namespace: str, tag: str) -> None:
    """更新某個知識空間所有 chunk 的 tag。"""
    try:
        supabase.table("knowledge_chunks").update({"tag": tag}).eq("namespace", namespace).execute()
    except Exception as e:
        st.toast(f"更新標籤失敗：{e}", icon="⚠️")


# ─── 主要 UI ──────────────────────────────────────────────────────────────────

# 先撈彙總資料（三個 tab 都會用到）
summary_df = load_all_namespace_summary()
available_ns = (
    list(summary_df["namespace"].unique())
    if not summary_df.empty and "namespace" in summary_df.columns
    else []
)

tab_upload, tab_search, tab_manage = st.tabs(["📤 上傳", "🔍 搜尋測試", "📚 管理"])


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1：上傳
# ═══════════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.caption(":small[:gray[拖曳或點選，可多選。OCR 勾選只對 PDF 有效；Namespace 可逐檔修改。]]")
    uploaded = st.file_uploader(
        "上傳文件",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        file_namespace_map: Dict[str, str] = st.session_state.kg_file_namespaces
        file_ocr_map: Dict[str, bool] = st.session_state.kg_file_ocr
        file_tag_map: Dict[str, str] = st.session_state.kg_file_tags

        # 新檔案設預設值
        for f in uploaded:
            stem = os.path.splitext(f.name)[0]
            ext = os.path.splitext(f.name)[1].lower()
            if f.name not in file_namespace_map:
                file_namespace_map[f.name] = stem
            if f.name not in file_ocr_map:
                file_ocr_map[f.name] = ext in (".png", ".jpg", ".jpeg")
            if f.name not in file_tag_map:
                file_tag_map[f.name] = "未分類"

        # 建 DataFrame
        rows_data = []
        for f in uploaded:
            ext = os.path.splitext(f.name)[1].lower()
            ns = file_namespace_map.get(f.name, os.path.splitext(f.name)[0])
            file_key = f"{f.name}::{ns}"
            rows_data.append({
                "OCR": file_ocr_map.get(f.name, False),
                "檔名": f.name,
                "類型": ext.lstrip(".").upper(),
                "大小(KB)": round(f.size / 1024, 1),
                "Namespace": ns,
                "標籤": file_tag_map.get(f.name, "未分類"),
                "狀態": "✅ 已存入" if file_key in st.session_state.kg_processed_files else "⏳ 待處理",
            })

        edited = st.data_editor(
            pd.DataFrame(rows_data),
            hide_index=True,
            use_container_width=True,
            key="kg_file_editor",
            column_config={
                "OCR": st.column_config.CheckboxColumn(
                    "OCR",
                    help="PDF：勾選後用 GPT-4o Vision OCR（適合掃描版，費用較高）\n圖片：永遠 OCR\n其他格式：不適用",
                    width="small",
                ),
                "檔名": st.column_config.TextColumn("檔名", disabled=True, width="large"),
                "類型": st.column_config.TextColumn("類型", disabled=True, width="small"),
                "大小(KB)": st.column_config.NumberColumn("大小(KB)", disabled=True, format="%.1f", width="small"),
                "Namespace": st.column_config.TextColumn("Namespace（可編輯）", width="medium"),
                "標籤": st.column_config.TextColumn("分類標籤（可編輯）", width="medium",
                                                    help="用來在管理頁將知識庫分組，例如：房地產、財報、研究報告"),
                "狀態": st.column_config.TextColumn("狀態", disabled=True, width="small"),
            },
            disabled=["檔名", "類型", "大小(KB)", "狀態"],
        )

        # 回寫 session state
        for _, row in edited.iterrows():
            fname = row["檔名"]
            ext = os.path.splitext(fname)[1].lower()
            new_ns = (str(row["Namespace"]) or "").strip() or os.path.splitext(fname)[0]
            file_namespace_map[fname] = new_ns
            file_tag_map[fname] = (str(row.get("標籤", "")) or "").strip() or "未分類"
            if ext in (".png", ".jpg", ".jpeg"):
                file_ocr_map[fname] = True
            elif ext == ".pdf":
                file_ocr_map[fname] = bool(row["OCR"])
            else:
                file_ocr_map[fname] = False

        # 整理待處理 vs 已處理
        file_entries = []
        for _, row in edited.iterrows():
            fname = row["檔名"]
            ns_row = (str(row["Namespace"]) or "").strip() or os.path.splitext(fname)[0]
            tag_row = (str(row.get("標籤", "")) or "").strip() or "未分類"
            use_ocr = file_ocr_map.get(fname, False)
            file_key = f"{fname}::{ns_row}"
            f_obj = next((f for f in uploaded if f.name == fname), None)
            if f_obj:
                file_entries.append((f_obj, ns_row, use_ocr, tag_row, file_key))

        new_entries = [(f, ns, use_ocr, tag, key) for f, ns, use_ocr, tag, key in file_entries
                       if key not in st.session_state.kg_processed_files]

        if new_entries:
            if st.button(f"🚀 建立知識庫（{len(new_entries)} 個待處理）", type="primary"):
                total_saved = 0
                for f, ns, use_ocr, tag, key in new_entries:
                    file_bytes = f.getvalue()

                    with st.status(f"處理 **{f.name}**（namespace：{ns}，標籤：{tag}）...", expanded=True) as status:
                        st.write("🔎 萃取文字中...")
                        pages, ocr_used = extract_text_from_file(file_bytes, f.name, use_ocr=use_ocr)
                        total_chars = sum(len(t) for _, t in pages if t)

                        if total_chars == 0:
                            status.update(label=f"⚠️ {f.name}：無法萃取文字", state="error")
                            continue

                        ocr_label = "（GPT-4o OCR）" if ocr_used else ""
                        st.write(f"✅ {len(pages)} 頁 · {total_chars:,} 字元 {ocr_label}")

                        # 萃取文字預覽
                        preview_text = " ".join(t for _, t in pages if t)[:300]
                        with st.expander("📄 文字預覽（前 300 字）"):
                            st.text(preview_text + ("…" if len(preview_text) >= 300 else ""))

                        all_chunks: List[str] = []
                        for _, page_text in pages:
                            if page_text:
                                all_chunks.extend(chunk_text(page_text))
                        st.write(f"✂️ 切割為 {len(all_chunks)} 個段落")

                        st.write(f"💾 向量化並存入知識空間「{ns}」（標籤：{tag}）...")
                        saved = batch_embed_and_save(all_chunks, f.name, ns, tag=tag)
                        total_saved += saved
                        st.session_state.kg_processed_files.add(key)

                        status.update(
                            label=f"✅ {f.name}：{saved} 個段落已存入 [{ns}]",
                            state="complete",
                        )

                st.toast(f"完成！共存入 {total_saved} 個段落", icon="🎉")
                st.rerun()
        else:
            st.caption("✅ 所有已上傳檔案均已處理完成。")


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2：搜尋測試
# ═══════════════════════════════════════════════════════════════════════════════
with tab_search:
    st.caption(":small[:gray[輸入問題，驗證知識庫會回傳哪些段落——確認知識品質後再連接對話頁面。]]")

    if not available_ns:
        st.info("尚無知識空間資料，請先在「📤 上傳」頁上傳文件。", icon="💡")
    else:
        s_col1, s_col2 = st.columns([2, 1])
        with s_col1:
            search_ns = st.selectbox("知識空間", options=available_ns, key="search_ns")
        with s_col2:
            search_top_k = st.slider("回傳數量", min_value=3, max_value=10, value=5, key="search_top_k")

        search_query = st.text_input(
            "輸入問題或關鍵字",
            placeholder="例：什麼是量化寬鬆政策？",
            key="search_query",
        )
        search_threshold = st.slider(
            "相似度門檻",
            min_value=0.30,
            max_value=0.95,
            value=0.50,
            step=0.05,
            help="數值越高代表只回傳高相似度的段落；若找不到結果，可嘗試降低門檻。",
            key="search_threshold",
        )

        if st.button("🔍 搜尋", type="primary", key="search_btn") and search_query.strip():
            with st.spinner("向量化查詢並搜尋..."):
                try:
                    qvec = embeddings_model.embed_query(search_query.strip())
                    result = supabase.rpc(
                        "match_knowledge_chunks",
                        {
                            "query_embedding": qvec,
                            "match_threshold": float(search_threshold),
                            "match_count": int(search_top_k),
                            "namespace_filter": search_ns,
                        },
                    ).execute()
                    hits = result.data or []
                except Exception as search_err:
                    st.error(f"搜尋失敗：{search_err}")
                    hits = []

            if hits:
                st.success(f"找到 {len(hits)} 個相關段落", icon="✅")
                for i, hit in enumerate(hits, 1):
                    sim = hit.get("similarity", 0)
                    fname = hit.get("filename") or "未知檔案"
                    chunk_idx = hit.get("chunk_index", "?")
                    content = hit.get("content") or ""
                    ext = os.path.splitext(fname)[1].lower()
                    icon = (
                        "🖼️" if ext in (".png", ".jpg", ".jpeg")
                        else "📄" if ext == ".pdf"
                        else "📝"
                    )
                    label = f"#{i}  {icon} {fname}（段落 {chunk_idx}）— 相似度 {sim:.3f}"
                    with st.expander(label, expanded=(i == 1)):
                        st.markdown(content)
            else:
                st.warning(
                    f"在知識空間「{search_ns}」中找不到相似度 ≥ {search_threshold:.2f} 的段落。\n\n"
                    "建議：\n- 降低相似度門檻\n- 換個問法\n- 確認文件已正確存入",
                    icon="🔍",
                )


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 3：管理（卡片式 + 依標籤分組）
# ═══════════════════════════════════════════════════════════════════════════════
with tab_manage:
    if summary_df.empty:
        st.info("目前沒有任何知識空間資料，請先在「📤 上傳」頁上傳文件。", icon="💡")
    else:
        # ── 頂部總覽 metrics ──────────────────────────────────────────────
        total_ns_count = len(summary_df)
        total_files_count = int(summary_df["檔案數"].sum())
        total_chunks_count = int(summary_df["chunk 數"].sum())
        all_tags = sorted(summary_df["tag"].fillna("未分類").unique().tolist())

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("📚 知識空間", total_ns_count)
        mc2.metric("🏷️ 分類數", len(all_tags))
        mc3.metric("📄 文件總數", total_files_count)
        mc4.metric("🧩 段落總數", f"{total_chunks_count:,}")

        hc1, hc2, hc3 = st.columns([3, 2, 1])
        with hc1:
            # 標籤篩選（pills 風格用 radio horizontal）
            filter_opts = ["全部"] + all_tags
            selected_filter = st.radio(
                "篩選分類",
                filter_opts,
                horizontal=True,
                label_visibility="collapsed",
                key="manage_tag_filter",
            )
        with hc2:
            sort_mode = st.radio(
                "排序",
                ["最新優先", "名稱 A-Z"],
                horizontal=True,
                label_visibility="collapsed",
                key="manage_sort_mode",
            )
        with hc3:
            if st.button("🔄 重新整理", use_container_width=True):
                st.rerun()

        st.divider()

        # ── 一次撈出所有 namespace 的檔案列表（避免 N+1 查詢）────────────
        all_files_map = load_all_files_map()

        # ── 決定要顯示哪些 tag groups ─────────────────────────────────────
        if selected_filter == "全部":
            tags_to_show = all_tags
            filtered_df = summary_df
        else:
            tags_to_show = [selected_filter]
            filtered_df = summary_df[summary_df["tag"] == selected_filter]

        # ── 逐 tag 分組顯示手風琴列表 ─────────────────────────────────────
        for tag_group in tags_to_show:
            group_rows = filtered_df[filtered_df["tag"] == tag_group]
            if group_rows.empty:
                continue

            st.markdown(f"**🏷️ {tag_group}**　:small[:gray[（{len(group_rows)} 個知識庫）]]")
            st.divider()

            # 組內排序
            if sort_mode == "最新優先":
                group_rows = group_rows.sort_values("上傳時間", ascending=False)
            else:
                group_rows = group_rows.sort_values("namespace")

            for ns_info in group_rows.to_dict("records"):
                ns_name = ns_info["namespace"]
                ns_tag = str(ns_info.get("tag") or "未分類")
                ns_chunks = int(ns_info["chunk 數"])
                ns_file_count = int(ns_info["檔案數"])
                ns_file_list = all_files_map.get(ns_name, [])

                ts_raw = ns_info.get("上傳時間")
                try:
                    ts_str = pd.Timestamp(ts_raw).tz_convert("Asia/Taipei").strftime("%Y-%m-%d %H:%M") if pd.notna(ts_raw) else "—"
                except Exception:
                    ts_str = "—"

                exp_label = (
                    f"📖 {ns_name}"
                    f"　:small[:gray[{ns_file_count} 個檔案 · {ns_chunks:,} 段落 · {ts_str}]]"
                )
                with st.expander(exp_label, expanded=False):
                    # ── 緊湊檔案列表 ──
                    for frow in ns_file_list:
                        fext = os.path.splitext(frow["filename"])[1].lower()
                        ficon = (
                            "🖼️" if fext in (".png", ".jpg", ".jpeg")
                            else "📄" if fext == ".pdf"
                            else "📝"
                        )
                        st.caption(f"{ficon} {frow['filename']}　·　{frow['chunks']} 段落")

                    st.markdown("")

                    # ── 標籤編輯 + 操作按鈕 ──
                    col_inp, col_save, col_srch, col_del = st.columns([4, 2, 1, 1])
                    with col_inp:
                        new_tag_val = st.text_input(
                            "標籤",
                            value=ns_tag,
                            key=f"tag_inp_{ns_name}",
                            placeholder="輸入分類標籤...",
                            label_visibility="collapsed",
                        )
                    with col_save:
                        if st.button(
                            "💾 更新標籤",
                            key=f"tag_save_{ns_name}",
                            use_container_width=True,
                        ):
                            t = new_tag_val.strip() or "未分類"
                            update_namespace_tag(ns_name, t)
                            st.toast(f"「{ns_name}」標籤已更新為「{t}」", icon="🏷️")
                            st.rerun()
                    with col_srch:
                        if st.button(
                            "🔍",
                            key=f"srch_ns_{ns_name}",
                            help="切換到搜尋測試並預選此知識空間",
                            use_container_width=True,
                        ):
                            st.session_state["search_ns"] = ns_name
                            st.toast(f"請切換到「搜尋測試」頁，已預選「{ns_name}」", icon="🔍")
                    with col_del:
                        if st.button(
                            "🗑️",
                            key=f"del_ns_{ns_name}",
                            help="刪除此知識空間（不可還原）",
                            use_container_width=True,
                            type="secondary",
                        ):
                            delete_namespace_chunks(ns_name)
                            st.session_state.kg_processed_files = {
                                key for key in st.session_state.kg_processed_files
                                if not key.endswith(f"::{ns_name}")
                            }
                            st.toast(f"已刪除知識空間「{ns_name}」", icon="🗑️")
                            st.rerun()

            st.markdown("")  # 每個 tag group 之間留空行
