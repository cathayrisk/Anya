"""
Cowork — 任務型 Agent（對話式介面）
多步驟自主 Agent，以聊天方式輸入任務，自動規劃、研究、整合並產出報告。
"""
from __future__ import annotations

import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import streamlit as st
from langchain_core.tools import tool
from openai import OpenAI

# ── Context Engineering middleware（deepagents / langchain）────────────────────
# @dynamic_prompt：每次 LLM 呼叫前動態注入 system prompt 補充
# @wrap_model_call：每次 LLM 呼叫前過濾工具清單
_HAS_CE_MW = False
try:
    from langchain.agents.middleware import (
        dynamic_prompt as _dynamic_prompt_deco,
        wrap_model_call as _wrap_model_call_deco,
        ModelRequest as _ModelRequest,
    )
    _HAS_CE_MW = True
except ImportError:
    try:
        from deepagents.middleware import (
            dynamic_prompt as _dynamic_prompt_deco,
            wrap_model_call as _wrap_model_call_deco,
            ModelRequest as _ModelRequest,
        )
        _HAS_CE_MW = True
    except ImportError:
        _HAS_CE_MW = False

from docstore import (
    FileRow,
    build_file_row_from_bytes,
    build_indices_incremental,
    doc_list_payload,
)

# ── 頁面設定 ───────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Cowork", page_icon="🥜", layout="wide")

# ── API Key ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = (
    st.secrets.get("OPENAI_API_KEY")
    or st.secrets.get("OPENAI_KEY")
    or os.getenv("OPENAI_API_KEY")
)
if not OPENAI_API_KEY:
    st.error("找不到 OpenAI API Key，請在 .streamlit/secrets.toml 設定 OPENAI_API_KEY。")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
_oai = OpenAI(api_key=OPENAI_API_KEY)

from langchain_openai import ChatOpenAI as _ChatOpenAI

_main_llm = _ChatOpenAI(
    model="gpt-5.2",
    api_key=OPENAI_API_KEY,
    use_responses_api=True,
)

# ── Supabase 知識庫（選用）────────────────────────────────────────────────────
_HAS_KB = False
_kb_supabase = None
_kb_embeddings = None
try:
    from supabase import create_client as _sb_create_client
    from langchain_openai import OpenAIEmbeddings as _OAIEmb
    if st.secrets.get("SUPABASE_URL") and st.secrets.get("SUPABASE_KEY"):
        _kb_supabase = _sb_create_client(
            st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"]
        )
        _kb_embeddings = _OAIEmb(
            openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small"
        )
        _HAS_KB = True
except Exception:
    _HAS_KB = False

COWORK_DIR = Path(__file__).parent.parent / "cowork"

# ── Module-level DocStore（避免 deepagents worker thread 讀不到 st.session_state）
class _DS:
    store = None  # 每次 invoke 前在主 thread 設定


# ══════════════════════════════════════════════════════════════════════════════
# CONTEXT ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CoworkContext:
    """型別化 Runtime Context，每次 Agent 執行時傳入，讓 middleware 知道目前環境狀態。"""
    has_documents: bool = False    # 是否有已建立索引的文件
    doc_chunk_count: int = 0       # 索引的 chunks 數量
    has_kb: bool = False           # Supabase 知識庫是否啟用


# ── @dynamic_prompt：每次 LLM 呼叫前動態注入環境資訊到 system prompt ──────────
if _HAS_CE_MW:
    @_dynamic_prompt_deco
    def _cowork_dynamic_prompt(request: _ModelRequest) -> str:
        """注入今日日期、文件狀態、長對話提醒，讓 Agent 不需使用者告知即可掌握環境。"""
        ctx: CoworkContext | None = getattr(request, "context", None)
        lines: list[str] = [f"📅 今日日期：{datetime.now().strftime('%Y-%m-%d %H:%M')}"]

        # 文件索引 / KB 狀態：只在 ctx 有效時注入
        # ctx=None 表示 context= 參數未能正確傳遞；此時靜默，由 user message 的 env prefix 負責
        if ctx is not None:
            if ctx.has_documents:
                lines.append(f"📚 已建立文件索引：{ctx.doc_chunk_count} chunks 可用，請優先使用 docstore_search 搜尋。")
            else:
                lines.append("📚 目前無已索引文件，請勿呼叫 docstore_search。")
            if not ctx.has_kb:
                lines.append("🏢 公司知識庫未啟用，請勿呼叫 company_knowledge_search。")

        # 長對話精簡提醒
        if len(getattr(request, "messages", [])) > 20:
            lines.append("⚠️ 對話已很長，請精簡回答，避免重複前面已說過的內容。")

        return "\n".join(lines)

    # ── @wrap_model_call：依據 context 過濾工具，避免 Agent 呼叫無法使用的工具 ──
    @_wrap_model_call_deco
    def _filter_tools(request: _ModelRequest, handler: Callable) -> Any:
        """移除當前狀態下不可用的工具，減少 Agent 的無效呼叫與幻覺。"""
        ctx: CoworkContext | None = getattr(request, "context", None)
        tools = list(request.tools)

        if ctx is not None:
            if not ctx.has_documents:
                tools = [t for t in tools if getattr(t, "name", "") != "docstore_search"]
            if not ctx.has_kb:
                tools = [t for t in tools if getattr(t, "name", "") != "company_knowledge_search"]

        return handler(request.override(tools=tools))

# ── Session State 初始化 ───────────────────────────────────────────────────────
_SS_DEFAULTS: dict = {
    "cowork_chat_history": [],   # list of {role, content, todos, tool_calls_log, web_sources, report_content, files}
    "cowork_file_rows": [],
    "cowork_file_bytes": {},
    "cowork_ds_store": None,
    "cowork_ds_processed_keys": set(),
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── URL 解析 regex ─────────────────────────────────────────────────────────────
_URL_SRC_RE = re.compile(r"- \[(.+?)\]\((https?://[^\)]+)\)")

def _parse_sources(text: str) -> list[dict]:
    return [{"title": m.group(1), "url": m.group(2)} for m in _URL_SRC_RE.finditer(text)]


# ══════════════════════════════════════════════════════════════════════════════
# TOOLS
# ══════════════════════════════════════════════════════════════════════════════

@tool
def web_search(query: str) -> str:
    """Search the web for current information using OpenAI web search.
    Always use think after each search to evaluate: Did I find enough?
    What's missing? Should I search again with different keywords?"""
    try:
        response = _oai.responses.create(
            model="gpt-4.1",
            tools=[{"type": "web_search_preview"}],
            input=query,
        )
        text = response.output_text or "（搜尋無結果）"

        # 提取 URL 引用
        sources: list[dict] = []
        for item in response.output:
            for block in getattr(item, "content", None) or []:
                for ann in getattr(block, "annotations", None) or []:
                    if getattr(ann, "type", "") == "url_citation":
                        url = getattr(ann, "url", "")
                        title = getattr(ann, "title", url)
                        if url and not any(s["url"] == url for s in sources):
                            sources.append({"title": title, "url": url})

        if sources:
            src_md = "\n".join(f"- [{s['title']}]({s['url']})" for s in sources)
            return f"{text}\n\n**🔗 搜尋來源：**\n{src_md}"
        return text
    except Exception as e:
        return f"搜尋失敗：{e}"


@tool
def think(thought: str) -> str:
    """Think through reasoning step by step.
    Use after web_search to evaluate:
    - What did I find? Is it relevant and credible?
    - What gaps remain? Should I search again?
    Use before writing reports to plan structure and coverage."""
    return thought


@tool
def docstore_search(query: str) -> str:
    """Search user-uploaded documents (FAISS + BM25 hybrid retrieval).
    Call MULTIPLE TIMES with different focused queries for comprehensive coverage.
    Never use '*' or vague terms — use specific topic keywords."""
    ds = _DS.store
    if ds is None or not getattr(ds, "chunks", None):
        return "目前沒有已建立索引的文件。請先在上方「📚 上傳文件」區塊建立索引。"
    try:
        emb_resp = _oai.embeddings.create(model="text-embedding-3-small", input=[query])
        qvec = np.array(emb_resp.data[0].embedding, dtype="float32")
        qvec /= np.linalg.norm(qvec) + 1e-9
        qvec = qvec.reshape(1, -1)
        results = ds.search_hybrid(query, qvec, k=10)
        if not results:
            return "在上傳文件中找不到相關內容。"
        return "\n\n".join(f"[相似度 {score:.2f}] {chunk.text}" for score, chunk in results)
    except Exception as e:
        return f"文件搜尋失敗：{e}"


@tool
def company_knowledge_search(query: str) -> str:
    """Search company internal knowledge base (SOPs, regulations, ESG, product docs)."""
    if not _HAS_KB or _kb_supabase is None or _kb_embeddings is None:
        return "公司知識庫未啟用。請確認 SUPABASE_URL 和 SUPABASE_KEY 已設定。"
    try:
        qvec = _kb_embeddings.embed_query(query)
        result = _kb_supabase.rpc(
            "match_knowledge_chunks",
            {"query_embedding": qvec, "match_threshold": 0.30,
             "match_count": 8, "namespace_filter": None},
        ).execute()
        rows = result.data or []
        if not rows:
            return "公司知識庫中找不到相關內容。"
        return "\n\n".join(
            f"[{r.get('similarity', 0):.2f}] {r.get('filename', '')}:\n{(r.get('content') or '')[:500]}"
            for r in rows[:6]
        )
    except Exception as e:
        return f"知識庫搜尋失敗：{e}"


# ── Research Sub-Agent ────────────────────────────────────────────────────────
RESEARCHER_PROMPT = """You are a focused research assistant conducting targeted web research.

<Instructions>
1. Read the research topic carefully
2. Execute web_search with a well-crafted query
3. ALWAYS use think after each search:
   - What did I find? Is it relevant, credible, and sufficient?
   - What gaps remain?
   - Should I search again with different or narrower keywords?
4. Stop when you have 3+ good sources OR after 5 searches maximum
5. Structure your findings clearly with headings and cite all sources
</Instructions>

<Output Format>
Use clear headings. Cite inline as [1][2][3].
End with:
### Sources
[1] Title — URL
[2] Title — URL
</Output Format>"""

_research_llm = _ChatOpenAI(
    model="gpt-5.2",
    api_key=OPENAI_API_KEY,
    use_responses_api=True,
    reasoning_effort="low",
)

research_sub_agent = {
    "name": "research-agent",
    "description": (
        "Delegate COMPREHENSIVE multi-step research to this agent — NOT simple factual lookups. "
        "Use when you need: multiple searches, critical evaluation, structured findings with citations. "
        "For quick single-fact queries, use web_search directly instead. "
        "Give ONE focused research topic per delegation."
    ),
    "system_prompt": RESEARCHER_PROMPT,
    "tools": [web_search, think],
    "model": _research_llm,
}


# ── Agent 建立（per-session，保留 thread 跨回合對話記憶）────────────────────────
def _get_agent_and_workspace() -> tuple:
    if "cowork_agent" not in st.session_state:
        workspace = tempfile.mkdtemp(prefix="cowork_")
        st.session_state.cowork_workspace = workspace
        st.session_state.cowork_thread_id = str(uuid.uuid4())

        from deepagents import create_deep_agent
        from deepagents.backends import FilesystemBackend
        from langgraph.checkpoint.memory import InMemorySaver

        backend = FilesystemBackend(root_dir=workspace, virtual_mode=True)

        # SummarizationMiddleware 已在 create_deep_agent 預設 stack 中，
        # 不可重複傳入（會觸發 AssertionError: duplicate middleware）。
        # 只傳入我們自訂的 Context Engineering middleware。
        ce_middleware: list = []
        if _HAS_CE_MW:
            ce_middleware = [_cowork_dynamic_prompt, _filter_tools]

        agent = create_deep_agent(
            model=_main_llm,
            middleware=ce_middleware,
            context_schema=CoworkContext,  # 型別化 runtime context
            memory=[str(COWORK_DIR / "AGENTS.md")],
            skills=[str(COWORK_DIR / "skills")],
            tools=[web_search, think, docstore_search, company_knowledge_search],
            subagents=[research_sub_agent],
            backend=backend,
            checkpointer=InMemorySaver(),
        )
        st.session_state.cowork_agent = agent

    return st.session_state.cowork_agent, st.session_state.cowork_workspace


def _reset_conversation():
    """清除對話記憶，開啟新 thread（保留已上傳文件索引）。"""
    for k in ["cowork_agent", "cowork_workspace", "cowork_thread_id"]:
        st.session_state.pop(k, None)
    st.session_state.cowork_chat_history = []


# ── UI 短期記憶修剪 ───────────────────────────────────────────────────────────
TRIM_LAST_N_TURNS = 20  # 保留最近 N 則訊息（user + assistant 合計）

def _trim_chat_history() -> None:
    """修剪 UI 聊天歷史，防止 session_state 無限成長（類似 Home.py TRIM_LAST_N_USER_TURNS）。"""
    hist = st.session_state.cowork_chat_history
    if len(hist) > TRIM_LAST_N_TURNS:
        st.session_state.cowork_chat_history = hist[-TRIM_LAST_N_TURNS:]


# ── UI 常數 ───────────────────────────────────────────────────────────────────
TOOL_ICONS: dict[str, str] = {
    "web_search": "🔍", "think": "🤔", "write_file": "📝",
    "read_file": "📖", "edit_file": "✏️", "task": "🤖",
    "docstore_search": "📚", "company_knowledge_search": "🏢",
    "glob": "🗂️", "grep": "🔎", "ls": "📂",
    "research-agent": "🔬", "write_todos": "📋",
}
TODO_ICONS: dict[str, str] = {
    "completed": "✅", "in_progress": "🔄", "pending": "⬜",
}
REPORT_NAMES = ["final_report.md", "analysis_report.md", "report.md"]


# ── 歷史訊息渲染（collapsed，用 expander）────────────────────────────────────
def _render_history_assistant(msg: dict, msg_idx: int = 0) -> None:
    todos = msg.get("todos", [])
    tool_calls_log = msg.get("tool_calls_log", [])
    web_sources = msg.get("web_sources", [])
    report = msg.get("report_content", "")
    content = msg.get("content", "")
    files = msg.get("files", {})

    if todos:
        with st.expander("📋 任務進度", expanded=False):
            for t in todos:
                icon = TODO_ICONS.get(t.get("status", "pending"), "⬜")
                st.markdown(f"{icon} {t.get('content', '')}")

    if tool_calls_log:
        with st.expander("🔧 工具呼叫紀錄", expanded=False):
            for tc in tool_calls_log:
                icon = TOOL_ICONS.get(tc["name"], "🔧")
                label = f"{icon} **{tc['name']}**"
                if tc.get("summary"):
                    label += f"：{tc['summary']}"
                st.markdown(label)

    if web_sources:
        with st.expander("🔗 網路來源", expanded=False):
            for s in web_sources:
                st.markdown(f"- [{s['title']}]({s['url']})")

    if report:
        st.markdown(report)
    if content and not report:
        st.markdown(content)
    elif content and report:
        with st.expander("💬 Agent 最終回應", expanded=False):
            st.markdown(content)

    if files:
        with st.expander("📁 工作區檔案", expanded=False):
            for fpath, file_data in files.items():
                filename = Path(fpath).name
                cn, cb = st.columns([4, 1])
                cn.markdown(f"📄 `{filename}`")
                raw = (
                    file_data if isinstance(file_data, bytes)
                    else file_data.encode() if isinstance(file_data, str)
                    else str(file_data).encode()
                )
                cb.download_button(
                    "下載", data=raw, file_name=filename,
                    key=f"dl_h{msg_idx}_{filename}",
                )


# ══════════════════════════════════════════════════════════════════════════════
# 主頁面
# ══════════════════════════════════════════════════════════════════════════════
st.title("🥜 Cowork — 任務型 Agent")
st.caption("輸入複合任務，Agent 將自動規劃、研究、整合並產出報告。支援上下文對話。")

# ── 頂部操作列：文件上傳 + 清除對話 ──────────────────────────────────────────
_ds_store = st.session_state.cowork_ds_store
_has_index = (
    _ds_store is not None
    and getattr(_ds_store, "index", None) is not None
    and _ds_store.index.ntotal > 0
)
doc_label = (
    f"📚 上傳文件（已建索引：{len(_ds_store.chunks)} chunks）"
    if _has_index else "📚 上傳文件"
)

with st.expander(doc_label, expanded=not _has_index):
    st.caption("檔案只存在本次 session。建立索引後，Agent 才能搜尋文件內容。")

    uploaded = st.file_uploader(
        "上傳文件",
        type=["pdf", "docx", "doc", "pptx", "xlsx", "xls", "txt", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="cowork_file_uploader",
    )
    if uploaded:
        existing = {(r.name, r.bytes_len) for r in st.session_state.cowork_file_rows}
        for f in uploaded:
            data = f.read()
            if (f.name, len(data)) in existing:
                continue
            row = build_file_row_from_bytes(filename=f.name, data=data)
            st.session_state.cowork_file_rows.append(row)
            st.session_state.cowork_file_bytes[row.file_id] = data

    rows = st.session_state.cowork_file_rows
    if rows:
        payload = doc_list_payload(rows, st.session_state.cowork_ds_store)
        items = payload.get("items", [])
        id_to_row = {r.file_id: r for r in rows}
        key_to_file_id = {
            (os.path.splitext(r.name)[0], r.ext): r.file_id for r in rows
        }

        def _short(name: str, n: int = 48) -> str:
            name = (name or "").strip()
            return name if len(name) <= n else name[:n] + "…"

        _blank_row = FileRow(
            file_id="", file_sig="", name="", ext="", bytes_len=0,
            pages=None, extracted_chars=0, token_est=0,
            blank_pages=None, blank_ratio=None, text_pages=None,
            text_pages_ratio=None, likely_scanned=False, use_ocr=False,
        )
        df = pd.DataFrame([
            {
                "OCR": bool(
                    id_to_row.get(
                        key_to_file_id.get((it.get("title"), it.get("ext"))),
                        _blank_row,
                    ).use_ocr
                ) if it.get("ext") == ".pdf" else False,
                "檔名": _short(f"{it.get('title')}{it.get('ext')}"),
                "類型": (it.get("ext") or "").lstrip(".").upper(),
                "頁數": it.get("pages"),
                "chunks": int(it.get("chunks") or 0),
                "_file_id": key_to_file_id.get((it.get("title"), it.get("ext"))),
            }
            for it in items
        ])
        edited = st.data_editor(
            df, hide_index=True, width="stretch",
            key="cowork_file_list_editor",
            column_config={
                "_file_id": st.column_config.TextColumn("_file_id", disabled=True, width="small"),
                "檔名": st.column_config.TextColumn("檔名", disabled=True, width="large"),
                "類型": st.column_config.TextColumn("類型", disabled=True, width="small"),
                "頁數": st.column_config.NumberColumn("頁數", disabled=True, width="small"),
                "chunks": st.column_config.NumberColumn("chunks", disabled=True, width="small"),
                "OCR": st.column_config.CheckboxColumn(
                    "OCR", help="僅 PDF；掃描 PDF 用視覺 OCR（較慢）", width="small"
                ),
            },
            disabled=["_file_id", "檔名", "類型", "頁數", "chunks"],
        )
        try:
            for rec in edited.to_dict(orient="records"):
                fid = rec.get("_file_id")
                if fid and fid in id_to_row:
                    r = id_to_row[fid]
                    r.use_ocr = bool(rec.get("OCR")) if r.ext == ".pdf" else False
        except Exception:
            pass

        caps = payload.get("capabilities", {}) or {}
        st.caption(
            f"BM25={'on' if caps.get('bm25') else 'off'} · "
            f"FlashRank={'on' if caps.get('flashrank') else 'off'} · "
            f"Unstructured={'on' if caps.get('unstructured_loaders') else 'off'} · "
            f"PyMuPDF={'on' if caps.get('pymupdf') else 'off'}"
        )
    else:
        st.caption("（尚未上傳任何文件）")

    cb1, cb2 = st.columns(2)
    build_btn = cb1.button("🚀 建立/更新索引", type="primary", width="stretch", key="cowork_build_idx")
    if cb2.button("🧹 清空文件庫", width="stretch", key="cowork_clear_docs"):
        st.session_state.cowork_file_rows = []
        st.session_state.cowork_file_bytes = {}
        st.session_state.cowork_ds_store = None
        st.session_state.cowork_ds_processed_keys = set()
        st.rerun()

    if build_btn and rows:
        with st.status("建索引中（文字抽取 + embeddings）...", expanded=True) as s:
            _store, _stats, _pkeys = build_indices_incremental(
                _oai,
                file_rows=st.session_state.cowork_file_rows,
                file_bytes_map=st.session_state.cowork_file_bytes,
                store=st.session_state.cowork_ds_store,
                processed_keys=st.session_state.cowork_ds_processed_keys,
            )
            st.session_state.cowork_ds_store = _store
            st.session_state.cowork_ds_processed_keys = _pkeys
            s.write(f"新增文件：{_stats.get('new_reports', 0)}　新增 chunks：{_stats.get('new_chunks', 0)}")
            if _stats.get("errors"):
                s.warning("\n".join(f"- {e}" for e in _stats["errors"][:5]))
            s.update(state="complete")
        st.rerun()

    if _has_index:
        st.success(f"已建立索引：{len(st.session_state.cowork_ds_store.chunks)} chunks")
    elif rows:
        st.info("尚未建立索引（點「建立/更新索引」）")

st.divider()

# ── 對話狀態說明 ───────────────────────────────────────────────────────────────
_ch = st.session_state.cowork_chat_history
if _ch:
    st.caption(f"對話共 {len(_ch)} 則訊息 · 相同 session 保有短期記憶")
else:
    st.caption("💡 輸入任務或問題，Agent 會自動規劃並執行。支援多輪對話。")

# ── 顯示歷史對話 ──────────────────────────────────────────────────────────────
for _i, _msg in enumerate(st.session_state.cowork_chat_history):
    with st.chat_message(_msg["role"]):
        if _msg["role"] == "user":
            st.markdown(_msg["content"])
        else:
            _render_history_assistant(_msg, msg_idx=_i)

# ══════════════════════════════════════════════════════════════════════════════
# Chat Input → Agent 執行
# ══════════════════════════════════════════════════════════════════════════════
if prompt := st.chat_input(
    "輸入任務或問題… 例：分析上傳文件找出關鍵風險 / 研究 AI Agent 趨勢並產出報告",
    key="cowork_chat_input",
):
    # ── 1. 顯示使用者訊息 ─────────────────────────────────────────────────────
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.cowork_chat_history.append({"role": "user", "content": prompt})

    # ── 2. 取得 Agent + 建立 CoworkContext ────────────────────────────────────
    agent, workspace = _get_agent_and_workspace()
    _DS.store = st.session_state.cowork_ds_store  # 保留 module-level ref（工具 thread 安全用）

    _ds_ref = st.session_state.cowork_ds_store

    # 使用 chunks（與 docstore_search 工具內部相同的判斷），比 index.ntotal 更穩定
    _doc_chunks = list(getattr(_ds_ref, "chunks", None) or []) if _ds_ref else []
    _has_idx = len(_doc_chunks) > 0

    # fallback：如果 FAISS 狀態異常但使用者曾建立過索引，仍應嘗試搜尋
    _has_processed = bool(st.session_state.cowork_ds_processed_keys)

    _runtime_ctx = CoworkContext(
        has_documents=_has_idx,
        doc_chunk_count=len(_doc_chunks),
        has_kb=_HAS_KB,
    )

    # ── 環境提示前綴（直接注入 user message，不依賴 CE middleware）────────────
    _env_lines: list[str] = [f"📅 今日日期：{datetime.now().strftime('%Y-%m-%d')}"]
    if _has_idx:
        _env_lines.append(
            f"📚 文件索引：已就緒，共 {len(_doc_chunks)} chunks。"
            "使用者問到文件/附件時，**你必須立即呼叫 `docstore_search` 工具**取得內容，"
            "不得說找不到文件或要求使用者再次提供。"
        )
    elif _has_processed:
        # 曾建立過索引但目前 store 物件狀態異常 → 仍鼓勵嘗試
        _env_lines.append(
            "📚 文件索引：使用者本次 session 曾上傳並建立索引。"
            "若問到附件內容，請先呼叫 `docstore_search` 確認是否可用。"
        )
    else:
        _env_lines.append(
            "📚 文件索引：目前無已索引文件。"
            "若使用者提到附件，請提示他先在上方上傳並點「建立/更新索引」。"
        )

    if not _HAS_KB:
        _env_lines.append("🏢 公司知識庫：未啟用，請勿呼叫 company_knowledge_search。")

    _env_prefix = "<系統環境資訊>\n" + "\n".join(_env_lines) + "\n</系統環境資訊>\n\n"
    _agent_prompt = _env_prefix + prompt

    # ── 3. Assistant 回應區塊 ─────────────────────────────────────────────────
    with st.chat_message("assistant"):
        # 狀態指示器（即時更新工具呼叫 + 任務進度）
        status = st.status("思考中…✨", expanded=True)
        status_steps_ph = status.empty()   # 工具呼叫列表
        status_todos_ph  = status.empty()  # 任務進度 TodoList

        # 主要回應佔位（在 status 之下）
        response_ph = st.empty()

        # ── 狀態追蹤（用可變容器避免 closure 重新賦值問題）
        step_log: list[str]       = []
        current_todos: list[dict] = []
        tool_calls_log: list[dict] = []
        web_sources: list[dict]   = []

        def _refresh_status() -> None:
            """更新 status 內的步驟列表與任務進度。"""
            if step_log:
                status_steps_ph.markdown(
                    "**🔧 執行步驟**\n" + "\n".join(f"- {s}" for s in step_log[-10:])
                )
            if current_todos:
                lines = [
                    f"{TODO_ICONS.get(t.get('status', 'pending'), '⬜')} {t.get('content', '')}"
                    for t in current_todos
                ]
                status_todos_ph.markdown("**📋 任務進度**\n\n" + "\n\n".join(lines))

        # ── 執行 Agent（LangGraph stream_mode="values"）────────────────────────
        all_messages: list = []
        try:
            cfg = {"configurable": {"thread_id": st.session_state.cowork_thread_id}}

            # 取得本次執行前的訊息數量，避免 stream 時重複處理舊輪次的 tool calls
            # （stream_mode="values" 返回完整累積 state，不是本輪 delta）
            try:
                _pre_state = agent.get_state(cfg)
                last_msg_count = len((_pre_state.values or {}).get("messages", []))
            except Exception:
                last_msg_count = 0

            final_chunk = None

            for chunk in agent.stream(
                {"messages": [{"role": "user", "content": _agent_prompt}]},
                config=cfg,
                context=_runtime_ctx,  # CoworkContext：傳遞環境狀態給 middleware 和工具
                stream_mode="values",  # 完整 state dict；避免 Overwrite 物件問題
            ):
                final_chunk = chunk
                all_msgs = chunk.get("messages", [])
                new_msgs = all_msgs[last_msg_count:]
                last_msg_count = len(all_msgs)

                for msg in new_msgs:
                    # ── AI 訊息：工具呼叫 ─────────────────────────────────
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            name = tc.get("name", "")
                            args = tc.get("args", {})

                            if name == "write_todos":
                                # 更新任務進度（原地清空 + 填入，避免 closure 重新賦值）
                                current_todos.clear()
                                current_todos.extend(args.get("todos", []))
                            else:
                                icon = TOOL_ICONS.get(name, "🔧")
                                summary_raw = (
                                    args.get("query")
                                    or args.get("thought", "")
                                    or args.get("description", "")
                                )
                                summary = str(summary_raw)[:80] if summary_raw else ""
                                step_str = f"{icon} {name}" + (f"：{summary}" if summary else "")
                                step_log.append(step_str)
                                tool_calls_log.append({"name": name, "summary": summary})

                            _refresh_status()

                    # ── Tool 回應訊息：擷取 web_search 來源 URL ────────────
                    msg_name = getattr(msg, "name", "")
                    if msg_name == "web_search":
                        content_text = getattr(msg, "content", "")
                        if isinstance(content_text, str):
                            for s in _parse_sources(content_text):
                                if not any(x["url"] == s["url"] for x in web_sources):
                                    web_sources.append(s)

            all_messages = final_chunk.get("messages", []) if final_chunk else []
            status.update(label="完成 ✅", state="complete", expanded=False)

        except Exception as exc:
            status.update(label="執行失敗 ❌", state="error", expanded=False)
            st.error(f"Agent 執行失敗：{exc}")
            st.stop()

        # ── 網路來源（expanded，最優先顯示）──────────────────────────────
        if web_sources:
            with st.expander("🔗 網路來源", expanded=True):
                for s in web_sources:
                    st.markdown(f"- [{s['title']}]({s['url']})")

        # ── 任務進度（最終狀態）──────────────────────────────────────────
        if current_todos:
            with st.expander("📋 任務進度", expanded=True):
                for t in current_todos:
                    icon = TODO_ICONS.get(t.get("status", "pending"), "⬜")
                    st.markdown(f"{icon} {t.get('content', '')}")

        # 注意：工具呼叫紀錄（tool_calls_log）仍保留在 session_state，供歷史訊息渲染使用。
        # 當前回應不另顯示，因 status 展開的「執行步驟」已包含相同資訊。

        # ── 收集工作區檔案 ────────────────────────────────────────────────
        workspace_path = Path(workspace)
        files: dict = {}
        for _f in workspace_path.rglob("*"):
            if _f.is_file():
                rel = str(_f.relative_to(workspace_path))
                files[rel] = _f.read_bytes()

        # ── 研究報告直接展示（優先）──────────────────────────────────────
        report_content = ""
        for rname in REPORT_NAMES:
            rdata = files.get(rname)
            if rdata:
                report_content = (
                    rdata.decode("utf-8", errors="replace")
                    if isinstance(rdata, bytes) else str(rdata)
                )
                st.subheader("📄 研究報告")
                st.markdown(report_content)
                break

        # ── 最終 Agent 文字回應 ───────────────────────────────────────────
        final_msg = all_messages[-1] if all_messages else None
        response_text = ""
        if final_msg and hasattr(final_msg, "content") and final_msg.content:
            content = final_msg.content
            if isinstance(content, list):
                content = "\n".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            response_text = content
            if not report_content:
                response_ph.markdown(response_text)
            else:
                response_ph.empty()
                with st.expander("💬 Agent 最終回應", expanded=False):
                    st.markdown(response_text)
        else:
            response_ph.empty()

        # ── 工作區檔案下載 ────────────────────────────────────────────────
        if files:
            with st.expander("📁 工作區檔案", expanded=False):
                for fpath, file_data in files.items():
                    filename = Path(fpath).name
                    cn, cb = st.columns([4, 1])
                    cn.markdown(f"📄 `{filename}`")
                    raw = (
                        file_data if isinstance(file_data, bytes)
                        else file_data.encode() if isinstance(file_data, str)
                        else str(file_data).encode()
                    )
                    cb.download_button(
                        "下載", data=raw, file_name=filename,
                        key=f"dl_new_{filename}",
                    )

        # ── 對話記憶摘要（SummarizationMiddleware 生成，存於 workspace）────
        _thread_id = st.session_state.get("cowork_thread_id", "")
        _mem_file = Path(workspace) / "conversation_history" / f"{_thread_id}.md"
        if _mem_file.exists():
            with st.expander("🧠 對話記憶摘要", expanded=False):
                st.caption("由 SummarizationMiddleware 自動生成，當對話超過 8K tokens 時觸發。")
                st.markdown(_mem_file.read_text(encoding="utf-8", errors="replace"))

        # ── 儲存 assistant 訊息到對話歷史，並修剪 UI 歷史長度 ────────────
        st.session_state.cowork_chat_history.append({
            "role": "assistant",
            "content": response_text,
            "todos": list(current_todos),
            "tool_calls_log": list(tool_calls_log),
            "web_sources": list(web_sources),
            "report_content": report_content,
            "files": files,
        })
        _trim_chat_history()  # 保留最近 TRIM_LAST_N_TURNS 則，防止 session_state 過大
