"""
Cowork — 任務型 Agent
類 Claude Code 的多步驟任務 Agent，使用 deepagents。
"""

import os
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from langchain_core.tools import tool
from openai import OpenAI

# ── DocStore imports ──────────────────────────────────────────────────────────
from docstore import (
    FileRow,
    build_file_row_from_bytes,
    build_indices_incremental,
    doc_list_payload,
)

# ── 頁面設定 ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Cowork", page_icon="🥜", layout="wide")

# ── Session State 初始化 ───────────────────────────────────────────────────────
if "cowork_file_rows" not in st.session_state:
    st.session_state.cowork_file_rows = []
if "cowork_file_bytes" not in st.session_state:
    st.session_state.cowork_file_bytes = {}
if "cowork_ds_store" not in st.session_state:
    st.session_state.cowork_ds_store = None
if "cowork_ds_processed_keys" not in st.session_state:
    st.session_state.cowork_ds_processed_keys = set()

# ── API Key ───────────────────────────────────────────────────────────────────
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

# ── Supabase 知識庫初始化（選用）────────────────────────────────────────────
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

# ── Cowork 路徑 ───────────────────────────────────────────────────────────────
COWORK_DIR = Path(__file__).parent.parent / "cowork"

# ── Module-level DocStore 參考（避免 deepagents 在子執行緒中讀不到 st.session_state）
class _DS:
    store = None  # 在 invoke() 前由主執行緒更新


# ── 工具定義 ──────────────────────────────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """Search the web for current information using OpenAI web search.
    Use for factual queries, recent events, or topics not in internal knowledge."""
    try:
        response = _oai.responses.create(
            model="gpt-4.1-mini",
            tools=[{"type": "web_search_preview"}],
            input=query,
        )
        return response.output_text or "（搜尋無結果）"
    except Exception as e:
        return f"搜尋失敗：{e}"


@tool
def think(thought: str) -> str:
    """Think through your reasoning step by step.
    Use after gathering information to reflect: What did I find? What's missing?
    Do I have enough to write the report, or should I search more?"""
    return thought


@tool
def docstore_search(query: str) -> str:
    """Search documents uploaded by the user in the Cowork document library (FAISS + BM25 hybrid).
    Use when the user has uploaded files and wants information from them."""
    ds = _DS.store
    if ds is None or not getattr(ds, "chunks", None):
        return "目前沒有已建立索引的文件。請先在上方的「📚 上傳文件」區塊上傳並建立索引。"
    try:
        emb_resp = _oai.embeddings.create(model="text-embedding-3-small", input=[query])
        qvec = np.array(emb_resp.data[0].embedding, dtype="float32")
        qvec /= np.linalg.norm(qvec) + 1e-9
        results = ds.search_hybrid(query, qvec, k=5)
        if not results:
            return "在上傳文件中找不到相關內容。"
        return "\n\n".join(f"[相似度 {score:.2f}] {chunk.text}" for score, chunk in results)
    except Exception as e:
        return f"文件搜尋失敗：{e}"


@tool
def company_knowledge_search(query: str) -> str:
    """Search the company internal knowledge base (SOPs, regulations, product documents, ESG).
    Use when the user asks about internal company information."""
    if not _HAS_KB or _kb_supabase is None or _kb_embeddings is None:
        return "公司知識庫未啟用。請確認 SUPABASE_URL 和 SUPABASE_KEY 已設定。"
    try:
        qvec = _kb_embeddings.embed_query(query)
        result = _kb_supabase.rpc(
            "match_knowledge_chunks",
            {
                "query_embedding": qvec,
                "match_threshold": 0.30,
                "match_count": 8,
                "namespace_filter": None,
            },
        ).execute()
        rows = result.data or []
        if not rows:
            return "公司知識庫中找不到相關內容。"
        parts = []
        for row in rows[:6]:
            fname = row.get("filename", "unknown")
            content = (row.get("content") or "")[:500]
            score = row.get("similarity", 0)
            parts.append(f"[{score:.2f}] {fname}\n{content}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"知識庫搜尋失敗：{e}"


# ── Research Sub-Agent ────────────────────────────────────────────────────────

RESEARCHER_PROMPT = """You are a research assistant conducting targeted web research.

<Task>
Use web_search and think tools to gather information on the given topic.
Call tools in series or parallel to research comprehensively.
</Task>

<Instructions>
1. Read the research topic carefully
2. Start with a broad search query
3. Use think after each search: What did I find? What's missing? Need more searches?
4. Execute narrower searches to fill gaps
5. Stop when you can answer confidently
</Instructions>

<Hard Limits>
- Simple queries: 2-3 searches maximum
- Complex queries: up to 5 searches maximum
- Always stop after 5 searches
- Stop immediately when you have 3+ relevant sources
</Hard Limits>

<Output Format>
Structure your findings with clear headings.
Cite sources inline: [1], [2], [3]
End with:
### Sources
[1] Title: URL
[2] Title: URL
</Output Format>"""

research_sub_agent = {
    "name": "research-agent",
    "description": (
        "Delegate web research to this agent. "
        "Give ONE focused research topic at a time. "
        "Returns structured findings with citations."
    ),
    "system_prompt": RESEARCHER_PROMPT,
    "tools": [web_search, think],
    "model": "openai:gpt-4.1-mini",
}

# ── Agent 建立（per session）─────────────────────────────────────────────────

def _get_agent_and_workspace():
    if "cowork_agent" not in st.session_state:
        workspace = tempfile.mkdtemp(prefix="cowork_")
        st.session_state.cowork_workspace = workspace
        st.session_state.cowork_thread_id = str(uuid.uuid4())

        from deepagents import create_deep_agent
        from deepagents.backends import FilesystemBackend
        from langgraph.checkpoint.memory import InMemorySaver

        agent = create_deep_agent(
            model="openai:gpt-4.1",
            memory=[str(COWORK_DIR / "AGENTS.md")],
            skills=[str(COWORK_DIR / "skills")],
            tools=[web_search, think, docstore_search, company_knowledge_search],
            subagents=[research_sub_agent],
            backend=FilesystemBackend(root_dir=workspace),
            checkpointer=InMemorySaver(),
        )
        st.session_state.cowork_agent = agent

    return st.session_state.cowork_agent, st.session_state.cowork_workspace


def _reset_task():
    """清除任務結果，保留已上傳文件。"""
    for key in ["cowork_agent", "cowork_workspace", "cowork_thread_id",
                "cowork_result", "cowork_todos", "cowork_files"]:
        st.session_state.pop(key, None)


# ── 結果解析 ──────────────────────────────────────────────────────────────────

def _extract_todos(messages: list) -> list:
    todos = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("name") == "write_todos":
                    todos = tc.get("args", {}).get("todos", [])
    return todos


def _extract_tool_calls(messages: list) -> list:
    calls = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.get("name", "")
                args = tc.get("args", {})
                if name == "write_todos":
                    continue
                summary = ""
                if name == "web_search":
                    summary = args.get("query", "")[:60]
                elif name == "think":
                    summary = args.get("thought", "")[:60]
                elif name in ("write_file", "read_file", "edit_file"):
                    summary = args.get("file_path", "")
                elif name == "task":
                    summary = args.get("description", "")[:60]
                elif name in ("docstore_search", "company_knowledge_search"):
                    summary = args.get("query", "")[:60]
                calls.append({"name": name, "summary": summary})
    return calls


TOOL_ICONS = {
    "web_search": "🔍", "think": "🤔", "write_file": "📝",
    "read_file": "📖", "edit_file": "✏️", "task": "🤖",
    "docstore_search": "📚", "company_knowledge_search": "🏢",
    "glob": "🗂️", "grep": "🔎", "ls": "📂",
}

TODO_ICONS = {"completed": "✅", "in_progress": "🔄", "pending": "⬜"}


# ── 主頁面 ────────────────────────────────────────────────────────────────────

st.title("🥜 Cowork — 任務型 Agent")
st.caption("輸入複合任務，Agent 將自動規劃、研究、整合並產出報告。")

st.divider()

# ── 文件上傳區 ────────────────────────────────────────────────────────────────
_ds_store = st.session_state.cowork_ds_store
_has_index = (
    _ds_store is not None
    and getattr(_ds_store, "index", None) is not None
    and _ds_store.index.ntotal > 0
)
doc_label = (
    f"📚 上傳文件（已建索引：{len(_ds_store.chunks)} chunks）"
    if _has_index
    else "📚 上傳文件"
)

with st.expander(doc_label, expanded=not _has_index):
    st.caption("檔案只存在本次 session。建索引後，Agent 才能搜尋文件內容。")
    st.caption(":small[:gray[拖曳檔案到這裡，或點一下選取。]]")

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
    store = st.session_state.cowork_ds_store

    if rows:
        payload = doc_list_payload(rows, store)
        items = payload.get("items", [])

        # 建 file_id 查找表
        id_to_row = {r.file_id: r for r in st.session_state.cowork_file_rows}
        key_to_file_id = {}
        for r in st.session_state.cowork_file_rows:
            title = os.path.splitext(r.name)[0]
            key_to_file_id[(title, r.ext)] = r.file_id

        def _short(name: str, n: int = 48) -> str:
            name = (name or "").strip()
            return name if len(name) <= n else (name[:n] + "…")

        df = pd.DataFrame([
            {
                "OCR": bool(
                    id_to_row.get(
                        key_to_file_id.get((it.get("title"), it.get("ext"))),
                        FileRow(
                            file_id="", file_sig="", name="", ext="", bytes_len=0,
                            pages=None, extracted_chars=0, token_est=0,
                            blank_pages=None, blank_ratio=None, text_pages=None,
                            text_pages_ratio=None, likely_scanned=False, use_ocr=False,
                        ),
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

        st.markdown("### 📄 文件清單")
        st.caption("OCR 勾選只對 PDF 生效；非 PDF 會自動忽略。")

        edited = st.data_editor(
            df,
            hide_index=True,
            use_container_width=True,
            key="cowork_file_list_editor",
            column_config={
                "_file_id": st.column_config.TextColumn("_file_id", disabled=True, width="small"),
                "檔名": st.column_config.TextColumn("檔名", disabled=True, width="large"),
                "類型": st.column_config.TextColumn("類型", disabled=True, width="small"),
                "頁數": st.column_config.NumberColumn("頁數", disabled=True, width="small"),
                "chunks": st.column_config.NumberColumn("chunks", disabled=True, width="small"),
                "OCR": st.column_config.CheckboxColumn(
                    "OCR",
                    help="僅 PDF 可用；用 OCR 抽取掃描 PDF 文字（較慢）",
                    width="small",
                ),
            },
            disabled=["_file_id", "檔名", "類型", "頁數", "chunks"],
        )

        # 回寫 OCR 設定（只對 PDF 生效）
        try:
            for rec in edited.to_dict(orient="records"):
                fid = rec.get("_file_id")
                if not fid or fid not in id_to_row:
                    continue
                r = id_to_row[fid]
                r.use_ocr = bool(rec.get("OCR")) if r.ext == ".pdf" else False
        except Exception:
            pass

        # 能力摘要
        caps = payload.get("capabilities", {}) or {}
        st.markdown(
            ":small[:gray[能力："
            f"BM25={'on' if caps.get('bm25') else 'off'} · "
            f"FlashRank={'on' if caps.get('flashrank') else 'off'} · "
            f"Unstructured={'on' if caps.get('unstructured_loaders') else 'off'} · "
            f"PyMuPDF={'on' if caps.get('pymupdf') else 'off'}"
            "]]"
        )
    else:
        st.markdown(":small[（尚未上傳任何文件）]")

    cb1, cb2 = st.columns([1, 1])
    build_btn = cb1.button("🚀 建立/更新索引", type="primary", use_container_width=True, key="cowork_build_idx")
    clear_docs_btn = cb2.button("🧹 清空文件庫", use_container_width=True, key="cowork_clear_docs")

    if clear_docs_btn:
        st.session_state.cowork_file_rows = []
        st.session_state.cowork_file_bytes = {}
        st.session_state.cowork_ds_store = None
        st.session_state.cowork_ds_processed_keys = set()
        st.rerun()

    if build_btn and rows:
        with st.status("建索引中（抽文/OCR + embeddings）...", expanded=True) as s:
            store, stats, processed_keys = build_indices_incremental(
                _oai,
                file_rows=st.session_state.cowork_file_rows,
                file_bytes_map=st.session_state.cowork_file_bytes,
                store=st.session_state.cowork_ds_store,
                processed_keys=st.session_state.cowork_ds_processed_keys,
            )
            st.session_state.cowork_ds_store = store
            st.session_state.cowork_ds_processed_keys = processed_keys
            s.write(f"新增文件數：{stats.get('new_reports', 0)}")
            s.write(f"新增 chunks：{stats.get('new_chunks', 0)}")
            if stats.get("errors"):
                s.warning("部分檔案失敗：\n" + "\n".join(f"- {e}" for e in stats["errors"][:5]))
            s.update(state="complete")
        st.rerun()

    if _has_index:
        st.success(f"已建立索引：chunks={len(st.session_state.cowork_ds_store.chunks)}")
    elif rows:
        st.info("尚未建立索引（或索引為空）。")

st.divider()

# ── 任務輸入 ──────────────────────────────────────────────────────────────────
task_input = st.text_area(
    "任務描述",
    placeholder=(
        "例如：研究 LangGraph 的最新功能，整理成 markdown 報告並存成 final_report.md\n"
        "例如：比較 Claude、GPT-4、Gemini 在程式碼生成上的差異\n"
        "例如：分析上傳的 PDF 文件，找出關鍵風險點"
    ),
    height=120,
    key="task_input",
    label_visibility="collapsed",
)

col1, col2 = st.columns([1, 6])
with col1:
    run_btn = st.button("🚀 開始任務", type="primary", use_container_width=True)
with col2:
    if st.button("🗑 清除任務", use_container_width=False):
        _reset_task()
        st.rerun()

# ── 執行 ──────────────────────────────────────────────────────────────────────

if run_btn:
    task = (task_input or "").strip()
    if not task:
        st.warning("請輸入任務描述。")
    else:
        _reset_task()
        agent, workspace = _get_agent_and_workspace()

        # ★ 在 invoke 前把 DocStore 傳給 module-level 參考，確保子執行緒可讀到
        _DS.store = st.session_state.cowork_ds_store

        with st.spinner("⏳ Cowork Agent 正在處理任務，請稍候..."):
            try:
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": task}]},
                    config={"configurable": {"thread_id": st.session_state.cowork_thread_id}},
                )
                st.session_state.cowork_result = result
                st.session_state.cowork_todos = _extract_todos(result.get("messages", []))
                st.session_state.cowork_files = result.get("files", {})
            except Exception as e:
                st.error(f"Agent 執行失敗：{e}")

# ── 顯示結果 ──────────────────────────────────────────────────────────────────

if "cowork_result" in st.session_state:
    result = st.session_state.cowork_result
    todos = st.session_state.get("cowork_todos", [])
    files = st.session_state.get("cowork_files", {})
    messages = result.get("messages", [])

    st.divider()

    # Todo 清單
    if todos:
        st.subheader("📋 任務進度")
        for todo in todos:
            status = todo.get("status", "pending")
            icon = TODO_ICONS.get(status, "⬜")
            st.markdown(f"{icon} {todo.get('content', '')}")
        st.divider()

    # 工具呼叫紀錄
    tool_calls = _extract_tool_calls(messages)
    if tool_calls:
        with st.expander("🔧 工具呼叫紀錄", expanded=False):
            for tc in tool_calls:
                icon = TOOL_ICONS.get(tc["name"], "🔧")
                label = f"{icon} **{tc['name']}**"
                if tc["summary"]:
                    label += f"：{tc['summary']}"
                st.markdown(label)

    # 最終回應
    final_msg = messages[-1] if messages else None
    if final_msg and hasattr(final_msg, "content") and final_msg.content:
        st.subheader("💬 Agent 回應")
        content = final_msg.content
        if isinstance(content, list):
            content = "\n".join(
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        st.markdown(content)

    # 工作區檔案下載
    if files:
        st.divider()
        st.subheader("📁 工作區檔案")
        for path, file_data in files.items():
            filename = Path(path).name
            col_name, col_btn = st.columns([4, 1])
            col_name.markdown(f"📄 `{filename}`")
            if isinstance(file_data, bytes):
                raw = file_data
            elif hasattr(file_data, "encode"):
                raw = file_data.encode("utf-8")
            else:
                try:
                    from deepagents.backends.utils import file_data_to_string
                    raw = file_data_to_string(file_data).encode("utf-8")
                except Exception:
                    raw = str(file_data).encode("utf-8")
            col_btn.download_button(
                "下載",
                data=raw,
                file_name=filename,
                key=f"dl_{filename}",
            )
