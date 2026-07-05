# pages/8_Deep_Research.py
# -*- coding: utf-8 -*-
"""
深度研究報告生成頁面
- 簡單模式：gpt-researcher research_report + OpenAI WebSearch custom retriever
- 深度模式：gpt-researcher report_type="deep"（樹狀遞迴，可調廣度/深度）
- 支援三種資料來源：🌐 網路、📄 文件、🔀 混合
- 文件上傳 + DocStore 索引（仿 Home.py 模式）
- 探索文件主題後自動填入研究題目
"""

import os
import sys
import json
import time
import asyncio
import logging
import queue
import threading
from datetime import datetime

import streamlit as st
from openai import OpenAI

# ── 確保專案根目錄在 path（讓 Deep_Research package 可被 import）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── 環境變數（必須在 gpt-researcher import 前設定）
OPENAI_API_KEY = (
    st.secrets.get("OPENAI_API_KEY")
    or st.secrets.get("OPENAI_KEY")
    or os.getenv("OPENAI_API_KEY", "")
)
if not OPENAI_API_KEY:
    st.set_page_config(page_title="深度研究", page_icon="🔬", layout="wide")
    st.error("找不到 OpenAI API Key，請在 .streamlit/secrets.toml 設定 OPENAI_API_KEY。")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["LLM_PROVIDER"] = "openai"
os.environ["SMART_LLM_MODEL"] = "gpt-4.1"
os.environ["FAST_LLM_MODEL"] = "gpt-4.1-mini"
os.environ["EMBEDDING"] = "openai"
os.environ["EMBEDDING_MODEL"] = "text-embedding-3-small"
os.environ["RETRIEVER"] = "custom"
os.environ["RETRIEVER_ARG"] = "Deep_Research.openai_retriever.OpenAIWebRetriever"

# ── DocStore imports
from docstore import (
    FileRow,
    build_file_row_from_bytes,
    build_indices_incremental,
    doc_search_payload,
)

# ── LangChain Documents（hybrid 搜尋用）
try:
    from langchain_core.documents import Document as LCDocument
    HAS_LC = True
except ImportError:
    HAS_LC = False

# ── gpt-researcher
try:
    from gpt_researcher import GPTResearcher
    HAS_GPTR = True
except ImportError:
    HAS_GPTR = False

# ────────────────────────────────────────────────
# 頁面設定
# ────────────────────────────────────────────────
st.set_page_config(page_title="深度研究", page_icon="🔬", layout="wide")

client = OpenAI(api_key=OPENAI_API_KEY)

# ────────────────────────────────────────────────
# Session state 初始化
# ────────────────────────────────────────────────
st.session_state.setdefault("dr_file_rows", [])
st.session_state.setdefault("dr_file_bytes", {})
st.session_state.setdefault("dr_store", None)
st.session_state.setdefault("dr_processed_keys", set())
st.session_state.setdefault("dr_topics", [])
st.session_state.setdefault("dr_query", "")
st.session_state.setdefault("dr_report", "")
st.session_state.setdefault("dr_sources", [])
st.session_state.setdefault("dr_logs", [])

# ────────────────────────────────────────────────
# 工具函式
# ────────────────────────────────────────────────

def _has_store() -> bool:
    store = st.session_state.get("dr_store")
    try:
        return bool(store and getattr(store, "index", None) and store.index.ntotal > 0)
    except Exception:
        return False


def get_lc_docs(store, query: str, k: int = 20) -> list:
    """DocStore → LangChain Documents（供 gpt-researcher hybrid 模式使用）"""
    if not HAS_LC or not store or store.index.ntotal == 0:
        return []
    hits = doc_search_payload(client, store, query, k=k, difficulty="hard").get("hits", [])
    return [
        LCDocument(
            page_content=h["snippet"],
            metadata={"source": h.get("citation_token", ""), "title": h["title"]},
        )
        for h in hits
    ]


def discover_doc_topics(n_topics: int = 8) -> list[str]:
    """從 DocStore chunks 取樣，呼叫 LLM 萃取關鍵研究主題。"""
    store = st.session_state.get("dr_store")
    if not _has_store():
        return []
    sample_chunks = [c.text[:300] for c in store.chunks[:30]]
    sample_text = "\n---\n".join(sample_chunks)
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{
                "role": "user",
                "content": (
                    f"以下是文件節選：\n\n{sample_text}\n\n"
                    f"請從這些文件中萃取 {n_topics} 個適合深度研究的主題，"
                    "每個主題用 4–12 字描述，以 JSON 格式回傳，key 為 \"topics\"，"
                    "例如：{{\"topics\": [\"主題一\", \"主題二\"]}}"
                ),
            }],
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        for key in ("topics", "items", "results", "data"):
            if key in data and isinstance(data[key], list):
                return [str(t) for t in data[key][:n_topics]]
        for v in data.values():
            if isinstance(v, list):
                return [str(t) for t in v[:n_topics]]
    except Exception as e:
        st.warning(f"探索主題失敗：{e}")
    return []


# ── 背景執行緒 runner

class _LogSink(logging.Handler):
    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q

    def emit(self, record: logging.LogRecord):
        try:
            self.q.put_nowait(self.format(record))
        except Exception:
            pass


def _run_coro_in_thread(coro, out: dict):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        out["result"] = loop.run_until_complete(coro)
    except Exception as e:
        out["error"] = e
    finally:
        loop.close()


def _build_gptr_kwargs(query: str, research_mode: str, report_type: str) -> dict:
    """建立 GPTResearcher 的共用 kwargs，處理文件/混合模式。"""
    store = st.session_state.get("dr_store")
    kwargs: dict = {"query": query, "report_type": report_type}

    if research_mode == "docs" and _has_store():
        lc_docs = get_lc_docs(store, query)
        if lc_docs:
            kwargs["report_source"] = "langchain_documents"
            kwargs["documents"] = lc_docs
    elif research_mode == "hybrid" and _has_store():
        lc_docs = get_lc_docs(store, query)
        if lc_docs:
            kwargs["report_source"] = "hybrid"
            kwargs["documents"] = lc_docs

    return kwargs


# ── 簡單模式：research_report

async def _simple_research(query: str, research_mode: str):
    if not HAS_GPTR:
        raise ImportError("gpt-researcher 未安裝，請執行 pip install gpt-researcher")
    kwargs = _build_gptr_kwargs(query, research_mode, "research_report")
    researcher = GPTResearcher(**kwargs)
    await researcher.conduct_research()
    report = await researcher.write_report()
    try:
        sources = researcher.get_source_urls()
    except Exception:
        sources = []
    return report, sources


# ── 深度模式：report_type="deep"（樹狀遞迴探索）

async def _deep_research(
    query: str,
    research_mode: str,
    breadth: int = 4,
    depth: int = 2,
    concurrency: int = 4,
):
    if not HAS_GPTR:
        raise ImportError("gpt-researcher 未安裝，請執行 pip install gpt-researcher")

    # 深度模式參數必須在 GPTResearcher 初始化前設定
    os.environ["DEEP_RESEARCH_BREADTH"] = str(breadth)
    os.environ["DEEP_RESEARCH_DEPTH"] = str(depth)
    os.environ["DEEP_RESEARCH_CONCURRENCY"] = str(concurrency)

    kwargs = _build_gptr_kwargs(query, research_mode, "deep")
    researcher = GPTResearcher(**kwargs)
    await researcher.conduct_research()
    report = await researcher.write_report()
    try:
        sources = researcher.get_source_urls()
    except Exception:
        sources = []
    return report, sources


# ────────────────────────────────────────────────
# 主 UI
# ────────────────────────────────────────────────

st.title("🔬 深度研究報告")
st.caption("gpt-researcher + OpenAI Web Search，自動生成深度研究報告。")

# ── 文件上傳區（可選）
with st.expander("📁 文件上傳（可選，用於文件或混合研究模式）",
                  expanded=not _has_store()):

    uploaded = st.file_uploader(
        "上傳文件",
        type=["pdf", "docx", "doc", "pptx", "xlsx", "xls", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        existing = {(r.name, r.bytes_len) for r in st.session_state.dr_file_rows}
        new_count = 0
        for f in uploaded:
            data = f.read()
            if (f.name, len(data)) not in existing:
                row = build_file_row_from_bytes(filename=f.name, data=data)
                st.session_state.dr_file_rows.append(row)
                st.session_state.dr_file_bytes[row.file_id] = data
                new_count += 1
        if new_count:
            st.toast(f"已新增 {new_count} 份文件")

    col_build, col_clear = st.columns([2, 1])

    with col_build:
        if st.session_state.dr_file_rows:
            if st.button("⚙️ 建立文件索引", use_container_width=True):
                with st.spinner("建立索引中..."):
                    store, stats, pkeys = build_indices_incremental(
                        client,
                        file_rows=st.session_state.dr_file_rows,
                        file_bytes_map=st.session_state.dr_file_bytes,
                        store=st.session_state.dr_store,
                        processed_keys=st.session_state.dr_processed_keys,
                    )
                    st.session_state.dr_store = store
                    st.session_state.dr_processed_keys = pkeys
                    st.session_state.dr_topics = []
                new_chunks = stats.get("new_chunks", 0)
                st.success(f"索引完成！新增 {new_chunks} 個 chunks。")
        else:
            st.info("尚未上傳文件")

    with col_clear:
        if st.session_state.dr_file_rows and st.button("🗑️ 清除", use_container_width=True):
            st.session_state.dr_file_rows = []
            st.session_state.dr_file_bytes = {}
            st.session_state.dr_store = None
            st.session_state.dr_processed_keys = set()
            st.session_state.dr_topics = []
            st.session_state.dr_query = ""
            st.rerun()

    # 索引狀態
    if _has_store():
        store = st.session_state.dr_store
        n_files = len(st.session_state.dr_file_rows)
        n_chunks = store.index.ntotal
        st.success(f"✅ 已索引 {n_files} 份文件 / {n_chunks} 個 chunks")

        if st.button("🔍 探索文件主題", use_container_width=False):
            with st.spinner("分析文件主題中..."):
                topics = discover_doc_topics()
            if topics:
                st.session_state.dr_topics = topics
                st.rerun()
            else:
                st.warning("無法萃取主題，請確認文件已正確索引。")

# ── 探索到的主題（點選填入）
if st.session_state.dr_topics:
    st.markdown("**點選主題自動填入研究題目：**")
    n = len(st.session_state.dr_topics)
    cols = st.columns(min(4, n))
    for i, topic in enumerate(st.session_state.dr_topics):
        if cols[i % 4].button(topic, key=f"dr_topic_{i}"):
            st.session_state.dr_query = topic
            st.rerun()

st.divider()

# ── 研究主題輸入
query = st.text_input(
    "研究主題",
    value=st.session_state.dr_query,
    placeholder="輸入研究主題，或上傳文件後點選「探索文件主題」自動填入",
)
st.session_state.dr_query = query

# ── 模式選擇
col_src, col_agent = st.columns(2)
with col_src:
    src_options = ["🌐 網路", "📄 文件", "🔀 混合"]
    research_mode_label = st.radio(
        "資料來源",
        src_options,
        horizontal=True,
        help="文件/混合模式需先上傳文件並建立索引",
    )
    research_mode = {"🌐 網路": "web", "📄 文件": "docs", "🔀 混合": "hybrid"}[research_mode_label]

with col_agent:
    agent_options = ["⚡ 簡單模式（2–4 分鐘）", "🌲 深度模式（5–10 分鐘）"]
    agent_mode_label = st.radio(
        "研究模式",
        agent_options,
        horizontal=True,
        help="深度模式採樹狀遞迴探索，廣度×深度路徑同時進行，報告更全面",
        disabled=not HAS_GPTR,
    )

# ── 深度模式參數（僅在選擇深度模式時顯示）
breadth, depth_lvl, concurrency = 4, 2, 4
if agent_mode_label.startswith("🌲"):
    col_b, col_d, col_c = st.columns(3)
    breadth = col_b.slider(
        "探索廣度 breadth", 1, 8, 4,
        help="平行探索的路徑數量，越大越全面但越慢"
    )
    depth_lvl = col_d.slider(
        "遞迴深度 depth", 1, 4, 2,
        help="每條路徑的遞迴層數，越深越詳細但越慢"
    )
    concurrency = col_c.slider(
        "並發數 concurrency", 1, 8, 4,
        help="最大同時執行的搜尋數"
    )

# ── 可用性提示
if not HAS_GPTR:
    st.warning("gpt-researcher 未安裝，請執行 `pip install gpt-researcher`")
if research_mode in ("docs", "hybrid") and not _has_store():
    st.info("已選文件模式，請先上傳文件並建立索引；若未建立索引將自動改為網路搜尋。")

st.divider()

# ── 開始研究按鈕
start_btn = st.button("🔍 開始研究", type="primary", disabled=not query.strip())

if start_btn and query.strip():
    st.session_state.dr_report = ""
    st.session_state.dr_sources = []
    st.session_state.dr_logs = []

    progress_q: queue.Queue = queue.Queue()

    # 掛載 log sink
    _sink = _LogSink(progress_q)
    _sink.setFormatter(logging.Formatter("%(message)s"))
    for logger_name in ("gpt_researcher", "gpt_researcher.master.research_agent"):
        _logger = logging.getLogger(logger_name)
        if not any(isinstance(h, _LogSink) for h in _logger.handlers):
            _logger.addHandler(_sink)

    # 選擇 coroutine
    if agent_mode_label.startswith("🌲"):
        coro = _deep_research(
            query.strip(), research_mode,
            breadth=breadth, depth=depth_lvl, concurrency=concurrency,
        )
    else:
        coro = _simple_research(query.strip(), research_mode)

    out: dict = {}
    t = threading.Thread(target=_run_coro_in_thread, args=(coro, out), daemon=True)
    t.start()

    # ── 進度顯示
    with st.status("研究進行中...", expanded=True) as status_box:
        ph_log = st.empty()
        logs: list[str] = []
        while t.is_alive():
            while not progress_q.empty():
                try:
                    logs.append(progress_q.get_nowait())
                except queue.Empty:
                    break
            if logs:
                ph_log.markdown("\n".join(f"- {l}" for l in logs[-15:]))
            time.sleep(0.35)

        while not progress_q.empty():
            try:
                logs.append(progress_q.get_nowait())
            except queue.Empty:
                break

        if "error" in out:
            status_box.update(label="研究失敗", state="error")
            st.error(f"錯誤：{out['error']}")
        else:
            status_box.update(label="研究完成！", state="complete")
            st.session_state.dr_logs = logs

    # ── 儲存結果
    if "result" in out:
        result = out["result"]
        if isinstance(result, tuple):
            report, sources = result
        else:
            report, sources = str(result), []
        st.session_state.dr_report = report
        st.session_state.dr_sources = sources or []

# ────────────────────────────────────────────────
# 報告輸出區
# ────────────────────────────────────────────────
if st.session_state.dr_report:
    st.divider()
    st.subheader("📄 研究報告")
    st.markdown(st.session_state.dr_report)

    if st.session_state.dr_sources:
        with st.expander(f"🔗 參考來源（{len(st.session_state.dr_sources)} 筆）"):
            for i, url in enumerate(st.session_state.dr_sources, 1):
                st.markdown(f"{i}. {url}")

    st.download_button(
        "⬇️ 下載報告（.md）",
        data=st.session_state.dr_report,
        file_name=f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
    )

    if st.session_state.dr_logs:
        with st.expander("📋 研究過程 log"):
            st.text("\n".join(st.session_state.dr_logs))
