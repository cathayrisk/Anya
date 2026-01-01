# pages/deep_agents.py
# -*- coding: utf-8 -*-
import os
import re
import json
import uuid
import asyncio
import tempfile
import threading
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Literal, Optional

import streamlit as st
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    TextLoader,
)

from langgraph.checkpoint.memory import MemorySaver
from deepagents import create_deep_agent


# =========================================================
# ✅ 壓掉 Streamlit ScriptRunContext 洗版 warning（不影響功能）
# =========================================================
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.state.session_state_proxy").setLevel(logging.ERROR)


# =========================================================
# Global constraints
# =========================================================
BUDGET_LIMIT = 3            # KB/Web 各 3 次/輪
MAX_VERIFY_ROUNDS = 1       # 補強 1 輪（保留）
MAX_REPLAN_ROUNDS = 1       # replan 1 次（replan 會重置預算）
WEB_MEM_MAX = 500           # ✅ 只保留最近 500 筆 web 記憶做 embeddings 檢索


# =========================================================
# Persistence (跨對話短期記憶：jsonl + raw files + FAISS index)
# =========================================================
MEM_ROOT = Path(".agent_memory")
MEM_ROOT.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    # ✅ 修掉 utcnow() deprecation：timezone-aware UTC
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _jsonl_append(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _jsonl_tail(path: Path, max_lines: int = 200) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()[-max_lines:]
    out = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out


def _mem_paths(namespace: str) -> Dict[str, Path]:
    base = MEM_ROOT / namespace
    return {
        "base": base,
        "chat": base / "chat.jsonl",
        "web": base / "web.jsonl",
        "kb": base / "kb.jsonl",
        "failures": base / "failures.jsonl",
        "web_raw_dir": base / "web_raw",
        "kb_raw_dir": base / "kb_raw",
        # ✅ web 記憶 embeddings index
        "web_index_dir": base / "web_mem_index",
        "web_index_manifest": base / "web_mem_index" / "manifest.json",
    }


# =========================================================
# Session init
# =========================================================
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # UI chat only

if "main_thread_id" not in st.session_state:
    st.session_state["main_thread_id"] = str(uuid.uuid4())

# ✅ 依你指定：main/web_tool/reasoning 全部 gpt-5.2
st.session_state.setdefault(
    "settings",
    {
        "web_allowed": True,
        "main_model": "gpt-5.2",
        "web_tool_model": "gpt-5.2",
        "reasoning_model": "gpt-5.2",
        "max_parallel_todos": 3,
        "direct_do_planning": False,
        "memory_namespace": st.secrets.get("MEMORY_NAMESPACE", "default"),
    },
)

# 每次使用者提問 = 一個 run_id（跨對話遞增）
if "run_seq" not in st.session_state:
    st.session_state["run_seq"] = 0

# round 只用於同一 run 的 initial/replan
if "budget" not in st.session_state:
    st.session_state["budget"] = {"vector_calls": 0, "web_calls": 0, "round": 0, "run_id": 0}

# ✅ lock 用 setdefault，比較不會因為 thread 時序/初始化而缺 key
st.session_state.setdefault("budget_lock", threading.Lock())

# UI todo list
if "last_todos" not in st.session_state:
    st.session_state["last_todos"] = []

# 記憶（會從 jsonl 載入）
if "memory" not in st.session_state:
    st.session_state["memory"] = {"web": [], "kb": [], "failures": []}

# tool context：讓 tool 記錄 todo_id、round、run_id、focus
if "tool_context" not in st.session_state:
    st.session_state["tool_context"] = {"run_id": 0, "round": 0, "todo_id": None, "focus": ""}

# ✅ web 記憶 embeddings index（FAISS）
if "web_mem" not in st.session_state:
    st.session_state["web_mem"] = {
        "vectorstore": None,     # FAISS vectorstore
        "doc_ids": [],           # 對應順序（同時也用來判斷是否已索引）
        "loaded": False,
    }

st.session_state.setdefault("web_mem_lock", threading.Lock())

# agents cache
if "agents" not in st.session_state:
    st.session_state["agents"] = None

if "agents_settings_key" not in st.session_state:
    st.session_state["agents_settings_key"] = None

# OpenAI key
if "OPENAI_API_KEY" not in os.environ and "OPENAI_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_KEY"]


# =========================================================
# Helpers
# =========================================================
def _main_model() -> str:
    return str(st.session_state["settings"].get("main_model", "gpt-5.2"))


def _web_tool_model() -> str:
    return str(st.session_state["settings"].get("web_tool_model", "gpt-5.2"))


def _reasoning_model() -> str:
    return str(st.session_state["settings"].get("reasoning_model", "gpt-5.2"))


def _web_allowed() -> bool:
    return bool(st.session_state["settings"].get("web_allowed", True))


def _direct_do_planning() -> bool:
    return bool(st.session_state["settings"].get("direct_do_planning", False))


def _namespace() -> str:
    ns = str(st.session_state["settings"].get("memory_namespace", "default")).strip() or "default"
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", ns)


def _reset_budget_new_round(round_id: int):
    st.session_state["budget"]["vector_calls"] = 0
    st.session_state["budget"]["web_calls"] = 0
    st.session_state["budget"]["round"] = round_id
    st.session_state["tool_context"]["round"] = round_id


def _start_new_run():
    st.session_state["run_seq"] += 1
    st.session_state["budget"]["run_id"] = st.session_state["run_seq"]
    st.session_state["tool_context"]["run_id"] = st.session_state["run_seq"]
    _reset_budget_new_round(round_id=0)


def _get_budget_lock() -> threading.Lock:
    # ✅ 就算某些 thread 讀 session_state 出狀況，也不要炸
    try:
        lock = st.session_state.get("budget_lock", None)
    except Exception:
        lock = None
    return lock or threading.Lock()


def _try_inc_budget(kind: Literal["vector", "web"]) -> bool:
    with _get_budget_lock():
        if kind == "vector":
            if st.session_state["budget"]["vector_calls"] >= BUDGET_LIMIT:
                return False
            st.session_state["budget"]["vector_calls"] += 1
            return True
        else:
            if st.session_state["budget"]["web_calls"] >= BUDGET_LIMIT:
                return False
            st.session_state["budget"]["web_calls"] += 1
            return True


def render_todos_md(todos: List[Dict[str, Any]]) -> str:
    lines = ["### Todo 進度\n"]
    for t in todos:
        status = t.get("status", "pending")
        checked = "x" if status == "completed" else " "
        suffix = ""
        if status == "failed":
            suffix = "  **(failed)**"
        elif status == "in_progress":
            suffix = "  _(in progress)_"
        lines.append(f"- [{checked}] {t['id']}: {t['todo']}{suffix}")
    return "\n".join(lines)


def _content_to_text(content: Any) -> str:
    """
    ✅ Responses API 可能回傳 content blocks（list[dict]），這裡統一抽成可 parse 的字串。
    - str: 原樣回傳
    - list: 抽所有 type=text 的 text 串起來
    - dict: 取 text 或 json.dumps
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                t = item.get("type")
                if t == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    # 其他 block（reasoning/tool_call/result）通常不含 JSON；但保底
                    if "text" in item:
                        parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join([p for p in parts if p.strip()]).strip()
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text", ""))
        try:
            return json.dumps(content, ensure_ascii=False)
        except Exception:
            return str(content)
    return str(content)


def safe_json_from_text(text: Any) -> Optional[dict]:
    """
    ✅ 修正你遇到的 TypeError：
    text 可能是 list（Responses content blocks），先轉字串再 parse。
    """
    s = _content_to_text(text)
    if not s:
        return None
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def memory_digest(max_web: int = 5, max_kb: int = 5, max_fail: int = 10) -> Dict[str, Any]:
    web = st.session_state["memory"]["web"][-max_web:]
    kb = st.session_state["memory"]["kb"][-max_kb:]
    failures = st.session_state["memory"]["failures"][-max_fail:]
    return {"web_recent": web, "kb_recent": kb, "failures_recent": failures}


def load_persistent_memory():
    paths = _mem_paths(_namespace())
    st.session_state["memory"]["web"] = _jsonl_tail(paths["web"], 600)       # 多載一些，index 會自己截 500
    st.session_state["memory"]["kb"] = _jsonl_tail(paths["kb"], 200)
    st.session_state["memory"]["failures"] = _jsonl_tail(paths["failures"], 400)


def persist_chat_message(role: str, content: str):
    paths = _mem_paths(_namespace())
    rec = {"ts": _now_iso(), "role": role, "content": content}
    _jsonl_append(paths["chat"], rec)


def persist_failure(kind: str, detail: str, todo_id: Optional[str] = None):
    paths = _mem_paths(_namespace())
    rec = {
        "ts": _now_iso(),
        "run_id": st.session_state["tool_context"]["run_id"],
        "round": st.session_state["tool_context"]["round"],
        "todo_id": todo_id or st.session_state["tool_context"].get("todo_id"),
        "type": kind,
        "detail": detail,
    }
    _jsonl_append(paths["failures"], rec)
    st.session_state["memory"]["failures"].append(rec)


def persist_kb_record(query: str, sources: List[str], snippets: List[str], todo_id: Optional[str] = None):
    paths = _mem_paths(_namespace())
    rec = {
        "ts": _now_iso(),
        "run_id": st.session_state["tool_context"]["run_id"],
        "round": st.session_state["tool_context"]["round"],
        "todo_id": todo_id or st.session_state["tool_context"].get("todo_id"),
        "focus": st.session_state["tool_context"].get("focus", "")[:200],
        "query": query,
        "sources": list(dict.fromkeys([s for s in sources if s]))[:10],
        "snippets": snippets[:5],
    }
    _jsonl_append(paths["kb"], rec)
    st.session_state["memory"]["kb"].append(rec)


def persist_web_record(rec: dict):
    """rec 必須包含 doc_id/raw_path/summary_5_lines/citations/query 等。"""
    paths = _mem_paths(_namespace())
    _jsonl_append(paths["web"], rec)
    st.session_state["memory"]["web"].append(rec)


# =========================================================
# LLM factory
# - ✅ 全部使用 Responses API（use_responses_api=True）
# - ✅ 只有「推理需求高」才加 reasoning={"effort":"medium"}
# =========================================================
def llm_main(temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(
        model=_main_model(),
        temperature=temperature,
        use_responses_api=True,
        max_completion_tokens=None,
    )


def llm_web_tool(temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(
        model=_web_tool_model(),
        temperature=temperature,
        use_responses_api=True,
        max_completion_tokens=None,
    )


def llm_reasoning(temperature: float = 0) -> ChatOpenAI:
    # ✅ 只設定 effort=medium（其餘不設定）
    reasoning = {"effort": "medium"}
    return ChatOpenAI(
        model=_reasoning_model(),
        temperature=temperature,
        use_responses_api=True,
        reasoning=reasoning,
        max_completion_tokens=None,
    )


# =========================================================
# ✅ Web 記憶 embeddings index（FAISS）管理
# =========================================================
def _web_mem_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small")


def _web_mem_record_to_text(rec: dict) -> str:
    """
    拿來做 embeddings 的「可檢索文字」：
    - 不 embedding raw 全文（避免太慢、也避免太長）
    - 用 query + focus + 5 行摘要 + citations title/url
    """
    q = (rec.get("query") or "").strip()
    focus = (rec.get("focus") or "").strip()
    s5 = (rec.get("summary_5_lines") or "").strip()
    cites = rec.get("citations") or []
    cite_text = " | ".join([f"{c.get('title','')} {c.get('url','')}".strip() for c in cites[:5]])
    return f"query: {q}\nfocus: {focus}\nsummary:\n{s5}\ncitations: {cite_text}".strip()


def _web_mem_manifest_load(paths: Dict[str, Path]) -> List[str]:
    if not paths["web_index_manifest"].exists():
        return []
    try:
        return json.loads(paths["web_index_manifest"].read_text(encoding="utf-8"))
    except Exception:
        return []


def _web_mem_manifest_save(paths: Dict[str, Path], doc_ids: List[str]) -> None:
    paths["web_index_dir"].mkdir(parents=True, exist_ok=True)
    paths["web_index_manifest"].write_text(json.dumps(doc_ids, ensure_ascii=False), encoding="utf-8")


def web_mem_load_or_build():
    with st.session_state["web_mem_lock"]:
        if st.session_state["web_mem"]["loaded"]:
            return

        paths = _mem_paths(_namespace())
        emb = _web_mem_embeddings()
        idx_dir = paths["web_index_dir"]

        if idx_dir.exists():
            try:
                vs = FAISS.load_local(
                    str(idx_dir),
                    emb,
                    allow_dangerous_deserialization=True,
                )
                doc_ids = _web_mem_manifest_load(paths)
                st.session_state["web_mem"]["vectorstore"] = vs
                st.session_state["web_mem"]["doc_ids"] = doc_ids
                st.session_state["web_mem"]["loaded"] = True
                return
            except Exception:
                pass

        rebuild_web_mem_index()


def rebuild_web_mem_index():
    paths = _mem_paths(_namespace())
    emb = _web_mem_embeddings()

    records = (st.session_state["memory"]["web"] or [])[-WEB_MEM_MAX:]
    docs: List[Document] = []
    doc_ids: List[str] = []

    for rec in records:
        doc_id = rec.get("doc_id")
        raw_path = rec.get("raw_path")
        if not doc_id or not raw_path:
            continue
        t = _web_mem_record_to_text(rec)
        docs.append(Document(page_content=t, metadata={"doc_id": doc_id, "raw_path": raw_path}))
        doc_ids.append(doc_id)

    if docs:
        vs = FAISS.from_documents(docs, emb)
    else:
        vs = FAISS.from_texts(["(empty)"], emb, metadatas=[{"doc_id": "__empty__", "raw_path": ""}])
        doc_ids = ["__empty__"]

    paths["web_index_dir"].mkdir(parents=True, exist_ok=True)
    vs.save_local(str(paths["web_index_dir"]))
    _web_mem_manifest_save(paths, doc_ids)

    st.session_state["web_mem"]["vectorstore"] = vs
    st.session_state["web_mem"]["doc_ids"] = doc_ids
    st.session_state["web_mem"]["loaded"] = True


def add_web_mem_record_to_index(rec: dict):
    with st.session_state["web_mem_lock"]:
        vs: Optional[FAISS] = st.session_state["web_mem"]["vectorstore"]
        doc_ids: List[str] = st.session_state["web_mem"]["doc_ids"]

        doc_id = rec.get("doc_id")
        raw_path = rec.get("raw_path")
        if not doc_id or not raw_path:
            return

        if vs is None or not st.session_state["web_mem"]["loaded"]:
            web_mem_load_or_build()
            vs = st.session_state["web_mem"]["vectorstore"]
            doc_ids = st.session_state["web_mem"]["doc_ids"]

        if doc_id in doc_ids:
            return

        if len([x for x in doc_ids if x != "__empty__"]) >= WEB_MEM_MAX:
            rebuild_web_mem_index()
            return

        text = _web_mem_record_to_text(rec)
        vs.add_documents([Document(page_content=text, metadata={"doc_id": doc_id, "raw_path": raw_path})])
        doc_ids.append(doc_id)

        st.session_state["web_mem"]["vectorstore"] = vs
        st.session_state["web_mem"]["doc_ids"] = doc_ids

        paths = _mem_paths(_namespace())
        paths["web_index_dir"].mkdir(parents=True, exist_ok=True)
        vs.save_local(str(paths["web_index_dir"]))
        _web_mem_manifest_save(paths, doc_ids)


# =========================================================
# Tools
# =========================================================
@tool
def kb_available() -> bool:
    """檢查目前是否已建立/載入知識庫向量庫（st.session_state['vectorstore']）。"""
    return st.session_state.get("vectorstore") is not None


@tool
def web_allowed() -> bool:
    """回傳目前設定是否允許使用網路搜尋（OpenAI web_search_preview）。"""
    return _web_allowed()


@tool
def web_memory_search(query: str, k: int = 5) -> Dict[str, Any]:
    """
    ✅ embeddings 記憶檢索：只查「web 記憶 index」，不消耗 web budget。
    回傳 top-k 的 raw_path + doc_id + summary_5_lines + citations + score。
    """
    web_mem_load_or_build()
    vs: Optional[FAISS] = st.session_state["web_mem"]["vectorstore"]
    if vs is None:
        return {"ok": False, "error": "web 記憶 index 尚未建立。", "results": []}

    try:
        docs_scores = vs.similarity_search_with_score(query, k=k)
    except Exception as e:
        return {"ok": False, "error": f"web 記憶檢索失敗：{e}", "results": []}

    web_records = st.session_state["memory"]["web"]

    def find_rec(doc_id: str) -> Optional[dict]:
        for r in reversed(web_records):
            if r.get("doc_id") == doc_id:
                return r
        return None

    results = []
    for doc, score in docs_scores:
        meta = doc.metadata or {}
        doc_id = meta.get("doc_id", "")
        raw_path = meta.get("raw_path", "")
        r = find_rec(doc_id) or {}
        results.append({
            "doc_id": doc_id,
            "raw_path": raw_path or r.get("raw_path", ""),
            "query": r.get("query", ""),
            "focus": r.get("focus", ""),
            "summary_5_lines": r.get("summary_5_lines", ""),
            "citations": r.get("citations", []),
            "score": score,
        })

    return {"ok": True, "results": results, "k": k}


@tool
def vector_search(query: str, k: int = 6) -> Dict[str, Any]:
    """對使用者上傳的 KB 向量庫做檢索（會消耗 KB budget）。"""
    if not _try_inc_budget("vector"):
        err = f"KB 檢索超過硬性預算（上限 {BUDGET_LIMIT} 次/輪）。"
        persist_failure("budget_exceeded_kb", err)
        return {"ok": False, "error": err, "results": []}

    vs = st.session_state.get("vectorstore")
    if vs is None:
        err = "目前沒有向量庫（尚未上傳文件）。"
        persist_failure("kb_missing", err)
        return {"ok": False, "error": err, "results": []}

    retriever = vs.as_retriever(search_kwargs={"k": k})
    docs: List[Document] = retriever.invoke(query) or []

    results = []
    sources: List[str] = []
    snippets: List[str] = []

    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "")
        src_short = os.path.basename(str(src).replace("\\", "/")) if src else ""
        content = (d.page_content or "").strip()
        results.append({"rank": i, "source": src_short, "content": content, "metadata": d.metadata})
        if src_short:
            sources.append(src_short)
        if content:
            snippets.append(content[:300])

    persist_kb_record(query=query, sources=sources, snippets=snippets)
    return {"ok": True, "query": query, "results": results, "budget": dict(st.session_state["budget"])}


def _extract_citations_from_content_blocks(content_blocks: List[dict]) -> List[dict]:
    seen = set()
    citations: List[dict] = []
    for b in content_blocks or []:
        if b.get("type") != "text":
            continue
        for a in b.get("annotations", []) or []:
            if a.get("type") != "citation":
                continue
            url = a.get("url") or ""
            title = a.get("title") or ""
            key = (title, url)
            if key in seen:
                continue
            seen.add(key)
            citations.append({"title": title, "url": url})
    return citations


def _summarize_5_lines(text: str, focus: str) -> str:
    llm = llm_main(temperature=0)
    system = "你是摘要器。請用繁中，輸出剛好 5 行，每行一個重點，不要多也不要少。"
    human = f"""焦點（本輪在意的點）：
{focus}

內容：
{text[:6000]}

請輸出 5 行重點（每行一點）。"""
    resp = llm.invoke([("system", system), ("human", human)])
    out = getattr(resp, "text", None)
    if not out:
        out = _content_to_text(getattr(resp, "content", None))
    lines = [ln.strip() for ln in str(out).splitlines() if ln.strip()]
    return "\n".join(lines[:5]) if lines else "（摘要失敗）"


@tool
def openai_web_search(query: str) -> Dict[str, Any]:
    """使用 OpenAI 的 web_search_preview 做即時網路搜尋（會消耗 Web budget），並把結果存入 web 記憶。"""
    if not web_allowed():
        err = "網路搜尋已被禁用（web_allowed=false）。"
        persist_failure("web_disabled", err)
        return {"ok": False, "error": err, "text": "", "citations": []}

    if not _try_inc_budget("web"):
        err = f"Web 搜尋超過硬性預算（上限 {BUDGET_LIMIT} 次/輪）。"
        persist_failure("budget_exceeded_web", err)
        return {"ok": False, "error": err, "text": "", "citations": []}

    tool_def = {"type": "web_search_preview"}
    llm = llm_web_tool(temperature=0).bind_tools([tool_def])

    prompt = (
        "請使用 web search 搜尋並整理答案。\n"
        f"Query: {query}\n\n"
        "輸出要求：\n"
        "1) 5-10 點重點（條列）\n"
        "2) 內容要可被 citations 支持（annotations 會附 citation）\n"
        "3) 不要輸出無根據推測\n"
    )
    response = llm.invoke(prompt)
    content_blocks = getattr(response, "content_blocks", None)

    if content_blocks is None:
        citations: List[dict] = []
        text = getattr(response, "text", None) or _content_to_text(getattr(response, "content", None))
    else:
        citations = _extract_citations_from_content_blocks(content_blocks)
        text = getattr(response, "text", None) or ""

    paths = _mem_paths(_namespace())
    paths["web_raw_dir"].mkdir(parents=True, exist_ok=True)
    doc_id = f"run{st.session_state['tool_context']['run_id']}_r{st.session_state['tool_context']['round']}_{uuid.uuid4().hex[:8]}"
    raw_path = paths["web_raw_dir"] / f"{doc_id}.txt"
    raw_path.write_text(text, encoding="utf-8")

    focus = st.session_state["tool_context"].get("focus", "")
    summary_5 = _summarize_5_lines(text=text, focus=focus)

    rec = {
        "ts": _now_iso(),
        "run_id": st.session_state["tool_context"]["run_id"],
        "round": st.session_state["tool_context"]["round"],
        "todo_id": st.session_state["tool_context"].get("todo_id"),
        "focus": focus[:200],
        "doc_id": doc_id,
        "query": query,
        "summary_5_lines": summary_5,
        "citations": citations[:10],
        "raw_path": str(raw_path),
    }
    persist_web_record(rec)
    add_web_mem_record_to_index(rec)

    return {
        "ok": True,
        "query": query,
        "text": text,
        "citations": citations,
        "summary_5_lines": summary_5,
        "raw_path": str(raw_path),
        "doc_id": doc_id,
        "budget": dict(st.session_state["budget"]),
    }


@tool
def resummarize_web(raw_path: str, focus: str) -> Dict[str, Any]:
    """針對既有的 web raw 檔，依新的 focus 重新產生 5 行摘要（不耗 web budget）。"""
    p = Path(raw_path)
    if not p.exists():
        err = f"找不到 raw 檔：{raw_path}"
        persist_failure("resummarize_missing_raw", err)
        return {"ok": False, "error": err, "summary_5_lines": ""}

    text = p.read_text(encoding="utf-8", errors="ignore")
    summary_5 = _summarize_5_lines(text=text, focus=focus)
    return {"ok": True, "raw_path": raw_path, "summary_5_lines": summary_5}


# =========================================================
# Mode classify
# =========================================================
class NeedModeOut(BaseModel):
    mode: Literal["direct", "rag"] = Field(...)
    reason: str = Field(...)


async def classify_need_mode(user_question: str) -> NeedModeOut:
    # ✅ 推理需求高 → reasoning=medium
    llm = llm_reasoning(temperature=0).with_structured_output(NeedModeOut)
    system = """你是流程分類器：判斷這題要走 direct 或 rag。
- direct：純聊天/推理/寫作/格式化/程式建議（不一定需要外部證據）。
- rag：需要引用文件或外部來源才可靠（規範/數據/最新資訊/需查證）。
"""
    return llm.invoke([("system", system), ("human", user_question)])


# =========================================================
# Agents builder (cached)
# =========================================================
def build_agents() -> Dict[str, Any]:
    # ✅ 推理需求高的 agents → gpt-5.2 + reasoning medium
    reasoning_model = llm_reasoning(temperature=0)
    # 其餘不設定 reasoning（但仍用 Responses API）
    base_model = llm_main(temperature=0)

    checkpointer = MemorySaver()

    planner_system = """你是 Planning Agent（繁中）。你有 planning（write_todos）與 filesystem（write_file）。
請產生 3~7 個 todo，寫入 /workspace/todos.json（JSON 格式：{"todos":[{"id":"T1","todo":"..."}]}），最後回覆：已完成規劃。"""
    planner_agent = create_deep_agent(model=reasoning_model, tools=[], system_prompt=planner_system, checkpointer=checkpointer, name="planner", debug=False)

    todo_worker_system = f"""你是 Todo Worker（繁中）。工具：
- vector_search（KB）
- web_memory_search（Web 記憶 embeddings 檢索，不耗 web budget）
- resummarize_web（針對舊 raw 用新 focus 重摘要）
- openai_web_search（真的上網，耗 web budget）
- kb_available、web_allowed

策略（強制，按順序）：
1) 若 kb_available=true：先用 vector_search（1 次；必要時最多 2 次）。
2) 若 KB 明顯不足：先用 web_memory_search 找「以前查過的網路資料」(top 3~5)。
   - 若找到合適的 raw_path：用 resummarize_web(raw_path, focus) 針對本次重點重摘要，補齊 todo。
3) 若仍不足且 web_allowed=true：最後才用 openai_web_search（1 次；最多 2 次不建議）。
4) 若 todo 不需要查證（整理/改寫/推理/寫作），可以不使用工具直接完成（省預算）。

輸出 JSON（盡量只輸出 JSON）：
{{
  "summary": "...",
  "used_kb": true/false,
  "used_web_memory": true/false,
  "used_web_live": true/false,
  "sources": {{
    "kb_sources": ["..."],
    "web_urls": ["https://..."],
    "web_raw_paths_used": ["..."]
  }},
  "notes": "..."
}}

硬預算提醒：KB/Web live 各最多 {BUDGET_LIMIT} 次/輪（工具會拒絕超額）。
"""

    todo_worker_agent = create_deep_agent(
        model=base_model,
        tools=[kb_available, web_allowed, vector_search, web_memory_search, resummarize_web, openai_web_search],
        system_prompt=todo_worker_system,
        checkpointer=checkpointer,
        name="todo_worker",
        debug=False,
    )

    synth_system = """你是統整回答 Agent（繁中）。
輸入：user_question + todo_results + budget + memory_digest（含過去錯誤與 web/kb 記憶摘要）
輸出格式：
A. 回答
B. 依據/來源（檔名或 URL）
C. 限制與建議"""
    synth_agent = create_deep_agent(model=reasoning_model, tools=[], system_prompt=synth_system, checkpointer=checkpointer, name="synth", debug=False)

    direct_fast_system = """你是 Direct 回應助理（繁中）。
你可用工具：vector_search、web_memory_search、resummarize_web、openai_web_search、kb_available、web_allowed。
策略（強制，按順序）：先 KB → 再 web_memory_search → 不足才 web_search（若允許）；不需查證就不要用工具。
輸出：最終回答 +（若有）來源。"""
    direct_fast_agent = create_deep_agent(
        model=reasoning_model,
        tools=[kb_available, web_allowed, vector_search, web_memory_search, resummarize_web, openai_web_search],
        system_prompt=direct_fast_system,
        checkpointer=checkpointer,
        name="direct_fast",
        debug=False,
    )

    verifier_system = """你是 Verifier（繁中），不使用工具。
輸入：user_question、draft_answer、todo_results、budget、web_allowed、memory_digest。
輸出 JSON：
{"needs_fix":true/false,"fix_type":"rewrite_only"|"need_more_research"|"replan","missing_points":[...],"supplemental_todos":[...], "replan_goal":"...", "notes":"..."}"""
    verifier_agent = create_deep_agent(model=reasoning_model, tools=[], system_prompt=verifier_system, checkpointer=checkpointer, name="verifier", debug=False)

    replanner_system = """你是 Replanner（繁中），不使用工具。
輸入：user_question、previous_todos、todo_results、verifier_report、memory_digest（含 failures）。
輸出 JSON：{"todos":[{"id":"R1","todo":"..."}]}"""
    replanner_agent = create_deep_agent(model=reasoning_model, tools=[], system_prompt=replanner_system, checkpointer=checkpointer, name="replanner", debug=False)

    refiner_system = """你是 Refiner（繁中），不使用工具。
輸入：user_question、draft_answer、todo_results、verifier_report、budget、memory_digest。
輸出最終答案（非 JSON），格式：
A. 回答
B. 依據/來源
C. 限制與建議"""
    refiner_agent = create_deep_agent(model=reasoning_model, tools=[], system_prompt=refiner_system, checkpointer=checkpointer, name="refiner", debug=False)

    return {
        "planner": planner_agent,
        "todo_worker": todo_worker_agent,
        "synth": synth_agent,
        "direct_fast": direct_fast_agent,
        "verifier": verifier_agent,
        "replanner": replanner_agent,
        "refiner": refiner_agent,
    }


def ensure_agents():
    settings_key = tuple(sorted(st.session_state["settings"].items()))
    if st.session_state["agents"] is None or st.session_state["agents_settings_key"] != settings_key:
        st.session_state["agents"] = build_agents()
        st.session_state["agents_settings_key"] = settings_key


# =========================================================
# Planning -> todos
# =========================================================
async def plan_todos(user_question: str, history_messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    ensure_agents()
    planner = st.session_state["agents"]["planner"]
    config = {"configurable": {"thread_id": st.session_state["main_thread_id"] + ":planner"}}
    result = await planner.ainvoke({"messages": history_messages + [{"role": "user", "content": user_question}]}, config=config)

    files = result.get("files") or {}
    raw = files.get("/workspace/todos.json", "")
    todos: List[Dict[str, str]] = []

    if raw:
        try:
            data = json.loads(raw)
            for i, t in enumerate(data.get("todos", []), start=1):
                tid = (t.get("id") or f"T{i}").strip()
                todo = (t.get("todo") or "").strip()
                if todo:
                    todos.append({"id": tid, "todo": todo})
        except Exception:
            pass

    if not todos:
        data = safe_json_from_text(getattr(result["messages"][-1], "content", None)) or {}
        for i, t in enumerate(data.get("todos", []), start=1):
            tid = (t.get("id") or f"T{i}").strip()
            todo = (t.get("todo") or "").strip()
            if todo:
                todos.append({"id": tid, "todo": todo})

    if not todos:
        todos = [{"id": "T1", "todo": user_question.strip()}]

    return todos


# =========================================================
# Todo execution (parallel) + auto tick + focus
# =========================================================
async def solve_one_todo(todo_item: Dict[str, Any], user_question: str) -> Dict[str, Any]:
    ensure_agents()
    todo_worker = st.session_state["agents"]["todo_worker"]
    tid = todo_item["id"]
    todo = todo_item["todo"]

    st.session_state["tool_context"]["todo_id"] = tid
    st.session_state["tool_context"]["focus"] = f"User question: {user_question}\nTodo: {todo}"

    config = {"configurable": {"thread_id": f"{st.session_state['main_thread_id']}:todo:{tid}:run{st.session_state['budget']['run_id']}:r{st.session_state['budget']['round']}"}}
    prompt = f"""User question:
{user_question}

Todo:
{todo}

請依策略（先 KB → web_memory_search → 不足才 web_search）輸出 JSON。"""

    res = await todo_worker.ainvoke({"messages": [{"role": "user", "content": prompt}]}, config=config)
    msg = res["messages"][-1]
    text = _content_to_text(getattr(msg, "content", None))
    parsed = safe_json_from_text(text)
    if parsed is None:
        persist_failure("todo_output_not_json", text[:400], todo_id=tid)

    return {"id": tid, "todo": todo, "raw": text, "parsed": parsed, "ok": True}


async def solve_todos_parallel_with_progress(todos: List[Dict[str, Any]], user_question: str, todo_placeholder) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(int(st.session_state["settings"].get("max_parallel_todos", 3)))

    def _set_status(tid: str, status: str, result: Optional[dict] = None):
        for item in st.session_state["last_todos"]:
            if item.get("id") == tid:
                item["status"] = status
                if result is not None:
                    item["result"] = result
                break

    async def _wrapped(t: Dict[str, Any]):
        tid = t["id"]
        _set_status(tid, "in_progress")
        todo_placeholder.markdown(render_todos_md(st.session_state["last_todos"]))

        try:
            async with sem:
                r = await solve_one_todo(t, user_question)
            return tid, r, None
        except Exception as e:
            return tid, None, e

    st.session_state["last_todos"] = [{**t, "status": "pending", "result": None} for t in todos]
    todo_placeholder.markdown(render_todos_md(st.session_state["last_todos"]))

    tasks: List[asyncio.Task] = [asyncio.create_task(_wrapped(t)) for t in todos]

    results: List[Dict[str, Any]] = []
    for fut in asyncio.as_completed(tasks):
        tid, r, err = await fut

        if err is None:
            results.append(r)
            _set_status(tid, "completed", r)
        else:
            persist_failure("todo_exception", str(err), todo_id=tid)
            fail_rec = {"id": tid, "ok": False, "error": str(err)}
            results.append(fail_rec)
            _set_status(tid, "failed", fail_rec)

        todo_placeholder.markdown(render_todos_md(st.session_state["last_todos"]))

    st.session_state["tool_context"]["todo_id"] = None
    st.session_state["tool_context"]["focus"] = ""
    return results


# =========================================================
# Synth / Direct / Verify / Replan / Refine
# =========================================================
async def synthesize_answer(user_question: str, todo_results: List[Dict[str, Any]]) -> str:
    ensure_agents()
    synth = st.session_state["agents"]["synth"]
    config = {"configurable": {"thread_id": st.session_state["main_thread_id"] + f":synth:run{st.session_state['budget']['run_id']}:r{st.session_state['budget']['round']}"}}
    payload = {
        "user_question": user_question,
        "budget": st.session_state["budget"],
        "todo_results": todo_results,
        "memory_digest": memory_digest(),
    }
    prompt = "請根據以下 JSON 統整最終回答：\n\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    res = await synth.ainvoke({"messages": [{"role": "user", "content": prompt}]}, config=config)
    return _content_to_text(getattr(res["messages"][-1], "content", None))


async def direct_fast_answer(user_question: str, history_messages: List[Dict[str, str]]) -> str:
    ensure_agents()
    agent = st.session_state["agents"]["direct_fast"]
    config = {"configurable": {"thread_id": st.session_state["main_thread_id"] + f":direct:run{st.session_state['budget']['run_id']}:r{st.session_state['budget']['round']}"}}
    res = await agent.ainvoke({"messages": history_messages + [{"role": "user", "content": user_question}]}, config=config)
    return _content_to_text(getattr(res["messages"][-1], "content", None))


class VerifierReport(BaseModel):
    needs_fix: bool
    fix_type: Literal["rewrite_only", "need_more_research", "replan"]
    missing_points: List[str] = Field(default_factory=list)
    supplemental_todos: List[str] = Field(default_factory=list)
    replan_goal: str = ""
    notes: str = ""


async def verify_answer(user_question: str, draft_answer: str, todo_results: List[Dict[str, Any]]) -> VerifierReport:
    ensure_agents()
    verifier = st.session_state["agents"]["verifier"]
    config = {"configurable": {"thread_id": st.session_state["main_thread_id"] + f":verifier:run{st.session_state['budget']['run_id']}:r{st.session_state['budget']['round']}"}}
    payload = {
        "user_question": user_question,
        "draft_answer": draft_answer,
        "todo_results": todo_results,
        "budget": st.session_state["budget"],
        "web_allowed": _web_allowed(),
        "memory_digest": memory_digest(),
    }
    prompt = "請審查並輸出 JSON：\n\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    res = await verifier.ainvoke({"messages": [{"role": "user", "content": prompt}]}, config=config)

    # ✅ 修正：content 可能是 list（Responses content blocks）
    parsed = safe_json_from_text(getattr(res["messages"][-1], "content", None)) or {}
    try:
        return VerifierReport(**parsed)
    except Exception:
        return VerifierReport(needs_fix=False, fix_type="rewrite_only", notes="Verifier 解析失敗，跳過。")


async def replan_todos(user_question: str, previous_todos: List[Dict[str, str]], todo_results: List[Dict[str, Any]], report: VerifierReport) -> List[Dict[str, str]]:
    ensure_agents()
    replanner = st.session_state["agents"]["replanner"]
    config = {"configurable": {"thread_id": st.session_state["main_thread_id"] + f":replanner:run{st.session_state['budget']['run_id']}:r{st.session_state['budget']['round']}"}}
    payload = {
        "user_question": user_question,
        "previous_todos": previous_todos,
        "todo_results": todo_results,
        "verifier_report": report.model_dump(),
        "memory_digest": memory_digest(),
    }
    prompt = "請輸出新的 todos JSON：\n\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    res = await replanner.ainvoke({"messages": [{"role": "user", "content": prompt}]}, config=config)

    data = safe_json_from_text(getattr(res["messages"][-1], "content", None)) or {}
    out = []
    for i, t in enumerate(data.get("todos", []), start=1):
        tid = (t.get("id") or f"R{i}").strip()
        todo = (t.get("todo") or "").strip()
        if todo:
            out.append({"id": tid, "todo": todo})
    return out or [{"id": "R1", "todo": "重新釐清需求與可用資料來源，提出可行解法與限制"}]


async def refine_answer(user_question: str, draft_answer: str, todo_results: List[Dict[str, Any]], report: VerifierReport) -> str:
    ensure_agents()
    refiner = st.session_state["agents"]["refiner"]
    config = {"configurable": {"thread_id": st.session_state["main_thread_id"] + f":refiner:run{st.session_state['budget']['run_id']}:r{st.session_state['budget']['round']}"}}
    payload = {
        "user_question": user_question,
        "draft_answer": draft_answer,
        "todo_results": todo_results,
        "verifier_report": report.model_dump(),
        "budget": st.session_state["budget"],
        "memory_digest": memory_digest(),
    }
    prompt = "請修訂最終答案：\n\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    res = await refiner.ainvoke({"messages": [{"role": "user", "content": prompt}]}, config=config)
    return _content_to_text(getattr(res["messages"][-1], "content", None))


def _alloc_ids(prefix: str, used_ids: List[str], n: int) -> List[str]:
    used = set(used_ids)
    out = []
    i = 1
    while len(out) < n:
        tid = f"{prefix}{i}"
        if tid not in used:
            out.append(tid)
            used.add(tid)
        i += 1
    return out


async def run_full_pipeline(user_question: str, history_messages: List[Dict[str, str]], todo_placeholder) -> str:
    mode = await classify_need_mode(user_question)

    previous_todos: List[Dict[str, str]] = []
    todo_results: List[Dict[str, Any]] = []

    if mode.mode == "direct" and not _direct_do_planning():
        draft = await direct_fast_answer(user_question, history_messages)
    else:
        previous_todos = await plan_todos(user_question, history_messages)
        todo_results = await solve_todos_parallel_with_progress(previous_todos, user_question, todo_placeholder)
        draft = await synthesize_answer(user_question, todo_results)

    report = await verify_answer(user_question, draft, todo_results)
    if not report.needs_fix:
        return draft

    if report.fix_type == "rewrite_only":
        return await refine_answer(user_question, draft, todo_results, report)

    if report.fix_type == "need_more_research":
        supplemental = (report.supplemental_todos or [])[:2]
        if supplemental:
            used_ids = [t["id"] for t in st.session_state["last_todos"]]
            new_ids = _alloc_ids("S", used_ids, len(supplemental))
            sup_todos = [{"id": tid, "todo": t} for tid, t in zip(new_ids, supplemental)]
            st.session_state["last_todos"].extend([{**t, "status": "pending", "result": None} for t in sup_todos])
            todo_placeholder.markdown(render_todos_md(st.session_state["last_todos"]))
            sup_results = await solve_todos_parallel_with_progress(sup_todos, user_question, todo_placeholder)
            todo_results.extend(sup_results)
        return await refine_answer(user_question, draft, todo_results, report)

    if report.fix_type == "replan":
        for _ in range(MAX_REPLAN_ROUNDS):
            _reset_budget_new_round(round_id=1)
            new_todos = await replan_todos(user_question, previous_todos, todo_results, report)
            st.session_state["last_todos"].append({"id": "—", "todo": "【Replan：重新規劃後的 Todo】", "status": "completed", "result": None})
            st.session_state["last_todos"].extend([{**t, "status": "pending", "result": None} for t in new_todos])
            todo_placeholder.markdown(render_todos_md(st.session_state["last_todos"]))

            new_results = await solve_todos_parallel_with_progress(new_todos, user_question, todo_placeholder)
            todo_results = todo_results + new_results
            draft2 = await synthesize_answer(user_question, todo_results)

            report2 = await verify_answer(user_question, draft2, todo_results)
            if report2.needs_fix:
                return await refine_answer(user_question, draft2, todo_results, report2)
            return draft2

    return draft


def run_async(coro):
    """
    Streamlit 有時候環境已經有 event loop（或某些 runtime 包起來），asyncio.run 會噴。
    這裡做 fallback：nest_asyncio 可用就用它。
    """
    try:
        return asyncio.run(coro)
    except RuntimeError as e:
        msg = str(e)
        if "cannot be called from a running event loop" not in msg:
            raise
        try:
            import nest_asyncio  # type: ignore
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(coro)
        except Exception:
            raise


# =========================================================
# UI
# =========================================================
st.title("Agentic RAG（Web 記憶 embeddings 檢索 + Replan + GPT-5.2 Responses API）")

with st.popover("設定（記憶/模型/網路）", use_container_width=True):
    st.session_state["settings"]["memory_namespace"] = st.text_input(
        "記憶ID（同一個ID會載入同一份跨對話記憶）",
        value=st.session_state["settings"].get("memory_namespace", "default"),
    )
    st.session_state["settings"]["web_allowed"] = st.toggle(
        "允許使用網路（OpenAI web_search_preview）",
        value=_web_allowed(),
    )
    st.session_state["settings"]["direct_do_planning"] = st.toggle(
        "Direct 模式也先規劃 Todo",
        value=_direct_do_planning(),
    )
    st.session_state["settings"]["max_parallel_todos"] = st.number_input(
        "同時並行 Todo 上限",
        min_value=1, max_value=10,
        value=int(st.session_state["settings"].get("max_parallel_todos", 3)),
        step=1,
    )
    st.caption(
        f"每輪預算：KB {BUDGET_LIMIT} 次 + Web live {BUDGET_LIMIT} 次。"
        f"Web 記憶 embeddings 檢索保留最近 {WEB_MEM_MAX} 筆（不耗 web 預算）。"
    )

with st.popover("上傳知識文件（建立向量庫）", use_container_width=True):
    uploaded_files = st.file_uploader(
        "上傳知識文件（PDF, Word, PPT, Excel, TXT）",
        type=["pdf", "docx", "doc", "pptx", "xlsx", "xls", "txt"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            if file_ext == ".pdf":
                loader = PyMuPDFLoader(tmp_path)
            elif file_ext in [".docx", ".doc"]:
                loader = UnstructuredWordDocumentLoader(tmp_path, mode="single")
            elif file_ext == ".pptx":
                loader = UnstructuredPowerPointLoader(tmp_path, mode="single")
            elif file_ext in [".xlsx", ".xls"]:
                loader = UnstructuredExcelLoader(tmp_path, mode="single")
            elif file_ext == ".txt":
                loader = TextLoader(tmp_path)
            else:
                st.warning(f"不支援的檔案格式：{uploaded_file.name}")
                continue

            docs = loader.load()
            if not docs:
                st.warning(f"檔案 {uploaded_file.name} 沒有可讀內容，請換個檔案！")
                continue

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
            splits = splitter.split_documents(docs)

            embd = OpenAIEmbeddings(model="text-embedding-3-small")
            if st.session_state["vectorstore"] is None:
                st.session_state["vectorstore"] = FAISS.from_documents(splits, embd)
            else:
                st.session_state["vectorstore"].add_documents(splits)

            st.success(f"已建立/更新向量庫：{uploaded_file.name}")

# 載入跨對話記憶（依 memory_namespace）
load_persistent_memory()
# ✅ 載入/建立 web 記憶 index（最多 500）
web_mem_load_or_build()

# 顯示歷史對話（UI）
for entry in st.session_state["chat_history"]:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])

prompt = st.chat_input("請輸入需求 / 問題")
if prompt:
    ensure_agents()
    _start_new_run()

    st.session_state["chat_history"].append({"role": "user", "content": prompt})
    persist_chat_message("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    todo_placeholder = st.empty()

    with st.status("執行中：initial → verifier →（補強或 replan）→ 最終答案", expanded=True) as status:
        history_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state["chat_history"]]
        final_answer = run_async(run_full_pipeline(prompt, history_messages, todo_placeholder))
        status.write({"run_id": st.session_state["budget"]["run_id"], "budget": st.session_state["budget"]})
        status.update(label="完成！", state="complete")

    st.session_state["chat_history"].append({"role": "assistant", "content": final_answer})
    persist_chat_message("assistant", final_answer)
    with st.chat_message("assistant"):
        st.markdown(final_answer)

    with st.expander("Web 記憶 embeddings index 狀態", expanded=False):
        st.write({
            "indexed_docs": len([x for x in st.session_state["web_mem"]["doc_ids"] if x != "__empty__"]),
            "max": WEB_MEM_MAX,
            "index_dir": str(_mem_paths(_namespace())["web_index_dir"]),
        })

    with st.expander("短期記憶（分開存）", expanded=False):
        st.markdown("#### Web memory（最近 5 筆）")
        st.json(st.session_state["memory"]["web"][-5:])
        st.markdown("#### KB memory（最近 5 筆）")
        st.json(st.session_state["memory"]["kb"][-5:])
        st.markdown("#### Failures（最近 10 筆）")
        st.json(st.session_state["memory"]["failures"][-10:])

    with st.expander("本輪 Todo 詳細（debug）", expanded=False):
        st.json(st.session_state.get("last_todos", []))
