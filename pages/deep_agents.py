# pages/deep_agents.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import io
import uuid
import math
import time
import json
import base64
import hashlib
import threading
import ast
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
from functools import lru_cache
import inspect

import streamlit as st
import numpy as np
import pandas as pd
import faiss
from pypdf import PdfReader

from openai import OpenAI
from langgraph.errors import GraphRecursionError

# LangChain: splitter + BM25 + FlashRank (rerank)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

HAS_FLASHRANK = False
FlashrankRerank = None
try:
    from langchain_community.document_compressors import FlashrankRerank as _FlashrankRerank
    FlashrankRerank = _FlashrankRerank
    HAS_FLASHRANK = True
except Exception:
    HAS_FLASHRANK = False

try:
    import fitz  # pymupdf
    HAS_PYMUPDF = True
except Exception:
    HAS_PYMUPDF = False

# Optional: Unstructured loaders for Office docs
HAS_UNSTRUCTURED_LOADERS = False
UNSTRUCTURED_IMPORT_ERRORS: list[str] = []
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
except Exception as e:
    UNSTRUCTURED_IMPORT_ERRORS.append(repr(e))
    HAS_UNSTRUCTURED_LOADERS = False


# =========================
# Streamlit config
# =========================
st.set_page_config(page_title="研究報告助手（DeepAgent）", layout="wide")
st.title("研究報告助手（DeepAgent）")


# =========================
# DeepAgents / LangChain imports（可診斷版）
# =========================
HAS_DEEPAGENTS = False
DEEPAGENTS_IMPORT_ERRORS: list[str] = []

create_deep_agent = None
init_chat_model = None
ChatOpenAI = None

try:
    from deepagents import create_deep_agent as _create_deep_agent
    create_deep_agent = _create_deep_agent
except Exception as e:
    DEEPAGENTS_IMPORT_ERRORS.append(f"deepagents import failed: {repr(e)}")

try:
    from langchain.chat_models import init_chat_model as _init_chat_model
    init_chat_model = _init_chat_model
except Exception as e:
    DEEPAGENTS_IMPORT_ERRORS.append(f"langchain.chat_models.init_chat_model import failed: {repr(e)}")

try:
    from langchain_openai import ChatOpenAI as _ChatOpenAI
    ChatOpenAI = _ChatOpenAI
except Exception as e:
    DEEPAGENTS_IMPORT_ERRORS.append(f"langchain_openai.ChatOpenAI import failed: {repr(e)}")

HAS_DEEPAGENTS = (create_deep_agent is not None) and ((init_chat_model is not None) or (ChatOpenAI is not None))


def _require_deepagents() -> None:
    if HAS_DEEPAGENTS:
        return
    st.error("DeepAgent 依賴載入失敗（可能是版本/依賴不相容）。")
    if DEEPAGENTS_IMPORT_ERRORS:
        st.markdown("### 依賴錯誤細節")
        for msg in DEEPAGENTS_IMPORT_ERRORS:
            st.code(msg)
    st.stop()


def _make_langchain_llm(model_name: str, temperature: float = 0.0, reasoning_effort: Optional[str] = None):
    """
    回傳 LangChain 的 chat model instance：
    - 優先 init_chat_model
    - fallback ChatOpenAI
    """
    if init_chat_model is not None:
        if model_name.startswith("openai:"):
            return init_chat_model(model=model_name, temperature=temperature)
        return init_chat_model(model=f"openai:{model_name}", temperature=temperature)

    if ChatOpenAI is not None:
        if model_name.startswith("openai:"):
            model_name = model_name.split("openai:", 1)[1]
        kwargs = dict(
            model=model_name,
            temperature=temperature,
            use_responses_api=True,
            max_completion_tokens=None,
        )
        if reasoning_effort in ("low", "medium", "high"):
            kwargs["reasoning"] = {"effort": reasoning_effort}
        return ChatOpenAI(**kwargs)

    raise RuntimeError("No LangChain LLM factory available.")


# =========================
# 模型設定
# =========================
EMBEDDING_MODEL = "text-embedding-3-small"

MODEL_MAIN = "gpt-5.2"
MODEL_GRADER = "gpt-5.2"
MODEL_WEB = "gpt-5.2"

REASONING_EFFORT = "medium"


# =========================
# Skills / Memory（session-only：以 invoke/stream 的 files= 注入）
# =========================
AGENTS_MD = """
# AGENTS.md（本檔內容會被注入 system prompt：session-only）

你是安妮亞風格的研究/分析助理，但在研究與引用任務時必須專業嚴謹。

## 核心目標
- 幫使用者從文件中找證據、形成可核對的主張（claims），並輸出可直接拿去工作的建議。
- 你可以提供「Decision Memo」與「下一步清單（含 DoD）」，提升工作推進效率。

## 硬規則（務必遵守）
1) 語言：繁體中文（台灣用語）
2) 禁止洩漏內部流程/檔名：不得出現
   - /evidence、/analysis、/workspace、doc_、web_、Budget exceeded、chunk_id 等字樣
3) 引用格式
   - 文件引用 token： [報告名稱 pN] 或 [報告名稱 p-]
   - 網路引用 token： [WebSearch:<domain> p-]
4) 不要揭露模型內在思考（不要 chain-of-thought）
   - 可以呈現：Todo / facets / evidence / claims / 反思 / 決策取捨（這些是工作產物）
5) 若證據不足：清楚說「資料不足」，並列出需要補的資訊（<=3項），不要腦補。

## 你要輸出的固定區塊（建議）
- 重點結論（每點句尾要有引用 token）
- Decision Memo（目標/現況與限制/選項/建議/取捨/反思/下一步）
- 下一步（3~8項，含 DoD）
- 需要我補的資訊（<=3項）
""".strip()

SKILL_CLAIMS_FIRST = """---
name: claims-first
description: Use this skill when you need rigorous, citation-backed reasoning (claims-first) before writing a final answer.
---
# Claims-first workflow（強推理）
當任務需要嚴謹推導、比對、或要產出可核對結論時，先做 claims，再寫成品。

## Steps
1) 先從 /evidence/ 擷取可用證據（不要發明新事實）
2) 產出 /analysis/claims.json：
   - claim：一句可核對的主張
   - citations：必須是 [報告名稱 pN] 或 [WebSearch:domain p-]
   - assumptions：必要假設（可空）
   - confidence：0~1
3) 產出 /analysis/reflections.json（反思）：
   - 盲點/風險/反例/需驗證點
   - 若沒有引用，needs_validation=true
4) 最後 writer 只能依據 claims/reflections + evidence 寫 /draft.md
""".strip()

SKILL_DECISION_MEMO = """---
name: decision-memo
description: Use this skill when the user wants actionable work guidance, planning, or decision support with trade-offs and reflections.
---
# Decision Memo（含反思）
輸出時請包含以下小節：

## Decision Memo
- 目標：
- 現況與限制：
- 選項（Option A/B/…）：
- 建議（Recommendation）：
- 取捨（Trade-offs）：
- 反思（Reflections）：
  - 風險、盲點、反例、需驗證點（可引用；無引用要標明需驗證）
- 下一步（Next Steps）：
  - 每項要有 DoD（完成條件）

Tips：
- 管理/流程（30%）：里程碑、stakeholder、風險控管、依賴
- 技術/分析（70%）：假設檢驗、資料需求、方法、驗證步驟
""".strip()

SKILL_REPORT_COMPARE = """---
name: report-compare
description: Use this skill when comparing multiple reports, finding differences, contradictions, or synthesizing across documents.
---
# 跨報告比較（report-compare）
1) 先列比較維度（2~6個）：定義、結論、數據、假設、方法、限制
2) 對每維度找 evidence（每點都要引用 token）
3) 若有矛盾：列出矛盾點 + 各自依據 + 可能原因（版本/口徑/範圍），不要硬融合
4) 最終輸出要有可用建議：下一步補什麼資料/找誰確認/如何驗證
""".strip()

SKILL_ACTION_PLAN = """---
name: action-plan
description: Use this skill when turning analysis into a concrete plan with milestones, stakeholders, and risks.
---
# 推進計畫（action-plan）
輸出「下一步清單」時：
- 3~8項為主
- 每項包含：要做什麼 / 產出物 / DoD / owner（未知可留空）/ 風險
另外可加「里程碑」小節：
- M1/M2/M3（每個里程碑一句話+驗收條件）
""".strip()


# ========= [替換 3] build_seed_files_for_deepagents：整個函式替換 =========
def build_seed_files_for_deepagents() -> dict:
    """
    DeepAgents 的 files seed（session-only）。
    除了 memory/skills，也塞入 runtime 設定（scope / threshold / question_kind）。
    """
    seed: dict[str, str] = {}

    # runtime（每次 run 都可能不同）
    runtime = {
        "scope_title": (st.session_state.get("selected_report_title") or "All"),
        "web_threshold": float(WEB_EVIDENCE_THRESHOLD),
        "allow_web": bool(st.session_state.get("enable_web_search_agent", True)),
        "question_kind": str(st.session_state.get("current_question_kind", QUESTION_KIND_CHAT) or QUESTION_KIND_CHAT),
    }
    seed["/runtime/runtime.json"] = json.dumps(runtime, ensure_ascii=False, indent=2)

    if st.session_state.get("da_enable_memory", True):
        seed["/memory/AGENTS.md"] = AGENTS_MD

    if st.session_state.get("da_enable_skills", True):
        if st.session_state.get("da_skill_claims_first", True):
            seed["/skills/claims-first/SKILL.md"] = SKILL_CLAIMS_FIRST
        if st.session_state.get("da_skill_decision_memo", True):
            seed["/skills/decision-memo/SKILL.md"] = SKILL_DECISION_MEMO
        if st.session_state.get("da_skill_report_compare", True):
            seed["/skills/report-compare/SKILL.md"] = SKILL_REPORT_COMPARE
        if st.session_state.get("da_skill_action_plan", True):
            seed["/skills/action-plan/SKILL.md"] = SKILL_ACTION_PLAN

    return seed


# =========================
# 系統提示（精簡版）
# =========================
ANYA_SYSTEM_PROMPT = """
你是安妮亞風格的助理，但在學術/研究/引用任務時要專業嚴謹。
規則：
- 用繁體中文（台灣用語）。
- 若有給 Context（文件摘錄/證據），只能依據 Context 回答；不足就說資料不足並提出需要什麼。
- 禁止洩漏內部流程/檔名（/evidence、doc_、web_、Budget exceeded、chunk_id 等）。
- 若要引用：文件引用格式必須是 [報告名稱 pN]；網路引用 token 為 [WebSearch:<domain> p-]。
- 回答盡量結構化（標題/條列），先結論後細節。
""".strip()

DIRECT_EVIDENCE_SYSTEM_PROMPT = """
你是研究助理。你必須使用 web_search 先蒐集證據，然後只輸出『證據筆記』。
輸出格式必須固定：
### EVIDENCE
- 最多 8 點，每點一句、可核對（含日期/人名/機構/數字）。
### SOURCES
- 最多 12 行：- <domain> | <title> | <url>
規則：
- 只寫來源中確實看到的內容，不確定就不要寫。
- 不要提工具流程/額度/內部字樣。
""".strip()

DIRECT_WRITER_SYSTEM_PROMPT = """
你是寫作整理者。你會收到：使用者問題 + EVIDENCE + SOURCES。
規則：
- 只能用 EVIDENCE 內的事實寫作，不可腦補。
- 正文不要貼 URL，不要提「我查到/我蒐證」等流程。
- 內文引用用（來源：domain），domain 必須出現在 SOURCES。
- 若沒有來源能對應的段落，直接刪掉不寫。
輸出建議：
## 重點摘要
- 3~6 點（來源：domain）
## 已知
- 3~8 點（來源：domain）
## 待確認
- 2~6 點（可不附來源）
""".strip()

FORMATTER_SYSTEM_PROMPT = r"""
你是 Markdown formatter：只整理版面，不改內容、不新增事實、不刪改引用 token。
引用 token（如 [報告 p12]、[WebSearch:xx p-]）不可改寫不可刪。
不要提內部流程/檔名/額度。
只輸出排版後 Markdown。
""".strip()


# =========================
# 效能/策略參數
# =========================
EMBED_BATCH_SIZE = 256
OCR_MAX_WORKERS = 2

DA_MAX_DOC_SEARCH_CALLS = 14
DA_MAX_WEB_SEARCH_CALLS = 4
DA_MAX_REWRITE_ROUNDS = 2
DA_MAX_CLAIMS = 10

DEFAULT_RECURSION_LIMIT = 200
DEFAULT_CITATION_STALL_STEPS = 12
DEFAULT_CITATION_STALL_MIN_CHARS = 450

DEFAULT_SOURCES_BADGE_MAX_TITLES_INLINE = 4
DEFAULT_SOURCES_BADGE_MAX_PAGES_PER_TITLE = 10

ENABLE_FLASHRANK_RERANK = True
FLASHRANK_CANDIDATES = 30

UI_MAX_EVIDENCE_PREVIEW_CHARS = 900
UI_MAX_DRAFT_PREVIEW_CHARS = 1200
UI_MAX_DOC_SEARCH_LOG = 8

# ========= [新增/替換 1] 放在「效能/策略參數」附近（例如 DEFAULT_* 那區） =========

WEB_EVIDENCE_THRESHOLD = 0.55  # 你拍板：grader < 0.55 才允許開 web（保守、成本低）

# question kind（用來決定 Q2=C：聊天式 vs Decision Memo）
QUESTION_KIND_CHAT = "chat"
QUESTION_KIND_MEMO = "memo"
# =========================
# Regex / 內部洩漏防護
# =========================
CHUNK_ID_LEAK_PAT = re.compile(r"(chunk_id\s*=\s*|_p(?:na|\d+)_c\d+)", re.IGNORECASE)
EVIDENCE_PATH_IN_CIT_RE = re.compile(r"\[(?:/)?evidence/[^ \]]+?\s+p(\d+|-)\s*\]", re.IGNORECASE)
INTERNAL_LEAK_PAT = re.compile(
    r"(Budget exceeded|/evidence|/analysis|/workspace|doc_[\w\-]+\.md|web_[\w\-]+\.md|額度不足|占位|向量庫|內部文件|工作流|流程|工具預算|chunk_id)",
    re.IGNORECASE,
)


# =========================
# 小工具
# =========================
def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def estimate_tokens_from_chars(n_chars: int) -> int:
    if n_chars <= 0:
        return 0
    return max(1, int(math.ceil(n_chars / 3.6)))


@lru_cache(maxsize=16)
def _get_recursive_splitter(chunk_size: int, overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=int(chunk_size),
        chunk_overlap=int(overlap),
        length_function=len,
        separators=[
            "\n\n", "\n",
            "。", "！", "？",
            ".", "!", "?",
            "；", ";",
            "，", ",",
            " ", "",
        ],
    )


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    text = norm_space(text)
    if not text:
        return []
    try:
        splitter = _get_recursive_splitter(chunk_size, overlap)
        docs = splitter.create_documents([text])
        out = []
        for d in docs:
            t = norm_space(d.page_content)
            if t:
                out.append(t)
        return out
    except Exception:
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


def truncate_filename(name: str, max_len: int = 44) -> str:
    if len(name) <= max_len:
        return name
    base, ext = os.path.splitext(name)
    keep = max(10, max_len - len(ext) - 1)
    return f"{base[:keep]}…{ext}"


def _dedup_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _hash_norm_text(s: str) -> str:
    return sha1_bytes(norm_space(s).encode("utf-8"))


def strip_internal_process_lines(md: str) -> str:
    lines = (md or "").splitlines()
    kept = []
    for line in lines:
        if INTERNAL_LEAK_PAT.search(line):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def get_recent_chat_messages(max_messages: int = 15) -> list[dict]:
    msgs: list[dict] = []
    for m in st.session_state.get("chat_history", []):
        if m.get("kind") != "text":
            continue
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if len(content) > 2000:
            content = content[:2000] + "…"
        msgs.append({"role": role, "content": content})
    return msgs[-max_messages:]


def build_run_messages(prompt: str, max_messages: int = 15) -> list[dict]:
    msgs = get_recent_chat_messages(max_messages=max_messages)
    if msgs and msgs[-1].get("role") == "user" and (msgs[-1].get("content") or "").strip() == (prompt or "").strip():
        return msgs
    msgs.append({"role": "user", "content": (prompt or "").strip()})
    return msgs


def _domain(u: str) -> str:
    try:
        host = urlparse(u).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host or "web"
    except Exception:
        return "web"


def has_visible_citations(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    # 只要有 [xxx pN] 就算
    return bool(re.search(r"\[[^\]]+?\s+p(\d+|-)\s*\]", raw))


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


def _to_messages(system: str, user: Any) -> list[Dict[str, Any]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _try_parse_json_or_py_literal(text: str) -> Optional[Any]:
    t = (text or "").strip()
    if not t:
        return None
    if t.startswith("{") or t.startswith("["):
        try:
            return json.loads(t)
        except Exception:
            pass
    if t.startswith("{") and t.endswith("}"):
        try:
            return ast.literal_eval(t)
        except Exception:
            return None
    return None


def call_gpt(
    client: OpenAI,
    *,
    model: str,
    system: str,
    user: Any,
    reasoning_effort: Optional[str] = None,
    tools: Optional[list] = None,
    include_sources: bool = False,
    tool_choice: Optional[Any] = None,
) -> Tuple[str, Optional[list[Dict[str, Any]]]]:
    messages = _to_messages(system, user)
    if tool_choice is None:
        tc = "auto" if tools else "none"
    else:
        tc = tool_choice

    resp = client.responses.create(
        model=model,
        input=messages,
        tools=tools,
        tool_choice=tc,
        parallel_tool_calls=True if tools else None,
        reasoning={"effort": reasoning_effort} if reasoning_effort in ("low", "medium", "high") else None,
        include=["web_search_call.action.sources"] if (tools and include_sources) else None,
        truncation="auto",
    )

    out_text = resp.output_text
    sources = None

    if tools and include_sources:
        sources_list: list[Dict[str, Any]] = []

        def _as_dict(x: Any) -> dict:
            if isinstance(x, dict):
                return x
            d = getattr(x, "__dict__", None)
            return d if isinstance(d, dict) else {}

        try:
            for item in (getattr(resp, "output", None) or []):
                d = _as_dict(item)
                typ = d.get("type") or getattr(item, "type", None)
                if typ == "web_search_call":
                    action = d.get("action") or getattr(item, "action", None) or {}
                    action_d = _as_dict(action)
                    ss = action_d.get("sources") or []
                    for s in ss:
                        sd = _as_dict(s)
                        url = (sd.get("url") or "").strip()
                        title = (sd.get("title") or sd.get("source") or "source").strip()
                        if url:
                            sources_list.append({"title": title, "url": url})
        except Exception:
            pass

        if sources_list:
            seen = set()
            uniq = []
            for s in sources_list:
                u = (s.get("url") or "").strip()
                if not u or u in seen:
                    continue
                seen.add(u)
                uniq.append(s)
            sources = uniq if uniq else None

    return out_text, sources


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


# =========================
# Web sources helpers
# =========================
def web_sources_from_openai_sources(sources: Optional[list[dict]]) -> Dict[str, List[Tuple[str, str]]]:
    out: Dict[str, List[Tuple[str, str]]] = {}
    if not sources:
        return out
    for s in sources:
        if not isinstance(s, dict):
            continue
        title = (s.get("title") or s.get("source") or "source").strip()
        url = (s.get("url") or "").strip()
        if not url:
            continue
        dom = _domain(url)
        out.setdefault(dom, []).append((title, url))

    for dom in list(out.keys()):
        seen = set()
        uniq: List[Tuple[str, str]] = []
        for t, u in out[dom]:
            if u in seen:
                continue
            seen.add(u)
            uniq.append((t, u))
        out[dom] = uniq
    return out


def render_web_sources_list(
    web_sources: Dict[str, List[Tuple[str, str]]],
    max_domains: int = 6,
    max_per_domain: int = 6,
    max_path_len: int = 80,
) -> None:
    if not web_sources:
        return

    def _path(u: str) -> str:
        try:
            p = urlparse(u)
            path = (p.path or "/").strip()
            if p.query:
                path = f"{path}?{p.query}"
            if len(path) > max_path_len:
                path = path[:max_path_len] + "…"
            return path
        except Exception:
            return "/"

    domains = sorted(web_sources.keys())
    show = domains[:max_domains]
    more = domains[max_domains:]

    def _build_md(domains_list: list[str]) -> str:
        lines: list[str] = []
        for dom in domains_list:
            items = web_sources.get(dom, []) or []
            if not items:
                continue
            lines.append(f"- **{dom}**")
            for title, url in items[:max_per_domain]:
                t = (title or "").strip() or dom
                u = (url or "").strip()
                if not u:
                    continue
                lines.append(f"  - [{t}]({u})")
                lines.append(f"    :small[`{dom}{_path(u)}`]")
        return "\n".join(lines).strip()

    st.markdown("#### Web Sources")
    md_main = _build_md(show)
    if md_main:
        st.markdown(md_main)
    if more:
        md_more = _build_md(more)
        with st.expander(f"更多 Web Sources（{len(more)}）", expanded=False):
            st.markdown(md_more if md_more else "（無）")


# =========================
# OCR / PDF / Image
# =========================
def extract_pdf_text_pages_pypdf(pdf_bytes: bytes) -> list[Tuple[int, str]]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    out: list[Tuple[int, str]] = []
    for i, p in enumerate(reader.pages):
        try:
            t = p.extract_text() or ""
        except Exception:
            t = ""
        out.append((i + 1, norm_space(t)))
    return out


def extract_pdf_text_pages_pymupdf(pdf_bytes: bytes) -> list[Tuple[int, str]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    out: list[Tuple[int, str]] = []
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


def _img_bytes_to_data_url(img_bytes: bytes, mime: str = "image/png") -> str:
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def ocr_image_bytes(client: OpenAI, image_bytes: bytes, mime: str = "image/png") -> str:
    system = "你是 OCR。只輸出可見文字/表格（表格用 Markdown 表格），不要評論。"
    user_content = [
        {"type": "input_text", "text": "請擷取圖片中所有可見文字（含小字/註腳）。"},
        {"type": "input_image", "image_url": _img_bytes_to_data_url(image_bytes, mime=mime)},
    ]
    text, _ = call_gpt(client, model=MODEL_GRADER, system=system, user=user_content, reasoning_effort=None)
    return text


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
        futs = {ex.submit(ocr_image_bytes, client, img_bytes, "image/png"): page_no for page_no, img_bytes in page_imgs}
        for fut in as_completed(futs):
            page_no = futs[fut]
            try:
                results[page_no] = norm_space(fut.result())
            except Exception:
                results[page_no] = ""

    return [(p, results.get(p, "")) for p, _ in page_imgs]


# =========================
# Optional Office extraction (via Unstructured loaders)
# =========================
def _write_temp_file(data: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        return tmp.name


def extract_office_text_blocks(filename: str, ext: str, data: bytes) -> list[Tuple[Optional[int], str]]:
    """
    回傳 [(block_no, text)]，block_no 當作 pN 的 N（位置序）。
    注意：這裡先用「單一 block」回傳，交給 chunk_text 切；能穩就先穩。
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


def infer_loc_kind_from_ext(ext: str) -> str:
    ext = (ext or "").lower()
    if ext == ".pdf":
        return "page"
    if ext == ".pptx":
        return "slide"
    if ext in (".doc", ".docx"):
        return "para"
    if ext in (".xls", ".xlsx"):
        return "block"
    if ext == ".txt":
        return "na"
    if ext in (".png", ".jpg", ".jpeg"):
        return "na"
    return "na"


# =========================
# Hybrid retrieval (BM25 + semantic + RRF + hard-only FlashRank)
# =========================
def bm25_preprocess_zh_en(text: str) -> list[str]:
    t = (text or "").lower()
    return re.findall(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*|[\u4e00-\u9fff]", t)


def rrf_scores(rank_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for rl in rank_lists:
        for rank, cid in enumerate(rl, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return scores


@dataclass
class Chunk:
    chunk_id: str
    report_id: str
    title: str
    page: Optional[int]    # token 上仍叫 pN，但 UI 會顯示成「位置」
    text: str
    ext: str
    loc_kind: str


class FaissStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []
        self.bm25: Optional[BM25Retriever] = None
        self._flashrank: Optional[Any] = None

    def _rebuild_bm25(self) -> None:
        if not self.chunks:
            self.bm25 = None
            return
        docs = [
            Document(
                page_content=(c.text or ""),
                metadata={"chunk_id": c.chunk_id, "title": c.title, "page": c.page if c.page is not None else "-"},
            )
            for c in self.chunks
        ]
        self.bm25 = BM25Retriever.from_documents(
            docs,
            k=24,
            preprocess_func=bm25_preprocess_zh_en,
        )

    def add(self, vecs: np.ndarray, chunks: list[Chunk]) -> None:
        self.index.add(vecs)
        self.chunks.extend(chunks)
        self._rebuild_bm25()

    def search_semantic(self, qvec: np.ndarray, k: int = 10) -> list[Tuple[float, Chunk]]:
        if self.index.ntotal == 0:
            return []
        scores, idx = self.index.search(qvec.astype(np.float32), k)
        out: list[Tuple[float, Chunk]] = []
        for s, i in zip(scores[0], idx[0]):
            if i < 0 or i >= len(self.chunks):
                continue
            out.append((float(s), self.chunks[i]))
        return out

    def search_bm25(self, query: str, k: int = 16) -> list[Chunk]:
        if not self.bm25:
            return []
        self.bm25.k = max(1, int(k))
        docs = self.bm25.invoke(query)
        cid_to_chunk = {c.chunk_id: c for c in self.chunks}
        out: list[Chunk] = []
        for d in (docs or []):
            cid = (d.metadata or {}).get("chunk_id")
            if cid and cid in cid_to_chunk:
                out.append(cid_to_chunk[cid])
        return out

    def _rerank_flashrank(self, query: str, candidates: list[Chunk], top_k: int) -> list[Tuple[float, Chunk]]:
        if not (ENABLE_FLASHRANK_RERANK and HAS_FLASHRANK and FlashrankRerank is not None):
            return [(0.0, c) for c in candidates[:top_k]]

        try:
            if self._flashrank is None:
                self._flashrank = FlashrankRerank()
            docs = [
                Document(
                    page_content=(c.text or "")[:2400],
                    metadata={"chunk_id": c.chunk_id, "title": c.title, "page": c.page if c.page is not None else "-"},
                )
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
        difficulty = (difficulty or "medium").strip().lower()
        do_rerank = bool(ENABLE_FLASHRANK_RERANK) and (difficulty == "hard")

        sem_hits = self.search_semantic(qvec, k=max(10, k))
        bm_chunks = self.search_bm25(query, k=max(16, k * 2))

        sem_rank = [ch.chunk_id for _, ch in sem_hits]
        bm_rank = [ch.chunk_id for ch in bm_chunks]
        fused = rrf_scores([sem_rank, bm_rank], k=60)

        cid_to_chunk: dict[str, Chunk] = {}
        for _, ch in sem_hits:
            cid_to_chunk[ch.chunk_id] = ch
        for ch in bm_chunks:
            cid_to_chunk.setdefault(ch.chunk_id, ch)

        items = list(cid_to_chunk.items())
        items.sort(key=lambda kv: fused.get(kv[0], 0.0), reverse=True)

        if not do_rerank:
            out: list[Tuple[float, Chunk]] = []
            for cid, ch in items[:k]:
                out.append((float(fused.get(cid, 0.0)), ch))
            return out

        candidates = [ch for _, ch in items[: max(FLASHRANK_CANDIDATES, k)]]
        return self._rerank_flashrank(query, candidates, top_k=k)


# =========================
# File rows + indexing
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
    text_pages: Optional[int]
    text_pages_ratio: Optional[float]
    blank_pages: Optional[int]
    blank_ratio: Optional[float]
    likely_scanned: bool
    use_ocr: bool


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

    stats = {"new_reports": 0, "new_chunks": 0, "errors": []}

    st.session_state.setdefault("title_to_loc_kind", {})

    new_chunks: list[Chunk] = []
    new_texts: list[str] = []

    to_process: list[FileRow] = []
    for r in file_rows:
        key = (r.file_sig, bool(r.use_ocr))
        if key not in processed_keys:
            to_process.append(r)

    for row in to_process:
        data = file_bytes_map[row.file_id]
        report_id = row.file_id
        title = os.path.splitext(row.name)[0]
        ext = (row.ext or "").lower()
        loc_kind = infer_loc_kind_from_ext(ext)
        st.session_state["title_to_loc_kind"][title] = loc_kind
        st.session_state.setdefault("title_to_max_page", {})
        if isinstance(row.pages, int) and row.pages > 0:
            st.session_state["title_to_max_page"][title] = int(row.pages)
        else:
            st.session_state["title_to_max_page"].setdefault(title, None)
        stats["new_reports"] += 1

        pages: list[Tuple[Optional[int], str]] = []
        try:
            if ext == ".pdf":
                pdf_pages = ocr_pdf_pages_parallel(client, data) if row.use_ocr else extract_pdf_text_pages(data)
                pages = [(pno, txt) for pno, txt in pdf_pages]
            elif ext == ".txt":
                pages = [(None, norm_space(data.decode("utf-8", errors="ignore")))]
            elif ext in (".png", ".jpg", ".jpeg"):
                mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
                txt = norm_space(ocr_image_bytes(client, data, mime=mime))
                pages = [(None, txt)]
            elif ext in (".doc", ".docx", ".pptx", ".xls", ".xlsx"):
                pages = extract_office_text_blocks(row.name, ext, data)
            else:
                pages = [(None, "")]
        except Exception as e:
            stats["errors"].append(f"{row.name}: {repr(e)}")
            pages = [(None, "")]

        for page_no, page_text in pages:
            if not page_text:
                continue
            chunks = chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
            for i, ch in enumerate(chunks):
                cid = f"{report_id}_p{page_no if page_no else 'na'}_c{i}"
                new_chunks.append(
                    Chunk(
                        chunk_id=cid,
                        report_id=report_id,
                        title=title,
                        page=page_no if isinstance(page_no, int) else None,
                        text=ch,
                        ext=ext,
                        loc_kind=loc_kind,
                    )
                )
                new_texts.append(ch)

        processed_keys.add((row.file_sig, bool(row.use_ocr)))

    if new_texts:
        vecs_list: list[np.ndarray] = []
        for i in range(0, len(new_texts), EMBED_BATCH_SIZE):
            vecs_list.append(embed_texts(client, new_texts[i:i + EMBED_BATCH_SIZE]))
        vecs = np.vstack(vecs_list)
        store.add(vecs, new_chunks)

    stats["new_chunks"] = len(new_chunks)
    return store, stats, processed_keys


# =========================
# Rendering / citations
# =========================
def _badge_directive(label: str, color: str) -> str:
    safe = label.replace("[", "(").replace("]", ")")
    return f":{color}-badge[{safe}]"


def _strip_citations_from_text(text: str) -> str:
    if not text:
        return ""
    pat = re.compile(r"[ \t]*\[[^\]]*?\s+p(\d+|-)(?:-\d+)?[^\]]*?\][ \t]*")
    out_lines: list[str] = []
    for line in text.splitlines():
        out_lines.append(pat.sub("", line).rstrip())
    return "\n".join(out_lines).strip()


def _extract_citation_items(text: str) -> list[tuple[str, str]]:
    if not text:
        return []
    items: list[tuple[str, str]] = []
    for m in re.finditer(r"\[([^\]]+)\]", text):
        inner = (m.group(1) or "").strip()
        if not inner:
            continue
        parts = [p.strip() for p in re.split(r"[;；]", inner) if p.strip()]
        for p in parts:
            mm = re.search(r"^(.*)\s+p(\d+(?:-\d+)?|-)\s*$", p)
            if not mm:
                continue
            title = norm_space(mm.group(1))
            page = mm.group(2).strip()
            if EVIDENCE_PATH_IN_CIT_RE.search(f"[{title} p{page}]"):
                continue
            items.append((title, page))
    return items


def _format_location_pages(pages: list[str], loc_kind: str) -> str:
    pages = _dedup_keep_order([p.strip() for p in pages if p.strip()])
    if not pages:
        return "p-"
    if pages == ["-"]:
        return "p-"

    if loc_kind == "page":
        prefix = "頁"
    elif loc_kind == "slide":
        prefix = "投影片"
    elif loc_kind == "para":
        prefix = "段落"
    elif loc_kind == "block":
        prefix = "區塊"
    else:
        prefix = "p"

    max_pages = int(st.session_state.get("sources_badge_max_pages_per_title", DEFAULT_SOURCES_BADGE_MAX_PAGES_PER_TITLE))
    if len(pages) <= max_pages:
        return f"位置:{prefix}" + ",".join(pages)
    return f"位置:{prefix}" + ",".join(pages[:max_pages]) + "…"


def render_markdown_answer_with_sources_badges(answer_text: str) -> None:
    raw = strip_internal_process_lines((answer_text or "").strip())
    if raw and CHUNK_ID_LEAK_PAT.search(raw):
        raw = CHUNK_ID_LEAK_PAT.sub("", raw)

    cit_items = _extract_citation_items(raw)
    clean = _strip_citations_from_text(raw)
    st.markdown(clean if clean else "（無內容）")

    if not cit_items:
        return

    grouped: dict[str, list[str]] = {}
    for title, page in cit_items:
        grouped.setdefault(title, []).append(page)

    st.markdown("### 來源")
    max_inline = int(st.session_state.get("sources_badge_max_titles_inline", DEFAULT_SOURCES_BADGE_MAX_TITLES_INLINE))
    title_to_loc_kind = st.session_state.get("title_to_loc_kind", {}) or {}

    titles = sorted(grouped.keys(), key=lambda x: (x.strip().lower().startswith("websearch:"), x.lower()))
    inline_titles = titles[:max_inline]
    extra_titles = titles[max_inline:]

    def _render(titles_list: list[str]) -> None:
        badges = []
        for t in titles_list:
            pages = grouped.get(t, []) or []
            if t.lower().startswith("websearch:"):
                label = f"{t} p-"
                color = "violet"
            else:
                loc_kind = str(title_to_loc_kind.get(t, "na") or "na")
                label = f"{t} {_format_location_pages(pages, loc_kind)}"
                color = "green"
            badges.append(_badge_directive(label, color))
        if badges:
            st.markdown(" ".join(badges))

    _render(inline_titles)
    if extra_titles:
        with st.expander(f"更多來源（{len(extra_titles)}）", expanded=False):
            _render(extra_titles)


def format_markdown_output_preserve_citations(client: OpenAI, md: str) -> str:
    raw = (md or "").strip()
    if not raw:
        return ""
    out, _ = call_gpt(
        client,
        model=MODEL_MAIN,
        system=FORMATTER_SYSTEM_PROMPT,
        user=raw,
        reasoning_effort=None,
        tools=None,
        include_sources=False,
    )
    return (out or "").strip() or raw


def render_chunks_for_model(chunks: list[Chunk], max_chars_each: int = 900) -> str:
    parts = []
    for c in chunks:
        head = f"[{c.title} p{c.page if c.page is not None else '-'}]"
        parts.append(head + "\n" + (c.text or "")[:max_chars_each])
    return "\n\n".join(parts)


# =========================
# Router（規則優先 + LLM 補強）
# =========================
@dataclass
class RoutePlan:
    mode: str  # "smalltalk" | "direct" | "rag" | "deepagent" | "advisor" | "clarify"
    difficulty: str  # "easy" | "medium" | "hard"
    allow_web: bool
    enable_web: bool
    doc_top_k: int
    facets: int
    needs_clarification: bool
    clarifying_questions: list[str]
    reason: str


ADVISOR_KEYWORDS = [
    "規劃", "計畫", "roadmap", "里程碑", "下一步", "怎麼推進", "怎麼做", "策略", "建議",
    "風險", "要注意", "決策", "選項", "取捨", "stakeholder", "優先順序", "時程",
    "待辦", "todo", "拆任務", "排程", "decision memo", "memo",
]
DEEPAGENT_KEYWORDS = [
    "比較", "差異", "對照", "彙整", "交叉驗證", "多份", "跨文件", "矛盾",
    "引用", "出處", "證據", "依據",
]
RAG_KEYWORDS = [
    "條款", "定義", "這段在說什麼", "在哪裡提到", "章", "節",
]


def rule_route_mode(question: str, has_index: bool) -> Optional[str]:
    q = (question or "").lower()

    if any(k.lower() in q for k in ADVISOR_KEYWORDS):
        return "advisor"

    if not has_index:
        return None

    if any(k.lower() in q for k in DEEPAGENT_KEYWORDS):
        return "deepagent"

    if any(k.lower() in q for k in RAG_KEYWORDS):
        return "rag"

    return None

# ========= [新增 5] 新增：doc_intent / question_kind / scope 同步 helper
# 建議放在 Router 區塊附近（例如 rule_route_mode 之後，decide_route_plan 之前） =========

SMALLTALK_HINTS = [
    "你好", "嗨", "哈囉", "早安", "午安", "晚安",
    "謝謝", "感謝", "哈哈", "在嗎",
    "你是誰", "你會做什麼",
]

DOC_INTENT_STRONG = [
    "報告", "文件", "上傳", "附件", "引用", "出處", "根據文件", "根據報告",
    "第幾頁", "頁碼", "哪一頁", "在哪裡提到", "條款", "定義", "章", "節",
    "摘要", "彙整", "整理", "對照", "比較",
]

MEMO_INTENT_HINTS = [
    "比較", "差異", "對照", "彙整", "交叉驗證", "矛盾",
    "決策", "選項", "取捨", "風險", "策略", "規劃", "里程碑", "下一步", "roadmap", "memo",
]

def classify_question_kind(question: str) -> str:
    q = (question or "").strip()
    ql = q.lower()
    if any(k.lower() in ql for k in MEMO_INTENT_HINTS):
        return QUESTION_KIND_MEMO
    return QUESTION_KIND_CHAT

def looks_like_smalltalk(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return True
    if len(q) <= 8 and any(h in q for h in SMALLTALK_HINTS):
        return True
    # 很短、沒有名詞線索的也當閒聊
    if len(q) <= 12 and not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", q):
        return True
    return False

def decide_doc_intent(
    client: OpenAI,
    question: str,
    *,
    has_index: bool,
    scope_title: Optional[str],
    run_messages: Optional[list[dict]] = None,
) -> bool:
    """
    你拍板：敏感版
    - has_index 且像在問內容（摘要/解釋概念）也要開 DeepAgent
    - 但要避免把明顯閒聊誤判成 doc 任務
    """
    if not has_index:
        return False

    q = (question or "").strip()
    if not q:
        return False

    # 1) scope 被鎖定（非 All）→ 一律視為想用文件
    if scope_title:
        return True

    # 2) 明顯閒聊 → 不用文件
    if looks_like_smalltalk(q):
        return False

    # 3) 強訊號關鍵字 → 用文件
    ql = q.lower()
    if any(k.lower() in ql for k in DOC_INTENT_STRONG):
        return True

    # 4) 模糊：交給 router LLM 判斷一次（輕量、只要 true/false）
    hist = ""
    if run_messages:
        lines = []
        for m in run_messages[-8:]:
            role = (m.get("role") or "").strip()
            content = (m.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                if len(content) > 500:
                    content = content[:500] + "…"
                lines.append(f"{role.upper()}: {content}")
        hist = "\n".join(lines).strip()

    system = (
        "你是分類器，只輸出 JSON：{\"doc_intent\":true|false,\"reason\":\"...\"}\n"
        "doc_intent=true 表示：使用者是在問『已上傳文件/報告』的內容（含摘要/解釋概念）。\n"
        "規則：\n"
        "- 明顯生活閒聊/社交用語 → false\n"
        "- 問題像在要你從文件找答案/摘要/解釋 → true\n"
        "只輸出 JSON，不要多字。"
    )
    user = (
        f"has_index=true\n"
        + (f"recent_history:\n{hist}\n\n" if hist else "")
        + f"question:\n{q}\n"
    )
    out, _ = call_gpt(client, model=MODEL_GRADER, system=system, user=user, reasoning_effort="low")
    data = _try_parse_json_or_py_literal(out) or {}
    return bool(data.get("doc_intent", False))

def get_titles_from_store(store: Optional[FaissStore]) -> list[str]:
    return list_report_titles_from_store(store)

def sync_scope_from_prompt_and_ui(prompt: str, store: Optional[FaissStore]) -> Optional[str]:
    """
    回傳 scope_title：
    - None 表示 All
    - "某 title" 表示鎖定該文件

    你要求：若 prompt 明確提到某 title，要「同步更新下拉選單」。
    """
    titles = get_titles_from_store(store)
    ui_sel = str(st.session_state.get("selected_report_title", "All") or "All").strip()
    last_title = st.session_state.get("last_report_title")

    # prompt 明確匹配 title（最優先）
    target = guess_target_title(prompt, titles, last_title=last_title)

    if target and target in titles:
        # ✅ 同步更新 UI
        st.session_state["selected_report_title"] = target
        return target

    # UI 選單（持久化）
    if ui_sel and ui_sel != "All" and ui_sel in titles:
        return ui_sel

    return None

def decide_route_plan_llm(
    client: OpenAI,
    question: str,
    *,
    has_index: bool,
    allow_web: bool,
    run_messages: Optional[list[dict]] = None,
) -> RoutePlan:
    hist = ""
    if run_messages:
        lines = []
        for m in run_messages[-10:]:
            role = (m.get("role") or "").strip()
            content = (m.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                if len(content) > 700:
                    content = content[:700] + "…"
                lines.append(f"{role.upper()}: {content}")
        hist = "\n".join(lines).strip()

    system = (
        "你是 RAG 路由器，只輸出 JSON。\n"
        "schema：{\n"
        ' "mode":"smalltalk"|"direct"|"rag"|"deepagent"|"advisor"|"clarify",\n'
        ' "difficulty":"easy"|"medium"|"hard",\n'
        ' "enable_web":true|false,\n'
        ' "doc_top_k":6~12,\n'
        ' "facets":2~4,\n'
        ' "needs_clarification":true|false,\n'
        ' "clarifying_questions":["..."],\n'
        ' "reason":"..."\n'
        "}\n"
        "規則：\n"
        "- has_index=false：不得選 rag/deepagent\n"
        "- clarify：缺關鍵約束，先問 1~3 題\n"
        "- rag：文件內單點可答且需引用\n"
        "- deepagent：多面向整合/跨多段證據/嚴格引用\n"
        "- advisor：偏工作推進/決策/規劃，但必要時仍可查文件\n"
        "difficulty 定義：hard 只在多面向比較/多文件整合才用。\n"
        "- enable_web 只有 allow_web=true 且確實需要外部最新/法規/新聞/即時資訊才可 true\n"
    )
    user = (
        f"has_index={str(has_index).lower()}\n"
        f"allow_web={str(allow_web).lower()}\n\n"
        + (f"對話脈絡：\n{hist}\n\n" if hist else "")
        + f"問題：{question}"
    )
    out, _ = call_gpt(client, model=MODEL_MAIN, system=system, user=user, reasoning_effort=REASONING_EFFORT)
    data = _try_parse_json_or_py_literal(out) or {}

    mode = str(data.get("mode", "")).strip().lower()
    difficulty = str(data.get("difficulty", "medium")).strip().lower()
    enable_web = bool(data.get("enable_web", False)) and bool(allow_web)
    doc_top_k = max(6, min(12, int(data.get("doc_top_k", 10) or 10)))
    facets = max(2, min(4, int(data.get("facets", 3) or 3)))
    needs_clarification = bool(data.get("needs_clarification", False))
    clarifying_questions = data.get("clarifying_questions", []) or []
    clarifying_questions = [str(x).strip() for x in clarifying_questions if str(x).strip()][:3]
    reason = str(data.get("reason", "")).strip() or "（router 未提供原因）"

    if difficulty not in ("easy", "medium", "hard"):
        difficulty = "medium"
    if not has_index and mode in ("rag", "deepagent"):
        mode = "direct"
    if mode not in ("smalltalk", "direct", "rag", "deepagent", "advisor", "clarify"):
        mode = "deepagent" if has_index else "direct"
    if mode == "clarify":
        needs_clarification = True
        if not clarifying_questions:
            clarifying_questions = [
                "你要我比較/整合的是哪些報告（或全部上傳報告）？",
                "你希望輸出是：差異比較表、重點摘要、還是結論+證據？",
            ]

    return RoutePlan(
        mode=mode,
        difficulty=difficulty,
        allow_web=bool(allow_web),
        enable_web=enable_web,
        doc_top_k=doc_top_k,
        facets=facets,
        needs_clarification=needs_clarification,
        clarifying_questions=clarifying_questions,
        reason=reason,
    )


def decide_route_plan(
    client: OpenAI,
    question: str,
    *,
    has_index: bool,
    allow_web: bool,
    run_messages: Optional[list[dict]] = None,
) -> RoutePlan:
    rule_mode = rule_route_mode(question, has_index=has_index)
    if rule_mode:
        difficulty = "hard" if rule_mode == "deepagent" and any(k in (question or "") for k in ("比較", "差異", "對照", "跨")) else "medium"
        return RoutePlan(
            mode=rule_mode,
            difficulty=difficulty,
            allow_web=bool(allow_web),
            enable_web=False,
            doc_top_k=10,
            facets=3,
            needs_clarification=False,
            clarifying_questions=[],
            reason=f"rule_route:{rule_mode}",
        )
    return decide_route_plan_llm(client, question, has_index=has_index, allow_web=allow_web, run_messages=run_messages)


def grade_doc_evidence_sufficiency(client: OpenAI, question: str, ctx: str) -> float:
    if not (ctx or "").strip():
        return 0.0
    system = "你是檢索品質評分器。評估文件摘錄是否足以回答問題。只輸出 0~1 小數。"
    user = f"問題：{question}\n\n文件摘錄：\n{ctx}\n\n分數："
    out, _ = call_gpt(client, model=MODEL_GRADER, system=system, user=user, reasoning_effort="low")
    s = (out or "").strip()
    m = re.search(r"(0\.\d+|1(?:\.0+)?|0(?:\.0+)?)", s)
    if not m:
        return 0.45
    try:
        return max(0.0, min(1.0, float(m.group(1))))
    except Exception:
        return 0.45

# ========= [A] 新增：放在 Router 區塊附近（utilities） =========
def has_built_index() -> bool:
    store = st.session_state.get("store", None)
    try:
        return bool(store is not None and getattr(store, "index", None) is not None and store.index.ntotal > 0)
    except Exception:
        return False


def force_mode_when_indexed(plan_mode: str, has_index: bool) -> str:
    """
    你拍板的規則：只要有索引，一律走 advisor（讓 DeepAgent 自己評估怎麼回）。
    """
    if has_index:
        return "advisor"
    return plan_mode
    
# =========================
# Fallback RAG
# =========================
def fallback_answer_from_store(
    client: OpenAI,
    store: Optional[FaissStore],
    question: str,
    *,
    k: int = 10,
    difficulty: str = "medium",
) -> str:
    q = (question or "").strip()
    if not q:
        return "（系統：問題為空，無法產生回答）"

    if store is None or getattr(store, "index", None) is None or store.index.ntotal == 0:
        system = ANYA_SYSTEM_PROMPT
        ans, _ = call_gpt(client, model=MODEL_MAIN, system=system, user=q, reasoning_effort=REASONING_EFFORT)
        return ans or "（系統：無索引且模型未產出內容）"

    qvec = embed_texts(client, [q])
    hits = store.search_hybrid(q, qvec, k=max(4, min(12, int(k))), difficulty=difficulty)
    chunks = [ch for _, ch in hits]
    ctx = render_chunks_for_model(chunks, max_chars_each=900)

    system = (
        "你是嚴謹研究助理，只能根據資料回答，不可腦補。\n"
        "輸出：純 bullet（每行 -），每個 bullet 句尾必有引用 [報告名稱 pN]。\n"
        "注意：不要輸出內部流程字樣。\n"
    )
    user = f"問題：{q}\n\n資料：\n{ctx}\n"
    out, _ = call_gpt(client, model=MODEL_MAIN, system=system, user=user, reasoning_effort=REASONING_EFFORT)
    out = strip_internal_process_lines((out or "").strip())
    return out or "（系統：fallback RAG 未產出內容）"


# =========================
# DeepAgents create compat + runner
# =========================
def _require_deepagents() -> None:
    if HAS_DEEPAGENTS:
        return
    st.error("DeepAgent 依賴載入失敗（可能是版本/依賴不相容）。")
    if DEEPAGENTS_IMPORT_ERRORS:
        st.markdown("### 依賴錯誤細節")
        for msg in DEEPAGENTS_IMPORT_ERRORS:
            st.code(msg)
    st.stop()


# ========= [A] 整段替換：_create_deep_agent_compat（用這個版本蓋掉原本的） =========
def _create_deep_agent_compat(**kwargs):
    """
    兼容不同版本的 create_deep_agent。
    目標：
    - system prompt 參數命名差異相容
    - subagents 參數命名差異相容（subagents / sub_agents / agents）
    """
    sig = inspect.signature(create_deep_agent)
    allowed = set(sig.parameters.keys())

    payload = {}

    # 1) 先處理 subagents 參數命名差異
    if "subagents" in kwargs:
        if "subagents" in allowed:
            payload["subagents"] = kwargs["subagents"]
        elif "sub_agents" in allowed:
            payload["sub_agents"] = kwargs["subagents"]
        elif "agents" in allowed:
            payload["agents"] = kwargs["subagents"]
        # 若都不支援，就不塞（至少主 agent 還能回 message）

    # 2) 一般參數：只傳該版本有的
    for k, v in kwargs.items():
        if k in ("subagents",):
            continue
        if k in allowed:
            payload[k] = v

    # 3) system prompt 參數命名差異
    if "system_prompt" in kwargs:
        sp = kwargs["system_prompt"]
        if "system_prompt" in allowed:
            payload["system_prompt"] = sp
        else:
            # 常見替代命名
            if "prompt" in allowed:
                payload["prompt"] = sp
            elif "system_message" in allowed:
                payload["system_message"] = sp
            elif "instructions" in allowed:
                payload["instructions"] = sp
            elif "state_modifier" in allowed:
                payload["state_modifier"] = sp
            elif "messages_modifier" in allowed:
                payload["messages_modifier"] = sp

    return create_deep_agent(**payload)


def _agent_stream_with_files(agent, input_state: dict, *, files_seed: Optional[dict], stream_mode: str, config: dict):
    """
    deepagents docs：StateBackend 下 skills 需用 invoke(files=...) 提供。([reference.langchain.com](https://reference.langchain.com/python/deepagents/graph/))
    這裡做相容：先嘗試 agent.stream(..., files=seed)，不行再 fallback input_state["files"]=seed
    """
    if files_seed:
        try:
            return agent.stream(input_state, stream_mode=stream_mode, config=config, files=files_seed)
        except TypeError:
            # fallback：某些 runnable 不接受 files= kwarg
            state2 = dict(input_state)
            state2["files"] = files_seed
            return agent.stream(state2, stream_mode=stream_mode, config=config)

    return agent.stream(input_state, stream_mode=stream_mode, config=config)

# ========= (3) 新增：通用「選文件 + 檢索 + 回答」工具（放在 fallback_answer_from_store() 後面、ensure_deep_agent() 前面） =========
REPORT_REF_KWS = ["這份報告", "這一份報告", "本報告", "該報告", "此報告", "這篇報告", "這份文件", "本文件", "該文件", "此文件"]

def mentions_report_reference(q: str) -> bool:
    q = (q or "").strip()
    return any(k in q for k in REPORT_REF_KWS)

def list_report_titles_from_store(store: Optional[FaissStore]) -> list[str]:
    if store is None:
        return []
    seen = set()
    titles = []
    for c in (store.chunks or []):
        t = (c.title or "").strip()
        if not t or t in seen:
            continue
        seen.add(t)
        titles.append(t)
    return sorted(titles)

def guess_target_title(prompt: str, titles: list[str], *, last_title: Optional[str]) -> Optional[str]:
    """
    選文件順序：
    1) 問題文字明確包含 title -> 用最長匹配
    2) 只有一份 -> 用那份
    3) 有「這份/本報告」等指涉字 -> 用 last_title（若存在）
    4) 否則 None（表示跨文件問答，或需要你指定）
    """
    titles = [t for t in (titles or []) if (t or "").strip()]
    q = (prompt or "").strip()
    if not titles:
        return None

    q_low = q.lower()
    matched = [t for t in titles if t.lower() in q_low or norm_space(t).lower() in norm_space(q).lower()]
    if matched:
        matched.sort(key=lambda x: len(x), reverse=True)
        return matched[0]

    if len(titles) == 1:
        return titles[0]

    if mentions_report_reference(q) and last_title and last_title in titles:
        return last_title

    return None

def retrieve_hits(
    client: OpenAI,
    store: FaissStore,
    query: str,
    *,
    title: Optional[str],
    k: int,
    difficulty: str,
) -> list[tuple[float, Chunk]]:
    """
    通用檢索：先全庫搜，再視需要過濾到指定 title。
    若指定 title 過濾後太少，會再做一次「title + query」的 bias 搜尋補強。
    """
    q = (query or "").strip()
    if not q:
        return []

    k = max(4, min(12, int(k)))
    qvec = embed_texts(client, [q])
    hits = store.search_hybrid(q, qvec, k=k, difficulty=difficulty)

    if not title:
        return hits

    title = (title or "").strip()
    filtered = [(s, ch) for (s, ch) in hits if (ch.title or "").strip() == title]

    # 不夠就用 title bias 再搜一次補強
    if len(filtered) < max(2, k // 2):
        q2 = f"{title}\n{q}"
        q2vec = embed_texts(client, [q2])
        hits2 = store.search_hybrid(q2, q2vec, k=k, difficulty=difficulty)
        filtered2 = [(s, ch) for (s, ch) in hits2 if (ch.title or "").strip() == title]

        # merge（去重 chunk_id）
        seen = set()
        merged: list[tuple[float, Chunk]] = []
        for s, ch in (filtered + filtered2):
            if ch.chunk_id in seen:
                continue
            seen.add(ch.chunk_id)
            merged.append((s, ch))
        return merged[:k]

    return filtered[:k]

def render_retriever_hits_expander(hits: list[tuple[float, Chunk]], *, label: str = "🔎 Retriever 命中內容（節錄）") -> None:
    if not st.session_state.get("show_retriever_hits_expander", True):
        return
    expanded = bool(st.session_state.get("retriever_hits_expanded_by_default", False))
    max_show = int(st.session_state.get("retriever_hits_max_per_query", 6) or 6)

    with st.expander(label, expanded=expanded):
        if not hits:
            st.markdown("（沒有命中任何內容）")
            return
        for score, ch in hits[:max_show]:
            head = f"[{ch.title} p{ch.page if ch.page is not None else '-'}]"
            snippet = (ch.text or "").strip().replace("\n", " ")
            if len(snippet) > 520:
                snippet = snippet[:520] + "…"
            try:
                s = f"{float(score):.3f}"
            except Exception:
                s = str(score)
            st.markdown(f"- **{head}** score={s}：{snippet}")

def answer_from_hits(
    client: OpenAI,
    question: str,
    hits: list[tuple[float, Chunk]],
) -> str:
    """
    真正的「解問題」：把命中的 chunk 當 Context，要求模型只能依據 Context 回答並附引用。
    """
    chunks = [ch for _s, ch in (hits or [])]
    ctx = render_chunks_for_model(chunks, max_chars_each=900)

    if not ctx.strip():
        return "資料不足：目前檢索不到可用的文件段落來回答這個問題。你可以：\n- 換個問法（加上關鍵字/章節/頁碼線索）\n- 或指定要看哪一份報告"

    system = (
        "你是嚴謹的文件問答助理，只能根據「文件摘錄」回答，不可腦補。\n"
        "用繁體中文（台灣用語）。\n"
        "輸出規則：\n"
        "- 先直接回答問題（條列為主）。\n"
        "- 只要是事實/判斷/引用文件內容的句子，句尾都要有引用 token：[報告名稱 pN]（N 可為 -）。\n"
        "- 若文件摘錄不足以支持，就明確寫『資料不足』並說需要補什麼（<=3 點）。\n"
        "- 不要只給大綱；要回答使用者的問句。\n"
    )
    user = f"問題：{question}\n\n文件摘錄：\n{ctx}\n"
    out, _ = call_gpt(client, model=MODEL_MAIN, system=system, user=user, reasoning_effort=REASONING_EFFORT)
    out = strip_internal_process_lines((out or "").strip())
    return out or "（系統：模型未產出內容）"

def ensure_deep_agent(client: OpenAI, store: FaissStore, enable_web: bool):
    _require_deepagents()
    from langchain_core.tools import BaseTool, StructuredTool

    st.session_state.setdefault("deep_agent", None)
    st.session_state.setdefault("deep_agent_web_flag", None)
    st.session_state.setdefault("da_usage", {"doc_search_calls": 0, "web_search_calls": 0})

    st.session_state.setdefault("ui_doc_search_log", [])
    st.session_state.setdefault("ui_last_doc_list", "")

    cfg_sig = json.dumps(
        {
            "enable_web": bool(enable_web),
            "enable_skills": bool(st.session_state.get("da_enable_skills", True)),
            "enable_memory": bool(st.session_state.get("da_enable_memory", True)),
            "skill_claims_first": bool(st.session_state.get("da_skill_claims_first", True)),
            "skill_decision_memo": bool(st.session_state.get("da_skill_decision_memo", True)),
            "skill_report_compare": bool(st.session_state.get("da_skill_report_compare", True)),
            "skill_action_plan": bool(st.session_state.get("da_skill_action_plan", True)),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    st.session_state.setdefault("deep_agent_cfg_sig", None)

    if (st.session_state.deep_agent is not None) and (st.session_state.deep_agent_cfg_sig == cfg_sig):
        return st.session_state.deep_agent

    lock = threading.Lock()
    usage = {"doc_search_calls": 0, "web_search_calls": 0}
    st.session_state["da_usage"] = usage

    def _inc(name: str, limit: int) -> bool:
        with lock:
            usage[name] += 1
            st.session_state["da_usage"] = usage
            return usage[name] <= limit

    def _get_usage_fn() -> str:
        with lock:
            return json.dumps(usage, ensure_ascii=False)

    def _doc_list_fn() -> str:
        by_title: Dict[str, int] = {}
        for c in store.chunks:
            by_title[c.title] = by_title.get(c.title, 0) + 1
        lines = [f"- {t} (chunks={n})" for t, n in sorted(by_title.items(), key=lambda x: x[0])]
        out = "\n".join(lines) if lines else "（目前沒有任何已索引文件）"
        st.session_state["ui_last_doc_list"] = out
        return out

    def _doc_search_fn(query: str, k: int = 8, title_filter: str = "") -> str:
        """
        Hybrid search。新增 title_filter：
        - 空字串/All => 不過濾（跨所有文件）
        - 否則只回傳該 title 的 chunks（硬限制 scope）
        """
        if not _inc("doc_search_calls", DA_MAX_DOC_SEARCH_CALLS):
            return json.dumps({"hits": [], "error": f"Budget exceeded: doc_search_calls > {DA_MAX_DOC_SEARCH_CALLS}"}, ensure_ascii=False)

        q = (query or "").strip()
        if not q:
            return json.dumps({"hits": []}, ensure_ascii=False)

        tf = (title_filter or "").strip()
        if tf.lower() == "all":
            tf = ""

        qvec = embed_texts(client, [q])
        k2 = max(1, min(24, int(k)))
        difficulty = str(st.session_state.get("current_difficulty", "medium") or "medium").lower()

        # 先多抓一些候選，避免 scope 過濾後不夠
        hits = store.search_hybrid(q, qvec, k=max(k2, 16), difficulty=difficulty)

        if tf:
            hits = [(s, ch) for (s, ch) in hits if (ch.title or "").strip() == tf]

        hits = hits[: min(k2, 12)]

        payload = {"hits": []}
        ui_hits = []
        for score, ch in hits:
            citation_token = f"[{ch.title} p{ch.page if ch.page is not None else '-'}]"
            item = {
                "title": ch.title,
                "page": str(ch.page) if ch.page is not None else "-",
                "citation_token": citation_token,
                "chunk_id": ch.chunk_id,  # internal only（不得寫入 evidence）
                "text": (ch.text or "")[:1200],
                "score": float(score),
            }
            payload["hits"].append(item)
            ui_hits.append(
                {
                    "title": ch.title,
                    "page": str(ch.page) if ch.page is not None else "-",
                    "text": (ch.text or "")[:260],
                    "score": float(score),
                }
            )

        with lock:
            log = st.session_state.get("ui_doc_search_log", []) or []
            log.append({"query": q, "k": k2, "hits": ui_hits[:6], "title_filter": tf or "All"})
            st.session_state["ui_doc_search_log"] = log[-UI_MAX_DOC_SEARCH_LOG:]

        return json.dumps(payload, ensure_ascii=False)

    def _doc_get_chunk_fn(chunk_id: str, max_chars: int = 2600) -> str:
        cid = (chunk_id or "").strip()
        if not cid:
            return ""
        for c in store.chunks:
            if c.chunk_id == cid:
                return (c.text or "")[:max_chars]
        return ""

    def _grade_doc_evidence_fn(question: str, evidence: str) -> str:
        q = (question or "").strip()
        ev = (evidence or "").strip()
        s = grade_doc_evidence_sufficiency(client, q, ev)
        return json.dumps({"score": float(s)}, ensure_ascii=False)

    def _mk_tool(fn, name: str, description: str) -> BaseTool:
        return StructuredTool.from_function(fn, name=name, description=description)

    # ✅ 一定要先定義 tools，下面 subagents 才能引用
    tool_get_usage = _mk_tool(_get_usage_fn, "get_usage", "Get current tool usage counters as JSON (budget/debug).")
    tool_doc_list = _mk_tool(_doc_list_fn, "doc_list", "List indexed documents and chunk counts.")
    tool_doc_search = _mk_tool(_doc_search_fn, "doc_search", "Hybrid search over indexed chunks. Returns JSON hits with title/page/citation_token/chunk_id/text.")
    tool_doc_get_chunk = _mk_tool(_doc_get_chunk_fn, "doc_get_chunk", "Fetch full text for a given chunk_id for close reading. Returns text only.")
    tool_grade_doc = _mk_tool(_grade_doc_evidence_fn, "grade_doc_evidence", "Grade whether document evidence is sufficient. Returns JSON {score}.")

    tools: list[BaseTool] = [tool_get_usage, tool_doc_list, tool_doc_search, tool_doc_get_chunk, tool_grade_doc]

    tool_web_search_summary: Optional[BaseTool] = None
    if enable_web:
        def _web_search_summary_fn(query: str) -> str:
            if not _inc("web_search_calls", DA_MAX_WEB_SEARCH_CALLS):
                return "[WebSearch:web p-]\nSources:"

            q = (query or "").strip()
            if not q:
                return "[WebSearch:web p-]\nSources:"

            system = (
                "你是研究助理。輸出格式固定：\n"
                "1) 3~8 個 bullets 摘要（每點一句，必要時含日期/數字）。\n"
                "2) Sources: 之後列來源，每行：- <domain> | <title> | <url>\n"
                "規則：不要提工具流程/額度。\n"
            )
            user = f"Search term: {q}"
            text, sources = call_gpt(
                client,
                model=MODEL_WEB,
                system=system,
                user=user,
                reasoning_effort=None,
                tools=[{"type": "web_search"}],
                include_sources=True,
            )

            src_lines = []
            for s in (sources or [])[:10]:
                if isinstance(s, dict):
                    t = (s.get("title") or s.get("source") or "source").strip()
                    u = (s.get("url") or "").strip()
                    if u:
                        src_lines.append(f"- {_domain(u)} | {t} | {u}")

            out_text = (text or "").strip()
            if "Sources:" not in out_text:
                out_text = (out_text + "\n\nSources:").strip()
            if src_lines:
                out_text = (out_text + "\n" + "\n".join(src_lines)).strip()

            primary_domain = "web"
            if sources and isinstance(sources, list):
                u0 = ((sources[0] or {}).get("url") or "").strip() if isinstance(sources[0], dict) else ""
                if u0:
                    primary_domain = _domain(u0)

            return f"[WebSearch:{primary_domain} p-]\n" + out_text[:2400]

        tool_web_search_summary = _mk_tool(
            _web_search_summary_fn,
            "web_search_summary",
            "Run web_search and return a short Traditional Chinese summary with sources.",
        )
        tools.append(tool_web_search_summary)

    # ===== Subagent prompts =====
    retriever_prompt = f"""
你是文件檢索專家（只能用 doc_list/doc_search/doc_get_chunk/get_usage）。
你會收到 runtime 設定在 /runtime/runtime.json（scope_title/web_threshold/question_kind/allow_web）。

任務：針對 facet 找證據，寫入 /evidence/doc_<facet_slug>.md

facet 格式：
facet_slug: <英文小寫_底線>
facet_goal: <要回答什麼>
hints: <關鍵字可空>

檢索規則（很重要）：
- 若 scope_title != "All"：doc_search 時要帶 title_filter=scope_title（硬限制只查那份文件）
- 每個 facet 至少做 2 次 doc_search（multi-query：原句 + 精簡關鍵詞/同義詞）
- evidence 裡每則引用只能使用 doc_search hits 給的 citation_token（例如 [報告名 p12]）
- chunk_id 絕對不能寫進 evidence（只能 internal 使用）

evidence 內容格式（固定）：
1) 一行引用標頭：<citation_token>
2) 一段原文片段（可截斷）
3) 一行說明「這段支持什麼」

最後回 orchestrator：≤150 字摘要（找到什麼 + 最大缺口）
""".strip()

    analyst_prompt = f"""
你是推理分析專家（不做檢索；只讀 evidence 產物）。
你要做兩份結構化產物：

1) /analysis/claims.json
- JSON array，最多 {DA_MAX_CLAIMS} 條
- 每條包含：
  - claim（可核對的一句話）
  - citations（array；元素必須是像 [報告名稱 pN] 或 [WebSearch:domain p-]）
  - assumptions（array，可空）
  - confidence（0~1 float）

2) /analysis/reflections.json（反思）
- JSON array，至少 2 條
- 每條包含：
  - reflection（盲點/風險/反例/需驗證點）
  - citations（array；可空）
  - needs_validation（boolean；若 citations 空，必須 true）
  - impact（若成立會如何影響結論/決策）

硬規則：
- 只能依據 evidence 內看到的內容
- 禁止任何內部字樣（chunk_id、/evidence、Budget exceeded 等）
""".strip()

    web_prompt = """
你是網路搜尋專家（只允許 web_search_summary/get_usage；不允許 doc_*）。
對每個 facet：寫入 /evidence/web_<facet_slug>.md
硬規則：
- 每段要保留引用標頭 [WebSearch:<domain> p-]
- 禁止捏造來源；不要寫工具流程/額度字樣
""".strip()

    writer_prompt = """
你是寫作/整理專家。你會收到：
- /runtime/runtime.json（question_kind: chat|memo）
- evidence（doc_*.md 與可選 web_*.md）
- claims/reflections
→ 產生 /draft.md

輸出風格：
- question_kind="chat"：一般聊天式回答；只有「關鍵事實句」句尾要引用 token
- question_kind="memo"：Decision Memo；重點結論句尾必有引用 token

硬規則：
- 引用 token 必須是 evidence 中出現過的 token（或 [WebSearch:* p-]）
- 不可腦補；不足就寫「資料不足」並列 1~3 項需要補的資訊
- /draft.md 不得出現內部字樣（/evidence、/analysis、doc_、web_、Budget exceeded、chunk_id）
""".strip()

    verifier_prompt = f"""
你是審稿查核專家：檢查 /draft.md 是否符合引用覆蓋與禁則，做最少改動修正。
規則：
- 不得出現 chunk_id、/evidence、/analysis、doc_、web_、Budget exceeded 等內部字樣
- draft 中的引用 token 必須能在 evidence 裡找到同樣的 token（或是 [WebSearch:* p-]）
- 若出現不存在的引用：優先刪掉該句或改成「資料不足」，不要亂補頁碼/標題
最多修正 {DA_MAX_REWRITE_ROUNDS} 輪。
""".strip()

    orchestrator_prompt = f"""
你是 Deep Supervisor（Agentic RAG）。請先讀 /runtime/runtime.json（scope_title/web_threshold/allow_web/question_kind）。

核心策略（gate）：
- 先文件後網路：每個 facet 先做 doc 蒐證
- 只有當「文件 evidence 不足」且 allow_web=true 時，才可以考慮 web
- 必須用工具 grade_doc_evidence(question, evidence) 評分：
  - score < web_threshold（你這裡會看到 runtime 內的門檻） 才能派 web-researcher
  - score >= web_threshold 則禁止 web

固定流程：
0) todos（7~12 步）
1) facets（2~4 個）
2) 平行派 retriever
3) analyst → claims/reflections
4) 視需要用 grade_doc_evidence 決定是否派 web-researcher
5) writer → /draft.md
6) verifier 修稿（最多 {DA_MAX_REWRITE_ROUNDS} 輪）
7) read_file /draft.md 作為最終回答

注意：
- 最終輸出不可提內部流程/檔名/額度
- 不要輸出 chain-of-thought
""".strip()

    llm = _make_langchain_llm(model_name=f"openai:{MODEL_MAIN}", temperature=0.0, reasoning_effort=REASONING_EFFORT)

    # ✅ subagents：你的 deepagents 版本「一定要」system_prompt
    # 同時保留 prompt 欄位做跨版本相容（有些版本可能吃 prompt）
    def _mk_subagent(
        *,
        name: str,
        description: str,
        system_prompt: str,
        tools: list,
        model: str,
    ) -> dict:
        return {
            "name": name,
            "description": description,
            "system_prompt": system_prompt,  # ✅ 你這版 deepagents 會讀這個 key
            "prompt": system_prompt,         # ✅ 保留給其他可能吃 prompt 的版本（不影響）
            "tools": tools,
            "model": model,
        }

    subagents = [
        _mk_subagent(
            name="retriever",
            description="從文件索引找證據，寫 /evidence/doc_*.md（不含 chunk_id）",
            system_prompt=retriever_prompt,
            tools=[tool_get_usage, tool_doc_list, tool_doc_search, tool_doc_get_chunk],
            model=f"openai:{MODEL_MAIN}",
        ),
        _mk_subagent(
            name="analyst",
            description="claims-first 推理分析，產出 claims/reflections",
            system_prompt=analyst_prompt,
            tools=[],
            model=f"openai:{MODEL_MAIN}",
        ),
        _mk_subagent(
            name="writer",
            description="整合 evidence + claims/reflections → 產生 draft",
            system_prompt=writer_prompt,
            tools=[],
            model=f"openai:{MODEL_MAIN}",
        ),
        _mk_subagent(
            name="verifier",
            description="檢查引用覆蓋並修稿 draft",
            system_prompt=verifier_prompt,
            tools=[],
            model=f"openai:{MODEL_MAIN}",
        ),
    ]

    if enable_web and tool_web_search_summary is not None:
        subagents.insert(
            1,
            _mk_subagent(
                name="web-researcher",
                description="用 web_search 補外部背景，寫 /evidence/web_*.md",
                system_prompt=web_prompt,
                tools=[tool_web_search_summary, tool_get_usage],
                model=f"openai:{MODEL_MAIN}",
            ),
        )
    # 若沒啟用 web，把 web-researcher 拿掉（避免 subagent 誤用）
    if not (enable_web and tool_web_search_summary is not None):
        subagents = [a for a in subagents if a.get("name") != "web-researcher"]

    memory = ["/memory/AGENTS.md"] if st.session_state.get("da_enable_memory", True) else None
    skills = ["/skills/"] if st.session_state.get("da_enable_skills", True) else None

    agent = _create_deep_agent_compat(
        model=llm,
        tools=tools,
        system_prompt=orchestrator_prompt,
        subagents=subagents,
        debug=False,
        name="deep-doc-agent",
        memory=memory,
        skills=skills,
    ).with_config({"recursion_limit": int(st.session_state.get("langgraph_recursion_limit", DEFAULT_RECURSION_LIMIT))})

    st.session_state.deep_agent = agent
    st.session_state.deep_agent_cfg_sig = cfg_sig
    st.session_state.deep_agent_web_flag = bool(enable_web)
    return agent


def _safe_json_preview(text: str, max_chars: int = 1400) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    if len(t) > max_chars:
        return t[:max_chars] + "…"
    return t

# ========= [1/2] 新增：放在 deep_agent_run_with_live_status 之前（建議放在 _safe_json_preview 後面） =========
def _coerce_file_text(v: Any) -> str:
    """
    deepagents state["files"][path] 可能是：
    - str（最常見）
    - dict/list（已解析的 JSON）
    - 其他型別（少見）
    統一轉成「可顯示/可 .strip()」的字串，避免 AttributeError。
    """
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, (dict, list)):
        try:
            return json.dumps(v, ensure_ascii=False, indent=2)
        except Exception:
            return str(v)
    try:
        return str(v)
    except Exception:
        return ""

# ========= [B] 新增：放在 deep_agent_run_with_live_status 之前（你之前加的 _coerce_file_text 可以保留；我這裡補一個讀 messages 的 helper） =========
def _extract_last_assistant_text_from_state(state: Optional[dict]) -> str:
    """
    deepagents / langgraph 回傳的 state["messages"] 可能是：
    - list[dict]  ({"role":"assistant","content":...})
    - list[BaseMessage]（AIMessage/HumanMessage...）
    這裡統一抓最後一則 assistant 的文字內容。
    """
    if not state or not isinstance(state, dict):
        return ""
    msgs = state.get("messages") or []
    if not isinstance(msgs, list):
        return ""

    for m in reversed(msgs):
        # dict 格式
        if isinstance(m, dict):
            role = (m.get("role") or "").strip().lower()
            if role == "assistant":
                c = m.get("content")
                return (c if isinstance(c, str) else str(c or "")).strip()
            continue

        # LangChain message 物件
        role = ""
        try:
            role = (getattr(m, "type", "") or getattr(m, "role", "") or "").strip().lower()
        except Exception:
            role = ""

        if role in ("ai", "assistant"):
            try:
                c = getattr(m, "content", "")
                return (c if isinstance(c, str) else str(c or "")).strip()
            except Exception:
                return ""

    return ""

# ========= [C] 整段替換：deep_agent_run_with_live_status（用這個版本蓋掉你目前那個） =========
def deep_agent_run_with_live_status(agent, user_text: str, run_messages: list[dict], client: OpenAI, status=None) -> Tuple[str, Optional[dict]]:
    """
    ✅ 共用同一個 st.status
    ✅ 沒有 /draft.md 時：改抓最後 assistant message
    ✅ 若連 message 都沒有：fallback RAG（避免 UI 出現「沒產出內容」）
    """
    final_state = None
    st.session_state["last_run_forced_end"] = None

    recursion_limit = int(st.session_state.get("langgraph_recursion_limit", DEFAULT_RECURSION_LIMIT))
    stall_steps = int(st.session_state.get("citation_stall_steps", DEFAULT_CITATION_STALL_STEPS))
    stall_min_chars = int(st.session_state.get("citation_stall_min_chars", DEFAULT_CITATION_STALL_MIN_CHARS))

    draft_unchanged_streak = 0
    draft_no_citation_streak = 0
    last_draft_hash: Optional[str] = None

    def set_phase(s, phase: str):
        mapping = {
            "start": ("DeepAgent：啟動中…", "running"),
            "plan": ("DeepAgent：規劃中（todos/facets）…", "running"),
            "evidence": ("DeepAgent：蒐證中（doc_search）…", "running"),
            "analysis": ("DeepAgent：推理中（claims/反思）…", "running"),
            "draft": ("DeepAgent：寫作中（draft）…", "running"),
            "review": ("DeepAgent：審稿/補引用中（review）…", "running"),
            "done": ("DeepAgent：完成", "complete"),
            "error": ("DeepAgent：發生錯誤", "error"),
        }
        label, state = mapping.get(phase, ("DeepAgent：執行中…", "running"))
        s.update(label=label, state=state, expanded=bool(st.session_state.get("da_status_expanded", False)))

    msgs_for_agent = list(run_messages or [])
    if not msgs_for_agent or msgs_for_agent[-1].get("role") != "user":
        msgs_for_agent.append({"role": "user", "content": user_text})
    elif (msgs_for_agent[-1].get("content") or "").strip() != (user_text or "").strip():
        msgs_for_agent.append({"role": "user", "content": user_text})

    seed_files = build_seed_files_for_deepagents()

    show_debug = bool(st.session_state.get("da_show_status_debug", True))
    show_files = bool(st.session_state.get("da_show_status_files", True))
    show_doc_hits = bool(st.session_state.get("da_show_status_doc_hits", True))

    if status is None:
        status_cm = st.status("DeepAgent：啟動中…", expanded=bool(st.session_state.get("da_status_expanded", False)))
        s = status_cm.__enter__()
        _need_exit = True
    else:
        s = status
        _need_exit = False

    try:
        set_phase(s, "start")

        memo_ph = st.empty()
        doc_hits_ph = st.empty()
        files_ph = st.empty()

        def _safe(s0: str, max_chars: int = 1200) -> str:
            s0 = (s0 or "").strip()
            return s0 if len(s0) <= max_chars else s0[:max_chars] + "…"

        def _get_text(files: dict, path: str) -> str:
            return _coerce_file_text((files or {}).get(path)).strip()

        def _render_doc_hits():
            if not show_doc_hits:
                return
            log = st.session_state.get("ui_doc_search_log", []) or []
            if not log:
                doc_hits_ph.markdown(":small[（尚未觸發 doc_search）]")
                return
            lines = ["#### 🔎 最近文件檢索命中（Top3 節錄）"]
            for item in log[-UI_MAX_DOC_SEARCH_LOG:][::-1]:
                q = item.get("query") or ""
                lines.append(f"- **Query**：{q}")
                for h in (item.get("hits") or [])[:3]:
                    title = h.get("title") or ""
                    page = h.get("page") or "-"
                    snippet = (h.get("text") or "").replace("\n", " ")
                    score = h.get("score")
                    try:
                        score_s = f"{float(score):.3f}"
                    except Exception:
                        score_s = str(score)
                    lines.append(f"  - [{title} p{page}] score={score_s}：{snippet}")
            doc_hits_ph.markdown("\n".join(lines))

        def _render_memo(files: dict):
            if not show_debug:
                return
            todos = _get_text(files, "/workspace/todos.json")
            facets = _get_text(files, "/workspace/facets.json")
            claims = _get_text(files, "/analysis/claims.json")
            refl = _get_text(files, "/analysis/reflections.json")

            blocks = []
            if todos:
                blocks.append("#### 📝 Todos\n```json\n" + _safe(todos, 1400) + "\n```")
            if facets:
                blocks.append("#### 🧭 Facets\n```json\n" + _safe(facets, 1400) + "\n```")
            if claims and st.session_state.get("da_show_claims", True):
                blocks.append("#### 🧠 Claims\n```json\n" + _safe(claims, 1600) + "\n```")
            if refl and st.session_state.get("da_show_reflections", True):
                blocks.append("#### 🤔 反思\n```json\n" + _safe(refl, 1600) + "\n```")

            memo_ph.markdown("\n\n".join(blocks) if blocks else ":small[（尚未產生 todos/facets/claims/反思）]")

        def _render_files_preview(files: dict):
            if not (show_files and show_debug):
                return
            keys = sorted([k for k in (files or {}).keys() if isinstance(k, str)])
            evidence_keys = [k for k in keys if k.startswith("/evidence/")][:12]
            draft = _get_text(files, "/draft.md")
            review = _get_text(files, "/review.md")

            lines = []
            if evidence_keys:
                lines.append("#### 📎 Evidence（節錄）")
                for k in evidence_keys:
                    t = _coerce_file_text((files or {}).get(k))
                    if not isinstance(t, str) or not t.strip():
                        continue
                    lines.append(f"- `{k}`\n\n```text\n{t[:UI_MAX_EVIDENCE_PREVIEW_CHARS]}\n```")
            if draft.strip():
                lines.append("#### 🧾 Draft（節錄）\n```markdown\n" + draft[:UI_MAX_DRAFT_PREVIEW_CHARS] + "\n```")
            if review.strip():
                lines.append("#### ✅ Review（節錄）\n```text\n" + review[:900] + "\n```")

            files_ph.markdown("\n\n".join(lines) if lines else ":small[（尚未產生 evidence/draft/review）]")

        set_phase(s, "plan")

        stream_iter = _agent_stream_with_files(
            agent,
            {"messages": msgs_for_agent},
            files_seed=seed_files,
            stream_mode="values",
            config={"recursion_limit": recursion_limit},
        )

        saw_any_state = False

        for state in stream_iter:
            saw_any_state = True
            final_state = state

            files = state.get("files") or {}
            files = files if isinstance(files, dict) else {}
            file_keys = set(files.keys())

            if "/analysis/claims.json" in file_keys or "/analysis/reflections.json" in file_keys:
                set_phase(s, "analysis")
            if any(isinstance(k, str) and k.startswith("/evidence/") for k in file_keys):
                set_phase(s, "evidence")
            if "/draft.md" in file_keys:
                set_phase(s, "draft")
            if "/review.md" in file_keys:
                set_phase(s, "review")

            _render_doc_hits()
            _render_memo(files)
            _render_files_preview(files)

            draft_txt = _coerce_file_text(files.get("/draft.md"))
            draft_norm = norm_space(draft_txt) if isinstance(draft_txt, str) else ""
            if draft_norm and len(draft_norm) >= stall_min_chars:
                h = _hash_norm_text(draft_norm)
                if last_draft_hash == h:
                    draft_unchanged_streak += 1
                else:
                    draft_unchanged_streak = 0
                    last_draft_hash = h

                if has_visible_citations(draft_norm):
                    draft_no_citation_streak = 0
                else:
                    draft_no_citation_streak += 1

                if (draft_unchanged_streak >= stall_steps) and (draft_no_citation_streak >= stall_steps):
                    set_phase(s, "error")
                    st.session_state["last_run_forced_end"] = "citation_stall"
                    s.warning("判定卡住（引用未生成），已改用 fallback。")
                    diff = str(st.session_state.get("current_difficulty", "medium") or "medium")
                    answer = fallback_answer_from_store(client, st.session_state.get("store", None), user_text, k=10, difficulty=diff)
                    return answer, (files if files else None)

        # stream 沒吐任何 state（少見）：直接 fallback
        if not saw_any_state:
            set_phase(s, "error")
            st.session_state["last_run_forced_end"] = "no_stream"
            diff = str(st.session_state.get("current_difficulty", "medium") or "medium")
            answer = fallback_answer_from_store(client, st.session_state.get("store", None), user_text, k=10, difficulty=diff)
            return answer, None

        files = (final_state or {}).get("files") or {}
        files = files if isinstance(files, dict) else {}
        draft = _coerce_file_text(files.get("/draft.md"))
        draft = strip_internal_process_lines(draft if isinstance(draft, str) else "")

        # ✅ 1) 有 draft 就用 draft
        if draft.strip():
            set_phase(s, "done")
            return draft.strip(), (files if files else None)

        # ✅ 2) 沒 draft：改抓最後 assistant message（很常見！）
        msg_text = _extract_last_assistant_text_from_state(final_state)
        msg_text = strip_internal_process_lines(msg_text)
        if msg_text.strip():
            set_phase(s, "done")
            return msg_text.strip(), (files if files else None)

        # ✅ 3) 連 message 都沒有：fallback（至少你一定拿得到答案）
        set_phase(s, "error")
        st.session_state["last_run_forced_end"] = "empty_output"
        diff = str(st.session_state.get("current_difficulty", "medium") or "medium")
        answer = fallback_answer_from_store(client, st.session_state.get("store", None), user_text, k=10, difficulty=diff)
        return answer, (files if files else None)

    except GraphRecursionError:
        set_phase(s, "error")
        st.session_state["last_run_forced_end"] = "recursion_limit"

        files = (final_state or {}).get("files") or {}
        files = files if isinstance(files, dict) else {}
        draft = _coerce_file_text(files.get("/draft.md"))
        draft = strip_internal_process_lines(draft if isinstance(draft, str) else "")
        if draft.strip():
            return draft.strip(), (files if files else None)

        msg_text = _extract_last_assistant_text_from_state(final_state)
        msg_text = strip_internal_process_lines(msg_text)
        if msg_text.strip():
            return msg_text.strip(), (files if files else None)

        diff = str(st.session_state.get("current_difficulty", "medium") or "medium")
        answer = fallback_answer_from_store(client, st.session_state.get("store", None), user_text, k=10, difficulty=diff)
        return answer, (files if files else None)

    finally:
        if _need_exit:
            status_cm.__exit__(None, None, None)


# =========================
# UI helpers
# =========================
def render_run_badges(*, mode: str, enable_web: bool, usage: dict, difficulty: str, scope_title: Optional[str] = None) -> None:
    badges: List[str] = []
    badges.append(_badge_directive(f"Mode:{mode}", "gray"))
    badges.append(_badge_directive(f"Diff:{difficulty}", "blue"))

    scope = (scope_title or st.session_state.get("selected_report_title") or "All")
    badges.append(_badge_directive(f"Scope:{scope}", "green" if scope != "All" else "gray"))

    doc_calls = int((usage or {}).get("doc_search_calls", 0) or 0)
    web_calls = int((usage or {}).get("web_search_calls", 0) or 0)
    badges.append(_badge_directive(f"DB:{doc_calls}", "green" if doc_calls else "gray"))
    badges.append(_badge_directive(f"Web:{web_calls}" if enable_web else "Web:off", "violet" if enable_web else "gray"))
    st.markdown(" ".join(badges))


def build_files_df(rows: list[FileRow]) -> pd.DataFrame:
    data = []
    for r in rows:
        data.append(
            {
                "file_id": r.file_id,
                "name": r.name,
                "ext": r.ext,
                "size_kb": round(r.bytes_len / 1024, 1),
                "pages": r.pages,
                "token_est": r.token_est,
                "likely_scanned": bool(r.likely_scanned),
                "blank_ratio": (None if r.blank_ratio is None else round(float(r.blank_ratio), 3)),
                "use_ocr": bool(r.use_ocr),
            }
        )
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values(["ext", "name"], ascending=[True, True]).reset_index(drop=True)
    return df


# =========================
# Session init
# =========================
OPENAI_API_KEY = get_openai_api_key()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ.setdefault("OPENAI_KEY", OPENAI_API_KEY)
client = get_client(OPENAI_API_KEY)

st.session_state.setdefault("file_rows", [])
st.session_state.setdefault("file_bytes", {})
st.session_state.setdefault("store", None)
st.session_state.setdefault("processed_keys", set())
st.session_state.setdefault("chat_history", [])

st.session_state.setdefault("enable_web_search_agent", True)

st.session_state.setdefault("langgraph_recursion_limit", DEFAULT_RECURSION_LIMIT)
st.session_state.setdefault("citation_stall_steps", DEFAULT_CITATION_STALL_STEPS)
st.session_state.setdefault("citation_stall_min_chars", DEFAULT_CITATION_STALL_MIN_CHARS)
st.session_state.setdefault("last_run_forced_end", None)

st.session_state.setdefault("enable_output_formatter", True)
st.session_state.setdefault("sources_badge_max_titles_inline", DEFAULT_SOURCES_BADGE_MAX_TITLES_INLINE)
st.session_state.setdefault("sources_badge_max_pages_per_title", DEFAULT_SOURCES_BADGE_MAX_PAGES_PER_TITLE)

st.session_state.setdefault("current_difficulty", "medium")

# Skills / Debug toggles（popover 內用）
st.session_state.setdefault("da_enable_skills", True)
st.session_state.setdefault("da_enable_memory", True)
st.session_state.setdefault("da_skill_claims_first", True)
st.session_state.setdefault("da_skill_decision_memo", True)
st.session_state.setdefault("da_skill_report_compare", True)
st.session_state.setdefault("da_skill_action_plan", True)

st.session_state.setdefault("da_show_status_debug", True)
st.session_state.setdefault("da_show_status_files", True)
st.session_state.setdefault("da_show_status_doc_hits", True)
st.session_state.setdefault("da_show_claims", True)
st.session_state.setdefault("da_show_reflections", True)
st.session_state.setdefault("da_status_expanded", False)
# ========= (1) Session init：新增幾個 session_state（放在你現有的 st.session_state.setdefault(...) 那一大段附近） =========
st.session_state.setdefault("last_report_title", None)

# retriever hits 顯示控制
st.session_state.setdefault("show_retriever_hits_expander", True)
st.session_state.setdefault("retriever_hits_expanded_by_default", False)
st.session_state.setdefault("retriever_hits_max_per_query", 6)

# ========= [新增/替換 2] Session init：放在你那串 st.session_state.setdefault(...) 附近 =========
st.session_state.setdefault("selected_report_title", "All")  # UI 下拉：All / 某份 title（持久化）
st.session_state.setdefault("title_to_max_page", {})          # title -> max page（PDF 用；Office 多半 1 或 None）
st.session_state.setdefault("current_question_kind", QUESTION_KIND_CHAT)

# =========================
# Popover：文件管理 / Skills / Debug（依你要求重新排版）
# =========================
with st.popover("📦 文件管理 / Skills / Debug"):
    st.caption("Session-only：檔案與索引只存在於本次 session。Office 解析需安裝 unstructured loaders。")

    has_index = (
        st.session_state.store is not None
        and getattr(st.session_state.store, "index", None) is not None
        and st.session_state.store.index.ntotal > 0
    )
    if has_index:
        st.success(f"已建立索引：檔案數={len(st.session_state.file_rows)} / chunks={len(st.session_state.store.chunks)}")
    else:
        st.info("目前沒有索引：你仍可直接聊天；需要引用文件再建立索引。")

    # 1) Web Search
    st.markdown("### 🌐 Web Search")
    st.session_state.enable_web_search_agent = st.checkbox(
        "允許使用網路搜尋（文件不足才會用）",
        value=bool(st.session_state.enable_web_search_agent),
    )

    # 2) 上傳文件（移到 Web Search 下面）
    st.markdown("---")
    st.markdown("### 📤 上傳文件（按一次建立索引）")
    uploaded = st.file_uploader(
        "上傳文件",
        type=["pdf", "docx", "doc", "pptx", "xlsx", "xls", "txt", "png", "jpg", "jpeg"],
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

            if ext in (".png", ".jpg", ".jpeg"):
                use_ocr = True
            elif ext == ".txt":
                use_ocr = False
            else:
                # 只對 PDF 建議 OCR；Office 不做 OCR
                use_ocr = bool(likely_scanned) if ext == ".pdf" else False

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
            st.session_state["last_report_title"] = os.path.splitext(f.name)[0]

    # 表格顯示（pandas）
    df_files = build_files_df(st.session_state.file_rows)
    if not df_files.empty:
        st.markdown("#### 文件清單")
        st.dataframe(
            df_files.drop(columns=["file_id"]),
            width="stretch",
            hide_index=True,
        )

        # PDF OCR 勾選（獨立區）
        st.markdown("#### PDF OCR（建議掃描件才開）")
        for r in st.session_state.file_rows:
            if r.ext != ".pdf":
                continue
            cols = st.columns([3, 1, 1])
            cols[0].write(truncate_filename(r.name, 64))
            cols[1].write("建議OCR" if r.likely_scanned else "")
            r.use_ocr = cols[2].checkbox("OCR", value=bool(r.use_ocr), key=f"ocr_{r.file_id}")

    if (not HAS_UNSTRUCTURED_LOADERS) and any(r.ext in (".doc", ".docx", ".pptx", ".xls", ".xlsx") for r in st.session_state.file_rows):
        st.warning("你上傳了 Office 檔，但環境缺少 unstructured loaders，可能索引不到文字。建議安裝或先轉成 PDF/TXT 再上傳。")

    # —— 新增：文件範圍（持久化直到切回 All）——
    st.markdown("---")
    st.markdown("### 🎯 文件範圍（回答時要用哪些文件）")

    titles_for_scope = sorted([os.path.splitext(r.name)[0] for r in st.session_state.file_rows if (r.name or "").strip()])
    titles_for_scope = _dedup_keep_order([t for t in titles_for_scope if t.strip()])
    scope_options = ["All"] + titles_for_scope

    # 防呆：如果目前選到的 title 不存在（例如清空/重建），就回到 All
    cur_sel = str(st.session_state.get("selected_report_title", "All") or "All")
    if cur_sel not in scope_options:
        cur_sel = "All"
        st.session_state["selected_report_title"] = "All"

    st.session_state.selected_report_title = st.selectbox(
        "選擇文件範圍（會持續套用到之後的提問，直到你切回 All）",
        options=scope_options,
        index=scope_options.index(cur_sel),
    )
    
    col1, col2 = st.columns([1, 1])
    build_btn = col1.button("🚀 建立索引", type="primary", use_container_width=True)
    clear_btn = col2.button("🧹 清空全部（含聊天）", use_container_width=True)

    if clear_btn:
        st.session_state.file_rows = []
        st.session_state.file_bytes = {}
        st.session_state.store = None
        st.session_state.processed_keys = set()
        st.session_state.chat_history = []
        st.session_state.deep_agent = None
        st.session_state.deep_agent_cfg_sig = None
        st.session_state.deep_agent_web_flag = None
        st.session_state.da_usage = {"doc_search_calls": 0, "web_search_calls": 0}
        st.session_state.ui_doc_search_log = []
        st.session_state["last_run_forced_end"] = None
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
            s.write(f"新增報告數：{stats.get('new_reports')}")
            s.write(f"新增 chunks：{stats.get('new_chunks')}")
            if stats.get("errors"):
                s.warning("部分檔案抽取失敗：\n" + "\n".join([f"- {e}" for e in stats["errors"][:8]]))
            s.write(f"耗時：{time.perf_counter() - t0:.2f}s")
            s.update(state="complete")

        st.session_state.deep_agent = None
        st.session_state.deep_agent_cfg_sig = None
        st.session_state.deep_agent_web_flag = None
        st.rerun()

    # 3) Skills + Debug（收在一個 expander）
    st.markdown("---")
    with st.expander("🧩 Skills / Debug（展開設定）", expanded=False):
        st.markdown("#### 🧩 Skills / Memory（DeepAgents）")
        st.session_state.da_enable_memory = st.checkbox("啟用 Memory（/memory/AGENTS.md）", value=bool(st.session_state.da_enable_memory))
        st.session_state.da_enable_skills = st.checkbox("啟用 Skills（/skills/*/SKILL.md）", value=bool(st.session_state.da_enable_skills))

        if st.session_state.da_enable_skills:
            colA, colB = st.columns(2)
            with colA:
                st.session_state.da_skill_claims_first = st.checkbox("Skill: claims-first（強推理）", value=bool(st.session_state.da_skill_claims_first))
                st.session_state.da_skill_decision_memo = st.checkbox("Skill: decision-memo（含反思）", value=bool(st.session_state.da_skill_decision_memo))
            with colB:
                st.session_state.da_skill_report_compare = st.checkbox("Skill: report-compare（跨報告比對）", value=bool(st.session_state.da_skill_report_compare))
                st.session_state.da_skill_action_plan = st.checkbox("Skill: action-plan（推進計畫）", value=bool(st.session_state.da_skill_action_plan))

        st.markdown("#### 🧪 Debug / Status 顯示（st.status）")
        st.session_state.da_status_expanded = st.checkbox("st.status 預設展開", value=bool(st.session_state.da_status_expanded))
        st.session_state.da_show_status_debug = st.checkbox("顯示 Agent Memo（todos/facets/claims/反思）", value=bool(st.session_state.da_show_status_debug))
        st.session_state.da_show_status_doc_hits = st.checkbox("顯示最近 doc_search 命中段落", value=bool(st.session_state.da_show_status_doc_hits))
        st.session_state.da_show_status_files = st.checkbox("顯示 evidence/draft/review 節錄", value=bool(st.session_state.da_show_status_files))
        st.session_state.da_show_claims = st.checkbox("顯示 claims.json", value=bool(st.session_state.da_show_claims))
        st.session_state.da_show_reflections = st.checkbox("顯示 reflections.json", value=bool(st.session_state.da_show_reflections))


# =========================
# History render
# =========================
for msg in st.session_state.chat_history:
    role = msg.get("role", "assistant")
    with st.chat_message(role):
        if role == "user":
            st.markdown(msg.get("content", ""))
            continue
        meta = msg.get("meta", {}) or {}
        render_run_badges(
            mode=meta.get("mode", "unknown"),
            enable_web=bool(meta.get("enable_web", False)),
            usage=meta.get("usage", {}) or {},
            difficulty=str(meta.get("difficulty", "medium") or "medium"),
        )
        render_markdown_answer_with_sources_badges(msg.get("content", ""))
        render_web_sources_list(meta.get("web_sources", {}) or {})


# =========================
# Chat main
# =========================
prompt = st.chat_input("請輸入問題（也可貼草稿要我查核/除錯）。")
if prompt:
    st.session_state.chat_history.append({"role": "user", "kind": "text", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        has_index = (
            st.session_state.store is not None
            and getattr(st.session_state.store, "index", None) is not None
            and st.session_state.store.index.ntotal > 0
        )

        allow_web = bool(st.session_state.enable_web_search_agent)

        run_messages = build_run_messages(prompt, max_messages=15)

        # ========= [B] 修改：Chat main 中「取得 plan 後」到「分支處理」這段（整段替換） =========
        plan = decide_route_plan(
            client,
            prompt,
            has_index=has_index,
            allow_web=allow_web,
            run_messages=run_messages,
        )

# ========= [替換 8] Chat main（prompt if prompt: 區塊中，從「plan = decide_route_plan(...)」開始到各分支處理）
# 這段很長，你可以直接用下面這段「完整替換」原本那一大段路由/分支（保留上面的 run_messages/build messages 等前置即可）。 =========

        store = st.session_state.get("store", None)

        # 1) scope：同步「文字指定檔名」與 UI 下拉（你要求要同步更新）
        scope_title = sync_scope_from_prompt_and_ui(prompt, store)  # None 表示 All

        # 2) 判斷題型（Q2=C）
        q_kind = classify_question_kind(prompt)
        st.session_state["current_question_kind"] = q_kind

        # 3) 判斷 doc_intent（敏感版：像在問內容也要開 DeepAgent）
        doc_intent = decide_doc_intent(
            client,
            prompt,
            has_index=has_index,
            scope_title=scope_title,
            run_messages=run_messages,
        )

        # 4) difficulty：memo 題偏 hard，其餘 medium
        difficulty = "hard" if q_kind == QUESTION_KIND_MEMO else "medium"
        st.session_state["current_difficulty"] = difficulty

        allow_web = bool(st.session_state.enable_web_search_agent)

        # 5) 分支：DeepAgent（主線） vs Direct chat
        if has_index and doc_intent:
            enable_web = bool(allow_web)  # ✅ web gate 在 DeepAgent 內部用 grade_doc_evidence < 0.55 控制
            agent = ensure_deep_agent(client, store, enable_web=enable_web)

            with st.status("DeepAgent：執行中…", expanded=bool(st.session_state.get("da_status_expanded", False))) as main_status:
                answer_text, _files = deep_agent_run_with_live_status(
                    agent,
                    user_text=prompt,
                    run_messages=run_messages,
                    client=client,
                    status=main_status,
                )

            usage = st.session_state.get("da_usage", {"doc_search_calls": 0, "web_search_calls": 0}) or {}
            used_web = bool(enable_web) and int(usage.get("web_search_calls", 0) or 0) > 0

            # ✅ 末尾 badges（你要求放回應結束後）
            scope_label = scope_title if scope_title else "All"
            tail_badges = " ".join([
                _badge_directive("Mode:DeepAgent", "gray"),
                _badge_directive(f"Scope:{scope_label}", "green" if scope_label != "All" else "gray"),
                _badge_directive(f"Docs:{int(usage.get('doc_search_calls', 0) or 0)}", "green"),
                _badge_directive(f"Web:{int(usage.get('web_search_calls', 0) or 0)}" if enable_web else "Web:off", "violet" if used_web else "gray"),
            ])

            # scope 鎖定但文件不足：提醒（你要「要，或提醒改以Web」）
            # 這句放在 badges 前面，讓使用者看得懂再看 badge
            reminder = ""
            if scope_title:
                # 如果完全沒有引用（很可能證據不足/沒命中），提醒切回 All 或用 Web
                if not has_visible_citations(answer_text):
                    if allow_web:
                        reminder = f"\n\n:small[提示：目前範圍鎖定在「{scope_title}」。若找不到答案，可切回 All 再問，或允許我改用 Web 補足。]\n"
                    else:
                        reminder = f"\n\n:small[提示：目前範圍鎖定在「{scope_title}」。若找不到答案，可切回 All 再問。]\n"

            answer_text = (answer_text or "").strip()
            answer_text = strip_internal_process_lines(answer_text)
            answer_text = (answer_text + reminder + "\n\n" + tail_badges).strip()

            if st.session_state.get("enable_output_formatter", True):
                answer_text = format_markdown_output_preserve_citations(client, answer_text)
            answer_text = strip_internal_process_lines(answer_text)

            meta = {
                "mode": "deepagent",
                "enable_web": bool(enable_web),
                "usage": usage,
                "difficulty": difficulty,
                "web_sources": {},   # deepagent 的 web sources 目前沒集中回傳（先留空）
                "scope_title": scope_label,
            }

            # 既有 UI：仍會在 history render 顯示上方 badges（OK），但你要的尾端 badges 也已經加了
            render_run_badges(mode=meta["mode"], enable_web=meta["enable_web"], usage=meta["usage"], difficulty=meta["difficulty"])
            render_markdown_answer_with_sources_badges(answer_text)

            st.session_state.chat_history.append({"role": "assistant", "kind": "text", "content": answer_text, "meta": meta})
            st.stop()

        # 否則：一般聊天（direct）
        web_sources: Dict[str, List[Tuple[str, str]]] = {}
        usage = {"doc_search_calls": 0, "web_search_calls": 0}

        history_msgs = run_messages[:-1]
        history_block = "\n".join(
            [f"{m['role'].upper()}: {m['content']}" for m in history_msgs if m.get("role") in ("user", "assistant") and (m.get("content") or "").strip()]
        ).strip()
        user_text = prompt if not history_block else f"對話脈絡（最近）：\n{history_block}\n\n目前問題：\n{prompt}"

        ans, _ = call_gpt(
            client,
            model=MODEL_MAIN,
            system=ANYA_SYSTEM_PROMPT,
            user=user_text,
            reasoning_effort=REASONING_EFFORT,
        )
        answer_text = (ans or "").strip()

        # 尾端 badges（聊天模式也顯示 scope，方便你知道現在 UI 鎖定狀態）
        scope_label = (scope_title if scope_title else "All")
        tail_badges = " ".join([
            _badge_directive("Mode:Chat", "gray"),
            _badge_directive(f"Scope:{scope_label}", "green" if scope_label != "All" else "gray"),
            _badge_directive("Web:off", "gray"),
        ])
        answer_text = (answer_text + "\n\n" + tail_badges).strip()

        if st.session_state.get("enable_output_formatter", True):
            answer_text = format_markdown_output_preserve_citations(client, answer_text)
        answer_text = strip_internal_process_lines(answer_text)

        meta = {
            "mode": "direct",
            "enable_web": False,
            "usage": usage,
            "difficulty": "easy",
            "web_sources": web_sources,
            "scope_title": scope_label,
        }
        render_run_badges(mode=meta["mode"], enable_web=meta["enable_web"], usage=meta["usage"], difficulty=meta["difficulty"])
        render_markdown_answer_with_sources_badges(answer_text)

        st.session_state.chat_history.append({"role": "assistant", "kind": "text", "content": answer_text, "meta": meta})
