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
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import streamlit as st
import numpy as np
import pandas as pd
import faiss
from pypdf import PdfReader

from openai import OpenAI
from langgraph.errors import GraphRecursionError

try:
    import fitz  # pymupdf
    HAS_PYMUPDF = True
except Exception:
    HAS_PYMUPDF = False


# =========================
# Streamlit config
# =========================
st.set_page_config(page_title="研究報告助手（DeepAgent + Badges）", layout="wide")
st.title("研究報告助手（DeepAgent + Badges）")
# ✅ 依你要求：不注入任何 CSS（回到 Streamlit 預設排版）


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
    st.error("DeepAgent 依賴載入失敗（不一定是沒安裝，可能是版本/依賴不相容）。")
    if DEEPAGENTS_IMPORT_ERRORS:
        st.markdown("### 依賴錯誤細節（請把這段貼給我，我就能精準指你該裝哪個版本）")
        for msg in DEEPAGENTS_IMPORT_ERRORS:
            st.code(msg)
    else:
        st.info("（沒有捕捉到錯誤細節）")
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
# 模型設定（依你要求：gpt-5.2）
# =========================
EMBEDDING_MODEL = "text-embedding-3-small"

MODEL_MAIN = "gpt-5.2"
MODEL_GRADER = "gpt-5.2"
MODEL_WEB = "gpt-5.2"

REASONING_EFFORT = "medium"

ANYA_SYSTEM_PROMPT = """
Developer: 
# Agentic Reminders
- Persistence：確保回應完整，直到用戶問題解決才結束，避免只分析不給具體結論或建議。
- Tool-calling：必要時使用可用工具，不要依空腦測；在決定是否使用工具前，先簡短思考判斷。
- Failure-mode mitigations：
  • 若無足夠資訊使用工具，請先向用戶詢問關鍵補充資訊（最多 1–3 個問題）。
  • 變換範例用語，避免在不同回合重複相同句型或模板。

# Role & Objective
你是安妮亞（Anya Forger），來自《SPY×FAMILY 間諜家家酒》的小女孩。你天真可愛、開朗樂觀，說話直接帶點呆萌，喜歡用可愛語氣和表情回應。你很愛家人和朋友，渴望被愛，也很喜歡花生。

- 在一般、輕鬆主題時，可以自然展現安妮亞的可愛語氣與 emoji。
- 遇到法律、醫療、財經、學術等重要嚴肅主題時，**優先確保內容準確與清楚**：
  - 語氣仍然可以溫和、友善，但明顯降低「呆萌」與玩笑成分。
  - 避免使用彩色徽章、繽紛模式與過多 emoji，以專業、可讀性為主。

Begin with a concise checklist（3–7 bullets）of what you will do; keep items conceptual, not implementation-level。
- 若用戶問題非常簡單（例如只問一個定義或單一事實），可以將 checklist 縮短為 2–3 點，或在明顯不需要時省略。

# Instructions
**若用戶要求翻譯，或明確表示需要將內容轉換語言（不論是否精確使用「翻譯」、「請翻譯」、「幫我翻譯」等字眼，只要語意明確表示需要翻譯），請暫時不用安妮亞的語氣，直接正式逐句翻譯。**
- 若用戶同時要求「翻譯＋說明／評論」，請分兩個明確區塊：
  1) 先以正式語氣完成完整逐句翻譯（不加可愛語氣、不使用條列式）。
  2) 再以安妮亞的語氣，額外用條列式或摘要方式說明或評論。
After each tool call or code edit, validate result in 1-2 lines and proceed or self-correct if validation fails。

# 回答語言與風格
- 務必以正體中文回應，並遵循台灣用語習慣。
- 回答時要友善、熱情、謙虛，並適時加入 emoji。
- 回答要有安妮亞的語氣回應，簡單、直接、可愛，偶爾加入「哇～」「安妮亞覺得…」「這個好厲害！」等語句。
- 若回答不完全正確，請主動道歉並表達會再努力。

### 工具使用決策原則
- 下列情況「優先使用 web_search」：
  - 用戶明確詢問「最新、現在、今年、目前」等時間敏感資訊。
  - 涉及法律、醫療、財經、政府政策等高風險領域，且需要具體數據或規範。
  - 問題牽涉到特定網站、文件、或外部服務狀態。
- 下列情況「優先不使用工具」，直接依內部知識回答：
  - 純概念解釋、基礎知識、學習方法、生活建議、創作發想。
  - 用戶明確要求「不要上網查」或只想要腦力激盪。
- 若不確定是否需要工具，可先用 1–2 句說明你的判斷，再決定是否呼叫 web_search。

Before any significant tool call, state in one line: purpose + minimal inputs。

---
## 搜尋工具使用進階指引
<web_search_rules>
# 角色定位
- 你是可靠的網路研究助理：以正確、可追溯、可驗證為最高優先。
- 只要外部事實可能不確定/過時/版本差異/需要來源佐證，就優先使用「可用的網路搜尋工具」，不要靠印象補。

# 研究門檻（Research bar）與停止條件：做到邊際收益下降才停
- 先在心中拆成子問題，確保每個子問題都有依據。
- 核心結論：
  - 盡量用 ≥2 個獨立可靠來源交叉驗證。
  - 若只能找到單一來源：要明講「證據薄弱/尚待更多來源」。
- 遇到矛盾：至少再找 1–2 個高品質來源來釐清（版本/日期/定義/地域差異）。
- 停止條件：再搜尋已不太可能改變主要結論、或只能增加低價值重複資訊。

# 查詢策略（怎麼搜）
- 多 query：至少 2–4 組不同關鍵字（同義詞/正式名稱/縮寫/可能拼字變體）。
- 多語言：以中文 + 英文為主；必要時加原文語言（例如日文官方資訊）。
- 二階線索：看到高品質文章引用官方文件/公告/論文/規格時，優先追到一手來源。

# 來源品質（Source quality）
- 優先順序（一般情況）：
  1) 一手官方來源（政府/標準機構/公司公告/產品文件/原始論文）
  2) 權威媒體/大型機構整理（可回溯一手來源者更佳）
  3) 專家文章（需看作者可信度與引用）
  4) 論壇/社群（只當線索或經驗談，不可作為唯一依據）
- 若只能找到低品質來源：要明講可信度限制，避免用肯定語氣下定論。

# 時效性（Recency）
- 對可能變動的資訊（價格、版本、政策、法規、時間表、人事等）：
  - 必須標註來源日期或「截至何時」。
  - 優先採用最新且官方的資訊；若資訊可能過期要提醒。

# 矛盾處理（Non-negotiable）
- 不要把矛盾硬融合成一句話。
- 要列出差異點、各自依據、可能原因（版本/日期/定義/地區），並說明你採用哪個結論與理由。

# 不問釐清問題（Prompting guild 建議）
- 進入 web research 模式時：不要問使用者釐清問題。
- 改為涵蓋 2–3 個最可能的使用者意圖並分段標註：
  - 「若你想問 A：...」
  - 「若你想問 B：...」
  - 其餘較不可能延伸放「可選延伸」一小段，避免失焦。

# 引用規則（Citations）
- 凡是網路得來的事實/數字/政策/版本/聲明：都要附引用。
- 引用放在該段落末尾；核心結論盡量用 2 個來源。
- 不得捏造引用；找不到就說找不到。

# 輸出形狀（Output shape & tone）
- 預設用 Markdown：
  - 先給 3–6 點重點結論
  - 再給「證據/來源整理」與必要背景
  - 需要比較就用表格
- 首次出現縮寫要展開；能給具體例子就給 1 個。
- 口吻：自然、好懂、像安妮亞陪你一起查資料，但內容要專業可靠、不要油滑或諂媚。
</web_search_rules>

# 格式化規則
- 根據內容選擇最合適的 Markdown 格式及彩色徽章（colored badges）元素表達。

# Markdown 格式與 emoji/顏色用法說明
## 基本原則
- 根據內容選擇最合適的強調方式，讓回應清楚、易讀、有層次，避免過度使用彩色文字。
- 只用 Streamlit 支援的 Markdown 語法，不要用 HTML 標籤。

## 功能與語法
- **粗體**：`**重點**` → **重點**
- *斜體*：`*斜體*` → *斜體*
- 標題：`# 大標題`、`## 小標題`
- 分隔線：`---`
- 表格（僅部分平台支援，建議用條列式）
- 引用：`> 這是重點摘要`
- emoji：直接輸入或貼上，如 😄
- Material Symbols：如`:material/star:`
- LaTeX 數學公式：`$公式$` 或 `$$公式$$`
- 彩色文字：`:orange[重點]`、`:blue[說明]`
- 彩色背景：`:orange-background[警告內容]`
- 彩色徽章：`:orange-badge[重點]`、`:blue-badge[資訊]`
- 小字：`:small[這是輔助說明]`
- 彩色文字與彩色徽章使用原則：
  - 一則回應中，建議彩色徽章區塊不超過 2–3 個。
  - 嚴肅主題時，避免使用彩色文字與徽章，只使用基本粗體、標題與條列式。
  - 以提升可讀性為主，若文字已足夠清楚，不必強行加顏色。

## 顏色名稱及建議用途（條列式，跨平台穩定）
- **blue**：資訊、一般重點
- **green**：成功、正向、通過
- **orange**：警告、重點、溫暖
- **red**：錯誤、警告、危險
- **violet**：創意、次要重點
- **gray/grey**：輔助說明、備註
- **rainbow**：彩色強調、活潑
- **primary**：依主題色自動變化

**注意：**
- 只能使用上述顏色。**請勿使用 yellow（黃色）**，如需黃色效果，請改用 orange 或黃色 emoji（🟡、✨、🌟）強調。
- 不支援 HTML 標籤，請勿使用 `<span>`、`<div>` 等語法。
- 建議只用標準 Markdown 語法，保證跨平台顯示正常。

# 回答步驟
1. **若用戶的問題包含「翻譯」、「請翻譯」或「幫我翻譯」等字眼，請直接完整逐句翻譯內容為正體中文，不要摘要、不用可愛語氣、不用條列式，直接正式翻譯。**
2. 若非翻譯需求，先用安妮亞的語氣簡單回應或打招呼。
3. 若非翻譯需求，條列式摘要或回答重點，語氣可愛、簡單明瞭；對於非常簡單的問題，整體回答以 3–6 句內為原則，避免不必要的冗長。
4. 根據內容自動選擇最合適的Markdown格式，並靈活組合。
5. 若有數學公式，正確使用 $$Latex$$ 格式。
6. 若有使用 web_search，在答案最後用 `## 來源` 列出所有參考網址。
7. 適時穿插 emoji。
8. 結尾可用「安妮亞回答完畢！」、「還有什麼想問安妮亞嗎？」等可愛語句。
9. 請先思考再作答，確保每一題都用最合適的格式呈現。

# 《SPY×FAMILY 間諜家家酒》彩蛋模式
- 若不是在討論法律、醫療、財經、學術等重要嚴肅主題，安妮亞可在回答中穿插趣味元素，但不要影響正確性與可讀性。

請先思考再作答，確保每一題都用最合適的格式呈現。
"""

# =========================
# 效能參數（固定預設；不提供 UI 調整）
# =========================
EMBED_BATCH_SIZE = 256
OCR_MAX_WORKERS = 2

CORPUS_DEFAULT_MAX_CHUNKS = 24
CORPUS_PER_REPORT_QUOTA = 6

DA_MAX_DOC_SEARCH_CALLS = 14
DA_MAX_WEB_SEARCH_CALLS = 4
DA_MAX_REWRITE_ROUNDS = 2
DA_MAX_CLAIMS = 10

# 固定預設（不提供 UI）
DEFAULT_RECURSION_LIMIT = 200
DEFAULT_CITATION_STALL_STEPS = 12
DEFAULT_CITATION_STALL_MIN_CHARS = 450

DEFAULT_SOURCES_BADGE_MAX_TITLES_INLINE = 4
DEFAULT_SOURCES_BADGE_MAX_PAGES_PER_TITLE = 10


# =========================
# Regex / citations
# =========================
CHUNK_ID_LEAK_PAT = re.compile(r"(chunk_id\s*=\s*|_p(?:na|\d+)_c\d+)", re.IGNORECASE)

# 重要：內部 evidence 不該出現在有效引用
EVIDENCE_PATH_IN_CIT_RE = re.compile(r"\[(?:/)?evidence/[^ \]]+?\s+p(\d+|-)\s*\]", re.IGNORECASE)

CIT_RE = re.compile(r"\[[^\]]+?\s+p(\d+|-)\s*\]")
BULLET_RE = re.compile(r"^\s*(?:[-•*]|\d+\.)\s+")
CIT_PARSE_RE = re.compile(r"\[([^\]]+?)\s+p(\d+|-)\s*\]")

# 移除內部流程/檔名洩漏（只留「查得到的」內容）
INTERNAL_LEAK_PAT = re.compile(
    r"(Budget exceeded|/evidence|doc_[\w\-]+\.md|web_[\w\-]+\.md|額度不足|占位|向量庫|內部文件|工作流|流程|工具預算)",
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


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
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


def has_visible_citations(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    cits = [m.group(0) for m in re.finditer(r"\[[^\]]+?\s+p(\d+|-)\s*\]", raw)]
    cits = [c for c in cits if not EVIDENCE_PATH_IN_CIT_RE.search(c)]
    return bool(cits)


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

def get_recent_chat_messages(max_messages: int = 15) -> list[dict]:
    """
    取最近 N 則「text」訊息當短期記憶（排除 default 大包輸出），避免 prompt 爆長。
    回傳格式：[{role:"user"/"assistant", content:"..."}]
    """
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
        # 避免太長（可視需要調）
        if len(content) > 2000:
            content = content[:2000] + "…"
        msgs.append({"role": role, "content": content})

    return msgs[-max_messages:]


def _domain(u: str) -> str:
    try:
        host = urlparse(u).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host or "web"
    except Exception:
        return "web"


def web_sources_from_openai_sources(sources: Optional[list[dict]]) -> Dict[str, List[Tuple[str, str]]]:
    """
    將 OpenAI web_search sources 轉成：
    {domain: [(title, url), ...]}
    """
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

    # 去重
    for dom in list(out.keys()):
        seen = set()
        uniq: List[Tuple[str, str]] = []
        for t, u in out[dom]:
            key = (t, u)
            if key in seen:
                continue
            seen.add(key)
            uniq.append((t, u))
        out[dom] = uniq
    return out


def render_web_sources_list(
    web_sources: Dict[str, List[Tuple[str, str]]],
    max_domains: int = 6,
    max_per_domain: int = 6,
) -> None:
    """
    B 方案：
    - badge：只顯示 domain（你已經做到了）
    - Web Sources：列「可點連結」，顯示 domain + path 為主
    """
    if not web_sources:
        return

    st.markdown("#### Web Sources")
    domains = sorted(web_sources.keys())
    show = domains[:max_domains]
    more = domains[max_domains:]

    def _render(domains_list: list[str]):
        for dom in domains_list:
            items = web_sources.get(dom, [])
            if not items:
                continue

            st.markdown(f"- **{dom}**")
            for _title, url in items[:max_per_domain]:
                label = url_to_domain_path(url)
                # ✅ 可點連結
                st.markdown(f"  - [{label}]({url})")

    _render(show)
    if more:
        with st.expander(f"更多 Web Sources（{len(more)}）", expanded=False):
            _render(more)

def ensure_web_citation_token(text: str, domain: str) -> str:
    """
    保證回答中至少有一個 [WebSearch:<domain> p-]，
    讓你的 UI 能顯示 web badge。
    """
    if not text:
        return text
    if re.search(r"\[WebSearch:[^\]]+\s+p-\s*\]", text, re.IGNORECASE):
        return text
    dom = (domain or "web").strip() or "web"
    return (text.rstrip() + f"\n\n[WebSearch:{dom} p-]").strip()

def _domain_from_url(u: str) -> str:
    try:
        host = urlparse(u).netloc or ""
        host = host.lower()
        if host.startswith("www."):
            host = host[4:]
        return host or "web"
    except Exception:
        return "web"


def strip_internal_process_lines(md: str) -> str:
    """
    你指定「只寫有查到的」，所以：
    - 任何主要在講額度不足/內部檔名/流程的行，直接移除
    """
    lines = (md or "").splitlines()
    kept = []
    for line in lines:
        if INTERNAL_LEAK_PAT.search(line):
            continue
        kept.append(line)
    return "\n".join(kept).strip()

def url_to_domain_path(url: str, max_len: int = 72) -> str:
    """
    把 URL 轉成適合顯示的 label：<domain><path>
    - 不顯示 query/fragment（避免太長、也避免 utm 之類雜訊）
    - 過長就截斷
    """
    try:
        u = urlparse(url)
        host = (u.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        path = u.path or "/"
        label = f"{host}{path}"
        if len(label) > max_len:
            label = label[: max_len - 1] + "…"
        return label or url
    except Exception:
        return url

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


def call_gpt(
    client: OpenAI,
    *,
    model: str,
    system: str,
    user: Any,
    reasoning_effort: Optional[str] = None,
    tools: Optional[list] = None,
    include_sources: bool = False,
    tool_choice: Optional[Any] = None,  # ✅ 新增
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

        # (A) 嘗試從 web_search_call.action.sources 抓
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

        # (B) ✅ 再從 message.content[].annotations(url_citation) 抓（很多模型會走這裡）
        try:
            for item in (getattr(resp, "output", None) or []):
                d = _as_dict(item)
                typ = d.get("type") or getattr(item, "type", None)
                if typ != "message":
                    continue

                content = d.get("content") or getattr(item, "content", None) or []
                for part in (content or []):
                    pd = _as_dict(part)
                    annotations = pd.get("annotations") or getattr(part, "annotations", None) or []
                    for ann in (annotations or []):
                        ad = _as_dict(ann)
                        at = ad.get("type") or getattr(ann, "type", None)
                        if at != "url_citation":
                            continue
                        url = (ad.get("url") or "").strip()
                        title = (ad.get("title") or ad.get("source") or ad.get("name") or "source").strip()
                        if url:
                            sources_list.append({"title": title, "url": url})
        except Exception:
            pass

        # 去重（以 url 為準）
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
# OCR / PDF
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
    system = "你是一個OCR工具。只輸出可見文字與表格內容（若有表格用 Markdown 表格）。中文請用繁體中文。不要加評論。"
    user_content = [
        {"type": "input_text", "text": "請擷取圖片中所有可見文字（包含小字/註腳）。若無法辨識請標記[無法辨識]。"},
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
        self.chunks: list[Chunk] = []

    def add(self, vecs: np.ndarray, chunks: list[Chunk]) -> None:
        self.index.add(vecs)
        self.chunks.extend(chunks)

    def search(self, qvec: np.ndarray, k: int = 8) -> list[Tuple[float, Chunk]]:
        if self.index.ntotal == 0:
            return []
        scores, idx = self.index.search(qvec.astype(np.float32), k)
        out: list[Tuple[float, Chunk]] = []
        for s, i in zip(scores[0], idx[0]):
            if i < 0 or i >= len(self.chunks):
                continue
            out.append((float(s), self.chunks[i]))
        return out


# =========================
# File rows
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

    stats = {"new_reports": 0, "new_chunks": 0}
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
        stats["new_reports"] += 1

        if row.ext == ".pdf":
            pages = ocr_pdf_pages_parallel(client, data) if row.use_ocr else extract_pdf_text_pages(data)
        elif row.ext == ".txt":
            pages = [(None, norm_space(data.decode("utf-8", errors="ignore")))]
        elif row.ext in (".png", ".jpg", ".jpeg"):
            mime = "image/jpeg" if row.ext in (".jpg", ".jpeg") else "image/png"
            txt = norm_space(ocr_image_bytes(client, data, mime=mime))
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
        vecs_list: list[np.ndarray] = []
        for i in range(0, len(new_texts), EMBED_BATCH_SIZE):
            vecs_list.append(embed_texts(client, new_texts[i:i + EMBED_BATCH_SIZE]))
        vecs = np.vstack(vecs_list)
        store.add(vecs, new_chunks)

    stats["new_chunks"] = len(new_chunks)
    return store, stats, processed_keys


# =========================
# citations / rendering
# =========================
def file_to_text(file_obj: Any) -> str:
    if file_obj is None:
        return ""

    if isinstance(file_obj, dict):
        if "data" in file_obj:
            return file_to_text(file_obj.get("data"))
        if "content" in file_obj:
            return file_to_text(file_obj.get("content"))
        for k in ("text", "answer", "final", "output", "message"):
            if k in file_obj:
                return file_to_text(file_obj.get(k))
        try:
            return json.dumps(file_obj, ensure_ascii=False, indent=2)
        except Exception:
            return str(file_obj)

    if isinstance(file_obj, (bytes, bytearray)):
        return file_obj.decode("utf-8", errors="ignore")

    if isinstance(file_obj, str):
        return file_obj

    if isinstance(file_obj, (list, tuple)):
        parts: list[str] = []
        for x in file_obj:
            t = file_to_text(x).strip()
            if t:
                parts.append(t)
        return "\n".join(parts)

    return str(file_obj)


def get_files_text(files: Optional[dict], key: str) -> str:
    if not isinstance(files, dict) or key not in files:
        return ""
    return file_to_text(files.get(key)).strip()


def _badge_directive(label: str, color: str) -> str:
    safe = label.replace("[", "(").replace("]", ")")
    return f":{color}-badge[{safe}]"


def _extract_main_text_from_payload(payload: Any) -> Optional[str]:
    if isinstance(payload, dict):
        for k in ("content", "answer", "final", "output", "text", "message"):
            if k not in payload:
                continue
            v = payload.get(k)
            if isinstance(v, str) and v.strip():
                return v
            if isinstance(v, (list, tuple)):
                joined = file_to_text(v).strip()
                if joined:
                    return joined
        msgs = payload.get("messages")
        if isinstance(msgs, list) and msgs:
            last = msgs[-1]
            if isinstance(last, dict):
                c = last.get("content")
                if isinstance(c, (str, list, tuple, dict)):
                    out = file_to_text(c).strip()
                    if out:
                        return out
            out = file_to_text(last).strip()
            return out or None
        return None

    if isinstance(payload, list):
        out = file_to_text(payload).strip()
        return out or None

    return None


def _strip_citations_from_text(text: str) -> str:
    """
    移除引用 token，但保留換行（避免 Markdown 黏成一坨）
    """
    if not text:
        return ""
    pat = re.compile(r"[ \t]*\[[^\]]*?\s+p(\d+|-)(?:-\d+)?[^\]]*?\][ \t]*")
    out_lines: list[str] = []
    for line in text.splitlines():
        out_lines.append(pat.sub("", line).rstrip())
    return "\n".join(out_lines).strip()


def _extract_citation_items(text: str) -> list[tuple[str, str]]:
    """
    支援：
    - [Title p12]
    - [Title p2-3; Another Title p8]
    - [A p2][B p8]
    """
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


def _title_to_display_domain(title: str) -> str:
    """
    B 方案：badge 只顯示 domain
    - doc: 그대로用 title
    - web: [WebSearch:domain p-] -> 顯示 domain
    """
    t = (title or "").strip()
    low = t.lower()
    if low.startswith("websearch:"):
        dom = t.split(":", 1)[1].strip() if ":" in t else "web"
        return dom or "web"
    return t


def render_markdown_answer_with_sources_badges(answer_text: str) -> None:
    raw = (answer_text or "").strip()

    if raw and CHUNK_ID_LEAK_PAT.search(raw):
        raw = CHUNK_ID_LEAK_PAT.sub("", raw)

    payload = _try_parse_json_or_py_literal(raw)
    if payload is not None:
        extracted = _extract_main_text_from_payload(payload)
        if extracted is not None:
            raw = extracted.strip()

    # ✅ 最終輸出前再保險：去內部流程/檔名
    raw = strip_internal_process_lines(raw)

    cit_items = _extract_citation_items(raw)

    clean = _strip_citations_from_text(raw)
    st.markdown(clean if clean else "（無內容）")

    if not cit_items:
        return

    grouped: dict[str, list[str]] = {}
    for title, page in cit_items:
        grouped.setdefault(title, []).append(page)

    def _key(p: str):
        if p.isdigit():
            return (0, int(p))
        if re.fullmatch(r"\d+-\d+", p):
            a, b = p.split("-", 1)
            return (1, int(a), int(b))
        if p == "-":
            return (9, 10**9)
        return (10, p)

    for t in list(grouped.keys()):
        pages = _dedup_keep_order([p.strip() for p in grouped[t] if p.strip()])
        grouped[t] = sorted(pages, key=_key)

    titles_sorted = sorted(grouped.keys(), key=lambda x: (x.strip().lower().startswith("websearch:"), x.lower()))
    max_inline = int(st.session_state.get("sources_badge_max_titles_inline", DEFAULT_SOURCES_BADGE_MAX_TITLES_INLINE))
    max_pages = int(st.session_state.get("sources_badge_max_pages_per_title", DEFAULT_SOURCES_BADGE_MAX_PAGES_PER_TITLE))

    inline_titles = titles_sorted[:max_inline]
    extra_titles = titles_sorted[max_inline:]

    def _pages_str(pages: list[str]) -> str:
        if not pages:
            return "p-"
        if len(pages) <= max_pages:
            return "p" + ",".join(pages)
        return "p" + ",".join(pages[:max_pages]) + "…"

    def _render_badges(titles: list[str]) -> None:
        doc_badges: list[str] = []
        web_badges: list[str] = []
        for title in titles:
            show_title = _title_to_display_domain(title)
            label = f"{show_title} {_pages_str(grouped.get(title, []))}"
            if title.strip().lower().startswith("websearch:"):
                web_badges.append(_badge_directive(label, "violet"))
            else:
                doc_badges.append(_badge_directive(label, "green"))
        if doc_badges:
            st.markdown(" ".join(doc_badges))
        if web_badges:
            st.markdown(" ".join(web_badges))

    st.markdown("### 來源")
    _render_badges(inline_titles)
    if extra_titles:
        with st.expander(f"更多來源（{len(extra_titles)}）", expanded=False):
            _render_badges(extra_titles)


# =========================
# Formatter（可愛但克制）
# =========================
FORMATTER_SYSTEM_PROMPT = r"""
你是安妮亞（Anya Forger，《SPY×FAMILY》）風格的「可靠小幫手」，但你的本職是：Markdown 輸出排版美化（formatter）。
風格目標：可愛但克制、重點先行、不出錯。

任務（只做版面，不改內容）：
- 調整標題層級（# / ## / ###）、補空行、分段
- 過長段落可改成 bullets（每點一件事）
- 統一章節結構、讓閱讀更清楚

嚴格禁止：
- 新增任何事實、數字、日期、主張、推論、案例
- 改變原文意思
- 捏造/補上不存在的引用

引用 token 硬規則（必須逐字保留）：
- 形如 [報告名稱 p頁]、[WebSearch:... p-]、[A p2][B p8] 的 token 不可改寫、不可刪除、不可合併
- 若你把段落改成 bullet，引用 token 要放回對應 bullet 句尾

另外：
- 不得提及內部流程、檔名、額度不足、Budget exceeded 等字樣。
- 若遇到「缺口/不足」描述，請直接省略該段落（使用者只想看有查到的內容）。

輸出：
- 只輸出排版後的 Markdown，不要解釋、不加前言

# 格式化規則
- 根據內容選擇最合適的 Markdown 格式及彩色徽章（colored badges）元素表達。
- 可愛語氣與彩色元素是輔助閱讀的裝飾，而不是主要結構；**不可取代清楚的標題、條列與段落組織**。

# Markdown 格式與 emoji／顏色用法說明
## 基本原則
- 根據內容選擇最合適的強調方式，讓回應清楚、易讀、有層次，避免過度使用彩色文字與 emoji 造成視覺負擔。
- 只用 Streamlit 支援的 Markdown 語法，不要用 HTML 標籤。

## 功能與語法
- **粗體**：`**重點**` → **重點**
- *斜體*：`*斜體*` → *斜體*
- 標題：`# 大標題`、`## 小標題`
- 分隔線：`---`
- 表格（僅部分平台支援，建議用條列式）
- 引用：`> 這是重點摘要`
- emoji：直接輸入或貼上，如 😄
- Material Symbols：`:material/star:`
- LaTeX 數學公式：`$公式$` 或 `$$公式$$`
- 彩色文字：`:orange[重點]`、`:blue[說明]`
- 彩色背景：`:orange-background[警告內容]`
- 彩色徽章：`:orange-badge[重點]`、`:blue-badge[資訊]`
- 小字：`:small[這是輔助說明]`

## 顏色名稱及建議用途（條列式，跨平台穩定）
- **blue**：資訊、一般重點
- **green**：成功、正向、通過
- **orange**：警告、重點、溫暖
- **red**：錯誤、警告、危險
- **violet**：創意、次要重點
- **gray/grey**：輔助說明、備註
- **rainbow**：彩色強調、活潑
- **primary**：依主題色自動變化

**注意：**
- 只能使用上述顏色。**請勿使用 yellow（黃色）**，如需黃色效果，請改用 orange 或黃色 emoji（🟡、✨、🌟）強調。
- 不支援 HTML 標籤，請勿使用 `<span>`、`<div>` 等語法。
- 建議只用標準 Markdown 語法，保證跨平台顯示正常。
"""


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


# =========================
# Default outputs (summary/claims/chain)（保留原樣）
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


def pick_corpus_chunks_for_default(all_chunks: list[Chunk]) -> list[Chunk]:
    by_title: Dict[str, list[Chunk]] = {}
    for c in all_chunks:
        by_title.setdefault(c.title, []).append(c)

    kw = re.compile(
        r"(outlook|risk|implication|forecast|scenario|inflation|rate|credit|spread|cap rate|vacancy|supply|demand|rental|office|retail|residential|logistics|hotel|reits)",
        re.I,
    )

    def score(c: Chunk) -> float:
        s = 0.0
        if kw.search(c.text or ""):
            s += 6.0
        if c.page is not None:
            s += max(0.0, 2.0 - min(2.0, float(c.page) / 6.0))
        s += min(2.0, len(c.text) / 1400.0)
        return s

    chosen: list[Chunk] = []
    for title, chunks in sorted(by_title.items(), key=lambda x: x[0]):
        by_page: Dict[int, list[Chunk]] = {}
        for c in chunks:
            p = c.page if c.page is not None else 0
            by_page.setdefault(p, []).append(c)

        page_best: list[Chunk] = []
        for p, cs in by_page.items():
            cs = sorted(cs, key=score, reverse=True)
            page_best.append(cs[0])

        page_best = sorted(page_best, key=score, reverse=True)
        chosen.extend(page_best[:CORPUS_PER_REPORT_QUOTA])

    chosen = sorted(chosen, key=score, reverse=True)[:CORPUS_DEFAULT_MAX_CHUNKS]
    return chosen


def render_chunks_for_model(chunks: list[Chunk], max_chars_each: int = 900) -> str:
    parts = []
    for c in chunks:
        head = f"[{c.title} p{c.page if c.page is not None else '-'}]"
        parts.append(head + "\n" + c.text[:max_chars_each])
    return "\n\n".join(parts)


def bullets_all_have_citations(md: str) -> bool:
    lines = (md or "").splitlines()
    if not any(BULLET_RE.match(l) for l in lines):
        return False
    for line in lines:
        if BULLET_RE.match(line):
            cits = [m.group(0) for m in re.finditer(r"\[[^\]]+?\s+p(\d+|-)\s*\]", line)]
            cits = [c for c in cits if not EVIDENCE_PATH_IN_CIT_RE.search(c)]
            if not cits:
                return False
    return True


def generate_default_outputs_bundle(client: OpenAI, title: str, ctx: str, max_retries: int = 2) -> Dict[str, str]:
    system = (
        "你是嚴謹的研究助理，只能根據我提供的資料回答，不可腦補。\n"
        "硬性規則：\n"
        "1) 你必須輸出三個區塊，且順序/標題固定：### SUMMARY、### CLAIMS、### CHAIN。\n"
        "2) 每個區塊都必須是純 bullet（每行以 - 開頭），不要段落。\n"
        "3) 每個 bullet 句尾必須附引用，格式固定：[報告名稱 p頁]\n"
        "4) 引用中的『報告名稱』必須是資料片段方括號內的那個名稱。\n"
        "5) 不可使用 /evidence/*.md 當作報告名稱。\n"
        "6) 若同一句需要多個引用，請用多個方括號連續附在句尾，例如：[A p2][B p8]，不要在同一對 [] 內用分號塞多筆。\n"
    )
    user = (
        f"請針對《{title}》一次輸出三份內容（融合多份報告）：\n"
        f"- SUMMARY：8~14 bullets\n"
        f"- CLAIMS：8~14 bullets\n"
        f"- CHAIN：6~12 bullets\n\n"
        f"資料：\n{ctx}\n"
    )

    last = ""
    for _ in range(max_retries + 1):
        out, _ = call_gpt(client, model=MODEL_MAIN, system=system, user=user, reasoning_effort=REASONING_EFFORT)
        parts = _split_default_bundle(out)
        ok = bullets_all_have_citations(parts["summary"]) and bullets_all_have_citations(parts["claims"]) and bullets_all_have_citations(parts["chain"])
        if ok:
            return parts
        last = out
        user += "\n\n【強制修正】整份重寫：三區塊皆為純 bullet，且每個 bullet 句尾都有 [報告名稱 p頁]；不得出現 /evidence/*.md。多引用用 [A p2][B p8]。"

    return _split_default_bundle(last)


# =========================
# Direct mode todos.json（你要求 direct 也要產）
# =========================
def build_todos_json_for_question(
    client: OpenAI,
    question: str,
    *,
    enable_web: bool,
    has_index: bool,
    planned_mode: str,
    run_messages: Optional[list[dict]] = None,   # ✅ 新增
    max_history_chars: int = 3200,               # ✅ 避免 todo prompt 過長
) -> str:
    system = (
        "你是任務規劃器。請輸出 todos.json（JSON array of strings）。\n"
        "目的：用來規劃『如何回答使用者問題』。\n"
        "硬規則：\n"
        "1) 只輸出 JSON array（不要 markdown、不要解釋）。\n"
        "2) 5~9 個 steps。\n"
        "3) 第 1 步要說明 enable_web/has_index/planned_mode 與本題目標。\n"
        "4) 需考慮對話脈絡（若提供）。\n"
    )

    # ✅ 把 run_messages（最近對話）壓成「對話脈絡」給 todo 參考
    history_block = ""
    if run_messages:
        # 排除最後一則（通常就是本次 question），避免重複
        hist = run_messages[:-1]
        lines = []
        for m in hist:
            role = (m.get("role") or "").strip()
            content = (m.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                lines.append(f"{role.upper()}: {content}")
        history_block = "\n".join(lines).strip()

        if len(history_block) > max_history_chars:
            history_block = history_block[:max_history_chars] + "…"

    user = (
        f"enable_web={str(enable_web).lower()}\n"
        f"has_index={str(has_index).lower()}\n"
        f"planned_mode={planned_mode}\n\n"
        + (f"對話脈絡（最近）：\n{history_block}\n\n" if history_block else "")
        + f"目前問題：\n{question}"
    )

    out, _ = call_gpt(
        client,
        model=MODEL_MAIN,
        system=system,
        user=user,
        reasoning_effort=REASONING_EFFORT,
        tools=None,
        include_sources=False,
    )

    data = _try_parse_json_or_py_literal(out)
    if isinstance(data, list):
        steps = [str(x) for x in data if str(x).strip()]
        return json.dumps(steps[:12], ensure_ascii=False, indent=2)

    steps = [
        f"說明 enable_web={str(enable_web).lower()}、has_index={str(has_index).lower()}、planned_mode={planned_mode} 與本題目標",
        "回顧對話脈絡：確認使用者目前的上下文、限制與偏好（若有）",
        "釐清問題範圍與關鍵名詞，避免誤解",
        "列出需要引用/查證的資訊點",
        "若啟用網搜：列出要查的子問題與優先來源（官方/主流媒體/一手文件）",
        "整理可用證據後，產出結構化回覆（結論先行 + 分點）",
        "最終檢查：不提內部流程與檔名，只保留有來源支撐的內容",
    ]
    return json.dumps(steps, ensure_ascii=False, indent=2)


# =========================
# DeepAgent
# =========================
def ensure_deep_agent(client: OpenAI, store: FaissStore, enable_web: bool):
    _require_deepagents()
    from langchain_core.tools import BaseTool, StructuredTool

    st.session_state.setdefault("deep_agent", None)
    st.session_state.setdefault("deep_agent_web_flag", None)
    st.session_state.setdefault("da_usage", {"doc_search_calls": 0, "web_search_calls": 0})

    if (st.session_state.deep_agent is not None) and (st.session_state.deep_agent_web_flag == bool(enable_web)):
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
        return "\n".join(lines) if lines else "（目前沒有任何已索引文件）"

    def _doc_search_fn(query: str, k: int = 8) -> str:
        if not _inc("doc_search_calls", DA_MAX_DOC_SEARCH_CALLS):
            return json.dumps({"hits": [], "error": f"Budget exceeded: doc_search_calls > {DA_MAX_DOC_SEARCH_CALLS}"}, ensure_ascii=False)

        q = (query or "").strip()
        if not q:
            return json.dumps({"hits": []}, ensure_ascii=False)

        qvec = embed_texts(client, [q])
        hits = store.search(qvec, k=max(1, min(12, int(k))))

        payload = {"hits": []}
        for score, ch in hits:
            payload["hits"].append({
                "title": ch.title,
                "page": str(ch.page) if ch.page is not None else "-",
                "chunk_id": ch.chunk_id,  # internal only
                "text": (ch.text or "")[:1200],
                "score": float(score),
            })
        return json.dumps(payload, ensure_ascii=False)

    def _doc_get_chunk_fn(chunk_id: str, max_chars: int = 2600) -> str:
        cid = (chunk_id or "").strip()
        if not cid:
            return ""
        for c in store.chunks:
            if c.chunk_id == cid:
                return (c.text or "")[:max_chars]
        return ""

    def _mk_tool(fn, name: str, description: str) -> BaseTool:
        return StructuredTool.from_function(fn, name=name, description=description)

    tool_get_usage = _mk_tool(_get_usage_fn, "get_usage", "Get current tool usage counters as JSON (budget/debug).")
    tool_doc_list = _mk_tool(_doc_list_fn, "doc_list", "List indexed documents and chunk counts.")
    tool_doc_search = _mk_tool(_doc_search_fn, "doc_search", "Semantic search over indexed chunks. Returns JSON hits with title/page/chunk_id/text.")
    tool_doc_get_chunk = _mk_tool(_doc_get_chunk_fn, "doc_get_chunk", "Fetch full text for a given chunk_id for close reading. Returns text only.")

    tools: list[BaseTool] = [tool_get_usage, tool_doc_list, tool_doc_search, tool_doc_get_chunk]

    tool_web_search_summary: Optional[BaseTool] = None
    if enable_web:
        def _web_search_summary_fn(query: str) -> str:
            if not _inc("web_search_calls", DA_MAX_WEB_SEARCH_CALLS):
                # ✅ 不要把 Budget exceeded 的文字塞進 evidence/草稿，避免污染最終輸出
                return "[WebSearch:web p-]\nSources:"

            q = (query or "").strip()
            if not q:
                return "[WebSearch:web p-]\nSources:"

            system = (
                "你是研究助理。用繁體中文（台灣用語）整理 web_search 結果。\n"
                "輸出格式必須固定：\n"
                "1) 先用 3~8 個 bullets 摘要（每點一句，清楚、帶日期/數字則保留）。\n"
                "2) 最後一定要有一段 Sources:，用條列列出來源。\n"
                "   每列格式：- <domain> | <title> | <url>\n"
                "規則：\n"
                "- 不要提到工具流程/額度/Budget exceeded。\n"
                "- 若找不到可靠來源：摘要可為空，但仍要輸出 Sources:。\n"
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

            def _domain(u: str) -> str:
                try:
                    host = urlparse(u).netloc.lower()
                    if host.startswith("www."):
                        host = host[4:]
                    return host or "web"
                except Exception:
                    return "web"

        # ✅ 用第一個來源的 domain 當 citation header： [WebSearch:<domain> p-]
            primary_domain = "web"
            if isinstance(sources, list) and sources:
                first = sources[0] if isinstance(sources[0], dict) else {}
                u0 = (first.get("url") or "").strip()
                if u0:
                    primary_domain = _domain(u0)

            src_lines = []
            for s in (sources or [])[:10]:
                if isinstance(s, dict):
                    t = (s.get("title") or s.get("source") or "source").strip()
                    u = (s.get("url") or "").strip()
                    if u:
                        src_lines.append(f"- {_domain(u)} | {t} | {u}")

            out_text = (text or "").strip()
            if src_lines:
                out_text = (out_text + "\n\nSources:\n" + "\n".join(src_lines)).strip()
            else:
                # 仍保留 Sources: 標頭，讓 downstream 好解析
                out_text = (out_text + "\n\nSources:").strip()

            return f"[WebSearch:{primary_domain} p-]\n" + out_text[:2400]

        tool_web_search_summary = _mk_tool(
            _web_search_summary_fn,
            "web_search_summary",
            "Run web_search and return a short Traditional Chinese summary with sources (domain|title|url).",
        )
        tools.append(tool_web_search_summary)
    
    retriever_prompt = f"""
你是文件檢索專家（只允許使用 doc_list/doc_search/doc_get_chunk/get_usage）。

facet 子任務格式：
facet_slug: <英文小寫_底線>
facet_goal: <這個面向要回答什麼>
hints: <可能的關鍵字/指標/名詞（可空）>

硬規則：
- 你要寫入 /evidence/doc_<facet_slug>.md
- evidence 內容只能包含：
  1) 引用標頭：[報告名稱 p頁]（不得包含 /evidence/ 路徑；不得用 doc_*.md 當報告名稱）
  2) 原文片段（可截斷）
  3) 一行說明「這段支持什麼」
- 你可以用 doc_search 拿到 chunk_id，然後用 doc_get_chunk(chunk_id=...) 精讀，
  但 chunk_id 絕對不能寫進 evidence。

若遇到 Budget exceeded：停止，且不要把錯誤字串抄進 evidence（只要停止即可）。
最後回覆 orchestrator：≤150 字摘要（找到什麼 + 最大缺口）
"""

    writer_prompt = f"""
你是寫作/整理專家（用 read_file/glob/grep/write_file/edit_file/ls）。
你必須整合 /evidence/ 底下所有檔案（doc_*.md 與可選 web_*.md）。

任務類型判斷：
- 完整報告/章節 → REPORT
- 單題回答 → QA
- 整理知識脈絡 → KNOWLEDGE
- 使用者貼草稿要查核/除錯 → VERIFY_DRAFT（最多 {DA_MAX_CLAIMS} 條主張）

引用規則（嚴格）：
- QA：純 bullet（每行 -），每個 bullet 句尾必有引用 [報告名稱 p頁] 或 [WebSearch:* p-]
- REPORT/KNOWLEDGE/VERIFY：Markdown；每個非標題段落至少 1 個引用
- enable_web=false：不得出現 WebSearch
- draft 絕對不能出現 chunk_id
- 報告名稱不得是 /evidence/*.md
- 若同一句需要多個引用，請用多個方括號連續附在句尾，例如：[A p2][B p8]

輸出禁則（非常重要）：
- 成品 /draft.md 不得提及內部流程或檔名：不得出現「/evidence」、「doc_*.md」、「web_*.md」、「Budget exceeded」、「額度不足」、「占位」、「向量庫」等字樣。
- 若某面向找不到可引用來源：在 /draft.md 直接省略該面向，不要寫「證據不足/額度不足/未能取得」等段落。
- 只寫你有引用支撐、能說清楚的內容。

# 格式化規則
- 根據內容選擇最合適的 Markdown 格式及彩色徽章（colored badges）元素表達。
- 可愛語氣與彩色元素是輔助閱讀的裝飾，而不是主要結構；**不可取代清楚的標題、條列與段落組織**。

# Markdown 格式與 emoji／顏色用法說明
## 基本原則
- 根據內容選擇最合適的強調方式，讓回應清楚、易讀、有層次，避免過度使用彩色文字與 emoji 造成視覺負擔。
- 只用 Streamlit 支援的 Markdown 語法，不要用 HTML 標籤。

## 功能與語法
- **粗體**：`**重點**` → **重點**
- *斜體*：`*斜體*` → *斜體*
- 標題：`# 大標題`、`## 小標題`
- 分隔線：`---`
- 表格（僅部分平台支援，建議用條列式）
- 引用：`> 這是重點摘要`
- emoji：直接輸入或貼上，如 😄
- Material Symbols：`:material/star:`
- LaTeX 數學公式：`$公式$` 或 `$$公式$$`
- 彩色文字：`:orange[重點]`、`:blue[說明]`
- 彩色背景：`:orange-background[警告內容]`
- 彩色徽章：`:orange-badge[重點]`、`:blue-badge[資訊]`
- 小字：`:small[這是輔助說明]`

## 顏色名稱及建議用途（條列式，跨平台穩定）
- **blue**：資訊、一般重點
- **green**：成功、正向、通過
- **orange**：警告、重點、溫暖
- **red**：錯誤、警告、危險
- **violet**：創意、次要重點
- **gray/grey**：輔助說明、備註
- **rainbow**：彩色強調、活潑
- **primary**：依主題色自動變化

**注意：**
- 只能使用上述顏色。**請勿使用 yellow（黃色）**，如需黃色效果，請改用 orange 或黃色 emoji（🟡、✨、🌟）強調。
- 不支援 HTML 標籤，請勿使用 `<span>`、`<div>` 等語法。
- 建議只用標準 Markdown 語法，保證跨平台顯示正常。

## 最終把結果寫到 /draft.md
"""

    verifier_prompt = f"""
你是審稿查核專家（用 read_file/edit_file/grep）。
任務：檢查 /draft.md 是否符合引用覆蓋，並做『最少改動』修正。

規則：
- QA：每個 bullet 句尾必有 [.. p..]
- 其他：每個非標題段落至少 1 個引用 [.. p..]
- enable_web=false：不得出現 WebSearch
- 若 /draft.md 出現 chunk_id 痕跡（chunk_id= 或 _p*_c*），必須移除。
- 引用標頭不得使用 /evidence/*.md

額外規則：
- /draft.md 若出現「/evidence、doc_、web_、Budget exceeded、額度不足、占位、向量庫」等內部字樣，必須移除。
- 若整段主要在講「找不到資料/額度不足」，請刪除該段落（使用者只想看有查到的內容）。

最多修正 {DA_MAX_REWRITE_ROUNDS} 輪：
- 每輪：read /draft.md → edit_file 修正 → write /review.md 記錄
"""

    subagents = [
        {
            "name": "retriever",
            "description": "從上傳文件向量庫找證據，寫 /evidence/doc_*.md（不含 chunk_id）",
            "system_prompt": retriever_prompt,
            "tools": [tool_get_usage, tool_doc_list, tool_doc_search, tool_doc_get_chunk],
            "model": f"openai:{MODEL_MAIN}",
        },
        {
            "name": "writer",
            "description": "整合 /evidence/ → 產生 /draft.md",
            "system_prompt": writer_prompt,
            "tools": [],
            "model": f"openai:{MODEL_MAIN}",
        },
        {
            "name": "verifier",
            "description": "檢查引用覆蓋並修稿 /draft.md，寫 /review.md",
            "system_prompt": verifier_prompt,
            "tools": [],
            "model": f"openai:{MODEL_MAIN}",
        },
    ]

    if enable_web:
        web_prompt = """
你是網路搜尋專家（只允許 web_search_summary/get_usage；不允許 doc_*）。
facet 子任務格式同 retriever。

硬規則：
- 寫入 /evidence/web_<facet_slug>.md
- 每段要保留引用標頭 [WebSearch:<domain> p-]
- 禁止捏造來源
- 不要寫「Budget exceeded/額度不足/占位」到 evidence；若沒有來源就不要寫那段
"""
        subagents.insert(
            1,
            {
                "name": "web-researcher",
                "description": "用 OpenAI 內建 web_search 做少量高品質搜尋，寫 /evidence/web_*.md",
                "system_prompt": web_prompt,
                "tools": [tool_web_search_summary, tool_get_usage] if tool_web_search_summary else [tool_get_usage],
                "model": f"openai:{MODEL_MAIN}",
            },
        )

    orchestrator_prompt = f"""
你是 Deep Doc Orchestrator（文件優先；enable_web={str(enable_web).lower()}）。

固定流程（必做）：
0) 立刻用 write_file 建立 /workspace/todos.json（JSON array of strings；5~9 步），並在第一步說明 enable_web 與目標。
1) write_file /evidence/README.md 記錄本次需求與 enable_web
2) 拆 2–4 個 facets（面向，不是章節）
3) 平行派工：
   - 每個 facet 至少派 1 個 retriever
   - enable_web=true 且需要外部背景時，對同 facet 再派 1 個 web-researcher
4) 叫 writer 產生 /draft.md
5) 叫 verifier 修稿（最多 {DA_MAX_REWRITE_ROUNDS} 輪）
6) read_file /draft.md 作為最終回答

引用與隱私規則：
- /evidence 與 /draft 絕對不能出現 chunk_id
- 引用只能用 [報告名稱 p頁] 或 [WebSearch:<domain> p-]
- 報告名稱不得使用 /evidence/*.md 當作來源名稱
- 多引用請用 [A p2][B p8]，不要在同一對 [] 內塞多筆
- 成品不得提及內部流程/檔名/額度不足；只保留有引用支撐的內容
"""

    llm = _make_langchain_llm(model_name=f"openai:{MODEL_MAIN}", temperature=0.0, reasoning_effort=REASONING_EFFORT)

    agent = create_deep_agent(
        model=llm,
        tools=tools,
        system_prompt=orchestrator_prompt,
        subagents=subagents,
        debug=False,
        name="deep-doc-agent",
    ).with_config({"recursion_limit": int(st.session_state.get("langgraph_recursion_limit", DEFAULT_RECURSION_LIMIT))})

    st.session_state.deep_agent = agent
    st.session_state.deep_agent_web_flag = bool(enable_web)
    return agent

# =========================
# Fallback (RAG)
# =========================
def fallback_answer_from_store(
    client: OpenAI,
    store: Optional[FaissStore],
    question: str,
    *,
    k: int = 10,
) -> str:
    q = (question or "").strip()
    if not q:
        return "（系統：問題為空，無法產生回答）"

    if store is None or getattr(store, "index", None) is None or store.index.ntotal == 0:
        system = "你是助理。用繁體中文（台灣用語）回答，結構清楚。"
        ans, _ = call_gpt(client, model=MODEL_MAIN, system=system, user=q, reasoning_effort=None, tools=None)
        return ans or "（系統：無索引且模型未產出內容）"

    qvec = embed_texts(client, [q])
    hits = store.search(qvec, k=max(4, min(12, int(k))))
    chunks = [ch for _, ch in hits]
    ctx = render_chunks_for_model(chunks, max_chars_each=900)

    system = (
        "你是嚴謹的研究助理，只能根據我提供的資料回答，不可腦補。\n"
        "輸出要求：\n"
        "1) 純 bullet，每行以 - 開頭。\n"
        "2) 每個 bullet 句尾必須附引用，格式固定：[報告名稱 p頁]。\n"
        "3) 引用中的報告名稱必須來自我提供的資料片段標頭（例如 [XXX p12]）。\n"
        "4) 不可使用 /evidence/*.md 當作報告名稱。\n"
        "5) 若同一句需要多個引用，請用多個方括號連續附在句尾，例如：[A p2][B p8]。\n"
    )
    user = f"問題：{q}\n\n資料：\n{ctx}\n"

    out, _ = call_gpt(
        client,
        model=MODEL_MAIN,
        system=system,
        user=user,
        reasoning_effort=REASONING_EFFORT,
        tools=None,
        include_sources=False,
    )
    out = (out or "").strip()
    out = strip_internal_process_lines(out)
    return out or "（系統：fallback RAG 未產出內容）"


# =========================
# DeepAgent run + stall detect
# =========================
# === Patch 1) 新增一個共用 helper：建立本次 run 的 messages（避免重複塞 prompt） ===
# ------------------------------------------------------------
# (1) 新增/確認：共用 run_messages 的 helper（若你已經加過 build_run_messages，略過）
#     放在 get_recent_chat_messages() 附近最順
# ------------------------------------------------------------
def build_run_messages(prompt: str, max_messages: int = 15) -> list[dict]:
    """
    共用短期記憶（同一份給 direct / deepagent / todo 用）。
    - 從 chat_history 抽最近 max_messages 則 kind=text 的 user/assistant
    - 確保最後一則一定是本次 prompt 的 user 訊息（避免漏或重複）
    """
    msgs = get_recent_chat_messages(max_messages=max_messages)

    # 若 chat_history 已經 append 本次 prompt，最後一則就是它 → 直接回傳
    if msgs and msgs[-1].get("role") == "user" and (msgs[-1].get("content") or "").strip() == (prompt or "").strip():
        return msgs

    msgs.append({"role": "user", "content": (prompt or "").strip()})
    return msgs

# === Patch 2) 修改 deep_agent_run_with_live_status：多收一個 run_messages，並餵給 agent.stream ===
def deep_agent_run_with_live_status(agent, user_text: str, run_messages: list[dict]) -> Tuple[str, Optional[dict]]:
    final_state = None
    todos_preview_written = False

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
            "plan": ("DeepAgent：規劃中…", "running"),
            "evidence": ("DeepAgent：蒐證中…", "running"),
            "draft": ("DeepAgent：寫作中…", "running"),
            "review": ("DeepAgent：審稿/補引用中…", "running"),
            "done": ("DeepAgent：完成", "complete"),
            "error": ("DeepAgent：發生錯誤", "error"),
        }
        label, state = mapping.get(phase, ("DeepAgent：執行中…", "running"))
        s.update(label=label, state=state, expanded=False)

    # ✅ 深度記憶：deepagent 直接吃 messages list（與 direct 共用同一份）
    # 保險：確保最後是 user_text
    msgs_for_agent = list(run_messages or [])
    if not msgs_for_agent or msgs_for_agent[-1].get("role") != "user":
        msgs_for_agent.append({"role": "user", "content": user_text})
    elif (msgs_for_agent[-1].get("content") or "").strip() != (user_text or "").strip():
        msgs_for_agent.append({"role": "user", "content": user_text})

    with st.status("DeepAgent：啟動中…", expanded=False) as s:
        set_phase(s, "start")
        set_phase(s, "plan")

        try:
            for state in agent.stream(
                {"messages": [{"role": "user", "content": user_text}]},
                stream_mode="values",
                config={"recursion_limit": recursion_limit},
            ):
                final_state = state
                files = state.get("files") or {}
                file_keys = set(files.keys()) if isinstance(files, dict) else set()

                if (not todos_preview_written) and isinstance(files, dict) and "/workspace/todos.json" in files:
                    todos_txt = get_files_text(files, "/workspace/todos.json")
                    if todos_txt:
                        s.write("### 本次 Todo（規劃結果預覽）")
                        s.code(todos_txt[:4000], language="json")
                        todos_preview_written = True

                if any(k.startswith("/evidence/") for k in file_keys):
                    set_phase(s, "evidence")
                if "/draft.md" in file_keys:
                    set_phase(s, "draft")
                if "/review.md" in file_keys:
                    set_phase(s, "review")

                # ✅ 卡住判定：draft 夠長後才開始（不變 + 無引用）連續 N 步
                if isinstance(files, dict) and "/draft.md" in files:
                    draft_txt = get_files_text(files, "/draft.md")
                    draft_norm = norm_space(draft_txt)
                    if len(draft_norm) >= stall_min_chars:
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
                            s.warning(
                                f"判定卡住：/draft.md 內容連續 {draft_unchanged_streak} 步未變、且連續 {draft_no_citation_streak} 步無引用。"
                                "已強制結束 DeepAgent，改用 fallback 產出答案。"
                            )
                            answer = fallback_answer_from_store(client, st.session_state.get("store", None), user_text, k=10)
                            return answer, files if isinstance(files, dict) and files else None

        except GraphRecursionError:
            set_phase(s, "error")
            st.session_state["last_run_forced_end"] = "recursion_limit"

            files = (final_state or {}).get("files") or {}
            draft = get_files_text(files, "/draft.md") if isinstance(files, dict) else ""
            draft = strip_internal_process_lines(draft)
            if draft.strip():
                s.warning(f"已達步數上限（recursion_limit={recursion_limit}），回傳目前 /draft.md。")
                return draft.strip(), (files if isinstance(files, dict) and files else None)

            s.warning(f"已達步數上限（recursion_limit={recursion_limit}），改用 fallback 產生回答。")
            answer = fallback_answer_from_store(client, st.session_state.get("store", None), user_text, k=10)
            return answer, (files if isinstance(files, dict) and files else None)

        except Exception as e:
            msg = str(e)
            if "Budget exceeded" in msg:
                set_phase(s, "evidence")
                s.update(label="DeepAgent：已達工具預算上限（停止加搜證）", state="running", expanded=False)
            else:
                set_phase(s, "error")
                raise

        files = (final_state or {}).get("files") or {}
        final_text = get_files_text(files, "/draft.md")

        if not final_text:
            msgs = (final_state or {}).get("messages") or []
            if msgs:
                last = msgs[-1]
                content = getattr(last, "content", None)
                final_text = (file_to_text(content) or file_to_text(last)).strip()

        if final_text and CHUNK_ID_LEAK_PAT.search(final_text):
            final_text = CHUNK_ID_LEAK_PAT.sub("", final_text)

        final_text = strip_internal_process_lines(final_text)

        set_phase(s, "done")

    return final_text or "（DeepAgent 沒有產出內容）", files if isinstance(files, dict) and files else None

# =========================
# need_todo decision
# =========================
def decide_need_todo(client: OpenAI, question: str) -> Tuple[bool, str]:
    system = (
        "你是路由器。請判斷這個問題是否需要做『Todo + 文件/網路檢索』。\n"
        "規則：\n"
        "- 需要：涉及具體事實、數據、政策、版本、引用、研究、比較、出處；或需要引用上傳文件。\n"
        "- 不需要：純意見/寫作/改寫/腦力激盪/不要求引用的泛泛解釋。\n"
        "請輸出 JSON：{\"need_todo\": true/false, \"reason\": \"...\"}（只輸出 JSON）"
    )
    out, _ = call_gpt(
        client,
        model=MODEL_MAIN,
        system=system,
        user=question,
        reasoning_effort=REASONING_EFFORT,
        tools=None,
        include_sources=False,
    )
    data = _try_parse_json_or_py_literal(out) or {}
    need = bool(data.get("need_todo", False))
    reason = str(data.get("reason", "")).strip() or "（未提供原因）"
    return need, reason


# =========================
# UI badges
# =========================
def render_run_badges(
    *,
    mode: str,
    need_todo: bool,
    reason: str,
    usage: dict,
    enable_web: bool,
    todo_file_present: Optional[bool] = None,
    forced_end: Optional[str] = None,
) -> None:
    badges: List[str] = []
    badges.append(_badge_directive(f"Mode:{mode}", "gray"))

    # ✅ 你要求：不管模式都要 Todo 分析（所以這裡永遠顯示 Todo:需要）
    badges.append(_badge_directive("Todo:需要", "blue"))

    if todo_file_present is True:
        badges.append(_badge_directive("Todos.json:有", "blue"))
    elif todo_file_present is False:
        badges.append(_badge_directive("Todos.json:無(流程異常)", "orange"))

    if forced_end:
        mapping = {
            "citation_stall": "ForcedStop:卡住(引用未生成)",
            "recursion_limit": "ForcedStop:步數上限",
        }
        label = mapping.get(forced_end, f"ForcedStop:{forced_end}")
        badges.append(_badge_directive(label, "orange"))
        badges.append(_badge_directive("Fallback:RAG", "orange"))

    doc_calls = int((usage or {}).get("doc_search_calls", 0) or 0)
    web_calls = int((usage or {}).get("web_search_calls", 0) or 0)

    badges.append(_badge_directive(f"DB:{'used' if doc_calls else 'unused'}({doc_calls})" if doc_calls else "DB:unused", "green" if doc_calls else "gray"))
    if enable_web:
        badges.append(_badge_directive(f"Web:used({web_calls})" if web_calls else "Web:unused", "violet" if web_calls else "gray"))
    else:
        badges.append(_badge_directive("Web:disabled", "gray"))

    st.markdown(" ".join(badges))


def get_forced_end() -> Optional[str]:
    return st.session_state.get("last_run_forced_end", None)


# =========================
# Debug panel (selectable)
# =========================
def render_debug_panel(files: Optional[dict]) -> None:
    if not files or not isinstance(files, dict):
        st.write("（沒有 files）")
        return

    all_keys = sorted([k for k in files.keys() if isinstance(k, str)])
    evidence_keys = [k for k in all_keys if k.startswith("/evidence/")]
    doc_evidence_keys = [k for k in evidence_keys if k.startswith("/evidence/doc_")]
    web_evidence_keys = [k for k in evidence_keys if k.startswith("/evidence/web_")]

    todos = get_files_text(files, "/workspace/todos.json") if "/workspace/todos.json" in files else ""
    readme = get_files_text(files, "/evidence/README.md") if "/evidence/README.md" in files else ""
    draft = get_files_text(files, "/draft.md") if "/draft.md" in files else ""
    review = get_files_text(files, "/review.md") if "/review.md" in files else ""

    tab_overview, tab_orch, tab_retr, tab_web, tab_writer, tab_verifier, tab_browser = st.tabs(
        ["總覽", "Orchestrator", "Retriever(evidence)", "Web(evidence)", "Writer(draft)", "Verifier(review)", "Files browser"]
    )

    with tab_overview:
        st.write(f"files keys：{len(all_keys)}")
        st.write(f"evidence：{len(evidence_keys)}")
        st.code("\n".join(all_keys[:800]), language="text")

    with tab_orch:
        st.markdown("### /workspace/todos.json")
        st.code((todos or "（無）")[:20000], language="json")
        st.divider()
        st.markdown("### /evidence/README.md")
        st.code((readme or "（無）")[:20000], language="markdown")

    with tab_retr:
        if not doc_evidence_keys:
            st.write("（沒有 /evidence/doc_*.md）")
        else:
            pick = st.selectbox("選擇 retriever evidence", doc_evidence_keys, index=0)
            st.code(get_files_text(files, pick)[:60000], language="markdown")

    with tab_web:
        if not web_evidence_keys:
            st.write("（沒有 /evidence/web_*.md）")
        else:
            pick = st.selectbox("選擇 web evidence", web_evidence_keys, index=0)
            st.code(get_files_text(files, pick)[:60000], language="markdown")

    with tab_writer:
        st.code((draft or "（沒有 /draft.md）")[:60000], language="markdown")

    with tab_verifier:
        st.code((review or "（沒有 /review.md）")[:60000], language="markdown")

    with tab_browser:
        if not all_keys:
            st.write("（files 為空）")
        else:
            pick = st.selectbox("選擇任一檔案（files）", all_keys, index=0)
            st.code(get_files_text(files, pick)[:60000], language="text")


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
st.session_state.setdefault("default_outputs", None)
st.session_state.setdefault("chat_history", [])

# Popover 只留 web 開關
st.session_state.setdefault("enable_web_search_agent", False)

# 其餘固定預設（不提供 UI）
st.session_state.setdefault("langgraph_recursion_limit", DEFAULT_RECURSION_LIMIT)
st.session_state.setdefault("citation_stall_steps", DEFAULT_CITATION_STALL_STEPS)
st.session_state.setdefault("citation_stall_min_chars", DEFAULT_CITATION_STALL_MIN_CHARS)
st.session_state.setdefault("last_run_forced_end", None)

st.session_state.setdefault("enable_output_formatter", True)
st.session_state.setdefault("sources_badge_max_titles_inline", DEFAULT_SOURCES_BADGE_MAX_TITLES_INLINE)
st.session_state.setdefault("sources_badge_max_pages_per_title", DEFAULT_SOURCES_BADGE_MAX_PAGES_PER_TITLE)

ENABLE_FORMATTER_FOR_DIRECT = True
ENABLE_FORMATTER_FOR_DEEPAGENT = False

# =========================
# File table helpers
# =========================
def file_rows_to_df(rows: list[FileRow]) -> pd.DataFrame:
    recs = []
    for r in rows:
        if r.ext == ".pdf":
            pages_str = "-" if r.pages is None else str(r.pages)
            text_pages_str = "-" if r.text_pages is None else str(r.text_pages)
            text_ratio_str = "-" if r.text_pages_ratio is None else f"{r.text_pages_ratio:.0%}"
        else:
            pages_str = "-" if r.pages is None else str(r.pages)
            text_pages_str = "-"
            text_ratio_str = "-"

        if r.ext == ".pdf" and r.likely_scanned:
            suggest = "建議 OCR"
        elif r.ext in (".png", ".jpg", ".jpeg"):
            suggest = "必 OCR"
        else:
            suggest = ""

        recs.append({
            "_file_id": r.file_id,
            "使用OCR": bool(r.use_ocr),
            "檔名": truncate_filename(r.name, 52),
            "格式": r.ext.replace(".", ""),
            "頁數": pages_str,
            "文字頁": text_pages_str,
            "文字%": text_ratio_str,
            "token估算": int(r.token_est),
            "建議": suggest,
        })
    return pd.DataFrame(recs)


def sync_df_to_file_rows(df: pd.DataFrame, rows: list[FileRow]) -> None:
    id_to_row_idx = {r.file_id: i for i, r in enumerate(rows)}
    for _, rec in df.iterrows():
        fid = rec.get("_file_id")
        if fid not in id_to_row_idx:
            continue
        i = id_to_row_idx[fid]

        ext = rows[i].ext
        if ext in (".png", ".jpg", ".jpeg"):
            rows[i].use_ocr = True
        elif ext == ".txt":
            rows[i].use_ocr = False
        else:
            rows[i].use_ocr = bool(rec.get("使用OCR", rows[i].use_ocr))


# =========================
# Popover：文件管理（只留網搜開關；其餘 UI 不顯示）
# =========================
with st.popover("📦 文件管理（上傳 / OCR / 建索引 / DeepAgent設定）"):
    st.caption("支援 PDF/TXT/PNG/JPG。PDF 若文字抽取偏少會建議 OCR（逐檔可勾選）。")
    st.caption("✅ 不上傳文件也能聊天；只有你需要引用文件時才需要建立索引。")

    has_index = (
        st.session_state.store is not None
        and getattr(st.session_state.store, "index", None) is not None
        and st.session_state.store.index.ntotal > 0
    )
    if has_index:
        st.success(f"已建立索引：檔案數={len(st.session_state.file_rows)} / chunks={len(st.session_state.store.chunks)}")
        st.caption("來源以 badge 顯示（文件：檔名 + 頁碼；網路：domain + p-）。")
    else:
        st.info("目前沒有索引：你仍可直接聊天（純 LLM）。若需要引用文件，再在此處上傳並建立索引。")

    # ✅ 只留這個
    st.session_state.enable_web_search_agent = st.checkbox(
        "啟用網路搜尋（會增加成本）",
        value=bool(st.session_state.enable_web_search_agent),
    )

    uploaded = st.file_uploader(
        "上傳文件",
        type=["pdf", "txt", "png", "jpg", "jpeg"],
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
                use_ocr = bool(likely_scanned)

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

    st.markdown("### 文件清單（可逐檔勾選 OCR）")

    if not st.session_state.file_rows:
        st.info("尚未上傳文件。")
    else:
        df = file_rows_to_df(st.session_state.file_rows)
        edited = st.data_editor(
            df.drop(columns=["_file_id"]),
            key="file_table_editor",
            width="stretch",
            hide_index=True,
            disabled=["檔名", "格式", "頁數", "文字頁", "文字%", "token估算", "建議"],
            column_config={
                "使用OCR": st.column_config.CheckboxColumn(
                    "使用OCR",
                    help="逐檔選擇是否啟用 OCR（PDF 可選；圖檔固定OCR；TXT固定不OCR）",
                ),
            },
        )

        df_for_sync = df.copy()
        df_for_sync["使用OCR"] = edited["使用OCR"].values
        sync_df_to_file_rows(df_for_sync, st.session_state.file_rows)

    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])
    build_btn = col1.button("🚀 建立索引", type="primary", width="stretch")
    default_btn = col2.button("🧾 產生預設輸出", width="stretch")
    clear_btn = col3.button("🧹 清空全部", width="stretch")

    if clear_btn:
        st.session_state.file_rows = []
        st.session_state.file_bytes = {}
        st.session_state.store = None
        st.session_state.processed_keys = set()
        st.session_state.default_outputs = None
        st.session_state.chat_history = []
        st.session_state.deep_agent = None
        st.session_state.deep_agent_web_flag = None
        st.session_state.da_usage = {"doc_search_calls": 0, "web_search_calls": 0}
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
            s.write(f"新增報告數：{stats['new_reports']}")
            s.write(f"新增 chunks：{stats['new_chunks']}")
            s.write(f"耗時：{time.perf_counter() - t0:.2f}s")
            s.update(state="complete")

        st.session_state.deep_agent = None
        st.session_state.deep_agent_web_flag = None
        st.rerun()

    if default_btn:
        if st.session_state.store is None or st.session_state.store.index.ntotal == 0:
            st.warning("尚未建立索引或沒有 chunks，請先按「建立索引」。")
        else:
            with st.status("產生預設輸出（摘要/主張/推論鏈）...", expanded=True) as s2:
                chosen = pick_corpus_chunks_for_default(st.session_state.store.chunks)
                ctx = render_chunks_for_model(chosen)
                bundle = generate_default_outputs_bundle(client, "整體融合（全部上傳報告）", ctx, max_retries=2)
                st.session_state.default_outputs = bundle
                s2.update(state="complete")

            st.session_state.chat_history.append({
                "role": "assistant",
                "kind": "default",
                "title": "整體融合（全部上傳報告）",
                **(st.session_state.default_outputs or {}),
            })
            st.rerun()


# =========================
# 主畫面：Chat
# =========================
has_index = (
    st.session_state.store is not None
    and getattr(st.session_state.store, "index", None) is not None
    and st.session_state.store.index.ntotal > 0
)

# 顯示歷史（✅ user 不顯示 badge）
for msg in st.session_state.chat_history:
    role = msg.get("role", "assistant")
    with st.chat_message(role):
        if role == "user":
            st.markdown(msg.get("content", ""))
            continue

        if msg.get("kind") == "default":
            st.markdown(f"## 預設輸出：{msg.get('title','')}")
            st.markdown("### 1) 報告摘要")
            st.code((msg.get("summary", "") or "")[:20000], language="markdown")
            st.markdown("### 2) 核心主張")
            st.code((msg.get("claims", "") or "")[:20000], language="markdown")
            st.markdown("### 3) 推論鏈")
            st.code((msg.get("chain", "") or "")[:20000], language="markdown")
        else:
            meta = msg.get("meta", {}) or {}
            render_run_badges(
                mode=meta.get("mode", "unknown"),
                need_todo=True,
                reason=str(meta.get("reason", "") or ""),
                usage=meta.get("usage", {}) or {},
                enable_web=bool(meta.get("enable_web", False)),
                todo_file_present=meta.get("todo_file_present", None),
                forced_end=meta.get("forced_end", None),
            )
            render_markdown_answer_with_sources_badges(msg.get("content", ""))
            render_web_sources_list(meta.get("web_sources", {}) or {})


prompt = st.chat_input("請輸入問題（也可貼草稿要我查核/除錯）。")
if prompt:
    st.session_state.chat_history.append({"role": "user", "kind": "text", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        enable_web = bool(st.session_state.enable_web_search_agent)

        # ✅ 不管 direct/deepagent，都做 todo 判斷（你要求）
        need_todo, reason = decide_need_todo(client, prompt)
        run_messages = build_run_messages(prompt, max_messages=15)

        planned_mode = "deepagent" if (has_index and need_todo) else "direct"
        todos_json_text = build_todos_json_for_question(
            client,
            prompt,
            enable_web=enable_web,
            has_index=has_index,
            planned_mode=planned_mode,
            run_messages=run_messages,  # ✅ 新增：Todo 吃共用記憶
        )

        # direct
        if planned_mode == "direct":
            system = ANYA_SYSTEM_PROMPT

            # ✅ 短期記憶（N=10）：把最近對話壓成 context（不改 call_gpt wrapper 也能用）
            memory_msgs = run_messages[:-1]  # 排除本次 prompt
            history_block = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in memory_msgs if m.get("role") in ("user", "assistant") and m.get("content")])
            user_text = prompt
            if history_block.strip():
                user_text = f"對話脈絡（最近）：\n{history_block}\n\n目前問題：\n{prompt}"

            web_sources = {}

            if enable_web:
                # ✅ direct 也支援 web_search（並回收 sources）
                answer_text, sources = call_gpt(
                    client,
                    model=MODEL_MAIN,
                    system=system,
                    user=user_text,
                    reasoning_effort=REASONING_EFFORT,
                    tools=[{"type": "web_search"}],
                    include_sources=True,
                    tool_choice="required",  # ✅ 關鍵：保證真的 call web_search 才會有 sources
                )

                web_sources = web_sources_from_openai_sources(sources)

                # ✅ 保證至少有一個 WebSearch 引用 token，讓 badge 能顯示
                primary_domain = "web"
                if web_sources:
                    primary_domain = sorted(web_sources.keys())[0]  # 取一個穩定的 domain
                    answer_text = ensure_web_citation_token(answer_text or "", primary_domain)
                else:
                    # 沒抓到 sources，就不要硬塞 [WebSearch:web p-]，避免誤導
                    pass

                # usage 記一個（call_gpt 內部 tool 可能多次 call，但 direct 我們至少標示有用 web）
                usage = {"doc_search_calls": 0, "web_search_calls": 1}
            else:
                answer_text, _ = call_gpt(
                    client,
                    model=MODEL_MAIN,
                    system=system,
                    user=user_text,
                    reasoning_effort=REASONING_EFFORT,
                    tools=None,
                )
                usage = {"doc_search_calls": 0, "web_search_calls": 0}

            # formatter（只留 direct） + 去內部流程
            if ENABLE_FORMATTER_FOR_DIRECT and st.session_state.get("enable_output_formatter", True):
                answer_text = format_markdown_output_preserve_citations(client, answer_text)
            answer_text = strip_internal_process_lines(answer_text)

            # ✅ direct 也造出 files（供 debug / badges）
            files = {"/workspace/todos.json": todos_json_text}

            meta = {
                "mode": "direct",
                "need_todo": True,
                "reason": reason,
                "usage": usage,
                "enable_web": enable_web,
                "todo_file_present": True,
                "forced_end": None,
                "web_sources": web_sources,  # ✅ 之後你要在 UI 列 URL 就用這個
            }

            render_run_badges(
                mode=meta["mode"],
                need_todo=True,
                reason=reason,
                usage=meta["usage"],
                enable_web=enable_web,
                todo_file_present=True,
                forced_end=None,
            )
            render_markdown_answer_with_sources_badges(answer_text)

            # （可選）如果你已經有 render_web_sources_list，就可以直接把 URL 列出來
            # render_web_sources_list(web_sources)

            with st.expander("Debug", expanded=False):
                st.markdown("### 本次 Todo（direct 產生）")
                st.code(todos_json_text[:20000], language="json")

                if enable_web:
                    st.divider()
                    st.markdown("### 本次 Web Sources（direct）")
                    st.code(json.dumps(web_sources, ensure_ascii=False, indent=2)[:20000], language="json")

            st.session_state.chat_history.append({"role": "assistant", "kind": "text", "content": answer_text, "meta": meta})
            st.stop()
        # deepagent
        agent = ensure_deep_agent(client=client, store=st.session_state.store, enable_web=enable_web)
        answer_text, files = deep_agent_run_with_live_status(agent, prompt, run_messages)

        if ENABLE_FORMATTER_FOR_DEEPAGENT and st.session_state.get("enable_output_formatter", True):
            answer_text = format_markdown_output_preserve_citations(client, answer_text)
        answer_text = strip_internal_process_lines(answer_text)

        # deepagent：若沒產 todos.json，用 direct 計畫補上（保證有）
        if not isinstance(files, dict):
            files = {}
        if "/workspace/todos.json" not in files:
            files["/workspace/todos.json"] = todos_json_text

        todo_file_present = isinstance(files, dict) and ("/workspace/todos.json" in files)

        meta = {
            "mode": "deepagent",
            "need_todo": True,
            "reason": reason,
            "usage": dict(st.session_state.get("da_usage", {"doc_search_calls": 0, "web_search_calls": 0})),
            "enable_web": enable_web,
            "todo_file_present": bool(todo_file_present),
            "forced_end": get_forced_end(),
        }

        render_run_badges(
            mode=meta["mode"],
            need_todo=True,
            reason=reason,
            usage=meta["usage"],
            enable_web=enable_web,
            todo_file_present=meta["todo_file_present"],
            forced_end=meta.get("forced_end"),
        )
        render_markdown_answer_with_sources_badges(answer_text)

        with st.expander("Debug", expanded=False):
            todos_txt = get_files_text(files, "/workspace/todos.json") if isinstance(files, dict) else ""
            if todos_txt:
                st.markdown("### 本次 Todo（完整）")
                st.code(todos_txt[:20000], language="json")
                st.divider()
            render_debug_panel(files)

    st.session_state.chat_history.append({"role": "assistant", "kind": "text", "content": answer_text, "meta": meta})
