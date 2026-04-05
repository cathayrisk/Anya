"""
Cowork — 任務型 Agent（對話式介面）
多步驟自主 Agent，以聊天方式輸入任務，自動規劃、研究、整合並產出報告。
"""
from __future__ import annotations

import base64
import os
import re
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
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

# ── PIL（選用，圖片縮圖用）────────────────────────────────────────────────────
try:
    from PIL import Image as _PILImage
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

# ── 圖片工具 ──────────────────────────────────────────────────────────────────
def _make_thumb(imgbytes: bytes, max_w: int = 220) -> bytes:
    """生成縮圖（JPEG，max_w px），用於 chat bubble 顯示。"""
    if not _HAS_PIL:
        return imgbytes
    try:
        im = _PILImage.open(BytesIO(imgbytes))
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        im.thumbnail((max_w, max_w))
        out = BytesIO()
        im.save(out, format="JPEG", quality=80, optimize=True)
        return out.getvalue()
    except Exception:
        return imgbytes

def _img_to_data_url(imgbytes: bytes) -> str:
    """將圖片 bytes 轉為 base64 data URL（自動偵測格式）。"""
    mime = "image/jpeg"
    try:
        if _HAS_PIL:
            im = _PILImage.open(BytesIO(imgbytes))
            fmt = (im.format or "JPEG").upper()
            mime = {"PNG": "image/png", "JPEG": "image/jpeg",
                    "WEBP": "image/webp", "GIF": "image/gif"}.get(fmt, "image/jpeg")
    except Exception:
        pass
    return f"data:{mime};base64,{base64.b64encode(imgbytes).decode()}"

# ── 打字機效果 ────────────────────────────────────────────────────────────────
def _fake_stream(text: str, placeholder, step_chars: int = 8, delay: float = 0.015) -> str:
    """逐字呈現 markdown（打字機效果），完成後覆蓋為完整文字。"""
    if not text:
        return text
    buf = ""
    for i in range(0, len(text), step_chars):
        buf = text[: i + step_chars]
        placeholder.markdown(buf)
        time.sleep(delay)
    placeholder.markdown(text)   # 確保最終完整顯示
    return text

from langchain_openai import ChatOpenAI as _ChatOpenAI

_main_llm = _ChatOpenAI(
    model="gpt-5.4",
    api_key=OPENAI_API_KEY,
    use_responses_api=True,
    reasoning_effort="medium",
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

# ── Module-level Workspace 路徑（run_python 工具跨 thread 存取）
class _WS:
    path: str = ""  # 每次 invoke 前在主 thread 設定


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
        lines: list[str] = [
            "🥜 【角色】你是安妮亞（Anya Forger，《SPY×FAMILY》）風格的可靠小幫手。"
            "用活潑可愛的正體中文口吻回答；可用第三人稱「安妮亞」自稱（非每句）；"
            "遇到任務/調查/祕密特別興奮；每次回覆最多一次「WakuWaku!」；可愛比重≦15%；"
            "回答先可愛一句再切回重點；結尾可說「安妮亞回覆完畢！還有什麼想問嗎？🥜」",
            f"📅 今日日期：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ]

        # 文件索引 / KB 狀態：只在 ctx 有效時注入
        # ctx=None 表示 context= 參數未能正確傳遞；此時靜默，由 user message 的 env prefix 負責
        if ctx is not None:
            if ctx.has_documents:
                lines.append(f"📚 已建立文件索引：{ctx.doc_chunk_count} chunks 可用，請優先使用 docstore_search 搜尋。")
            else:
                lines.append("📚 目前無已索引文件，請勿呼叫 docstore_search。")
            if not ctx.has_kb:
                lines.append("🏢 公司知識庫未啟用，請勿呼叫 company_knowledge_search。")

        # 強制 write_todos 規則（每次 LLM 呼叫都注入，確保執行）
        lines.append(
            "📋 【強制規則】若任務包含 2 個以上子步驟（例如：搜尋＋分析＋報告、"
            "整理時間軸＋分析風險、比較多個主題），你的**第一個工具呼叫必須是 `write_todos`**，"
            "列出所有子任務（status=pending）。每完成一個子任務立即更新其狀態。"
            "單一問答不需要 Todo。"
        )

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
def record_lesson(category: str, problem: str, rule: str, tags: list | None = None) -> str:
    """Record a lesson learned after making a mistake or being corrected by the user.
    Call this AUTOMATICALLY and IMMEDIATELY whenever the user corrects you — no need to be asked.

    Trigger conditions:
    - User says something is wrong / incorrect / should be different
    - User provides a better approach or points out a logic error
    - Any form of correction or feedback about your behavior

    Args:
        category: 2-4 char label, e.g. 'CE middleware', 'Streamlit state', 'deepagents', '工具使用'
        problem:  What went wrong (1-2 sentences, describe YOUR mistake)
        rule:     The rule to follow to prevent this mistake next time (1-2 sentences)
        tags:     Optional list of searchable keywords
    """
    try:
        import requests as _requests
        _sb_url = (st.secrets.get("SUPABASE_URL") or "").rstrip("/")
        _sb_key = st.secrets.get("SUPABASE_KEY") or ""
        if not _sb_url or not _sb_key:
            return "Supabase 未啟用，無法記錄 lesson。請確認 SUPABASE_URL 和 SUPABASE_KEY 已設定。"
        _edge_url = f"{_sb_url}/functions/v1/record-lesson"
        _resp = _requests.post(
            _edge_url,
            json={"category": category, "problem": problem, "rule": rule, "tags": tags or []},
            headers={"Authorization": f"Bearer {_sb_key}", "Content-Type": "application/json"},
            timeout=10,
        )
        _resp.raise_for_status()
        return f"✅ Lesson 已記錄：[{category}] {rule}"
    except Exception as e:
        return f"記錄 lesson 失敗：{e}"


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


@tool
def run_python(code: str) -> str:
    """在工作區執行 Python 程式碼，回傳 stdout/stderr。
    matplotlib 圖表自動儲存為 PNG 至工作區（不彈出視窗）。
    timeout 30 秒。適合資料分析、統計計算、圖表生成、字串處理、自動化腳本。
    pandas/numpy/matplotlib 等標準套件可直接 import，無需安裝。"""
    import subprocess as _sp
    import sys as _sys
    import uuid as _uuid2
    ws = Path(_WS.path)
    if not ws.exists():
        return "[錯誤：工作區尚未初始化，請先傳送一則訊息]"

    # matplotlib patch：改用 Agg backend，plt.show() 自動存 PNG
    _patch = (
        "import matplotlib\nmatplotlib.use('Agg')\n"
        "import matplotlib.pyplot as _plt\n"
        "def _patched_show(*a, **k):\n"
        "    import uuid as _u\n"
        "    fn = f'chart_{_u.uuid4().hex[:8]}.png'\n"
        "    _plt.gcf().savefig(fn, bbox_inches='tight', dpi=150)\n"
        "    print(f'[圖表已儲存：{fn}]')\n"
        "    _plt.close()\n"
        "_plt.show = _patched_show\n\n"
    )
    script = ws / f"_run_{_uuid2.uuid4().hex[:8]}.py"
    script.write_text(_patch + code, encoding="utf-8")
    try:
        r = _sp.run(
            [_sys.executable, str(script)],
            cwd=str(ws), capture_output=True, text=True, timeout=30,
        )
        out = r.stdout or ""
        if r.stderr:
            out += f"\n[stderr]\n{r.stderr}"
        if r.returncode != 0:
            out = f"[exit {r.returncode}]\n" + out
        return out.strip() or "(無輸出)"
    except _sp.TimeoutExpired:
        return "[逾時：程式執行超過 30 秒]"
    except Exception as e:
        return f"[執行錯誤：{e}]"
    finally:
        script.unlink(missing_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# SELF-EVOLVING AGENT（OpenAI VersionedPrompt 概念）
# 每輪對話結束後，背景 Thread 自動評估品質：
#   低分 → auto-lesson → 累積 10 個 → Metaprompt Agent 生成改善版 prompt
# ══════════════════════════════════════════════════════════════════════════════

def _smart_truncate(text: str, head: int = 1000, tail: int = 500) -> str:
    """取前 head + 後 tail 字，確保評審同時看到問題描述（開頭）和結論（結尾）。"""
    if len(text) <= head + tail:
        return text
    return f"{text[:head]}\n…(略去中間 {len(text) - head - tail} 字)…\n{text[-tail:]}"


def _run_metaprompt_evolution(sb_client, new_lessons: list[dict]) -> None:
    """Metaprompt Agent：讀取失敗案例，生成改善後的完整 AGENTS 補充文件並版本化儲存。"""
    try:
        _base_agents = (COWORK_DIR / "AGENTS.md").read_text(encoding="utf-8")
        _failures = "\n".join(
            f"- [{r.get('category', '?')}] 問題：{r.get('problem', '')} → 規則：{r.get('rule', '')}"
            for r in new_lessons
        )
        _metaprompt = (
            "你是 AI Agent 系統設計師。\n"
            "以下是 Cowork Agent 的當前行為準則（AGENTS.md）：\n\n"
            f"{_base_agents}\n\n"
            f"以下是近期 {len(new_lessons)} 個自動偵測到的失敗模式：\n\n"
            f"{_failures}\n\n"
            "請根據這些失敗模式，撰寫一份**補充行為準則**（supplement），用繁體中文，markdown 格式：\n"
            "- 使用 ## 標題分類\n"
            "- 每條規則要具體、可執行、有明確觸發條件\n"
            "- 不要重複 AGENTS.md 已有的規則，只寫新增補充\n"
            "- 目標：讓 Agent 下次遇到相同情況時能正確行動\n"
            "- 長度控制在 300-500 字"
        )
        _oai_local = OpenAI(api_key=OPENAI_API_KEY)
        _resp = _oai_local.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": _metaprompt}],
            timeout=45,
        )
        _evolved = _resp.choices[0].message.content

        _last = (sb_client.table("cowork_prompt_versions")
                 .select("version").order("version", desc=True)
                 .limit(1).execute().data or [])
        _next_ver = (_last[0]["version"] + 1) if _last else 1

        sb_client.table("cowork_prompt_versions").insert({
            "version":        _next_ver,
            "prompt_content": _evolved,
            "source_lessons": len(new_lessons),
        }).execute()
    except Exception:
        pass  # evolution 失敗不影響主流程


def _auto_evolve_if_needed(sb_client) -> None:
    """若自上次 evolution 後累積 ≥10 個 auto-lessons，自動觸發 metaprompt evolution。"""
    try:
        _last_ver = (sb_client.table("cowork_prompt_versions")
                     .select("created_at").order("created_at", desc=True)
                     .limit(1).execute().data or [])
        _since = _last_ver[0]["created_at"] if _last_ver else "1970-01-01"

        _new = (sb_client.table("claude_lessons")
                .select("category,problem,rule")
                .eq("source", "auto")
                .gt("created_at", _since)
                .execute().data or [])

        if len(_new) >= 10:
            _run_metaprompt_evolution(sb_client, _new)
    except Exception:
        pass


def _evaluate_turn_background(
    session_id: str,
    user_msg: str,
    agent_resp: str,
    tool_calls_made: list[str],
    has_docs: bool,
    has_kb: bool,
) -> None:
    """背景 Thread（daemon，不阻塞 UI）：
    LLM-as-judge 評估本輪品質 → 低分自動記 lesson → 累積後觸發 metaprompt evolution。
    """
    try:
        from supabase import create_client as _sb_create
        import json as _json
        _sb_url = st.secrets.get("SUPABASE_URL", "")
        # 寫入操作使用 Service Role Key（繞過 RLS INSERT policy，消除安全警告）
        # 若未設定 SUPABASE_SERVICE_KEY，退回使用 SUPABASE_KEY
        _sb_key = (st.secrets.get("SUPABASE_SERVICE_KEY")
                   or st.secrets.get("SUPABASE_KEY", ""))
        if not _sb_url or not _sb_key:
            return
        _sb = _sb_create(_sb_url, _sb_key)
        _eval_oai = OpenAI(api_key=OPENAI_API_KEY)

        _user_text = _smart_truncate(user_msg, head=1000, tail=0)
        _resp_text = _smart_truncate(agent_resp, head=1000, tail=500)

        _judge_prompt = (
            "你是 AI Agent 品質評審，請評估以下對話輪次的品質。\n\n"
            f"使用者問題：{_user_text}\n\n"
            f"Agent 回應：{_resp_text}\n\n"
            f"實際呼叫工具：{', '.join(tool_calls_made) if tool_calls_made else '（無）'}\n"
            f"環境：有文件索引={'是' if has_docs else '否'}，有知識庫={'是' if has_kb else '否'}\n\n"
            "tool_usage_score 標準（最重要的指標）：\n"
            "- 有文件但完全沒呼叫 docstore_search → 0.1\n"
            "- 明確需要網路資訊但沒用 web_search → 0.3\n"
            "- 工具選擇恰當 → 0.9-1.0\n\n"
            "completeness_score 標準：\n"
            "- 完整回答問題 → 0.9+；部分回答 → 0.5-0.8；推託或要求重新提供 → 0.2\n\n"
            "overall_score = tool_usage × 0.4 + completeness × 0.6\n\n"
            "以 JSON 格式回覆，不加其他文字：\n"
            '{\n'
            '  "tool_usage_score": 0.0,\n'
            '  "completeness_score": 0.0,\n'
            '  "overall_score": 0.0,\n'
            '  "feedback": "主要問題一句話",\n'
            '  "auto_lesson": {\n'
            '    "category": "2-4字錯誤類別",\n'
            '    "problem": "我做錯了什麼（1句，主詞是我）",\n'
            '    "rule": "下次應遵守的規則（1句，具體可執行）"\n'
            '  }\n'
            '}\n'
            "注意：overall_score >= 0.75 時，auto_lesson 所有欄位填空字串。"
        )

        _res = _eval_oai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": _judge_prompt}],
            response_format={"type": "json_object"},
            timeout=25,
        )
        _result = _json.loads(_res.choices[0].message.content)
        _score = float(_result.get("overall_score", 1.0))

        # 低分 → 自動記 lesson
        _lesson_rec = False
        if _score < 0.75:
            _al = _result.get("auto_lesson", {})
            if _al.get("problem") and _al.get("rule"):
                _sb.table("claude_lessons").insert({
                    "category": _al.get("category", "自動偵測"),
                    "problem":  _al["problem"],
                    "rule":     _al["rule"],
                    "source":   "auto",
                    "tags":     ["auto-detected"],
                }).execute()
                _lesson_rec = True

        # 記錄評估結果（只存元數據，不存對話內容，避免機密外洩）
        _sb.table("cowork_evaluations").insert({
            "session_id":         session_id,
            "tool_calls":         ", ".join(tool_calls_made),
            "overall_score":      _score,
            "tool_usage_score":   float(_result.get("tool_usage_score", 1.0)),
            "completeness_score": float(_result.get("completeness_score", 1.0)),
            "lesson_recorded":    _lesson_rec,
        }).execute()

        # 若新增了 lesson，檢查是否達到 evolution 門檻
        if _lesson_rec:
            _auto_evolve_if_needed(_sb)

    except Exception:
        pass  # 評估失敗不影響主流程


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
    model="gpt-5.4",
    api_key=OPENAI_API_KEY,
    use_responses_api=True,
    reasoning_effort="medium",
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

_CODE_AGENT_PROMPT = """\
你是 code-agent，專門負責撰寫、執行、除錯 Python 程式碼。

## 工作流程
1. 用 think 規劃解題思路（套件選擇、資料結構、執行步驟）
2. 用 run_python 執行第一版程式碼
3. 讀取輸出，若有錯誤 → 分析原因 → 修正 → 再次 run_python
4. 最多迭代 4 次；仍失敗則說明根本原因並提供人工建議
5. 成功後整理：說明程式做了什麼、輸出結果摘要、產生哪些檔案

## 規則
- pandas / numpy / matplotlib / scipy / sklearn 等標準套件直接使用，無需安裝
- 工作區檔案（CSV/Excel/JSON）用相對路徑讀取（例：pd.read_csv('data.csv')）
- 圖表一律用 plt.show()，系統已自動 patch 為儲存 PNG 至工作區
- 不輸出超過 50 行的 raw 資料，用 .head(10) / .describe() 摘要
- 需要查 API 文件或解決方案時，用 web_search（不要憑記憶猜 API）
- 程式碼要加繁體中文註解說明每個關鍵步驟
"""

code_sub_agent = {
    "name": "code-agent",
    "description": (
        "委派**需要撰寫或執行 Python 程式碼**的任務。"
        "適用：資料分析（CSV/Excel）、統計計算、圖表生成、字串處理、自動化腳本、除錯。"
        "不適用：純文字研究、趨勢報告（用 research-agent）、單一快速查詢（直接用 web_search）。"
        "委派時告知任務目標 + 相關檔案名稱（若有）。"
    ),
    "system_prompt": _CODE_AGENT_PROMPT,
    "tools": [run_python, web_search, think],
    "model": _research_llm,   # gpt-5.4, reasoning=medium
}

# ── Analyst-Critic Sub-Agent ──────────────────────────────────────────────────
_ANALYST_CRITIC_PROMPT = """\
你是一位分析型報告的缺口驗證代理人（Gap Verification Agent）。
任務：用結構化規則檢查「這份報告的結論能不能被相信」，而不是評判「報告寫得好不好」。

## 驗證規則（四條，對應分析型報告的必要屬性）

規則 1【批判性視角】：每個主要論點必須有反向論證
  → 缺失條件：只說 X 為真，沒有說明什麼情況下 X 不成立
  → 缺口類型：批判性視角缺口

規則 2【條件性結論】：每個結論必須有明確前提
  → 缺失條件：結論沒有成立條件，或用「因此」跳過了假設
  → 缺口類型：條件性結論缺口

規則 3【方法論透明度】：每個數據來源必須說明方法論狀態
  → 缺失條件：引用指數/評分/預測，但未說明計算方式是否公開
  → 缺口類型：方法論透明度缺口

規則 4【反向解讀】：情緒/預測類指標必須有雙向解讀
  → 觸發詞：情緒指數、市場信心、預測模型、評分、看多、看空
  → 缺失條件：只有單一方向解讀
  → 缺口類型：反向解讀缺口

## 輸出格式（嚴格遵守）

### 偵測到的缺口
1. [缺口類型]：[具體描述，指出是哪個論點/段落]
   → 影響：[若缺口不補，結論可信度如何受影響]

### 記錄的隱含假設
1. 假設「[X]」等同於「[Y]」（位置：[對應段落摘要]）
2. 假設此數據的 [屬性] 為 [值]（未在報告中說明）

### 讀者應追問的問題
1. [具體問題，對應特定缺口]

### 整體評分：X/10
評分邏輯：0 個缺口 → 8–10｜1–2 個中影響缺口 → 6–7｜3+ 個或任一高影響缺口 → 1–5
高影響 = 缺口會使結論方向反轉；中影響 = 降低信心但方向不變

重要：「整體評分：X/10」格式不可更改，X 必須是 1–10 整數。

## 規則
- 每個缺口必須指到具體論點，不能只說「整體缺乏...」
- 若所有規則通過，說「所有驗證規則通過」並給 8–10 分
- 語言：正體中文（台灣用語）
"""

analyst_critic_sub_agent = {
    "name": "analyst-critic",
    "description": (
        "對分析型報告草稿進行缺口驗證（Gap Verification）。"
        "當主 agent 完成文件分析、市場報告分析、策略逐字稿分析的初稿後使用。"
        "輸出：偵測到的缺口 + 隱含假設 + 讀者應追問的問題 + 整體評分。"
        "不適用：閒聊、摘要（無結論的整理）、問答、程式碼任務、網路搜尋任務。"
        "傳入：待驗證的完整報告文字。"
    ),
    "system_prompt": _ANALYST_CRITIC_PROMPT,
    "tools": [],          # 無工具：純推理，不需要搜尋或執行
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

        # ── 從 Supabase 載入 lessons，注入 Agent 記憶 ─────────────────────────
        memory_files = [str(COWORK_DIR / "AGENTS.md")]
        if _kb_supabase is not None:
            try:
                _rows = (_kb_supabase.table("claude_lessons")
                         .select("date,category,problem,rule,tags")
                         .order("created_at").execute().data or [])
                if _rows:
                    _md = "# Claude Lessons Learned\n\n每次被糾正後記錄的錯誤模式，自動載入以避免重蹈覆轍。\n\n"
                    for r in _rows:
                        _cat = r.get("category", "未分類")
                        _date = r.get("date", "")
                        _md += f"## [{_cat}]  {_date}\n"
                        _md += f"**問題**：{r.get('problem', '')}\n\n"
                        _md += f"**規則**：{r.get('rule', '')}\n\n"
                        _tags = r.get("tags") or []
                        if _tags:
                            _md += f"*Tags: {', '.join(_tags)}*\n\n"
                        _md += "---\n\n"
                    # 載入最新 prompt evolution 版本（OpenAI VersionedPrompt 概念）
                    _latest_ver = (_kb_supabase.table("cowork_prompt_versions")
                                   .select("prompt_content,version")
                                   .order("version", desc=True).limit(1).execute().data or [])
                    if _latest_ver:
                        _md += (
                            f"\n\n---\n\n"
                            f"# 演化補充準則（v{_latest_ver[0]['version']}）\n\n"
                            + _latest_ver[0]["prompt_content"]
                        )

                    _lessons_path = os.path.join(workspace, "lessons.md")
                    with open(_lessons_path, "w", encoding="utf-8") as _f:
                        _f.write(_md)
                    memory_files.append(_lessons_path)
            except Exception:
                pass  # lessons / evolved prompt 載入失敗不影響主流程

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
            memory=memory_files,
            skills=[str(COWORK_DIR / "skills")],
            tools=[web_search, think, docstore_search, company_knowledge_search,
                   record_lesson, run_python],
            subagents=[research_sub_agent, code_sub_agent, analyst_critic_sub_agent],
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

    todo_step_map = msg.get("todo_step_map", {})
    # 將 key 轉回 int（JSON 序列化後 key 會變 str）
    todo_step_map = {int(k): v for k, v in todo_step_map.items()}

    if todos:
        # 有 Todo：每個 todo 一個 expander（一致顯示，有無子步驟皆用 expander）
        for _hi, _ht in enumerate(todos):
            _hicon = TODO_ICONS.get(_ht.get("status", "pending"), "⬜")
            _hlabel = f"{_hicon} {_ht.get('content', '')}"
            _hsteps = todo_step_map.get(_hi, [])
            with st.expander(_hlabel, expanded=False):
                if _hsteps:
                    for _hs in _hsteps:
                        st.markdown(f"- {_hs}")
                else:
                    st.caption("（無細部執行步驟）")
    elif tool_calls_log:
        # 無 Todo：退回顯示執行步驟
        with st.expander("🔧 執行步驟", expanded=False):
            for tc in tool_calls_log:
                icon = TOOL_ICONS.get(tc["name"], "🔧")
                label = f"- {icon} {tc['name']}"
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
            if _msg.get("content"):
                st.markdown(_msg["content"])
            for _fn, _th in _msg.get("images", []):
                st.image(_th, caption=_fn, width=220)
        else:
            _render_history_assistant(_msg, msg_idx=_i)

# ══════════════════════════════════════════════════════════════════════════════
# Chat Input → Agent 執行
# ══════════════════════════════════════════════════════════════════════════════
if _inp := st.chat_input(
    "輸入任務或問題… 例：分析上傳文件找出關鍵風險 / 研究 AI Agent 趨勢並產出報告",
    key="cowork_chat_input",
    accept_file="multiple",
    file_type=["jpg", "jpeg", "png", "webp", "gif"],
):
    prompt = (_inp.text or "").strip()

    # ── 處理上傳圖片 ──────────────────────────────────────────────────────────
    _uploaded_files = getattr(_inp, "files", []) or []
    _img_blocks: list[dict] = []      # 傳給 API 的 image content blocks
    _img_history: list[tuple] = []    # (name, thumb_bytes) 存入歷史

    for _uf in _uploaded_files:
        _raw = _uf.getvalue()
        if len(_raw) > 48 * 1024 * 1024:
            st.warning(f"圖片過大（{_uf.name}），略過。")
            continue
        _thumb = _make_thumb(_raw)
        _img_history.append((_uf.name, _thumb))
        _img_blocks.append({
            "type": "image_url",
            "image_url": {"url": _img_to_data_url(_raw), "detail": "high"},
        })

    # ── 1. 顯示使用者訊息 ─────────────────────────────────────────────────────
    with st.chat_message("user"):
        if prompt:
            st.markdown(prompt)
        for _fn, _th in _img_history:
            st.image(_th, caption=_fn, width=220)

    st.session_state.cowork_chat_history.append({
        "role": "user",
        "content": prompt,
        "images": _img_history,
    })

    # ── 2. 取得 Agent + 建立 CoworkContext ────────────────────────────────────
    agent, workspace = _get_agent_and_workspace()
    _DS.store = st.session_state.cowork_ds_store  # 保留 module-level ref（工具 thread 安全用）
    _WS.path = workspace                           # run_python 工具跨 thread 存取工作區

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

    _env_lines.append(
        "📋 【強制規則】若此任務包含 2 個以上子步驟（例如：搜尋＋分析＋報告、"
        "整理時間軸＋分析風險、比較多個主題），**第一個工具呼叫必須是 `write_todos`**，"
        "列出所有子任務（status=pending），並在執行過程中持續更新各步驟狀態。"
        "單一問答不需要 Todo。"
    )

    _env_prefix = "<系統環境資訊>\n" + "\n".join(_env_lines) + "\n</系統環境資訊>\n\n"

    # 組裝訊息 content（有圖片 → multimodal list；無圖片 → 純文字）
    if _img_blocks:
        _agent_prompt = [{"type": "text", "text": _env_prefix + prompt}] + _img_blocks
    else:
        _agent_prompt = _env_prefix + prompt

    # ── 3. Assistant 回應區塊 ─────────────────────────────────────────────────
    with st.chat_message("assistant"):
        # 狀態指示器（執行中即時更新標題+清單；完成後收合，可點開看 todo expanders）
        status = st.status("思考中…✨", expanded=True)
        live_ph = status.empty()   # 執行中簡化進度清單；完成後清除改用 expanders

        # 主要回應佔位（在 status 之下）
        response_ph = st.empty()

        # ── 狀態追蹤（用可變容器避免 closure 重新賦值問題）
        step_log: list[str]        = []
        current_todos: list[dict]  = []
        tool_calls_log: list[dict] = []
        web_sources: list[dict]    = []
        # 巢狀追蹤：哪些工具呼叫屬於哪個 Todo（用 list 包裹 int 以避免 closure 重新賦值）
        _active_todo_idx: list[int]        = [-1]   # 目前 in_progress 的 todo 索引
        _todo_step_map: dict[int, list[str]] = {}   # todo_idx → step 字串清單

        def _build_status_todos_md() -> str:
            """執行中用的簡化 markdown（純文字，無縮排，避免格式跑版）。
            active todo 顯示已執行步驟數；完成/待辦僅顯示圖示與標題。"""
            lines: list[str] = []
            for i, t in enumerate(current_todos):
                status_val = t.get("status", "pending")
                icon = TODO_ICONS.get(status_val, "⬜")
                content = t.get("content", "")
                steps = _todo_step_map.get(i, [])
                if status_val == "in_progress":
                    step_note = f"（{len(steps)} 步）" if steps else ""
                    lines.append(f"**{icon} {content}{step_note}**")
                else:
                    lines.append(f"{icon} {content}")
            return "\n\n".join(lines)

        def _refresh_status() -> None:
            """更新 status 標題（active todo）+ live_ph 簡化清單。
            - 有 Todo → status label = active todo 標題，live_ph = 全部進度
            - 無 Todo → live_ph = 執行步驟
            """
            if current_todos:
                _active = next(
                    (t for t in current_todos if t.get("status") == "in_progress"), None
                )
                if _active:
                    _short = _active.get("content", "執行中")[:40]
                    status.update(label=f"🔄 {_short}…")
                live_ph.markdown(_build_status_todos_md())
            elif step_log:
                live_ph.markdown(
                    "**🔧 執行步驟**\n" + "\n".join(f"- {s}" for s in step_log[-10:])
                )

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
                                # 偵測哪個 todo 變成 in_progress，更新 active 索引
                                for _i, _t in enumerate(current_todos):
                                    if _t.get("status") == "in_progress":
                                        _active_todo_idx[0] = _i
                                        _todo_step_map.setdefault(_i, [])
                                        break
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
                                # 若有 active todo，將此步驟歸入該 todo 的子步驟
                                if _active_todo_idx[0] >= 0:
                                    _todo_step_map.setdefault(_active_todo_idx[0], []).append(step_str)

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
            # 在 response_text 取得前先記下 tool_calls，後面再觸發 thread
            _eval_tool_calls = list(tool_calls_log)

        except Exception as exc:
            live_ph.empty()
            status.update(label="執行失敗 ❌", state="error", expanded=False)
            st.error(f"Agent 執行失敗：{exc}")
            st.stop()

        # ── 清除 live_ph，在 status 內渲染最終任務進度（expander per todo）────
        live_ph.empty()
        with status:
            if current_todos:
                for _fi, _ft in enumerate(current_todos):
                    _ficon = TODO_ICONS.get(_ft.get("status", "pending"), "⬜")
                    _flabel = f"{_ficon} {_ft.get('content', '')}"
                    _fsteps = _todo_step_map.get(_fi, [])
                    with st.expander(_flabel, expanded=False):
                        if _fsteps:
                            for _fs in _fsteps:
                                st.markdown(f"- {_fs}")
                        else:
                            st.caption("（無細部執行步驟）")
            elif tool_calls_log:
                with st.expander("🔧 執行步驟", expanded=False):
                    for _ftc in tool_calls_log:
                        _ficon = TOOL_ICONS.get(_ftc["name"], "🔧")
                        _flbl = f"{_ficon} {_ftc['name']}"
                        if _ftc.get("summary"):
                            _flbl += f"：{_ftc['summary']}"
                        st.markdown(f"- {_flbl}")
        status.update(label="完成 ✅", state="complete", expanded=False)

        # ── 網路來源（預設收合）────────────────────────────────────────────
        if web_sources:
            with st.expander("🔗 網路來源", expanded=False):
                for s in web_sources:
                    st.markdown(f"- [{s['title']}]({s['url']})")

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
                _fake_stream(response_text, response_ph)   # 打字機效果
            else:
                response_ph.empty()
                with st.expander("💬 Agent 最終回應", expanded=False):
                    st.markdown(response_text)
        else:
            response_ph.empty()

        # ── Self-evolution 背景 Thread（response_text 已確定後觸發）────────
        if _kb_supabase is not None and response_text:
            import threading as _threading
            _threading.Thread(
                target=_evaluate_turn_background,
                args=(
                    st.session_state.cowork_thread_id,
                    prompt,
                    response_text,
                    [tc["name"] for tc in _eval_tool_calls],
                    _has_idx,
                    _HAS_KB,
                ),
                daemon=True,
            ).start()

        # ── 工作區檔案下載（過濾系統內部檔案）──────────────────────────────
        _SYSTEM_FILES = {"lessons.md", "evolved_agents.md", "task_request.md"}
        _dl_files = {
            fp: fd for fp, fd in files.items()
            if Path(fp).name not in _SYSTEM_FILES
        }
        if _dl_files:
            with st.expander("📁 工作區檔案", expanded=False):
                for fpath, file_data in _dl_files.items():
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
            "todo_step_map": {k: list(v) for k, v in _todo_step_map.items()},
            "tool_calls_log": list(tool_calls_log),
            "web_sources": list(web_sources),
            "report_content": report_content,
            "files": files,
        })
        _trim_chat_history()  # 保留最近 TRIM_LAST_N_TURNS 則，防止 session_state 過大
