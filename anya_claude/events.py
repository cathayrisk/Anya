# -*- coding: utf-8 -*-
"""把 graph.stream(stream_mode=["messages","custom","updates"]) 的輸出轉成畫面事件。

純函式、無 streamlit 依賴——頁面（主執行緒）拿事件去畫，測試可以直接餵假 chunk。

Claude 的串流 chunk 是 Anthropic 原生內容區塊：text（可能帶 citations）、thinking、
server_tool_use（網路搜尋／讀網頁開始）、web_search_tool_result、web_fetch_tool_result。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk


@dataclass
class Ev:
    kind: str                       # text | thinking | tool_start | search_results | fetch_result
    node: str = ""                  # code_result | citations | escalate | usage | answer
    text: str = ""
    data: dict = field(default_factory=dict)


def _citations(block: dict) -> list[dict]:
    out = []
    for c in block.get("citations") or []:
        if isinstance(c, dict):
            out.append({"url": c.get("url") or "", "title": c.get("title") or "",
                        "cited_text": c.get("cited_text") or "", "type": c.get("type") or ""})
    return out


def parse_message_chunk(chunk: Any, node: str) -> list[Ev]:
    """只處理串流中的 AIMessageChunk。完整的 AIMessage（節點輸出被 LangGraph 再送一次）略過，
    否則同一段文字會畫兩次——最終答案改從 updates 拿。"""
    if not isinstance(chunk, AIMessageChunk):
        return []
    content = chunk.content
    if isinstance(content, str):
        return [Ev("text", node, text=content)] if content else []
    evs: list[Ev] = []
    for b in content or []:
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t == "text":
            if b.get("text"):
                evs.append(Ev("text", node, text=b["text"]))
            cites = _citations(b)
            if cites:
                evs.append(Ev("citations", node, data={"items": cites}))
        elif t in ("thinking", "reasoning"):
            s = b.get("thinking") or b.get("reasoning") or ""
            if s:
                evs.append(Ev("thinking", node, text=s))
        elif t == "server_tool_use":
            evs.append(Ev("tool_start", node, data={"name": b.get("name") or "", "input": b.get("input") or {}}))
        elif t == "web_search_tool_result":
            evs.append(Ev("search_results", node, data=_search_payload(b.get("content"))))
        elif t == "web_fetch_tool_result":
            evs.append(Ev("fetch_result", node, data=_fetch_payload(b.get("content"))))
        elif t in _CODE_RESULT_TYPES:
            # Opus 的 web_search_20260209 會用 code_execution 包住搜尋、篩選結果（2026-10-23 實測）
            evs.append(Ev("code_result", node))
    return evs


_CODE_RESULT_TYPES = ("code_execution_tool_result", "bash_code_execution_tool_result",
                      "text_editor_code_execution_tool_result")


def _search_payload(content: Any) -> dict:
    """成功時 content 是結果 list；失敗時是一個 error 物件（server tool 失敗不會拋例外）。"""
    if isinstance(content, list):
        res = [{"url": r.get("url") or "", "title": r.get("title") or ""}
               for r in content if isinstance(r, dict) and r.get("url")]
        return {"status": "ok" if res else "empty", "results": res}
    code = content.get("error_code") if isinstance(content, dict) else None
    return {"status": "error", "results": [], "error": code or "unknown"}


def _fetch_payload(content: Any) -> dict:
    if isinstance(content, dict) and content.get("type") == "web_fetch_result":
        return {"status": "ok", "url": content.get("url") or ""}
    code = content.get("error_code") if isinstance(content, dict) else None
    return {"status": "error", "url": "", "error": code or "unknown"}


def parse_stream_item(mode: str, data: Any) -> list[Ev]:
    if mode == "messages":
        chunk, meta = data
        return parse_message_chunk(chunk, (meta or {}).get("langgraph_node", ""))
    if mode == "custom" and isinstance(data, dict):
        kind = data.get("type")
        if kind in ("escalate", "usage"):
            return [Ev(kind, data={k: v for k, v in data.items() if k != "type"})]
        return []
    if mode == "updates" and isinstance(data, dict):
        evs = []
        for node, upd in data.items():
            if isinstance(upd, dict) and isinstance(upd.get("answer"), AIMessage):
                evs.append(Ev("answer", node, data={"message": upd["answer"],
                                                    "model": upd.get("model_used") or ""}))
        return evs
    return []


def paragraph_break(prev: str) -> str:
    """接在 prev 後面、讓下一段另起一段要補的換行（已經有換行就補到兩個為止）。"""
    if not prev.strip():
        return ""
    if prev.endswith("\n\n"):
        return ""
    return "\n" if prev.endswith("\n") else "\n\n"


def answer_text(msg: AIMessage | None) -> str:
    """把最終 AIMessage 的文字區塊接起來（server tool、thinking 區塊不算）。

    中間夾著工具區塊（例如「我去查一下」→ 搜尋 → 回答）時另起一段；2026-10-23 線上測試發現原本直接相接，
    搜尋前那句和回答黏成同一行。連續的文字區塊（引用會把一段切成好幾塊）照舊直接接起來。
    """
    if msg is None:
        return ""
    if isinstance(msg.content, str):
        return msg.content
    out, gap = "", False
    for b in msg.content:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text":
            if gap:
                out += paragraph_break(out)
            out += b.get("text", "")
            gap = False
        else:
            gap = True
    return out


def answer_citations(msg: AIMessage | None) -> list[dict]:
    if msg is None or isinstance(msg.content, str):
        return []
    out = []
    for b in msg.content:
        if isinstance(b, dict) and b.get("type") == "text":
            out += _citations(b)
    return out
