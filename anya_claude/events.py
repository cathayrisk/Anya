# -*- coding: utf-8 -*-
"""把 graph.stream(stream_mode=["messages","custom","updates"]) 的輸出轉成畫面事件。

純函式、無 streamlit 依賴——頁面（主執行緒）拿事件去畫，測試可以直接餵假 chunk。

Claude 的串流 chunk 是 Anthropic 原生內容區塊：text（可能帶 citations）、thinking、
server_tool_use（網路搜尋／讀網頁開始）、web_search_tool_result、web_fetch_tool_result。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage


@dataclass
class Ev:
    kind: str                       # text | thinking | tool_start | search_results | fetch_result | code_result
    node: str = ""                  # tool_call | tool_args | tool_result | citations | escalate | usage | reflect
                                    # | evidence | answer
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
    """只處理串流中的 AIMessageChunk 與工具結果（ToolMessage）。完整的 AIMessage（節點輸出被
    LangGraph 再送一次）略過，否則同一段文字會畫兩次——最終答案改從 updates 拿。"""
    if isinstance(chunk, ToolMessage):
        # 階段 2：deep agent 的自訂工具（氣象署、skills 的 read_file、think…）跑完
        text = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
        return [Ev("tool_result", node, data={"name": chunk.name or "", "status": chunk.status or "success",
                                              "text": text[:2000]})]
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
        elif t == "tool_use" and b.get("name"):
            # 自訂（client）工具開始：串流第一塊帶名稱，參數之後才以 input_json_delta 陸續送來
            evs.append(Ev("tool_call", node, data={"name": b["name"], "id": b.get("id") or ""}))
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


CUSTOM_KINDS = ("escalate", "usage", "reflect", "evidence", "widget", "rubric_start", "rubric", "calc")


def parse_item(item: tuple) -> list[Ev]:
    """graph.stream 的一筆：subgraphs=False 是 (mode, data)，subgraphs=True 是 (namespace, mode, data)。
    階段 2 的 deep agent 是節點裡的子圖，要 subgraphs=True 才拿得到它的逐字與工具事件（p2_stream_probe 實測）。"""
    if len(item) == 3:
        ns, mode, data = item
        # 第二層以下的子圖（例如寫完再檢查的 grader agent）：它的逐字和節點更新不是回答，不能畫進答案區。
        # custom 事件照收（工具、middleware 主動送的）。子代理目前關著，第二層沒有其他正當來源。
        if len(ns) >= 2 and mode in ("messages", "updates"):
            return []
        return parse_stream_item(mode, data)
    return parse_stream_item(*item)


def parse_stream_item(mode: str, data: Any) -> list[Ev]:
    if mode == "messages":
        chunk, meta = data
        return parse_message_chunk(chunk, (meta or {}).get("langgraph_node", ""))
    if mode == "custom" and isinstance(data, dict):
        kind = data.get("type")
        if kind in CUSTOM_KINDS:
            return [Ev(kind, data={k: v for k, v in data.items() if k != "type"})]
        return []
    if mode == "updates" and isinstance(data, dict):
        evs = []
        for node, upd in data.items():
            if not isinstance(upd, dict):
                continue
            if isinstance(upd.get("answer"), AIMessage):
                evs.append(Ev("answer", node, data={"message": upd["answer"],
                                                    "model": upd.get("model_used") or ""}))
            # deep agent 的模型節點跑完：工具參數這時才完整（串流時只拿得到名稱）——時間軸要用它寫「讀了哪個 skill」
            for m in upd.get("messages") or []:
                if isinstance(m, AIMessage) and m.tool_calls:
                    evs.append(Ev("tool_args", node, data={"calls": [
                        {"name": tc.get("name") or "", "args": tc.get("args") or {}, "id": tc.get("id") or ""}
                        for tc in m.tool_calls]}))
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
