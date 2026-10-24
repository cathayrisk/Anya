# -*- coding: utf-8 -*-
"""一個回合的 LangGraph 圖：Haiku（一般）→（需要時）升級 → Opus（深思）。

⚠️ 本檔與它呼叫的一切都**不可碰 streamlit**：頁面用 stream_mode="messages" 消費這張圖時，
節點跑在背景執行緒（spike T1 實測，v3 也一樣），在這裡呼叫 st.* 會壞或靜默失效。
節點要讓畫面知道的事（升級、用量）一律用 get_stream_writer() 送 custom 事件。

第 1 階段：Opus 只有 server tools（網路搜尋、讀網頁）。
第 2 階段：opus 節點改成呼叫 deepagents 的 deep agent（anya_claude/deep.py）——它是節點裡的子圖，
頁面要用 subgraphs=True 串流才拿得到它的逐字與工具事件（tools/claude_spike/p2_stream_probe.py 實測）。
"""
from __future__ import annotations

import operator
import queue
import threading
import time
from typing import Annotated, Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from typing_extensions import NotRequired, TypedDict

from . import config as CFG
from . import prompts as P
from .deep_mw import final_answer, research_not_started, web_search_count
from .tools import calculate, run_calculate

CACHE = {"type": "ephemeral", "ttl": "1h"}   # 同事常隔超過 5 分鐘才問下一題，5m 快取會一直失效


@tool
def escalate_to_deep(reason: str, socratic: bool = False, research: bool = False) -> str:
    """把這題交給深思模式（Opus）。符合升級規則時呼叫，呼叫前不要先寫任何回答。

    Args:
        reason: 一句話說明為什麼要升級。
        socratic: 使用者卡住、需要被引導思考（而不是要答案）時設為 true。
        research: 使用者要一份研究報告、深度研究或文獻回顧時設為 true（會派研究員分頭上網查）。
    """
    return "已轉交深思模式"


class TurnState(TypedDict):
    messages: list[BaseMessage]          # 歷史＋本回合使用者訊息（不含系統提示）
    route: str                           # "fast" | "deep"
    socratic: bool
    research: NotRequired[bool]          # 深度研究：主持人派研究員（階段 4）
    context_block: NotRequired[str]      # 災防 prefetch 的官方資料；空字串＝沒有
    allow_escalate: NotRequired[bool]    # False＝使用者指定「改用一般模式重答」，Haiku 不能自己升級
    escalate_reason: NotRequired[str]
    model_used: NotRequired[str]
    answer: NotRequired[AIMessage]
    calls: Annotated[list[dict], operator.add]


def _models() -> dict[str, Any]:
    # 每次建圖時建一次；ChatAnthropic 本身可跨執行緒共用
    haiku = ChatAnthropic(model=CFG.HAIKU, max_tokens=CFG.HAIKU_MAX_TOKENS, effort=CFG.HAIKU_EFFORT,
                          thinking=CFG.HAIKU_THINKING)
    return {
        "haiku": haiku.bind_tools([escalate_to_deep, calculate, *CFG.HAIKU_SERVER_TOOLS]),
        "haiku_solo": haiku.bind_tools([calculate, *CFG.HAIKU_SERVER_TOOLS]),
    }


def _deep_agents() -> dict[str, Any]:
    from . import deep   # 延遲 import：deepagents 很重，只有建圖時才載入
    return {"assist": deep.build_deep_agent(socratic=False), "socratic": deep.build_deep_agent(socratic=True),
            "research": deep.build_research_agent()}


def _deep_files() -> dict:
    from . import deep
    return dict(deep.skill_files())


def turn_note(state: TurnState, now=None, lessons_block: str | None = None) -> str:
    """deep agent 每回合的系統訊息：日期＋長期記憶＋災防資料＋一般模式轉交的理由（常帶技能名稱）。
    都會變，不放進固定系統提示（快取才不會失效）。"""
    if lessons_block is None:
        from . import lessons
        lessons_block = lessons.block()
    reason = (state.get("escalate_reason") or "").strip()
    parts = (P.today_line(now), lessons_block, (state.get("context_block") or "").strip(),
             f"一般模式轉交這題的理由：{reason}" if reason else "")
    return "\n\n".join(x for x in parts if x)


NO_ESCALATE_NOTE = "使用者指定這一題用一般模式回答：這回合不能升級，請直接盡力完整回答。"


def compose(system: str, state: TurnState, *, note: str = "") -> list[BaseMessage]:
    """系統提示（快取斷點）＋歷史＋本回合；有災防資料就以對話中的系統訊息附在最後。

    Haiku 5.5／Opus 5.5 都支援對話中的 role=system（必須接在 user 後面、且是最後一則），
    這是「營運方指示」的正式管道，比舊版偽造一問一答乾淨，也不會被當成使用者的話。
    """
    msgs: list[BaseMessage] = [SystemMessage([{"type": "text", "text": system, "cache_control": CACHE}])]
    msgs += list(state["messages"])
    block = "\n\n".join(x for x in ((state.get("context_block") or "").strip(), note) if x)
    if block:
        msgs.append(SystemMessage(block))
    return msgs


def _invoke(model, msgs: list[BaseMessage], *, label: str, model_id: str) -> tuple[AIMessage, list[dict]]:
    """呼叫一次；server tool 迴圈太長回 pause_turn 時，把回應接回去續跑（最多 N 次）。"""
    writer = get_stream_writer()
    calls: list[dict] = []
    parts: list[AIMessage] = []
    convo = list(msgs)
    for i in range(CFG.MAX_PAUSE_CONTINUES + 1):
        resp = model.invoke(convo, cache_control=CACHE)
        parts.append(resp)
        rec = {"label": label if i == 0 else f"{label} 續跑{i}", "model": model_id,
               "usage": resp.usage_metadata or {}, "web": web_search_count(resp)}
        calls.append(rec)
        writer({"type": "usage", **rec})
        if (resp.response_metadata or {}).get("stop_reason") != "pause_turn":
            break
        convo = convo + [resp]
    return _merge(parts), calls


def _merge(parts: list[AIMessage], *, tool_calls=None) -> AIMessage:
    """幾段回應接成一則（pause_turn 續跑、Haiku 呼叫 calculate 前後）。tool_calls 預設沿用最後一段。"""
    if len(parts) == 1 and tool_calls is None:
        return parts[0]
    content: list = []
    for p in parts:
        content += p.content if isinstance(p.content, list) else [{"type": "text", "text": str(p.content)}]
    return AIMessage(content=content, tool_calls=parts[-1].tool_calls if tool_calls is None else tool_calls,
                     response_metadata=parts[-1].response_metadata)


CALC_ROUNDS = 3   # Haiku 一回合最多呼叫幾次 calculate（每次多一輪模型呼叫，約 +1 秒、不到 $0.001）


def build_graph(*, deep_agents: dict[str, Any] | None = None, deep_files=None):
    """deep_agents／deep_files 給測試注入假的；正式使用時建 deepagents 的兩個 agent（一般／提問引導）。"""
    models = _models()
    agents = deep_agents if deep_agents is not None else _deep_agents()
    files_for = deep_files or _deep_files

    def haiku_node(state: TurnState):
        """Haiku 回答；要算數字時呼叫 calculate（在這裡執行、把結果接回去再問一次），要升級就交給 opus。"""
        solo = state.get("allow_escalate") is False
        model = models["haiku_solo" if solo else "haiku"]
        convo = compose(P.haiku_system(), state, note=NO_ESCALATE_NOTE if solo else "")
        parts: list[AIMessage] = []
        calls: list[dict] = []
        for i in range(CALC_ROUNDS + 1):
            resp, more = _invoke(model, convo, label="Haiku 一般" if i == 0 else f"Haiku 計算後{i}",
                                 model_id=CFG.HAIKU)
            parts.append(resp)
            calls += more
            esc = [tc for tc in (resp.tool_calls or []) if tc.get("name") == "escalate_to_deep"]
            if esc:
                args = esc[0].get("args") or {}
                socratic = bool(state.get("socratic")) or bool(args.get("socratic"))
                research = not socratic and bool(args.get("research"))
                reason = str(args.get("reason") or "").strip()[:200]
                get_stream_writer()({"type": "escalate", "reason": reason, "socratic": socratic, "research": research})
                return Command(goto="opus", update={"escalate_reason": reason, "socratic": socratic,
                                                    "research": research, "calls": calls})
            todo = [tc for tc in (resp.tool_calls or []) if tc.get("name") == "calculate"]
            if not todo or i == CALC_ROUNDS:
                break
            convo = convo + [resp] + [
                ToolMessage(content=run_calculate((tc.get("args") or {}).get("expressions") or []),
                            tool_call_id=tc.get("id") or "", name="calculate") for tc in todo]
        return Command(goto=END, update={"answer": _merge(parts, tool_calls=[]), "model_used": CFG.HAIKU,
                                         "calls": calls})

    def opus_node(state: TurnState):
        # deep agent 是子圖：它的逐字、工具、用量事件（UsageEventsMiddleware）都從子圖串出去
        socratic = bool(state.get("socratic"))
        research = not socratic and bool(state.get("research"))
        agent = agents["socratic"] if socratic else agents.get("research", agents["assist"]) if research \
            else agents["assist"]
        msgs = list(state["messages"]) + [SystemMessage(turn_note(state))]
        inp = {"messages": msgs, "files": files_for()}
        if not socratic and not research:   # 研究流程由魔鬼代言人審查，不再跑通用的寫完再檢查
            inp["rubric"] = P.DEEP_RUBRIC   # 寫完再檢查（RubricMiddleware 看到 rubric 才啟動）
        out = agent.invoke(inp)
        if research and research_not_started(out.get("messages") or []):
            # 2026-10-24 線上偶發：主持人沒派研究員、只吐出幾行簡體字；同一題重問就正常。自動重跑一次（失敗那次約 $0.05）
            get_stream_writer()({"type": "retry", "reason": "第一次沒有派出研究員、回答很短，重新開始一次"})
            out = agent.invoke(inp)
        try:
            from . import lessons
            lessons.maybe_distill(out.get("messages") or [])   # 卡關後又解決 → 背景萃取搜尋教訓
        except Exception:
            pass
        answer = final_answer(out.get("messages") or [])
        return {"answer": answer or AIMessage(content=""), "model_used": CFG.OPUS}

    g = StateGraph(TurnState)
    g.add_node("haiku", haiku_node, destinations=("opus", END))
    g.add_node("opus", opus_node)
    g.add_conditional_edges(START, lambda s: "opus" if s.get("route") == "deep" else "haiku",
                            {"opus": "opus", "haiku": "haiku"})
    g.add_edge("opus", END)
    return g.compile()


STREAM_MODES = ["messages", "custom", "updates"]


def stream_in_thread(graph, state: TurnState) -> tuple[queue.Queue, threading.Event]:
    """在背景執行緒跑 graph.stream，項目丟進 queue：("item", (產生時間, (namespace, mode, data))) /
    ("error", exc) / ("end", None)。

    為什麼：Streamlit 只在 st.* 呼叫時檢查「停止」。直接在主執行緒 for 迴圈等 graph.stream，Opus
    默默跑程式、搜尋時沒有事件，主執行緒卡在等下一筆，按停止要等下一個事件才生效（實測晚了 70 秒，
    這段時間輸入框一直鎖著）。改成主執行緒用 queue.get(timeout) 等，空檔時自己碰一下畫面。
    halt.set() 之後，背景執行緒在下一筆項目時停止讀取（已送出的 API 呼叫會自己跑完）。
    產生時間：時間軸的秒數照它算。斷線接續時，已收到的事件一口氣重播、斷線期間堆著的事件一口氣讀出來，
    用讀到的時間算會全擠在同一刻（2026-10-24 線上：占星題接續後出現「想了 0.0 秒」）。
    """
    q: queue.Queue = queue.Queue()
    halt = threading.Event()

    def work():
        try:
            # subgraphs=True：deep agent（opus 節點裡的子圖）的事件才串得出來；每筆變成 (namespace, mode, data)
            for item in graph.stream(state, stream_mode=STREAM_MODES, subgraphs=True):
                if halt.is_set():
                    break
                q.put(("item", (time.time(), item)))
        except BaseException as e:   # noqa: BLE001 — 交給主執行緒原樣處理
            q.put(("error", e))
        finally:
            q.put(("end", None))

    threading.Thread(target=work, name="anya-graph-stream", daemon=True).start()
    return q, halt


def initial_state(messages: list[BaseMessage], *, route: str, socratic: bool,
                  context_block: str = "", allow_escalate: bool = True, research: bool = False) -> TurnState:
    return {"messages": messages, "route": route, "socratic": socratic, "research": research,
            "context_block": context_block, "allow_escalate": allow_escalate, "calls": []}


def user_message(text: str, images: list[dict] | None = None) -> HumanMessage:
    """images：[{"mime": "image/png", "b64": "..."}]。圖片放在文字前面（Anthropic 建議的順序）。"""
    if not images:
        return HumanMessage(text)
    blocks: list[dict] = [{"type": "image", "source": {"type": "base64", "media_type": im["mime"],
                                                       "data": im["b64"]}} for im in images]
    blocks.append({"type": "text", "text": text or "（請看圖）"})
    return HumanMessage(blocks)
