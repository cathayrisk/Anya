# -*- coding: utf-8 -*-
"""階段 2：深思模式（deepagents）用的 middleware。純 Python，無 streamlit 依賴（跑在背景執行緒）。

階段 1 的 opus_node 自己處理 pause_turn、自己送用量事件；換成 deepagents 之後，這兩件事改由
middleware 在 agent 的模型節點前後做：
- PauseTurnMiddleware：server tool（搜尋、讀網頁、程式）跑太久時 Claude 回 stop_reason=pause_turn，
  回答還沒寫完。agent 預設看到「沒有 tool_calls」就結束 → 會把半截回答當成最終答案。這裡改成跳回模型續跑。
- UsageEventsMiddleware：每次模型呼叫完送一個 custom 事件（頁面算錢、dev 面板），和階段 1 的格式一樣。
"""
from __future__ import annotations

import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.config import get_stream_writer

from . import config as CFG
from .docs import has_shelf, is_shelf


def stop_reason(msg: Any) -> str:
    return str((getattr(msg, "response_metadata", None) or {}).get("stop_reason") or "")


def web_search_count(msg: AIMessage) -> int:
    meta = getattr(msg, "response_metadata", {}) or {}
    stu = ((meta.get("usage") or {}).get("server_tool_use") or {})
    n = stu.get("web_search_requests") if isinstance(stu, dict) else getattr(stu, "web_search_requests", 0)
    if n:
        return int(n)
    blocks = msg.content if isinstance(msg.content, list) else []
    return sum(1 for b in blocks if isinstance(b, dict) and b.get("type") == "server_tool_use"
               and b.get("name") == "web_search")


def final_answer(messages: list[BaseMessage]) -> AIMessage | None:
    """這回合的最終回答：通常是最後一段連續的 AIMessage。萬一最後一則不是回答（例如寫完再檢查的回饋
    沒被處理到），退回最後一則有文字的 AIMessage——寧可交出第一版，也不要交出空白。"""
    tail = merge_trailing_ai(messages)
    if tail is not None:
        return tail
    for m in reversed(messages or []):
        if isinstance(m, HumanMessage) and m.additional_kwargs.get("lc_source") is None:
            break   # 不往上一回合找
        if isinstance(m, AIMessage) and (m.text if isinstance(m.text, str) else "").strip():
            return m
    return None


def merge_trailing_ai(messages: list[BaseMessage]) -> AIMessage | None:
    """最後一段連續的 AIMessage 併成一則（pause_turn 續跑會留下好幾則，內容要接起來才是完整回答）。"""
    tail: list[AIMessage] = []
    for m in reversed(messages or []):
        if not isinstance(m, AIMessage):
            break
        tail.append(m)
    if not tail:
        return None
    tail.reverse()
    if len(tail) == 1:
        return tail[0]
    content: list = []
    for p in tail:
        content += p.content if isinstance(p.content, list) else [{"type": "text", "text": str(p.content)}]
    return AIMessage(content=content, tool_calls=tail[-1].tool_calls, response_metadata=tail[-1].response_metadata)


class PauseTurnMiddleware(AgentMiddleware):
    def __init__(self, max_continues: int = CFG.MAX_PAUSE_CONTINUES):
        super().__init__()
        self.max_continues = max_continues

    @hook_config(can_jump_to=["model"])
    def after_model(self, state, runtime) -> dict[str, Any] | None:
        paused = 0
        for m in reversed(state.get("messages") or []):
            if isinstance(m, AIMessage) and stop_reason(m) == "pause_turn":
                paused += 1
            else:
                break
        if 0 < paused <= self.max_continues:
            return {"jump_to": "model"}
        return None


# 這個斷點只給同一回合的下一次呼叫用（下一回合的歷史是純文字重建），5m 就夠，寫入比 1h 便宜
CACHE_TURN = {"type": "ephemeral", "ttl": "5m"}


def _with_cache_control(msg: BaseMessage) -> BaseMessage:
    """在訊息最後一個內容區塊加快取斷點（tool_result 的子區塊會被 langchain-anthropic 提到 tool_result 上）。"""
    content = msg.content
    if isinstance(content, str):
        if not content:
            return msg
        blocks = [{"type": "text", "text": content, "cache_control": CACHE_TURN}]
    elif isinstance(content, list) and content and isinstance(content[-1], dict) \
            and content[-1].get("type") in ("text", "image", "document", "tool_result"):
        if content[-1].get("cache_control"):
            return msg
        blocks = list(content[:-1]) + [{**content[-1], "cache_control": CACHE_TURN}]
    else:
        return msg
    return msg.model_copy(update={"content": blocks})


class PrevRequestBreakpointMiddleware(AgentMiddleware):
    """在「上一次模型呼叫的最後一則輸入」打一個明確的快取斷點。

    2026-10-23 實測：一次回應裡搜尋好幾次（搜尋結果、文字一塊塊交錯）會超過快取往回找的 20 個位置，
    下一次呼叫的自動斷點找不到上一次寫的快取 → 對話部分整段重寫（地震題第三次呼叫只讀到 11K、重寫 28K）。
    官方建議：把斷點放在上一回合最後一個快取位置附近。上一次呼叫的最後輸入＝最新那則 AIMessage 的前一則。
    斷點總數：工具、系統提示、自動（最後一塊）＋這一個＝4，剛好在上限內。
    """

    def wrap_model_call(self, request, handler):
        msgs = list(request.messages)
        last_ai = next((i for i in range(len(msgs) - 1, -1, -1) if isinstance(msgs[i], AIMessage)), None)
        if not last_ai or isinstance(msgs[last_ai - 1], SystemMessage):
            return handler(request)        # 第一次呼叫，或前一則是系統訊息（還在 20 格內，自動斷點找得到）
        tagged = _with_cache_control(msgs[last_ai - 1])
        if tagged is msgs[last_ai - 1]:
            return handler(request)
        msgs[last_ai - 1] = tagged
        return handler(request.override(messages=msgs))


def _emit(payload: dict) -> None:
    try:
        get_stream_writer()(payload)
    except Exception:
        pass


# ── 寫完再檢查只用在長交付物（2026-10-23 使用者決定：報告、程式、用到 skill 的任務；一般問答不要跑）──
# 寫完才判斷（看這回合實際做了什麼），不在路由時猜：Haiku 自己升級的任務事先不知道是什麼。
REPORT_MIN_CHARS = 3000
REPORT_MIN_CHARS_WITH_HEADINGS = 1500
REPORT_MIN_HEADINGS = 4
CODE_MIN_LINES_UNTAGGED = 5
_NOT_CODE_LANGS = {"mermaid", "text", "txt", "markdown", "md", "plaintext"}
_FENCE_RE = re.compile(r"^[ \t]*```([\w+#.-]*)[^\n]*\n(.*?)^[ \t]*```", re.M | re.S)
_HEADING_RE = re.compile(r"^#{1,4}\s+\S", re.M)


def _last_user_index(messages: list) -> int | None:
    """最後一則「使用者真的說的」HumanMessage 的位置（評分回饋也是 HumanMessage，要跳過）。"""
    from deepagents.middleware.rubric import RUBRIC_GRADER_MESSAGE_SOURCE
    for i in range(len(messages or []) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, HumanMessage) and m.additional_kwargs.get("lc_source") != RUBRIC_GRADER_MESSAGE_SOURCE:
            return i
    return None


def _this_turn(messages: list) -> list:
    """這回合的訊息：最後一則使用者提問之後。"""
    i = _last_user_index(messages)
    return list(messages or []) if i is None else list(messages[i + 1:])


# 占星解讀不跑寫完再檢查：它是象徵詮釋、不是事實交付，評分一直要它「標計算基準日、把解讀標成推估」
# （2026-10-24 線上兩次都補寫後仍不過，每次多 30–50 秒）。用了占星 skill 或占星工具就整回合略過。
NO_RUBRIC_SKILL_PREFIXES = ("astro-",)
NO_RUBRIC_TOOLS = {"get_natal_chart", "get_astro_forecast", "get_synastry", "get_astro_reference"}


def deliverable_reason(messages: list) -> str:
    """這回合算不算長交付物；是的話回傳理由（給時間軸），不是回空字串。占星回合、有上傳文件的對話一律不算：
    評分員只看得到文字轉錄、看不到文件，會把根據文件寫的內容都評成「沒有根據」。文件的根據改由引用負責。"""
    if has_shelf(messages):
        return ""
    turn = _this_turn(messages)
    skills = []
    for m in turn:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls or []:
                if tc.get("name") in NO_RUBRIC_TOOLS:
                    return ""
                path = str((tc.get("args") or {}).get("file_path") or (tc.get("args") or {}).get("path") or "")
                if path.startswith(tuple(f"/skills/{p}" for p in NO_RUBRIC_SKILL_PREFIXES)):
                    return ""
                if tc.get("name") == "read_file" and path.startswith("/skills/") and not path.startswith("/skills/widget-"):
                    name = path.split("/")[2] if path.count("/") >= 3 else path
                    if name not in skills:
                        skills.append(name)
    if skills:
        return f"用了技能：{'、'.join(skills[:3])}"
    final = merge_trailing_ai(turn)
    text = final.text if final is not None and isinstance(final.text, str) else ""
    for lang, body in _FENCE_RE.findall(text):
        lang = lang.lower()
        if lang in _NOT_CODE_LANGS:
            continue
        if lang or len([ln for ln in body.splitlines() if ln.strip()]) >= CODE_MIN_LINES_UNTAGGED:
            return "回答裡有程式碼"
    n_chars, n_heads = len(text), len(_HEADING_RE.findall(text))
    if n_chars >= REPORT_MIN_CHARS or (n_heads >= REPORT_MIN_HEADINGS and n_chars >= REPORT_MIN_CHARS_WITH_HEADINGS):
        return f"報告型長回答（{n_chars} 字）"
    return ""


def make_deliverable_rubric(rubric_cls):
    """RubricMiddleware 的子類：不是長交付物就不評分（不呼叫 grader、不多等）。只覆寫公開的 after_agent。

    ⚠️ 覆寫的方法一定要重新標 @hook_config(can_jump_to=["model"])：LangChain 是看方法上的標記決定能不能
    「跳回模型重寫」，覆寫後標記就不見了——2026-10-23 VBA 驗收實測：評分說要補寫，卻直接結束，
    最後一則是評分回饋、回答變成空的。tests/test_claude_deep.py 盯著這個標記。
    """

    class DeliverableRubricMiddleware(rubric_cls):
        @hook_config(can_jump_to=["model"])
        def after_agent(self, state, runtime):
            if not deliverable_reason(state.get("messages") or []):
                return None
            return super().after_agent(state, runtime)

        @hook_config(can_jump_to=["model"])
        async def aafter_agent(self, state, runtime):
            if not deliverable_reason(state.get("messages") or []):
                return None
            return await super().aafter_agent(state, runtime)

    return DeliverableRubricMiddleware


class RubricStartMiddleware(AgentMiddleware):
    """寫完再檢查開始時送一個事件（RubricMiddleware 只在評完才有回呼）。

    after_agent 依 middleware 順序倒著跑：這個要排在 RubricMiddleware「後面」，才會在評分前先送出。
    每次 agent 準備結束（第一次寫完、補寫完）都會送一次；不是長交付物就不送（也不會評）。
    """

    def after_agent(self, state, runtime) -> dict[str, Any] | None:
        if state.get("rubric"):
            reason = deliverable_reason(state.get("messages") or [])
            if reason:
                _emit({"type": "rubric_start", "reason": reason})
        return None


def _server_tool_digest(msg: AIMessage) -> str:
    """把一則回應裡的 server tool（網路搜尋、讀網頁）整理成文字：查了什麼、拿到哪些標題與網址。"""
    lines: list[str] = []
    names: dict[str, str] = {}
    for b in msg.content if isinstance(msg.content, list) else []:
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t == "server_tool_use":
            names[b.get("id") or ""] = b.get("name") or ""
            inp = b.get("input") or {}
            if not inp and b.get("partial_json"):
                try:
                    import json
                    inp = json.loads(b["partial_json"])
                except ValueError:
                    inp = {}
            q = inp.get("query") or inp.get("url") or ""
            lines.append(f"- 呼叫 {b.get('name')}：{str(q)[:120]}")
        elif t == "web_search_tool_result":
            content = b.get("content")
            if isinstance(content, list):
                hits = [f"{r.get('title') or ''}（{r.get('url') or ''}）" for r in content if isinstance(r, dict)][:8]
                lines.append("  結果：" + ("；".join(hits) if hits else "沒有結果"))
            else:
                lines.append(f"  結果：搜尋失敗（{(content or {}).get('error_code') if isinstance(content, dict) else '?'}）")
        elif t == "web_fetch_tool_result":
            content = b.get("content") or {}
            ok = isinstance(content, dict) and content.get("type") == "web_fetch_result"
            lines.append(f"  結果：{'讀到 ' + str(content.get('url')) if ok else '讀取失敗'}")
    return ("[這次回應裡的網路查詢紀錄（伺服器端工具，內容已省略）]\n" + "\n".join(lines)) if lines else ""


PRIOR_QUESTION_CHARS = 200


def _prior_note(prior: list) -> str:
    """前一題縮成一句：只讓 grader 看得懂「上面那份改短」這類接續的提問。
    只放提問、不放回答：2026-10-24 線上驗收，前一題回答開頭寫著「依據 Microsoft Learn…安妮亞讀過原文」，
    標明了不評分，grader 照樣評它「找不到查詢動作」，下一題（VBA 效能）因此補寫一輪，多花 $0.4、50 秒。"""
    q = next((m for m in reversed(prior) if isinstance(m, HumanMessage) and not is_shelf(m)), None)
    if q is None:
        return ""
    return ("（先前的對話，只供理解這次的提問，不在評分範圍）\n"
            f"上一題：{q.text.strip()[:PRIOR_QUESTION_CHARS]}")


def grader_view(messages: list) -> list:
    """給寫完再檢查的 grader 看的對話。兩件事：

    - 只給這一題：deepagents 的轉錄一定保留整段對話的「第一則」提問當原始問題。2026-10-24 線上：先問 SOP
      再問流年，流年那題被評「SOP 標示已查證卻沒有查詢紀錄」（前一題的紀錄不會帶進歷史），Opus 還照回饋
      去補查勞保法規。前一題縮成一小段併進這題的提問，標明不評分。
    - 補上伺服器端的搜尋／讀網頁：deepagents 的轉錄只收文字與自訂工具呼叫，看不到它們，會誤判「說查過但
      沒有工具紀錄」（2026-10-23 實測）。在每則有查詢的回應前面補一段查詢紀錄。"""
    msgs = list(messages or [])
    start = _last_user_index(msgs)
    if start:
        note = _prior_note(msgs[:start])
        q = msgs[start]
        if note:
            content = (f"{note}\n\n這次的提問：{q.content}" if isinstance(q.content, str)
                       else [{"type": "text", "text": note + "\n\n這次的提問："}, *q.content])
            q = q.model_copy(update={"content": content})
        msgs = [q, *msgs[start + 1:]]
    out = []
    for m in msgs:
        if isinstance(m, AIMessage):
            digest = _server_tool_digest(m)
            if digest:
                content = m.content if isinstance(m.content, list) else [{"type": "text", "text": str(m.content)}]
                m = m.model_copy(update={"content": [{"type": "text", "text": digest}, *content]})
        out.append(m)
    return out


def rubric_event(evaluation: dict) -> None:
    """RubricMiddleware 的 on_evaluation：評分結果送給畫面（時間軸、要不要清掉第一版回答）。"""
    try:
        crit = evaluation.get("criteria") or []
        _emit({"type": "rubric", "iteration": evaluation.get("iteration", 0),
               "result": evaluation.get("result"), "explanation": str(evaluation.get("explanation") or "")[:300],
               "passed": sum(1 for c in crit if c.get("passed")), "total": len(crit),
               "gaps": [str(c.get("gap") or "")[:160] for c in crit if not c.get("passed")][:4]})
    except Exception:
        pass


def server_sources(msg: AIMessage) -> tuple[int, int, list[dict]]:
    """一則回應裡的網路搜尋次數、讀網頁次數，與拿到的 [{url, title}]（搜尋結果＋讀到的網頁）。"""
    searches, fetches, found = 0, 0, []
    for b in msg.content if isinstance(msg.content, list) else []:
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t == "server_tool_use":
            searches += b.get("name") == "web_search"
            fetches += b.get("name") == "web_fetch"
        elif t == "web_search_tool_result" and isinstance(b.get("content"), list):
            found += [{"url": r.get("url") or "", "title": r.get("title") or ""}
                      for r in b["content"] if isinstance(r, dict) and r.get("url")]
        elif t == "web_fetch_tool_result":
            c = b.get("content") or {}
            if isinstance(c, dict) and c.get("type") == "web_fetch_result" and c.get("url"):
                found.append({"url": c["url"], "title": ((c.get("content") or {}).get("title") or c["url"])})
    return searches, fetches, found


class ResearchSourcesMiddleware(AgentMiddleware):
    """研究員每次回應後送出「查了幾次、拿到哪些網址」（custom 事件 research_sources）。
    子代理的逐字與搜尋事件在畫面上看不到（events.parse_item 擋掉第二層以下的 messages），這裡讓時間軸看得到進度，
    也讓頁面在收尾時核對：報告列的網址都要真的出現在研究員的搜尋結果裡（research.check_sources）。"""

    def __init__(self, *, who: str):
        super().__init__()
        self.who = who

    def after_model(self, state, runtime) -> dict[str, Any] | None:
        msgs = state.get("messages") or []
        last = msgs[-1] if msgs else None
        if isinstance(last, AIMessage):
            searches, fetches, found = server_sources(last)
            if searches or fetches or found:
                try:
                    get_stream_writer()({"type": "research_sources", "who": self.who, "searches": searches,
                                         "fetches": fetches, "results": found})
                except Exception:
                    pass
        return None


class UsageEventsMiddleware(AgentMiddleware):
    def __init__(self, *, label: str, model_id: str):
        super().__init__()
        self.label = label
        self.model_id = model_id
        self._n = 0

    def after_model(self, state, runtime) -> dict[str, Any] | None:
        msgs = state.get("messages") or []
        last = msgs[-1] if msgs else None
        if isinstance(last, AIMessage):
            self._n += 1
            try:
                get_stream_writer()({"type": "usage", "label": self.label, "model": self.model_id,
                                     "usage": last.usage_metadata or {}, "web": web_search_count(last)})
            except Exception:
                pass   # 不在串流裡（例如 invoke）就沒有 writer；用量事件不影響回答
        return None
