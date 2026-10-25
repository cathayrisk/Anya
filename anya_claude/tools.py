# -*- coding: utf-8 -*-
"""階段 2：深思模式（deepagents）的自訂工具。純 Python，無 streamlit 依賴——跑在背景執行緒。

移植自 Home.py（盤點見 2026-10-23 session）：只搬「資料」，UI 副作用（_status、_gif、render_todo_panel）
全部丟掉，改成送 custom 事件給頁面畫。
- 氣象署三支：utils/cwa_weather 的 *_impl。金鑰由頁面啟動時放進環境變數 CWA_API_KEY
  （cwa_weather 先讀 st.secrets、再讀環境變數；背景執行緒讀 st.secrets 沒驗證過，不靠它）。
- think：舊版的硬性反思工具。完整度連續偏低時回傳策略提示（規則照搬 Home.py:2448）；
  前幾次的完整度從這回合的對話紀錄（AIMessage.tool_calls）裡找，不靠 session_state。
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.config import get_stream_writer

from utils import evidence as EV

from . import calc as CALC


def _emit(payload: dict) -> None:
    try:
        get_stream_writer()(payload)
    except Exception:
        pass   # 不在串流裡（測試、invoke）就沒有 writer


def _evidence(tool_name: str, scope: str, status: str, detail: Any = None) -> None:
    _emit({"type": "evidence", "tool": tool_name, "scope": scope, "status": status, "detail": detail})


# ───────── calculate（精確計算）─────────
def run_calculate(expressions: list[str]) -> str:
    """算式交給 calc（安全、Excel 慣例），結果送 calc 事件給時間軸，回傳給模型的 JSON。Haiku 節點也直接呼叫這個。"""
    results = CALC.evaluate(expressions)
    _emit({"type": "calc", "results": [{k: r[k] for k in ("expr", "text", "error") if k in r} for r in results]})
    return json.dumps([{k: r[k] for k in ("expr", "name", "value", "text", "error") if k in r} for r in results],
                      ensure_ascii=False, default=str)


@tool
def calculate(expressions: list[str]) -> str:
    """精確計算數字，不要心算。利息、報酬、試算、月付、百分比、加總、換算都用這個工具算，回答裡的數字照結果寫。
    expressions：一行一個算式，可以先定義變數再用，例如 ["r = 0.05/12", "n = 20*12", "fv(r, n, -10000)"]。
    支援 + - * / // % **、括號、round/abs/min/max/sum/sqrt/exp/log/log10/ceil/floor，以及和 Excel 相同的財務函式：
    fv(rate, nper, pmt, pv=0, when=0)、pv(rate, nper, pmt, fv=0, when=0)、pmt(rate, nper, pv, fv=0, when=0)、
    nper(rate, pmt, pv, fv=0, when=0)、npv(rate, [CF0, CF1, …])（CF0 在第 0 期不折現）、irr([CF0, CF1, …])。
    付出為負、收到為正；when=1 表示期初。百分比用小數（5% 寫 0.05），數字不要加千分位逗號。"""
    return run_calculate(expressions)


# ───────── 氣象署 ─────────
def _cwa(impl_name: str):
    from utils import cwa_weather   # 延遲 import：測試可以不碰網路模組
    return getattr(cwa_weather, impl_name)


@tool
def get_weather(location: str = "") -> str:
    """查中央氣象署的即時觀測、36 小時預報、天氣特報與雨量。location 是縣市或鄉鎮名（例如「台北市」「花蓮縣」），
    空字串＝全台概況。問天氣、下雨、溫度、特報時一定要先呼叫，不可用記憶回答。"""
    try:
        out = _cwa("get_weather_impl")(location)
        _evidence("get_weather", EV.SCOPE_WEATHER, EV.STATUS_OK, {"location": location})
        return json.dumps(out, ensure_ascii=False, default=str)
    except Exception as e:
        _evidence("get_weather", EV.SCOPE_WEATHER, EV.STATUS_ERROR, {"error": type(e).__name__})
        return json.dumps({"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}, ensure_ascii=False)


@tool
def get_earthquake_info() -> str:
    """查中央氣象署最新的有感地震報告（時間、震央、規模、深度、各地震度）。問地震時一定要先呼叫。
    回傳 found=false 只代表這次沒拿到可顯示的事件，不等於近期沒有地震。"""
    try:
        out = _cwa("get_earthquake_impl")()
        _evidence("get_earthquake_info", EV.SCOPE_EARTHQUAKE,
                  EV.STATUS_OK if out.get("found") else EV.STATUS_EMPTY, None)
        return json.dumps(out, ensure_ascii=False, default=str)
    except Exception as e:
        _evidence("get_earthquake_info", EV.SCOPE_EARTHQUAKE, EV.STATUS_ERROR, {"error": type(e).__name__})
        return json.dumps({"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}, ensure_ascii=False)


@tool
def get_typhoon_info() -> str:
    """查中央氣象署的颱風警報與追蹤中的熱帶性低氣壓。問颱風時一定要先呼叫。
    has_active_taiwan_warning=false 是官方「目前沒有警報」的有效答案。"""
    try:
        out = _cwa("get_typhoon_impl")()
        _evidence("get_typhoon_info", EV.SCOPE_TYPHOON, EV.STATUS_OK, None)
        return json.dumps(out, ensure_ascii=False, default=str)
    except Exception as e:
        _evidence("get_typhoon_info", EV.SCOPE_TYPHOON, EV.STATUS_ERROR, {"error": type(e).__name__})
        return json.dumps({"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}, ensure_ascii=False)


CWA_TOOLS = [get_weather, get_earthquake_info, get_typhoon_info]


# ───────── create_widget（互動元件）─────────
# 驗證規則照搬 Home.py:2893；畫面由頁面在主執行緒畫（components.html），這裡只送 widget 事件
WIDGET_MAX_CHARS = 20_000
_WIDGET_EXTERNAL_RE = re.compile(r"""(?:src\s*=\s*["']?|url\(\s*["']?)\s*https?://""", re.IGNORECASE)


def _retry_hint() -> str:
    """舊版的提示叫模型用 load_skill；深思模式的模板在 skills 虛擬檔案裡，改成 read_file 的說法。"""
    from utils import widget_audit as WAUDIT
    return WAUDIT.RETRY_HINT.replace("用 load_skill 載入對應的 widget_* 模板",
                                     "用 read_file 讀對應的 /skills/widget-<名稱>/SKILL.md 模板")


def _this_turn_tool_results(messages: list, name: str) -> list[str]:
    from langchain_core.messages import ToolMessage
    return [m.content if isinstance(m.content, str) else str(m.content)
            for m in messages or [] if isinstance(m, ToolMessage) and m.name == name]


@tool
def create_widget(title: str, height: int, html: str, runtime: ToolRuntime) -> str:
    """生成互動 HTML 小元件，顯示在這則回答下方（自包含、零外部資源）。

    何時用：「互動比敘述更能回答」時——數字試算／參數探索、多維度比較、研究來源瀏覽、從文件做學習卡。
    單純敘述性答案不要用。
    流程：先 read_file 讀對應的 /skills/widget-<名稱>/SKILL.md 模板，只替換資料區、不改結構與 JS，再呼叫本工具。
    限制：每回合最多 1 個；HTML 完全自包含（CSS／JS inline），禁止外部資源與 fetch／XHR；height 依內容估 200–800（px）。
    """
    from utils import widget_audit as WAUDIT
    from utils.widget_state import content_fingerprint

    earlier = _this_turn_tool_results((runtime.state or {}).get("messages") if runtime else [], "create_widget")
    if any('"ok": true' in r for r in earlier):
        return json.dumps({"error": "本回合已生成過互動元件（上限 1 個），請直接完成文字回答。"}, ensure_ascii=False)
    h = (html or "").strip()
    if not h:
        return json.dumps({"error": "html 不可為空"}, ensure_ascii=False)
    if len(h) > WIDGET_MAX_CHARS:
        return json.dumps({"error": f"HTML 過長（{len(h)} > {WIDGET_MAX_CHARS} 字元），請精簡資料量後重試。"},
                          ensure_ascii=False)
    low = h.lower()
    if "<script src" in low or "fetch(" in low or "xmlhttprequest" in low or _WIDGET_EXTERNAL_RE.search(h):
        return json.dumps({"error": "偵測到外部資源載入或網路請求（<script src> / src=http / url(http / fetch / XHR）。"
                                    "元件必須完全自包含：把資源改為 inline 後重試。"}, ensure_ascii=False)
    # 狀態流失稽核：自己手寫的 HTML 不會接 AnyaState，使用者一操作、頁面一重跑就全部重置。只擋一次（判定是啟發式的）
    audit = WAUDIT.audit_widget_html(h)
    hint = _retry_hint()
    if audit.will_lose_state and not any(hint[:30] in r for r in earlier):
        return json.dumps({"error": hint}, ensure_ascii=False)
    try:
        hh = max(200, min(800, int(height)))
    except (TypeError, ValueError):
        hh = 420
    # 每個元件一個不可變的 id：歷史回放沿用同一個，狀態才對得上（不可用標題或內容 hash，見 utils/widget_state.py）
    widget = {"title": (title or "互動元件")[:60], "html": h, "height": hh,
              "wid": uuid.uuid4().hex[:16], "fp": content_fingerprint(h), "audit": audit.as_dict()}
    _emit({"type": "widget", **widget})
    return json.dumps({"ok": True, "note": "元件已顯示在回答下方。正文用 1–2 句說明元件用途與操作方式即可，"
                                          "不要用文字重複元件內容。"}, ensure_ascii=False)


# ───────── think（搜尋後反思）─────────
NEXT_ACTIONS = ("繼續搜尋", "換工具", "直接作答")

HINT_VERY_LOW = ("⚠️ 策略警告（系統）：這次完整度 < 30，搜尋方向可能完全錯了。請重新審視問題本身："
                 "1. 改用英文關鍵字 2. 把問題拆成更小的子問題 3. 直接讀相關官方網頁（web_fetch）。"
                 "不要再用意思相近的關鍵字重搜。")
HINT_STUCK = ("🚨 強化策略警告（系統）：連續 3 次完整度 ≤ 55，搜尋策略卡住了。立刻擇一："
              "1. 改用英文關鍵字 2. 換完全不同的概念或同義詞 3. 直接讀已知的相關網址 "
              "4. 都做不到就用現有資料作答（next_action='直接作答'），並列出還沒查到的部分。")
HINT_LOW = ("⚠️ 策略警告（系統）：連續 2 次完整度 ≤ 55，關鍵字策略沒效。下一步先診斷再改："
            "關鍵字太專門或太模糊？該改用英文？該換角度或同義詞？下次 think 的 reflection 要寫出你改了什麼。")


def strategy_hint(confidences: list[int]) -> str | None:
    """confidences：這回合到目前為止（含這次）的完整度。規則照搬 Home.py 的 think。"""
    if not confidences:
        return None
    if confidences[-1] < 30:
        return HINT_VERY_LOW
    if len(confidences) >= 2 and all(c <= 55 for c in confidences[-2:]):
        if len(confidences) >= 3 and all(c <= 55 for c in confidences[-3:]):
            return HINT_STUCK
        return HINT_LOW
    return None


def _previous_confidences(messages: list) -> list[int]:
    out = []
    for m in messages or []:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls or []:
                if tc.get("name") == "think":
                    try:
                        out.append(int((tc.get("args") or {}).get("confidence", 0)))
                    except (TypeError, ValueError):
                        out.append(0)
    return out


@tool
def think(reflection: str, key_finding: str, next_action: str, confidence: int, runtime: ToolRuntime) -> str:
    """策略性反思：在搜尋或讀網頁之後、作答之前，停下來評估資料夠不夠回答問題，並決定下一步（不會取得新資訊）。

    何時用：每輪搜尋告一段落、準備作答前，呼叫一次；同一輪多次搜尋合併成一次。沒有搜尋的回合不用呼叫。
    reflection：寫五件事——1.發現摘要 2.和原本假設的對比 3.來源之間的矛盾 4.還缺的資訊 5.下一步策略。
    reflection 和 key_finding 用正體中文寫，會顯示在使用者的畫面上。
    key_finding：1–2 句，這一輪最重要的發現。
    next_action：「繼續搜尋」「換工具」「直接作答」三選一。
    confidence：0–100，目前能完整回答問題的程度；≥80 就直接作答。
    """
    try:
        conf = max(0, min(100, int(confidence)))
    except (TypeError, ValueError):
        conf = 0
    action = next_action if next_action in NEXT_ACTIONS else "繼續搜尋"
    # 這一次的 tool_call 已經在 messages 裡（模型先送出呼叫，才輪到工具執行）
    history = _previous_confidences((runtime.state or {}).get("messages") if runtime else [])
    confs = history if history and history[-1] == conf else history + [conf]
    hint = strategy_hint(confs)
    _emit({"type": "reflect", "n": len(confs), "confidence": conf, "next_action": action,
           "key_finding": (key_finding or "").strip()[:200], "reflection": (reflection or "").strip()[:1500],
           "hint": bool(hint)})
    out: dict[str, Any] = {"ok": True}
    if hint:
        out["strategy_hint"] = hint
    return json.dumps(out, ensure_ascii=False)
