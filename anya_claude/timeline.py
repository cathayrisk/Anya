# -*- coding: utf-8 -*-
"""狀態列展開後的「步驟時間軸」：這回合實際做了哪些事。純資料，無 streamlit 依賴。

2026-10-23 使用者選了 st.status(type="step") 時間軸。分工：
- 收合時那一行標題＝安妮亞心裡話（monologue.py，好玩）
- 展開後的時間軸＝事實（這裡）：收到→（查氣象署）→搜尋／讀網頁→升級→深思→撰寫→完成
畫面層（view.py）只負責把 steps 畫出來；這裡的邏輯（合併連續搜尋、算秒數、路由講人話）可以單獨測。

實測得來的規則：
- Streamlit 1.63：沒有內容的 step 會結束時間軸，所以除了最後的「完成」，每一步都要有 detail。
- Opus 的 web_search_20260209 會用 code_execution 包住搜尋再篩選結果（串流順序：程式開始 → 搜尋 → 結果 →
  程式結果）。緊鄰搜尋的程式併進搜尋那一步，不然一題會冒出好幾個「搜尋 0 筆」；單獨跑的程式才算「整理資料」。
- 思考摘要是在下一個動作前一次送來的，從第一個摘要字算會變成「想了 0.1 秒」；改從上一步結束算。
- Opus 的思考摘要是英文的，2026-10-23 使用者決定不顯示：「想了 N 秒」只記時間和接著做了什麼；
  想完緊接著呼叫 think 工具時併成同一步，底下放它寫的中文反思。
- 搜尋過、帶引用的回答，API 是寫完才整段一次送來（2026-10-23 實測：510 字的片段同一刻到）。
  所以「寫好了」從上一步結束算、「想了」算到摘要送達（摘要在想完時才送）；反思說「直接作答」
  就先亮出撰寫回答，不然那幾秒畫面像卡住。
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

MODE_NAME = {"fast": "一般模式（Haiku 5.5）", "deep": "深思模式（Opus 5.5）",
             "socratic": "提問引導（Opus 5.5）", "cwa": "氣象署直答"}
MODEL_NAME = {"fast": "Haiku 5.5", "deep": "Opus 5.5", "socratic": "Opus 5.5"}

# router.Route.reason → 給使用者看的「為什麼這樣分」
ROUTE_WHY = {
    "default_fast": "一般問題",
    "translation": "翻譯、摘要這類貼上文字的請求",
    "hazard_hint": "災防題，要查即時資料",
    "verify_hint": "你要求查證",
    "user_requested_deep": "你要求深思",
    "keyword_hint": "需要深入處理的任務",
    "contains_image": "有附圖片",
    "has_docs": "有附文件",
    "has_url": "附了網址，要讀原文",
    "socratic_mode": "提問引導模式中",
    "user_redo_deep": "你要求改用深思模式重答",
    "user_redo_fast": "你要求改用一般模式重答",
}
MAX_TITLES = 5
THINKING = "在想下一步怎麼做"      # 深思中那一步的內容（英文摘要不顯示；step 不能沒內容）
REFLECT_LINES = 8
CALC_LINES = 10

# deep agent 的自訂工具 → 時間軸的哪一類步驟（group, 圖示, 動作）
CLIENT_TOOLS = {
    "get_weather": ("cwa", "🌐", "查氣象署天氣"),
    "get_earthquake_info": ("cwa", "🌐", "查氣象署地震"),
    "get_typhoon_info": ("cwa", "🌐", "查氣象署颱風"),
    "read_file": ("skill", "📚", "讀技能說明"),
    "ls": ("files", "📂", "翻資料夾"),
    "glob": ("files", "📂", "翻資料夾"),
    "grep": ("files", "📂", "在資料裡找字"),
    "write_file": ("notes", "🗒️", "寫筆記"),
    "edit_file": ("notes", "🗒️", "改筆記"),
    "write_todos": ("todo", "📝", "列待辦"),
    "think": ("reflect", "🧠", "寫下反思"),
    "calculate": ("calc", "🧮", "精確計算"),
    "create_widget": ("widget", "🧩", "做互動元件"),
    "save_lesson": ("lesson", "🧠", "記下教訓"),
    "get_natal_chart": ("astro", "🔮", "排本命盤"),
    "get_astro_forecast": ("astro", "🔮", "推算流年"),
    "get_synastry": ("astro", "🔮", "計算合盤"),
    "get_astro_reference": ("astro_ref", "📖", "查占星原型文章"),
}
_WAIT_SEARCH = "等待搜尋結果…"
_WAIT_FETCH = "等待網頁內容…"


def route_why(reason: str) -> str:
    if reason.startswith("skill_hint:"):
        return f"需要專門技能（{reason.split(':', 1)[1]}）"
    return ROUTE_WHY.get(reason, reason)


def _plain(s: str, limit: int) -> str:
    """放進 markdown 連結文字／caption 的字串：拿掉會打壞語法的方括號、換行，限長。"""
    s = re.sub(r"\s+", " ", (s or "").replace("[", "（").replace("]", "）")).strip()
    return s[:limit] + ("…" if len(s) > limit else "")


def _secs(s: float) -> str:
    return f"{s:.0f} 秒" if s >= 10 else f"{s:.1f} 秒"


def _unshimmer(label: str) -> str:
    return re.sub(r":shimmer\[(.*)\]", r"\1", label)


def _verb(label: str) -> str:
    """「🔎 :shimmer[上網搜尋…]」→「上網搜尋」"""
    return _unshimmer(label).split(" ", 1)[-1].rstrip("…")


def _expr_view(expr: str, limit: int = 60) -> str:
    """算式放進 markdown caption：* 會被當成斜體，換成 × 與 ^；方括號跳脫就好——_plain 換成全形括號，
    清單會變成「irr(（-100, 25）)」看不出是清單（2026-10-24 線上）。"""
    s = re.sub(r"\s+", " ", expr or "").strip()
    s = s[:limit] + ("…" if len(s) > limit else "")
    return s.replace("**", "^").replace("*", "×").replace("[", "\\[").replace("]", "\\]")


def _reflection_lines(text) -> list[str]:
    """think 工具的中文反思（照五點寫）：保留分行，拿掉標題記號，每行限長，最多 REFLECT_LINES 行。"""
    lines = [_plain(re.sub(r"^#+\s*", "", ln.strip()), 160) for ln in str(text or "").splitlines()]
    lines = [ln for ln in lines if ln]
    return lines[:REFLECT_LINES] + (["…"] if len(lines) > REFLECT_LINES else [])


@dataclass
class Step:
    kind: str                    # received | cwa | search | fetch | code | escalate | think | write | done | error | stopped
    label: str                   # markdown；進行中的會包 :shimmer[]
    state: str = "running"       # running | complete | error
    detail: list[str] = field(default_factory=list)
    started: float = 0.0
    ended: float = 0.0
    calls: int = 0               # 搜尋／讀網頁的次數；撰寫回答：0＝先亮出來、字還沒到
    hits: int = 0                # 搜尋結果筆數
    last: float = 0.0            # 深思：最後一個摘要片段到的時間（＝想完的時間）

    def record(self) -> dict:
        return {"label": self.label, "state": self.state, "detail": list(self.detail)}


class Timeline:
    """一回合一個。每次變動 version +1，畫面層看 version 決定要不要重畫。"""

    def __init__(self, *, route_reason: str, mode: str, clock=time.time):
        self.clock = clock
        self.mode = mode
        self.version = 0
        now = clock()
        self.steps: list[Step] = [Step("received", "🥜 收到問題", "complete",
                                       [f"交給{MODE_NAME.get(mode, mode)}：{route_why(route_reason)}"],
                                       started=now, ended=now)]

    @property
    def current(self) -> Step:
        return self.steps[-1]

    # ── 氣象署 ──
    def cwa_start(self) -> None:
        self._open(Step("cwa", "🌐 :shimmer[查氣象署即時資料…]", detail=["中央氣象署開放資料"]))

    def cwa_done(self, results: list[dict]) -> None:
        if self.current.kind != "cwa":
            self.cwa_start()
        names = {"earthquake": "地震", "typhoon": "颱風"}
        cur = self.current
        cur.detail = [f"{names.get(r.get('scope'), r.get('scope'))}："
                      f"{'✓ 有拿到' if r.get('status') == 'ok' else '✗ ' + str(r.get('status'))}"
                      for r in results or []] or ["沒有要查的項目"]
        cur.label = "🌐 查了氣象署即時資料"
        self._complete(cur)

    # ── 搜尋 ──
    def search_start(self) -> None:
        cur = self.current
        if cur.kind == "search":                      # 連續搜尋併成一步
            cur.state, cur.calls = "running", cur.calls + 1
            cur.label = f"🔎 :shimmer[上網搜尋（第 {cur.calls} 次）…]"
            self._bump()
        elif cur.kind == "code" and cur.state == "running":   # 包著搜尋的程式：改成搜尋那一步
            cur.kind, cur.calls, cur.label, cur.detail = "search", 1, "🔎 :shimmer[上網搜尋…]", [_WAIT_SEARCH]
            self._bump()
        else:
            self._open(Step("search", "🔎 :shimmer[上網搜尋…]", calls=1, detail=[_WAIT_SEARCH]))

    def search_done(self, results: list[dict], status: str = "ok") -> None:
        if self.current.kind != "search":
            self.search_start()
        cur = self.current
        if cur.detail == [_WAIT_SEARCH]:
            cur.detail = []
        cur.hits += len(results or [])
        for r in results or []:
            if sum(1 for d in cur.detail if d.startswith("- ")) >= MAX_TITLES:
                break
            cur.detail.append(f"- [{_plain(r.get('title') or r.get('url'), 40)}]({r.get('url')})")
        if status != "ok" and not results:
            cur.detail.append("（這次沒有拿到結果）" if status == "empty" else f"（搜尋失敗：{_plain(status, 20)}）")
        cur.label = self._search_label(cur)
        self._complete(cur)

    # ── 讀網頁 ──
    def fetch_start(self) -> None:
        cur = self.current
        if cur.kind == "fetch":
            cur.state, cur.calls = "running", cur.calls + 1
            cur.label = f"📄 :shimmer[讀第 {cur.calls} 個網頁…]"
            self._bump()
        elif cur.kind == "code" and cur.state == "running":    # 和搜尋一樣，包著讀網頁的程式併進來
            cur.kind, cur.calls, cur.label, cur.detail = "fetch", 1, "📄 :shimmer[讀網頁…]", [_WAIT_FETCH]
            self._bump()
        else:
            self._open(Step("fetch", "📄 :shimmer[讀網頁…]", calls=1, detail=[_WAIT_FETCH]))

    def fetch_done(self, url: str, status: str = "ok") -> None:
        if self.current.kind != "fetch":
            self.fetch_start()
        cur = self.current
        if cur.detail == [_WAIT_FETCH]:
            cur.detail = []
        host = urlparse(url or "").netloc or "（網址不明）"
        cur.detail.append(f"- {_plain(host, 40)}" + ("" if status == "ok" else f"（失敗：{_plain(status, 20)}）"))
        cur.label = self._fetch_label(cur)
        self._complete(cur)

    # ── 程式（Opus 篩選搜尋結果用）──
    def code_start(self) -> None:
        cur = self.current
        if cur.kind in ("search", "fetch"):
            cur.state = "running"
            cur.label = "🔎 :shimmer[整理搜尋結果…]" if cur.kind == "search" else "📄 :shimmer[整理網頁內容…]"
            self._bump()
        elif cur.kind == "code" and cur.state == "running":
            return
        else:
            self._open(Step("code", "🧮 :shimmer[用程式整理資料…]", detail=["在沙盒裡跑程式篩選、計算資料"]))

    def code_done(self) -> None:
        cur = self.current
        if cur.state != "running" or cur.kind not in ("search", "fetch", "code"):
            return
        cur.label = {"search": self._search_label, "fetch": self._fetch_label}.get(
            cur.kind, lambda _s: "🧮 用程式整理了資料")(cur)
        self._complete(cur)

    # ── deep agent 的自訂工具（階段 2）──
    # 串流順序：tool_call（只有名稱）→ tool_args（模型節點跑完，參數完整）→ tool_result（工具跑完）
    def tool_call(self, name: str) -> None:
        group, icon, verb = CLIENT_TOOLS.get(name, ("tool", "🧰", f"用工具 {name}"))
        cur = self.current
        if group == "reflect" and cur.kind == "think" and cur.state == "running":
            # 想完緊接著寫反思：併成同一步，秒數從開始想算起，內容換成中文反思
            cur.kind, cur.calls = "reflect", 1
            cur.label, cur.detail = f"{icon} :shimmer[{verb}…]", ["檢查資料夠不夠回答問題"]
            self._bump()
            return
        if cur.kind == group and group != "reflect":           # 連續同類（例如讀好幾個檔）併成一步
            cur.state, cur.calls = "running", cur.calls + 1
            cur.label = f"{icon} :shimmer[{verb}…]"
            self._bump()
            return
        detail = {"reflect": ["檢查資料夠不夠回答問題"], "cwa": ["中央氣象署開放資料"]}.get(group, [verb])
        self._open(Step(group, f"{icon} :shimmer[{verb}…]", calls=1, detail=list(detail)))

    def tool_args(self, calls: list[dict]) -> None:
        cur = self.current
        changed = False
        for c in calls or []:
            name, args = c.get("name") or "", c.get("args") or {}
            if name == "read_file" and cur.kind == "skill":
                path = str(args.get("file_path") or args.get("path") or "")
                skill = path.split("/")[2] if path.startswith("/skills/") and path.count("/") >= 3 else ""
                line = f"- {skill}" if skill else f"- {_plain(path, 60)}"
                cur.detail = [d for d in cur.detail if d != "讀技能說明"]
                if line not in cur.detail:
                    cur.detail.append(line)
                    changed = True
            elif name == "save_lesson" and cur.kind == "lesson":
                cur.detail = [f"[{_plain(args.get('category'), 20)}] {_plain(args.get('summary'), 80)}"]
                changed = True
            elif name == "write_todos" and cur.kind == "todo":
                mark = {"completed": "✅", "in_progress": "▶️"}
                cur.detail = [f"- {mark.get(t.get('status'), '⬜')} {_plain(t.get('content'), 60)}"
                              for t in (args.get("todos") or [])][:10] or cur.detail
                changed = True
        if changed:
            self._bump()

    def tool_result(self, name: str, status: str = "success", text: str = "") -> None:
        group = CLIENT_TOOLS.get(name, ("tool",))[0]
        cur = self.current
        if cur.kind != group or group in ("reflect", "calc"):   # think 由 reflect 事件、calculate 由 calc 事件收尾
            return
        failed = status == "error" or '"ok": false' in (text or "")[:200]
        if group == "cwa":
            what = CLIENT_TOOLS[name][2].replace("查氣象署", "")
            cur.detail = [d for d in cur.detail if d != "中央氣象署開放資料"] + [f"{what}：{'✗ 沒拿到' if failed else '✓ 有拿到'}"]
            cur.label = "🌐 查了氣象署即時資料"
        elif group == "skill":
            skills = [d[2:] for d in cur.detail if d.startswith("- ")]
            cur.label = f"📚 讀了技能：{'、'.join(skills[:3])}" if skills else "📚 讀了技能說明"
        elif group == "lesson":
            cur.label = "🧠 記住了（之後的對話都會參考）" if '"ok": true' in (text or "") else "🧠 這條沒記成功"
        elif group == "widget":
            ok = '"ok": true' in (text or "")
            cur.label = "🧩 做好了互動元件" if ok else "🧩 互動元件被退回重做"
            if not ok:
                try:
                    cur.detail = [_plain(str(json.loads(text).get("error") or text), 80)]
                except (ValueError, AttributeError):
                    cur.detail = [_plain(text, 80)]
        else:
            cur.label = _unshimmer(cur.label).rstrip("…") + ("（失敗）" if failed else "")
        self._complete(cur)

    # ── 寫完再檢查（RubricMiddleware）──
    def rubric_start(self, reason: str = "") -> None:
        detail = ["對照：有沒有漏答、查過的有沒有根據、會變動的有沒有日期、沒查到的有沒有說"]
        if reason:
            detail.insert(0, f"為什麼要檢查：{_plain(reason, 60)}")
        self._open(Step("rubric", "🔍 :shimmer[自我檢查回答…]", detail=detail))

    def rubric_done(self, data: dict) -> None:
        if self.current.kind != "rubric" or self.current.state != "running":
            self.rubric_start()
        cur = self.current
        total, passed = int(data.get("total") or 0), int(data.get("passed") or 0)
        gaps = [f"- {_plain(g, 120)}" for g in data.get("gaps") or [] if g]
        result = data.get("result")
        if result == "satisfied":
            cur.label = f"🔍 自我檢查：{total} 項全部通過" if total else "🔍 自我檢查：通過"
        elif result == "needs_revision":
            cur.label = f"🔍 自我檢查：{total - passed} 項要補，重寫中"
            cur.detail = gaps or cur.detail
        elif result == "max_iterations_reached":
            cur.label = f"🔍 補寫後仍有 {total - passed} 項沒過（照現況交出）"
            cur.detail = gaps or cur.detail
        else:
            cur.label = "🔍 自我檢查沒跑完（不影響回答）"
        self._complete(cur)

    def widget(self, title: str) -> None:
        """create_widget 通過檢查、送出元件（在 tool_result 之前到）：時間軸寫上元件標題。"""
        if self.current.kind == "widget":
            self.current.detail = [_plain(title, 60)]
            self._bump()

    def calc(self, results: list[dict]) -> None:
        """calculate 算完（工具裡送的 calc 事件）：時間軸列出算式與結果，讓人看得到數字從哪來。連續算幾次併成一步。"""
        if self.current.kind != "calc":
            self.tool_call("calculate")
        cur = self.current
        if cur.detail == ["精確計算"]:
            cur.detail = []
        for r in results or []:
            expr = _expr_view(r.get("expr") or "")
            cur.detail.append(f"- {expr} ＝ {r['text']}" if "text" in r else f"- {expr}：{_plain(r.get('error'), 40)}")
            cur.hits += "text" in r
        cur.detail = cur.detail[-CALC_LINES:]
        cur.label = f"🧮 精確計算・算了 {cur.hits} 筆"
        self._complete(cur)

    def reflect(self, data: dict) -> None:
        if self.current.kind != "reflect" or self.current.state != "running":
            self.tool_call("think")
        cur = self.current
        cur.label = (f"🧠 想了 {_secs(self.clock() - cur.started)}・第 {data.get('n', 1)} 次反思・"
                     f"完整度 {data.get('confidence', 0)}%・{data.get('next_action') or '繼續'}")
        cur.detail = ([f"💡 {_plain(data.get('key_finding') or '（沒寫發現）', 120)}"]
                      + _reflection_lines(data.get("reflection")))
        if data.get("hint"):
            cur.detail.append("⚠️ 完整度偏低，系統提示改變查法")
        self._complete(cur)
        if data.get("next_action") == "直接作答":
            self.write(pending=True)

    # ── 模型 ──
    def escalate(self, *, reason: str, socratic: bool) -> None:
        label = "🧭 交給提問引導模式（Opus 5.5）" if socratic else "↗️ 升級到深思模式（Opus 5.5）"
        preamble = self.mode == "fast" and self.current.kind == "write"
        self.mode = "socratic" if socratic else "deep"
        self._finish_current()
        now = self.clock()
        step = Step("escalate", label, "complete", [_plain(reason, 80) or "Haiku 判斷這題需要深入處理"],
                    started=now, ended=now)
        if preamble:
            # Haiku 升級前先寫了幾個字（畫面上已經清掉）：那一格「寫好了 N 秒」直接換成升級（2026-10-24 實測常見）
            self.steps[-1] = step
        else:
            self.steps.append(step)
        self._bump()

    def think(self, text: str = "") -> None:
        """模型在想。摘要內容（英文）不顯示，只開一步記時間；想完做什麼由下一步補上。"""
        cur = self.current
        if self._pending_write(cur):
            return                                    # 決定作答之後的思考算進撰寫回答
        if cur.kind == "think" and cur.state == "running":
            cur.last = self.clock()
            return
        self._open(Step("think", "🧠 :shimmer[深思中…]", detail=[THINKING]))
        self.current.started = cur.ended or self.current.started   # 模型在上一步結束後就開始想了
        self.current.last = self.clock()

    def write(self, *, pending: bool = False) -> None:
        """開始寫回答。pending：反思說直接作答，字還沒到就先亮出來。"""
        cur = self.current
        if cur.kind == "write" and cur.state == "running":
            cur.calls = cur.calls or int(not pending)
            return
        self._open(Step("write", "✍️ :shimmer[撰寫回答…]", calls=int(not pending),
                        detail=[f"由 {MODEL_NAME.get(self.mode, '模型')} 撰寫"]))
        self.current.started = cur.ended or self.current.started   # 從上一步結束算：回答常是寫完才整段送來

    # ── 結束 ──
    def finish(self, elapsed_s: float) -> None:
        self._finish_current()
        now = self.clock()
        self.steps.append(Step("done", f"🏁 完成・共 {_secs(elapsed_s)}", "complete", started=now, ended=now))
        self._bump()

    def fail(self) -> None:
        self._interrupt("error")
        now = self.clock()
        self.steps.append(Step("error", "⚠️ 沒有完成", "error", started=now, ended=now))
        self._bump()

    def stop(self) -> None:
        """使用者按了停止：畫面已經不能動（StopException），這裡只整理要存進歷史的紀錄。"""
        self._interrupt("complete")
        now = self.clock()
        self.steps.append(Step("stopped", "⏹️ 你按了停止", "complete", started=now, ended=now))
        self._bump()

    def records(self) -> list[dict]:
        return [s.record() for s in self.steps]

    # ── 內部 ──
    @staticmethod
    def _search_label(cur: Step) -> str:
        times = f"查了 {cur.calls} 次・" if cur.calls > 1 else ""
        return f"🔎 上網搜尋・{times}找到 {cur.hits} 筆"

    @staticmethod
    def _fetch_label(cur: Step) -> str:
        return f"📄 讀了 {cur.calls} 個網頁"

    def _interrupt(self, state: str) -> None:
        cur = self.current
        if cur.state == "running":
            cur.state, cur.ended = state, self.clock()
            cur.label = _unshimmer(cur.label).rstrip("…") + "（中斷）"

    @staticmethod
    def _pending_write(step: Step) -> bool:
        return step.kind == "write" and step.state == "running" and not step.calls

    def _open(self, step: Step) -> None:
        prev = self.current
        if self._pending_write(prev):
            # 先亮出的撰寫回答還沒有字，模型又去用工具：原地換成新的那一步（畫面上沿用同一格）
            step.started = self.clock()
            self.steps[-1] = step
            self._bump()
            return
        self._finish_current()
        if prev.kind == "think":
            prev.detail = [f"接著：{_verb(step.label)}"]
        step.started = self.clock()
        self.steps.append(step)
        self._bump()

    def _finish_current(self) -> None:
        cur = self.current
        if cur.state != "running":
            return
        end = cur.last if cur.kind == "think" and cur.last else self.clock()
        took = _secs(end - cur.started)
        done = {"cwa": "🌐 查了氣象署即時資料", "search": self._search_label(cur),
                "fetch": self._fetch_label(cur), "code": "🧮 用程式整理了資料",
                "think": f"🧠 想了 {took}", "write": f"✍️ 寫好了・{took}",
                "skill": "📚 讀了技能說明", "files": "📂 翻了資料", "notes": "🗒️ 寫了筆記", "todo": "📝 列了待辦"}
        cur.label = done.get(cur.kind, _unshimmer(cur.label).rstrip("…"))
        cur.state, cur.ended = "complete", end

    def _complete(self, step: Step) -> None:
        step.state, step.ended = "complete", self.clock()
        self._bump()

    def _bump(self) -> None:
        self.version += 1
