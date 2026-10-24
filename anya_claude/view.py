# -*- coding: utf-8 -*-
"""主執行緒的畫面：把 events.Ev 畫成 Streamlit 元素。只有這裡和頁面本身可以用 st.*。

和舊版 ShimmerStreamRenderer 的差別：一律直接串 markdown（邊收邊排版），不再有
「先純文字流光、最後淡化成 markdown」的兩段式，也沒有收尾的 0.9 秒等待。

狀態列（2026-10-23）：收合的那一行是安妮亞心裡話（monologue.py），展開是步驟時間軸
（timeline.py，st.status(type="step")）。英文的深思摘要不顯示；think 工具的中文反思放在「想了 N 秒」那一步裡。
"""
from __future__ import annotations

import time
from contextlib import contextmanager

import streamlit as st
import streamlit.components.v1 as components

from utils import evidence as EV
from utils.widget_state import inject_state_helper

from . import mdfix as M
from .events import Ev, paragraph_break
from .monologue import Monologue
from .timeline import Timeline
from .usage import TurnUsage

# 等待第一個字時的骨架閃光（移植自 Home.py 的 _SKELETON_CSS）
_WAIT_CSS = (
    "<style>"
    ".rjsk{display:flex;flex-direction:column;gap:9px;padding:6px 0;}"
    ".rjsk .b{height:11px;border-radius:6px;"
    "background:linear-gradient(90deg,#efe7e3 0%,#faf4f1 50%,#efe7e3 100%);"
    "background-size:220% 100%;animation:rjShine 1.15s linear infinite;}"
    "@keyframes rjShine{0%{background-position:120% 0;}100%{background-position:-120% 0;}}"
    "@media (prefers-reduced-motion:reduce){.rjsk .b{animation:none;}}"
    "</style>"
)
_BARS = ("<div class='rjsk'><span class='b' style='width:92%'></span><span class='b' style='width:78%'>"
         "</span><span class='b' style='width:85%'></span><span class='b' style='width:55%'></span></div>")

MIN_REDRAW = 0.06   # 重畫節流（秒）
THINK_REDRAW = 0.15
_TOOL_STAGE = {"web_search": "searching", "web_fetch": "fetching"}
_CLIENT_STAGE = {"get_weather": "cwa", "get_earthquake_info": "cwa", "get_typhoon_info": "cwa",
                 "read_file": "skill", "think": "reflect", "task": "searching"}
# 拿到資料就算「查到」的自訂工具（回答裡說「查到」是實話，見 mdfix.strip_claims）。
# 氣象署工具不列：它們有 evidence 事件，成功與否看那邊
_LOOKUP_TOOLS = {"get_natal_chart", "get_astro_forecast", "get_synastry", "get_astro_reference"}
# 狀態列台詞多久換一句（秒）。背景執行緒不能動 st.*，所以是主執行緒「有事件進來、或每 0.5 秒
# 的 pulse()」時順便檢查。
ROTATE_WAIT = 4.0
ROTATE_WRITING = 10.0
_WAIT_STAGES = {"received", "thinking", "deep_thinking", "searching", "fetching", "organizing", "skill",
                "reflect", "escalate", "socratic"}


def _step_detail(rec: dict) -> str:
    return "  \n".join(rec.get("detail") or [])


def render_widget(widget: dict) -> None:
    """互動元件（create_widget 送來的自包含 HTML）。

    一定要用 components.html（吃 HTML）；st.iframe 吃的是 URL——舊版 2026-09-05 換過一次，線上全部壞掉
    （見 Home.py render_widget_html 的註解）。wid／fp 注入狀態保存小工具，歷史回放沿用同一個 wid。
    """
    try:
        components.html(inject_state_helper(widget["html"], widget.get("wid"), widget.get("fp")),
                        height=int(widget.get("height") or 420), scrolling=True)
    except Exception as e:
        st.caption(f"⚠️ 互動元件顯示失敗：{type(e).__name__}")


def render_saved_status(header: str, state: str, steps: list[dict]) -> None:
    """歷史訊息重畫時用：收合的狀態列＋裡面的時間軸（不再有流光、不會輪替）。"""
    with st.status(header or "✅ 回答完成", type="compact", state=state or "complete", expanded=False):
        for rec in steps or []:
            h = st.status(rec.get("label") or "", type="step", state=rec.get("state") or "complete")
            detail = _step_detail(rec)
            if detail:
                h.caption(detail)


class TurnView:
    """一個回合的助理訊息區塊。用法：建立 → 逐一 on(ev) → finish(...)。"""

    def __init__(self, *, mode: str, user_text: str = "", serious: bool = False,
                 route_reason: str = "default_fast", started_at: float | None = None, docs_line: str = ""):
        """started_at：斷線接續時，原本那一輪開始的時間（「收到問題」和秒數從那時算）。
        docs_line：這一題帶著的上傳文件（寫在「收到問題」那一步）。"""
        self._at = started_at              # 正在處理的事件是什麼時候發生的；None＝現在（見 _now）
        self.mono = Monologue(user_text, serious=serious)
        self.tl = Timeline(route_reason=route_reason, mode=mode, clock=self._now,
                           extra=[docs_line] if docs_line else None)
        self.header = self.mono.opening()
        # compact：收合時只有一行心裡話；一般模式先收著（通常一兩秒就答完），深思模式一開始就展開時間軸
        self._expanded = mode != "fast"
        self.status = st.status(self.header, expanded=self._expanded, type="compact")
        self._steps: list[list] = []       # 每步一組 [status 容器, detail 佔位, label, state, detail, 攤開]
        self._tl_ver = -1
        self._last_step_draw = 0.0
        self._sync_steps()
        self.stage = "received"
        self._stage_t = time.time()
        self.badges_ph = st.empty()
        self.think_ph = st.empty()
        self.answer_ph = st.empty()
        self.banner_ph = st.empty()
        self.widget_ph = st.empty()
        self.widget: dict | None = None
        self._pulse_ph = st.empty()   # 看不見；pulse() 用它讓停止要求即時生效
        self.mode = mode
        self.buf = ""
        self._gap = False   # 剛經過工具（搜尋、讀網頁）：下一段文字要另起一段
        self.node = ""
        self.escalated = False
        self.escalate_reason = ""
        self.socratic = False
        self.research = False                    # Haiku 升級時要求走深度研究（階段 4）
        self.retried = False                     # 研究沒啟動、自動重跑過
        self.sources: list[dict] = []
        self.evidence: list[dict] = []
        self.usage = TurnUsage()
        self.answer_msg = None
        self.answer_model = ""
        self.header_state = "running"
        self.trace: list[tuple] = []
        self._last_draw = 0.0
        self._t0 = self._now()
        self._at = None
        self.first_text_s: float | None = None   # 首字時間（進 dev 與 telemetry；驗收看這個）
        self._await_text = False                 # 已先亮出「撰寫回答」、第一個字還沒到
        self.looked_up = False                   # 這回合真的查到資料（搜尋、讀網頁、氣象署、占星工具）
        self.research_urls: set[str] = set()     # 研究員搜尋結果與讀到的網頁（收尾時核對報告列的網址）
        self._show_wait()

    def _now(self) -> float:
        return time.time() if self._at is None else self._at

    @contextmanager
    def _clock_at(self, at: float | None):
        """這段期間時間軸用事件發生的時間算秒數，不用現在（斷線接續時重播的事件才不會全擠在同一刻）。"""
        self._at = at
        try:
            yield
        finally:
            self._at = None

    # ── 等待中 ──
    def _show_wait(self):
        self.think_ph.markdown(_WAIT_CSS + _BARS, unsafe_allow_html=True)

    # ── 事件 ──
    def on(self, ev: Ev, at: float | None = None) -> None:
        """at：事件在背景執行緒產生的時間（graph.stream_in_thread 記的）；沒給就當作現在。"""
        with self._clock_at(at):
            if ev.kind not in ("text", "thinking") or not self.trace or self.trace[-1][1] != ev.kind:
                # 事件順序（dev 面板看得到）：連續的 text／thinking 只記第一筆
                self.trace.append((round(self._now() - self._t0, 1), ev.kind, ev.data.get("name") or ev.node))
            self._on(ev)
        if ev.kind not in ("usage", "answer"):   # 這兩個在回合尾端，接著就是 finish，不必再換台詞
            self._rotate()   # 放在事件之後：剛因事件換階段的不會馬上又被輪替（第一個字到時會閃一下）
        self._sync_steps(throttle=ev.kind == "thinking")

    def _on(self, ev: Ev) -> None:
        k = ev.kind
        if k == "text":
            if ev.node and ev.node != self.node:   # 換了一個節點在說話（例如升級後的 Opus）：重新開始
                self.node, self.buf, self._gap = ev.node, "", False
            if self._gap and ev.text.strip():      # 工具前後的文字另起一段（和 events.answer_text 一致）
                self.buf += paragraph_break(self.buf)
                self._gap = False
            if self.first_text_s is None and ev.text.strip():
                self.first_text_s = round(self._now() - self._t0, 2)
            if (self.stage != "writing" or self._await_text) and ev.text.strip():
                if self.stage != "writing":
                    self.set_stage("writing")
                self._await_text = False
                self.tl.write()
                self._expand(False)            # 開始寫字就把時間軸收起來，不擋回答
            if ev.text.strip():
                self.tl.wrote()
            self.buf += ev.text
            self._draw()
        elif k == "thinking":
            self.tl.think(ev.text)
        elif k == "tool_start":
            self._gap = True
            name = ev.data.get("name")
            if name == "web_search":
                self.tl.search_start()
            elif name == "web_fetch":
                self.tl.fetch_start()
            else:                                  # code_execution 等：Opus 用程式包住搜尋、篩選結果
                self.tl.code_start()
            self.set_stage(_TOOL_STAGE.get(name, "organizing"))
            self._expand(True)
        elif k == "code_result":
            self.tl.code_done()
        elif k == "tool_call":                      # 階段 2：deep agent 的自訂工具
            name = ev.data.get("name") or ""
            if name == "escalate_to_deep":          # Haiku 的升級工具由 escalate 事件處理
                return
            self._gap = True
            self.tl.tool_call(name)
            self.set_stage(_CLIENT_STAGE.get(name, "organizing"))
            self._expand(True)
        elif k == "tool_args":
            self.tl.tool_args(ev.data.get("calls") or [])
        elif k == "tool_result":
            name, status, text = ev.data.get("name") or "", ev.data.get("status") or "success", ev.data.get("text") or ""
            self.tl.tool_result(name, status, text)
            if name in _LOOKUP_TOOLS and status != "error" and '"ok": false' not in text[:200]:
                self.looked_up = True
        elif k == "calc":
            self.tl.calc(ev.data.get("results") or [])
        elif k == "retry":                          # 研究沒啟動、自動重跑：清掉第一次的回答（偶發亂碼），等重跑的結果
            self.retried = True
            self.tl.retry(ev.data.get("reason") or "")
            self.buf, self._gap = "", False
            self.answer_ph.empty()
            self._show_wait()
            self.stage, self._stage_t = "deep_thinking", time.time()
            self._expand(True)
        elif k == "research_sources":               # 階段 4：研究員查了一輪（子代理內部，畫面只看得到進度）
            results = ev.data.get("results") or []
            self.research_urls.update(r["url"] for r in results if r.get("url"))
            self.looked_up |= bool(results)
            self.tl.research_progress(int(ev.data.get("searches") or 0) + int(ev.data.get("fetches") or 0),
                                      len(results))
        elif k == "reflect":
            self.tl.reflect(ev.data)
            if self.tl.current.kind == "write":     # 反思說直接作答：先換成寫字的台詞（帶引用的回答寫完才整段送來）
                self.set_stage("writing")
                self._await_text = True
        elif k == "rubric_start":                   # 寫完再檢查：回答留在畫面上，標題換成「檢查中」
            self.tl.rubric_start(ev.data.get("reason") or "")
            self.set_stage("reflect")
        elif k == "rubric":
            self.tl.rubric_done(ev.data)
            if ev.data.get("result") == "needs_revision":
                # 第一版沒過：清掉、等補寫（最終答案只取補寫後那一版）；攤開時間軸讓人看到要補什麼
                self.buf, self._gap = "", False
                self.answer_ph.empty()
                self._show_wait()
                self.stage, self._stage_t = "reflect", time.time()
                self._expand(True)
        elif k == "widget":                         # create_widget 通過檢查：主執行緒畫出來
            self.widget = {key: ev.data.get(key) for key in ("title", "html", "height", "wid", "fp")}
            self.tl.widget(self.widget.get("title") or "")
            with self.widget_ph.container():
                render_widget(self.widget)
        elif k == "evidence":                       # 氣象署工具的查證紀錄 → banner
            self._evidence(ev.data.get("tool") or "", ev.data.get("scope") or "", ev.data.get("status") or "",
                           ev.data.get("detail"))
        elif k == "search_results":
            results = ev.data.get("results") or []
            self.sources += results
            self.tl.search_done(results, ev.data.get("status", "error"))
            self._evidence("web_search", EV.SCOPE_WEB, ev.data.get("status", "error"),
                           {"n": len(results), "error": ev.data.get("error")})
        elif k == "fetch_result":
            self.tl.fetch_done(ev.data.get("url") or "", ev.data.get("status", "error"))
            self._evidence("web_fetch", EV.SCOPE_PAGE, ev.data.get("status", "error"),
                           {"url": ev.data.get("url"), "error": ev.data.get("error")})
        elif k == "citations":
            self.sources += [c for c in ev.data.get("items") or [] if c.get("url")]
        elif k == "escalate":
            self.escalated = True
            self.escalate_reason = ev.data.get("reason") or ""
            self.socratic = bool(ev.data.get("socratic"))
            self.research = bool(ev.data.get("research"))
            self.mode = "socratic" if self.socratic else "deep"
            self.tl.escalate(reason=self.escalate_reason, socratic=self.socratic)
            self.buf, self._gap = "", False
            self.answer_ph.empty()
            self._show_wait()
            self.stage, self._stage_t = ("socratic" if self.socratic else "escalate"), time.time()
            self._header(self.mono.escalate(socratic=self.socratic, reason=self.escalate_reason))
            self._expand(True)
        elif k == "usage":
            self.usage.add(ev.data.get("label", ""), ev.data.get("model", ""), ev.data.get("usage"),
                           ev.data.get("web", 0))
        elif k == "answer":
            self.answer_msg = ev.data.get("message")
            self.answer_model = ev.data.get("model", "")

    def pulse(self) -> None:
        """等模型的空檔（頁面每 0.5 秒呼叫一次）：台詞照常輪替，並送一個看不見的更新——
        Streamlit 只在 st.* 呼叫時檢查「停止」，沒有這一下，按停止要等到下一個事件才生效。"""
        self._rotate()
        try:
            self._pulse_ph.empty()
        except Exception:
            pass

    # 頁面在模型之前先查氣象署時用（at：斷線接續重畫時，原本查的時間）
    def cwa_start(self, at: float | None = None) -> None:
        with self._clock_at(at):
            self.tl.cwa_start()
        self.set_stage("cwa")
        self._expand(True)
        self._sync_steps()

    def cwa_done(self, results: list[dict], at: float | None = None) -> None:
        with self._clock_at(at):
            self.tl.cwa_done(results)
        self._sync_steps()

    def add_evidence(self, events: list[dict]) -> None:
        self.evidence += list(events or [])
        self.looked_up |= any(e.get("status") == EV.STATUS_OK for e in events or [])

    def _evidence(self, tool: str, scope: str, status: str, detail) -> None:
        self.looked_up |= status == EV.STATUS_OK
        try:
            self.evidence.append(EV.make_event(tool=tool, scope=scope, status=status, detail=detail))
        except Exception:
            pass   # 憑證記錄失敗不該連帶弄壞回答

    def set_stage(self, stage: str) -> None:
        """換階段就換一句台詞（Monologue 決定說什麼，這裡只管什麼時候說）。"""
        self.stage, self._stage_t = stage, time.time()
        self._header(self.mono.pick(stage))

    def _rotate(self) -> None:
        """同一階段待太久就換一句；剛收到問題太久沒動靜，就進入「思考中」。"""
        wait = ROTATE_WRITING if self.stage == "writing" else ROTATE_WAIT
        if time.time() - self._stage_t < wait:
            return
        if self.stage == "writing" or self.stage in _WAIT_STAGES:
            nxt = self.stage
            if self.stage in ("received", "escalate", "socratic"):
                nxt = "thinking" if self.mode == "fast" else "deep_thinking"
            self.set_stage(nxt)

    def _header(self, label: str, state: str = "running") -> None:
        self.header, self.header_state = label, state
        try:
            # expanded 一定要明給：1.63 實測只改 label 會把展開的狀態列收起來（4 秒一次的台詞輪替會一直收）
            self.status.update(label=label, state=state, expanded=self._expanded)
        except Exception:
            pass

    def _expand(self, on: bool) -> None:
        if on == self._expanded:
            return
        self._expanded = on
        try:
            self.status.update(expanded=on)
        except Exception:
            pass

    def _sync_steps(self, throttle: bool = False) -> None:
        """把 Timeline 的變動畫到時間軸上：新的步驟新增、改過的步驟更新。"""
        if self.tl.version == self._tl_ver:
            return
        now = time.time()
        if throttle and now - self._last_step_draw < THINK_REDRAW:
            return
        self._tl_ver, self._last_step_draw = self.tl.version, now
        try:
            # 最新一次寫好的反思攤開（看得到中文反思）；有新的反思時，舊的收起來
            last_reflect = max((i for i, s in enumerate(self.tl.steps)
                                if s.kind == "reflect" and s.state == "complete"), default=-1)
            for i, s in enumerate(self.tl.steps):
                detail = "  \n".join(s.detail)
                open_ = i == last_reflect
                if i < len(self._steps):
                    row = self._steps[i]
                    if (row[2], row[3], row[5]) != (s.label, s.state, open_):
                        row[0].update(label=s.label, state=s.state, expanded=open_)
                    if row[4] != detail and row[1] is not None:
                        row[1].caption(detail)
                    row[2:] = [s.label, s.state, detail, open_]
                else:
                    h = self.status.status(s.label, type="step", state=s.state, expanded=open_)
                    ph = None
                    if detail:   # 沒有內容的 step 會結束時間軸，所以只有最後一步可以空著
                        ph = h.empty()
                        ph.caption(detail)
                    self._steps.append([h, ph, s.label, s.state, detail, open_])
        except Exception:
            pass   # 時間軸畫壞不該連帶弄壞回答

    def _draw(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_draw < MIN_REDRAW:
            return
        self._last_draw = now
        shown = M.display_text(self.buf, final=False, looked_up=self.looked_up)
        if shown.strip():
            self.think_ph.empty()
            self.answer_ph.markdown(M.render_ready(shown), unsafe_allow_html=True)

    # ── 收尾 ──
    def finish(self, final_text: str, *, badges: str, banner: str, elapsed_s: float) -> None:
        self.close(elapsed_s)
        self.draw_finish(final_text, badges=badges, banner=banner)

    def close(self, elapsed_s: float) -> None:
        """收尾的資料部分（不碰 st.*）：時間軸加上「完成」、標題換成完成台詞。
        頁面先 close、存進歷史，再 draw_finish——畫到一半連線斷了，回答也已經存好。"""
        if self.tl.current.kind != "done":
            self.tl.finish(elapsed_s)
        self.header, self.header_state = self.mono.done(self.mode), "complete"

    def draw_finish(self, final_text: str, *, badges: str, banner: str) -> None:
        self.think_ph.empty()
        self.answer_ph.markdown(M.render_ready(final_text) if final_text else "安妮亞這次沒有拿到回答 🙏",
                                unsafe_allow_html=True)
        self.badges_ph.markdown(badges)
        if banner:
            self.banner_ph.caption(banner)
        self._sync_steps()
        self._expanded = False
        self._header(self.header, "complete")

    def fail(self, message: str) -> None:
        self.think_ph.empty()
        self.answer_ph.markdown(message)
        self.tl.fail()
        self._sync_steps()
        self._expanded = True   # 出錯時攤開時間軸，看得出卡在哪一步
        self._header(self.mono.failed(), "error")

    def saved_status(self) -> dict:
        """存進歷史的狀態列（重畫時用 render_saved_status）。"""
        out = {"status": self.header, "status_state": self.header_state, "steps": self.tl.records()}
        if self.widget:
            out["widget"] = self.widget
        return out

    def stopped_record(self, *, stopped: bool = True) -> dict:
        """使用者按了停止（或程式中途出了沒接住的錯）之後：不能再呼叫 st.*，只把已經收到的部分
        整理成歷史項目。"""
        if stopped:
            self.tl.stop()
        else:
            self.tl.fail()
        out = {"status": "⏹️ 已停止" if stopped else "⚠️ 這次沒有完成",
               "status_state": "complete" if stopped else "error", "steps": self.tl.records(),
               "text": M.display_text(self.buf, final=True, looked_up=self.looked_up), "mode": self.mode}
        if self.widget:
            out["widget"] = self.widget
        return out
