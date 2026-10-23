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
"""
from __future__ import annotations

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
    "socratic_mode": "提問引導模式中",
    "user_redo_deep": "你要求改用深思模式重答",
    "user_redo_fast": "你要求改用一般模式重答",
}
MAX_TITLES = 5
THINK_TAIL = 240
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


@dataclass
class Step:
    kind: str                    # received | cwa | search | fetch | code | escalate | think | write | done | error | stopped
    label: str                   # markdown；進行中的會包 :shimmer[]
    state: str = "running"       # running | complete | error
    detail: list[str] = field(default_factory=list)
    started: float = 0.0
    ended: float = 0.0
    calls: int = 0               # 搜尋／讀網頁的次數
    hits: int = 0                # 搜尋結果筆數
    text: str = ""               # 深思摘要累積

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

    # ── 模型 ──
    def escalate(self, *, reason: str, socratic: bool) -> None:
        label = "🧭 交給提問引導模式（Opus 5.5）" if socratic else "↗️ 升級到深思模式（Opus 5.5）"
        self.mode = "socratic" if socratic else "deep"
        self._finish_current()
        now = self.clock()
        self.steps.append(Step("escalate", label, "complete",
                               [_plain(reason, 80) or "Haiku 判斷這題需要深入處理"], started=now, ended=now))
        self._bump()

    def think(self, text: str) -> None:
        cur = self.current
        if cur.kind != "think" or cur.state != "running":
            prev = cur
            self._open(Step("think", "🧠 :shimmer[深思中…]", detail=["💭 …"]))
            cur = self.current
            cur.started = prev.ended or cur.started   # 模型在上一步結束後就開始想了
        cur.text += text or ""
        tail = re.sub(r"\s+", " ", cur.text).strip()
        if len(tail) > THINK_TAIL:
            tail = "… " + tail[-THINK_TAIL:]
        cur.detail = [f"💭 {_plain(tail, THINK_TAIL + 2)}"] if tail else ["💭 …"]
        self._bump()

    def write(self) -> None:
        if self.current.kind == "write" and self.current.state == "running":
            return
        self._open(Step("write", "✍️ :shimmer[撰寫回答…]",
                        detail=[f"由 {MODEL_NAME.get(self.mode, '模型')} 撰寫"]))

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

    def _open(self, step: Step) -> None:
        self._finish_current()
        step.started = self.clock()
        self.steps.append(step)
        self._bump()

    def _finish_current(self) -> None:
        cur = self.current
        if cur.state != "running":
            return
        took = _secs(self.clock() - cur.started)
        done = {"cwa": "🌐 查了氣象署即時資料", "search": self._search_label(cur),
                "fetch": self._fetch_label(cur), "code": "🧮 用程式整理了資料",
                "think": f"🧠 想了 {took}", "write": f"✍️ 寫好了・{took}"}
        cur.label = done.get(cur.kind, _unshimmer(cur.label))
        cur.state, cur.ended = "complete", self.clock()

    def _complete(self, step: Step) -> None:
        step.state, step.ended = "complete", self.clock()
        self._bump()

    def _bump(self) -> None:
        self.version += 1
