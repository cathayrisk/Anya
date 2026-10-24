# -*- coding: utf-8 -*-
"""Anya（Claude 版）——2026-10 起和舊 Home.py 並存的新主聊天。

階段 1（骨架＋一般模式）：
- 第一層路由照搬 Home.py（anya_claude/router.py）；沒命中就交給 Haiku 5.5，
  Haiku 覺得要深入時呼叫 escalate_to_deep 工具，升級到 Opus 5.5（取代 [[ESCALATE]] 暗號）。
- 災防：分類 → controller prefetch（一定查、一定進 context、一定留憑證）→ 純即時題由程式直答。
- 真串流：直接串 markdown；誠實剝除每次重畫都套（anya_claude/mdfix.display_text）。
- 不設每日 Opus 上限（2026-10-23 使用者決定）；額度用完時畫面講人話（anya_claude/usage.py）。
- 互動（2026-10-23 使用者挑的 Streamlit 新功能）：狀態列＝心裡話＋步驟時間軸（view／monologue／timeline）、
  停止鈕（chat_input submit_mode="stop"，圖在背景執行緒跑、主執行緒每 0.5 秒 pulse 才停得即時）、
  最新回答的「⋯」選單（重答／換模式重答，舊回答收進「前一版回答」）、回答裡的 Mermaid 圖（mdfix）。
計畫：Anya-main(New)/anya-claude-改寫計畫-20261023.md。

⚠️ 分層：anya_claude/ 裡除了 view.py 都不准 import streamlit——圖的節點跑在背景執行緒。
"""
from __future__ import annotations

import base64
import json
import os
import queue
import sys
import time
import traceback
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Anya（Claude 版）", page_icon="🥜", layout="wide")

from anya_claude import events as E  # noqa: E402
from anya_claude import graph as G  # noqa: E402
from anya_claude import history as H  # noqa: E402
from anya_claude import mdfix as M  # noqa: E402
from anya_claude import router as R  # noqa: E402
from anya_claude import usage as U  # noqa: E402
from anya_claude.view import TurnView, render_saved_status, render_widget  # noqa: E402
from utils import evidence as EV  # noqa: E402
from utils import evidence_banner as EVB  # noqa: E402
from utils import hazard_prefetch as HZPF  # noqa: E402
from utils import hazard_render as HZR  # noqa: E402
from utils.cwa_weather import get_earthquake_impl, get_typhoon_impl  # noqa: E402
from utils.hazard_intent import classify_hazard_intent  # noqa: E402

try:
    from utils.rich_styles import inject_rich_styles
    inject_rich_styles()
except Exception:
    pass

DEV = str(st.query_params.get("dev", "")).strip() == "1"
AVATAR = {"fast": "⚡", "deep": "💬", "socratic": "🧭", "cwa": "🌐"}
MODE_LABEL = {"fast": "一般模式", "deep": "深思模式", "socratic": "提問引導", "cwa": "氣象署直答"}
IMAGE_TYPES = ["jpg", "jpeg", "png", "webp", "gif"]
PULSE_SECS = 0.5   # 等模型時多久碰一次畫面（「停止」的反應時間上限）
# 最新一則回答的「⋯」選單（2026-10-23 使用者要的：重答、換模式重答）
REDO_SAME, REDO_DEEP, REDO_FAST = "🔁 重新回答", "🧠 改用深思模式重答", "⚡ 改用一般模式重答"


def _secret_to_env(name: str) -> bool:
    """st.secrets → 環境變數（不印出）。圖在背景執行緒跑，工具一律從環境變數讀金鑰。"""
    if os.environ.get(name):
        return True
    try:
        val = st.secrets.get(name)
    except Exception:
        val = None
    if val:
        os.environ[name] = str(val)
        return True
    return False


def _ensure_api_key() -> bool:
    _secret_to_env("CWA_API_KEY")          # 階段 2 深思模式的氣象署工具用；沒有也不擋頁面
    _secret_to_env("SUPABASE_URL")         # 長期記憶（全體共用）；沒有就停用 save_lesson 與注入
    _secret_to_env("SUPABASE_KEY")
    return _secret_to_env("ANTHROPIC_API_KEY")


@st.cache_resource(show_spinner=False)
def get_graph():
    return G.build_graph()


def _pf_earthquake():
    o = get_earthquake_impl()
    # found=False 是「CWA 這次沒回傳可顯示的事件」，**不等於「近期沒有地震」**
    return json.dumps(o, ensure_ascii=False, default=str), (EV.STATUS_OK if o.get("found") else EV.STATUS_EMPTY)


def _pf_typhoon():
    o = get_typhoon_impl()
    # has_active_taiwan_warning 是明確布林，「目前無警報」是官方的有效答案
    return json.dumps(o, ensure_ascii=False, default=str), EV.STATUS_OK


def _render_message(msg: dict) -> None:
    if msg.get("superseded"):
        # 被「重答」取代的舊回答：收起來留著比對，不帶進之後的對話
        with st.expander(f"📜 前一版回答（{MODE_LABEL.get(msg.get('mode'), '')}）", expanded=False):
            st.markdown(M.render_ready(msg.get("text") or "（沒有內容）"), unsafe_allow_html=True)
        return
    if msg.get("steps"):
        render_saved_status(msg.get("status", ""), msg.get("status_state", "complete"), msg["steps"])
    if msg.get("badges"):
        st.markdown(msg["badges"])
    if msg.get("banner"):
        st.caption(msg["banner"])
    if msg.get("text"):
        st.markdown(M.render_ready(msg["text"]), unsafe_allow_html=True)
    if msg.get("widget"):
        render_widget(msg["widget"])   # 同一個 wid：使用者在元件裡的操作（sessionStorage）回放時對得上
    for name in msg.get("images") or []:
        st.caption(f"🖼 {name}")
    if msg.get("stopped") and msg.get("role") == "assistant":
        st.markdown(":gray-badge[⏹️ 已停止] :small[:gray[沒寫完的這則不會帶進之後的對話。]]")
    elif msg.get("pending") and not msg.get("stopped") and not msg.get("inflight"):
        st.markdown(":orange-badge[⏳ 沒有拿到回答] :small[:gray[這則不會帶進之後的對話。]]")
    if DEV and msg.get("dev"):
        with st.expander("🔧 [dev] 本回合", expanded=False):
            st.json(msg["dev"])


def _redo_options(msg: dict) -> list[str]:
    mode = msg.get("mode")
    if mode == "cwa":
        return [REDO_SAME]
    return [REDO_SAME, REDO_FAST if mode in ("deep", "socratic") else REDO_DEEP]


def _redo_menu(index: int, msg: dict) -> None:
    """只有最新一則回答有選單：重答會把它收進「前一版回答」，再用同一個提問重跑。
    key 用歷史索引——回合剛跑完時畫的、和下一輪歷史迴圈畫的是同一個元件。"""
    key = f"ac_redo_menu_{index}"
    st.menu_button("", options=_redo_options(msg), icon=":material/more_horiz:", type="tertiary",
                   key=key, help="重新回答／換模式重答", on_click=_on_redo, args=(key,))


def _on_redo(key: str) -> None:
    """menu_button 的 callback（在重跑之前執行）：舊回答標成被取代，下一輪重跑時重答。"""
    choice = st.session_state.get(key)
    kind = {REDO_SAME: "same", REDO_DEEP: "deep", REDO_FAST: "fast"}.get(choice)
    if not kind:
        return
    for h in reversed(st.session_state.get("ac_history", [])):
        if h.get("role") == "assistant" and not h.get("superseded"):
            h["superseded"] = True
            h["pending"] = True
            break
    st.session_state["ac_redo"] = kind


def _telemetry(rec: dict) -> None:
    try:
        print("TELEMETRY_CLAUDE " + json.dumps(rec, ensure_ascii=False, default=str), flush=True)
    except Exception:
        pass


# ── 連線中斷後接著答（2026-10-24）──
# 線上偶爾整題變成「⏹️ 已停止」，沒人按停止：Streamlit 在連線斷掉時會停掉正在跑的題目
# （websocket_session_manager.disconnect_session → request_script_stop），前端自動重連後同一個
# 工作階段重畫，就看到已停止（本機在題目中途關掉 websocket 可以重現）。Reboot 後第一題、跑很久的
# 題目最常碰到。圖本來就在背景執行緒跑，所以連線斷掉時讓它繼續跑，重連後接著畫：
# - 模型已經開始：已收到的事件重播一次，再接著讀還沒讀的（resume）
# - 模型還沒開始就斷了：重連後整題重跑（restart，最多 MAX_RESTARTS 次）
# 只有使用者自己按停止才真的停。2026-10-24 線上又抓到一次沒斷線也被停（流年題第 47 秒，連線沒換）：
# fastReruns 下任何重跑要求（同一條連線送來的、或伺服器自己發的）都會停掉正在跑的題目，觸發點查不到。
# 所以改成反過來判斷：在 Streamlit 處理「停止」按鈕的地方做記號，沒有記號的停止一律接著答。
MAX_RESTARTS = 2


@st.cache_resource(show_spinner=False)
def _user_stops() -> dict:
    """{session_id: 按停止的時間}。包一次 AppSession._handle_stop_script_request（使用者按停止鈕＝stop_script
    指令走這裡；斷線、重跑要求不走這裡）。私有函式，streamlit 釘在 <1.64，tests 盯著它還在。"""
    from streamlit.runtime.app_session import AppSession
    stops: dict = {}
    original = AppSession._handle_stop_script_request

    def marked(self):
        stops[self.id] = time.time()
        return original(self)

    AppSession._handle_stop_script_request = marked
    return stops


def _session_id() -> str:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx().session_id
    except Exception:
        return ""


def _stopped_by_user(ctx: dict) -> bool:
    """這次的停止是使用者按了停止鈕（回合開始後才按的）？記號拿不到時當作是，照舊停下來。"""
    stops, sid = ctx.get("user_stops"), ctx.get("sid")
    if stops is None or not sid:
        return True
    return stops.get(sid, 0) >= ctx["t0_local"]


def run_turn(user_entry: dict, text: str, images: list[dict], route: R.Route, hist: list[dict], *,
             allow_escalate: bool = True, resume: dict | None = None) -> None:
    """跑一回合（resume＝連線中斷後接著畫同一題）。使用者按「停止」時 Streamlit 在下一個 st.* 呼叫丟
    StopException（BaseException，1.63 實測 finally 會執行）：這時畫面已經不能動，只把收到的部分存進
    歷史、標成已停止。連線斷掉造成的停止則交給下一次重畫接著做（見上面）。"""
    user_entry["pending"] = True   # 答完才拿掉；中途失敗或按停止，這則提問就不帶進之後的對話
    ctx = {"view": None, "done": False, "t0": (resume or {}).get("t0") or time.time(), "t0_local": time.time(),
           "user_stops": _user_stops(), "sid": _session_id(), "box": st.session_state["ac_box"],
           "turn": {"user_entry": user_entry, "text": text, "images": images, "route": route,
                    "allow_escalate": allow_escalate, "restarts": (resume or {}).get("restarts", 0)}}
    ctx["box"]["active"] = True   # 回合進行中（含停止後的收尾）；見頁面開頭的等待
    try:
        if resume and resume.get("q") is not None:
            _resume_turn(ctx, resume, hist)
        else:
            _run_turn(ctx, user_entry, text, images, route, hist, allow_escalate)
    except BaseException as e:
        try:
            _on_turn_stopped(ctx, e, user_entry, route, hist)
        finally:
            ctx["box"].pop("active", None)
        raise
    ctx["box"].pop("active", None)


def _on_turn_stopped(ctx: dict, e: BaseException, user_entry: dict, route: R.Route, hist: list[dict]) -> None:
    """回合被中斷：連線斷掉的寄放起來等重連接續，其他的（按停止、出錯）照舊存成已停止。
    停止中不碰 st.*：寄放用回合開始時拿好的 dict。"""
    if ctx["done"]:
        return
    view = ctx["view"]
    stopped = type(e).__name__ == "StopException"
    if stopped and not _stopped_by_user(ctx):
        turn = ctx["turn"]
        if turn.get("q") is not None or turn["restarts"] < MAX_RESTARTS:
            if turn.get("q") is None:
                turn["restarts"] += 1
            ctx["box"]["turn"] = turn
            user_entry["inflight"] = True
            _telemetry({"route": route.reason, "interrupted": True, "resume": turn.get("q") is not None,
                        "secs": round(time.time() - ctx["t0"], 2)})
            return
    if view is not None:
        rec = view.stopped_record(stopped=stopped)
        hist.append({"role": "assistant", **rec, "pending": True, "stopped": True})
        user_entry["stopped"] = True
        _telemetry({"route": route.reason, "stopped": stopped,
                    "error": None if stopped else f"{type(e).__name__}: {str(e)[:300]}",
                    "secs": round(time.time() - ctx["t0"], 2), "usage": view.usage.as_rows()})


def _wait_for_previous_turn(box: dict, timeout: float = 3.0) -> None:
    """連線斷掉、前端重連時，Streamlit 會立刻開新的一輪，舊的一輪可能還在停止後的收尾（判斷是不是斷線、
    寄放題目）。先等它收完再畫，不然新的一輪看不到寄放的題目（本機重現過）。"""
    end = time.time() + timeout
    while box.get("active") and time.time() < end:
        time.sleep(0.05)
    box.pop("active", None)


def _resume_turn(ctx: dict, turn: dict, hist: list[dict]) -> None:
    """重連後接著畫同一題：同樣的狀態列，先重播已收到的事件，再接著讀背景執行緒還沒讀的。"""
    with st.chat_message("assistant", avatar=AVATAR[turn["start_mode"]]):
        # 重播用事件當時的時間：時間軸的秒數才會和原本那一輪一樣（不然全擠在重播那一刻，顯示「想了 0.0 秒」）
        view = TurnView(mode=turn["start_mode"], user_text=turn["text"], serious=turn["serious"],
                        route_reason=turn["route"].reason, started_at=turn["t0"])
        ctx["view"] = view
        if turn.get("pf_results"):
            pf_start, pf_end = turn.get("pf_at") or (turn["t0"], turn["t0"])
            view.cwa_start(at=pf_start)
            view.cwa_done(turn["pf_results"], at=pf_end)
            view.add_evidence(turn["pf_evidence"])
        for at, ev in turn["events"]:
            view.on(ev, at)
        turn["resumed"] = True
        ctx["turn"] = turn
        _consume_and_finish(ctx, view, turn, hist)


def _run_turn(ctx: dict, user_entry: dict, text: str, images: list[dict], route: R.Route, hist: list[dict],
              allow_escalate: bool) -> None:
    t0 = ctx["t0"]
    intent, pure, pure_why = None, False, ""
    try:
        # 翻譯請求不做災防分類（文章提到颱風不代表要查即時警報）；長訊息只看頭尾的指示部分
        if route.reason != "translation":
            inst = R.instruction_part(text)
            intent = classify_hazard_intent(inst)
            pure, pure_why = HZR.is_pure_live(inst, intent)
    except Exception:
        pass
    start_mode = "socratic" if route.socratic else route.mode
    # 災防題的狀態列用正經版台詞（問的人可能正在擔心，不開玩笑）
    serious = route.reason == "hazard_hint" or (intent is not None and intent.should_prefetch)
    with st.chat_message("assistant", avatar=AVATAR["cwa"] if pure else AVATAR[start_mode]):
        view = TurnView(mode=start_mode, user_text=text, serious=serious, route_reason=route.reason)
        ctx["view"] = view

        # ── 災防 controller prefetch：把「要不要查」從模型手上收回來 ──
        pf_results: list[dict] = []
        pf_evidence: list[dict] = []
        pf_at: tuple[float, float] | None = None
        block = ""
        if intent is not None and intent.should_prefetch:
            scopes = HZPF.prefetch_scopes(intent.scopes)
            if scopes:
                pf_start = time.time()
                view.cwa_start()
                pf_results = HZPF.run_prefetch(scopes, {EV.SCOPE_EARTHQUAKE: _pf_earthquake,
                                                        EV.SCOPE_TYPHOON: _pf_typhoon})
                view.cwa_done(pf_results)
                pf_at = (pf_start, time.time())   # 斷線接續重畫時照原本的時間畫
                pf_evidence = [EV.make_event(tool="prefetch/" + HZPF.SCOPE_TOOLS.get(r["scope"], r["scope"]),
                                             scope=r["scope"], status=r["status"],
                                             detail={"prefetch": True, "error": r["error"]})
                               for r in pf_results]
                view.add_evidence(pf_evidence)
                block = HZPF.build_context_block(pf_results)

        dev = {"route": route.reason, "start_mode": start_mode,
               "hazard": None if intent is None else {"state": intent.state, "scopes": list(intent.scopes),
                                                      "pure_live": pure, "pure_reason": pure_why,
                                                      "prefetched": [{k: r[k] for k in ("scope", "status", "error")}
                                                                     for r in pf_results]}}

        # ── 純即時災防題：程式直接渲染，不經模型 ──
        if pure and pf_results:
            payloads = {}
            for r in pf_results:
                if r.get("status") == EV.STATUS_OK and r.get("payload"):
                    try:
                        payloads[r["scope"]] = json.loads(r["payload"])
                    except Exception:
                        pass
            hr_text = HZR.render(pf_results, payloads=payloads)
            if hr_text:
                view.mode = "cwa"
                dev.update({"final_mode": "cwa", "escalated": False, "secs": round(time.time() - t0, 2), "usd": 0.0})
                badges = M.badges_markdown(mode="cwa", elapsed_s=time.time() - t0)
                view.close(time.time() - t0)       # 先存進歷史再畫：畫到一半斷線也不會丟
                hist.append({"role": "assistant", "text": hr_text, "mode": "cwa", "badges": badges,
                             "dev": dev, **view.saved_status()})
                ctx["done"] = True
                user_entry.pop("pending", None)
                # 自帶來源 footer，刻意不加 banner（沿用舊版）
                view.draw_finish(hr_text, badges=badges, banner="")
                _telemetry(dev)
                _redo_menu(len(hist) - 1, hist[-1])
                return
            # render 回空字串＝欄位長相變了，退回讓模型作答（資料仍在 context）

        # ── 模型回合：Haiku →（升級）→ Opus ──
        # 只帶這則提問之前的歷史（重答時，後面會有被取代的舊回答，那些不送）
        upto = next(i for i in range(len(hist) - 1, -1, -1) if hist[i] is user_entry)
        state = G.initial_state(H.to_messages(hist[:upto]) + [G.user_message(text, images)],
                                route=route.mode, socratic=route.socratic, context_block=block,
                                allow_escalate=allow_escalate)
        q, halt = G.stream_in_thread(get_graph(), state)
        ctx["turn"].update(q=q, halt=halt, events=[], start_mode=start_mode, serious=serious, dev=dev,
                           pf_results=pf_results, pf_evidence=pf_evidence, pf_at=pf_at, t0=t0)
        _consume_and_finish(ctx, view, ctx["turn"], hist)


def _consume_and_finish(ctx: dict, view: TurnView, turn: dict, hist: list[dict]) -> None:
    """讀背景執行緒的事件畫到畫面上，讀完收尾存進歷史。新回合和重連接續共用。"""
    q, halt, route, user_entry, dev, t0 = (turn["q"], turn["halt"], turn["route"], turn["user_entry"],
                                           turn["dev"], turn["t0"])
    try:
        while True:
            try:
                kind, payload = q.get(timeout=PULSE_SECS)
            except queue.Empty:
                view.pulse()   # 空檔也碰一下畫面：讓「停止」即時生效、台詞照樣輪替
                continue
            if kind == "end":
                break
            if kind == "error":
                raise payload
            at, item = payload              # at：事件在背景執行緒產生的時間
            for ev in E.parse_item(item):
                turn["events"].append((at, ev))   # 連線中斷後重畫時要重播
                view.on(ev, at)
    except Exception as e:
        halt.set()
        view.fail(U.friendly_error(e))
        ctx["done"] = True             # 提問維持 pending：沒答到的不帶進之後的對話（舊版「孤兒訊息污染」的教訓）
        user_entry.pop("inflight", None)
        _telemetry({"route": route.reason, "error": f"{type(e).__name__}: {str(e)[:300]}",
                    "usage": view.usage.as_rows()})
        if DEV:
            st.code(traceback.format_exc()[-3000:])
        return
    except BaseException:
        # 按了停止：背景執行緒不再往下讀。其他原因的停止不在這裡停——run_turn 寄放起來交給下一輪接續
        if _stopped_by_user(ctx):
            halt.set()
        raise

    if view.escalated and view.socratic:
        # Haiku 判定使用者「卡住需要引導」→ 進入 sticky 引導模式，之後的回合維持到退出詞（沿用舊版）
        st.session_state["ac_mode_sticky"] = "socratic"
    msg = view.answer_msg
    final = M.display_text(E.answer_text(msg), final=True, looked_up=view.looked_up)
    if not final.strip():
        view.fail("安妮亞這次沒有拿到回答 🙏 請再試一次。")
        ctx["done"] = True
        user_entry.pop("inflight", None)
        _telemetry({"route": route.reason, "error": "empty_answer", "usage": view.usage.as_rows()})
        return
    footer = M.build_web_sources_footer(view.sources + E.answer_citations(msg))
    # 只在「這回合真的查過東西」（網路搜尋、讀網頁、氣象署）時才標示；什麼都沒查的
    # 「💡 本回覆未經任何查證…」不顯示——2026-10-23 使用者決定，每則都掛很怪。
    banner = EVB.build_banner(view.evidence) if view.evidence else ""
    elapsed = time.time() - t0
    badges = M.badges_markdown(mode=view.mode, web_calls=view.usage.web_searches,
                               elapsed_s=elapsed, escalated=view.escalated)
    view.close(elapsed)   # 先存進歷史再畫：畫到一半連線斷了，重連後歷史裡就有完整回答
    dev.update({"final_mode": view.mode, "escalated": view.escalated,
                "escalate_reason": view.escalate_reason, "secs": round(elapsed, 2),
                "first_text_s": view.first_text_s, "resumed": bool(turn.get("resumed")),
                "usd": round(view.usage.usd, 5), "calls": view.usage.as_rows(),
                "evidence": EV.summarize(view.evidence), "trace": view.trace})
    hist.append({"role": "assistant", "text": final + footer, "mode": view.mode,
                 "badges": badges, "banner": banner, "dev": dev, **view.saved_status()})
    ctx["done"] = True
    user_entry.pop("pending", None)
    user_entry.pop("inflight", None)
    view.draw_finish(final + footer, badges=badges, banner=banner)
    _telemetry(dev)
    _redo_menu(len(hist) - 1, hist[-1])
    if DEV:
        with st.expander("🔧 [dev] 本回合", expanded=False):
            st.json(dev)


# ───────────────────────── 頁面 ─────────────────────────
st.markdown("### 🥜 安妮亞（Claude 版・測試中）")
st.caption("一般問題由 Haiku 5.5 快速回答；需要深入時會自動升級到 Opus 5.5。舊版主頁照常可用。")

if not _ensure_api_key():
    st.error("尚未設定 ANTHROPIC_API_KEY（Streamlit Cloud 的 Secrets），這一頁暫時不能用。")
    st.stop()

st.session_state.setdefault("ac_history", [])
st.session_state.setdefault("ac_mode_sticky", "assist")   # 'assist' | 'socratic'（狀態放頁面，不靠模型記得）
st.session_state.setdefault("ac_box", {})                 # 連線中斷時寄放還沒答完的那一題（run_turn）
history: list[dict] = st.session_state["ac_history"]
_wait_for_previous_turn(st.session_state["ac_box"])

# 圖（含 deepagents）在開頁時就建好：冷啟動要匯入一大堆套件，放到第一題才建，那段時間連線最容易斷
with st.spinner("安妮亞準備中…", show_time=True):
    get_graph()

# 輸入框先讀（它固定在頁面底部，呼叫順序不影響位置）：這一輪要跑新回合的話，舊的最後一則不畫選單，
# 不然新回答畫出自己的選單後，畫面上會同時有兩個「⋯」
# submit_mode="stop"：送出後按鈕變成停止鈕（1.59 起），深思模式跑太久可以自己喊停
prompt = st.chat_input("wakuwaku！輸入你的問題吧～", accept_file="multiple", file_type=IMAGE_TYPES,
                       submit_mode="stop")
redo = st.session_state.pop("ac_redo", None)

for i, m in enumerate(history):
    role = m.get("role", "assistant")
    with st.chat_message(role, avatar=AVATAR.get(m.get("mode")) if role == "assistant" else None):
        _render_message(m)
        if i == len(history) - 1 and role == "assistant" and not m.get("superseded") and not prompt:
            _redo_menu(i, m)

# 連線中斷時還沒答完的那一題：重連後接著畫；使用者已經改問別的就放棄它
inflight = st.session_state["ac_box"].pop("turn", None)
if inflight is not None:
    entry = inflight["user_entry"]
    if prompt or redo:
        if inflight.get("halt") is not None:
            inflight["halt"].set()
        entry.pop("inflight", None)
        entry["stopped"] = True
        rec = {"role": "assistant", "text": "", "mode": inflight["route"].mode, "pending": True, "stopped": True}
        history.append(rec)
        with st.chat_message("assistant", avatar=AVATAR.get(rec["mode"])):
            _render_message(rec)
    else:
        run_turn(entry, inflight["text"], inflight["images"], inflight["route"], history,
                 allow_escalate=inflight["allow_escalate"], resume=inflight)

sticky_drawn = st.session_state["ac_mode_sticky"]   # 模式列是在回合之前畫的（見檔尾）
if sticky_drawn == "socratic":
    left, right = st.columns([5, 1])
    left.markdown(":violet-badge[🧭 提問引導模式] "
                  ":small[:gray[安妮亞只提問不給答案，幫你自己想清楚；說「直接說」或按右邊按鈕結束。]]")
    if right.button("直接說", key="ac_socratic_exit", width="stretch"):
        st.session_state["ac_mode_sticky"] = "assist"
        st.rerun()

if prompt:
    text = (getattr(prompt, "text", None) if not isinstance(prompt, str) else prompt) or ""
    text = text.strip()
    images = []
    for f in (getattr(prompt, "files", None) or []):
        try:
            images.append({"name": f.name, "mime": f.type or "image/png",
                           "b64": base64.standard_b64encode(f.getvalue()).decode()})
        except Exception:
            pass
    if text or images:
        with st.chat_message("user"):
            if text:
                st.markdown(text)
            for im in images:
                st.image(base64.standard_b64decode(im["b64"]), caption=im["name"], width=220)
        entry = {"role": "user", "text": text or "（請看圖）", "raw_text": text,
                 "images": [im["name"] for im in images]}
        history.append(entry)
        st.session_state["ac_last_images"] = images   # 只留最新一則的圖（重答用），不累積
        sticky = R.next_socratic_state(text, st.session_state["ac_mode_sticky"])
        st.session_state["ac_mode_sticky"] = sticky
        route = R.decide(text, socratic_active=(sticky == "socratic"), has_images=bool(images))
        run_turn(entry, text, images, route, history)
elif redo:
    entry = next((h for h in reversed(history) if h.get("role") == "user"), None)
    if entry is not None:
        text = entry.get("raw_text", entry.get("text", ""))
        images = st.session_state.get("ac_last_images", []) if entry.get("images") else []
        entry.pop("stopped", None)
        if redo == "deep":
            route = R.Route("deep", "user_redo_deep")
        elif redo == "fast":
            route = R.Route("fast", "user_redo_fast")
        else:
            route = R.decide(text, socratic_active=(st.session_state["ac_mode_sticky"] == "socratic"),
                             has_images=bool(images))
        run_turn(entry, text, images, route, history, allow_escalate=(redo != "fast"))

# 這一輪進入或離開了引導模式：上面的模式列（徽章與「直接說」按鈕）是回合前畫的，重畫一次才對得上。
# 2026-10-24 實測：剛被判定要引導的那一輪看不到按鈕，要到下一輪才出現。
if st.session_state["ac_mode_sticky"] != sticky_drawn:
    st.rerun()
