# -*- coding: utf-8 -*-
"""跨 session 長期記憶（Supabase anya_lessons）。純 Python，無 streamlit 依賴——跑在背景執行緒。

2026-10-23 使用者決定：lessons_store 保留、全體共用一份（不做每人一份）。移植自 Home.py:2578：
- save_lesson 工具：每回合最多 2 次；成功後讓注入的快取失效
- 開場注入：最多 8 條，5 分鐘快取（不要每回合都打 Supabase）。深思模式放在每回合的系統訊息（graph.turn_note），
  不放固定系統提示——教訓會變，放系統提示會讓快取一直失效
- 搜尋教訓蒸餾：這回合 think 曾收到策略警告（卡關）、最後完整度回到 ≥70（解決了）→ 背景用 Haiku 萃取一條
  search_strategy 教訓寫進去；失敗靜默
金鑰：頁面啟動時把 st.secrets 的 SUPABASE_URL／SUPABASE_KEY 放進環境變數；沒有就整個停用（不註冊工具、不注入）。
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

MAX_PER_TURN = 2
INJECT_LIMIT = 8
CACHE_TTL_S = 300
DISTILL_MIN_CONFIDENCE = 70

_lock = threading.Lock()
_store: Any = None
_store_ready = False
_cache: tuple[float, list] | None = None


def get_store():
    """LessonsStore 或 None（沒設定 Supabase、套件缺、初始化失敗都回 None）。"""
    global _store, _store_ready
    with _lock:
        if not _store_ready:
            _store_ready = True
            url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY")
            if url and key:
                try:
                    from utils.lessons_store import LessonsStore
                    _store = LessonsStore(url, key)
                except Exception:
                    _store = None
        return _store


def enabled() -> bool:
    return get_store() is not None


def invalidate() -> None:
    global _cache
    with _lock:
        _cache = None


def block(now: float | None = None) -> str:
    """注入用的文字；沒啟用或沒資料回空字串。格式沿用 Home.py。"""
    global _cache
    store = get_store()
    if store is None:
        return ""
    t = time.time() if now is None else now
    with _lock:
        cached = _cache
    if not (cached and t - cached[0] < CACHE_TTL_S):
        try:
            rows = store.fetch(limit=INJECT_LIMIT) or []
        except Exception:
            rows = []
        cached = (t, rows)
        with _lock:
            _cache = cached
    rows = cached[1]
    if not rows:
        return ""
    lines = "\n".join(f"- [{r.get('category')}] {r.get('summary')}" for r in rows)
    return ("【長期記憶（過往 session 的教訓，開場自動載入）】\n" + lines +
            "\n（這些是過往經驗；與使用者當前明確指示衝突時，一律以當前指示為準。）")


@tool
def save_lesson(category: str, summary: str, content: str, runtime: ToolRuntime) -> str:
    """跨 session 長期記憶：記下「之後每次對話都仍然適用」的教訓或使用者偏好（存 Supabase，重開對話仍在，所有同事共用）。

    何時用：使用者糾正你的做法或表達偏好（category='user_pref'）、確認了某種固定作法（'workflow'）、
    你發現某領域的重要查證要點（'domain'）。
    何時不用：這次任務的一次性事實、對話歷史已記錄的內容、與使用者無關的常識。
    summary：一句話（≤60 字）：觸發情境＋該怎麼做。content：2–4 句補充為什麼。同主題已有記憶時會自動合併更新。
    """
    store = get_store()
    if store is None:
        return json.dumps({"error": "長期記憶未啟用（未設定 Supabase）"}, ensure_ascii=False)
    msgs = (runtime.state or {}).get("messages") if runtime else []
    saved = sum(1 for m in msgs or [] if isinstance(m, ToolMessage) and m.name == "save_lesson"
                and '"ok": true' in str(m.content))
    if saved >= MAX_PER_TURN:
        return json.dumps({"error": f"本回合 save_lesson 已達上限（{MAX_PER_TURN} 次）"}, ensure_ascii=False)
    try:
        out = store.save(category, summary, content)
    except Exception as e:
        out = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:120]}"}
    if out.get("ok"):
        invalidate()
    return json.dumps(out, ensure_ascii=False)


def _think_trail(messages: list) -> list[dict]:
    """這回合每次 think 的參數＋有沒有收到策略警告（從對話紀錄找，不靠 session_state）。"""
    hinted = {m.tool_call_id for m in messages or []
              if isinstance(m, ToolMessage) and m.name == "think" and "strategy_hint" in str(m.content)}
    trail = []
    for m in messages or []:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls or []:
                if tc.get("name") == "think":
                    a = tc.get("args") or {}
                    try:
                        conf = int(a.get("confidence", 0))
                    except (TypeError, ValueError):
                        conf = 0
                    trail.append({"confidence": conf, "key_finding": str(a.get("key_finding") or ""),
                                  "hint": tc.get("id") in hinted})
    return trail


def should_distill(messages: list) -> bool:
    trail = _think_trail(messages)
    return bool(trail) and any(t["hint"] for t in trail) and trail[-1]["confidence"] >= DISTILL_MIN_CONFIDENCE


def maybe_distill(messages: list, llm=None) -> bool:
    """卡關後又解決 → 背景萃取一條搜尋教訓。回傳有沒有啟動（測試用）。"""
    store = get_store()
    if store is None or not should_distill(messages):
        return False
    trail = _think_trail(messages)
    payload = "搜尋任務軌跡（過程曾卡關、最終解決）：\n" + "\n".join(
        f"- 反思{i + 1}（完整度 {t['confidence']}%）：{t['key_finding'][:150]}"
        + ("\n  ↳ 系統給了策略警告" if t["hint"] else "") for i, t in enumerate(trail))

    def work():
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            model = llm
            if model is None:
                from langchain_anthropic import ChatAnthropic

                from . import config as CFG
                model = ChatAnthropic(model=CFG.HAIKU, max_tokens=600, effort="low", thinking=CFG.HAIKU_THINKING)
            resp = model.invoke([SystemMessage(
                "你是教訓萃取員。根據一次搜尋任務的卡關與解決軌跡，萃取「下次遇到類似情況可直接套用」的通則教訓。"
                "第一行：一句話教訓（≤60 字，具體可執行）。第二行起：2–3 行說明原本卡在哪、後來怎麼解。"
                "若軌跡沒有可一般化的教訓，只輸出 NONE。正體中文、台灣用語。"), HumanMessage(payload)])
            text = resp.text if isinstance(resp.text, str) else str(resp.content)
            text = text.strip()
            if not text or text.upper().startswith("NONE"):
                return
            lines = [ln for ln in text.splitlines() if ln.strip()]
            store.save("search_strategy", lines[0].strip()[:120], "\n".join(lines[1:]).strip()[:800] or lines[0])
            invalidate()
        except Exception:
            pass   # 背景蒸餾失敗不影響回答

    threading.Thread(target=work, name="anya-lesson-distill", daemon=True).start()
    return True
