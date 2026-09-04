# -*- coding: utf-8 -*-
"""PREMIUM 輪替池：把「每天只有 20 次」的 Gemini flash 從一顆變五顆（純函式，無 streamlit 依賴）。

2026-09-04 AI Studio rate-limit dashboard 實證：gemini-3-flash-preview / 3.5 / 3.6 / 3.7 / 3.8-flash
在免費層都是 **每天 20 次 + 每分鐘 5 次**，而且配額 **per model 各自獨立** → 輪著用合計 100 次/天。
RPD 在太平洋午夜重置（= 台灣 15:00），所以「用盡」要記到太平洋日，不是台灣日。

狀態是一個可序列化的 dict（放 st.session_state）：
  {"rr": 下一顆的索引, "exhausted": {model: 太平洋日字串}, "dead": [model, ...]}
呼叫端每次拿 candidates() 依序試，成功後 advance()；429 PerDay → mark_exhausted；404 → mark_dead。
"""
from __future__ import annotations

import datetime as _dt
from typing import Iterable

try:
    from zoneinfo import ZoneInfo
    _PACIFIC = ZoneInfo("America/Los_Angeles")
except Exception:  # 沒有 tzdata 的環境：退成固定 UTC-8（夏令時會差一小時，只影響重置時刻）
    _PACIFIC = _dt.timezone(_dt.timedelta(hours=-8))


def pacific_day(now: _dt.datetime | None = None) -> str:
    """RPD 配額的「今天」：太平洋時間的日期字串。naive datetime 視為 UTC。"""
    now = now or _dt.datetime.now(_dt.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=_dt.timezone.utc)
    return now.astimezone(_PACIFIC).date().isoformat()


def quota_scope(quota_info: dict | None) -> str | None:
    """從 telemetry.parse_quota_failure() 的輸出判斷 429 是日額（'day'）還是分鐘窗（'minute'）。
    抓不到明細回 None——呼叫端應保守當日額處理（RPD 型重打同一顆毫無意義）。"""
    for v in (quota_info or {}).get("quota_violations") or []:
        qid = str(v.get("quotaId") or "")
        if "PerDay" in qid:
            return "day"
        if "PerMinute" in qid:
            return "minute"
    return None


class PremiumPool:
    def __init__(self, models: Iterable[str], state: dict):
        self.models = tuple(models)
        self.state = state
        state.setdefault("rr", 0)
        state.setdefault("exhausted", {})
        state.setdefault("dead", [])

    def _live(self, model: str, day: str) -> bool:
        return model not in self.state["dead"] and self.state["exhausted"].get(model) != day

    def candidates(self, now: _dt.datetime | None = None) -> list[str]:
        """從輪替指標開始、繞一圈，剔除已死與今日已用盡的。空 list = 全池不可用。"""
        n = len(self.models)
        if not n:
            return []
        day = pacific_day(now)
        start = self.state["rr"] % n
        order = [self.models[(start + i) % n] for i in range(n)]
        return [m for m in order if self._live(m, day)]

    def advance(self, model: str) -> None:
        """成功用了 model → 下次從它的下一顆開始（讓五顆平均消耗，而不是把第一顆打到 20 才換）。"""
        if model in self.models:
            self.state["rr"] = (self.models.index(model) + 1) % len(self.models)

    def mark_exhausted(self, model: str, now: _dt.datetime | None = None) -> None:
        self.state["exhausted"][model] = pacific_day(now)

    def mark_dead(self, model: str) -> None:
        if model not in self.state["dead"]:
            self.state["dead"].append(model)

    def exhausted_today(self, now: _dt.datetime | None = None) -> list[str]:
        day = pacific_day(now)
        return [m for m in self.models if self.state["exhausted"].get(m) == day]
