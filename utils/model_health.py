# -*- coding: utf-8 -*-
"""模型健康度：連續 503 的模型暫時隔離，讓備援鏈自動跳過（純函式，無 streamlit 依賴）。

為什麼需要（2026-09-05 線上實測）：
`gemini-3.5-flash-lite` 連續 30 小時以上全部回 503「high demand」。它當時是
`general` 鏈的最後一格，於是 gemma 撞 16K TPM 降級後掉到它身上，而
`downgrade_model()` 因為「已經是最後一格」回 False → 退避階梯反覆重打同一顆死模型 →
使用者看到思考軌跡跑完、然後什麼都沒有。7 個回合有 5 個這樣靜默失敗。

`_mark_model_dead()` 不適用：那是給 404（模型 ID 不存在）用的**永久**標記。
503 是暫時的容量問題，模型隨時可能回來，所以要用「連續 N 次才隔離、隔離 M 秒後自動放行」。

狀態是可序列化的 dict（放 st.session_state）：
  {"strikes": {model: 連續 503 次數}, "until": {model: 隔離到期的 epoch 秒}}
"""
from __future__ import annotations

import time as _time

OVERLOAD_STRIKES = 3          # 連續幾次 503 才判定暫時不可用（單次尖峰不該讓模型下線）
OVERLOAD_QUARANTINE_SECS = 600  # 隔離多久後再試一次（Google 的 503 多為分鐘～小時級）


class ModelHealth:
    def __init__(self, state: dict):
        self.state = state
        state.setdefault("strikes", {})
        state.setdefault("until", {})

    def record_overload(self, model: str, now: float | None = None) -> bool:
        """記一次 503。回傳 True 表示這一次讓它進入隔離。"""
        now = _time.time() if now is None else now
        n = self.state["strikes"].get(model, 0) + 1
        self.state["strikes"][model] = n
        if n >= OVERLOAD_STRIKES:
            self.state["until"][model] = now + OVERLOAD_QUARANTINE_SECS
            return True
        return False

    def record_success(self, model: str) -> None:
        """成功一次就把連續計數歸零（隔離到期後的第一次成功也會清掉紀錄）。"""
        self.state["strikes"].pop(model, None)
        self.state["until"].pop(model, None)

    def is_quarantined(self, model: str, now: float | None = None) -> bool:
        now = _time.time() if now is None else now
        until = self.state["until"].get(model)
        if until is None:
            return False
        if now >= until:                      # 到期自動放行，並清掉計數重新觀察
            self.state["until"].pop(model, None)
            self.state["strikes"].pop(model, None)
            return False
        return True

    def quarantined_now(self, now: float | None = None) -> list[str]:
        return [m for m in list(self.state["until"]) if self.is_quarantined(m, now)]


def pick_model(chain: tuple[str, ...], tier: int, dead: set, health: ModelHealth,
               now: float | None = None) -> str:
    """從 tier 起往後找第一個「沒永久死、沒被隔離」的模型。

    全部都不可用時**回到鏈的最前面**找第一個沒永久死的——而不是沿用最後一格。
    理由：最後一格通常是 lite（被隔離代表它 503 中，再打也是白打），而第一格是
    gemma，它的 TPM 是分鐘窗、等一下就會恢復。回頭打 gemma 至少有機會成功。
    """
    if not chain:
        return ""
    for i in range(max(0, tier), len(chain)):
        m = chain[i]
        if m not in dead and not health.is_quarantined(m, now):
            return m
    for m in chain:                 # 全被隔離：退回第一個沒永久死的（通常是主力 gemma）
        if m not in dead:
            return m
    return chain[-1]
