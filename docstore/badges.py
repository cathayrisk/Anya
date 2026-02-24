# badges.py
# -*- coding: utf-8 -*-
from __future__ import annotations

def _badge(label: str, color: str) -> str:
    safe = (label or "").replace("[", "(").replace("]", ")")
    return f":{color}-badge[{safe}]"

_MODE_DISPLAY = {
    "fast":     "⚡ **Fast Route**",
    "general":  "💬 **General Route**",
    "research": "🔬 **Research Route**",
}

def badges_markdown(
    *,
    mode: str,
    db_used: bool,
    web_used: bool,
    doc_calls: int = 0,
    web_calls: int = 0,
    elapsed_s: float | None = None,
) -> str:
    mode_norm = (mode or "").strip().lower()
    if mode_norm not in ("fast", "general", "research"):
        mode_norm = "general"

    items = [
        st.caption(_MODE_DISPLAY.get(mode_norm, f"{mode_norm.title()}")),
        _badge(f"DB:{doc_calls}" if db_used else "DB:off", "green" if db_used else "gray"),
        _badge(f"Web:{web_calls}" if web_used else "Web:off", "violet" if web_used else "gray"),
    ]
    if elapsed_s is not None:
        items.append(_badge(f"⏱ {elapsed_s:.1f}s", "gray"))
    return " ".join(items)
