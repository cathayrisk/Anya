# badges.py
# -*- coding: utf-8 -*-
from __future__ import annotations

def _badge(label: str, color: str) -> str:
    safe = (label or "").replace("[", "(").replace("]", ")")
    return f":{color}-badge[{safe}]"

def badges_markdown(
    *,
    mode: str,
    db_used: bool,
    web_used: bool,
    doc_calls: int = 0,
    web_calls: int = 0,
) -> str:
    mode_norm = (mode or "").strip().lower()
    if mode_norm not in ("fast", "general", "research"):
        mode_norm = "general"

    mode_color = {"fast": "violet", "general": "blue", "research": "orange"}[mode_norm]
    items = [
        _badge(f"Mode:{mode_norm}", mode_color),
        _badge(f"DB:{doc_calls}" if db_used else "DB:off", "green" if db_used else "gray"),
        _badge(f"Web:{web_calls}" if web_used else "Web:off", "violet" if web_used else "gray"),
    ]
    return " ".join(items)
