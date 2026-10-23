# -*- coding: utf-8 -*-
"""頁面上的對話歷史 → LangChain messages。純函式，無 streamlit 依賴。

沿用現有做法（2026-10-23 使用者決定）：只用當次 session 的對話，每回合從存下來的
**純文字**重建。附帶的好處：不回傳 thinking block，就不會踩到 Claude 新模型
「改動前文會讓 thinking block 失效（400）」的規則。
"""
from __future__ import annotations

import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from . import config as CFG

_HTML_COMMENT_RE = re.compile(r"\n*<!--.*?-->", re.DOTALL)


def to_messages(history: list[dict], *, max_user_turns: int = CFG.HISTORY_USER_TURNS) -> list[BaseMessage]:
    """history 項目：{"role": "user"|"assistant", "text": str, ...}。

    - 只保留最近 max_user_turns 個使用者回合（從那則使用者訊息開始切，避免開頭是孤兒回答）
    - 空文字、尚未送達（pending）的訊息略過
    - 來源 footer 是系統附的，不送回模型（不然模型會學著自己寫來源清單）
    """
    items = [h for h in (history or []) if (h.get("text") or "").strip() and not h.get("pending")]
    user_idx = [i for i, h in enumerate(items) if h.get("role") == "user"]
    if len(user_idx) > max_user_turns:
        items = items[user_idx[-max_user_turns]:]
    out: list[BaseMessage] = []
    for h in items:
        text = _HTML_COMMENT_RE.sub("", h["text"]).strip()
        if h.get("role") == "user":
            out.append(HumanMessage(text))
        else:
            out.append(AIMessage(strip_sources_footer(text)))
    return out


_FOOTER_RE = re.compile(r"\n+---\n\*\*🌐 來源\*\*\n(?:- .*\n?)+\s*$")


def strip_sources_footer(text: str) -> str:
    return _FOOTER_RE.sub("", text or "").rstrip()
