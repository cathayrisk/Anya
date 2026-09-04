# -*- coding: utf-8 -*-
"""LLM 呼叫例外分類（純函式，無 streamlit 依賴，可離線測試）。

Home.py 的 invoke_with_backoff 原本把分類規則寫在 except 區塊裡，測不到。
這裡逐字搬出來，並新增 is_overloaded（503／UNAVAILABLE）：

2026-09-04 實測 gemini-3.5-flash-lite 連續 21 次 503「high demand」跨 8 分鐘，
而舊版只把 503 標成 retriable、不換模型——同一顆沿 BACKOFF_DELAYS 整條階梯重打
（每階內 SDK 還自帶 API_MAX_RETRIES 次重試）才放棄，備援鏈完全沒用上。
Google 對 503 的建議只有「稍後再試」；有備援時換一顆永遠比等快。

500／Internal error 刻意**不**列入 overloaded：原始碼註解記錄 gemma 池的 500 是
「Google 端暫時性、實測常見」的單次抖動，短退避重試同一顆通常就好，降級反而白白掉品質。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMErrorKind:
    is_quota: bool       # 429 / 配額：有備援直接換池，沒有就等官方建議秒數
    is_stuck: bool       # 逾時／連線中斷：再等也是白等，換模型
    is_overloaded: bool  # 503 / UNAVAILABLE：模型壅塞，換模型（新增）
    is_dead: bool        # 404 / 不支援：標記死亡，立刻換下一格
    retriable: bool      # 以上任一，或 500 類暫時性錯誤


def classify_llm_error(exc: BaseException) -> LLMErrorKind:
    name = type(exc).__name__
    msg = str(exc)
    low = msg.lower()
    is_quota = (
        "429" in msg
        or "ResourceExhausted" in name
        or "rate" in low
        or "quota" in low
        or "exhausted" in low
    )
    is_stuck = (
        "timeout" in low
        or "timed out" in low
        or "deadline" in low
        or "DeadlineExceeded" in name
        or name in ("TimeoutError", "ReadTimeout", "ConnectTimeout")
    )
    is_overloaded = (
        "503" in msg
        or "UNAVAILABLE" in msg
        or "ServiceUnavailable" in name
    )
    is_dead = (
        "404" in msg
        or "NOT_FOUND" in msg
        or "not found" in low
        or "is not supported" in low
    )
    retriable = (
        is_quota
        or is_stuck
        or is_overloaded
        or "500" in msg
        or "Internal error" in msg
        or "InternalServerError" in name
    )
    return LLMErrorKind(is_quota, is_stuck, is_overloaded, is_dead, retriable)
