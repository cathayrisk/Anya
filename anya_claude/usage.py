# -*- coding: utf-8 -*-
"""每次模型呼叫的 token 與美元成本，以及 API 額度錯誤的辨識。純函式，無 streamlit 依賴。

2026-10-23 使用者決定不設每日 Opus 上限，唯一的防線是 Console 的 spend limit——
所以碰到額度錯誤時，畫面一定要講人話，不能丟 traceback（is_spend_limit_error）。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from . import config as CFG


@dataclass
class Call:
    label: str
    model: str
    input_tokens: int
    cache_read: int
    cache_write: int
    output_tokens: int
    web_searches: int
    usd: float


def cost_usd(model: str, usage: dict | None, web_searches: int = 0, *, cache_ttl: str = "1h") -> float:
    """usage = LangChain 的 usage_metadata（input_tokens 已含快取部分）。"""
    usage = usage or {}
    details = usage.get("input_token_details") or {}
    cache_read = int(details.get("cache_read") or 0)
    cache_write = int(details.get("cache_creation") or 0)
    total_in = int(usage.get("input_tokens") or 0)
    out = int(usage.get("output_tokens") or 0)
    base, w5, w1, rd, o = CFG.PRICES.get(model, CFG.PRICES[CFG.OPUS])
    write_price = w1 if cache_ttl == "1h" else w5
    plain = max(0, total_in - cache_read - cache_write)
    return (plain * base + cache_write * write_price + cache_read * rd + out * o) / 1e6 \
        + web_searches * CFG.WEB_SEARCH_USD


@dataclass
class TurnUsage:
    calls: list[Call] = field(default_factory=list)

    def add(self, label: str, model: str, usage: dict | None, web_searches: int = 0) -> Call:
        u = usage or {}
        d = u.get("input_token_details") or {}
        c = Call(label=label, model=model, input_tokens=int(u.get("input_tokens") or 0),
                 cache_read=int(d.get("cache_read") or 0), cache_write=int(d.get("cache_creation") or 0),
                 output_tokens=int(u.get("output_tokens") or 0), web_searches=int(web_searches or 0),
                 usd=cost_usd(model, u, web_searches))
        self.calls.append(c)
        return c

    @property
    def usd(self) -> float:
        return sum(c.usd for c in self.calls)

    @property
    def web_searches(self) -> int:
        return sum(c.web_searches for c in self.calls)

    def as_rows(self) -> list[dict]:
        return [asdict(c) for c in self.calls]


def is_spend_limit_error(exc: BaseException) -> bool:
    """Console spend limit（400：You have reached your specified API usage limits）或
    方案月上限（429：enforced_spend_limit_reached）。兩種都不是重試能解決的。"""
    msg = f"{type(exc).__name__}: {exc}"
    return ("usage limits" in msg) or ("enforced_spend_limit_reached" in msg) or ("credit balance" in msg)


def friendly_error(exc: BaseException) -> str:
    if is_spend_limit_error(exc):
        return ("安妮亞這個月的 API 額度用完了，暫時沒辦法回答 🥲\n\n"
                "請管理者到 Claude Console 調整用量上限；下個月額度會自動恢復。")
    name = type(exc).__name__
    if "RateLimit" in name or "Overloaded" in name or "529" in str(exc):
        return "Claude 現在有點忙（流量限制或伺服器過載），請過幾秒再試一次。"
    if "Timeout" in name or "Connection" in name:
        return "連線到 Claude 的時候逾時了，請再試一次。"
    return "安妮亞這次出錯了，沒有拿到回答 🙏 請再試一次；一直發生的話請告訴管理者。"
