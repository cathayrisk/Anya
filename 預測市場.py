# pages/預測市場.py
# -*- coding: utf-8 -*-
"""
Polymarket 財經向儀表板（官方 Gamma / CLOB REST）

設計原則：主題優先（topic-first）——Polymarket 全站成交量約四成是體育賽事，
但體育市場壽命只有數小時、走勢圖幾乎沒東西可看，對總體經濟／風險監控無用。
因此本頁只收財經相關主題（Finance / Geopolitics 為主），明確跳過 Sports 與 Culture。

資料來源與取數規則：
- 事件清單 → Gamma /events?tag_id=...&related_tags=true（一個 event 內嵌 markets[]，
  多結果事件天然聚合，不會像 /markets 那樣被同一事件的各選項洗版）。
- 機率 → 依 Polymarket 官方規則：預設取 (bestBid + bestAsk) / 2 的中價；
  只有買賣價差 > $0.10 時才退回 lastTradePrice。介面一律揭露價差與價格來源。
- 多結果事件的原始價格加總（overround）中位數實測 1.1，加總落在 NORMALIZE_BAND
  內才做乘法正規化到 1，並把原始加總當「隱含抽水」揭露；落在區間外代表這組結果
  不是互斥窮盡的，一律原樣顯示並在介面上警告，絕不硬算。
- 走勢 → CLOB /prices-history（interval 的 1m 是「近一個月」不是一分鐘，實測涵蓋 744 小時）。

實測校正（2026-08-30 於 Streamlit Cloud，見 pages/預測市場_診斷.py）：
- /events 的 limit 上限是 100，不是坊間教學說的 500；超過會靜默回 100 筆 + HTTP 200。
- 多結果事件的價格加總並非都接近 1（實測樣本低到 0.091），所以正規化必須設防呆區間。
- bestBid 覆蓋率 ~91%、bestAsk 100%、outcomePrices ~90%；價差 > $0.10 的 market 佔 ~21%。
- negRisk 欄位存在但**不能當判準**：覆蓋率 92%，且 126 個多結果事件裡有 8 個 negRisk=True
  卻加總落在區間外、6 個 negRisk=False 卻加總接近 1。維持以價格加總為準，negRisk 只作揭露。
- 多結果事件有 **51%** 加總落在區間外（「by date」階梯市場與複選市場），
  正規化防呆擋下的是一半的事件，不是零星例外。

已知環境限制：部分境內網路會把 *.polymarket.com 的 DNS 導向非官方 IP，
造成 TLS 自簽憑證錯誤。那是網路層攔截，不是程式問題——錯誤訊息會分類指出。
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# 可用環境變數覆寫：本機因 DNS 攔截連不到官方端點時能指向代理，
# 也讓 UI 能對著 mock server 被實際點過一遍（見 tests/ 與 scratchpad 的 mock）。
GAMMA = os.getenv("POLYMARKET_GAMMA_BASE", "https://gamma-api.polymarket.com").rstrip("/")
CLOB = os.getenv("POLYMARKET_CLOB_BASE", "https://clob.polymarket.com").rstrip("/")

TPE = ZoneInfo("Asia/Taipei")

# Polymarket 官方 constants/categories.ts 的 tag_id（polymarket.com 自己在用的那份）。
# 刻意不收 Sports(100639) 與 Culture(596)：本頁定位是財經／總體經濟。
CATEGORIES: list[tuple[str, tuple[int, ...]]] = [
    ("財經綜合", (120, 100265, 2, 21, 1401)),
    ("Finance", (120,)),
    ("Geopolitics", (100265,)),
    ("Politics", (2,)),
    ("Crypto", (21,)),
    ("Tech", (1401,)),
]
CATEGORY_MAP = dict(CATEGORIES)

RANGE_MAP = {
    "1H": "1h",
    "6H": "6h",
    "1D": "1d",
    "1W": "1w",
    "1M": "1m",   # 官方語意：近一個月
    "ALL": "max",
}

# 官方 UI 規則：買賣價差超過這個門檻才改看最後成交價。
# EPS 是必要的：0.55 - 0.45 在浮點下等於 0.10000000000000003，
# 沒有它，剛好 $0.10 的價差會被誤判成「寬價差」而退回最後成交價。
SPREAD_FALLBACK = 0.10
SPREAD_EPS = 1e-9

# 實測（2026-08-30，Streamlit Cloud）：/events 的 limit 上限是 100，不是坊間教學說的 500。
# 傳 500 或 1000 都靜默回 100 筆、HTTP 200，沒有任何「還有更多」的提示。
# offset 分頁實測可用（相鄰兩頁 id 無重疊），真要拿更多就得靠它。
LIMIT_CAP = 100

# 只有當多結果事件的原始價格加總落在這個區間，才把它當「互斥且窮盡」來正規化。
# 實測樣本裡有加總只有 0.091 的事件——那是把幾個獨立問題綁在一起的 event，
# 硬做正規化等於把每個價格乘以 11 倍，會產生看起來很正常的假機率。
NORMALIZE_BAND = (0.90, 1.30)

# 實測一個事件的完整 JSON 約 70KB（description／圖片 URL／解析規則佔大宗）。
# 財經綜合 5 個 tag × 100 筆 ≈ 35MB，每 60 秒重抓一次會把 Cloud 的記憶體吃光，
# 所以進快取前先砍成只留用得到的欄位。
EVENT_KEEP = ("id", "slug", "title", "volume24hr", "volume", "liquidity", "endDate", "negRisk")
MARKET_KEEP = (
    "outcomes", "outcomePrices", "clobTokenIds", "bestBid", "bestAsk",
    "lastTradePrice", "spread", "groupItemTitle", "question", "enableOrderBook",
    # 診斷頁倒出完整欄位後才發現這三個一直在，只是沒取：
    "acceptingOrders",        # 此刻能不能下單（enableOrderBook 只說市場型別支援掛單簿）
    "umaResolutionStatuses",  # 解析狀態註記＝解析風險，先前誤以為盤面看不到
    "oneMonthPriceChange",    # API 自帶的一個月價格變化，免費，不用打 prices-history
    # 實測 65% 的 market 收費（財經類佔最大宗），先前「Polymarket 零手續費」的認知已過時
    "feesEnabled", "feeType", "feeSchedule",
)

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


# -----------------------
# HTTP 層
# -----------------------
@st.cache_resource(show_spinner=False)
def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "Anya-PredictionMarket/2.0 (streamlit)"})
    return s


def _http_get_json(url: str, params: dict, timeout: int = 10, retries: int = 3, backoff: float = 0.7):
    """只對 429／5xx 與連線層錯誤重試；其餘 4xx 直接拋（重試 400 只是浪費三倍時間）。"""
    sess = _session()
    last_exc: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            r = sess.get(url, params=params, timeout=timeout)
        except requests.exceptions.RequestException as e:
            last_exc = e
            if attempt == retries:
                raise
            time.sleep(backoff * (2 ** (attempt - 1)))
            continue

        if r.status_code in _RETRYABLE_STATUS:
            if attempt == retries:
                r.raise_for_status()
            wait = backoff * (2 ** (attempt - 1))
            retry_after = r.headers.get("Retry-After")
            if retry_after:
                try:
                    wait = max(wait, min(float(retry_after), 10.0))
                except ValueError:
                    pass
            time.sleep(wait)
            continue

        r.raise_for_status()
        return r.json()

    raise last_exc if last_exc else RuntimeError("unreachable")


def explain_network_error(exc: Exception) -> str:
    """把例外翻成人看得懂的分類，不要全部混成一句『可能是 DNS 抖動』。"""
    if isinstance(exc, requests.exceptions.SSLError):
        return (
            "TLS 憑證驗證失敗。若你在境內網路，很可能是 *.polymarket.com 的 DNS "
            "被導向非官方 IP（可用 `nslookup gamma-api.polymarket.com 8.8.8.8` 對照確認）。"
            "這是網路層攔截，不是程式問題——部署在 Streamlit Cloud 上不受影響。"
        )
    if isinstance(exc, requests.exceptions.ConnectTimeout):
        return "連線逾時：連不上 Polymarket，請確認網路或稍後再試。"
    if isinstance(exc, requests.exceptions.ReadTimeout):
        return "讀取逾時：Polymarket 有回應但太慢，稍後再試或把主題範圍縮小。"
    if isinstance(exc, requests.exceptions.HTTPError):
        code = getattr(exc.response, "status_code", None)
        if code == 429:
            return "被 Polymarket 限流（HTTP 429）。等一分鐘再試，或降低查詢範圍。"
        if code and 500 <= code < 600:
            return f"Polymarket 伺服器錯誤（HTTP {code}），是對方的問題，稍後再試。"
        return f"HTTP {code}：請求被拒。可能是 API 參數或 schema 已變更。"
    if isinstance(exc, requests.exceptions.ConnectionError):
        return "連線失敗：DNS 解析或 TCP 連線出錯。"
    if isinstance(exc, ValueError):
        return "回應不是合法 JSON，Polymarket API 可能改版了。"
    return f"未預期的錯誤：{type(exc).__name__}: {exc}"


# -----------------------
# 資料抓取（快取只存成功結果；失敗讓例外往外拋，避免把空值鎖進快取）
# -----------------------
def slim_event(ev: dict) -> dict:
    """只留下渲染真的會用到的欄位，讓快取存的是 KB 級而不是 MB 級。"""
    out = {k: ev.get(k) for k in EVENT_KEEP if k in ev}
    out["markets"] = [
        {k: m.get(k) for k in MARKET_KEEP if k in m}
        for m in (ev.get("markets") or [])
        if isinstance(m, dict)
    ]
    return out


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_events_by_tag(gamma_base: str, tag_id: int, limit: int) -> list[dict]:
    params = {
        "limit": min(int(limit), LIMIT_CAP),
        "offset": 0,
        "order": "volume24hr",
        "ascending": "false",
        "closed": "false",
        "active": "true",
        "archived": "false",
        "tag_id": int(tag_id),
        "related_tags": "true",          # 不加的話撈不到子分類（如 Crypto 底下的 BTC / ETH）
    }
    data = _http_get_json(f"{gamma_base}/events", params=params)
    if not isinstance(data, list):
        return []
    return [slim_event(ev) for ev in data if isinstance(ev, dict)]


def fetch_events(tag_ids: tuple[int, ...], limit_per_tag: int = LIMIT_CAP) -> list[dict]:
    """多個 tag 合併去重。每個 tag 各自快取，切換主題時能重用。"""
    seen: set[str] = set()
    merged: list[dict] = []
    for tid in tag_ids:
        for ev in _fetch_events_by_tag(GAMMA, tid, limit_per_tag):
            if not isinstance(ev, dict):
                continue
            eid = str(ev.get("id") or ev.get("slug") or "")
            if not eid or eid in seen:
                continue
            seen.add(eid)
            merged.append(ev)
    return merged


def _prices_history_raw(clob_base: str, token_id: str, interval: str, fidelity: int) -> list[dict]:
    """給執行緒池用的純函式（不掛 st 裝飾器，避免 ScriptRunContext 警告）。"""
    j = _http_get_json(
        f"{clob_base}/prices-history",
        params={"market": token_id, "interval": interval, "fidelity": fidelity},
        timeout=10,
    )
    return j.get("history", []) if isinstance(j, dict) else []


@st.cache_data(ttl=30, show_spinner=False)
def prices_history(clob_base: str, token_id: str, interval: str, fidelity: int = 5) -> list[dict]:
    return _prices_history_raw(clob_base, token_id, interval, fidelity)


@st.cache_data(ttl=120, show_spinner=False)
def scan_histories(
    clob_base: str, token_ids: tuple[str, ...], interval: str, fidelity: int
) -> dict[str, list[dict]]:
    """併發抓多個 token 的走勢。單一 token 失敗不影響其他。"""
    out: dict[str, list[dict]] = {}
    if not token_ids:
        return out
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {
            ex.submit(_prices_history_raw, clob_base, t, interval, fidelity): t
            for t in token_ids
        }
        for fut in as_completed(futs):
            tid = futs[fut]
            try:
                out[tid] = fut.result()
            except Exception:
                out[tid] = []
    return out


# -----------------------
# 解析
# -----------------------
def as_list(x) -> list:
    """Gamma 很多欄位是 JSON 字串而不是 list（outcomes / outcomePrices / clobTokenIds）。"""
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        s = x.strip()
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return v
        except Exception:
            pass
        s = s.strip("[]")
        if not s:
            return []
        return [p.strip().strip('"').strip("'") for p in s.split(",")]
    return []


def to_float(x) -> float | None:
    try:
        if x is None or x == "":
            return None
        v = float(x)
        return None if pd.isna(v) else v
    except (TypeError, ValueError):
        return None


def month_change_pp(x) -> float | None:
    """
    API 自帶的一個月價格變化 → 百分點。

    單位尚未實測確認（診斷頁探針 8 會拿它跟真實一個月走勢對帳）。
    其他價格欄位都是 0–1，所以先假設它也是價格空間的差值；
    上面那道 1.5 的保險絲是防「萬一 API 給的已經是百分點」被誤放大 100 倍。
    """
    v = to_float(x)
    if v is None:
        return None
    return round(v * 100, 1) if abs(v) <= 1.5 else round(v, 1)


def uma_flag(m: dict) -> str | None:
    """解析狀態註記的原始序列，例如 proposed → disputed → proposed。"""
    v = m.get("umaResolutionStatuses")
    if v in (None, "", [], "[]"):
        return None
    items = as_list(v) if isinstance(v, str) else (v if isinstance(v, list) else [v])
    items = [str(x).strip() for x in items if str(x).strip()]
    return " → ".join(items) if items else None


def uma_level(m: dict) -> str | None:
    """
    把註記分級。實測值只有兩種（2026-08-30，1563 個 market）：
      proposed  解析提案中，正常流程的一部分（266 個）
      disputed  有人挑戰提案 → **真正的解析風險**（9 個，常見 proposed→disputed→proposed）
    """
    txt = uma_flag(m)
    if not txt:
        return None
    return "disputed" if "disputed" in txt.lower() else "proposed"


def fee_info(m: dict) -> str | None:
    """
    手續費摘要。實測 65% 的 market 收費，財經類（finance_prices_fees）佔最大宗。

    注意 `takerBaseFee` 在所有收費 market 上都是固定的 1000，但 feeSchedule.rate
    有 0.04 / 0.05 兩種——所以 takerBaseFee 不是實際費率，rate 才是。
    """
    if not m.get("feesEnabled"):
        return None
    sched = m.get("feeSchedule")
    if isinstance(sched, str):
        try:
            sched = json.loads(sched)
        except Exception:
            sched = None
    rate = to_float(sched.get("rate")) if isinstance(sched, dict) else None
    taker_only = bool(sched.get("takerOnly")) if isinstance(sched, dict) else False
    kind = str(m.get("feeType") or "").strip()
    if rate is None:
        return f"有手續費（{kind or '費率未知'}）"
    return f"{rate * 100:.1f}%{'（僅 taker）' if taker_only else ''}" + (f"｜{kind}" if kind else "")


def pick_price(m: dict, fallback: float | None = None) -> tuple[float | None, str, float | None]:
    """官方規則：預設中價；價差 > $0.10 才退回最後成交價。回傳 (價格, 來源, 價差)。"""
    bid = to_float(m.get("bestBid"))
    ask = to_float(m.get("bestAsk"))
    last = to_float(m.get("lastTradePrice"))

    if bid is not None and ask is not None and ask >= bid:
        spread = round(ask - bid, 4)
        if spread <= SPREAD_FALLBACK + SPREAD_EPS:
            return (bid + ask) / 2.0, "mid", spread
        if last is not None:
            return last, "last(價差寬)", spread
        return (bid + ask) / 2.0, "mid(價差寬)", spread

    # 實測 bestBid 只有 90% 覆蓋率（冷門選項沒人掛買單），但 Gamma 自己的 spread 欄位是 100%。
    # 算不出中價時，至少把價差顯示出來，不要讓這一欄空著。
    reported = to_float(m.get("spread"))
    reported = round(reported, 4) if reported is not None else None
    if last is not None:
        return last, "last(無雙邊報價)", reported
    if fallback is not None:
        return fallback, "outcomePrices", reported
    return None, "n/a", reported


def event_outcomes(ev: dict) -> tuple[pd.DataFrame, float | None, bool]:
    """
    把一個 event 攤成結果表。回傳 (表, 原始價格加總, 是否做了正規化)。

    - 單一 market 的事件 → 兩列（Yes / No）
    - 多 market 的事件（誰會當選這類）→ 每個 market 一列，取其 Yes 價
    加總即 overround，>1 的部分是價差造成的隱含抽水。
    """
    mkts = [m for m in (ev.get("markets") or []) if isinstance(m, dict)]
    rows: list[dict] = []

    if len(mkts) == 1:
        m = mkts[0]
        names = [str(x) for x in as_list(m.get("outcomes"))] or ["Yes", "No"]
        prices = [to_float(x) for x in as_list(m.get("outcomePrices"))]
        tokens = [str(x) for x in as_list(m.get("clobTokenIds"))]
        p0, src0, spread0 = pick_price(m, fallback=prices[0] if prices else None)
        for i, nm in enumerate(names):
            if i == 0:
                raw, src = p0, src0
            else:
                raw, src = (prices[i] if i < len(prices) else None), "outcomePrices"
            # 價差是「這個掛單簿」的屬性，不是某個結果的屬性——Yes 和 No 共用同一本簿子。
            # 之前只給 index 0，導致「領先結果是 No」的二元事件在列表上價差永遠顯示 None
            # （實測 6 個事件裡有 2 個中招，真實資料中凡是機率低於 50% 的市場都會）。
            spd = spread0
            rows.append({
                "outcome": nm, "raw": raw, "來源": src, "價差": spd,
                "token_id": tokens[i] if i < len(tokens) else None,
                "可交易": m.get("acceptingOrders"),
                "解析註記": uma_flag(m),
                "月變化_pp": month_change_pp(m.get("oneMonthPriceChange")) if i == 0 else None,
                "解析等級": uma_level(m),
                "手續費": fee_info(m),
            })
    else:
        for m in mkts:
            prices = [to_float(x) for x in as_list(m.get("outcomePrices"))]
            tokens = [str(x) for x in as_list(m.get("clobTokenIds"))]
            raw, src, spd = pick_price(m, fallback=prices[0] if prices else None)
            label = str(m.get("groupItemTitle") or m.get("question") or "?").strip()
            rows.append({
                "outcome": label, "raw": raw, "來源": src, "價差": spd,
                "token_id": tokens[0] if tokens else None,
                "可交易": m.get("acceptingOrders"),
                "解析註記": uma_flag(m),
                "月變化_pp": month_change_pp(m.get("oneMonthPriceChange")),
                "解析等級": uma_level(m),
                "手續費": fee_info(m),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df, None, False

    vals = pd.to_numeric(df["raw"], errors="coerce")
    total = float(vals.sum()) if vals.notna().any() else None
    df["原始_%"] = (vals * 100).round(1)

    # 只有加總落在合理區間，才把這組結果當成「互斥且窮盡」來正規化。
    # 實測有事件加總只有 0.091（幾個獨立問題被綁成一個 event），
    # 對它做正規化等於乘以 11 倍，會生出看起來很正常的假機率——寧可原樣呈現。
    lo, hi = NORMALIZE_BAND
    normalized = total is not None and lo <= total <= hi
    df["機率_%"] = (vals / total * 100).round(1) if normalized else df["原始_%"]
    return df, total, normalized


def is_binary_event(ev: dict) -> bool:
    """單一 market ＝ 二元事件（Yes / No 兩個結果共用一本掛單簿）。"""
    return len([m for m in (ev.get("markets") or []) if isinstance(m, dict)]) == 1


def primary_row(odf: pd.DataFrame, binary: bool) -> pd.Series | None:
    """
    挑出代表這個事件的那一列。

    二元事件固定取 Yes（第 0 列），**不是**取機率較高的那一邊——
    一個 3% 的尾部風險事件如果顯示成「No 97%」，語意整個反過來，
    而且列表裡會混雜 Yes 機率與 No 機率，沒辦法上下掃描比較。
    Polymarket 官方介面顯示的也是「3% chance」。
    多結果事件沒有 Yes/No 的概念，才取機率最高者。
    """
    if odf.empty:
        return None
    if binary:
        return odf.iloc[0]
    if odf["機率_%"].notna().any():
        return odf.loc[odf["機率_%"].idxmax()]
    return odf.iloc[0]


def worst_uma_level(odf: pd.DataFrame) -> str | None:
    """一個事件裡只要有任一結果被挑戰，整個事件就該標成 disputed。"""
    if odf.empty or "解析等級" not in odf.columns:
        return None
    levels = set(odf["解析等級"].dropna())
    if "disputed" in levels:
        return "disputed"
    return "proposed" if levels else None


def events_to_frame(events: list[dict]) -> pd.DataFrame:
    rows = []
    for ev in events:
        odf, total, normalized = event_outcomes(ev)
        binary = is_binary_event(ev)
        top = primary_row(odf, binary)
        top_label, top_prob, top_spread = None, None, None
        top_month, top_tradable = None, None
        if top is not None:
            top_label = str(top["outcome"])
            top_prob = to_float(top["機率_%"])
            top_spread = to_float(top["價差"])
            top_month = to_float(top.get("月變化_pp"))
            top_tradable = top.get("可交易")
        rows.append({
            "event_id": str(ev.get("id") or ev.get("slug") or ""),
            "title": str(ev.get("title") or ev.get("slug") or "(無標題)"),
            "二元": binary,
            "主要結果": top_label,
            "機率_%": top_prob,
            "月變化_pp": top_month,
            "可交易": top_tradable,
            "解析風險": worst_uma_level(odf),
            "手續費": (sorted({x for x in odf.get("手續費", pd.Series(dtype=object)).dropna()}) or [None])[0]
                      if not odf.empty else None,
            "價差": top_spread,
            "結果數": int(len(odf)),
            "overround": round(total, 4) if total else None,
            "已正規化": normalized,
            "volume24hr": to_float(ev.get("volume24hr")),
            "volume": to_float(ev.get("volume")),
            "liquidity": to_float(ev.get("liquidity")),
            "endDate": ev.get("endDate"),
            "slug": str(ev.get("slug") or ""),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["endDate_台北"] = df["endDate"].apply(to_taipei_date)
    return df


# -----------------------
# 走勢與統計
# -----------------------
def build_series(hist: list[dict]) -> pd.DataFrame:
    if not hist:
        return pd.DataFrame()
    hdf = pd.DataFrame(hist)
    if "t" not in hdf.columns or "p" not in hdf.columns:
        return pd.DataFrame()
    hdf["timestamp"] = pd.to_datetime(
        pd.to_numeric(hdf["t"], errors="coerce"), unit="s", utc=True
    ).dt.tz_convert(TPE)
    # prices-history 的 p 依官方文件就是 0–1，直接換算，不做啟發式猜測。
    hdf["prob_%"] = pd.to_numeric(hdf["p"], errors="coerce") * 100
    hdf = hdf.dropna(subset=["timestamp", "prob_%"]).sort_values("timestamp")
    return hdf[["timestamp", "prob_%"]]


def series_stats(series: pd.DataFrame) -> dict | None:
    """淨變化是方向性漂移；全距與標準差才是波動。三個都給，讓使用者自己選排序。"""
    if series.empty or len(series) < 2:
        return None
    p = series["prob_%"]
    return {
        "現值_%": round(float(p.iloc[-1]), 1),
        "淨變化_pp": round(float(p.iloc[-1] - p.iloc[0]), 1),
        "全距_pp": round(float(p.max() - p.min()), 1),
        "波動度_pp": round(float(p.std(ddof=0)), 2),
        "樣本數": int(len(p)),
    }


# -----------------------
# 格式化
# -----------------------
def to_taipei_date(v):
    if v is None:
        return None
    try:
        ts = pd.to_datetime(v, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.tz_convert(TPE).date()
    except Exception:
        return None


def compact_number(x, digits: int = 1) -> str:
    """
    5413482.27 -> 541.3萬；None/NaN -> N/A

    用萬／億而不是 K／M：表格欄位設了 format="compact"，Streamlit 會依語系
    在地化成「246萬」，指標若還用 K／M 就會出現同一頁兩套單位制。
    """
    v = to_float(x)
    if v is None:
        return "N/A"
    sign = "-" if v < 0 else ""
    v = abs(v)
    unit, scale = "", 1.0
    for u, sc in [("", 1.0), ("萬", 1e4), ("億", 1e8), ("兆", 1e12)]:
        unit, scale = u, sc
        if v < sc * 10_000:
            break
    val = v / scale
    if unit == "":
        return f"{sign}{val:,.0f}"
    return f"{sign}{val:.{digits}f}{unit}" if val < 100 else f"{sign}{val:.0f}{unit}"


# -----------------------
# 明細
# -----------------------
def render_event_detail(ev: dict, row: pd.Series, key_prefix: str) -> None:
    odf, total, normalized = event_outcomes(ev)

    with st.container(border=True):
        st.markdown(f"### {row.get('title', '')}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("主要結果" if not row.get("二元") else "結果", str(row.get("主要結果") or "N/A"))
        prob = to_float(row.get("機率_%"))
        c2.metric(
            "機率（正規化）" if normalized else "機率（原始價）",
            f"{prob:.1f}%" if prob is not None else "N/A",
        )
        c3.metric("24h 成交量", compact_number(row.get("volume24hr")))
        end_d = row.get("endDate_台北")
        c4.metric("結束（台北）", end_d.isoformat() if end_d else "N/A")

        # 解析風險與可交易性：先前的免責聲明說「解析風險不會反映在盤面上」，
        # 那是因為沒去取欄位，不是真的看不到。
        level = worst_uma_level(odf)
        notes = sorted({x for x in odf.get("解析註記", pd.Series(dtype=object)).dropna()})
        if level == "disputed":
            st.error(
                "🔴 **解析提案已被挑戰（disputed）**——有人對結果判定提出異議，最終結果可能與現在的"
                "盤面預期不同。這是實測樣本裡最罕見也最嚴重的狀態（1563 個 market 中僅 9 個）。"
                + ("　狀態序列：`" + "`、`".join(notes) + "`" if notes else "")
            )
        elif level == "proposed":
            st.warning(
                "🟡 **解析提案中（proposed）**——已有人提出結果判定，還在挑戰期。這是正常流程"
                "（實測 266 個 market 處於此狀態），但代表這個市場接近尾聲，價格已不太會動。"
            )

        fees = sorted({x for x in odf.get("手續費", pd.Series(dtype=object)).dropna()})
        if fees:
            st.info(
                "💰 這個市場收取手續費：" + "、".join(f"`{x}`" for x in fees) +
                "。下表的機率是**盤面價格、未扣費**——實際成交後的有效機率會更差。"
                "費用如何折進機率需要 feeSchedule 的完整結構（含 rebateRate），尚未實測，故本頁不做調整。"
            )
        if "可交易" in odf.columns and (odf["可交易"] == False).any():   # noqa: E712
            n_no = int((odf["可交易"] == False).sum())                   # noqa: E712
            st.info(f"ℹ️ {n_no}/{len(odf)} 個結果目前暫停接單（acceptingOrders=false）——價格看得到但成交不了。")

        # 實測 outcomePrices 覆蓋率只有 90%，所以「加總偏低」有兩種完全不同的成因：
        # 真的不互斥窮盡，或只是有幾個 market 缺價格。混為一談會冤枉後者。
        priced = int(odf["raw"].notna().sum())
        n_out = len(odf)
        neg = ev.get("negRisk")

        if total is None:
            pass
        elif normalized:
            st.caption(
                f"原始價格加總 **{total:.3f}** → 超出 1 的 {(total - 1) * 100:.1f}% 是價差造成的"
                "隱含抽水（overround）。下表「機率_%」已做乘法正規化。"
            )
            if neg is False:
                st.caption(
                    "註：這個事件的 `negRisk` 是 False，與「加總接近 1」的判斷不一致。"
                    "實測 negRisk 與價格加總在 126 個多結果事件裡有 14 個彼此矛盾，以價格加總為準。"
                )
        elif priced < n_out:
            st.warning(
                f"原始價格加總 **{total:.3f}**，但 {n_out} 個結果裡只有 {priced} 個抓得到價格"
                f"（缺 {n_out - priced} 個）——**加總偏低很可能是資料不全，不是結果不互斥**。"
                "無論何者都不該正規化，下表顯示原始價格。"
            )
        else:
            extra = ""
            if neg is True:
                extra = ("（註：這個事件的 `negRisk` 標為 True，與價格加總矛盾；"
                         "實測有 8 個事件如此，以價格加總為準。）")
            st.warning(
                f"原始價格加總 **{total:.3f}**，落在合理區間 "
                f"{NORMALIZE_BAND[0]}–{NORMALIZE_BAND[1]} 之外，且每個結果都有價格——"
                "這組結果**不是互斥且窮盡**的。常見於「by date」階梯市場"
                "（會不會在某日前發生）與複選市場。硬做正規化會生出假機率，"
                f"所以下表直接顯示原始價格，未做任何調整。{extra}"
            )

        if odf.empty:
            st.warning("這個事件沒有可解析的結果資料。")
            return

        st.dataframe(
            odf[["outcome", "機率_%", "原始_%", "月變化_pp", "價差", "來源", "可交易"]],
            hide_index=True,
            column_config={
                "outcome": st.column_config.TextColumn("結果", width="large"),
                "機率_%": st.column_config.NumberColumn("機率 %（正規化）", format="%.1f"),
                "原始_%": st.column_config.NumberColumn("原始 %", format="%.1f"),
                "月變化_pp": st.column_config.NumberColumn("月變化 pp", format="%+.1f"),
                "價差": st.column_config.NumberColumn("買賣價差", format="%.3f"),
                "來源": st.column_config.TextColumn("價格來源"),
                "可交易": st.column_config.CheckboxColumn("可交易"),
            },
        )

        tradable = odf[odf["token_id"].notna()]
        if tradable.empty:
            st.warning("這個事件沒有 clobTokenIds，畫不出走勢（可能不是掛單簿市場）。")
            return

        ctrl = st.columns([3, 2, 2])
        with ctrl[0]:
            labels = tradable["outcome"].tolist()
            pick_label = st.selectbox("看哪個結果的走勢", labels, key=f"{key_prefix}_outcome")
        with ctrl[1]:
            range_ui = st.segmented_control(
                "區間", list(RANGE_MAP.keys()), default="1M",
                key=f"{key_prefix}_range",
            ) or "1M"
        with ctrl[2]:
            fidelity = st.slider("fidelity（分鐘）", 1, 60, 10, 1, key=f"{key_prefix}_fid")

        token_id = str(tradable.loc[tradable["outcome"] == pick_label, "token_id"].iloc[0])

        try:
            hist = prices_history(CLOB, token_id, RANGE_MAP[range_ui], int(fidelity))
        except Exception as e:
            st.error(explain_network_error(e))
            return

        series = build_series(hist)
        if series.empty:
            st.warning("這個區間沒有足夠成交資料。試 ALL，或把 fidelity 調大。")
            return

        stats = series_stats(series)
        if stats:
            s1, s2, s3 = st.columns(3)
            s1.metric("區間淨變化", f"{stats['淨變化_pp']:+.1f} pp")
            s2.metric("區間全距", f"{stats['全距_pp']:.1f} pp")
            s3.metric("波動度（標準差）", f"{stats['波動度_pp']:.2f} pp")

        fig = px.line(series, x="timestamp", y="prob_%")
        fig.update_traces(line=dict(width=2, color="#AD4746"), hovertemplate="%{y:.1f}%<extra></extra>")
        fig.update_layout(
            template="plotly_white", height=420,
            margin=dict(l=40, r=20, t=10, b=40), hovermode="x", showlegend=False,
        )
        fig.update_yaxes(range=[0, 100], title="市場價格 (%)", ticksuffix="%",
                         showgrid=True, gridcolor="rgba(0,0,0,0.06)")
        fig.update_xaxes(title="", showgrid=False, hoverformat="%Y-%m-%d %H:%M（台北）")
        st.plotly_chart(fig)

        if row.get("slug"):
            st.caption(f"原始頁面：https://polymarket.com/event/{row['slug']}")


# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="Polymarket 財經儀表板", layout="wide", initial_sidebar_state="collapsed")
st.title("Polymarket 財經儀表板")
st.caption(
    "只收財經相關主題（Finance / Geopolitics 為主），刻意跳過 Sports 與 Culture——"
    "體育佔全站約四成成交量，但市場壽命只有數小時，對總體經濟觀察沒有訊號價值。"
)

top = st.columns([5, 2, 2])
with top[0]:
    cat_name = st.pills(
        "主題", options=[c for c, _ in CATEGORIES], selection_mode="single", default="財經綜合"
    ) or "財經綜合"
with top[1]:
    sort_by = st.selectbox("排序依據", ["volume24hr", "liquidity", "volume"], index=0)
with top[2]:
    min_liq = st.number_input("最低流動性（USD）", min_value=0, value=1000, step=500,
                              help="流動性太低的市場報價噪音大於訊號，預設濾掉。")

only_tradable = st.checkbox(
    "只看此刻可下單的市場（acceptingOrders）", value=True,
    help="關掉會一併顯示暫停接單的市場——那些價格看得到但成交不了。",
)

kw = st.text_input("關鍵字（事件標題包含）", "", placeholder="例：Fed、Taiwan、tariff")

try:
    events = fetch_events(CATEGORY_MAP[cat_name])
except Exception as e:
    st.error(explain_network_error(e))
    st.stop()

if not events:
    st.warning(f"「{cat_name}」目前沒有進行中的事件。換個主題試試。")
    st.stop()

events_by_id = {str(ev.get("id") or ev.get("slug") or ""): ev for ev in events}
df = events_to_frame(events)

if kw:
    df = df[df["title"].str.contains(kw, case=False, na=False)]
if min_liq > 0:
    df = df[df["liquidity"].fillna(0) >= float(min_liq)]

# 過濾一定要講出來砍了幾筆，不能靜默縮水
n_before_trade = len(df)
if only_tradable and "可交易" in df.columns:
    df = df[df["可交易"] != False]      # noqa: E712 — None（欄位缺）視為未知，保留
n_untradable = n_before_trade - len(df)

if df.empty:
    st.warning("篩選後沒有結果。放寬關鍵字或降低流動性門檻。")
    st.stop()

df = df.sort_values(sort_by, ascending=False, na_position="last").reset_index(drop=True)

tab_hot, tab_move = st.tabs(["熱門事件", "異常變動"])

# ---- Tab 1：熱門事件 ----
with tab_hot:
    msg = f"主題「{cat_name}」共 {len(df)} 個進行中事件。點一列看明細。"
    if n_untradable:
        msg += f" 另有 {n_untradable} 個暫停接單已濾掉（取消勾選上方選項可看）。"
    n_disputed = int((df["解析風險"] == "disputed").sum()) if "解析風險" in df.columns else 0
    n_fee = int(df["手續費"].notna().sum()) if "手續費" in df.columns else 0
    if n_disputed:
        msg += f" 🔴 其中 {n_disputed} 個的解析提案已被挑戰。"
    if n_fee:
        msg += f" {n_fee} 個收取手續費（顯示的機率未扣費）。"
    st.caption(msg)

    view = df.copy()
    view["解析風險"] = view["解析風險"].map({"disputed": "🔴 已被挑戰", "proposed": "🟡 解析中"})
    show = view[["title", "主要結果", "機率_%", "月變化_pp", "已正規化", "可交易",
                 "解析風險", "手續費", "價差", "結果數", "volume24hr", "liquidity", "endDate_台北"]]
    sel = st.dataframe(
        show.head(80),
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="hot_table",
        column_config={
            "title": st.column_config.TextColumn("事件", width="large"),
            "主要結果": st.column_config.TextColumn(
                "主要結果",
                help="二元事件固定顯示 Yes 的機率（不顯示 No），多結果事件顯示機率最高者。",
            ),
            "機率_%": st.column_config.NumberColumn("機率 %", format="%.1f"),
            "月變化_pp": st.column_config.NumberColumn(
                "月變化 pp", format="%+.1f",
                help="API 自帶的一個月價格變化，不需另外抓走勢。單位待探針 8 對帳確認。",
            ),
            "可交易": st.column_config.CheckboxColumn(
                "可交易",
                help="acceptingOrders：此刻能不能下單。enableOrderBook 只說市場型別支援掛單簿。",
            ),
            "解析風險": st.column_config.TextColumn(
                "解析風險",
                help="🔴 已被挑戰＝有人 dispute 解析提案，結果可能翻盤；🟡 解析中＝已提案，正常流程。",
            ),
            "手續費": st.column_config.TextColumn(
                "手續費",
                help="實測 65% 的 market 收費。顯示的機率是盤面價格，未扣費——實際成交的有效機率會更差。",
            ),
            "已正規化": st.column_config.CheckboxColumn(
                "已正規化",
                help="未打勾＝各結果加總落在合理區間外，很可能不是互斥窮盡的一組，顯示的是原始價格。",
            ),
            "價差": st.column_config.NumberColumn("價差", format="%.3f",
                                                 help="買賣價差。> 0.10 時價格改用最後成交價。"),
            "結果數": st.column_config.NumberColumn("結果數", format="%d"),
            "volume24hr": st.column_config.NumberColumn("24h 量", format="compact"),
            "liquidity": st.column_config.NumberColumn("流動性", format="compact"),
            "endDate_台北": st.column_config.DateColumn("結束（台北）"),
        },
    )

    # 表格是 canvas 畫的：合成點擊進不去，無障礙層的元素尺寸是 0×0，
    # 等於「點一列」這個主要互動既寫不了 e2e 測試、螢幕閱讀器也用不了。
    # 所以另外給一個純 DOM 的下拉當備援入口，兩者共用同一份選取狀態。
    NONE_LABEL = "（未選取）"
    titles = df["title"].head(80).tolist()
    options = [NONE_LABEL] + titles

    picked = sel.selection.rows if sel and sel.selection else []
    prev = st.session_state.get("_hot_prev_rows")
    if picked != prev:
        # 表格這一輪有動作 → 表格優先，並把下拉同步過去
        st.session_state["_hot_prev_rows"] = picked
        st.session_state["hot_pick"] = str(df.iloc[picked[0]]["title"]) if picked else NONE_LABEL
    if st.session_state.get("hot_pick") not in options:
        st.session_state["hot_pick"] = NONE_LABEL   # 換主題後舊標題可能已不存在

    chosen = st.selectbox(
        "或用下拉選取（表格點選在螢幕閱讀器與自動化環境下不可用）",
        options, key="hot_pick",
    )

    if chosen == NONE_LABEL:
        st.info("點選上表任一列，或用上面的下拉選單挑一個事件。")
    else:
        match = df[df["title"] == chosen].head(1)
        if match.empty:
            st.warning("找不到該事件，請重新選一次。")
        else:
            row = match.iloc[0]
            ev = events_by_id.get(row["event_id"])
            if ev is None:
                st.warning("找不到該事件的原始資料，請重新整理。")
            else:
                render_event_detail(ev, row, key_prefix="hot")

# ---- Tab 2：異常變動 ----
with tab_move:
    st.caption(
        "兩段式：先用 API 自帶的 `oneMonthPriceChange` 立刻排序（零請求），"
        "再只對你挑的前幾名抓走勢，算出該區間的全距與標準差。"
    )

    # ── 第一段：零請求的即時排行 ──
    quick = df[df["月變化_pp"].notna()].copy()
    if quick.empty:
        st.warning("這個主題的事件都沒有 oneMonthPriceChange，只能走第二段抓走勢。")
    else:
        quick["月變化絕對值"] = quick["月變化_pp"].abs()
        quick = quick.sort_values("月變化絕對值", ascending=False, na_position="last")
        n_missing = len(df) - len(quick)
        st.caption(
            f"即時排行：{len(quick)} 個事件有月變化資料"
            + (f"（另 {n_missing} 個沒有，未列入）。" if n_missing else "。")
        )
        st.dataframe(
            quick[["title", "主要結果", "機率_%", "月變化_pp", "可交易", "volume24hr"]].head(30),
            hide_index=True,
            column_config={
                "title": st.column_config.TextColumn("事件", width="large"),
                "機率_%": st.column_config.NumberColumn("機率 %", format="%.1f"),
                "月變化_pp": st.column_config.NumberColumn("月變化 pp", format="%+.1f"),
                "可交易": st.column_config.CheckboxColumn("可交易"),
                "volume24hr": st.column_config.NumberColumn("24h 量", format="compact"),
            },
        )

    st.markdown("---")
    st.markdown("**細部走勢**（只對前 N 名發請求）")

    m1, m2, m3, m4 = st.columns([1.6, 1, 1, 1.4])
    with m1:
        mv_range = st.segmented_control("區間", list(RANGE_MAP.keys()), default="1D", key="mv_range") or "1D"
    with m2:
        top_k = st.number_input("抓前幾名", min_value=3, max_value=40, value=10, step=1, key="mv_k")
    with m3:
        mv_fid = st.slider("fidelity", 1, 60, 10, 1, key="mv_fid")
    with m4:
        rank_by = st.selectbox("排序依據", ["全距_pp", "波動度_pp", "淨變化_pp（絕對值）"],
                               index=0, key="mv_rank")

    if st.button("抓走勢細節"):
        base_rank = quick if not quick.empty else df
        cands = base_rank.head(int(top_k))
        pairs: list[tuple[str, str, str]] = []
        skipped: list[tuple[str, str]] = []
        for _, r in cands.iterrows():
            ev = events_by_id.get(r["event_id"])
            if not ev:
                skipped.append((str(r["title"]), "找不到原始事件"))
                continue
            odf, _, _ = event_outcomes(ev)
            if odf.empty:
                skipped.append((str(r["title"]), "無可解析的結果"))
                continue
            valid = odf[odf["token_id"].notna()]
            if valid.empty:
                skipped.append((str(r["title"]), "沒有 clobTokenIds（非掛單簿市場）"))
                continue
            # 二元事件追 Yes：追 No 等於在監控一條鏡像曲線，漲跌方向剛好相反。
            best = primary_row(valid.reset_index(drop=True), bool(r.get("二元")))
            if best is None:
                skipped.append((str(r["title"]), "找不到可追蹤的結果"))
                continue
            pairs.append((str(best["token_id"]), r["event_id"], str(best["outcome"])))

        if not pairs:
            st.warning("這批事件沒有可用的掛單簿 token，換個主題或提高名次。")
        else:
            with st.spinner(f"併發抓取 {len(pairs)} 個走勢…"):
                try:
                    hist_map = scan_histories(
                        CLOB, tuple(p[0] for p in pairs), RANGE_MAP[mv_range], int(mv_fid)
                    )
                except Exception as e:
                    hist_map = {}
                    st.error(explain_network_error(e))

            results, empty_n = [], 0
            for token_id, event_id, label in pairs:
                stats = series_stats(build_series(hist_map.get(token_id, [])))
                if not stats:
                    empty_n += 1
                    continue
                b = df[df["event_id"] == event_id].head(1)
                results.append({
                    "event_id": event_id,
                    "title": b["title"].iloc[0] if not b.empty else event_id,
                    "結果": label,
                    **stats,
                    "月變化_pp": b["月變化_pp"].iloc[0] if not b.empty else None,
                    "volume24hr": b["volume24hr"].iloc[0] if not b.empty else None,
                })

            # 存進 session_state：不然一動下方任何 widget，button 就變 False，整塊結果會消失
            st.session_state["mv_results"] = results
            st.session_state["mv_empty_n"] = empty_n
            st.session_state["mv_skipped"] = skipped
            st.session_state["mv_scanned"] = len(cands)
            st.session_state["mv_at"] = datetime.now(TPE).strftime("%H:%M:%S")

    results = st.session_state.get("mv_results")
    if results is None:
        st.info("上面的即時排行不需要任何請求。要看區間內的全距與標準差，按「抓走勢細節」。")
    elif not results:
        st.warning("沒有算出結果——該區間成交太少，或 API 回空。試 ALL 區間。")
    else:
        res = pd.DataFrame(results)
        res["淨變化_pp（絕對值）"] = res["淨變化_pp"].abs()
        res = res.sort_values(rank_by, ascending=False, na_position="last")

        empty_n = st.session_state.get("mv_empty_n", 0)
        skipped = st.session_state.get("mv_skipped", [])
        scanned = st.session_state.get("mv_scanned", 0)
        st.caption(
            f"抓取於 {st.session_state.get('mv_at', '?')}（台北）｜範圍 {scanned} 個事件 → "
            f"成功 {len(results)}、無足夠成交資料 {empty_n}、無法抓取 {len(skipped)}。"
        )
        if skipped:
            with st.expander(f"被跳過的 {len(skipped)} 個事件（點開看原因）"):
                st.dataframe(pd.DataFrame(skipped, columns=["事件", "原因"]), hide_index=True)

        st.dataframe(
            res[["title", "結果", "現值_%", "月變化_pp", "淨變化_pp", "全距_pp",
                 "波動度_pp", "樣本數", "volume24hr"]].head(30),
            hide_index=True,
            column_config={
                "title": st.column_config.TextColumn("事件", width="large"),
                "現值_%": st.column_config.NumberColumn("現值 %", format="%.1f"),
                "月變化_pp": st.column_config.NumberColumn(
                    "月變化 pp", format="%+.1f",
                    help="API 自帶的一個月變化，與右邊實際抓來的區間統計可互相對照。"),
                "淨變化_pp": st.column_config.NumberColumn("淨變化 pp", format="%+.1f",
                                                        help="方向性漂移（末值 − 首值），不是波動。"),
                "全距_pp": st.column_config.NumberColumn("全距 pp", format="%.1f",
                                                       help="區間內最高 − 最低，衡量擺盪幅度。"),
                "波動度_pp": st.column_config.NumberColumn("標準差 pp", format="%.2f"),
                "樣本數": st.column_config.NumberColumn("點數", format="%d"),
                "volume24hr": st.column_config.NumberColumn("24h 量", format="compact"),
            },
        )

        pick_title = st.selectbox("看哪一個的明細", res["title"].tolist(), key="mv_pick")
        if pick_title:
            eid = res.loc[res["title"] == pick_title, "event_id"].iloc[0]
            b = df[df["event_id"] == eid].head(1)
            ev = events_by_id.get(eid)
            if ev is not None and not b.empty:
                render_event_detail(ev, b.iloc[0], key_prefix="mvdet")

st.divider()
st.caption(
    "⚠️ 市場價格不等於校準後的真實機率：預測市場普遍存在 favorite-longshot bias"
    "（低機率端系統性高估、高機率端低估，且離到期越遠越嚴重），本頁未做校準調整。"
    "另外解析風險不會反映在盤面價格上——本頁已把 UMA 解析狀態註記標示出來，"
    "但規則措辭的模糊性仍需自行到原始頁面判讀。"
)
st.caption("本頁僅為資訊呈現，不構成投資建議。")
