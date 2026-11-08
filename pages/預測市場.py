# pages/預測市場.py
# -*- coding: utf-8 -*-
"""
Polymarket 財經向儀表板（官方 Gamma／Data API v2／CLOB REST）

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
- 走勢 → data-api 的 /v2/prices-history（2026-09-04 起官方以它取代 CLOB /prices-history）；
  新版失敗或回空才退回舊版 CLOB 路由，這次用的是哪一個在 ?dev=1 看得到。
  interval 的 1m 是「近一個月」不是一分鐘（舊版實測涵蓋 744 小時）。
- 市場代號 → 依 market 的 `version` 選：v2 用 `positionIds`（陣列），v1 用 `clobTokenIds`（JSON 字串）；
  兩欄可能同時有值，不能看哪欄有值來決定（官方 Protocol V2 遷移文件）。
- 未平倉／累計單邊成交／結算 → data-api 的 /v2/oi、/v2/live-volume、/v2/resolutions（一次最多 20 個代號）。
  /v2/live-volume 是**開市以來的累計** taker 成交量，不是 24 小時；24 小時成交量仍用 Gamma 的 volume24hr。

實測校正（2026-08-30 於 Streamlit Cloud，見 pages/預測市場_診斷.py）：
- /events 的 limit 上限是 100，不是坊間教學說的 500；超過會靜默回 100 筆 + HTTP 200。
- 多結果事件的價格加總並非都接近 1（實測樣本低到 0.091），所以正規化必須設防呆區間。
- bestBid 覆蓋率 ~91%、bestAsk 100%、outcomePrices ~90%；價差 > $0.10 的 market 佔 ~21%。
- negRisk 欄位存在但**不能當判準**：覆蓋率 92%，且 126 個多結果事件裡有 8 個 negRisk=True
  卻加總落在區間外、6 個 negRisk=False 卻加總接近 1。維持以價格加總為準，negRisk 只作揭露。
- 多結果事件有 **51%** 加總落在區間外（「by date」階梯市場與複選市場），
  正規化防呆擋下的是一半的事件，不是零星例外。

已知環境限制：*.polymarket.com 在台灣依法院命令封鎖（封鎖頁寫明涉違反公職人員選舉罷免法，
苗栗地院 115 年聲扣字第 4 號、南投地院 115 年度聲扣字第 5 號刑事裁定）：台灣網路的 DNS 會被導到
封鎖頁、TLS 憑證是自簽的。本機開發一律用測試資料；正式資料由部署在 Streamlit Cloud 的伺服器取得。
polymarket.com 的原始頁面在台灣也打不開，頁面上的連結只能當參考。
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

# 可用環境變數覆寫：讓 UI 能對著 mock server 被實際點過一遍（見 tests/ 與 scratchpad 的 mock）。
# 台灣網路依法院命令封鎖 *.polymarket.com：本機開發只用 mock，不要為了連線去繞過封鎖。
GAMMA = os.getenv("POLYMARKET_GAMMA_BASE", "https://gamma-api.polymarket.com").rstrip("/")
CLOB = os.getenv("POLYMARKET_CLOB_BASE", "https://clob.polymarket.com").rstrip("/")
DATA = os.getenv("POLYMARKET_DATA_BASE", "https://data-api.polymarket.com").rstrip("/")

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
    # 2026-11 Protocol V2：代號依 version 選（v2＝positionIds）；conditionId 給 Data API v2 查未平倉與結算
    "version", "positionIds", "conditionId",
)

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# 走勢（/v2/prices-history）：bucket_seconds 官方限 60–86400 秒，max／all／1m 至少 600、1w 至少 300；
# max／all 給太細的格子只會回最近 30 天，所以這兩個不給 bucket，讓服務自己選解析度。
HISTORY_WINDOW_S = {"1h": 3600, "6h": 21600, "1d": 86400, "1w": 604800, "1m": 2592000}
HISTORY_MIN_BUCKET_S = {"1w": 300, "1m": 600, "max": 600, "all": 600}
HISTORY_MAX_POINTS = 600     # 一條線大約不超過這麼多點，避免拖慢圖表
HISTORY_MAX_PAGES = 10       # 分頁上限，防止無限翻頁
# Data API v2 的 condition／event_id 參數一次最多 20 個不同的值（官方文件）
V2_SELECTOR_CAP = 20
RESOLUTION_KEEP = ("status", "expected_settlement_time", "settlement_time_basis",
                   "was_disputed", "resolution_source", "resolved_at")


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
            "TLS 憑證驗證失敗：*.polymarket.com 在台灣依法院命令封鎖，台灣網路的 DNS 會被導到"
            "封鎖頁、憑證是自簽的。在台灣本機開發連不到是預期的；部署在 Streamlit Cloud 上不受影響。"
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


# -----------------------
# 走勢：新版 /v2/prices-history 為主，舊版 CLOB 路由備援
# -----------------------
def history_bucket_seconds(interval: str, fidelity_min: int) -> int | None:
    """
    把 UI 的 fidelity（分鐘）換成新版的 bucket_seconds。

    - max／all：不給，讓服務自己選解析度（官方建議；給太細的格子只會回最近 30 天）
    - 其他：取「使用者選的 fidelity」「官方下限」「整段大約不超過 HISTORY_MAX_POINTS 點」
      三者的最大值，進位到整分鐘，上限 86400 秒。
    """
    iv = str(interval or "").strip().lower()
    if iv in ("max", "all"):
        return None
    b = max(60, int(fidelity_min or 1) * 60, HISTORY_MIN_BUCKET_S.get(iv, 60))
    win = HISTORY_WINDOW_S.get(iv)
    if win:
        b = max(b, -(-win // HISTORY_MAX_POINTS))
    b = -(-b // 60) * 60
    return int(min(b, 86400))


def _ts_seconds(v) -> int | None:
    """epoch 秒（或毫秒）、數字字串、ISO 字串 → epoch 秒；認不得回 None。"""
    if v is None or isinstance(v, bool) or v == "":
        return None
    if isinstance(v, (int, float)):
        x = float(v)
    else:
        s = str(v).strip()
        try:
            x = float(s)
        except ValueError:
            ts = pd.to_datetime(s, utc=True, errors="coerce")
            return None if pd.isna(ts) else int(ts.timestamp())
    if x != x:          # NaN
        return None
    if x > 1e12:        # 毫秒
        x /= 1000.0
    return int(x)


def parse_history_v2(payload) -> tuple[list[dict], str | None]:
    """
    /v2/prices-history 的回應 → ([{"t": epoch 秒, "p": 0–1}], 下一頁 cursor)。

    官方文件有兩種寫法：{data:[{timestamp, price}], pagination:{has_more, next_cursor}}
    與 {points:[...], next_cursor}（時間是 ISO 字串、價格是字串）。兩種都收，
    轉成和舊版一樣的 t／p，下游的 build_series 不用改。
    """
    if not isinstance(payload, dict):
        return [], None
    rows = payload.get("data")
    if not isinstance(rows, list):
        rows = payload.get("points") if isinstance(payload.get("points"), list) else []
    out: list[dict] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        t = _ts_seconds(r.get("timestamp", r.get("t", r.get("ts"))))
        p = to_float(r.get("price", r.get("p")))
        if t is None or p is None:
            continue
        out.append({"t": t, "p": p})
    pag = payload.get("pagination")
    cursor = None
    if isinstance(pag, dict):
        if pag.get("has_more", True):
            cursor = pag.get("next_cursor")
    else:
        cursor = payload.get("next_cursor")
    return out, (str(cursor) if cursor else None)


def _prices_history_v2_raw(data_base: str, token_id: str, interval: str, fidelity: int) -> list[dict]:
    params: dict = {"token_id": token_id, "interval": interval}
    bucket = history_bucket_seconds(interval, fidelity)
    if bucket is not None:
        params["bucket_seconds"] = bucket
    points: list[dict] = []
    cursor: str | None = None
    for _ in range(HISTORY_MAX_PAGES):
        q = dict(params)
        if cursor:
            q["cursor"] = cursor
        page, cursor = parse_history_v2(
            _http_get_json(f"{data_base}/v2/prices-history", params=q, timeout=10)
        )
        points.extend(page)
        if not cursor:
            break
    by_t = {pt["t"]: pt for pt in points}          # 分頁交界可能重複同一點
    return [by_t[t] for t in sorted(by_t)]


def _prices_history_legacy_raw(clob_base: str, token_id: str, interval: str, fidelity: int) -> list[dict]:
    """舊版 CLOB /prices-history（2026-09 起官方標為被取代，留作備援）。"""
    j = _http_get_json(
        f"{clob_base}/prices-history",
        params={"market": token_id, "interval": interval, "fidelity": fidelity},
        timeout=10,
    )
    return j.get("history", []) if isinstance(j, dict) else []


def _history_raw(data_base: str, clob_base: str, token_id: str, interval: str,
                 fidelity: int) -> tuple[list[dict], str]:
    """
    先新版、再舊版。回傳 (點, 來源)，來源是 "v2"、"legacy" 或 "empty"（兩邊都沒資料）。
    新版出錯而舊版沒救回來時，拋新版的例外（比較接近現況）；新版正常但沒資料時不拋，回空。
    給執行緒池用：不掛 st 裝飾器，避免 ScriptRunContext 警告。
    """
    v2_exc: Exception | None = None
    try:
        pts = _prices_history_v2_raw(data_base, token_id, interval, fidelity)
        if pts:
            return pts, "v2"
    except Exception as e:      # 新版路由或格式有變：記下來，改試舊版
        v2_exc = e
    try:
        pts = _prices_history_legacy_raw(clob_base, token_id, interval, fidelity)
        if pts:
            return pts, "legacy"
    except Exception:
        if v2_exc is not None:
            raise v2_exc
    if v2_exc is not None:
        raise v2_exc
    return [], "empty"


@st.cache_data(ttl=30, show_spinner=False)
def prices_history(data_base: str, clob_base: str, token_id: str, interval: str,
                   fidelity: int = 5) -> dict:
    """{"points": [...], "source": "v2"／"legacy"／"empty"}；失敗直接拋例外（快取不存失敗）。"""
    pts, src = _history_raw(data_base, clob_base, token_id, interval, fidelity)
    return {"points": pts, "source": src}


@st.cache_data(ttl=120, show_spinner=False)
def scan_histories(data_base: str, clob_base: str, token_ids: tuple[str, ...],
                   interval: str, fidelity: int) -> dict:
    """
    併發抓多個 token 的走勢。單一 token 失敗不影響其他。
    回傳 {"points": {token: [...]}, "sources": {"v2": 幾條, "legacy": 幾條, "error": 幾條, …}}。
    """
    points: dict[str, list[dict]] = {}
    sources: dict[str, int] = {}
    if not token_ids:
        return {"points": points, "sources": sources}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {
            ex.submit(_history_raw, data_base, clob_base, t, interval, fidelity): t
            for t in token_ids
        }
        for fut in as_completed(futs):
            tid = futs[fut]
            try:
                pts, src = fut.result()
            except Exception:
                pts, src = [], "error"
            points[tid] = pts
            sources[src] = sources.get(src, 0) + 1
    return {"points": points, "sources": sources}


# -----------------------
# Data API v2：未平倉、累計單邊成交、結算
# -----------------------
def _distinct(ids) -> list[str]:
    """去空白、去重、保留順序。"""
    out: list[str] = []
    for x in ids or ():
        s = str(x).strip() if x is not None else ""
        if s and s not in out:
            out.append(s)
    return out


def _chunks(xs: list[str], n: int):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


@st.cache_data(ttl=300, show_spinner=False)
def fetch_open_interest(data_base: str, condition_ids: tuple[str, ...]) -> dict[str, float]:
    """/v2/oi：每個 condition 的未平倉金額（美元）；沒人持有的官方回 0。"""
    out: dict[str, float] = {}
    for part in _chunks(_distinct(condition_ids), V2_SELECTOR_CAP):
        j = _http_get_json(f"{data_base}/v2/oi", params={"condition": ",".join(part)})
        for r in (j.get("data") if isinstance(j, dict) else None) or []:
            if isinstance(r, dict) and r.get("condition_id"):
                v = to_float(r.get("value"))
                if v is not None:
                    out[str(r["condition_id"])] = v
    return out


@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_volume(data_base: str, event_ids: tuple[str, ...]) -> dict[str, float]:
    """
    /v2/live-volume：每個 condition 開市以來的**累計**單邊（taker）成交量。
    官方明說是累計值，不是 24 小時；只算 taker 一邊，不會像雙邊加總那樣重複計算。
    event_id 只收正整數（全都不合法時官方回 400），先在這裡濾掉，全濾光就不發請求。
    """
    out: dict[str, float] = {}
    ids = [e for e in _distinct(event_ids) if e.isdigit()]
    for part in _chunks(ids, V2_SELECTOR_CAP):
        j = _http_get_json(f"{data_base}/v2/live-volume", params={"event_id": ",".join(part)})
        data = j.get("data") if isinstance(j, dict) else None
        for r in (data.get("conditions") if isinstance(data, dict) else None) or []:
            if isinstance(r, dict) and r.get("condition_id"):
                v = to_float(r.get("taker_volume"))
                if v is not None:
                    out[str(r["condition_id"])] = v
    return out


@st.cache_data(ttl=600, show_spinner=False)
def fetch_resolutions(data_base: str, condition_ids: tuple[str, ...]) -> dict[str, dict]:
    """/v2/resolutions：每個 condition 的結算狀態與預計結算時間（同一個 condition 只取第一列）。"""
    out: dict[str, dict] = {}
    for part in _chunks(_distinct(condition_ids), V2_SELECTOR_CAP):
        j = _http_get_json(f"{data_base}/v2/resolutions", params={"condition": ",".join(part)})
        for r in (j.get("data") if isinstance(j, dict) else None) or []:
            if isinstance(r, dict) and r.get("condition_id"):
                out.setdefault(str(r["condition_id"]),
                               {k: r.get(k) for k in RESOLUTION_KEEP if k in r})
    return out


def try_fetch(fn, *args):
    """(結果, None) 或 (None, 例外)：明細裡每一項各自失敗，不要一項壞掉整塊都不見。"""
    try:
        return fn(*args), None
    except Exception as e:
        return None, e


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


def outcome_ids(m: dict) -> list[str | None]:
    """
    依 market 的 version 取各結果的市場代號（官方 Protocol V2 遷移文件）：
      v2 → positionIds（已經是陣列）；v1 → clobTokenIds（JSON 字串，要先解析）。
    兩欄可能同時有值，**不能看哪欄有值來決定**。沒有 version 的舊資料照 v1 處理；
    其他版本官方說不支援，回空（畫面會說沒有可用代號）。
    官方的代號是十進位數字字串，但本頁只讀不下單：格式不認得也照收（交給 API 判斷），
    只有空的那一格放 None——位置要和 outcomes 對得上，不能往前擠。
    """
    ver = str(m.get("version") or "").strip().lower()
    if ver == "v2":
        raw = m.get("positionIds")
        ids = raw if isinstance(raw, list) else as_list(raw)
    elif ver in ("", "v1"):
        ids = as_list(m.get("clobTokenIds"))
    else:
        return []
    out: list[str | None] = []
    for x in ids:
        s = str(x).strip() if x is not None else ""
        out.append(s or None)
    return out


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
        tokens = outcome_ids(m)
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
            tokens = outcome_ids(m)
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


def event_market_facts(ev: dict, oi: dict | None, vol: dict | None, res: dict | None) -> dict:
    """
    把一個事件底下各 market 的未平倉、累計單邊成交、結算資訊合成事件層級的數字。
    某一項整個抓不到（None）或對不上任何 market 就回 None，畫面顯示「—」，不把缺資料算成 0。
    預計結算取所有 market 裡最早的那個，換成台北時間。
    """
    cids = [str(m.get("conditionId")) for m in (ev.get("markets") or [])
            if isinstance(m, dict) and m.get("conditionId")]

    def _sum(mp):
        if not mp:
            return None
        vals = [mp[c] for c in cids if c in mp]
        return float(sum(vals)) if vals else None

    settle_ts: list[int] = []
    statuses: list[str] = []
    disputed = False
    for c in cids:
        r = (res or {}).get(c)
        if not isinstance(r, dict):
            continue
        t = _ts_seconds(r.get("expected_settlement_time"))
        if t:
            settle_ts.append(t)
        s = str(r.get("status") or "").strip()
        if s and s not in statuses:
            statuses.append(s)
        disputed = disputed or bool(r.get("was_disputed"))
    settle = None
    if settle_ts:
        settle = datetime.fromtimestamp(min(settle_ts), TPE).strftime("%Y-%m-%d %H:%M")
    return {
        "open_interest": _sum(oi),
        "taker_volume": _sum(vol),
        "expected_settlement": settle,
        "statuses": statuses,
        "was_disputed": disputed,
        "n_markets": len(cids),
    }


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

        # 新版 Data API（2026-11 接上）：未平倉、累計單邊成交、預計結算。
        # 三項各自抓，任一項失敗只顯示「—」，不影響下面的內容。
        cids = tuple(str(m.get("conditionId")) for m in (ev.get("markets") or [])
                     if isinstance(m, dict) and m.get("conditionId"))
        eid = str(ev.get("id") or "")
        oi_map, oi_err = try_fetch(fetch_open_interest, DATA, cids) if cids else ({}, None)
        vol_map, vol_err = try_fetch(fetch_live_volume, DATA, (eid,)) if eid.isdigit() else ({}, None)
        res_map, res_err = try_fetch(fetch_resolutions, DATA, cids) if cids else ({}, None)
        facts = event_market_facts(ev, oi_map, vol_map, res_map)
        g1, g2, g3 = st.columns(3)
        oi_v, vol_v = facts["open_interest"], facts["taker_volume"]
        g1.metric("未平倉量", f"US${compact_number(oi_v)}" if oi_v is not None else "—",
                  help="押在這個事件上、還沒結算的金額（/v2/oi）。比成交量更能代表大家實際押了多少。")
        g2.metric("累計單邊成交", f"US${compact_number(vol_v)}" if vol_v is not None else "—",
                  help="開市以來的 taker 成交量（/v2/live-volume）：只算一邊、不會重複計算；不是 24 小時的量。")
        g3.metric("預計結算（台北）", facts["expected_settlement"] or "—",
                  help="官方預計的結算時間（/v2/resolutions），可能晚於上面的「結束」日期。")
        if facts["was_disputed"]:
            st.caption("結算紀錄（/v2/resolutions）顯示這個事件曾被提出異議。")
        if DEV:
            errs = {k: explain_network_error(e) for k, e in
                    (("未平倉", oi_err), ("累計成交", vol_err), ("結算", res_err)) if e is not None}
            st.caption(f"Data API v2：{facts['n_markets']} 個 condition｜狀態 {facts['statuses'] or '—'}"
                       + (f"｜錯誤 {errs}" if errs else ""))

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
            st.warning("這個事件沒有可用的市場代號，畫不出走勢（可能不是掛單簿市場，或新版代號還沒提供）。")
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
            hist = prices_history(DATA, CLOB, token_id, RANGE_MAP[range_ui], int(fidelity))
        except Exception as e:
            st.error(explain_network_error(e))
            return

        series = build_series(hist.get("points", []))
        if DEV:
            st.caption(f"走勢來源：{hist.get('source')}｜{len(hist.get('points', []))} 點｜代號 {token_id[:10]}…")
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
            st.caption(f"原始頁面：https://polymarket.com/event/{row['slug']}"
                       "（polymarket.com 在台灣依法院命令封鎖，台灣網路打不開）")


# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="Polymarket 財經儀表板", layout="wide", initial_sidebar_state="collapsed")
DEV = str(st.query_params.get("dev", "")).strip() == "1"   # ?dev=1：顯示走勢來源與 Data API 診斷
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
                skipped.append((str(r["title"]), "沒有可用的市場代號（非掛單簿市場，或新版代號還沒提供）"))
                continue
            # 二元事件追 Yes：追 No 等於在監控一條鏡像曲線，漲跌方向剛好相反。
            best = primary_row(valid.reset_index(drop=True), bool(r.get("二元")))
            if best is None:
                skipped.append((str(r["title"]), "找不到可追蹤的結果"))
                continue
            pairs.append((str(best["token_id"]), r["event_id"], str(best["outcome"])))

        if not pairs:
            st.warning("這批事件沒有可用的市場代號，換個主題或提高名次。")
        else:
            with st.spinner(f"併發抓取 {len(pairs)} 個走勢…"):
                try:
                    scanned = scan_histories(
                        DATA, CLOB, tuple(p[0] for p in pairs), RANGE_MAP[mv_range], int(mv_fid)
                    )
                    hist_map = scanned.get("points", {})
                    st.session_state["mv_sources"] = scanned.get("sources", {})
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
    "但規則措辭的模糊性仍需自行判讀（polymarket.com 原始頁面在台灣依法院命令封鎖，打不開）。"
)
st.caption("本頁僅為資訊呈現，不構成投資建議。")

if DEV:
    with st.expander("開發者診斷（?dev=1）", expanded=True):
        _versions: dict[str, int] = {}
        _with_ids = 0
        for _ev in events:
            for _m in (_ev.get("markets") or []):
                if not isinstance(_m, dict):
                    continue
                _k = str(_m.get("version") or "（無）")
                _versions[_k] = _versions.get(_k, 0) + 1
                _with_ids += 1 if any(outcome_ids(_m)) else 0
        st.write({
            "market 的 version 分布": _versions,
            "取得到市場代號的 market 數": _with_ids,
            "異常變動最近一次的走勢來源": st.session_state.get("mv_sources"),
        })
