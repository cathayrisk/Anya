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
- bestBid 覆蓋率 90%、bestAsk 100%；價差 > $0.10 的 market 佔 25%，中價規則會實際 fallback。

已知環境限制：部分境內網路會把 *.polymarket.com 的 DNS 導向非官方 IP，
造成 TLS 自簽憑證錯誤。那是網路層攔截，不是程式問題——錯誤訊息會分類指出。
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

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


def pick_price(m: dict, fallback: float | None = None) -> tuple[float | None, str, float | None]:
    """官方規則：預設中價；價差 > $0.10 才退回最後成交價。回傳 (價格, 來源, 價差)。"""
    bid = to_float(m.get("bestBid"))
    ask = to_float(m.get("bestAsk"))
    last = to_float(m.get("lastTradePrice"))

    if bid is not None and ask is not None and ask >= bid:
        spread = ask - bid
        if spread <= SPREAD_FALLBACK + SPREAD_EPS:
            return (bid + ask) / 2.0, "mid", spread
        if last is not None:
            return last, "last(價差寬)", spread
        return (bid + ask) / 2.0, "mid(價差寬)", spread

    # 實測 bestBid 只有 90% 覆蓋率（冷門選項沒人掛買單），但 Gamma 自己的 spread 欄位是 100%。
    # 算不出中價時，至少把價差顯示出來，不要讓這一欄空著。
    reported = to_float(m.get("spread"))
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
                raw, src, spd = p0, src0, spread0
            else:
                raw, src, spd = (prices[i] if i < len(prices) else None), "outcomePrices", None
            rows.append({
                "outcome": nm, "raw": raw, "來源": src, "價差": spd,
                "token_id": tokens[i] if i < len(tokens) else None,
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


def events_to_frame(events: list[dict]) -> pd.DataFrame:
    rows = []
    for ev in events:
        odf, total, normalized = event_outcomes(ev)
        top_label, top_prob, top_spread = None, None, None
        if not odf.empty and odf["機率_%"].notna().any():
            top = odf.loc[odf["機率_%"].idxmax()]
            top_label = str(top["outcome"])
            top_prob = float(top["機率_%"])
            top_spread = to_float(top["價差"])
        rows.append({
            "event_id": str(ev.get("id") or ev.get("slug") or ""),
            "title": str(ev.get("title") or ev.get("slug") or "(無標題)"),
            "領先結果": top_label,
            "機率_%": top_prob,
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


def compact_number(x, digits: int = 2) -> str:
    """5413482.27 -> 5.41M；None/NaN -> N/A"""
    v = to_float(x)
    if v is None:
        return "N/A"
    sign = "-" if v < 0 else ""
    v = abs(v)
    unit, scale = "", 1.0
    for u, s in [("", 1.0), ("K", 1e3), ("M", 1e6), ("B", 1e9), ("T", 1e12)]:
        unit, scale = u, s
        if v < s * 1000:
            break
    val = v / scale
    return f"{sign}{val:,.0f}" if unit == "" else f"{sign}{val:.{digits}f}{unit}"


# -----------------------
# 明細
# -----------------------
def render_event_detail(ev: dict, row: pd.Series, key_prefix: str) -> None:
    odf, total, normalized = event_outcomes(ev)

    with st.container(border=True):
        st.markdown(f"### {row.get('title', '')}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("領先結果", str(row.get("領先結果") or "N/A"))
        prob = to_float(row.get("機率_%"))
        c2.metric("機率（正規化）", f"{prob:.1f}%" if prob is not None else "N/A")
        c3.metric("24h 成交量", compact_number(row.get("volume24hr")))
        end_d = row.get("endDate_台北")
        c4.metric("結束（台北）", end_d.isoformat() if end_d else "N/A")

        if total is None:
            pass
        elif normalized:
            st.caption(
                f"原始價格加總 **{total:.3f}** → 超出 1 的 {(total - 1) * 100:.1f}% 是價差造成的"
                "隱含抽水（overround）。下表「機率_%」已做乘法正規化。"
            )
        else:
            st.warning(
                f"原始價格加總 **{total:.3f}**，落在合理區間 "
                f"{NORMALIZE_BAND[0]}–{NORMALIZE_BAND[1]} 之外——這組結果很可能**不是互斥且窮盡**的"
                "（例如把幾個獨立問題綁成同一個事件）。硬做正規化會生出假機率，"
                "所以下表直接顯示原始價格，未做任何調整。"
            )

        if odf.empty:
            st.warning("這個事件沒有可解析的結果資料。")
            return

        st.dataframe(
            odf[["outcome", "機率_%", "原始_%", "價差", "來源"]],
            hide_index=True,
            column_config={
                "outcome": st.column_config.TextColumn("結果", width="large"),
                "機率_%": st.column_config.NumberColumn("機率 %（正規化）", format="%.1f"),
                "原始_%": st.column_config.NumberColumn("原始 %", format="%.1f"),
                "價差": st.column_config.NumberColumn("買賣價差", format="%.3f"),
                "來源": st.column_config.TextColumn("價格來源"),
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
            range_ui = st.radio(
                "區間", list(RANGE_MAP.keys()), horizontal=True, index=4,
                key=f"{key_prefix}_range",
            )
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

if df.empty:
    st.warning("篩選後沒有結果。放寬關鍵字或降低流動性門檻。")
    st.stop()

df = df.sort_values(sort_by, ascending=False, na_position="last").reset_index(drop=True)

tab_hot, tab_move = st.tabs(["熱門事件", "異常變動"])

# ---- Tab 1：熱門事件 ----
with tab_hot:
    st.caption(f"主題「{cat_name}」共 {len(df)} 個進行中事件。點一列看明細。")

    show = df[["title", "領先結果", "機率_%", "已正規化", "價差", "結果數",
               "volume24hr", "liquidity", "endDate_台北"]]
    sel = st.dataframe(
        show.head(80),
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="hot_table",
        column_config={
            "title": st.column_config.TextColumn("事件", width="large"),
            "領先結果": st.column_config.TextColumn("領先結果"),
            "機率_%": st.column_config.NumberColumn("機率 %", format="%.1f"),
            "已正規化": st.column_config.CheckboxColumn(
                "已正規化",
                help="未打勾＝各結果加總落在合理區間外，很可能不是互斥窮盡的一組，顯示的是原始價格。",
            ),
            "價差": st.column_config.NumberColumn("價差", format="%.3f",
                                                 help="買賣價差。> 0.10 時價格改用最後成交價。"),
            "結果數": st.column_config.NumberColumn("結果數", format="%d"),
            "volume24hr": st.column_config.NumberColumn("24h 量", format="%.0f"),
            "liquidity": st.column_config.NumberColumn("流動性", format="%.0f"),
            "endDate_台北": st.column_config.DateColumn("結束（台北）"),
        },
    )

    picked = sel.selection.rows if sel and sel.selection else []
    if not picked:
        st.info("點選上表任一列，下方顯示結果分布與走勢。")
    else:
        row = df.iloc[picked[0]]
        ev = events_by_id.get(row["event_id"])
        if ev is None:
            st.warning("找不到該事件的原始資料，請重新整理。")
        else:
            render_event_detail(ev, row, key_prefix="hot")

# ---- Tab 2：異常變動 ----
with tab_move:
    st.caption("對主題內成交量前 K 名事件抓走勢，計算區間內的淨變化 / 全距 / 標準差。")

    m1, m2, m3, m4 = st.columns([1.4, 1, 1, 1.6])
    with m1:
        mv_range = st.radio("區間", list(RANGE_MAP.keys()), horizontal=True, index=2, key="mv_range")
    with m2:
        top_k = st.number_input("Top K", min_value=5, max_value=80, value=25, step=5, key="mv_k")
    with m3:
        mv_fid = st.slider("fidelity", 1, 60, 10, 1, key="mv_fid")
    with m4:
        rank_by = st.selectbox("排序依據", ["全距_pp", "波動度_pp", "淨變化_pp（絕對值）"],
                               index=0, key="mv_rank")

    if st.button("開始掃描"):
        cands = df.head(int(top_k))
        pairs: list[tuple[str, str, str]] = []   # (token_id, event_id, 結果標籤)
        for _, r in cands.iterrows():
            ev = events_by_id.get(r["event_id"])
            if not ev:
                continue
            odf, _, _ = event_outcomes(ev)
            if odf.empty:
                continue
            valid = odf[odf["token_id"].notna()]
            if valid.empty:
                continue
            best = valid.loc[valid["機率_%"].idxmax()] if valid["機率_%"].notna().any() else valid.iloc[0]
            pairs.append((str(best["token_id"]), r["event_id"], str(best["outcome"])))

        if not pairs:
            st.warning("這批事件沒有可用的掛單簿 token，換個主題或提高 Top K。")
        else:
            with st.spinner(f"併發抓取 {len(pairs)} 個走勢…"):
                try:
                    hist_map = scan_histories(
                        CLOB, tuple(p[0] for p in pairs), RANGE_MAP[mv_range], int(mv_fid)
                    )
                except Exception as e:
                    hist_map = {}
                    st.error(explain_network_error(e))

            results = []
            empty_n = 0
            for token_id, event_id, label in pairs:
                stats = series_stats(build_series(hist_map.get(token_id, [])))
                if not stats:
                    empty_n += 1
                    continue
                base = df[df["event_id"] == event_id].head(1)
                results.append({
                    "event_id": event_id,
                    "title": base["title"].iloc[0] if not base.empty else event_id,
                    "結果": label,
                    **stats,
                    "volume24hr": base["volume24hr"].iloc[0] if not base.empty else None,
                })

            # 存進 session_state：不然一動下方任何 widget，button 就變 False，整塊結果會消失
            st.session_state["mv_results"] = results
            st.session_state["mv_empty_n"] = empty_n

    results = st.session_state.get("mv_results")
    if results is None:
        st.info("按「開始掃描」計算。")
    elif not results:
        st.warning("沒有算出結果——該區間成交太少，或 API 回空。試 ALL 區間。")
    else:
        res = pd.DataFrame(results)
        res["淨變化_pp（絕對值）"] = res["淨變化_pp"].abs()
        res = res.sort_values(rank_by, ascending=False, na_position="last")

        empty_n = st.session_state.get("mv_empty_n", 0)
        if empty_n:
            st.caption(f"註：另有 {empty_n} 個事件在此區間沒有足夠成交資料，已排除（非靜默截斷）。")

        st.dataframe(
            res[["title", "結果", "現值_%", "淨變化_pp", "全距_pp", "波動度_pp", "樣本數", "volume24hr"]].head(30),
            hide_index=True,
            column_config={
                "title": st.column_config.TextColumn("事件", width="large"),
                "現值_%": st.column_config.NumberColumn("現值 %", format="%.1f"),
                "淨變化_pp": st.column_config.NumberColumn("淨變化 pp", format="%+.1f",
                                                        help="方向性漂移（末值 − 首值），不是波動。"),
                "全距_pp": st.column_config.NumberColumn("全距 pp", format="%.1f",
                                                       help="區間內最高 − 最低，衡量擺盪幅度。"),
                "波動度_pp": st.column_config.NumberColumn("標準差 pp", format="%.2f"),
                "樣本數": st.column_config.NumberColumn("點數", format="%d"),
                "volume24hr": st.column_config.NumberColumn("24h 量", format="%.0f"),
            },
        )

        pick_title = st.selectbox("看哪一個的明細", res["title"].tolist(), key="mv_pick")
        if pick_title:
            eid = res.loc[res["title"] == pick_title, "event_id"].iloc[0]
            base = df[df["event_id"] == eid].head(1)
            ev = events_by_id.get(eid)
            if ev is not None and not base.empty:
                render_event_detail(ev, base.iloc[0], key_prefix="mvdet")

st.divider()
st.caption(
    "⚠️ 市場價格不等於校準後的真實機率：預測市場普遍存在 favorite-longshot bias"
    "（低機率端系統性高估、高機率端低估，且離到期越遠越嚴重），本頁未做校準調整。"
    "另外解析風險（UMA 爭議、規則措辭）不會反映在盤面價格上。"
)
st.caption("本頁僅為資訊呈現，不構成投資建議。")
