import json
import time
import requests
import pandas as pd
import streamlit as st
import plotly.express as px

GAMMA = "https://gamma-api.polymarket.com"
CLOB  = "https://clob.polymarket.com"

RANGE_MAP = {
    "1H": "1h",
    "6H": "6h",
    "1D": "1d",
    "1W": "1w",
    "1M": "1m",
    "ALL": "max",
}

# -----------------------
# Network helpers (retry + backoff)
# -----------------------
def _http_get_json(url: str, params: dict, timeout: int = 30, retries: int = 3, backoff: float = 0.7):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            last_exc = e
            if attempt < retries:
                time.sleep(backoff * (2 ** (attempt - 1)))
            else:
                raise
    raise last_exc


# -----------------------
# Cached API functions (avoid hashing `self`)
# -----------------------
@st.cache_data(ttl=60, show_spinner=False)
def _cached_markets(
    gamma_base: str,
    limit: int = 400,
    offset: int = 0,
    closed: bool | None = False,
    order: str = "volume24hr",
    ascending: bool = False,
) -> pd.DataFrame:
    params = {
        "limit": limit,
        "offset": offset,
        "order": order,
        "ascending": str(ascending).lower(),
    }
    if closed is not None:
        params["closed"] = str(closed).lower()

    try:
        data = _http_get_json(f"{gamma_base}/markets", params=params, timeout=30, retries=3, backoff=0.7)
    except Exception:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    for c in ["volume24hr", "volume", "liquidity", "lastTradePrice"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "endDate" in df.columns:
        df["endDate"] = pd.to_datetime(df["endDate"], errors="coerce", utc=True)

    if "lastTradePrice" in df.columns:
        df["implied_prob_%"] = (df["lastTradePrice"] * 100).round(1)

    return df


@st.cache_data(ttl=30, show_spinner=False)
def _cached_prices_history(
    clob_base: str,
    token_id: str,
    interval: str,
    fidelity: int = 5,
) -> list[dict]:
    params = {"market": token_id, "interval": interval, "fidelity": fidelity}
    try:
        j = _http_get_json(f"{clob_base}/prices-history", params=params, timeout=30, retries=3, backoff=0.7)
    except Exception:
        return []
    return j.get("history", [])


# -----------------------
# Tiny "SDK" wrapper
# -----------------------
class PolySDK:
    def __init__(self, gamma_base=GAMMA, clob_base=CLOB):
        self.gamma_base = gamma_base
        self.clob_base = clob_base

    @staticmethod
    def _safe_list(x):
        """Gamma 有些欄位是 JSON 字串或 list，統一轉成 list[str]."""
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

    def markets(self, limit=400, offset=0, closed=False, order="volume24hr", ascending=False):
        return _cached_markets(
            self.gamma_base,
            limit=limit,
            offset=offset,
            closed=closed,
            order=order,
            ascending=ascending,
        )

    def prices_history(self, token_id: str, interval: str, fidelity: int = 5):
        return _cached_prices_history(
            self.clob_base,
            token_id=token_id,
            interval=interval,
            fidelity=fidelity,
        )

    def market_token_ids(self, row: pd.Series):
        return [str(x) for x in self._safe_list(row.get("clobTokenIds"))]

    def market_outcomes(self, row: pd.Series):
        return [str(x) for x in self._safe_list(row.get("outcomes"))]


# -----------------------
# Helpers
# -----------------------
def to_prob_percent(p: pd.Series) -> pd.Series:
    p = pd.to_numeric(p, errors="coerce")
    mx = p.max()
    if pd.notna(mx) and mx <= 1.5:
        return p * 100
    return p


def build_series(hist: list[dict]) -> pd.DataFrame:
    if not hist:
        return pd.DataFrame()
    hdf = pd.DataFrame(hist)
    if "t" not in hdf.columns or "p" not in hdf.columns:
        return pd.DataFrame()
    hdf["timestamp"] = pd.to_datetime(pd.to_numeric(hdf["t"], errors="coerce"), unit="s", utc=True)
    hdf["prob_%"] = to_prob_percent(hdf["p"])
    hdf = hdf.dropna(subset=["timestamp", "prob_%"]).sort_values("timestamp")
    return hdf[["timestamp", "prob_%"]]


def metric_delta(series: pd.DataFrame):
    if series.empty:
        return None
    first_v = float(series["prob_%"].iloc[0])
    last_v  = float(series["prob_%"].iloc[-1])
    delta_pp = last_v - first_v
    delta_rel = (last_v / first_v - 1) * 100 if first_v not in [0.0] else None
    return last_v, delta_pp, delta_rel


def pick_sort_col(df: pd.DataFrame, preferred: str) -> str | None:
    if preferred in df.columns:
        return preferred
    return None


def _format_enddate_ymd(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        ts = pd.to_datetime(v, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.date()
    except Exception:
        return None


def _format_compact_number(x, digits: int = 2) -> str:
    """把 5413482.27 -> 5.41M；None/NaN -> N/A"""
    try:
        if x is None:
            return "N/A"
        if isinstance(x, float) and pd.isna(x):
            return "N/A"
        v = float(x)
    except Exception:
        return "N/A"

    sign = "-" if v < 0 else ""
    v = abs(v)

    units = [("", 1.0), ("K", 1e3), ("M", 1e6), ("B", 1e9), ("T", 1e12)]
    unit = ""
    scale = 1.0
    for u, s in units:
        unit = u
        scale = s
        if v < s * 1000:
            break

    val = v / scale
    if unit == "":
        return f"{sign}{val:,.0f}"
    return f"{sign}{val:.{digits}f}{unit}"


def _add_latest_label(fig, series: pd.DataFrame):
    """加上你截圖那種：最後一點紅點 + 右側 Latest 標籤 + 水平虛線"""
    if series.empty:
        return fig

    last_ts = series["timestamp"].iloc[-1]
    last_v = float(series["prob_%"].iloc[-1])

    # 最後一點 marker（紅點）
    fig.add_scatter(
        x=[last_ts],
        y=[last_v],
        mode="markers",
        marker=dict(size=9, color="#E45756", line=dict(width=1, color="white")),
        hovertemplate="%{x}<br>Latest: %{y:.2f}%<extra></extra>",
        showlegend=False,
    )

    # 水平虛線
    fig.add_shape(
        type="line",
        xref="paper",
        x0=0,
        x1=1,
        yref="y",
        y0=last_v,
        y1=last_v,
        line=dict(color="rgba(0,0,0,0.22)", width=1, dash="dot"),
    )

    # 右側標籤（白底小框）
    fig.add_annotation(
        xref="paper",
        x=1.0,
        yref="y",
        y=last_v,
        text=f"Latest {last_v:.1f}%",
        showarrow=False,
        xanchor="right",
        yanchor="middle",
        font=dict(size=12, color="rgba(0,0,0,0.78)"),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="rgba(0,0,0,0.18)",
        borderwidth=1,
        borderpad=4,
    )
    return fig


# -----------------------
# Detail Renderer (clean top + latest label like screenshot)
# -----------------------
def _render_market_detail(sdk: PolySDK, picked_row: pd.Series):
    picked_q = str(picked_row.get("question", "") or "")

    token_ids = sdk.market_token_ids(picked_row)
    outcomes = sdk.market_outcomes(picked_row)

    with st.container(border=True):
        st.markdown("### 走勢詳情")

        if not token_ids:
            st.warning("這個事件沒有 clobTokenIds（可能不是 orderbook 市場或資料缺失）。換一個試試。")
            return

        if outcomes and len(outcomes) == len(token_ids):
            labels = [str(outcomes[i]) for i in range(len(outcomes))]
        else:
            labels = [f"Outcome {i}" for i in range(len(token_ids))]

        # Controls row (no hint text)
        ctrl = st.columns([2, 2, 2, 3])
        with ctrl[0]:
            outcome_idx = st.radio(
                "Outcome",
                options=list(range(len(token_ids))),
                format_func=lambda i: labels[i],
                horizontal=True,
                index=0,
                key="detail_outcome_idx",
            )
        with ctrl[1]:
            range_ui = st.radio(
                "區間",
                ["1H", "6H", "1D", "1W", "1M", "ALL"],
                horizontal=True,
                index=5,
                key="detail_range_ui",
            )
        with ctrl[2]:
            fidelity = st.slider("fidelity（分鐘）", 1, 60, 5, 1, key="detail_fidelity")
        with ctrl[3]:
            st.write("")  # 留白

        token_id = token_ids[outcome_idx]
        interval = RANGE_MAP[range_ui]
        hist = sdk.prices_history(token_id=token_id, interval=interval, fidelity=int(fidelity))
        series = build_series(hist)

        if series.empty:
            st.warning("這個區間沒有足夠成交資料畫圖。試試看切到 ALL 或把 fidelity 調大/調小。")
            return

        # Title alone line
        st.markdown(f"## {picked_q}")

        # Metrics: same row
        m = metric_delta(series)
        end_ymd = _format_enddate_ymd(picked_row.get("endDate"))
        vol24 = picked_row.get("volume24hr")

        c1, c2, c3, c4 = st.columns(4)
        if m is None:
            c1.metric("chance", "N/A")
            c2.metric("變化(pp)", "N/A")
        else:
            last_v, delta_pp, _ = m
            c1.metric("chance", f"{last_v:.1f}%")
            c2.metric("變化(pp)", f"{delta_pp:+.1f}")
        c3.metric("24h Vol", _format_compact_number(vol24, digits=2))
        c4.metric("End", end_ymd.isoformat() if end_ymd else "N/A")

        # Chart (simple + latest label)
        fig = px.line(series, x="timestamp", y="prob_%")
        fig.update_traces(
            line=dict(width=2, color="#1f77b4"),
            hovertemplate="%{x}<br>Chance: %{y:.2f}%<extra></extra>",
        )
        fig = _add_latest_label(fig, series)

        fig.update_layout(
            template="plotly_white",
            height=440,
            margin=dict(l=40, r=20, t=10, b=40),
            hovermode="x unified",
            showlegend=False,
        )
        fig.update_yaxes(
            range=[0, 100],
            title="Chance (%)",
            ticksuffix="%",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.06)",
        )
        fig.update_xaxes(title="", showgrid=False)

        st.plotly_chart(fig, use_container_width=True)

        # Hint moved under chart
        st.caption("提示：ALL 沒資料時可改 1W/1M；或把 fidelity 調大/調小。")


# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Polymarket Dashboard", layout="wide", initial_sidebar_state="collapsed")
st.title("Polymarket 互動儀表板（官方 REST）")

sdk = PolySDK()

# Filters moved from sidebar to main, wrapped in expander
with st.expander("篩選 / 設定", expanded=False):
    f1, f2, f3 = st.columns([2.2, 1.4, 2.4])
    with f1:
        kw = st.text_input("關鍵字（question 包含）", "")
    with f2:
        only_open = st.checkbox("只看未關閉", value=True)
    with f3:
        only_orderbook = st.checkbox("只看可用 Orderbook（enableOrderBook=true）", value=True)

tabs = st.tabs(["Trending 熱門", "Volatility 波動"])

# Load markets
df = sdk.markets(limit=450, closed=False if only_open else None)
if df.empty:
    st.error("目前無法連線到 Polymarket API（可能是 DNS 暫時解析失敗 / 網路抖動）。請稍後重新整理再試。")
    st.stop()

if "question" in df.columns:
    df["question"] = df["question"].astype(str)
else:
    st.error("Gamma /markets 沒回傳 question 欄位，無法顯示。")
    st.stop()

if kw:
    df = df[df["question"].fillna("").str.contains(kw, case=False, na=False)]

if only_orderbook and "enableOrderBook" in df.columns:
    df = df[df["enableOrderBook"] == True]

df_view = df.reset_index(drop=True).copy()
df_view["_rowid"] = range(len(df_view))

if "endDate" in df_view.columns:
    df_view["endDate_ymd"] = df_view["endDate"].apply(_format_enddate_ymd)

# Session state
if "selected_rowid" not in st.session_state:
    st.session_state.selected_rowid = None
if "editor_nonce" not in st.session_state:
    st.session_state.editor_nonce = 0

if st.session_state.selected_rowid is not None:
    if st.session_state.selected_rowid not in set(df_view["_rowid"].tolist()):
        st.session_state.selected_rowid = None

# -----------------------
# Tab 1: Trending
# -----------------------
with tabs[0]:
    st.subheader("熱門事件排行")

    pill = st.pills(
        "排行依據",
        options=["volume24hr", "liquidity", "volume"],
        selection_mode="single",
        default="volume24hr",
    )

    sort_col = pick_sort_col(df_view, pill)
    work = df_view.copy()
    if sort_col:
        work = work.sort_values(sort_col, ascending=False, na_position="last")

    show_cols = [c for c in ["question", "implied_prob_%", "volume24hr", "liquidity", "volume"] if c in work.columns]
    if "endDate_ymd" in work.columns:
        show_cols.append("endDate_ymd")

    chip_left, chip_right = st.columns([6, 1])
    with chip_left:
        if st.session_state.selected_rowid is None:
            st.caption("目前未選取事件：請在表格勾選「選取」")
        else:
            sel_df = work[work["_rowid"] == st.session_state.selected_rowid].head(1)
            if sel_df.empty:
                st.caption("目前選取事件：N/A（可能被篩選條件過濾掉）")
            else:
                sel_q = str(sel_df.iloc[0].get("question", "") or "")
                st.markdown(f"**目前選取：** `{sel_q}`")
    with chip_right:
        if st.button("清除選取", use_container_width=True, disabled=(st.session_state.selected_rowid is None)):
            st.session_state.selected_rowid = None
            st.session_state.editor_nonce += 1
            st.rerun()

    topn = work.head(50).set_index("_rowid")[show_cols].copy()
    topn.insert(0, "選取", False)
    if st.session_state.selected_rowid is not None and st.session_state.selected_rowid in topn.index:
        topn.loc[st.session_state.selected_rowid, "選取"] = True

    editor_key = f"trending_table_editor_{st.session_state.editor_nonce}"
    edited = st.data_editor(
        topn,
        use_container_width=True,
        hide_index=True,
        disabled=[c for c in topn.columns if c != "選取"],
        column_config={
            "選取": st.column_config.CheckboxColumn("選取", help="勾選後在下方顯示走勢", default=False),
            "question": st.column_config.TextColumn("question", width="large"),
            "implied_prob_%": st.column_config.NumberColumn("implied_prob_%", format="%.1f"),
            "volume24hr": st.column_config.NumberColumn("volume24hr", format="%.0f"),
            "liquidity": st.column_config.NumberColumn("liquidity", format="%.0f"),
            "volume": st.column_config.NumberColumn("volume", format="%.0f"),
            "endDate_ymd": st.column_config.DateColumn("endDate", help="只顯示年月日"),
        },
        key=editor_key,
    )

    selected_ids = edited.index[edited["選取"] == True].tolist()
    if not selected_ids:
        st.session_state.selected_rowid = None
        st.info("勾選一個事件，就會在下面看到走勢。")
    else:
        if len(selected_ids) > 1:
            st.warning("你一次勾選了多個事件；我先顯示第一個（建議只勾一個）。")

        picked_rowid = int(selected_ids[0])
        st.session_state.selected_rowid = picked_rowid

        picked_row_df = work[work["_rowid"] == picked_rowid].head(1)
        if picked_row_df.empty:
            st.warning("找不到該事件（可能排序/篩選變動）。請重新勾選一次。")
        else:
            _render_market_detail(sdk, picked_row_df.iloc[0])

# -----------------------
# Tab 2: Volatility
# -----------------------
with tabs[1]:
    st.subheader("波動最大事件（用 prices-history 計算）")
    st.caption("做法：對熱門 Top K 市場抓 prices-history，計算該區間 first→last 的變化（pp）。")

    colA, colB, colC = st.columns([1, 1, 2])
    with colA:
        range_ui = st.radio("區間", ["1H", "6H", "1D", "1W", "1M", "ALL"], horizontal=True, index=2, key="vol_range")
    with colB:
        top_k = st.number_input("計算 Top K（越大越慢）", min_value=5, max_value=80, value=25, step=5, key="vol_topk")
    with colC:
        fidelity = st.slider("fidelity（分鐘）", 1, 60, 10, 1, key="vol_fidelity")

    base_sort = "volume24hr" if "volume24hr" in df_view.columns else ("volume" if "volume" in df_view.columns else None)
    base = df_view.copy()
    if base_sort:
        base = base.sort_values(base_sort, ascending=False, na_position="last")

    if st.button("開始計算波動排行"):
        interval = RANGE_MAP[range_ui]
        work2 = base.head(int(top_k)).copy()
        results = []

        progress = st.progress(0)
        for i, (_, r) in enumerate(work2.iterrows(), start=1):
            token_ids = sdk.market_token_ids(r)
            outcomes = sdk.market_outcomes(r)
            if not token_ids:
                progress.progress(i / int(top_k))
                continue

            token_id = token_ids[0]
            try:
                hist = sdk.prices_history(token_id=token_id, interval=interval, fidelity=int(fidelity))
                series = build_series(hist)
                m = metric_delta(series)
                if m is None:
                    progress.progress(i / int(top_k))
                    continue

                last_v, delta_pp, _ = m
                results.append({
                    "_rowid": r.get("_rowid"),
                    "question": r.get("question"),
                    "outcome": outcomes[0] if outcomes else "outcome0",
                    "current_%": round(last_v, 1),
                    "delta_pp": round(delta_pp, 1),
                    "abs_delta_pp": round(abs(delta_pp), 1),
                    "volume24hr": r.get("volume24hr"),
                    "liquidity": r.get("liquidity"),
                })
            except Exception:
                pass

            progress.progress(i / int(top_k))
            time.sleep(0.03)

        res = pd.DataFrame(results)
        if res.empty:
            st.warning("沒有算出結果（可能該區間成交太少或 API 回傳空 / 網路不穩）。試試看：改 ALL 或提高 Top K。")
        else:
            res = res.sort_values("abs_delta_pp", ascending=False, na_position="last")
            st.dataframe(res.head(30), use_container_width=True, hide_index=True)

            opts2 = res.dropna(subset=["question"])
            if not opts2.empty:
                picked2 = st.selectbox("點一個波動事件（回到 Trending 會自動選取並顯示走勢）", options=opts2["question"].tolist())
                if picked2:
                    rid = opts2[opts2["question"] == picked2]["_rowid"].iloc[0]
                    try:
                        st.session_state.selected_rowid = int(rid)
                        st.info("已同步選取：請切回「Trending 熱門」看走勢。")
                    except Exception:
                        pass
