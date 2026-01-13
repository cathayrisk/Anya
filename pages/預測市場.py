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
    """
    對外 HTTP GET（requests）加重試：處理 DNS 暫時失敗、網路抖動等。
    - retries: 總嘗試次數
    - backoff: 每次失敗後 sleep 秒數（指數退避）
    """
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
    # 理論上不會走到這裡
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
        # 網路/DNS 失敗：回傳空 df，讓 UI 端顯示友善錯誤並 stop
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # numeric clean
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
        # 網路/DNS 失敗：回傳空 list，UI 會顯示「沒資料」提示
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
            # try JSON
            try:
                v = json.loads(s)
                if isinstance(v, list):
                    return v
            except Exception:
                pass
            # fallback: split
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
    """把 endDate 轉成 YYYY-MM-DD（處理 NaT/None）。"""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        ts = pd.to_datetime(v, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.date()
    except Exception:
        return None


def _render_market_detail(sdk: PolySDK, picked_row: pd.Series):
    """在表格下方渲染走勢與控制項（Outcome / Range / Fidelity）。"""
    picked_q = str(picked_row.get("question", "") or "")

    with st.container(border=True):
        st.subheader("走勢詳情")

        token_ids = sdk.market_token_ids(picked_row)
        outcomes = sdk.market_outcomes(picked_row)

        if not token_ids:
            st.warning("這個事件沒有 clobTokenIds（可能不是 orderbook 市場或資料缺失）。換一個試試。")
            return

        # outcome selector
        if outcomes and len(outcomes) == len(token_ids):
            labels = [f"{outcomes[i]} (idx={i})" for i in range(len(outcomes))]
        else:
            labels = [f"Outcome idx={i}" for i in range(len(token_ids))]

        top = st.columns([2, 2, 2, 3])
        with top[0]:
            outcome_idx = st.radio(
                "Outcome",
                options=list(range(len(token_ids))),
                format_func=lambda i: labels[i],
                horizontal=True,
                index=0,
                key="detail_outcome_idx",
            )
        with top[1]:
            range_ui = st.radio(
                "區間",
                ["1H", "6H", "1D", "1W", "1M", "ALL"],
                horizontal=True,
                index=5,
                key="detail_range_ui",
            )
        with top[2]:
            fidelity = st.slider("fidelity（分鐘）", 1, 60, 5, 1, key="detail_fidelity")
        with top[3]:
            st.markdown("####")
            st.caption("提示：ALL 沒資料時可改 1W/1M；或把 fidelity 調大/調小。")

        token_id = token_ids[outcome_idx]
        interval = RANGE_MAP[range_ui]
        hist = sdk.prices_history(token_id=token_id, interval=interval, fidelity=int(fidelity))
        series = build_series(hist)

        header_left, header_right = st.columns([3, 2])
        with header_left:
            st.markdown(f"### {picked_q}")

            meta = st.columns(3)
            if "volume24hr" in picked_row.index:
                meta[0].write(f"**24h Vol**：{picked_row.get('volume24hr')}")
            if "endDate" in picked_row.index:
                end_ymd = _format_enddate_ymd(picked_row.get("endDate"))
                meta[1].write(f"**End**：{end_ymd.isoformat() if end_ymd else 'N/A'}")
            meta[2].write(f"**token_id**：{token_id}")

        with header_right:
            m = metric_delta(series)
            if m is None:
                st.metric("chance", "N/A")
                st.metric("變化（pp）", "N/A")
            else:
                last_v, delta_pp, _ = m
                st.metric("chance", f"{last_v:.1f}%")
                st.metric("變化（pp）", f"{delta_pp:+.1f} pp")

        if series.empty:
            st.warning("這個區間沒有足夠成交資料畫圖。試試看切到 ALL 或把 fidelity 調大/調小。")
            return

        fig = px.line(series, x="timestamp", y="prob_%")
        fig.update_yaxes(range=[0, 100], title="Chance (%)")
        fig.update_xaxes(title="")
        st.plotly_chart(fig, use_container_width=True)


# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Polymarket Dashboard", layout="wide")
st.title("Polymarket 互動儀表板（官方 REST）")

sdk = PolySDK()

# Sidebar (global)
st.sidebar.header("全域篩選")
kw = st.sidebar.text_input("關鍵字（question 包含）", "")
only_open = st.sidebar.checkbox("只看未關閉（closed=false）", value=True)
only_orderbook = st.sidebar.checkbox("只看可用 Orderbook（enableOrderBook=true）", value=True)

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

# 加一個穩定 rowid（用來做勾選與 session_state）
df_view = df.reset_index(drop=True).copy()
df_view["_rowid"] = range(len(df_view))

# endDate 只顯示年月日（表格用）
if "endDate" in df_view.columns:
    df_view["endDate_ymd"] = df_view["endDate"].apply(_format_enddate_ymd)

# Session state
if "selected_rowid" not in st.session_state:
    st.session_state.selected_rowid = None
if "editor_nonce" not in st.session_state:
    st.session_state.editor_nonce = 0

# 如果篩選後原本選的 row 不存在了，就清掉
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

    # 顯示欄位（endDate 改成 endDate_ymd）
    show_cols = [c for c in ["question", "implied_prob_%", "volume24hr", "liquidity", "volume"] if c in work.columns]
    if "endDate_ymd" in work.columns:
        show_cols.append("endDate_ymd")

    # 目前選取 chip + 清除按鈕（你要的第 1 點）
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
            st.session_state.editor_nonce += 1  # 重置 data_editor 狀態
            st.rerun()

    st.caption("在表格勾選「選取」後，下方會直接顯示走勢（建議只勾一個最清楚）。")

    # 用 index 當 rowid，這樣可以 hide_index=True 讓 ID 不出現在表格
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
            "volume24hr": st.column_config.NumberColumn("volume24hr"),
            "liquidity": st.column_config.NumberColumn("liquidity"),
            "volume": st.column_config.NumberColumn("volume"),
            "endDate_ymd": st.column_config.DateColumn("endDate", help="只顯示年月日"),
        },
        key=editor_key,
    )

    selected_ids = edited.index[edited["選取"] == True].tolist()
    if not selected_ids:
        # 允許使用者把勾選全部取消
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

            token_id = token_ids[0]  # 預設 outcome 0
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
