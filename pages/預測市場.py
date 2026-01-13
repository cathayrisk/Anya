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
# Tiny "SDK" wrapper
# -----------------------
class PolySDK:
    def__init__(self, gamma_base=GAMMA, clob_base=CLOB):
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

    @st.cache_data(ttl=60)
    def markets(self, limit=400, offset=0, closed=False, order="volume24hr", ascending=False):
        params = {
            "limit": limit,
            "offset": offset,
            "order": order,
            "ascending": str(ascending).lower(),
        }
        if closed is not None:
            params["closed"] = str(closed).lower()

        r = requests.get(f"{self.gamma_base}/markets", params=params, timeout=30)
        r.raise_for_status()
        df = pd.DataFrame(r.json())

        # numeric clean
        for c in ["volume24hr", "volume", "liquidity", "lastTradePrice"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        if "endDate" in df.columns:
            df["endDate"] = pd.to_datetime(df["endDate"], errors="coerce", utc=True)

        if "lastTradePrice" in df.columns:
            df["implied_prob_%"] = (df["lastTradePrice"] * 100).round(1)

        return df

    @st.cache_data(ttl=30)
    def prices_history(self, token_id: str, interval: str, fidelity: int = 5):
        params = {"market": token_id, "interval": interval, "fidelity": fidelity}
        r = requests.get(f"{self.clob_base}/prices-history", params=params, timeout=30)
        r.raise_for_status()
        j = r.json()
        return j.get("history", [])

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

tabs = st.tabs(["Trending 熱門", "Volatility 波動", "Market 詳情"])

# Load markets
df = sdk.markets(limit=450, closed=False if only_open else None)

if "question" in df.columns:
    df["question"] = df["question"].astype(str)
else:
    st.error("Gamma /markets 沒回傳 question 欄位，無法顯示。")
    st.stop()

if kw:
    df = df[df["question"].fillna("").str.contains(kw, case=False, na=False)]

if only_orderbook and "enableOrderBook" in df.columns:
    df = df[df["enableOrderBook"] == True]

# Session state
if "selected_question" not in st.session_state:
    st.session_state.selected_question = None

# -----------------------
# Tab 1: Trending
# -----------------------
with tabs[0]:
    st.subheader("熱門事件排行")

    # pills: choose ranking metric
    pill = st.pills(
        "排行依據",
        options=["volume24hr", "liquidity", "volume"],
        selection_mode="single",
        default="volume24hr",
    )

    sort_col = pick_sort_col(df, pill)
    work = df.copy()
    if sort_col:
        work = work.sort_values(sort_col, ascending=False, na_position="last")

    show_cols = [c for c in ["question", "implied_prob_%", "volume24hr", "liquidity", "volume", "endDate"] if c in work.columns]
    st.dataframe(work[show_cols].head(50), use_container_width=True, hide_index=True)

    st.markdown("### 選一個事件看詳情")
    options = work["question"].fillna("").head(200).tolist()
    picked = st.selectbox("事件", options=options, index=0 if options else None)
    if picked:
        st.session_state.selected_question = picked
        st.info("已選擇，切到「Market 詳情」分頁看走勢。")

# -----------------------
# Tab 2: Volatility
# -----------------------
with tabs[1]:
    st.subheader("波動最大事件（用 prices-history 計算）")
    st.caption("做法：對熱門 Top K 市場抓 prices-history，計算該區間 first→last 的變化（pp）。")

    colA, colB, colC = st.columns([1, 1, 2])
    with colA:
        range_ui = st.radio("區間", ["1H", "6H", "1D", "1W", "1M", "ALL"], horizontal=True, index=2)
    with colB:
        top_k = st.number_input("計算 Top K（越大越慢）", min_value=5, max_value=80, value=25, step=5)
    with colC:
        fidelity = st.slider("fidelity（分鐘）", 1, 60, 10, 1)

    # 用 volume24hr 當作「熱門」基準（沒有就用 volume）
    base_sort = "volume24hr" if "volume24hr" in df.columns else ("volume" if "volume" in df.columns else None)
    base = df.copy()
    if base_sort:
        base = base.sort_values(base_sort, ascending=False, na_position="last")

    if st.button("開始計算波動排行"):
        interval = RANGE_MAP[range_ui]
        work = base.head(int(top_k)).copy()
        results = []

        progress = st.progress(0)
        for i, (_, r) in enumerate(work.iterrows(), start=1):
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
            st.warning("沒有算出結果（可能該區間成交太少或 API 回傳空）。試試看：改 ALL 或提高 Top K。")
        else:
            res = res.sort_values("abs_delta_pp", ascending=False, na_position="last")
            st.dataframe(res.head(30), use_container_width=True, hide_index=True)

            picked2 = st.selectbox("點一個波動事件，帶到 Market 詳情", options=res["question"].dropna().tolist())
            if picked2:
                st.session_state.selected_question = picked2
                st.info("已選擇，切到「Market 詳情」分頁看走勢。")

# -----------------------
# Tab 3: Market detail
# -----------------------
with tabs[2]:
    st.subheader("Market 詳情（機率走勢）")

    # fallback selector
    if st.session_state.selected_question is None:
        options = df["question"].fillna("").head(200).tolist()
        st.session_state.selected_question = st.selectbox("先選一個事件", options=options)

    picked_q = st.session_state.selected_question
    row_df = df[df["question"] == picked_q].head(1)
    if row_df.empty:
        st.warning("找不到該事件（可能被篩選條件過濾掉了）。回到 Trending 重新選。")
        st.stop()
    row = row_df.iloc[0]

    token_ids = sdk.market_token_ids(row)
    outcomes = sdk.market_outcomes(row)

    if not token_ids:
        st.warning("這個事件沒有 clobTokenIds（可能不是 orderbook 市場或資料缺失）。換一個熱門事件試試。")
        st.stop()

    # outcome selector
    if outcomes and len(outcomes) == len(token_ids):
        labels = [f"{outcomes[i]} (idx={i})" for i in range(len(outcomes))]
    else:
        labels = [f"Outcome idx={i}" for i in range(len(token_ids))]

    outcome_idx = st.radio(
        "Outcome",
        options=list(range(len(token_ids))),
        format_func=lambda i: labels[i],
        horizontal=True,
        index=0,
    )
    token_id = token_ids[outcome_idx]

    range_ui = st.radio("區間", ["1H", "6H", "1D", "1W", "1M", "ALL"], horizontal=True, index=5)
    fidelity = st.slider("fidelity（分鐘）", 1, 60, 5, 1)

    interval = RANGE_MAP[range_ui]
    hist = sdk.prices_history(token_id=token_id, interval=interval, fidelity=int(fidelity))
    series = build_series(hist)

    top_left, top_right = st.columns([2, 1])

    with top_left:
        st.markdown(f"## {picked_q}")
        meta = st.columns(3)
        if "volume24hr" in df.columns:
            meta[0].write(f"**24h Vol**：{row.get('volume24hr')}")
        if "endDate" in df.columns:
            meta[1].write(f"**End**：{row.get('endDate')}")
        meta[2].write(f"**token_id**：{token_id}")

    with top_right:
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
    else:
        fig = px.line(series, x="timestamp", y="prob_%")
        fig.update_yaxes(range=[0, 100], title="Chance (%)")
        fig.update_xaxes(title="")
        st.plotly_chart(fig, use_container_width=True)
