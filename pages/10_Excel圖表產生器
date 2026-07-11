# pages/10_Excel圖表產生器.py
# -*- coding: utf-8 -*-
"""
Excel 智慧圖表產生器 — 上傳 Excel 後互動式產生 Plotly 圖表

功能：
  1. 上傳 .xlsx / .xls，選擇工作表，預覽前 10 行與統計摘要
  2. 自動偵測欄位型別，智慧推薦最適圖表類型
  3. 支援 8 種圖表：折線、長條、散點、圓餅/甜甜圈、面積、盒鬚、熱力圖、直方圖
  4. 豐富細部調整面板（色彩、字型、座標軸、圖例、版面、圖表專屬設定）
  5. Dashboard 模式：最多 4 圖並排顯示
  6. AI Agent 協助（OpenAI gpt-4.1-mini）：自然語言描述需求，自動填入設定
  7. 匯出：互動式 HTML、靜態 PNG（需 kaleido）、原始 CSV
"""

# ── 標準函式庫 ──────────────────────────────────────────────────────────────────
import io
import json
import os

import streamlit as st

# ─── 頁面設定（必須是第一個 Streamlit 呼叫）────────────────────────────────────
st.set_page_config(
    page_title="Excel 圖表產生器",
    page_icon="📊",
    layout="wide",
)

from utils.weather_toast import render_weather_toast_watcher
render_weather_toast_watcher()

# ─── 標題（立即渲染，確保頁面不空白）──────────────────────────────────────────
st.title("📊 Excel 智慧圖表產生器")
st.caption("上傳 Excel → 選欄位 → 細部調整 → 匯出")

# ─── 第三方套件 ─────────────────────────────────────────────────────────────────
try:
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from openai import OpenAI
except ImportError as _e:
    st.error(f"缺少必要套件：{_e}，請執行 pip install pandas plotly openai")
    st.stop()

# kaleido（靜態圖片匯出，非強制）
HAS_KALEIDO = False
try:
    import kaleido  # noqa: F401
    HAS_KALEIDO = True
except ImportError:
    pass

# ─── OpenAI API Key ─────────────────────────────────────────────────────────────
try:
    _OPENAI_KEY = st.secrets.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_KEY")
except Exception:
    _OPENAI_KEY = None
if not _OPENAI_KEY:
    _OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")

# ─── Session State 初始化 ────────────────────────────────────────────────────────
_DEFAULTS: dict = {
    # 資料
    "excel_df": None,
    "excel_sheets": [],
    "excel_file_name": "",
    # Dashboard 模式
    "chart_configs": [],
    "dashboard_mode": False,
    # AI 對話
    "ai_messages": [],
    # 基本圖表設定
    "chart_type": "bar",
    "x_col": None,
    "y_cols": [],
    "color_col": "",
    "size_col": "",
    "chart_title": "",
    "x_label": "",
    "y_label": "",
    "theme": "plotly_white",
    # ── 細部調整：色彩 ──
    "color_palette": "Plotly",
    "opacity": 1.0,
    "plot_bgcolor": "#ffffff",
    # ── 細部調整：字型 ──
    "font_family": "Open Sans",
    "font_size": 14,
    "title_font_size": 20,
    "font_color": "#000000",
    # ── 細部調整：座標軸 ──
    "x_range_min": "",
    "x_range_max": "",
    "y_range_min": "",
    "y_range_max": "",
    "x_axis_type": "auto",
    "y_axis_type": "auto",
    "tick_angle": 0,
    "x_tickformat": "",
    "y_tickformat": "",
    "show_xgrid": True,
    "show_ygrid": True,
    "show_zeroline": True,
    # ── 細部調整：圖例 ──
    "show_legend": True,
    "legend_position": "右上",
    "legend_orientation": "v",
    "legend_bgcolor": "#ffffff",
    "legend_borderwidth": 0,
    # ── 細部調整：版面 ──
    "chart_height": 500,
    "margin_t": 60,
    "margin_b": 60,
    "margin_l": 60,
    "margin_r": 40,
    # ── 細部調整：資料標籤 ──
    "show_labels": False,
    "label_position": "outside",
    "label_template": "",
    "label_font_size": 12,
    # ── 折線 / 面積 ──
    "line_dash": "solid",
    "line_shape": "linear",
    "line_width": 2,
    "show_markers": False,
    "marker_size_line": 8,
    # ── 長條圖 ──
    "barmode": "group",
    "bargap": 0.15,
    "bargroupgap": 0.1,
    "bar_border_width": 0,
    # ── 散點圖 ──
    "marker_symbol": "circle",
    "marker_size": 10,
    "marker_opacity": 0.8,
    "marker_border_width": 0,
    "trendline": "none",
    # ── 圓餅 / 甜甜圈 ──
    "pie_hole": 0.0,
    "pie_direction": "clockwise",
    "pie_sort": True,
    "pie_border_width": 1,
    # ── 熱力圖 ──
    "heatmap_colorscale": "Viridis",
    "heatmap_showscale": True,
    "heatmap_zmin": "",
    "heatmap_zmax": "",
    "heatmap_text_auto": True,
    "heatmap_xside": "bottom",
    # ── 直方圖 ──
    "hist_nbins": 30,
    "hist_norm": "",
    "hist_logy": False,
    "hist_opacity": 0.8,
    # ── 盒鬚圖 ──
    "box_points": "outliers",
    "box_notched": False,
    "boxmode": "group",
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ═══════════════════════════════════════════════════════════════════════════════
# 輔助函式
# ═══════════════════════════════════════════════════════════════════════════════

CHART_TYPES = {
    "bar":       "📊 長條圖",
    "line":      "📈 折線圖",
    "scatter":   "🔵 散點圖",
    "pie":       "🥧 圓餅/甜甜圈",
    "area":      "🏔️ 面積圖",
    "box":       "📦 盒鬚圖",
    "heatmap":   "🌡️ 熱力圖",
    "histogram": "📉 直方圖",
}

PALETTE_MAP: dict = {
    "Plotly":   px.colors.qualitative.Plotly,
    "D3":       px.colors.qualitative.D3,
    "G10":      px.colors.qualitative.G10,
    "T10":      px.colors.qualitative.T10,
    "Set1":     px.colors.qualitative.Set1,
    "Set2":     px.colors.qualitative.Set2,
    "Vivid":    px.colors.qualitative.Vivid,
    "Safe":     px.colors.qualitative.Safe,
    "Pastel":   px.colors.qualitative.Pastel,
    "Bold":     px.colors.qualitative.Bold,
    "Antique":  px.colors.qualitative.Antique,
    "Light24":  px.colors.qualitative.Light24,
    "Dark24":   px.colors.qualitative.Dark24,
    "Alphabet": px.colors.qualitative.Alphabet,
}

LEGEND_POS_MAP = {
    "右上": dict(x=1.01, y=1,    xanchor="left",   yanchor="top"),
    "右下": dict(x=1.01, y=0,    xanchor="left",   yanchor="bottom"),
    "左上": dict(x=0,    y=1,    xanchor="right",  yanchor="top"),
    "左下": dict(x=0,    y=0,    xanchor="right",  yanchor="bottom"),
    "頂中": dict(x=0.5,  y=1.05, xanchor="center", yanchor="bottom"),
    "底中": dict(x=0.5,  y=-0.2, xanchor="center", yanchor="top"),
    "圖內右上": dict(x=0.98, y=0.98, xanchor="right", yanchor="top"),
}


@st.cache_data(show_spinner="🔍 偵測欄位型別...")
def detect_col_types(df: "pd.DataFrame") -> dict[str, str]:
    """回傳 {欄位名: 'numeric' | 'datetime' | 'categorical'}"""
    result = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            result[col] = "numeric"
        else:
            # 嘗試轉日期
            try:
                converted = pd.to_datetime(df[col], errors="raise", infer_datetime_format=True)
                if converted.notna().sum() > len(df) * 0.5:
                    result[col] = "datetime"
                else:
                    result[col] = "categorical"
            except Exception:
                result[col] = "categorical"
    return result


def recommend_chart(x_type: str, y_type: str) -> str:
    mapping = {
        ("datetime",    "numeric"):     "line",
        ("categorical", "numeric"):     "bar",
        ("numeric",     "numeric"):     "scatter",
        ("categorical", "categorical"): "pie",
    }
    return mapping.get((x_type, y_type), "bar")


def _safe_float(val) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def build_chart(df: "pd.DataFrame", ss: dict) -> "go.Figure":
    """依 session_state 設定建立 Plotly Figure。"""
    ct       = ss["chart_type"]
    x        = ss["x_col"]
    ys       = ss["y_cols"] or []
    color    = ss["color_col"] or None
    size     = ss["size_col"] or None
    title    = ss["chart_title"] or ""
    theme    = ss["theme"]
    palette  = PALETTE_MAP.get(ss["color_palette"], px.colors.qualitative.Plotly)

    # --- 折線圖 ---
    if ct == "line":
        if not ys:
            return go.Figure()
        fig = px.line(
            df, x=x, y=ys if len(ys) > 1 else ys[0],
            color=color,
            color_discrete_sequence=palette,
            template=theme,
            title=title,
            line_shape=ss["line_shape"],
            markers=ss["show_markers"],
        )
        if ss["show_labels"]:
            fig.update_traces(textposition="top center", mode="lines+markers+text",
                              text=df[ys[0]].round(2) if ys else None)
        # 時間序列 rangeslider
        col_types = detect_col_types(df)
        if col_types.get(x) == "datetime":
            fig.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(buttons=[
                    dict(count=1,  label="1M",  step="month",  stepmode="backward"),
                    dict(count=3,  label="3M",  step="month",  stepmode="backward"),
                    dict(count=6,  label="6M",  step="month",  stepmode="backward"),
                    dict(count=1,  label="YTD", step="year",   stepmode="todate"),
                    dict(count=1,  label="1Y",  step="year",   stepmode="backward"),
                    dict(step="all", label="All"),
                ])
            )

    # --- 長條圖 ---
    elif ct == "bar":
        if not ys:
            return go.Figure()
        fig = px.bar(
            df, x=x, y=ys if len(ys) > 1 else ys[0],
            color=color,
            color_discrete_sequence=palette,
            template=theme,
            title=title,
            barmode=ss["barmode"],
            text_auto=ss["show_labels"],
        )
        if ss["show_labels"]:
            fig.update_traces(textposition=ss["label_position"])
        fig.update_layout(bargap=ss["bargap"], bargroupgap=ss["bargroupgap"])
        if ss["bar_border_width"] > 0:
            fig.update_traces(marker_line_width=ss["bar_border_width"])

    # --- 散點圖 ---
    elif ct == "scatter":
        if not ys:
            return go.Figure()
        tl_map = {"none": None, "OLS": "ols", "LOWESS": "lowess", "Exponential": "expanding"}
        tl = tl_map.get(ss["trendline"])
        fig = px.scatter(
            df, x=x, y=ys[0],
            color=color,
            size=size,
            color_discrete_sequence=palette,
            template=theme,
            title=title,
            trendline=tl,
            symbol=None,
        )
        fig.update_traces(
            marker=dict(
                symbol=ss["marker_symbol"],
                size=ss["marker_size"],
                opacity=ss["marker_opacity"],
                line=dict(width=ss["marker_border_width"]),
            )
        )

    # --- 圓餅 / 甜甜圈 ---
    elif ct == "pie":
        if not x or not ys:
            return go.Figure()
        fig = px.pie(
            df, names=x, values=ys[0],
            color_discrete_sequence=palette,
            template=theme,
            title=title,
            hole=ss["pie_hole"],
        )
        fig.update_traces(
            direction=ss["pie_direction"],
            sort=ss["pie_sort"],
            marker=dict(line=dict(width=ss["pie_border_width"])),
        )

    # --- 面積圖 ---
    elif ct == "area":
        if not ys:
            return go.Figure()
        fig = px.area(
            df, x=x, y=ys if len(ys) > 1 else ys[0],
            color=color,
            color_discrete_sequence=palette,
            template=theme,
            title=title,
            line_shape=ss["line_shape"],
        )

    # --- 盒鬚圖 ---
    elif ct == "box":
        if not ys:
            return go.Figure()
        pts_map = {"outliers": "outliers", "all": "all",
                   "suspectedoutliers": "suspectedoutliers", "不顯示": False}
        fig = px.box(
            df, x=x, y=ys[0],
            color=color,
            color_discrete_sequence=palette,
            template=theme,
            title=title,
            points=pts_map.get(ss["box_points"], "outliers"),
            notched=ss["box_notched"],
            boxmode=ss["boxmode"],
        )

    # --- 熱力圖 ---
    elif ct == "heatmap":
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) < 2:
            st.warning("熱力圖需要至少 2 個數值欄位（自動使用相關矩陣）。")
            return go.Figure()
        corr = df[num_cols].corr()
        kw: dict = dict(
            color_continuous_scale=ss["heatmap_colorscale"],
            template=theme,
            title=title or "相關矩陣",
            text_auto=ss["heatmap_text_auto"],
            aspect="auto",
        )
        if ss["heatmap_zmin"] != "":
            z = _safe_float(ss["heatmap_zmin"])
            if z is not None:
                kw["zmin"] = z
        if ss["heatmap_zmax"] != "":
            z = _safe_float(ss["heatmap_zmax"])
            if z is not None:
                kw["zmax"] = z
        fig = px.imshow(corr, **kw)
        if not ss["heatmap_showscale"]:
            fig.update_coloraxes(showscale=False)
        if ss["heatmap_xside"] == "top":
            fig.update_xaxes(side="top")

    # --- 直方圖 ---
    elif ct == "histogram":
        if not ys:
            return go.Figure()
        norm = ss["hist_norm"] if ss["hist_norm"] else None
        fig = px.histogram(
            df, x=ys[0],
            color=color,
            color_discrete_sequence=palette,
            template=theme,
            title=title,
            nbins=ss["hist_nbins"],
            histnorm=norm,
            log_y=ss["hist_logy"],
            opacity=ss["hist_opacity"],
        )

    else:
        return go.Figure()

    # ── 套用通用細部調整 ──────────────────────────────────────────────────────
    _apply_fine_tuning(fig, ss)
    return fig


def _apply_fine_tuning(fig: "go.Figure", ss: dict) -> None:
    """就地套用通用細部調整（字型、軸、圖例、版面等）。"""
    # 軸標籤
    x_label = ss.get("x_label") or ""
    y_label = ss.get("y_label") or ""

    layout_kw: dict = dict(
        height=ss["chart_height"],
        margin=dict(t=ss["margin_t"], b=ss["margin_b"],
                    l=ss["margin_l"], r=ss["margin_r"]),
        font=dict(
            family=ss["font_family"],
            size=ss["font_size"],
            color=ss["font_color"],
        ),
        title_font_size=ss["title_font_size"],
        showlegend=ss["show_legend"],
        paper_bgcolor=ss["plot_bgcolor"],
        plot_bgcolor=ss["plot_bgcolor"],
    )

    # 圖例設定
    if ss["show_legend"]:
        pos = LEGEND_POS_MAP.get(ss["legend_position"], LEGEND_POS_MAP["右上"])
        layout_kw["legend"] = dict(
            orientation=ss["legend_orientation"],
            bgcolor=ss["legend_bgcolor"],
            borderwidth=ss["legend_borderwidth"],
            **pos,
        )

    fig.update_layout(**layout_kw)

    # 透明度（非圓餅）
    if ss["chart_type"] not in ("pie",):
        fig.update_traces(opacity=ss["opacity"])

    # X 軸
    x_ax: dict = dict(
        title_text=x_label,
        showgrid=ss["show_xgrid"],
        zeroline=ss["show_zeroline"],
        tickangle=ss["tick_angle"],
    )
    if ss["x_tickformat"]:
        x_ax["tickformat"] = ss["x_tickformat"]
    if ss["x_axis_type"] != "auto":
        x_ax["type"] = ss["x_axis_type"]
    xmin = _safe_float(ss["x_range_min"])
    xmax = _safe_float(ss["x_range_max"])
    if xmin is not None and xmax is not None:
        x_ax["range"] = [xmin, xmax]
    fig.update_xaxes(**x_ax)

    # Y 軸
    y_ax: dict = dict(
        title_text=y_label,
        showgrid=ss["show_ygrid"],
        zeroline=ss["show_zeroline"],
    )
    if ss["y_tickformat"]:
        y_ax["tickformat"] = ss["y_tickformat"]
    if ss["y_axis_type"] != "auto":
        y_ax["type"] = ss["y_axis_type"]
    ymin = _safe_float(ss["y_range_min"])
    ymax = _safe_float(ss["y_range_max"])
    if ymin is not None and ymax is not None:
        y_ax["range"] = [ymin, ymax]
    fig.update_yaxes(**y_ax)

    # 折線寬度 / 線型
    if ss["chart_type"] in ("line", "area"):
        fig.update_traces(
            line=dict(dash=ss["line_dash"], width=ss["line_width"])
        )

    # 資料標籤格式模板
    if ss["show_labels"] and ss["label_template"] and ss["chart_type"] not in ("pie", "heatmap"):
        fig.update_traces(texttemplate=ss["label_template"],
                          textfont_size=ss["label_font_size"])


# ═══════════════════════════════════════════════════════════════════════════════
# 側邊欄
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ 全域設定")
    st.session_state["theme"] = st.selectbox(
        "Plotly 主題",
        ["plotly_white", "plotly_dark", "seaborn", "ggplot2", "simple_white", "plotly"],
        index=["plotly_white", "plotly_dark", "seaborn", "ggplot2", "simple_white", "plotly"]
              .index(st.session_state["theme"]),
    )
    st.session_state["dashboard_mode"] = st.toggle(
        "📋 Dashboard 模式（多圖並排）",
        value=st.session_state["dashboard_mode"],
    )
    if st.session_state["dashboard_mode"]:
        st.caption("最多可新增 4 張圖表並排顯示")

    st.divider()
    st.header("🤖 AI 圖表助理")
    ai_enabled = bool(_OPENAI_KEY)
    if not ai_enabled:
        st.warning("未設定 OPENAI_API_KEY，AI 功能停用")

# ═══════════════════════════════════════════════════════════════════════════════
# 區塊 B — 上傳檔案
# ═══════════════════════════════════════════════════════════════════════════════

uploaded = st.file_uploader(
    "上傳 Excel 檔案（.xlsx / .xls）",
    type=["xlsx", "xls"],
    key="excel_uploader",
)

if uploaded is None:
    st.info(
        "📂 **尚未上傳檔案**\n\n"
        "請上傳 .xlsx 或 .xls 格式的 Excel 檔案，即可開始互動式圖表製作。\n\n"
        "**支援圖表類型**：折線圖 · 長條圖 · 散點圖 · 圓餅/甜甜圈 · 面積圖 · 盒鬚圖 · 熱力圖 · 直方圖\n\n"
        "**使用流程**\n"
        "1. 上傳 Excel 檔案\n"
        "2. 選擇工作表\n"
        "3. 設定圖表類型與欄位\n"
        "4. 微調樣式細節\n"
        "5. 下載 HTML / PNG / CSV"
    )
    st.stop()

# ── 讀取 Excel ──────────────────────────────────────────────────────────────────
if uploaded.name != st.session_state["excel_file_name"]:
    try:
        xls = pd.ExcelFile(uploaded)
        st.session_state["excel_sheets"] = xls.sheet_names
        st.session_state["excel_file_name"] = uploaded.name
        # 預設讀第一張 sheet
        st.session_state["excel_df"] = xls.parse(xls.sheet_names[0])
        # 重設圖表設定
        for _reset_k in ("x_col", "y_cols", "color_col", "size_col",
                          "chart_title", "x_label", "y_label"):
            st.session_state[_reset_k] = _DEFAULTS[_reset_k]
        st.session_state["chart_configs"] = []
        st.session_state["ai_messages"] = []
    except Exception as exc:
        st.error(f"❌ 無法讀取 Excel 檔案：{exc}")
        st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# 區塊 C — 資料預覽
# ═══════════════════════════════════════════════════════════════════════════════

df: "pd.DataFrame" = st.session_state["excel_df"]
sheets = st.session_state["excel_sheets"]

col_sheet, col_stats = st.columns([2, 3])

with col_sheet:
    if len(sheets) > 1:
        chosen_sheet = st.selectbox("📋 選擇工作表", sheets)
        try:
            xls2 = pd.ExcelFile(uploaded)
            df = xls2.parse(chosen_sheet)
            st.session_state["excel_df"] = df
        except Exception:
            pass

with col_stats:
    col_types = detect_col_types(df)
    type_emoji = {"numeric": "🔢", "datetime": "📅", "categorical": "🔤"}
    st.markdown(
        f"**{len(df):,} 行 × {len(df.columns)} 欄** ｜ "
        + " · ".join(f"{type_emoji.get(t, '❓')} {c}" for c, t in col_types.items())
    )

with st.expander("📋 資料預覽（前 10 行）", expanded=True):
    st.dataframe(df.head(10), use_container_width=True)

with st.expander("📊 基本統計摘要", expanded=False):
    st.dataframe(df.describe(include="all"), use_container_width=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# 區塊 D — 圖表設定表單 + 細部調整
# ═══════════════════════════════════════════════════════════════════════════════

all_cols    = list(df.columns)
num_cols    = [c for c in all_cols if col_types.get(c) == "numeric"]
cat_cols    = [c for c in all_cols if col_types.get(c) == "categorical"]
date_cols   = [c for c in all_cols if col_types.get(c) == "datetime"]

cfg_col, preview_col = st.columns([1, 2], gap="medium")

with cfg_col:
    st.subheader("🖊️ 圖表設定")

    # ── 基本設定 ─────────────────────────────────────────────────────────────
    # 自動推薦提示
    x_default   = st.session_state["x_col"] or (all_cols[0] if all_cols else None)
    y_default   = st.session_state["y_cols"] or ([num_cols[0]] if num_cols else [])

    st.session_state["x_col"] = st.selectbox(
        "X 軸欄位",
        all_cols,
        index=all_cols.index(x_default) if x_default in all_cols else 0,
    )
    st.session_state["y_cols"] = st.multiselect(
        "Y 軸欄位（可多選）",
        all_cols,
        default=[c for c in y_default if c in all_cols],
    )

    # 自動推薦圖表
    x_type = col_types.get(st.session_state["x_col"], "categorical")
    y_type = col_types.get(st.session_state["y_cols"][0], "numeric") if st.session_state["y_cols"] else "numeric"
    recommended = recommend_chart(x_type, y_type)

    chart_options = list(CHART_TYPES.keys())
    chart_labels  = [
        f"⭐ {CHART_TYPES[k]}（推薦）" if k == recommended else CHART_TYPES[k]
        for k in chart_options
    ]
    current_ct = st.session_state["chart_type"]
    st.session_state["chart_type"] = chart_options[
        st.selectbox(
            "圖表類型",
            range(len(chart_options)),
            index=chart_options.index(current_ct) if current_ct in chart_options else 0,
            format_func=lambda i: chart_labels[i],
        )
    ]

    # 色彩分組 / 大小
    color_opts = [""] + all_cols
    size_opts  = [""] + num_cols
    st.session_state["color_col"] = st.selectbox(
        "色彩分組欄位（可選）",
        color_opts,
        index=color_opts.index(st.session_state["color_col"])
              if st.session_state["color_col"] in color_opts else 0,
    )
    if st.session_state["chart_type"] == "scatter":
        st.session_state["size_col"] = st.selectbox(
            "氣泡大小欄位（可選）",
            size_opts,
            index=size_opts.index(st.session_state["size_col"])
                  if st.session_state["size_col"] in size_opts else 0,
        )

    # 標題 / 軸標籤
    st.session_state["chart_title"] = st.text_input("圖表標題", value=st.session_state["chart_title"])
    st.session_state["x_label"]     = st.text_input("X 軸標籤", value=st.session_state["x_label"])
    st.session_state["y_label"]     = st.text_input("Y 軸標籤", value=st.session_state["y_label"])

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # 細部調整面板
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("🎛️ 細部調整")

    # ── 色彩設定 ─────────────────────────────────────────────────────────────
    with st.expander("🎨 色彩設定"):
        st.session_state["color_palette"] = st.selectbox(
            "離散色盤",
            list(PALETTE_MAP.keys()),
            index=list(PALETTE_MAP.keys()).index(st.session_state["color_palette"]),
        )
        st.session_state["opacity"] = st.slider("整體透明度", 0.1, 1.0,
                                                  float(st.session_state["opacity"]), 0.05)
        st.session_state["plot_bgcolor"] = st.color_picker(
            "背景色", value=st.session_state["plot_bgcolor"])

    # ── 字型設定 ─────────────────────────────────────────────────────────────
    with st.expander("🔤 字型設定"):
        font_opts = ["Open Sans", "Arial", "Courier New", "Times New Roman", "Verdana"]
        st.session_state["font_family"] = st.selectbox(
            "字型",
            font_opts,
            index=font_opts.index(st.session_state["font_family"])
                  if st.session_state["font_family"] in font_opts else 0,
        )
        st.session_state["font_size"]        = st.slider("全域字級", 10, 24, int(st.session_state["font_size"]))
        st.session_state["title_font_size"]  = st.slider("標題字級", 12, 40, int(st.session_state["title_font_size"]))
        st.session_state["font_color"]       = st.color_picker("字體顏色", value=st.session_state["font_color"])

    # ── 座標軸 ────────────────────────────────────────────────────────────────
    with st.expander("📐 座標軸"):
        ax_c1, ax_c2 = st.columns(2)
        with ax_c1:
            st.session_state["x_range_min"] = st.text_input("X 軸最小值", value=str(st.session_state["x_range_min"]))
            st.session_state["x_range_max"] = st.text_input("X 軸最大值", value=str(st.session_state["x_range_max"]))
        with ax_c2:
            st.session_state["y_range_min"] = st.text_input("Y 軸最小值", value=str(st.session_state["y_range_min"]))
            st.session_state["y_range_max"] = st.text_input("Y 軸最大值", value=str(st.session_state["y_range_max"]))

        axis_type_opts = ["auto", "linear", "log", "date", "category"]
        at_c1, at_c2 = st.columns(2)
        with at_c1:
            st.session_state["x_axis_type"] = st.selectbox(
                "X 軸型態", axis_type_opts,
                index=axis_type_opts.index(st.session_state["x_axis_type"]))
        with at_c2:
            st.session_state["y_axis_type"] = st.selectbox(
                "Y 軸型態", axis_type_opts,
                index=axis_type_opts.index(st.session_state["y_axis_type"]))

        st.session_state["tick_angle"]   = st.slider("Tick 角度", -90, 90, int(st.session_state["tick_angle"]))
        tc1, tc2 = st.columns(2)
        with tc1:
            st.session_state["x_tickformat"] = st.text_input("X Tick 格式（d3-format）",
                                                               value=st.session_state["x_tickformat"],
                                                               help='例如 ".1f" ".0%" ",.0f"')
        with tc2:
            st.session_state["y_tickformat"] = st.text_input("Y Tick 格式（d3-format）",
                                                               value=st.session_state["y_tickformat"])
        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            st.session_state["show_xgrid"]   = st.toggle("X 格線",    value=st.session_state["show_xgrid"])
        with gc2:
            st.session_state["show_ygrid"]   = st.toggle("Y 格線",    value=st.session_state["show_ygrid"])
        with gc3:
            st.session_state["show_zeroline"] = st.toggle("零線",     value=st.session_state["show_zeroline"])

    # ── 圖例 ─────────────────────────────────────────────────────────────────
    with st.expander("📋 圖例"):
        st.session_state["show_legend"] = st.toggle("顯示圖例", value=st.session_state["show_legend"])
        if st.session_state["show_legend"]:
            lp_opts = list(LEGEND_POS_MAP.keys())
            st.session_state["legend_position"] = st.selectbox(
                "圖例位置", lp_opts,
                index=lp_opts.index(st.session_state["legend_position"])
                      if st.session_state["legend_position"] in lp_opts else 0,
            )
            lo_c1, lo_c2 = st.columns(2)
            with lo_c1:
                st.session_state["legend_orientation"] = st.radio(
                    "方向", ["v", "h"],
                    format_func=lambda v: "垂直" if v == "v" else "水平",
                    index=["v","h"].index(st.session_state["legend_orientation"]),
                )
            with lo_c2:
                st.session_state["legend_borderwidth"] = st.slider(
                    "邊框寬度", 0, 4, int(st.session_state["legend_borderwidth"]))
            st.session_state["legend_bgcolor"] = st.color_picker(
                "圖例背景色", value=st.session_state["legend_bgcolor"])

    # ── 版面間距 ──────────────────────────────────────────────────────────────
    with st.expander("📏 版面間距"):
        st.session_state["chart_height"] = st.slider("圖表高度（px）", 300, 1200,
                                                       int(st.session_state["chart_height"]), 50)
        mg_c = st.columns(4)
        for _i, (_mk, _ml) in enumerate(zip(
            ["margin_t", "margin_b", "margin_l", "margin_r"],
            ["上邊距", "下邊距", "左邊距", "右邊距"]
        )):
            with mg_c[_i]:
                st.session_state[_mk] = st.number_input(_ml, 0, 300, int(st.session_state[_mk]), step=10)

    # ── 資料標籤 ─────────────────────────────────────────────────────────────
    with st.expander("🏷️ 資料標籤"):
        st.session_state["show_labels"] = st.toggle("顯示資料標籤", value=st.session_state["show_labels"])
        if st.session_state["show_labels"]:
            lpos_opts = ["outside", "inside", "auto", "none",
                         "top center", "bottom center", "middle center"]
            st.session_state["label_position"] = st.selectbox(
                "標籤位置", lpos_opts,
                index=lpos_opts.index(st.session_state["label_position"])
                      if st.session_state["label_position"] in lpos_opts else 0,
            )
            st.session_state["label_template"] = st.text_input(
                "格式模板（texttemplate）",
                value=st.session_state["label_template"],
                help='例如 "%{y:.1f}" "%{y:,.0f}" "%{percent:.1%}"',
            )
            st.session_state["label_font_size"] = st.slider(
                "標籤字級", 8, 24, int(st.session_state["label_font_size"]))

    # ── 圖表類型專屬調整 ──────────────────────────────────────────────────────
    ct = st.session_state["chart_type"]

    if ct in ("line", "area"):
        with st.expander("📈 折線設定"):
            dash_opts  = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
            shape_opts = ["linear", "spline", "hv", "vh", "hvh", "vhv"]
            ld_c1, ld_c2 = st.columns(2)
            with ld_c1:
                st.session_state["line_dash"] = st.selectbox(
                    "線型", dash_opts,
                    index=dash_opts.index(st.session_state["line_dash"])
                          if st.session_state["line_dash"] in dash_opts else 0,
                )
            with ld_c2:
                st.session_state["line_shape"] = st.selectbox(
                    "線形狀", shape_opts,
                    index=shape_opts.index(st.session_state["line_shape"])
                          if st.session_state["line_shape"] in shape_opts else 0,
                )
            st.session_state["line_width"]      = st.slider("線寬（px）", 1, 10, int(st.session_state["line_width"]))
            st.session_state["show_markers"]    = st.toggle("顯示資料點", value=st.session_state["show_markers"])
            if st.session_state["show_markers"]:
                st.session_state["marker_size_line"] = st.slider(
                    "點大小", 4, 24, int(st.session_state["marker_size_line"]))

    elif ct == "bar":
        with st.expander("📊 長條圖設定"):
            bm_opts = ["group", "stack", "overlay", "relative"]
            bm_labels = {"group": "分組", "stack": "堆疊", "overlay": "重疊", "relative": "相對"}
            st.session_state["barmode"] = st.radio(
                "群組模式",
                bm_opts,
                format_func=lambda v: bm_labels.get(v, v),
                index=bm_opts.index(st.session_state["barmode"])
                      if st.session_state["barmode"] in bm_opts else 0,
                horizontal=True,
            )
            bg_c1, bg_c2, bg_c3 = st.columns(3)
            with bg_c1:
                st.session_state["bargap"]         = st.slider("Bar 間距", 0.0, 0.5, float(st.session_state["bargap"]), 0.05)
            with bg_c2:
                st.session_state["bargroupgap"]    = st.slider("群組間距", 0.0, 0.5, float(st.session_state["bargroupgap"]), 0.05)
            with bg_c3:
                st.session_state["bar_border_width"] = st.slider("邊框寬度", 0, 5, int(st.session_state["bar_border_width"]))

    elif ct == "scatter":
        with st.expander("🔵 散點圖設定"):
            sym_opts = ["circle", "square", "diamond", "cross", "x",
                        "triangle-up", "triangle-down", "star", "hexagon", "pentagon"]
            st.session_state["marker_symbol"] = st.selectbox(
                "點形狀", sym_opts,
                index=sym_opts.index(st.session_state["marker_symbol"])
                      if st.session_state["marker_symbol"] in sym_opts else 0,
            )
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.session_state["marker_size"]          = st.slider("點大小", 4, 40, int(st.session_state["marker_size"]))
            with sc2:
                st.session_state["marker_opacity"]       = st.slider("點透明度", 0.1, 1.0, float(st.session_state["marker_opacity"]), 0.05)
            with sc3:
                st.session_state["marker_border_width"]  = st.slider("邊框寬度", 0, 5, int(st.session_state["marker_border_width"]))
            tl_opts = ["none", "OLS", "LOWESS", "Exponential"]
            tl_labels = {"none": "無", "OLS": "線性（OLS）", "LOWESS": "LOWESS 平滑", "Exponential": "指數"}
            st.session_state["trendline"] = st.selectbox(
                "趨勢線",
                tl_opts,
                format_func=lambda v: tl_labels.get(v, v),
                index=tl_opts.index(st.session_state["trendline"])
                      if st.session_state["trendline"] in tl_opts else 0,
            )

    elif ct == "pie":
        with st.expander("🍩 圓餅/甜甜圈設定"):
            st.session_state["pie_hole"]       = st.slider("中心孔洞（0=圓餅，>0=甜甜圈）", 0.0, 0.9, float(st.session_state["pie_hole"]), 0.05)
            pi_c1, pi_c2, pi_c3 = st.columns(3)
            with pi_c1:
                st.session_state["pie_direction"] = st.radio(
                    "排列方向", ["clockwise", "counterclockwise"],
                    format_func=lambda v: "順時針" if v == "clockwise" else "逆時針",
                    index=["clockwise","counterclockwise"].index(st.session_state["pie_direction"]),
                )
            with pi_c2:
                st.session_state["pie_sort"]      = st.toggle("依大小排序", value=st.session_state["pie_sort"])
            with pi_c3:
                st.session_state["pie_border_width"] = st.slider("邊框寬度", 0, 5, int(st.session_state["pie_border_width"]))

    elif ct == "heatmap":
        with st.expander("🌡️ 熱力圖設定"):
            hm_scales = ["Viridis", "Plasma", "Inferno", "Magma", "RdBu", "RdBu_r",
                         "Blues", "Greens", "Reds", "Greys", "Hot", "Jet", "Bluered"]
            st.session_state["heatmap_colorscale"] = st.selectbox(
                "色階",
                hm_scales,
                index=hm_scales.index(st.session_state["heatmap_colorscale"])
                      if st.session_state["heatmap_colorscale"] in hm_scales else 0,
            )
            hm_c1, hm_c2 = st.columns(2)
            with hm_c1:
                st.session_state["heatmap_showscale"]  = st.toggle("顯示色軸", value=st.session_state["heatmap_showscale"])
                st.session_state["heatmap_text_auto"]  = st.toggle("顯示數值", value=st.session_state["heatmap_text_auto"])
            with hm_c2:
                st.session_state["heatmap_zmin"] = st.text_input("Z 最小值", value=str(st.session_state["heatmap_zmin"]))
                st.session_state["heatmap_zmax"] = st.text_input("Z 最大值", value=str(st.session_state["heatmap_zmax"]))
            st.session_state["heatmap_xside"] = st.radio(
                "X 軸位置", ["bottom", "top"],
                format_func=lambda v: "下方" if v == "bottom" else "上方",
                index=["bottom","top"].index(st.session_state["heatmap_xside"]),
                horizontal=True,
            )

    elif ct == "histogram":
        with st.expander("📉 直方圖設定"):
            st.session_state["hist_nbins"]   = st.slider("Bin 數量", 5, 200, int(st.session_state["hist_nbins"]))
            norm_opts = ["", "percent", "probability", "density", "probability density"]
            norm_labels = {"": "計數（count）", "percent": "百分比", "probability": "機率",
                           "density": "密度", "probability density": "機率密度"}
            st.session_state["hist_norm"] = st.selectbox(
                "正規化",
                norm_opts,
                format_func=lambda v: norm_labels.get(v, v),
                index=norm_opts.index(st.session_state["hist_norm"])
                      if st.session_state["hist_norm"] in norm_opts else 0,
            )
            hi_c1, hi_c2 = st.columns(2)
            with hi_c1:
                st.session_state["hist_logy"]    = st.toggle("Log Y 軸", value=st.session_state["hist_logy"])
            with hi_c2:
                st.session_state["hist_opacity"] = st.slider("透明度", 0.3, 1.0, float(st.session_state["hist_opacity"]), 0.05)

    elif ct == "box":
        with st.expander("📦 盒鬚圖設定"):
            pts_opts   = ["outliers", "all", "suspectedoutliers", "不顯示"]
            pts_labels = {"outliers": "離群值", "all": "全部資料點",
                          "suspectedoutliers": "可疑離群值", "不顯示": "不顯示"}
            bx_c1, bx_c2, bx_c3 = st.columns(3)
            with bx_c1:
                st.session_state["box_points"] = st.selectbox(
                    "顯示點",
                    pts_opts,
                    format_func=lambda v: pts_labels.get(v, v),
                    index=pts_opts.index(st.session_state["box_points"])
                          if st.session_state["box_points"] in pts_opts else 0,
                )
            with bx_c2:
                st.session_state["box_notched"] = st.toggle("缺口（Notch）", value=st.session_state["box_notched"])
            with bx_c3:
                bxm_opts = ["group", "overlay"]
                st.session_state["boxmode"] = st.radio(
                    "群組模式",
                    bxm_opts,
                    format_func=lambda v: "分組" if v == "group" else "重疊",
                    index=bxm_opts.index(st.session_state["boxmode"]),
                )

    # ── Dashboard：新增至儀表板 ──────────────────────────────────────────────
    if st.session_state["dashboard_mode"]:
        st.divider()
        dash_c1, dash_c2 = st.columns(2)
        with dash_c1:
            if st.button("➕ 加入儀表板", use_container_width=True):
                if len(st.session_state["chart_configs"]) < 4:
                    # 快照目前設定
                    snapshot = {k: st.session_state[k] for k in _DEFAULTS if k not in ("excel_df",)}
                    st.session_state["chart_configs"].append(snapshot)
                    st.success(f"已加入第 {len(st.session_state['chart_configs'])} 張圖")
                else:
                    st.warning("最多 4 張圖")
        with dash_c2:
            if st.button("🗑️ 清除儀表板", use_container_width=True):
                st.session_state["chart_configs"] = []


# ═══════════════════════════════════════════════════════════════════════════════
# 區塊 E — AI Agent 協助
# ═══════════════════════════════════════════════════════════════════════════════

with preview_col:
    if _OPENAI_KEY:
        with st.expander("🤖 AI 圖表助理", expanded=False):
            # 顯示對話歷史
            for msg in st.session_state["ai_messages"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            user_input = st.chat_input("描述您想要的圖表（例如：幫我比較各月份業績）")
            if user_input:
                st.session_state["ai_messages"].append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)

                # 系統提示包含欄位資訊
                col_info = "\n".join(
                    f"- {c}（{col_types.get(c, 'unknown')}）" for c in all_cols
                )
                system_prompt = (
                    "你是一位資料視覺化專家。使用者上傳了以下 Excel 欄位：\n"
                    f"{col_info}\n\n"
                    "請根據使用者描述，建議最適合的 Plotly 圖表設定，"
                    "以 JSON 格式回傳以下欄位（只回傳 JSON，不要其他文字）：\n"
                    '{"chart_type":"bar|line|scatter|pie|area|box|heatmap|histogram",'
                    '"x_col":"欄位名稱","y_cols":["欄位名稱"],'
                    '"color_col":"欄位名稱或空字串",'
                    '"chart_title":"建議標題","reason":"推薦理由（繁體中文，1-2句）"}'
                )

                with st.spinner("AI 分析中..."):
                    try:
                        client = OpenAI(api_key=_OPENAI_KEY)
                        resp = client.chat.completions.create(
                            model="gpt-4.1-mini",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                *[{"role": m["role"], "content": m["content"]}
                                  for m in st.session_state["ai_messages"]],
                            ],
                            temperature=0.3,
                            max_tokens=400,
                        )
                        raw = resp.choices[0].message.content.strip()
                        # 嘗試解析 JSON
                        try:
                            # 提取 JSON 區塊
                            import re as _re
                            json_match = _re.search(r'\{.*\}', raw, _re.DOTALL)
                            suggestion = json.loads(json_match.group() if json_match else raw)
                            # 套用建議到 session_state
                            for _key in ("chart_type", "x_col", "y_cols", "color_col", "chart_title"):
                                if _key in suggestion and suggestion[_key]:
                                    val = suggestion[_key]
                                    if _key == "x_col" and val in all_cols:
                                        st.session_state[_key] = val
                                    elif _key == "y_cols":
                                        valid = [c for c in (val if isinstance(val, list) else [val]) if c in all_cols]
                                        if valid:
                                            st.session_state[_key] = valid
                                    elif _key == "color_col":
                                        st.session_state[_key] = val if val in all_cols else ""
                                    elif _key == "chart_type" and val in CHART_TYPES:
                                        st.session_state[_key] = val
                                    elif _key == "chart_title":
                                        st.session_state[_key] = val

                            reason = suggestion.get("reason", "")
                            reply = (
                                f"✅ 已套用建議設定：\n"
                                f"- 圖表類型：**{CHART_TYPES.get(suggestion.get('chart_type',''), suggestion.get('chart_type',''))}**\n"
                                f"- X 軸：**{suggestion.get('x_col','')}**\n"
                                f"- Y 軸：**{', '.join(suggestion.get('y_cols',[]))}**\n"
                                f"\n💡 {reason}"
                            )
                        except Exception:
                            reply = raw  # 直接顯示原始回應

                        st.session_state["ai_messages"].append({"role": "assistant", "content": reply})
                        with st.chat_message("assistant"):
                            st.markdown(reply)
                        st.rerun()
                    except Exception as ai_err:
                        st.error(f"AI 呼叫失敗：{ai_err}")

    # ═══════════════════════════════════════════════════════════════════════════
    # 區塊 F — 圖表渲染
    # ═══════════════════════════════════════════════════════════════════════════

    st.subheader("📊 圖表預覽")

    if not st.session_state["dashboard_mode"]:
        # 單圖模式
        try:
            fig = build_chart(df, st.session_state)
            st.plotly_chart(fig, use_container_width=True, key="main_chart")
        except Exception as chart_err:
            st.error(f"圖表渲染失敗：{chart_err}")
            fig = None

    else:
        # Dashboard 模式
        configs = st.session_state["chart_configs"]
        if not configs:
            # 先顯示目前設定的預覽
            try:
                fig = build_chart(df, st.session_state)
                st.plotly_chart(fig, use_container_width=True, key="dash_preview")
            except Exception as chart_err:
                st.error(f"圖表渲染失敗：{chart_err}")
                fig = None
            st.info("👆 請點擊左側「➕ 加入儀表板」將圖表加入多圖排版")
        else:
            # 2-column grid
            grid_cols = st.columns(min(2, len(configs)), gap="small")
            for i, cfg in enumerate(configs):
                with grid_cols[i % 2]:
                    try:
                        dash_fig = build_chart(df, cfg)
                        st.plotly_chart(dash_fig, use_container_width=True, key=f"dash_{i}")
                    except Exception as dash_err:
                        st.error(f"圖表 {i+1} 失敗：{dash_err}")

    # ═══════════════════════════════════════════════════════════════════════════
    # 區塊 G — 匯出
    # ═══════════════════════════════════════════════════════════════════════════

    st.divider()
    st.subheader("💾 匯出")

    try:
        _export_fig = build_chart(df, st.session_state)
    except Exception:
        _export_fig = None

    exp_c1, exp_c2, exp_c3 = st.columns(3)

    with exp_c1:
        if _export_fig is not None:
            html_str = _export_fig.to_html(include_plotlyjs="cdn", full_html=True)
            st.download_button(
                label="🌐 下載 HTML（互動式）",
                data=html_str,
                file_name=f"{st.session_state['excel_file_name'].rsplit('.',1)[0]}_chart.html",
                mime="text/html",
                use_container_width=True,
            )

    with exp_c2:
        if _export_fig is not None:
            if HAS_KALEIDO:
                try:
                    png_bytes = _export_fig.to_image(format="png", scale=2)
                    st.download_button(
                        label="🖼️ 下載 PNG（2x）",
                        data=png_bytes,
                        file_name=f"{st.session_state['excel_file_name'].rsplit('.',1)[0]}_chart.png",
                        mime="image/png",
                        use_container_width=True,
                    )
                except Exception as png_err:
                    st.warning(f"PNG 匯出失敗：{png_err}")
            else:
                st.button("🖼️ 下載 PNG（需安裝 kaleido）",
                          disabled=True, use_container_width=True,
                          help="請執行 pip install kaleido")

    with exp_c3:
        csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            label="📄 下載原始 CSV",
            data=csv_bytes,
            file_name=f"{st.session_state['excel_file_name'].rsplit('.',1)[0]}_data.csv",
            mime="text/csv",
            use_container_width=True,
        )
