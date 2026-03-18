# utils/rich_styles.py
# -*- coding: utf-8 -*-
"""
Anya 富文本輸出美化工具
靈感來自 Compose Richtext（https://halilibo.com/compose-richtext/）
色彩主題：Anya Forger（Spy x Family）

  珊瑚粉  #D97B72 — 主標題、清單符號
  金邊黃  #C8A43A — blockquote 邊框
  深褐色  #4A2F1A — 表格標頭底色（制服色）
  淡珊瑚  #FFF5F2 — 背景色調
  淡金色  #FFF8E8 — 行內程式碼底色

公開函式：
  inject_rich_styles()                       — 全域 CSS 注入（冪等）
  render_report_header(query, mode_label,
                       source_count, word_count) — 研究報告卡片標頭
  render_source_chips(sources)               — 來源 URL chip 連結行
"""

from __future__ import annotations

from datetime import datetime
from urllib.parse import urlparse

import streamlit as st

# ── Anya Forger 色彩常數
# 取自 Spy x Family Anya Forger 海報
_CORAL   = "#D97B72"   # 珊瑚粉（海報背景）
_GOLD    = "#C8A43A"   # 金邊黃（制服金邊）
_BROWN   = "#4A2F1A"   # 深褐色（制服底色）
_LIGHT   = "#FFF5F2"   # 淡珊瑚（背景色調）
_STRIPE  = "#FDF0ED"   # 斑馬紋偶數列
_BORDER  = "#F2D5CF"   # 表格分隔線
_CODE_BG = "#FFF8E8"   # 行內程式碼底色（淡金）

# ── CSS（只套用在 .stMarkdown 範圍，不影響 widget）
_RICH_CSS = f"""
<style>
/* ── 標題 — Anya 珊瑚粉 ── */
.stMarkdown h1 {{
    color: {_CORAL};
    border-bottom: 2px solid {_BORDER};
    padding-bottom: .3em;
    margin-top: 1.2em;
}}
.stMarkdown h2 {{
    color: {_CORAL};
    margin-top: 1em;
}}
.stMarkdown h3,
.stMarkdown h4 {{
    color: {_BROWN};
}}

/* ── Blockquote — 金邊框 + 淡珊瑚背景 ── */
/* 靈感自 Compose Richtext BlockQuote 元件 */
.stMarkdown blockquote {{
    border-left: 4px solid {_GOLD};
    background: {_LIGHT};
    padding: .7em 1.2em;
    border-radius: 0 8px 8px 0;
    margin: .8em 0;
    color: {_BROWN};
}}

/* ── 表格 — 深褐色標頭 + 斑馬紋 ── */
.stMarkdown table {{
    border-collapse: collapse;
    width: 100%;
}}
.stMarkdown th {{
    background: {_BROWN};
    color: {_GOLD};          /* 金色文字搭深褐底，如制服金邊 */
    padding: 8px 12px;
    text-align: left;
    letter-spacing: .03em;
}}
.stMarkdown td {{
    padding: 8px 12px;
    border-bottom: 1px solid {_BORDER};
}}
.stMarkdown tr:nth-child(even) td {{
    background: {_STRIPE};
}}

/* ── 行內程式碼 — 淡金底色 ── */
.stMarkdown code:not(pre > code) {{
    background: {_CODE_BG};
    padding: 2px 6px;
    border-radius: 4px;
    font-size: .88em;
    color: {_BROWN};
}}

/* ── 清單項目符號 — 珊瑚粉 ── */
.stMarkdown ul li::marker {{ color: {_CORAL}; }}
.stMarkdown ol li::marker {{ color: {_CORAL}; font-weight: bold; }}
</style>
"""

_INJECTED_KEY = "_rich_styles_injected"


def inject_rich_styles() -> None:
    """將品牌富文本 CSS 注入頁面（冪等：只注入一次）。

    在每個頁面的 set_page_config() 之後呼叫即可。
    效果（Anya Forger 主題）：
      - h1/h2 顯示珊瑚粉 #D97B72
      - h3/h4 顯示深褐色 #4A2F1A
      - blockquote 顯示金色左邊框 + 淡珊瑚背景
      - table 標頭為深褐底金色字（制服配色）+ 斑馬紋
      - 行內 `code` 顯示淡金底圓角
      - 清單項目符號為珊瑚粉
    """
    if st.session_state.get(_INJECTED_KEY):
        return
    st.markdown(_RICH_CSS, unsafe_allow_html=True)
    st.session_state[_INJECTED_KEY] = True


def render_report_header(
    query: str,
    mode_label: str,
    source_count: int,
    word_count: int,
) -> None:
    """渲染研究報告卡片標頭（使用 Streamlit 原生元件）。

    Args:
        query:        研究主題文字
        mode_label:   研究模式標籤，如「🌲 深度模式（5–10 分鐘）」
        source_count: 來源數量
        word_count:   報告字數（空白分割）
    """
    with st.container(border=True):
        col_title, col_date = st.columns([5, 1])
        col_title.markdown(":material/description: **研究報告**")
        col_date.markdown(
            f":small[:gray[{datetime.now().strftime('%Y-%m-%d')}]]",
            text_alignment="right",
        )

        st.markdown(f"**主題：** {query}")

        col_a, col_b, col_c = st.columns(3)
        col_a.markdown(f":material/travel_explore: :small[{mode_label}]")
        col_b.markdown(f":material/link: :small[{source_count} 筆來源]")
        col_c.markdown(f":material/article: :small[約 {word_count:,} 字]")


def render_source_chips(sources: list[str], max_show: int = 20) -> None:
    """將來源 URL 列表顯示為 chip 連結行。

    只取 domain 部分（netloc），最多顯示 max_show 筆，
    超過顯示「+N 更多」。

    Args:
        sources:  來源 URL 字串列表
        max_show: 最多顯示幾筆（預設 20）
    """
    if not sources:
        return

    shown = sources[:max_show]
    extra = len(sources) - max_show if len(sources) > max_show else 0

    chips: list[str] = []
    for url in shown:
        try:
            domain = urlparse(url).netloc or url
        except Exception:
            domain = url
        chips.append(f"[{domain}]({url})")

    chip_line = " · ".join(chips)
    if extra > 0:
        chip_line += f" :gray[+{extra} 更多]"

    st.markdown(
        f":material/link: **參考來源：** {chip_line}",
        help="點擊連結開啟原始頁面",
    )
