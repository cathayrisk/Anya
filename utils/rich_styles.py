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

import markdown as _md_lib
import streamlit as st

# ── Anya Forger 色彩常數
# 取自 Spy x Family Anya Forger 海報
_CORAL   = "#C05A50"   # 珊瑚深紅（海報背景加深）
_GOLD    = "#C8A43A"   # 金邊黃（制服金邊）
_BROWN   = "#4A2F1A"   # 深褐色（制服底色）
_LIGHT   = "#FFF5F2"   # 淡珊瑚（背景色調）
_STRIPE  = "#FDF0ED"   # 斑馬紋偶數列
_BORDER  = "#F2D5CF"   # 表格分隔線
_CODE_BG = "#EDE8E6"   # 行內程式碼底色（淡暖灰，不搶眼）

# ── CSS（只套用在 .stMarkdown 範圍，不影響 widget）
_RICH_CSS = f"""
<style>
/* ── 標題 — Anya 珊瑚粉 ── */
.stMarkdown h1 {{
    color: {_CORAL};
    border-bottom: 1px solid {_BORDER};   /* 改細：1px 比 2px 輕盈 */
    padding-bottom: .25em;
    margin-top: 0.9em;
    margin-bottom: 0.2em;
}}
.stMarkdown h2 {{
    color: {_CORAL};
    margin-top: 0.8em;
    margin-bottom: 0.2em;
}}
.stMarkdown h3,
.stMarkdown h4 {{
    color: #7A4030;   /* 暖褐色：比 h1/h2 深但比正文有層次感 */
    margin-top: .7em;
    margin-bottom: 0.1em;
}}
/* Tab / container 內第一個標題不需要上方空白 */
.stMarkdown h1:first-child,
.stMarkdown h2:first-child,
.stMarkdown h3:first-child {{
    margin-top: 0.2em;
}}

/* ── 有序清單 — 層級縮排加強 ── */
.stMarkdown ol {{
    padding-left: 1.6em;
    line-height: 1.7;
}}
.stMarkdown ol li {{
    margin-bottom: 0.35em;
}}
.stMarkdown ol li ul,
.stMarkdown ol li ol {{
    margin-top: 0.25em;
    padding-left: 1.4em;
}}

/* ── 無序清單 — 與 ol 統一縮排與間距 ── */
.stMarkdown ul {{
    padding-left: 1.6em;
    line-height: 1.7;
}}
.stMarkdown ul li {{
    margin-bottom: 0.35em;
}}
.stMarkdown ul li ul,
.stMarkdown ul li ol {{
    margin-top: 0.25em;
    padding-left: 1.4em;
}}

/* ── li 內部 p 標籤（Markdown parser 有時自動包 p）不加額外間距 ── */
.stMarkdown li > p {{
    margin: 0;
    padding: 0;
}}

/* ── 段落間距 ── */
.stMarkdown p {{
    margin-top: 0.3em;
    margin-bottom: 0.5em;
    line-height: 1.7;
}}

/* ── Blockquote — 金邊框 + 淡珊瑚背景 ── */
/* 靈感自 Compose Richtext BlockQuote 元件 */
.stMarkdown blockquote {{
    border-left: 4px solid {_GOLD};
    background: {_LIGHT};
    padding: .6em 1.2em;
    border-radius: 0 8px 8px 0;
    margin-top: .6em;
    margin-bottom: .3em;   /* 縮小下方間距，避免與後續段落缺口過大 */
    color: {_BROWN};
}}
/* blockquote 內的 p 標籤不加額外間距 */
.stMarkdown blockquote p {{
    margin: 0;
    padding: 0;
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

/* ── 行內程式碼 — 淡暖灰底色 ── */
.stMarkdown code:not(pre > code) {{
    background: {_CODE_BG};
    padding: 2px 6px;
    border-radius: 4px;
    font-size: .88em;
    color: {_BROWN};
}}

/* ── 程式碼區塊（triple-backtick）— 覆蓋 Streamlit 預設黃底 ── */
.stMarkdown pre {{
    background: #F5F1EE !important;   /* 暖灰白，取代 Streamlit 預設黃底 */
    border: 1px solid {_BORDER} !important;
    border-radius: 8px;
    padding: 1em 1.4em;
    line-height: 1.65;
    overflow-x: auto;
}}
.stMarkdown pre > code {{
    background: transparent !important;
    font-size: 0.93em;                /* 比預設略小但清晰可讀 */
    color: #2E1F18;                   /* 深褐近黑，最高對比 */
    padding: 0;
    border-radius: 0;
}}

/* ── 清單項目符號 — 珊瑚粉 ── */
.stMarkdown ul li::marker {{ color: {_CORAL}; }}
.stMarkdown ol li::marker {{ color: {_CORAL}; font-weight: bold; }}
</style>
"""

def inject_rich_styles() -> None:
    """將 Anya Forger 主題 CSS 注入頁面。

    在每個頁面的 set_page_config() 之後呼叫即可。
    Streamlit 每次 rerun 都會重建 DOM，所以 CSS 每次都需要重新注入。

    效果（Anya Forger 主題）：
      - h1/h2 顯示珊瑚粉 #D97B72
      - h3/h4 顯示深褐色 #4A2F1A
      - blockquote 顯示金色左邊框 + 淡珊瑚背景
      - table 標頭為深褐底金色字（制服配色）+ 斑馬紋
      - 行內 `code` 顯示淡金底圓角
      - 清單項目符號為珊瑚粉
    """
    st.markdown(_RICH_CSS, unsafe_allow_html=True)


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


def copy_html_button(text: str, key: str = "copy") -> None:
    """一鍵複製帶格式 HTML：貼入 Word 保留標題 / 粗體 / 清單 / 表格。

    把 markdown 轉成含 inline style 的 HTML 放入剪貼簿（text/html MIME）。
    Word 貼上時讀 inline style → 標題大字、粗體、清單符號、表格格線皆保留。

    st.html() 直接插入頁面主 DOM（非 iframe），可存取 Clipboard API。
    """
    # 1. markdown → HTML（tables/extra extension 支援表格與巢狀格式）
    html = _md_lib.markdown(text, extensions=["tables", "extra"])

    # 2. 補 inline style — Word 只讀 inline style，不讀 CSS class
    html = (
        html
        .replace("<h1>", '<h1 style="color:#C05A50;font-size:20pt;font-weight:bold;margin-top:1em;">')
        .replace("<h2>", '<h2 style="color:#C05A50;font-size:16pt;font-weight:bold;margin-top:.8em;">')
        .replace("<h3>", '<h3 style="color:#7A4030;font-size:13pt;font-weight:bold;margin-top:.6em;">')
        .replace("<h4>", '<h4 style="color:#7A4030;font-size:12pt;font-weight:bold;">')
        .replace("<th>", '<th style="background:#4A2F1A;color:#C8A43A;padding:6px 10px;text-align:left;">')
        .replace("<td>", '<td style="padding:6px 10px;border-bottom:1px solid #F2D5CF;">')
        .replace("<table>", '<table style="border-collapse:collapse;width:100%;">')
        .replace("<blockquote>", '<blockquote style="border-left:4px solid #C8A43A;padding:.5em 1em;background:#FFF5F2;color:#4A2F1A;margin:.5em 0;">')
    )

    # 3. 用 st.html() 注入 JS 按鈕（在主 DOM，無 same-origin iframe 限制）
    escaped = html.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    st.html(f"""
    <button id="cpbtn-{key}"
        style="font-size:13px;padding:4px 14px;border:1px solid #C05A50;border-radius:6px;
               background:#fff;color:#C05A50;cursor:pointer;margin-top:8px;"
        onclick="
            navigator.clipboard.write([
                new ClipboardItem({{'text/html': new Blob([`{escaped}`], {{type:'text/html'}})}})
            ]).then(function() {{
                var b = document.getElementById('cpbtn-{key}');
                b.textContent = '✓ 已複製';
                setTimeout(function() {{ b.textContent = '複製'; }}, 2000);
            }}).catch(function(e) {{
                alert('複製失敗：' + e.message);
            }});
        ">複製</button>
    """)
