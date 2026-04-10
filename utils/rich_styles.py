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
/* ════════════════════════════════════════════════════════
   Anya Forger Markdown Theme — 完整元素定義
   參考：GitHub Markdown CSS (sindresorhus.com/github-markdown-css)
   色調：珊瑚粉 / 金邊黃 / 深褐色（Spy x Family Anya 配色）
   ════════════════════════════════════════════════════════ */

/* ── 全域：字型 + 文字換行 ── */
.stMarkdown {{
    word-break: break-word;
    overflow-wrap: break-word;
}}
.stMarkdown, .stMarkdown *:not(code):not(pre):not(kbd) {{
    font-family:
        'SF Pro Rounded',        /* 本地靜態字型（config.toml fontFaces 載入）*/
        'PingFang TC',           /* macOS/iOS 繁中系統字型（SF Pro 無 CJK，此為 fallback）*/
        'Microsoft JhengHei',    /* Windows 繁中 */
        'Noto Sans CJK TC',      /* Linux 繁中 */
        'WenQuanYi Micro Hei',   /* Linux 中文備用 */
        sans-serif;
}}
/* 程式碼專用等寬字型（系統 monospace，與 config.toml codeFont = "monospace" 一致）*/
.stMarkdown code,
.stMarkdown pre,
.stMarkdown kbd {{
    font-family:
        'Cascadia Code', 'Consolas', 'Monaco', 'Menlo',
        'DejaVu Sans Mono', monospace;
}}

/* ════════════════════
   標題 h1–h6
   ════════════════════ */
.stMarkdown h1 {{
    font-size: 1.9em;
    font-weight: 700;
    color: {_CORAL};
    border-bottom: 2px solid {_BORDER};
    padding-bottom: .25em;
    margin-top: 1em;
    margin-bottom: 0.4em;
}}
.stMarkdown h2 {{
    font-size: 1.5em;
    font-weight: 700;
    color: {_CORAL};
    border-bottom: 1px solid {_BORDER};
    padding-bottom: .2em;
    margin-top: 0.9em;
    margin-bottom: 0.3em;
}}
.stMarkdown h3 {{
    font-size: 1.22em;
    font-weight: 600;
    color: #7A4030;
    margin-top: .8em;
    margin-bottom: 0.2em;
}}
.stMarkdown h4 {{
    font-size: 1.05em;
    font-weight: 600;
    color: #7A4030;
    margin-top: .7em;
    margin-bottom: 0.15em;
}}
.stMarkdown h5 {{
    font-size: 0.93em;
    font-weight: 600;
    color: #9A5040;
    margin-top: .6em;
    margin-bottom: 0.1em;
}}
.stMarkdown h6 {{
    font-size: 0.85em;
    font-weight: 600;
    color: #B07060;   /* 最淺色，表層級最低 */
    margin-top: .5em;
    margin-bottom: 0.1em;
}}
/* 容器內第一個標題不加上方空白 */
.stMarkdown h1:first-child,
.stMarkdown h2:first-child,
.stMarkdown h3:first-child,
.stMarkdown h4:first-child,
.stMarkdown h5:first-child,
.stMarkdown h6:first-child {{
    margin-top: 0.2em;
}}

/* ════════════════════
   段落 / 換行
   ════════════════════ */
.stMarkdown p {{
    margin-top: 0.3em;
    margin-bottom: 0.6em;
    line-height: 1.75;
}}

/* ════════════════════
   文字強調
   ════════════════════ */
.stMarkdown strong {{ font-weight: 700; }}
.stMarkdown em     {{ font-style: italic; }}
.stMarkdown strong em,
.stMarkdown em strong {{ font-weight: 700; font-style: italic; }}

/* 刪除線 — 灰色，暗示「已廢棄」 */
.stMarkdown del {{
    color: #999;
    text-decoration: line-through;
}}

/* ════════════════════
   連結
   ════════════════════ */
.stMarkdown a {{
    color: {_CORAL};
    text-decoration: none;
    border-bottom: 1px solid transparent;
    transition: border-color 0.15s;
}}
.stMarkdown a:hover {{
    border-bottom-color: {_CORAL};
}}

/* ════════════════════
   水平分隔線
   ════════════════════ */
.stMarkdown hr {{
    border: none;
    border-top: 2px solid {_BORDER};
    margin: 1.6em 0;
    opacity: 0.8;
}}

/* ════════════════════
   圖片
   ════════════════════ */
.stMarkdown img {{
    max-width: 100%;
    height: auto;
    border-radius: 6px;
    display: block;
    margin: 0.6em auto;
}}

/* ════════════════════
   有序 / 無序清單
   ════════════════════ */
.stMarkdown ol,
.stMarkdown ul {{
    padding-left: 1.6em;
    line-height: 1.75;
    margin-top: 0.2em;
    margin-bottom: 0.6em;
}}
.stMarkdown ol li,
.stMarkdown ul li {{
    margin-bottom: 0.35em;
}}
/* 巢狀清單 */
.stMarkdown ol li ol, .stMarkdown ol li ul,
.stMarkdown ul li ul, .stMarkdown ul li ol {{
    margin-top: 0.25em;
    margin-bottom: 0;
    padding-left: 1.4em;
}}
/* li 內部 p 不加額外間距 */
.stMarkdown li > p {{
    margin: 0;
    padding: 0;
}}
/* 清單符號顏色 */
.stMarkdown ul li::marker {{ color: {_CORAL}; }}
.stMarkdown ol li::marker {{ color: {_CORAL}; font-weight: 700; }}

/* Task list（GFM checkbox）*/
.stMarkdown .task-list-item {{
    list-style: none;
    padding-left: 0;
}}
.stMarkdown .task-list-item input[type="checkbox"] {{
    margin-right: 0.5em;
    margin-left: -1.6em;
    accent-color: {_CORAL};
    vertical-align: middle;
}}

/* ════════════════════
   Blockquote
   ════════════════════ */
.stMarkdown blockquote {{
    border-left: 4px solid {_GOLD};
    background: {_LIGHT};
    padding: .6em 1.2em;
    border-radius: 0 8px 8px 0;
    margin: .6em 0 .3em 0;
    color: {_BROWN};
}}
.stMarkdown blockquote p {{
    margin: 0;
    padding: 0;
    line-height: 1.7;
}}
/* 巢狀 blockquote — 金色變淡，區分層級 */
.stMarkdown blockquote blockquote {{
    border-left-color: #E0C87A;
    background: #FFFBF0;
    margin: .4em 0;
}}

/* ════════════════════
   表格
   ════════════════════ */
/* 外層可橫向捲動（防止窄螢幕溢出）*/
.stMarkdown table {{
    display: block;
    overflow-x: auto;
    border-collapse: collapse;
    width: max-content;
    max-width: 100%;
    border: 1px solid {_BORDER};
    border-radius: 8px;
    margin-bottom: 0.8em;
}}
.stMarkdown thead tr {{
    background: {_BROWN};
}}
.stMarkdown th {{
    color: {_GOLD};
    padding: 8px 14px;
    text-align: left;
    font-weight: 600;
    letter-spacing: .03em;
    white-space: nowrap;
}}
.stMarkdown td {{
    padding: 8px 14px;
    border-top: 1px solid {_BORDER};
    vertical-align: top;
}}
.stMarkdown tr:nth-child(even) td {{
    background: {_STRIPE};
}}
.stMarkdown tr:hover td {{
    background: #FAEAE7;   /* hover 淡珊瑚 */
}}

/* ════════════════════
   行內程式碼
   ════════════════════ */
.stMarkdown code:not(pre > code) {{
    background: {_CODE_BG};
    padding: 2px 6px;
    border-radius: 4px;
    font-size: .87em;
    color: {_BROWN};
    border: 1px solid {_BORDER};
}}

/* ════════════════════
   程式碼區塊
   ════════════════════ */
.stMarkdown pre {{
    background: #F5F1EE !important;
    border: 1px solid {_BORDER} !important;
    border-radius: 8px;
    padding: 1em 1.4em;
    line-height: 1.65;
    overflow-x: auto;
    margin: 0.6em 0;
}}
.stMarkdown pre > code {{
    background: transparent !important;
    font-size: 0.92em;
    color: #2E1F18;
    padding: 0;
    border: none;
    border-radius: 0;
}}

/* ════════════════════
   鍵盤按鍵 <kbd>
   ════════════════════ */
.stMarkdown kbd {{
    background: #F0EBE8;
    border: 1px solid {_BORDER};
    border-bottom-width: 3px;       /* 立體感 */
    border-radius: 4px;
    padding: 2px 7px;
    font-size: 0.83em;
    color: {_BROWN};
    white-space: nowrap;
}}

/* ════════════════════
   <details> / <summary>（摺疊區塊）
   ════════════════════ */
.stMarkdown details {{
    background: {_LIGHT};
    border: 1px solid {_BORDER};
    border-radius: 8px;
    padding: 0.5em 1em;
    margin: 0.7em 0;
}}
.stMarkdown summary {{
    cursor: pointer;
    font-weight: 600;
    color: {_CORAL};
    user-select: none;
    outline: none;
    list-style: none;
}}
.stMarkdown summary::before {{
    content: '▶ ';
    font-size: 0.75em;
    transition: transform 0.2s;
}}
.stMarkdown details[open] summary::before {{
    content: '▼ ';
}}
.stMarkdown details[open] summary {{
    margin-bottom: 0.5em;
    border-bottom: 1px solid {_BORDER};
    padding-bottom: 0.3em;
}}

/* ════════════════════
   定義清單 <dl>
   ════════════════════ */
.stMarkdown dl {{
    margin: 0.6em 0;
}}
.stMarkdown dt {{
    font-weight: 700;
    color: {_CORAL};
    margin-top: 0.6em;
}}
.stMarkdown dd {{
    margin-left: 1.6em;
    color: {_BROWN};
    margin-bottom: 0.3em;
}}

/* ════════════════════
   腳注 footnote
   ════════════════════ */
.stMarkdown .footnotes {{
    border-top: 1px solid {_BORDER};
    margin-top: 1.5em;
    padding-top: 0.8em;
    font-size: 0.88em;
    color: #666;
}}
.stMarkdown .footnote-ref a,
.stMarkdown .footnotes a {{
    color: {_CORAL};
    font-size: 0.85em;
    vertical-align: super;
}}
</style>
"""

def inject_rich_styles() -> None:
    """將 Anya Forger 主題 CSS 注入頁面。

    在每個頁面的 set_page_config() 之後呼叫即可。
    Streamlit 每次 rerun 都會重建 DOM，所以 CSS 每次都需要重新注入。

    覆蓋元素（Anya Forger 主題，參考 GitHub Markdown CSS）：
      - h1–h6：珊瑚粉漸層，h1/h2 有底線
      - p / strong / em / del / a / hr / img
      - ul / ol（含巢狀）/ task list / dl / 腳注
      - blockquote（含巢狀） / details+summary
      - table（橫向捲動 + hover）/ th / td / 斑馬紋
      - inline code / pre code block / kbd
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
        .replace("<h1>", '<h1 style="color:#C05A50;font-size:20pt;font-weight:bold;margin-top:1em;border-bottom:2px solid #F2D5CF;padding-bottom:.2em;">')
        .replace("<h2>", '<h2 style="color:#C05A50;font-size:16pt;font-weight:bold;margin-top:.8em;border-bottom:1px solid #F2D5CF;padding-bottom:.15em;">')
        .replace("<h3>", '<h3 style="color:#7A4030;font-size:13pt;font-weight:bold;margin-top:.6em;">')
        .replace("<h4>", '<h4 style="color:#7A4030;font-size:12pt;font-weight:bold;">')
        .replace("<h5>", '<h5 style="color:#9A5040;font-size:11pt;font-weight:bold;">')
        .replace("<h6>", '<h6 style="color:#B07060;font-size:10pt;font-weight:bold;">')
        .replace("<th>", '<th style="background:#4A2F1A;color:#C8A43A;padding:6px 10px;text-align:left;font-weight:600;">')
        .replace("<td>", '<td style="padding:6px 10px;border-top:1px solid #F2D5CF;vertical-align:top;">')
        .replace("<table>", '<table style="border-collapse:collapse;width:100%;border:1px solid #F2D5CF;border-radius:8px;">')
        .replace("<blockquote>", '<blockquote style="border-left:4px solid #C8A43A;padding:.5em 1em;background:#FFF5F2;color:#4A2F1A;margin:.5em 0;border-radius:0 6px 6px 0;">')
        .replace("<hr>", '<hr style="border:none;border-top:2px solid #F2D5CF;margin:1.2em 0;">')
        .replace("<del>", '<del style="color:#999;">')
        .replace("<kbd>", '<kbd style="background:#F0EBE8;border:1px solid #F2D5CF;border-radius:4px;padding:1px 5px;font-family:monospace;font-size:.85em;">')
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
