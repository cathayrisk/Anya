# -*- coding: utf-8 -*-
"""把模型吐出的 LaTeX 數學語法轉成純文字（純函式，無 streamlit 依賴）。

為什麼需要（2026-09-05 使用者實測截圖）：
1. **紅色原始碼外洩**：畫面出現
   `\\text{下週數據 \\rightarrow 溫和} \\implies \\text{Waller 獲勝} \\implies ...`
   KaTeX 解析失敗時的行為就是「把原始碼用紅色印出來」。`\\text{}` 裡包中文、
   再加上 `\\implies`，是典型會讓 KaTeX 炸掉的組合。
2. **黑色方塊（tofu）**：`現在升息 25bp ▮ CPI 立即降到 2%`——語意上該是「⇏／≠」。
   normalize_markdown_for_streamlit 的 NFC 只修「分解形式」（= + U+0338），
   修不了本來就是精撰碼位、但字型沒有該字的情況（⇏ U+21CF 之類）。

Home.py:3596 的系統提示早就寫了「數學公式：不用 LaTeX，用 inline code 包起」，
模型照樣用——同 Fix C 的教訓：**對弱模型下的格式禁令不可靠，要在後處理層兜底。**

刻意不做的事：不嘗試渲染真正的數學。這個 app 的定位是金融／VBA 助理，
出現的 LaTeX 幾乎都是「用數學符號寫散文」（A ⇒ B ⇒ C），轉成 Unicode 就夠讀。
"""
from __future__ import annotations

import re

# LaTeX 指令 → Unicode。只收「散文裡會用到」的符號；真正的數學排版不在守備範圍。
_CMD_MAP = {
    r"\implies": "⇒", r"\Rightarrow": "⇒", r"\rightarrow": "→", r"\to": "→",
    r"\impliedby": "⇐", r"\Leftarrow": "⇐", r"\leftarrow": "←",
    r"\Leftrightarrow": "⇔", r"\leftrightarrow": "↔", r"\iff": "⇔",
    # 否定形式一定要收：截圖裡的黑框就是 \nRightarrow（⇏）。沒收錄會被當成無效指令刪掉，
    # 留下一個空白，讀者看到的是「升息 25bp　CPI 降到 2%」——語意整個反過來，比黑框更糟。
    r"\nRightarrow": "≠", r"\nrightarrow": "≠", r"\nLeftarrow": "≠",
    r"\nleftrightarrow": "≠", r"\nLeftrightarrow": "≠", r"\not": "≠",
    r"\geq": "≥", r"\ge": "≥", r"\leq": "≤", r"\le": "≤",
    r"\neq": "≠", r"\ne": "≠", r"\approx": "≈", r"\sim": "～",
    r"\times": "×", r"\div": "÷", r"\pm": "±", r"\cdot": "・",
    r"\alpha": "α", r"\beta": "β", r"\Delta": "Δ", r"\delta": "δ",
    r"\infty": "∞", r"\%": "%", r"\$": "$", r"\&": "&",
    r"\ldots": "…", r"\dots": "…", r"\quad": " ", r"\qquad": "  ",
    r"\left": "", r"\right": "", r"\,": " ", r"\;": " ", r"\!": "",
}

# 字型常缺 → 會變黑框。換成看得懂又一定有字的寫法。
_TOFU_MAP = {
    "⇏": "≠", "⇎": "≠", "↛": "≠", "↮": "≠", "⊭": "≠", "≢": "≠",
    "⟹": "⇒", "⟸": "⇐", "⟺": "⇔", "⟶": "→", "⟵": "←",
}

_TEXT_WRAP_RE = re.compile(r"\\(?:text|mathrm|mathbf|textbf|mathit|textit|operatorname)\s*\{([^{}]*)\}")
_MATH_SPAN_RE = re.compile(
    r"\$\$(?P<dd>.+?)\$\$"          # $$...$$
    r"|\$(?P<d>[^$\n]+?)\$"         # $...$（不跨行，避免把金額 $100 ... $200 誤配）
    r"|\\\((?P<p>.+?)\\\)"          # \(...\)
    r"|\\\[(?P<b>.+?)\\\]",         # \[...\]
    re.S,
)
_LEFTOVER_CMD_RE = re.compile(r"\\[A-Za-z]+")
_CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff\u3040-\u30ff]")


def _convert_commands(s: str) -> str:
    for _ in range(3):                       # \text{\mathbf{x}} 這種巢狀最多拆三層
        new = _TEXT_WRAP_RE.sub(r"\1", s)
        if new == s:
            break
        s = new
    # 長指令先換，避免 \le 先吃掉 \leq 的前綴
    for cmd in sorted(_CMD_MAP, key=len, reverse=True):
        s = s.replace(cmd, _CMD_MAP[cmd])
    s = s.replace("{", "").replace("}", "")
    return re.sub(r"[ \t]{2,}", " ", s).strip()


def _fix_tofu(s: str) -> str:
    for a, b in _TOFU_MAP.items():
        s = s.replace(a, b)
    return s


def latex_to_plain(text: str) -> str:
    """把散文裡的 LaTeX 轉成純文字。程式碼區塊（``` 與 `）原封不動。

    規則：
    - `$...$` / `$$...$$` / `\\(...\\)` / `\\[...\\]` 一律拆掉外框，內容轉 Unicode。
    - 框外裸露的 `\\rightarrow`、`\\text{...}` 也一併轉（模型常忘了加框，KaTeX 就印紅字）。
    - 字型會缺字的符號（⇏ 等）換成一定有字的等義符號。
    """
    if not text:
        return ""
    stash: list[str] = []

    def _keep(m: re.Match) -> str:
        stash.append(m.group(0))
        return "\x00%d\x00" % (len(stash) - 1)

    t = re.sub(r"```.*?```", _keep, text, flags=re.S)
    t = re.sub(r"`[^`\n]*`", _keep, t)

    def _span(m: re.Match) -> str:
        inner = m.group("dd") or m.group("d") or m.group("p") or m.group("b") or ""
        # 純數字/貨幣的 $...$ 不是數學（例：$100），原樣保留
        if m.group("d") is not None and not re.search(r"\\|[⇒→≥≤≠]", inner):
            return m.group(0)
        return _convert_commands(inner)

    t = _MATH_SPAN_RE.sub(_span, t)
    t = _convert_commands(t)          # 收框外裸露的指令
    t = _fix_tofu(t)
    t = _LEFTOVER_CMD_RE.sub("", t)   # 沒收錄的指令直接拿掉，總比印紅字好
    t = re.sub(r"\n{3,}", "\n\n", t)
    return re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], t)
