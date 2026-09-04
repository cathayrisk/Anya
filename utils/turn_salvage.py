# -*- coding: utf-8 -*-
"""回合收尾的兩個補救（純函式，無 streamlit 依賴）。

2026-09-05 使用者截圖到的兩個 widget 相關問題：

**Bug A：工作全做完了，卻只回一句道歉。**
畫面顯示「抱歉，安妮亞這次沒有取得回應，請再試一次。」，但同一則訊息下方
互動 widget 正常渲染、任務清單三項全部打勾（market-research／widget／報告）。
也就是 tool loop 成功、產物都在，只有「最終那段文字」是空的。
原本 [Home.py] 的空回應分支只看 `not ai_text`，不看這回合到底做了什麼 →
使用者一邊看著可用的元件、一邊被告知「沒有取得回應」，還會重試一次白燒配額。

**Bug B：widget 原始碼被當成散文印出來。**
模型有時把 `widget_*` 模板的 `<script> const DATA = {...} </script>` 直接寫進
markdown，而不是呼叫 `create_widget`；後面還接一句「請點選上方的互動表格」，
但上方根本沒有元件。使用者看到的是一段 JS 原始碼加一個指向空氣的指示。
`create_widget` 的 docstring 早就寫了「必須…再呼叫本工具」——同 Fix C／LaTeX 的教訓，
**對弱模型的流程指令不可靠，要在後處理層兜底。**
"""
from __future__ import annotations

import re

# widget 模板的特徵：自包含的 <script> 區塊，且帶模板的資料變數
_WIDGET_SRC_RE = re.compile(
    r"(?:<!--[^\n]*?互動[^\n]*?-->\s*)?"
    r"```[a-zA-Z]*\s*\n?(?=(?:[^`]*?<script)|(?:[^`]*?const\s+DATA\s*=))[^`]*?```"
    r"|(?:<!--[^\n]*?互動[^\n]*?-->\s*)?<script\b[^>]*>.*?</script>",
    re.S,
)
# 指向不存在元件的句子（模型以為自己建了）
_DANGLING_RE = re.compile(
    r"[（(]?\s*(?:請)?點[選擊][^。\n）)]{0,20}(?:上方|上面)[^。\n）)]{0,20}"
    r"(?:互動|元件|表格|矩陣)[^。\n）)]{0,30}[）)]?[。\n]?"
    r"|上方(?:是|為)[^。\n]{0,20}互動[^。\n]{0,20}[。\n]?"
)


def strip_orphan_widget_source(text: str, widget_created: bool) -> tuple[str, bool]:
    """widget 沒建成卻把模板原始碼寫進答案時，把原始碼與「請點上方元件」的指示拿掉。

    widget_created=True（這回合真的呼叫過 create_widget）時原樣返回——
    那種情況下「上方是互動比較矩陣」是正確的敘述，不可誤刪。

    回傳 (清理後文字, 是否有動過)。
    """
    if not text or widget_created:
        return text or "", False
    if not _WIDGET_SRC_RE.search(text):
        return text, False
    out = _WIDGET_SRC_RE.sub("", text)
    out = _DANGLING_RE.sub("", out)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out, True


def describe_completed_work(*, widget_title: str | None = None,
                            todos: list | None = None,
                            has_report: bool = False,
                            n_web: int = 0, n_doc: int = 0) -> str:
    """最終文字是空的，但這回合其實有產物時，講清楚做了什麼——取代乾巴巴的道歉。

    沒有任何產物就回空字串，呼叫端沿用原本的道歉文案。
    """
    done = [t for t in (todos or [])
            if isinstance(t, dict) and str(t.get("status")) in ("completed", "done")]
    bits = []
    if widget_title:
        bits.append(f"互動元件「{widget_title}」（已顯示在上方）")
    if has_report:
        bits.append("結構化報告")
    if n_doc:
        bits.append(f"{n_doc} 次文件檢索")
    if n_web:
        bits.append(f"{n_web} 次網路搜尋")
    if not bits and not done:
        return ""
    lines = ["安妮亞這回合的**文字總結沒有生成出來**，但前面的工作有完成："]
    for b in bits:
        lines.append(f"- ✅ {b}")
    for t in done[:6]:
        lines.append(f"- ✅ {str(t.get('content') or '')[:60]}")
    lines.append("")
    lines.append("要安妮亞把上面的結果寫成文字總結嗎？（直接說「幫我總結」就可以，不用重問一次）")
    return "\n".join(lines)
