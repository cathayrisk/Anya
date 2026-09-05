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

**Bug B2（2026-09-05 線上測試補上）：連原始碼都沒有，只有一句口頭宣稱。**
請它做 VBA 抽認卡，它載入了模板、卻**完全沒有呼叫 `create_widget`**，直接寫散文並附上
「（註：如果 widget 沒有顯示，請確認您的瀏覽器支援 iframe。）」——上方沒有任何元件。
原本的 `strip_orphan_widget_source` 接不住，因為它開頭就 `if not _WIDGET_SRC_RE.search(text)`
提早返回：**沒有原始碼可剝，指向空氣的句子那段就永遠跑不到**。
兩種樣態的共通點是「這回合沒有 widget」，所以判斷條件應該只看 `widget_created`。
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
# 指向不存在元件的句子（模型以為自己建了）。
# ⚠️ 只在 widget_created=False 時套用，所以「下方」也收得安全——真的有元件時
# 「下方的抽認卡」是正確敘述，那條路徑根本不會走到這裡。
_POS = r"(?:上方|上面|下方|下面|以下|底下)"
_DANGLING_RE = re.compile(
    r"[（(]?\s*(?:請)?點[選擊][^。\n）)]{0,20}" + _POS + r"[^。\n）)]{0,20}"
    r"(?:互動|元件|表格|矩陣|卡片|閃卡|抽認卡)[^。\n）)]{0,30}[）)]?[。\n]?"
    + r"|" + _POS + r"(?:是|為)[^。\n]{0,20}互動[^。\n]{0,20}[。\n]?"
    # Bug B2 的實際長相：「（註：如果 widget 沒有顯示，請確認您的瀏覽器支援 iframe。）」
    # 這句預設「元件應該在那裡」，是最誤導的一種——它把「沒做出來」講成「你的瀏覽器有問題」。
    + r"|[（(]?\s*(?:註\s*[：:]\s*)?如果[^。\n]{0,12}(?:widget|元件|互動元件)[^。\n]{0,12}"
      r"(?:沒有顯示|沒顯示|未顯示|看不到|無法顯示)[^。\n]{0,50}[。]?\s*[）)]?",
    re.IGNORECASE,
)


def strip_orphan_widget_source(text: str, widget_created: bool) -> tuple[str, bool]:
    """這回合沒有 widget，卻寫了模板原始碼或指向元件的句子時，把它們拿掉。

    `widget_created=True`（真的呼叫過 create_widget）時原樣返回——
    那種情況下「上方是互動比較矩陣」是正確敘述，不可誤刪。**唯一的判斷依據就是它**：
    系統知道這回合到底有沒有建成元件，不需要（也不該）去猜模型的措辭是真是假。

    ⚠️ 原本這裡有一行 `if not _WIDGET_SRC_RE.search(text): return` 提早返回，
    於是「沒有原始碼、只有一句口頭宣稱」（Bug B2）永遠接不住。兩種樣態要各自獨立判斷。

    回傳 (清理後文字, 是否有動過)。
    """
    if not text or widget_created:
        return text or "", False
    out, changed = text, False
    if _WIDGET_SRC_RE.search(out):
        out = _WIDGET_SRC_RE.sub("", out)
        changed = True
    if _DANGLING_RE.search(out):
        out = _DANGLING_RE.sub("", out)
        changed = True
    if not changed:
        return text, False
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
