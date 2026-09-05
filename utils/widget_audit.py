# -*- coding: utf-8 -*-
"""判斷一段 widget HTML 會不會「靜默流失操作狀態」（純函式，無 streamlit 依賴）。

## 起因

2026-09-05 測試 T20：請它把 VBA 觀念做成閃卡，翻到第 3 張後送下一則訊息，
widget 重置回第 1 張。追查發現——

- `AnyaState` 有正常注入（key 已建、手動 `save`/`load` 都正常）
- 但那個 iframe 的 srcdoc **完全沒有 `HAS_STATE`**

也就是模型**自己手寫了 HTML**，沒有用 `widget_templates.py` 的 `widget_flashcards`。
`create_widget` 接受任意 HTML，自製的那些沒有任何呼叫 `AnyaState` 的程式碼，
於是狀態就這樣不見了，而且**沒有任何訊號**——使用者只看到「我剛剛翻到第 3 張，
怎麼又回到第 1 張」。

## 為什麼不能靠 prompt

`WIDGET_RULES` 早就寫了「必須先 load_skill 載入模板」，模型照樣自己寫。
這是本專案第六次遇到同一件事：硬性 invariant 交給機率模型執行不會成立。

## 也不能自動補救

想過在注入的 helper 裡做通用 DOM 狀態存取，但**對這個 case 沒用**：
flashcards 的狀態（`pos`／`pool`）活在 JS closure 裡，不在任何表單控制項上。
就算把 `innerHTML` 存下來還原，closure 仍停在第 1 張，下次點擊會跳回第 2 張——
畫面與內部狀態不一致，比重置更糟。

## 所以做法是：偵測 → 擋一次 → 放行

`create_widget` 在**渲染之前**檢查；判定會流失狀態時回一個帶指引的錯誤，
模型可以在同一回合改用模板重來（此時 widget 名額還沒被佔用）。
**同一回合只擋一次**：第二次一律放行。理由是判定必然是啟發式的，
擋兩次以上的風險（無限迴圈、使用者完全拿不到元件）比「狀態流失」這個
不便本身更大——這不是安全性問題，處理力道要相稱。

## 模板怎麼認

用**根 id**（`#anya-fc`、`#anya-cmx`…）而不是加註解標記：
根 id 是 CSS 與 `getElementById` 都在用的**結構性**識別，模型抄模板時拿不掉；
註解則可能在改寫時被順手刪掉。等於用「拔掉就會壞」的東西當指紋。

`widget_calculator` 與 `widget_natal_chart` **刻意沒有接狀態**（值是當下輸入，
還原沒有意義），所以只要認出是模板就直接放行，不看有沒有 `AnyaState`。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# 根 id → 模板名。這些 id 同時被 CSS 與 JS 使用，拔掉模板就壞了。
TEMPLATE_ROOT_IDS = {
    "anya-nat": "widget_natal_chart",
    "anya-cmx": "widget_comparison_matrix",
    "anya-calc": "widget_calculator",
    "anya-src": "widget_source_browser",
    "anya-fc": "widget_flashcards",
}

# 有互動才談得上「狀態」；純展示的圖表流失不了東西
_INTERACTIVE_RE = re.compile(
    r"addEventListener\s*\(\s*['\"](?:click|change|input|keydown|keyup|submit)"
    r"|\bon(?:click|change|input|keydown)\s*="
    r"|<input\b|<select\b|<textarea\b|<button\b",
    re.IGNORECASE,
)

RETRY_HINT = (
    "這個元件有互動但沒有接狀態保存，使用者操作後只要畫面重新整理就會全部重置"
    "（實測會發生：翻到第 3 張卡片，送出下一則訊息後跳回第 1 張）。"
    "請改用 load_skill 載入對應的 widget_* 模板，只替換資料區的 DATA、"
    "不要自己重寫 HTML/JS，然後重新呼叫 create_widget。"
    "若這個需求真的沒有現成模板可用，就再呼叫一次本工具（同一回合只會擋這一次）。"
)


@dataclass(frozen=True)
class WidgetAudit:
    template: str | None      # 認出來的模板名；None＝模型自製
    interactive: bool
    has_state: bool           # HTML 自己有沒有呼叫 AnyaState
    will_lose_state: bool     # 自製 ＋ 有互動 ＋ 沒接狀態

    def as_dict(self) -> dict:
        return {"template": self.template, "interactive": self.interactive,
                "has_state": self.has_state, "will_lose_state": self.will_lose_state}


def detect_template(html: str) -> str | None:
    """`id="anya-fc"` 或 CSS 的 `#anya-fc` 都算。

    要加尾端邊界：裸的子字串比對會讓 `#anya-fcx`／`#anya-calculator` 這類
    自製名稱誤中模板，於是真正該擋的元件被放行。
    """
    h = html or ""
    for root_id, name in TEMPLATE_ROOT_IDS.items():
        if re.search(r'id=["\']' + re.escape(root_id) + r'["\']', h):
            return name
        if re.search(r"#" + re.escape(root_id) + r"(?![\w-])", h):
            return name
    return None


def audit_widget_html(html: str) -> WidgetAudit:
    h = html or ""
    template = detect_template(h)
    interactive = bool(_INTERACTIVE_RE.search(h))
    has_state = "AnyaState" in h
    # 認出模板就放行：wired 的三個本來就有；calculator／natal_chart 是刻意不接的。
    will_lose = (template is None) and interactive and not has_state
    return WidgetAudit(template=template, interactive=interactive,
                       has_state=has_state, will_lose_state=will_lose)
