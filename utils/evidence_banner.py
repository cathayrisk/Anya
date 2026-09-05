# -*- coding: utf-8 -*-
"""由憑證帳本生成回覆下方的查證標示（純函式，無 streamlit 依賴）。

## 為什麼要重做

原本的 banner 條件是 `if not web_happened:`——只認得「有沒有跑網路搜尋」，於是：

1. **General 路徑根本沒有 banner。** 09-05 實測問「宏碁跟宏達電的曝險差異」，模型
   把宏達電整段寫成緯創，格式專業、語氣肯定、零標示。對金融使用者最危險的一種錯。
2. **第 4 步之後這個條件開始說假話。** controller prefetch 讓 Fast 也可能握有氣象署
   官方資料，banner 卻仍寫「內容來自模型既有知識」——**主動誤導，比沒有 banner 更糟**。

正解是讓 banner 讀**這一回合實際發生過什麼**，也就是 `utils/evidence.py` 的帳本。
帳本已經記了 scope／status／completed_at，這裡只負責把它翻成一句話。

## 措辭上的兩條紅線

- **網路搜尋成功 ≠ 已查證。** authority 分 official／open_web／internal 就是為了這件事；
  搜到東西只證明搜尋跑過，不保證內容對。文字上一律寫成「做過網路搜尋」而不是「已查證」。
- **empty ≠ 沒發生。** 「氣象署這次沒回傳地震」跟「近期沒有地震」是兩件事，
  混為一談就是拿資料缺漏冒充事實。

## 這個模組不做的事

不決定要不要顯示、不決定顯示在哪——呼叫端的政策。純函式好測。
"""
from __future__ import annotations

import datetime as _dt
from typing import Iterable

from utils import evidence as EV

_TPE = _dt.timezone(_dt.timedelta(hours=8))

# 官方 scope 的中文名（banner 要能說出「查了什麼」，不能只說「查了」）
_OFFICIAL_NAMES = {
    EV.SCOPE_EARTHQUAKE: "地震",
    EV.SCOPE_TYPHOON: "颱風",
    EV.SCOPE_WEATHER: "天氣",
}

NOTHING = ("本回覆未經任何查證，內容來自模型既有知識，可能不是最新；"
           "需要查證可以再問一次並要求搜尋。")
TAIL = "其餘內容仍來自模型既有知識。"


def _latest_time(events: Iterable[dict]) -> str:
    """帳本裡最後一筆的時間（台灣時間 HH:MM）。banner 要寫「已於 X 查詢」就得有它。"""
    stamps = [e.get("completed_at") for e in events or [] if e.get("completed_at")]
    if not stamps:
        return ""
    try:
        return _dt.datetime.fromisoformat(max(stamps)).astimezone(_TPE).strftime("%H:%M")
    except Exception:
        return ""


def build_banner(events: Iterable[dict]) -> str:
    """回傳要放進 `st.caption` 的那一行；"" 表示不顯示（目前不會發生——
    什麼都沒查也要說「未經任何查證」，那正是最需要標示的情況）。"""
    events = list(events or [])
    summary = EV.summarize(events)          # {scope: 最能代表現況的 status}
    if not summary:
        return "💡 " + NOTHING

    def scopes_with(status, pool):
        return [sc for sc in pool if summary.get(sc) == status]

    official = list(_OFFICIAL_NAMES)
    ok_off = scopes_with(EV.STATUS_OK, official)
    empty_off = scopes_with(EV.STATUS_EMPTY, official)
    err_off = scopes_with(EV.STATUS_ERROR, official)

    parts: list[str] = []
    hhmm = _latest_time(events)
    if ok_off:
        names = "、".join(_OFFICIAL_NAMES[s] for s in ok_off)
        parts.append((f"已於 {hhmm} " if hhmm else "已") + f"直接查詢中央氣象署（{names}）")
    if empty_off:
        names = "、".join(_OFFICIAL_NAMES[s] for s in empty_off)
        # ⚠️ 不可寫成「目前沒有地震」——那是拿資料缺漏冒充事實
        parts.append(f"向中央氣象署查詢{names}但沒有回傳資料（取不到，不等於沒發生）")
    if err_off:
        names = "、".join(_OFFICIAL_NAMES[s] for s in err_off)
        parts.append(f"氣象署{names}查詢失敗")

    # ⚠️ 網路搜尋成功只證明「搜尋跑過」，不等於已查證——措辭必須守住這條線
    if summary.get(EV.SCOPE_WEB) == EV.STATUS_OK:
        parts.append("做過網路搜尋（搜尋到資料不等於已查證）")
    elif summary.get(EV.SCOPE_WEB) == EV.STATUS_EMPTY:
        parts.append("做過網路搜尋，但沒有取得來源")
    elif summary.get(EV.SCOPE_WEB) == EV.STATUS_ERROR:
        parts.append("網路搜尋失敗")

    if summary.get(EV.SCOPE_PAGE) == EV.STATUS_OK:
        parts.append("擷取過網頁內容")
    elif summary.get(EV.SCOPE_PAGE) == EV.STATUS_ERROR:
        parts.append("網頁擷取失敗")

    if summary.get(EV.SCOPE_DOC) == EV.STATUS_OK:
        parts.append("檢索過你上傳的文件")
    elif summary.get(EV.SCOPE_DOC) == EV.STATUS_EMPTY:
        parts.append("檢索過你上傳的文件，但沒有命中")

    if not parts:
        return "💡 " + NOTHING

    # 圖示依「最需要使用者注意的事」決定：失敗 > 官方資料 > 其他檢索
    if err_off or EV.STATUS_ERROR in summary.values():
        icon = "⚠️"
    elif ok_off:
        icon = "🌐"
    else:
        icon = "🔎"
    return f"{icon} 本回合" + "、".join(parts) + "。" + TAIL
