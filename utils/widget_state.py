# -*- coding: utf-8 -*-
"""互動 widget 的狀態保存（純函式，無 streamlit 依賴）。

## 問題
[Home.py] 的歷史回放註解自承「元件內操作狀態不跨 rerun 保留」——使用者在比較矩陣上
勾了維度、點了優勝方，只要送下一則訊息，整段歷史重播、iframe 重建，操作全部消失。
`components.html` 是**單向**的（app 讀不到 iframe 內部），所以狀態只能存在 iframe 那一側。

## 為什麼是 sessionStorage（2026-09-05 production 實機 probe）
iframe 是 `srcdoc` + sandbox 含 `allow-same-origin`：
  - `location.origin` 回 `"null"`（srcdoc 的不透明來源）**但 storage 可用**，
    且與父層 app **共用同一分區**——別被 origin=null 誤導成不能用。
  - 實測：送新訊息（iframe 重建）後狀態 SURVIVED、新 iframe 讀得到；重新整理也 SURVIVED。
  - `localStorage` 已有 9 個 **Streamlit 自己的鍵**（`appSessionId-*`、`machineId`、
    `stActiveTheme-*`、`stSidebarCollapsed-*` 等），`sessionStorage` 則是空的。
    → 用 sessionStorage，避免模板寫錯鍵名蓋掉 Streamlit 的狀態。

## 設計約束（採納 OAI adversarial review）
1. **不可變 UUID**：每個 widget occurrence 建立時產生，與訊息一起持久化。
   不可用 title 或內容 hash 當 id——同標題不同資料會互相污染。
2. **contentFingerprint**：id 穩定不代表**內容**穩定。模板改版、資料更新、歷史被改過，
   舊 state 套到新內容會產生錯誤選取或越界索引——**畫面看似還原成功，呈現的卻是錯的**。
   還原前比對 fingerprint，不符即丟棄。
3. **只存 UI 檢視狀態**（顯示哪些維度、哪格被標記），**絕不存使用者輸入的數值**。
   storage 與其他 widget 及 app 共用，而 widget HTML 是模型產生的。
   （`create_widget` 已擋 `<script src>`/fetch/XHR，資料出不了網路，風險是本地污染。）
4. 大小上限；資料損壞就刪除，不要靜默留著。
"""
from __future__ import annotations

import hashlib
import json

SCHEMA_VERSION = 1
STATE_KEY_PREFIX = "anya:w:"
MAX_STATE_CHARS = 4096          # envelope 序列化後的上限


def content_fingerprint(html: str) -> str:
    """模型產生的 HTML 的指紋。內容一變（換資料、換模板）指紋就變，舊 state 自動作廢。"""
    return hashlib.sha1((html or "").encode("utf-8", "replace")).hexdigest()[:12]


def build_state_script(wid: str, fingerprint: str) -> str:
    """注入到 widget HTML 尾端的共用小工具，提供 `AnyaState.load()` / `AnyaState.save(obj)`。

    刻意全部 try/catch 包起來——無痕模式、storage 被停用、配額滿都可能丟例外，
    但**失敗只能降級成「沒有還原」，絕不能讓 widget 本身壞掉**。
    """
    key = json.dumps(STATE_KEY_PREFIX + wid)
    fp = json.dumps(fingerprint)
    return (
        "<script>(function(){"
        f"var K={key},FP={fp},V={SCHEMA_VERSION},MAX={MAX_STATE_CHARS};"
        "function S(){try{return window.sessionStorage;}catch(e){return null;}}"
        "function drop(){try{var s=S();if(s)s.removeItem(K);}catch(e){}}"
        "window.AnyaState={"
        "load:function(){var s=S();if(!s)return null;try{"
        "var raw=s.getItem(K);if(!raw)return null;var env=JSON.parse(raw);"
        # 版本或內容指紋不符 → 丟棄（避免舊 state 套到新資料上）
        "if(!env||env.v!==V||env.fp!==FP){drop();return null;}"
        "return env.state;}catch(e){drop();return null;}},"
        "save:function(st){var s=S();if(!s)return false;try{"
        "var raw=JSON.stringify({v:V,fp:FP,state:st});"
        "if(raw.length>MAX)return false;"
        "s.setItem(K,raw);return true;}catch(e){return false;}}"
        "};})();</script>"
    )


def inject_state_helper(html: str, wid: str | None, fingerprint: str | None) -> str:
    """把小工具插在模板 JS **之前**——模板載入時就要能呼叫 `AnyaState.load()`。

    wid 為空（舊歷史還沒補號）時原樣返回：沒有 id 就沒有穩定的 key，
    硬給一個會讓不同 widget 互相污染，寧可不還原。
    """
    if not html or not wid or not fingerprint:
        return html or ""
    return build_state_script(wid, fingerprint) + html
