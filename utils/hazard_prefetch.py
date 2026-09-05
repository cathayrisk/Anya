# -*- coding: utf-8 -*-
"""災防資料的 controller prefetch：把「要不要查」從模型手上收回來（純函式，無 streamlit 依賴）。

## 為什麼

一小時測試 T5：問「最近台灣有地震嗎？有沒有颱風要來？」→ 路由正確升級 General、
工具也確實掛著，但**模型就是沒呼叫**，憑記憶編出半年前的地震並宣稱查過氣象署。
2026-09-05 線上複驗時同一句話卻正常呼叫了兩支工具——所以這是**間歇性**的。

間歇正是最麻煩的地方：靠 prompt 叮嚀、靠測一次通過，都不構成保證。這就是這個
專案反覆踩到的「gating ≠ retrieval」——把硬性 invariant 交給機率模型執行。

修法是把檢索移出模型的決策範圍：**分類器判定為即時災防題時，程式在第一次 LLM
呼叫之前就直接打 CWA，把結果放進 context**。之後模型呼不呼叫工具都無所謂，
資料已經在了，憑證帳本也一定留得下記錄。

## 這一步保證什麼、不保證什麼

保證（程式層，與模型行為無關）：
  1. 官方資料一定被查詢
  2. 查詢結果一定進入本回合 context
  3. 憑證帳本一定有對應事件（ok／empty／error 都記）

**不保證**模型會正確使用這些資料——那需要程式渲染（第 5 步）。這裡塞進 block 的
那幾條「規則：…」是**軟偏好**，寫下來能提高命中率，但絕不是閘門；本專案已經有
五次紀錄證明 prompt 硬規則對弱模型不可靠。不要把它當保證看。

## 為什麼只 prefetch 地震與颱風

`get_earthquake_info` / `get_typhoon_info` **免參數**，程式可以無歧義地代呼叫。
`get_weather` 需要地點：使用者沒指名時 prefetch 只能餵預設地點，等於用台北的天氣
回答高雄的問題——那是換一種方式編造，比不查更難察覺。天氣題目前由 Fast 的
sentinel 自行升級後呼叫工具，實測會拿到真實 CWA 資料，不在這一步動它。

海嘯／土石流／淹水沒有任何工具可查（見 `hazard_intent.UNCOVERED_ENTITIES`），
這裡也不處理——「沒有資料源」要靠第 5、6 步的渲染與 banner 講清楚。
"""
from __future__ import annotations

import datetime as _dt
from typing import Callable, Iterable

from utils import evidence as EV

# 免參數、可由程式無歧義代呼叫的 scope。加東西進來之前先確認它不需要從自然語言
# 猜參數——猜錯的代價是「看起來查過、其實答錯對象」。
PREFETCHABLE = (EV.SCOPE_EARTHQUAKE, EV.SCOPE_TYPHOON)

SCOPE_LABELS = {
    EV.SCOPE_EARTHQUAKE: "地震 · 最新一筆顯著有感地震",
    EV.SCOPE_TYPHOON: "颱風 · 目前警報與追蹤中的熱帶氣旋",
    EV.SCOPE_WEATHER: "天氣 · 即時觀測與特報",
}
SCOPE_TOOLS = {
    EV.SCOPE_EARTHQUAKE: "get_earthquake_info",
    EV.SCOPE_TYPHOON: "get_typhoon_info",
}

# 單一 scope 的 payload 上限。實測地震 433 字、颱風 1,034 字，離上限很遠；
# 但颱風季 tracked_cyclones 會長出多組預報點，而 gemma 的 input TPM 只有 16K，
# 沒有上限的話一個回合就可能被單一 payload 吃掉。
MAX_PAYLOAD_CHARS = 2500
_TPE = _dt.timezone(_dt.timedelta(hours=8))


def prefetch_scopes(scopes: Iterable[str]) -> tuple[str, ...]:
    """從分類器給的 scope 裡挑出「程式可以代呼叫」的那些，保持原順序。"""
    seen, out = set(), []
    for s in scopes or ():
        if s in PREFETCHABLE and s not in seen:
            seen.add(s)
            out.append(s)
    return tuple(out)


def clip(payload: str) -> str:
    if len(payload) <= MAX_PAYLOAD_CHARS:
        return payload
    return payload[:MAX_PAYLOAD_CHARS] + f"\n…（已截斷，原始長度 {len(payload)} 字）"


def run_prefetch(scopes: Iterable[str],
                 impls: dict[str, Callable[[], tuple[str, str]]]) -> list[dict]:
    """依序呼叫（**刻意不用執行緒**）。

    平行化省得到的只有約 0.5 秒（實測地震 0.63s、颱風 0.44s），但 `utils/cwa_weather`
    會讀 `st.secrets`，而本專案已經被「在沒有 ScriptRunContext 的執行緒裡碰 st.*」
    咬過好幾次（`st.write` 還會靜默 no-op，最難查）。這種交換不划算。

    每個 impl 回傳 `(payload_str, status)`——status 由呼叫端判定，因為
    **`empty` 必須與 `ok` 分開**：地震的 `found=False` 是「CWA 這次沒回傳事件」，
    不等於「近期沒有地震」，措辭混為一談就等於編造。

    回傳每個 scope 一筆 `{scope, status, payload, error}`；任何例外都收斂成
    `status=error`，不往外拋——查詢失敗不該連帶弄壞整個回合。
    """
    results = []
    for sc in scopes or ():
        fn = impls.get(sc)
        if fn is None:
            continue
        try:
            payload, status = fn()
            if status not in (EV.STATUS_OK, EV.STATUS_EMPTY):
                status = EV.STATUS_OK
            results.append({"scope": sc, "status": status,
                            "payload": clip(payload or ""), "error": None})
        except Exception as e:
            results.append({"scope": sc, "status": EV.STATUS_ERROR,
                            "payload": None, "error": f"{type(e).__name__}: {str(e)[:160]}"})
    return results


def build_context_block(results: list[dict], *, now: _dt.datetime | None = None) -> str:
    """把 prefetch 結果組成要塞進 context 的那段文字。回傳 "" 表示沒東西可塞。

    語氣上刻意寫明「由程式直接查詢、不是你的記憶、也不是使用者提供的」——模型分不清
    context 裡的東西從哪來，講清楚才不會把官方資料當成使用者的說法去反駁或忽略。
    """
    results = [r for r in (results or []) if r.get("scope")]
    if not results:
        return ""
    now = now or _dt.datetime.now(_TPE)
    ts = now.astimezone(_TPE).strftime("%Y-%m-%d %H:%M")

    lines = [f"（系統：以下是本回合**由程式直接向中央氣象署（CWA）查詢**取得的即時資料，"
             f"查詢時間 {ts}（台灣時間）。這不是你的記憶，也不是使用者提供的內容。）", ""]
    for r in results:
        label = SCOPE_LABELS.get(r["scope"], r["scope"])
        st_ = r.get("status")
        if st_ == EV.STATUS_OK:
            lines.append(f"【{label}】查詢成功")
            lines.append(r.get("payload") or "")
        elif st_ == EV.STATUS_EMPTY:
            # 這一行的措辭是刻意的：查詢有成功，但氣象署沒回傳可顯示的事件。
            # 講成「近期沒有地震」就是拿資料缺漏冒充事實。
            lines.append(f"【{label}】查詢成功，但氣象署**這次沒有回傳可顯示的事件**"
                         "（這代表資料暫時取不到，**不等於**近期沒有發生）")
            lines.append(r.get("payload") or "")
        else:
            lines.append(f"【{label}】**查詢失敗**：{r.get('error') or '未知錯誤'}")
        lines.append("")

    done = "、".join(SCOPE_TOOLS[r["scope"]] for r in results if r["scope"] in SCOPE_TOOLS)
    lines += [
        "（回答須知：",
        "- 講到即時災害狀況時，只能依據上面這段資料，不要補上你記憶中的任何事件、時間或規模。",
        f"- 這些項目本回合已經查過了，不需要再呼叫 {done}。" if done else "",
        "- 上面標「查詢失敗」的項目，就直說這次查不到，不要用既有知識填補。",
        "- 上面標「沒有回傳可顯示的事件」的項目，要說成「這次取不到資料」，"
        "**不可以**說成「近期沒有發生」。",
        "- 使用者只問其中一項時，只回答那一項即可。）",
    ]
    return "\n".join(x for x in lines if x is not None).strip()
