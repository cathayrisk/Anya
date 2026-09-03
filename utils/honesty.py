# -*- coding: utf-8 -*-
"""誠實性後處理：系統握有「本回合沒搜尋」的事實時，不信任模型的措辭。

為什麼需要（2026-09-03 線上驗證 V3）：Fast 模式、Web:off、系統 banner 明寫「本回覆未經網路查證」，
模型下一行照樣寫「安妮亞幫你查好了！」＋一個 2024 年的過時數字。寫進 FAST_GEMMA_PROMPT 的
「禁止宣稱查證動作」硬規則對 flash-lite 級模型無效——這是同一天第三次印證「prompt 指令擋不住
gating／誠實性問題」（另兩次：General 捏造引文、報告否認文件裡存在的數字）。

所以改在後處理層做：呼叫端只在 `web_happened == False` 時呼叫，用 regex 剝掉「自稱已查證」的句子
與小標。只動措辭、不動內容；純函式、零 API 呼叫、與模型無關，可離線測（tests/test_routing_honesty.py）。

不做什麼：
- 不動 General 路徑——General 有工具時「查好了」可能是真話；General 的對應修法是 evidence receipt（另案）。
- 不判斷內容真偽——「目前無颱風」這種斷言留給 banner 標示，這裡只剝「我查過了」這個動作宣稱。
"""
from __future__ import annotations

import re

# 否定語境不剝：「沒有查證過」「未查證」「無法查證」是誠實話，正是我們要的。
_NEG = r"(?<!沒有)(?<!未)(?<!沒)(?<!不曾)(?<!無法)"
# 「檢查了一下文法」「審查了」「調查了」不是查證宣稱。
_NOT_INSPECT = r"(?<![檢審調])"

# 兩種樣態：
#   1. 整句宣稱：「安妮亞幫你查好了！🥜」「WakuWaku! 安妮亞幫你查好囉！」「安妮亞查了一下，目前無颱風。」
#      前綴最多 20 字、不跨 CJK 句界（。！？）；ASCII 的 ! ? 允許（「WakuWaku!」是感嘆語尾不是句界）。
#      句尾吃掉標點與 emoji／空白（[^\w\n]：\w 在 Python 3 含 CJK，所以不會吃到下一句文字）。
#   2. 小標／引導語：「🔍 查證結果」「查詢結果：」「經查」——剝標籤，保留其後的內容。
_FALSE_VERIFY_RE = re.compile(
    r"^[^\n。！？]{0,20}?(?:幫你|為你|替你|已經|已|都)?" + _NEG + _NOT_INSPECT +
    r"(?:查好|查過|查證過|查到|確認過|核實過|查了一下|查了)"
    r"[^\n。！!]*[。！!]?[^\w\n]*\n?"
    r"|^[^\w\n]*(?:查詢結果|查證結果|查證資訊|經查)[：:]?[^\w\n]*\n?",
    re.MULTILINE,
)


def strip_false_verification_claims(text: str) -> str:
    """移除模型自稱已查證的句子與小標；其餘原樣保留。空字串進、空字串出。"""
    if not text:
        return text
    out = _FALSE_VERIFY_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", out).strip()
