# -*- coding: utf-8 -*-
"""災防提問的意圖分類：即時查詢 vs 純知識（純函式，無 streamlit 依賴）。

## 為什麼要分三態，而不是二元
一小時測試 T5：口語問「最近台灣有地震嗎？有沒有颱風要來？」→ `HAZARD_HINT_RE` 正確
命中並升級 General，但**模型沒呼叫 CWA 工具**，憑記憶編出半年前的地震資料。
第 4 步要做的 controller prefetch 需要知道「這題該不該強制查」，就得先分類。

而 `HAZARD_HINT_RE` 太寬——它只要看到「地震」兩個字就命中，於是
「地震規模是怎麼定義的？」也會被強制 prefetch，若之後接上程式渲染還會直接答錯題。

**但也不能只用二元切分。** 錯誤成本是**不對稱**的：
- 誤判成 live 的代價：多打一次本地 CWA API（0.4–0.9 秒，不耗 LLM 配額）
- 誤判成 knowledge 的代價：回到 T5——憑記憶編造即時災害資訊

所以第三態 `UNCERTAIN` 存在的意義是：**不確定時偏向去查**，但不把整題改寫成災防模板。
這個不對稱是刻意寫進設計的，不是靠 regex 調到剛好。

## 判斷順序（live 優先）
1. 有災害實體詞嗎？沒有 → `NONE`
2. 有即時訊號（時間詞／狀態動作詞／「有沒有＋災害詞」問法）→ `EXPLICIT_LIVE`
   即使同時出現知識詞也算 live——「為什麼**最近**地震這麼多」需要真實資料才答得好。
3. 否則有知識詞 → `CLEAR_KNOWLEDGE`
4. 其餘 → `UNCERTAIN`

## 涵蓋範圍的誠實邊界
只有地震、颱風、天氣有對應的 CWA 工具（`get_earthquake_info` / `get_typhoon_info` /
`get_weather`）。**海嘯沒有任何工具**——偵測到會標在 `uncovered` 裡，讓呼叫端知道
「這題屬於災防、但我們沒有可強制的資料源」，而不是假裝已涵蓋。

刻意用可讀的 pattern list 而不是一條巨大 regex：`HAZARD_HINT_RE` 就是疊成一條之後
沒人看得出它會誤抓知識題。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

STATE_LIVE = "explicit_live"
STATE_KNOWLEDGE = "clear_knowledge"
STATE_UNCERTAIN = "uncertain"
STATE_NONE = "none"

# ── 災害實體 → 對應的 evidence scope（與 utils/evidence.py 的 enum 一致）──────
ENTITY_SCOPES: dict[str, tuple[str, ...]] = {
    "earthquake_latest": ("地震", "震度", "餘震", "震央", "芮氏"),
    "typhoon_active": ("颱風", "台風", "颶風", "熱帶氣旋", "熱帶性低氣壓"),
    # 「天氣／氣象」刻意收進來——與 HAZARD_HINT_RE 的取捨不同：那條是決定**要不要升級
    # General**（誤升級要燒 gemma 的 16K TPM，所以刻意排除「天氣」），這裡只決定
    # **要不要多打一次本地 CWA API**（0.4–0.9 秒、不耗 LLM 配額）。成本結構不同，取捨就不同。
    "weather_current": ("天氣", "氣象", "豪雨", "大雨", "特報", "陣雨", "降雨機率", "氣溫"),
}
# 屬於災防、但**沒有任何工具可查**——不可假裝已涵蓋
UNCOVERED_ENTITIES = ("海嘯", "土石流", "淹水")

# ── 即時訊號 ────────────────────────────────────────────────────────────────
LIVE_TIME = ("現在", "目前", "最近", "近期", "今天", "今日", "昨天", "剛剛", "剛才",
             "最新", "這幾天", "這陣子", "這禮拜", "這週", "本週")
LIVE_ACTION = ("發布", "發佈", "警報", "特報", "解除", "登陸", "生效", "要來", "來襲",
               "靠近", "查證", "查一下", "實際查", "查查", "有無發布")
# 「有沒有」單獨不算 live（「地震有沒有分級」會誤判），必須緊貼災害詞
LIVE_QUESTION_RE = re.compile(
    r"(?:有沒有|有無|是否有|會不會有)\s*(?:大|強烈|明顯|什麼|比較大的)?"
    r"(?:地震|餘震|颱風|台風|颶風|警報|特報|豪雨|大雨)"
    # ⚠️ 第二式的結尾詞必須夠specific：原本收裸的「沒」，結果
    # 「地震有沒有分級？」被判成 live（地震＋「有」＋「沒」在 4 字內）。
    # 改成只認完整的疑問尾／來襲詞。
    r"|(?:地震|餘震|颱風|台風|警報|特報)[^\n]{0,3}(?:嗎[？?]?|要來|來了|了沒|來襲|靠近)"
)

# ── 知識訊號 ────────────────────────────────────────────────────────────────
KNOWLEDGE = ("定義", "怎麼形成", "如何形成", "為什麼會", "為何會", "原理", "機制",
             "如何測量", "怎麼測", "怎麼算", "分級", "級距", "等級制", "差別", "區別",
             "是什麼意思", "意思是", "歷史上", "防災", "怎麼準備", "如何準備",
             "避難", "科普", "原因是")


@dataclass(frozen=True)
class HazardIntent:
    state: str
    scopes: tuple[str, ...]          # 該題涉及、且**有工具可查**的 scope
    uncovered: tuple[str, ...]       # 涉及但沒有工具的災害詞（如海嘯）
    signals: dict                    # 命中了哪些詞——shadow log 診斷用，不參與判斷

    @property
    def should_prefetch(self) -> bool:
        """第 4 步會用到：live 與 uncertain 都查（錯誤成本不對稱），knowledge 不查。"""
        return self.state in (STATE_LIVE, STATE_UNCERTAIN) and bool(self.scopes)


def _hits(text: str, words) -> tuple[str, ...]:
    return tuple(w for w in words if w in text)


def classify_hazard_intent(text: str) -> HazardIntent:
    t = text or ""
    scopes, ent_hits = [], []
    for scope, words in ENTITY_SCOPES.items():
        h = _hits(t, words)
        if h:
            scopes.append(scope)
            ent_hits.extend(h)
    uncovered = _hits(t, UNCOVERED_ENTITIES)

    if not scopes and not uncovered:
        return HazardIntent(STATE_NONE, (), (), {})

    time_hits = _hits(t, LIVE_TIME)
    action_hits = _hits(t, LIVE_ACTION)
    q = LIVE_QUESTION_RE.search(t)
    know_hits = _hits(t, KNOWLEDGE)
    signals = {"entities": tuple(ent_hits), "time": time_hits, "action": action_hits,
               "question": q.group(0) if q else None, "knowledge": know_hits}

    # live 優先：「為什麼**最近**地震這麼多」需要真實資料才答得好
    if time_hits or action_hits or q:
        state = STATE_LIVE
    elif know_hits:
        state = STATE_KNOWLEDGE
    else:
        state = STATE_UNCERTAIN
    return HazardIntent(state, tuple(scopes), uncovered, signals)
