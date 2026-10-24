# -*- coding: utf-8 -*-
"""模型、工具與價目設定（2026-10 實測與官方價目；改這裡就好，不要散落各處）。"""
from __future__ import annotations

HAIKU = "claude-haiku-5-5"
OPUS = "claude-opus-5-5"

# effort：Haiku 用 low（spike T2：簡單題首字約 0.7 秒），Opus 預設本來就是 medium，明寫避免日後預設變動
HAIKU_EFFORT = "low"
OPUS_EFFORT = "medium"
# Haiku 關掉思考：加上安妮亞系統提示＋綁工具後，effort=low 也幾乎每次先思考，首字被拖到 2–4.6 秒；
# 關掉後 0.9 秒左右。升級判斷照樣 10/10（tools/claude_spike/p1_escalate_thinking.py，2026-10-23）。
# Haiku 5.5 只在 effort ≤ high 時接受 disabled，effort 往上調時要一起拿掉這行。
HAIKU_THINKING = {"type": "disabled"}
HAIKU_MAX_TOKENS = 16000   # 翻譯整篇文章時輸出可達數千 token（4000 會截斷）；串流下沒有逾時問題
OPUS_MAX_TOKENS = 16000

# Haiku 用基本版搜尋：spike T2 實測 20250305 比 20260209 快（7s vs 12s）、input 少一半（12.6K vs 21.5K）
HAIKU_SERVER_TOOLS = [
    {"type": "web_search_20250305", "name": "web_search", "max_uses": 2},
]
OPUS_SERVER_TOOLS = [
    {"type": "web_search_20260209", "name": "web_search", "max_uses": 5},
    {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 3},
]
# 階段 2 的深思模式（deepagents，有自訂工具）用「沒有動態篩選」的版本。2026-10-23 實測：20260209 會在沙盒裡
# 跑程式、從程式裡呼叫我們的自訂工具（氣象署），程式暫停等結果；續跑要把沙盒 container 帶回去，deepagents／
# LangChain 都沒帶 → 第二次呼叫 400「code_execution tool use … without a corresponding result」，或模型在
# 程式裡呼叫失敗後說「有查但沒拿到」。基本版沒有沙盒，模型只能直接呼叫工具：可靠、時間軸也看得到。
# ⚠️ web_fetch 不要加 max_content_tokens：2026-10-24 實測，Haiku 單獨呼叫時有效（22,080 → 5,977 token），
# 但深思模式（Opus＋deepagents）只要工具定義帶這個參數，研究報告題就輸出幾個亂碼字、完全不呼叫工具
# （連續 3 次；拿掉後立刻正常，對照組見當次 scratchpad ifrs_isolate.py）。讀網頁的費用只能靠 max_uses 控制。
DEEP_SERVER_TOOLS = [
    {"type": "web_search_20250305", "name": "web_search", "max_uses": 5},
    {"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 3},
]

# server tool 的迴圈太長時 API 會回 stop_reason="pause_turn"，要把回應接回去再送一次
MAX_PAUSE_CONTINUES = 3

# 歷史只帶最近 N 個使用者回合（沿用 Home.py 的 TRIM_LAST_N_USER_TURNS）
HISTORY_USER_TURNS = 18

# $/MTok：base input, 5m cache write, 1h cache write, cache read, output（Haiku 為 ≤100K prompt 的價目）
PRICES = {
    HAIKU: (0.10, 0.125, 0.20, 0.01, 0.50),
    OPUS: (4.00, 5.00, 8.00, 0.20, 20.00),
}
WEB_SEARCH_USD = 10 / 1000

MODEL_LABEL = {HAIKU: "Haiku 5.5", OPUS: "Opus 5.5"}
