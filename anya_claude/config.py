# -*- coding: utf-8 -*-
"""模型、工具與價目設定（2026-10 實測與官方價目；改這裡就好，不要散落各處）。"""
from __future__ import annotations

HAIKU = "claude-haiku-5-5"
OPUS = "claude-opus-5-5"
SONNET = "claude-sonnet-5-5"   # 階段 4 魔鬼代言人
# 階段 4 研究員：2026-10-24 使用者先選 Sonnet（單一子題試算 $0.08），完整試跑 IFRS 研究題研究員合計 $0.9、
# 整題 $1.6–1.7；換 Haiku 整題 $0.83、研究員合計 $0.20，內容一樣扎實 → 使用者改選 Haiku
RESEARCHER_MODEL = HAIKU

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

# 階段 4 研究流程（deepagents 的 deep_research 範例：最多 3 位平行、最多 3 輪；這裡收緊成 2 輪控制時間與費用）
MAX_RESEARCHERS = 3            # 一輪最多派幾位研究員（平行）
MAX_RESEARCH_ROUNDS = 2        # 最多派幾輪（第二輪補缺口）
RESEARCHER_MAX_CALLS = 8       # 每位研究員最多呼叫模型幾次（搜尋＋反思交錯，約 5 次搜尋）
# 研究沒啟動的判準：主持人一位研究員都沒派、回答又短於這個字數（deep_mw.research_not_started）。
# 2026-10-24 線上偶發：重啟後第一題只輸出幾行簡體字（「1 个参与者对话」），31 秒 $0.05；正常報告都上萬字
RESEARCH_MIN_CHARS = 400
RESEARCHER_MAX_TOKENS = 8000
# 研究員用基本版（理由同 DEEP_SERVER_TOOLS）。讀網頁不加 max_content_tokens（見上面的 ⚠️）。
# 2026-10-24 第一次試跑：搜尋 5＋讀網頁 2 時，一位研究員單次輸入 12.6 萬 token，3 位合計 $0.90（主持人才 $0.62）
# → 收緊成搜尋 4、讀網頁 1
RESEARCH_SERVER_TOOLS = [
    {"type": "web_search_20250305", "name": "web_search", "max_uses": 4},
    {"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 1},
]

# server tool 的迴圈太長時 API 會回 stop_reason="pause_turn"，要把回應接回去再送一次
MAX_PAUSE_CONTINUES = 3

# 歷史只帶最近 N 個使用者回合（沿用 Home.py 的 TRIM_LAST_N_USER_TURNS）
HISTORY_USER_TURNS = 18

# $/MTok：base input, 5m cache write, 1h cache write, cache read, output（Haiku 為 ≤100K prompt 的價目）
PRICES = {
    HAIKU: (0.10, 0.125, 0.20, 0.01, 0.50),
    OPUS: (4.00, 5.00, 8.00, 0.20, 20.00),
    SONNET: (2.00, 2.50, 4.00, 0.10, 10.00),
}
# Haiku 5.5 的 prompt 超過 10 萬 token（含快取讀寫）整次改用這組價目（2026-10 官方價目表）。
# 大份文件、研究員連續搜尋都會超過；之前一律用低價算，用量記錄會少算到 1/5
PRICES_LONG = {HAIKU: (0.50, 0.625, 1.00, 0.05, 2.50)}
LONG_PROMPT_TOKENS = 100_000
WEB_SEARCH_USD = 10 / 1000

MODEL_LABEL = {HAIKU: "Haiku 5.5", OPUS: "Opus 5.5", SONNET: "Sonnet 5.5"}
