# -*- coding: utf-8 -*-
"""狀態列的「安妮亞心裡話」：每個階段一組台詞，隨機挑、偶爾冒彩蛋。純函式，無 streamlit 依賴。

2026-10-23 使用者要求：st.status 加一些安妮亞心裡的話或小巧思，讓對話更有趣。
- 素材只用《SPY×FAMILY 間諜家家酒》台灣譯名查得到的設定：讀心（是秘密）、愛吃花生、
  「安妮亞喜歡這個」、彭德（會預知）／彭德曼、伊甸學園的星星與閃電、韓德森老師的「優雅」、
  爸爸代號黃昏、貝琪、奇美拉、企鵝先生。台詞全部自寫，不搬劇中原句。
- 災防題（地震、颱風）一律用正經版，不開玩笑——問的人可能正在擔心。
- 每句都要看得出「現在在做什麼」（搜尋、讀網頁、升級深思），好玩不能蓋過資訊。
- 回傳的是 st.status 標籤用的 markdown：進行中包 :shimmer[]（1.57+ 的流光指令）。
  台詞裡不可有方括號，否則會截斷指令。
"""
from __future__ import annotations

import datetime as _dt
import random
import re

from .router import instruction_part

_TPE = _dt.timezone(_dt.timedelta(hours=8))
EGG_RATE = 0.12          # 等待中的台詞有多少機率換成隨機彩蛋
TIME_RATE = 0.5          # 開場沒有關鍵字彩蛋時，多少機率用時段台詞

# 階段 → (圖示, 台詞)
LINES: dict[str, tuple[str, tuple[str, ...]]] = {
    "received": ("🥜", (
        "安妮亞收到了！wakuwaku～",
        "安妮亞偷偷讀一下你的心…噓，這是秘密",
        "安妮亞把花生先放下，認真聽",
        "收到！安妮亞的秘密任務開始",
        "安妮亞立正站好，準備回答",
    )),
    "thinking": ("🤔", (
        "安妮亞動腦中…",
        "安妮亞在想…答對可以拿星星嗎",
        "安妮亞努力裝成大人的樣子思考",
        "嗯嗯…安妮亞快想到了",
    )),
    "deep_thinking": ("🧠", (
        "安妮亞開啟認真模式，腦袋轉轉轉",
        "這題值得一顆星星，安妮亞好好想",
        "安妮亞學黃昏那樣冷靜分析中",
        "安妮亞把線索排成一排，一個一個看",
        "腦袋快冒煙了…再一下下就好",
        "安妮亞努力不打瞌睡中",
    )),
    "searching": ("🔎", (
        "安妮亞出動偵查！上網找線索",
        "安妮亞學彭德曼潛入網路蒐集情報",
        "安妮亞在網路上翻資料中",
    )),
    "fetching": ("📄", (
        "安妮亞在讀網頁…字好多",
        "安妮亞一行一行讀情報中",
    )),
    "organizing": ("🧮", (
        "安妮亞把線索整理成小抄",
        "安妮亞拿出計算機算一算",
        "安妮亞在幫資料排隊",
    )),
    "skill": ("📚", (
        "安妮亞翻開秘笈看做法",
        "安妮亞照著說明書一步一步來",
        "伊甸學園的課本拿出來了",
    )),
    "reflect": ("💭", (
        "安妮亞停下來想想夠不夠",
        "安妮亞檢查還缺哪一塊拼圖",
        "這些線索夠不夠拿星星呢",
    )),
    "escalate": ("↗️", (
        "這題好難，安妮亞去請爸爸幫忙・深思模式",
        "重要任務！安妮亞換成深思模式",
        "安妮亞換上認真模式・深思模式",
    )),
    "socratic": ("🧭", (
        "安妮亞換成提問模式，陪你一起想",
        "安妮亞來當小老師，換你想想看・提問引導",
    )),
    "writing": ("✍️", (
        "安妮亞開始寫了！",
        "安妮亞一個字一個字寫給你",
        "安妮亞寫字中…這次要拿星星",
        "安妮亞邊寫邊檢查，很認真",
    )),
    "cwa": ("🌐", (
        "安妮亞先去氣象署查即時資料",
    )),
}

# 正經版（災防題）：每階段一句，不開玩笑
SERIOUS: dict[str, tuple[str, str]] = {
    "received": ("🥜", "安妮亞收到了，馬上處理"),
    "thinking": ("🤔", "安妮亞思考中"),
    "deep_thinking": ("🧠", "深思模式思考中"),
    "searching": ("🔎", "安妮亞上網搜尋中"),
    "fetching": ("📄", "安妮亞在讀網頁"),
    "organizing": ("🧮", "整理資料中"),
    "skill": ("📚", "讀技能說明中"),
    "reflect": ("💭", "檢查資料夠不夠"),
    "escalate": ("↗️", "升級到深思模式"),
    "socratic": ("🧭", "交給提問引導模式"),
    "writing": ("✍️", "安妮亞整理回答中"),
    "cwa": ("🌐", "安妮亞先去氣象署查即時資料"),
}

# 等待中偶爾冒出來的彩蛋（只在 received／thinking／deep_thinking 出現）
EGGS: tuple[tuple[str, str], ...] = (
    ("🐶", "彭德好像預見了什麼…"),
    ("🎩", "韓德森老師在看，要優雅"),
    ("🧸", "奇美拉也來幫忙了"),
    ("🐧", "企鵝先生也在旁邊聽"),
    ("🥜", "安妮亞偷偷吃了一顆花生"),
    ("🎀", "貝琪說這題安妮亞一定會"),
    ("🤫", "不能讓別人知道安妮亞會讀心…"),
    ("⚡", "千萬不能拿到閃電…"),
)
_EGG_STAGES = {"received", "thinking", "deep_thinking"}

# 開場的關鍵字彩蛋：只看指示部分（長文章內文提到花生不算），先命中先用
KEYWORD_EGGS: tuple[tuple[re.Pattern, str, str], ...] = (
    (re.compile(r"花生"), "🥜", "花生！安妮亞喜歡這個！"),
    (re.compile(r"間諜|特務|\bspy\b", re.IGNORECASE), "🕵️", "間諜！？wakuwaku！"),
    (re.compile(r"心情.{0,3}(?:不好|差|低落|很糟)|難過|好累|好煩|壓力好大|不開心|好想哭"), "🫂", "安妮亞先給你摸摸頭"),
    (re.compile(r"考試|作業|功課|期末|期中"), "📚", "考試…安妮亞最怕這個了"),
    (re.compile(r"彭德|狗狗|小狗"), "🐶", "彭德也想一起聽"),
    (re.compile(r"世界和平"), "🕊️", "為了世界和平，安妮亞出動！"),
    (re.compile(r"優雅"), "🎩", "優雅！韓德森老師會稱讚的"),
    (re.compile(r"星星"), "⭐", "星星！安妮亞想要星星！"),
    (re.compile(r"早安|早啊"), "☀️", "早安！安妮亞今天也很有精神"),
    (re.compile(r"晚安"), "🌙", "晚安…安妮亞也有點想睡了"),
    (re.compile(r"謝謝|感謝|thx|thanks", re.IGNORECASE), "💌", "安妮亞收到你的謝謝了！"),
)

# 台灣時間的時段台詞（開場用）：(起, 迄) 小時，含起不含迄
TIME_LINES: tuple[tuple[int, int, str, str], ...] = (
    (0, 5, "🌙", "這麼晚還在忙…安妮亞陪你"),
    (5, 8, "🌅", "這麼早就開工，安妮亞佩服"),
    (12, 13, "🍱", "午餐時間了，安妮亞想吃花生"),
)

DONE: dict[str, tuple[str, ...]] = {
    "fast": ("安妮亞想要花生當獎勵", "任務成功！", "安妮亞表現很優雅吧", "安妮亞可以拿星星嗎"),
    "deep": ("腦袋要吃花生補充一下", "認真任務完成！", "安妮亞努力想好了"),
    "socratic": ("換你想想看", "安妮亞等你的答案"),
}
DONE_LABEL = {"fast": "一般模式", "deep": "深思模式", "socratic": "提問引導"}
CWA_DONE = "✅ 已直接取自中央氣象署（未經模型）"


def _clean(s: str, limit: int = 30) -> str:
    """模型給的升級理由要接在標籤後面：拿掉會打壞 markdown 指令的字元、限長。"""
    s = re.sub(r"[\[\]*_`#<>|\\:]", "", re.sub(r"\s+", " ", s or "")).strip()
    return s[:limit] + ("…" if len(s) > limit else "")


class Monologue:
    """一回合一個。rng／now 可注入，方便測試。"""

    def __init__(self, user_text: str = "", *, serious: bool = False, rng: random.Random | None = None,
                 now: _dt.datetime | None = None, egg_rate: float = EGG_RATE, time_rate: float = TIME_RATE):
        self.text = instruction_part(user_text or "")
        self.serious = serious
        self.rng = rng or random.Random()
        self.now = (now or _dt.datetime.now(_TPE)).astimezone(_TPE)
        self.egg_rate = egg_rate
        self.time_rate = time_rate
        self._used: set[str] = set()
        self._last = ""

    @staticmethod
    def running(icon: str, line: str) -> str:
        return f"{icon} :shimmer[{line}]"

    def opening(self) -> str:
        if not self.serious:
            for pat, icon, line in KEYWORD_EGGS:
                if pat.search(self.text):
                    return self._mark(icon, line)
            for start, end, icon, line in TIME_LINES:
                if start <= self.now.hour < end and self.rng.random() < self.time_rate:
                    return self._mark(icon, line)
        return self.pick("received")

    def pick(self, stage: str) -> str:
        if self.serious or stage not in LINES:
            icon, line = SERIOUS.get(stage, SERIOUS["thinking"])
            return self._mark(icon, line)
        if stage in _EGG_STAGES and self.rng.random() < self.egg_rate:
            eggs = [e for e in EGGS if e[1] not in self._used] or list(EGGS)
            return self._mark(*self.rng.choice(eggs))
        icon, pool = LINES[stage]
        fresh = [s for s in pool if s not in self._used] or [s for s in pool if s != self._last] or list(pool)
        return self._mark(icon, self.rng.choice(fresh))

    def escalate(self, *, socratic: bool, reason: str = "") -> str:
        label = self.pick("socratic" if socratic else "escalate")
        why = _clean(reason)
        return label + (f" :gray[{why}]" if why else "")

    def done(self, mode: str) -> str:
        if mode == "cwa":
            return CWA_DONE
        name = DONE_LABEL.get(mode, "")
        head = f"✅ 回答完成（{name}）" if name else "✅ 回答完成"
        if self.serious or mode not in DONE:
            return head
        return f"{head} :gray[・{self.rng.choice(DONE[mode])}]"

    def failed(self) -> str:
        return "⚠️ 這次沒有完成" + ("" if self.serious else " :gray[・安妮亞會再努力]")

    def _mark(self, icon: str, line: str) -> str:
        self._used.add(line)
        self._last = line
        return self.running(icon, line)
