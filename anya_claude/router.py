# -*- coding: utf-8 -*-
"""第一層路由：不呼叫模型，命中就直接送深思（Opus）。純函式，無 streamlit 依賴。

規則**照搬 Home.py**（2026-10-23 使用者決定：現有升級方式沿用），順序也一樣；
另外加一條「使用者明講要深思」。regex 的取捨理由見 Home.py 原處註解，這裡只留摘要。
第 5 階段替換 Home.py 時，這份會成為唯一來源（現在兩份並存是過渡期）。
第二層（Haiku 自己判斷）在 graph.py 的 escalate_to_deep 工具。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

try:
    from widget_templates import WIDGET_HINT_RE
except Exception:  # 缺檔時降級：不因互動元件關鍵詞升級
    WIDGET_HINT_RE = None
try:   # 口語地震說法（有在搖、地牛翻身…）：和災防分類器共用同一條
    from utils.hazard_intent import QUAKE_COLLOQUIAL_RE
except Exception:
    QUAKE_COLLOQUIAL_RE = None

DEEP_RESEARCH_HINT_RE = re.compile(
    r"深度研究|研究報告|文獻回顧|文獻探討|系統性(?:回顧|調查)|deep\s*research|literature\s*review"
)
# 明確的「寫程式」請求
CODING_HINT_RE = re.compile(
    r"(?:寫|幫我做|產生|生成).{0,8}(?:程式|腳本|函式|巨集|macro|script|function)|"
    r"(?:python|vba|pandas).{0,12}(?:程式|腳本|函式|巨集|寫)|debug|除錯|重構",
    re.IGNORECASE,
)
GENERAL_HINT_RE = re.compile(
    r"系統性比較|深入分析|完整報告|全面評估|交叉比對|魔鬼代言人|蘇格拉底|專家團隊"
)
# 災防：寧可多花一次深思，也不能拿模型的記憶當即時警報
HAZARD_HINT_RE = re.compile(
    r"地震|震度|餘震|海嘯|颱風|台風|颶風|豪雨|大雨特報|豪雨特報|陸上警報|海上警報|氣象署|氣象局"
)
# 使用者刻意講出口的查證動詞（保持窄）
VERIFY_HINT_RE = re.compile(r"查證|求證|核實|查核|事實查核|fact\s*check", re.IGNORECASE)
# 新增：使用者明講要深思。刻意不收單獨的「深入」「仔細」（「深入淺出」「仔細看一下」會誤升級）
FORCE_DEEP_RE = re.compile(
    r"深思|深入(?:想|思考)|仔細(?:想|思考)|認真(?:想|思考)|好好想(?:想|一下)?|多想一下|想清楚一點"
    r"|用\s*opus|思考模式",
    re.IGNORECASE,
)

# 翻譯請求：交給 Haiku（2026-10-23 使用者回報：翻譯一篇文章被送去深思，應該只有特殊任務才用深思）
TRANSLATE_RE = re.compile(r"翻譯|翻成|譯成|翻為|譯為|中翻英|英翻中|日翻中|translate", re.IGNORECASE)

# 長訊息多半是「指示＋貼上的文章」。關鍵詞只看頭尾的指示部分——文章內文提到「地震」「研究報告」
# 「估值」不代表使用者要那類服務（舊版整則比對，貼文章很容易誤升級）。
LONG_TEXT_CHARS = 300
EDGE_CHARS = 150


def instruction_part(text: str) -> str:
    t = text or ""
    if len(t) <= LONG_TEXT_CHARS:
        return t
    return t[:EDGE_CHARS] + "\n" + t[-EDGE_CHARS:]


# 生日或星座：「合不合」旁邊要有這個才算問配對
_ASTRO_PERSON = (r"(?:星座|星盤|生日|出生|\d{4}[/.\-年]\d{1,2}"
                 r"|(?:牡羊|金牛|雙子|巨蟹|獅子|處女|天秤|天蠍|射手|摩羯|水瓶|雙魚)座)")

# per-skill 高精度關鍵詞（命中即升級——Haiku 沒有 skill）
SKILL_HINT_RES: dict[str, re.Pattern] = {
    "market-research": re.compile(r"TAM|SAM|SOM|市場規模|市場調查|市場區隔", re.IGNORECASE),
    "statistical-analyst": re.compile(r"假設檢定|顯著性|信賴區間|p\s*值|迴歸分析|統計檢定"),
    "financial-analyst": re.compile(r"DCF|現金流折現|估值|財報比率|CAC|LTV|ARR|MRR", re.IGNORECASE),
    "data-quality-auditor": re.compile(r"資料品質|缺值|離群值|資料清理"),
    "astro-natal": re.compile(
        r"星盤|本命盤|命盤|占星|上升星座|太陽星座|月亮星座"
        r"|(?:太陽|月亮|水星|金星|火星|木星|土星|天王星|海王星|冥王星|凱龍|"
        r"上升|天頂|下降|天底|[南北]交點)"
        r"[^。！？\n]{0,10}?(?:星座|落[在入]|[一二三四五六七八九十幾]+宮|\d{1,2}\s*宮|"
        r"牡羊|金牛|雙子|巨蟹|獅子|處女|天秤|天蠍|射手|摩羯|水瓶|雙魚|"
        r"合相|三分|四分|六分|對分|梅花|刑|拱|相位|逆行)"
        r"|(?:合相|三分|四分|六分|對分|刑|拱)[^。！？\n]{0,6}?"
        r"(?:太陽|月亮|水星|金星|火星|木星|土星|天王星|海王星|冥王星|凱龍|上升|天頂)"),
    # 2026-10-24 實測漏接：「我明年的運勢如何？」「我跟男友合不合？我 1990/5/3…」→ Haiku 沒排盤、婉拒式自答
    "astro-forecast": re.compile(
        r"流年|行運|返照|小限|運勢|(?:今年|明年|下半年|上半年|這個月|下個月|\d{4}\s*年).{0,6}運氣"),
    # 「合不合」單獨出現太寬（「合不合適」「合不合理」），要旁邊有生日或星座才算
    "astro-relationship": re.compile(
        r"合盤|兩人.{0,4}星盤|配對.{0,4}星盤|契合度"
        r"|(?:合不合|配不配|速配)(?![適理法規用格]).{0,40}?" + _ASTRO_PERSON
        + r"|" + _ASTRO_PERSON + r".{0,40}?(?:合不合|配不配|速配)(?![適理法規用格])"),
    "excel-vba": re.compile(
        r"\bvba\b|巨集|\.xlsm\b"
        r"|\bmacros?\b[\s\S]{0,24}(?:excel|workbook|worksheet|\.xlsm|vbe|module|\bsub\b|code"
        r"|程式碼|模組|活頁簿|工作表|跑不動|當掉|不能跑|無法執行|除錯|撰寫)"
        r"|(?:excel|workbook|worksheet|\.xlsm|vbe|module|\bsub\b|code"
        r"|程式碼|模組|活頁簿|工作表|跑不動|當掉|不能跑|無法執行|除錯|撰寫)[\s\S]{0,24}\bmacros?\b",
        re.IGNORECASE),
}

# 提問引導（蘇格拉底）模式：狀態放在頁面（session），不指望模型跨回合記得
SOCRATIC_ENTER_RE = re.compile(
    r"陪我想|幫我釐清|別給答案|不要給答案|不要直接(?:給|說)|引導我(?:想|思考)?|蘇格拉底"
)
SOCRATIC_EXIT_RE = re.compile(
    r"直接說|直接講|直接寫|給我範例|幫我列選項|給我答案|直接給(?:我)?(?:答案|結論)"
)


def match_skill_hint(text: str) -> str | None:
    for name, pat in SKILL_HINT_RES.items():
        if pat.search(text or ""):
            return name
    return None


@dataclass(frozen=True)
class Route:
    mode: str               # "fast"（Haiku）| "deep"（Opus）
    reason: str             # 進 dev 面板與 log，日後調規則才知道是哪條命中
    socratic: bool = False


def next_socratic_state(text: str, current: str) -> str:
    """回傳本回合的引導模式狀態（"assist" | "socratic"）。退出詞與「要具體交付物」優先。"""
    t = instruction_part(text)
    if (SOCRATIC_EXIT_RE.search(t) or CODING_HINT_RE.search(t) or DEEP_RESEARCH_HINT_RE.search(t)
            or TRANSLATE_RE.search(t)
            or (WIDGET_HINT_RE is not None and WIDGET_HINT_RE.search(t))):
        return "assist"
    if SOCRATIC_ENTER_RE.search(t):
        return "socratic"
    return current if current in ("assist", "socratic") else "assist"


def decide(text: str, *, socratic_active: bool = False, has_images: bool = False,
           has_docs: bool = False) -> Route:
    t = instruction_part(text)
    if socratic_active:
        return Route("deep", "socratic_mode", socratic=True)
    if has_images:
        return Route("deep", "contains_image")
    if has_docs:
        return Route("deep", "has_docs")
    if TRANSLATE_RE.search(t) and not FORCE_DEEP_RE.search(t):
        return Route("fast", "translation")
    if HAZARD_HINT_RE.search(t) or (QUAKE_COLLOQUIAL_RE is not None and QUAKE_COLLOQUIAL_RE.search(t)):
        return Route("deep", "hazard_hint")
    if VERIFY_HINT_RE.search(t):
        return Route("deep", "verify_hint")
    if FORCE_DEEP_RE.search(t):
        return Route("deep", "user_requested_deep")
    if (DEEP_RESEARCH_HINT_RE.search(t) or GENERAL_HINT_RE.search(t) or CODING_HINT_RE.search(t)
            or (WIDGET_HINT_RE is not None and WIDGET_HINT_RE.search(t))):
        return Route("deep", "keyword_hint")
    skill = match_skill_hint(t)
    if skill:
        return Route("deep", f"skill_hint:{skill}")
    return Route("fast", "default_fast")
