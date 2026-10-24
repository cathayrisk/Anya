# -*- coding: utf-8 -*-
"""回答文字的顯示前處理（純函式，無 streamlit 依賴）。

normalize_markdown_for_streamlit／_emphasis_to_html／build_web_sources_footer 從 Home.py §F 原樣移植
（第 5 階段替換 Home.py 後這裡是唯一來源）。新增的是串流專用的 display_text()：
改成真串流後，誠實剝除不能等到最後才做——否則使用者會先看到「我查好了」再看它消失。
"""
from __future__ import annotations

import re
import unicodedata

from utils.honesty import _FALSE_VERIFY_RE as _CLAIM_SENTENCE_RE
from utils.mathtext import latex_to_plain

CODE_FENCE_WHOLE_BLOCK_RE = re.compile(
    r"^\s*```(?:markdown|md|text)?\s*\r?\n([\s\S]*?)\r?\n```\s*$",
    flags=re.IGNORECASE,
)
_ST_COLORS = "blue|green|orange|red|violet|gray|grey|rainbow|primary"
_MISSING_COLON_RE = re.compile(r"(?<![:\w-])(" + _ST_COLORS + r")(-badge|-background)?\[")


def _strip_unbalanced_code_fences(text: str) -> str:
    if not text or text.count("```") % 2 == 0:
        return text
    return "\n".join(ln for ln in text.splitlines() if not re.match(r"^\s*```", ln))


def _maybe_unindent_indented_block(text: str) -> str:
    """整段被 4-space 縮排導致 Streamlit 誤判為 code block 時，移除縮排。"""
    if not text:
        return text
    lines = text.splitlines()
    non_empty = [ln for ln in lines if ln.strip() != ""]
    if not non_empty:
        return text
    indented = sum(1 for ln in non_empty if ln.startswith("    ") or ln.startswith("\t"))
    if (indented / len(non_empty)) < 0.7:
        return text
    out = []
    for ln in lines:
        if ln.startswith("    "):
            out.append(ln[4:])
        elif ln.startswith("\t"):
            out.append(ln[1:])
        else:
            out.append(ln)
    return "\n".join(out)


def _repair_color_directives(text: str) -> str:
    """把漏冒號的顏色指令補回去（orange[重點] → :orange[重點]）；程式碼內一律不動。"""
    if not text:
        return text
    stash: list = []

    def _hold(m):
        stash.append(m.group(0))
        return "\x00c%d\x00" % (len(stash) - 1)

    t = re.sub(r"```.*?```", _hold, text, flags=re.S)
    t = re.sub(r"`[^`]*`", _hold, t)
    t = _MISSING_COLON_RE.sub(lambda m: ":" + m.group(0), t)
    return re.sub(r"\x00c(\d+)\x00", lambda m: stash[int(m.group(1))], t)


def normalize_markdown_for_streamlit(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFC", text).strip("﻿")
    m = CODE_FENCE_WHOLE_BLOCK_RE.match(t.strip())
    if m:
        t = m.group(1)
    t = _strip_unbalanced_code_fences(t)
    t = _maybe_unindent_indented_block(t)
    t = re.sub(r"\\([*_`])", r"\1", t)
    t = latex_to_plain(t)
    t = _escape_dollars(t)
    t = _repair_color_directives(t)
    return t


def _escape_dollars(t: str) -> str:
    """latex_to_plain 之後還留著的 $ 都是貨幣（數學式已經轉成純文字），一律跳脫成 \\$。
    2026-10-24 實測：「NT$10M … NT$10M」兩個 $ 中間被 Streamlit 當成數學式，畫成一個字一行的斜體字母。
    程式碼（``` 與 `）原樣不動。"""
    if "$" not in t:
        return t
    stash: list[str] = []

    def _hold(m: re.Match) -> str:
        stash.append(m.group(0))
        return "\x00d%d\x00" % (len(stash) - 1)

    t = re.sub(r"```.*?```", _hold, t, flags=re.S)
    t = re.sub(r"`[^`\n]*`", _hold, t)
    t = re.sub(r"(?<!\\)\$", r"\\$", t)
    return re.sub(r"\x00d(\d+)\x00", lambda m: stash[int(m.group(1))], t)


def emphasis_to_html(t: str) -> str:
    """**粗體** → <strong>（避開程式碼），解決 CommonMark 對 CJK 粗體不渲染的問題。"""
    if not t:
        return t
    blocks = []

    def _stash(m):
        blocks.append(m.group(0))
        return "\x00%d\x00" % (len(blocks) - 1)

    t = re.sub(r"```.*?```", _stash, t, flags=re.S)
    t = re.sub(r"`[^`]*`", _stash, t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t, flags=re.S)
    return re.sub(r"\x00(\d+)\x00", lambda m: blocks[int(m.group(1))], t)


def render_ready(text: str) -> str:
    """最終要丟給 st.markdown(unsafe_allow_html=True) 的字串。"""
    return emphasis_to_html(fix_mermaid_blocks(normalize_markdown_for_streamlit(text)))


# ── Mermaid（Streamlit 1.59 起 st.markdown 直接畫 ```mermaid 區塊）──
# 語法錯一個字就整張變成紅色錯誤框（1.63 實測），最常見的是節點文字裡有括號、冒號。
# 流程圖的方括號／大括號節點與連線文字一律補上雙引號——加了引號永遠合法，沒加才會出事。
_MERMAID_BLOCK_RE = re.compile(r"(^[ \t]*(`{3,})mermaid[ \t]*\r?\n)([\s\S]*?)(\r?\n[ \t]*\2[ \t]*$)",
                               re.M | re.I)
_MM_SQUARE_RE = re.compile(r"(?<=[\w一-鿿])\[(?![\[(/\\\"])([^\[\]\"\n]+)\]")
_MM_CURLY_RE = re.compile(r"(?<=[\w一-鿿])\{(?![{\"])([^{}\"\n]+)\}")
_MM_EDGE_RE = re.compile(r"\|(?!\")([^|\"\n]+)\|")
_MM_SKIP_PREFIX = ("%%", "classDef", "class ", "style ", "linkStyle", "click ")


def fix_mermaid(code: str) -> str:
    """只處理 flowchart／graph；其他圖種（sequenceDiagram、timeline…）語法不同，原樣不動。"""
    lines = (code or "").splitlines()
    head = next((ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("%%")), "")
    if not re.match(r"(?:flowchart|graph)\b", head, re.I):
        return code
    out = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith(_MM_SKIP_PREFIX) or s == head:
            out.append(ln)
            continue
        ln = _MM_SQUARE_RE.sub(lambda m: '["' + m.group(1).strip() + '"]', ln)
        ln = _MM_CURLY_RE.sub(lambda m: '{"' + m.group(1).strip() + '"}', ln)
        ln = _MM_EDGE_RE.sub(lambda m: '|"' + m.group(1).strip() + '"|', ln)
        out.append(ln)
    return "\n".join(out)


def fix_mermaid_blocks(text: str) -> str:
    if "mermaid" not in (text or ""):
        return text
    return _MERMAID_BLOCK_RE.sub(lambda m: m.group(1) + fix_mermaid(m.group(3)) + m.group(4), text)


# 串流時暫不顯示的尾巴：最後一個句界之後、長度在這個以內的部分。
# 誠實剝除的 regex 以句子為單位，句子沒寫完就比對不到 → 先藏起來，寫完再一起判斷。
HOLD_BACK_MAX = 40
_SENTENCE_END_RE = re.compile(r"[。！？!?\n]")
_FENCE_LINE_RE = re.compile(r"^[ \t]*`{3,}([\w-]*)", re.M)
MERMAID_DRAWING = "\n\n:gray[🎨 安妮亞在畫圖…]"


# ── 漏進正文的英文寫作規劃 ──
# 2026-10-23 VBA 驗收：回答開頭是一段英文的「Make sure to hit the main points…Keep it concise.」，
# 後面才是中文正文——模型把規劃寫進了回答。人設已經寫「不要把思考過程寫進回答」仍然發生，
# 照舊版教訓（提示詞擋不住、後處理才有效）在這裡剝掉。條件刻意窄：第一段幾乎全英文、不是程式碼／表格／
# 標題／清單、後面接著中文正文。使用者自己用英文問時後面也是英文，不會被剝。
_CJK_RE = re.compile(r"[一-鿿]")
PLAN_MIN_CHARS = 40
PLAN_MAX_CJK = 6
PLAN_MIN_ASCII_RATIO = 0.6
REST_MIN_CJK = 20


PLAN_MIN_WORDS = 8
_WORD_RE = re.compile(r"[A-Za-z]{2,}")


def _plan_shaped(para: str) -> bool:
    """像一段英文散文：夠長、大多是英文字母、有好幾個英文單字、幾乎沒有中文，且不是程式碼／表格／標題／清單。"""
    p = para.strip()
    if len(p) < PLAN_MIN_CHARS or p.startswith(("```", "#", "|", "-", "*", ">", "1.")):
        return False
    letters = sum(1 for c in p if c.isascii() and c.isalpha())
    return (letters / len(p) >= PLAN_MIN_ASCII_RATIO and len(_CJK_RE.findall(p)) <= PLAN_MAX_CJK
            and len(_WORD_RE.findall(p)) >= PLAN_MIN_WORDS)


# ── 漏進正文的中文內部判斷 ──
# 2026-10-24 Haiku 占星題：回答開頭「這是占星／算命類的問題，屬於沒有科學依據的領域。依照我的規則，這類問題
# 不需要升級，也不需要搜尋，我直接回答。」——把路由判斷寫進了正文。用語刻意窄（「系統不需要升級」這種正常
# 回答不能誤剝），只看第一段：短段整段拿掉，長段只拿掉那幾句。
_META_RE = re.compile(
    r"(?:依照|根據|按照)我的(?:規則|設定|指示)|(?:這類問題|這題|這個問題)[^。！？\n]{0,6}不(?:需要|用|必)升級"
    r"|escalate_to_deep|(?:不需要|不用|無需)(?:搜尋|上網)[^。！？\n]{0,12}(?:直接|我就)(?:回答|作答)")
META_PARA_MAX = 160
_SENTENCE_RE = re.compile(r"[^。！？!?]+[。！？!?]?")


def _drop_meta_sentences(para: str) -> str:
    return "".join(s for s in _SENTENCE_RE.findall(para) if not _META_RE.search(s)).strip()


def strip_leaked_plan(text: str, *, final: bool = True) -> str:
    t = text or ""
    lead = len(t) - len(t.lstrip())
    cut = t.find("\n\n", lead)
    if cut < 0:
        # 第一段還沒寫完：串流中看起來像規劃就先藏著（等下一段出現再判斷）；定稿時只拿掉內部判斷那幾句
        if not final and (_plan_shaped(t) or _META_RE.search(t)):
            return ""
        return (_drop_meta_sentences(t) or t) if _META_RE.search(t) else t
    first, rest = t[lead:cut], t[cut:].lstrip("\n")
    if _META_RE.search(first):
        if len(first.strip()) > META_PARA_MAX:
            return _drop_meta_sentences(first) + t[cut:]
        return rest if (rest.strip() or not final) else t
    if not _plan_shaped(first):
        return t
    if len(_CJK_RE.findall(rest)) >= REST_MIN_CJK:
        return rest
    return rest if not final else t      # 串流中後面還不夠判斷：先藏第一段；定稿仍不像中文回答就還原


# ── 自稱查證的措辭 ──
# 句子的偵測沿用舊版 utils/honesty.py（形狀是線上一個個補出來的），刪法這裡自己決定；舊版 Home 照舊用
# finalize_response，不受影響。2026-10-24 線上：天氣題「限制」標題底下空白——舊刪法把含「查到」的句子整句
# 刪到句號，「安妮亞查到的資料有點亂，…所以先講結論：明天很可能會下雨」連結論一起不見（本機同題 6 次：1 次
# 結論整段消失、1 次「這是今天查到的預報，明天可能會變」被刪）。改成：
# - 這回合真的查到資料（looked_up）：「查到」「查了」是實話，不刪；只刪「查證過」「核實過」這類過頭的說法。
# - 沒查：照樣刪，但只刪宣稱所在的那一小段（到逗號、冒號為止），同一句後面的內容留著。
# - 問句（「要不要我幫你查？」）、否定（「沒辦法幫你查」）是誠實話，不刪。
_CLAIM_VERB_RE = re.compile(r"查證過|確認過|核實過|查好|查過|查到|查了一下|查了|查查|盯了一下|盯了"
                            r"|(?:幫你|為你|替你)查")
_OVERCLAIM_RE = re.compile(r"查證過|確認過|核實過")
_CLAIM_LABEL_RE = re.compile(r"^[^\w\n]*(查詢結果|查證結果|查證資訊|查證完成|查詢完成|經查)")
_TRUE_LABELS = ("查詢結果", "查詢完成")      # 真的查過時，這兩個小標是實話
_ASK_RE = re.compile(r"[？?]|嗎|要不要|需不需要|想要我|需要我")
_NEG_RE = re.compile(r"沒|未|無法|不能|不會|不曾")
_CLAUSE_MARKS = "，,：:；;"
_SENTENCE_MARKS = "。！？!?"


def _fix_claim(m: re.Match, looked_up: bool) -> str:
    s = m.group(0)
    label = _CLAIM_LABEL_RE.match(s)
    if label:                                   # 小標「查證結果：」：只拿掉標籤，內容在後面、不受影響
        return s if looked_up and label.group(1) in _TRUE_LABELS else ""
    v = (_OVERCLAIM_RE if looked_up else _CLAIM_VERB_RE).search(s)
    if v is None or _ASK_RE.search(s) or _NEG_RE.search(s[:v.start()]):
        return s
    # 宣稱所在的那一小段：從前一個逗號／冒號之後（「結論：」這種開頭留著），到後面第一個標點為止
    start = max(s.rfind(c, 0, v.start()) for c in _CLAUSE_MARKS) + 1
    ends = [i for i in (s.find(c, v.end()) for c in _CLAUSE_MARKS + _SENTENCE_MARKS) if i >= 0]
    end = min(ends) + 1 if ends else len(s)
    rest = s[end:].lstrip()
    if not re.search(r"\w", rest):              # 這一句只有宣稱：整句拿掉（和舊版一樣，連句尾的 emoji、換行）
        return ""
    return s[:start] + rest


def strip_claims(text: str, *, looked_up: bool) -> str:
    """拿掉模型自稱查證的措辭（Claude 頁面用；規則見上）。"""
    if not text:
        return ""
    out = _CLAIM_SENTENCE_RE.sub(lambda m: _fix_claim(m, looked_up), text)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def display_text(buf: str, *, final: bool, looked_up: bool = False) -> str:
    """串流中／定稿時要顯示的文字（尚未做 markdown 修復，交給 render_ready）。
    looked_up：這回合真的查到資料（搜尋、讀網頁、氣象署、占星工具），回答說「查到」是實話。"""
    # U+FFFD（無法顯示的字元）：2026-10-23 實測 Opus 在正式回答前送出只有這個字的文字區塊，畫面上變成一個「�」
    t = (buf or "").replace("�", "")
    t = strip_leaked_plan(strip_claims(t, looked_up=looked_up), final=final)
    if not final and t:
        if t.count("```") % 2 == 1:
            # 畫到一半的 Mermaid 會變成紅色錯誤框（每次重畫都閃一次），整塊先藏起來，畫完才出現
            fences = list(_FENCE_LINE_RE.finditer(t))
            if fences and len(fences) % 2 == 1 and fences[-1].group(1).lower() == "mermaid":
                return t[:fences[-1].start()].rstrip() + MERMAID_DRAWING
            # 串流到一半的程式碼區塊：程式碼不需要誠實剝除，不藏尾巴；補上收尾，避免後文被吃進去
            return t + "\n```"
        ends = [m.end() for m in _SENTENCE_END_RE.finditer(t)]
        cut = ends[-1] if ends else 0
        if len(t) - cut <= HOLD_BACK_MAX:
            t = t[:cut]
    return t


def build_web_sources_footer(sources: list[dict], max_n: int = 6) -> str:
    """來源 footer 由系統產生，不靠模型（沿用 Home.py 的格式）。"""
    seen = set()
    lines = []
    for s in sources or []:
        url = (s.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        title = " ".join(((s.get("title") or url).strip()).split())
        if len(title) > 60:
            title = title[:60] + "…"
        lines.append(f"- [{title}]({url})")
        if len(lines) >= max_n:
            break
    return ("\n\n---\n**🌐 來源**\n" + "\n".join(lines)) if lines else ""


_MODE_BADGE = {
    "fast": ("⚡ 一般 · Haiku 5.5", "green"),
    "deep": ("💬 深思 · Opus 5.5", "blue"),
    "socratic": ("🧭 引導 · Opus 5.5", "violet"),
    "cwa": ("🌐 氣象署直答", "orange"),
}


def _badge(label: str, color: str) -> str:
    return f":{color}-badge[{(label or '').replace('[', '(').replace(']', ')')}]"


def badges_markdown(*, mode: str, web_calls: int = 0, elapsed_s: float | None = None,
                    escalated: bool = False) -> str:
    label, color = _MODE_BADGE.get(mode, _MODE_BADGE["deep"])
    items = [_badge(label, color)]
    if escalated:
        items.append(_badge("↑ 自動升級", "orange"))
    items.append(_badge(f"Web:{web_calls}" if web_calls else "Web:off", "violet" if web_calls else "gray"))
    if elapsed_s is not None:
        items.append(_badge(f"⏱ {elapsed_s:.1f}s", "gray"))
    return " ".join(items)
