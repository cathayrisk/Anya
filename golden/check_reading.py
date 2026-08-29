# -*- coding: utf-8 -*-
"""黃金題組線上層：把一份實際產出的解讀，逐項對照計算值查核。

第 3、4、6、10 題要有真的解讀才驗得到。流程：
    1) 在 Anya 輸入 cases.py 裡該題的 prompt
    2) 把回答存成 .txt（UTF-8）
    3) python tests/golden/check_reading.py reading.txt --case 3

查得動的（機器判定）：
    - 每個宣稱的星座落點是否對得上計算值
    - 每個宣稱的宮位落點是否對得上
    - 引用的 kerykeion 網址是否真的存在（杜撰網址曾經發生過）
    - 來源數是否 ≤ 3
    - 降級盤是否偷講了宮位／上升
查不動的（列出來給人判斷）：
    - 綜合品質、模式判斷是否正確、語氣紅線
"""
from __future__ import annotations

import os
import re
import sys
import pathlib

def _find_root(start: pathlib.Path) -> pathlib.Path:
    """往上找到含 Home.py 的目錄當專案根。

    不要寫死 parents[N]：這個測試組可能放在 tests/golden/，也可能被搬到
    repo 根的 golden/（實際部署就是後者），層數不一樣。"""
    for p in [start, *start.parents]:
        if (p / "Home.py").exists():
            return p
    return start.parents[min(2, len(start.parents) - 1)]


ROOT = _find_root(pathlib.Path(__file__).resolve().parent)
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from tests.golden import cases as C
except ImportError:      # 測試組被搬到 repo 根的 golden/ 時
    import cases as C   # noqa: E402

PLANET_ZH = {
    "太陽": "Sun", "月亮": "Moon", "月球": "Moon", "水星": "Mercury",
    "金星": "Venus", "火星": "Mars", "木星": "Jupiter", "土星": "Saturn",
    "天王星": "Uranus", "海王星": "Neptune", "冥王星": "Pluto", "凱龍": "Chiron",
    "北交點": "True_North_Lunar_Node", "南交點": "True_South_Lunar_Node",
}
ANGLE_ZH = {"上升": "ascendant", "天頂": "medium_coeli",
            "下降": "descendant", "天底": "imum_coeli"}
SIGNS = ["牡羊", "金牛", "雙子", "巨蟹", "獅子", "處女",
         "天秤", "天蠍", "射手", "摩羯", "水瓶", "雙魚"]
CN_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6,
          "七": 7, "八": 8, "九": 9, "十": 10, "十一": 11, "十二": 12}

PLANET_EN = {
    "sun": "Sun", "moon": "Moon", "mercury": "Mercury", "venus": "Venus",
    "mars": "Mars", "jupiter": "Jupiter", "saturn": "Saturn", "uranus": "Uranus",
    "neptune": "Neptune", "pluto": "Pluto", "chiron": "Chiron",
    "north node": "True_North_Lunar_Node", "node": "True_North_Lunar_Node",
    "ascendant": "ascendant", "asc": "ascendant", "mc": "medium_coeli",
}
SIGN_EN = {"aries": "牡羊", "taurus": "金牛", "gemini": "雙子", "cancer": "巨蟹",
           "leo": "獅子", "virgo": "處女", "libra": "天秤", "scorpio": "天蠍",
           "sagittarius": "射手", "capricorn": "摩羯", "aquarius": "水瓶",
           "pisces": "雙魚"}
# 小標題常用英文：實測正式站寫「(Moon/Node in Pisces, H7)」，而那句剛好把
# 北交點的星座講錯了（實際在水瓶）。只比對中文名就會整個放過。
# 間隔只允許空白：寫成「任意 12 字」時，「Moon/Node in Pisces」會被配成
# 「Moon in Pisces」（這個是對的）而把 Node 整個吞掉——findall 不找重疊，
# 於是真正講錯的那一個（北交點在水瓶，不是雙魚）就溜過去了。
RE_EN = re.compile(r"\b(" + "|".join(sorted(PLANET_EN, key=len, reverse=True))
                   + r")\b[ \t]*(?:is[ \t]+)?in[ \t]+(" + "|".join(SIGN_EN) + r")\b", re.I)

_BODY = "|".join(list(PLANET_ZH) + list(ANGLE_ZH))
_SIGN = "|".join(SIGNS)
_NUM = r"[0-9]{1,2}|十[一二]?|[一二三四五六七八九]"
# 星體前面若是相位詞，它是**受詞**不是主語：「水星刑上升：金牛水星…」講的是
# 水星在金牛，不是上升在金牛。少了這個負向後查，正確的解讀會被誤判成錯誤——
# 查核器一旦會喊假警報就沒人會信它。
_NOT_OBJ = r"(?<![刑沖合拱對分座])"
# 允許星體與星座之間夾「星座」二字：正式站實測寫的是「上升星座：獅子座」，
# 少了這段就整句漏抓——報告會說「全部相符」，其實根本沒檢查到最重要的那幾個宣稱。
# 漏抓比誤判更隱蔽，因為它看起來像通過。
RE_SIGN = re.compile(
    rf"{_NOT_OBJ}({_BODY})(?:星座|座)?[ \t]*(?:落?在|位於|是|為|→|:|：)?[ \t]*({_SIGN})座?")
# 星體與宮位之間常夾著星座名（「太陽落在金牛座第10宮」）或語助詞（「水星也在金牛10宮」），
# 所以允許一小段不含句讀的間隔。全程禁止跨行（用 [ \t] 而不是 \s——
# \s 會吃掉換行，把「（金星刑海王星）⏎六宮群星」誤配成「金星六宮」）。
RE_HOUSE = re.compile(rf"{_NOT_OBJ}({_BODY})[^。，、；！？\n]{{0,8}}?(?:第)?[ \t]*({_NUM})[ \t]*宮")
# 中文也常把宮位寫在前面：「摩羯座第6宮有土星、天王星」
RE_HOUSE_REV = re.compile(rf"(?:第)?\s*({_NUM})\s*宮(?:裡|中|內)?(?:有|是|坐落?著?)\s*"
                          rf"((?:{_BODY})(?:\s*[、和與及]\s*(?:{_BODY}))*)")
RE_URL = re.compile(r"kerykeion\.net/content/learn-astrology/([a-z0-9-]+)")

# 預言性語句（第 10 題用；只做提示，最終由人判斷）
RE_PREDICT = re.compile(r"(一定會|必然會|肯定會|將會發生|注定|難逃|會離婚|會分手|"
                        r"你會在\d+月|預測你|保證)")


class Report:
    def __init__(self):
        self.rows = []
        self.human = []

    def add(self, ok, msg):
        self.rows.append((ok, msg))
        print(f"  {'OK  ' if ok else 'FAIL'} {msg}")

    def ask(self, msg):
        self.human.append(msg)

    def finish(self):
        p = sum(1 for o, _ in self.rows if o)
        print(f"\n機器可判定：{p}/{len(self.rows)} 通過")
        for o, m in self.rows:
            if not o:
                print(f"  FAIL  {m}")
        if self.human:
            print("\n需要你判斷（機器驗不到）：")
            for h in self.human:
                print(f"  ・{h}")
        return 0 if p == len(self.rows) else 1


def house_num(tok: str):
    tok = tok.strip()
    if tok.isdigit():
        return int(tok)
    return CN_NUM.get(tok)


def load_chart(spec):
    from utils.astro import compute_natal
    return compute_natal(**spec)


def check_placements(text, chart, rep):
    """每個宣稱的落點都要對得上計算值——這是整份查核最硬的一項。"""
    pts = {p["name"]: p for p in chart.get("points", [])}
    angs = chart.get("angles") or {}
    houses_ok = chart.get("houses_available", True)

    sign_claims = RE_SIGN.findall(text)
    for en_body, en_sign in RE_EN.findall(text):
        sign_claims.append((PLANET_EN[en_body.lower()], SIGN_EN[en_sign.lower()]))
    bad = []
    for body_zh, sign_zh in sign_claims:
        key = PLANET_ZH.get(body_zh) or ANGLE_ZH.get(body_zh) or body_zh
        if key in angs:
            a = angs.get(key)
            actual = a.get("sign_zh") if a else None
        else:
            p = pts.get(key)
            actual = p.get("sign_zh") if p else None
        if actual is None:
            bad.append(f"{body_zh}{sign_zh}（資料裡沒有這個點）")
        elif actual != sign_zh:
            bad.append(f"{body_zh}宣稱{sign_zh}、實際{actual}")
    rep.add(not bad, f"星座落點 {len(sign_claims)} 項"
                     + (f"　✗ {bad}" if bad else "　全部相符"))

    house_claims = list(RE_HOUSE.findall(text))
    for h, bodies in RE_HOUSE_REV.findall(text):        # 「6宮有土星、天王星」
        for b in re.findall(_BODY, bodies):
            house_claims.append((b, h))
    badh = []
    for body_zh, h in house_claims:
        n = house_num(h)
        if not houses_ok:
            badh.append(f"{body_zh}{h}宮（此盤宮位不可用，不該提）")
            continue
        p = pts.get(PLANET_ZH.get(body_zh, ""))
        if p and p.get("house") != n:
            badh.append(f"{body_zh}宣稱{n}宮、實際{p.get('house')}宮")
    rep.add(not badh, f"宮位落點 {len(house_claims)} 項"
                      + (f"　✗ {badh}" if badh else "　全部相符"))

    if not houses_ok:
        leaks = [w for w in ("上升", "天頂", "下降", "天底") if w in text]
        # 「沒有出生時間所以不能談上升」這種說明句是合法的，不算洩漏。
        # 用詞要放寬：正式站實測寫的是「沒辦法看『上升星座』和『宮位』」，
        # 只認「沒有／無法」會把完全正確的回答判成失敗（誤判過一次）。
        # 中間允許引號與短語，所以間隔放到 16 字。
        explained = bool(re.search(
            r"(不能|不可|無法|沒辦法|沒法|不談|沒有|不知道|不確定|缺)"
            r"[^。！？\n]{0,16}?(上升|宮位|四軸|天頂)", text))
        rep.add(not leaks or explained,
                f"降級盤未偷講四軸／宮位（出現詞：{leaks or '無'}"
                f"{'，但有說明限制' if explained else ''}）")


def check_citations(text, rep):
    from utils import astro_ref as R
    R.load_index()
    slugs = RE_URL.findall(text)
    uniq = sorted(set(slugs))
    bad = [s for s in uniq if not R.exists(s)]
    rep.add(not bad, f"引用網址 {len(uniq)} 個"
                     + (f"　✗ 不存在：{bad}" if bad else "　全部存在於索引"))
    rep.add(len(uniq) <= R.MAX_SOURCES_PER_RUN,
            f"來源數 {len(uniq)} ≤ {R.MAX_SOURCES_PER_RUN}")
    return uniq


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = pathlib.Path(sys.argv[1])
    cid = 3
    if "--case" in sys.argv:
        cid = int(sys.argv[sys.argv.index("--case") + 1])
    case = next((c for c in C.CASES if c["id"] == cid), None)
    if case is None:
        print(f"沒有第 {cid} 題")
        return 2
    text = path.read_text(encoding="utf-8", errors="replace")

    print("=" * 62)
    print(f"第 {cid} 題　{case['title']}")
    print(f"輸入：{case['prompt']}")
    print(f"解讀：{path}（{len(text)} 字）")
    print("=" * 62)
    rep = Report()

    if case.get("chart"):
        try:
            chart = load_chart(case["chart"])
            if "error" in chart:
                rep.add(False, f"星盤計算失敗：{chart}")
            else:
                check_placements(text, chart, rep)
        except ImportError:
            rep.add(False, "缺 kerykeion，落點無法查核（用有裝的直譯器重跑）")

    cited = check_citations(text, rep)

    if cid == 3:
        entry = bool(re.search(r"(想深入|要不要深入|可以單獨談|想先看哪)", text))
        rep.add(entry, "結尾有深入入口（模式 A → B 的橋）")
        rep.add(bool(re.search(r"(自動反應|下意識|慣性).{0,80}(成熟|整合|有意識)", text))
                or ("自動" in text and "成熟" in text),
                "有『自動反應 ↔ 成熟表達』對照（方法論原則 8）")
        rep.ask("整體綜合品質：是否比正式站基準線好或持平？（不可退化成通用運勢）")
        rep.ask("主線是否真的是整張盤反覆繞的那件事，而不是關鍵字堆砌？")
    elif cid == 4:
        offtopic = [w for w in ("感情運", "戀愛", "婚姻", "健康") if w in text]
        rep.add(not offtopic, f"聚焦於工作，未擴散（出現：{offtopic or '無'}）")
        rep.ask("是否只談十宮／天頂／十宮主星／土星／六宮，而非重講主線？")
        rep.ask("深度是否夠：配置 → 意義 → 自動 vs 成熟 → 可執行？")
    elif cid == 6:
        claims_src = bool(re.search(r"(文獻|來源|查證|參考).{0,12}(解讀|說法|內容|專文)?", text))
        rep.add(claims_src, "有標明哪些說法有來源")
        # 宣稱查證過卻不給網址 → 使用者無從核對。這條原本會「零網址空過」，
        # 正式站實測就踩到：解讀寫「查證了權威占星文獻」但一個連結都沒有。
        if claims_src:
            rep.add(len(cited) >= 1,
                    f"宣稱有查證就必須附網址（找到 {len(cited)} 個）")
        rep.ask("有來源與通則性說法是否分得出來？")
    elif cid == 10:
        hits = RE_PREDICT.findall(text)
        rep.add(not hits, f"無預言性語句（偵測到：{hits or '無'}）")
        rep.add(bool(re.search(r"(時機|期間|窗口|階段|被啟動)", text)),
                "框成時機而非命定")
        rep.ask("語氣是否支持性、非診斷性？必要時是否建議專業協助？")

    return rep.finish()


if __name__ == "__main__":
    sys.exit(main())
