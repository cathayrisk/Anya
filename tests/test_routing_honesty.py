# -*- coding: utf-8 -*-
"""路由誠實性測試（P0-1）。

背景：2026-09-03 線上測試發現，問「最近有地震嗎？有沒有颱風要來？」會留在 Fast、
零工具、憑記憶回答，還在開頭寫「安妮亞幫你查好了 🔍」——而正上方的系統 banner 同時
寫著「本回覆未經網路查證」。同一則訊息自相矛盾，且對外表現成查證過的事實。

這組測試守兩件事：
1. 災防詞與查證詞一定要把 mode 打到 General（工具只掛在那一邊）。
2. 一般閒聊／翻譯／摘要不可以被誤升級（免費層限流很緊，誤升級是有成本的）。

刻意不 import Home.py（會拉進 streamlit），改用與 tests/golden/run_golden.py 相同的
「從原始碼抽 regex」手法，維持秒級、離線、零費用。

跑法（專案根目錄）：
    python -m pytest tests/test_routing_honesty.py -v
    python tests/test_routing_honesty.py          # 不裝 pytest 也能跑
"""
from __future__ import annotations

import re
import pathlib

import pytest


def _find_root(start: pathlib.Path) -> pathlib.Path:
    """往上找到含 Home.py 的目錄當專案根（不寫死 parents[N]，測試組可能被搬位置）。"""
    for p in [start, *start.parents]:
        if (p / "Home.py").exists():
            return p
    return start.parents[min(1, len(start.parents) - 1)]


ROOT = _find_root(pathlib.Path(__file__).resolve().parent)
SRC = (ROOT / "Home.py").read_text(encoding="utf-8")


def _grab(name: str) -> re.Pattern:
    """從 Home.py 原始碼取出頂層 `NAME = re.compile(r"...")` 的樣式並編譯。

    樣式可能拆成多個相鄰 raw string，只認單行會整條抓不到 → 誤判成「沒有這個路由」。
    """
    m = re.search(
        rf'^{name} = re\.compile\(\s*((?:\s*r"[^"]*"\s*\|?)+),?\s*(?:re\.IGNORECASE,?\s*)?\)',
        SRC, re.M,
    )
    assert m, f"在 Home.py 抓不到 {name}——是不是改名或改寫法了？"
    return re.compile("".join(re.findall(r'r"([^"]*)"', m.group(1))), re.IGNORECASE)


HAZARD = _grab("HAZARD_HINT_RE")
VERIFY = _grab("VERIFY_HINT_RE")
GENERAL = _grab("GENERAL_HINT_RE")


# ── 必須升級 General：工具只在那一邊，落 Fast 就只能憑記憶編 ──────────────────
ESCALATE_CASES = [
    ("最近台灣有發生地震嗎？有沒有颱風要來？", HAZARD),      # 線上實測的原始失敗案例
    ("請你實際查證後回答：氣象署有沒有發布地震報告？", HAZARD),  # 明講查證仍未升級的案例
    ("剛剛有感覺到震度嗎", HAZARD),
    ("現在有發布豪雨特報嗎", HAZARD),
    ("台北有海上警報嗎", HAZARD),
    ("這個數字幫我查證一下", VERIFY),
    ("這則新聞可以核實嗎", VERIFY),
    ("幫我做事實查核", VERIFY),
]

# ── 不可誤升級：這些問題 Fast 答得好，升級只是白燒配額 ───────────────────────
STAY_FAST_CASES = [
    "幫我把這段翻成英文",
    "今天天氣真好",            # 「天氣」刻意不放進 HAZARD，就是為了守住這一條
    "推薦幾間台北的咖啡店",
    "把這篇文章做成摘要",
    "1+1 等於多少",
    "幫我想一個產品名字",
]


@pytest.mark.parametrize("text,rx", ESCALATE_CASES)
def test_hazard_and_verify_escalate(text, rx):
    assert rx.search(text), f"「{text}」應該升級 General，卻沒有命中"


@pytest.mark.parametrize("text", STAY_FAST_CASES)
def test_no_false_escalation(text):
    hit = HAZARD.search(text) or VERIFY.search(text) or GENERAL.search(text)
    assert not hit, f"「{text}」不該升級，卻被「{hit.group(0) if hit else ''}」誤中"


def test_routing_branches_wired_in():
    """regex 定義了但沒接進 mode 判斷等於沒改——確認兩個分支都真的在路由鏈上。"""
    assert 'escalate_reason = "hazard_hint"' in SRC, "HAZARD_HINT_RE 沒接進 mode 判斷"
    assert 'escalate_reason = "verify_hint"' in SRC, "VERIFY_HINT_RE 沒接進 mode 判斷"
    assert "HAZARD_HINT_RE.search(user_text" in SRC
    assert "VERIFY_HINT_RE.search(user_text" in SRC


def test_fast_prompt_forbids_claiming_verification():
    """Fast prompt 必須有「禁止宣稱查證動作」硬規則，且列出實際踩過的措辭。"""
    assert "禁止宣稱查證動作" in SRC, "Fast prompt 少了禁止宣稱查證的硬規則"
    for phrase in ("查好了", "我查了一下", "查詢結果顯示", "根據最新資料"):
        assert phrase in SRC, f"禁用措辭清單少了「{phrase}」"


# ── MODEL_CHAINS 衛生：不准有已知下架的模型 ID ────────────────────────────────────
# 2026-09-04 用使用者的 key 實測（tools/list-google-models.py + 單次 smoke call）：
# 這些 ID 回 404「no longer available to new users」。留在鏈裡會被 _mark_model_dead 跳過，
# 但每次都先白浪費一次呼叫（原作者 Home.py:160 註解即警告）。
DEAD_MODEL_IDS = ("gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro")


def _model_chains_block() -> str:
    m = re.search(r"^MODEL_CHAINS[^\n]*=\s*\{(.*?)^\}", SRC, re.S | re.M)
    assert m, "找不到 MODEL_CHAINS 定義"
    return m.group(1)


@pytest.mark.parametrize("dead", DEAD_MODEL_IDS)
def test_no_dead_model_in_chains(dead):
    assert f'"{dead}"' not in _model_chains_block(), f"MODEL_CHAINS 仍含已下架的 {dead}"


def _chain(name: str) -> list[str]:
    m = re.search(rf'"{name}":\s*\(([^)]*)\)', _model_chains_block())
    assert m, f"找不到 {name} 鏈"
    return [e.strip() for e in m.group(1).split(",") if e.strip()]


# ── 2026-09-04 AI Studio dashboard 實證：Gemini 3.x flash 非 lite 全系列每天 20 次 + 5 RPM。
# 28 天用量：3.5-flash 30/20、3.6-flash 27/20——當自動備援一波就燒光，之後整天不在。
# 自動鏈一律不放；它們只給 PREMIUM_MODEL 的單次高價值呼叫。
RPD20_MODEL_IDS = ("gemini-3-flash-preview", "gemini-3.5-flash", "gemini-3.6-flash",
                   "gemini-3.7-flash", "gemini-3.8-flash", "gemini-flash-latest")


@pytest.mark.parametrize("rpd", RPD20_MODEL_IDS)
def test_no_rpd20_model_in_any_chain(rpd):
    assert f'"{rpd}"' not in _model_chains_block(), f"MODEL_CHAINS 仍含每天只有 20 次的 {rpd}"


def test_chains_after_2026_09_04_reorg():
    assert _chain("fast") == ['FAST_MODEL', '"gemini-3.5-flash-lite"']
    assert _chain("general") == ['GENERAL_MODEL', 'BACKGROUND_MODEL', '"gemini-3.5-flash-lite"']
    assert _chain("chore") == ['CHORE_MODEL', 'BACKGROUND_MODEL']
    # socratic／research 刻意不含 lite（實測會漏答案／拉低報告品質）：兩顆 gemma 撞牆就等分鐘窗
    for name in ("socratic", "research"):
        c = _chain(name)
        assert c == ['GENERAL_MODEL', 'BACKGROUND_MODEL'], c
        assert not any("lite" in e for e in c)


# ── 背景雜活搬離 gemma（2026-09-04）────────────────────────────────────────────
# dashboard 實測：兩顆 gemma 各只有 16K input token/分（全表最緊），而雜活跟 General 主腦
# 搶的正是這個分鐘窗——主腦 31b 溢出要退 26b 時，26b 常正被雜活佔用。
def test_chore_model_is_lite_not_gemma():
    m = re.search(r'^CHORE_MODEL\s*=\s*"([^"]+)"', SRC, re.M)
    assert m, "找不到 CHORE_MODEL"
    assert "lite" in m.group(1) and "gemma" not in m.group(1), m.group(1)


def test_no_stale_background_llm_getter():
    """舊 getter 名稱殘留 = 有呼叫點沒搬到雜活鏈。"""
    assert "get_background_llm" not in SRC


def test_chore_calls_carry_purpose():
    """沒帶 purpose 的 invoke_with_backoff 不會換模型：429／503 只能把退避階梯燒完才放棄，
    摘要那條會直接卡住整個回合（2026-09-04 3.5-flash-lite 連續 21 次 503 的教訓）。"""
    for m in re.finditer(r"invoke_with_backoff\(lambda: get_chore_llm\(\)", SRC):
        tail = SRC[m.start(): m.start() + 2000]
        assert 'purpose="chore"' in tail.split("\n    except")[0], "雜活呼叫少了 purpose=\"chore\""
    assert SRC.count('purpose="chore"') == 2


# ── Fix C：Fast 後處理層剝除「自稱已查證」措辭（utils/honesty.py）────────────────
# 2026-09-03 線上驗證 V3：Fast、Web:off、banner 寫「未經網路查證」，模型仍寫「幫你查好了！」。
# prompt 禁令對 flash-lite 無效 → 改在系統握有 web_happened == False 事實的後處理層做。
STRIP_POS = [  # (模型原文, 期望輸出)
    ("安妮亞幫你查好了！🥜\n\n台灣目前的法定基本工資月薪為 27,470 元。", "台灣目前的法定基本工資月薪為 27,470 元。"),  # V3 實例
    ("哇～安妮亞幫你查好了！🔍\n\n根據中央氣象署截至今天的資訊：", "根據中央氣象署截至今天的資訊："),            # T4 實例
    ("WakuWaku! 安妮亞幫你查好囉！🥜\n\n目前沒有颱風警報。", "目前沒有颱風警報。"),                              # V1 開場句型
    ("🔍 查證結果\n\n1. 颱風消息：目前無警報", "1. 颱風消息：目前無警報"),                                     # 小標
    ("安妮亞查了一下，目前無颱風。\n\n說明如下。", "說明如下。"),
    ("查詢結果：目前無警報。", "目前無警報。"),                                                                # 剝標籤、留斷言
    ("", ""),
]
STRIP_NEG = [  # 不可誤傷
    "這句英文的文法我幫你檢查了一下，沒問題。",        # 「檢查」不是查證宣稱
    "建議可至勞動部官網查詢最新公告。",                # 建議使用者去查
    "安妮亞不太確定，這個數字沒有查證過喔（未查證）。",  # 誠實的否定
    "我無法查證這個數字，請以官方為準。",
    "這件事需要調查了解後才能回答。",
    "根據既有知識，基本工資約為 2 萬 7 千元。",
    "Hello! 這是一般回答。",
]


def _strip():
    import sys as _s
    if str(ROOT) not in _s.path:
        _s.path.insert(0, str(ROOT))
    from utils.honesty import strip_false_verification_claims
    return strip_false_verification_claims


@pytest.mark.parametrize("text,want", STRIP_POS)
def test_strip_false_verification_claims(text, want):
    assert _strip()(text) == want


@pytest.mark.parametrize("text", STRIP_NEG)
def test_strip_does_not_touch_honest_text(text):
    assert _strip()(text) == text.strip()


def test_fix_c_wired_into_fast_postprocess():
    """模組寫了但沒接進 Fast 收尾等於沒改：確認 import 與呼叫都在，且 banner 不再只看年份/%。"""
    assert "from utils.honesty import strip_false_verification_claims" in SRC
    assert "fast_text = strip_false_verification_claims(fast_text)" in SRC
    assert 're.search(r"20\\d{2}|％|%", fast_text)' not in SRC, "banner 觸發條件仍只看年份/%"


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    fails = 0
    print("── 應升級 General ──")
    for text, rx in ESCALATE_CASES:
        ok = bool(rx.search(text))
        fails += not ok
        print(f"  {'OK  ' if ok else 'FAIL'} {text}")
    print("── 應保持 Fast ──")
    for text in STAY_FAST_CASES:
        hit = HAZARD.search(text) or VERIFY.search(text) or GENERAL.search(text)
        fails += bool(hit)
        print(f"  {'OK  ' if not hit else 'FAIL'} {text}"
              + (f"   ← 誤中「{hit.group(0)}」" if hit else ""))
    print("── 接線與 prompt ──")
    for name, fn in (("路由分支已接上", test_routing_branches_wired_in),
                     ("Fast prompt 禁止宣稱查證", test_fast_prompt_forbids_claiming_verification)):
        try:
            fn()
            print(f"  OK   {name}")
        except AssertionError as e:
            fails += 1
            print(f"  FAIL {name}：{e}")
    print(f"\n{'全部通過' if not fails else f'{fails} 項失敗'}")
    sys.exit(1 if fails else 0)
