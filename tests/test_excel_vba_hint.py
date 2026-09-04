# -*- coding: utf-8 -*-
"""excel-vba skill 的路由與索引測試（純函式，零 LLM 呼叫）。

為什麼要這組測試：SKILL_HINT_RES 命中會強制升級 General 模式，而免費層非 lite 模型
只有 20 RPD——誤升級的代價是使用者當天問不到第 21 句話。反例集刻意塞滿台灣金融
語境的陷阱詞（宏碁／宏達電／宏觀／macro outlook），改 pattern 時這裡必須全綠。
"""
import re

import pytest

from Home import SKILL_HINT_RES, SKILL_SUGGEST_RES, match_skill_hint, suggest_unused_skills

HINT_RE: re.Pattern = SKILL_HINT_RES["excel-vba"]

# ── 應該命中（強制升級 General，模型才拿得到 load_skill）──────────────────────
SHOULD_MATCH = [
    "這段 VBA 為什麼跑不動",
    "我的巨集會當掉",
    "幫我看 Excel 巨集哪裡有問題",
    "幫我寫一個巨集把 A 欄重複值標紅",
    "risk_report.xlsm 打不開",
    "VBA",
    "Excel macro crashes on startup",
    "Please review this macro code",
    "這個 macro 跑不動",
    "the macro in Sheet1 module fails",
    "可以幫我改這個 macro 的程式碼嗎",
]

# ── 不該命中（誤升級＝燒 RPD 配額）────────────────────────────────────────────
SHOULD_NOT_MATCH = [
    "宏碁財報有錯誤",           # 中國用語「宏」刻意不收，否則台股公司名整片誤觸
    "宏達電營收有問題",
    "宏觀經濟資料處理很慢",
    "總體經濟很慢",
    "Excel 公式怎麼寫",         # 公式不是 VBA
    "幫我做樞紐分析表",
    "macroeconomic outlook",
    "Taiwan macro outlook",     # 金融語境的 macro，不是巨集
    "macro risk review for Taiwan banks",
    "fix the macro forecast error",
    "macro data is failing to load",
    "Excel 的 VLOOKUP 一直錯誤",
    "請幫我 debug 這個 Python 程式",
]


@pytest.mark.parametrize("text", SHOULD_MATCH)
def test_hint_matches(text):
    assert HINT_RE.search(text), f"應命中卻沒中：{text}"


@pytest.mark.parametrize("text", SHOULD_NOT_MATCH)
def test_hint_does_not_match(text):
    assert not HINT_RE.search(text), f"誤命中（會浪費 RPD 配額）：{text}"


@pytest.mark.parametrize("text", SHOULD_MATCH)
def test_match_skill_hint_returns_excel_vba(text):
    """經過 match_skill_hint 的「首個命中生效」順序後，仍要選到 excel-vba。"""
    assert match_skill_hint(text) == "excel-vba", f"被前面的 skill 搶走：{text}"


def test_suggest_res_entry_exists():
    assert "excel-vba" in SKILL_SUGGEST_RES


@pytest.mark.parametrize("text", ["幫我寫一個巨集", "這個 VBA 有問題", "excel 自動化"])
def test_suggest_covers_vba(text):
    """回合後建議：available 顯式傳入，避免相依於執行環境掃不掃得到 skills/。"""
    names = suggest_unused_skills(text, used=[], available={"excel-vba": {"description": "x"}})
    assert "excel-vba" in names


def test_skill_file_is_registered():
    """SKILL.md 的 frontmatter name 必須與白名單 key 一致，否則永遠載不到。"""
    from utils.skill_loader import SKILL_WHITELIST, discover_skills

    assert "excel-vba" in SKILL_WHITELIST
    assert "excel-vba" in discover_skills(), "skills/excel-vba/SKILL.md 掃不到或 name 對不上"
