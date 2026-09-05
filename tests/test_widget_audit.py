# -*- coding: utf-8 -*-
"""widget 狀態流失稽核（utils/widget_audit.py，2026-09-05 P1-1）。

起因 T20：模型自己手寫閃卡 HTML（沒用 `widget_flashcards` 模板），
翻到第 3 張後送下一則訊息就重置回第 1 張，而且完全沒有訊號。

這組測試守兩件事：判定本身不要誤傷真模板，以及 Home.py 真的接在**渲染之前**
（接在之後就沒有補救機會了——widget 名額一被佔用，第二次呼叫會被拒）。

跑法：python -m pytest tests/test_widget_audit.py -v
"""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = next(p for p in [pathlib.Path(__file__).resolve().parent, *pathlib.Path(__file__).resolve().parents]
            if (p / "Home.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.widget_audit import (  # noqa: E402
    TEMPLATE_ROOT_IDS, audit_widget_html, detect_template,
)
from widget_templates import WIDGET_TEMPLATES  # noqa: E402

SRC = (ROOT / "Home.py").read_text(encoding="utf-8")

# T20 實際發生的長相：自製、有互動、完全沒有 AnyaState
HANDROLLED = """
<div id="anya-flashcards-container"><div class="card">Q</div>
<button class="next">下一張</button></div>
<script>(function(){let pos=0;
document.querySelector(".next").addEventListener("click", function(){ pos++; render(); });
})();</script>
"""
STATIC = '<div><h3>營收占比</h3><svg width="200" height="100"><rect width="50" height="80"/></svg></div>'


# ── 判定 ────────────────────────────────────────────────────────────────────
def test_the_t20_handrolled_widget_is_caught():
    a = audit_widget_html(HANDROLLED)
    assert a.template is None and a.interactive and not a.has_state
    assert a.will_lose_state


def test_static_widget_is_not_flagged():
    """純展示的東西流失不了狀態，擋它只是白白多一次工具呼叫。"""
    a = audit_widget_html(STATIC)
    assert not a.interactive and not a.will_lose_state


@pytest.mark.parametrize("name", sorted(WIDGET_TEMPLATES))
def test_every_real_template_is_recognised_and_passes(name):
    """⚠️ 最重要的一條：誤判模板會讓每個正常 widget 都被擋一次。
    `widget_calculator` / `widget_natal_chart` **刻意沒接狀態**（值是當下輸入，
    還原沒有意義），所以認出是模板就要直接放行，不看有沒有 AnyaState。"""
    html = WIDGET_TEMPLATES[name]["content"]
    a = audit_widget_html(html)
    assert a.template == name, f"{name} 沒被認出來 → 會被誤擋"
    assert not a.will_lose_state


def test_template_ids_match_the_actual_templates():
    """根 id 改名而這裡沒跟著改，稽核會靜默失效（每個模板都變成「自製」）。"""
    for root_id, name in TEMPLATE_ROOT_IDS.items():
        assert name in WIDGET_TEMPLATES, name
        assert f"#{root_id}" in WIDGET_TEMPLATES[name]["content"], root_id


def test_template_detected_by_structural_id_not_a_comment():
    """用根 id 當指紋是刻意的：它是 CSS 與 getElementById 都在用的東西，
    拔掉模板就壞了；註解標記則可能在改寫時被順手刪掉。"""
    assert detect_template('<div id="anya-fc">…</div>') == "widget_flashcards"
    assert detect_template("<style>#anya-cmx{color:red}</style>") == "widget_comparison_matrix"
    assert detect_template("<!-- widget_flashcards -->") is None, "註解不算指紋"


@pytest.mark.parametrize("html", [
    '<div id="anya-flashcards-container">',      # T20 自製元件真的用過的名字
    "<style>#anya-fcx{}</style>",
    "<style>#anya-calculator{}</style>",
])
def test_similar_looking_ids_do_not_count_as_templates(html):
    """裸的子字串比對會讓這些誤中模板，於是真正該擋的元件被放行——
    誤放行比誤擋更糟，因為它是靜默的。"""
    assert detect_template(html) is None, html


def test_handrolled_but_state_wired_is_allowed():
    """自製但有自己接 AnyaState → 沒問題，不該擋。"""
    a = audit_widget_html(HANDROLLED + "<script>AnyaState.save({pos:1});</script>")
    assert a.has_state and not a.will_lose_state


# ── Home.py 接線 ────────────────────────────────────────────────────────────
def test_audit_runs_before_render_so_a_retry_is_still_possible():
    """接在渲染之後就沒救了：`rt["widget"]` 一被填上，第二次呼叫會被
    「本回合已生成過互動元件」擋掉，模型再也沒機會改用模板。"""
    i = SRC.index("_wa = WAUDIT.audit_widget_html(h)")
    assert i < SRC.index('rt["widget"] = {"title"')
    assert i < SRC.index("render_widget_html(h, height=hh")


def test_blocks_only_once_per_turn():
    """判定必然是啟發式的。擋兩次以上的風險（迴圈、使用者完全拿不到元件）
    比「狀態流失」這個不便本身更大——這不是安全性問題，力道要相稱。"""
    i = SRC.index("_wa = WAUDIT.audit_widget_html(h)")
    region = SRC[i: i + 700]
    assert 'not rt.get("widget_state_nudged")' in region
    assert 'rt["widget_state_nudged"] = True' in region
    assert 'rt["widget_state_nudged"] = False' in SRC, "每回合要重置，否則只擋得到第一輪"


def test_the_nudge_tells_the_model_what_to_do():
    from utils.widget_audit import RETRY_HINT
    assert "load_skill" in RETRY_HINT and "widget_" in RETRY_HINT
    assert "再呼叫一次" in RETRY_HINT, "要講明第二次會放行，否則模型可能放棄做元件"


def test_audit_is_observable():
    """比照第 3 步的 shadow log：要先知道模型多常自製，才知道這個力道要不要調。"""
    assert "[widget_audit]" in SRC, "stdout 沒記就看不到真實比例"
    assert "🔧 [dev] widget audit" in SRC


def test_widget_rules_mention_state():
    from widget_templates import WIDGET_RULES
    assert "AnyaState" in WIDGET_RULES and "重置" in WIDGET_RULES


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
