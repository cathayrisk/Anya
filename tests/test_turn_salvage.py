# -*- coding: utf-8 -*-
"""回合收尾補救（utils/turn_salvage.py，2026-09-05 使用者截圖的兩個 bug）。

跑法：python -m pytest tests/test_turn_salvage.py -v
"""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = next(p for p in [pathlib.Path(__file__).resolve().parent, *pathlib.Path(__file__).resolve().parents]
            if (p / "Home.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.turn_salvage import strip_orphan_widget_source, describe_completed_work  # noqa: E402

SRC = (ROOT / "Home.py").read_text(encoding="utf-8")

# 使用者截圖的原文形狀
LEAKED = """安妮亞幫你整理了這兩位官員的觀點差異：

```html
<!-- 互動比較矩陣 -->
<script>
const DATA = {
  title: "聯準會官員觀點比較：Waller vs. Warsh",
  items: ["Christopher Waller (Governor)", "Kevin Warsh (Chairman)"]
};
</script>
```

(請點選上方的互動表格查看維度詳情與優勝方標記)

## 3. 決策觀察報告
這場博弈的結論是：市場目前的預期已進入「等待驗證」模式。"""


# ── Bug B：widget 原始碼外洩 ─────────────────────────────────────────────────
def test_strips_leaked_widget_source_when_no_widget():
    out, changed = strip_orphan_widget_source(LEAKED, widget_created=False)
    assert changed
    assert "<script>" not in out and "const DATA" not in out
    assert "點選上方的互動表格" not in out
    # 正文必須留著
    assert "決策觀察報告" in out and "等待驗證" in out


def test_keeps_everything_when_widget_really_created():
    """widget 真的建成時，「上方是互動比較矩陣」是正確敘述，不可誤刪。"""
    txt = "上方是互動比較矩陣，點擊膠囊可以切換維度喔！\n\n## 摘要\n兩人分歧在於資料解讀。"
    out, changed = strip_orphan_widget_source(txt, widget_created=True)
    assert not changed and out == txt


@pytest.mark.parametrize("txt", [
    "一般回答，沒有任何元件相關內容。",
    "```vba\nSub A()\n  Rows(1).Delete\nEnd Sub\n```\n這段程式碼沒問題。",  # 一般程式碼不可誤刪
    "",
])
def test_does_not_touch_normal_answers(txt):
    out, changed = strip_orphan_widget_source(txt, widget_created=False)
    assert not changed and out == txt


# ── Bug B2：連原始碼都沒有，只有一句口頭宣稱（2026-09-05 線上抓到）──────────
# 實際發生：請它做 VBA 抽認卡 → 載入了模板、**完全沒呼叫 create_widget**、
# 直接寫散文並附這一句。上方沒有任何元件。
B2_CLAIM = """WakuWaku! 安妮亞收到任務了！✨

### 💡 複習重點
- 陣列讀寫：不要一格一格點。

(註：如果 widget 沒有顯示，請確認您的瀏覽器支援 iframe。)

安妮亞回覆完畢！🥜"""


def test_strips_a_bare_claim_with_no_source_at_all():
    """⚠️ 修之前接不住：原本開頭 `if not _WIDGET_SRC_RE.search(text): return`，
    沒有原始碼可剝就提早返回，指向空氣的句子那段永遠跑不到。"""
    out, changed = strip_orphan_widget_source(B2_CLAIM, widget_created=False)
    assert changed
    assert "iframe" not in out and "沒有顯示" not in out
    assert "陣列讀寫" in out and "安妮亞回覆完畢" in out, "只剝宣稱，內容要留著"


def test_the_iframe_excuse_is_the_worst_kind_of_claim():
    """這句把「沒做出來」講成「你的瀏覽器有問題」——比單純漏做更誤導。"""
    out, changed = strip_orphan_widget_source(
        "已完成。\n如果元件沒有顯示，請確認瀏覽器支援 iframe。\n結束", widget_created=False)
    assert changed and "瀏覽器" not in out


def test_bare_claim_is_kept_when_the_widget_really_exists():
    """有 widget 時這句是真話（元件可能真的被瀏覽器擋掉），不可誤刪。
    判斷依據只有 widget_created——系統知道事實，不必去猜措辭。"""
    out, changed = strip_orphan_widget_source(B2_CLAIM, widget_created=True)
    assert not changed and out == B2_CLAIM


@pytest.mark.parametrize("pos", ["上方", "下方", "以下"])
def test_pointing_at_a_missing_widget_in_any_direction(pos):
    """元件是渲染在回答**下方**的，所以「下方」才是模型的自然說法；
    原本只收「上方／上面」，最常見的那個方向反而漏掉。"""
    out, changed = strip_orphan_widget_source(
        f"整理好了。\n請點擊{pos}的抽認卡開始複習。\n加油！", widget_created=False)
    assert changed and pos not in out


@pytest.mark.parametrize("txt", [
    "Streamlit 的 components.html 會產生一個 iframe。",          # 講技術，不是宣稱
    "互動元件適合多維度比較，但這題用文字講比較清楚。",           # 講適用性，沒說有做
])
def test_talking_about_widgets_is_not_claiming_one_exists(txt):
    out, changed = strip_orphan_widget_source(txt, widget_created=False)
    assert not changed and out == txt


# ── Bug A：做完了卻只回道歉 ──────────────────────────────────────────────────
def test_describes_work_instead_of_bare_apology():
    msg = describe_completed_work(
        widget_title="聯準會官員觀點比較矩陣",
        todos=[{"content": "執行 market-research：分析談話對市場預期的影響", "status": "completed"},
               {"content": "建立 widget_comparison_matrix", "status": "completed"},
               {"content": "還沒做的事", "status": "pending"}],
        has_report=True, n_web=2, n_doc=0,
    )
    assert "聯準會官員觀點比較矩陣" in msg
    assert "結構化報告" in msg and "2 次網路搜尋" in msg
    assert "market-research" in msg
    assert "還沒做的事" not in msg, "未完成的 todo 不該被說成已完成"
    assert "幫我總結" in msg, "要給使用者一個比『再試一次』更省配額的下一步"


def test_returns_empty_when_nothing_was_produced():
    """真的什麼都沒做成時回空字串，呼叫端沿用原本的道歉文案。"""
    assert describe_completed_work() == ""
    assert describe_completed_work(todos=[{"content": "x", "status": "pending"}]) == ""


# ── 接線 ────────────────────────────────────────────────────────────────────
def test_wired_into_home():
    assert "from utils.turn_salvage import strip_orphan_widget_source, describe_completed_work" in SRC
    assert "ai_text, _stripped = strip_orphan_widget_source(ai_text, bool(_w))" in SRC
    assert "ai_text = describe_completed_work(" in SRC
    # 仍要保留最後的道歉當 fallback
    assert '"抱歉，安妮亞這次沒有取得回應，請再試一次。"' in SRC

# ── st.components.v1.html 棄用（2026-09-05 Cloud 警告；移除日期 2026-06-01 已過）──
def test_widget_render_does_not_use_st_iframe():
    """2026-09-05 線上事故：`st.iframe` 是 components.iframe（吃 URL）的公開版，
    **不是 components.html（吃 HTML）的替代品**。誤用導致 create_widget 全部失敗：
        TypeError: IframeMixin.iframe() got an unexpected keyword argument 'scrolling'
    更危險的是——若簽名剛好相容，整段 HTML 會被當成 URL、靜默渲染空白而不報錯。
    在驗證新 API 確實接受 raw HTML 之前，這條守門不准再引入 st.iframe。"""
    assert "def render_widget_html(" in SRC
    body = SRC[SRC.index("def render_widget_html("):]
    body = body[: body.index("\n\n# ---")]
    assert "components.html(html" in body
    # 只禁「呼叫」，docstring 裡引用名稱說明事故是必要的
    assert "st.iframe(" not in body, "不可再呼叫 st.iframe（見函式 docstring）"
    assert 'getattr(st, "iframe"' not in body, "不可再用 getattr 探測 st.iframe"
    # 兩個渲染點都要走 shim，不可留下直接呼叫
    others = SRC.replace(body, "")
    assert "components.html(" not in others, "還有直接呼叫 components.html 的地方"
    assert others.count("render_widget_html(") >= 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
