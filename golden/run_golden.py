# -*- coding: utf-8 -*-
"""占星黃金題組：離線測試組。

跑法（專案根目錄）：
    python tests/golden/run_golden.py
    python tests/golden/run_golden.py --live     # 額外做真實網站煙霧測試

需要 kerykeion。本機若只裝在 natal-reading-kit 的 venv，用它的直譯器跑：
    "C:/Users/Patrick/Desktop/natal-reading-kit/natal-reading-kit/.venv/Scripts/python.exe" tests/golden/run_golden.py

離線題（1,2,5,7,8,9）完全不碰網路也不呼叫模型：秒級、免費、可以每次改完就跑。
第 3、4、6、10 題要真的產出一份解讀才驗得到 → 交給 check_reading.py。
"""
from __future__ import annotations

import os
import re
import sys
import ast
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
    import cases as C           # noqa: E402
try:
    from tests.golden import fixtures as F
except ImportError:
    import fixtures as F        # noqa: E402

_results: list[tuple[bool, str]] = []


def check(cond, msg):
    _results.append((bool(cond), msg))
    print(f"    {'OK  ' if cond else 'FAIL'} {msg}")
    return bool(cond)


def head(case):
    print(f"\n[{case['id']}] {case['title']}　（{case['tier']}）")
    print(f"    輸入：{case['prompt'][:60]}")


def need_kerykeion():
    try:
        import kerykeion  # noqa: F401
        return True
    except Exception:
        print("\n⚠️  找不到 kerykeion —— 需要星盤計算的題目將略過。")
        print("    用裝了 kerykeion 的直譯器重跑，例如 natal-reading-kit 的 venv。")
        return False


# ---------------------------------------------------------------- 路由
def skill_hint_match(text: str) -> list[str]:
    """不 import Home.py（會拉進 streamlit），直接從原始碼取出 SKILL_HINT_RES。"""
    src = (ROOT / "Home.py").read_text(encoding="utf-8")
    blk = re.search(r"SKILL_HINT_RES.*?\n\}", src, re.S)
    if not blk:
        return []
    hits = []
    for k, pat in re.findall(r'"([a-z0-9-]+)":\s*re\.compile\(r"([^"]+)"', blk.group(0)):
        if re.search(pat, text):
            hits.append(k)
    return hits


# ---------------------------------------------------------------- 題目
def case_1(HAS_K):
    c = next(x for x in C.CASES if x["id"] == 1)
    head(c)
    check("astro-natal" in skill_hint_match(c["prompt"]),
          "路由命中 astro-natal（強制升級 General）")
    if not HAS_K:
        return
    from utils.astro import compute_natal, ANGLE_NAMES
    o = compute_natal(**c["chart"])
    check(o.get("houses_available") is False, "houses_available = False")
    check("angles" not in o, "輸出無 angles 欄位")
    check(not any(a["p1"] in ANGLE_NAMES or a["p2"] in ANGLE_NAMES
                  for a in o.get("aspects", [])), "相位清單不含四軸")
    check(not any("house" in p for p in o.get("points", [])), "星體不帶宮位")
    mu = o.get("moon_uncertainty") or {}
    check(mu.get("crosses_sign") is True, "偵測到當天月亮跨星座")
    signs = {mu.get("range_start", {}).get("sign_zh"), mu.get("range_end", {}).get("sign_zh")}
    check(signs == {"水瓶", "雙魚"}, f"兩種可能為水瓶／雙魚（得到 {signs}）")
    check(o.get("birth_time") is None, "birth_time 回 None，不是假的 12:00")


def case_2(HAS_K):
    c = next(x for x in C.CASES if x["id"] == 2)
    head(c)
    from utils.astro import parse_time
    for txt, hh, mm in [("上午10點15分", 10, 15), ("下午3點", 15, 0), ("晚上8點30分", 20, 30),
                        ("凌晨2點", 2, 0), ("中午12點", 12, 0), ("上午12點", 0, 0),
                        ("10:15", 10, 15), ("9點", 9, 0)]:
        r = parse_time(txt)
        check(r.get("hour") == hh and r.get("minute") == mm,
              f"「{txt}」→ {r.get('hour')}:{r.get('minute'):02d}"
              if "error" not in r else f"「{txt}」解析失敗：{r}")
    if not HAS_K:
        return
    from utils.astro import compute_natal
    o = compute_natal(**c["chart"])
    asc = (o.get("angles") or {}).get("ascendant") or {}
    check(asc.get("sign_zh") == "獅子" and abs(asc.get("deg", 0) - 7.38) < 0.05,
          f"上升獅子 7.38°（得到 {asc.get('sign_zh')} {asc.get('deg')}）")


def case_5(HAS_K):
    c = next(x for x in C.CASES if x["id"] == 5)
    head(c)
    if not HAS_K:
        return
    from utils.astro import compute_natal
    from utils import astro_state as S
    sess = {}
    f1 = S.build_facts(compute_natal(**c["prev_chart"]), c["prev_chart"])
    cid1, ch1 = S.put(sess, f1)
    check(ch1 is False, "第一張盤不算換盤")
    sess["gm_history_summary"] = {"count": 5, "summary": "小P 的上升是獅子，六宮有摩羯三星…"}
    f2 = S.build_facts(compute_natal(**c["chart"]), c["chart"])
    cid2, ch2 = S.put(sess, f2)
    check(cid1 != cid2, f"chart_id 不同（{cid1} → {cid2}）")
    check(ch2 is True, "偵測到換盤")
    if ch2:
        S.on_active_chart_change(sess)
    check("gm_history_summary" not in sess, "舊盤的歷史摘要已失效")
    proj = S.project(S.active(sess))
    check(cid2 in proj and cid1 not in proj, "正典投影已切換到新盤")
    check("獅子" not in proj.split("四軸：")[1].split("\n")[0]
          if "四軸：" in proj else True, "投影不含舊盤的上升")


def case_7():
    c = next(x for x in C.CASES if x["id"] == 7)
    head(c)
    from utils import astro_ref as R
    b = R.Broker(fetcher=F.fake_fetcher)
    r = b.get(c["slug"])
    check(r["status"] == "not_found", f"回 not_found（得到 {r['status']}）")
    check(bool(r.get("suggestions")), f"附上建議：{r.get('suggestions', [])[:3]}")
    check(all(R.exists(s) for s in r.get("suggestions", [])), "每個建議都真實存在")
    check(b.attempts == 0, "未消耗抓取次數（查無不該花配額）")
    # 杜撰網址也要擋
    r2 = b.get("transit-saturn-square-natal-moon")
    check(r2["status"] == "not_found", "先前杜撰過的網址同樣被擋")
    check(any("transit-saturn" in s for s in r2.get("suggestions", [])),
          "建議裡有正確的 transit-saturn-*")


def case_8(HAS_K):
    c = next(x for x in C.CASES if x["id"] == 8)
    head(c)
    if not HAS_K:
        return
    from utils.astro import compute_synastry, ANGLE_NAMES
    o = compute_synastry(a_name="Me", a_birthdate="1990-05-18", a_birth_time=None,
                         b_name="Him", b_birthdate="1985-03-02", b_birth_time=None)
    if "error" in o:
        check(False, f"合盤計算失敗：{o}")
        return
    check(o["a"].get("houses_available") is False, "A 方 houses_available=False")
    check(o["b"].get("houses_available") is False, "B 方 houses_available=False")
    check("angles" not in o["a"] and "angles" not in o["b"], "雙方皆無 angles")
    check(not any(a["p1"] in ANGLE_NAMES or a["p2"] in ANGLE_NAMES
                  for a in o.get("cross_aspects", [])), "交互相位不含四軸")


def case_9(HAS_K):
    c = next(x for x in C.CASES if x["id"] == 9)
    head(c)
    src = (ROOT / "Home.py").read_text(encoding="utf-8")
    check("ASTRO_STATE.project(_facts)" in src and "instructions +=" in src,
          "正典投影注入 system instructions（不進訊息串）")
    bl = src[src.index("def build_lc_messages"):src.index("def estimate_tokens_for_lc_messages")]
    check("ASTRO_STATE" not in bl, "build_lc_messages 不碰正典（訊息串乾淨）")
    check("note_for_summarizer" in src, "摘要器已被指示不要寫入星盤數字")
    if not HAS_K:
        return
    from utils.astro import compute_natal
    from utils import astro_state as S
    proj = S.project(S.build_facts(compute_natal(**c["chart"]), c["chart"]))
    check("雙魚0.2" in proj, "月亮度數在投影中仍精確（雙魚 0.2°）")
    check(round(len(proj) / 3.6) < 400, f"投影 ~{round(len(proj)/3.6)} tok（每回合成本可接受）")


def case_broker_budget():
    """跨題共用：中介的預算與 fail-closed，第 3／6 題的前提。"""
    print("\n[*] 中介預算與 fail-closed（第 3、6 題的前提）")
    from utils import astro_ref as R
    b = R.Broker(fetcher=F.fake_fetcher)
    got = [b.get(s, aspect="trine") for s in
           ("natal-sun-saturn-aspects", "natal-moon-nessus-aspects",
            "foundation-saturn", "natal-sun-fourth-house")]
    check(sum(1 for g in got if g["status"] == "ok") <= R.MAX_SOURCES_PER_RUN,
          f"來源上限 {len(b.sources)}/{R.MAX_SOURCES_PER_RUN}")
    check(got[-1]["status"] == "budget", "第 4 篇被預算擋下")
    check(b.tokens <= R.MAX_EVIDENCE_TOKENS,
          f"素材總量 {b.tokens}/{R.MAX_EVIDENCE_TOKENS} tok")
    check(got[0]["matched_aspect"] is True, "五相位文章有切中 trine 段")
    check("The Trine" in got[0]["material"] and "The Square" not in got[0]["material"],
          "只留 trine，其餘四段已丟")
    check("不可執行" in got[0]["material"], "素材框成不可信資料")
    # fail closed
    b2 = R.Broker(fetcher=F.fake_fetcher)
    R.load_index().update({"__huge__", "__flat__", "__404__"})   # 讓病態案例進得了索引
    check(b2.get("__huge__", aspect="trine")["status"] == "miss", "單段爆預算 → miss 而非盲切")
    check(b2.get("__404__")["status"] == "miss", "404 頁 → miss")
    r = b2.get("__flat__")
    check(r["status"] in ("ok", "miss"), "無標題頁不會拋例外")


def live_smoke():
    print("\n[live] 真實網站煙霧測試")
    from utils import astro_ref as R
    b = R.Broker()
    r = b.get("natal-sun-saturn-aspects", aspect="trine")
    check(r["status"] == "ok", f"實抓成功（{r.get('status')}）")
    if r["status"] == "ok":
        check(r["matched_aspect"], "切中 trine 段")
        check(r["tokens"] <= R.PER_SOURCE_HARD, f"{r['tokens']} tok 在單篇上限內")
        check("The Trine" in r["material"], "內容確實是 trine 段")


def main():
    live = "--live" in sys.argv
    HAS_K = need_kerykeion()
    print("=" * 62)
    print("占星黃金題組 — 離線層")
    print("=" * 62)
    case_1(HAS_K)
    case_2(HAS_K)
    case_5(HAS_K)
    case_7()
    case_8(HAS_K)
    case_9(HAS_K)
    case_broker_budget()
    if live:
        live_smoke()

    passed = sum(1 for ok_, _ in _results if ok_)
    total = len(_results)
    print("\n" + "=" * 62)
    print(f"通過 {passed}/{total}")
    for ok_, msg in _results:
        if not ok_:
            print(f"  FAIL  {msg}")
    print("\n第 3、4、6、10 題需要真的產出一份解讀：")
    print("  1) 在 Anya 依序輸入 cases.py 裡那幾題的 prompt")
    print("  2) 把回答貼進檔案，跑：python tests/golden/check_reading.py <檔案> --case 3")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
