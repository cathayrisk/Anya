# pages/預測市場_診斷.py
# -*- coding: utf-8 -*-
"""
Polymarket API 契約診斷頁（唯讀，可安全部署）

用途：pages/預測市場.py 有四個假設在開發機上驗證不了（本機 DNS 對 *.polymarket.com
被導向非官方 IP）。把這頁推到 Streamlit Cloud 就能一次確認：

  1. /events 是否真的接受並套用 order / ascending / active / archived / closed
  2. tag_id 是否仍是現行值，且過濾真的生效
  3. limit 的上限，以及超過上限時是否靜默截斷
  4. 巢狀 markets[] 上 bestBid / bestAsk / groupItemTitle 等欄位的實際覆蓋率
  5. negRisk 旗標是否存在，且是否真的等於「互斥且窮盡」（能否取代價格加總啟發式）

設計原則：**每個探針都要有負向對照**。
最危險的失敗不是 API 回 4xx——那會被看見；而是 API 收下了參數卻默默忽略，
讓你拿到一份看起來正常、其實沒過濾也沒排序的資料。所以每個「有帶參數」的請求
都配一個「沒帶／帶假值」的請求來對照，兩者結果一樣就是參數沒生效。

刻意不 import pages/預測市場.py 的任何東西：這頁要測的是 API 本身的行為，
不是我們包裝過後的行為。HTTP 呼叫一律用最原始的 requests。

全部是公開端點的 GET，不需要金鑰，不會寫入任何東西。
用完把檔名結尾的 .py 拿掉即可停用（本專案慣例）。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import requests
import streamlit as st

DEFAULT_GAMMA = "https://gamma-api.polymarket.com"
DEFAULT_CLOB = "https://clob.polymarket.com"
UA = {"User-Agent": "Anya-PredictionMarket-Diagnostics/1.0"}

# 待驗證的對照表（來自 Polymarket 官方 constants/categories.ts）
TAGS_UNDER_TEST: list[tuple[str, int]] = [
    ("Finance", 120),
    ("Geopolitics", 100265),
    ("Politics", 2),
    ("Crypto", 21),
    ("Tech", 1401),
]
# 負向對照：一個幾乎不可能存在的 tag_id。
# 若它也回一堆資料，代表 tag_id 過濾根本沒生效，上面五個的結果全部不可信。
BOGUS_TAG_ID = 999_999_999

SPREAD_FALLBACK = 0.10

VERDICT_ICON = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "ERROR": "💥"}


def network_dead(p: "Probe") -> bool:
    """
    這個探針有沒有任何一個請求真的抵達伺服器？

    沒有的話，任何領域結論都是瞎猜——探針 3 曾經在純網路失敗時斷言「對照表已過期」，
    那會把人帶去改一份根本沒問題的 tag_id。連不上就只能說連不上。
    """
    return bool(p.calls) and all("status" not in c for c in p.calls)


@dataclass
class Probe:
    name: str
    verdict: str = "ERROR"
    summary: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    calls: list[dict] = field(default_factory=list)


# -----------------------
# 最原始的 HTTP，不做任何包裝
# -----------------------
def api_get(url: str, params: dict, timeout: int = 25) -> tuple[requests.Response | None, int, str | None]:
    t0 = time.perf_counter()
    try:
        r = requests.get(url, params=params, timeout=timeout, headers=UA)
        return r, int((time.perf_counter() - t0) * 1000), None
    except Exception as e:
        return None, int((time.perf_counter() - t0) * 1000), f"{type(e).__name__}: {e}"


def call_json(probe: Probe, url: str, params: dict, label: str) -> Any:
    """呼叫並把請求本身記進 probe.calls，讓證據可回溯。"""
    r, ms, err = api_get(url, params)
    rec = {"label": label, "params": dict(params), "ms": ms}
    if err:
        rec["error"] = err
        probe.calls.append(rec)
        return None
    rec["status"] = r.status_code
    rec["bytes"] = len(r.content)
    rec["url"] = r.url
    try:
        data = r.json()
    except Exception:
        rec["error"] = "回應不是合法 JSON"
        rec["body_head"] = r.text[:200]
        probe.calls.append(rec)
        return None
    rec["n"] = len(data) if isinstance(data, list) else None
    probe.calls.append(rec)
    return data if r.status_code == 200 else None


def f(x) -> float | None:
    try:
        if x is None or x == "":
            return None
        v = float(x)
        return None if pd.isna(v) else v
    except (TypeError, ValueError):
        return None


def as_list(x) -> list:
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            v = json.loads(x)
            return v if isinstance(v, list) else []
        except Exception:
            return []
    return []


def markets_of(events: list[dict]) -> list[dict]:
    out = []
    for ev in events or []:
        if isinstance(ev, dict):
            out.extend([m for m in (ev.get("markets") or []) if isinstance(m, dict)])
    return out


# -----------------------
# 探針 1：order / ascending 是否真的生效
# -----------------------
def probe_order(gamma: str) -> Probe:
    p = Probe("1. order / ascending 是否真的套用")
    base = {"limit": 25, "closed": "false"}

    desc = call_json(p, f"{gamma}/events", {**base, "order": "volume24hr", "ascending": "false"}, "order=volume24hr desc")
    asc = call_json(p, f"{gamma}/events", {**base, "order": "volume24hr", "ascending": "true"}, "order=volume24hr asc")
    # 負向對照：亂填一個欄位名。回 4xx 代表有驗證；回 200 代表會靜默忽略。
    bogus = call_json(p, f"{gamma}/events", {**base, "order": "definitelyNotAField", "ascending": "false"}, "order=亂填（負向對照）")

    if not isinstance(desc, list) or not desc:
        p.verdict, p.summary = "FAIL", "帶 order=volume24hr 的請求沒有回傳資料。"
        return p

    vols = [f(e.get("volume24hr")) for e in desc]
    known = [v for v in vols if v is not None]
    is_desc = all(known[i] >= known[i + 1] for i in range(len(known) - 1)) if len(known) > 1 else False

    asc_ids = [str(e.get("id")) for e in asc] if isinstance(asc, list) else []
    desc_ids = [str(e.get("id")) for e in desc]
    flips = bool(asc_ids) and asc_ids[:5] != desc_ids[:5]

    bogus_status = p.calls[2].get("status")
    bogus_ignored = bogus_status == 200

    p.evidence = {
        "前10筆 volume24hr": known[:10],
        "是否遞減": is_desc,
        "ascending 切換後前5筆是否改變": flips,
        "有 volume24hr 欄位的比例": f"{len(known)}/{len(vols)}",
        "亂填 order 的 HTTP": bogus_status,
        "亂填 order 是否被靜默忽略": bogus_ignored,
    }

    if is_desc and flips:
        p.verdict = "PASS"
        p.summary = "order 與 ascending 都確實生效（結果遞減，且切換方向後順序改變）。"
        if bogus_ignored:
            p.verdict = "WARN"
            p.summary += " 但亂填欄位名也回 200——代表 order 打錯字不會報錯，只會靜默失去排序。"
    elif is_desc and not flips:
        p.verdict, p.summary = "WARN", "結果是遞減的，但切 ascending=true 順序沒變——排序可能是預設行為而非參數生效。"
    else:
        p.verdict, p.summary = "FAIL", "結果不是遞減——order=volume24hr 沒有生效，主頁的熱門排行不可信。"
    return p


# -----------------------
# 探針 2：active / archived / closed 是否真的過濾
# -----------------------
def probe_filters(gamma: str) -> Probe:
    p = Probe("2. active / archived / closed 是否真的過濾")
    filtered = call_json(
        p, f"{gamma}/events",
        {"limit": 100, "order": "volume24hr", "ascending": "false",
         "closed": "false", "active": "true", "archived": "false"},
        "帶全部過濾參數",
    )
    naked = call_json(p, f"{gamma}/events", {"limit": 100, "order": "volume24hr", "ascending": "false"}, "不帶過濾參數（負向對照）")
    closed_only = call_json(p, f"{gamma}/events", {"limit": 25, "closed": "true"}, "closed=true（反向對照）")

    if not isinstance(filtered, list) or not filtered:
        p.verdict, p.summary = "FAIL", "帶過濾參數的請求沒有回傳資料。"
        return p

    def violations(rows: list[dict]) -> dict:
        return {
            "closed=True 的筆數": sum(1 for e in rows if e.get("closed") is True),
            "active=False 的筆數": sum(1 for e in rows if e.get("active") is False),
            "archived=True 的筆數": sum(1 for e in rows if e.get("archived") is True),
        }

    v = violations(filtered)
    total_viol = sum(v.values())

    naked_ids = [str(e.get("id")) for e in naked] if isinstance(naked, list) else []
    filt_ids = [str(e.get("id")) for e in filtered]
    same_as_naked = bool(naked_ids) and naked_ids[:10] == filt_ids[:10]

    closed_ok = None
    if isinstance(closed_only, list) and closed_only:
        closed_ok = sum(1 for e in closed_only if e.get("closed") is True) / len(closed_only)

    p.evidence = {
        "違反過濾的筆數": v,
        "回傳筆數": len(filtered),
        "與『不帶參數』前10筆是否完全相同": same_as_naked,
        "closed=true 時實際 closed 的比例": f"{closed_ok:.0%}" if closed_ok is not None else "無法判定",
        "事件是否帶 active/archived 欄位": {
            "active": sum(1 for e in filtered if "active" in e),
            "archived": sum(1 for e in filtered if "archived" in e),
            "closed": sum(1 for e in filtered if "closed" in e),
        },
    }

    if total_viol == 0 and not same_as_naked:
        p.verdict, p.summary = "PASS", "過濾參數生效：沒有任何違反條件的事件，且與不帶參數的結果不同。"
    elif total_viol == 0 and same_as_naked:
        p.verdict, p.summary = "WARN", "沒有違反條件的事件，但結果與不帶參數時完全相同——可能只是預設就這樣，參數未必生效。"
    else:
        p.verdict, p.summary = "FAIL", f"有 {total_viol} 筆違反過濾條件——參數沒有被套用，主頁可能混入已結束／已封存的市場。"
    return p


# -----------------------
# 探針 3：tag_id 是否仍有效（含負向對照）
# -----------------------
def probe_tags(gamma: str) -> Probe:
    p = Probe("3. tag_id 是否仍為現行值且過濾生效")
    rows = []
    tag_ok = 0

    for name, tid in TAGS_UNDER_TEST:
        data = call_json(
            p, f"{gamma}/events",
            {"limit": 20, "tag_id": tid, "related_tags": "true",
             "closed": "false", "active": "true",
             "order": "volume24hr", "ascending": "false"},
            f"tag_id={tid}（{name}）",
        )
        n = len(data) if isinstance(data, list) else 0
        # 回傳的事件自己的 tags[] 裡有沒有出現這個 id？沒有的話代表過濾沒生效。
        hit = 0
        sample_title = None
        if isinstance(data, list) and data:
            sample_title = str(data[0].get("title") or "")[:60]
            for ev in data:
                ids = {str(t.get("id")) for t in (ev.get("tags") or []) if isinstance(t, dict)}
                if str(tid) in ids:
                    hit += 1
        rows.append({
            "主題": name, "tag_id": tid, "回傳筆數": n,
            "tags[] 含該 id 的筆數": hit,
            "自我一致": (n > 0 and hit > 0),
            "範例標題": sample_title,
        })
        if n > 0:
            tag_ok += 1

    bogus = call_json(
        p, f"{gamma}/events",
        {"limit": 20, "tag_id": BOGUS_TAG_ID, "related_tags": "true", "closed": "false"},
        f"tag_id={BOGUS_TAG_ID}（假 id，負向對照）",
    )
    bogus_n = len(bogus) if isinstance(bogus, list) else 0

    p.evidence = {
        "各主題": rows,
        "假 tag_id 回傳筆數": bogus_n,
        "有資料的主題數": f"{tag_ok}/{len(TAGS_UNDER_TEST)}",
    }

    if bogus_n > 0:
        p.verdict = "FAIL"
        p.summary = (
            f"假 tag_id 也回了 {bogus_n} 筆——tag_id 過濾沒有生效，"
            "上面所有主題的結果都只是『全部事件』，主頁的主題切換等於裝飾。"
        )
    elif tag_ok == len(TAGS_UNDER_TEST):
        p.verdict, p.summary = "PASS", "五個財經向 tag_id 全部有資料，假 id 回 0 筆——過濾確實生效。"
    elif tag_ok == 0:
        p.verdict, p.summary = "FAIL", "五個 tag_id 全部沒資料——對照表已過期，需要改用 /tags 端點重新查。"
    else:
        p.verdict = "WARN"
        p.summary = f"只有 {tag_ok}/{len(TAGS_UNDER_TEST)} 個主題有資料，其餘 tag_id 可能已變更。"
    return p


# -----------------------
# 探針 4：limit 上限與靜默截斷
# -----------------------
def probe_limit(gamma: str) -> Probe:
    p = Probe("4. limit 上限與是否靜默截斷")
    base = {"order": "volume24hr", "ascending": "false", "closed": "false"}
    counts = {}
    for lim in (100, 500, 1000):
        data = call_json(p, f"{gamma}/events", {**base, "limit": lim}, f"limit={lim}")
        counts[lim] = len(data) if isinstance(data, list) else None

    # offset 是否可用來翻頁（若上限真的是 500，這是唯一的出路）
    page2 = call_json(p, f"{gamma}/events", {**base, "limit": 100, "offset": 100}, "limit=100&offset=100")
    page1 = call_json(p, f"{gamma}/events", {**base, "limit": 100, "offset": 0}, "limit=100&offset=0")
    offset_works = None
    if isinstance(page1, list) and isinstance(page2, list) and page1 and page2:
        ids1 = {str(e.get("id")) for e in page1}
        ids2 = {str(e.get("id")) for e in page2}
        offset_works = len(ids1 & ids2) == 0

    p.evidence = {
        "各 limit 實際回傳筆數": counts,
        "limit=1000 的 HTTP": p.calls[2].get("status"),
        "offset 分頁是否可用（兩頁無重疊）": offset_works,
    }

    c500, c1000 = counts.get(500), counts.get(1000)
    if c1000 is None and p.calls[2].get("status") not in (200, None):
        p.verdict, p.summary = "PASS", f"limit=1000 被拒（HTTP {p.calls[2].get('status')}）——有明確上限，不會靜默截斷。"
    elif c500 and c1000 and c1000 <= c500:
        p.verdict = "WARN"
        p.summary = (
            f"limit=1000 只回 {c1000} 筆且回 200——確認會**靜默截斷**。"
            "主頁每個 tag 抓 150 筆遠低於上限，目前安全；但若之後放大取數，必須改走 offset 分頁。"
        )
    elif c1000 and c500 and c1000 > c500:
        p.verdict, p.summary = "PASS", f"limit=1000 回了 {c1000} 筆，上限高於 500。"
    else:
        p.verdict, p.summary = "WARN", "無法判定上限行為，請看下方原始呼叫紀錄。"
    return p


# -----------------------
# 探針 5：巢狀 markets[] 欄位覆蓋率
# -----------------------
def probe_fields(gamma: str, sample_n: int) -> Probe:
    p = Probe("5. 巢狀 markets[] 的欄位覆蓋率")
    events = call_json(
        p, f"{gamma}/events",
        {"limit": sample_n, "order": "volume24hr", "ascending": "false",
         "closed": "false", "active": "true", "tag_id": 120, "related_tags": "true"},
        f"Finance 前 {sample_n} 筆取樣",
    )
    if not isinstance(events, list) or not events:
        p.verdict, p.summary = "FAIL", "取樣請求沒有資料。"
        return p

    mkts = markets_of(events)
    if not mkts:
        p.verdict, p.summary = "FAIL", "事件裡沒有巢狀 markets[]——/events 的結構跟預期不同，主頁的解析全錯。"
        return p

    def cover(key: str) -> str:
        n = sum(1 for m in mkts if m.get(key) not in (None, "", [], "[]"))
        return f"{n}/{len(mkts)}（{n / len(mkts):.0%}）"

    critical = ["bestBid", "bestAsk", "lastTradePrice", "outcomes", "outcomePrices", "clobTokenIds"]
    coverage = {k: cover(k) for k in critical + ["groupItemTitle", "spread", "enableOrderBook"]}

    # 中價規則實際會 fallback 幾成？
    spreads = []
    for m in mkts:
        b, a = f(m.get("bestBid")), f(m.get("bestAsk"))
        if b is not None and a is not None and a >= b:
            spreads.append(a - b)
    wide = sum(1 for s in spreads if s > SPREAD_FALLBACK)

    # 多結果事件的 overround 真的落在 1.05–1.12 嗎？
    single = sum(1 for ev in events if len(ev.get("markets") or []) == 1)
    multi_over = []
    for ev in events:
        ms = [m for m in (ev.get("markets") or []) if isinstance(m, dict)]
        if len(ms) < 2:
            continue
        tot = 0.0
        ok = False
        for m in ms:
            pr = [f(x) for x in as_list(m.get("outcomePrices"))]
            if pr and pr[0] is not None:
                tot += pr[0]
                ok = True
        if ok:
            multi_over.append(round(tot, 3))

    p.evidence = {
        "取樣": f"{len(events)} 個事件 / {len(mkts)} 個 market",
        "欄位覆蓋率": coverage,
        "單一 market 的事件": f"{single}/{len(events)}",
        "多 market 的事件": f"{len(events) - single}/{len(events)}",
        "有雙邊報價的 market": f"{len(spreads)}/{len(mkts)}",
        "價差 > 0.10（會退回 last）": f"{wide}/{len(spreads)}" if spreads else "無資料",
        "價差中位數": round(pd.Series(spreads).median(), 4) if spreads else None,
        "多結果 overround 樣本": sorted(multi_over)[:15],
        "overround 中位數": round(pd.Series(multi_over).median(), 3) if multi_over else None,
    }
    p.evidence["_raw_market_sample"] = {k: mkts[0].get(k) for k in coverage}

    missing = [k for k in critical if k not in mkts[0]]
    bid_ask_ok = sum(1 for m in mkts if m.get("bestBid") is not None and m.get("bestAsk") is not None)

    if missing:
        p.verdict, p.summary = "FAIL", f"關鍵欄位缺失：{missing}——主頁的中價規則會整個退回 lastTradePrice。"
    elif bid_ask_ok / len(mkts) < 0.5:
        p.verdict = "WARN"
        p.summary = f"只有 {bid_ask_ok}/{len(mkts)} 個 market 有雙邊報價，中價規則有一半以上會 fallback。"
    else:
        p.verdict, p.summary = "PASS", f"關鍵欄位齊全，{bid_ask_ok}/{len(mkts)} 個 market 有雙邊報價可算中價。"
    return p


# -----------------------
# 探針 7：negRisk 是否存在，且真的等於「互斥且窮盡」
# -----------------------
def probe_negrisk(gamma: str, sample_n: int) -> Probe:
    """
    主頁目前用「價格加總落在 0.90–1.30」當互斥窮盡的啟發式門檻。
    negRisk 若真是官方的正式旗標，就能取代它。

    但「欄位存在」不等於「可用」——關鍵在交叉驗證：
    negRisk=True 的事件，價格加總是否真的都落在區間內？
    只要出現一個 negRisk=True 卻加總 0.09 的事件，這個旗標對我們就沒有用。
    """
    p = Probe("7. negRisk 是否存在，且真的等於「互斥且窮盡」")
    events: list[dict] = []
    for tid, name in [(120, "Finance"), (100265, "Geopolitics"), (2, "Politics")]:
        data = call_json(
            p, f"{gamma}/events",
            {"limit": sample_n, "order": "volume24hr", "ascending": "false",
             "closed": "false", "active": "true", "tag_id": tid, "related_tags": "true"},
            f"tag_id={tid}（{name}）取樣",
        )
        if isinstance(data, list):
            events.extend(x for x in data if isinstance(x, dict))

    seen: set[str] = set()
    uniq: list[dict] = []
    for e in events:
        eid = str(e.get("id"))
        if eid and eid not in seen:
            seen.add(eid)
            uniq.append(e)

    if not uniq:
        p.verdict, p.summary = "FAIL", "取樣沒有資料。"
        return p

    has_field = sum(1 for e in uniq if "negRisk" in e)
    value_counts: dict[str, int] = {}
    for e in uniq:
        k = repr(e.get("negRisk")) if "negRisk" in e else "(欄位不存在)"
        value_counts[k] = value_counts.get(k, 0) + 1

    LO, HI = 0.90, 1.30
    cont = {"negRisk=True 且加總在區間內": 0, "negRisk=True 但加總在區間外": 0,
            "negRisk=False 但加總在區間內": 0, "negRisk=False 且加總在區間外": 0}
    outliers: list[dict] = []
    multi_n = 0

    for e in uniq:
        ms = [m for m in (e.get("markets") or []) if isinstance(m, dict)]
        if len(ms) < 2:
            continue
        tot, ok = 0.0, False
        for m in ms:
            pr = [f(x) for x in as_list(m.get("outcomePrices"))]
            if pr and pr[0] is not None:
                tot += pr[0]
                ok = True
        if not ok:
            continue
        multi_n += 1
        neg = e.get("negRisk")
        inband = LO <= tot <= HI
        if neg is True:
            cont["negRisk=True 且加總在區間內" if inband else "negRisk=True 但加總在區間外"] += 1
        elif neg is False:
            cont["negRisk=False 但加總在區間內" if inband else "negRisk=False 且加總在區間外"] += 1
        if not inband:
            outliers.append({
                "事件": str(e.get("title") or "")[:46],
                "negRisk": repr(neg),
                "market 數": len(ms),
                "價格加總": round(tot, 3),
            })

    p.evidence = {
        "取樣事件數": len(uniq),
        "有 negRisk 欄位": f"{has_field}/{len(uniq)}（{has_field / len(uniq):.0%}）",
        "值分布": value_counts,
        "多 market 事件數": multi_n,
        "交叉驗證": cont,
        "加總落在區間外的事件": sorted(outliers, key=lambda r: r["價格加總"])[:12],
        # 順便把完整欄位清單倒出來，看看還有沒有別的可用旗標
        "事件層可用欄位": sorted(uniq[0].keys()),
        "market 層可用欄位": sorted((uniq[0].get("markets") or [{}])[0].keys()),
    }

    bad = cont["negRisk=True 但加總在區間外"]
    caught = cont["negRisk=False 且加總在區間外"]
    missed = cont["negRisk=False 但加總在區間內"]

    if has_field == 0:
        p.verdict, p.summary = "FAIL", "事件上沒有 negRisk 欄位——維持現在的價格加總啟發式。"
    elif has_field < len(uniq):
        p.verdict = "WARN"
        p.summary = f"只有 {has_field}/{len(uniq)} 個事件帶 negRisk，覆蓋率不足以單獨當判準，只能當輔助。"
    elif multi_n == 0:
        p.verdict, p.summary = "WARN", "取樣裡沒有多 market 事件，無法交叉驗證。加大取樣再試。"
    elif bad > 0:
        p.verdict = "WARN"
        p.summary = (
            f"有 {bad} 個事件 negRisk=True 但加總落在區間外——旗標不足以保證互斥窮盡，"
            "必須與價格加總門檻併用（兩者都通過才正規化）。"
        )
    else:
        p.verdict = "PASS"
        p.summary = (
            f"negRisk 覆蓋 100%，且沒有任何 negRisk=True 的事件加總落在區間外"
            f"（區間外的 {caught} 個全部是 negRisk=False）。可升級成正式判準。"
        )
        if missed:
            p.summary += f" 註：另有 {missed} 個 negRisk=False 但加總在區間內，換判準後這些會停止正規化。"
    return p


# -----------------------
# 探針 6：prices-history 的 interval 語意
# -----------------------
def probe_history(gamma: str, clob: str) -> Probe:
    p = Probe("6. prices-history 的 interval 語意（1m 是月不是分）")
    events = call_json(
        p, f"{gamma}/events",
        {"limit": 5, "order": "volume24hr", "ascending": "false", "closed": "false",
         "active": "true", "tag_id": 120, "related_tags": "true"},
        "取一個 Finance 事件當樣本",
    )
    token = None
    title = None
    for ev in events or []:
        for m in ev.get("markets") or []:
            ids = as_list(m.get("clobTokenIds"))
            if ids:
                token, title = str(ids[0]), str(ev.get("title") or "")[:60]
                break
        if token:
            break

    if not token:
        p.verdict, p.summary = "FAIL", "取樣事件裡找不到 clobTokenIds，畫不出走勢。"
        return p

    spans = {}
    for ui, iv in [("1H", "1h"), ("1D", "1d"), ("1W", "1w"), ("1M", "1m"), ("ALL", "max")]:
        data = call_json(p, f"{clob}/prices-history", {"market": token, "interval": iv, "fidelity": 10}, f"interval={iv}")
        hist = data.get("history", []) if isinstance(data, dict) else []
        ts = [x.get("t") for x in hist if isinstance(x, dict) and x.get("t")]
        hours = round((max(ts) - min(ts)) / 3600, 1) if len(ts) > 1 else 0
        spans[ui] = {"點數": len(hist), "涵蓋小時": hours}

    p.evidence = {"樣本事件": title, "token_id": token[:24] + "…", "各區間": spans}

    m_hours = spans.get("1M", {}).get("涵蓋小時", 0)
    d_hours = spans.get("1D", {}).get("涵蓋小時", 0)
    if m_hours > max(d_hours, 48):
        p.verdict, p.summary = "PASS", f"1m 涵蓋 {m_hours} 小時，遠大於 1d 的 {d_hours} 小時——確認是『近一個月』。"
    elif spans.get("1M", {}).get("點數", 0) == 0:
        p.verdict, p.summary = "WARN", "1m 回空，無法判定語意（可能該市場太新）。換一個成交量更大的市場再試。"
    else:
        p.verdict, p.summary = "FAIL", f"1m 只涵蓋 {m_hours} 小時——語意可能不是一個月，RANGE_MAP 要改。"
    return p


# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="Polymarket API 診斷", layout="wide", initial_sidebar_state="collapsed")
st.title("Polymarket API 契約診斷")
st.caption(
    "驗證 pages/預測市場.py 依賴的 API 假設。全部是公開端點的 GET，唯讀、免金鑰、不寫入任何東西。"
    "每個探針都配負向對照——因為最危險的失敗是 API 收下參數卻默默忽略。"
)

c1, c2, c3 = st.columns([3, 3, 2])
with c1:
    gamma = st.text_input("Gamma base", DEFAULT_GAMMA)
with c2:
    clob = st.text_input("CLOB base", DEFAULT_CLOB)
with c3:
    sample_n = st.number_input("欄位取樣事件數", min_value=10, max_value=200, value=60, step=10)

if st.button("開始診斷", type="primary"):
    probes: list[Probe] = []
    steps = [
        ("排序參數", lambda: probe_order(gamma)),
        ("過濾參數", lambda: probe_filters(gamma)),
        ("tag_id", lambda: probe_tags(gamma)),
        ("limit 上限", lambda: probe_limit(gamma)),
        ("欄位覆蓋率", lambda: probe_fields(gamma, int(sample_n))),
        ("走勢區間", lambda: probe_history(gamma, clob)),
        ("negRisk", lambda: probe_negrisk(gamma, int(sample_n))),
    ]
    bar = st.progress(0.0, text="準備中…")
    for i, (label, fn) in enumerate(steps, start=1):
        bar.progress((i - 1) / len(steps), text=f"執行中：{label}…")
        try:
            pr = fn()
            if network_dead(pr):
                first = next((c.get("error") for c in pr.calls if c.get("error")), "未知")
                pr.verdict = "ERROR"
                pr.summary = f"完全連不到 API，無法判定任何事。首個錯誤：{first[:120]}"
            probes.append(pr)
        except Exception as e:
            probes.append(Probe(label, "ERROR", f"探針自己爆了：{type(e).__name__}: {e}"))
    bar.empty()
    # 存進 session_state，不然一動下方任何 widget 結果就整塊消失
    st.session_state["diag_probes"] = probes

probes: list[Probe] | None = st.session_state.get("diag_probes")

if probes is None:
    st.info("按「開始診斷」。全部跑完約 15–40 秒，會發出約 28 個請求。")
    st.stop()

# 總覽
counts = {v: sum(1 for p in probes if p.verdict == v) for v in ("PASS", "WARN", "FAIL", "ERROR")}
o1, o2, o3, o4 = st.columns(4)
o1.metric("✅ PASS", counts["PASS"])
o2.metric("⚠️ WARN", counts["WARN"])
o3.metric("❌ FAIL", counts["FAIL"])
o4.metric("💥 ERROR", counts["ERROR"])

if counts["ERROR"] == len(probes):
    st.error(
        "所有探針都連不到 Polymarket——這是網路層問題，不是 API 契約問題。"
        "在本機通常是 DNS 攔截；在 Streamlit Cloud 出現這個結果才值得緊張。"
    )
elif counts["FAIL"] or counts["ERROR"]:
    st.error("有探針失敗——主頁的對應假設不成立，先看下面哪一項掛了。")
elif counts["WARN"]:
    st.warning("全部通過但有警告，值得看一眼。")
else:
    st.success("所有假設確認成立，主頁可以照現在的寫法上線。")

for p in probes:
    icon = VERDICT_ICON.get(p.verdict, "❔")
    with st.expander(f"{icon} {p.name} — {p.summary}", expanded=(p.verdict in ("FAIL", "ERROR"))):
        if p.evidence:
            raw_sample = p.evidence.pop("_raw_market_sample", None)
            for k, v in p.evidence.items():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    st.dataframe(pd.DataFrame(v), hide_index=True)
                else:
                    st.markdown(f"- **{k}**：`{v}`")
            if raw_sample:
                st.markdown("**單一 market 的原始欄位取樣**")
                st.json(raw_sample, expanded=False)
        if p.calls:
            st.markdown("**原始呼叫紀錄**")
            st.dataframe(pd.DataFrame(p.calls), hide_index=True)

# 可貼回來的摘要
st.divider()
st.subheader("診斷摘要（複製這段貼回對話）")
summary = {
    "gamma": gamma,
    "clob": clob,
    "tally": counts,
    "probes": [
        {"name": p.name, "verdict": p.verdict, "summary": p.summary,
         "evidence": p.evidence,
         "calls": [{k: v for k, v in c.items() if k != "url"} for c in p.calls]}
        for p in probes
    ],
}
st.code(json.dumps(summary, ensure_ascii=False, indent=2, default=str), language="json")
st.caption("用完可把檔名結尾的 .py 拿掉來停用這頁（本專案慣例），或直接刪除。")
