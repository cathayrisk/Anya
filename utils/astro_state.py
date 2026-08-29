# -*- coding: utf-8 -*-
"""占星正典狀態（astro_facts）：星盤事實的唯一真相來源。

為什麼要有這一層
----------------
歷史摘要器（gemma-4-26b）會把舊訊息壓成 300 字。星盤數字一旦被摘要
「大致轉述」，後續回合就會拿轉述當事實推理——實測過模型憑印象講月亮星座，
那天月亮中午前後才跨星座，等於擲硬幣。

所以：**星盤事實不放進訊息串**，改放這裡，每回合以正典形式重新注入。
摘要器只摘對話，不摘星盤；摘要與正典衝突時，正典贏。

還有一個更陰險的情況（對抗審查的壓力測試打出來的）：使用者先問自己的盤，
再問「幫我看阿明的盤」——**舊盤的解讀還躺在歷史裡**。宣告「正典優先」
只是提示詞層級的一句話，擋不住模型錨定舊文字。因此換盤時要讓摘要失效，
見 `on_active_chart_change()`。

本模組不 import streamlit，session 由呼叫端傳進來（好測試）。
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

# 注入用的緊湊序列化：同樣資訊，JSON 要 1,002 tokens，這裡約 400。
# 壓縮的是格式不是內容——數字一個都不能少（審查明確反對選擇性投影：
# 占星的依賴關係無法機械閉合，模型無法索取它不知道自己缺少的東西）。

_ANGLE_ABBR = {
    "ascendant": "ASC", "medium_coeli": "MC",
    "descendant": "DSC", "imum_coeli": "IC",
}


def chart_id(spec: Dict[str, Any]) -> str:
    """由計算規格算出穩定的 chart_id。

    納入所有會改變盤面的東西：人、時間、地點、時區、宮位制、黃道類型。
    只要其中一項變了就是另一張盤，歷史裡的舊解讀必須失效。
    """
    parts = [
        str(spec.get("name") or ""),
        str(spec.get("birthdate") or ""),
        str(spec.get("birth_time") or "-"),
        f"{spec.get('lat')},{spec.get('lng')}",
        str(spec.get("tz") or ""),
        str(spec.get("houses_system") or "P"),
        str(spec.get("zodiac_type") or "Tropical"),
        str(spec.get("kind") or "natal"),
    ]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]


def _fmt_point(p: Dict[str, Any]) -> str:
    """單一星體：名稱 星座度數 [宮位] [逆行] [近交界]"""
    bits = [str(p.get("name_zh") or p.get("name") or "?")]
    sign = p.get("sign_zh") or p.get("sign") or ""
    deg = p.get("deg")
    bits.append(f"{sign}{deg}" if deg is not None else str(sign))
    if p.get("house"):
        bits.append(f"H{p['house']}")
    if p.get("retrograde"):
        bits.append("R")
    if p.get("near_cusp"):
        bits.append("~界")          # 離星座交界不到 1 度，星座可能翻掉
    if p.get("out_of_bounds"):
        bits.append("OOB")
    return " ".join(bits)


def build_facts(tool_out: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
    """把計算工具的輸出收斂成正典事實物件。

    不做詮釋，只做結構化與標記。`derived` 欄位裡的東西（命主星）明確標為推導值，
    免得下游把它跟實際計算出來的位置混為一談。
    """
    cid = chart_id(spec)
    facts: Dict[str, Any] = {
        "chart_id": cid,
        "revision": 1,
        "kind": tool_out.get("kind") or spec.get("kind") or "natal",
        "spec": {
            "name": spec.get("name"),
            "birthdate": spec.get("birthdate"),
            "birth_time": tool_out.get("birth_time"),   # 降級時工具回 None，照實記
            "lat": spec.get("lat"), "lng": spec.get("lng"), "tz": spec.get("tz"),
            "houses_system": spec.get("houses_system") or "Placidus",
            "zodiac_type": spec.get("zodiac_type") or "Tropical",
        },
        "points": tool_out.get("points") or [],
        "angles": tool_out.get("angles") or {},
        "aspects": tool_out.get("aspects") or [],
        "distribution": tool_out.get("distribution") or {},
        "lunar_phase": tool_out.get("lunar_phase") or {},
        "houses_available": bool(tool_out.get("houses_available", True)),
        "flags": {},
        "derived": {},
    }
    for k in ("warning", "moon_uncertainty", "houses_unavailable_reason",
              "near_cusp_warning", "context_withheld"):
        if tool_out.get(k):
            facts["flags"][k] = tool_out[k]

    asc = (facts["angles"].get("ascendant") or {}).get("sign")
    if asc and facts["houses_available"]:
        ruler = _CHART_RULERS.get(asc)
        if ruler:
            pt = next((p for p in facts["points"]
                       if str(p.get("name", "")).lower() == ruler), None)
            if pt:
                facts["derived"]["chart_ruler"] = {
                    "body": ruler, "sign": pt.get("sign_zh") or pt.get("sign"),
                    "house": pt.get("house"), "note": "推導值（上升星座的傳統主星）",
                }
    return facts


_CHART_RULERS = {
    "Ari": "mars", "Tau": "venus", "Gem": "mercury", "Can": "moon",
    "Leo": "sun", "Vir": "mercury", "Lib": "venus", "Sco": "pluto",
    "Sag": "jupiter", "Cap": "saturn", "Aqu": "uranus", "Pis": "neptune",
    "Aries": "mars", "Taurus": "venus", "Gemini": "mercury", "Cancer": "moon",
    "Virgo": "mercury", "Libra": "venus", "Scorpio": "pluto",
    "Sagittarius": "jupiter", "Capricorn": "saturn", "Aquarius": "uranus",
    "Pisces": "neptune",
}


def project(facts: Dict[str, Any]) -> str:
    """正典事實 → 注入用的緊湊文字。

    刻意**全量**投影（不是只給相關星體）：審查指出占星的依賴關係無法機械閉合，
    命主星、定位星、盤面整體張力都可能相關，而模型無法索取它不知道自己缺少的東西。
    全量的代價約 400 tokens，遠低於一篇未切片文章（7,392）。
    """
    if not facts:
        return ""
    sp = facts.get("spec") or {}
    L: List[str] = []
    when = f"{sp.get('birthdate') or '?'} {sp.get('birth_time') or '時間未知'}"
    L.append(f"【星盤正典 {facts.get('chart_id')}】{sp.get('name') or ''} {when}"
             f" {sp.get('tz') or ''} {sp.get('houses_system') or ''}")
    L.append("※ 以下數字為唯一真相；與對話摘要或先前敘述衝突時，一律以此為準。")

    pts = facts.get("points") or []
    if pts:
        L.append("星體：" + "｜".join(_fmt_point(p) for p in pts))

    if facts.get("houses_available"):
        ang = facts.get("angles") or {}
        if ang:
            L.append("四軸：" + "｜".join(
                f"{_ANGLE_ABBR.get(k, k)} {(v.get('sign_zh') or v.get('sign') or '')}{v.get('deg', '')}"
                for k, v in ang.items()))
    else:
        L.append("宮位／四軸：**無資料**（出生時間或地點不可信，已從計算中移除）。"
                 "不可提及任何宮位、上升或天頂，也不可推測。")

    asps = facts.get("aspects") or []
    if asps:
        L.append("相位（由緊到鬆）：" + "｜".join(
            f"{a.get('p1')}{ASPECT_ZH.get(a.get('aspect'), a.get('aspect'))}"
            f"{a.get('p2')} {a.get('orb')}"
            for a in asps[:20]))

    lp = facts.get("lunar_phase") or {}
    if lp.get("name"):
        # 方法論原則 10：月相是獨立的一層（處理開始與結束的節奏）
        L.append(f"月相：{lp['name']}")

    dist = facts.get("distribution") or {}
    if dist:
        seg = []
        for grp in ("element", "mode", "quality", "polarity"):
            if dist.get(grp):
                seg.append(" ".join(f"{k}{v}" for k, v in dist[grp].items()))
        if seg:
            L.append("分布：" + "｜".join(seg))

    cr = (facts.get("derived") or {}).get("chart_ruler")
    if cr:
        L.append(f"命主星（推導）：{cr['body']} 在 {cr.get('sign')} "
                 f"{('H' + str(cr['house'])) if cr.get('house') else ''}")

    for k, v in (facts.get("flags") or {}).items():
        if isinstance(v, dict):
            # moon_uncertainty 只在真的跨星座時才值得佔版面；其餘 dict 旗標
            # （near_cusp_warning 等）一律取 note。這裡用 dict 分支而不是
            # 只認字串——near_cusp_warning 就是 dict，寫成字串判斷會靜默漏掉。
            if k == "moon_uncertainty" and not v.get("crosses_sign"):
                continue
            note = v.get("note")
            if note:
                L.append(f"⚠️ {note}")
        elif isinstance(v, str):
            L.append("⚠️ " + v)
    return "\n".join(L)


ASPECT_ZH = {
    "conjunction": "合", "sextile": "六分", "square": "四分",
    "trine": "三分", "opposition": "對分", "quintile": "五分",
    "semi-sextile": "半六分", "quincunx": "梅花", "sesquiquadrate": "補八分",
    "semi-square": "半四分", "biquintile": "倍五分",
}


# ---------------------------------------------------------------- session 膠水

def put(session: Dict[str, Any], facts: Dict[str, Any]) -> Tuple[str, bool]:
    """存入正典並設為使用中。回傳 (chart_id, 是否換了盤)。

    「換了盤」是重要的訊號：呼叫端必須據此讓歷史摘要失效，
    否則舊盤的解讀會透過摘要活下來，模型會拿它跟新盤混講。
    """
    cid = facts.get("chart_id") or ""
    store = session.setdefault("astro_facts", {})
    prev = session.get("astro_active_chart")
    store[cid] = facts
    session["astro_active_chart"] = cid
    if len(store) > 6:                      # 只留最近 6 張，避免 session 無限長大
        for k in list(store)[:-6]:
            store.pop(k, None)
    return cid, bool(prev and prev != cid)


def active(session: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    cid = session.get("astro_active_chart")
    return (session.get("astro_facts") or {}).get(cid) if cid else None


def on_active_chart_change(session: Dict[str, Any]) -> None:
    """換盤時清掉歷史摘要快取。

    摘要是舊盤事實的有損轉述；留著它，模型就會在新盤的對話裡讀到舊盤的數字。
    清掉的成本只是下回合重摘一次（背景池、獨立配額），很便宜。
    """
    session.pop("gm_history_summary", None)


def note_for_summarizer() -> str:
    """給摘要器的指令片段：不要把星盤數字寫進摘要。"""
    return ("星盤的行星位置、宮位、相位度數**不要寫進摘要**"
            "（另有正典來源會單獨提供）；只摘使用者的關注點與已達成的結論。")
