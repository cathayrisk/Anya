# -*- coding: utf-8 -*-
"""純即時災防題的確定性渲染：該類問題的答案由程式產生，不經模型（純函式，無 streamlit 依賴）。

## 為什麼還要這一步

第 4 步的 controller prefetch 保證了「一定查、一定進 context、一定留憑證」，但
**沒有保證模型會正確使用那些資料**。實測 09-05 就看到模型把 payload 裡的
`moving_direction: "SSW"` 寫成「向西南西南西方向移動」——這次只是措辭亂掉，
但同一個機制寫錯規模或時間也是完全可能的，而那看起來會跟正確答案一模一樣。

對「最近有地震嗎」這種題，模型其實沒有加值空間：答案就是把官方欄位唸出來。
既然如此就不要讓它有出錯的機會——**直接由程式渲染**。額外好處：
- 零 LLM 配額（免費層一天 20 次的模型很珍貴）
- 0.5 秒回答（實測經模型要 7～30 秒）
- 順手修掉方位詞：SSW → 西南偏南，程式查表不會亂寫

## 但適用範圍必須很窄

程式渲染只會唸欄位。碰到「最近地震這麼多，地震規模是怎麼定義的？」這種混合題，
模板會把知識那一半整個吃掉——**那是另一種答錯，而且使用者不會察覺少了什麼**。

所以 `is_pure_live()` 的設計原則是「有疑慮就交回模型」：模型手上已經有 prefetch
的官方資料（第 4 步保證的），交回去的成本只是慢一點、可能措辭不精確；而誤判成
純即時題的成本是答非所問。兩邊不對稱，閘門就該偏嚴。

判定為不純的常見原因（都刻意保留給模型）：
- 出現知識詞（定義、分級、怎麼形成…）
- 出現縣市名——地震 payload 是全國最新一筆，程式沒辦法只答「台南的部分」
- 出現接續詞（然後、另外、以及…）——通常代表第二個題目
- 扣掉災害詞／時間詞／語助詞之後還剩下東西，表示句子裡有模板涵蓋不到的內容
"""
from __future__ import annotations

import datetime as _dt
import re
from typing import Any

from utils import evidence as EV

_TPE = _dt.timezone(_dt.timedelta(hours=8))

# 縣市名出現 → 交回模型。地震資料是「全國最新一筆顯著有感地震」，
# 程式沒有能力回答「台南有沒有」，硬套模板會變成答非所問。
COUNTIES = ("臺北", "台北", "新北", "桃園", "臺中", "台中", "臺南", "台南", "高雄",
            "基隆", "新竹", "苗栗", "彰化", "南投", "雲林", "嘉義", "屏東", "宜蘭",
            "花蓮", "臺東", "台東", "澎湖", "金門", "連江", "馬祖")

# 接續詞：幾乎都代表句子裡還有第二個題目
CONJUNCTIONS = ("然後", "另外", "以及", "還有", "順便", "同時", "而且", "並且", "此外")

# 扣掉這些之後若還剩東西，就表示有模板涵蓋不到的內容。
# 注意這裡**不放**接續詞——那些要留著讓殘留量變大。
FILLER = ("台灣", "臺灣", "全台", "全臺", "幫我", "幫忙", "請問", "請", "想知道",
          "知道", "告訴我", "查一下", "查查", "查", "問一下", "一下", "目前",
          "狀況", "情況", "資訊", "消息", "報告", "如何", "怎樣", "多大", "多少",
          "在哪", "哪裡", "是不是", "有沒有", "有無", "是否", "會不會",
          "嗎", "呢", "喔", "啊", "吧", "了", "的", "有", "沒", "我", "你", "他",
          "要", "來", "去", "看", "說", "跟", "和", "與", "或")

MAX_RESIDUAL_CHARS = 4

_COMPASS = {
    "N": "北", "NNE": "東北偏北", "NE": "東北", "ENE": "東北偏東",
    "E": "東", "ESE": "東南偏東", "SE": "東南", "SSE": "東南偏南",
    "S": "南", "SSW": "西南偏南", "SW": "西南", "WSW": "西南偏西",
    "W": "西", "WNW": "西北偏西", "NW": "西北", "NNW": "西北偏北",
}


def _residual(text: str, intent) -> str:
    """扣掉災害詞、時間詞、動作詞、語助詞、標點與數字之後還剩什麼。"""
    t = text or ""
    sig = getattr(intent, "signals", {}) or {}
    words = list(sig.get("entities") or ()) + list(sig.get("time") or ()) + \
        list(sig.get("action") or ()) + list(FILLER)
    # 長詞先扣，否則「有沒有」會被「有」拆掉
    for w in sorted(words, key=len, reverse=True):
        t = t.replace(w, "")
    return re.sub(r"[\s\d\W_]+", "", t, flags=re.UNICODE)


def is_pure_live(text: str, intent) -> tuple[bool, str]:
    """能不能用程式模板直接回答？回傳 `(可以嗎, 原因)`——原因會進 dev 面板，
    之後要調閘門時才知道實際是被哪一條擋下來的。"""
    from utils import hazard_intent as HI
    from utils import hazard_prefetch as PF

    if getattr(intent, "state", None) != HI.STATE_LIVE:
        return False, f"state={getattr(intent, 'state', None)}（只有 explicit_live 才渲染）"
    if not intent.scopes:
        return False, "沒有可查的 scope"
    if intent.uncovered:
        return False, f"含無資料源的災害詞：{list(intent.uncovered)}"
    if any(sc not in PF.PREFETCHABLE for sc in intent.scopes):
        return False, "含程式無法代呼叫的 scope（如天氣需要地點）"
    if (intent.signals or {}).get("knowledge"):
        return False, f"含知識詞：{list(intent.signals['knowledge'])}"
    hit = [c for c in COUNTIES if c in (text or "")]
    if hit:
        return False, f"含縣市名 {hit}——地震資料是全國最新一筆，程式無法只答該縣市"
    hit = [c for c in CONJUNCTIONS if c in (text or "")]
    if hit:
        return False, f"含接續詞 {hit}，可能還有第二個問題"
    res = _residual(text, intent)
    if len(res) > MAX_RESIDUAL_CHARS:
        return False, f"殘留 {len(res)} 字「{res}」超過門檻，句子裡有模板涵蓋不到的內容"
    return True, "純即時災防題"


# ── 渲染 ────────────────────────────────────────────────────────────────────
def _fmt_time(raw: Any) -> str:
    s = str(raw or "").strip()
    if not s:
        return "—"
    try:
        return _dt.datetime.fromisoformat(s).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return s


def _squash(v: Any) -> str:
    """CWA 的 location 夾雜多個連續空白：「臺東縣政府東南東方  44.9  公里」。"""
    return re.sub(r"\s+", " ", str(v or "—")).strip()


def _earthquake(payload: dict) -> list[str]:
    areas, seen = [], set()
    for a in payload.get("shaking_areas") or []:
        key = (a.get("county"), a.get("intensity"))
        if key in seen:          # CWA 實測會回重複的縣市
            continue
        seen.add(key)
        areas.append(f"{a.get('county')} {a.get('intensity')}")
    out = [
        f"- 發生時間：{_fmt_time(payload.get('origin_time'))}",
        "- 震央：" + _squash(payload.get("location")),
        f"- 規模：芮氏 {payload.get('magnitude')}　深度 {payload.get('depth_km')} 公里",
    ]
    if areas:
        out.append("- 最大震度：" + "、".join(areas))
    if payload.get("report_image_uri"):
        out.append(f"- 震度圖：{payload['report_image_uri']}")
    return out


def _typhoon(payload: dict) -> list[str]:
    out = []
    if payload.get("has_active_taiwan_warning"):
        out.append(f"- ⚠️ **目前有對台生效的颱風警報**：{payload.get('last_bulletin_headline') or '—'}"
                   f"（發布時間 {_fmt_time(payload.get('last_bulletin_time'))}）")
        if payload.get("affected_areas"):
            out.append("- 影響區域：" + "、".join(str(a) for a in payload["affected_areas"] if a))
        for sec in (payload.get("description") or [])[:3]:
            val = str(sec.get("value") or "").strip().replace("\n", " ")
            out.append(f"- {sec.get('title') or '說明'}：{val[:200]}")
    else:
        out.append(f"- 目前**沒有**對台生效的颱風警報"
                   f"（最近一次公告：{payload.get('last_bulletin_headline') or '—'}，"
                   f"{_fmt_time(payload.get('last_bulletin_time'))}）")
    for c in (payload.get("tracked_cyclones") or [])[:3]:
        pos = c.get("latest_position") or {}
        zh, en = c.get("cwa_name"), c.get("name")
        name = f"{zh}（{en}）" if zh and en else (zh or en or "—")
        # 方位詞查表——模型實測會把 SSW 寫成「西南西南西」，程式不會
        d = _COMPASS.get(str(pos.get("moving_direction") or "").upper().strip())
        move = f"向{d}移動" if d else (f"移動方向 {pos.get('moving_direction')}"
                                    if pos.get("moving_direction") else "移動方向不明")
        speed = f"，時速 {pos['moving_speed_kmh']} 公里" if pos.get("moving_speed_kmh") else ""
        out.append(f"- 追蹤中的熱帶氣旋：**{name}**——位置 {pos.get('lat')}°N {pos.get('lon')}°E，"
                   f"近中心最大風速 {pos.get('max_wind_mps')} m/s，{move}{speed}"
                   f"（{_fmt_time(pos.get('time'))}）")
    if not (payload.get("tracked_cyclones") or []):
        out.append("- 目前西太平洋沒有追蹤中的熱帶氣旋。")
    return out


_SECTIONS = {
    EV.SCOPE_EARTHQUAKE: ("🌏 地震", _earthquake,
                          "氣象署這次沒有回傳可顯示的地震資料——**這代表資料暫時取不到，"
                          "不等於近期沒有發生地震**。"),
    EV.SCOPE_TYPHOON: ("🌀 颱風", _typhoon,
                       "氣象署這次沒有回傳颱風資料——**這代表資料暫時取不到，"
                       "不等於目前沒有颱風**。"),
}


def render(results: list[dict], *, payloads: dict, now: _dt.datetime | None = None) -> str:
    """把 prefetch 結果渲染成完整回答。`payloads` 是 scope → 已解析的 dict。

    回傳 "" 表示渲染不出東西（呼叫端應退回讓模型作答）。
    """
    now = now or _dt.datetime.now(_TPE)
    ts = now.astimezone(_TPE).strftime("%Y-%m-%d %H:%M")
    blocks = []
    for r in results or []:
        sec = _SECTIONS.get(r.get("scope"))
        if not sec:
            continue
        title, fn, empty_msg = sec
        lines = [f"**{title}**"]
        if r.get("status") == EV.STATUS_OK:
            try:
                lines += fn(payloads.get(r["scope"]) or {})
            except Exception:
                # 欄位長相變了就退回讓模型作答，不要吐半截模板
                return ""
        elif r.get("status") == EV.STATUS_EMPTY:
            lines.append(f"- {empty_msg}")
        else:
            lines.append(f"- **查詢失敗**：{r.get('error') or '未知錯誤'}。"
                         "這次沒有取得氣象署資料，請稍後再試。")
        blocks.append("\n".join(lines))
    if not blocks:
        return ""
    return ("WakuWaku！\n\n" + "\n\n".join(blocks)
            + f"\n\n:small[:gray[以上內容於 {ts} 由程式直接取自中央氣象署開放資料，"
            "未經模型改寫。]]\n\n安妮亞回覆完畢！🥜")
