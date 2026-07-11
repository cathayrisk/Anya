# pages/11_氣象通知.py
# -*- coding: utf-8 -*-
"""
氣象通知專屬頁面

設計原則：例外才浮出（exception-based）——颱風/特報等「偶而才有」的警示只在
生效時以橫幅出現，不佔常駐版面；常駐內容是每天都有用的地點天氣卡。

資料來源：
- 儀表板顯示（現在天氣/預報/降雨/地震/颱風/特報）→ 直接向 CWA 即時查詢（需 CWA_API_KEY）。
  比收集器快照更即時，且不依賴 Supabase。
- 「通知歷史」與全頁 st.toast → Supabase（GitHub Actions 收集器寫入的推播事件流）。

沒有 CWA_API_KEY 時自動切換示範資料（頂部會標示），?demo=1 也可強制預覽。
"""

from __future__ import annotations

import html as _html
import os
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# 讓本頁可獨立執行（streamlit run pages/11_氣象通知.py）：把專案根目錄加進 sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from utils.cwa_weather import (
    _nearest_county,
    get_current_conditions,
    get_cwa_api_key,
    get_earthquake_impl,
    get_forecast_periods,
    get_health_indices,
    get_rain,
    get_typhoon_impl,
    get_warning_details,
    get_warnings,
)
from utils.rich_styles import inject_rich_styles
from utils import theme_tokens as tt
from utils.weather_toast import (
    _get_weather_supabase_client,
    get_secret_safe,
    render_weather_toast_watcher,
)

st.set_page_config(page_title="氣象通知", page_icon="🌦️", layout="wide")
inject_rich_styles()
render_weather_toast_watcher()

RAIN_STATE_LABEL = {"raining": "🌧️ 降雨中", "soon_rain": "☁️ 即將降雨", "dry": "☀️ 無降雨"}

# 頁面監控的地點（座標）。想加地點就往這裡加。
_LOCATION_COORDS: dict[str, tuple[float, float]] = {
    "預設地點": (25.037438, 121.553563),
}

# 降雨狀態門檻（對齊收集器 config.yaml 的預設；頁面只顯示現況、不做通知去重）
RAIN_NOW_MM = 0.5
RAIN_SOON_MM = 1.0


def _rain_state(observed, forecast) -> str:
    if (observed or 0.0) >= RAIN_NOW_MM:
        return "raining"
    if (forecast or 0.0) >= RAIN_SOON_MM:
        return "soon_rain"
    return "dry"


# ──────────────────────────────────────────────────────────────────────────
# 即時資料載入（直接向 CWA；各自快取）
# ──────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner="讀取即時天氣中…")
def _live_location(name: str, lat: float, lon: float) -> dict:
    county = _nearest_county(lat, lon)
    out = {
        "name": name, "county": county, "current": None, "periods": [],
        "rain": {"observed_mm": 0.0, "forecast_mm": 0.0, "state": "dry"},
        "warnings": [], "data_time": None, "health": {"primary": None, "extra": []},
    }
    try:
        out["current"] = get_current_conditions(lat, lon)
    except Exception:
        pass
    try:
        out["periods"] = get_forecast_periods(county)
    except Exception:
        pass
    try:
        r = get_rain(lat, lon)
        obs, fc = r.get("observed_past_1hr_mm"), r.get("forecast_next_1hr_mm")
        out["rain"] = {
            "observed_mm": obs if obs is not None else 0.0,
            "forecast_mm": fc if fc is not None else 0.0,
            "state": _rain_state(obs, fc),
        }
        out["data_time"] = r.get("data_time")
    except Exception:
        pass
    try:
        out["warnings"] = get_warnings(county)
    except Exception:
        pass
    try:
        out["health"] = get_health_indices(lat, lon)
    except Exception:
        out["health"] = {"primary": None, "extra": []}
    return out


@st.cache_data(ttl=120, show_spinner=False)
def _live_earthquake():
    try:
        e = get_earthquake_impl()
    except Exception:
        return None
    if not e.get("found"):
        return None
    return {
        "location": e.get("location"), "magnitude": e.get("magnitude"),
        "depth": e.get("depth_km"), "origin_time": e.get("origin_time"),
        "report_image_uri": e.get("report_image_uri"),
    }


@st.cache_data(ttl=300, show_spinner=False)
def _live_typhoon():
    try:
        t = get_typhoon_impl()
    except Exception:
        return None
    return {
        "active": t.get("has_active_taiwan_warning", False),
        "headline": t.get("last_bulletin_headline"),
        "effective": t.get("last_bulletin_time"),
        # 警報生效時才有值：公告段落（颱風動態/警戒區域…）與影響區域
        "description": t.get("description") or [],
        "affected_areas": t.get("affected_areas") or [],
    }


@st.cache_data(ttl=300, show_spinner=False)
def _live_warning_details() -> list:
    """全台各別特報公告（W-C0033-002），含內容全文與影響區域。失敗回空 list。"""
    try:
        return get_warning_details()
    except Exception:
        return []


@st.cache_data(ttl=60, show_spinner=False)
def _load_alerts() -> list:
    client = _get_weather_supabase_client()
    return (
        client.table("weather_alerts")
        .select("id, ts, category, title, body, extra_url")
        .order("id", desc=True)
        .limit(20)
        .execute()
    ).data or []


def load_alerts():
    """通知歷史（Supabase）。未設定 Supabase 回傳 None、查詢失敗也回 None。"""
    if not (get_secret_safe("SUPABASE_URL") and get_secret_safe("SUPABASE_KEY")):
        return None
    try:
        return _load_alerts()
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────
# 示範資料（沒有 CWA_API_KEY 或 ?demo=1 時）
# ──────────────────────────────────────────────────────────────────────────
def _demo_current_conditions() -> dict:
    now = datetime.now().astimezone()
    return {
        "station_name": "臺北", "station_distance_km": 1.8,
        "obs_time": (now - timedelta(minutes=12)).isoformat(timespec="seconds"),
        "weather": "陰", "air_temperature_c": "27.4", "feels_like_c": 31.5,
        "relative_humidity_pct": "88", "wind_speed_mps": "2.3", "uv_index": "3",
    }


def _demo_forecast_periods() -> list[dict]:
    now = datetime.now().astimezone()
    base = now.replace(minute=0, second=0, microsecond=0)

    def iso(dt):
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    return [
        {"start_time": iso(base), "end_time": iso(base + timedelta(hours=12)),
         "weather": "短暫陣雨", "pop": "70", "min_temp": "26", "max_temp": "30",
         "min_feel": "28", "max_feel": "34", "comfort": "舒適至悶熱"},
        {"start_time": iso(base + timedelta(hours=12)), "end_time": iso(base + timedelta(hours=24)),
         "weather": "多雲時晴", "pop": "30", "min_temp": "25", "max_temp": "32",
         "min_feel": "27", "max_feel": "38", "comfort": "悶熱"},
        {"start_time": iso(base + timedelta(hours=24)), "end_time": iso(base + timedelta(hours=36)),
         "weather": "晴時多雲", "pop": "10", "min_temp": "26", "max_temp": "33",
         "min_feel": "28", "max_feel": "39", "comfort": "悶熱"},
    ]


def _demo_bundle():
    now = datetime.now().astimezone()

    def ago(**kw):
        return (now - timedelta(**kw)).isoformat(timespec="seconds")

    locations = [{
        "name": "預設地點", "county": "臺北市",
        "current": _demo_current_conditions(),
        "periods": _demo_forecast_periods(),
        "rain": {"observed_mm": 4.2, "forecast_mm": 6.8, "state": "raining"},
        "warnings": [{"phenomena": "大雨", "significance": "特報",
                      "start_time": ago(hours=2), "end_time": ago(hours=-4)}],
        "data_time": ago(minutes=8),
        "health": {
            "primary": {"kind": "heat", "label": "熱傷害", "emoji": "🥵",
                        "index": 30, "warning": "", "peak_index_24h": 35,
                        "peak_warning_24h": "警戒", "peak_time_24h": ago(hours=-18)},
            "extra": [{"kind": "tempdiff", "label": "溫差提醒", "emoji": "🌡️",
                       "index": 8, "warning": "注意", "peak_index_24h": 9,
                       "peak_warning_24h": "警戒", "peak_time_24h": ago(hours=-6)}],
        },
    }]
    earthquake = {
        "location": "花蓮縣政府南南西方 41.0 公里 (位於花蓮縣豐濱鄉)",
        "magnitude": 5.1, "depth": 15.5, "origin_time": ago(hours=2),
        "report_image_uri": "https://scweb.cwa.gov.tw/webdata/OLDEQ/202607/2026070207060640048_H.png",
    }
    typhoon = {
        "active": True, "headline": "海上颱風警報", "effective": ago(hours=5),
        "description": [
            {"title": "颱風動態", "value": "第7號颱風目前位於鵝鑾鼻東南方海面，向西北移動，暴風圈正逐漸接近臺灣東南部海面。"},
            {"title": "注意事項", "value": "臺灣東南部海面航行及作業船隻應嚴加戒備。"},
        ],
        "affected_areas": ["臺灣東南部海面", "巴士海峽"],
    }
    warning_details = [{
        "title": "豪雨特報",
        "content": "一、概述：受鋒面影響，臺北市、新北市山區有局部大雨或豪雨發生的機率，請注意坍方及落石。\n\n二、注意(警戒)事項：山區請防範坍方、落石、土石流；低窪地區請慎防淹水。",
        "affected": ["臺北市", "新北市"],
        "issue_time": ago(hours=2), "start_time": ago(hours=2), "end_time": ago(hours=-4),
    }]
    alerts = [
        {"id": 5, "ts": ago(minutes=8), "category": "rain",
         "title": "降雨提醒 - 預設地點", "body": "偵測到降雨中,過去1小時累積雨量約 4.2mm。", "extra_url": None},
        {"id": 4, "ts": ago(minutes=42), "category": "rain",
         "title": "降雨提醒 - 預設地點", "body": "預計1小時內可能開始下雨,預測累積雨量約 2.1mm。", "extra_url": None},
        {"id": 3, "ts": ago(hours=2), "category": "warning",
         "title": "天氣特報 - 臺北市", "body": "大雨特報生效中，有效至今晚。", "extra_url": None},
        {"id": 2, "ts": ago(hours=2, minutes=10), "category": "earthquake",
         "title": "地震通知 - 預設地點", "body": "花蓮縣豐濱鄉發生規模 5.1 地震（深度 15.5km），臺北市 測得震度 4級。",
         "extra_url": "https://scweb.cwa.gov.tw/webdata/OLDEQ/202607/2026070207060640048_H.png"},
        {"id": 1, "ts": ago(hours=5), "category": "typhoon",
         "title": "颱風公告 - 海上颱風警報", "body": "第7號颱風海上警報發布，暴風圈朝台灣東部海面接近，請注意後續動態。", "extra_url": None},
    ]
    return locations, earthquake, typhoon, warning_details, alerts


_TAIPEI = ZoneInfo("Asia/Taipei")


def _fmt_time(value) -> str:
    """格式化顯示時間，統一轉成台北時間。
    Supabase 的 ts 存的是 UTC（帶 +00:00），CWA 自己的欄位大多是不含時區的
    naive 字串但本來就是台北當地時間——只對「有時區資訊」的值做轉換，
    naive 值視為已經是台北時間，不再二次位移。"""
    if not value:
        return "—"
    try:
        dt = datetime.fromisoformat(str(value))
        if dt.tzinfo is not None:
            dt = dt.astimezone(_TAIPEI)
        return dt.strftime("%m/%d %H:%M")
    except ValueError:
        return str(value)


_WEEKDAY = ["一", "二", "三", "四", "五", "六", "日"]


def _period_label(start_str: str) -> str:
    """預報時段 → 「7/7（一）晚上」這種實際日期標籤。"""
    try:
        start = datetime.fromisoformat(start_str)
    except (TypeError, ValueError):
        return str(start_str)
    if 6 <= start.hour < 18:
        part = "白天"
    elif start.hour < 6:
        part = "凌晨"
    else:
        part = "晚上"
    return f"{start.month}/{start.day}（{_WEEKDAY[start.weekday()]}）{part}"


def _period_hours(start_str: str, end_str: str) -> str:
    try:
        s = datetime.fromisoformat(start_str)
        e = datetime.fromisoformat(end_str)
        return f"{s.hour:02d}:00–{e.hour:02d}:00"
    except (TypeError, ValueError):
        return ""


# ──────────────────────────────────────────────────────────────────────────
# 地點天氣卡（自製 HTML/CSS：st.metric 卡片高度不齊，改用 grid 完全對齊）
# ──────────────────────────────────────────────────────────────────────────
_WX_EMOJI = [("雷", "⛈️"), ("雨", "🌧️"), ("雪", "❄️"), ("霧", "🌫️"), ("陰", "☁️"), ("多雲", "⛅"), ("晴", "☀️")]


def _weather_emoji(desc: str) -> str:
    for kw, emo in _WX_EMOJI:
        if kw in (desc or ""):
            return emo
    return "🌤️"


def _feel_emoji(max_feel, min_feel) -> str:
    """體感溫度 emoji：依高溫端判斷熱、低溫端判斷冷。"""
    try:
        hi = float(max_feel)
        lo = float(min_feel)
    except (TypeError, ValueError):
        return "🌡️"
    if hi >= 38:
        return "🥵"
    if lo <= 12:
        return "🥶"
    if hi >= 32:
        return "😓"
    if lo <= 18:
        return "🧥"
    return "🙂"


# 舒適度描述（寒→熱）→ emoji；比對關鍵字，越後面越嚴重優先
_COMFORT_EMOJI = [
    ("中暑", "🥵"), ("悶熱", "😓"), ("舒適", "😊"),
    ("寒意", "🧥"), ("寒冷", "🥶"), ("寒", "🥶"),
]


def _comfort_emoji(desc: str) -> str:
    d = desc or ""
    for kw, emo in _COMFORT_EMOJI:
        if kw in d:
            return emo
    return "😐"


# 設計原則:警報層(danger/watch/notice)沿用消防/交通號誌等級的紅/橘/藍語意色,
# 不強行套品牌珊瑚色,安全警示的辨識度優先;其餘卡片表面回到全站珊瑚粉/金邊/深褐
# 主題,與 Anya Forger 人格化風格一致;內文對比一律 ≥4.5:1。
# 顏色一律引用 utils/theme_tokens.py（單一事實來源），不在這裡重複硬編碼 hex——
# 用 %()s 而非 f-string，避免跟 CSS 本身滿版的 { } 起衝突。
_WX_CSS_TEMPLATE = """
<style>
/* 卡片坐在淡粉近白頁面底上，用淡珊瑚描邊分離。 */
.wx-card{border:1px solid %(border)s;border-radius:14px;background:linear-gradient(180deg,#FFFFFF 0%%,%(bg_card_end)s 100%%);
  padding:18px 20px 14px;margin-bottom:14px;}
.wx-head{display:flex;align-items:baseline;gap:10px;margin-bottom:14px;}
.wx-name{font-size:1.05rem;font-weight:700;color:%(brown)s;}
.wx-county{font-size:.85rem;color:%(muted)s;}
.wx-body{display:flex;flex-direction:column;gap:14px;}
.wx-now{display:flex;flex-wrap:wrap;align-items:center;gap:22px;
  padding:18px 24px;border-radius:12px;background:%(bg_panel)s;}
.wx-now-primary{display:flex;align-items:center;gap:14px;}
.wx-now-emoji{font-size:2.6rem;line-height:1;}
.wx-now-temp{font-size:2.7rem;font-weight:700;color:%(brown)s;line-height:1;}
.wx-now-desc-wrap{display:flex;flex-direction:column;gap:2px;}
.wx-now-desc{font-size:1.05rem;font-weight:600;color:%(coral_text)s;}
.wx-now-feel{font-size:.8rem;color:%(muted)s;}
.wx-now-mid{flex:1;min-width:200px;display:flex;flex-direction:column;gap:6px;}
.wx-now-range{font-size:.8rem;color:%(muted)s;display:flex;gap:8px;}
.wx-now-stats{display:flex;flex-wrap:wrap;column-gap:14px;row-gap:4px;font-size:.84rem;color:%(muted)s;}
.wx-now-stats span{white-space:nowrap;}
.wx-now-side{display:flex;flex-direction:column;gap:6px;align-items:flex-start;min-width:160px;}
.wx-chip{display:inline-block;margin-top:4px;padding:4px 10px;border-radius:99px;font-size:.8rem;font-weight:600;width:fit-content;}
.wx-chip.rain-on{background:%(rain_blue_bg)s;color:%(rain_blue)s;}
.wx-chip.rain-soon{background:%(warn_amber_bg)s;color:%(warn_amber)s;}
.wx-chip.rain-off{background:%(neutral_chip_bg)s;color:%(muted)s;}
/* 健康警示：固定一個位置(stats排之後、降雨chip之前)，狀態改變只換樣式深淺，不搬家 */
.wx-health-slot{display:flex;flex-direction:column;gap:3px;margin-top:2px;}
.wx-health-calm{font-size:.78rem;color:%(muted)s;}
.wx-health-section-label{font-size:.7rem;font-weight:700;color:%(muted)s;letter-spacing:.03em;margin-top:2px;}
.wx-health-row{display:flex;align-items:center;justify-content:space-between;gap:8px;
  padding:4px 8px;border-radius:8px;font-size:.8rem;}
.wx-health-row.active{background:%(health_active_bg)s;}
.wx-health-row .label{display:flex;align-items:center;gap:5px;font-weight:600;color:%(brown)s;white-space:nowrap;}
.wx-health-row .meta{font-size:.74rem;color:%(muted)s;white-space:nowrap;}
.wx-health-escalate{font-size:.7rem;color:%(warn_amber)s;padding:0 8px 2px;}
.wx-badge{font-size:.72rem;font-weight:700;padding:2px 8px;border-radius:99px;white-space:nowrap;}
.wx-badge.fill-danger{background:%(danger_bg)s;color:%(danger)s;}
.wx-badge.fill-warn{background:%(warn_amber_bg)s;color:%(warn_amber)s;}
.wx-badge.fill-caution{background:%(caution_bg)s;color:%(caution)s;}
.wx-badge.fill-safe{background:%(neutral_chip_bg)s;color:%(muted)s;}
.wx-badge.outline-danger{background:transparent;border:1px solid %(outline_danger)s;color:%(danger)s;}
.wx-badge.outline-warn{background:transparent;border:1px solid %(outline_warn)s;color:%(warn_amber)s;}
.wx-badge.outline-caution{background:transparent;border:1px solid %(outline_caution)s;color:%(caution)s;}
.wx-badge.outline-safe{background:transparent;border:1px solid %(outline_safe)s;color:%(muted)s;}
.wx-periods{width:100%%;display:grid;grid-template-columns:repeat(3,1fr);gap:10px;}
.wx-period{border:1px solid %(border)s;border-radius:12px;background:#FFFFFF;padding:12px 14px;
  display:flex;flex-direction:column;gap:5px;}
.wx-p-head{display:flex;justify-content:space-between;align-items:baseline;gap:6px;}
.wx-p-label{font-size:.85rem;font-weight:700;color:%(brown)s;}
.wx-p-pop{font-size:.78rem;color:%(info_blue)s;white-space:nowrap;}
.wx-p-hours{font-size:.72rem;color:%(muted)s;}
.wx-p-wx{font-size:.98rem;font-weight:600;color:%(coral_text)s;display:flex;align-items:center;gap:6px;}
.wx-p-temp{font-size:.82rem;color:%(muted)s;}
.wx-p-feel{font-size:.78rem;color:%(muted)s;}
.wx-p-comfort{font-size:.76rem;color:%(muted)s;}
.wx-src{margin-top:12px;font-size:.74rem;color:%(muted)s;}
.wx-t-row{display:flex;align-items:center;gap:7px;font-size:.8rem;color:%(muted)s;}
.wx-t-row .t{font-weight:600;}
.wx-t-track{flex:1;height:6px;border-radius:99px;background:%(neutral_chip_bg)s;position:relative;min-width:50px;}
.wx-t-fill{position:absolute;top:0;height:100%%;border-radius:99px;}
.wx-wk{display:flex;flex-direction:column;gap:2px;}
.wx-wk-row{display:grid;grid-template-columns:104px 26px minmax(70px,1fr) 52px 34px minmax(80px,1.4fr) 34px;
  align-items:center;gap:8px;padding:7px 10px;border-radius:10px;font-size:.86rem;color:%(brown)s;}
.wx-wk-row:nth-child(even){background:%(stripe)s;}
.wx-wk-day{font-weight:600;}
.wx-wk-desc{color:%(muted)s;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.wx-wk-pop{color:%(info_blue)s;font-size:.8rem;text-align:right;white-space:nowrap;}
.wx-wk-t{color:%(muted)s;font-weight:600;text-align:right;}
@media (max-width:720px){.wx-now{flex-direction:column;align-items:flex-start;gap:12px;}
  .wx-periods{grid-template-columns:1fr;}
  .wx-wk-row{grid-template-columns:88px 24px 46px 30px 1fr 30px;}.wx-wk-desc{display:none;}}
/* ── 警報嚴重度系統:明文等級+圖示+成對色,不只靠顏色 ── */
.al-banner{display:flex;gap:12px;align-items:flex-start;border:1px solid;border-radius:12px;
  padding:12px 16px;margin-bottom:8px;}
.al-danger{background:%(alert_danger_bg)s;border-color:%(alert_danger_border)s;}
.al-watch{background:%(alert_watch_bg)s;border-color:%(alert_watch_border)s;}
.al-notice{background:%(alert_notice_bg)s;border-color:%(alert_notice_border)s;}
.al-icon{font-size:1.35rem;line-height:1.25;}
.al-main{flex:1;display:flex;flex-direction:column;gap:3px;min-width:0;}
.al-row1{display:flex;align-items:center;gap:8px;flex-wrap:wrap;}
.al-chip{font-size:.75rem;font-weight:700;padding:2px 10px;border-radius:99px;color:#FFFFFF;white-space:nowrap;}
.al-danger .al-chip{background:%(danger)s;}
.al-watch .al-chip{background:%(alert_watch_chip)s;}
.al-notice .al-chip{background:%(alert_notice_chip)s;}
.al-title{font-weight:700;color:%(brown)s;font-size:.98rem;}
.al-local{font-size:.75rem;font-weight:700;color:%(danger)s;background:#FFFFFF;border:1px solid %(alert_danger_border)s;
  padding:1px 8px;border-radius:99px;white-space:nowrap;}
.al-meta{font-size:.82rem;color:%(muted)s;}
.al-meta b{color:%(brown)s;}
/* 在地決策列:此刻、此地、有沒有事 */
.al-strip{display:flex;align-items:center;gap:10px;border:1px solid;border-radius:12px;
  padding:12px 16px;margin-bottom:12px;font-size:.95rem;}
.al-strip.ok{background:%(success_bg)s;border-color:%(alert_success_border)s;color:%(success)s;}
.al-strip.hit{background:%(alert_danger_bg)s;border-color:%(alert_danger_border)s;color:%(alert_strip_hit_text)s;}
.al-strip b{font-size:1.02rem;}
</style>
""" % {
    "border": tt.BORDER,
    "bg_card_end": tt.BG_CARD_END,
    "brown": tt.BROWN,
    "muted": tt.MUTED,
    "bg_panel": tt.BG_PANEL,
    "coral_text": tt.CORAL_TEXT,
    "rain_blue_bg": tt.RAIN_BLUE_BG,
    "rain_blue": tt.RAIN_BLUE,
    "warn_amber_bg": tt.WARN_AMBER_BG,
    "warn_amber": tt.WARN_AMBER,
    "neutral_chip_bg": tt.NEUTRAL_CHIP_BG,
    "health_active_bg": tt.HEALTH_ACTIVE_BG,
    "danger_bg": tt.DANGER_RED_BG,
    "danger": tt.DANGER_RED,
    "caution_bg": tt.CAUTION_GOLD_BG,
    "caution": tt.CAUTION_GOLD,
    "outline_danger": tt.OUTLINE_DANGER_BORDER,
    "outline_warn": tt.OUTLINE_WARN_BORDER,
    "outline_caution": tt.OUTLINE_CAUTION_BORDER,
    "outline_safe": tt.OUTLINE_SAFE_BORDER,
    "info_blue": tt.INFO_BLUE,
    "stripe": tt.STRIPE,
    "alert_danger_bg": tt.ALERT_DANGER_BG,
    "alert_danger_border": tt.ALERT_DANGER_BORDER,
    "alert_watch_bg": tt.ALERT_WATCH_BG,
    "alert_watch_border": tt.ALERT_WATCH_BORDER,
    "alert_watch_chip": tt.ALERT_WATCH_CHIP,
    "alert_notice_bg": tt.ALERT_NOTICE_BG,
    "alert_notice_border": tt.ALERT_NOTICE_BORDER,
    "alert_notice_chip": tt.ALERT_NOTICE_CHIP,
    "success_bg": tt.SUCCESS_GREEN_BG,
    "success": tt.SUCCESS_GREEN,
    "alert_success_border": tt.ALERT_SUCCESS_BORDER,
    "alert_strip_hit_text": tt.ALERT_STRIP_HIT_TEXT,
}
_WX_CSS = _WX_CSS_TEMPLATE

_RAIN_CHIP_CLASS = {"raining": "rain-on", "soon_rain": "rain-soon", "dry": "rain-off"}


# ──────────────────────────────────────────────────────────────────────────
# 警報嚴重度元件(等級=本頁自訂的相關性×災種映射,非 CWA 官方分級)
# ──────────────────────────────────────────────────────────────────────────
def _remaining_label(end_str) -> str:
    """有效期限剩不到 3 小時 → 顯示倒數,時間成為一級資訊。"""
    try:
        end = datetime.fromisoformat(str(end_str))
    except (TypeError, ValueError):
        return ""
    now = datetime.now(_TAIPEI) if end.tzinfo else datetime.now()
    secs = (end - now).total_seconds()
    if secs <= 0:
        return ""
    if secs <= 3 * 3600:
        return f"(剩約 {secs / 3600:.0f} 小時)" if secs >= 3600 else f"(剩約 {secs / 60:.0f} 分)"
    return ""


def _affected_summary(affected: list, watched: set) -> str:
    """地名牆 → 摘要:命中的監控縣市排最前,其餘收成「等 N 地」。"""
    if not affected:
        return "詳見內容全文"
    hits = [c for c in affected if c in watched]
    lead = hits[0] if hits else affected[0]
    return lead if len(affected) == 1 else f"{lead} 等 {len(affected)} 地"


def _alert_banner_html(sev: str, icon: str, title: str, affected_txt: str, time_txt: str, local: bool) -> str:
    """嚴重度橫幅:明文等級 chip + 圖示 + 在地標記,不只靠顏色編碼。"""
    e = _html.escape
    label = {"danger": "危險", "watch": "警戒", "notice": "注意"}[sev]
    local_tag = "<span class='al-local'>📍 影響你的地點</span>" if local else ""
    meta_bits = [b for b in (affected_txt, time_txt) if b]
    meta = f"<div class='al-meta'>{'　'.join(meta_bits)}</div>" if meta_bits else ""
    return (
        f"<div class='al-banner al-{sev}'><span class='al-icon'>{icon}</span>"
        f"<div class='al-main'><div class='al-row1'><span class='al-chip'>{label}</span>"
        f"<span class='al-title'>{e(title)}</span>{local_tag}</div>{meta}</div></div>"
    )

# 溫度 → 端點顏色（冷藍 → 金 → 暖橘 → 珊瑚紅），溫度帶兩端各取自己溫度的顏色
_TEMP_STOPS = tt.TEMP_GRADIENT_STOPS


def _temp_color(t: float) -> str:
    if t <= _TEMP_STOPS[0][0]:
        rgb = _TEMP_STOPS[0][1]
    elif t >= _TEMP_STOPS[-1][0]:
        rgb = _TEMP_STOPS[-1][1]
    else:
        rgb = _TEMP_STOPS[-1][1]
        for (t0, c0), (t1, c1) in zip(_TEMP_STOPS, _TEMP_STOPS[1:]):
            if t0 <= t <= t1:
                k = (t - t0) / (t1 - t0)
                rgb = tuple(round(a + (b - a) * k) for a, b in zip(c0, c1))
                break
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def _temp_scale(periods: list[dict]) -> tuple[float, float] | None:
    """卡片/週列共用刻度：取所有時段的最低~最高溫，兩端各留 1°C。"""
    vals = []
    for p in periods:
        for k in ("min_temp", "max_temp"):
            try:
                vals.append(float(p.get(k)))
            except (TypeError, ValueError):
                pass
    if not vals:
        return None
    return min(vals) - 1.0, max(vals) + 1.0


def _temp_bar_html(min_t, max_t, scale_min: float, scale_max: float) -> str:
    """共同刻度上的溫度帶；溫度缺漏時回空字串（呼叫端退回純文字）。"""
    try:
        lo, hi = float(min_t), float(max_t)
    except (TypeError, ValueError):
        return ""
    span = max(scale_max - scale_min, 1e-6)
    left = min(max(0.0, (lo - scale_min) / span * 100), 96.0)
    width = max(4.0, (hi - lo) / span * 100)
    width = min(width, 100.0 - left)
    return (
        f"<span class='wx-t-track'><span class='wx-t-fill' style='left:{left:.1f}%;width:{width:.1f}%;"
        f"background:linear-gradient(90deg,{_temp_color(lo)},{_temp_color(hi)});'></span></span>"
    )


_SEVERITY_RANK = {"": 0, "注意": 1, "警戒": 2, "危險": 3}


def _severity_rank(warning: str) -> int:
    # 未知字串（CWA 未來若換新詞）保守當作至少「注意」等級，不要低估
    return _SEVERITY_RANK.get(warning or "", 1)


def _health_severity_class(warning: str) -> str:
    """官方警示文字 → 嚴重度樣式代碼。健康指數的數字大小不代表嚴重度（冷傷害
    指數越高反而越安全），一律以 CWA 給的警示文字判斷，不自己編級距。"""
    w = warning or ""
    if "危險" in w:
        return "danger"
    if "警戒" in w:
        return "warn"
    if "注意" in w:
        return "caution"
    return "safe"


def _health_relative_day(dt: datetime, now: datetime) -> str:
    delta = (dt.date() - now.date()).days
    if delta == 0:
        return "今天"
    if delta == 1:
        return "明天"
    return f"{dt.month}/{dt.day}"


def _health_peak_when(h: dict, now: datetime) -> str:
    """peak_time_24h 是『預測最嚴重的那個時間點』，不是警示開始生效的時刻，
    用詞避免『起』這種暗示持續生效的字眼。"""
    try:
        dt = datetime.fromisoformat(h.get("peak_time_24h"))
        if dt.tzinfo is not None:
            dt = dt.astimezone(_TAIPEI)
        return f"{_health_relative_day(dt, now)} {dt.strftime('%H:%M')}"
    except (TypeError, ValueError):
        return "稍後"


def _health_row_html(h: dict, now: datetime) -> str:
    """單一指數一列：現在生效中→實心 badge（若預測會再惡化，底下多一行升級提示，
    不會因為現在只是『注意』就丟失『稍後會到危險』這個資訊）；現在安全但24h內會
    達標→outline badge 標時間。固定放在同一個位置，只有樣式深淺隨急迫程度變化。"""
    e = _html.escape
    emoji, label = h.get("emoji", "🩺"), h.get("label", "健康指數")
    warning, peak = h.get("warning") or "", h.get("peak_warning_24h") or ""

    if warning:
        sev = _health_severity_class(warning)
        row = (
            f"<div class='wx-health-row active'><span class='label'>{emoji} {e(label)}</span>"
            f"<span class='wx-badge fill-{sev}'>{e(warning)}中</span></div>"
        )
        if peak and _severity_rank(peak) > _severity_rank(warning):
            row += f"<div class='wx-health-escalate'>{e(_health_peak_when(h, now))} 預計升至{e(peak)}</div>"
        return row

    if peak:
        sev = _health_severity_class(peak)
        return (
            f"<div class='wx-health-row'><span class='label'>{emoji} {e(label)}</span>"
            f"<span class='meta'>{e(_health_peak_when(h, now))} 預計達</span>"
            f"<span class='wx-badge outline-{sev}'>{e(peak)}</span></div>"
        )
    return ""


def _health_slot_html(health: dict) -> str:
    """固定位置的健康警示區塊：一定會顯示東西（沒事時是一行安靜確認，不是完全消失
    ——沉默無法區分『查過沒事』跟『這功能沒在運作』）。多項時分組列出，不擠成一句
    長文或一排 pill 雲。"""
    if not health:
        return ""
    items = [h for h in [health.get("primary"), *(health.get("extra") or [])] if h]
    if not items:
        return ""
    now = datetime.now(_TAIPEI)

    active = [h for h in items if h.get("warning")]
    upcoming = [h for h in items if not h.get("warning") and h.get("peak_warning_24h")]

    if not active and not upcoming:
        return "<div class='wx-health-slot'><div class='wx-health-calm'>✓ CWA 健康警示｜24小時內無警示</div></div>"

    rows = []
    if active:
        rows.append(f"<div class='wx-health-section-label'>健康警示・現在（{len(active)}項）</div>" if len(active) > 1 else "")
        rows.extend(_health_row_html(h, now) for h in active)
    if upcoming:
        rows.append("<div class='wx-health-section-label'>健康提醒・未來24小時</div>")
        rows.extend(_health_row_html(h, now) for h in upcoming)
    return f"<div class='wx-health-slot'>{''.join(rows)}</div>"


def _build_location_card_html(name, county, current, rain, periods, source_bits, health=None) -> str:
    e = _html.escape

    health_html = _health_slot_html(health or {})

    rain_state = rain.get("state", "dry")
    # 乾的時候不列一串 0.0mm——數字只在真的有雨量意義時才出現
    if rain_state == "raining":
        rain_amounts = (
            f"　{e(str(rain.get('observed_mm', 0)))} mm"
            f"・未來1hr {e(str(rain.get('forecast_mm', 0)))} mm"
        )
    elif rain_state == "soon_rain":
        rain_amounts = f"　未來1hr 約 {e(str(rain.get('forecast_mm', 0)))} mm"
    else:
        rain_amounts = ""
    rain_chip = (
        f"<span class='wx-chip {_RAIN_CHIP_CLASS.get(rain_state, 'rain-off')}'>"
        f"{e(RAIN_STATE_LABEL.get(rain_state, '降雨'))}{rain_amounts}</span>"
    )

    if current:
        desc = current.get("weather") or "—"
        stats = [
            f"<span>💧 {e(str(current.get('relative_humidity_pct') or '—'))}%</span>",
            f"<span>🍃 {e(str(current.get('wind_speed_mps') or '—'))} m/s</span>",
        ]
        if current.get("uv_index") is not None:
            stats.append(f"<span>🔆 UV {e(str(current.get('uv_index')))}</span>")
        if current.get("peak_gust_mps") is not None:
            stats.append(f"<span>💨 陣風 {e(str(current.get('peak_gust_mps')))} m/s</span>")
        feels = current.get("feels_like_c")
        # O-A0003-001 即時觀測本身沒有體感溫度欄位，用氣溫/濕度/風速依 CWA 公式自算
        feel_html = (
            f"<div class='wx-now-feel'>{_feel_emoji(feels, feels)} 體感 {e(str(feels))}°C</div>"
            if feels is not None else ""
        )
        hi, lo = current.get("daily_high_c"), current.get("daily_low_c")
        hi_time, lo_time = _fmt_time(current.get("daily_high_time")), _fmt_time(current.get("daily_low_time"))
        range_parts = []
        if hi is not None:
            range_parts.append(f"<span title='{e(hi_time)} 測得'>↑{e(str(hi))}°</span>")
        if lo is not None:
            range_parts.append(f"<span title='{e(lo_time)} 測得'>↓{e(str(lo))}°</span>")
        range_html = f"<div class='wx-now-range'>🌡️ 今日 {' '.join(range_parts)}</div>" if range_parts else ""
        now_html = f"""
        <div class="wx-now">
          <div class="wx-now-primary">
            <span class="wx-now-emoji">{_weather_emoji(desc)}</span>
            <span class="wx-now-temp">{e(str(current.get('air_temperature_c', '—')))}°</span>
            <div class="wx-now-desc-wrap">
              <div class="wx-now-desc">{e(desc)}</div>
              {feel_html}
            </div>
          </div>
          <div class="wx-now-mid">
            {range_html}
            <div class="wx-now-stats">{''.join(stats)}</div>
          </div>
          <div class="wx-now-side">
            {health_html}
            {rain_chip}
          </div>
        </div>"""
    else:
        now_html = f"""
        <div class="wx-now">
          <div class="wx-now-primary">
            <div class="wx-now-desc-wrap"><div class="wx-now-desc">降雨現況</div></div>
          </div>
          <div class="wx-now-mid">
            <span class="wx-now-temp">{e(str(rain.get('observed_mm', 0)))}</span>
            <span class="wx-now-desc">mm</span>
          </div>
          <div class="wx-now-side">
            {health_html}
            {rain_chip}
          </div>
        </div>"""

    # 同一張卡的三個時段共用刻度，溫度帶位置/長度可直接互相比較
    card_scale = _temp_scale(periods[:3])

    def _period_cell(p: dict) -> str:
        hours = _period_hours(p.get("start_time"), p.get("end_time"))
        hours_html = f"<div class='wx-p-hours'>{e(hours)}</div>" if hours else ""
        comfort = p.get("comfort")
        comfort_html = (
            f"<div class='wx-p-comfort'>{_comfort_emoji(comfort)} {e(comfort)}</div>" if comfort else ""
        )
        feel_html = ""
        if p.get("min_feel") and p.get("max_feel"):
            feel_html = (
                f"<div class='wx-p-feel'>{_feel_emoji(p.get('max_feel'), p.get('min_feel'))} "
                f"體感 {e(str(p.get('min_feel')))} ~ {e(str(p.get('max_feel')))}°C</div>"
            )
        bar = _temp_bar_html(p.get("min_temp"), p.get("max_temp"), *card_scale) if card_scale else ""
        if bar:
            temp_html = (
                f"<div class='wx-t-row'><span class='t'>{e(str(p.get('min_temp')))}°</span>{bar}"
                f"<span class='t'>{e(str(p.get('max_temp')))}°</span></div>"
            )
        else:
            temp_html = (
                f"<div class='wx-p-temp'>🌡️ {e(str(p.get('min_temp', '—')))} ~ {e(str(p.get('max_temp', '—')))}°C</div>"
            )
        return f"""
        <div class="wx-period">
          <div class="wx-p-head"><span class="wx-p-label">{e(_period_label(p.get('start_time')))}</span>
            <span class="wx-p-pop">☔ {e(str(p.get('pop', '—')))}%</span></div>
          {hours_html}
          <div class="wx-p-wx">{_weather_emoji(p.get('weather', ''))} {e(p.get('weather', '—'))}</div>
          {temp_html}
          {feel_html}
          {comfort_html}
        </div>"""

    period_cells = "".join(_period_cell(p) for p in periods[:3])

    county_html = f"<span class='wx-county'>{e(county)}</span>" if county else ""
    raw = f"""
    <div class="wx-card">
      <div class="wx-head"><span class="wx-name">{e(name)}</span>{county_html}</div>
      <div class="wx-body">
        {now_html}
        <div class="wx-periods">{period_cells}</div>
      </div>
      <div class="wx-src">{e('　｜　'.join(source_bits))}</div>
    </div>"""
    # markdown 會把縮排 4 格以上的行當 code block，HTML 每行前導空白必須拿掉
    return "".join(line.strip() for line in raw.splitlines())


# ──────────────────────────────────────────────────────────────────────────
# 頁面
# ──────────────────────────────────────────────────────────────────────────
has_cwa = bool(get_cwa_api_key())
is_demo = (st.query_params.get("demo") == "1") or not has_cwa

if is_demo:
    locations, earthquake, typhoon, warning_details, alerts = _demo_bundle()
else:
    locations = [_live_location(n, lat, lon) for n, (lat, lon) in _LOCATION_COORDS.items()]
    earthquake = _live_earthquake()
    typhoon = _live_typhoon()
    warning_details = _live_warning_details()
    alerts = load_alerts()

header_left, header_right = st.columns([5, 1], vertical_alignment="bottom")
with header_left:
    st.title("🌦️ 氣象通知")
with header_right:
    # 次要操作不佔滿欄寬——過寬的單一按鈕會搶走警示層的視覺權重
    if st.button("🔄 重新整理", width="content"):
        _live_location.clear()
        _live_earthquake.clear()
        _live_typhoon.clear()
        _live_warning_details.clear()
        _load_alerts.clear()
        st.rerun()

if is_demo:
    if not has_cwa:
        st.warning(
            "目前顯示**示範資料**——本機未設定 `CWA_API_KEY`。"
            "在 `.streamlit/secrets.toml` 加入 `CWA_API_KEY` 後即可顯示即時天氣。",
            icon="🧪",
        )
    else:
        st.warning("目前顯示**示範資料**（`?demo=1`）。移除網址參數即可看即時資料。", icon="🧪")
else:
    newest = max((l.get("data_time") or "" for l in locations), default="")
    # 資料連線狀態用中性灰——綠色保留給「安全」語意，避免在警報上方形成假安全訊號
    st.caption(f":gray-badge[即時 CWA 資料]　降雨網格時間：{_fmt_time(newest)}（頁面每次載入即時查詢）")

# ── 警示層：在地決策列 + 嚴重度橫幅（例外才浮出；等級=相關性×災種，非官方分級） ──
st.markdown(_WX_CSS, unsafe_allow_html=True)

_WARN_ICON = [("雨", "🌧️"), ("風", "💨"), ("低溫", "🥶"), ("高溫", "🥵"), ("濃霧", "🌫️")]
watched_counties = {loc.get("county") for loc in locations if loc.get("county")}
watched_label = "、".join(sorted(watched_counties)) or "我的地點"

typhoon_active = bool(typhoon and typhoon.get("active"))
# 同一颱風常同時出現在颱風公報與 W-C0033 特報清單 → 只呈現一次，特報版併入颱風區塊
typhoon_dups = [wd for wd in (warning_details or []) if typhoon_active and "颱風" in (wd.get("title") or "")]
other_details = [wd for wd in (warning_details or []) if wd not in typhoon_dups]

typhoon_affected = list((typhoon or {}).get("affected_areas") or [])
for wd in typhoon_dups:
    for c in wd.get("affected") or []:
        if c not in typhoon_affected:
            typhoon_affected.append(c)
typhoon_local = any(c in watched_counties for c in typhoon_affected)

local_warns = [wd for wd in other_details if any(c in watched_counties for c in wd.get("affected") or [])]
other_warns = [wd for wd in other_details if wd not in local_warns]

# 在地決策列：先回答「此刻、此地、有沒有事」，再往下看是什麼事
n_local = len(local_warns) + (1 if (typhoon_active and typhoon_local) else 0)
if n_local:
    st.markdown(
        f"<div class='al-strip hit'>📍 <b>{_html.escape(watched_label)}</b>"
        f"<span>{n_local} 則警報影響中，詳見下方</span></div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"<div class='al-strip ok'>📍 <b>{_html.escape(watched_label)}</b>"
        f"<span>目前無生效中的警報特報</span></div>",
        unsafe_allow_html=True,
    )


def _render_warning_banner(wd: dict, sev: str) -> None:
    icon = next((emo for kw, emo in _WARN_ICON if kw in wd.get("title", "")), "⚠️")
    time_txt = f"有效至 <b>{_fmt_time(wd.get('end_time'))}</b> {_remaining_label(wd.get('end_time'))}"
    st.markdown(
        _alert_banner_html(
            sev, icon, wd.get("title", "天氣特報"),
            f"影響：{_affected_summary(wd.get('affected') or [], watched_counties)}",
            time_txt, sev == "watch",
        ),
        unsafe_allow_html=True,
    )


if typhoon_active:
    st.markdown(
        _alert_banner_html(
            "danger", "🌀", typhoon.get("headline", "颱風警報"),
            f"影響：{_affected_summary(typhoon_affected, watched_counties)}",
            f"公告 <b>{_fmt_time(typhoon.get('effective'))}</b>", typhoon_local,
        ),
        unsafe_allow_html=True,
    )
    with st.expander("📄 颱風警報內容全文與完整影響範圍"):
        for sec in typhoon.get("description") or []:
            if sec.get("title"):
                st.markdown(f"**{sec['title']}**")
            if sec.get("value"):
                st.text(sec["value"])
        if typhoon_affected:
            st.markdown(f"**警戒區域**　{'、'.join(a for a in typhoon_affected if a)}")
        for wd in typhoon_dups:
            if wd.get("content"):
                st.markdown(f"**{wd.get('title', '特報')}**")
                st.text(wd["content"])

for wd in local_warns:
    _render_warning_banner(wd, "watch")
    if wd.get("content"):
        with st.expander(f"📄 {wd.get('title', '特報')}內容全文與完整影響範圍"):
            st.text(wd["content"])
            if wd.get("affected"):
                st.markdown(f"**影響範圍**　{'、'.join(wd['affected'])}")
            st.caption(f":small[:gray[發布 {_fmt_time(wd.get('issue_time'))}]]")

# 沒打到監控地點的特報整批收合，不佔首屏
if other_warns:
    with st.expander(f"🗺️ 其他地區特報 {len(other_warns)} 則（未影響你的地點）"):
        for wd in other_warns:
            _render_warning_banner(wd, "notice")
            if wd.get("content"):
                st.text(wd["content"])
            if wd.get("affected"):
                st.caption(f"影響範圍：{'、'.join(wd['affected'])}")

# ── 我的地點（常駐：現在觀測 + 降雨現況 + 今明36小時預報） ──────────────
st.subheader("📍 我的地點")
if not locations:
    st.caption("沒有設定監控地點。")

for loc in locations:
    current = loc.get("current")
    periods = loc.get("periods") or []
    rain = loc.get("rain") or {}
    county = loc.get("county")

    source_bits = []
    if loc.get("data_time"):
        source_bits.append(f"降雨網格 {_fmt_time(loc.get('data_time'))}")
    if current:
        source_bits.append(
            f"現在天氣：{current.get('station_name', '—')} 測站"
            f"（約 {current.get('station_distance_km', '—')} km・{_fmt_time(current.get('obs_time'))} 觀測）"
        )
    source_bits.append("預報：CWA 一週鄉鎮天氣預報")

    st.markdown(
        _build_location_card_html(loc["name"], county, current, rain, periods, source_bits, loc.get("health")),
        unsafe_allow_html=True,
    )

    # 卡片只放最近 3 個時段；一週預報（一天一列，白天/晚上合併）收進 expander
    if len(periods) > 3:
        days: dict[str, dict] = {}
        for p in periods:
            try:
                start = datetime.fromisoformat(p.get("start_time"))
            except (TypeError, ValueError):
                continue
            d = days.setdefault(
                start.strftime("%Y-%m-%d"),
                {"date": start, "min": None, "max": None, "pops": [], "day_wx": None, "any_wx": None},
            )
            try:
                v = float(p.get("min_temp"))
                d["min"] = v if d["min"] is None else min(d["min"], v)
            except (TypeError, ValueError):
                pass
            try:
                v = float(p.get("max_temp"))
                d["max"] = v if d["max"] is None else max(d["max"], v)
            except (TypeError, ValueError):
                pass
            pop = p.get("pop")
            if pop not in (None, "", " ", "-"):
                try:
                    d["pops"].append(int(pop))
                except ValueError:
                    pass
            wx = p.get("weather")
            if wx:
                if d["any_wx"] is None:
                    d["any_wx"] = wx
                if 6 <= start.hour < 18:
                    d["day_wx"] = wx  # 一天的代表天氣以白天時段為準

        wk_scale = _temp_scale(periods)
        rows_html = []
        for key in sorted(days):
            d = days[key]
            dt = d["date"]
            wx = d["day_wx"] or d["any_wx"] or "—"
            pop_txt = f"{max(d['pops'])}%" if d["pops"] else "—"
            min_txt = f"{d['min']:.0f}°" if d["min"] is not None else "—"
            max_txt = f"{d['max']:.0f}°" if d["max"] is not None else "—"
            bar = (
                _temp_bar_html(d["min"], d["max"], *wk_scale)
                if wk_scale and d["min"] is not None and d["max"] is not None
                else "<span></span>"
            )
            rows_html.append(
                f"<div class='wx-wk-row'>"
                f"<span class='wx-wk-day'>{dt.month}/{dt.day}（{_WEEKDAY[dt.weekday()]}）</span>"
                f"<span>{_weather_emoji(wx)}</span>"
                f"<span class='wx-wk-desc'>{_html.escape(wx)}</span>"
                f"<span class='wx-wk-pop'>☔ {pop_txt}</span>"
                f"<span class='wx-wk-t'>{min_txt}</span>{bar}<span class='wx-wk-t'>{max_txt}</span>"
                f"</div>"
            )

        st.markdown(
            f"<div class='wx-card'><div class='wx-head'><span class='wx-name'>📅 一週預報</span>"
            f"<span class='wx-county'>{_html.escape(county or loc['name'])}</span></div>"
            f"<div class='wx-wk'>{''.join(rows_html)}</div></div>",
            unsafe_allow_html=True,
        )

# ── 最近地震 + 通知歷史（併成雙欄，減少常駐版面高度） ────────────────────
col_eq, col_hist = st.columns(2)

with col_eq:
    st.subheader("🌐 最近地震")
    if earthquake:
        st.markdown(
            f"{_fmt_time(earthquake.get('origin_time'))}　**{earthquake.get('location', '未知位置')}**　"
            f"規模 **{earthquake.get('magnitude', '—')}**　深度 {earthquake.get('depth', '—')} km"
        )
        if earthquake.get("report_image_uri"):
            with st.expander("查看震度分布圖"):
                st.image(earthquake["report_image_uri"], caption="CWA 震度分布圖", width="stretch")
    else:
        st.caption("目前沒有地震資料。")

with col_hist:
    st.subheader("🔔 通知歷史")
    if alerts is None:
        st.caption("通知歷史來自收集器寫入 Supabase 的推播事件流；本機未設定 Supabase，暫不顯示。")
    elif not alerts:
        st.caption("還沒有任何通知紀錄。")
    else:
        _CATEGORY_BADGE = {
            "earthquake": ":red-badge[🌐 地震]",
            "typhoon": ":orange-badge[🌀 颱風]",
            "warning": ":orange-badge[⚠️ 特報]",
            "rain": ":blue-badge[🌧️ 降雨]",
            "forecast": ":gray-badge[☀️ 預報]",
        }
        # 固定高度、內部捲動：歷史再多也不會把頁面越拉越長
        with st.container(height=320, border=True):
            for a in alerts:
                badge = _CATEGORY_BADGE.get(a["category"], ":gray-badge[🔔]")
                line = f"{badge}　**{a['title']}**"
                if a.get("extra_url"):
                    line += f"　[圖]({a['extra_url']})"
                st.markdown(line)
                st.caption(f":small[:gray[{a['body']}　{_fmt_time(a.get('ts'))}]]")

# ── 通知規則 ────────────────────────────────────────────────────────────
with st.expander("📋 通知規則說明"):
    st.markdown(
        """
本頁**儀表板**（現在天氣、預報、降雨、地震、颱風、特報）為**即時查詢 CWA**。
以下規則是**收集器**（GitHub Actions → Supabase）產生**推播通知（st.toast）與通知歷史**的邏輯：

| 類別 | 觸發條件 | 去重方式 |
|---|---|---|
| 🌐 地震 | 新地震報告，且**所在縣市震度 ≥ 設定門檻**（預設 4級）才推播 | 地震編號 |
| ⚠️ 天氣特報 | 全台發布**新的**特報公告（大雨/豪雨/低溫/強風…），通知只含現象＋影響區域＋期限，公告全文在本頁特報區展開看 | (公告標題, 發布時間)，到期自動重置 |
| 🌧️ 降雨 | 狀態**轉換**時通知一次：無雨 → 即將降雨（未來1小時預測 ≥ 1.0mm）、→ 降雨中（過去1小時實測 ≥ 0.5mm） | 每個地點各自的狀態機 |
| ☀️ 預報 | 預報**內容有變化**才通知 | 與上一次快照比對 |
| 🌀 颱風 | 氣象署發布**新的颱風公告**（含發布與解除） | 公告時戳，保留30天 |

- 收集器由 cron-job.org 定時觸發 GitHub Actions，約 **每 10 分鐘一輪**；門檻值（震度、雨量）在收集器的 `config.yaml` 調整。
- 通知會即時出現在 Anya **所有頁面**的右下角 toast（不自動消失，按掉才解除）；本頁「通知歷史」顯示最近 20 筆，資料庫保留 30 天後自動清除。
"""
    )
