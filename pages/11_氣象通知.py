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
    get_rain,
    get_typhoon_impl,
    get_warnings,
)
from utils.rich_styles import inject_rich_styles
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
        "warnings": [], "data_time": None,
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
    }


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
        "weather": "陰", "air_temperature_c": "27.4",
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
    }]
    earthquake = {
        "location": "花蓮縣政府南南西方 41.0 公里 (位於花蓮縣豐濱鄉)",
        "magnitude": 5.1, "depth": 15.5, "origin_time": ago(hours=2),
        "report_image_uri": "https://scweb.cwa.gov.tw/webdata/OLDEQ/202607/2026070207060640048_H.png",
    }
    typhoon = {"active": True, "headline": "海上颱風警報", "effective": ago(hours=5)}
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
    return locations, earthquake, typhoon, alerts


def _fmt_time(value) -> str:
    if not value:
        return "—"
    try:
        return datetime.fromisoformat(str(value)).strftime("%m/%d %H:%M")
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


_WX_CSS = """
<style>
.wx-card{border:1px solid #F2D5CF;border-radius:14px;background:linear-gradient(180deg,#FFFFFF 0%,#FFF7F4 100%);
  padding:18px 20px 14px;margin-bottom:14px;}
.wx-head{display:flex;align-items:baseline;gap:10px;margin-bottom:14px;}
.wx-name{font-size:1.05rem;font-weight:700;color:#4A2F1A;}
.wx-county{font-size:.85rem;color:#9A8478;}
.wx-body{display:flex;gap:16px;flex-wrap:wrap;}
.wx-now{flex:0 0 230px;display:flex;flex-direction:column;justify-content:center;gap:6px;
  padding:14px 18px;border-radius:12px;background:#FFF1EC;}
.wx-now-main{display:flex;align-items:center;gap:12px;}
.wx-now-emoji{font-size:2.4rem;line-height:1;}
.wx-now-temp{font-size:2.5rem;font-weight:700;color:#4A2F1A;line-height:1;}
.wx-now-desc{font-size:1rem;font-weight:600;color:#C05A50;}
.wx-now-stats{display:flex;gap:14px;font-size:.82rem;color:#7A6A5F;}
.wx-chip{display:inline-block;margin-top:4px;padding:4px 10px;border-radius:99px;font-size:.8rem;font-weight:600;width:fit-content;}
.wx-chip.rain-on{background:#E3F2FD;color:#1565C0;}
.wx-chip.rain-soon{background:#FFF3E0;color:#B26A00;}
.wx-chip.rain-off{background:#F1EDEA;color:#8A7A6E;}
.wx-periods{flex:1;min-width:280px;display:grid;grid-template-columns:repeat(3,1fr);gap:12px;}
.wx-period{border:1px solid #F2D5CF;border-radius:12px;background:#FFFFFF;padding:12px 14px;
  display:flex;flex-direction:column;gap:5px;}
.wx-p-head{display:flex;justify-content:space-between;align-items:baseline;gap:6px;}
.wx-p-label{font-size:.85rem;font-weight:700;color:#4A2F1A;}
.wx-p-pop{font-size:.78rem;color:#5B8DB8;white-space:nowrap;}
.wx-p-hours{font-size:.72rem;color:#B0A399;}
.wx-p-wx{font-size:.98rem;font-weight:600;color:#C05A50;display:flex;align-items:center;gap:6px;}
.wx-p-temp{font-size:.82rem;color:#7A6A5F;}
.wx-p-feel{font-size:.78rem;color:#B26A55;}
.wx-p-comfort{font-size:.76rem;color:#8A7A6E;}
.wx-src{margin-top:12px;font-size:.72rem;color:#B0A399;}
.wx-t-row{display:flex;align-items:center;gap:7px;font-size:.8rem;color:#7A6A5F;}
.wx-t-row .t{font-weight:600;}
.wx-t-track{flex:1;height:6px;border-radius:99px;background:#F1EDEA;position:relative;min-width:50px;}
.wx-t-fill{position:absolute;top:0;height:100%;border-radius:99px;}
.wx-wk{display:flex;flex-direction:column;gap:2px;}
.wx-wk-row{display:grid;grid-template-columns:104px 26px minmax(70px,1fr) 46px 34px minmax(80px,1.4fr) 34px;
  align-items:center;gap:8px;padding:7px 10px;border-radius:10px;font-size:.86rem;color:#4A2F1A;}
.wx-wk-row:nth-child(even){background:#FDF0ED;}
.wx-wk-day{font-weight:600;}
.wx-wk-desc{color:#7A6A5F;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.wx-wk-pop{color:#5B8DB8;font-size:.8rem;text-align:right;}
.wx-wk-t{color:#7A6A5F;font-weight:600;text-align:right;}
@media (max-width:720px){.wx-now{flex:1 1 100%;}.wx-periods{grid-template-columns:1fr;}
  .wx-wk-row{grid-template-columns:88px 24px 42px 30px 1fr 30px;}.wx-wk-desc{display:none;}}
</style>
"""

_RAIN_CHIP_CLASS = {"raining": "rain-on", "soon_rain": "rain-soon", "dry": "rain-off"}

# 溫度 → 端點顏色（冷藍 → 金 → 暖橘 → 珊瑚紅），溫度帶兩端各取自己溫度的顏色
_TEMP_STOPS = [(12, (110, 159, 197)), (22, (200, 164, 58)), (30, (224, 138, 99)), (36, (192, 90, 80))]


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


def _build_location_card_html(name, county, current, rain, periods, source_bits) -> str:
    e = _html.escape

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
        now_html = f"""
        <div class="wx-now">
          <div class="wx-now-main">
            <span class="wx-now-emoji">{_weather_emoji(desc)}</span>
            <span class="wx-now-temp">{e(str(current.get('air_temperature_c', '—')))}°</span>
          </div>
          <div class="wx-now-desc">{e(desc)}</div>
          <div class="wx-now-stats">{''.join(stats)}</div>
          {rain_chip}
        </div>"""
    else:
        now_html = f"""
        <div class="wx-now">
          <div class="wx-now-desc">降雨現況</div>
          <div class="wx-now-main"><span class="wx-now-temp">{e(str(rain.get('observed_mm', 0)))}</span>
            <span class="wx-now-desc">mm</span></div>
          {rain_chip}
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
    locations, earthquake, typhoon, alerts = _demo_bundle()
else:
    locations = [_live_location(n, lat, lon) for n, (lat, lon) in _LOCATION_COORDS.items()]
    earthquake = _live_earthquake()
    typhoon = _live_typhoon()
    alerts = load_alerts()

header_left, header_right = st.columns([4, 1], vertical_alignment="bottom")
with header_left:
    st.title("🌦️ 氣象通知")
with header_right:
    if st.button("🔄 重新整理", width="stretch"):
        _live_location.clear()
        _live_earthquake.clear()
        _live_typhoon.clear()
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
    st.caption(f":green-badge[即時 CWA 資料]　降雨網格時間：{_fmt_time(newest)}（頁面每次載入即時查詢）")

# ── 警示橫幅：只在有事時出現 ────────────────────────────────────────────
has_alert = False

if typhoon and typhoon.get("active"):
    has_alert = True
    st.error(
        f"🌀 **{typhoon.get('headline', '颱風警報')}** 生效中"
        f"　:small[:gray[公告 {_fmt_time(typhoon.get('effective'))}]]",
        icon="⚠️",
    )

for loc in locations:
    for hazard in loc.get("warnings", []):
        has_alert = True
        st.error(
            f"**{loc['name']}**：{hazard.get('phenomena', '')}{hazard.get('significance', '')}生效中"
            f"　有效至 {_fmt_time(hazard.get('end_time'))}",
            icon="🌧️" if "雨" in hazard.get("phenomena", "") else "⚠️",
        )

if not has_alert:
    # 「沒事」是常態，不佔一整條橫幅——例外才浮出
    st.markdown(":small[:green[✅ 目前無生效中的颱風警報與天氣特報]]")

# ── 我的地點（常駐：現在觀測 + 降雨現況 + 今明36小時預報） ──────────────
st.subheader("📍 我的地點")
if not locations:
    st.caption("沒有設定監控地點。")
else:
    st.markdown(_WX_CSS, unsafe_allow_html=True)

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
        _build_location_card_html(loc["name"], county, current, rain, periods, source_bits),
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

        with st.expander(f"📅 一週預報（{county or loc['name']}）"):
            st.markdown(f"<div class='wx-wk'>{''.join(rows_html)}</div>", unsafe_allow_html=True)

# ── 最近地震（一行式，震度圖收進 expander） ─────────────────────────────
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

# ── 通知歷史（來自 Supabase 收集器；本機未設定 Supabase 時不顯示） ──────
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
    with st.container(border=True):
        for a in alerts:
            badge = _CATEGORY_BADGE.get(a["category"], ":gray-badge[🔔]")
            line = f"{badge}　**{a['title']}**　{a['body']}"
            if a.get("extra_url"):
                line += f"　[圖]({a['extra_url']})"
            st.markdown(line)
            st.caption(f":small[:gray[{_fmt_time(a.get('ts'))}]]")

# ── 通知規則 ────────────────────────────────────────────────────────────
with st.expander("📋 通知規則說明"):
    st.markdown(
        """
本頁**儀表板**（現在天氣、預報、降雨、地震、颱風、特報）為**即時查詢 CWA**。
以下規則是**收集器**（GitHub Actions → Supabase）產生**推播通知（st.toast）與通知歷史**的邏輯：

| 類別 | 觸發條件 | 去重方式 |
|---|---|---|
| 🌐 地震 | 新地震報告，且**所在縣市震度 ≥ 設定門檻**（預設 4級）才推播 | 地震編號 |
| ⚠️ 天氣特報 | 所在縣市出現**新的**特報（大雨/豪雨/低溫/強風…） | (縣市, 現象, 起始時間)，到期自動重置 |
| 🌧️ 降雨 | 狀態**轉換**時通知一次：無雨 → 即將降雨（未來1小時預測 ≥ 1.0mm）、→ 降雨中（過去1小時實測 ≥ 0.5mm） | 每個地點各自的狀態機 |
| ☀️ 預報 | 預報**內容有變化**才通知 | 與上一次快照比對 |
| 🌀 颱風 | 氣象署發布**新的颱風公告**（含發布與解除） | 公告時戳，保留30天 |

- 收集器由 cron-job.org 定時觸發 GitHub Actions，約 **每 10 分鐘一輪**；門檻值（震度、雨量）在收集器的 `config.yaml` 調整。
- 通知會即時出現在 Anya **所有頁面**的右下角 toast；本頁「通知歷史」保留最近 20 筆。
"""
    )
