# pages/11_氣象通知.py
# -*- coding: utf-8 -*-
"""
氣象通知專屬頁面

資料來源：外部 GitHub Actions 收集器輪詢 CWA 開放資料後寫入 Supabase 的
`weather_latest`（現況快照）與 `weather_alerts`（事件流，交給 st.toast 用）。
這裡只負責讀取 + 顯示，不做任何輪詢/收集邏輯。
"""

from __future__ import annotations

from collections import defaultdict

import streamlit as st

from utils.rich_styles import inject_rich_styles
from utils.weather_toast import _get_weather_supabase_client, render_weather_toast_watcher

st.set_page_config(page_title="氣象通知", page_icon="🌦️", layout="wide")
inject_rich_styles()
render_weather_toast_watcher()

st.title("🌦️ 氣象通知")

if not (st.secrets.get("SUPABASE_URL") and st.secrets.get("SUPABASE_KEY")):
    st.error("找不到 Supabase 設定，請在 .streamlit/secrets.toml 設定 SUPABASE_URL / SUPABASE_KEY。")
    st.stop()

try:
    client = _get_weather_supabase_client()
    resp = client.table("weather_latest").select("location_name, category, payload, updated_at").execute()
    rows = resp.data or []
except Exception as e:
    st.error(f"讀取氣象資料失敗：{e}")
    st.stop()

by_location: dict[str, dict[str, dict]] = defaultdict(dict)
for row in rows:
    by_location[row["location_name"]][row["category"]] = row

# ── 最近地震 ──────────────────────────────────────────────────────────────
st.subheader("🌐 最近地震")
earthquake_row = by_location.get("ALL", {}).get("earthquake")
if earthquake_row:
    payload = earthquake_row["payload"]
    st.info(
        f"{payload.get('location', '未知位置')} 發生規模 {payload.get('magnitude')} 地震"
        f"（深度 {payload.get('depth')}km）"
    )
    if payload.get("report_image_uri"):
        st.image(payload["report_image_uri"], caption="震度圖", width=400)
    st.caption(f"地震時間：{payload.get('origin_time', '—')}　｜　最後更新：{earthquake_row.get('updated_at', '—')}")
else:
    st.caption("目前沒有地震資料。")

st.divider()

# ── 各地點現況 ────────────────────────────────────────────────────────────
st.subheader("📍 各地點現況")
location_names = sorted(name for name in by_location if name != "ALL")

if not location_names:
    st.caption("目前沒有地點現況資料，等收集器跑過第一輪之後就會出現。")

for name in location_names:
    categories = by_location[name]
    st.markdown(f"### {name}")
    col_forecast, col_warning, col_rain = st.columns(3)

    with col_forecast:
        st.markdown("**預報**")
        forecast = categories.get("forecast")
        if forecast:
            p = forecast["payload"]
            st.metric("天氣", p.get("weather", "—"))
            st.caption(
                f"降雨機率 {p.get('pop_percent', '—')}%　"
                f"氣溫 {p.get('min_temp', '—')}~{p.get('max_temp', '—')}°C"
            )
            st.caption(f"最後更新：{forecast.get('updated_at', '—')}")
        else:
            st.caption("尚無資料")

    with col_warning:
        st.markdown("**天氣特報**")
        warning = categories.get("warning")
        active = (warning or {}).get("payload", {}).get("active", [])
        if active:
            for hazard in active:
                st.error(f"{hazard['phenomena']}{hazard['significance']}　有效至 {hazard['end_time']}")
        else:
            st.success("目前無特報")
        if warning:
            st.caption(f"最後更新：{warning.get('updated_at', '—')}")

    with col_rain:
        st.markdown("**降雨現況**")
        rain = categories.get("rain")
        if rain:
            p = rain["payload"]
            state_label = {"raining": "🌧️ 降雨中", "soon_rain": "☁️ 即將降雨", "dry": "☀️ 無降雨"}.get(
                p.get("state"), p.get("state", "—")
            )
            st.metric(state_label, f"{p.get('observed_mm', 0)}mm（過去1小時）")
            st.caption(f"未來1小時預測：{p.get('forecast_mm', 0)}mm")
            st.caption(f"最後更新：{rain.get('updated_at', '—')}")
        else:
            st.caption("尚無資料")

    st.divider()
