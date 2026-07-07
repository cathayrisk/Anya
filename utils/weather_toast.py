# utils/weather_toast.py
# -*- coding: utf-8 -*-
"""
氣象通知 st.toast watcher（全 App 共用）

資料來源：外部 GitHub Actions 收集器輪詢 CWA 開放資料後寫入 Supabase 的
`weather_alerts` 表。這裡只負責「讀取 + st.toast」，不做任何輪詢/收集邏輯。

公開函式：
  render_weather_toast_watcher()  — 在每個頁面 st.set_page_config() 之後呼叫一次
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from supabase import Client

_CATEGORY_ICON = {
    "earthquake": "🌐",
    "typhoon": "🌀",
    "warning": "⚠️",
    "rain": "🌧️",
    "forecast": "☀️",
}


def get_secret_safe(name: str):
    """st.secrets.get 在 secrets.toml 完全不存在時會直接拋例外（不是回傳 None），
    本機沒設定 secrets 跑 demo 時會炸頁面，包一層。"""
    try:
        return st.secrets.get(name)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _get_weather_supabase_client() -> Client:
    # 惰性 import：supabase 套件缺失時只讓氣象通知靜默停用（呼叫端 try/except 接住），
    # 不在模組載入期炸掉 import 本模組的所有頁面
    from supabase import create_client
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])


def _latest_alert_id(client: Client) -> int:
    resp = client.table("weather_alerts").select("id").order("id", desc=True).limit(1).execute()
    return resp.data[0]["id"] if resp.data else 0


def _fetch_new_alerts(client: Client, since_id: int) -> list[dict]:
    resp = (
        client.table("weather_alerts")
        .select("id, category, title, body, extra_url")
        .gt("id", since_id)
        .order("id")
        .execute()
    )
    return resp.data or []


@st.fragment(run_every="15s")
def _weather_toast_fragment() -> None:
    if not (get_secret_safe("SUPABASE_URL") and get_secret_safe("SUPABASE_KEY")):
        return  # Supabase not configured in this environment (e.g. local dev) — stay silent

    try:
        client = _get_weather_supabase_client()

        if "_weather_last_alert_id" not in st.session_state:
            # First run for this session: don't replay historical alerts, just
            # remember where "new" starts from.
            st.session_state["_weather_last_alert_id"] = _latest_alert_id(client)
            return

        new_rows = _fetch_new_alerts(client, st.session_state["_weather_last_alert_id"])
    except Exception:
        return  # network/Supabase hiccup shouldn't break the page

    for row in new_rows:
        icon = _CATEGORY_ICON.get(row["category"], "🔔")
        # duration="infinite"：不自動消失，使用者要自己按掉右上角關閉鈕才會解除
        st.toast(f"{row['title']}：{row['body']}", icon=icon, duration="infinite")
        st.session_state["_weather_last_alert_id"] = row["id"]


def render_weather_toast_watcher() -> None:
    """Call near the top of every page (right after st.set_page_config) to get
    app-wide st.toast alerts for new CWA earthquake/warning/rain/forecast
    events collected externally into Supabase."""
    _weather_toast_fragment()
