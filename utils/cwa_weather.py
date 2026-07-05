# utils/cwa_weather.py
# -*- coding: utf-8 -*-
"""
中央氣象署(CWA)開放資料 — 即時查詢工具（給 Anya_Gemma 的 @tool 用）

跟 weather/ 收集器是兩個獨立的東西：這裡只做「使用者問一次、答一次」的即時查詢，
不做輪詢/去重/狀態機，所以不依賴 weather/ repo，也不需要 Supabase。

公開函式：
  get_weather_impl(location)   — 任意地點的天氣現況（預報＋特報＋降雨網格）
  get_earthquake_impl()        — 最新地震
  get_typhoon_impl()           — 目前颱風狀態（含追蹤中但尚未對台發布警報的熱帶氣旋）
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import requests
import streamlit as st

REST_BASE = "https://opendata.cwa.gov.tw/api/v1/rest/datastore"
FILE_BASE = "https://opendata.cwa.gov.tw/fileapi/v1/opendataapi"
NODATA = -99.0

_WANTED_FORECAST_ELEMENTS = {"Wx": "weather", "PoP": "pop", "MinT": "min_temp", "MaxT": "max_temp"}

# 22 縣市官方名稱 → 縣市政府座標（geocode 的 fallback）。
# 名稱已用 CWA 實際回傳資料核對過（W-C0033-001 / W-C0034-001 的 areaDesc）。
_COUNTY_CENTROIDS: dict[str, tuple[float, float]] = {
    "基隆市": (25.1276, 121.7392),
    "臺北市": (25.0330, 121.5654),
    "新北市": (25.0169, 121.4627),
    "宜蘭縣": (24.7021, 121.7378),
    "桃園市": (24.9936, 121.3010),
    "新竹縣": (24.8385, 121.0025),
    "新竹市": (24.8138, 120.9675),
    "苗栗縣": (24.5602, 120.8214),
    "臺中市": (24.1477, 120.6736),
    "彰化縣": (24.0518, 120.5161),
    "南投縣": (23.9157, 120.6870),
    "雲林縣": (23.7092, 120.4313),
    "嘉義縣": (23.4518, 120.2555),
    "嘉義市": (23.4801, 120.4491),
    "臺南市": (22.9998, 120.2269),
    "高雄市": (22.6273, 120.3014),
    "屏東縣": (22.5519, 120.5487),
    "臺東縣": (22.7583, 121.1444),
    "花蓮縣": (23.9871, 121.6015),
    "澎湖縣": (23.5711, 119.5794),
    "金門縣": (24.4324, 118.3171),
    "連江縣": (26.1505, 119.9497),
}


# --------------------------------------------------------------------------
# secret 讀取（沿用 Anya_Gemma.py 的 fallback 順序：st.secrets → env → .env）
# --------------------------------------------------------------------------
def _get_secret(name: str) -> Optional[str]:
    try:
        return st.secrets.get(name)
    except Exception:
        return None


def _load_key_from_dotenv(name: str) -> Optional[str]:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(root, ".env")
    if not os.path.exists(env_path):
        return None
    try:
        for line in open(env_path, encoding="utf-8"):
            line = line.strip()
            if line.startswith(name):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return None


def get_cwa_api_key() -> Optional[str]:
    return (
        _get_secret("CWA_API_KEY")
        or os.getenv("CWA_API_KEY")
        or _load_key_from_dotenv("CWA_API_KEY")
    )


# --------------------------------------------------------------------------
# 低階 API 呼叫
# --------------------------------------------------------------------------
def _rest(resource_id: str, **params) -> dict:
    api_key = get_cwa_api_key()
    if not api_key:
        raise RuntimeError("找不到 CWA_API_KEY，請在 .streamlit/secrets.toml 或 .env 設定。")
    query = {"Authorization": api_key, **params}
    resp = requests.get(f"{REST_BASE}/{resource_id}", params=query, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if data.get("success") != "true":
        raise RuntimeError(f"CWA API {resource_id} 回傳失敗：{data}")
    return data


@dataclass
class GridDataset:
    start_lon: float
    start_lat: float
    resolution: float
    dim_x: int
    dim_y: int
    data_time: str
    values: list

    @classmethod
    def from_raw(cls, raw: dict) -> "GridDataset":
        dataset = raw["cwaopendata"]["dataset"]
        params = dataset["datasetInfo"]["parameterSet"]
        content = dataset["contents"]["content"]
        values = [float(v) for v in content.split(",") if v.strip()]
        return cls(
            start_lon=float(params["StartPointLongitude"]),
            start_lat=float(params["StartPointLatitude"]),
            resolution=float(params["GridResolution"]),
            dim_x=int(params["GridDimensionX"]),
            dim_y=int(params["GridDimensionY"]),
            data_time=params["DateTime"],
            values=values,
        )

    def value_at(self, lat: float, lon: float):
        col = round((lon - self.start_lon) / self.resolution)
        row = round((lat - self.start_lat) / self.resolution)
        if not (0 <= col < self.dim_x and 0 <= row < self.dim_y):
            return None
        index = row * self.dim_x + col
        value = self.values[index]
        return None if value <= NODATA else value


def _grid(resource_id: str) -> GridDataset:
    api_key = get_cwa_api_key()
    if not api_key:
        raise RuntimeError("找不到 CWA_API_KEY，請在 .streamlit/secrets.toml 或 .env 設定。")
    query = {"Authorization": api_key, "format": "JSON"}
    resp = requests.get(f"{FILE_BASE}/{resource_id}", params=query, timeout=30)
    resp.raise_for_status()
    return GridDataset.from_raw(resp.json())


# --------------------------------------------------------------------------
# geocode：自由地名 → (縣市官方名稱, lat, lon)
# --------------------------------------------------------------------------
def _match_county_in_text(text: str) -> Optional[str]:
    for county in _COUNTY_CENTROIDS:
        if county in text:
            return county
    return None


def _nearest_county(lat: float, lon: float) -> str:
    return min(
        _COUNTY_CENTROIDS,
        key=lambda c: (lat - _COUNTY_CENTROIDS[c][0]) ** 2 + (lon - _COUNTY_CENTROIDS[c][1]) ** 2,
    )


def _nominatim_geocode(location: str):
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": f"{location}, Taiwan", "format": "json", "limit": 1},
            headers={"User-Agent": "AnyaWeatherTool/1.0 (personal use)"},
            timeout=5,
        )
        resp.raise_for_status()
        results = resp.json()
        if results:
            return float(results[0]["lat"]), float(results[0]["lon"])
    except Exception:
        pass
    return None


def geocode(location: str) -> tuple[str, float, float]:
    """回傳 (縣市官方名稱, lat, lon)。找不到會丟 ValueError。"""
    location = (location or "").strip()
    if not location:
        raise ValueError("location 不可為空")

    county = _match_county_in_text(location)
    coords = _nominatim_geocode(location)

    if coords is not None:
        lat, lon = coords
        if county is None:
            county = _nearest_county(lat, lon)
        return county, lat, lon

    if county is not None:
        lat, lon = _COUNTY_CENTROIDS[county]
        return county, lat, lon

    raise ValueError(f"無法辨識地點「{location}」，請提供更明確的縣市名稱。")


# --------------------------------------------------------------------------
# 天氣
# --------------------------------------------------------------------------
def get_forecast(county: str):
    raw = _rest("F-C0032-001", locationName=county)
    locations = raw.get("records", {}).get("location", [])
    if not locations:
        return None
    loc = locations[0]
    values: dict = {}
    window: dict = {}
    for el in loc.get("weatherElement", []):
        name = el.get("elementName")
        if name not in _WANTED_FORECAST_ELEMENTS:
            continue
        periods = el.get("time", [])
        if not periods:
            continue
        first = periods[0]
        values[_WANTED_FORECAST_ELEMENTS[name]] = first["parameter"]["parameterName"]
        window = {"start_time": first.get("startTime"), "end_time": first.get("endTime")}
    if not values:
        return None
    return {**values, **window}


def get_warnings(county: str) -> list:
    raw = _rest("W-C0033-001")
    warnings = []
    for loc in raw.get("records", {}).get("location", []):
        if loc.get("locationName") != county:
            continue
        for hazard in loc.get("hazardConditions", {}).get("hazards") or []:
            info = hazard.get("info", {})
            valid_time = hazard.get("validTime", {})
            warnings.append({
                "phenomena": info.get("phenomena", ""),
                "significance": info.get("significance", ""),
                "start_time": valid_time.get("startTime", ""),
                "end_time": valid_time.get("endTime", ""),
            })
    return warnings


def get_rain(lat: float, lon: float) -> dict:
    forecast_grid = _grid("F-B0046-001")
    observed_grid = _grid("O-B0045-001")
    return {
        "observed_past_1hr_mm": observed_grid.value_at(lat, lon),
        "forecast_next_1hr_mm": forecast_grid.value_at(lat, lon),
        "data_time": forecast_grid.data_time,
    }


def get_weather_impl(location: str) -> dict:
    county, lat, lon = geocode(location)
    return {
        "location_input": location,
        "resolved_county": county,
        "coordinates": {"lat": lat, "lon": lon},
        "forecast": get_forecast(county),
        "warnings": get_warnings(county),
        "rain": get_rain(lat, lon),
    }


# --------------------------------------------------------------------------
# 地震
# --------------------------------------------------------------------------
def get_earthquake_impl() -> dict:
    raw = _rest("E-A0015-001", limit=1)
    events = raw.get("records", {}).get("Earthquake", [])
    if not events:
        return {"found": False}
    e = events[0]
    info = e["EarthquakeInfo"]
    shaking_areas = [
        {"county": a["CountyName"], "intensity": a["AreaIntensity"]}
        for a in e.get("Intensity", {}).get("ShakingArea", [])
    ]
    return {
        "found": True,
        "earthquake_no": e["EarthquakeNo"],
        "origin_time": info["OriginTime"],
        "location": info["Epicenter"]["Location"],
        "magnitude": info["EarthquakeMagnitude"]["MagnitudeValue"],
        "depth_km": info["FocalDepth"],
        "report_image_uri": e.get("ReportImageURI"),
        "shaking_areas": shaking_areas,
    }


# --------------------------------------------------------------------------
# 颱風
# --------------------------------------------------------------------------
def _is_bulletin_active(bulletin: dict) -> bool:
    headline = bulletin.get("headline", "")
    if not headline or "解除" in headline:
        return False
    expires = bulletin.get("expires", "")
    if expires:
        try:
            expires_dt = datetime.fromisoformat(expires)
            now = datetime.now(expires_dt.tzinfo) if expires_dt.tzinfo else datetime.now()
            if expires_dt <= now:
                return False
        except Exception:
            pass
    return True


def get_typhoon_impl() -> dict:
    warning_raw = _rest("W-C0034-001")
    infos = warning_raw.get("records", {}).get("info", [])
    bulletin = infos[0] if infos else {}
    has_active_warning = bool(bulletin) and _is_bulletin_active(bulletin)

    result: dict = {
        "has_active_taiwan_warning": has_active_warning,
        "last_bulletin_headline": bulletin.get("headline") or None,
        "last_bulletin_time": bulletin.get("effective") or bulletin.get("onset"),
    }
    if has_active_warning:
        sections = bulletin.get("description", {}).get("section", [])
        result["description"] = [
            {"title": s.get("title"), "value": s.get("value")} for s in sections
        ]
        result["affected_areas"] = [a.get("areaDesc") for a in bulletin.get("area", [])]

    # 追蹤中的熱帶氣旋（不論是否已對台灣發布警報 —— 例如系統仍在遠洋、尚未達發布門檻）
    try:
        track_raw = _rest("W-C0034-005")
        cyclones = track_raw.get("records", {}).get("TropicalCyclones", {}).get("TropicalCyclone", [])
        tracked = []
        for c in cyclones:
            fixes = c.get("AnalysisData", {}).get("Fix", [])
            forecasts = c.get("ForecastData", {}).get("Fix", [])
            latest = fixes[-1] if fixes else None
            tracked.append({
                "name": c.get("TyphoonName"),
                "cwa_name": c.get("CwaTyphoonName"),
                "latest_position": None if not latest else {
                    "time": latest.get("DateTime"),
                    "lat": latest.get("CoordinateLatitude"),
                    "lon": latest.get("CoordinateLongitude"),
                    "max_wind_mps": latest.get("MaxWindSpeed"),
                    "max_gust_mps": latest.get("MaxGustSpeed"),
                    "pressure_hpa": latest.get("Pressure"),
                    "moving_direction": latest.get("MovingDirection"),
                    "moving_speed_kmh": latest.get("MovingSpeed"),
                },
                "forecast_points": [
                    {
                        "forecast_hour": f.get("ForecastHour"),
                        "based_on_time": f.get("InitialTime"),
                        "lat": f.get("CoordinateLatitude"),
                        "lon": f.get("CoordinateLongitude"),
                        "max_wind_mps": f.get("MaxWindSpeed"),
                    }
                    for f in forecasts[:5]
                ],
            })
        result["tracked_cyclones"] = tracked
    except Exception:
        result["tracked_cyclones"] = []

    return result
