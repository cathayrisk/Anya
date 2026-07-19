# utils/cwa_weather.py
# -*- coding: utf-8 -*-
"""
中央氣象署(CWA)開放資料 — 即時查詢工具（給 Anya_Gemma 的 @tool 用）

跟 weather/ 收集器是兩個獨立的東西：這裡只做「使用者問一次、答一次」的即時查詢，
不做輪詢/去重/狀態機，所以不依賴 weather/ repo，也不需要 Supabase。

公開函式：
  get_weather_impl(location)   — 任意地點的天氣現況（即時觀測＋預報＋特報＋降雨網格）
  get_earthquake_impl()        — 最新地震
  get_typhoon_impl()           — 目前颱風狀態（含追蹤中但尚未對台發布警報的熱帶氣旋）
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import requests
import streamlit as st

REST_BASE = "https://opendata.cwa.gov.tw/api/v1/rest/datastore"
FILE_BASE = "https://opendata.cwa.gov.tw/fileapi/v1/opendataapi"
NODATA = -99.0

_WANTED_FORECAST_ELEMENTS = {
    "Wx": "weather", "PoP": "pop", "MinT": "min_temp", "MaxT": "max_temp", "CI": "comfort",
}

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
# 地點解析：自由地名 → 縣市 + 座標，並區分「台灣以外」「查不到」
# --------------------------------------------------------------------------
# 使用者沒指定地點時的預設地點（臺北市大安區；對應收集器與頁面的預設地點）。
DEFAULT_LOCATION_NAME = "預設地點"
DEFAULT_COORDS = (25.037438, 121.553563)


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


def _nominatim(query: str, tw_only: bool):
    """回傳 Nominatim 第一筆結果（含 address），失敗回 None。"""
    params = {"q": query, "format": "json", "limit": 1, "addressdetails": 1}
    if tw_only:
        params["countrycodes"] = "tw"
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params=params,
            headers={"User-Agent": "AnyaWeatherTool/1.0 (personal use)"},
            timeout=6,
        )
        resp.raise_for_status()
        results = resp.json()
        if results:
            return results[0]
    except Exception:
        pass
    return None


def resolve_location(location: str) -> dict:
    """把使用者說的地名解析成天氣查詢用的座標。回傳：
    - {"status":"ok", "county", "lat", "lon", "is_default"}
    - {"status":"outside_taiwan", "country"}   台灣以外的真實地點
    - {"status":"not_found"}                    查不到、無法確定在台灣哪裡

    location 空字串 → 用預設地點。
    """
    location = (location or "").strip()
    if not location:
        lat, lon = DEFAULT_COORDS
        return {"status": "ok", "county": _nearest_county(lat, lon),
                "lat": lat, "lon": lon, "is_default": True}

    # 快速路徑：文字含 22 縣市官方名稱 → 一定在台灣；再抓精確座標，抓不到用縣市中心
    county = _match_county_in_text(location)
    if county is not None:
        hit = _nominatim(f"{location}, Taiwan", tw_only=True)
        if hit:
            return {"status": "ok", "county": county,
                    "lat": float(hit["lat"]), "lon": float(hit["lon"]), "is_default": False}
        lat, lon = _COUNTY_CENTROIDS[county]
        return {"status": "ok", "county": county, "lat": lat, "lon": lon, "is_default": False}

    # 限台灣查（countrycodes=tw，避免「板橋」被解到東京板橋区）＋不限國家查（判斷是否台灣以外）
    hit_tw = _nominatim(location, tw_only=True)
    hit_any = _nominatim(location, tw_only=False)

    def _imp(h) -> float:
        try:
            return float(h.get("importance", 0))
        except (TypeError, ValueError):
            return 0.0

    def _cc(h) -> str:
        return ((h.get("address", {}) or {}).get("country_code") or "").lower()

    # 全球最佳解在台灣以外，且明顯比台灣解更強（或台灣根本查不到）→ 台灣以外。
    # 例：「日本」台灣只撈到含「日本」的 POI(imp≈0.42)，全球是日本國(imp≈0.94)，差距明顯。
    # 「板橋」全球最佳解本身就在台灣，不會誤判。
    if hit_any and _cc(hit_any) != "tw" and (hit_tw is None or _imp(hit_any) >= _imp(hit_tw) + 0.25):
        address = hit_any.get("address", {}) or {}
        return {"status": "outside_taiwan", "country": address.get("country") or location}

    if hit_tw:
        lat, lon = float(hit_tw["lat"]), float(hit_tw["lon"])
        return {"status": "ok", "county": _nearest_county(lat, lon),
                "lat": lat, "lon": lon, "is_default": False}

    if hit_any and _cc(hit_any) == "tw":
        lat, lon = float(hit_any["lat"]), float(hit_any["lon"])
        return {"status": "ok", "county": _nearest_county(lat, lon),
                "lat": lat, "lon": lon, "is_default": False}

    return {"status": "not_found"}


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


# 一週鄉鎮預報 F-D0047-091：新式 JSON（Locations/Location/WeatherElement/Time），
# 12小時時段，內容比 36 小時的 F-C0032-001 豐富（含體感溫度、綜合描述）。
# (ElementName, ElementValue 的鍵, 回傳欄位名)
_FD0047_ELEMENTS = {
    "天氣現象": ("Weather", "weather"),
    "12小時降雨機率": ("ProbabilityOfPrecipitation", "pop"),
    "最高溫度": ("MaxTemperature", "max_temp"),
    "最低溫度": ("MinTemperature", "min_temp"),
    "最高體感溫度": ("MaxApparentTemperature", "max_feel"),
    "最低體感溫度": ("MinApparentTemperature", "min_feel"),
    "最大舒適度指數": ("MaxComfortIndexDescription", "comfort_max"),
    "最小舒適度指數": ("MinComfortIndexDescription", "comfort_min"),
    "天氣預報綜合描述": ("WeatherDescription", "description"),
}


def get_forecast_periods(county: str) -> list:
    """一週鄉鎮預報（F-D0047-091）的全部 12 小時時段，依時間排序。
    每段含天氣現象/降雨機率/溫度/體感溫度/舒適度/綜合描述。
    頁面通常只取前幾段當「今明」預報。"""
    raw = _rest("F-D0047-091", locationName=county)
    groups = raw.get("records", {}).get("Locations", [])
    if not groups:
        return []
    loc_list = groups[0].get("Location", [])
    loc = next((l for l in loc_list if l.get("LocationName") == county), loc_list[0] if loc_list else None)
    if not loc:
        return []

    by_start: dict[str, dict] = {}
    for el in loc.get("WeatherElement", []):
        mapping = _FD0047_ELEMENTS.get(el.get("ElementName"))
        if not mapping:
            continue
        value_key, field = mapping
        for t in el.get("Time", []):
            start = t.get("StartTime")
            values = t.get("ElementValue") or []
            if not start or not values:
                continue
            slot = by_start.setdefault(start, {"start_time": start, "end_time": t.get("EndTime")})
            slot[field] = values[0].get(value_key)

    result = []
    for start in sorted(by_start):
        p = by_start[start]
        cmin, cmax = p.pop("comfort_min", None), p.pop("comfort_max", None)
        if cmin and cmax and cmin != cmax:
            p["comfort"] = f"{cmin}至{cmax}"
        else:
            p["comfort"] = cmax or cmin
        result.append(p)
    return result


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


def _reflectivity_label(dbz: float) -> Optional[str]:
    """dBZ → 強度描述（氣象業界常用門檻，非CWA官方分級，比照特報橫幅「相關性×災種」
    自訂等級的做法）。<20 視為無明顯回波，回傳 None 讓呼叫端保持安靜。"""
    if dbz < 20:
        return None
    if dbz < 35:
        return "一般降雨"
    if dbz < 45:
        return "中到大雨"
    if dbz < 55:
        return "豪雨等級"
    return "對流旺盛"


def get_reflectivity(lat: float, lon: float) -> Optional[dict]:
    """整合回波圖（O-A0059-001）：全台 dBZ 網格，跟降雨網格共用同一套 GridDataset
    取樣機制（已用樹林站實測值 47.8dBZ／觀測降雨1.75mm 交叉驗證過對應關係正確）。
    回波比地面雨量計早一步：空中已有回波、地面雨量還沒累積出來時能提早示警。
    只給儀表板用（單次約9MB，比降雨網格重很多），不接進 get_rain()／聊天工具，
    避免每次問天氣都多付這筆網路成本。"""
    try:
        dbz = _grid("O-A0059-001").value_at(lat, lon)
    except Exception:
        return None
    if dbz is None:
        return None
    return {"dbz": round(dbz, 1), "label": _reflectivity_label(dbz)}


def get_radar_image() -> Optional[dict]:
    """樹林雷達回波圖（O-A0084-001）：CWA 已經畫好的 PNG（半徑150km，涵蓋台北盆地），
    直接回傳圖片網址供 st.image 顯示，不需要自己畫。"""
    api_key = get_cwa_api_key()
    if not api_key:
        return None
    try:
        resp = requests.get(
            f"{FILE_BASE}/O-A0084-001",
            params={"Authorization": api_key, "format": "JSON"},
            timeout=20,
        )
        resp.raise_for_status()
        dataset = resp.json()["cwaopendata"]["dataset"]
        return {"image_url": dataset["resource"]["ProductURL"], "data_time": dataset.get("DateTime")}
    except Exception:
        return None


def get_warning_details() -> list:
    """全台各別特報公告（W-C0033-002）：含公告標題、完整內容文字、影響區域與有效期間。
    跟 get_warnings(county)（W-C0033-001，只回答某縣市現在有無特報）互補——
    這裡是「發布尺度」，就算沒影響到查詢縣市的特報也會列出來。無特報時回空 list。"""
    raw = _rest("W-C0033-002")
    details = []
    for record in raw.get("records", {}).get("record", []) or []:
        ds = record.get("datasetInfo", {}) or {}
        title = ds.get("datasetDescription", "")
        if not title:
            continue
        valid = ds.get("validTime", {}) or {}

        content = (
            ((record.get("contents", {}) or {}).get("content", {}) or {}).get("contentText") or ""
        ).strip()

        affected: list[str] = []
        hazards = ((record.get("hazardConditions", {}) or {}).get("hazards", {}) or {}).get("hazard", []) or []
        for hz in hazards:
            info = hz.get("info", {}) or {}
            for loc in (info.get("affectedAreas", {}) or {}).get("location", []) or []:
                name = loc.get("locationName")
                if name and name not in affected:
                    affected.append(name)

        details.append(
            {
                "title": title,
                "content": content,
                "affected": affected,
                "issue_time": ds.get("issueTime", ""),
                "start_time": valid.get("startTime", ""),
                "end_time": valid.get("endTime", ""),
            }
        )
    return details


# 健康氣象傷害指數：三個資料集（熱/冷/溫差）結構完全相同——鄉鎮級、逐3小時，
# 每個時間點 WeatherElements 裡有一個數值指數 + 一個警示文字（達標才有值）。
_HEALTH_TZ = ZoneInfo("Asia/Taipei")
_HEALTH_INDEX = {
    "heat": {"rid": "M-A0085-001", "index_key": "HeatInjuryIndex",
             "warn_key": "HeatInjuryWarning", "label": "熱傷害", "emoji": "🥵"},
    "cold": {"rid": "F-A0085-003", "index_key": "ColdInjuryIndex",
             "warn_key": "ColdInjuryWarning", "label": "冷傷害", "emoji": "🥶"},
    "tempdiff": {"rid": "F-A0085-005", "index_key": "TemperatureDifferenceIndex",
                 "warn_key": "TemperatureDifferenceWarning", "label": "溫差提醒", "emoji": "🌡️"},
}


def _nearest_township(raw: dict, lat: float, lon: float) -> Optional[dict]:
    """健康指數資料是全台鄉鎮，用座標找最近的鄉鎮（同 get_current_conditions 的最近測站邏輯）。"""
    best, best_km = None, float("inf")
    for grp in raw.get("records", {}).get("Locations", []) or []:
        for town in grp.get("Location", []) or []:
            try:
                tlat, tlon = float(town["Latitude"]), float(town["Longitude"])
            except (KeyError, ValueError, TypeError):
                continue
            km = _haversine_km(lat, lon, tlat, tlon)
            if km < best_km:
                best_km, best = km, town
    return best


def _health_time_entries(town: dict) -> list:
    """鄉鎮的 Time[] → [(datetime, WeatherElements)]，依時間排序。IssueTime 有的帶時區、
    有的是 naive 台北時間，統一補上台北時區再比較。"""
    entries = []
    for t in town.get("Time", []) or []:
        it = t.get("IssueTime")
        try:
            dt = datetime.fromisoformat(it)
        except (TypeError, ValueError):
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_HEALTH_TZ)
        entries.append((dt, t.get("WeatherElements", {}) or {}))
    entries.sort(key=lambda x: x[0])
    return entries


def get_health_index(lat: float, lon: float, kind: str) -> Optional[dict]:
    """單一健康傷害指數（kind: 'heat'/'cold'/'tempdiff'）在座標最近鄉鎮的
    「目前時段值」與「未來24小時內的最高值/是否出現警示」。查無資料回 None。"""
    cfg = _HEALTH_INDEX.get(kind)
    if cfg is None:
        return None
    try:
        raw = _rest(cfg["rid"])
    except Exception:
        return None
    town = _nearest_township(raw, lat, lon)
    if not town:
        return None
    entries = _health_time_entries(town)
    if not entries:
        return None

    now = datetime.now(_HEALTH_TZ)
    idx_key, warn_key = cfg["index_key"], cfg["warn_key"]

    # 目前時段 = 最後一個 IssueTime <= now，都在未來則取最早一筆
    current = None
    for dt, we in entries:
        if dt <= now:
            current = (dt, we)
        elif current is None:
            current = (dt, we)
            break
        else:
            break
    if current is None:
        current = entries[0]
    cur_dt, cur_we = current

    # 未來24小時內（含目前）的最高指數與是否出現警示——回答「今天稍後會不會更嚴重」
    horizon = now.timestamp() + 24 * 3600
    peak_index, peak_warning, peak_time = cur_we.get(idx_key), "", None
    for dt, we in entries:
        if dt < cur_dt or dt.timestamp() > horizon:
            continue
        v = we.get(idx_key)
        try:
            if peak_index is None or (v is not None and float(v) > float(peak_index)):
                peak_index, peak_time = v, dt.isoformat()
        except (TypeError, ValueError):
            pass
        w = (we.get(warn_key) or "").strip()
        if w and not peak_warning:
            peak_warning = w

    return {
        "kind": kind, "label": cfg["label"], "emoji": cfg["emoji"],
        "town": town.get("TownName"),
        "index": cur_we.get(idx_key),
        "warning": (cur_we.get(warn_key) or "").strip(),
        "time": cur_dt.isoformat(),
        "peak_index_24h": peak_index,
        "peak_warning_24h": peak_warning,
        "peak_time_24h": peak_time,
    }


def get_health_indices(lat: float, lon: float) -> dict:
    """依當季挑主指數（暖季→熱傷害、冷季→冷傷害），另外任何指數有生效警示就一併帶出。
    回傳 {'primary': {...}|None, 'extra': [{...}, ...]}，供儀表板決定顯示哪些 chip。"""
    month = datetime.now(_HEALTH_TZ).month
    primary_kind = "heat" if 5 <= month <= 10 else "cold"
    secondary_kind = "cold" if primary_kind == "heat" else "heat"

    primary = get_health_index(lat, lon, primary_kind)
    extra = []
    # 反季節的傷害指數、以及溫差提醒——只有「出現警示」時才浮出，平時不佔版面
    for kind in (secondary_kind, "tempdiff"):
        h = get_health_index(lat, lon, kind)
        if h and (h.get("warning") or h.get("peak_warning_24h")):
            extra.append(h)
    return {"primary": primary, "extra": extra}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _apparent_temperature(temp_c: float, rh_pct: float, wind_mps: float) -> float:
    """CWA 官方公開的體感溫度公式（Steadman's Apparent Temperature，與澳洲氣象局
    同一套，CWA 預報產品 MaxApparentTemperature/MinApparentTemperature 也是用這個
    算出來的）。O-A0003-001 即時觀測本身沒有體感溫度欄位，用同一公式自己算，
    才能跟預報卡片的「體感」數字方法一致、可比較。
    AT = Ta + 0.33e − 0.70×風速 − 4.00，e 為水氣壓（hPa）。"""
    e = (rh_pct / 100) * 6.105 * math.exp(17.27 * temp_c / (237.7 + temp_c))
    return temp_c + 0.33 * e - 0.70 * wind_mps - 4.00


def get_current_conditions(lat: float, lon: float):
    """最近測站的即時觀測（O-A0003-001，10分鐘綜觀氣象資料）——跟 get_forecast 不同，
    這是「現在實際量到的」而不是預報值。找不到座標可用的測站就回傳 None。"""
    raw = _rest("O-A0003-001")
    stations = raw.get("records", {}).get("Station", [])

    nearest = None
    nearest_km = float("inf")
    for s in stations:
        coords = s.get("GeoInfo", {}).get("Coordinates", [])
        wgs84 = next((c for c in coords if c.get("CoordinateName") == "WGS84"), None)
        if not wgs84:
            continue
        try:
            s_lat = float(wgs84["StationLatitude"])
            s_lon = float(wgs84["StationLongitude"])
        except (KeyError, ValueError, TypeError):
            continue
        dist_km = _haversine_km(lat, lon, s_lat, s_lon)
        if dist_km < nearest_km:
            nearest_km = dist_km
            nearest = s

    if nearest is None:
        return None

    def _val(v):
        # 這個測站類型沒有該感測器時，CWA 回傳字串 "-99" 當無效值標記。
        try:
            return None if float(v) <= NODATA else v
        except (TypeError, ValueError):
            return v

    we = nearest.get("WeatherElement", {})
    temp = _val(we.get("AirTemperature"))
    rh = _val(we.get("RelativeHumidity"))
    wind = _val(we.get("WindSpeed"))

    feels_like_c = None
    try:
        if temp is not None and rh is not None and wind is not None:
            feels_like_c = round(_apparent_temperature(float(temp), float(rh), float(wind)), 1)
    except (TypeError, ValueError):
        pass

    daily = we.get("DailyExtreme", {}) or {}
    daily_high_info = (daily.get("DailyHigh", {}) or {}).get("TemperatureInfo", {}) or {}
    daily_low_info = (daily.get("DailyLow", {}) or {}).get("TemperatureInfo", {}) or {}
    gust = we.get("GustInfo", {}) or {}

    return {
        "station_name": nearest.get("StationName"),
        "station_distance_km": round(nearest_km, 1),
        "obs_time": nearest.get("ObsTime", {}).get("DateTime"),
        "weather": _val(we.get("Weather")),
        "air_temperature_c": temp,
        "feels_like_c": feels_like_c,
        "relative_humidity_pct": rh,
        "wind_speed_mps": wind,
        "wind_direction_deg": _val(we.get("WindDirection")),
        "air_pressure_hpa": _val(we.get("AirPressure")),
        "uv_index": _val(we.get("UVIndex")),
        "precipitation_now_mm": _val(we.get("Now", {}).get("Precipitation")),
        "visibility": _val(we.get("VisibilityDescription")),
        "daily_high_c": _val(daily_high_info.get("AirTemperature")),
        "daily_high_time": (daily_high_info.get("Occurred_at", {}) or {}).get("DateTime"),
        "daily_low_c": _val(daily_low_info.get("AirTemperature")),
        "daily_low_time": (daily_low_info.get("Occurred_at", {}) or {}).get("DateTime"),
        "peak_gust_mps": _val(gust.get("PeakGustSpeed")),
        "peak_gust_time": (gust.get("Occurred_at", {}) or {}).get("DateTime"),
    }


def get_weather_impl(location: str = "") -> dict:
    """location 空 → 用預設地點；台灣以外或查不到 → 回傳對應 status 讓 agent 處理。
    正常情況一次給足所有資料，讓 agent 依使用者需求自行取用。"""
    loc = resolve_location(location)

    if loc["status"] == "outside_taiwan":
        return {
            "status": "outside_taiwan",
            "location_input": location,
            "message": f"「{location}」在台灣以外（{loc.get('country', '')}），"
                       f"本服務只提供台灣的氣象資料。",
        }
    if loc["status"] == "not_found":
        return {
            "status": "not_found",
            "location_input": location,
            "message": f"無法確定「{location}」在台灣的哪個縣市，"
                       f"請補充更明確的地名（例如縣市或鄉鎮）。",
        }

    county, lat, lon = loc["county"], loc["lat"], loc["lon"]
    return {
        "status": "ok",
        "location_input": location or DEFAULT_LOCATION_NAME,
        "used_default_location": loc.get("is_default", False),
        "resolved_county": county,
        "coordinates": {"lat": lat, "lon": lon},
        "current_conditions": get_current_conditions(lat, lon),
        "forecast": get_forecast_periods(county),
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
