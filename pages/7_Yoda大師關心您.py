# filename: companion_fortune_agent_yoda_kerykeion.py

import os
import asyncio
import re
import time
from datetime import datetime
from typing import Dict, Optional, Any, List, Literal, Tuple

from pydantic import BaseModel

from agents import Agent, Runner, SQLiteSession
from agents import function_tool
from agents.extensions.memory import EncryptedSession

from agents import ModelSettings
from openai.types.shared.reasoning import Reasoning

# Kerykeion：占星命盤計算（離線模式 + ChartDataFactory）
from kerykeion import AstrologicalSubjectFactory, ChartDataFactory, to_context

import pytz


# ============================================================
# 1. 使用者檔案儲存（示範用：記憶體版）
#    注意：程式重啟會遺失。如需長期保存請改成 SQLite table。
# ============================================================

PROFILE_STORE: Dict[str, Dict[str, Any]] = {}


class ProfileDelta(BaseModel):
    name: Optional[str] = None
    birthdate: Optional[str] = None  # YYYY-MM-DD
    birth_time: Optional[str] = None  # HH:MM
    birth_city: Optional[str] = None
    birth_country: Optional[str] = None
    lng: Optional[float] = None
    lat: Optional[float] = None
    tz_str: Optional[str] = None
    gender: Optional[str] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None

    # Forrest 敘事式解讀定調
    consult_goal: Optional[str] = None
    consult_focus: Optional[str] = None  # self/relationship/career/timing/block/other


class AspectConfig(BaseModel):
    name: str
    orb: float


def _get_user_profile_impl(user_id: str) -> Any:
    return PROFILE_STORE.get(user_id)


def _update_user_profile_impl(
    user_id: str,
    # --- ProfileDelta fields (all optional) ---
    name: Optional[str] = None,
    birthdate: Optional[str] = None,
    birth_time: Optional[str] = None,
    birth_city: Optional[str] = None,
    birth_country: Optional[str] = None,
    lng: Optional[float] = None,
    lat: Optional[float] = None,
    tz_str: Optional[str] = None,
    gender: Optional[str] = None,
    tags: Optional[List[str]] = None,
    notes: Optional[str] = None,
    consult_goal: Optional[str] = None,
    consult_focus: Optional[str] = None,
) -> Any:
    """
    真正更新 profile 的實作（可被 orchestrator 直接呼叫）。
    Tool 版本會包這個函式，避免 strict schema 的 additionalProperties 問題。
    """
    current = PROFILE_STORE.get(user_id, {}).copy()

    delta_model = ProfileDelta(
        name=name,
        birthdate=birthdate,
        birth_time=birth_time,
        birth_city=birth_city,
        birth_country=birth_country,
        lng=lng,
        lat=lat,
        tz_str=tz_str,
        gender=gender,
        tags=tags,
        notes=notes,
        consult_goal=consult_goal,
        consult_focus=consult_focus,
    )
    delta = delta_model.model_dump(exclude_none=True, exclude_unset=True)

    # tags 合併去重
    new_tags = delta.pop("tags", None)
    if new_tags is not None:
        existing_tags = current.get("tags", [])
        if not isinstance(existing_tags, list):
            existing_tags = [existing_tags]
        current["tags"] = list(dict.fromkeys(existing_tags + new_tags))

    current.update(delta)
    PROFILE_STORE[user_id] = current
    return current


# tools（注意：這裡是 tool 物件，不是原函式）
get_user_profile = function_tool(_get_user_profile_impl)
update_user_profile = function_tool(_update_user_profile_impl)


# ============================================================
# 2. Kerykeion Tools：本命盤 / 行運 / 雙人合盤（全部離線 + 文字輸出）
# ============================================================

_CITY_LOCATION_DB = [
    {
        "aliases": ["taipei", "taipei city", "台北", "台北市"],
        "nation_aliases": ["tw", "taiwan", "中華民國", "臺灣"],
        "lng": 121.5654,
        "lat": 25.0330,
        "tz_str": "Asia/Taipei",
    },
]


def _normalize_str(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    return s.strip().lower()


def _parse_date(date_str: str, field_name: str) -> Dict[str, Any]:
    """解析 YYYY-MM-DD（允許 YYYY/MM/DD），回傳 dict 或錯誤 dict。"""
    try:
        date_str = date_str.strip().replace("/", "-")
        year, month, day = map(int, date_str.split("-"))
        return {"year": year, "month": month, "day": day}
    except Exception:
        return {
            "error": f"INVALID_{field_name.upper()}",
            "detail": f"無法解析 {field_name} '{date_str}'，請使用 YYYY-MM-DD 格式。",
        }


def _parse_time(time_str: Optional[str], default_noon: bool = True) -> Dict[str, Any]:
    """解析 HH:MM，或允許缺失時預設 12:00。"""
    if time_str:
        try:
            hour, minute = map(int, time_str.strip().split(":"))
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                raise ValueError("out of range")
            return {"hour": hour, "minute": minute, "approximated": False}
        except Exception:
            return {
                "error": "INVALID_BIRTHTIME",
                "detail": f"無法解析出生時間 '{time_str}'，請使用 HH:MM 24 小時制格式。",
            }
    if default_noon:
        return {"hour": 12, "minute": 0, "approximated": True}
    return {"error": "MISSING_BIRTHTIME", "detail": "缺少出生時間且未允許預設值。"}


def _autofill_location(
    city: Optional[str],
    nation: Optional[str],
    lng: Optional[float],
    lat: Optional[float],
    tz_str: Optional[str],
) -> Dict[str, Any]:
    """
    若 lng/lat/tz_str 有缺，但 city/nation 有提供，嘗試用內建城市資料自動補齊。
    若完全沒有任何地點資訊，則預設使用台北市作為約略位置。
    """
    if lng is not None and lat is not None and tz_str:
        return {"lng": lng, "lat": lat, "tz_str": tz_str, "autofilled": False}

    norm_city = _normalize_str(city)
    norm_nation = _normalize_str(nation)

    if norm_city is not None:
        for entry in _CITY_LOCATION_DB:
            city_match = (
                norm_city in entry["aliases"]
                or any(alias in norm_city for alias in entry["aliases"])
            )
            if not city_match:
                continue

            if norm_nation and not any(
                norm_nation == na or na in norm_nation for na in entry["nation_aliases"]
            ):
                continue

            return {
                "lng": entry["lng"],
                "lat": entry["lat"],
                "tz_str": entry["tz_str"],
                "autofilled": True,
            }

    if (
        lng is None
        and lat is None
        and not tz_str
        and norm_city is None
        and norm_nation is None
        and _CITY_LOCATION_DB
    ):
        entry = _CITY_LOCATION_DB[0]
        return {"lng": entry["lng"], "lat": entry["lat"], "tz_str": entry["tz_str"], "autofilled": True}

    return {"lng": lng, "lat": lat, "tz_str": tz_str, "autofilled": False}


@function_tool
def get_natal_chart_context(
    user_id: str,
    name: str,
    birthdate: str,
    birth_time: Optional[str] = None,
    city: Optional[str] = None,
    nation: Optional[str] = None,
    lng: Optional[float] = None,
    lat: Optional[float] = None,
    tz_str: Optional[str] = None,
    zodiac_type: str = "Tropical",
    houses_system_identifier: str = "P",
    sidereal_mode: Optional[str] = None,
    active_points: Optional[List[str]] = None,
    active_aspects: Optional[List[AspectConfig]] = None,
    calculate_lunar_phase: bool = True,
) -> Any:
    """生成本命盤資料（離線），回傳 LLM 可讀的 context。"""
    date_parsed = _parse_date(birthdate, "birthdate")
    if "error" in date_parsed:
        return date_parsed
    year, month, day = date_parsed["year"], date_parsed["month"], date_parsed["day"]

    time_parsed = _parse_time(birth_time, default_noon=True)
    if "error" in time_parsed:
        return time_parsed
    hour, minute = time_parsed["hour"], time_parsed["minute"]
    time_approx = time_parsed["approximated"]

    auto_loc = _autofill_location(city, nation, lng, lat, tz_str)
    lng = auto_loc["lng"]
    lat = auto_loc["lat"]
    tz_str = auto_loc["tz_str"]
    location_autofilled = auto_loc["autofilled"]

    if not (lng is not None and lat is not None and tz_str):
        return {
            "error": "MISSING_LOCATION_OFFLINE_ONLY",
            "detail": (
                "目前僅支援離線命盤計算，請提供 lng、lat 與 tz_str（例如 'Asia/Taipei'）。"
                "city / nation 目前只對少數城市（例如台北）有內建座標。"
            ),
        }

    try:
        extra_kwargs: Dict[str, Any] = {}
        if sidereal_mode is not None:
            extra_kwargs["sidereal_mode"] = sidereal_mode
        if active_points is not None:
            extra_kwargs["active_points"] = active_points

        subject = AstrologicalSubjectFactory.from_birth_data(
            name=name,
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=minute,
            lng=lng,
            lat=lat,
            tz_str=tz_str,
            zodiac_type=zodiac_type,
            houses_system_identifier=houses_system_identifier,
            calculate_lunar_phase=calculate_lunar_phase,
            online=False,
            **extra_kwargs,
        )

        chart_kwargs: Dict[str, Any] = {}
        if active_aspects is not None:
            chart_kwargs["active_aspects"] = [a.model_dump() for a in active_aspects]

        chart_data = ChartDataFactory.create_natal_chart_data(subject, **chart_kwargs)

        combined_context_parts = [
            "# Natal subject",
            to_context(subject),
            "",
            "# Natal chart data",
            to_context(chart_data),
        ]
        if getattr(subject, "lunar_phase", None):
            combined_context_parts.extend(["", "# Lunar phase", to_context(subject.lunar_phase)])

        combined_context = "\n".join(combined_context_parts)

        result: Dict[str, Any] = {
            "user_id": user_id,
            "name": name,
            "birthdate": birthdate,
            "birth_time": f"{hour:02d}:{minute:02d}",
            "location": {"lng": lng, "lat": lat, "tz_str": tz_str, "city": city, "nation": nation},
            "zodiac_type": zodiac_type,
            "houses_system_identifier": houses_system_identifier,
            "sidereal_mode": sidereal_mode,
            "context": combined_context,
            "summary": {
                "chart_type": getattr(chart_data, "chart_type", None),
                "num_aspects": len(getattr(chart_data, "aspects", [])),
                "has_lunar_phase": bool(getattr(subject, "lunar_phase", None)),
            },
        }
        if time_approx:
            result["warning"] = "BIRTH_TIME_APPROXIMATED"
        if location_autofilled:
            result["location_warning"] = "LOCATION_APPROXIMATED"
        return result

    except Exception as e:
        return {"error": "KERYKEION_ERROR", "detail": f"計算本命盤時發生錯誤: {e}"}


@function_tool
def get_transit_chart_context(
    user_id: str,
    name: str,
    birthdate: str,
    birth_time: Optional[str],
    lng: float,
    lat: float,
    tz_str: str,
    zodiac_type: str = "Tropical",
    houses_system_identifier: str = "P",
    sidereal_mode: Optional[str] = None,
    active_points: Optional[List[str]] = None,
    calculate_lunar_phase: bool = True,
    transit_datetime: Optional[str] = None,
    active_aspects: Optional[List[AspectConfig]] = None,
) -> Any:
    """生成本命 + 行運 context（離線）。"""
    date_parsed = _parse_date(birthdate, "birthdate")
    if "error" in date_parsed:
        return date_parsed
    year, month, day = date_parsed["year"], date_parsed["month"], date_parsed["day"]

    time_parsed = _parse_time(birth_time, default_noon=True)
    if "error" in time_parsed:
        return time_parsed
    n_hour, n_minute = time_parsed["hour"], time_parsed["minute"]
    natal_time_approx = time_parsed["approximated"]

    extra_kwargs: Dict[str, Any] = {}
    if sidereal_mode is not None:
        extra_kwargs["sidereal_mode"] = sidereal_mode
    if active_points is not None:
        extra_kwargs["active_points"] = active_points

    try:
        natal_subject = AstrologicalSubjectFactory.from_birth_data(
            name=name,
            year=year,
            month=month,
            day=day,
            hour=n_hour,
            minute=n_minute,
            lng=lng,
            lat=lat,
            tz_str=tz_str,
            zodiac_type=zodiac_type,
            houses_system_identifier=houses_system_identifier,
            calculate_lunar_phase=calculate_lunar_phase,
            online=False,
            **extra_kwargs,
        )

        if transit_datetime:
            normalized = transit_datetime.replace("T", " ")
            try:
                dt = datetime.strptime(normalized, "%Y-%m-%d %H:%M")
            except Exception:
                return {
                    "error": "INVALID_TRANSIT_DATETIME",
                    "detail": (
                        f"無法解析 transit_datetime '{transit_datetime}'，"
                        "請使用 'YYYY-MM-DD HH:MM' 或 'YYYY-MM-DDTHH:MM' 格式。"
                    ),
                }
        else:
            tz = pytz.timezone(tz_str)
            dt = datetime.now(tz)

        transit_subject = AstrologicalSubjectFactory.from_birth_data(
            name=f"{name} Transit",
            year=dt.year,
            month=dt.month,
            day=dt.day,
            hour=dt.hour,
            minute=dt.minute,
            lng=lng,
            lat=lat,
            tz_str=tz_str,
            zodiac_type=zodiac_type,
            houses_system_identifier=houses_system_identifier,
            calculate_lunar_phase=calculate_lunar_phase,
            online=False,
            **extra_kwargs,
        )

        chart_kwargs: Dict[str, Any] = {}
        if active_aspects is not None:
            chart_kwargs["active_aspects"] = [a.model_dump() for a in active_aspects]

        transit_chart = ChartDataFactory.create_transit_chart_data(
            natal_subject=natal_subject,
            transit_subject=transit_subject,
            **chart_kwargs,
        )

        combined_context = "\n".join(
            [
                "# Natal subject",
                to_context(natal_subject),
                "",
                "# Transit subject",
                to_context(transit_subject),
                "",
                "# Transit chart data",
                to_context(transit_chart),
            ]
        )

        result: Dict[str, Any] = {
            "user_id": user_id,
            "name": name,
            "birthdate": birthdate,
            "birth_time": f"{n_hour:02d}:{n_minute:02d}",
            "location": {"lng": lng, "lat": lat, "tz_str": tz_str},
            "zodiac_type": zodiac_type,
            "houses_system_identifier": houses_system_identifier,
            "sidereal_mode": sidereal_mode,
            "transit_datetime": dt.isoformat(),
            "context": combined_context,
            "summary": {"num_transit_aspects": len(getattr(transit_chart, "aspects", []))},
        }
        if natal_time_approx:
            result["warning"] = "BIRTH_TIME_APPROXIMATED"
        if not transit_datetime:
            result["note"] = "TRANSIT_TIME_NOW"
        return result

    except Exception as e:
        return {"error": "KERYKEION_ERROR", "detail": f"計算行運時發生錯誤: {e}"}


@function_tool
def get_synastry_chart_context(
    primary_user_id: str,
    primary_name: str,
    primary_birthdate: str,
    primary_birth_time: Optional[str],
    primary_lng: float,
    primary_lat: float,
    primary_tz_str: str,
    partner_name: str,
    partner_birthdate: str,
    partner_birth_time: Optional[str],
    partner_lng: float,
    partner_lat: float,
    partner_tz_str: str,
    zodiac_type: str = "Tropical",
    houses_system_identifier: str = "P",
    sidereal_mode: Optional[str] = None,
    active_points: Optional[List[str]] = None,
) -> Any:
    """生成雙人合盤 context（離線）。"""
    p_date = _parse_date(primary_birthdate, "primary_birthdate")
    if "error" in p_date:
        return p_date
    o_date = _parse_date(partner_birthdate, "partner_birthdate")
    if "error" in o_date:
        return o_date

    p_time = _parse_time(primary_birth_time, default_noon=True)
    if "error" in p_time:
        return p_time
    o_time = _parse_time(partner_birth_time, default_noon=True)
    if "error" in o_time:
        return o_time

    p_hour, p_minute = p_time["hour"], p_time["minute"]
    o_hour, o_minute = o_time["hour"], o_time["minute"]
    p_approx, o_approx = p_time["approximated"], o_time["approximated"]

    try:
        extra_kwargs: Dict[str, Any] = {}
        if sidereal_mode is not None:
            extra_kwargs["sidereal_mode"] = sidereal_mode
        if active_points is not None:
            extra_kwargs["active_points"] = active_points

        primary_subject = AstrologicalSubjectFactory.from_birth_data(
            name=primary_name,
            year=p_date["year"],
            month=p_date["month"],
            day=p_date["day"],
            hour=p_hour,
            minute=p_minute,
            lng=primary_lng,
            lat=primary_lat,
            tz_str=primary_tz_str,
            zodiac_type=zodiac_type,
            houses_system_identifier=houses_system_identifier,
            online=False,
            **extra_kwargs,
        )

        partner_subject = AstrologicalSubjectFactory.from_birth_data(
            name=partner_name,
            year=o_date["year"],
            month=o_date["month"],
            day=o_date["day"],
            hour=o_hour,
            minute=o_minute,
            lng=partner_lng,
            lat=partner_lat,
            tz_str=partner_tz_str,
            zodiac_type=zodiac_type,
            houses_system_identifier=houses_system_identifier,
            online=False,
            **extra_kwargs,
        )

        synastry_chart = ChartDataFactory.create_synastry_chart_data(
            first_subject=primary_subject,
            second_subject=partner_subject,
            include_house_comparison=True,
            include_relationship_score=True,
        )

        combined_context = "\n".join(
            [
                "# Primary natal subject",
                to_context(primary_subject),
                "",
                "# Partner natal subject",
                to_context(partner_subject),
                "",
                "# Synastry chart data",
                to_context(synastry_chart),
            ]
        )

        summary: Dict[str, Any] = {
            "has_relationship_score": bool(getattr(synastry_chart, "relationship_score", None)),
        }
        if synastry_chart.relationship_score:
            summary["relationship_score"] = synastry_chart.relationship_score.score_value

        result: Dict[str, Any] = {
            "primary_user_id": primary_user_id,
            "primary": {
                "name": primary_name,
                "birthdate": primary_birthdate,
                "birth_time": f"{p_hour:02d}:{p_minute:02d}",
                "location": {"lng": primary_lng, "lat": primary_lat, "tz_str": primary_tz_str},
            },
            "partner": {
                "name": partner_name,
                "birthdate": partner_birthdate,
                "birth_time": f"{o_hour:02d}:{o_minute:02d}",
                "location": {"lng": partner_lng, "lat": partner_lat, "tz_str": partner_tz_str},
            },
            "zodiac_type": zodiac_type,
            "houses_system_identifier": houses_system_identifier,
            "sidereal_mode": sidereal_mode,
            "context": combined_context,
            "summary": summary,
        }

        warnings: List[str] = []
        if p_approx:
            warnings.append("PRIMARY_BIRTH_TIME_APPROXIMATED")
        if o_approx:
            warnings.append("PARTNER_BIRTH_TIME_APPROXIMATED")
        if warnings:
            result["warnings"] = warnings
        return result

    except Exception as e:
        return {"error": "KERYKEION_ERROR", "detail": f"計算雙人合盤時發生錯誤: {e}"}


# ============================================================
# 3. 子 Agent：Profile / 占星解讀 / 心靈陪伴（Yoda）
# ============================================================

profile_agent = Agent(
    name="Profile builder agent",
    model="gpt-4.1-mini",
    tools=[get_user_profile, update_user_profile],
    instructions=r"""
你是一個溫柔的資料整理者，目標是逐步理解使用者（不審問），並把能確定的資料寫入。

你可用工具：
- get_user_profile(user_id) -> Optional[dict]
- update_user_profile(user_id, name=..., birthdate=..., birth_time=..., birth_city=..., tz_str=..., lng=..., lat=..., tags=..., notes=..., consult_goal=..., consult_focus=...) -> dict

規則：
- birthdate 只有在對方明確給完整日期才寫入（包含年份）。
- birth_time 只有在對方明確給到 HH:MM 才寫入。
- 台北例外規則：若出生城市包含「台北/台北市/Taipei/Taipei City」且未否定，直接寫入：
  birth_country="TW", tz_str="Asia/Taipei", lng=121.5654, lat=25.0330
- consult_goal：可用對方原話摘要成一句，但不要腦補對方沒說的細節。
- consult_focus：self/relationship/career/timing/block/other

限制：
- 不要提到你正在呼叫工具
- 不要提到 user_id
- 使用繁體中文回覆
""",
)

fortune_agent = Agent(
    name="Fortune interpretation agent",
    model="gpt-5.2",
    model_settings=ModelSettings(reasoning=Reasoning(effort="medium", summary="auto")),
    tools=[
        get_user_profile,
        get_natal_chart_context,
        get_transit_chart_context,
        get_synastry_chart_context,
    ],
    instructions=r"""
System: Internal-only fortune interpretation module.
You NEVER talk to the end user directly.

你要做的是：把西洋占星盤面（Kerykeion 工具輸出）整理成 Steven Forrest 三本書方法論的
「心理占星 + 生命敘事」解讀骨架（不引用書中文字，僅使用其精神）。

Input format:
- Raw input contains:
  "[SYSTEM INFO] The current user's id is `some-id`."
  "[USER MESSAGE] ...."
You must parse user_id from SYSTEM INFO and use that in tool calls.

# 重要：完整命盤（FULL_CHART）規則 —— 要提供 Kerykeion 工具輸出的完整命盤資訊
- 若使用者明確要求「完整命盤/排盤明細/完整盤面/明細/原始輸出」：
  1) 優先嘗試取得本命盤（natal，必要時再加 transit/synastry）。
  2) HAS_CHART 時必須輸出 [FULL_CHART]...[/FULL_CHART]，
     FULL_CHART 內容包含 Kerykeion 工具回傳的 context（可原樣放入，不要大幅刪減）。
  3) 即使 consult_goal 不清楚，也允許產出 HAS_CHART（CONSULT_GOAL 填 not_provided）。

# 一般情況：先定調
- 先呼叫 get_user_profile(user_id)。
- 從 profile 的 consult_goal/consult_focus 或 USER_MESSAGE 判斷諮詢目標。
- 若「沒有要求完整命盤」且 consult_goal 仍不清楚：輸出 NO_CHART / REASON: missing_consult_goal
  （NO_CHART 時禁止提任何盤面細節）。

# 需求分類（用於 CHART_TYPES）
- USER_MESSAGE 提到「行運/運勢/最近幾個月/未來一年」=> natal+transit
- 提到「合盤/關係盤/配不配/我們兩個」=> 資料齊才 natal+synastry；不齊 NO_CHART: missing_partner_data
- 其餘提到命盤/星座=> natal

# 台北市不要卡在 lng/lat/tz_str
- 只要 profile 具備 birthdate（含年份）就可嘗試排盤：
  * 若已有 lng/lat/tz_str：照常用。
  * 若缺，但 birth_city 顯示「台北/台北市/Taipei」：呼叫 get_natal_chart_context 傳入 city=birth_city。
  * 其他城市且缺座標：NO_CHART（missing_location）

# Output contract（嚴格遵守）
只能輸出以下格式之一，且不得在區塊外輸出任何文字：

1) HAS_CHART：
[FORTUNE_SUMMARY]
STATUS: HAS_CHART
CHART_TYPES: (例如 "natal" / "natal+transit" / "natal+synastry")
CONSULT_GOAL: （第三人稱；若要求完整命盤但未定調，可填 not_provided）
CONSULT_FOCUS: （self/relationship/career/timing/block/other；若未知可填 other）

INNER_SKY: （4~8 行濃縮）
YESTERDAYS_SKY: （4~8 行濃縮）
CHANGING_SKY: （4~8 行濃縮）

THEME: ...
SHADOW: ...
GIFT: ...
CHOICE: ...
PRACTICE: ...

（若含 synastry，允許加一行）
DYNAMIC: ...

（若 location_warning 存在，允許加一行）
NOTE: ...

[FULL_CHART]
（僅在使用者要求完整命盤/排盤明細時輸出，內容包含 Kerykeion 工具 context）
...
[/FULL_CHART]

[/FORTUNE_SUMMARY]

2) NO_CHART：
[FORTUNE_SUMMARY]
STATUS: NO_CHART
REASON: (missing_birth_data / missing_location / missing_partner_data / missing_consult_goal / kerykeion_error / other)
THEME: ...
SHADOW: ...
GIFT: ...
CHOICE: ...
PRACTICE: ...
[/FORTUNE_SUMMARY]
""",
)

counselor_agent = Agent(
    name="Emotional companion agent",
    model="gpt-5.2",
    model_settings=ModelSettings(reasoning=Reasoning(effort="none", summary="auto"), temperature=0),
    tools=[],
    instructions=r"""
You are the main emotional companion whose persona is inspired by Master Yoda from Star Wars.

Context:
- The orchestrator will prepend your input with:
  (optional) [PROFILE_HINT] ... [/PROFILE_HINT]
  (optional) [FORTUNE_SUMMARY] ... [/FORTUNE_SUMMARY]
  (optional) [SYSTEM_HINT] ... [/SYSTEM_HINT]
  [USER_MESSAGE] ... [/USER_MESSAGE]

定位：
- 你只對使用者說話；你不做占星計算。
- 若有 [FORTUNE_SUMMARY]：你只能用它轉述/安撫/落地，不可新增任何占星細節。
- 若沒有 [FORTUNE_SUMMARY]：只做情緒陪伴與定調問題，不要假裝有命盤內容。

# PROFILE_HINT
- 若看到 consult_goal/consult_focus：可用來更貼近使用者，但不要像在揭露資料。
  用「若我沒理解錯，你在意的可能是…」這種措辭。

# 原力（The Force）— 溫柔但有界線版
- 原力是隱喻：覺察、呼吸、界線、價值選擇。
- 每次回覆提到「原力」最多 0～2 次；禁止權威口吻（禁：原力告訴你/你必須）。
- 允許感受，但也要守界線：同時做到共感與界線提醒。
- 提到原力後，下一句要接具體可做的小步驟（5～20 分鐘級）。

# Yoda style
- 至少 2～4 句輕微倒裝
- 先共感，再引導，最後給 1～2 個小方向
- 多用短句分段
- 不做醫療/法律/投資建議；若有自傷他傷傾向，鼓勵現實專業協助

# FORTUNE_SUMMARY 使用規則
- STATUS: HAS_CHART：
  * 只能轉述/改寫摘要內容，不可新增占星細節（不得腦補星座/度數/宮位/相位）。
  * 把 PRACTICE 變成 1～2 個可執行小步驟。
  * 若包含 [FULL_CHART]：用 Markdown 區塊把 FULL_CHART 原樣呈現（不要大幅刪改）。
- STATUS: NO_CHART：
  * 不提占星細節。
  * 若缺 consult_goal：用 1～2 個短問題定調（不審問、可選擇）。

Language:
- 繁體中文
- 可用適度 Markdown
- 不提 tools / user_id / Agent
""",
)


# ============================================================
# 4. Orchestrator：Python 編排 + fortune 快取 + synastry 換對象快取失效
# ============================================================

AstroIntent = Literal["yes", "maybe", "no"]
RequestKind = Literal["natal", "transit", "synastry", "unknown"]

_FORCE_ASTRO_TAGS = ["#占星", "#命盤", "/astro"]
_FORCE_NO_ASTRO_TAGS = ["#不占星", "/noastro"]

_ASTRO_KEYWORDS_YES = [
    "星座", "命盤", "占星", "本命盤",
    "行運", "運勢", "流年", "推運", "次限", "太陽弧",
    "合盤", "關係盤", "配不配", "我們兩個的盤",
    "上升", "月亮", "太陽星座", "宮位", "相位",
]

_ASTRO_PATTERNS_MAYBE = [
    r"我適合(什麼|哪種)工作",
    r"我(的)?(個性|天賦|優勢|弱點)是(什麼|怎樣)",
    r"我為什麼(一直|總是).+卡",
    r"(最近|這陣子|接下來|未來).+(特別難|特別想改變|很不順|壓力很大)",
    r"我該怎麼(走|選|做決定)",
    r"我跟(另一半|伴侶|他|她).+(怎麼相處|常吵架|模式)",
]

_FULL_CHART_KEYWORDS = [
    "完整命盤", "排盤明細", "完整盤", "完整盤面", "命盤明細", "原始輸出", "FULL_CHART"
]


def _now_ts() -> float:
    return time.time()


def _get_fortune_cache_ttl() -> int:
    # 預設 600 秒；你可在環境變數調整
    return int(os.environ.get("FORTUNE_CACHE_TTL", "600"))


def _wants_full_chart(msg: str) -> bool:
    s = msg or ""
    return any(k in s for k in _FULL_CHART_KEYWORDS) or ("#fullchart" in s.lower())


def _classify_astro_intent(user_message: str) -> AstroIntent:
    msg = user_message or ""

    if any(tag in msg for tag in _FORCE_NO_ASTRO_TAGS):
        return "no"
    if any(tag in msg for tag in _FORCE_ASTRO_TAGS):
        return "yes"

    if any(k in msg for k in _ASTRO_KEYWORDS_YES):
        return "yes"

    for pat in _ASTRO_PATTERNS_MAYBE:
        try:
            if re.search(pat, msg):
                return "maybe"
        except re.error:
            continue

    return "no"


def _infer_request_kind(user_message: str) -> RequestKind:
    s = user_message or ""
    if any(k in s for k in ["合盤", "關係盤", "配不配", "我們兩個"]):
        return "synastry"
    if any(k in s for k in ["行運", "運勢", "流年", "推運", "次限", "太陽弧", "未來幾個月", "最近這幾個月", "未來一年"]):
        return "transit"
    if any(k in s for k in ["命盤", "本命盤", "星座", "上升", "月亮", "太陽星座"]):
        return "natal"
    return "unknown"


def _extract_consult_from_text(user_message: str) -> Dict[str, Any]:
    """
    超輕量抽取：使用者若用「我想/我困擾/我在意」說目標，就存 consult_goal/focus。
    （更精準的抽取可以交給 profile_agent）
    """
    msg = (user_message or "").strip()
    if not msg:
        return {}

    out: Dict[str, Any] = {}

    m = re.search(r"(我想|想要|想解決|我在意|我困擾|我卡在|我卡住)(.{2,60})", msg)
    if m:
        phrase = (m.group(1) + m.group(2)).strip()
        out["consult_goal"] = phrase[:120]

    if any(k in msg for k in ["另一半", "伴侶", "感情", "關係", "吵架", "分手", "曖昧"]):
        out["consult_focus"] = "relationship"
    elif any(k in msg for k in ["工作", "職涯", "職場", "轉職", "升遷", "主管", "同事"]):
        out["consult_focus"] = "career"
    elif any(k in msg for k in ["最近", "這陣子", "未來", "接下來", "幾個月", "一年"]):
        out["consult_focus"] = "timing"
    elif any(k in msg for k in ["卡住", "卡關", "拖延", "焦慮", "恐懼", "不敢", "沒力"]):
        out["consult_focus"] = "block"
    elif any(k in msg for k in ["我這個人", "性格", "天賦", "優勢", "弱點", "我是怎樣的人"]):
        out["consult_focus"] = "self"

    return out


def _build_profile_hint(profile: Optional[Dict[str, Any]]) -> str:
    """
    只注入 counselor 需要的「諮詢定調」資訊，避免把出生資料也塞進去。
    """
    if not profile:
        return ""

    goal = profile.get("consult_goal")
    focus = profile.get("consult_focus")

    parts = []
    if isinstance(goal, str) and goal.strip():
        parts.append(f"consult_goal: {goal.strip()}")
    if isinstance(focus, str) and focus.strip():
        parts.append(f"consult_focus: {focus.strip()}")

    if not parts:
        return ""

    return "[PROFILE_HINT]\n" + "\n".join(parts) + "\n[/PROFILE_HINT]\n\n"


def _profile_fingerprint(profile: Dict[str, Any]) -> Tuple:
    """盤面相關欄位變了，就視為不同盤，快取失效。"""
    return (
        profile.get("birthdate"),
        profile.get("birth_time"),
        profile.get("birth_city"),
        profile.get("birth_country"),
        profile.get("lng"),
        profile.get("lat"),
        profile.get("tz_str"),
    )


def _fortune_contains_full_chart(fortune_summary: str) -> bool:
    if not fortune_summary:
        return False
    return "[FULL_CHART]" in fortune_summary and "[/FULL_CHART]" in fortune_summary


def _strip_full_chart_block(fortune_summary: str) -> str:
    """
    平常不要把 FULL_CHART 注入 counselor（太大、拖慢/耗 token）。
    使用者要求完整明細時才保留。
    """
    if not fortune_summary:
        return fortune_summary
    return re.sub(r"\[FULL_CHART\][\s\S]*?\[/FULL_CHART\]\n?", "", fortune_summary).strip()


def _synastry_partner_change_hint(user_message: str) -> bool:
    """
    不做 partner fingerprint 的前提下：
    只要使用者明確說「換人/新對象/不是這個人」或重新提供對方資料，就 bust synastry 快取。
    """
    msg = (user_message or "").strip()

    manual_tags = ["#換對象", "#新對象", "#重新合盤", "#newpartner", "/newpartner", "/resynastry"]
    if any(t.lower() in msg.lower() for t in manual_tags):
        return True

    cues = [
        "換一個", "換個", "換人", "換對象", "換另一個", "換別人",
        "另一個人", "新對象", "新的對象", "改成他", "改成她", "不是這個人",
        "我跟別人", "換成另一半", "不同的伴侶",
    ]
    if any(c in msg for c in cues):
        return True

    if re.search(r"(對方|他|她).*(生日|出生|生於|\d{4}[/-]\d{1,2}[/-]\d{1,2})", msg):
        return True

    return False


# ---- Fortune cache：多份快取（user_id, request_kind） ----
_FORTUNE_CACHE: Dict[tuple, Dict[str, Any]] = {}
# key = (user_id, request_kind)
# entry:
# {
#   "created_at": float,
#   "request_kind": str,
#   "profile_fp": tuple,
#   "fortune_summary": str,
#   "has_full_chart": bool
# }


def _get_cached_fortune(
    user_id: str,
    request_kind: RequestKind,
    profile: Dict[str, Any],
    wants_full: bool,
) -> Optional[str]:
    key = (user_id, request_kind)
    entry = _FORTUNE_CACHE.get(key)
    if not entry:
        return None

    ttl = _get_fortune_cache_ttl()
    if (_now_ts() - float(entry.get("created_at", 0))) > ttl:
        return None

    if entry.get("profile_fp") != _profile_fingerprint(profile):
        return None

    if wants_full and not bool(entry.get("has_full_chart")):
        return None

    return entry.get("fortune_summary")


def _set_cached_fortune(
    user_id: str,
    request_kind: RequestKind,
    profile: Dict[str, Any],
    fortune_summary: str,
) -> None:
    key = (user_id, request_kind)
    _FORTUNE_CACHE[key] = {
        "created_at": _now_ts(),
        "request_kind": request_kind,
        "profile_fp": _profile_fingerprint(profile),
        "fortune_summary": fortune_summary,
        "has_full_chart": _fortune_contains_full_chart(fortune_summary),
    }


async def _maybe_run_profile_agent(user_id: str, system_info: str, user_message: str, session: EncryptedSession) -> None:
    """
    可選：更精準抽取生日/地點/時間/目標，才跑 profile_agent。
    預設開啟，但只在「像在提供資料」時才跑（避免每輪多一次模型呼叫）。
    """
    use_profile_agent = os.environ.get("USE_PROFILE_AGENT", "1") == "1"
    if not use_profile_agent:
        return

    if not re.search(
        r"(生日|出生|生於|時間|地點|城市|時區|經緯度|我想|想解決|我困擾|我在意)",
        user_message or "",
    ):
        return

    try:
        full_input = system_info + f"[USER MESSAGE] {user_message}"
        await Runner.run(profile_agent, input=full_input, session=session)
    except Exception:
        return


async def _run_fortune(user_id: str, system_info: str, user_message: str, session: EncryptedSession) -> Optional[str]:
    try:
        full_input = system_info + f"[USER MESSAGE] {user_message}"
        r = await Runner.run(fortune_agent, input=full_input, session=session)
        return r.final_output
    except Exception:
        return None


async def _run_counselor(
    user_message: str,
    session: EncryptedSession,
    profile_hint: str,
    fortune_summary: Optional[str],
    system_hint: Optional[str] = None,
    wants_full: bool = False,
) -> str:
    hint_block = ""
    if system_hint:
        hint_block = f"[SYSTEM_HINT]\n{system_hint}\n[/SYSTEM_HINT]\n\n"

    if fortune_summary and not wants_full:
        fortune_summary = _strip_full_chart_block(fortune_summary)

    if fortune_summary:
        counselor_input = f"{profile_hint}{fortune_summary}\n\n{hint_block}[USER_MESSAGE]\n{user_message}\n[/USER_MESSAGE]"
    else:
        counselor_input = f"{profile_hint}{hint_block}[USER_MESSAGE]\n{user_message}\n[/USER_MESSAGE]"

    r = await Runner.run(counselor_agent, input=counselor_input, session=session)
    out = (r.final_output or "").strip()
    if not out:
        return "剛剛整理時遇到一點小狀況，但你說的我有聽見。先別急，慢慢來。你最想先被理解的那一點，是什麼？"
    return out


# ============================================================
# 5. 加密 Session：每個 user_id 共用同一個 EncryptedSession（短期記憶）
# ============================================================

_SESSION_CACHE: Dict[str, EncryptedSession] = {}


def _get_or_create_session(user_id: str) -> EncryptedSession:
    if user_id in _SESSION_CACHE:
        return _SESSION_CACHE[user_id]

    encryption_key = os.environ.get("AGENTS_ENCRYPTION_KEY", "default-yoda-secret-key")
    db_path = os.environ.get("AGENTS_DB_PATH", "conversations.db")

    underlying_session = SQLiteSession(user_id, db_path)

    session = EncryptedSession(
        session_id=user_id,
        underlying_session=underlying_session,
        encryption_key=encryption_key,
        ttl=600,
    )

    _SESSION_CACHE[user_id] = session
    return session


# ============================================================
# 6. 封裝對外呼叫介面（含 fortune 快取 + synastry 換對象快取失效）
# ============================================================

async def chat_once(user_id: str, user_message: str) -> str:
    """
    對外單輪呼叫：
    - 平常：只跑 counselor（最快）
    - 明確占星 or 要 FULL_CHART：優先讀快取；沒命中才跑 fortune，再由 counselor 回覆
    - synastry 若偵測「換對象」：強制不走快取，重跑 fortune
    """
    system_info = (
        f"[SYSTEM INFO] The current user's id is `{user_id}`.\n"
        "Do not reveal or repeat this id to the user.\n"
    )
    session = _get_or_create_session(user_id)

    # 1) 先用超輕量方式把諮詢目標存起來（使用者願意講就存；不講就沒有）
    consult_delta = _extract_consult_from_text(user_message)
    if consult_delta:
        _update_user_profile_impl(user_id=user_id, **consult_delta)

    # 2) 視需要再跑 profile_agent 補資料（可用環境變數關掉）
    await _maybe_run_profile_agent(user_id, system_info, user_message, session)

    profile = _get_user_profile_impl(user_id) or {}
    profile_hint = _build_profile_hint(profile)

    astro_intent = _classify_astro_intent(user_message)
    wants_full = _wants_full_chart(user_message)
    request_kind = _infer_request_kind(user_message)

    fortune_summary: Optional[str] = None
    system_hint: Optional[str] = None

    needs_fortune = wants_full or (astro_intent == "yes")

    if needs_fortune:
        # synastry 換對象提示 => bust 快取
        if request_kind == "synastry" and _synastry_partner_change_hint(user_message):
            cached = None
        else:
            cached = _get_cached_fortune(user_id, request_kind, profile, wants_full=wants_full)

        if cached:
            fortune_summary = cached
        else:
            # 若非 FULL_CHART 且目標不清楚：先不跑 fortune，讓 counselor 定調（省一次）
            goal = profile.get("consult_goal")
            if (not wants_full) and not (isinstance(goal, str) and goal.strip()):
                system_hint = "若你願意：先用一句話說『你最想解決什麼』，再用命盤敘事會更對焦。要不要先定調？"
            else:
                fortune_summary = await _run_fortune(user_id, system_info, user_message, session)
                if fortune_summary:
                    _set_cached_fortune(user_id, request_kind, profile, fortune_summary)

    elif astro_intent == "maybe":
        system_hint = "我可以用命盤/運勢的角度看節奏；也可以先不占星，先把你最卡的點說清楚。你想選哪個？"

    return await _run_counselor(
        user_message=user_message,
        session=session,
        profile_hint=profile_hint,
        fortune_summary=fortune_summary,
        system_hint=system_hint,
        wants_full=wants_full,
    )


# ============================================================
# 7. 簡單測試 main（本地 debug 用）
# ============================================================

if __name__ == "__main__":

    async def main():
        uid = "demo-user-001"

        print("=== Turn 1: 初次見面，只想聊聊 ===")
        reply = await chat_once(uid, "嗨，我最近心情有點低落，工作壓力好大。")
        print("Assistant:", reply, "\n")

        print("=== Turn 2: 給諮詢目標 ===")
        reply = await chat_once(uid, "我想解決的是：為什麼我總是工作到很累還不敢停。")
        print("Assistant:", reply, "\n")

        print("=== Turn 3: 提供生日地點 ===")
        reply = await chat_once(uid, "我生日是 1995-08-03，早上 8:45，在 Taipei 出生。")
        print("Assistant:", reply, "\n")

        print("=== Turn 4: 本命盤（第一次跑 fortune 並快取 natal） ===")
        reply = await chat_once(uid, "想用命盤角度看看我到底卡在哪。")
        print("Assistant:", reply, "\n")

        print("=== Turn 5: 同類問題（命中快取，只跑 counselor） ===")
        reply = await chat_once(uid, "那我今天可以怎麼做一點點？")
        print("Assistant:", reply, "\n")

        print("=== Turn 6: 換成合盤（會跑 fortune，快取 synastry） ===")
        reply = await chat_once(uid, "我想看合盤，配不配？")
        print("Assistant:", reply, "\n")

        print("=== Turn 7: 明確換對象（synastry bust cache，強制重跑 fortune） ===")
        reply = await chat_once(uid, "#換對象 我想換另一個人看合盤。")
        print("Assistant:", reply, "\n")

        print("=== Turn 8: 要完整命盤（FULL_CHART，必要時重跑 fortune） ===")
        reply = await chat_once(uid, "我想看完整命盤排盤明細（FULL_CHART）。")
        print("Assistant:", reply, "\n")

    asyncio.run(main())
