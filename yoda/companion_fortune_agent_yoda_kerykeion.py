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

from kerykeion import AstrologicalSubjectFactory, ChartDataFactory, to_context
import pytz


# ============================================================
# 0. 產品決策：地點/時區一律固定台北（避免追問）
# ============================================================

DEFAULT_CITY = "台北市"
DEFAULT_COUNTRY = "TW"
DEFAULT_TZ = "Asia/Taipei"
DEFAULT_LNG = 121.5654
DEFAULT_LAT = 25.0330


# ============================================================
# 1. 使用者檔案儲存（示範用：記憶體版）
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

    consult_goal: Optional[str] = None
    consult_focus: Optional[str] = None  # self/relationship/career/timing/block/other


class AspectConfig(BaseModel):
    name: str
    orb: float


def _get_user_profile_impl(user_id: str) -> Any:
    return PROFILE_STORE.get(user_id)


def _ensure_default_taipei_fields(profile: Dict[str, Any]) -> Dict[str, Any]:
    if not profile.get("birth_city"):
        profile["birth_city"] = DEFAULT_CITY
    if not profile.get("birth_country"):
        profile["birth_country"] = DEFAULT_COUNTRY
    if not profile.get("tz_str"):
        profile["tz_str"] = DEFAULT_TZ
    if profile.get("lng") is None:
        profile["lng"] = DEFAULT_LNG
    if profile.get("lat") is None:
        profile["lat"] = DEFAULT_LAT
    return profile


def _update_user_profile_impl(
    user_id: str,
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

    new_tags = delta.pop("tags", None)
    if new_tags is not None:
        existing_tags = current.get("tags", [])
        if not isinstance(existing_tags, list):
            existing_tags = [existing_tags]
        current["tags"] = list(dict.fromkeys(existing_tags + new_tags))

    current.update(delta)
    current = _ensure_default_taipei_fields(current)

    PROFILE_STORE[user_id] = current
    return current


get_user_profile = function_tool(_get_user_profile_impl)
update_user_profile = function_tool(_update_user_profile_impl)


# ============================================================
# 2. Kerykeion Tools：本命盤 / 行運 / 合盤（離線 + 文字輸出）
# ============================================================

def _parse_date(date_str: str, field_name: str) -> Dict[str, Any]:
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
    date_parsed = _parse_date(birthdate, "birthdate")
    if "error" in date_parsed:
        return date_parsed
    year, month, day = date_parsed["year"], date_parsed["month"], date_parsed["day"]

    time_parsed = _parse_time(birth_time, default_noon=True)
    if "error" in time_parsed:
        return time_parsed
    hour, minute = time_parsed["hour"], time_parsed["minute"]
    time_approx = time_parsed["approximated"]

    lng = DEFAULT_LNG if lng is None else lng
    lat = DEFAULT_LAT if lat is None else lat
    tz_str = DEFAULT_TZ if not tz_str else tz_str
    city = DEFAULT_CITY if not city else city
    nation = DEFAULT_COUNTRY if not nation else nation

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
                return {"error": "INVALID_TRANSIT_DATETIME", "detail": f"無法解析 transit_datetime '{transit_datetime}'"}
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

        return {
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

    except Exception as e:
        return {"error": "KERYKEION_ERROR", "detail": f"計算雙人合盤時發生錯誤: {e}"}


# ============================================================
# 3. Agents
# ============================================================

profile_agent = Agent(
    name="Profile builder agent",
    model="gpt-4.1-mini",
    tools=[get_user_profile, update_user_profile],
    instructions=r"""
你是溫柔的資料整理者。
注意：產品決策已固定使用台北預設，因此不要追問任何地點/時區相關問題。

偏好（可選）：
- 若使用者表達「不想要作業/小步驟」，可把 tags 加上 pref_no_steps。
""",
)

fortune_agent = Agent(
    name="Fortune interpretation agent",
    model="gpt-5.2",
    model_settings=ModelSettings(reasoning=Reasoning(effort="medium", verbosity="low")),
    tools=[get_user_profile, get_natal_chart_context, get_transit_chart_context, get_synastry_chart_context],
    instructions=r"""
System: Internal-only fortune interpretation module.
You NEVER talk to the end user directly.

你要把 Steven Forrest 三書的方法當作「鏡頭」而非「每次固定三段報告」：
- Inner Sky 鏡頭：本命核心劇本（我是誰/我怎麼運作）
- Yesterday’s Sky 鏡頭：南北交點敘事（舊路/熟悉慣性 vs 今生方向）
- Changing Sky 鏡頭：行運作為季節（不做事件預言，改成選擇建議）

=== 禁止事項 ===
- 禁止出現：出生地、時區、DST、日光節約、日光節約時間
- 若要談精準度，只能用「盤面精準度」。
- 禁止英文段標：INNER_SKY / YESTERDAYS_SKY / CHANGING_SKY
- 禁止盤面幻想：只能用工具 context 中真的出現的配置；不存在就不要寫。

=== 何時用哪些鏡頭（重點） ===
0) 永遠先回答問題本身（聚焦回應）。
1) 只在需要時補鏡頭（最多 1–2 個）：
   - 問「我為什麼會一直重複同樣模式/卡關/像命運循環」=> 補 Yesterday’s Sky（交點敘事）
   - 問「最近/接下來」=> 補 Changing Sky（若有 transit）
   - 問「我到底是什麼樣的人/天賦是什麼」=> 補 Inner Sky
2) 只有使用者明確要求「全面整理/完整解讀/完整解析」才用三段展開（全面整理模式）。

=== 昨日的天空（交點敘事）強制規則 ===
只要你有寫「交點敘事」：
- 必須提到：南交點/北交點/月交點 至少其一
- 主線是：南交點=舊路/熟悉慣性（可用前世/業力作象徵敘事，但提醒不是可驗證歷史）
         北交點=今生方向
- 相位只有工具輸出有才可提

=== consult_goal ===
consult_goal 缺失不能造成 NO_CHART：
- 預設 CONSULT_GOAL=全面整理（預設），CONSULT_FOCUS=other

=== NO_CHART 僅允許於 ===
- 缺 birthdate（missing_birth_data）
- synastry 缺對方資料（missing_partner_data）
- kerykeion_error / other

=== 落地（不要每次都像交作業） ===
- 若 profile.tags 有 pref_no_steps：優先 QUESTIONS 或 ANCHOR/BOUNDARY_REMINDER
- 使用者明確要行動清單才用 PRACTICE/ACTIONS

=== 輸出格式（嚴格：只能輸出一個區塊） ===

[FORTUNE_SUMMARY]
STATUS: HAS_CHART 或 NO_CHART
CHART_TYPES: "natal" / "natal+transit" / "natal+synastry"
CONSULT_GOAL: ...
CONSULT_FOCUS: ...

（HAS_CHART：兩種模式擇一）

【聚焦回應模式】
聚焦回應：
（2–10 行：先回答問題本身）
（可選）交點敘事：
（2–8 行：若出現則需交點字樣）
（可選）更成熟的選擇：
（2–8 行：若含 transit 用季節/天氣隱喻，不做事件預言）

【全面整理模式】
你內在的核心劇本：
（4–10 行）
你曾用來活下來的方式：
（4–10 行：交點敘事為主）
你接下來更成熟的選擇：
（4–10 行：行運=季節，不做事件預言）

THEME: ...
SHADOW: ...
GIFT: ...
CHOICE: ...

（落地段：至少一種）
PRACTICE: ...
ACTIONS:
- 1) ...
- 2) ...
或
QUESTIONS:
- 1) ...
- 2) ...
或
ANCHOR: ...
BOUNDARY_REMINDER: ...

（使用者要求完整命盤時才加）
[FULL_CHART]
...（放入 Kerykeion context）
[/FULL_CHART]

[/FORTUNE_SUMMARY]

NO_CHART：
[FORTUNE_SUMMARY]
STATUS: NO_CHART
REASON: missing_birth_data / missing_partner_data / kerykeion_error / other
CONSULT_GOAL: 全面整理（預設）
CONSULT_FOCUS: other
THEME: ...
SHADOW: ...
GIFT: ...
CHOICE: ...
QUESTIONS:
- 1) ...
- 2) ...
[/FORTUNE_SUMMARY]
""",
)

counselor_agent = Agent(
    name="Emotional companion agent",
    model="gpt-5.2",
    model_settings=ModelSettings(reasoning=Reasoning(effort="none"), temperature=0),
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

# 原力（The Force）— 溫柔但有界線版
- 原力是隱喻：覺察、呼吸、界線、價值選擇。
- 每次回覆提到「原力」最多 0～2 次；禁止權威口吻（禁：原力告訴你/你必須）。

# Yoda style
- 至少 2～4 句輕微倒裝
- 先共感，再引導，最後給 1～2 個小方向
- 多用短句分段
- 不做醫療/法律/投資建議；若有自傷他傷傾向，鼓勵現實專業協助

# 硬性禁詞
- 回覆中禁止出現：出生地、時區、DST、日光節約、日光節約時間
- 若要談精準度，只能說「盤面精準度」。

# 回覆策略（避免無趣）
- 以使用者問題為主，不要硬拆三段。
- fortune 是「聚焦回應模式」：就照它的節奏，先回答，再溫柔承接。
- fortune 是「全面整理模式」：才用三段中文敘事回應。

# 落地方式（避免作業感）
- 若 fortune 有 QUESTIONS：挑 1–2 題溫柔問
- 若有 ANCHOR/BOUNDARY_REMINDER：給一句能帶著走的話（不要求做事）
- 只有使用者明確要步驟時，才轉成可做的小步驟
""",
)


# ============================================================
# 4. 輸出格式檢查器 + 自動重試（依本輪是否要求全面整理決定嚴格度）
# ============================================================

_BANNED_STRINGS = [
    "出生地", "時區", "DST", "日光節約", "日光節約時間",
    "INNER_SKY", "YESTERDAYS_SKY", "CHANGING_SKY",
]

_REQUIRED_FIELDS_BASE = ["STATUS:", "CHART_TYPES:", "CONSULT_GOAL:", "CONSULT_FOCUS:"]
_REQUIRED_FIELDS_HAS_CHART_CORE = ["THEME:", "SHADOW:", "GIFT:", "CHOICE:"]
_ALLOWED_GROUNDING_MARKERS = ["PRACTICE:", "ACTIONS:", "QUESTIONS:", "ANCHOR:", "BOUNDARY_REMINDER:"]

_REQUIRED_SECTIONS_TRILOGY = [
    "你內在的核心劇本：",
    "你曾用來活下來的方式：",
    "你接下來更成熟的選擇：",
]


def _normalize_fortune_block(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"\[\s*FORTUNE_SUMMARY\s*\]", "[FORTUNE_SUMMARY]", t)
    t = re.sub(r"\[\s*/\s*FORTUNE_SUMMARY\s*\]", "[/FORTUNE_SUMMARY]", t)
    t = re.sub(r"\[\s*FULL_CHART\s*\]", "[FULL_CHART]", t)
    t = re.sub(r"\[\s*/\s*FULL_CHART\s*\]", "[/FULL_CHART]", t)
    return t.strip()


def _extract_fortune_summary_block(text: str) -> Optional[str]:
    t = _normalize_fortune_block(text)
    m = re.search(r"\[FORTUNE_SUMMARY\][\s\S]*?\[/FORTUNE_SUMMARY\]", t)
    return m.group(0).strip() if m else None


def _is_only_one_fortune_block(text: str) -> bool:
    t = _normalize_fortune_block(text)
    block = _extract_fortune_summary_block(t)
    return bool(block) and (t == block)


def _parse_status(block: str) -> Optional[str]:
    m = re.search(r"STATUS:\s*(HAS_CHART|NO_CHART)\b", block)
    return m.group(1) if m else None


def _has_any_grounding(block: str) -> bool:
    return any(marker in block for marker in _ALLOWED_GROUNDING_MARKERS)


def _validate_fortune_output(raw_text: str, require_trilogy: bool) -> Tuple[bool, List[str], Optional[str]]:
    problems: List[str] = []
    t = _normalize_fortune_block(raw_text)
    block = _extract_fortune_summary_block(t)

    if not block:
        return False, ["缺少 [FORTUNE_SUMMARY]...[/FORTUNE_SUMMARY] 區塊"], None

    if not _is_only_one_fortune_block(t):
        problems.append("輸出包含 fortune 區塊以外的多餘文字（必須只輸出 fortune 區塊）")

    for s in _BANNED_STRINGS:
        if s in block:
            problems.append(f"包含禁詞/禁段標：{s}")

    for key in _REQUIRED_FIELDS_BASE:
        if key not in block:
            problems.append(f"缺少欄位：{key}")

    status = _parse_status(block)
    if status is None:
        problems.append("STATUS 必須是 HAS_CHART 或 NO_CHART")
        return False, problems, block

    if status == "HAS_CHART":
        for key in _REQUIRED_FIELDS_HAS_CHART_CORE:
            if key not in block:
                problems.append(f"HAS_CHART 缺少欄位：{key}")

        if not _has_any_grounding(block):
            problems.append("HAS_CHART 必須包含落地段：PRACTICE/ACTIONS 或 QUESTIONS 或 ANCHOR/BOUNDARY_REMINDER 其一")

        if require_trilogy:
            for sec in _REQUIRED_SECTIONS_TRILOGY:
                if sec not in block:
                    problems.append(f"全面整理模式缺少段落標題：{sec}")
        else:
            # 聚焦模式：至少要有「聚焦回應：」
            if "聚焦回應：" not in block:
                problems.append("聚焦回應模式缺少『聚焦回應：』段落（請先回答問題本身）")

        # 若出現交點敘事，至少要提到交點字樣
        if "交點敘事：" in block:
            if ("南交點" not in block) and ("北交點" not in block) and ("月交點" not in block):
                problems.append("交點敘事需至少提到南交點/北交點/月交點其中之一")

    else:
        if "REASON:" not in block:
            problems.append("NO_CHART 缺少欄位：REASON:")
        for k in ["THEME:", "SHADOW:", "GIFT:", "CHOICE:"]:
            if k not in block:
                problems.append(f"NO_CHART 缺少欄位：{k}")
        if ("QUESTIONS:" not in block) and ("ANCHOR:" not in block):
            problems.append("NO_CHART 需包含 QUESTIONS 或 ANCHOR 其一（避免空泛）")

    return (len(problems) == 0), problems, block


async def _run_fortune_checked(
    user_id: str,
    system_info: str,
    user_message: str,
    session: EncryptedSession,
    require_trilogy: bool,
    max_attempts: int = 2,
) -> Optional[str]:
    last_block: Optional[str] = None
    last_problems: List[str] = []

    for attempt in range(1, max_attempts + 1):
        format_hint = ""
        if attempt > 1 and last_problems:
            format_hint = (
                "[FORMAT_HINT]\n"
                "上一次輸出未通過格式檢查，這次務必完全修正。\n"
                "問題如下（逐一修正）：\n"
                + "\n".join([f"- {p}" for p in last_problems])
                + "\n要求：只能輸出一個 [FORTUNE_SUMMARY] 區塊；不得出現禁詞/英文段標。\n"
                + ("（本次為全面整理：三段中文段標必須齊全）\n" if require_trilogy else "（本次為聚焦回應：必須先回答問題本身）\n")
                + "[/FORMAT_HINT]\n"
            )

        full_input = system_info + format_hint + f"[USER MESSAGE] {user_message}"
        r = await Runner.run(fortune_agent, input=full_input, session=session)
        raw = (r.final_output or "").strip()

        ok, problems, block = _validate_fortune_output(raw, require_trilogy=require_trilogy)
        last_problems = problems
        last_block = block

        if ok and block:
            return block

    return last_block


# ============================================================
# 5. Orchestrator：快取 + synastry bust + 偏好
# ============================================================

AstroIntent = Literal["yes", "maybe", "no"]
RequestKind = Literal["natal", "transit", "synastry", "unknown"]

_ASTRO_KEYWORDS_YES = [
    "星座", "命盤", "占星", "本命盤", "解讀", "看盤", "排盤", "排盤解析",
    "行運", "運勢", "流年", "推運", "次限", "太陽弧",
    "合盤", "關係盤", "配不配", "我們兩個",
    "上升", "月亮", "太陽星座", "宮位", "相位",
    "全面整理", "完整解讀", "完整解析", "全盤",
]

_FULL_CHART_KEYWORDS = ["完整命盤", "排盤明細", "完整盤", "命盤明細", "原始輸出", "FULL_CHART"]

_NO_STEPS_CUES = [
    "不要小步驟", "不要作業", "別叫我做", "不要練習", "不要任務", "不要行動清單",
    "不要每天", "不要每次", "不想做小步驟", "不想被指派",
]

_WANTS_ACTION_CUES = ["怎麼做", "具體要做什麼", "給我行動", "行動方案", "步驟", "清單", "實作", "可執行"]


def _get_fortune_cache_ttl() -> int:
    return int(os.environ.get("FORTUNE_CACHE_TTL", "600"))


def _wants_full_chart(msg: str) -> bool:
    s = msg or ""
    return any(k in s for k in _FULL_CHART_KEYWORDS) or ("#fullchart" in s.lower())


def _wants_full_reading(msg: str) -> bool:
    return bool(re.search(r"(全面整理|完整解讀|完整解析|全盤|全面|完整看盤)", msg or ""))


def _classify_astro_intent(user_message: str) -> AstroIntent:
    msg = user_message or ""
    if any(k in msg for k in _ASTRO_KEYWORDS_YES):
        return "yes"
    return "maybe" if re.search(r"(最近|這陣子|未來|卡住|適合|性格|天賦|壓力|關係|職涯)", msg) else "no"


def _infer_request_kind(user_message: str) -> RequestKind:
    s = user_message or ""
    if any(k in s for k in ["合盤", "關係盤", "配不配", "我們兩個"]):
        return "synastry"
    if any(k in s for k in ["行運", "運勢", "流年", "推運", "次限", "太陽弧", "未來幾個月", "最近這幾個月", "未來一年"]):
        return "transit"
    if any(k in s for k in ["命盤", "本命盤", "星座", "上升", "月亮", "太陽星座", "解讀", "看盤", "排盤", "排盤解析", "全面整理", "完整解讀", "完整解析", "全盤"]):
        return "natal"
    return "unknown"


def _synastry_partner_change_hint(user_message: str) -> bool:
    msg = (user_message or "").strip()
    manual_tags = ["#換對象", "#新對象", "#重新合盤", "#newpartner", "/newpartner", "/resynastry"]
    if any(t.lower() in msg.lower() for t in manual_tags):
        return True
    cues = ["換一個", "換個", "換人", "新對象", "不是這個人", "另一個人", "換別人"]
    return any(c in msg for c in cues)


def _extract_birth_date_time(msg: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    m = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", msg or "")
    if m:
        out["birthdate"] = f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    t = re.search(r"\b(\d{1,2}):(\d{2})\b", msg or "")
    if t:
        hh, mi = int(t.group(1)), int(t.group(2))
        if 0 <= hh <= 23 and 0 <= mi <= 59:
            out["birth_time"] = f"{hh:02d}:{mi:02d}"
    return out


def _wants_no_steps(msg: str) -> bool:
    s = msg or ""
    return any(cue in s for cue in _NO_STEPS_CUES)


def _wants_action_mode(msg: str) -> bool:
    s = msg or ""
    return any(cue in s for cue in _WANTS_ACTION_CUES)


def _profile_fingerprint(profile: Dict[str, Any]) -> Tuple:
    tags = profile.get("tags") or []
    if not isinstance(tags, list):
        tags = [tags]
    pref_no_steps = "pref_no_steps" in tags
    return (profile.get("birthdate"), profile.get("birth_time"), pref_no_steps)


def _fortune_contains_full_chart(fortune_summary: str) -> bool:
    return bool(fortune_summary) and ("[FULL_CHART]" in fortune_summary and "[/FULL_CHART]" in fortune_summary)


def _strip_full_chart_block(fortune_summary: str) -> str:
    if not fortune_summary:
        return fortune_summary
    return re.sub(r"\[FULL_CHART\][\s\S]*?\[/FULL_CHART\]\n?", "", fortune_summary).strip()


_FORTUNE_CACHE: Dict[tuple, Dict[str, Any]] = {}
# key = (user_id, request_kind)


def _get_cached_fortune(user_id: str, request_kind: RequestKind, profile: Dict[str, Any], wants_full: bool) -> Optional[str]:
    key = (user_id, request_kind)
    entry = _FORTUNE_CACHE.get(key)
    if not entry:
        return None
    if (time.time() - float(entry.get("created_at", 0))) > _get_fortune_cache_ttl():
        return None
    if entry.get("profile_fp") != _profile_fingerprint(profile):
        return None
    if wants_full and not bool(entry.get("has_full_chart")):
        return None
    return entry.get("fortune_summary")


def _set_cached_fortune(user_id: str, request_kind: RequestKind, profile: Dict[str, Any], fortune_summary: str) -> None:
    key = (user_id, request_kind)
    _FORTUNE_CACHE[key] = {
        "created_at": time.time(),
        "request_kind": request_kind,
        "profile_fp": _profile_fingerprint(profile),
        "fortune_summary": fortune_summary,
        "has_full_chart": _fortune_contains_full_chart(fortune_summary),
    }


def _build_profile_hint(profile: Dict[str, Any]) -> str:
    tags = profile.get("tags") or []
    if not isinstance(tags, list):
        tags = [tags]
    pref_no_steps = "pref_no_steps" in tags
    parts = []
    if pref_no_steps:
        parts.append("pref_no_steps: true")
    return "[PROFILE_HINT]\n" + "\n".join(parts) + "\n[/PROFILE_HINT]\n\n" if parts else ""


async def _run_counselor(user_message: str, session: EncryptedSession, fortune_summary: Optional[str], wants_full: bool, profile_hint: str) -> str:
    if fortune_summary and not wants_full:
        fortune_summary = _strip_full_chart_block(fortune_summary)

    if fortune_summary:
        counselor_input = f"{profile_hint}{fortune_summary}\n\n[USER_MESSAGE]\n{user_message}\n[/USER_MESSAGE]"
    else:
        counselor_input = f"{profile_hint}[USER_MESSAGE]\n{user_message}\n[/USER_MESSAGE]"

    r = await Runner.run(counselor_agent, input=counselor_input, session=session)
    return (r.final_output or "").strip() or "剛剛有點小狀況，但我有聽見你。先別急，慢慢來。"


# ============================================================
# 6. 加密 Session（短期記憶）
# ============================================================

_SESSION_CACHE: Dict[str, EncryptedSession] = {}


def _get_or_create_session(user_id: str) -> EncryptedSession:
    if user_id in _SESSION_CACHE:
        return _SESSION_CACHE[user_id]

    encryption_key = os.environ.get("AGENTS_ENCRYPTION_KEY", "default-yoda-secret-key")
    db_path = os.environ.get("AGENTS_DB_PATH", "conversations.db")

    session = EncryptedSession(
        session_id=user_id,
        underlying_session=SQLiteSession(user_id, db_path),
        encryption_key=encryption_key,
        ttl=600,
    )
    _SESSION_CACHE[user_id] = session
    return session


# ============================================================
# 7. 對外單輪呼叫
# ============================================================

async def chat_once(user_id: str, user_message: str) -> str:
    system_info = (
        f"[SYSTEM INFO] The current user's id is `{user_id}`.\n"
        "Do not reveal or repeat this id to the user.\n"
    )
    session = _get_or_create_session(user_id)

    _update_user_profile_impl(user_id=user_id)

    dt_delta = _extract_birth_date_time(user_message)
    if dt_delta:
        _update_user_profile_impl(user_id=user_id, **dt_delta)

    if _wants_no_steps(user_message):
        profile = _get_user_profile_impl(user_id) or {}
        tags = profile.get("tags") or []
        if not isinstance(tags, list):
            tags = [tags]
        if "pref_no_steps" not in tags:
            _update_user_profile_impl(user_id=user_id, tags=tags + ["pref_no_steps"])

    if _wants_action_mode(user_message):
        profile = _get_user_profile_impl(user_id) or {}
        tags = profile.get("tags") or []
        if not isinstance(tags, list):
            tags = [tags]
        if "pref_no_steps" in tags:
            _update_user_profile_impl(user_id=user_id, tags=[t for t in tags if t != "pref_no_steps"])

    profile = _get_user_profile_impl(user_id) or {}
    profile = _ensure_default_taipei_fields(profile)
    profile_hint = _build_profile_hint(profile)

    astro_intent = _classify_astro_intent(user_message)
    wants_full = _wants_full_chart(user_message)
    request_kind = _infer_request_kind(user_message)
    require_trilogy = _wants_full_reading(user_message)

    needs_fortune = wants_full or (astro_intent == "yes")

    fortune_summary: Optional[str] = None
    if needs_fortune:
        cached: Optional[str] = None
        if not (request_kind == "synastry" and _synastry_partner_change_hint(user_message)):
            cached = _get_cached_fortune(user_id, request_kind, profile, wants_full=wants_full)

        if cached:
            ok, _, _ = _validate_fortune_output(cached, require_trilogy=require_trilogy)
            if ok:
                fortune_summary = cached
            else:
                cached = None

        if not cached:
            fortune_summary = await _run_fortune_checked(
                user_id=user_id,
                system_info=system_info,
                user_message=user_message,
                session=session,
                require_trilogy=require_trilogy,
                max_attempts=int(os.environ.get("FORTUNE_FORMAT_RETRY", "2")),
            )
            if fortune_summary:
                ok, _, _ = _validate_fortune_output(fortune_summary, require_trilogy=require_trilogy)
                if ok:
                    _set_cached_fortune(user_id, request_kind, profile, fortune_summary)

    return await _run_counselor(user_message, session, fortune_summary, wants_full=wants_full, profile_hint=profile_hint)


# ============================================================
# 8. 本地 debug
# ============================================================

if __name__ == "__main__":
    async def main():
        uid = "demo-user-001"
        print(await chat_once(uid, "我的生日是2012/09/03 出生時間在13:30 幫我排盤解析：我為什麼很容易焦慮？"))
        print(await chat_once(uid, "可以幫我全面整理嗎？"))
    asyncio.run(main())
