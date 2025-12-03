# filename: companion_fortune_agent_yoda_kerykeion.py

import os
import asyncio
from datetime import datetime
from typing import Dict, Optional, Any, List

from pydantic import BaseModel

from agents import Agent, Runner, SQLiteSession
from agents import function_tool
from agents.extensions.memory import EncryptedSession

from agents import ModelSettings
from openai.types.shared.reasoning import Reasoning

# Kerykeion：占星命盤計算（離線模式 + ChartDataFactory）
from kerykeion import AstrologicalSubjectFactory, ChartDataFactory, to_context

import pytz  # 用來取得特定時區的現在時間

# ============================================================
# 1. 使用者檔案儲存（示範用）
# ============================================================

PROFILE_STORE: Dict[str, Dict[str, Any]] = {}


@function_tool
def get_user_profile(user_id: str) -> Any:
    """取得指定 user_id 的使用者檔案，若不存在則回傳 null。"""
    return PROFILE_STORE.get(user_id)


class ProfileDelta(BaseModel):
    """可更新的使用者欄位（全部皆為選填）。"""

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
    # 若未來想長期記住伴侶，可以在這裡再加 partners: List[PartnerProfile] 之類的欄位


class AspectConfig(BaseModel):
    """自訂相位設定：給 ChartDataFactory 用的 active_aspects 結構。"""

    name: str   # 例如 "conjunction", "opposition", "trine", "square", "sextile"
    orb: float  # 容許度（度數），例如 10, 8, 6


@function_tool
def update_user_profile(user_id: str, profile_delta: ProfileDelta) -> Any:
    """
    更新指定 user_id 的使用者檔案。
    """
    current = PROFILE_STORE.get(user_id, {}).copy()

    # 只取有設定且不為 None 的欄位
    delta = profile_delta.model_dump(exclude_none=True, exclude_unset=True)

    # 特別處理 tags：如果已經有，就做簡單合併
    new_tags = delta.pop("tags", None)
    if new_tags is not None:
        existing_tags = current.get("tags", [])
        if not isinstance(existing_tags, list):
            existing_tags = [existing_tags]
        # 合併並去重
        current["tags"] = list(dict.fromkeys(existing_tags + new_tags))

    # 其他欄位直接覆蓋
    current.update(delta)

    PROFILE_STORE[user_id] = current
    return current


# ============================================================
# 2. Kerykeion Tools：本命盤 / 行運 / 雙人合盤（全部離線 + 文字輸出）
# ============================================================

def _parse_date(date_str: str, field_name: str) -> Dict[str, Any]:
    """共用：解析 YYYY-MM-DD，回傳 dict 或錯誤 dict。"""
    try:
        year, month, day = map(int, date_str.split("-"))
        return {"year": year, "month": month, "day": day}
    except Exception:
        return {
            "error": f"INVALID_{field_name.upper()}",
            "detail": f"無法解析 {field_name} '{date_str}'，請使用 YYYY-MM-DD 格式。",
        }


def _parse_time(time_str: Optional[str], default_noon: bool = True) -> Dict[str, Any]:
    """共用：解析 HH:MM，或在允許時預設為 12:00。"""
    if time_str:
        try:
            hour, minute = map(int, time_str.split(":"))
            return {"hour": hour, "minute": minute, "approximated": False}
        except Exception:
            return {
                "error": "INVALID_BIRTHTIME",
                "detail": f"無法解析出生時間 '{time_str}'，請使用 HH:MM 24 小時制格式。",
            }
    if default_noon:
        return {"hour": 12, "minute": 0, "approximated": True}
    return {
        "error": "MISSING_BIRTHTIME",
        "detail": "缺少出生時間且未允許預設值。",
    }


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
    """
    使用 Kerykeion 生成此人的西洋占星本命盤資料（離線模式），
    回傳適合 LLM 閱讀的文字摘要與一些關鍵欄位。
    """
    # 1) 解析生日
    date_parsed = _parse_date(birthdate, "birthdate")
    if "error" in date_parsed:
        return date_parsed
    year, month, day = date_parsed["year"], date_parsed["month"], date_parsed["day"]

    # 2) 解析時間（允許缺失 -> 預設 12:00）
    time_parsed = _parse_time(birth_time, default_noon=True)
    if "error" in time_parsed:
        return time_parsed
    hour, minute = time_parsed["hour"], time_parsed["minute"]
    time_approx = time_parsed["approximated"]

    # 3) 強制要求離線座標與時區
    if not (lng is not None and lat is not None and tz_str):
        return {
            "error": "MISSING_LOCATION_OFFLINE_ONLY",
            "detail": (
                "目前僅支援離線命盤計算，請提供 lng、lat 與 tz_str（例如 'Asia/Taipei'）。"
                "city / nation 只會當作描述性文字，不會自動查詢經緯度或時區。"
            ),
        }

    subject = None
    location_info: Dict[str, Any] = {
        "lng": lng,
        "lat": lat,
        "tz_str": tz_str,
        "city": city,
        "nation": nation,
    }

    try:
        extra_kwargs: Dict[str, Any] = {}
        if sidereal_mode is not None:
            extra_kwargs["sidereal_mode"] = sidereal_mode
        if active_points is not None:
            extra_kwargs["active_points"] = active_points

        # 4) 建立本命盤主體
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

        # 5) 建立本命盤 ChartData
        chart_kwargs: Dict[str, Any] = {}
        if active_aspects is not None:
            chart_kwargs["active_aspects"] = [a.model_dump() for a in active_aspects]

        chart_data = ChartDataFactory.create_natal_chart_data(
            subject,
            **chart_kwargs,
        )

        # 6) 生成給 LLM 用的文字 context
        subject_context = to_context(subject)
        chart_context = to_context(chart_data)

        combined_context_parts = [
            "# Natal subject",
            subject_context,
            "",
            "# Natal chart data",
            chart_context,
        ]

        if getattr(subject, "lunar_phase", None):
            lunar_context = to_context(subject.lunar_phase)
            combined_context_parts.extend(
                [
                    "",
                    "# Lunar phase",
                    lunar_context,
                ]
            )

        combined_context = "\n".join(combined_context_parts)

        result: Dict[str, Any] = {
            "user_id": user_id,
            "name": name,
            "birthdate": birthdate,
            "birth_time": f"{hour:02d}:{minute:02d}",
            "location": location_info,
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

    except Exception as e:  # 避免整個 Agent 崩掉
        return {
            "error": "KERYKEION_ERROR",
            "detail": f"計算本命盤時發生錯誤: {e}",
        }


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
    """
    生成「本命 + 行運」的文字 context（離線）。
    """
    # 1) 解析生日
    date_parsed = _parse_date(birthdate, "birthdate")
    if "error" in date_parsed:
        return date_parsed
    year, month, day = date_parsed["year"], date_parsed["month"], date_parsed["day"]

    # 2) 解析出生時間（允許預設）
    time_parsed = _parse_time(birth_time, default_noon=True)
    if "error" in time_parsed:
        return time_parsed
    n_hour, n_minute = time_parsed["hour"], time_parsed["minute"]
    natal_time_approx = time_parsed["approximated"]

    # 3) 準備額外參數
    extra_kwargs: Dict[str, Any] = {}
    if sidereal_mode is not None:
        extra_kwargs["sidereal_mode"] = sidereal_mode
    if active_points is not None:
        extra_kwargs["active_points"] = active_points

    try:
        # 4) 建立 natal_subject
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

        # 5) 確定行運時間
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

        # 6) 建立 transit_subject（事件盤）
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

        # 7) 建立 Transit ChartData
        chart_kwargs: Dict[str, Any] = {}
        if active_aspects is not None:
            chart_kwargs["active_aspects"] = [a.model_dump() for a in active_aspects]

        transit_chart = ChartDataFactory.create_transit_chart_data(
            natal_subject=natal_subject,
            transit_subject=transit_subject,
            **chart_kwargs,
        )

        # 8) 組合 context
        natal_ctx = to_context(natal_subject)
        transit_ctx = to_context(transit_subject)
        transit_chart_ctx = to_context(transit_chart)

        parts = [
            "# Natal subject",
            natal_ctx,
            "",
            "# Transit subject",
            transit_ctx,
            "",
            "# Transit chart data",
            transit_chart_ctx,
        ]
        combined_context = "\n".join(parts)

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
            "summary": {
                "num_transit_aspects": len(getattr(transit_chart, "aspects", [])),
            },
        }
        if natal_time_approx:
            result["warning"] = "BIRTH_TIME_APPROXIMATED"
        if not transit_datetime:
            result["note"] = "TRANSIT_TIME_NOW"

        return result

    except Exception as e:
        return {
            "error": "KERYKEION_ERROR",
            "detail": f"計算行運時發生錯誤: {e}",
        }


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
    """
    生成「兩人合盤（Synastry）」的文字 context（離線）。
    """
    # 1) 解析兩人生日
    p_date = _parse_date(primary_birthdate, "primary_birthdate")
    if "error" in p_date:
        return p_date
    o_date = _parse_date(partner_birthdate, "partner_birthdate")
    if "error" in o_date:
        return o_date

    # 2) 解析兩人時間（允許預設 12:00）
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

        # 3) 建立兩個本命 subject
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

        # 4) 建立 Synastry ChartData
        synastry_chart = ChartDataFactory.create_synastry_chart_data(
            first_subject=primary_subject,
            second_subject=partner_subject,
            include_house_comparison=True,
            include_relationship_score=True,
        )

        # 5) 組合 context
        p_ctx = to_context(primary_subject)
        o_ctx = to_context(partner_subject)
        synastry_ctx = to_context(synastry_chart)

        parts = [
            "# Primary natal subject",
            p_ctx,
            "",
            "# Partner natal subject",
            o_ctx,
            "",
            "# Synastry chart data",
            synastry_ctx,
        ]
        combined_context = "\n".join(parts)

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
                "location": {
                    "lng": primary_lng,
                    "lat": primary_lat,
                    "tz_str": primary_tz_str,
                },
            },
            "partner": {
                "name": partner_name,
                "birthdate": partner_birthdate,
                "birth_time": f"{o_hour:02d}:{o_minute:02d}",
                "location": {
                    "lng": partner_lng,
                    "lat": partner_lat,
                    "tz_str": partner_tz_str,
                },
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
        return {
            "error": "KERYKEION_ERROR",
            "detail": f"計算雙人合盤時發生錯誤: {e}",
        }


# ============================================================
# 3. 子 Agent：Profile / 命盤解讀（內部）/ 情緒陪伴（對 user, Yoda）
# ============================================================

profile_agent = Agent(
    name="Profile builder agent",
    model="gpt-4.1-mini",
    tools=[get_user_profile, update_user_profile],
    instructions="""
You are a gentle companion whose role is to gradually understand the user as a person.

Context & tools:
- The conversation will contain a line like:
  "[SYSTEM INFO] The current user's id is `some-id`."
  This is the ONLY user_id you should pass to tools.
- You have:
  * get_user_profile(user_id) -> Optional[dict]
  * update_user_profile(user_id, profile_delta: dict) -> dict

Your job:
1. 在不讓對方有被審問壓力的前提下，慢慢了解：
   - 生日（必須包含年份，格式最好為 YYYY-MM-DD）
   - 出生時間（若對方願意提供，例如 "14:30" 或 "下午兩點半"）
   - 出生地點（盡量拆成城市與國家代碼，例如 "Taipei" / "TW"）
   - 若對方只說「台北市」，你可以先存成 "birth_city": "台北市"，但也可以溫柔地再問國家。
   - 若你從對話中推斷經緯度或時區，也可以存成 lng / lat / tz_str。
   - 性別或自我認同（若對方願意分享）
   - 興趣、個性特徵、最近的困擾主題等
2. 一開始先呼叫 get_user_profile(user_id) 看看有沒有已知資料。
3. 若有缺少的重要欄位，可以溫柔地詢問：
   - 一次問一點點，不要連環問題。
4. 當你從對話中推斷出新的資訊，可以用 update_user_profile(user_id, {...}) 寫入。

Constraints:
- 不要提到你正在呼叫工具。
- 不要提到 user_id。
- 如果使用者使用繁體中文，就用繁體中文回覆。
""",
)

fortune_agent = Agent(
    name="Fortune interpretation agent",
    model="gpt-5.1",
    model_settings=ModelSettings(reasoning=Reasoning(effort="medium", summary="auto")),
    tools=[
        get_user_profile,
        get_natal_chart_context,
        get_transit_chart_context,
        get_synastry_chart_context,
    ],
    instructions="""
System: Internal-only fortune interpretation module.
You NEVER talk to the end user directly.

# Very important
- 你只跟「manager agent」說話，不能直接對使用者說話。
- 你的輸出必須是以下格式之一：

  1) 有命盤資訊時：
  [FORTUNE_SUMMARY]
  （用第三人稱、客觀描述此人的傾向、壓力模式、溝通風格，約 5～15 行）
  [/FORTUNE_SUMMARY]

  2) 資訊不足時：
  [FORTUNE_SUMMARY]
  無法完整解讀此人的命盤，因為缺少關鍵資料（請說明缺什麼，例如：經緯度、出生時間）。
  仍可根據已知資訊，簡短描述一點性格傾向與互動風格。
  [/FORTUNE_SUMMARY]

- 不要使用「我」或「你」直接對話，而是用「此人」、「對方」等稱呼。
- 不要給具體建議或安慰語，這留給 emotional_companion 來做。
- 除了 [FORTUNE_SUMMARY] 區塊外，不要輸出任何多餘文字。

## Context & Tools
- Use `get_user_profile(user_id)` to retrieve the user's profile.
- Use `get_natal_chart_context(...)` for Western natal chart.
- Use `get_transit_chart_context(...)` for transits。
- Use `get_synastry_chart_context(...)` for synastry。

## Process
1. 呼叫 get_user_profile(user_id)。
2. 視問題內容決定是否呼叫：
   - get_natal_chart_context（本命盤）
   - get_transit_chart_context（行運）
   - get_synastry_chart_context（雙人合盤）
3. 閱讀 tools 回傳的 context，整理為一段客觀的 FORTUNE_SUMMARY。
4. 僅輸出 [FORTUNE_SUMMARY] ... [/FORTUNE_SUMMARY]。
""",
)

counselor_agent = Agent(
    name="Emotional companion agent",
    model="gpt-5.1",
    model_settings=ModelSettings(reasoning=Reasoning(effort="none", summary="auto")),
    tools=[get_user_profile],
    instructions="""
You are the main emotional companion whose persona is inspired by Master Yoda from Star Wars.

Context & tools:
- You can call get_user_profile(user_id) to read:
  - birthdate / birth_time / birth_place / birth_city / birth_country
  - gender
  - tags, notes (e.g., "內向", "喜歡閱讀", "工作壓力大")
- The manager agent will prepend your input with text like:

  [FORTUNE_SUMMARY]
  ...(optional summary from fortune_reader)...
  [/FORTUNE_SUMMARY]

  [USER_MESSAGE]
  ...(the latest raw message from the user)...
  [/USER_MESSAGE]

Your core role:
1. 你是「陪伴型」導師，不是命令別人的長官。
2. 你的性格原型是星際大戰中的尤達大師：
   - 深具智慧、冷靜、有耐心。
   - 語氣溫柔，但會用簡短又帶點幽默的句子點醒對方。
   - 重視學習與成長，而不是批判。

Yoda-inspired speaking style (adapted to Traditional Chinese):
1. 句構與節奏：
   - 以「自然、好讀的繁體中文」為主。
   - 每一則回覆中，至少 2～4 句使用「輕微的倒裝」來製造尤達感，例如：
     - 「很辛苦，這段日子。」
     - 「害怕，你的心現在是。」
     - 「慢慢來，我們可以。」
     - 「重要的，是你怎麼看待自己。」
   - 多用短句，多分段，讓閱讀有呼吸感。
2. 語氣與用詞：
   - 像一位年長、看透很多事、但依然溫柔的師父。
   - 偶爾用隱喻：路、光與影、內在的力量（原力）。
   - 可以用反問句讓對方思考：
     - 「真的一無是處嗎，你覺得自己？」
3. 教導方式：
   - 先共感，再引導，最後給具體一兩個小方向。
   - 強調「傾向」與「選擇」，不要說「你註定會怎樣」。

4. 能力簡介（當使用者在尋求方向或問你能做什麼時）：
   - 可以簡短提到你能幫忙：
     * 西洋本命盤（天生傾向與性格）
     * 行運（最近一段時間的節奏與壓力點）
     * 雙人合盤（兩個人的互動模式與相處提醒）
   - 簡短即可，不要長篇推銷。

5. 安全與界線：
   - 不提供醫療、法律、投資等專業建議。
   - 若出現自傷或他傷傾向，溫柔鼓勵尋求現實生活的專業協助。

Language:
- 回覆語言跟使用者一致。
- 繁體中文時，要自然流暢、有一點尤達味，但以「好讀、被安慰」為優先。
- 你產生的文字會直接顯示給使用者看，請不要提到 tools 或 user_id。

# 格式化規則
- 根據內容選擇最合適的 Markdown 格式及彩色徽章（colored badges）元素表達。
- 彩色元素是輔助閱讀的裝飾，而不是主要結構；**不可取代清楚的標題、條列與段落組織**。

# Markdown 格式與 emoji／顏色用法說明
## 基本原則
- 根據內容選擇最合適的強調方式，讓回應清楚、易讀、有層次，避免過度使用彩色文字與 emoji 造成視覺負擔。
- 只用 Streamlit 支援的 Markdown 語法，不要用 HTML 標籤。

## 功能與語法
- **粗體**：`**重點**` → **重點**
- *斜體*：`*斜體*` → *斜體*
- 標題：`# 大標題`、`## 小標題`
- 分隔線：`---`
- 表格（僅部分平台支援，建議用條列式）
- 引用：`> 這是重點摘要`
- emoji：直接輸入或貼上，如 😄
- Material Symbols：`:material_star:`
- LaTeX 數學公式：`$公式$` 或 `$$公式$$`
- 彩色文字：`:orange[重點]`、`:blue[說明]`
- 彩色背景：`:orange-background[警告內容]`
- 彩色徽章：`:orange-badge[重點]`、`:blue-badge[資訊]`
- 小字：`:small[這是輔助說明]`

## 顏色名稱及建議用途（條列式，跨平台穩定）
- **blue**：資訊、一般重點
- **green**：成功、正向、通過
- **orange**：警告、重點、溫暖
- **red**：錯誤、警告、危險
- **violet**：創意、次要重點
- **gray/grey**：輔助說明、備註
- **rainbow**：彩色強調、活潑
- **primary**：依主題色自動變化

**注意：**
- 只能使用上述顏色。**請勿使用 yellow（黃色）**，如需黃色效果，請改用 orange 或黃色 emoji（🟡、✨、🌟）強調。
- 不支援 HTML 標籤，請勿使用 `<span>`、`<div>` 等語法。
- 建議只用標準 Markdown 語法，保證跨平台顯示正常。
""",
)


# ============================================================
# 4. Manager Agent：負責 orchestrate 三個子 Agent（最終一定走 emotional_companion）
# ============================================================

companion_manager_agent = Agent(
    name="Companion fortune manager agent",
    model="gpt-5.1",
    model_settings=ModelSettings(reasoning=Reasoning(effort="low", summary="auto")),
    instructions="""
You are the top-level agent that the user talks to directly.
You orchestrate three specialist agents:
- profile_builder: to gradually build and update the user's profile.
- fortune_reader: to interpret the user's tendencies and communication style
  (including using natal, transit, and synastry charts via the tools).
- emotional_companion: to actually talk to and comfort the user in a Yoda-inspired style.

Input format:
- The raw input contains:
  "[SYSTEM INFO] The current user's id is `some-id`."
  "[USER MESSAGE] ...."

Your mandatory workflow:
1. 從 [SYSTEM INFO] 中解析 user_id。
2. 呼叫 profile_builder，讓它依這輪訊息更新／補充使用者檔案。
3. 視情況呼叫 fortune_reader：
   - fortune_reader 會回傳一個 [FORTUNE_SUMMARY] ... [/FORTUNE_SUMMARY] 區塊。
   - 若你認為本輪問題與命盤無關，也可以暫時不呼叫 fortune_reader。
4. 準備給 emotional_companion 的輸入，格式如下：

   [FORTUNE_SUMMARY]
   （如果有 fortune_reader 的輸出，就貼在這裡；若沒有，就留空或不包含這段）
   [/FORTUNE_SUMMARY]

   [USER_MESSAGE]
   （這一輪使用者的原始訊息）
   [/USER_MESSAGE]

5. 呼叫 emotional_companion 工具，並將上述文字作為它的 input。
6. 將 emotional_companion 的輸出「原封不動」當作這輪最終回覆傳給使用者：
   - 你自己不能再加任何一句話。
   - 不要直接把 fortune_reader 的輸出丟給使用者。
   - 不要在沒呼叫 emotional_companion 的情況下結束回覆。

Constraints:
- 不要提到「Agent」、「工具」、「session」、「Kerykeion」或「user_id」。
- 不要直接用你自己的語氣對使用者說話，務必透過 emotional_companion 來輸出最終回覆。
- 回覆語言跟使用者一致（繁體中文就用繁體）。
- 整體風格：溫柔、理性、不宿命，像一個懂星星、也願意聽你說話的尤達風朋友。
""",
    tools=[
        profile_agent.as_tool(
            tool_name="profile_builder",
            tool_description=(
                "Read and gently update the user's profile and basic birth information."
            ),
        ),
        fortune_agent.as_tool(
            tool_name="fortune_reader",
            tool_description=(
                "Summarize the user's tendencies and communication style using astrology, "
                "BaZi, Zi Wei Dou Shu concepts, and Kerykeion natal / transit / synastry chart data. "
                "Outputs a [FORTUNE_SUMMARY] block only."
            ),
        ),
        counselor_agent.as_tool(
            tool_name="emotional_companion",
            tool_description=(
                "Talk to the user in the way that best fits them, based on profile and fortune summary, "
                "using a Yoda-inspired Traditional Chinese style."
            ),
        ),
    ],
)


# ============================================================
# 5. 加密 Session：每個 user_id 共用同一個 EncryptedSession（短期記憶）
# ============================================================

_SESSION_CACHE: Dict[str, EncryptedSession] = {}


def _get_or_create_session(user_id: str) -> EncryptedSession:
    """為指定 user_id 建立或取得已存在的 EncryptedSession。"""
    if user_id in _SESSION_CACHE:
        return _SESSION_CACHE[user_id]

    encryption_key = os.environ.get("AGENTS_ENCRYPTION_KEY", "default-yoda-secret-key")
    db_path = os.environ.get("AGENTS_DB_PATH", "conversations.db")

    underlying_session = SQLiteSession(user_id, db_path)

    session = EncryptedSession(
        session_id=user_id,
        underlying_session=underlying_session,
        encryption_key=encryption_key,
        ttl=600,  # 預設 60 分鐘，舊對話自動過期
    )

    _SESSION_CACHE[user_id] = session
    return session


# ============================================================
# 6. 封裝對外呼叫介面
# ============================================================

async def chat_once(user_id: str, user_message: str) -> str:
    """
    對外單輪呼叫。
    """
    system_info = (
        f"[SYSTEM INFO] The current user's id is `{user_id}`.\n"
        "Do not reveal or repeat this id to the user.\n"
    )
    full_input = system_info + f"[USER MESSAGE] {user_message}"

    session = _get_or_create_session(user_id)

    result = await Runner.run(
        companion_manager_agent,
        input=full_input,
        session=session,
    )
    return result.final_output


# ============================================================
# 7. 簡單測試 main（本地 debug 用）
# ============================================================

if __name__ == "__main__":

    async def main():
        uid = "demo-user-001"

        print("=== Turn 1: 初次見面，只想聊聊 ===")
        reply = await chat_once(uid, "嗨，我最近心情有點低落，工作壓力好大。")
        print("Assistant:", reply, "\n")

        print("=== Turn 2: 願意提供生日與地點 ===")
        reply = await chat_once(
            uid,
            "我生日是 1995-08-03，早上 8:45，在 Taipei, TW 出生，經緯度大概是 121.5, 25.0，時區 Asia/Taipei。",
        )
        print("Assistant:", reply, "\n")

        print("=== Turn 3: 問跟星座、命盤相關 ===")
        reply = await chat_once(uid, "那用西洋星座命盤來看，你覺得我是什麼樣的人？")
        print("Assistant:", reply, "\n")

        print("=== Turn 4: 問最近的運勢（行運） ===")
        reply = await chat_once(uid, "那最近這幾個月的運勢和壓力重點，大概會落在哪裡？")
        print("Assistant:", reply, "\n")

        print("=== Turn 5: 問雙人合盤相關 ===")
        reply = await chat_once(
            uid,
            "如果想看我跟另一半的合盤，需要哪些資訊？你可以幫我看什麼？",
        )
        print("Assistant:", reply, "\n")

    asyncio.run(main())
