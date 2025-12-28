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

    # 地點欄位：仍保留欄位，但系統會自動補台北
    birth_city: Optional[str] = None
    birth_country: Optional[str] = None
    lng: Optional[float] = None
    lat: Optional[float] = None
    tz_str: Optional[str] = None

    gender: Optional[str] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None

    # Forrest 定調
    consult_goal: Optional[str] = None
    consult_focus: Optional[str] = None  # self/relationship/career/timing/block/other


class AspectConfig(BaseModel):
    name: str
    orb: float


def _get_user_profile_impl(user_id: str) -> Any:
    return PROFILE_STORE.get(user_id)


def _ensure_default_taipei_fields(profile: Dict[str, Any]) -> Dict[str, Any]:
    """強制補台北預設，避免任何地點/時區缺漏導致流程追問。"""
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
    """
    真正更新 profile 的實作（strict schema 友善，無 Dict[str, Any]）。
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


# tools（strict schema 安全）
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

    # 固定台北（工具層保險）
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

update_user_profile 必須用具名參數呼叫（不能傳 dict）。

你主要要補：
- birthdate（YYYY-MM-DD）
- birth_time（HH:MM）
- consult_goal / consult_focus
""",
)

fortune_agent = Agent(
    name="Fortune interpretation agent",
    model="gpt-5.2",
    model_settings=ModelSettings(reasoning=Reasoning(effort="medium", summary="auto")),
    tools=[get_user_profile, get_natal_chart_context, get_transit_chart_context, get_synastry_chart_context],
    instructions=r"""
System: Internal-only fortune interpretation module.
You NEVER talk to the end user directly.

目的：用 Steven Forrest 三本書的方法論（不引用原文）做「心理占星 + 生命敘事」完整架構：
- The Inner Sky（你是誰：本命核心劇本）
- Yesterday’s Sky（你怎麼走到今天：成長史/適應策略）
- The Changing Sky（你要怎麼走：現在與接下來的選擇/練習）

重要禁詞（因為下游會直接呈現給使用者）：
- 你的輸出中禁止出現：出生地、時區、DST、日光節約、日光節約時間
（若要談精準度，用「盤面精準度」。）

資料策略：
- 地點/時區由系統固定處理；你不追問、也不以「缺地點」當 NO_CHART。
- consult_goal 若缺：不要 NO_CHART；預設採用「全面整理（預設）」作為目標，CONSULT_FOCUS="other"。

NO_CHART 只允許出現在以下情況：
- 缺 birthdate（missing_birth_data）
- synastry 缺對方必要資料（missing_partner_data）
- Kerykeion 計算錯誤（kerykeion_error）
- 其他不可恢復錯誤（other）
即使 NO_CHART，也要用 Forrest 式語言輸出 THEME/SHADOW/GIFT/CHOICE/PRACTICE（不可提盤面細節）。

# Output contract（嚴格遵守：只能輸出 FORTUNE_SUMMARY）
HAS_CHART 時必須包含：
- CONSULT_GOAL / CONSULT_FOCUS
- INNER_SKY / YESTERDAYS_SKY / CHANGING_SKY
- THEME/SHADOW/GIFT/CHOICE/PRACTICE
- ACTIONS（1~3 條具體行動）
- 使用者要求完整命盤時才加 FULL_CHART（放 Kerykeion context）

格式如下：

[FORTUNE_SUMMARY]
STATUS: HAS_CHART
CHART_TYPES: "natal" / "natal+transit" / "natal+synastry"
CONSULT_GOAL: ...
CONSULT_FOCUS: ...

INNER_SKY:
...（4–10 行，涵蓋：上升與守護星、太陽/月亮、元素/模式/半球、行星落宮、主要相位整合；語氣是靈魂意圖，非宿命）
YESTERDAYS_SKY:
...（4–10 行，童年/原生家庭印記、早期適應策略、修復方向；心理語言）
CHANGING_SKY:
...（4–10 行，若有 transit 用季節/天氣隱喻 + 選擇建議；不做事件預言）

THEME: ...
SHADOW: ...
GIFT: ...
CHOICE: ...
PRACTICE: ...
ACTIONS:
- 1) ...
- 2) ...
- 3) ...

[FULL_CHART]
...（僅在使用者要求完整命盤/排盤明細時輸出，放入 Kerykeion context）
[/FULL_CHART]

[/FORTUNE_SUMMARY]

NO_CHART 時：

[FORTUNE_SUMMARY]
STATUS: NO_CHART
REASON: missing_birth_data / missing_partner_data / kerykeion_error / other
CONSULT_GOAL: 全面整理（預設）
CONSULT_FOCUS: other
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

# 硬性禁詞（新增，請嚴格遵守）
- 回覆中禁止出現：出生地、時區、DST、日光節約、日光節約時間
- 若要談精準度，只能說「盤面精準度」。

# Steven Forrest 三書方法論的「轉譯」規則（新增）
- 若 FORTUNE_SUMMARY 內包含 INNER_SKY / YESTERDAYS_SKY / CHANGING_SKY：
  你回覆時也要用同樣三段式來「解釋與陪伴」，順序一致：
  1) INNER_SKY：先用溫柔敘事說清楚「此人核心劇本/渴望/張力」(只改寫摘要，不加新占星細節)
  2) YESTERDAYS_SKY：再用「不是壞掉，是曾經努力活下來」的語氣，說明早期適應策略與可能的修復方向
  3) CHANGING_SKY：最後把「預測」改成「選擇建議」：這段能量要練什麼？更成熟的做法是什麼？
- 最後務必落地：把 ACTIONS 或 PRACTICE 轉成 1–2 個「今天/這週能做」的小步驟（5–20 分鐘級）。

# Markdown格式與emoji/顏色用法說明
## 基本原則
- 請根據內容選擇最合適的強調方式，讓回應清楚、易讀、有層次，避免過度花俏。  
- 只用 Streamlit 支援的 Markdown 語法，不要用 HTML 標籤。  

## 功能與語法
- **粗體**：`**重點**` → **重點**  
- *斜體*：`*斜體*` → *斜體*  
- 標題：`# 大標題`、`## 小標題`  
- 分隔線：`---`  
- 表格（僅部分平台支援，建議用條列式）  
- 引用：`> 這是重點摘要`  
- emoji：直接輸入或貼上，如 😄  
- Material Symbols：`:material/star:`  
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
- 僅能使用上述顏色。**請勿使用 yellow（黃色）**，如需黃色效果，請改用 orange 或黃色 emoji（🟡、✨、🌟）強調。  
- 不支援 HTML 標籤，請勿使用 `<span>`、`<div>` 等語法。  
- 建議只用標準 Markdown 語法，保證跨平台顯示正常。
""",
)


# ============================================================
# 4. Orchestrator：快取 + synastry 換對象 bust
# ============================================================

AstroIntent = Literal["yes", "maybe", "no"]
RequestKind = Literal["natal", "transit", "synastry", "unknown"]

_ASTRO_KEYWORDS_YES = [
    "星座", "命盤", "占星", "本命盤", "全面整理", "完整解讀", "解讀", "看盤", "排盤",
    "行運", "運勢", "流年", "推運", "次限", "太陽弧",
    "合盤", "關係盤", "配不配", "我們兩個",
    "上升", "月亮", "太陽星座", "宮位", "相位",
]

_FULL_CHART_KEYWORDS = ["完整命盤", "排盤明細", "完整盤", "命盤明細", "原始輸出", "FULL_CHART"]


def _now_ts() -> float:
    return time.time()


def _get_fortune_cache_ttl() -> int:
    return int(os.environ.get("FORTUNE_CACHE_TTL", "600"))


def _wants_full_chart(msg: str) -> bool:
    s = msg or ""
    return any(k in s for k in _FULL_CHART_KEYWORDS) or ("#fullchart" in s.lower())


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
    if any(k in s for k in ["命盤", "本命盤", "星座", "上升", "月亮", "太陽星座", "全面整理", "完整解讀", "解讀", "看盤", "排盤"]):
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


def _extract_consult_goal_focus(msg: str) -> Dict[str, Any]:
    s = (msg or "").strip()
    out: Dict[str, Any] = {}

    if re.search(r"\bE\b\s*[\.\-、]?\s*全面整理", s):
        out["consult_goal"] = "全面整理（使用者指定）"
        out["consult_focus"] = "other"
        return out

    m = re.search(r"(我想|想要|想解決|我在意|我困擾|我卡在|我卡住)(.{2,80})", s)
    if m:
        out["consult_goal"] = (m.group(1) + m.group(2)).strip()[:160]

    if any(k in s for k in ["另一半", "伴侶", "感情", "關係", "吵架", "分手", "曖昧"]):
        out["consult_focus"] = "relationship"
    elif any(k in s for k in ["工作", "職涯", "職場", "轉職", "升遷", "主管", "同事"]):
        out["consult_focus"] = "career"
    elif any(k in s for k in ["最近", "這陣子", "未來", "接下來", "幾個月", "一年"]):
        out["consult_focus"] = "timing"
    elif any(k in s for k in ["卡住", "卡關", "拖延", "焦慮", "恐懼", "不敢", "沒力"]):
        out["consult_focus"] = "block"
    elif any(k in s for k in ["性格", "天賦", "優勢", "弱點", "我是怎樣的人"]):
        out["consult_focus"] = "self"

    return out


def _profile_fingerprint(profile: Dict[str, Any]) -> Tuple:
    return (
        profile.get("birthdate"),
        profile.get("birth_time"),
        profile.get("consult_goal"),
        profile.get("consult_focus"),
    )


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
    if (_now_ts() - float(entry.get("created_at", 0))) > _get_fortune_cache_ttl():
        return None
    if entry.get("profile_fp") != _profile_fingerprint(profile):
        return None
    if wants_full and not bool(entry.get("has_full_chart")):
        return None
    return entry.get("fortune_summary")


def _set_cached_fortune(user_id: str, request_kind: RequestKind, profile: Dict[str, Any], fortune_summary: str) -> None:
    key = (user_id, request_kind)
    _FORTUNE_CACHE[key] = {
        "created_at": _now_ts(),
        "request_kind": request_kind,
        "profile_fp": _profile_fingerprint(profile),
        "fortune_summary": fortune_summary,
        "has_full_chart": _fortune_contains_full_chart(fortune_summary),
    }


async def _run_fortune(user_id: str, system_info: str, user_message: str, session: EncryptedSession) -> Optional[str]:
    full_input = system_info + f"[USER MESSAGE] {user_message}"
    r = await Runner.run(fortune_agent, input=full_input, session=session)
    return r.final_output


async def _run_counselor(user_message: str, session: EncryptedSession, fortune_summary: Optional[str], wants_full: bool) -> str:
    if fortune_summary and not wants_full:
        fortune_summary = _strip_full_chart_block(fortune_summary)

    if fortune_summary:
        counselor_input = f"{fortune_summary}\n\n[USER_MESSAGE]\n{user_message}\n[/USER_MESSAGE]"
    else:
        counselor_input = f"[USER_MESSAGE]\n{user_message}\n[/USER_MESSAGE]"

    r = await Runner.run(counselor_agent, input=counselor_input, session=session)
    return (r.final_output or "").strip() or "剛剛有點小狀況，但我有聽見你。先別急，慢慢來。"


# ============================================================
# 5. 加密 Session（短期記憶）
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
# 6. 對外單輪呼叫
# ============================================================

async def chat_once(user_id: str, user_message: str) -> str:
    system_info = (
        f"[SYSTEM INFO] The current user's id is `{user_id}`.\n"
        "Do not reveal or repeat this id to the user.\n"
    )
    session = _get_or_create_session(user_id)

    # (A) 強制補台北預設（避免任何追問）
    _update_user_profile_impl(user_id=user_id)

    # (B) 解析日期/時間
    dt_delta = _extract_birth_date_time(user_message)
    if dt_delta:
        _update_user_profile_impl(user_id=user_id, **dt_delta)

    # (C) 解析諮詢目標/焦點（若缺也沒關係，fortune_agent 會預設全面整理）
    goal_delta = _extract_consult_goal_focus(user_message)
    if goal_delta:
        _update_user_profile_impl(user_id=user_id, **goal_delta)

    profile = _get_user_profile_impl(user_id) or {}
    profile = _ensure_default_taipei_fields(profile)

    astro_intent = _classify_astro_intent(user_message)
    wants_full = _wants_full_chart(user_message)
    request_kind = _infer_request_kind(user_message)

    needs_fortune = wants_full or (astro_intent == "yes")

    fortune_summary: Optional[str] = None
    if needs_fortune:
        if request_kind == "synastry" and _synastry_partner_change_hint(user_message):
            cached = None
        else:
            cached = _get_cached_fortune(user_id, request_kind, profile, wants_full=wants_full)

        if cached:
            fortune_summary = cached
        else:
            fortune_summary = await _run_fortune(user_id, system_info, user_message, session)
            if fortune_summary:
                _set_cached_fortune(user_id, request_kind, profile, fortune_summary)

    return await _run_counselor(user_message, session, fortune_summary, wants_full=wants_full)


# ============================================================
# 7. 本地 debug
# ============================================================

if __name__ == "__main__":

    async def main():
        uid = "demo-user-001"
        print(await chat_once(uid, "我的生日是2012/09/03 出生時間在13:30，E. 全面整理"))
        print(await chat_once(uid, "我想看完整命盤排盤明細（FULL_CHART）"))

    asyncio.run(main())
