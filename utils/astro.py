# -*- coding: utf-8 -*-
"""占星計算（kerykeion / Swiss Ephemeris）——三層架構的第一層：算資料。

設計原則（對齊 natal-reading-kit 的方法論）：
- **算資料是程式的事，解讀是 LLM 的事。** 這個模組只吐決定性的數據，
  絕不做任何象徵詮釋；詮釋規則放在 skills/astrology/*。
- **絕不拋例外。** 呼叫端是 LLM 工具，未捕捉的例外會讓整個回合失敗。
  一律回 {"error": CODE, "detail": ...}。
- **惰性 import。** kerykeion 缺失時只讓占星功能停用，不擋整頁載入
  （與專案既有的 supabase / skill_loader 降級慣例一致）。
- **預設回精簡摘要，不回完整 context。** kerykeion 的 to_context() 本命盤
  約 10K 字元（≈5K tokens），而本專案 HISTORY_SUMMARY_TRIGGER_TOKENS 只有
  6000 → 一次工具呼叫就會把脈絡撐爆、加速撞免費層限流。
  需要全文時明確傳 detail=True。

驗證：本機 Python 3.11 venv 與正式站 Python 3.12 Linux 皆為 kerykeion 5.12.9，
同一組出生資料輸出逐字元相同（Swiss Ephemeris 為決定性計算）。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# 台北預設值（使用者沒給地點時用；與 yoda agent 既有慣例一致）
DEFAULT_LAT = 25.0330
DEFAULT_LNG = 121.5654
DEFAULT_TZ = "Asia/Taipei"
DEFAULT_CITY = "Taipei"
DEFAULT_NATION = "TW"

# 中文對照（只做名詞翻譯，不涉及任何詮釋）
SIGN_ZH = {
    "Ari": "牡羊", "Tau": "金牛", "Gem": "雙子", "Can": "巨蟹",
    "Leo": "獅子", "Vir": "處女", "Lib": "天秤", "Sco": "天蠍",
    "Sag": "射手", "Cap": "摩羯", "Aqu": "水瓶", "Pis": "雙魚",
}
_HOUSE_NUM = {
    "First": 1, "Second": 2, "Third": 3, "Fourth": 4, "Fifth": 5, "Sixth": 6,
    "Seventh": 7, "Eighth": 8, "Ninth": 9, "Tenth": 10, "Eleventh": 11, "Twelfth": 12,
}
# 年度小限的宮主星（傳統七行星系統）
SIGN_RULER_ZH = {
    "Ari": "火星", "Tau": "金星", "Gem": "水星", "Can": "月亮",
    "Leo": "太陽", "Vir": "水星", "Lib": "金星", "Sco": "火星",
    "Sag": "木星", "Cap": "土星", "Aqu": "土星", "Pis": "木星",
}
MAIN_BODIES = ["sun", "moon", "mercury", "venus", "mars",
               "jupiter", "saturn", "uranus", "neptune", "pluto",
               "mean_node", "chiron"]
ANGLES = ["ascendant", "medium_coeli", "descendant", "imum_coeli"]


# ---------------------------------------------------------------- 基礎工具
def _err(code: str, detail: str) -> Dict[str, Any]:
    return {"error": code, "detail": detail}


def _kerykeion():
    """惰性載入。缺套件時回 None，呼叫端一律回 error 而不是炸掉。"""
    try:
        import kerykeion  # noqa: F401
        return kerykeion
    except Exception:
        return None


def parse_date(s: str) -> Dict[str, Any]:
    """接受 YYYY-MM-DD / YYYY/MM/DD。純函式可測。"""
    # 也接受「1990年5月15日」「1990.5.15」——那是使用者最自然的寫法，
    # 逼他改成 ISO 格式只是把工具的限制推給使用者。
    txt = str(s or "").strip()
    txt = re.sub(r"[年月\.]", "-", txt).replace("日", "").strip().rstrip("-")
    m = re.match(r"^\s*(\d{4})[-/](\d{1,2})[-/](\d{1,2})\s*$", txt)
    if not m:
        return _err("BAD_DATE", f"日期格式應為 YYYY-MM-DD，收到：{s!r}")
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return _err("BAD_DATE", f"月份或日期超出範圍：{s!r}")
    return {"year": y, "month": mo, "day": d}


def parse_time(s: Optional[str]) -> Dict[str, Any]:
    """接受 HH:MM。缺值時退回中午 12:00 並標記 approximated——
    出生時間會直接影響上升點與宮位，不能默默假裝知道。純函式可測。"""
    if not s or not str(s).strip():
        return {"hour": 12, "minute": 0, "approximated": True}
    txt = str(s).strip()
    txt = re.sub(r"[點時]", ":", txt).replace("分", "").strip().rstrip(":")
    if re.match(r"^\d{1,2}$", txt):        # 只給「9」→ 視為 9:00
        txt += ":00"
    m = re.match(r"^\s*(\d{1,2})[:：](\d{1,2})\s*$", txt)
    if not m:
        return _err("BAD_TIME", f"時間格式應為 HH:MM，收到：{s!r}")
    hh, mm = int(m.group(1)), int(m.group(2))
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        return _err("BAD_TIME", f"時間超出範圍：{s!r}")
    return {"hour": hh, "minute": mm, "approximated": False}


def house_num(house_name: Any) -> Optional[int]:
    """把 kerykeion 的 'Ninth_House' 轉成 9。純函式可測。"""
    if not house_name:
        return None
    return _HOUSE_NUM.get(str(house_name).split("_")[0])


def sign_zh(sign: Any) -> str:
    return SIGN_ZH.get(str(sign or ""), str(sign or ""))


def _point(obj: Any) -> Optional[Dict[str, Any]]:
    """把 kerykeion 的星體物件壓成精簡 dict。"""
    if obj is None:
        return None
    sign = getattr(obj, "sign", None)
    out = {
        "name": getattr(obj, "name", None),
        "sign": sign,
        "sign_zh": sign_zh(sign),
        "deg": round(float(getattr(obj, "position", 0.0)), 2),
        "abs_deg": round(float(getattr(obj, "abs_pos", 0.0)), 2),
        "house": house_num(getattr(obj, "house", None)),
        "retrograde": bool(getattr(obj, "retrograde", False)),
    }
    if getattr(obj, "out_of_bounds", None):
        out["out_of_bounds"] = True
    return out


def _aspects(chart_data: Any, limit: int = 18) -> List[Dict[str, Any]]:
    """取相位並依容許度由緊到鬆排序——最緊的相位資訊量最高。"""
    rows = []
    for a in (getattr(chart_data, "aspects", None) or []):
        try:
            rows.append({
                "p1": getattr(a, "p1_name", None),
                "p2": getattr(a, "p2_name", None),
                "aspect": getattr(a, "aspect", None),
                "orb": round(float(getattr(a, "orbit", 0.0)), 2),
            })
        except Exception:
            continue
    rows.sort(key=lambda r: abs(r.get("orb") or 99))
    return rows[:limit]


def _distribution(points: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """元素／模式分布（只算主要行星，不含軸點）。純統計，不做詮釋。"""
    elem = {"火": 0, "土": 0, "風": 0, "水": 0}
    mode = {"開創": 0, "固定": 0, "變動": 0}
    E = {"Ari": "火", "Leo": "火", "Sag": "火", "Tau": "土", "Vir": "土", "Cap": "土",
         "Gem": "風", "Lib": "風", "Aqu": "風", "Can": "水", "Sco": "水", "Pis": "水"}
    M = {"Ari": "開創", "Can": "開創", "Lib": "開創", "Cap": "開創",
         "Tau": "固定", "Leo": "固定", "Sco": "固定", "Aqu": "固定",
         "Gem": "變動", "Vir": "變動", "Sag": "變動", "Pis": "變動"}
    for p in points:
        s = p.get("sign")
        if s in E:
            elem[E[s]] += 1
        if s in M:
            mode[M[s]] += 1
    return {"element": elem, "mode": mode}


def _build_subject(name: str, birthdate: str, birth_time: Optional[str],
                   lat: Optional[float], lng: Optional[float], tz: Optional[str],
                   zodiac_type: str = "Tropical",
                   houses_system: str = "P") -> Tuple[Any, Dict[str, Any]]:
    """建立 kerykeion subject。回傳 (subject, meta)；失敗時 subject 為 None、
    meta 是 error dict。"""
    d = parse_date(birthdate)
    if "error" in d:
        return None, d
    t = parse_time(birth_time)
    if "error" in t:
        return None, t

    from kerykeion import AstrologicalSubjectFactory
    subject = AstrologicalSubjectFactory.from_birth_data(
        name=name or "Subject",
        year=d["year"], month=d["month"], day=d["day"],
        hour=t["hour"], minute=t["minute"],
        lng=DEFAULT_LNG if lng is None else float(lng),
        lat=DEFAULT_LAT if lat is None else float(lat),
        tz_str=tz or DEFAULT_TZ,
        zodiac_type=zodiac_type,
        houses_system_identifier=houses_system,
        calculate_lunar_phase=True,
        online=False,
    )
    return subject, {"time_approximated": t["approximated"],
                     "location_defaulted": (lat is None and lng is None),
                     "birth_time": f"{t['hour']:02d}:{t['minute']:02d}"}


def build_warning(meta: Dict[str, Any], who: str = "") -> str:
    """把「資料不完整」組成一句給模型看的警告。

    地點預設特別危險：實測同一時間在台北 vs 東京，上升會從獅子變處女——
    整張盤的解讀基礎就錯了。缺時間至少還會提醒，地點若靜默預設就完全無跡可循。
    純函式可測。
    """
    who = f"{who} " if who else ""
    parts = []
    if meta.get("time_approximated"):
        parts.append(f"{who}出生時間未提供，已用中午 12:00 計算")
    if meta.get("location_defaulted"):
        parts.append(f"{who}出生地點未提供，已預設台北")
    if not parts:
        return ""
    tail = ("上升點與宮位會因此不準確——**必須先向使用者說明這個限制、並詢問正確資料**，"
            "再決定要不要繼續解讀。若出生地不在台灣，上升星座可能整個不同。")
    return "；".join(parts) + "。" + tail


def _subject_summary(subject: Any) -> Dict[str, Any]:
    pts = []
    for b in MAIN_BODIES:
        p = _point(getattr(subject, b, None))
        if p:
            pts.append(p)
    angles = {}
    for a in ANGLES:
        p = _point(getattr(subject, a, None))
        if p:
            angles[a] = {"sign": p["sign"], "sign_zh": p["sign_zh"], "deg": p["deg"]}
    lp = getattr(subject, "lunar_phase", None)
    out: Dict[str, Any] = {
        "points": pts,
        "angles": angles,
        "distribution": _distribution(pts),
    }
    if lp is not None:
        out["lunar_phase"] = {
            "name": getattr(lp, "moon_phase_name", None),
            "illumination": getattr(lp, "degrees_between_s_m", None),
        }
    oob = [p["name"] for p in pts if p.get("out_of_bounds")]
    if oob:
        out["out_of_bounds"] = oob
    return out


# ---------------------------------------------------------------- 對外 API
def compute_natal(name: str, birthdate: str, birth_time: Optional[str] = None,
                  lat: Optional[float] = None, lng: Optional[float] = None,
                  tz: Optional[str] = None, detail: bool = False) -> Dict[str, Any]:
    """本命盤。detail=True 才附上 kerykeion 完整 context（約 10K 字元）。"""
    if _kerykeion() is None:
        return _err("NO_KERYKEION", "伺服器未安裝 kerykeion，占星功能暫時無法使用。")
    try:
        subject, meta = _build_subject(name, birthdate, birth_time, lat, lng, tz)
        if subject is None:
            return meta
        from kerykeion import ChartDataFactory
        cd = ChartDataFactory.create_natal_chart_data(subject)
        out: Dict[str, Any] = {
            "kind": "natal",
            "name": name,
            "birthdate": birthdate,
            "birth_time": meta["birth_time"],
            "location": {"lat": DEFAULT_LAT if lat is None else lat,
                         "lng": DEFAULT_LNG if lng is None else lng,
                         "tz": tz or DEFAULT_TZ},
            **_subject_summary(subject),
            "aspects": _aspects(cd),
        }
        w = build_warning(meta)
        if w:
            out["warning"] = w
        if detail:
            from kerykeion import to_context
            out["context"] = to_context(cd)
        return out
    except Exception as e:
        return _err("KERYKEION_ERROR", f"計算本命盤失敗：{type(e).__name__}: {e}")


def compute_synastry(a_name: str, a_birthdate: str, a_birth_time: Optional[str],
                     b_name: str, b_birthdate: str, b_birth_time: Optional[str],
                     a_lat: Optional[float] = None, a_lng: Optional[float] = None,
                     a_tz: Optional[str] = None,
                     b_lat: Optional[float] = None, b_lng: Optional[float] = None,
                     b_tz: Optional[str] = None,
                     with_composite: bool = True) -> Dict[str, Any]:
    """合盤：synastry 相位 + 組合中點盤 + 相容性評分（Ciro Discepolo）。"""
    if _kerykeion() is None:
        return _err("NO_KERYKEION", "伺服器未安裝 kerykeion，占星功能暫時無法使用。")
    try:
        s1, m1 = _build_subject(a_name, a_birthdate, a_birth_time, a_lat, a_lng, a_tz)
        if s1 is None:
            return m1
        s2, m2 = _build_subject(b_name, b_birthdate, b_birth_time, b_lat, b_lng, b_tz)
        if s2 is None:
            return m2
        from kerykeion import ChartDataFactory
        scd = ChartDataFactory.create_synastry_chart_data(s1, s2)
        out: Dict[str, Any] = {
            "kind": "synastry",
            "a": {"name": a_name, **_subject_summary(s1)},
            "b": {"name": b_name, **_subject_summary(s2)},
            "cross_aspects": _aspects(scd, limit=20),
        }
        warns = [w for w in (build_warning(m1, a_name), build_warning(m2, b_name)) if w]
        if warns:
            out["warning"] = " ".join(warns)

        # 相容性評分：上游 API 變動時只降級這一項，不讓整個合盤失敗
        try:
            from kerykeion import RelationshipScoreFactory
            rs = RelationshipScoreFactory(s1, s2).get_relationship_score()
            out["relationship_score"] = {
                "score": getattr(rs, "score_value", None) or getattr(rs, "score", None),
                "description": getattr(rs, "score_description", None),
            }
        except Exception as e:
            out["relationship_score_error"] = f"{type(e).__name__}: {e}"

        if with_composite:
            try:
                from kerykeion import CompositeSubjectFactory
                comp = CompositeSubjectFactory(s1, s2).get_midpoint_composite_subject_model()
                ccd = ChartDataFactory.create_composite_chart_data(comp)
                out["composite"] = {**_subject_summary(comp), "aspects": _aspects(ccd, limit=12)}
            except Exception as e:
                out["composite_error"] = f"{type(e).__name__}: {e}"
        return out
    except Exception as e:
        return _err("KERYKEION_ERROR", f"計算合盤失敗：{type(e).__name__}: {e}")


def compute_profection(birthdate: str, target_year: int,
                       natal_asc_sign: Optional[str] = None) -> Dict[str, Any]:
    """年度小限（Annual Profection）。純算術：年齡 mod 12 決定啟動宮，
    該宮所在星座的傳統主星即為年主星。不需要 kerykeion。純函式可測。"""
    d = parse_date(birthdate)
    if "error" in d:
        return d
    try:
        ty = int(target_year)
    except Exception:
        return _err("BAD_YEAR", f"年份需為整數，收到：{target_year!r}")
    age = ty - d["year"]
    if age < 0:
        return _err("BAD_YEAR", f"目標年份 {ty} 早於出生年 {d['year']}。")
    house = (age % 12) + 1
    out: Dict[str, Any] = {
        "kind": "profection",
        "target_year": ty,
        "age_at_birthday": age,
        "activated_house": house,
        "note": "以生日為界；生日前仍屬前一年的小限宮。",
    }
    if natal_asc_sign:
        order = list(SIGN_ZH.keys())
        try:
            idx = order.index(str(natal_asc_sign)[:3])
        except ValueError:
            return out
        sign = order[(idx + house - 1) % 12]
        out["activated_sign"] = sign
        out["activated_sign_zh"] = sign_zh(sign)
        out["year_lord_zh"] = SIGN_RULER_ZH.get(sign)
    return out


def wheel_data(chart: Dict[str, Any]) -> Dict[str, Any]:
    """把 compute_natal 的輸出壓成 widget 畫輪盤用的最小資料集。
    只保留繪圖需要的欄位，避免 widget HTML 超過 20K 字元上限。純函式可測。"""
    pts = []
    for p in (chart.get("points") or []):
        if p.get("abs_deg") is None:
            continue
        pts.append({"n": p.get("name"), "d": p.get("abs_deg"),
                    "s": p.get("sign_zh"), "h": p.get("house"),
                    "r": 1 if p.get("retrograde") else 0})
    asc = ((chart.get("angles") or {}).get("ascendant") or {})
    return {
        "points": pts,
        "asc_deg": asc.get("deg"),
        "aspects": [{"a": a["p1"], "b": a["p2"], "t": a["aspect"], "o": a["orb"]}
                    for a in (chart.get("aspects") or [])[:12]],
        "distribution": chart.get("distribution"),
    }


# ---------------------------------------------------------------- 預測
MAJOR_ASPECTS = ("conjunction", "opposition", "trine", "square", "sextile")
SLOW_BODIES = ("Jupiter", "Saturn", "Uranus", "Neptune", "Pluto", "Chiron")


def compute_transits(name: str, birthdate: str, birth_time: Optional[str],
                     start_date: str, days: int = 30, step: int = 2,
                     lat: Optional[float] = None, lng: Optional[float] = None,
                     tz: Optional[str] = None, limit: int = 25) -> Dict[str, Any]:
    """行運：在區間內掃出「行運星 → 本命星」的主要相位。

    只回精華（依容許度排序、慢行星優先），不回每一天的完整格點——
    完整資料量會把 LLM 脈絡撐爆，而且對解讀沒有增益。
    """
    if _kerykeion() is None:
        return _err("NO_KERYKEION", "伺服器未安裝 kerykeion，占星功能暫時無法使用。")
    try:
        natal, meta = _build_subject(name, birthdate, birth_time, lat, lng, tz)
        if natal is None:
            return meta
        sd = parse_date(start_date)
        if "error" in sd:
            return sd
        try:
            days = max(1, min(int(days), 400))     # 上限防呆：太長會算很久也讀不完
            step = max(1, min(int(step), 30))
        except Exception:
            return _err("BAD_RANGE", "days／step 需為整數。")

        from datetime import datetime, timedelta
        from kerykeion.ephemeris_data_factory import EphemerisDataFactory
        from kerykeion.transits_time_range_factory import TransitsTimeRangeFactory

        tzs = tz or DEFAULT_TZ
        _lat = DEFAULT_LAT if lat is None else float(lat)
        _lng = DEFAULT_LNG if lng is None else float(lng)
        start = datetime(sd["year"], sd["month"], sd["day"])
        end = start + timedelta(days=days)

        eph = EphemerisDataFactory(start_datetime=start, end_datetime=end,
                                   step_type="days", step=step,
                                   lat=_lat, lng=_lng, tz_str=tzs)
        points = eph.get_ephemeris_data_as_astrological_subjects()
        res = TransitsTimeRangeFactory(natal_chart=natal,
                                       ephemeris_data_points=points).get_transit_moments()

        # kerykeion 的時間戳是 UTC。直接截字串會讓 UTC+8 的日期整批早一天，
        # 讓整份預測的時機悄悄偏移（natal-reading-kit 踩過並修掉的坑）。
        # 用 pytz 而非 zoneinfo：Windows 沒有內建 tz 資料庫，tzdata 未必裝得到。
        import pytz
        local_tz = pytz.timezone(tzs)

        def _local_date(iso: str) -> str:
            return datetime.fromisoformat(iso).astimezone(local_tz).strftime("%Y-%m-%d")

        rows: List[Dict[str, Any]] = []
        for mom in (getattr(res, "transits", None) or []):
            try:
                date_s = _local_date(str(getattr(mom, "date", "")))
            except Exception:
                date_s = str(getattr(mom, "date", ""))[:10]
            for asp in (getattr(mom, "aspects", None) or []):
                d = asp.model_dump() if hasattr(asp, "model_dump") else dict(asp)
                if d.get("aspect") not in MAJOR_ASPECTS:
                    continue
                rows.append({
                    "date": date_s,
                    "transiting": d.get("p1_name"),
                    "aspect": d.get("aspect"),
                    "natal": d.get("p2_name"),
                    "orb": round(abs(float(d.get("orbit", 0) or 0)), 2),
                    "slow": bool(d.get("p1_name") in SLOW_BODIES),
                })

        # 慢行星優先、再依容許度由緊到鬆：慢行星的行運才是「這段期間的主題」
        rows.sort(key=lambda r: (0 if r["slow"] else 1, r["orb"]))
        out: Dict[str, Any] = {
            "kind": "transits",
            "name": name,
            "range": {"start": start_date, "days": days, "step_days": step, "tz": tzs},
            "highlights": rows[:limit],
            "total_found": len(rows),
        }
        w = build_warning(meta)
        if w:
            out["warning"] = w
        return out
    except Exception as e:
        return _err("KERYKEION_ERROR", f"計算行運失敗：{type(e).__name__}: {e}")


def compute_solar_return(name: str, birthdate: str, birth_time: Optional[str],
                         target_year: int,
                         lat: Optional[float] = None, lng: Optional[float] = None,
                         tz: Optional[str] = None,
                         return_type: str = "Solar") -> Dict[str, Any]:
    """太陽／月亮返照盤。以「現居地」起盤（返照盤看的是當下所在地）。"""
    if _kerykeion() is None:
        return _err("NO_KERYKEION", "伺服器未安裝 kerykeion，占星功能暫時無法使用。")
    try:
        natal, meta = _build_subject(name, birthdate, birth_time, lat, lng, tz)
        if natal is None:
            return meta
        bd = parse_date(birthdate)
        try:
            ty = int(target_year)
        except Exception:
            return _err("BAD_YEAR", f"年份需為整數，收到：{target_year!r}")

        from kerykeion.planetary_return_factory import PlanetaryReturnFactory
        from kerykeion import ChartDataFactory

        rf = PlanetaryReturnFactory(natal,
                                    lng=DEFAULT_LNG if lng is None else float(lng),
                                    lat=DEFAULT_LAT if lat is None else float(lat),
                                    tz_str=tz or DEFAULT_TZ, online=False)
        # 從「生日當天」往後找最近一次返照
        ret = rf.next_return_from_date(ty, bd["month"], bd["day"], return_type=return_type)
        rcd = ChartDataFactory.create_natal_chart_data(ret)
        out: Dict[str, Any] = {
            "kind": "solar_return" if return_type == "Solar" else "lunar_return",
            "name": name,
            "target_year": ty,
            "return_type": return_type,
            **_subject_summary(ret),
            "aspects": _aspects(rcd, limit=14),
        }
        w = build_warning(meta)
        if w:
            out["warning"] = w
        return out
    except Exception as e:
        return _err("KERYKEION_ERROR", f"計算返照盤失敗：{type(e).__name__}: {e}")
