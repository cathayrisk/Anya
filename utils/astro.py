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
# ⚠️ 節點的屬性名是 true_north_lunar_node，不是 mean_node。
# 寫錯的話 getattr 靜默回 None → 南北交點從資料裡整個消失，
# 而方法論多處要求解讀它 → 模型只能省略或編造。
MAIN_BODIES = ["sun", "moon", "mercury", "venus", "mars",
               "jupiter", "saturn", "uranus", "neptune", "pluto",
               "chiron", "true_north_lunar_node", "true_south_lunar_node"]
# 元素／模式分布只算行星，不算交點（交點是軸不是天體，算進去會扭曲統計）
DISTRIBUTION_EXCLUDE = {"True_North_Lunar_Node", "True_South_Lunar_Node",
                        "Mean_North_Lunar_Node", "Mean_South_Lunar_Node"}
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
    # 上午／下午前綴：正式站實測「上午10點15分」會解析失敗，模型得自己重試才成功，
    # 白白多花一次工具呼叫。這是使用者最自然的講法，應該由工具吸收。
    pm = bool(re.search(r"下午|晚上|傍晚|夜(?:間|裡)?|pm", txt, re.I))
    am = bool(re.search(r"上午|早上|凌晨|清晨|am", txt, re.I))
    noon = bool(re.search(r"中午|正午", txt))
    txt = re.sub(r"上午|早上|凌晨|清晨|下午|晚上|傍晚|夜間|夜裡|夜|中午|正午|am|pm", "", txt, flags=re.I).strip()
    txt = re.sub(r"[點時]", ":", txt).replace("分", "").strip().rstrip(":")
    if noon and not txt:
        txt = "12:00"
    if re.match(r"^\d{1,2}$", txt):        # 只給「9」→ 視為 9:00
        txt += ":00"
    m = re.match(r"^\s*(\d{1,2})[:：](\d{1,2})\s*$", txt)
    if not m:
        return _err("BAD_TIME", f"時間格式應為 HH:MM，收到：{s!r}")
    hh, mm = int(m.group(1)), int(m.group(2))
    if pm and hh < 12:
        hh += 12
    elif am and hh == 12:      # 上午12點 = 半夜 0 點
        hh = 0
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
    # 逼近星座交界：出生時間只要有一點誤差，星座就會翻掉。
    # 實測小P的月亮在雙魚 0.2°，換算成時間只離交界 22 分鐘 ——
    # 就算使用者「有給時間」，這種配置的星座仍然脆弱，必須標出來。
    if out["deg"] < 1.0 or out["deg"] > 29.0:
        out["near_cusp"] = True
    return out


ANGLE_NAMES = {"Ascendant", "Descendant", "Medium_Coeli", "Imum_Coeli"}
MAJOR_ASPECTS = {"conjunction", "opposition", "square", "trine", "sextile"}

# 軸的另一端。上升／下降、天頂／天底、南北交點各自是**一條軸的兩頭**，
# 所以「水星四分上升」與「水星四分下降」講的是同一件事，kerykeion 兩筆都給。
# 不去重的話，前 18 名會被鏡像佔掉一半——實測小P 的清單裡光交點與四軸的鏡像
# 就吃掉 7 個名額，把「太陽四分月亮」（orb 3.3）整個擠出去。
AXIS_PARTNER = {
    "Descendant": "Ascendant",
    "Imum_Coeli": "Medium_Coeli",
    "True_South_Lunar_Node": "True_North_Lunar_Node",
    "Mean_South_Lunar_Node": "Mean_North_Lunar_Node",
}
# 端點換到軸的另一頭時，相位跟著鏡射（180° 之差）
_MIRROR_ASPECT = {"conjunction": "opposition", "opposition": "conjunction",
                  "trine": "sextile", "sextile": "trine", "square": "square"}


def _dedupe_axis_mirrors(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把軸點鏡像收斂成一筆，保留指向主要端（上升／天頂／北交點）的那筆。"""
    seen, out = set(), []
    for r in rows:
        p1, p2, asp = r.get("p1"), r.get("p2"), r.get("aspect")
        flips = 0
        c1 = AXIS_PARTNER.get(p1, p1)
        c2 = AXIS_PARTNER.get(p2, p2)
        flips += (c1 != p1) + (c2 != p2)
        casp = asp
        if flips == 1:                       # 只有一端被鏡射 → 相位要跟著換
            casp = _MIRROR_ASPECT.get(asp, asp)
        key = (frozenset((c1, c2)), casp, r.get("orb"))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _reported_bodies(subject: Any, houses_ok: bool = True) -> set:
    """實際會出現在 points/angles 裡的名稱集合——相位只能提到這些。"""
    names = {getattr(getattr(subject, b, None), "name", None) for b in MAIN_BODIES}
    if houses_ok:
        names |= ANGLE_NAMES
    return {n for n in names if n}


def _aspects(chart_data: Any, limit: int = 18, houses_ok: bool = True,
             allowed: Optional[set] = None) -> List[Dict[str, Any]]:
    """取相位並依容許度由緊到鬆排序——最緊的相位資訊量最高。

    houses_ok=False 時**必須濾掉牽涉四軸的相位**。
    這裡曾經漏過：摘要區塊已經拿掉宮位與上升，相位清單卻還留著
    「Moon 對分 Ascendant 0.72」——等於把剛拿掉的東西從後門還回去，
    而且帶著看起來很精確的容許度。降級要在每一個出口都做，不是只做主要那個。

    allowed 用來把相位限縮在有回報位置的星體上：若相位提到 Mean_Lilith
    但星體清單裡沒有它，模型看得到關係卻不知道它在哪，只能靠猜。
    """
    rows = []
    for a in (getattr(chart_data, "aspects", None) or []):
        try:
            p1 = getattr(a, "p1_name", None)
            p2 = getattr(a, "p2_name", None)
            if not houses_ok and (p1 in ANGLE_NAMES or p2 in ANGLE_NAMES):
                continue
            if allowed is not None and not (p1 in allowed and p2 in allowed):
                continue
            rows.append({
                "p1": p1, "p2": p2,
                "aspect": getattr(a, "aspect", None),
                "orb": round(float(getattr(a, "orbit", 0.0)), 2),
            })
        except Exception:
            continue
    rows = _dedupe_axis_mirrors(rows)
    # 主相位排在次相位前面，各自再依 orb。
    # 純以 orb 排序會讓五分相（0.73°）壓過日月四分相（3.3°），
    # 而清單只取前 18 個 → 對照 natal-reading-kit 實測，
    # 小P 的「太陽四分月亮」就是這樣被兩個五分相擠出去的。
    # 方法論把這張清單當重要性排序在用，所以順序本身就是判斷。
    rows.sort(key=lambda r: (r.get("aspect") not in MAJOR_ASPECTS,
                             abs(r.get("orb") or 99)))
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
        if (p.get("name") or "") in DISTRIBUTION_EXCLUDE:
            continue
        s = p.get("sign")
        if s in E:
            elem[E[s]] += 1
        if s in M:
            mode[M[s]] += 1
    return {"element": elem, "mode": mode}


def moon_uncertainty(birthdate: str, lat: Optional[float] = None,
                     lng: Optional[float] = None, tz: Optional[str] = None) -> Dict[str, Any]:
    """未知出生時間時，月亮當天的可能範圍。

    月亮一天移動約 13 度，落在星座交界附近時，**星座本身就是未知的**。
    實測 1990-05-18：00:00 是水瓶 24.6°、23:59 是雙魚 7.8° —— 講哪一個都是猜。
    比較當日首尾兩端，若跨星座就明白標示。失敗回空 dict（不擋主流程）。
    """
    try:
        d = parse_date(birthdate)
        if "error" in d:
            return {}
        from kerykeion import AstrologicalSubjectFactory
        ends = []
        for hh, mm in ((0, 0), (23, 59)):
            sub = AstrologicalSubjectFactory.from_birth_data(
                name="_moonprobe", year=d["year"], month=d["month"], day=d["day"],
                hour=hh, minute=mm,
                lng=DEFAULT_LNG if lng is None else float(lng),
                lat=DEFAULT_LAT if lat is None else float(lat),
                tz_str=tz or DEFAULT_TZ, zodiac_type="Tropical",
                houses_system_identifier="P", calculate_lunar_phase=False, online=False)
            m = getattr(sub, "moon", None)
            ends.append((getattr(m, "sign", None), round(float(getattr(m, "position", 0.0)), 2)))
        (s0, d0), (s1, d1) = ends
        out = {
            "range_start": {"sign": s0, "sign_zh": sign_zh(s0), "deg": d0},
            "range_end": {"sign": s1, "sign_zh": sign_zh(s1), "deg": d1},
            "crosses_sign": s0 != s1,
        }
        if s0 != s1:
            out["note"] = (f"當天月亮從{sign_zh(s0)}跨到{sign_zh(s1)}——"
                           "**沒有出生時間就無法確定月亮星座**，"
                           "必須告訴使用者兩種可能，不可任選一個當事實。")
        else:
            out["note"] = (f"當天月亮都在{sign_zh(s0)}（{d0}°→{d1}°），星座確定，"
                           "但度數與相位容許度仍有不確定性。")
        return out
    except Exception:
        return {}


# 常見出生地。正式站實測：使用者說「台北出生」，但模型沒把座標填進工具參數，
# 於是 lat/lng 皆為 None → location_defaulted=True → 宮位與四軸被整組移除。
# 使用者**明明給了地點**，卻拿到降級盤。地名是人類最自然的講法，該由工具吸收。
CITIES = {
    "台北": (25.0330, 121.5654, "Asia/Taipei"), "臺北": (25.0330, 121.5654, "Asia/Taipei"),
    "新北": (25.0169, 121.4628, "Asia/Taipei"), "基隆": (25.1276, 121.7392, "Asia/Taipei"),
    "桃園": (24.9936, 121.3010, "Asia/Taipei"), "新竹": (24.8138, 120.9675, "Asia/Taipei"),
    "苗栗": (24.5602, 120.8214, "Asia/Taipei"), "台中": (24.1477, 120.6736, "Asia/Taipei"),
    "臺中": (24.1477, 120.6736, "Asia/Taipei"), "彰化": (24.0518, 120.5161, "Asia/Taipei"),
    "南投": (23.9609, 120.9719, "Asia/Taipei"), "雲林": (23.7092, 120.4313, "Asia/Taipei"),
    "嘉義": (23.4801, 120.4491, "Asia/Taipei"), "台南": (22.9999, 120.2269, "Asia/Taipei"),
    "臺南": (22.9999, 120.2269, "Asia/Taipei"), "高雄": (22.6273, 120.3014, "Asia/Taipei"),
    "屏東": (22.5519, 120.5487, "Asia/Taipei"), "宜蘭": (24.7021, 121.7378, "Asia/Taipei"),
    "花蓮": (23.9871, 121.6015, "Asia/Taipei"), "台東": (22.7583, 121.1444, "Asia/Taipei"),
    "臺東": (22.7583, 121.1444, "Asia/Taipei"), "澎湖": (23.5712, 119.5793, "Asia/Taipei"),
    "金門": (24.4493, 118.3767, "Asia/Taipei"),
    # 海外（缺地點最危險的情況——同一時間台北是獅子上升、東京是處女上升）
    "東京": (35.6762, 139.6503, "Asia/Tokyo"), "大阪": (34.6937, 135.5023, "Asia/Tokyo"),
    "首爾": (37.5665, 126.9780, "Asia/Seoul"), "香港": (22.3193, 114.1694, "Asia/Hong_Kong"),
    "新加坡": (1.3521, 103.8198, "Asia/Singapore"), "上海": (31.2304, 121.4737, "Asia/Shanghai"),
    "北京": (39.9042, 116.4074, "Asia/Shanghai"), "曼谷": (13.7563, 100.5018, "Asia/Bangkok"),
    "紐約": (40.7128, -74.0060, "America/New_York"),
    "洛杉磯": (34.0522, -118.2437, "America/Los_Angeles"),
    "舊金山": (37.7749, -122.4194, "America/Los_Angeles"),
    "溫哥華": (49.2827, -123.1207, "America/Vancouver"),
    "多倫多": (43.6532, -79.3832, "America/Toronto"),
    "倫敦": (51.5074, -0.1278, "Europe/London"), "巴黎": (48.8566, 2.3522, "Europe/Paris"),
    "雪梨": (-33.8688, 151.2093, "Australia/Sydney"),
    "墨爾本": (-37.8136, 144.9631, "Australia/Melbourne"),
    "taipei": (25.0330, 121.5654, "Asia/Taipei"), "kaohsiung": (22.6273, 120.3014, "Asia/Taipei"),
    "taichung": (24.1477, 120.6736, "Asia/Taipei"), "tainan": (22.9999, 120.2269, "Asia/Taipei"),
    "tokyo": (35.6762, 139.6503, "Asia/Tokyo"), "new york": (40.7128, -74.0060, "America/New_York"),
    "london": (51.5074, -0.1278, "Europe/London"),
}


def resolve_city(city: Optional[str]) -> Optional[Tuple[float, float, str]]:
    """地名 → (lat, lng, tz)。認不出來回 None（呼叫端該去問使用者，不要亂猜）。"""
    if not city:
        return None
    t = str(city).strip().lower()
    t = re.sub(r"(市|縣|區|巿)$", "", t).strip()
    if t in CITIES:
        return CITIES[t]
    for k, v in CITIES.items():          # 「台北市信義區」這種寫法
        if k in t:
            return v
    return None


def _build_subject(name: str, birthdate: str, birth_time: Optional[str],
                   lat: Optional[float], lng: Optional[float], tz: Optional[str],
                   zodiac_type: str = "Tropical",
                   houses_system: str = "P",
                   city: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
    """建立 kerykeion subject。回傳 (subject, meta)；失敗時 subject 為 None、
    meta 是 error dict。"""
    d = parse_date(birthdate)
    if "error" in d:
        return None, d
    t = parse_time(birth_time)
    if "error" in t:
        return None, t

    # 地名補位：使用者說「台北出生」而模型沒填座標時，這裡把它補上，
    # 並且**不算 location_defaulted**——使用者確實給了地點。
    city_hit = None
    if lat is None and lng is None:
        city_hit = resolve_city(city)
        if city_hit:
            lat, lng = city_hit[0], city_hit[1]
            tz = tz or city_hit[2]

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
                     "location_from_city": bool(city_hit),
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


def _subject_summary(subject: Any, houses_ok: bool = True) -> Dict[str, Any]:
    """houses_ok=False 時**完全不輸出**宮位與四軸。

    這是刻意的：出生時間或地點不可信時，宮位與上升是算得出來、但沒有意義的數字。
    留著只會讓模型拿去做看似精確的解讀。拿掉之後它想講也沒有資料。"""
    pts = []
    for b in MAIN_BODIES:
        p = _point(getattr(subject, b, None))
        if p:
            if not houses_ok:
                p.pop("house", None)
            pts.append(p)
    angles = {}
    if houses_ok:
        for a in ANGLES:
            p = _point(getattr(subject, a, None))
            if p:
                angles[a] = {"sign": p["sign"], "sign_zh": p["sign_zh"], "deg": p["deg"]}
    lp = getattr(subject, "lunar_phase", None)
    out: Dict[str, Any] = {
        "points": pts,
        "distribution": _distribution(pts),
        "houses_available": bool(houses_ok),
    }
    if houses_ok:
        out["angles"] = angles
    else:
        out["houses_unavailable_reason"] = (
            "出生時間或地點不可信，宮位與上升／天頂等四軸已從資料中移除。"
            "不可以推測或宣稱任何宮位配置。"
        )
    if lp is not None:
        out["lunar_phase"] = {
            "name": getattr(lp, "moon_phase_name", None),
            "illumination": getattr(lp, "degrees_between_s_m", None),
        }
    oob = [p["name"] for p in pts if p.get("out_of_bounds")]
    if oob:
        out["out_of_bounds"] = oob
    cusp = [f"{p['name']}（{p['sign_zh']}{p['deg']}°）" for p in pts if p.get("near_cusp")]
    if cusp:
        out["near_cusp_warning"] = {
            "points": cusp,
            "note": ("這些星體離星座交界不到 1 度。出生時間只要差幾十分鐘，"
                     "星座就會變成隔壁那個。解讀時要說明這個脆弱性，"
                     "若使用者的出生時間本身是聽說的、非戶籍精確時間，更要謹慎。"),
        }
    return out


# ---------------------------------------------------------------- 對外 API
def compute_natal(name: str, birthdate: str, birth_time: Optional[str] = None,
                  city: Optional[str] = None,
                  lat: Optional[float] = None, lng: Optional[float] = None,
                  tz: Optional[str] = None, detail: bool = False) -> Dict[str, Any]:
    """本命盤。detail=True 才附上 kerykeion 完整 context（約 10K 字元）。"""
    if _kerykeion() is None:
        return _err("NO_KERYKEION", "伺服器未安裝 kerykeion，占星功能暫時無法使用。")
    try:
        subject, meta = _build_subject(name, birthdate, birth_time, lat, lng, tz, city=city)
        if subject is None:
            return meta
        from kerykeion import ChartDataFactory
        cd = ChartDataFactory.create_natal_chart_data(subject)
        houses_ok = not (meta["time_approximated"] or meta["location_defaulted"])
        out: Dict[str, Any] = {
            "kind": "natal",
            "name": name,
            "birthdate": birthdate,
            "birth_time": meta["birth_time"] if not meta["time_approximated"] else None,
            "location": {"lat": DEFAULT_LAT if lat is None else lat,
                         "lng": DEFAULT_LNG if lng is None else lng,
                         "tz": tz or DEFAULT_TZ},
            **_subject_summary(subject, houses_ok=houses_ok),
            "aspects": _aspects(cd, houses_ok=houses_ok,
                                allowed=_reported_bodies(subject, houses_ok)),
        }
        w = build_warning(meta)
        if w:
            out["warning"] = w
        if meta["time_approximated"]:
            mu = moon_uncertainty(birthdate, lat, lng, tz)
            if mu:
                out["moon_uncertainty"] = mu
        if detail:
            if houses_ok:
                from kerykeion import to_context
                out["context"] = to_context(cd)
            else:
                # to_context 內含宮位與四軸，資料不可信時給了等於把剛拿掉的東西還回去
                out["context_withheld"] = (
                    "出生時間或地點不可信，完整技術脈絡含大量宮位資料，已不提供，"
                    "以免據以做出不可靠的解讀。")
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
        a_ok = not (m1["time_approximated"] or m1["location_defaulted"])
        b_ok = not (m2["time_approximated"] or m2["location_defaulted"])
        out: Dict[str, Any] = {
            "kind": "synastry",
            "a": {"name": a_name, **_subject_summary(s1, houses_ok=a_ok)},
            "b": {"name": b_name, **_subject_summary(s2, houses_ok=b_ok)},
            "cross_aspects": _aspects(scd, limit=20, houses_ok=(a_ok and b_ok)),
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
        houses_ok = not (meta["time_approximated"] or meta["location_defaulted"])
        out: Dict[str, Any] = {
            "kind": "solar_return" if return_type == "Solar" else "lunar_return",
            "name": name,
            "target_year": ty,
            "return_type": return_type,
            **_subject_summary(ret, houses_ok=houses_ok),
            "aspects": _aspects(rcd, limit=14, houses_ok=houses_ok,
                                allowed=_reported_bodies(ret, houses_ok)),
        }
        w = build_warning(meta)
        if w:
            out["warning"] = w
        return out
    except Exception as e:
        return _err("KERYKEION_ERROR", f"計算返照盤失敗：{type(e).__name__}: {e}")
