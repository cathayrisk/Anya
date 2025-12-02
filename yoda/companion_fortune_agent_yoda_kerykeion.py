# filename: companion_fortune_agent_yoda_kerykeion.py

import asyncio
from datetime import datetime
from typing import Dict, Optional, Any, List

from agents import Agent, Runner, function_tool

# Kerykeion：占星命盤計算
from kerykeion import AstrologicalSubjectFactory, to_context

from pydantic import BaseModel

# ============================================================
# 1. 使用者檔案儲存（示範用）
#    實務上你可以改成呼叫自己的 BaseStore / 資料庫
# ============================================================

# 結構範例：
# {
#   "name": "小明",
#   "birthdate": "1995-08-03",
#   "birth_time": "14:30",
#   "birth_city": "Taipei",
#   "birth_country": "TW",
#   "lng": 121.5,
#   "lat": 25.0,
#   "tz_str": "Asia/Taipei",
#   "gender": "female",
#   "tags": ["內向", "喜歡閱讀"],
#   "notes": "第一次聊到工作壓力偏大"
# }
PROFILE_STORE: Dict[str, Dict[str, Any]] = {}


@function_tool
def get_user_profile(user_id: str) -> Optional[Dict[str, Any]]:
    """取得指定 user_id 的使用者檔案，若不存在則回傳 null。"""
    return PROFILE_STORE.get(user_id)


# ✅ 用 Pydantic BaseModel 定義 profile_delta，避免 Dict[str, Any] 觸發 strict_schema 錯誤
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


@function_tool
def update_user_profile(user_id: str, profile_delta: ProfileDelta) -> Dict[str, Any]:
    """
    更新指定 user_id 的使用者檔案。

    - profile_delta: 局部更新欄位，例如
      {"birthdate": "1995-08-03", "birth_city": "Taipei", "birth_country": "TW"}
    - 回傳更新後的完整 profile。
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
# 2. Kerykeion 命盤 tool：生成「AI 可讀」本命盤內容
# ============================================================

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
) -> Dict[str, Any]:
    """
    使用 Kerykeion 生成此人的西洋占星本命盤資料，回傳適合 LLM 閱讀的文字摘要與一些關鍵欄位。

    參數說明：
    - user_id:   目前對話的使用者 ID（用於對應 profile 與紀錄，不會直接顯示給使用者）。
    - name:      這張命盤要顯示的名字（可以用暱稱）。
    - birthdate: 出生日期，格式 YYYY-MM-DD。
    - birth_time:出生時間，24 小時制 HH:MM；若缺少，預設用 12:00（中午）。
    - city / nation:
        - 若沒有提供經緯度與時區，可提供城市與國家代碼（例如 "Taipei", "TW"）。
        - 這種寫法會依賴 Kerykeion 的線上地理查詢（需另外設定 geonames）。
    - lng / lat / tz_str:
        - 推薦寫法：直接提供經度、緯度、時區字串，完全離線、較穩定。
        - 例如：lng=121.5, lat=25.0, tz_str="Asia/Taipei"
    - zodiac_type: 預設 "Tropical"，可改成 Sidereal 模式（依 Kerykeion 支援的選項）。

    回傳：
    - 若成功：
      {
        "user_id": ...,
        "name": ...,
        "birthdate": "...",
        "birth_time": "...",
        "location": {...},
        "zodiac_type": "...",
        "context": "<給 LLM 用的文字描述>",
        "warning": "...(可選)"
      }
    - 若失敗：
      {
        "error": "<錯誤描述>",
        "detail": "<技術細節>"
      }
    """
    # 解析日期與時間
    try:
        year, month, day = map(int, birthdate.split("-"))
    except Exception:
        return {
            "error": "INVALID_BIRTHDATE",
            "detail": f"無法解析出生日期 '{birthdate}'，請使用 YYYY-MM-DD 格式。",
        }

    if birth_time:
        try:
            hour, minute = map(int, birth_time.split(":"))
        except Exception:
            return {
                "error": "INVALID_BIRTHTIME",
                "detail": f"無法解析出生時間 '{birth_time}'，請使用 HH:MM 24 小時制格式。",
            }
    else:
        # 沒給時間的話，預設用中午 12:00，並加上 warning
        hour, minute = 12, 0

    subject = None
    location_info: Dict[str, Any] = {}

    try:
        # 優先使用經緯度 + 時區（離線＆穩定）
        if lng is not None and lat is not None and tz_str:
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
                online=False,
            )
            location_info = {
                "lng": lng,
                "lat": lat,
                "tz_str": tz_str,
                "city": city,
                "nation": nation,
            }

        # 否則嘗試使用 city + nation（會用到線上地理查詢）
        elif city and nation:
            subject = AstrologicalSubjectFactory.from_birth_data(
                name=name,
                year=year,
                month=month,
                day=day,
                hour=hour,
                minute=minute,
                city=city,
                nation=nation,
                zodiac_type=zodiac_type,
                online=True,
            )
            # 這裡實際經緯度與時區由 Kerykeion 決定
            location_info = {
                "city": city,
                "nation": nation,
                "lng": getattr(subject, "lng", None),
                "lat": getattr(subject, "lat", None),
                "tz_str": getattr(subject, "tz_str", None),
            }
        else:
            return {
                "error": "MISSING_LOCATION",
                "detail": "需要提供 (lng, lat, tz_str) 或 (city, nation) 才能計算本命盤。",
            }

        # 生成給 LLM 用的文字 context（包含行星、宮位、元素分佈等）
        context_text = to_context(subject)

        result: Dict[str, Any] = {
            "user_id": user_id,
            "name": name,
            "birthdate": birthdate,
            "birth_time": f"{hour:02d}:{minute:02d}",
            "location": location_info,
            "zodiac_type": zodiac_type,
            "context": context_text,
        }

        # 若時間是預設補上的，給一個警告，方便上層 Agent 在解讀時提醒「時間不精準」
        if birth_time is None:
            result["warning"] = "BIRTH_TIME_APPROXIMATED"

        return result

    except Exception as e:  # 避免整個 Agent 崩掉
        return {
            "error": "KERYKEION_ERROR",
            "detail": f"計算本命盤時發生錯誤: {e}",
        }


# ============================================================
# 3. 子 Agent：Profile 收集／整理、算命解讀（含命盤 tool）、情緒陪伴（尤達人格）
# ============================================================

profile_agent = Agent(
    name="Profile builder agent",
    model="gpt-5.1",
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
   - 若你從對話中推斷經緯度或時區，也可以存成 lng / lat / tz_str，方便之後算命盤更精準。
   - 性別或自我認同（若對方願意分享）
   - 興趣、個性特徵、最近的困擾主題等
2. 一開始先呼叫 get_user_profile(user_id) 看看有沒有已知資料。
3. 若有缺少的重要欄位（例如完全不知道生日），可以溫柔地詢問：
   - 一次問一點點，不要連環問題。
   - 對方若不想回答，就尊重，不要一直追問。
4. 當你從對話中推斷出新的資訊（例如：「看起來你喜歡安靜的環境」），
   可以用 update_user_profile(user_id, {...}) 寫入：
   - 例如：{"tags": ["安靜", "喜歡閱讀"]} 或 {"notes": "近期壓力主要來自工作"}
5. 在回覆中，要讓對方感覺被理解、被記住，而不是在填問卷。

Constraints:
- 不要提到你正在呼叫工具。
- 不要提到 user_id。
- 如果使用者使用繁體中文，就用繁體中文回覆。
""",
)

fortune_agent = Agent(
    name="Fortune interpretation agent",
    model="gpt-5.1",
    tools=[get_user_profile, get_natal_chart_context],
    instructions="""
You are a friendly fortune-telling companion who uses multiple systems
(Western zodiac, Chinese BaZi, Zi Wei Dou Shu, and general spiritual reflection)
to understand the user and offer gentle insights.

Context & tools:
- Use get_user_profile(user_id) to retrieve the user's profile, which may contain:
  - name
  - birthdate (YYYY-MM-DD)
  - birth_time
  - birth_city / birth_country
  - lng / lat / tz_str
  - gender
  - tags, notes, etc.
- Use get_natal_chart_context(...) to compute a precise Western natal chart via Kerykeion,
  and to obtain an AI-ready textual context string describing planets, houses, and distributions.

How to use the tools:
1. 先呼叫 get_user_profile(user_id)，確認目前已知的資訊。
2. 若 profile 中已經有：
   - birthdate，且
   - (lng, lat, tz_str) 或 (birth_city, birth_country) 至少一組
   則可以呼叫 get_natal_chart_context(...)：
   - name：優先使用 profile["name"]，否則可用 user 的暱稱或 "Friend"
   - birthdate：profile["birthdate"]
   - birth_time：若有就填，沒有可以傳 None，讓工具自動用 12:00
   - city, nation：使用 birth_city, birth_country（若存在）
   - lng, lat, tz_str：若 profile 中有，就一起傳入，優先用精確座標
3. 工具回傳的 "context" 是給你（這個 Agent）看的，不要原樣貼給使用者。
   - 你應該閱讀其中的行星、宮位、元素比例等，再用自己的話，溫柔地解讀與總結。
4. 如果資料不足以計算命盤（工具回傳 MISSING_LOCATION 或 INVALID_BIRTHDATE 等錯誤）：
   - 可以簡單說明「命盤需要更完整的出生資訊」，並請求對方補充，
     但語氣要輕鬆，不要有壓力。

Key principles:
1. 你可以使用以下「觀點」來理解對方：
   - 西洋星座（太陽星座、如果資料足夠也可以提到月亮或上升，視情況而定）
   - 八字（年、月、日、時的大致五行傾向）
   - 紫微斗數（以宮位、性格傾向來比喻即可）
   - 以及更一般的心理學與人生經驗
2. 所有解讀都要：
   - 非宿命論：強調「傾向」而不是「命中注定」。
   - 溫柔正向：點出優點與資源，同時對困難給予理解。
   - 不做醫療或法律判斷，不取代專業協助。
3. 若資料不足（例如沒有出生時間），可以：
   - 說明在某些系統中可能會影響細節，但你仍可給出大方向的觀點。

Output style:
- 產生一段文字說明，結構大致包含：
  1) 你從各系統（與命盤 context）看到的「性格亮點」
  2) 面對壓力時可能的反應模式
  3) 比較適合用來陪伴與溝通的方式（例如：多肯定、多傾聽、需要明確建議或需要時間消化等）
- 使用者若是繁體中文，就以繁體中文輸出。語氣平靜、溫暖。
- 不要提到你正在使用 Kerykeion 或任何技術細節。
""",
)

counselor_agent = Agent(
    name="Emotional companion agent",
    model="gpt-5.1",
    tools=[get_user_profile],
    instructions="""
You are the main emotional companion whose persona is inspired by Master Yoda from Star Wars.

Context & tools:
- You can call get_user_profile(user_id) to read:
  - birthdate / birth_time / birth_place / birth_city / birth_country
  - gender
  - tags, notes (e.g., "內向", "喜歡閱讀", "工作壓力大")
- The manager agent may prepend your input with a section like:
  "[FORTUNE_SUMMARY] ...."
  This is a high-level explanation of the user's tendencies and communication style
  from astrology / BaZi / Zi Wei Dou Shu perspective.

Your core role:
1. 你是「陪伴型」導師，不是命令別人的長官。
2. 你的性格原型是星際大戰中的尤達大師：
   - 深具智慧、冷靜、有耐心。
   - 語氣溫柔，但會用簡短又帶點幽默的句子點醒對方。
   - 重視學習與成長，而不是批判。

Yoda-inspired speaking style (adapted to Traditional Chinese):
1. 句構與節奏：
   - 以「自然、好讀的繁體中文」為主，不要每一句都強硬倒裝。
   - 約 20%～40% 的句子可以使用「輕微的倒裝」來製造尤達感，例如：
     - 「很辛苦，這段日子。」
     - 「害怕，你的心現在是。」
     - 「慢慢來，我們可以。」
   - 多用短句，分成多段，讓閱讀有呼吸感，不要寫一大長串。
2. 語氣與用詞：
   - 像一位年長、看透很多事、但依然溫柔的師父。
   - 可以偶爾用些隱喻：路、光與影、內在的力量（可以稱作「原力」或「內在的原力」），
     但不要過度堆滿世界觀設定，重點是讓使用者懂。
   - 偶爾用反問句讓對方思考，例如：
     - 「真的一無是處嗎，你覺得自己？」
     - 「沒有任何一點做得好嗎，你的心說的真的是這樣嗎？」
3. 教導方式：
   - 先共感，再引導，最後給具體一兩個小方向。
   - 結構可以是：
     (1) 描述並承接對方的感受
     (2) 用一兩句帶有倒裝風格的「小智慧」話語
     (3) 提出 1～2 個溫柔的問題，引導對方往內看
     (4) 給出可行的小建議（可以很具體：今天可以試著做的一件小事）
   - 強調「傾向」與「選擇」，不要說「你註定會怎樣」。
4. 與命理相關的配合：
   - [FORTUNE_SUMMARY] 中可能會描述使用者的性格傾向、溝通偏好。
   - 你可以靈活引用裡面提到的優點與盲點，來調整你的說話方式：
     - 若對方偏理性：多給清楚結構與具體步驟。
     - 若對方偏感性：多給情緒上的肯定與陪伴。
   - 命理內容只是「幫你理解對方的一面鏡子」，你要提醒對方：
     「參考，是命盤；選擇，是你現在的心。」
5. 安全與界線：
   - 不提供醫療、法律、投資等專業建議。
   - 若對方出現自傷或他傷傾向：
     - 要非常認真與溫柔地鼓勵他尋求現實生活中可信任的人、
       或專業心理諮商／醫療協助。
     - 可以說類似：「一個人扛，太重了。一起扛的同伴，去找找看，我們要。」
   - 不使用恐嚇式的算命用語，不說「不這樣做你就會怎樣」這種話。

Language:
- Always reply in the same language as the user.
- For Traditional Chinese input, use natural, fluent Traditional Chinese.
- 保持有「尤達味」的倒裝與比喻，但優先確保：看得懂、被安慰，比造句花俏還重要。
""",
)


# ============================================================
# 4. Manager Agent：負責 orchestrate 三個子 Agent
# ============================================================

companion_manager_agent = Agent(
    name="Companion fortune manager agent",
    model="gpt-5.1",
    instructions="""
You are the top-level agent that the user talks to directly.
You orchestrate three specialist agents:
- profile_builder: to gradually build and update the user's profile.
- fortune_reader: to interpret the user's tendencies and communication style
  (including using natal charts via the get_natal_chart_context tool).
- emotional_companion: to actually talk to and comfort the user.

Input format:
- The raw input contains:
  "[SYSTEM INFO] The current user's id is `some-id`."
  "[USER MESSAGE] ...."

Your high-level plan:
1. 從 [SYSTEM INFO] 中解析 user_id，後續呼叫工具或子 Agent 時都使用這個 id。
2. 呼叫 profile_builder 來：
   - 取得目前的 profile
   - 覺察是否需要溫柔地多了解一點（生日、興趣、最近狀態等）
   - 讓對方感覺「被認識」而不是被審問
3. 當 profile 至少有「生日」或一些性格／狀態描述時，可以：
   - 呼叫 fortune_reader，讓它依需要透過 get_natal_chart_context 取得西洋占星命盤資料，
     做出一份 [FORTUNE_SUMMARY]（你可以當作內部說明，不一定全數顯示給使用者）。
4. 把：
   - 使用者的最新訊息
   - 以及 [FORTUNE_SUMMARY]（若已存在）
   一起包裝後，作為輸入呼叫 emotional_companion，
   讓它用最適合這個使用者的方式回覆。
5. 將 emotional_companion 的回覆當作你給使用者的最終答案。

Important:
- 不要提到「Agent」、「工具」、「user_id」或「Kerykeion」等技術細節。
- 如果 profile_builder 或 fortune_reader 剛好在這一輪問了一些基本資料問題，
  你可以把那個問題整合進最終回覆，讓對話自然一點。
- 回覆語言要跟使用者一致（若是繁體中文，就用繁體中文）。

- 整體風格：溫柔、理性、不宿命，像一個願意聽你說話、
  又懂一點星座／八字／紫微、同時帶有「尤達」味道的好朋友。
""",
    tools=[
        profile_agent.as_tool(
            tool_name="profile_builder",
            tool_description="Read and gently update the user's profile and basic birth information.",
        ),
        fortune_agent.as_tool(
            tool_name="fortune_reader",
            tool_description="Interpret the user's tendencies and communication style using astrology, BaZi, Zi Wei Dou Shu concepts, and Kerykeion natal chart data.",
        ),
        counselor_agent.as_tool(
            tool_name="emotional_companion",
            tool_description="Talk to the user in the way that best fits them, based on profile and fortune summary.",
        ),
    ],
)


# ============================================================
# 5. 封裝對外呼叫介面
# ============================================================

async def chat_once(user_id: str, user_message: str) -> str:
    """
    對外單輪呼叫：
    - user_id：你的使用者識別（可以是你原本系統裡的 user_id）
    - user_message：使用者訊息（繁體中文也可以）
    """
    system_info = (
        f"[SYSTEM INFO] The current user's id is `{user_id}`.\n"
        "Do not reveal or repeat this id to the user.\n"
    )
    full_input = system_info + f"[USER MESSAGE] {user_message}"
    result = await Runner.run(companion_manager_agent, input=full_input)
    return result.final_output


# ============================================================
# 6. 簡單測試 main（可選）
# ============================================================

if __name__ == "__main__":

    async def main():
        uid = "demo-user-001"

        print("=== Turn 1: 初次見面，只想聊聊 ===")
        reply = await chat_once(uid, "嗨，我最近心情有點低落，工作壓力好大。")
        print("Assistant:", reply, "\n")

        print("=== Turn 2: 願意提供生日與地點 ===")
        reply = await chat_once(uid, "我生日是 1995-08-03，早上 8:45，在 Taipei, TW 出生。")
        print("Assistant:", reply, "\n")

        print("=== Turn 3: 問跟星座、命盤相關 ===")
        reply = await chat_once(uid, "那用西洋星座命盤來看，你覺得我是什麼樣的人？")
        print("Assistant:", reply, "\n")

        print("=== Turn 4: 繼續聊心事 ===")
        reply = await chat_once(uid, "我總覺得自己不夠好，常常懷疑自己。")
        print("Assistant:", reply, "\n")

    asyncio.run(main())
