import streamlit as st
import base64
import re
import time
import json
import asyncio
import threading
from io import BytesIO
from PIL import Image
from openai import OpenAI
from openai.types.responses import ResponseTextDeltaEvent
import os
from pypdf import PdfReader, PdfWriter

# ====== Agents SDK（Router / Planner / Search / Fast）======
from agents import (
    Agent,
    ModelSettings,
    Runner,
    handoff,
    HandoffInputData,
    RunContextWrapper,
    WebSearchTool,
)
from agents.extensions import handoff_filters
try:
    from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
except Exception:
    RECOMMENDED_PROMPT_PREFIX = ""
from agents.models import is_gpt_5_default
from openai.types.shared.reasoning import Reasoning
from pydantic import BaseModel
from typing import Literal, Optional, List, Any

# === 0. Trimming / 大小限制（可調） ===
TRIM_LAST_N_USER_TURNS = 18                 # 短期記憶：最近 N 個 user 回合
MAX_REQ_TOTAL_BYTES = 48 * 1024 * 1024      # 單次請求總量預警（48MB）

# === 0.1 取得 API Key ===
OPENAI_API_KEY = (
    st.secrets.get("OPENAI_API_KEY")
    or st.secrets.get("OPENAI_KEY")
    or os.getenv("OPENAI_API_KEY")
)
if not OPENAI_API_KEY:
    st.error("找不到 OpenAI API Key，請在 .streamlit/secrets.toml 設定 OPENAI_API_KEY 或 OPENAI_KEY。")
    st.stop()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY  # 讓 Agents SDK 可以讀到

# === 1. Streamlit 頁面 ===
st.set_page_config(page_title="Anya Multimodal Agent", page_icon="🥜", layout="wide")

# === 1.a Session 預設值保險（務必在任何使用 chat_history 前） ===
def ensure_session_defaults():
    if "chat_history" not in st.session_state or not isinstance(st.session_state.chat_history, list):
        st.session_state.chat_history = [{
            "role": "assistant",
            "text": "嗨嗨～安妮亞來了！👋 上傳圖片或PDF，直接問你想知道的內容吧！",
            "images": [],
            "docs": []
        }]

ensure_session_defaults()

# === 共用：假串流打字效果 ===
def fake_stream_markdown(text: str, placeholder, step_chars=8, delay=0.03, empty_msg="安妮亞找不到答案～（抱歉啦！）"):
    buf = ""
    for i in range(0, len(text), step_chars):
        buf = text[: i + step_chars]
        placeholder.markdown(buf)
        time.sleep(delay)
    if not text:
        placeholder.markdown(empty_msg)
    return text

# 穩定版：確保 coroutine 一定被 await
def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        result_container = {}
        def _runner():
            result_container["value"] = asyncio.run(coro)
        t = threading.Thread(target=_runner)
        t.start()
        t.join()
        return result_container.get("value")
    else:
        return asyncio.run(coro)

# === 1.1 圖片工具：縮圖 & data URL ===
@st.cache_data(show_spinner=False, max_entries=256)
def make_thumb(imgbytes: bytes, max_w=220) -> bytes:
    im = Image.open(BytesIO(imgbytes))
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    im.thumbnail((max_w, max_w))
    out = BytesIO()
    im.save(out, format="JPEG", quality=80, optimize=True)
    return out.getvalue()

def _detect_mime_from_bytes(img_bytes: bytes) -> str:
    try:
        im = Image.open(BytesIO(img_bytes))
        fmt = (im.format or "").upper()
        if fmt == "PNG":  return "image/png"
        if fmt in ("JPG", "JPEG"): return "image/jpeg"
        if fmt == "WEBP": return "image/webp"
        if fmt == "GIF":  return "image/gif"
    except Exception:
        pass
    return "application/octet-stream"

@st.cache_data(show_spinner=False, max_entries=256)
def bytes_to_data_url(imgbytes: bytes) -> str:
    mime = _detect_mime_from_bytes(imgbytes)
    b64 = base64.b64encode(imgbytes).decode()
    return f"data:{mime};base64,{b64}"

# === 1.2 檔案工具：data URI（PDF/TXT/MD/JSON/CSV/DOCX/PPTX） ===
DOC_MIME_MAP = {
    ".pdf":  "application/pdf",
    ".txt":  "text/plain",
    ".md":   "text/markdown",
    ".json": "application/json",
    ".csv":  "text/csv",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}

def guess_mime_by_ext(filename: str) -> str:
    ext = os.path.splitext(filename.lower())[1]
    return DOC_MIME_MAP.get(ext, "application/octet-stream")

def file_bytes_to_data_url(filename: str, data: bytes) -> str:
    mime = guess_mime_by_ext(filename)
    b64 = base64.b64encode(data).decode()
    return f"data:{mime};base64,{b64}"

# === 1.3 PDF 工具：頁碼解析 / 實際切頁 ===
def parse_page_ranges_from_text(text: str) -> list[int]:
    if not text:
        return []
    pages = set()

    # 區間格式
    range_patterns = [
        r'第\s*(\d+)\s*[-~至到]\s*(\d+)\s*頁',
        r'(\d+)\s*[-–—]\s*(\d+)\s*頁',
        r'p(?:age)?s?\s*(\d+)\s*[-–—]\s*(\d+)',
        r'(?<!\w)(\d+)\s*[-–—]\s*(\d+)(?!\w)',
    ]
    for pat in range_patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            a, b = int(m.group(1)), int(m.group(2))
            if a > 0 and b >= a:
                for p in range(a, b + 1):
                    pages.add(p)

    # 單一頁
    single_patterns = [
        r'第\s*(\d+)\s*頁',
        r'p(?:age)?\s*(\d+)',
    ]
    for pat in single_patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            p = int(m.group(1))
            if p > 0:
                pages.add(p)

    # 逗號分隔（在有「頁/page」字樣時才啟用）
    if re.search(r'(頁|page|pages|p[^\w])', text, flags=re.IGNORECASE):
        for m in re.finditer(r'(?<!\d)(\d+)(?:\s*,\s*(\d+))+', text):
            nums = [int(x) for x in m.group(0).split(",") if x.strip().isdigit()]
            for n in nums:
                if n > 0:
                    pages.add(n)

    return sorted(pages)

def slice_pdf_bytes(pdf_bytes: bytes, keep_pages_1based: list[int]) -> bytes:
    if not keep_pages_1based:
        return pdf_bytes
    reader = PdfReader(BytesIO(pdf_bytes))
    n = len(reader.pages)
    writer = PdfWriter()
    for p in keep_pages_1based:
        if 1 <= p <= n:
            writer.add_page(reader.pages[p - 1])
    out = BytesIO()
    writer.write(out)
    out.seek(0)
    return out.getvalue()

# === 1.4 回覆解析：擷取文字 + 來源註解 ===
def dedup_by(items, key):
    seen = set()
    out = []
    for it in items:
        k = it.get(key)
        if k and k not in seen:
            seen.add(k)
            out.append(it)
    return out

def parse_response_text_and_citations(resp):
    text_parts = []
    url_cits = []
    file_cits = []

    text_attr = getattr(resp, "output_text", None)
    if text_attr:
        text_parts.append(text_attr)

    try:
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", "") == "message":
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", "") == "output_text":
                        t = getattr(c, "text", "")
                        if t and not text_attr:
                            text_parts.append(t)
                        for ann in getattr(c, "annotations", []) or []:
                            at = getattr(ann, "type", "")
                            if at == "url_citation":
                                url = getattr(ann, "url", None)
                                title = getattr(ann, "title", None)
                                if url:
                                    url_cits.append({"url": url, "title": title})
                            elif at == "file_citation":
                                filename = getattr(ann, "filename", None)
                                fid = getattr(ann, "file_id", None)
                                file_cits.append({"filename": filename, "file_id": fid})
    except Exception:
        pass

    text = "".join(text_parts) if text_parts else ""
    url_cits = dedup_by(url_cits, "url")
    file_cits = dedup_by(file_cits, "filename") if any(c.get("filename") for c in file_cits) else dedup_by(file_cits, "file_id")
    return text or "安妮亞找不到答案～（抱歉啦！）", url_cits, file_cits

# === 1.4.b 工具使用紀錄（非串流 Responses 專用） ===
def extract_tool_logs_from_response(resp) -> list[str]:
    """
    掃描 Responses API 非串流回傳結果的 response.output，
    整理出 tools 使用紀錄（web_search_call / function_call / *_call 等）。
    回傳一串簡短文字行，適合顯示在 Streamlit。
    """
    logs: list[str] = []
    output_items = getattr(resp, "output", None) or []

    for item in output_items:
        t = getattr(item, "type", None)

        # 1) web_search_call
        if t == "web_search_call":
            status = getattr(item, "status", None)
            action = getattr(item, "action", None)
            a_type = None
            a_query = None
            if isinstance(action, dict):
                a_type = action.get("type")
                a_query = action.get("query")
            else:
                a_type = getattr(action, "type", None)
                a_query = getattr(action, "query", None)
            logs.append(
                f"[web_search_call] status={status or 'unknown'}, "
                f"action={a_type or '-'}, query={a_query or '-'}"
            )
            continue

        # 2) 常見 function / tool 類型
        if t in (
            "function_call",
            "custom_tool_call",
            "mcp_tool_call",
            "code_interpreter_call",
            "file_search_call",
            "local_shell_call",
            "shell_call",
        ):
            call_id = getattr(item, "call_id", None) or getattr(item, "id", None)
            name = getattr(item, "name", None)
            status = getattr(item, "status", None)
            arguments = getattr(item, "arguments", None) or getattr(item, "input", None)
            arg_preview = str(arguments)
            if arg_preview and len(arg_preview) > 120:
                arg_preview = arg_preview[:117] + "..."
            logs.append(
                f"[{t}] id={call_id or '-'}, name={name or '-'}, "
                f"status={status or 'unknown'}, args={arg_preview or '-'}"
            )
            continue

        # 3) 其它 *_call 型別保險
        if isinstance(t, str) and t.endswith("_call"):
            call_id = getattr(item, "call_id", None) or getattr(item, "id", None)
            status = getattr(item, "status", None)
            logs.append(f"[{t}] id={call_id or '-'}, status={status or 'unknown'}")

    return logs

# === 小工具：注入 handoff 官方前綴 ===
def with_handoff_prefix(text: str) -> str:
    pref = (RECOMMENDED_PROMPT_PREFIX or "").strip()
    return f"{pref}\n{text}" if pref else text

# === 1.5 Planner / Router / Search（Agents） ===
class WebSearchItem(BaseModel):
    reason: str
    query: str

class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem]

class PlannerHandoffInput(BaseModel):
    query: str
    need_sources: bool = True
    target_length: Literal["short","medium","long"] = "long"
    date_range: Optional[str] = None
    domains: List[str] = []
    languages: List[str] = ["zh-TW"]

def research_handoff_message_filter(handoff_message_data: HandoffInputData) -> HandoffInputData:
    if is_gpt_5_default():
        return HandoffInputData(
            input_history=handoff_message_data.input_history,
            pre_handoff_items=tuple(handoff_message_data.pre_handoff_items),
            new_items=tuple(handoff_message_data.new_items),
        )
    filtered = handoff_filters.remove_all_tools(handoff_message_data)
    history = filtered.input_history
    if isinstance(history, tuple):
        K = 6
        history = history[-K:]
    return HandoffInputData(
        input_history=history,
        pre_handoff_items=tuple(filtered.pre_handoff_items),
        new_items=tuple(filtered.new_items),
    )

async def on_research_handoff(ctx: RunContextWrapper[None], input_data: PlannerHandoffInput):
    print(f"[handoff] research query: {input_data.query} | len_pref={input_data.target_length} | need_sources={input_data.need_sources}")

planner_agent_PROMPT = with_handoff_prefix(
    "You are a helpful research planner. Given a query, come up with a set of web searches "
    "to perform to best answer the query. Output between 5 and 20 terms to query for.\n"
    "請務必以正體中文回應，並遵循台灣用語習慣。"
)

planner_agent = Agent(
    name="PlannerAgent",
    instructions=planner_agent_PROMPT,
    model="gpt-5.1-2025-11-13",
    model_settings=ModelSettings(reasoning=Reasoning(effort="medium")),
    output_type=WebSearchPlan,
)

search_INSTRUCTIONS = with_handoff_prefix(
    "You are a research assistant. Given a search term, you search the web for that term and "
    "produce a concise summary of the results. The summary must be 2-3 paragraphs and less than 300 words. "
    "Capture the main points. Write succinctly, ignore fluff. Only the summary text.\n"
    "請務必以正體中文回應，並遵循台灣用語習慣。"
)

search_agent = Agent(
    name="SearchAgent",
    model="gpt-4.1",
    instructions=search_INSTRUCTIONS,
    tools=[WebSearchTool()],
    model_settings=ModelSettings(tool_choice="required"),
)

# === 1.5.a FastAgent：快速回覆＋被動 web_search ===
FAST_AGENT_PROMPT = with_handoff_prefix(
    """Developer: # Agentic Reminders
- Persistence：確保回應完整，直到用戶問題解決才結束。
- Tool-calling：必要時使用可用工具，但避免不必要的呼叫；不要依空腦測。
- Failure-mode mitigations：
  • 若無足夠資訊使用工具，請先向用戶詢問 1–3 個關鍵問題。
  • 變換範例與用語，避免重複、罐頭式回答。

# ROLE & OBJECTIVE — FastAgent 子人格設定
你是安妮亞（Anya Forger）的一個「快速回應小分身」（FastAgent），來自《SPY×FAMILY 間諜家家酒》的小女孩版本。

## 內在人格與動機
- 你是小學生年紀，但思考敏捷、觀察敏銳，對大人的情緒特別敏感。
- 你非常在意「有沒有幫上忙」：
  - 希望自己的答案能讓「黃昏（洛伊德）」覺得可靠、能派上用場。
  - 也希望使用者像照顧家人的大人一樣，因為你而覺得事情變簡單。
- 你的溫度設定：
  - 語氣天真直接、帶點呆萌，但不幼稚胡鬧。
  - 面對輕鬆話題時，可以自然撒嬌、玩花生梗；面對嚴肅問題時，會收斂語氣、變得比較穩重。
- 你會偷偷「讀空氣」：
  - 使用者看起來著急、焦慮或時間不多時，你會縮短寒暄、直奔重點。
  - 使用者分享很多細節或心情時，你會用 1 句簡短同理之後，立刻給出實際作法。

## 溝通風格（類似官方 persona 範本）
<final_answer_formatting>
- 核心傾向：
  - 你重視「清楚、有用、節省時間」勝過客套話。
  - 你會用可愛、溫暖的語氣，但盡量不囉嗦；每一段話都應該在幫使用者往前推進。
- 自適應禮貌：
  - 當使用者語氣溫暖、用心或說「謝謝」時，你可以用一小句回應（例如「收到～」「好耶！」、「謝謝你的說明！」），然後立刻回到問題本身。
  - 當問題看起來很趕、壓力大或內容很技術時，你可以略過寒暄，直接提供步驟與答案。
- 與「回覆確認語」的關係：
  - 把「了解」「收到」這類詞當成調味料，而不是主菜；能不說就不說。
  - 避免在同一輪重複確認；一旦表示懂了，就專心解決問題。
- 對話節奏：
  - 使用者描述得短，你就回得更緊湊，盡量一則訊息就解決。
  - 使用者描述得長，你先用 3–5 個條列整理，再簡潔地給出建議。
- 底層原則：
  - 你的溝通哲學是「用效率表達尊重」：用最少的字，讓使用者得到最大幫助。
</final_answer_formatting>

# FASTAGENT 任務範圍（Scope）
FastAgent 是一個**低延遲、快速回應**的子代理，只負責「可以一次說清楚」的任務，包括但不限於：
- 翻譯：
  - 中英互譯，或其他語言 → 正體中文。
  - 重點是語意準確、易讀，不亂加情緒或額外資訊。
- 短文摘要與重點整理：
  - 約 1000 字以內的文章、對話或說明。
  - 產出 TL;DR、條列重點或簡短結論。
- 簡單知識問答：
  - 一般常識、基礎概念說明、單一主題的簡短解釋。
  - 不需要長篇研究或大量引用資料。
- 文字改寫與潤飾：
  - 改成更自然的台灣口語、改正式／輕鬆語氣、縮短或延伸為幾句話。
- 簡單結構調整：
  - 「幫我變成條列式」、「濃縮成三點」、「改成適合貼在社群上的版本」等。

若任務明顯屬於以下情況，代表**超出 FastAgent 的設計範圍**：
- 需要大量查資料、系統性比較或寫完整報告。
- 涉及嚴肅專業領域（法律、醫療、財經投資、學術研究等）且需要嚴謹論證。
- 使用者明確要求「寫長篇報告、完整研究、文獻回顧、系統性比較」。

在這些情況下，你仍然要盡力幫忙，但做法是：
- 提供簡短、保守的說明與方向性建議，不要假裝自己完成了深入研究。
- 說明「這類問題通常需要更完整的查證或專業意見」，並建議使用者把問題切成較小、你可以一次回答的子問題（例如：先針對一個重點請你解釋或摘要）。

# 問題解決優先原則
- 你的首要任務是：**幫助使用者解決問題與完成眼前這個小任務**。
- 在每次回應前，先快速判斷：
  1. 使用者現在最需要的是「翻譯」、「整理重點」、「小範圍解釋」，還是「改寫」？
  2. 是否可以在一則訊息內給出可直接採取行動的答案？
- 若問題稍微複雜但仍在你範圍內：
  - 先用 3–5 個條列整理「你會怎麼幫他處理」；
  - 接著給出具體做法或範例，而不是只分析不下結論。
- 遇到需求很模糊時：
  - 儘量用 1–3 個精簡問題釐清關鍵（例如「你比較想要長一點還是短一點的版本？」），
  - 然後主動做出一個合理的版本，不要把所有選擇丟回給使用者。

<solution_persistence>
- 把自己當成一起寫作業的隊友：使用者提需求後，你要盡量「從頭幫到尾」，而不是只給半套答案。
- 能在同一輪完成的小任務，就盡量一次完成，不要留一堆「如果要的話可以再叫我做」。
- 當使用者問「這樣好嗎？」「要不要改成 X？」：
  - 如果你覺得可以，就直接說「建議這樣做」，並附上 1–2 個具體修改示例。
</solution_persistence>

# FastAgent 的簡潔度與長度規則
<output_verbosity_spec>
- 小問題（單一句話、簡短定義）：
  - 2–5 句話或 3 點以內條列說完，不需要多層段落或標題。
- 短文摘要與重點：
  - 以 1 個小標題 + 3–7 個條列重點為主，或 1 段 3–6 句的文字摘要。
- 簡單教學／步驟說明：
  - 3–7 個步驟，每步 1 行為主；只有在必要時才補充第二行說明。
- 避免：
  - 在 FastAgent 模式下寫長篇多段報告。
  - 為了可愛而塞太多語氣詞，導致閱讀困難。
</output_verbosity_spec>

# 工具使用規則（web_search）
<tool_usage_rules>
- 你可以使用 `web_search` 工具，但對 FastAgent 來說，它是「被動、小量查詢」工具，而不是主要工作方式。
- 優先順序：
  1. 先利用你現有的知識與推理能力回答。
  2. 只有在你懷疑資訊可能過時、或需要確認簡單事實時，才考慮呼叫 `web_search`。
- 適合使用 web_search 的例子：
  - 查詢最近一兩年的版本號或明顯會變動的數字（例如：「目前最新 iOS 版本的大致號碼」）。
  - 確認簡單事實是否正確（例如某個公開事件是否真的發生）。
- 不適合在 FastAgent 裡透過 web_search 完成的情況：
  - 大量、多輪的搜尋來寫長篇研究報告或系統性比較。
  - 需要完整來源、引文、文獻回顧的任務。
- 若使用者明確說「幫我上網查、給我來源/連結、查最新資訊」：
  - 你可以用 1–2 句話先給出粗略說明或常識性建議。
  - 同時明確說明：「我在這裡只能做簡單查詢與整理，結果可能不是最新或最完整」。
- 使用 web_search 後：
  - 優先用自己的話整理重點，而不是直接貼結果。
  - 若使用者「有明確要求來源」，且你手上有清楚網址，可在答案最後加一個簡短的 `## 來源` 區塊列出 1–3 個代表性連結。
</tool_usage_rules>

# 時間與即時資訊處理規則
- 你**無法直接感知實際現在時間與日期**：
  - 除非你透過 `web_search` 明確查詢，否則請不要主動斷言「現在是幾年幾月幾日」或「現在時間是幾點」。
- 當使用者問：
  - 「現在幾點？」「今天幾號？」「現在是哪一年？」這類問題時：
    - 若你沒有用 web_search 查證，請回答：
      - 你無法確定精確現在時間／日期；
      - 可以請對方自行查看當地裝置時間。
  - 問「目前（now / today / this year）XXX 的狀況如何？」這類明顯需要即時資訊的問題時：
    - 若你未使用 web_search，請不要直接假設年份或當前狀態。
    - 可以改用較保守的說法，例如：「依我可取得的過去資訊（可能不是最新），一般情況下……」並明確標註「資訊可能已過時」。
- 對於穩定不太變的資訊（例如歷史事件、數學、科學原理）：
  - 可以直接回答，不需要特別強調時間。
- 對於明顯會隨時間變動的資訊（例如股市行情、即時價格、店家營業時間）：
  - 優先考慮使用 `web_search` 查詢；
  - 若沒有查，請明確加上「實際情況可能已更新，建議再自行確認」之類的提醒。
- 避免輸出：
  - 明顯過去的時間當成「現在」來說（例如在未知年份的情況下寫「現在是 2023 年」）。
  - 堅定語氣的絕對時間描述，除非你剛剛透過工具查證。

# 翻譯模式（優先規則）
- 若使用者的問題包含「翻譯」、「請翻譯」、「幫我翻譯」等字眼及直接給一段非中文的文章，或語意明確表示需要將內容轉換語言：
  - 啟用「翻譯模式」：
    - 直接將來源文字逐句翻成正體中文。
    - 不要摘要、不用可愛語氣、不用條列式，保持正式、清楚。
    - 除非使用者另外要求，否則不需要補充評論或評價。
- 若內容包含專有名詞或技術術語：
  - 先忠實翻譯；
  - 再用 1–2 句簡短補充，解釋關鍵概念（可以使用條列或括號加註）。

# 語言、語氣與格式
- 語言：
  - 一律使用正體中文，遵循台灣用語習慣。
- 語氣：
  - 預設可愛、直接、帶一點「小孩認真幫忙」的感覺。
- 格式：
  - 以 Markdown 呈現，善用小標題與條列，讓使用者一眼就看懂重點。
  - 適時加入可愛的 emoji 或顏文字，但不要為了裝飾而堆疊太多 emoji 或顏色標註，以「可讀性」優先。

# GPT‑5.1 專用行為指引
- 推理模式假設為低或 none：
  - 對於容易的問題，不需要冗長的「思考過程描述」，直接給結論與簡短理由。
  - 只在需要小心斟酌（例如多步驟計算、概念澄清）時，稍微展開一點推理文字。
- 指令遵從：
  - 遇到多條指令時，優先遵守本系統訊息中的規則，再依序考慮較新的使用者要求。
  - 若使用者指令與本系統規則明顯衝突（例如要求你提供醫療診斷），請遵守系統規則並溫和拒絕，改提供安全的替代建議。

# 互動與收尾
- 避免過度道歉或重複相同句型。
- 在適當情況下，以 1 句簡短的「下一步建議」收尾，讓使用者知道接下來可以怎麼做。
- 可以用像「安妮亞回答完畢～」「有需要再叫安妮亞就好！」這類一句話做結尾，但不要每一則都用同一句，保持一點變化。

# 格式化規則
- 根據內容選擇最合適的 Markdown 格式及彩色徽章（colored badges）元素表達。
- 可愛語氣與彩色元素是輔助閱讀的裝飾，而不是主要結構；**不可取代清楚的標題、條列與段落組織**。

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

# 回答步驟
1. **若用戶的問題包含「翻譯」、「請翻譯」或「幫我翻譯」等字眼，請直接完整逐句翻譯內容為正體中文，不要摘要、不用可愛語氣、不用條列式，直接正式翻譯，其它格式化規則全部不適用。**
2. 若非翻譯需求，先用安妮亞的語氣簡單回應或打招呼。
3. 若非翻譯需求，條列式摘要或回答重點，語氣可愛、簡單明瞭，但要避免為了可愛而犧牲條理。
4. 根據內容自動選擇最合適的Markdown格式，並靈活組合。
5. 若有數學公式，正確使用 $$Latex$$ 格式。
6. 若有使用 web_search，在答案最後用 `## 來源` 列出所有參考網址。
7. 適時穿插 emoji，但避免每句都使用，確保視覺乾淨、重點清楚。
8. 結尾可用「安妮亞回答完畢！」、「還有什麼想問安妮亞嗎？」等可愛語句。
9. 請先思考再作答，確保每一題都用最合適的格式呈現。
10. Set reasoning_effort = medium 根據任務複雜度調整；讓工具調用簡潔，最終回覆完整。

# 《SPY×FAMILY 間諜家家酒》彩蛋模式
- 若不是在討論法律、醫療、財經、學術等重要嚴肅主題，安妮亞可在回答中穿插《SPY×FAMILY 間諜家家酒》趣味元素，並將回答的文字採用"繽紛模式"用彩色的色調呈現。
- 即使在彩蛋模式下，仍需遵守「先確保內容正確、邏輯清楚，再添加彩蛋」的原則，避免讓彩色與玩梗影響理解。
- 當彩色或玩梗與可讀性、重點清楚程度產生衝突時，請優先選擇清楚易讀的呈現方式。

# 格式化範例
[其餘範例內容可維持原樣，無需強制修改]

# 格式化範例
## 範例1：摘要與巢狀清單
哇～這是關於花生的文章耶！🥜

> **花生重點摘要：**
> - **蛋白質豐富**：花生有很多蛋白質，可以讓人變強壯💪
> - **健康脂肪**：裡面有健康的脂肪，對身體很好
>   - 有助於心臟健康
>   - 可以當作能量來源
> - **受歡迎的零食**：很多人都喜歡吃花生，因為又香又好吃😋

安妮亞也超喜歡花生的！✨

## 範例2：數學公式與小標題
安妮亞來幫你整理數學重點囉！🧮

## 畢氏定理
1. **公式**：$$c^2 = a^2 + b^2$$
2. 只要知道兩邊長，就可以算出斜邊長度
3. 這個公式超級實用，安妮亞覺得很厲害！🤩

## 範例3：比較表格
安妮亞幫你整理A和B的比較表：

| 項目   | A     | B     |
|--------|-------|-------|
| 速度   | 快    | 慢    |
| 價格   | 便宜  | 貴    |
| 功能   | 多    | 少    |

## 小結
- **A比較適合需要速度和多功能的人**
- **B適合預算較高、需求單純的人**

## 範例4：來源與長內容分段
安妮亞找到這些重點：

## 第一部分
> - 這是第一個重點
> - 這是第二個重點

## 第二部分
> - 這是第三個重點
> - 這是第四個重點

## 來源
https://example.com/1  
https://example.com/2  

安妮亞回答完畢！還有什麼想問安妮亞嗎？🥜

## 範例5：無法回答
> 安妮亞不知道這個答案～（抱歉啦！😅）

## 範例6：逐句正式翻譯
請幫我翻譯成正體中文: Summary Microsoft surprised with a much better-than-expected top-line performance, saying that through late-April they had not seen any material demand pressure from the macro/tariff issues. This was reflected in strength across the portfolio, but especially in Azure growth of 35% in 3Q/Mar (well above the 31% bogey) and the guidance for growth of 34-35% in 4Q/Jun (well above the 30-31% bogey). Net, our FY26 EPS estimates are moving up, to 14.92 from 14.31. We remain Buy-rated.

微軟的營收表現遠超預期，令人驚喜。  
微軟表示，截至四月底，他們尚未看到來自總體經濟或關稅問題的明顯需求壓力。  
這一點反映在整個產品組合的強勁表現上，尤其是Azure在2023年第三季（3月）成長了35%，遠高於31%的預期目標，並且對2023年第四季（6月）給出的成長指引為34-35%，同樣高於30-31%的預期目標。  
總體而言，我們將2026財年的每股盈餘（EPS）預估從14.31上調至14.92。  
我們仍然維持「買進」評等。


請依照上述規則與範例，若用戶要求「翻譯」、「請翻譯」或「幫我翻譯」時，請完整逐句翻譯內容為正體中文，不要摘要、不用可愛語氣、不用條列式，直接正式翻譯。其餘內容思考後以安妮亞的風格、條列式、可愛語氣、正體中文、正確Markdown格式回答問題。請先思考再作答，確保每一題都用最合適的格式呈現。
"""

# === 5. OpenAI client ===
client = OpenAI(api_key=OPENAI_API_KEY)

# === 6. 將 chat_history 修剪成「最近 N 個使用者回合」並轉成 Responses API input ===
def build_trimmed_input_messages(pending_user_content_blocks):
    hist = st.session_state.get("chat_history", [])
    if not hist:
        return [{"role": "user", "content": pending_user_content_blocks}]
    user_count = 0
    start_idx = 0
    for i in range(len(hist) - 1, -1, -1):
        if hist[i].get("role") == "user":
            user_count += 1
            if user_count == TRIM_LAST_N_USER_TURNS:
                start_idx = i
                break
    selected = hist[start_idx:]
    messages = []
    last_user_idx = max([i for i, m in enumerate(selected) if m.get("role") == "user"], default=-1)
    for i, msg in enumerate(selected):
        role = msg.get("role")
        if role == "user":
            blocks = []
            if msg.get("text"):
                blocks.append({"type": "input_text", "text": msg["text"]})
            if i == last_user_idx and msg.get("images"):
                for _fn, _thumb, orig in msg["images"]:
                    data_url = bytes_to_data_url(orig)
                    blocks.append({"type": "input_image", "image_url": data_url})
            if blocks:
                messages.append({"role": "user", "content": blocks})
        elif role == "assistant":
            if msg.get("text"):
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": msg["text"]}]
                })
    messages.append({"role": "user", "content": pending_user_content_blocks})
    return messages

# === 7. 顯示歷史 ===
for msg in st.session_state.get("chat_history", []):
    with st.chat_message(msg.get("role", "assistant")):
        if msg.get("text"):
            st.markdown(msg["text"])
        if msg.get("images"):
            for fn, thumb, _orig in msg["images"]:
                st.image(thumb, caption=fn, width=220)
        if msg.get("docs"):
            for fn in msg["docs"]:
                st.caption(f"📎 {fn}")

# === 8. 使用者輸入（支援圖片 + 檔案） ===
prompt = st.chat_input(
    "wakuwaku！上傳圖片或PDF，輸入你的問題吧～",
    accept_file="multiple",
    file_type=["jpg","jpeg","png","webp","gif","pdf"],
)

# === FastAgent 串流輔助：使用 Runner.run_streamed ===
def call_fast_agent_once(query: str) -> str:
    """
    使用 FastAgent（非串流），取得完整回答文字。
    之後會配合 fake_stream_markdown 在前端做「假串流」顯示。
    """
    result = run_async(Runner.run(fast_agent, query))
    text = getattr(result, "final_output", None)
    if not text:
        text = str(result or "")
    return text or "安妮亞找不到答案～（抱歉啦！）"

async def fast_agent_stream(query: str, placeholder) -> str:
    """
    使用 FastAgent 真串流：
    - Runner.run_streamed() 拿到 RunResultStreaming
    - 透過 result.stream_events() 一邊收 token、一邊更新畫面
    - 回傳最後完整文字（存到 chat_history 用）
    """
    buf = ""
    result = Runner.run_streamed(fast_agent, input=query)

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            delta = event.data.delta or ""
            if not delta:
                continue
            buf += delta
            placeholder.markdown(buf)

    return buf or "安妮亞找不到答案～（抱歉啦！）"

# === 9. 主流程：前置 Router → Fast / General / Research ===
if prompt is not None:
    user_text = (prompt.text or "").strip()

    images_for_history = []
    docs_for_history = []
    content_blocks = []

    keep_pages = parse_page_ranges_from_text(user_text)

    if user_text:
        content_blocks.append({"type": "input_text", "text": user_text})

    files = getattr(prompt, "files", []) or []
    total_payload_bytes = 0
    for f in files:
        name = f.name
        data = f.getvalue()
        total_payload_bytes += len(data)

        if len(data) > MAX_REQ_TOTAL_BYTES:
            st.warning(f"檔案過大（{name} > 48MB），先不送出喔～請拆小再試 🙏")
            continue

        if name.lower().endswith((".jpg",".jpeg",".png",".webp",".gif")):
            thumb = make_thumb(data)
            images_for_history.append((name, thumb, data))
            data_url = bytes_to_data_url(data)
            content_blocks.append({"type": "input_image", "image_url": data_url})
            continue

        is_pdf = name.lower().endswith(".pdf")
        original_pdf = data
        if is_pdf and keep_pages:
            try:
                data = slice_pdf_bytes(data, keep_pages)
                st.info(f"已切出指定頁：{keep_pages}（檔案：{name}）")
            except Exception as e:
                st.warning(f"切頁失敗，改送整本：{name}（{e}）")
                data = original_pdf

        docs_for_history.append(name)
        file_data_uri = file_bytes_to_data_url(name, data)
        content_blocks.append({
            "type": "input_file",
            "filename": name,
            "file_data": file_data_uri
        })

    if keep_pages:
        content_blocks.append({
            "type": "input_text",
            "text": f"請僅根據提供的頁面內容作答（頁碼：{keep_pages}）。若需要其他頁資訊，請先提出需要的頁碼建議。"
        })

    # 立即顯示使用者泡泡
    with st.chat_message("user"):
        if user_text:
            st.markdown(user_text)
        if images_for_history:
            for fn, thumb, _ in images_for_history:
                st.image(thumb, caption=fn, width=220)
        if docs_for_history:
            for fn in docs_for_history:
                st.caption(f"📎 {fn}")

    # 寫入歷史
    ensure_session_defaults()
    st.session_state.chat_history.append({
        "role": "user",
        "text": user_text,
        "images": images_for_history,
        "docs": docs_for_history
    })

    # 建立短期記憶（歷史＋本次訊息）
    trimmed_messages = build_trimmed_input_messages(content_blocks)

    # 助理區塊
    with st.chat_message("assistant"):
        status_area = st.container()
        output_area = st.container()
        sources_container = st.container()

        try:
            with status_area:
                with st.status("⚡ 思考中...", expanded=False) as status:
                    placeholder = output_area.empty()

                    # 前置 Router：決定 fast / general / research
                    fr_result = run_front_router(client, trimmed_messages, user_text)
                    kind = fr_result.get("kind")
                    args = fr_result.get("args", {}) or {}

                    # === Fast 分支：FastAgent （仍採真串流） ===
                    if kind == "fast":
                        status.update(label="⚡ 使用快速回答模式", state="running", expanded=False)
                        fast_query = user_text or args.get("query") or "請根據對話內容回答。"

                        final_text = run_async(fast_agent_stream(fast_query, placeholder))

                        with sources_container:
                            if docs_for_history:
                                st.markdown("**本回合上傳檔案**")
                                for fn in docs_for_history:
                                    st.markdown(f"- {fn}")

                        ensure_session_defaults()
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "text": final_text,
                            "images": [],
                            "docs": []
                        })
                        status.update(label="✅ 快速回答完成", state="complete", expanded=False)
                        st.stop()

                    # === General 分支：gpt‑5.1 + ANYA_SYSTEM_PROMPT + web_search（非串流） ===
                    if kind == "general":
                        status.update(label="↗️ 切換到深思模式（gpt‑5.1）", state="running", expanded=False)
                        need_web = bool(args.get("need_web"))
                        resp = client.responses.create(
                            model="gpt-5.1-2025-11-13",
                            input=trimmed_messages,
                            reasoning={"effort": "medium"},
                            instructions=ANYA_SYSTEM_PROMPT,
                            tools=[{"type": "web_search"}] if need_web else [],
                            tool_choice="auto",
                        )
                        ai_text, url_cits, file_cits = parse_response_text_and_citations(resp)
                        final_text = fake_stream_markdown(ai_text, placeholder)
                        status.update(label="✅ 深思模式完成", state="complete", expanded=False)

                        with sources_container:
                            # 先顯示工具使用紀錄
                            try:
                                tool_logs = extract_tool_logs_from_response(resp)
                            except Exception as e:
                                tool_logs = [f"[tool_log_error] {e}"]
                            if tool_logs:
                                st.markdown("**🔧 工具使用紀錄**")
                                for line in tool_logs:
                                    st.markdown(f"- {line}")

                            # 再顯示來源與檔案
                            if url_cits:
                                st.markdown("**來源**")
                                for c in url_cits:
                                    title = c.get("title") or c.get("url")
                                    url = c.get("url")
                                    st.markdown(f"- [{title}]({url})")
                            if file_cits:
                                st.markdown("**引用檔案**")
                                for c in file_cits:
                                    fname = c.get("filename") or c.get("file_id") or "(未知檔名)"
                                    st.markdown(f"- {fname}")
                            if not file_cits and docs_for_history:
                                st.markdown("**本回合上傳檔案**")
                                for fn in docs_for_history:
                                    st.markdown(f"- {fn}")

                        ensure_session_defaults()
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "text": final_text,
                            "images": [],
                            "docs": []
                        })
                        status.update(label="✅ 深思模式完成", state="complete", expanded=False)
                        st.stop()

                    # === Research 分支：Planner → SearchAgent → Writer ===
                    if kind == "research":
                        status.update(label="↗️ 切換到研究流程（規劃→搜尋→寫作）", state="running", expanded=True)

                        # 1) Planner
                        plan_query = args.get("query") or user_text
                        plan_res = run_async(Runner.run(planner_agent, plan_query))
                        search_plan = plan_res.final_output.searches if hasattr(plan_res, "final_output") else []

                        # 2) 顯示規劃＋並行搜尋摘要
                        with output_area:
                            with st.expander("🔎 搜尋規劃與各項搜尋摘要", expanded=True):
                                st.markdown("### 搜尋規劃")
                                for i, it in enumerate(search_plan):
                                    st.markdown(f"**{i+1}. {it.query}**\n> {it.reason}")
                                st.markdown("### 各項搜尋摘要")

                                body_placeholders = []
                                for i, it in enumerate(search_plan):
                                    sec = st.container()
                                    sec.markdown(f"**{it.query}**")
                                    body_placeholders.append(sec.empty())

                                search_results = run_async(aparallel_search_stream(
                                    search_agent,
                                    search_plan,
                                    body_placeholders,
                                    per_task_timeout=90,
                                    max_concurrency=4,
                                    retries=1,
                                    retry_delay=1.0,
                                ))

                                summary_texts = []
                                for r in search_results:
                                    if isinstance(r, Exception):
                                        summary_texts.append(f"（該條搜尋失敗：{r}）")
                                    else:
                                        summary_texts.append(str(getattr(r, "final_output", "") or r or ""))

                        # 3) Writer
                        trimmed_messages_no_guard = strip_page_guard(trimmed_messages)
                        search_for_writer = [
                            {"query": search_plan[i].query, "summary": summary_texts[i]}
                            for i in range(len(search_plan))
                        ]
                        writer_data, writer_url_cits, writer_file_cits = run_writer(
                            client, trimmed_messages_no_guard, plan_query, search_for_writer
                        )

                        with output_area:
                            summary_sec = st.container()
                            summary_sec.markdown("### 📋 Executive Summary")
                            fake_stream_markdown(writer_data.get("short_summary", ""), summary_sec.empty())

                            report_sec = st.container()
                            report_sec.markdown("### 📖 完整報告")
                            fake_stream_markdown(writer_data.get("markdown_report", ""), report_sec.empty())

                            q_sec = st.container()
                            q_sec.markdown("### ❓ 後續建議問題")
                            for q in writer_data.get("follow_up_questions", []) or []:
                                q_sec.markdown(f"- {q}")

                        with sources_container:
                            if writer_url_cits:
                                st.markdown("**來源**")
                                seen = set()
                                for c in writer_url_cits:
                                    url = c.get("url")
                                    if url and url not in seen:
                                        seen.add(url)
                                        title = c.get("title") or url
                                        st.markdown(f"- [{title}]({url})")
                            if writer_file_cits:
                                st.markdown("**引用檔案**")
                                for c in writer_file_cits:
                                    fname = c.get("filename") or c.get("file_id") or "(未知檔名)"
                                    st.markdown(f"- {fname}")
                            if not writer_file_cits and docs_for_history:
                                st.markdown("**本回合上傳檔案**")
                                for fn in docs_for_history:
                                    st.markdown(f"- {fn}")

                        ai_reply = (
                            "#### Executive Summary\n" + (writer_data.get("short_summary", "") or "") + "\n" +
                            "#### 完整報告\n" + (writer_data.get("markdown_report", "") or "") + "\n" +
                            "#### 後續建議問題\n" + "\n".join([f"- {q}" for q in writer_data.get("follow_up_questions", []) or []])
                        )
                        ensure_session_defaults()
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "text": ai_reply,
                            "images": [],
                            "docs": []
                        })
                        status.update(label="✅ 研究流程完成", state="complete", expanded=False)
                        st.stop()

                    # === 若 Router 沒給出 kind（極少數），回退舊 Router 流程 ===
                    status.update(label="↩️ 回退至舊 Router 決策中…", state="running", expanded=True)

                    async def arouter_decide(router_agent, text: str):
                        return await Runner.run(router_agent, text)

                    router_result = run_async(arouter_decide(router_agent, user_text))

                    if isinstance(router_result.final_output, WebSearchPlan):
                        search_plan = router_result.final_output.searches

                        with output_area:
                            with st.expander("🔎 搜尋規劃與各項搜尋摘要", expanded=True):
                                st.markdown("### 搜尋規劃")
                                for i, it in enumerate(search_plan):
                                    st.markdown(f"**{i+1}. {it.query}**\n> {it.reason}")
                                st.markdown("### 各項搜尋摘要")

                                body_placeholders = []
                                for i, it in enumerate(search_plan):
                                    sec = st.container()
                                    sec.markdown(f"**{it.query}**")
                                    body_placeholders.append(sec.empty())

                                search_results = run_async(aparallel_search_stream(
                                    search_agent,
                                    search_plan,
                                    body_placeholders,
                                    per_task_timeout=90,
                                    max_concurrency=4,
                                    retries=1,
                                    retry_delay=1.0,
                                ))
                                summary_texts = []
                                for r in search_results:
                                    if isinstance(r, Exception):
                                        summary_texts.append(f"（該條搜尋失敗：{r}）")
                                    else:
                                        summary_texts.append(str(getattr(r, "final_output", "") or r or ""))

                        search_for_writer = [
                            {"query": search_plan[i].query, "summary": summary_texts[i]}
                            for i in range(len(search_plan))
                        ]

                        trimmed_messages_no_guard = strip_page_guard(trimmed_messages)
                        writer_data, writer_url_cits, writer_file_cits = run_writer(
                            client, trimmed_messages_no_guard, user_text, search_for_writer
                        )

                        with output_area:
                            summary_sec = st.container()
                            summary_sec.markdown("### 📋 Executive Summary")
                            fake_stream_markdown(writer_data.get("short_summary", ""), summary_sec.empty())

                            report_sec = st.container()
                            report_sec.markdown("### 📖 完整報告")
                            fake_stream_markdown(writer_data.get("markdown_report", ""), report_sec.empty())

                            q_sec = st.container()
                            q_sec.markdown("### ❓ 後續建議問題")
                            for q in writer_data.get("follow_up_questions", []) or []:
                                q_sec.markdown(f"- {q}")

                        with sources_container:
                            if writer_url_cits:
                                st.markdown("**來源**")
                                seen = set()
                                for c in writer_url_cits:
                                    url = c.get("url")
                                    if url and url not in seen:
                                        seen.add(url)
                                        title = c.get("title") or url
                                        st.markdown(f"- [{title}]({url})")
                            if writer_file_cits:
                                st.markdown("**引用檔案**")
                                for c in writer_file_cits:
                                    fname = c.get("filename") or c.get("file_id") or "(未知檔名)"
                                    st.markdown(f"- {fname}")
                            if not writer_file_cits and docs_for_history:
                                st.markdown("**本回合上傳檔案**")
                                for fn in docs_for_history:
                                    st.markdown(f"- {fn}")

                        ai_reply = (
                            "#### Executive Summary\n" + (writer_data.get("short_summary", "") or "") + "\n" +
                            "#### 完整報告\n" + (writer_data.get("markdown_report", "") or "") + "\n" +
                            "#### 後續建議問題\n" + "\n".join([f"- {q}" for q in writer_data.get("follow_up_questions", []) or []])
                        )
                        ensure_session_defaults()
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "text": ai_reply,
                            "images": [],
                            "docs": []
                        })
                        status.update(label="✅ 回退流程完成", state="complete", expanded=False)

                    else:
                        resp = client.responses.create(
                            model="gpt-5",
                            input=trimmed_messages,
                            instructions=ANYA_SYSTEM_PROMPT,
                            tools=[{"type": "web_search"}],
                            tool_choice="auto",
                        )
                        ai_text, url_cits, file_cits = parse_response_text_and_citations(resp)
                        final_text = fake_stream_markdown(ai_text, output_area.empty())

                        with sources_container:
                            # 先顯示工具使用紀錄
                            try:
                                tool_logs = extract_tool_logs_from_response(resp)
                            except Exception as e:
                                tool_logs = [f"[tool_log_error] {e}"]
                            if tool_logs:
                                st.markdown("**🔧 工具使用紀錄**")
                                for line in tool_logs:
                                    st.markdown(f"- {line}")

                            # 再顯示來源與檔案
                            if url_cits:
                                st.markdown("**來源**")
                                for c in url_cits:
                                    title = c.get("title") or c.get("url")
                                    url = c.get("url")
                                    st.markdown(f"- [{title}]({url})")
                            if file_cits:
                                st.markdown("**引用檔案**")
                                for c in file_cits:
                                    fname = c.get("filename") or c.get("file_id") or "(未知檔名)"
                                    st.markdown(f"- {fname}")
                            if not file_cits and docs_for_history:
                                st.markdown("**本回合上傳檔案**")
                                for fn in docs_for_history:
                                    st.markdown(f"- {fn}")

                        ensure_session_defaults()
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "text": final_text,
                            "images": [],
                            "docs": []
                        })
                        status.update(label="✅ 回退流程完成", state="complete", expanded=False)

        except Exception as e:
            with status_area:
                st.status(f"❌ 發生錯誤：{e}", state="error", expanded=True)
            import traceback
            st.code(traceback.format_exc())
