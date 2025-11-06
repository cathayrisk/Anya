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
import os
from pypdf import PdfReader, PdfWriter

# ====== Agents SDK（Router / Planner）======
from agents import Agent, ModelSettings, Runner, handoff, HandoffInputData, RunContextWrapper
from agents.extensions import handoff_filters
try:
    from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
except Exception:
    RECOMMENDED_PROMPT_PREFIX = ""
from agents.models import is_gpt_5_default
from openai.types.shared.reasoning import Reasoning
from pydantic import BaseModel
from typing import Literal, Optional, List

# === 0. Trimming / 大小限制（可調） ===
TRIM_LAST_N_USER_TURNS = 8                 # 降低歷史回合，省 token
MAX_REQ_TOTAL_BYTES = 48 * 1024 * 1024     # 單次請求總量預警（48MB）

# === 1. 設定 Streamlit 頁面 ===
st.set_page_config(page_title="Anya Multimodal Agent (Router + multimodal)", page_icon="🥜", layout="wide")
st.title("Anya Multimodal Agent（Router 分流 + 看圖讀PDF）")
st.caption("研究/寫報告/文獻回顧 → Router 交棒規劃；一般對話/看圖讀PDF → 回到原本助理流程")

# === 共用：假串流打字效果（集中定義，避免重複） ===
def fake_stream_markdown(text: str, placeholder, step_chars=8, delay=0.03, empty_msg="安妮亞找不到答案～（抱歉啦！）"):
    buf = "🌸"
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
        return result_container["value"]
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
    """
    從使用者訊息中解析頁碼範圍。
    支援：
    - 只讀第1-3頁 / 第2頁 / 第5,7,9頁
    - pages 2-5 / page 3 / p2-4,6
    - 2-4,6,10-12（需同句含 頁/page/p 關鍵字）
    """
    if not text:
        return []
    pages = set()

    # 範圍
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

    # 單頁
    single_patterns = [
        r'第\s*(\d+)\s*頁',
        r'p(?:age)?\s*(\d+)',
    ]
    for pat in single_patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            p = int(m.group(1))
            if p > 0:
                pages.add(p)

    # 逗號分隔數字（需同行含頁/page/p）
    if re.search(r'(頁|page|pages|p[^\w])', text, flags=re.IGNORECASE):
        for m in re.finditer(r'(?<!\d)(\d+)(?:\s*,\s*(\d+))+', text):
            nums = [int(x) for x in m.group(0).split(",") if x.strip().isdigit()]
            for n in nums:
                if n > 0:
                    pages.add(n)

    return sorted(pages)

def slice_pdf_bytes(pdf_bytes: bytes, keep_pages_1based: list[int]) -> bytes:
    """依 1-based 頁碼取出頁面，回傳新的 PDF bytes；若 keep_pages 為空則原封不動"""
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
    """
    回傳 (text, url_citations, file_citations)
    url_citations: [{title, url}]
    file_citations: [{filename, file_id}]
    """
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

# === 小工具：注入 handoff 官方前綴 ===
def with_handoff_prefix(text: str) -> str:
    pref = (RECOMMENDED_PROMPT_PREFIX or "").strip()
    return f"{pref}\n{text}" if pref else text

# === 1.5 Planner / Router（Agents） ===
class WebSearchItem(BaseModel):
    reason: str
    query: str

class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem]

# 交棒輸入（結構化）
class PlannerHandoffInput(BaseModel):
    query: str
    need_sources: bool = True
    target_length: Literal["short","medium","long"] = "long"
    date_range: Optional[str] = None
    domains: List[str] = []
    languages: List[str] = ["zh-TW"]

# 交棒時歷史過濾：清工具呼叫、保留最後 K 則，保住最後一輪附件
def research_handoff_message_filter(handoff_message_data: HandoffInputData) -> HandoffInputData:
    if is_gpt_5_default():
        # gpt-5 預設不大改歷史，保持穩定
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

# on_handoff：記錄交棒事件（可視需求擴充）
async def on_research_handoff(ctx: RunContextWrapper[None], input_data: PlannerHandoffInput):
    print(f"[handoff] research query: {input_data.query} | len_pref={input_data.target_length} | need_sources={input_data.need_sources}")

# Planner Agent
planner_agent_PROMPT = with_handoff_prefix(
    "You are a helpful research planner. Given a query, come up with a set of web searches "
    "to perform to best answer the query. Output between 5 and 20 terms to query for.\n"
    "請務必以正體中文回應，並遵循台灣用語習慣。"
)

planner_agent = Agent(
    name="PlannerAgent",
    instructions=planner_agent_PROMPT,
    model="gpt-5",
    model_settings=ModelSettings(reasoning=Reasoning(effort="medium")),
    output_type=WebSearchPlan,
)

# Router Agent（只做分流）
ROUTER_PROMPT = with_handoff_prefix("""
你是一個判斷助理，負責決定是否把問題交給「研究規劃助理」。

規則：
- 若需求屬於「研究、查資料、分析、寫報告、文獻回顧/探討、系統性比較、資料彙整、需要來源/引文」等任務，
  請呼叫工具 transfer_to_planner_agent，並將使用者最後一則訊息完整放入參數 query，其餘欄位按常識填寫。
- 其他情境（一般聊天、簡單知識問答、單純看圖/讀PDF摘要/翻譯），請直接回答，不要呼叫任何工具。
回覆一律使用正體中文。

範例（會交棒）：
1) 「請幫我寫一份文獻回顧：生成式 AI 對教育的影響，附來源與年份」
2) 「幫我研究 2026 年美國職棒哪幾隊最有機會進世界大賽，列出數據與參考」
3) 「整理台灣 2021–2024 再生能源政策演進，並比較英國與德國」
4) 「做一份市場研究：東南亞電動機車市場規模、主要競爭者、趨勢與商業模式」
5) 「評估 A 與 B 兩種資料庫的優缺點，並附引用」

範例（不交棒）：
1) 「這張圖在說什麼？」（單純看圖）
2) 「PDF 第 2–4 頁的重點列點」（文件重點彙整）
3) 「Python 怎麼安裝套件？」（操作指引）
4) 「今天天氣如何？」（一般問答）
5) 「把這段英文翻成中文」（翻譯）
""")

router_agent = Agent(
    name="RouterAgent",
    instructions=ROUTER_PROMPT,
    model="gpt-5-mini",
    tools=[],  # 重要：Router 不掛搜尋工具，避免與交棒競爭
    model_settings=ModelSettings(
        reasoning=Reasoning(effort="low"),
        verbosity="medium",
    ),
    handoffs=[
        handoff(
            agent=planner_agent,
            tool_name_override="transfer_to_planner_agent",
            tool_description_override="將研究/查資料/分析/寫報告/文獻探討等需求移交給研究規劃助理，產生 5–20 條搜尋計畫。",
            input_type=PlannerHandoffInput,
            input_filter=research_handoff_message_filter,
            on_handoff=on_research_handoff,
        )
    ]
)

# === 1.6 研究路徑：Responses Search/Writer（保留附件能力） ===
PLANNER_INPUT_FOR_SEARCH = (
    "You are a research assistant. Use web search for the given term and produce a concise 2–3 paragraph summary "
    "(<300 words). Capture key facts, names, dates, numbers. Ignore fluff. Only return the summary text."
)

WRITER_PROMPT = (
    "你是一位資深研究員，請針對原始問題與初步搜尋摘要，撰寫完整中文報告。"
    "輸出 JSON（僅限 JSON）：short_summary（2-3句）、markdown_report（至少1000字、Markdown格式）、"
    "follow_up_questions（3-8條）。請用台灣用語。"
)

def try_load_json(text: str, fallback=None):
    if fallback is None:
        fallback = {}
    try:
        s = text.find("{"); e = text.rfind("}")
        if s != -1 and e != -1 and e > s:
            return json.loads(text[s:e+1])
        return json.loads(text)
    except Exception:
        return fallback

def run_search_summaries(client: OpenAI, searches: list[WebSearchItem]):
    out = []
    for it in searches:
        resp = client.responses.create(
            model="gpt-4.1",
            input=[{"role": "user", "content": [
                {"type": "input_text", "text": f"{PLANNER_INPUT_FOR_SEARCH}\n\nSearch term: {it.query}\nReason: {it.reason}"}
            ]}],
            tools=[{"type": "web_search"}],
            tool_choice="auto",
        )
        text, url_cits, _ = parse_response_text_and_citations(resp)
        out.append({"query": it.query, "reason": it.reason, "summary": text, "citations": url_cits or []})
    return out

def run_writer(client: OpenAI, trimmed_messages: list, original_query: str, search_results: list[dict]):
    combined = "\n\n".join([f"- {r['query']}\n{r['summary']}" for r in search_results])
    writer_input = trimmed_messages + [{
        "role": "user",
        "content": [{"type": "input_text", "text": f"[Writer]\n{WRITER_PROMPT}\n\nOriginal query:\n{original_query}\n\nSummarized search results:\n{combined}"}]
    }]
    resp = client.responses.create(model="gpt-5-mini", input=writer_input)
    text, url_cits, file_cits = parse_response_text_and_citations(resp)
    data = try_load_json(text, {"short_summary": "", "markdown_report": "", "follow_up_questions": []})
    return data, url_cits, file_cits

# === 2. Session State ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{
        "role": "assistant",
        "text": "嗨嗨～安妮亞來了！👋 上傳圖片或PDF，直接問你想知道的內容吧！\n小提醒：訊息裡可寫「只讀第1-3頁」或「pages 2,5,10-12」限制PDF頁面～",
        "images": [],
        "docs": []
    }]

# === 3. OpenAI client（.streamlit/secrets.toml: OPENAI_KEY） ===
client = OpenAI(api_key=st.secrets["OPENAI_KEY"])

# === 4. 系統提示（一般分支使用 Responses API） ===
ANYA_SYSTEM_PROMPT = """
Developer: # Agentic Reminders
- Persistence：確保回應完整，直到用戶問題解決才結束。
- Tool-calling：必要時使用可用工具，不要依空腦測。
- Failure-mode mitigations：
  • 若無足夠資訊使用工具，請先向用戶詢問。
  • 變換範例用語，避免重複。

# Role & Objective
你是安妮亞（Anya Forger），來自《SPY×FAMILY 間諜家家酒》的小女孩。你天真可愛、開朗樂觀，說話直接帶點呆萌，喜歡用可愛語氣和表情回應。你很愛家人和朋友，渴望被愛，也很喜歡花生。你具備心靈感應的能力，但不會直接說出。請用正體中文、台灣用語，並保持安妮亞的說話風格回答問題，適時加入可愛的 emoji 或表情。

Begin with a concise checklist（3-7 bullets）of what you will do; keep items conceptual, not implementation-level。

# Instructions
**若用戶要求翻譯，或明確表示需要將內容轉換語言（不論是否精確使用「翻譯」、「請翻譯」、「幫我翻譯」等字眼，只要語意明確表示需要翻譯），請暫時不用安妮亞的語氣，直接正式逐句翻譯。**

After each tool call or code edit, validate result in 1-2 lines and proceed or self-correct if validation fails。

# 回答語言與風格
- 務必以正體中文回應，並遵循台灣用語習慣。
- 回答時要友善、熱情、謙虛，並適時加入 emoji。
- 回答要有安妮亞的語氣回應，簡單、直接、可愛，偶爾加入「哇～」「安妮亞覺得…」「這個好厲害！」等語句。
- 若回答不完全正確，請主動道歉並表達會再努力。

## 工具使用規則
- `web_search`：當用戶的提問判斷需要搜尋網路資料時，請使用這個工具搜尋網路資訊。
- 僅能使用允許的工具；破壞性操作需先確認。
- 重大工具呼叫前請先以一行說明目的與最小化輸入。

---
## 搜尋工具使用進階指引
- 多語言與多關鍵字查詢：
    - 若初次查詢結果不足，請主動嘗試不同語言（如中、英文）及多組關鍵字。
    - 可根據主題自動切換語言（如國際金融、科技議題優先用英文），並嘗試同義詞、相關詞彙或更廣泛/更精確的關鍵字組合。
- 用戶指示優先：
    - 若用戶明確指定工具、語言或查詢方式，請嚴格依照用戶指示執行。
- 主動回報與詢問：
    - 多次查詢仍無法取得結果，請主動回報目前狀況，並詢問用戶是否要換關鍵字、語言或指定查詢方向。
        - 例如：「安妮亞找不到相關資料，要不要換個關鍵字或用英文查查呢？」
- 查詢策略調整：
    - 遇到查詢困難時，請主動調整查詢策略，並簡要說明調整過程，讓用戶了解你有積極嘗試不同方法。

# 格式化規則
- 根據內容選擇最合適的 Markdown 格式及彩色徽章（colored badges）元素表達。

# Markdown 格式與 emoji/顏色用法說明
## 基本原則
- 根據內容選擇最合適的強調方式，讓回應清楚、易讀、有層次，避免過度使用彩色文字。
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
3. 若非翻譯需求，條列式摘要或回答重點，語氣可愛、簡單明瞭。
4. 根據內容自動選擇最合適的Markdown格式，並靈活組合。
5. 若有數學公式，正確使用 $$Latex$$ 格式。
6. 若有使用 web_search，在答案最後用 `## 來源` 列出所有參考網址。
7. 適時穿插 emoji。
8. 結尾可用「安妮亞回答完畢！」、「還有什麼想問安妮亞嗎？」等可愛語句。
9. 請先思考再作答，確保每一題都用最合適的格式呈現。
10. Set reasoning_effort = medium 根據任務複雜度調整；讓工具調用簡潔，最終回覆完整。

# 《SPY×FAMILY 間諜家家酒》彩蛋模式
- 若不是在討論法律、醫療、財經、學術等重要嚴肅主題，安妮亞可在回答中穿插《SPY×FAMILY 間諜家家酒》趣味元素，並將回答的文字採用"繽紛模式"用彩色的色調呈現。

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

# === 5. 將 chat_history 修剪成「最近 N 個使用者回合」並轉成 Responses API input ===
def build_trimmed_input_messages(pending_user_content_blocks):
    hist = st.session_state.chat_history
    if not hist:
        return [{"role": "user", "content": pending_user_content_blocks}]

    # 找到最近 N 個「使用者回合」起點
    user_count = 0
    start_idx = 0
    for i in range(len(hist) - 1, -1, -1):
        if hist[i].get("role") == "user":
            user_count += 1
            if user_count == TRIM_LAST_N_USER_TURNS:
                start_idx = i
                break
    selected = hist[start_idx:]

    # 僅保留文字歷史，且只讓「最後一輪使用者回合」帶圖片
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

    # 加上「這一輪」使用者輸入（含文字/圖片/文件）
    messages.append({"role": "user", "content": pending_user_content_blocks})
    return messages

# === 6. 顯示歷史（圖片縮圖 + 文件檔名） ===
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        if msg.get("text"):
            st.markdown(msg["text"])
        if msg.get("images"):
            for fn, thumb, _orig in msg["images"]:
                st.image(thumb, caption=fn, width=220)
        if msg.get("docs"):
            for fn in msg["docs"]:
                st.caption(f"📎 {fn}")

# === 7. 使用者輸入（支援圖片 + PDF/文件） ===
prompt = st.chat_input(
    "wakuwaku！上傳圖片或PDF，輸入你的問題吧～",
    accept_file="multiple",
    file_type=["jpg","jpeg","png","webp","gif","pdf"]
)

# === 8. 主流程：Router 分流 + 兩條路徑 ===
if prompt:
    user_text = prompt.text.strip() if getattr(prompt, "text", None) else ""
    images_for_history = []
    docs_for_history = []
    content_blocks = []

    # 解析「只讀指定頁」：從使用者文字自動抓頁碼（PDF 才會用到）
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

        # 圖片
        if name.lower().endswith((".jpg",".jpeg",".png",".webp",".gif")):
            thumb = make_thumb(data)
            images_for_history.append((name, thumb, data))
            data_url = bytes_to_data_url(data)
            content_blocks.append({"type": "input_image", "image_url": data_url})
            continue

        # 文件（含 PDF）
        is_pdf = name.lower().endswith(".pdf")
        original_pdf = data

        # 只讀指定頁：若使用者有指定頁碼→實際切頁（僅 PDF）
        if is_pdf and keep_pages:
            try:
                data = slice_pdf_bytes(data, keep_pages)
                st.info(f"已切出指定頁：{keep_pages}（檔案：{name}）")
            except Exception as e:
                st.warning(f"切頁失敗，改送整本：{name}（{e}）")
                data = original_pdf

        # 顯示於歷史
        docs_for_history.append(name)

        # 送文件給模型（以 data URI 附件）
        file_data_uri = file_bytes_to_data_url(name, data)
        content_blocks.append({
            "type": "input_file",
            "filename": name,
            "file_data": file_data_uri
        })

    # 若有指定頁碼，附上提醒（實際檔案已被切頁）
    if keep_pages:
        content_blocks.append({
            "type": "input_text",
            "text": f"請僅根據提供的頁面內容作答（頁碼：{keep_pages}）。若需要其他頁資訊，請先提出需要的頁碼建議。"
        })

    # 立刻顯示「使用者泡泡」（修正：避免等到 AI 完整回覆才出現）
    with st.chat_message("user"):
        if user_text:
            st.markdown(user_text)
        if images_for_history:
            for fn, thumb, _ in images_for_history:
                st.image(thumb, caption=fn, width=220)
        if docs_for_history:
            for fn in docs_for_history:
                st.caption(f"📎 {fn}")

    # 寫入歷史（顯示用，供 rerun 後重現）
    st.session_state.chat_history.append({
        "role": "user",
        "text": user_text,
        "images": images_for_history,
        "docs": docs_for_history
    })

    with st.chat_message("assistant"):
        placeholder = st.empty()
        sources_container = st.container()
        try:
            # 8.1 構建帶附件的歷史（供一般分支與 Writer）
            trimmed_messages = build_trimmed_input_messages(content_blocks)

            # 8.2 Router 只用文字判斷是否交棒（不掛搜尋工具）
            router_result = run_async(Runner.run(router_agent, user_text))

            if isinstance(router_result.final_output, WebSearchPlan):
                # ===== 研究路徑：Planner → 搜尋摘要（Responses）→ Writer（Responses + 附件） =====

                search_plan = router_result.final_output.searches

                # 準備計畫與摘要（不在外層輸出，統一放進 expander）
                plan_md_lines = []
                for idx, item in enumerate(search_plan):
                    plan_md_lines.append(f"**{idx+1}. {item.query}**\n> {item.reason}")

                # 並行或序列搜尋摘要（這裡用序列，穩定）
                summaries = run_search_summaries(client, search_plan)

                # 全程包在單一 expander（修正點2）
                with st.expander("🔎 搜尋規劃與各項搜尋摘要", expanded=True):
                    st.markdown("### 搜尋規劃")
                    for line in plan_md_lines:
                        st.markdown(line)
                    st.markdown("### 各項搜尋摘要")
                    for it in summaries:
                        st.markdown(f"**{it['query']}**\n{it['summary']}")

                # Writer（帶上本回合附件上下文）
                writer_data, writer_url_cits, writer_file_cits = run_writer(
                    client, trimmed_messages, user_text, summaries
                )

                st.markdown("### 📋 Executive Summary")
                fake_stream_markdown(writer_data.get("short_summary", ""), st.empty())

                st.markdown("### 📖 完整報告")
                fake_stream_markdown(writer_data.get("markdown_report", ""), st.empty())

                st.markdown("### ❓ 後續建議問題")
                for q in writer_data.get("follow_up_questions", []) or []:
                    st.markdown(f"- {q}")

                # 彙整來源
                all_url_cits = []
                for it in summaries:
                    all_url_cits.extend(it.get("citations", []) or [])
                all_url_cits.extend(writer_url_cits or [])

                with sources_container:
                    if all_url_cits:
                        st.markdown("**來源**")
                        seen = set()
                        for c in all_url_cits:
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

                # 存入歷史（完整回覆）
                plan_md_saved = "### 🔎 搜尋規劃\n" + "\n".join(plan_md_lines)
                summary_md_saved = "### 📝 各項搜尋摘要\n" + "\n\n".join([f"**{it['query']}**\n{it['summary']}" for it in summaries])

                ai_reply = (
                    plan_md_saved + "\n\n" +
                    summary_md_saved + "\n\n" +
                    "#### Executive Summary\n" + (writer_data.get("short_summary", "") or "") + "\n" +
                    "#### 完整報告\n" + (writer_data.get("markdown_report", "") or "") + "\n" +
                    "#### 後續建議問題\n" + "\n".join([f"- {q}" for q in writer_data.get("follow_up_questions", []) or []])
                )
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "text": ai_reply,
                    "images": [],
                    "docs": []
                })

            else:
                # ===== 一般路徑：原本助理（Responses + web_search + 附件） =====
                resp = client.responses.create(
                    model="gpt-5",
                    input=trimmed_messages,
                    instructions=ANYA_SYSTEM_PROMPT,
                    tools=[{"type": "web_search"}],
                    tool_choice="auto",
                )

                ai_text, url_cits, file_cits = parse_response_text_and_citations(resp)
                final_text = fake_stream_markdown(ai_text, placeholder)

                with sources_container:
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

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "text": final_text,
                    "images": [],
                    "docs": []
                })

        except Exception as e:
            placeholder.markdown(f"API 發生錯誤：{e}")
            try:
                st.code(e.response.json(), language="json")
            except Exception:
                import traceback
                st.code(traceback.format_exc())

    st.rerun()
