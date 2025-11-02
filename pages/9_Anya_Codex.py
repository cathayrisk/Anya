import streamlit as st
import asyncio
from pydantic import BaseModel
import os
import nest_asyncio
nest_asyncio.apply()

from openai.types.shared.reasoning import Reasoning
from agents import Agent, ModelSettings, WebSearchTool, Runner, handoff

# 追加入多模態需要的工具
import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI
import time

# =========================
# 基本環境設定
# =========================
st.set_page_config(page_title="AI 研究助理 Chat（多模態＋段落淡入）", layout="wide", page_icon="🤖")
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_KEY"]

def run_async(coro):
    # 在 Streamlit 中安全地跑 asyncio 協程
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)

# =========================
# 打字動畫（無停止功能、只有 emoji，節奏稍慢）
# =========================
def emoji_token_stream(
    full_text: str,
    emoji: str = "🌸",
    # 稍微放慢一點點（原本 28~140）
    min_cps: int = 20,
    max_cps: int = 110,
    short_len: int = 300,
    long_len: int = 1200,
    punctuation_pause: float = 0.50,  # 標點停頓略增
    preview_ratio: float = 0.40,      # emoji 預覽比重略增
    code_speedup: float = 1.8,
    ph=None
):
    """
    只用 emoji 做短暫預覽、無游標、無停止功能。
    """
    import time

    if not full_text:
        return ""

    # 使用字素叢集切分，避免切壞 emoji/合字
    try:
        import regex as re
        tokens = re.findall(r"\X", full_text)
    except Exception:
        tokens = list(full_text)

    n = len(tokens)

    # 依長度插值速度
    def lerp(a, b, t): return a + (b - a) * t
    if n <= short_len:
        base_cps = min_cps
    elif n >= long_len:
        base_cps = max_cps
    else:
        t = (n - short_len) / max(1, (long_len - short_len))
        base_cps = lerp(min_cps, max_cps, t)

    per_char_delay = 1.0 / max(1.0, base_cps)

    placeholder = ph or st.empty()

    out = []
    i = 0
    inside_code = False
    punct = set(".!?;:，。！？：、…\n")

    # 前段精緻、後段加速（稍慢版）
    def chunk_size(idx):
        if inside_code:
            return 8
        if idx < 80:
            return 1
        if idx < 240:
            return 2
        if idx < 900:
            return 3
        return 4

    def render(txt):
        placeholder.markdown(txt)

    while i < n:
        k = min(chunk_size(i), n - i)
        chunk_tokens = tokens[i:i + k]
        chunk_text = "".join(chunk_tokens)
        i += k

        # 偵測 ``` 程式碼區塊
        if "```" in chunk_text:
            flips = chunk_text.count("```")
            if flips % 2 == 1:
                inside_code = not inside_code

        intended = per_char_delay * k
        if inside_code:
            intended = max(intended / code_speedup, 0.002)

        last_char = chunk_tokens[-1]
        if last_char in punct and not inside_code:
            intended += per_char_delay * punctuation_pause

        start_t = time.monotonic()

        # 預覽：只有 emoji（不顯示游標）
        current_text = "".join(out)
        if not inside_code:
            render(current_text + emoji)
            time.sleep(min(intended * preview_ratio, 0.07))

        # 正式寫入
        out.append(chunk_text)
        render("".join(out))

        # 填滿剩餘時間，讓節奏穩定
        elapsed = time.monotonic() - start_t
        remain = max(0.0, intended - elapsed)
        time.sleep(remain)

    # 收尾，不帶 emoji
    render("".join(out))
    return "".join(out)

# =========================
# 段落淡入 + 逐字（方案2）
# =========================
def split_md_paragraphs(md: str):
    parts, buf, in_code = [], [], False
    for line in md.splitlines(keepends=True):
        if line.strip().startswith("```"):
            in_code = not in_code
            buf.append(line)
            continue
        if not in_code and line.strip() == "":
            if buf:
                parts.append("".join(buf).strip("\n")); buf=[]
        else:
            buf.append(line)
    if buf:
        parts.append("".join(buf).strip("\n"))
    return [p for p in parts if p.strip()]

def paragraph_type_with_fade(md_text: str, emoji: str = "🌸", fade_ms: int = 160):
    paragraphs = split_md_paragraphs(md_text)
    for para in paragraphs:
        ph = st.empty()
        # 1) 段落淡入（灰色幽靈）
        ph.markdown(f":grey[{para}]")
        time.sleep(fade_ms / 1000.0)
        # 2) 同一個 placeholder 逐字播放（會覆蓋灰色）
        emoji_token_stream(para, emoji=emoji, ph=ph)
        st.markdown("")  # 段落間距

# =========================
# 規劃 Agent（Planner）
# =========================
planner_agent_PROMPT = (
    "You are a helpful research assistant. Given a query, come up with a set of web searches "
    "to perform to best answer the query. Output between 5 and 20 terms to query for."
)

class WebSearchItem(BaseModel):
    reason: str
    "Your reasoning for why this search is important to the query."
    query: str
    "The search term to use for the web search."

class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem]
    """A list of web searches to perform to best answer the query."""

planner_agent = Agent(
    name="PlannerAgent",
    instructions=planner_agent_PROMPT,
    model="gpt-5",
    model_settings=ModelSettings(reasoning=Reasoning(effort="medium")),
    output_type=WebSearchPlan,
)

# =========================
# 搜尋 Agent（Search）
# =========================
INSTRUCTIONS = (
    "You are a research assistant. Given a search term, you search the web for that term and "
    "produce a concise summary of the results. The summary must be 2-3 paragraphs and less than 300 "
    "words. Capture the main points. Write succinctly, no need to have complete sentences or good "
    "grammar. This will be consumed by someone synthesizing a report, so its vital you capture the "
    "essence and ignore any fluff. Do not include any additional commentary other than the summary "
    "itself."
)

search_agent = Agent(
    name="Search agent",
    model="gpt-4.1",
    instructions=INSTRUCTIONS,
    tools=[WebSearchTool()],
    model_settings=ModelSettings(tool_choice="required"),
)

# =========================
# 寫作 Agent（Writer）
# =========================
writer_agent_PROMPT = (
    "You are a senior researcher tasked with writing a cohesive report for a research query. "
    "You will be provided with the original query, and some initial research done by a research "
    "assistant.\n"
    "You should first come up with an outline for the report that describes the structure and "
    "flow of the report. Then, generate the report and return that as your final output.\n"
    "The final output should be in markdown format, and it should be lengthy and detailed. Aim "
    "for 5-10 pages of content, at least 1000 words."
    "請務必以正體中文回應，並遵循台灣用語習慣"
)

class ReportData(BaseModel):
    short_summary: str
    """A short 2-3 sentence summary of the findings."""
    markdown_report: str
    """The final report"""
    follow_up_questions: list[str]
    """Suggested topics to research further"""

writer_agent = Agent(
    name="WriterAgent",
    instructions=writer_agent_PROMPT,
    model="gpt-5-mini",
    model_settings=ModelSettings(reasoning=Reasoning(effort="medium")),
    output_type=ReportData,
)

# =========================
# Router Agent（自動 handoff）
# =========================
ROUTER_PROMPT = """
你是一個智慧助理，會根據用戶的需求自動決定要怎麼處理問題。
- 如果用戶的問題是「需要研究、查資料、分析、寫報告、文獻探討」等，請使用 transfer_to_planner_agent 工具，把問題交給研究規劃助理。
- 如果只是一般聊天、知識問答、閒聊，請直接用你自己的知識回答。必要的時候可以使用WebSearchTool來搜尋網路資訊。
請根據用戶的輸入，自行判斷要不要 handoff。
"""

router_agent = Agent(
    name="RouterAgent",
    instructions=ROUTER_PROMPT,
    model="gpt-5",
    tools=[WebSearchTool()],
    model_settings=ModelSettings(
        reasoning=Reasoning(effort="low"),
        verbosity="medium",
    ),
    handoffs=[handoff(planner_agent)]
)

# =========================
# Multimodal：圖片理解模式（新）
# =========================
client = OpenAI(api_key=st.secrets["OPENAI_KEY"])

VISION_SYSTEM_PROMPT = """
你是一位多模態助理。收到圖片與（可選）文字指示時：
- 先描述圖片關鍵內容（物件、文字、關係、場景、版面）。
- 若有多張圖片，請比較差異或建立步驟推論。
- 適度結合OCR與推理；若與使用者提問相關，提供條列式結論與可行建議。
請以正體中文作答。
"""

st.title("AI 研究助理 Chat 版（多模態升級）")
st.write("用對話方式問研究問題，AI 會像聊天一樣幫你查資料、寫報告！另外也能上傳圖片，請 AI 幫你看圖說故事～")

with st.expander("🖼️ 圖片理解模式（多模態）", expanded=False):
    col1, col2 = st.columns([3, 2])
    with col1:
        vision_text = st.text_area("（可選）輸入你想讓 AI 針對圖片回答的問題或任務", placeholder="例如：幫我比對這兩張簡報圖的差異，整理成3點重點。")
    with col2:
        files = st.file_uploader("上傳 1～6 張圖片", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
    if st.button("分析圖片", type="primary", use_container_width=True, disabled=not files):
        # 準備內容區塊
        content_blocks = []
        if vision_text and vision_text.strip():
            content_blocks.append({"type": "input_text", "text": vision_text.strip()})
        imgs_preview = []
        for f in files[:6]:
            imgbytes = f.getvalue()
            mime = f"type" if hasattr(f, "type") and f.type else "image/png"
            b64 = base64.b64encode(imgbytes).decode()
            content_blocks.append({"type": "input_image", "image_url": f"data:{mime};base64,{b64}"})
            imgs_preview.append(imgbytes)

        with st.spinner("安妮亞看圖中…wakuwaku！🤩"):
            try:
                resp = client.responses.create(
                    model="gpt-5",
                    input=[{"role": "user", "content": content_blocks}],
                    instructions=VISION_SYSTEM_PROMPT,
                    parallel_tool_calls=True,
                    reasoning={"effort": "medium"},
                    text={"verbosity": "medium"},
                    store=False,
                    truncation="auto",
                )
                out_text = ""
                if hasattr(resp, "output") and resp.output:
                    for item in resp.output:
                        if hasattr(item, "content") and item.content:
                            for c in item.content:
                                if getattr(c, "type", None) == "output_text":
                                    out_text += c.text

                if not out_text.strip():
                    out_text = "安妮亞看過了，但沒有辨識到可以回答的重點，能不能補充一下你的期待呢？"

                # 顯示上傳圖片預覽
                st.markdown("#### 圖片預覽")
                st.image([Image.open(BytesIO(x)) for x in imgs_preview], width=260)

                st.markdown("#### 解析結果")
                paragraph_type_with_fade(out_text, emoji="🌸", fade_ms=140)

            except Exception as e:
                st.error(f"圖片分析失敗：{e}")

st.markdown("---")

# =========================
# 主要聊天介面（研究/一般對話）
# =========================
# 初始化對話歷史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 顯示歷史訊息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=msg.get("avatar")):
        st.markdown(msg["content"])

# 聊天輸入
user_input = st.chat_input("請輸入你想研究的問題或繼續追問...")

if user_input:
    # 顯示使用者訊息
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
    })
    with st.chat_message("user"):
        st.markdown(user_input)

    # AI 處理（顯示 spinner）
    with st.chat_message("assistant"):
        with st.spinner("AI 正在努力思考中..."):
            # 讓 Router 決定是否要 handoff
            router_result = run_async(Runner.run(router_agent, user_input))

            # 若 handoff 到規劃助理（需要研究工作）
            if isinstance(router_result.final_output, WebSearchPlan):
                # Step 1: 規劃
                search_plan = router_result.final_output.searches

                plan_md = "### 🔎 搜尋規劃\n"
                for idx, item in enumerate(search_plan):
                    plan_md += f"**{idx+1}. {item.query}**\n> {item.reason}\n"

                # Step 2: 並行搜尋
                search_tasks = [
                    Runner.run(search_agent, f"Search term: {item.query}\nReason: {item.reason}")
                    for item in search_plan
                ]
                search_results = run_async(asyncio.gather(*search_tasks))
                summaries = [str(r.final_output) for r in search_results]

                summary_md = "### 📝 各項搜尋摘要\n"
                for idx, summary in enumerate(summaries):
                    summary_md += f"**{search_plan[idx].query}**\n{summary}\n\n"

                with st.expander("🔎 搜尋規劃與各項搜尋摘要", expanded=True):
                    st.markdown("### 搜尋規劃")
                    for idx, item in enumerate(search_plan):
                        st.markdown(f"**{idx+1}. {item.query}**\n> {item.reason}")

                    st.markdown("### 各項搜尋摘要")
                    for idx, summary in enumerate(summaries):
                        st.markdown(f"**{search_plan[idx].query}**\n{summary}\n")

                # Step 3: 整合寫作
                writer_input = f"Original query: {user_input}\nSummarized search results: {summaries}"
                report = run_async(Runner.run(writer_agent, writer_input))

                st.markdown("### 📋 Executive Summary")
                # 短摘要保留純逐字
                emoji_token_stream(report.final_output.short_summary, emoji="🌟")

                st.markdown("### 📖 完整報告")
                # 長文改用「段落淡入 + 逐字」
                paragraph_type_with_fade(report.final_output.markdown_report, emoji="🌸", fade_ms=160)

                st.markdown("### ❓ 後續建議問題")
                for q in report.final_output.follow_up_questions:
                    emoji_token_stream(q, emoji="🥜")

                # 把 AI 回覆存進歷史
                ai_reply = (
                    plan_md + "\n" +
                    summary_md + "\n" +
                    "#### Executive Summary\n" + report.final_output.short_summary + "\n" +
                    "#### 完整報告\n" + report.final_output.markdown_report + "\n" +
                    "#### 後續建議問題\n" + "\n".join([f"- {q}" for q in report.final_output.follow_up_questions])
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": ai_reply,
                })

            else:
                # 一般對話：直接使用第一次的 router 結果，不要重跑
                full_text = str(router_result.final_output)
                emoji_token_stream(full_text, emoji="🌸")  # 速度已調慢
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_text,
                })
