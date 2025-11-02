import streamlit as st
import asyncio
from pydantic import BaseModel
import os
import nest_asyncio
nest_asyncio.apply()

from openai.types.shared.reasoning import Reasoning
from agents import Agent, ModelSettings, WebSearchTool, Runner, handoff

# 多模態需要的工具
import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI
import time

# =========================
# 基本環境設定
# =========================
st.set_page_config(page_title="AI 研究助理 Chat（st.chat_input 附件版）", layout="wide", page_icon="🤖")
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_KEY"]

def run_async(coro):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)

# =========================
# 打字動畫（無停止功能、只有 emoji，節奏稍慢）
# =========================
def emoji_token_stream(
    full_text: str,
    emoji: str = "🌸",
    min_cps: int = 20,
    max_cps: int = 110,
    short_len: int = 300,
    long_len: int = 1200,
    punctuation_pause: float = 0.50,
    preview_ratio: float = 0.40,
    code_speedup: float = 1.8,
    ph=None
):
    if not full_text:
        return ""
    try:
        import regex as re
        tokens = re.findall(r"\X", full_text)
    except Exception:
        tokens = list(full_text)

    n = len(tokens)
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
    out, i, inside_code = [], 0, False
    punct = set(".!?;:，。！？：、…\n")

    def chunk_size(idx):
        if inside_code: return 8
        if idx < 80:    return 1
        if idx < 240:   return 2
        if idx < 900:   return 3
        return 4

    def render(txt): placeholder.markdown(txt)

    while i < n:
        k = min(chunk_size(i), n - i)
        chunk_tokens = tokens[i:i+k]
        chunk_text = "".join(chunk_tokens)
        i += k

        if "```" in chunk_text:
            if chunk_text.count("```") % 2 == 1:
                inside_code = not inside_code

        intended = per_char_delay * k
        if inside_code:
            intended = max(intended / code_speedup, 0.002)

        last_char = chunk_tokens[-1]
        if last_char in punct and not inside_code:
            intended += per_char_delay * punctuation_pause

        start_t = time.monotonic()

        # 預覽：只有 emoji
        current_text = "".join(out)
        if not inside_code:
            render(current_text + emoji)
            time.sleep(min(intended * preview_ratio, 0.07))

        # 正式寫入
        out.append(chunk_text)
        render("".join(out))

        elapsed = time.monotonic() - start_t
        remain = max(0.0, intended - elapsed)
        time.sleep(remain)

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
            buf.append(line); continue
        if not in_code and line.strip() == "":
            if buf:
                parts.append("".join(buf).strip("\n")); buf=[]
        else:
            buf.append(line)
    if buf: parts.append("".join(buf).strip("\n"))
    return [p for p in parts if p.strip()]

def paragraph_type_with_fade(md_text: str, emoji: str = "🌸", fade_ms: int = 160):
    paragraphs = split_md_paragraphs(md_text)
    for para in paragraphs:
        ph = st.empty()
        ph.markdown(f":grey[{para}]")
        time.sleep(fade_ms / 1000.0)
        emoji_token_stream(para, emoji=emoji, ph=ph)
        st.markdown("")

# =========================
# 多模態系統提示
# =========================
client = OpenAI(api_key=st.secrets["OPENAI_KEY"])
VISION_SYSTEM_PROMPT = """
你是一位多模態助理。收到圖片與（可選）文字指示時：
- 先描述圖片關鍵內容（物件、文字、關係、場景、版面）。
- 若有多張圖片，請比較差異或建立步驟推論。
- 適度結合OCR與推理；若與使用者提問相關，提供條列式結論與可行建議。
請以正體中文作答。
"""

# =========================
# 規劃/搜尋/寫作/路由 Agents
# =========================
planner_agent_PROMPT = (
    "You are a helpful research assistant. Given a query, come up with a set of web searches "
    "to perform to best answer the query. Output between 5 and 20 terms to query for."
)

class WebSearchItem(BaseModel):
    reason: str
    query: str

class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem]

planner_agent = Agent(
    name="PlannerAgent",
    instructions=planner_agent_PROMPT if (planner_agent_PROMPT:=planner_agent_PROMPT) else "",
    model="gpt-5",
    model_settings=ModelSettings(reasoning=Reasoning(effort="medium")),
    output_type=WebSearchPlan,
)

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
    markdown_report: str
    follow_up_questions: list[str]

writer_agent = Agent(
    name="WriterAgent",
    instructions=writer_agent_PROMPT,
    model="gpt-5-mini",
    model_settings=ModelSettings(reasoning=Reasoning(effort="medium")),
    output_type=ReportData,
)

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
# 介面與主要流程（精簡聊天區＋附件）
# =========================
st.title("AI 研究助理 Chat 版（精簡聊天＋附件上傳）")
st.write("直接在輸入框貼文字，或一起上傳圖片，AI 會自動切換到多模態模式喔～")

# 初始化對話歷史（加入 images 欄位）
if "messages" not in st.session_state:
    st.session_state.messages = []

# 顯示歷史訊息（精簡：只在使用者訊息中顯示縮圖）
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=msg.get("avatar")):
        # 文字
        if msg.get("content"):
            st.markdown(msg["content"])
        # 圖片（只有 user 端可能有）
        if msg.get("images"):
            thumbs = []
            for name, imgbytes in msg["images"]:
                try:
                    thumbs.append(Image.open(BytesIO(imgbytes)))
                except Exception:
                    pass
            if thumbs:
                st.image(thumbs, width=220)

# 使用者輸入（支援多張圖片）
prompt = st.chat_input(
    "輸入問題，或一起上傳圖片讓我幫你看圖說故事～",
    accept_file="multiple",
    file_type=["png", "jpg", "jpeg", "webp"]
)

if prompt:
    # 兼容：有些版本回傳字串；有些回傳帶 text/files 的物件
    user_text = prompt.text.strip() if hasattr(prompt, "text") and prompt.text else (prompt.strip() if isinstance(prompt, str) else "")
    files = prompt.files if hasattr(prompt, "files") and prompt.files else []

    # 將圖片轉 base64，不在這裡顯示expander，保持聊天區簡潔
    content_blocks = []
    images_for_history = []

    if user_text:
        content_blocks.append({"type": "input_text", "text": user_text})

    for f in files:
        imgbytes = f.getbuffer()
        mime = getattr(f, "type", None) or "image/png"
        b64 = base64.b64encode(imgbytes).decode()
        content_blocks.append({
            "type": "input_image",
            "image_url": f"data:{mime};base64,{b64}"
        })
        images_for_history.append((getattr(f, "name", "image"), imgbytes))

    # 把使用者訊息（含縮圖）寫入歷史並顯示
    st.session_state.messages.append({
        "role": "user",
        "content": user_text,
        "images": images_for_history
    })
    with st.chat_message("user"):
        if user_text:
            st.markdown(user_text)
        if images_for_history:
            st.image([Image.open(BytesIO(b)) for _, b in images_for_history], width=220)

    # 助理回覆
    with st.chat_message("assistant"):
        with st.spinner("安妮亞努力思考中…"):
            # 如果有圖片，走多模態；否則走研究/一般聊天路線
            if any(block["type"] == "input_image" for block in content_blocks):
                try:
                    resp = client.responses.create(
                        model="gpt-5",
                        input=[{"role": "user", "content": content_blocks}],
                        instructions=VISION_SYSTEM_PROMPT,
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
                        out_text = "安妮亞看過了，但還沒抓到你想問的重點～可以再具體一點嗎？"

                    # 顯示：段落淡入＋逐字
                    paragraph_type_with_fade(out_text, emoji="🌸", fade_ms=140)

                    # 寫入歷史
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": out_text,
                        "images": []
                    })

                except Exception as e:
                    err = f"圖片分析失敗：{e}"
                    st.error(err)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": err,
                        "images": []
                    })

            else:
                # 純文字：讓 Router 決定是否 handoff 做研究
                router_result = run_async(Runner.run(router_agent, user_text))

                if isinstance(router_result.final_output, WebSearchPlan):
                    # Step 1 規劃
                    search_plan = router_result.final_output.searches
                    plan_md = "### 🔎 搜尋規劃\n"
                    for idx, item in enumerate(search_plan):
                        plan_md += f"**{idx+1}. {item.query}**\n> {item.reason}\n"

                    # Step 2 並行搜尋
                    tasks = [
                        Runner.run(search_agent, f"Search term: {item.query}\nReason: {item.reason}")
                        for item in search_plan
                    ]
                    results = run_async(asyncio.gather(*tasks))
                    summaries = [str(r.final_output) for r in results]

                    summary_md = "### 📝 各項搜尋摘要\n"
                    for idx, summary in enumerate(summaries):
                        summary_md += f"**{search_plan[idx].query}**\n{summary}\n\n"

                    with st.expander("🔎 搜尋規劃與各項搜尋摘要", expanded=False):
                        st.markdown("### 搜尋規劃")
                        for idx, item in enumerate(search_plan):
                            st.markdown(f"**{idx+1}. {item.query}**\n> {item.reason}")
                        st.markdown("### 各項搜尋摘要")
                        for idx, summary in enumerate(summaries):
                            st.markdown(f"**{search_plan[idx].query}**\n{summary}\n")

                    # Step 3 寫作
                    writer_input = f"Original query: {user_text}\nSummarized search results: {summaries}"
                    report = run_async(Runner.run(writer_agent, writer_input))

                    st.markdown("### 📋 Executive Summary")
                    emoji_token_stream(report.final_output.short_summary, emoji="🌟")

                    st.markdown("### 📖 完整報告")
                    paragraph_type_with_fade(report.final_output.markdown_report, emoji="🌸", fade_ms=160)

                    st.markdown("### ❓ 後續建議問題")
                    for q in report.final_output.follow_up_questions:
                        emoji_token_stream(q, emoji="🥜")

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
                        "images": []
                    })

                else:
                    # 一般對話
                    full_text = str(router_result.final_output)
                    emoji_token_stream(full_text, emoji="🌸")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_text,
                        "images": []
                    })
