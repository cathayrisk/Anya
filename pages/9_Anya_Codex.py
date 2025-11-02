import streamlit as st
import asyncio
from openai.types.shared.reasoning import Reasoning
from pydantic import BaseModel
import os
from agents import Agent, ModelSettings, WebSearchTool, Runner, handoff
from openai.types.responses import ResponseTextDeltaEvent
import time
import nest_asyncio
nest_asyncio.apply()

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_KEY"]

def stream_text_gen(result_streaming):
    async def gen():
        async for event in result_streaming.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                yield event.data.delta or ""
    return gen()

def emoji_token_stream(
    full_text: str,
    emoji: str = "🌸",
    # 速度設定（每秒字數的上下限，會隨長度自動插值）
    min_cps: int = 28,
    max_cps: int = 140,
    short_len: int = 300,
    long_len: int = 1200,
    # 標點停頓（以每字延遲為基準的倍數）
    punctuation_pause: float = 0.45,
    # 預覽emoji佔整體延遲的比例（越小越含蓄）
    preview_ratio: float = 0.35,
    # 程式碼區塊內是否略過emoji預覽並加速
    code_speedup: float = 1.8,
    # 可中止旗標（配合按鈕）
    cancel_key: str | None = None,
    # 顯示進度百分比
    show_progress: bool = False,
    # 外部顯示容器（可傳 st.empty() 進來，方便跳過後用同一格改成全文）
    ph=None
):
    """
    回傳 (text_shown, cancelled)
    - text_shown: 實際顯示文字（若中止可能是部分）
    - cancelled : True 表示中途被停止
    """
    import time
    import streamlit as st

    if not full_text:
        return "", False

    # 優先用 regex 的 \X 做「字素叢集」切分，避免切壞 emoji/合字
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
    prog_ph = st.empty() if show_progress else None

    out = []
    i = 0
    cancelled = False
    inside_code = False
    punct = set(".!?;:，。！？：、…\n")

    # 讓前段比較精緻、後段一次吐多一點字（更順）
    def chunk_size(idx):
        if inside_code:
            return 8
        if idx < 60:
            return 1
        if idx < 200:
            return 2
        if idx < 800:
            return 3
        return 4

    def render(txt):
        placeholder.markdown(txt)

    while i < n:
        if cancel_key and st.session_state.get(cancel_key, False):
            cancelled = True
            break

        k = min(chunk_size(i), n - i)
        chunk_tokens = tokens[i:i + k]
        chunk_text = "".join(chunk_tokens)
        i += k

        # 粗略偵測程式碼區塊（以 ``` 切換）
        if "```" in chunk_text:
            flips = chunk_text.count("```")
            if flips % 2 == 1:
                inside_code = not inside_code

        # 計算這個區塊應該花的時間
        intended = per_char_delay * k
        if inside_code:
            intended = max(intended / code_speedup, 0.002)

        # 標點微停頓（只看區塊最後一個字）
        last_char = chunk_tokens[-1]
        if last_char in punct and not inside_code:
            intended += per_char_delay * punctuation_pause

        start_t = time.monotonic()

        # 預覽：只加 emoji，不加游標（你說不要游標～）
        current_text = "".join(out)
        if not inside_code:
            render(current_text + emoji)
            # 預覽時間佔比，最多給一點點就好，避免閃太多
            preview_sleep = min(intended * preview_ratio, 0.06)
            time.sleep(preview_sleep)
        else:
            preview_sleep = 0.0

        # 正式寫入
        out.append(chunk_text)
        render("".join(out))

        # 把剩下的時間睡完，讓節奏穩定（扣掉前面預覽用掉的時間）
        elapsed = time.monotonic() - start_t
        remain = max(0.0, intended - elapsed)
        time.sleep(remain)

        if show_progress and (i % 60 == 0 or i == n):
            pct = int(i * 100 / n)
            prog_ph.caption(f"輸出中… {pct}%")

    # 收尾（保證最後不帶emoji）
    render("".join(out))
    if show_progress:
        prog_ph.empty()

    return "".join(out), cancelled

#---Planner
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

#----search_agent
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

#---writer_agent
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

#---RouterAgent（LLM自動判斷是否handoff）
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
        reasoning=Reasoning(effort="low"),  # "minimal", "low", "medium", "high"
        verbosity="medium",  # "low", "medium", "high"
    ),
    handoffs=[
        handoff(planner_agent)
    ]
)

def run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)

st.set_page_config(page_title="AI 研究助理 Chat", layout="wide", page_icon="🤖")

st.title("AI 研究助理 Chat 版")
st.write("用對話方式問研究問題，AI 會像聊天一樣幫你查資料、寫報告！")

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
            # 只呼叫 RouterAgent，讓 LLM 自己決定要不要 handoff
            loop = asyncio.get_event_loop()
            router_result = loop.run_until_complete(Runner.run(router_agent, user_input))

            # 如果 LLM handoff 給 planner_agent，則進行研究流程
            if isinstance(router_result.final_output, WebSearchPlan):
                # Step 1: 規劃
                plan_result = router_result  # 就是 planner_agent 的結果
                search_plan = plan_result.final_output.searches

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
                emoji_token_stream(report.final_output.short_summary, emoji="🌟")  # 用星星

                st.markdown("### 📖 完整報告")
                emoji_token_stream(report.final_output.markdown_report, emoji="🌸")  # 用花朵

                st.markdown("### ❓ 後續建議問題")
                for q in report.final_output.follow_up_questions:
                    emoji_token_stream(q, emoji="🥜")  # 用花生

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
                # 一般對話，直接顯示 RouterAgent 的回覆
                router_result = run_async(Runner.run(router_agent, user_input))
                full_text = str(router_result.final_output)
                emoji_token_stream(full_text)
                    
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_text,  # st.write_stream 回傳完整文字
                })
