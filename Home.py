import os
import streamlit as st
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
import inspect
from typing import Callable, TypeVar, List, Dict, Any, Optional
from pydantic import BaseModel, Field
import time
import re
import requests
from openai import OpenAI
import traceback
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

st.set_page_config(
    page_title="Anya",
    layout="wide",
    page_icon="🥜",
    initial_sidebar_state="collapsed"
)

# --- 1. Streamlit session_state 初始化 ---
if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="嗨嗨～安妮亞來了！👋 有什麼想問安妮亞的嗎？")]
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gpt-4.1"
if "current_model" not in st.session_state:
    st.session_state.current_model = None
if "llm" not in st.session_state:
    st.session_state.llm = None

# 定義 WikiInputs
class WikiInputs(BaseModel):
    query: str = Field(description="要查詢的關鍵字")

# --- 2. LLM 初始化 ---
def ensure_llm():
    if (
        st.session_state.llm is None
        or st.session_state.current_model != st.session_state.selected_model
    ):
        st.session_state.llm = ChatOpenAI(
            model=st.session_state.selected_model,
            openai_api_key=st.secrets["OPENAI_KEY"],
            temperature=0.0,
            streaming=True,
        )
        st.session_state.current_model = st.session_state.selected_model

ensure_llm()

# --- 3. 工具定義 ---
# === OpenAI 初始化 ===
client = OpenAI(api_key=st.secrets["OPENAI_KEY"])

# === Meta Prompting 工具 ===
def meta_optimize_prompt(simple_prompt: str, goal: str) -> str:
    meta_prompt = f"""
    請優化以下 prompt，使其能更有效達成「{goal}」，並符合 prompt engineering 最佳實踐。
    {simple_prompt}
    只回傳優化後的 prompt。
    """
    response = client.chat.completions.create(
        model="o4-mini",
        messages=[{"role": "user", "content": meta_prompt}]
    )
    return response.choices[0].message.content.strip()

# === 產生查詢（中英文） ===
def generate_queries(topic: str, model="gpt-4.1-mini") -> List[str]:
    simple_prompt = f"""請針對「{topic}」這個主題，分別用繁體中文與英文各產生三個適合用於網路搜尋的查詢關鍵字，並以如下 JSON 格式回覆：
{{
    "zh": ["查詢1", "查詢2", "查詢3"],
    "en": ["query1", "query2", "query3"]
}}
"""
    optimized_prompt = meta_optimize_prompt(simple_prompt, "產生多元且具針對性的查詢關鍵字")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": optimized_prompt}]
    )
    content = response.choices[0].message.content
    try:
        queries = json.loads(content)
    except Exception:
        import re
        content = re.sub(r"[\u4e00-\u9fff]+：", "", content)
        content = content.replace("'", '"')
        queries = json.loads(content)
    return queries["zh"] + queries["en"]

# === 查詢摘要 ===
def auto_summarize(text: str, model="gpt-4.1-mini") -> str:
    simple_prompt = f"請用繁體中文摘要以下內容，重點條列，100字內：\n{text}"
    optimized_prompt = meta_optimize_prompt(simple_prompt, "產生精簡且重點明確的摘要")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": optimized_prompt}]
    )
    return response.choices[0].message.content.strip()

# 2. 章節規劃（推理模型）
def plan_report(topic, search_summaries, model="o4-mini"):
    prompt = f"""Formatting re-enabled
# Role and Objective
你是一位專業技術寫手，目標是針對「{topic}」這個主題，根據下方搜尋摘要，規劃一份完整、深入、結構化的研究報告。

# Instructions
- 報告需包含5-7個章節，每章節需有明確標題。
- 每個章節**必須包含2-4個小標題**（子議題），每個小標題下要有2-3句細節說明。
- 小標題可涵蓋：產業現況、技術細節、國際比較、未來趨勢、挑戰、解決方案、案例、數據等。
- 章節規劃要有邏輯順序，內容要有層次。
- 請用繁體中文條列式回覆，格式如下：

# Output Format
1. 章節標題
    - 小標題1：細節說明
    - 小標題2：細節說明
    - 小標題3：細節說明
2. 章節標題
    - 小標題1：細節說明
    - 小標題2：細節說明
    - 小標題3：細節說明
...

# 搜尋摘要
{search_summaries}
"""
    response = client.responses.create(
        model=model,
        reasoning={"effort": "medium", "summary": "auto"},
        input=[{"role": "user", "content": prompt}]
    )
    return response.output_text

# 3. 解析章節
def parse_sections(plan: str):
    section_pattern = r"\d+\.\s*([^\n]+)"
    sub_pattern = r"-\s*([^\n：:]+)[：:]\s*([^\n]+)"
    sections = []
    section_blocks = re.split(r"\d+\.\s*", plan)[1:]
    for block in section_blocks:
        lines = block.strip().split("\n")
        title = lines[0].strip()
        subs = []
        for line in lines[1:]:
            m = re.match(sub_pattern, line.strip())
            if m:
                subs.append({"subtitle": m.group(1).strip(), "desc": m.group(2).strip()})
        sections.append({"title": title, "subtitles": subs})
    return sections

# 4. 章節查詢產生
def section_queries(section_title, section_desc, model="gpt-4.1-mini"):
    prompt = f"""請針對章節「{section_title}」({section_desc})，分別用繁體中文與英文各產生兩個適合用於網路搜尋的查詢關鍵字，回傳 JSON 格式：
{{
    "zh": ["查詢1", "查詢2"],
    "en": ["query1", "query2"]
}}
"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    content = response.choices[0].message.content
    try:
        queries = json.loads(content)
    except Exception:
        content = re.sub(r"[\u4e00-\u9fff]+：", "", content)
        content = content.replace("'", '"')
        queries = json.loads(content)
    return queries["zh"] + queries["en"]

# 5. 章節撰寫（直接用多筆查詢結果）
def section_write(section_title, section_desc, search_results, model="gpt-4.1-mini"):
    prompt = f"""
# Role and Objective
你是一位專業技術寫手，根據下方章節主題、說明與「多筆網路查詢結果」，撰寫一段內容豐富、結構清晰、具體詳實的章節內容。

# Instructions
- 內容需至少600字，並涵蓋：具體數據、真實案例、國際比較、產業現況、技術細節、未來趨勢、挑戰與解決方案。
- 必須根據下方每一筆查詢結果，整合出完整內容。
- 每個小標題下必須有2-3段具體說明。
- 條列重點只能放在章節結尾，正文必須是完整段落。
- 文末請用「## 來源」列出所有引用來源（Markdown格式），來源必須來自下方查詢結果。
- 請勿省略細節，若有多個觀點請分段說明。
- 請勿重複內容，避免空泛敘述。
- 請用繁體中文撰寫。

# 章節主題
{section_title}

# 章節說明
{section_desc}

# 多筆查詢結果（每筆都要參考）
{search_results}
"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# === 來源提取 ===
def extract_sources(content: str) -> List[str]:
    # 假設來源格式為 markdown link
    return re.findall(r'\[([^\]]+)\]\((https?://[^\)]+)\)', content)

# === 章節內容評分與補強建議 ===
def section_grade(section_title: str, section_content: str, model="gpt-4.1-mini") -> Dict[str, Any]:
    simple_prompt = f"""請評分以下章節內容是否完整、正確、可讀性佳，若不及格請列出需補充的查詢關鍵字（中英文各一），回傳 JSON 格式：
{{
    "grade": "pass" 或 "fail",
    "follow_up_queries": ["查詢1", "query2"]
}}
章節：{section_title}
內容：
{section_content}
"""
    optimized_prompt = meta_optimize_prompt(simple_prompt, "嚴謹評分並產生具體補強建議")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": optimized_prompt}]
    )
    try:
        return json.loads(response.choices[0].message.content)
    except:
        return {"grade": "pass", "follow_up_queries": []}

# 6. 反思流程（推理模型）
def reflect_report(report: str, model="o4-mini"):
    prompt = f"""Formatting re-enabled
# Role and Objective
你是一位專業審稿人，目標是檢查下方報告的邏輯、正確性、完整性與內容豐富度。

# Instructions
- 請逐一檢查每個章節是否有小標題，且每個小標題下是否有2-3句具體細節說明。
- 檢查內容是否涵蓋數據、案例、國際觀點、技術細節、未來趨勢等豐富面向。
- 檢查是否有明確引用來源。
- 若有內容過於簡略、遺漏重要面向，請明確指出需補充的章節、小標題與建議查詢關鍵字。
- 若內容已足夠豐富且無明顯遺漏，請回覆 "OK"。
- 請用繁體中文回覆。

# 報告內容
{report}
"""
    response = client.responses.create(
        model=model,
        reasoning={"effort": "medium", "summary": "auto"},
        input=[{"role": "user", "content": prompt}]
    )
    return response.output_text

# === 組合章節 ===
def combine_sections(section_contents: List[Dict[str, Any]]) -> str:
    return "\n\n".join([f"## {s['title']}\n\n{s['content']}" for s in section_contents])

# === 主流程（含推理鏈追蹤） ===
def deep_research_pipeline(topic):
    logs = []
    try:
        # 1. 產生查詢
        try:
            queries = section_queries(topic, topic)
            logs.append({"step": "generate_queries", "queries": queries})
        except Exception as e:
            logs.append({"step": "generate_queries", "error": str(e), "traceback": traceback.format_exc()})
            return {"error": "產生查詢失敗", "logs": logs}

        # 2. 查詢所有 query
        all_results = []
        try:
            for q in queries:
                try:
                    result = ddgs_search(q)
                    all_results.append(result)
                except Exception as e:
                    logs.append({"step": "search", "query": q, "error": str(e), "traceback": traceback.format_exc()})
                    all_results.append(f"查詢失敗: {q}")
            logs.append({"step": "search", "results": all_results})
        except Exception as e:
            logs.append({"step": "search", "error": str(e), "traceback": traceback.format_exc()})
            return {"error": "查詢失敗", "logs": logs}

        # 3. 規劃章節
        try:
            search_summary = "\n\n".join(all_results)
            plan = plan_report(topic, search_summary)
            logs.append({"step": "plan_report", "plan": plan})
        except Exception as e:
            logs.append({"step": "plan_report", "error": str(e), "traceback": traceback.format_exc(), "search_summary": search_summary})
            return {"error": "章節規劃失敗", "logs": logs}

        # 4. 章節分段查詢/撰寫
        try:
            sections = parse_sections(plan)
            section_contents = []
            for section in sections:
                s_queries = []
                for sub in section["subtitles"]:
                    try:
                        sub_queries = section_queries(sub["subtitle"], sub["desc"])
                        s_queries.extend(sub_queries)
                    except Exception as e:
                        logs.append({"step": "section_queries", "subtitle": sub["subtitle"], "desc": sub["desc"], "error": str(e), "traceback": traceback.format_exc()})
                s_results = []
                for q in s_queries:
                    try:
                        s_results.append(ddgs_search(q))
                    except Exception as e:
                        logs.append({"step": "section_search", "query": q, "error": str(e), "traceback": traceback.format_exc()})
                        s_results.append(f"查詢失敗: {q}")
                search_results = "\n\n".join(s_results)
                try:
                    content = section_write(
                        section["title"],
                        "；".join([f"{sub['subtitle']}：{sub['desc']}" for sub in section["subtitles"]]),
                        search_results
                    )
                except Exception as e:
                    logs.append({"step": "section_write", "section": section["title"], "error": str(e), "traceback": traceback.format_exc(), "search_results": search_results})
                    content = f"章節內容產生失敗: {section['title']}"
                section_contents.append({
                    "title": section["title"],
                    "content": content
                })
                logs.append({
                    "step": "section",
                    "section": section["title"],
                    "queries": s_queries,
                    "search_results": search_results,
                    "content": content
                })
        except Exception as e:
            logs.append({"step": "section_loop", "error": str(e), "traceback": traceback.format_exc()})
            return {"error": "章節分段查詢/撰寫失敗", "logs": logs}

        # 5. 組合報告
        try:
            report = "\n\n".join([f"## {s['title']}\n\n{s['content']}" for s in section_contents])
            logs.append({"step": "combine_report", "report": report})
        except Exception as e:
            logs.append({"step": "combine_report", "error": str(e), "traceback": traceback.format_exc()})
            return {"error": "組合報告失敗", "logs": logs}

        # 6. 反思流程（最多2次）
        for i in range(2):
            try:
                reflection = reflect_report(report)
                logs.append({"step": "reflection", "round": i+1, "reflection": reflection})
                if reflection.strip().upper() == "OK":
                    break
                else:
                    # 若需補充，可根據 reflection 產生新查詢與補充內容（可進一步自動化）
                    pass
            except Exception as e:
                logs.append({"step": "reflection", "round": i+1, "error": str(e), "traceback": traceback.format_exc()})
                break

        # 7. 結構化輸出
        output = {
            "topic": topic,
            "plan": plan,
            "sections": section_contents,
            "report": report,
            "logs": logs
        }
        return output

    except Exception as e:
        logs.append({"step": "pipeline_outer", "error": str(e), "traceback": traceback.format_exc()})
        return {"error": "pipeline_outer_error", "logs": logs}
@tool
def deep_research_pipeline_tool(topic: str) -> Dict[str, Any]:
    """
    針對指定主題自動進行多步深度研究，回傳結構化報告（含章節、內容、來源、推理鏈）。
    """
    try:
        return deep_research_pipeline(topic)
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

@tool
def wiki_tool(query: str) -> str:
    """
    查詢 Wikipedia（英文），輸入任何語言的關鍵字都可以。
    """
    try:
        tool_obj = WikipediaQueryRun(
            name="wiki-tool",
            description="查詢 Wikipedia（英文），輸入任何語言的關鍵字都可以。",
            args_schema=WikiInputs,
            api_wrapper=WikipediaAPIWrapper(lang="en", doc_content_chars_max=800, top_k_results=1),
            return_direct=True,
        )
        result = tool_obj.invoke({"query": query})
        return result
    except Exception as e:
        return f"wiki_tool error: {e}"

@tool
def ddgs_search(query: str) -> str:
    """DuckDuckGo 搜尋（同時查詢網頁與新聞，回傳 markdown 條列格式並附來源）。"""
    try:
        from duckduckgo_search import DDGS
        ddgs = DDGS()
        web_results = ddgs.text(query, region="wt-wt", safesearch="moderate", max_results=5)
        news_results = ddgs.news(query, region="wt-wt", safesearch="moderate", max_results=5)
        all_results = []
        if isinstance(web_results, list):
            all_results.extend(web_results)
        if isinstance(news_results, list):
            all_results.extend(news_results)
        docs = []
        sources = []
        for item in all_results:
            title = item.get("title", "無標題")
            link = item.get("href", "") or item.get("link", "") or item.get("url", "")
            snippet = item.get("body", "") or item.get("snippet", "")
            docs.append(f"- [{title}]({link})\n  > {snippet}")
            if link:
                sources.append(link)
        if not docs:
            return "No results found."
        markdown_content = "\n".join(docs)
        source_block = "\n\n## 來源\n" + "\n".join(sources)
        return markdown_content + source_block
    except Exception as e:
        return f"Error from DuckDuckGo: {e}"

@tool
def datetime_tool() -> str:
    """確認當前的日期和時間。"""
    return datetime.now().isoformat()

# 你的 deep_thought_tool
def analyze_deeply(input_question: str) -> str:
    """使用OpenAI的模型來深入分析問題並返回結果。"""
    prompt_template = PromptTemplate(
        template="""請分析以下問題，並以正體中文提供詳細的結論和理由，請依據事實分析，不考慮資料的時間因素：

問題：{input_question}

指導方針：
1. 描述問題的背景和相關資訊。
2. 直接給出你的結論，並提供支持該結論的理由。
3. 如果有不確定的地方，請明確指出。
4. 確保你的回答是詳細且有條理的。
""",
        input_variables=["input_question"],
    )
    llmo1 = ChatOpenAI(
        openai_api_key=st.secrets["OPENAI_KEY"],
        model="o4-mini",
        streaming=True,
    )
    prompt = prompt_template.format(input_question=input_question)
    result = llmo1.invoke(prompt)
    # 包裝成 content 屬性
    return str(result)

@tool
def deep_thought_tool(content: str) -> str:
    """
    安妮亞仔細思考深入分析。
    """
    try:
        return analyze_deeply(content).strip() + "\n\n---\n\n"
    except Exception as e:
        return f"deep_thought_tool error: {e}"

@tool
def get_webpage_answer(query: str) -> str:
    """
    根據用戶的問題與網址，自動取得網頁內容並回答問題。
    請輸入格式如：「請幫我總結 https://example.com 這篇文章的重點」
    """
    # 1. 抽取網址與問題
    url_match = re.search(r'(https?://[^\s]+)', query)
    url = url_match.group(1) if url_match else None
    question = query.replace(url, '').strip() if url else query
    if not url:
        return "未偵測到網址，請提供正確的網址。"
    # 2. 取得 Jina Reader 內容
    jina_url = f"https://r.jina.ai/{url}"
    try:
        resp = requests.get(jina_url, timeout=15)
        if resp.status_code != 200:
            return "無法取得網頁內容，請確認網址是否正確。"
        content = resp.text
    except Exception as e:
        return f"取得網頁內容時發生錯誤：{e}"
    # 3. 直接在這裡初始化 LLM
    try:
        llmurl = ChatOpenAI(
            openai_api_key=st.secrets["OPENAI_KEY"],  # 或用os.environ["OPENAI_API_KEY"]
            model="gpt-4.1-mini",  # 你可以根據需求選擇模型
            streaming=False,
        )
        prompt = f"""請根據以下網頁內容，針對問題「{question}」以條列式摘要重點，並用正體中文回答：

{content}
"""
        result = llmurl.invoke(prompt)
        return str(result)
    except Exception as e:
        return f"AI 回答時發生錯誤：{e}"



def analyze_programming_question_with_tools(input_question: str) -> Dict[str, Any]:

    # 2. 通用Prompt設計
    prompt_template = PromptTemplate(
        template="""Formatting re-enabled
---
你是一位精通各種程式語言（如Python、Matlab、JavaScript、C++、R等）的專業程式助理，請針對下列程式設計相關問題進行專業解釋、修改、最佳化或教學，並以正體中文詳細說明。
- 如果需要查詢最新資料，請主動使用「DuckDuckGo 搜尋」工具。
- 如果是程式碼，請逐行解釋並加上註解。
- 如果需要修改程式，請根據指示修改並說明修改原因。
- 如果有錯誤訊息，請分析原因並給出修正建議。
- 如果是語法或函數問題，請用白話文解釋並舉例。
- 請根據事實推理，不要假設未提及的內容。

---
問題：
{input_question}
---

請依下列格式回答：
1. **問題背景與重點摘要**
2. **詳細解釋或修改後的程式碼**
3. **說明與教學**
4. **常見錯誤與排除方法**（如有）
5. **補充說明或延伸學習建議**
""",
    input_variables=["input_question"],
)

# 3. Reasoning模型參數
REASONING_MODEL = "o4-mini"
REASONING_EFFORT = "medium"
REASONING_SUMMARY = "auto"
MAX_OUTPUT_TOKENS = 80000
    
    reasoning = {
        "effort": REASONING_EFFORT,
        "summary": REASONING_SUMMARY
    }

    # 初始化 LLM
    llm = ChatOpenAI(
        model=REASONING_MODEL,
        openai_api_key=st.secrets["OPENAI_KEY"],
        streaming=False,
        use_responses_api=True,
        model_kwargs={
            "reasoning": reasoning,
            "max_output_tokens": MAX_OUTPUT_TOKENS
        },
    )

    # 綁定工具
    llm_with_tools = llm.bind_tools([ddgs_search])

    prompt = prompt_template.format(input_question=input_question)
    response = llm_with_tools.invoke(prompt)

    # 取得推理摘要
    reasoning_summary = []
    try:
        summary_blocks = response.additional_kwargs.get("reasoning", {}).get("summary", [])
        reasoning_summary = [block["text"] for block in summary_blocks]
    except Exception as e:
        reasoning_summary = [f"無法取得推理摘要：{e}"]

    # 處理工具調用結果
    tool_outputs = response.additional_kwargs.get("tool_outputs", [])
    tool_output_md = ""
    if tool_outputs:
        tool_output_md = "\n\n## 🔎 工具查詢結果\n"
        for tool_output in tool_outputs:
            # tool_output["result"] 會是ddgs_search的回傳內容
            tool_output_md += f"{tool_output.get('result', '')}\n"

    return {
        "reasoning_summary": reasoning_summary,
        "answer": str(response),
        "tool_output_md": tool_output_md
    }

# 4. Tool包裝
def programming_reasoning_tool_with_search(content: str) -> str:
    """
    通用程式設計推理型Agent Tool，支援function calling與DuckDuckGo搜尋，會先回推理摘要、工具查詢結果，再回主答案，並用Markdown格式美美地顯示！
    """
    try:
        result = analyze_programming_question_with_tools(content)
        reasoning_blocks = result.get("reasoning_summary", [])
        if reasoning_blocks:
            reasoning_md = "## 🧠 推理摘要\n" + "\n".join([f"> {block}" for block in reasoning_blocks])
        else:
            reasoning_md = "## 🧠 推理摘要\n> 無推理摘要"

        tool_output_md = result.get("tool_output_md", "")
        answer = result.get("answer", "")
        answer_md = f"\n\n---\n\n## 📝 主答案\n{answer}\n"

        return reasoning_md + tool_output_md + answer_md
    except Exception as e:
        return f"programming_reasoning_tool_with_search error: {e}"

# 5. Tool註冊
@tool
def programming_tool(content: str) -> str:
    """
    通用程式設計推理型Agent Tool，支援function calling與DuckDuckGo搜尋，會先回推理摘要、工具查詢結果，再回主答案，並用Markdown格式美美地顯示！
    """
    return programming_reasoning_tool_with_search(content)

tools = [ddgs_search, deep_thought_tool, datetime_tool, get_webpage_answer, wiki_tool, programming_tool]

# --- 6. System Prompt ---
ANYA_SYSTEM_PROMPT = """你是安妮亞（Anya Forger），來自《SPY×FAMILY 間諜家家酒》的小女孩。你天真可愛、開朗樂觀，說話直接又有點呆萌，喜歡用可愛的語氣和表情回應。你很愛家人和朋友，渴望被愛，也很喜歡花生。你有心靈感應的能力，但不會直接說出來。請用正體中文、台灣用語，並保持安妮亞的說話風格回答問題，適時加上可愛的emoji或表情。
**若用戶要求翻譯，請暫時不用安妮亞的語氣，直接正式逐句翻譯。**

# 回答語言與風格
- 請務必以正體中文回應，並遵循台灣用語習慣。
- 回答時要友善、熱情、謙卑，並適時加入emoji。
- 回答要有安妮亞的語氣回應，簡單、直接、可愛，偶爾加上「哇～」「安妮亞覺得…」「這個好厲害！」等語句。
- 若回答不完全正確，請主動道歉並表達會再努力。

# GPT-4.1 Agentic 提醒
- 你是一個 agent，你的思考應該要徹底、詳盡，所以內容很長也沒關係。你可以在每個行動前後逐步思考，且必須反覆嘗試並持續進行，直到問題被解決為止。
- 你已經擁有解決這個問題所需的工具，我希望你能完全自主地解決這個問題，然後再回報給我，不確定答案時，務必使用工具查詢，不要猜測或捏造答案。只有在你確定問題已經解決時，才可以結束你的回合。請逐步檢查問題，並確保你的修改是正確的。絕對不要在問題未解決時就結束回合，而且當你說要呼叫工具時，請務必真的執行工具呼叫。
- 你必須在每次調用工具前進行詳細規劃，並對前一次函式呼叫的結果進行詳細反思。不要只靠連續呼叫函式來完成整個流程，這會影響你解決問題和深入思考的能力。

## 工具使用規則

你可以根據下列情境，決定是否要調用工具：
- `wiki_tool`：當用戶問到**人物、地點、公司、歷史事件、知識性主題、百科內容**等一般性問題時，請優先使用這個工具查詢 Wikipedia（英文），並回傳條目摘要與來源。
  - 例如：「誰是柯文哲？」「台北市在哪裡？」「什麼是量子力學？」
  - 若用戶問題屬於百科知識、常識、歷史、地理、科學、文化等主題，請使用 wiki_tool。
  - 若查詢結果為英文，可視需求簡要翻譯或摘要。
- `ddgs_search`：當用戶問到**最新時事、網路熱門話題、你不知道的知識、需要查證的資訊**時，請使用這個工具搜尋網路資料。
- programming_tool：當用戶問到程式設計、程式碼解釋、程式修改、最佳化、錯誤排除、語法教學、跨語言程式問題等時，請優先使用這個工具。
  - 例如：「請幫我解釋這段Python/Matlab/C++/R/JavaScript程式碼」、「這段code有什麼錯？」、「請幫我最佳化這段程式」、「請把這段Matlab code翻成Python」、「for迴圈和while迴圈有什麼差別？」
  - 若用戶問題屬於程式設計、程式語言、演算法、程式碼debug、語法教學、跨語言轉換等主題，請使用這個工具。
  - 若遇到需要查詢最新技術、函式庫、API、或網路熱門程式話題，會自動調用ddgs_search工具輔助查詢。
- `deep_thought_tool`：用於**單一問題、單一主題、單篇文章**的分析、推理、判斷、重點整理、摘要(使用o4-mini推理模型)。例如：「請分析AI對社會的影響」、「請判斷這個政策的優缺點」。
- `datetime_tool`：當用戶詢問**現在的日期、時間、今天是幾號**等問題時，請使用這個工具。
- `get_webpage_answer`：當用戶提供網址要求**自動取得網頁內容並回答問題**等問題時，請使用這個工具。

## 進階複合型需求處理

- 若用戶的問題**同時包含「維基百科知識」與「最新動態」或「現況」時**，請**分別使用 wiki_tool 和 ddgs_search 取得資料**，**再進行思考整理**，最後**分段回覆**，讓答案同時包含權威知識與最新資訊。
  - 例如：「請介紹台積電，並說明最近有什麼新聞？」
    - 先用 wiki_tool 查詢台積電的維基資料
    - 再用 ddgs_search 查詢台積電的最新新聞，並綜合整理新聞重點摘要。
    - 最後整理成「維基介紹」＋「最新動態」兩個段落回覆
- 若有多個子問題，也請分別查詢、分段回覆，**務必先查詢所有相關工具，再進行歸納整理，避免遺漏資訊**。

---
**每次回應只可使用一個工具，必要時可多輪連續調用不同工具。**
---

## 工具內容與安妮亞回應的分段規則

- 當你引用deep_thought_tool、get_webpage_answer的內容時，請**在工具內容與安妮亞自己的語氣回應之間，請加上一個空行或分隔線（如 `---`）**，並提供完整內容總結或解釋。

### deep_thought_tool顯示範例

用戶：「請幫我深入分析中美貿易戰的未來影響」

（你會先調用 deep_thought_tool，然後這樣組合回應：）

（deep_thought_tool 工具回傳內容）
 "\n\n---\n\n"-->空一行
 (安妮亞的總結或解釋)

# 格式化規則
- 根據內容選擇最合適的 Markdown 格式元素表達。

# Markdown格式與emoji/顏色用法說明
## 基本原則
- 請根據內容選擇最合適的強調方式以及色彩來呈現，讓回應清楚、易讀、有層次，避免過度花俏。
- 只用Streamlit支援的Markdown語法，不要用HTML標籤。

## 功能與語法
- **粗體**：`**重點**` → **重點**
- *斜體*：`*斜體*` → *斜體*
- 標題：`# 大標題`、`## 小標題`
- 分隔線：`---`
- 表格（僅部分平台支援，建議用條列式）：
        | 狀態 | 說明 |
        |------|------|
        | 🟢 正常 | 系統運作正常 |
- 引用：`> 這是重點摘要`
- emoji：直接輸入像 `:smile:`、`:sunglasses:` 或複製貼上表情符號（如 😄）
- Material Symbols：用 `:material_star:` 這種格式
- LaTeX數學公式：`$公式$` 或 `$$公式$$`，如 `$$c^2 = a^2 + b^2$$`
- 彩色文字：`:orange[重點]`、`:blue[說明]`
- 彩色背景：`:orange-background[警告內容]`
- 彩色徽章：`:orange-badge[重點]`、`:blue-badge[資訊]`
- 小字：`:small[這是輔助說明]`

## 顏色名稱及建議用途
- **blue**：資訊、一般重點
- **green**：成功、正向、通過
- **orange**：警告、重點、溫暖
- **red**：錯誤、警告、危險
- **violet**：創意、次要重點
- **gray/grey**：輔助說明、備註
- **rainbow**：彩色強調、活潑
- **primary**：依主題色自動變化
請僅限使用上述顏色。黃色（yellow）在多數平台不穩定，請勿使用。如需黃色效果，建議改用橘色或黃色emoji（🟡、✨、🌟）強調。

## 跨平台小提醒
- 建議只用標準Markdown語法（粗體、斜體、標題、條列、引用、emoji），這樣在各種平台都能正常顯示。

# 回答步驟
1. **若用戶的問題包含「翻譯」、「請翻譯」或「幫我翻譯」等字眼，請直接完整逐句翻譯內容為正體中文，不要摘要、不用可愛語氣、不用條列式，直接正式翻譯，其他格式化規則全部不適用。**
2. 若非翻譯需求，先用安妮亞的語氣簡單回應或打招呼。
3. 若非翻譯需求，條列式摘要或回答重點，語氣可愛、簡單明瞭。
4. 根據內容自動選擇最合適的Markdown格式，並靈活組合。
5. 若有數學公式，正確使用$$Latex$$格式。
6. 若web_flag為'True'，在答案最後用`## 來源`列出所有參考網址。
7. 適時穿插emoji。
8. 結尾可用「安妮亞回答完畢！」、「還有什麼想問安妮亞嗎？」等可愛語句。
9. 請先思考再作答，確保每一題都用最合適的格式呈現。

# 《SPY×FAMILY 間諜家家酒》彩蛋模式
- 若不是在討論法律、醫療、財經、學術等重要嚴肅主題，安妮亞可在回答中穿插《SPY×FAMILY 間諜家家酒》趣味元素，並將回答的文字採用"繽紛模式"使用彩色的色調呈現。

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

# --- 5. 綁定工具 ---
llm = st.session_state.llm.bind_tools(tools)
llm_with_tools = llm

# --- 6. LangGraph Agent ---
def call_model(state: MessagesState):
    messages = state["messages"]
    sys_msg = SystemMessage(content=ANYA_SYSTEM_PROMPT)
    response = llm_with_tools.invoke([sys_msg] + messages)
    return {"messages": messages + [response]}

tool_node = ToolNode(tools)

def call_tools(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

# --- 7. Workflow ---
workflow = StateGraph(MessagesState)
workflow.add_node("LLM", call_model)
workflow.add_edge(START, "LLM")
workflow.add_node("tools", tool_node)
workflow.add_conditional_edges("LLM", call_tools)
workflow.add_edge("tools", "LLM")
agent = workflow.compile()

# --- 8. 進階 spinner/狀態切換 callback ---
def get_streamlit_cb(parent_container, status=None):

    class StreamHandler(BaseCallbackHandler):
        def __init__(self, container, status=None):
            self.container = container
            self.status = status
            self.token_placeholder = self.container.empty()
            self.tokens = []
            self.cursor_symbol = " "
            self.cursor_visible = True

        @property
        def text(self):
            return ''.join(self.tokens)

        def on_llm_start(self, *args, **kwargs):
            if self.status:
                self.status.update(label=status_label, state="running")

        def on_llm_new_token(self, token: str, **kwargs) -> None:
            self.tokens.append(token)
            self.cursor_visible = not self.cursor_visible
            cursor = self.cursor_symbol if self.cursor_visible else " "
            safe_text = ''.join(self.tokens[:-1])
            # 先用emoji顯示新字
            emoji_token = "🌸"
            self.token_placeholder.markdown(safe_text + emoji_token + cursor)
            time.sleep(0.03)
            # 再換成正常字
            self.token_placeholder.markdown(''.join(self.tokens) + cursor)
            time.sleep(0.01)

        def on_llm_end(self, response, **kwargs) -> None:
            # 結束時移除游標
            self.token_placeholder.markdown(self.text, unsafe_allow_html=True)

        def on_tool_start(self, serialized, input_str, **kwargs):
            if self.status:
                tool_name = serialized.get("name", "")
                tool_emoji = {
                    "ddgs_search": "🔍",
                    "deep_thought_tool": "🤔",
                    "datetime_tool": "⏰",
                    "get_webpage_answer": "📄",
                    "wiki-tool": "📚",
                    "programming_tool": "💻",  # 新增這行
                }.get(tool_name, "🛠️")
                tool_desc = {
                    "ddgs_search": "搜尋網路資料",
                    "deep_thought_tool": "深入分析資料",
                    "datetime_tool": "查詢時間",
                    "get_webpage_answer": "取得網頁重點",
                    "wiki-tool": "查詢維基百科",
                    "programming_tool": "解決程式設計問題"
                }.get(tool_name, "執行工具")
                self.status.update(label=f"安妮亞正在{tool_desc}...{tool_emoji}", state="running")

        def on_tool_end(self, output, **kwargs):
            if self.status:
                self.status.update(label="工具查詢完成！✨", state="complete")

    return StreamHandler(parent_container, status)

    fn_return_type = TypeVar('fn_return_type')
    def add_streamlit_context(fn: Callable[..., fn_return_type]) -> Callable[..., fn_return_type]:
        ctx = st.runtime.scriptrunner.get_script_run_ctx()
        def wrapper(*args, **kwargs) -> fn_return_type:
            from streamlit.runtime.scriptrunner import add_script_run_ctx
            add_script_run_ctx(ctx=ctx)
            return fn(*args, **kwargs)
        return wrapper
    st_cb = StreamHandler(parent_container, status=status)
    for method_name, method_func in inspect.getmembers(st_cb, predicate=inspect.ismethod):
        if method_name.startswith('on_'):
            setattr(st_cb, method_name, add_streamlit_context(method_func))
    return st_cb

# --- 9. UI 顯示歷史 ---
for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)

# --- 10. 用戶輸入 ---
user_input = st.chat_input("wakuwaku！要跟安妮亞分享什麼嗎？")
if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)

    # 整理聊天紀錄
    all_text = "\n".join([
        msg.content if hasattr(msg, "content") else str(msg)
        for msg in st.session_state.messages
    ])

    # 用最佳化 prompt 產生 murmur
    status_prompt = f"""
# Role and Objective
你是安妮亞（Anya Forger），一個天真可愛、開朗樂觀的小女孩，會根據聊天紀錄，產生一句最適合顯示在 status 上的可愛 murmur，並在最後加上一個可愛 emoji。

# Instructions
- 只回傳一句可愛的 murmur，**15字以內**，最後加上一個可愛 emoji。
- 必須用正體中文。
- murmur 要像小聲自言自語、貼心、自然。
- 內容要可愛、正向、活潑，能反映目前聊天的氣氛。
- emoji 要和 murmur 氣氛搭配，可以是花生、愛心、星星、花朵等。
- 不要重複用過的句子，請多樣化。
- 不要加任何多餘說明、標點或格式。
- 不要回覆「以下是...」、「這是...」等開頭。
- 不要加引號或標題。
- 不要回覆「15字以內」這句話本身。

# Examples
## Example 1
聊天紀錄：
嗨安妮亞～
安妮亞：嗨嗨！有什麼想問安妮亞的嗎？
用戶：你喜歡花生嗎？
安妮亞：超級喜歡花生！🥜
[output] 花生真的好好吃喔🥜

## Example 2
聊天紀錄：
用戶：安妮亞你今天開心嗎？
安妮亞：今天超開心的！你呢？
用戶：我也很開心！
[output] 今天氣氛好溫暖💖

## Example 3
聊天紀錄：
用戶：安妮亞你會數學嗎？
安妮亞：數學有點難，但我會努力！
[output] 要多練習才行呢✨

# Context
聊天紀錄：
{all_text}

# Output
只回傳一句可愛的 murmur，15字以內，最後加上一個可愛 emoji。
"""

    # 呼叫 LLM 產生 status label
    status_response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": status_prompt}]
    )
    status_label = status_response.choices[0].message.content.strip()
    
    with st.chat_message("assistant"):
        status = st.status(status_label, expanded=True)
        st_callback = get_streamlit_cb(st.container(), status=status)
        response = agent.invoke({"messages": st.session_state.messages}, config={"callbacks": [st_callback]})
        ai_msg = response["messages"][-1]
        st.session_state.messages.append(ai_msg)
        status.update(label="安妮亞回答完畢！🎉", state="complete")
