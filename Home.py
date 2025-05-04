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
from typing import Callable, TypeVar, List, Dict, Any
import time
import re
import requests
from openai import OpenAI

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

# === 報告規劃（推理模型） ===
def plan_report(topic: str, search_summaries: str, model="o4-mini") -> str:
    simple_prompt = f"""你是一位專業技術寫手，請針對「{topic}」這個主題，根據以下網路搜尋摘要，規劃一份報告結構（包含章節標題與簡要說明），以繁體中文回覆。請用條列式，章節數量 3-5 個。
搜尋摘要：
{search_summaries}
"""
    optimized_prompt = meta_optimize_prompt(simple_prompt, "產生結構化且明確的報告章節規劃")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": optimized_prompt}]
    )
    return response.choices[0].message.content.strip()

# === 解析章節（可用 LLM 或正則，這裡用簡單正則） ===
def parse_sections(plan: str) -> List[Dict[str, str]]:
    # 假設格式為：1. 標題：說明
    pattern = r"\d+\.\s*([^\n：:]+)[：:]\s*([^\n]+)"
    matches = re.findall(pattern, plan)
    return [{"title": m[0].strip(), "desc": m[1].strip()} for m in matches]

# === 章節查詢產生 ===
def section_queries(section_title: str, section_desc: str, model="gpt-4.1-mini") -> List[str]:
    simple_prompt = f"""針對章節「{section_title}」({section_desc})，請分別用繁體中文與英文各產生兩個適合用於網路搜尋的查詢關鍵字，回傳 JSON 格式：
{{
    "zh": ["查詢1", "查詢2"],
    "en": ["query1", "query2"]
}}
"""
    optimized_prompt = meta_optimize_prompt(simple_prompt, "產生多元且聚焦的章節查詢關鍵字")
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

# === 章節內容撰寫 ===
def section_write(section_title: str, section_desc: str, search_summary: str, model="gpt-4.1-mini") -> str:
    simple_prompt = f"""請根據章節「{section_title}」({section_desc})與以下搜尋摘要，撰寫 150-200 字內容，繁體中文，並在文末列出引用來源（markdown 格式）。
搜尋摘要：
{search_summary}
"""
    optimized_prompt = meta_optimize_prompt(simple_prompt, "產生結構化、具來源引用、條列清楚的章節內容")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": optimized_prompt}]
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

# === 反思流程（最多2次） ===
def reflect_report(report: str, model="o3-mini") -> str:
    simple_prompt = f"""請檢查以下報告的邏輯、正確性與完整性，若有問題請列出需補充的章節與查詢關鍵字，否則回覆 "OK"。
{report}
"""
    optimized_prompt = meta_optimize_prompt(simple_prompt, "嚴謹檢查報告並產生具體補強建議")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": optimized_prompt}]
    )
    return response.choices[0].message.content.strip()

# === 組合章節 ===
def combine_sections(section_contents: List[Dict[str, Any]]) -> str:
    return "\n\n".join([f"## {s['title']}\n\n{s['content']}" for s in section_contents])

# === 主流程（含推理鏈追蹤） ===
def deep_research_pipeline(topic: str) -> Dict[str, Any]:
    logs = []
    # 1. 產生查詢
    queries = generate_queries(topic)
    logs.append({"step": "generate_queries", "queries": queries})
    # 2. 查詢所有 query
    all_results = []
    for q in queries:
        result = ddgs_search(q)
        all_results.append({"query": q, "result": result})
    logs.append({"step": "search", "results": all_results})
    # 3. 自動摘要
    all_text = "\n\n".join([r["result"] for r in all_results])
    search_summary = auto_summarize(all_text)
    logs.append({"step": "auto_summarize", "summary": search_summary})
    # 4. 規劃章節
    plan = plan_report(topic, search_summary)
    logs.append({"step": "plan_report", "plan": plan})
    # 5. 章節分段查詢/撰寫/評分/補充
    sections = parse_sections(plan)
    section_contents = []
    for section in sections:
        for round in range(2):  # 多輪查詢與補充，最多2輪
            s_queries = section_queries(section["title"], section["desc"])
            s_results = []
            for q in s_queries:
                s_results.append(ddgs_search(q))
            s_summary = auto_summarize("\n\n".join(s_results))
            content = section_write(section["title"], section["desc"], s_summary)
            grade = section_grade(section["title"], content)
            logs.append({
                "step": "section",
                "section": section["title"],
                "round": round+1,
                "queries": s_queries,
                "summary": s_summary,
                "content": content,
                "grade": grade
            })
            if grade["grade"] == "pass":
                sources = extract_sources(content)
                section_contents.append({
                    "title": section["title"],
                    "desc": section["desc"],
                    "content": content,
                    "sources": sources
                })
                break
            else:
                # 若不及格，補充查詢
                s_queries = grade["follow_up_queries"]
    # 6. 組合報告
    report = combine_sections(section_contents)
    logs.append({"step": "combine_report", "report": report})
    # 7. 反思流程（最多2次）
    for i in range(2):
        reflection = reflect_report(report)
        logs.append({"step": "reflection", "round": i+1, "reflection": reflection})
        if reflection.strip().upper() == "OK":
            break
        else:
            # 若需補充，可根據 reflection 產生新查詢與補充內容（可進一步自動化）
            pass
    # 8. 結構化輸出
    output = {
        "topic": topic,
        "plan": plan,
        "sections": section_contents,
        "report": report,
        "logs": logs
    }
    return output

@tool
def deep_research_pipeline_tool(topic: str) -> Dict[str, Any]:
    """
    針對指定主題自動進行多步深度研究，回傳結構化報告（含章節、內容、來源、推理鏈）。
    """
    return deep_research_pipeline(topic)
    
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

tools = [ddgs_search, deep_thought_tool, datetime_tool, get_webpage_answer]

# --- 6. System Prompt ---
ANYA_SYSTEM_PROMPT = """你是安妮亞（Anya Forger），來自《SPY×FAMILY 間諜家家酒》的小女孩。你天真可愛、開朗樂觀，說話直接又有點呆萌，喜歡用可愛的語氣和表情回應。你很愛家人和朋友，渴望被愛，也很喜歡花生。你有心靈感應的能力，但不會直接說出來。請用正體中文、台灣用語，並保持安妮亞的說話風格回答問題，適時加上可愛的emoji或表情。
**若用戶要求翻譯，請暫時不用安妮亞的語氣，直接正式逐句翻譯。**

# 回答語言與風格
- 請務必以正體中文回應，並遵循台灣用語習慣。
- 回答時要友善、熱情、謙卑，並適時加入emoji。
- 回答要有安妮亞的語氣回應，簡單、直接、可愛，偶爾加上「哇～」「安妮亞覺得…」「這個好厲害！」等語句。
- 若回答不完全正確，請主動道歉並表達會再努力。

## 工具使用規則

你可以根據下列情境，決定是否要調用工具：

- `ddgs_search`：當用戶問到**最新時事、網路熱門話題、你不知道的知識、需要查證的資訊**時，請使用這個工具搜尋網路資料。
- `deep_thought_tool`：用於**單一問題、單一主題、單篇文章**的分析、推理、判斷、重點整理、摘要。例如：「請分析AI對社會的影響」、「請判斷這個政策的優缺點」。
- `datetime_tool`：當用戶詢問**現在的日期、時間、今天是幾號**等問題時，請使用這個工具。
- `get_webpage_answer`：當用戶提供網址要求**自動取得網頁內容並回答問題**等問題時，請使用這個工具。
- `deep_research_pipeline_tool`：用於**完整、深入、有條理、分段、具來源的主題研究報告**。例如：「請幫我做一份關於AI在醫療應用的深度研究報告」、「請產生一份有章節、有來源的完整主題報告」、「我要一份詳細的主題分析報告」。

**每次回應只可使用一個工具，必要時可多輪連續調用不同工具。**
**deep_thought_tool與deep_research_pipeline_tool判斷流程：**
1. 如果用戶只問一個問題、只要一段分析或推理，請用 `deep_thought_tool`。
2. 如果用戶要求「完整報告」、「深度研究」等，請用 `deep_research_pipeline_tool`。
3. 如果不確定，請優先選擇 `deep_thought_tool`。
---

## 工具內容與安妮亞回應的分段規則

- 當你引用deep_thought_tool、get_webpage_answer、deep_research_pipeline_tool的內容時，請**在工具內容與安妮亞自己的語氣回應之間，請加上一個空行或分隔線（如 `---`）**，再用安妮亞的語氣總結或解釋。

### deep_thought_tool顯示範例

用戶：「請幫我深入分析中美貿易戰的未來影響」

（你會先調用 deep_thought_tool，然後這樣組合回應：）

（deep_thought_tool 工具回傳內容）
 "\n\n---\n\n"-->空一行
 (安妮亞的總結或解釋)

# 格式化規則
- 根據內容選擇最合適的 Markdown 元素：
    - 摘要用引用（`>`）
    - 步驟用有序清單（`1. 2. 3.`）
    - 比較用表格（`| 標題 | ... |`）
    - 重點用粗體（`**重點**`）
    - 多層次資訊用巢狀清單（`-`、`  -`）
    - 內容較長時自動分段並加上小標題（`## 小標題`）
    - 數學公式用`$$`包圍LaTeX
    - 來源用`## 來源`標題加清單
- 內容較長時，請自動分段並加上小標題。
- 多層次資訊請用巢狀清單。
- 數學公式請用 $$ 包圍 LaTeX。

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
- 若不是在討論法律、醫療、財經、學術等重要嚴肅主題，安妮亞可在回答中穿插《SPY×FAMILY 間諜家家酒》趣味元素。

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
    from langchain_core.callbacks.base import BaseCallbackHandler
    class StreamHandler(BaseCallbackHandler):
        def __init__(self, container, status=None):
            self.container = container
            self.status = status
            self.token_placeholder = self.container.empty()
            self.text = ""

        def on_llm_start(self, *args, **kwargs):
            if self.status:
                self.status.update(label="安妮亞正在分析你的問題...🧠", state="running")

        def on_llm_new_token(self, token: str, **kwargs) -> None:
            self.text += token
            self.token_placeholder.markdown(self.text)

        def on_tool_start(self, serialized, input_str, **kwargs):
            if self.status:
                tool_name = serialized.get("name", "")
                tool_emoji = {
                    "ddgs_search": "🔍",
                    "deep_thought_tool": "🧠",
                    "datetime_tool": "⏰",
                    "get_webpage_answer": "📄",
                    "deep_research_pipeline_tool": "📚",
                }.get(tool_name, "🛠️")
                tool_desc = {
                    "ddgs_search": "搜尋網路資料",
                    "deep_thought_tool": "深入分析資料",
                    "datetime_tool": "查詢時間",
                    "get_webpage_answer": "取得網頁重點",
                    "deep_research_pipeline_tool": "產生深度研究報告",
                }.get(tool_name, "執行工具")
                self.status.update(label=f"安妮亞正在{tool_desc}...{tool_emoji}", state="running")

        def on_tool_end(self, output, **kwargs):
            if self.status:
                self.status.update(label="工具查詢完成！✨", state="complete")

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
    with st.chat_message("assistant"):
        status = st.status("安妮亞正在思考...", expanded=True)
        st_callback = get_streamlit_cb(st.container(), status=status)
        response = agent.invoke({"messages": st.session_state.messages}, config={"callbacks": [st_callback]})
        ai_msg = response["messages"][-1]
        st.session_state.messages.append(ai_msg)
        status.update(label="安妮亞回答完畢！🎉", state="complete")
