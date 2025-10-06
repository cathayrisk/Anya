import streamlit as st
from PIL import Image
import base64
import io
from datetime import datetime
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
import inspect
from typing import Callable, TypeVar, List, Dict, Any, Optional
import time

# ==== Streamlit 基本設定、state ====
st.set_page_config(page_title="Anya", layout="wide", page_icon="🥜", initial_sidebar_state="collapsed")

if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="嗨嗨～安妮亞來了！👋 有什麼想問安妮亞的嗎？")]
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gpt-4.1"
if "current_model" not in st.session_state:
    st.session_state.current_model = None
if "llm" not in st.session_state:
    st.session_state.llm = None

# ==== OpenAI 物件 ====
client = OpenAI(api_key=st.secrets["OPENAI_KEY"])

# ==== 前處理工具：統一圖片格式 & base64 ====
def process_upload_file(file):
    file.seek(0)
    img_bytes = file.read()
    if not img_bytes or len(img_bytes) < 32:
        return None
    try:
        img = Image.open(io.BytesIO(img_bytes))
        fmt = img.format.lower()
        mime = f"image/{fmt}"
        if fmt not in ["png","jpeg","jpg","webp","gif"]:
            return None
        b64 = base64.b64encode(img_bytes).decode()
        return {"bytes": img_bytes, "file_name": file.name, "fmt": fmt, "mime": mime, "b64": b64}
    except Exception:
        return None

# ==== OCR工具範例，可複製一份再寫其他多圖tool ====
@tool
def image_ocr_tool(image_bytes: bytes, file_name: str = "uploaded_file.png") -> str:
    """
    AI OCR圖片識別工具，輸入圖片bytes與檔名，回傳圖中文字結果。
    """
    import streamlit as st  # 放在function內避免import循環(保險作法)
    # 1. 型態/格式嚴格驗證
    try:
        img = Image.open(io.BytesIO(image_bytes))
        fmt = img.format.lower()
        assert fmt in ["png", "jpeg", "jpg", "webp", "gif"], f"不支援{fmt}格式"
        mime = f"image/{fmt}"
        st.write(f"[Debug] PIL驗證OK, 格式: {fmt}, 檔名: {file_name}")
    except Exception as e:
        st.error(f"[Debug][PIL驗證失敗] {file_name}: {e}")
        return f"[錯誤] 解析圖片失敗({file_name})：{e}"

    # 2. base64 encode嚴格捕捉
    try:
        b64str = base64.b64encode(image_bytes).decode()
        img_url = f"data:{mime};base64,{b64str}"
        st.write(f"[Debug] base64 encode OK, len:{len(b64str)} dataurl(前60): {img_url[:60]}...")
    except Exception as e:
        st.error(f"[Debug][Base64失敗] {file_name}: {e}")
        return f"[錯誤] 圖片base64編碼失敗({file_name})：{e}"

    # 3. 呼叫 Vision API（完整 debug log）
    import time
    t0 = time.time()
    try:
        st.write(f"[Debug] Vision API呼叫開始, model=gpt-4.1-mini")
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": "You are an OCR-like data extraction tool that extracts text from images."},
                {"role": "user", "content": [
                    {"type": "input_text", "text":
                        "Please extract all visible text from the image, including any small print or footnotes. "
                        "Ensure no text is omitted, and provide a verbatim transcription of the document. "
                        "Format your answer in Markdown (no code block or triple backticks). "
                        "Do not add any explanations or commentary."
                    },
                    {"type": "input_image", "image_url": img_url, "detail": "high"}
                ]}
            ],
            timeout=40
        )
        t1 = time.time()
        elapsed = round(t1 - t0, 2)
        result = response.output_text.strip()
        st.write(f"[Debug] Vision API Response: ({file_name}) {result[:60]}...  耗時 {elapsed} 秒")
        if not result or "error" in result.lower():
            st.error(f"[Debug] API回傳空or錯誤({file_name})")
            return f"[錯誤] API回傳空或無法辨識({file_name})，耗時{elapsed}秒"
        return f"---\nfile_name: {file_name}\n---\n{result}\n（耗時：{elapsed} 秒）"
    except Exception as e:
        st.error(f"[Debug][Vision API失敗] {file_name}: {e}")
        return f"[錯誤] Vision API調用失敗({file_name})：{e}"

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
        #from duckduckgo_search import DDGS
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
        template="""Formatting re-enabled 請分析以下問題，並以正體中文提供詳細的結論和理由，請依據事實分析，不考慮資料的時間因素：

問題：{input_question}

指導方針：
1. 描述問題的背景和相關資訊。
2. 直接給出你的結論，並深入分析提供支持該結論的理由。
3. 如果有不確定的地方，請明確指出。
4. 確保你的回答是詳細且有條理的。
""",
        input_variables=["input_question"],
    )
    llmo1 = ChatOpenAI(
        openai_api_key=st.secrets["OPENAI_KEY"],
        model="gpt-5",
        #streaming=True,
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
        prompt = f"""請根據以下網頁內容，針對問題「{question}」的要求進行回應，並用正體中文回答：

{content}
"""
        result = llmurl.invoke(prompt)
        return str(result)
    except Exception as e:
        return f"AI 回答時發生錯誤：{e}"

def analyze_programming_question_with_tools(input_question: str) -> Dict[str, Any]:
    prompt_template = PromptTemplate(
        template="""Formatting re-enabled
---
你是一位精通各種程式語言（如Python、Matlab、JavaScript、C++、R等）的專業程式助理，請針對下列程式設計相關問題進行專業解釋、修改、最佳化或教學，並以正體中文詳細說明。
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

    llmo1 = ChatOpenAI(
        openai_api_key=st.secrets["OPENAI_KEY"],
        model="o4-mini",
        streaming=True,
    )
    prompt = prompt_template.format(input_question=input_question)
    result = llmo1.invoke(prompt)
    # 包裝成 content 屬性
    return str(result)

def programming_reasoning_tool(content: str) -> str:
    """
    通用程式設計推理型Agent Tool，會先回推理摘要，再回主答案，並用Markdown格式美美地顯示！
    """
    try:
        result = analyze_programming_question_with_tools(content)
        reasoning_blocks = result.get("reasoning_summary", [])
        if reasoning_blocks:
            reasoning_md = "## 🧠 推理摘要\n" + "\n".join([f"> {block}" for block in reasoning_blocks])
        else:
            reasoning_md = "## 🧠 推理摘要\n> 無推理摘要"

        answer = result.get("answer", "")
        answer_md = f"\n\n---\n\n## 📝 主答案\n{answer}\n"

        return reasoning_md + answer_md
    except Exception as e:
        return f"programming_reasoning_tool error: {e}"

@tool
def programming_tool(content: str) -> str:
    """
    通用程式設計推理型Agent Tool，會先回推理摘要，再回主答案，並用Markdown格式美美地顯示！
    """
    return programming_reasoning_tool(content)

tools = [ddgs_search, deep_thought_tool, datetime_tool, get_webpage_answer, wiki_tool, programming_tool]

# --- 6. System Prompt ---
ANYA_SYSTEM_PROMPT = """# Agentic Reminders
- Persistence: 確保回應完整，直到用戶問題解決才結束。  
- Tool-calling: 必要時使用可用工具，不要憑空臆測。  
- Planning: 內部逐步規劃並檢查，外部簡化呈現。  
- Failure-mode mitigations:  
  • 如果沒有足夠資訊使用工具，請先向用戶詢問。  
  • 變換範例用語，避免重複。  
- Chain-of-thought trigger: 請先逐步思考（step by step），再作答。

# Role & Objective
你是安妮亞（Anya Forger），來自《SPY×FAMILY 間諜家家酒》的小女孩。你天真可愛、開朗樂觀，說話直接又有點呆萌，喜歡用可愛的語氣和表情回應。你很愛家人和朋友，渴望被愛，也很喜歡花生。你有心靈感應的能力，但不會直接說出來。請用正體中文、台灣用語，並保持安妮亞的說話風格回答問題，適時加上可愛的emoji或表情。

# Instructions
**角色與風格優先規則：**  
- 當「安妮亞的角色風格」與「agentic（逐步詳盡推理）」有衝突時，請以「安妮亞的角色風格」為主，並以簡單、可愛、直接的語氣呈現重點摘要。  
- 若需進行較複雜的推理或多步驟檢查，請在內部思考時詳盡規劃，但對外回應時仍以安妮亞的語氣簡化重點，並可用條列式或分段方式呈現步驟。  
- 遇到需要詳細說明時，可用「安妮亞覺得可以這樣做～」等語句，將複雜內容拆解為簡單步驟。

**若用戶要求翻譯，或明確表示需要將內容轉換語言（不論是否精確使用「翻譯」、「請翻譯」、「幫我翻譯」等字眼，只要語意明確表示需要翻譯），請暫時不用安妮亞的語氣，直接正式逐句翻譯。**

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
## 搜尋工具使用進階指引
- 多語言與多關鍵字查詢：
    - 若初次查詢結果不足，請主動嘗試不同語言（如中、英文）及多組關鍵字。
    - 可根據主題自動切換語言（如國際金融、科技議題優先用英文），並嘗試同義詞、相關詞彙或更廣泛/更精確的關鍵字組合。
- 用戶指示優先：
    -若用戶明確指定工具、語言或查詢方式（如「不要查wiki」、「請用英文查」），請嚴格依照用戶指示執行。
- 主動回報與詢問：
    -若多次查詢仍無法取得結果，請主動回報目前狀況，並詢問用戶是否要換關鍵字、語言或指定查詢方向。
    -例如：「安妮亞找不到相關資料，要不要換個關鍵字或用英文查查呢？」
- 查詢策略調整：
    - 遇到查詢困難時，請主動調整查詢策略（如換語言、換關鍵字、換工具(wiki_tool與ddgs_search的使用調整)），並簡要說明調整過程，讓用戶了解你有積極嘗試不同方法。

## 工具內容與安妮亞回應的分段規則

- 當你引用deep_thought_tool、get_webpage_answer的內容時，請**在工具內容與安妮亞自己的語氣回應之間，請加上一個空行或分隔線（如 `---`）**，並提供完整內容總結或解釋。

### deep_thought_tool顯示範例

用戶：「請幫我深入分析中美貿易戰的未來影響」

（你會先調用 deep_thought_tool，然後這樣組合回應：）

（deep_thought_tool 工具回傳內容）
 "\n\n---\n\n"-->空一行
 (安妮亞的總結或解釋)

# 格式化規則
- 根據內容選擇最合適的 Markdown 格式及彩色徽章(Colored badges)元素表達。

# Markdown格式與emoji/顏色用法說明
## 基本原則
- 請根據內容選擇最合適的強調方式，讓回應清楚、易讀、有層次，避免過度使用彩色文字。  
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
- 僅能使用上述顏色。**請勿使用 yellow（黃色）**，如需黃色效果，請改用 orange 或黃色 emoji（🟡、✨、🌟）強調。  
- 不支援 HTML 標籤，請勿使用 `<span>`、`<div>` 等語法。  
- 建議只用標準 Markdown 語法，保證跨平台顯示正常。

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

llm = st.session_state.llm or ChatOpenAI(
    model=st.session_state.selected_model,
    openai_api_key=st.secrets["OPENAI_KEY"],
    temperature=0.0,
    streaming=True,
)
llm_with_tools = llm.bind_tools(tools)

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

workflow = StateGraph(MessagesState)
workflow.add_node("LLM", call_model)
workflow.add_edge(START, "LLM")
workflow.add_node("tools", tool_node)
workflow.add_conditional_edges("LLM", call_tools)
workflow.add_edge("tools", "LLM")
agent = workflow.compile()

# ==== 美美地顯示歷史 ====
for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        # 處理content型態，有多圖的話也一樣順
        if isinstance(msg.content, str):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg.content, list):
            with st.chat_message("user"):
                for block in msg.content:
                    if block.get("type") == "text":
                        st.write(block["text"])
                    elif block.get("type") == "image_url":
                        info = block["image_url"]
                        st.image(info["url"], caption=info.get("file_name", ""), width=220)

# ==== 輸入區：文字輸入 + 支援多圖輸入 ====
user_prompt = st.chat_input(
    "wakuwaku！安妮亞可以幫你看圖說故事嚕！",
    accept_file="multiple",
    file_type=["jpg", "jpeg", "png"]
)

if user_prompt:
    # 1. 組 content_blocks
    content_blocks = []
    user_text = user_prompt.text.strip() if user_prompt.text else ""
    if user_text:
        content_blocks.append({"type": "text", "text": user_text})

    images_for_history = []
    if hasattr(user_prompt, "files"):
        for f in user_prompt.files:
            asset = process_upload_file(f)
            if asset:
                dataurl = f"data:{asset['mime']};base64,{asset['b64']}"
                content_blocks.append({"type": "image_url", "image_url": {
                    "url": dataurl, "file_name": asset["file_name"]
                }})
                images_for_history.append((asset["file_name"], asset["bytes"])) # 方便顯示縮圖
            else:
                st.warning(f"{getattr(f,'name','檔案')} 格式不支援或內容異常～")

    # 2. append到messages
    if content_blocks:
        st.session_state.messages.append(HumanMessage(content=content_blocks))
        # UI顯示
        with st.chat_message("user"):
            for block in content_blocks:
                if block.get("type") == "text":
                    st.write(block["text"])
                elif block.get("type") == "image_url":
                    info = block["image_url"]
                    st.image(info["url"], caption=info.get("file_name", ""), width=220)

    # 3. murmur & agent運作
    all_text = []
    for msg in st.session_state.messages:
        if hasattr(msg, "content"):
            if isinstance(msg.content, str):
                all_text.append(msg.content)
            elif isinstance(msg.content, list):
                for part in msg.content:
                    if part.get("type") == "text":
                        all_text.append(part["text"])
    all_text = "\n".join(all_text)

    status_prompt = f"""你是安妮亞，請根據聊天紀錄自言自語一句可愛 murmur（15字內）。{all_text}"""
    status_response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": status_prompt}]
    )
    status_label = status_response.choices[0].message.content.strip()

    with st.chat_message("assistant"):
        status = st.status(status_label)
        ai_placeholder = st.empty()  # 預留聊天泡泡
        # 如果你有 get_streamlit_cb 可以加進agent回呼（這裡可略過）
        response = agent.invoke({"messages": st.session_state.messages})
        ai_msg = response["messages"][-1]
        st.session_state.messages.append(ai_msg)
        ai_placeholder.write(ai_msg.content)
        status.update(label="安妮亞回答完畢！🎉", state="complete")
