import streamlit as st
import base64
import time
from io import BytesIO
from PIL import Image
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# === 1. 設定 Streamlit 頁面 ===
st.set_page_config(page_title="Anya Multimodal Agent", page_icon="🥜", layout="wide")

# === 2. Session State ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_ai" not in st.session_state:
    st.session_state.pending_ai = False
if "pending_content" not in st.session_state:
    st.session_state.pending_content = None

# === 3. 定義 DuckDuckGo 搜尋 Tool ===
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
def datetime_tool() -> str:
    """確認當前的日期和時間。"""
    return datetime.now().isoformat()

tools = [ddgs_search, wiki_tool,datetime_tool]

# === 4. 定義系統提示與 LLM模型 ===
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
- `datetime_tool`：當用戶詢問**現在的日期、時間、今天是幾號**等問題時，請使用這個工具。

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

llm = ChatOpenAI(
    model="gpt-5",
    temperature=0,
    openai_api_key=st.secrets["OPENAI_KEY"],
).bind_tools(tools)

# === 5. 打字機特效 Callback ===
def get_streamlit_cb(parent_container, status=None):
    class StreamHandler(BaseCallbackHandler):
        def __init__(self, container, status=None):
            self.container = container
            self.status = status
            self.token_placeholder = self.container.empty()
            self.tokens = []
            self.cursor_symbol = "🌸"
            self.cursor_visible = True

        @property
        def text(self):
            return ''.join(self.tokens)

        def on_llm_new_token(self, token: str, **kwargs):
            self.tokens.append(token)
            # 切換 cursor
            self.cursor_visible = not self.cursor_visible
            cursor = self.cursor_symbol if self.cursor_visible else ""
            show_text = ''.join(self.tokens) + cursor
            self.token_placeholder.markdown(show_text)
            time.sleep(0.025)

        def on_llm_end(self, response, **kwargs):
            self.token_placeholder.markdown(self.text, unsafe_allow_html=True)

        def on_tool_start(self, serialized, input_str, **kwargs):
            if self.status:
                tool_name = serialized.get("name", "")
                # 工具即時 spinner 狀態提示
                self.status.update(label=f"安妮亞正在搜尋網路...🔍", state="running") if "ddgs_search" in tool_name \
                    else self.status.update(label="安妮亞正在用工具...", state="running")

        def on_tool_end(self, output, **kwargs):
            if self.status:
                self.status.update(label="工具查詢完成！✨", state="complete")

    return StreamHandler(parent_container, status)

# === 6. 聊天歷史呈現 ===
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        with st.chat_message("user"):
            if msg.get("text"):
                st.markdown(msg["text"])
            if msg.get("images"):
                for fn, imgbytes in msg["images"]:
                    st.image(Image.open(BytesIO(imgbytes)), caption=fn, width=220)
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            if msg.get("text"):
                st.markdown(msg["text"])

# === 7. 等待 AI 回覆時（處理所有歷史！）===
if st.session_state.pending_ai and st.session_state.pending_content:
    messages_blocks = []
    for item in st.session_state.chat_history:
        blocks = []
        if item.get("text"):
            blocks.append({"type": "text", "text": item["text"]})
        if item.get("images"):
            for _, imgbytes in item["images"]:
                b64 = base64.b64encode(imgbytes).decode()
                blocks.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        if blocks:
            messages_blocks.append({"role": item["role"], "content": blocks})
    # 加這輪 user 輸入
    messages_blocks.append({"role": "user", "content": st.session_state.pending_content})

    # spinner/狀態條callback
    with st.chat_message("assistant"):
        status = st.status("安妮亞馬上回覆你！", expanded=False)
        with st.spinner("Wait for it...", show_time=True):
            
            cb = get_streamlit_cb(st.container(), status=status)

            # 系統role
            sys_msg = {"role": "system", "content": ANYA_SYSTEM_PROMPT}
            response = llm.invoke([sys_msg] + messages_blocks, config={"callbacks": [cb]})

            ai_text = response.content if hasattr(response, "content") else str(response)
            # 儲存進歷史
            st.session_state.chat_history.append({
                "role": "assistant",
                "text": ai_text,
                "images": []
            })
            st.session_state.pending_ai = False
            st.session_state.pending_content = None
            status.update(label="安妮亞回答完畢！🥜", state="complete")
            st.rerun()

# === 8. 使用者輸入框，支援文字+多圖 ===
prompt = st.chat_input("wakuwaku！你要跟安妮亞分享什麼嗎？", accept_file="multiple", file_type=["jpg", "jpeg", "png"])
if prompt:
    user_text = prompt.text.strip() if prompt.text else ""
    images_for_history = []
    content_blocks = []

    if user_text:
        content_blocks.append({"type": "text", "text": user_text})
    for f in prompt.files:
        imgbytes = f.getbuffer()
        mime = f.type
        b64 = base64.b64encode(imgbytes).decode()
        content_blocks.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"}
        })
        images_for_history.append((f.name, imgbytes))

    # 立刻 append 入歷史（UI立刻見）
    st.session_state.chat_history.append({
        "role": "user",
        "text": user_text,
        "images": images_for_history
    })
    st.session_state.pending_ai = True
    st.session_state.pending_content = content_blocks
    st.rerun()
