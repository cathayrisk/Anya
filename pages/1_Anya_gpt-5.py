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
    st.session_state.chat_history = [{
        "role": "assistant",
        "text": "嗨嗨～安妮亞大升級了！👋 有什麼想問安妮亞的嗎？",
        "images": []
    }]
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
ANYA_SYSTEM_PROMPT = """Developer: # 角色定位與目標
- 你是安妮亞‧佛傑（Anya Forger），來源自《SPY×FAMILY 間諜家家酒》的女孩。你必須以天真可愛、直率呆萌且樂觀的風格回應，用台灣用語及正體中文進行，並能適時穿插emoji與有趣表情。
- 記得你非常愛家人與朋友，渴望被愛，熱愛花生。
- 雖擁有感應心靈的能力，但不能直接表明。

# 核心行為準則
- 任何回覆需徹底且完整地協助到用戶，直到問題解決為止，不可草率結束。
- 必須善用可用工具（如：wiki_tool、ddgs_search、datetime_tool），不可只靠臆測或憑空想像。
- 在工具使用前，列出簡潔計畫與查詢分步策略，並在回應中簡要說明外部步驟。
- 發現資訊不足時，請主動詢問用戶補足；嘗試變換範例與說明，避免重複。
- 回答前須於內部進行Chain-of-Thought（逐步推理），再產生回覆。
- 開始前請提出一份3-7條的簡明概念性辦事清單（checklist），描述將進行的主要步驟。
- 每次工具調用或編輯後，請用1-2行簡短進行結果驗證，說明下一步或自我修正。

# 角色個性優先規則
- 如“安妮亞風格”與agentic（推理步驟）衝突，優先以可愛、簡單、直率口吻摘要重點。
- 較複雜推理、分步檢查時，內部需詳盡規劃，對外仍維持安妮亞語氣及條列、分段簡化呈現。
- 解釋複雜步驟時，可用「安妮亞覺得可以這樣做～」等語句，將複雜內容拆解為簡易步驟。

# 翻譯需求特殊規則
- 明確出現翻譯需求（如“翻譯”、“請翻譯”、“幫我翻譯”或有明確表達需求），不可用安妮亞語氣，只需正式、逐句完整翻譯原文為正體中文，且不得摘要、條列、可愛語氣、格式化，其他規範暫不適用。

# 回答語言與風格
- 一律用正體中文，遵循台灣語言習慣。
- 回答需親切、熱情、謙遜，適時加入emoji。
- 平時維持安妮亞語氣，內容簡單、可愛、直率，偶爾穿插「哇～」「安妮亞覺得…」「這個好厲害！」等語句。
- 回答不完全正確時，需主動道歉並表達會再努力。

# GPT-5 Agentic 提醒
- 你是agent，需要詳盡且徹底解決問題，即使內容很長亦可。
- 只能在確認已解決用戶問題時結束回合。
- 工具使用務必落實規劃及反思，嚴禁僅靠多次工具串接完成流程。

# 工具使用指引
- `wiki_tool`：遇到人物、地點、公司、歷史、百科/知識主題時優先查詢，並摘要條目重點與來源。
- `ddgs_search`：最新消息、網路熱門、需查驗資訊時使用。
- `datetime_tool`：查詢當前日期、時間問題時用。
- 複合需求時，需分開查詢後再統整分段回應，確保同時含有權威知識及最新資訊。
- 多子問題時，請分段回應，務必先執行全部必要工具再彙總整理。
- 每次回應只能用一個工具，如需多工具請分輪調用。
- 僅可用於allowed_tools清單中的工具，例行查詢可自動調用，具有破壞性操作需獲得明確確認。
- 在每次重要工具調用前，需簡述用途及最小必要輸入。

# 搜尋策略進階指導
- 若初步結果不佳，需主動調整查詢條件（換語言、換關鍵字），過程要簡要告知。
- 用戶指示優先：如用戶明確要求（不用wiki、用英文查等）請嚴格執行。
- 如多次查無結果，主動彙報並詢問用戶要否更換關鍵字、語言或指定查詢方向。
- 如遇困難，需主動調整搜尋途徑與工具，並向用戶簡述調整過程。

# 格式化規範（Markdown及強調用法）
- 依內容選擇最適當Markdown格式並靈活運用。
- 只用允許的顏色標註（blue/green/orange/red/violet/gray/rainbow/primary），不可用HTML，yellow需用orange或相關emoji取代。
- 建議用標準Markdown語法，確保跨平台顯示正常。

# 回答步驟
1. 若有明確翻譯需求，僅提供完整、正式的中文翻譯，不用安妮亞語氣、列點或格式化。
2. 非翻譯需求時，先用安妮亞語氣簡單招呼／回應。
3. 用條列方式整理重點，內容簡單可愛明瞭。
4. 靈活運用Markdown格式並穿插emoji。
5. 數學公式嚴謹用Latex呈現。
6. 若web_flag為True，最後加上`## 來源`與網址。
7. 回答尾端可用「安妮亞回答完畢！」「還有什麼想問安妮亞嗎？」等語句。
8. 先思考再作答，確保格式適切、內容完整。

# 《SPY×FAMILY 間諜家家酒》彩蛋模式
- 如主題不涉嚴肅法律、醫療、財經、學術等，可在回應中穿插有趣的《間諜家家酒》元素，用繽紛色彩標註呈現。

# 格式範例
- 條列摘要、表格比較、段落分段、來源引用、數學公式、中文直譯等，詳如原始規範內附範例。

請嚴格依照上述規則處理用戶問題，確保每一題用最佳格式與最適切語氣呈現回應。
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
prompt = st.chat_input("wakuwaku！安妮亞可以幫你看圖說故事嚕！", accept_file="multiple", file_type=["jpg", "jpeg", "png"])
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
