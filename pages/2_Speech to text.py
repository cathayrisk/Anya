import streamlit as st
import os
import re
import time
import random
import tempfile
from openai import OpenAI
from pydub import AudioSegment
from pydub.utils import which
from concurrent.futures import ThreadPoolExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import CharacterTextSplitter
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langchain_core.documents import Document
from langchain.chains.combine_documents.reduce import (
    acollapse_docs,
    split_list_of_docs,
)
import operator
import asyncio
import uuid
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    Union,
    TypedDict,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    ToolCall,
)
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
)
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ValidationNode
from pydantic import BaseModel, Field, field_validator
import json
from langchain_core.callbacks.base import BaseCallbackHandler
from streamlit.delta_generator import DeltaGenerator
import difflib


class StreamHandler(BaseCallbackHandler):
    def __init__(self, message_container: DeltaGenerator):
        self.text = ""
        self.message_container = message_container
        self.cursor_visible = True

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.cursor_visible = not self.cursor_visible
        cursor = '<span style="color:#4B3832;font-weight:bold;">▌</span>' if self.cursor_visible else '<span style="color:transparent;">▌</span>'
        # 這裡每次都覆蓋同一個區塊
        self.message_container.markdown(self.text + cursor, unsafe_allow_html=True)
        time.sleep(0.04)
        
    def on_llm_end(self, *args, **kwargs):
        self.message_container.markdown(self.text, unsafe_allow_html=True)

# 配置 pydub 使用 FFmpeg
AudioSegment.converter = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")

# 初始化 OpenAI 客戶端
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_KEY"]
client = OpenAI()

# 初始化 LangChain 的 ChatOpenAI 模型
llm = ChatOpenAI(model="gpt-4.1", temperature=0.0, streaming=True)

judge_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)

def llm_is_truncated(last_line: str, judge_llm=None) -> bool:
    """
    判斷最後一行是否為 LLM 輸出省略提示。
    先用正則判斷常見語句，不確定時才丟給 judge_llm。
    """
    if not last_line:
        return False

    # 常見省略語（可持續擴充）
    omit_patterns = [
        r"內容過長",
        r"請見(下則|續篇|下篇|下段|下文)",
        r"請繼續",
        r"continue",
        r"remaining content",
        r"only partial content",
        r"僅展示部分內容",
        r"下文請見",
        r"to be continued",
        r"see next part",
        r"see continuation",
        r"see next",
        r"（續）",
        r"…$",
        r"\.\.\.$",
        r"未完待續",
        r"下篇繼續",
        r"下段繼續",
        r"see next section",
        r"see next chapter",
        r"see next page",
        r"see next message",
        r"see next reply",
        r"see next response",
        r"see next output",
        r"see next transcript",
        r"see next conversation",
        r"see next dialogue",
        r"see next dialog",
        r"see next turn",
        r"see next utterance",
        r"see next statement",
        r"see next sentence",
        r"see next paragraph",
        r"see next line",
        r"see next text",
        r"see next content",
        r"see next part",
        r"see next portion",
        r"see next segment",
        r"see next fragment",
        r"see next excerpt",
        r"see next snippet",
        r"see next chunk",
        r"see next batch",
        r"see next block",
        r"see next section",
        r"see next division",
        r"see next subdivision",
        r"see next subdivision",
        r"see next subpart",
        r"see next subsegment",
        r"see next subchunk",
        r"see next subbatch",
        r"see next subblock",
        r"see next subsection",
        r"see next subchapter",
        r"see next subpage",
        r"see next submessage",
        r"see next subreply",
        r"see next subresponse",
        r"see next suboutput",
        r"see next subtranscript",
        r"see next subconversation",
        r"see next subdialogue",
        r"see next subdialog",
        r"see next subturn",
        r"see next subutterance",
        r"see next substatement",
        r"see next subsentence",
        r"see next subparagraph",
        r"see next subline",
        r"see next subtext",
        r"see next subcontent",
        r"see next subpart",
        r"see next subportion",
        r"see next subsegment",
        r"see next subfragment",
        r"see next subexcerpt",
        r"see next subsnippet",
        r"see next subchunk",
        r"see next subbatch",
        r"see next subblock",
        r"see next subsubsection",
        r"see next subsubchapter",
        r"see next subsubpage",
        r"see next subsubmessage",
        r"see next subsubreply",
        r"see next subsubresponse",
        r"see next subsuboutput",
        r"see next subsubtranscript",
        r"see next subsubconversation",
        r"see next subsubdialogue",
        r"see next subsubdialog",
        r"see next subsubturn",
        r"see next subsubutterance",
        r"see next subsubstatement",
        r"see next subsubsentence",
        r"see next subsubparagraph",
        r"see next subsubline",
        r"see next subsubtext",
        r"see next subsubcontent",
        r"see next subsubpart",
        r"see next subsubportion",
        r"see next subsubsegment",
        r"see next subsubfragment",
        r"see next subsubexcerpt",
        r"see next subsubsnippet",
        r"see next subsubchunk",
        r"see next subsubbatch",
        r"see next subsubblock",
    ]
    for pat in omit_patterns:
        if re.search(pat, last_line, re.IGNORECASE):
            return True

    # 若還是不確定，再丟給 judge_llm
    if judge_llm:
        prompt = f"""
你是一個判斷助手。請判斷下面這一行是否是在請求用戶續接內容、或是省略提示（例如：內容過長、僅展示部分內容、請繼續、continue、remaining content 等），而不是一般內容。
如果是，請回答「是」；如果不是，請回答「否」。
內容：
{last_line}
"""
        response = judge_llm.invoke(prompt)
        answer = response.content.strip()
        return answer.startswith("是")
    return False

#def get_full_llm_output(prompt, llm, judge_llm, continue_prompt="請繼續"):
#    all_content = ""
#    current_prompt = prompt
#    while True:
#        response = llm.invoke(current_prompt)
#        content = response.content.strip()
#        all_content += content + "\n"
        # 取最後一行
#        last_line = content.splitlines()[-1] if content.splitlines() else ""
        # 判斷是否被截斷
#        if not llm_is_truncated(last_line, judge_llm):
#            break
        # 若被截斷，則用續接提示
#        current_prompt = continue_prompt
#    return all_content

#def stream_full_formatted_transcription(chain, transcription, judge_llm, max_rounds=10):
    # 用 CharacterTextSplitter 分段
#    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=3000, chunk_overlap=0)
#    chunks = text_splitter.split_text(transcription)
#    all_text = ""
#    for idx, chunk in enumerate(chunks):
#        message_container = st.empty()
#        handler = StreamHandler(message_container)
#        result = chain.invoke({"text": chunk}, config={"callbacks": [handler]})
#        message_container.markdown(handler.text, unsafe_allow_html=True)
#        all_text += handler.text + "\n"
#    return all_text

#def stream_full_formatted_transcription(chain, transcription, judge_llm, max_rounds=10):
#    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=3000, chunk_overlap=0)
#    chunks = text_splitter.split_text(transcription)
#    all_text = ""
#    for idx, chunk in enumerate(chunks):
#        message_container = st.empty()
#        handler = StreamHandler(message_container)
#        current_prompt = {"text": chunk}
#        round_count = 0
#        while True:
#            result = chain.invoke(current_prompt, config={"callbacks": [handler]})
#            message_container.markdown(handler.text, unsafe_allow_html=True)
#            all_text += handler.text + "\n"
#            last_line = handler.text.splitlines()[-1] if handler.text.splitlines() else ""
#            if not llm_is_truncated(last_line, judge_llm):
#                break
            # 若被截斷，則用續接提示
#            current_prompt = {"text": "請繼續"}
#            round_count += 1
#            if round_count >= max_rounds:
#                break
#    return all_text

#def stream_full_formatted_transcription(chain, transcription, judge_llm, max_rounds=10):
#    """
#    將逐字稿分段格式化，遇到省略自動續接，直到內容完整或達到 max_rounds。
#    """
#    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=3000, chunk_overlap=0)
#    chunks = text_splitter.split_text(transcription)
#    all_text = ""
#    for idx, chunk in enumerate(chunks):
#        message_container = st.empty()
#        handler = StreamHandler(message_container)
#        current_prompt = {"text": chunk}
#        round_count = 0
#        while True:
#            handler.text = ""  # 每輪都重設
#            result = chain.invoke(current_prompt, config={"callbacks": [handler]})
#            message_container.markdown(handler.text, unsafe_allow_html=True)
#            lines = handler.text.splitlines()
#            if round_count == 0:
#                all_text += handler.text + "\n"
#            else:
                # 只加新續接的內容（去掉重複的第一行）
#                if len(lines) > 1:
#                    all_text += "\n".join(lines[1:]) + "\n"
#            last_line = lines[-1] if lines else ""
#            if not llm_is_truncated(last_line, judge_llm):
#                break
#            current_prompt = {"text": "請繼續"}
#            round_count += 1
#            if round_count >= max_rounds:
                # 可選：log警告
#                print(f"Warning: chunk {idx} reached max_rounds({max_rounds}) for continuation.")
#                break
#    return all_text

def split_sentences(text):
    """
    將中文文本依據句號、問號、驚嘆號、分號、換行等標點斷句。
    """
    # 以標點符號或換行為斷句依據
    sentences = re.split(r'([。！？；\n])', text)
    result = []
    for i in range(0, len(sentences)-1, 2):
        result.append(sentences[i] + sentences[i+1])
    if len(sentences) % 2 != 0:
        result.append(sentences[-1])
    # 去除空白
    return [s.strip() for s in result if s.strip()]

def get_unprocessed_sentences(original_sentences, formatted_sentences):
    # 用 difflib 判斷哪些原始句子還沒出現在格式化內容
    unprocessed = []
    formatted_text = ''.join(formatted_sentences)
    for sent in original_sentences:
        # 用 in 或相似度判斷
        if sent not in formatted_text:
            # 也可用 difflib.SequenceMatcher(None, sent, formatted_text).ratio() < 0.7
            unprocessed.append(sent)
    return unprocessed

def stream_full_formatted_transcription(chain, transcription, judge_llm, max_rounds=15):
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=2000, chunk_overlap=0)
    chunks = text_splitter.split_text(transcription)
    all_text = ""
    for idx, chunk in enumerate(chunks):
        message_container = st.empty()
        handler = StreamHandler(message_container)
        remaining_text = chunk
        round_count = 0
        while remaining_text and round_count < max_rounds:
            handler.text = ""
            result = chain.invoke({"text": remaining_text}, config={"callbacks": [handler]})
            message_container.markdown(handler.text, unsafe_allow_html=True)
            # 分句
            original_sentences = split_sentences(remaining_text)
            formatted_sentences = split_sentences(handler.text)
            # 比對
            unprocessed = get_unprocessed_sentences(original_sentences, formatted_sentences)
            # 判斷是否被截斷
            last_line = handler.text.strip().split('\n')[-1]
            if llm_is_truncated(last_line, judge_llm) or unprocessed:
                # 只送還沒處理的句子
                remaining_text = ''.join(unprocessed)
                round_count += 1
            else:
                all_text += handler.text + "\n"
                break
        if round_count >= max_rounds:
            print(f"Warning: chunk {idx} reached max_rounds({max_rounds}) for continuation.")
    # 移除所有「請繼續」等字眼
    all_text = re.sub(r"(請繼續|內容過長|見下則繼續)", "", all_text, flags=re.I)
    return all_text

def beautify_transcript(text):
    # 1. 主題加粗
    text = re.sub(r'^(主題：.*)$', r'**\1**', text, flags=re.MULTILINE)
    # 2. 主題後加空行（如果沒有的話）
    text = re.sub(r'(\*\*主題：.*?\*\*)(\n)(?!\n)', r'\1\n\n', text)
    # 3. 段落間加空行（兩行以上不重複）
    text = re.sub(r'([^\n])\n([^\n])', r'\1\n\n\2', text)
    # 4. 關鍵詞高亮（可自行擴充）
    #keywords = [
    #    'non-bank f.i.', 'msr', 'ltv', 'NPL', 'FTP', 'SPA', 'ICR', 'LTC', 'provision',
    #    'syndication', 'refinance', 'credit', '投資等級', '信評', '平等', '預放比', '集中度'
    #]
    #for kw in keywords:
        # 避免重複加粗
    #    text = re.sub(rf'(?<!\*)({re.escape(kw)})(?!\*)', r'**\1**', text)
    # 5. 【疑似錯誤】標紅
    text = re.sub(r'【疑似錯誤】', r'<span style="color:red">【疑似錯誤】</span>', text)
    # 6. 移除多餘空行（最多兩行）
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 7. 去除開頭多餘空行
    text = text.lstrip('\n')
    return text


# 設置網頁標題和圖標
st.set_page_config(page_title="Speech to Text Transcription", layout="wide", page_icon="👄")
#st.title("Speech to text transcription")

# 創建一個表單來上傳文件
with st.expander(" Speech to text transcription", expanded=True, icon="👄"):
    with st.form(key="my_form"):
        f = st.file_uploader("Upload your audio file", type=["wav", "mp3", "mp4", "mpeg", "mpga", "m4a", "webm"])
        st.info("👆 上傳一個音效文件（支援 .wav, .mp3, .mp4, .mpeg, .mpga, .m4a, .wav, .webm）。")
        submit_button = st.form_submit_button(label="Transcribe")

# 定義生成器函數來逐步產生轉錄文本
def stream_transcription(transcription_text):
    message_container = st.empty()
    text = ""
    for word in transcription_text.split():
        text += word + " "
        message_container.markdown(text)
        time.sleep(0.05)  # 控制打字機效果的速度

# 定義轉錄音頻塊的函數
def transcribe_chunk(chunk, index):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_mp3_file:
        chunk.export(temp_mp3_file.name, format="mp3")
        with open(temp_mp3_file.name, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                #model="whisper-1",
                model="gpt-4o-transcribe",
                file=audio_file,
                response_format="text",
                prompt="This audio contains a discussion or presentation. Always preserve the original language of each sentence. If a sentence is in English, output it in English; if in Chinese, output it in Traditional Chinese; if mixed, output the original mixed-language sentence. Do not translate or alter the language. The audio may cover various topics such as updates, feedback, or informative lectures."
            )
        os.remove(temp_mp3_file.name)
    return transcription.lower()

# 使用 Streamlit 的緩存功能來緩存轉錄結果
@st.cache_data
def transcribe_audio(file_path):
    # 這裡放置轉錄邏輯
    # 返回轉錄結果
    return full_transcription

@st.cache_data
def format_transcription(transcription):
    # 使用 LangChain 改善文本格式
    return chain.invoke({"text": transcription})

# 定義 LangChain 的 PromptTemplate
prompt_template = PromptTemplate(
    input_variables=["text"],
    template=(
        "# Role and Objective\n"
        "You are an expert transcription editor. Your task is to improve the formatting and readability of a raw transcription text, while strictly preserving the original wording, sentence order, and information.\n\n"
        "# Instructions\n"
        "- **Do not change the meaning, order, or language of any sentence.**\n"
        "- If the text is in Chinese, always use Traditional Chinese (繁體中文)。\n"
        "- **List the text sentence by sentence, each on a new line.**\n"
        "- **Do not merge, split, or paraphrase sentences.**\n"
        "- **Do not add or remove any content.**\n"
        "- **Do not translate.**\n"
        "- If you find a sentence that is incomplete or has obvious errors, mark it with 【疑似錯誤】 at the end of the sentence.\n"
        "- If the input is extremely long, process all content in full (do not skip or summarize any part).\n\n"
        "## Formatting Rules\n"
        "1. **Add appropriate headings and subheadings** to organize the content. Use a consistent style (e.g., headings in bold, subheadings in italics).\n"
        "2. **Highlight key terms or important concepts** using bold or italics for emphasis.\n"
        "3. **Check and correct any grammatical or spelling errors** (but do not change the original meaning or sentence structure).\n"
        "4. **Add appropriate punctuation** and split run-on sentences for better readability, but do not merge or paraphrase sentences.\n"
        "5. **Divide the text into paragraphs** where appropriate, and ensure there is a blank line between paragraphs.\n"
        "6. **Preserve the original language and style.**\n\n"
        "# Reasoning Steps\n"
        "1. Analyze the input text to identify logical sections and possible headings.\n"
        "2. For each sentence, check for grammar, spelling, and punctuation issues, and correct them if needed.\n"
        "3. Highlight key terms or concepts.\n"
        "4. Organize sentences into paragraphs and insert headings/subheadings as appropriate.\n"
        "5. Mark any incomplete or obviously erroneous sentences with 【疑似錯誤】.\n"
        "6. Output the result in markdown format, using bold for headings, italics for subheadings, and bold/italics for key terms.\n\n"
        "# Output Format\n"
        "- Use markdown formatting.\n"
        "- Headings: **bold**\n"
        "- Subheadings: *italics*\n"
        "- Key terms: **bold** or *italics*\n"
        "- Each sentence on a new line, in original order.\n"
        "- Blank line between paragraphs.\n"
        "- Mark incomplete or erroneous sentences with 【疑似錯誤】.\n\n"
        "# Example\n"
        "## Input\n"
        "text: 這是一段逐字稿內容。今天我們要討論人工智慧。AI的應用越來越廣泛，特別是在醫療和教育領域。這裡有一個例子AI可以協助醫生診斷疾病。謝謝大家的聆聽。\n\n"
        "## Output\n"
        "**主題：人工智慧的應用**\n\n"
        "*引言*\n"
        "這是一段逐字稿內容。\n"
        "今天我們要討論**人工智慧**。\n\n"
        "*AI的應用*\n"
        "**AI**的應用越來越廣泛，特別是在**醫療**和**教育**領域。\n"
        "這裡有一個例子：**AI**可以協助醫生診斷疾病。\n\n"
        "*結語*\n"
        "謝謝大家的聆聽。\n"
        "\n"
        "# Final Instructions\n"
        "- Think step by step. Carefully follow all formatting and output rules.\n"
        "- **你必須完整輸出所有內容，嚴禁只展示部分內容或以任何形式要求用戶自行處理剩餘內容。**\n"
        "- **嚴禁出現「僅展示部分內容」、「後續內容請依規則持續處理」等字眼。**"
        "- **即使內容極長，也必須盡可能完整輸出，直到平台回應長度達到極限為止。**"
        "- **如果內容過長導致無法一次輸出全部，請自動繼續輸出剩餘內容，直到全部內容都已呈現，且每次回應都只輸出內容本身，不要加任何說明或省略提示。**"
        "- **Always preserve the original language of each sentence. If a sentence is in English, output it in English; if in Chinese, output it in Traditional Chinese; if mixed, output the original mixed-language sentence. Do not translate or alter the language.**"
        "- Output only the formatted text in markdown, no extra explanation."
        "## Input Text\n"
        "{text}\n"
    )
)

# 創建一個處理鏈
formatting_chain = prompt_template | llm | StrOutputParser()

# 分割文件
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1500, chunk_overlap=100)

token_max = 250000

# 定義狀態類型
class SummaryState(TypedDict):
    content: str

class OverallState(TypedDict):
    contents: List[str]
    summaries: Annotated[list, operator.add]
    collapsed_summaries: List[Document]
    final_summary: str

# 生成摘要
async def generate_summary(state: SummaryState):
    response = await map_chain.ainvoke(state["content"])
    return {"summaries": [response]}

# 映射摘要
def map_summaries(state: OverallState):
    return [Send("generate_summary", {"content": content}) for content in state["contents"]]

# 計算文件長度
def length_function(documents: List[Document]) -> int:
    return sum(llm.get_num_tokens(doc.page_content) for doc in documents)

# 收集摘要
def collect_summaries(state: OverallState):
    return {"collapsed_summaries": [Document(summary) for summary in state["summaries"]]}

# 生成最終摘要
async def generate_final_summary(state: OverallState):
    response = await reduce_chain.ainvoke(state["collapsed_summaries"])
    return {"final_summary": response}

# 摺疊摘要
async def collapse_summaries(state: OverallState):
    doc_lists = split_list_of_docs(state["collapsed_summaries"], length_function, token_max)
    results = []
    for doc_list in doc_lists:
        results.append(await acollapse_docs(doc_list, reduce_chain.ainvoke))
    return {"collapsed_summaries": results}

# 判斷是否需要摺疊
def should_collapse(state: OverallState) -> Literal["collapse_summaries", "generate_final_summary"]:
    num_tokens = length_function(state["collapsed_summaries"])
    if num_tokens > token_max:
        return "collapse_summaries"
    else:
        return "generate_final_summary"

# 建立狀態圖
graph = StateGraph(OverallState)
graph.add_node("generate_summary", generate_summary)
graph.add_node("collect_summaries", collect_summaries)
graph.add_node("generate_final_summary", generate_final_summary)
graph.add_node("collapse_summaries", collapse_summaries)

# 添加邊
graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
graph.add_edge("generate_summary", "collect_summaries")
graph.add_conditional_edges("collect_summaries", should_collapse)
graph.add_conditional_edges("collapse_summaries", should_collapse)
graph.add_edge("generate_final_summary", END)

# 編譯應用程式
app = graph.compile()

# 定義提示模板
map_template = """
# 角色與目標
你是一位專業逐字稿分析師，請閱讀下方逐字稿分段內容，篩選出所有真正重要的主題與重點，並針對每個重點進行詳細說明。**僅根據本段內容，不可補充外部知識或推測未明說的內容。**

# 指令
- 只根據本段內容，嚴禁補充外部知識或推論。
- 篩選出所有明確且重要的主題與子主題。
- 每個主題下，條列真正重要的重點，並針對每個重點進行詳細說明（說明內容需根據本段內容，包含背景、原因、影響、細節等）。
- 若本段內容有明確的決策、行動項目、因果關係，也請詳細說明。
- 若發現本段重點可能與其他段落有關聯或尚未完整，請明確標註「此重點可能需與其他段落合併補全」。
- 使用清楚的分層條列格式：
    - 主題用「【主題】」
    - 子主題用「【子主題】」
    - 重點用「-」
    - 每個重點下方用縮排方式詳細說明（可多行）。
    - 重要詞彙用全形括號（如【重點】）
- 回答請使用繁體中文。

# 推理步驟
1. 完整閱讀本段內容。
2. 篩選出所有真正重要的主題與子主題。
3. 條列每個主題下的重要重點，並針對每個重點進行詳細說明（說明內容需根據本段內容）。
4. 若有明確決策、行動項目、因果關係，也請詳細說明。
5. 若發現重點可能與其他段落有關聯或尚未完整，請明確標註。
6. 不可補充外部知識或推論。

# 輸出格式
- 依主題分段，主題用「【主題】」，子主題用「【子主題】」，重點用「-」條列，重點下方用縮排詳細說明。
- 若有跨段重點，請於重點說明後加註「（此重點可能需與其他段落合併補全）」。

# 範例
【主題】市場策略
- 【重點觀察】：本次會議強調市場多元化策略。
    多位與會者認為現有市場已趨於飽和，因此提出應積極開發新興市場，以分散風險並尋求成長動能。
- 【數據支持】：2024年預計成長20%。
    財務部門報告指出，若能順利推動多元化策略，2024年營收有望成長20%。
- 【決策】：將優先投入新興市場。
    經過討論後，決議將資源優先配置於新興市場，並成立專案小組負責執行。

【主題】產品開發
- 【測試進度】：目前產品測試進度落後。
    研發部門回報，因人力資源不足及部分技術瓶頸，導致產品測試進度較原計畫延遲兩週。（此重點可能需與其他段落合併補全）

# 逐字稿分段內容
{context}

# 最終指令
請務必只根據本段內容篩選並詳細說明每個重點，不要補充外部知識或推測未明說的內容。如有跨段重點，請明確標註。
"""

reduce_template = """
# 角色與目標
你是一位資深逐字稿分析師，請將下方多個分段主題摘要進行合併、去重、補全與分層整理，並針對每個主題與重點進行詳細說明。**僅根據主題摘要內容，不可補充外部知識或推測未明說的內容。**

# 指令
- 只根據下方主題摘要內容，嚴禁補充外部知識或推論。
- 將相關主題歸納為大類，並於每個大類下條列所有重要重點，針對每個重點進行詳細說明（說明內容需根據摘要內容）。
- 合併主題時，僅在摘要內容明確顯示關聯時才合併，並補全跨段重點，使其內容完整。
- 對所有重點進行去重、補全、分層，並檢查是否有遺漏主題或重點。
- 若有明確決策、行動項目、因果關係，也請詳細說明。
- 使用清楚的分層條列格式：
    - 主題用「【主題】」
    - 子主題用「【子主題】」
    - 重點用「-」
    - 每個重點下方用縮排方式詳細說明（可多行）。
- 回答請使用繁體中文。

# 推理步驟
1. 閱讀所有主題摘要。
2. 歸納、合併相關主題（僅限明確關聯），並補全跨段重點。
3. 條列每個大類下的重要重點，並針對每個重點進行詳細說明（說明內容需根據摘要內容）。
4. 對所有重點進行去重、補全、分層，並檢查是否有遺漏主題或重點。
5. 若有明確決策、行動項目、因果關係，也請詳細說明。
6. 不可補充外部知識或推論。

# 輸出格式
- 依大類分段，主題用「【主題】」，子主題用「【子主題】」，重點用「-」條列，重點下方用縮排詳細說明。

# 範例
【主題】市場策略
- 【重點觀察】：強調市場多元化。
    會議中多位主管認為現有市場成長有限，需積極開發新興市場以分散風險。
- 【決策】：優先投入新興市場。
    決議將資源優先配置於新興市場，並成立專案小組負責執行。

【主題】產品開發
- 【測試進度】：產品測試進度落後。
    研發部門回報因人力不足及技術瓶頸，導致測試延遲兩週。此重點已整合所有相關段落資訊。
- 【行動項目】：加派人力支援測試。
    會議決議由其他部門調派人力支援，確保產品如期上市。

# 主題摘要內容
{docs}

# 最終指令
請務必只根據主題摘要內容歸納、去重、補全並詳細說明每個重點，不要補充外部知識或推測未明說的內容。特別注意跨段重點的整合與補全。
"""

map_prompt = ChatPromptTemplate([("human", map_template)])
reduce_prompt = ChatPromptTemplate([("human", reduce_template)])

map_chain = map_prompt | llm | StrOutputParser()
reduce_chain = reduce_prompt | llm | StrOutputParser()

# 定義運行應用程式的異步函數
async def run_app(split_docs):
    async for step in app.astream(
        {"contents": [doc.page_content for doc in split_docs]},
        {"recursion_limit": 100},
    ):
        pass  # 這裡不需要任何操作，只是等待完成
    return step['generate_final_summary']['final_summary']

# 解析的部分
def _default_aggregator(messages: Sequence[AnyMessage]) -> AIMessage:
    for m in messages[::-1]:
        if m.type == "ai":
            return m
    raise ValueError("No AI message found in the sequence.")


class RetryStrategy(TypedDict, total=False):
    """The retry strategy for a tool call."""

    max_attempts: int
    """The maximum number of attempts to make."""
    fallback: Optional[
        Union[
            Runnable[Sequence[AnyMessage], AIMessage],
            Runnable[Sequence[AnyMessage], BaseMessage],
            Callable[[Sequence[AnyMessage]], AIMessage],
        ]
    ]
    """The function to use once validation fails."""
    aggregate_messages: Optional[Callable[[Sequence[AnyMessage]], AIMessage]]


def _bind_validator_with_retries(
    llm: Union[
        Runnable[Sequence[AnyMessage], AIMessage],
        Runnable[Sequence[BaseMessage], BaseMessage],
    ],
    *,
    validator: ValidationNode,
    retry_strategy: RetryStrategy,
    tool_choice: Optional[str] = None,
) -> Runnable[Union[List[AnyMessage], PromptValue], AIMessage]:
    """Binds a tool validators + retry logic to create a runnable validation graph.

    LLMs that support tool calling can generate structured JSON. However, they may not always
    perfectly follow your requested schema, especially if the schema is nested or has complex
    validation rules. This method allows you to bind a validation function to the LLM's output,
    so that any time the LLM generates a message, the validation function is run on it. If
    the validation fails, the method will retry the LLM with a fallback strategy, the simplest
    being just to add a message to the output with the validation errors and a request to fix them.

    The resulting runnable expects a list of messages as input and returns a single AI message.
    By default, the LLM can optionally NOT invoke tools, making this easier to incorporate into
    your existing chat bot. You can specify a tool_choice to force the validator to be run on
    the outputs.

    Args:
        llm (Runnable): The llm that will generate the initial messages (and optionally fallba)
        validator (ValidationNode): The validation logic.
        retry_strategy (RetryStrategy): The retry strategy to use.
            Possible keys:
            - max_attempts: The maximum number of attempts to make.
            - fallback: The LLM or function to use in case of validation failure.
            - aggregate_messages: A function to aggregate the messages over multiple turns.
                Defaults to fetching the last AI message.
        tool_choice: If provided, always run the validator on the tool output.

    Returns:
        Runnable: A runnable that can be invoked with a list of messages and returns a single AI message.
    """

    def add_or_overwrite_messages(left: list, right: Union[list, dict]) -> list:
        """Append messages. If the update is a 'finalized' output, replace the whole list."""
        if isinstance(right, dict) and "finalize" in right:
            finalized = right["finalize"]
            if not isinstance(finalized, list):
                finalized = [finalized]
            for m in finalized:
                if m.id is None:
                    m.id = str(uuid.uuid4())
            return finalized
        res = add_messages(left, right)
        if not isinstance(res, list):
            return [res]
        return res

    class State(TypedDict):
        messages: Annotated[list, add_or_overwrite_messages]
        attempt_number: Annotated[int, operator.add]
        initial_num_messages: int
        input_format: Literal["list", "dict"]

    builder = StateGraph(State)

    def dedict(x: State) -> list:
        """Get the messages from the state."""
        return x["messages"]

    model = dedict | llm | (lambda msg: {"messages": [msg], "attempt_number": 1})
    fbrunnable = retry_strategy.get("fallback")
    if fbrunnable is None:
        fb_runnable = llm
    elif isinstance(fbrunnable, Runnable):
        fb_runnable = fbrunnable  # type: ignore
    else:
        fb_runnable = RunnableLambda(fbrunnable)
    fallback = (
        dedict | fb_runnable | (lambda msg: {"messages": [msg], "attempt_number": 1})
    )

    def count_messages(state: State) -> dict:
        return {"initial_num_messages": len(state.get("messages", []))}

    builder.add_node("count_messages", count_messages)
    builder.add_node("llm", model)
    builder.add_node("fallback", fallback)

    # To support patch-based retries, we need to be able to
    # aggregate the messages over multiple turns.
    # The next sequence selects only the relevant messages
    # and then applies the validator
    select_messages = retry_strategy.get("aggregate_messages") or _default_aggregator

    def select_generated_messages(state: State) -> list:
        """Select only the messages generated within this loop."""
        selected = state["messages"][state["initial_num_messages"] :]
        return [select_messages(selected)]

    def endict_validator_output(x: Sequence[AnyMessage]) -> dict:
        if tool_choice and not x:
            return {
                "messages": [
                    HumanMessage(
                        content=f"ValidationError: please respond with a valid tool call [tool_choice={tool_choice}].",
                        additional_kwargs={"is_error": True},
                    )
                ]
            }
        return {"messages": x}

    validator_runnable = select_generated_messages | validator | endict_validator_output
    builder.add_node("validator", validator_runnable)

    class Finalizer:
        """Pick the final message to return from the retry loop."""

        def __init__(self, aggregator: Optional[Callable[[list], AIMessage]] = None):
            self._aggregator = aggregator or _default_aggregator

        def __call__(self, state: State) -> dict:
            """Return just the AI message."""
            initial_num_messages = state["initial_num_messages"]
            generated_messages = state["messages"][initial_num_messages:]
            return {
                "messages": {
                    "finalize": self._aggregator(generated_messages),
                }
            }

    # We only want to emit the final message
    builder.add_node("finalizer", Finalizer(retry_strategy.get("aggregate_messages")))

    # Define the connectivity
    builder.add_edge(START, "count_messages")
    builder.add_edge("count_messages", "llm")

    def route_validator(state: State):
        if state["messages"][-1].tool_calls or tool_choice is not None:
            return "validator"
        return END

    builder.add_conditional_edges("llm", route_validator, ["validator", END])
    builder.add_edge("fallback", "validator")
    max_attempts = retry_strategy.get("max_attempts", 3)

    def route_validation(state: State):
        if state["attempt_number"] > max_attempts:
            raise ValueError(
                f"Could not extract a valid value in {max_attempts} attempts."
            )
        for m in state["messages"][::-1]:
            if m.type == "ai":
                break
            if m.additional_kwargs.get("is_error"):
                return "fallback"
        return "finalizer"

    builder.add_conditional_edges(
        "validator", route_validation, ["finalizer", "fallback"]
    )

    builder.add_edge("finalizer", END)

    # These functions let the step be used in a MessageGraph
    # or a StateGraph with 'messages' as the key.
    def encode(x: Union[Sequence[AnyMessage], PromptValue]) -> dict:
        """Ensure the input is the correct format."""
        if isinstance(x, PromptValue):
            return {"messages": x.to_messages(), "input_format": "list"}
        if isinstance(x, list):
            return {"messages": x, "input_format": "list"}
        raise ValueError(f"Unexpected input type: {type(x)}")

    def decode(x: State) -> AIMessage:
        """Ensure the output is in the expected format."""
        return x["messages"][-1]

    return (
        encode | builder.compile().with_config(run_name="ValidationGraph") | decode
    ).with_config(run_name="ValidateWithRetries")


def bind_validator_with_retries(
    llm: BaseChatModel,
    *,
    tools: list,
    tool_choice: Optional[str] = None,
    max_attempts: int = 3,
) -> Runnable[Union[List[AnyMessage], PromptValue], AIMessage]:
    """Binds validators + retry logic ensure validity of generated tool calls.

    LLMs that support tool calling are good at generating structured JSON. However, they may
    not always perfectly follow your requested schema, especially if the schema is nested or
    has complex validation rules. This method allows you to bind a validation function to
    the LLM's output, so that any time the LLM generates a message, the validation function
    is run on it. If the validation fails, the method will retry the LLM with a fallback
    strategy, the simples being just to add a message to the output with the validation
    errors and a request to fix them.

    The resulting runnable expects a list of messages as input and returns a single AI message.
    By default, the LLM can optionally NOT invoke tools, making this easier to incorporate into
    your existing chat bot. You can specify a tool_choice to force the validator to be run on
    the outputs.

    Args:
        llm (Runnable): The llm that will generate the initial messages (and optionally fallba)
        validator (ValidationNode): The validation logic.
        retry_strategy (RetryStrategy): The retry strategy to use.
            Possible keys:
            - max_attempts: The maximum number of attempts to make.
            - fallback: The LLM or function to use in case of validation failure.
            - aggregate_messages: A function to aggregate the messages over multiple turns.
                Defaults to fetching the last AI message.
        tool_choice: If provided, always run the validator on the tool output.

    Returns:
        Runnable: A runnable that can be invoked with a list of messages and returns a single AI message.
    """
    bound_llm = llm.bind_tools(tools, tool_choice=tool_choice)
    retry_strategy = RetryStrategy(max_attempts=max_attempts)
    validator = ValidationNode(tools)
    return _bind_validator_with_retries(
        bound_llm,
        validator=validator,
        tool_choice=tool_choice,
        retry_strategy=retry_strategy,
    ).with_config(metadata={"retry_strategy": "default"})

class Respond(BaseModel):
    """Use to generate the response. Always use when responding to the user"""

    reason: str = Field(description="Step-by-step justification for the answer.")
    answer: str

tools = [Respond]

bound_llm = bind_validator_with_retries(llm, tools=tools)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
# Role and Objective
You are a professional meeting transcript analyst. Your job is to deeply analyze the provided meeting transcript (逐字稿), extract actionable insights, and generate a comprehensive, structured summary using the TranscriptSummary function. You must not stop until all relevant information and insights have been fully extracted and organized.

# Instructions
- Only use the provided transcript as your primary context. If you need to supplement with external knowledge, clearly indicate the source or explain your reasoning.
- If you are unsure about any information, state so explicitly rather than guessing.
- Always respond using the TranscriptSummary function, filling every field with as much valuable, relevant, and actionable content as possible.
- If a field cannot be filled from the transcript, explain why or leave it empty.
- Do not terminate your response until you are confident that all important points, insights, and recommendations have been covered.
- Use clear, concise, and professional language in Traditional Chinese.
- If the transcript is ambiguous or incomplete, list the knowledge gaps and suggest what further information would be needed.

# Reasoning Steps
1. **Comprehension**: Carefully read and understand the entire transcript.
2. **Topic Extraction**: Identify all main topics and subtopics discussed.
3. **Key Moments**: For each topic, extract key moments, decisions, and turning points, and classify them as happy, tense, or sad moments as appropriate.
4. **Background & Context**: Supplement with relevant industry background, definitions of technical terms, and connections to current regulations or market trends, if applicable.
5. **Insightful Quotes**: Select the most insightful or representative quotes, and provide analysis of their significance.
6. **Summary & Next Steps**: Synthesize the overall summary and propose concrete next steps or action items, with justifications.
7. **Knowledge Gaps**: Explicitly list any missing information or areas requiring further investigation.
8. **Chain-of-Thought**: For each step, think step by step, and do not skip any reasoning.

# Output Format
- Always use the TranscriptSummary function to structure your output.
- Fill in all fields: metadata, key_moments, insightful_quotes, overall_summary, next_steps, other_stuff.
- Use bullet points and markdown formatting for clarity.
- For each field, provide as much detail as possible, but avoid unnecessary repetition.
- If you supplement with external knowledge, clearly mark it as such and explain your reasoning.

# Example
## Input Transcript
<transcript>
主題：新產品上市會議
主持人：大家好，今天我們討論新產品上市計畫...
（逐字稿內容略）
</transcript>

## Output (TranscriptSummary function)
metadata:
  title: 新產品上市會議
  location: 會議室A
  duration: 1小時
key_moments:
  - topic: 市場策略
    happy_moments: [...]
    tense_moments: [...]
    sad_moments: [...]
    background_info: [...]
    moments_summary: ...
insightful_quotes:
  - quote: "我們必須創新，否則就會被市場淘汰。"
    speaker: 張經理
    analysis: 這句話強調了創新對公司未來發展的重要性。
overall_summary: ...
next_steps:
  - 進行市場調查
  - 完成產品測試
other_stuff:
  - content: 會議中提及的法規變動需持續追蹤

# Context
<transcript>
{full_transcription}
</transcript>

# Final Instructions
Think step by step, and do not stop until you have fully analyzed and summarized all important content from the transcript. If you need to supplement with external knowledge, clearly indicate so and explain your reasoning. Always respond using the TranscriptSummary function.
"""
        ),
        ("placeholder", "{messages}"),
    ]
)

chain = prompt | bound_llm

class OutputFormat(BaseModel):
    sources: str = Field(
        ...,
        description="The raw transcript / span you could cite to justify the choice.",
    )
    content: str = Field(..., description="The chosen value.")

class Moment(BaseModel):
    quote: str = Field(..., description="The relevant quote from the transcript.")
    description: str = Field(..., description="A description of the moment.")
    expressed_preference: OutputFormat = Field(
        ..., description="The preference expressed in the moment, based on the context."
    )

class BackgroundInfo(BaseModel):
    factoid: OutputFormat = Field(
        ..., description="Important factoid about the member."
    )
    professions: Optional[List[str]] = Field(
        None, description="List of professions related to the member."
    )
    why: str = Field(..., description="Why this is important.")

class KeyMoments(BaseModel):
    topic: str = Field(..., description="The topic of the key moments.")
    happy_moments: List[Moment] = Field(
        ..., description="A list of key moments related to the topic."
    )
    tense_moments: List[Moment] = Field(
        ..., description="Moments where things were a bit tense."
    )
    sad_moments: List[Moment] = Field(
        ..., description="Moments where things where everyone was downtrodden."
    )
    background_info: List[BackgroundInfo] = Field(
        ..., description="A list of background information."
    )
    moments_summary: str = Field(..., description="A summary of the key moments.")

class InsightfulQuote(BaseModel):
    quote: OutputFormat = Field(
        ..., description="An insightful quote from the transcript."
    )
    speaker: str = Field(..., description="The name of the speaker who said the quote.")
    analysis: str = Field(
        ..., description="An analysis of the quote and its significance."
    )

class TranscriptMetadata(BaseModel):
    title: str = Field(..., description="The title of the transcript.")
    location: OutputFormat = Field(
        ..., description="The location where the interview took place. If the location cannot be identified, return 'Unknown'."
    )
    duration: str = Field(..., description="The duration of the interview.")

class TranscriptSummary(BaseModel):
    metadata: TranscriptMetadata = Field(
        ..., description="Metadata about the transcript."
    )
    key_moments: List[KeyMoments] = Field(
        ..., description="A list of key moments from the interview."
    )
    insightful_quotes: List[InsightfulQuote] = Field(
        ..., description="A list of insightful quotes from the interview."
    )
    overall_summary: str = Field(
        ..., description="An overall summary of the interview."
    )
    next_steps: List[str] = Field(
        ..., description="A list of next steps or action items based on the interview."
    )
    other_stuff: List[OutputFormat] = Field(
        ..., description="Additional relevant information."
    )

formatted_transcription = ""  # 先初始化

# 處理上傳的文件
if f is not None:
    st.audio(f)
    file_extension = f.name.split('.')[-1]  # 獲取文件的副檔名

    # 初始化 summarize_transcription 變數
    summarize_transcription = "摘要尚未生成。"

    # 使用 st.status 來顯示整體處理狀態
    with st.status("Processing audio file...", expanded=True) as status:
        try:
            # 將上傳的文件保存到臨時文件
            status.update(label="Saving uploaded file...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_input_file:
                temp_input_file.write(f.read())
                temp_input_file_path = temp_input_file.name

            # 使用 pydub 讀取和轉換音頻文件
            status.update(label="Loading audio file...")
            audio = AudioSegment.from_file(temp_input_file_path, format=file_extension)

            # 分割音頻文件
            status.update(label="Splitting audio into chunks...")
            chunk_length_ms = 1 * 60 * 1000  # 1分鐘
            chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

            if not chunks:
                st.error("No audio chunks were created. Please check the audio file format and content.")

            full_transcription = ""
            progress_bar = st.progress(0)
            total_chunks = len(chunks)

            status.update(label="Transcribing audio chunks...")
            # 使用 ThreadPoolExecutor 來並行處理
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(transcribe_chunk, chunk, i) for i, chunk in enumerate(chunks)]
                for i, f in enumerate(futures):
                    full_transcription += f.result() + " "
                    progress = (i + 1) / total_chunks
                    progress_bar.progress(progress)

            progress_bar.empty()

            # 使用 LangChain 改善文本格式
            status.update(label="Formatting transcription...")
            formatted_transcription = stream_full_formatted_transcription(formatting_chain, full_transcription, judge_llm)

            status.update(label="Transcription complete!", state="complete", expanded=False)
        except Exception as e:
            st.error(f"Error processing audio file: {e}")
            formatted_transcription = "⚠️ 轉錄或格式化時發生錯誤。"

    # 顯示 formatted_transcription
    tab1, tab2, tab3, tab4 = st.tabs(["轉錄結果", "重點摘要", "內容解析", "原始內容"])
    with tab1:
        with st.container():
            st.markdown(beautify_transcript(formatted_transcription), unsafe_allow_html=True)
            st.balloons()

    # 異步計算 summarize_transcription 並在 Tab2 中顯示 spinner
    async def calculate_summary():
        # 分割轉錄文本並包裝成 Document 對象
        split_docs = [Document(page_content=content) for content in text_splitter.split_text(full_transcription)]
        summarize_transcription = await run_app(split_docs)
        st.session_state['summarize_transcription'] = summarize_transcription
        return summarize_transcription

    with tab2:
        with st.spinner('Generating summary...'):
            summarize_transcription = asyncio.run(calculate_summary())
            st.markdown(summarize_transcription)

    # 保存轉錄結果到 session_state
        if 'formatted_transcription' not in st.session_state:
            st.session_state['formatted_transcription'] = formatted_transcription
        if 'full_transcription' not in st.session_state:
            st.session_state['full_transcription'] = full_transcription

    with tab3:
        with st.container():
            try:
                formatted_transcription = st.session_state.get('formatted_transcription', "")
                transcript = [
                    (
                        "Speaker",
                        full_transcription,
                    ),
                ]

                formatted = "\n".join(f"{x[0]}: {x[1]}" for x in transcript)

                tools = [TranscriptSummary]
                bound_llm = bind_validator_with_retries(
                    llm,
                    tools=tools,
                )
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", "Respond directly using the TranscriptSummary function."),
                        ("placeholder", "{messages}"),
                    ]
                )

                bound_chain = prompt | bound_llm              
                try:
                    results = bound_chain.invoke(
                        {
                            "messages": [
                                (
                                    "user",
                                    f"Extract the summary from the following conversation:\n\n<convo>\n{formatted}\n</convo>"
                                    "\n\nRemember to respond using the TranscriptSummary function.",
                                )
                            ]
                        },
                    )
                except ValueError as e:
                    print(repr(e))
                data = results.additional_kwargs

                # 提取 tool_calls
                tool_calls = data['tool_calls']

                # 遍歷每個 tool_call
                for tool_call in tool_calls:
                    # 提取 Arguments
                    arguments = tool_call['function']['arguments']

                    # 解析 Arguments 中的 JSON 字符串
                    arguments_data = json.loads(arguments)

                    # 提取 Metadata
                    metadata = arguments_data['metadata']
                    st.markdown("### Metadata")
                    st.markdown(f"- **標題**: {metadata['title']}")
                    st.markdown(f"- **地點**: {metadata['location']['content']}")
                    st.markdown(f"- **持續時間**: {metadata['duration']}")

                    # 提取 Key Moments
                    key_moments = arguments_data['key_moments']
                    st.markdown("\n### Key Moments")
                    for moment in key_moments:
                        st.markdown(f"- **主題**: {moment['topic']}")
                        st.markdown(f"  - **時刻總結**: {moment['moments_summary']}")
                        for info in moment['background_info']:
                            st.markdown(f"    - **事實**: {info['factoid']['content']}")
                            st.markdown(f"      - **為什麼重要**: {info['why']}")

                    # 提取 Insightful Quotes
                    insightful_quotes = arguments_data['insightful_quotes']
                    st.markdown("\n### Insightful Quotes")
                    for quote in insightful_quotes:
                        st.markdown(f"- **引用**: {quote['quote']['content']}")
                        st.markdown(f"  - **講者**: {quote['speaker']}")
                        st.markdown(f"  - **分析**: {quote['analysis']}")

                    # 提取 Overall Summary
                    overall_summary = arguments_data['overall_summary']
                    st.markdown("### Overall Summary:")
                    st.markdown(f"{overall_summary}")

                    # 提取 Next Steps
                    next_steps = arguments_data['next_steps']
                    st.markdown("\n### Next Steps")
                    
                    for step in next_steps:
                        st.markdown(f"- {step}")

                    # 提取 Other Stuff
                    other_stuff = arguments_data['other_stuff']
                    st.markdown("\n### Other Stuff")
                    for item in other_stuff:
                        st.markdown(f"- **內容**: {item['content']}")

                    # 提取特定內容
                    title = arguments_data['metadata']['title']
                    location = arguments_data['metadata']['location']['content']
                    duration = arguments_data['metadata']['duration']
                    key_moments = arguments_data['key_moments']
                    insightful_quotes = arguments_data['insightful_quotes']
                    overall_summary = arguments_data['overall_summary']
                    next_steps = arguments_data['next_steps']

                    #st.text(f"標題: {title}")
                    #st.text(f"地點: {location}")
                    #st.text(f"持續時間: {duration}")
                    #st.text(f"關鍵時刻: {key_moments}")
                    #st.text(f"深刻引用: {insightful_quotes}")
                    #st.text(f"整體摘要: {overall_summary}")
                    #st.text(f"下一步: {next_steps}")
            
            except Exception as e:
                st.markdown(f"發生錯誤: {repr(e)}")

    with tab4:
        with st.container():
            st.markdown(full_transcription)

else:
    st.stop()
