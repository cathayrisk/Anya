import streamlit as st
import base64
import time
from io import BytesIO
from PIL import Image
from datetime import datetime
from openai import OpenAI
import os
import json

# === 0. Trimming 參數（可調） ===
# 只保留「最近 N 個使用者回合」做為上下文
TRIM_LAST_N_USER_TURNS = 30

# === 1. 設定 Streamlit 頁面 ===
st.set_page_config(page_title="Anya Multimodal Agent", page_icon="🥜", layout="wide")

def emoji_token_stream(full_text, emoji="🌸", cursor_symbol=" ", chunk=8):
    placeholder = st.empty()
    n = len(full_text)
    # 長文直接一次輸出，避免大量重繪
    if n > 1000:
        placeholder.markdown(full_text)
        return
    # 短文才做分塊動畫
    for i in range(0, n, chunk):
        shown = full_text[:i+chunk]
        placeholder.markdown(shown + cursor_symbol + emoji)
        time.sleep(0.03)
    placeholder.markdown(full_text)

# === 1.1 影像 MIME 偵測（用於回放舊回合圖片） ===
def _detect_mime_from_bytes(img_bytes: bytes) -> str:
    try:
        im = Image.open(BytesIO(img_bytes))
        fmt = (im.format or "").upper()
        if fmt == "PNG":
            return "image/png"
        if fmt in ("JPG", "JPEG"):
            return "image/jpeg"
        if fmt == "WEBP":
            return "image/webp"
        if fmt == "GIF":
            return "image/gif"
    except Exception:
        pass
    return "application/octet-stream"

# === 1.2 將 chat_history 修剪成「最近 N 個使用者回合」並轉成 Responses API input ===
def build_trimmed_input_messages(pending_user_content_blocks):
    """
    將 st.session_state.chat_history 修剪，只保留最近 N 個「使用者回合」，
    並把目前待送出的使用者訊息（pending_user_content_blocks）接在最後。
    回傳可直接丟進 client.responses.create(input=...) 的 messages 陣列。
    """
    hist = st.session_state.chat_history
    if not hist:
        # 首次對話：只送現在這一輪
        return [{"role": "user", "content": pending_user_content_blocks}]

    # 1) 找到「最近 N 個使用者回合」的起點索引
    user_count = 0
    start_idx = 0
    for i in range(len(hist) - 1, -1, -1):
        if hist[i].get("role") == "user":
            user_count += 1
            if user_count == TRIM_LAST_N_USER_TURNS:
                start_idx = i
                break
    # 如果少於 N 個 user 回合，就從最開頭開始

    selected = hist[start_idx:]

    # 2) 轉成 Responses API 的 messages 形狀
    messages = []
    for msg in selected:
        role = msg.get("role")
        if role == "user":
            blocks = []
            if msg.get("text"):
                blocks.append({"type": "input_text", "text": msg["text"]})
            # 將舊回合圖片一併帶入（如果你想更省 token，可以拿掉這段）
            if msg.get("images"):
                for fn, imgbytes in msg["images"]:
                    mime = _detect_mime_from_bytes(imgbytes)
                    b64 = base64.b64encode(imgbytes).decode()
                    blocks.append({
                        "type": "input_image",
                        "image_url": f"data:{mime};base64,{b64}"
                    })
            if blocks:
                messages.append({"role": "user", "content": blocks})
        elif role == "assistant":
            if msg.get("text"):
                # 以 assistant 的 output_text 形式放回上下文
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": msg["text"]}]
                })

    # 3) 加上「這一輪」使用者輸入（含圖片）
    messages.append({"role": "user", "content": pending_user_content_blocks})
    return messages

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

# 初始化（放在 === 2. Session State === 下面）
if "messages" not in st.session_state:
    st.session_state.messages = []  # 但這樣還要在送出時同步 append，工作量較大

# 不再使用 previous_response_id（改用 Trimming 手動餵上下文）
# if "previous_response_id" not in st.session_state:
#     st.session_state.previous_response_id = None

# === 3. OpenAI client ===
client = OpenAI(api_key=st.secrets["OPENAI_KEY"])

# === 4. 安妮亞系統提示 ===
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

你可以根據下列情境，決定是否要調用工具：
- `web_search`：當用戶的提問判斷需要搜尋網路資料時，請使用這個工具搜尋網路資訊。
- Use only tools listed in allowed_tools; for routine read-only tasks call automatically; for destructive ops require confirmation。

Before any significant tool call, state in one line: purpose + minimal inputs。

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

# 3. murmur（Responses API 版）& agent運作（無 BaseCallbackHandler）
# 3.1 匯總聊天文字（改用 chat_history，避免 messages 未初始化）
all_text = []
for msg in st.session_state.get("chat_history", []):
    if msg.get("text"):
        all_text.append(msg["text"])
all_text = "\n".join(all_text[-50:])  # 視需要保留最近幾則，避免太長

# 3.2 以 Responses API 產生 murmur（15字以內 + 可愛emoji）
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

# Context
聊天紀錄：
{all_text}

# Output
只回傳一句可愛的 murmur，15字以內，最後加上一個可愛 emoji。
""".strip()

try:
    murmur_resp = client.responses.create(
        model="gpt-4.1-nano",   # 也可用 gpt-4.1-mini
        input=[{"role": "user", "content": status_prompt}],
        timeout=12
    )
    status_label = (getattr(murmur_resp, "output_text", "") or "").strip()
    if not status_label:
        # 後備解析（避免不同 SDK 版型）
        if getattr(murmur_resp, "output", None):
            for item in murmur_resp.output:
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", "") in ("output_text", "text"):
                        status_label = (getattr(c, "text", "") or "").strip()
                        if status_label:
                            break
                if status_label:
                    break
    status_label = status_label.replace("\n", "").replace("\r", "").strip("「」\"' ")
    if len(status_label) > 15:
        status_label = status_label[:15]
except Exception:
    status_label = "今天氣氛好可愛✨"  # 兜底 murmur

# === 5. 聊天歷史呈現 ===
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

# === 6. 處理 AI 回覆（使用 Trimming；移除 spinner，只保留 status） ===
if st.session_state.pending_ai and st.session_state.pending_content:
    with st.chat_message("assistant"):
        status = st.status(status_label, expanded=False)
        try:
            # 依 Trimming 規則組裝上下文 + 這一輪使用者訊息
            status.update(label=f"{status_label}", state="running")
            trimmed_messages = build_trimmed_input_messages(st.session_state.pending_content)

            response = client.responses.create(
                model="gpt-5.1-2025-11-13",
                input=trimmed_messages,  # ← 不再用 previous_response_id，而是送修剪後的 messages
                tools=[{"type": "web_search"}],
                tool_choice="auto",
                parallel_tool_calls=True,
                reasoning={"effort": "medium"},
                text={"verbosity": "medium"},
                instructions=ANYA_SYSTEM_PROMPT,
                include=[
                    "web_search_call.action.sources",
                    "message.input_image.image_url"
                ],
                truncation="auto",
            )

            ai_text = ""
            if hasattr(response, "output") and response.output:
                for item in response.output:
                    if hasattr(item, "content") and item.content:
                        for c in item.content:
                            if getattr(c, "type", None) == "output_text":
                                ai_text += c.text
            if not ai_text:
                ai_text = "安妮亞找不到答案～（抱歉啦！）"

            # 狀態更新：正在輸出
            emoji_token_stream(ai_text, emoji="🌸", cursor_symbol=" ")
            status.update(label=f"{status_label}｜安妮亞回答完畢！🎉", state="complete")

        except Exception as e:
            ai_text = f"API 發生錯誤：{e}"
            status.update(label=f"{status_label}｜出現小狀況了…請再試一次🛠️", state="error")

        # 寫回歷史 & 收尾
        st.session_state.chat_history.append({
            "role": "assistant",
            "text": ai_text,
            "images": []
        })
        st.session_state.pending_ai = False
        st.session_state.pending_content = None
        status.update(label="安妮亞回答完畢！🥜", state="complete")
        st.rerun()

# === 7. 使用者輸入 ===
prompt = st.chat_input(
    "wakuwaku！安妮亞可以幫你看圖說故事嚕！",
    accept_file="multiple",
    file_type=["jpg", "jpeg", "png"]
)
if prompt:
    user_text = prompt.text.strip() if prompt.text else ""
    images_for_history = []
    content_blocks = []

    if user_text:
        content_blocks.append({"type": "input_text", "text": user_text})
    for f in prompt.files:
        imgbytes = f.getvalue()  # ← 直接 bytes
        mime = f.type or "image/png"
        b64 = base64.b64encode(imgbytes).decode()
        content_blocks.append({
            "type": "input_image",
            "image_url": f"data:{mime};base64,{b64}"
        })
        images_for_history.append((f.name, imgbytes))

    st.session_state.chat_history.append({
        "role": "user",
        "text": user_text,
        "images": images_for_history
    })
    st.session_state.pending_ai = True
    st.session_state.pending_content = content_blocks
    st.rerun()
