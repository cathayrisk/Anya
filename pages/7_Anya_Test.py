import streamlit as st
from PIL import Image
import base64
import io
from openai import OpenAI

# ---- 設置你的 OpenAI key ----
client = OpenAI(api_key=st.secrets["OPENAI_KEY"])  # 或直接寫 api_key="..."

st.set_page_config(page_title="安妮亞圖片OCR測試", page_icon="🥜", layout="centered")

st.title("🥜 單圖上傳 + Vision OCR Demo")

uploaded_file = st.file_uploader("請選擇一張圖片（支援jpg/png/gif/webp，建議不大於5MB）", type=["jpg", "jpeg", "png", "gif", "webp"])

if uploaded_file:
    # 1. 把file轉成bytes
    uploaded_file.seek(0)
    imgbytes = uploaded_file.read()
    # 2. 驗證 & 顯示縮圖
    try:
        img = Image.open(io.BytesIO(imgbytes))
        st.image(img, caption="你上傳的圖片", width=320)
        fmt = img.format.lower()
        mime = f"image/{fmt}"
        st.success(f"圖片格式偵測成功: {fmt}, 大小: {len(imgbytes)/1024:.1f}KB")
    except Exception as e:
        st.error(f"❌ 圖片格式錯誤：{e}")
        st.stop()
    # 3. base64 encode
    b64 = base64.b64encode(imgbytes).decode()
    img_url = f"data:{mime};base64,{b64}"

    # 4. OCR按鈕
    if st.button("辨識圖片文字（Vision OCR）"):
        with st.spinner("安妮亞努力辨識中...（Vision API如卡住請檢查API Key權限）"):
            # 5. Call OpenAI Vision
            try:
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
                    timeout=45
                )
                st.markdown("### 📄 辨識結果")
                st.write(response.output_text.strip())
            except Exception as e:
                st.error(f"❌ Vision API 錯誤：\n{e}")
