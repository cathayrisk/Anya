import streamlit as st
from PIL import Image
import base64
import io
from openai import OpenAI

# ---- 設置你的 OpenAI key ----
client = OpenAI(api_key=st.secrets["OPENAI_KEY"])  # 或直接寫 api_key="..."

st.set_page_config(page_title="多圖OCR", page_icon="🥜", layout="wide")
st.title("🥜 多圖上傳 + Vision OCR 批量辨識Demo")

uploaded_files = st.file_uploader(
    "請選擇多張圖片（jpg/png/gif/webp，單張建議小於5MB）", 
    type=["jpg","jpeg","png","gif","webp"], 
    accept_multiple_files=True
)

if uploaded_files:
    for idx, uploaded_file in enumerate(uploaded_files, 1):
        st.divider()
        st.write(f"### 第{idx}張：{uploaded_file.name}")

        # 1. 讀取bytes
        uploaded_file.seek(0)
        imgbytes = uploaded_file.read()

        # 2. 驗證圖片格式並顯示
        try:
            img = Image.open(io.BytesIO(imgbytes))
            st.image(img, caption=uploaded_file.name, width=320)
            fmt = img.format.lower()
            mime = f"image/{fmt}"
            st.info(f"圖片格式:{fmt}，大小:{len(imgbytes)//1024}KB")
        except Exception as e:
            st.error(f"❌ {uploaded_file.name} 不是有效圖片！錯誤：{e}")
            continue

        # 3. base64 encode
        b64 = base64.b64encode(imgbytes).decode()
        img_url = f"data:{mime};base64,{b64}"

        # 4. OCR按鈕（每張圖片獨立辨識）
        if st.button(f"辨識第{idx}張圖片文字（Vision OCR）", key=f"ocr_btn_{idx}"):
            with st.spinner("安妮亞努力辨識中..."):
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
                    st.success("🎉 OCR辨識完成！")
                    st.markdown("##### 📄 辨識結果")
                    st.write(response.output_text.strip())
                except Exception as e:
                    st.error(f"❌ Vision API 錯誤：{e}")
