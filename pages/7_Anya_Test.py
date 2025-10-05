import streamlit as st
from PIL import Image
import base64
import io
from openai import OpenAI

# 初始化 OpenAI
client = OpenAI(api_key=st.secrets["OPENAI_KEY"])

st.set_page_config(page_title="多圖多任務AI", page_icon="🥜", layout="wide")
st.title("🥜 多圖 & 多任務 Vision AI DEMO")

### =================  1️⃣ 統一前處理 ================= ###
def prepare_image_assets(files):
    assets = []
    for uf in files:
        uf.seek(0)
        img_bytes = uf.read()
        if not img_bytes or len(img_bytes) < 32:
            st.warning(f"{uf.name} 資料太小，略過")
            continue
        try:
            img = Image.open(io.BytesIO(img_bytes))
            fmt = img.format.lower()
            mime = f"image/{fmt}"
            if fmt not in ["png","jpeg","jpg","webp","gif"]:
                st.warning(f"{uf.name} 格式 {fmt} 不支援（略過）")
                continue
            b64 = base64.b64encode(img_bytes).decode()
            assets.append({
                "bytes": img_bytes,
                "file_name": uf.name,
                "fmt": fmt,
                "mime": mime,
                "b64": b64,
                "pil_image": img
            })
        except Exception as e:
            st.warning(f"{uf.name} 讀取錯誤：{e}")
            continue
    return assets

### =================  2️⃣ 各種AI工具DEMO def ================= ###
def ocr_tool(image_asset):
    """ Vision OCR辨識 """
    img_url = f"data:{image_asset['mime']};base64,{image_asset['b64']}"
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content":
                    "You are an OCR-like data extraction tool that extracts text from images."},
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
        return response.output_text.strip()
    except Exception as e:
        return f"OCR 失敗：{e}"

def dummy_caption_tool(image_asset):
    """假裝 Caption 任務（可換成真AI）"""
    # 這裡只是示範，可以串 Stable Diffusion, Gemini 等API
    return f"這是 {image_asset['file_name']} 的假caption描述（請換成自己的API）"

### =================  3️⃣ 主流程 UI ================= ###
uploaded_files = st.file_uploader(
    "請選擇多張圖片（jpg/png/webp/gif，max單張10MB）",
    type=["jpg","jpeg","png","gif","webp"],
    accept_multiple_files=True
)

if uploaded_files:
    assets = prepare_image_assets(uploaded_files)
    st.markdown(f"### 共預處理成功 {len(assets)} 張圖片")
    for idx, asset in enumerate(assets,1):
        with st.expander(f"第{idx}張：{asset['file_name']}"):
            # 顯示縮圖＋格式
            st.image(asset["pil_image"], width=280, caption=asset["file_name"])
            st.markdown(f"**格式:** {asset['fmt']} **大小：**{len(asset['bytes'])//1024}KB")

            # OCR按鈕
            if st.button(f"OCR辨識", key=f"OCR_btn_{idx}"):
                with st.spinner("提取中..."):
                    ocr_text = ocr_tool(asset)
                    st.markdown("#### 📋 Vision OCR 結果")
                    st.write(ocr_text)

            # Caption按鈕
            if st.button(f"圖片Caption", key=f"cap_btn_{idx}"):
                with st.spinner("產生中..."):
                    cap = dummy_caption_tool(asset)
                    st.markdown("#### 🖼️ Caption 結果")
                    st.write(cap)

    # Bonus：全部自動OCR
    if len(assets)>0 and st.button("全部自動OCR"):
        with st.spinner("全部圖片自動Batch OCR中..."):
            all_ocr = [ocr_tool(asset) for asset in assets]
            for asset, res in zip(assets, all_ocr):
                st.markdown(f"---\n**{asset['file_name']} OCR結果:**")
                st.write(res)
