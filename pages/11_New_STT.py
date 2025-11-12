import os
import re
import json
import difflib
import hashlib
import tempfile
import multiprocessing
from typing import List, Dict, Any

import streamlit as st
from openai import OpenAI
from pydub import AudioSegment, silence
from pydub.utils import which

# ========== 基本設定 ==========
st.set_page_config(page_title="會議錄音 → 直播逐字＋摘要", page_icon="📝", layout="wide")

# 自訂樣式（粉粉卡片、Tabs 視覺）
st.markdown("""
<style>
:root { --brand:#9c2b2f; --brand-weak:#9c2b2fcc; --bg:#FFF6F6; --border:#f2d9d9; }
.pink-card{background:var(--bg);border:1px solid var(--border);padding:14px 18px;border-radius:12px;}
.header-pill{display:flex;align-items:center;gap:12px;font-size:22px;font-weight:600;color:#2f2f2f;}
.header-pill .emoji{font-size:22px}
.success-card{display:flex;align-items:center;gap:10px;font-weight:600;color:#2f2f2f;}
/* Tabs */
.stTabs [data-baseweb="tab-list"]{gap:24px;border-bottom:1px solid #f0e2e2;margin-bottom:8px}
.stTabs [data-baseweb="tab"]{padding:10px 2px;color:var(--brand-weak);font-weight:600}
.stTabs [aria-selected="true"]{color:var(--brand);border-bottom:3px solid var(--brand)}
/* 讓段落更好讀 */
.block-container{padding-top:1.2rem}
.stMarkdown p{line-height:1.75}
</style>
""", unsafe_allow_html=True)

# 頂部卡片標題
st.markdown('<div class="pink-card header-pill"><span class="emoji">💋</span> Speech to text transcription</div>', unsafe_allow_html=True)

# 檢查 FFmpeg
AudioSegment.converter = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")
if not AudioSegment.converter or not AudioSegment.ffprobe:
    st.error("找不到 ffmpeg/ffprobe，請先於系統安裝後再試。")
    st.stop()

# 讀取 API Key
OPENAI_KEY = st.secrets.get("OPENAI_KEY", os.getenv("OPENAI_API_KEY"))
if not OPENAI_KEY:
    st.error("找不到 API Key，請在 Streamlit Secrets 設定 OPENAI_KEY 或環境變數 OPENAI_API_KEY。")
    st.stop()

client = OpenAI(api_key=OPENAI_KEY)

# ========== 參數 ==========
MODEL_STT = "gpt-4o-mini-transcribe"
MODEL_MAP = "gpt-4.1"
MODEL_REDUCE = "gpt-4.1"
DEFAULT_MAP_CHUNK_SIZE = 40
MIN_SILENCE_LEN_MS = 700
KEEP_SILENCE_MS = 300
SILENCE_DB_OFFSET = 16
OVERLAP_MS = 1200
MAX_STREAM_WORKERS = min(4, multiprocessing.cpu_count())

CACHE_DIR = ".stt_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ========== 工具函式 ==========
def _hash_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

def cache_get_text(key: str) -> str | None:
    path = os.path.join(CACHE_DIR, key + ".txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None

def cache_set_text(key: str, value: str):
    path = os.path.join(CACHE_DIR, key + ".txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(value)

def convert_to_wav(input_path: str, output_path: str, target_sr=16000):
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(target_sr).set_channels(1)
    audio.export(output_path, format="wav")
    return output_path

def split_audio_on_silence_safe(audio: AudioSegment) -> List[AudioSegment]:
    silence_thresh = audio.dBFS - SILENCE_DB_OFFSET
    raw_chunks = silence.split_on_silence(
        audio,
        min_silence_len=MIN_SILENCE_LEN_MS,
        silence_thresh=silence_thresh,
        keep_silence=KEEP_SILENCE_MS
    )
    if not raw_chunks:
        return []
    filtered = []
    for c in raw_chunks:
        if len(c) < 250:
            if filtered:
                filtered[-1] = filtered[-1] + c
            else:
                filtered.append(c)
        else:
            filtered.append(c)
    if not filtered:
        return []
    chunks = []
    for i, c in enumerate(filtered):
        if i == 0:
            chunks.append(c)
        else:
            prev = filtered[i - 1]
            safe_overlap = min(OVERLAP_MS, len(prev))
            if safe_overlap > 0:
                overlap = prev[-safe_overlap:]
                chunks.append(overlap + c)
            else:
                chunks.append(c)
    return chunks

def split_sentences(text: str) -> List[str]:
    parts = re.split(r'([。！？；\n])', text)
    result = []
    for i in range(0, len(parts) - 1, 2):
        s = (parts[i] + parts[i + 1]).strip()
        if s:
            result.append(s)
    if len(parts) % 2 != 0:
        tail = parts[-1].strip()
        if tail:
            result.append(tail)
    return result

def dedupe_against_prev(curr: List[str], prev: List[str], threshold=0.80) -> List[str]:
    out = []
    for s in curr:
        if all(difflib.SequenceMatcher(None, s, p).ratio() <= threshold for p in prev):
            out.append(s)
    return out

def stream_transcribe_all(chunks: List[AudioSegment], container, progress_bar):
    import time
    all_text = ""
    last_flush = 0.0
    FLUSH_INTERVAL = 0.15  # 150ms 節流

    for i, chunk in enumerate(chunks):
        chunk_hash = _hash_bytes(chunk.raw_data)
        cache_key = f"stt_{MODEL_STT}_{chunk_hash}"  # 包含模型名，避免汙染
        cached = cache_get_text(cache_key)
        if cached:
            all_text += cached + "\n"
            progress_bar.progress((i + 1) / len(chunks))
            container.markdown(all_text)
            continue

        full_text = ""
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp_path = tmp.name
                chunk.export(tmp_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
            with open(tmp_path, "rb") as audio_file:
                try:
                    stream = client.audio.transcriptions.create(
                        model=MODEL_STT,
                        file=audio_file,
                        response_format="text",
                        stream=True
                    )
                    for event in stream:
                        delta = getattr(event, "delta", None)
                        final_text = getattr(event, "text", None)
                        if delta:
                            full_text += delta
                            now = time.time()
                            if now - last_flush > FLUSH_INTERVAL:
                                container.markdown(all_text + full_text)
                                last_flush = now
                        elif final_text:
                            full_text = final_text
                            container.markdown(all_text + full_text)
                except Exception as e:
                    container.error(f"API 轉錄失敗：{e}")
        finally:
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

        cache_set_text(cache_key, full_text.strip())
        all_text += full_text + "\n"
        progress_bar.progress((i + 1) / len(chunks))
        container.markdown(all_text)

    return all_text.strip()

def map_summarize_blocks(flat_sentences: List[str], chunk_size=DEFAULT_MAP_CHUNK_SIZE) -> List[str]:
    blocks = []
    for idx in range(0, len(flat_sentences), chunk_size):
        part = flat_sentences[idx: idx + chunk_size]
        prompt = (
            "你是一位會議記錄小幫手，請將下列逐字稿整理為條列式重點（繁體中文）：\n"
            "- 每點盡量具體，避免空泛\n"
            "- 若有決策/風險/未決問題/行動項目，請清楚標記\n"
            "- 僅輸出條列重點，不要額外說明\n\n"
            + "\n".join(part)
        )
        try:
            resp = client.chat.completions.create(
                model=MODEL_MAP,
                messages=[{"role": "user", "content": prompt}]
            )
            content = resp.choices[0].message.content
            blocks.append(content.strip())
        except Exception as e:
            blocks.append(f"【API 摘要失敗：{e}】")
    return blocks

def reduce_finalize_json(map_blocks: List[str]) -> Dict[str, Any]:
    prompt = (
        "你是會議記錄總整專家。請將多個分段摘要合併成結構化 JSON，包含：\n"
        "- metadata: {title, date, location, participants[], duration}\n"
        "- topics[]: {title, key_points[], decisions[], risks[], open_questions[]}\n"
        "- decisions[]\n"
        "- risks[]\n"
        "- open_questions[]\n"
        "- action_items[]: {description, owner|null, due_date|null, priority|null (P0~P3), status, source_refs[]}\n"
        "- overall_summary: string\n"
        "要求：\n"
        "1) 嚴禁捏造來源沒有的資訊；未知欄位請留空或 Unknown。\n"
        "2) 去重、合併相近重點，但不得改變原意。\n"
        "3) 只輸出 JSON 物件，不要額外說明文字。\n"
        "4) 請確保輸出內容為合法 JSON，不能有任何說明或多餘文字。\n\n"
        "=== 分段摘要 ===\n"
        + "\n\n".join(f"[Part {i+1}]\n{blk}" for i, blk in enumerate(map_blocks))
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_REDUCE,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        s = resp.choices[0].message.content.strip()
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1:
            s = s[start:end+1]
        return json.loads(s)
    except Exception as e:
        return {"overall_summary": f"解析 JSON 失敗，請重試或調整提示。錯誤：{e}", "raw": ""}

def reduce_finalize_markdown(map_blocks: List[str]) -> str:
    prompt = (
        "你是會議記錄總整專家。請將多個分段摘要整併為『單一份最終會議紀錄（Markdown）』。\n"
        "要求：\n"
        "1) 僅根據提供的分段摘要整併，嚴禁捏造來源沒有的資訊。\n"
        "2) 不輸出 metadata（標題/日期/地點/參與者/時長），只要內容本體。\n"
        "3) 結構：\n"
        "   - 以一段「總結」開場，3~6 句，說清楚整體脈絡與結論。\n"
        "   - 之後用多個小節（## 主題名稱），每節採用短段落敘述為主，可穿插少量條列。\n"
        "   - 若有決策/風險/未決問題，於對應主題內以『決策：』『風險：』『未決：』行內標示。\n"
        "4) 只輸出純 Markdown 內容，不要額外說明。\n\n"
        "=== 分段摘要 ===\n"
        + "\n\n".join(f"[Part {i+1}]\n{blk}" for i, blk in enumerate(map_blocks))
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_REDUCE,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ 生成會議摘要失敗：{e}"

def render_topics_only(md: Dict[str, Any], st):
    st.markdown("#### 主題")
    topics = md.get("topics", [])
    for t in topics:
        st.markdown(f"##### {t.get('title','主題')}")
        kp = t.get("key_points", [])
        if kp:
            st.markdown("\n".join(f"- {x}" for x in kp))
        if t.get("decisions"):
            st.markdown("決策：\n" + "\n".join(f"- {x}" for x in t.get("decisions", [])))
        if t.get("risks"):
            st.markdown("風險：\n" + "\n".join(f"- {x}" for x in t.get("risks", [])))
        if t.get("open_questions"):
            st.markdown("未決問題：\n" + "\n".join(f"- {x}" for x in t.get("open_questions", [])))

# ========== 上傳區 ==========
with st.expander("上傳會議錄音檔案", expanded=True):
    f = st.file_uploader("請上傳音檔（.wav, .mp3, .m4a, .mp4, .webm）", type=["wav", "mp3", "m4a", "mp4", "webm"])
    start_btn = st.button("開始 Streaming 轉錄與摘要")

# 主畫面進階調整（僅顯示，預設即可）
with st.expander("進階調整（可選，不用也能順跑）", expanded=False):
    st.caption("不同錄音情境才需要微調，平常維持預設即可。")
    st.text(f"MIN_SILENCE_LEN_MS = {MIN_SILENCE_LEN_MS}")
    st.text(f"KEEP_SILENCE_MS = {KEEP_SILENCE_MS}")
    st.text(f"SILENCE_DB_OFFSET = {SILENCE_DB_OFFSET}")
    st.text(f"OVERLAP_MS = {OVERLAP_MS}")
    st.text(f"MAP_CHUNK_SIZE = {DEFAULT_MAP_CHUNK_SIZE}")

if not (f and start_btn):
    st.stop()

# ========== 主流程 ==========
raw_bytes = f.read()
st.audio(raw_bytes)

tab1, tab2, tab3, tab4 = st.tabs(["轉錄結果", "重點摘要", "內容解析", "原始內容"])

with st.status("處理中...", expanded=True) as status:
    # 0) 儲存與轉檔
    status.update(label="儲存與轉檔...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{f.name.split('.')[-1]}") as temp_input:
        temp_input.write(raw_bytes)
        temp_input_path = temp_input.name

    wav_path = temp_input_path
    if not f.name.lower().endswith(".wav"):
        wav_path = temp_input_path + ".wav"
        convert_to_wav(temp_input_path, wav_path)
    audio = AudioSegment.from_file(wav_path, format="wav")

    # 1) 靜音切段（安全重疊）
    status.update(label="靜音切段（安全重疊）...")
    chunks = split_audio_on_silence_safe(audio)
    if not chunks:
        st.error("無法切出有效音訊段，請檢查音檔或調整參數（可提高 keep_silence / 降低 silence_db_offset）。")
        st.stop()

    # 2) 逐段 Streaming 轉錄（顯示在 Tab1）
    with tab1:
        st.markdown("#### 轉錄結果")
        stream_container = st.empty()
        progress_bar = st.progress(0.0)
        all_text = stream_transcribe_all(chunks, stream_container, progress_bar)
        st.markdown('<div class="pink-card success-card">✅ Transcription complete!</div>', unsafe_allow_html=True)

    # 3) 分句與跨段去重
    status.update(label="分句與跨段去重...")
    grouped_sentences = []
    for i, txt in enumerate(all_text.split("\n")):
        sents = split_sentences(txt)
        if i == 0:
            grouped_sentences.append(sents)
        else:
            unique = dedupe_against_prev(sents, grouped_sentences[-1], threshold=0.80)
            grouped_sentences.append(unique)
    flat_sentences = [s for group in grouped_sentences for s in group]

    # 4) 原始內容（Tab4）
    with tab4:
        st.markdown("#### 原始內容")
        st.code("\n".join(flat_sentences), language="text")

    # 5) 分段摘要（僅計算，不顯示）
    status.update(label="整併重點（內部計算）...")
    map_blocks_text = map_summarize_blocks(flat_sentences)

    # 6) 最終會議摘要（Tab2：敘述版），內容解析（Tab3：主題重點）
    status.update(label="生成最終會議摘要與內容解析...")
    final_minutes = reduce_finalize_json(map_blocks_text)   # 結構化，給內容解析用
    final_md = reduce_finalize_markdown(map_blocks_text)    # 敘述版，給重點摘要用

    with tab2:
        st.markdown(final_md)
        st.download_button(
            "下載會議記錄 JSON",
            data=json.dumps(final_minutes, ensure_ascii=False, indent=2),
            file_name="meeting_minutes.json",
            mime="application/json"
        )

    with tab3:
        render_topics_only(final_minutes, st)

    status.update(label="全部完成！", state="complete", expanded=True)

# 清理暫存
try:
    os.remove(temp_input_path)
    if 'wav_path' in locals() and wav_path != temp_input_path:
        os.remove(wav_path)
except Exception:
    pass
