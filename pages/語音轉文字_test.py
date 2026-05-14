import os
import re
import json
import math
import time
import difflib
import hashlib
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
from openai import OpenAI
from pydub import AudioSegment, silence
from pydub.utils import which

# ========== 基本設定 ==========
st.set_page_config(page_title="會議錄音 → 直播逐字＋摘要", page_icon="📝", layout="wide")

# 自訂樣式（加大頂部內距避免標題被切到、Tabs 視覺、內文可讀性）
st.markdown("""
<style>
:root { --brand:#9c2b2f; --brand-weak:#9c2b2fcc; --bg:#FFF6F6; --border:#f2d9d9; }
.main .block-container{padding-top:2.2rem}
.pink-card{background:var(--bg);border:1px solid var(--border);padding:16px 22px;border-radius:12px;margin-bottom:12px;overflow:visible;}
.header-pill{display:flex;align-items:center;gap:12px;font-size:22px;font-weight:700;color:#2f2f2f;line-height:1.35;min-height:48px;}
.header-pill .emoji{font-size:22px;display:inline-block;transform:translateY(1px);}
.stTabs [data-baseweb="tab-list"]{gap:24px;border-bottom:1px solid #f0e2e2;margin-bottom:8px}
.stTabs [data-baseweb="tab"]{padding:10px 2px;color:var(--brand-weak);font-weight:600}
.stTabs [aria-selected="true"]{color:var(--brand);border-bottom:3px solid var(--brand)}
.stMarkdown p{line-height:1.8}
.transcript-readable{font-size:1.02rem;line-height:1.9;letter-spacing:0.02em;}
</style>
""", unsafe_allow_html=True)

# 頂部卡片標題
st.markdown('<div class="pink-card header-pill"><span class="emoji">✍️</span> 安妮亞開會不漏接：逐字 × 摘要</div>', unsafe_allow_html=True)

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
MODEL_STT    = "gpt-4o-mini-transcribe"   # 一般轉錄
MODEL_DIARIZE = "gpt-4o-transcribe-diarize"  # 說話人辨識轉錄
MODEL_MAP    = "gpt-5-mini"               # 分段摘要
MODEL_REDUCE = "gpt-4.1"                 # 總整/潤飾

MAX_WORKERS = 4      # 平行轉錄最大執行緒數
DIARIZE_MAX_SEC = 1400  # gpt-4o-transcribe-diarize API 上限（秒）

# 說話者顏色對應（最多支援 8 位說話者）
SPEAKER_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]
SPEAKER_EMOJIS = ["🔵", "🟢", "🟠", "🟣", "🔴", "🟡", "⚪", "🟤"]

BASE_PROMPT = (
    "This audio contains a discussion or presentation. "
    "Always preserve the original language of each sentence. "
    "If a sentence is in English, output it in English; "
    "if in Chinese, output it in Traditional Chinese; "
    "if mixed, output the original mixed-language sentence. "
    "Do not translate or alter the language. "
    "The audio may cover various topics such as updates, feedback, or informative lectures."
)

# 切段參數
MIN_SILENCE_LEN_MS = 700
KEEP_SILENCE_MS = 300
SILENCE_DB_OFFSET = 16
OVERLAP_MS = 1200

# 片段長度保護與回退
MAX_CHUNK_MS = 30_000   # 單段最長 30 秒
MIN_CHUNK_MS = 2_000    # 單段最短 2 秒
FALLBACK_WINDOW_MS = 20_000  # 找不到靜音時，固定切 20 秒

DEFAULT_MAP_CHUNK_SIZE = 40

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".stt_cache")
CACHE_MAX_FILES = 200   # 超過時刪除最舊的檔案
os.makedirs(CACHE_DIR, exist_ok=True)

def cache_cleanup():
    try:
        files = sorted(
            [os.path.join(CACHE_DIR, fn) for fn in os.listdir(CACHE_DIR) if fn.endswith(".txt")],
            key=os.path.getmtime
        )
        for old in files[:-CACHE_MAX_FILES]:
            os.remove(old)
    except Exception:
        pass

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

def normalize_loudness(audio: AudioSegment, target_dbfs: float = -20.0) -> AudioSegment:
    if audio.dBFS == float("-inf"):  # 全靜音，跳過正規化避免 apply_gain(inf) 崩潰
        return audio
    gain = target_dbfs - audio.dBFS
    return audio.apply_gain(gain)

def trim_leading_silence(audio: AudioSegment, silence_threshold_db: float = -30.0, chunk_ms: int = 10) -> AudioSegment:
    trim_ms = 0
    while trim_ms < len(audio) and audio[trim_ms:trim_ms+chunk_ms].dBFS < silence_threshold_db:
        trim_ms += chunk_ms
    return audio[trim_ms:]

def apply_filters(audio: AudioSegment, use_high_pass: bool = False, hp_hz: int = 100,
                  use_low_pass: bool = False, lp_hz: int = 9500) -> AudioSegment:
    out = audio
    if use_high_pass:
        out = out.high_pass_filter(hp_hz)
    if use_low_pass:
        out = out.low_pass_filter(lp_hz)
    return out

def split_audio_on_silence_safe(audio: AudioSegment) -> List[AudioSegment]:
    silence_thresh = audio.dBFS - SILENCE_DB_OFFSET
    raw_chunks = silence.split_on_silence(
        audio,
        min_silence_len=MIN_SILENCE_LEN_MS,
        silence_thresh=silence_thresh,
        keep_silence=KEEP_SILENCE_MS
    )

    if not raw_chunks:
        chunks = []
        i = 0
        while i < len(audio):
            end = min(i + FALLBACK_WINDOW_MS, len(audio))
            chunks.append(audio[i:end])
            i = end
    else:
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
            filtered = raw_chunks

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

    normalized = []
    for seg in chunks:
        if len(seg) <= MAX_CHUNK_MS:
            normalized.append(seg)
        else:
            start = 0
            while start < len(seg):
                end = min(start + MAX_CHUNK_MS, len(seg))
                normalized.append(seg[start:end])
                start = end

    final_chunks = []
    for seg in normalized:
        if final_chunks and len(seg) < MIN_CHUNK_MS:
            final_chunks[-1] = final_chunks[-1] + seg
        else:
            final_chunks.append(seg)

    return final_chunks

def split_sentences(text: str) -> List[str]:
    parts = re.split(r'([。！？；;.!?\n])', text)
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

def add_cjk_spacing(text: str) -> str:
    text = re.sub(r'([\u4e00-\u9fff])([A-Za-z0-9$%#@&])', r'\1 \2', text)
    text = re.sub(r'([A-Za-z0-9$%#@&])([\u4e00-\u9fff])', r'\1 \2', text)
    return text

def normalize_symbols(text: str) -> str:
    text = text.replace("％", "%").replace("＄", "$")
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("\u200b", "").replace("\u200c", "")
    return text

def pretty_format_sentences(sentences: List[str]) -> List[str]:
    pretty = []
    for s in sentences:
        s2 = add_cjk_spacing(s)
        s2 = normalize_symbols(s2)
        pretty.append(s2)
    return pretty

# 顯示層：逐行『潤飾＋必要時翻譯』為正體中文（台灣用語），穩定版（批次＋分隔符）
def refine_zh_tw_via_prompt(lines: List[str]) -> List[str]:
    """
    將多行句子逐行『潤飾＋必要時翻譯』為正體中文（台灣用語）。
    - 批次處理＋分隔符防走位；單批失敗只回退該批，不影響其他批。
    """
    if not lines:
        return lines

    SEP = "\u241E"  # ␞ 極少見的可視分隔符
    MAX_BATCH_CHARS = 9000  # 單批最大字數（保守）
    MAX_BATCH_LINES = 120   # 單批最多行數（保守）

    def _refine_batch(batch: List[str]) -> List[str]:
        blob = SEP.join(batch)
        dev_msg = (
            "你將收到多行逐字稿，請逐行『潤飾＋必要時翻譯』為正體中文（台灣用語）。\n"
            "要求：\n"
            "1) 保留原意，只做語句潤飾與正體翻譯，不得捏造資訊。\n"
            "2) 若該行是英文或混雜語言，翻譯為正體中文（台灣用語）。\n"
            "3) 嚴禁合併/拆分行；嚴禁插入或刪除空行；輸入幾行就輸出幾行。\n"
            "4) 保留數字、單位、時間、金額、emoji、網址、簡短代碼片段等非語意內容。\n"
            "5) 用詞採台灣慣用、口吻簡潔專業自然。\n"
            "6) 行與行由特殊分隔符 ␞（U+241E）連接；請務必保留相同數量的分隔符，不可新增或移除。\n"
            "只輸出最終文本，不要任何解釋。"
        )
        try:
            resp = client.responses.create(
                model=MODEL_REDUCE,
                input=[
                    {"role": "developer", "content": [{"type": "input_text", "text": dev_msg}]},
                    {"role": "user", "content": [{"type": "input_text", "text": blob}]},
                ],
                text={"format": {"type": "text"}},
                tools=[],
            )
            out = (resp.output_text or "").rstrip("\n")
            out_lines = out.split(SEP) if SEP in out else out.split("\n")
            return out_lines if len(out_lines) == len(batch) else batch
        except Exception:
            return batch

    # 分批處理
    refined_all: List[str] = []
    batch: List[str] = []
    size = 0
    for s in lines:
        if (len(batch) >= MAX_BATCH_LINES) or (size + len(s) + 1 > MAX_BATCH_CHARS):
            refined_all.extend(_refine_batch(batch))
            batch, size = [], 0
        batch.append(s)
        size += len(s) + 1
    if batch:
        refined_all.extend(_refine_batch(batch))

    return refined_all if refined_all else lines

# Prompt（若端點支援就用、不支援自動回退）
def build_prompt(prev_text: str, glossary: str, style_seed: str, max_chars: int = 800) -> str:
    # max_chars=800 約等於 220 tokens（中英混合估算），改用字元數截斷以正確處理 CJK
    parts = ["請全程使用正體中文（繁體，台灣用語）。"]
    if style_seed and style_seed.strip():
        parts.append(style_seed.strip())
    if glossary and glossary.strip():
        words = [w.strip() for w in glossary.splitlines() if w.strip()]
        if words:
            parts.append("Glossary: " + ", ".join(words))
    if prev_text and prev_text.strip():
        tail = prev_text.strip()
        if len(tail) > 1200:
            tail = tail[-1200:]
        parts.append(tail)

    prompt = "\n".join(parts).strip()
    if len(prompt) > max_chars:
        prompt = prompt[-max_chars:]
    return prompt

def _transcribe_chunk(chunk: AudioSegment, prompt: str, use_cache: bool = True) -> str:
    """單段音訊轉錄，供循序與平行模式共用。
    use_cache=False 時略過讀寫 cache（prompt 引導模式，每次 prompt 不同）。
    """
    cache_key = f"stt_{MODEL_STT}_{_hash_bytes(chunk.raw_data)}"
    if use_cache:
        cached = cache_get_text(cache_key)
        if cached:
            return cached

    full_text = ""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp_path = tmp.name
            chunk.export(tmp_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
        with open(tmp_path, "rb") as audio_file:
            try:
                resp = client.audio.transcriptions.create(
                    model=MODEL_STT,
                    file=audio_file,
                    response_format="text",
                    prompt=prompt,
                )
                full_text = resp if isinstance(resp, str) else (getattr(resp, "text", None) or "")
            except Exception:
                try:
                    resp = client.audio.transcriptions.create(
                        model=MODEL_STT,
                        file=audio_file,
                        response_format="text",
                        prompt=BASE_PROMPT,
                    )
                    full_text = resp if isinstance(resp, str) else (getattr(resp, "text", None) or "")
                except Exception:
                    full_text = ""
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    if use_cache:  # 只有無 prompt 引導的結果才寫入 cache
        cache_set_text(cache_key, full_text.strip())
    return full_text


def transcribe_all(
    chunks: List[AudioSegment],
    container,
    progress_bar,
    use_prompting: bool = False,
    glossary: str = "",
    style_seed: str = ""
) -> str:
    total = len(chunks)

    # ── 有 prompt 引導：循序執行以維持 rolling context ──
    if use_prompting:
        all_text = ""
        rolling_context = ""
        failed_indices: List[int] = []
        for i, chunk in enumerate(chunks):
            prompt_parts = [BASE_PROMPT]
            extra = build_prompt(rolling_context, glossary, style_seed, max_chars=800)
            if extra:
                prompt_parts.append(extra)
            text = _transcribe_chunk(chunk, "\n".join(prompt_parts), use_cache=False)
            # 首次失敗：等 2 秒後重試一次
            if not text.strip():
                time.sleep(2)
                text = _transcribe_chunk(chunk, "\n".join(prompt_parts), use_cache=False)
            if not text.strip():
                failed_indices.append(i + 1)
            all_text += text + "\n"
            rolling_context = (rolling_context + " " + text).strip()[-5000:]
            progress_bar.progress((i + 1) / total)
            container.markdown(f"*{i+1}/{total} 段完成...*\n\n" + all_text)
        if failed_indices:
            container.error(
                f"⛔ 第 {', '.join(map(str, failed_indices))} 段轉錄失敗（重試後仍無法取得內容），"
                "無法保證逐字稿完整性，已終止後續摘要生成。請重試或縮短音檔。"
            )
            st.stop()
        return all_text.strip()

    # ── 無 prompt 引導：平行執行，速度提升 3-5x ──
    results: Dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_transcribe_chunk, chunk, BASE_PROMPT, True): i
            for i, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            i = futures[future]
            try:
                results[i] = future.result() or ""
            except Exception:
                results[i] = ""
            done = len(results)
            progress_bar.progress(done / total)
            container.markdown(f"*{done}/{total} 段完成...*")

    # 重試空結果的段（一次，間隔 2 秒 backoff）
    empty_indices = [i for i in range(total) if not results.get(i, "").strip()]
    if empty_indices:
        time.sleep(2)
        for i in empty_indices:
            try:
                text = _transcribe_chunk(chunks[i], BASE_PROMPT, use_cache=False)
                if text.strip():
                    results[i] = text
            except Exception:
                pass

    # 統計重試後仍失敗的段，失敗時終止流程（避免以不完整內容生成摘要）
    still_failed = [i + 1 for i in range(total) if not results.get(i, "").strip()]
    if still_failed:
        container.error(
            f"⛔ 以下第 {', '.join(map(str, still_failed))} 段轉錄失敗（共 {len(still_failed)}/{total} 段），"
            "無法保證逐字稿完整性，已終止後續摘要生成。請重試或縮短音檔。"
        )
        st.stop()

    all_text = "\n".join(results.get(i, "") for i in range(total))
    container.markdown(all_text)
    return all_text.strip()

def _format_speaker_label(raw: str, speaker_map: Dict[str, str]) -> str:
    """將 SPEAKER_00 等原始標籤對應到 說話者 A/B/C...，首次出現時建立對應。"""
    if raw not in speaker_map:
        idx = len(speaker_map)
        label = SPEAKER_LABELS[idx] if idx < len(SPEAKER_LABELS) else str(idx)
        emoji = SPEAKER_EMOJIS[idx] if idx < len(SPEAKER_EMOJIS) else "🎙"
        speaker_map[raw] = f"{emoji} **[說話者 {label}]**"
    return speaker_map[raw]


def _parse_diarize_resp(resp) -> Tuple[str, str]:
    """回應物件 → (diarized_md, plain_text)，與 speaker_map 無關（每次獨立）。"""
    utterances = getattr(resp, "utterances", None)
    if utterances is None and hasattr(resp, "model_dump"):
        utterances = resp.model_dump().get("utterances") or resp.model_dump().get("segments")
    if utterances is None:
        plain = getattr(resp, "text", "") or ""
        return plain, plain

    speaker_map: Dict[str, str] = {}
    md_lines: List[str] = []
    plain_lines: List[str] = []
    for utt in utterances:
        if isinstance(utt, dict):
            raw_speaker = utt.get("speaker", "SPEAKER_00")
            text = utt.get("text", "").strip()
        else:
            raw_speaker = getattr(utt, "speaker", "SPEAKER_00")
            text = getattr(utt, "text", "").strip()
        if not text:
            continue
        label = _format_speaker_label(raw_speaker, speaker_map)
        md_lines.append(f"{label}　{text}")
        plain_lines.append(text)
    return "\n\n".join(md_lines), "\n".join(plain_lines)


def _call_diarize_api(tmp_wav_path: str) -> Tuple[str, str]:
    """對單一 WAV 檔呼叫 diarize API，回傳 (diarized_md, plain_text)。"""
    with open(tmp_wav_path, "rb") as audio_file:
        resp = client.audio.transcriptions.create(
            model=MODEL_DIARIZE,
            file=audio_file,
            response_format="diarized_json",
            chunking_strategy="auto",
        )
    return _parse_diarize_resp(resp)


def transcribe_diarize(wav_path: str, audio: AudioSegment, container, progress_bar) -> Tuple[str, str]:
    """
    說話人辨識轉錄。
    - 音檔 ≤ DIARIZE_MAX_SEC：單次 API 呼叫。
    - 音檔 > DIARIZE_MAX_SEC：自動切成 ≤1400 秒的段落分批辨識；
      各段說話者標籤獨立，不跨段比對，輸出附分段標題。
    """
    duration_sec = len(audio) / 1000

    # ── 短音檔：從已前處理的 audio 匯出後送出（確保與長音檔路徑行為一致）──
    if duration_sec <= DIARIZE_MAX_SEC:
        container.markdown("*正在送出整段音檔進行說話人辨識，請稍候...*")
        progress_bar.progress(0.3)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp_path = tmp.name
                audio.export(tmp_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
            diarized_md, plain_text = _call_diarize_api(tmp_path)
            progress_bar.progress(1.0)
        except Exception as e:
            container.error(f"說話人辨識失敗：{e}")
            return "", ""
        finally:
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
        container.markdown(diarized_md)
        return diarized_md, plain_text

    # ── 長音檔：自動分段 ──
    n_segs = math.ceil(duration_sec / DIARIZE_MAX_SEC)
    container.info(
        f"音檔長度 {int(duration_sec // 60)} 分 {int(duration_sec % 60)} 秒，"
        f"超過模型上限（{DIARIZE_MAX_SEC // 60} 分鐘），"
        f"自動分成 {n_segs} 段分別辨識。各段說話者標籤獨立，不跨段比對。"
    )

    all_diarized: List[str] = []
    all_plain: List[str] = []
    failed_segs: List[int] = []

    for seg_idx in range(n_segs):
        start_ms = seg_idx * DIARIZE_MAX_SEC * 1000
        end_ms = min((seg_idx + 1) * DIARIZE_MAX_SEC * 1000, len(audio))
        seg = audio[start_ms:end_ms]

        start_min, end_min = int(start_ms / 60000), int(end_ms / 60000)
        container.markdown(f"*處理第 {seg_idx + 1}/{n_segs} 段（{start_min}–{end_min} 分鐘）...*")

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp_path = tmp.name
                seg.export(tmp_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
            seg_md, seg_plain = _call_diarize_api(tmp_path)
            all_diarized.append(
                f"### 第 {seg_idx + 1} 段（{start_min}–{end_min} 分鐘）\n\n{seg_md}"
            )
            all_plain.append(seg_plain)
        except Exception as e:
            failed_segs.append(seg_idx + 1)
            all_diarized.append(f"### 第 {seg_idx + 1} 段（{start_min}–{end_min} 分鐘）\n\n*轉錄失敗：{e}*")
            all_plain.append("")
        finally:
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

        progress_bar.progress((seg_idx + 1) / n_segs)

    diarized_md = "\n\n---\n\n".join(all_diarized)
    plain_text = "\n".join(all_plain)
    container.markdown(diarized_md)

    # 任一段失敗時回傳空 plain_text，觸發呼叫端的 st.stop()
    if failed_segs:
        container.error(
            f"⛔ 第 {', '.join(map(str, failed_segs))} 段說話人辨識失敗，"
            "無法保證逐字稿完整性，已終止後續摘要生成。請重試或縮短音檔。"
        )
        return diarized_md, ""

    return diarized_md, plain_text


# ========== Map-Reduce（GPT‑5 + Responses API）==========
def map_summarize_blocks(flat_sentences: List[str], chunk_size=DEFAULT_MAP_CHUNK_SIZE) -> List[str]:
    blocks = []
    for idx in range(0, len(flat_sentences), chunk_size):
        part = flat_sentences[idx: idx + chunk_size]
        dev_msg = (
            "你是一位會議記錄小幫手，請將下列逐字稿整理為條列式重點（繁體中文）。"
            "要求：每點具體、避免空泛；若有決策/風險/未決問題/行動項目請清楚標記；"
            "只輸出條列重點，不要額外說明。"
        )
        user_msg = "\n".join(part)
        try:
            resp = client.responses.create(
                model=MODEL_MAP,
                input=[
                    {"role": "developer", "content": [{"type": "input_text", "text": dev_msg}]},
                    {"role": "user", "content": [{"type": "input_text", "text": user_msg}]},
                ],
                text={"format": {"type": "text"}},
                tools=[],
            )
            content = resp.output_text or ""
            blocks.append(content.strip())
        except Exception as e:
            blocks.append(f"【API 摘要失敗：{e}】")
    return blocks

def reduce_finalize_json(map_blocks: List[str]) -> Dict[str, Any]:
    dev_msg = (
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
        "4) 確保為合法 JSON。\n\n"
        "=== 分段摘要 ===\n"
        + "\n\n".join(f"[Part {i+1}]\n{blk}" for i, blk in enumerate(map_blocks))
    )
    try:
        resp = client.responses.create(
            model=MODEL_REDUCE,
            input=[{"role": "developer", "content": [{"type": "input_text", "text": dev_msg}]}],
            text={"format": {"type": "text"}},
            tools=[],
        )
        s = (resp.output_text or "").strip()
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1:
            s = s[start:end+1]
        return json.loads(s)
    except Exception as e:
        return {"overall_summary": f"解析 JSON 失敗，請重試或調整提示。錯誤：{e}", "raw": ""}

def reduce_finalize_markdown(map_blocks: List[str]) -> str:
    dev_msg = (
        "你是會議記錄總整專家。請將多個分段摘要整併為『單一份最終會議紀錄（Markdown）』。\n"
        "要求：\n"
        "1) 僅根據提供的分段摘要整併，嚴禁捏造來源沒有的資訊。\n"
        "2) 不輸出 metadata（標題/日期/地點/參與者/時長），只要內容本體。\n"
        "3) 結構：\n"
        "   - 以一段「總結」開場，3~6 句，說清楚整體脈絡與結論。\n"
        "   - 之後用多個小節（## 主題名稱），每節採用短段落敘述為主，可穿插少量條列。\n"
        "   - 若有決策/風險/未決問題，於對應主題內以『決策：』『風險：』『未決：』行內標示。\n"
        "4) 只輸出純 Markdown 內容，不要額外說明。"
        "\n\n=== 分段摘要 ===\n"
        + "\n\n".join(f"[Part {i+1}]\n{blk}" for i, blk in enumerate(map_blocks))
    )
    try:
        resp = client.responses.create(
            model=MODEL_REDUCE,
            input=[{"role": "developer", "content": [{"type": "input_text", "text": dev_msg}]}],
            text={"format": {"type": "text"}},
            tools=[],
        )
        return (resp.output_text or "").strip()
    except Exception as e:
        return f"⚠️ 生成會議摘要失敗：{e}"

# 顯示模式工具：段落群組（僅保留段落模式用）
def group_into_paragraphs(sentences: List[str], max_chars: int = 260, max_sents: int = 4) -> List[str]:
    paras, cur, length = [], [], 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if cur and (len(cur) >= max_sents or length + len(s) > max_chars):
            paras.append(" ".join(cur))
            cur, length = [s], len(s)
        else:
            cur.append(s)
            length += len(s)
    if cur:
        paras.append(" ".join(cur))
    return paras

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
    start_btn = st.button("開始轉錄與摘要")

# ========== 會議資訊（選填）==========
with st.expander("會議資訊（選填，加入摘要）", expanded=False):
    meta_cols = st.columns(2)
    with meta_cols[0]:
        meta_date = st.date_input("會議日期", value=None)
        meta_location = st.text_input("地點", placeholder="例：台北辦公室 / Google Meet")
    with meta_cols[1]:
        meta_participants = st.text_area(
            "參與者（每行一人）",
            height=100,
            placeholder="例：\n張三（主持）\n李四\nAimee"
        )
        meta_title = st.text_input("會議主題", placeholder="例：Q2 產品規劃會議")

# ========== 進階調整 ==========
with st.expander("進階調整（全部設定，可選）", expanded=False):
    st.caption("平常維持預設即可；只有音檔特性特殊時再開啟。")

    st.markdown("###### 轉錄模式")
    use_diarize = st.checkbox(
        "啟用說話人辨識（gpt-4o-transcribe-diarize）",
        value=False,
        help="辨識不同說話者並標記 [說話者 A / B / C...]。注意：此模式不支援 Prompt 引導，且費用較高。"
    )
    if use_diarize:
        st.info("說話人辨識模式：整段音檔一次送出，模型自動切段並標記說話者。")

    st.markdown("###### 音訊前處理")
    cols = st.columns(2)
    with cols[0]:
        do_trim_leading = st.checkbox("去前導靜音（建議開）", value=True)
        do_normalize = st.checkbox("音量正規化到 -20 dBFS（建議開）", value=True)
    with cols[1]:
        use_high_pass = st.checkbox("高通濾波（降低低頻噪）", value=False)
        hp_hz = st.slider("高通截止頻率 (Hz)", 60, 300, 100, 10, disabled=not use_high_pass)
        use_low_pass = st.checkbox("低通濾波（降高頻噪）", value=False)
        lp_hz = st.slider("低通截止頻率 (Hz)", 4000, 12000, 9500, 100, disabled=not use_low_pass)

    st.markdown("###### Prompt 引導（說話人辨識模式下無效）")
    use_prompting = st.checkbox(
        "啟用 Prompt 引導（改善專有名詞拼寫與風格一致）",
        value=False,
        disabled=use_diarize
    )
    glossary_input = st.text_area(
        "專有名詞拼寫清單（每行一個）",
        height=120,
        placeholder="例：\nAimee\nShawn\nBBQ\nZyntriQix",
        disabled=not use_prompting or use_diarize
    )
    style_seed = st.text_area(
        "風格示例（1～3 句示例文本，不是指令）",
        height=80,
        placeholder="例：\n保持簡潔、標點一致。例句：we discuss quarterly outlook and risks.",
        disabled=not use_prompting or use_diarize
    )

if not (f and start_btn):
    st.stop()

# ========== 主流程 ==========
cache_cleanup()
raw_bytes = f.read()

file_mb = len(raw_bytes) / (1024 * 1024)
if file_mb > 100:
    st.warning(f"檔案較大（{file_mb:.0f} MB），轉錄可能需要數分鐘，請耐心等候。")

st.audio(raw_bytes)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["轉錄結果", "重點摘要", "內容解析", "原始內容"])

with tab1:
    with st.status("處理中...", expanded=True) as status:
        status.update(label="儲存與轉檔...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{f.name.split('.')[-1]}") as temp_input:
            temp_input.write(raw_bytes)
            temp_input_path = temp_input.name

        wav_path = temp_input_path
        if not f.name.lower().endswith(".wav"):
            wav_path = temp_input_path + ".wav"
            convert_to_wav(temp_input_path, wav_path)

        status.update(label="載入音檔與前處理...")
        audio = AudioSegment.from_file(wav_path, format="wav")
        if do_trim_leading:
            audio = trim_leading_silence(audio, silence_threshold_db=-30.0, chunk_ms=10)
        if do_normalize:
            audio = normalize_loudness(audio, target_dbfs=-20.0)
        if use_high_pass or use_low_pass:
            audio = apply_filters(audio, use_high_pass=use_high_pass, hp_hz=hp_hz, use_low_pass=use_low_pass, lp_hz=lp_hz)

        duration_sec = len(audio) / 1000
        duration_str = f"{int(duration_sec // 60)} 分 {int(duration_sec % 60)} 秒"
        st.caption(f"音檔時長：{duration_str}　大小：{file_mb:.1f} MB")

        st.markdown("#### 轉錄結果")
        transcript_container = st.empty()
        progress_bar = st.progress(0.0)

        # ── 說話人辨識模式 ──
        if use_diarize:
            status.update(label="說話人辨識轉錄中（整段音檔送出）...")
            diarized_md, plain_text = transcribe_diarize(wav_path, audio, transcript_container, progress_bar)
            if not plain_text.strip():
                # transcribe_diarize 已顯示具體錯誤訊息，這裡直接終止
                st.stop()
            raw_output = plain_text
            # 辨識模式跳過切段去重，直接用 plain_text 做摘要
            flat_sentences = [s for s in split_sentences(plain_text) if s]
            final_md = diarized_md  # 顯示含說話者標籤的版本

        # ── 一般轉錄模式（平行 / 循序）──
        else:
            status.update(label="靜音切段（附最長/最短保護；找不到靜音會回退固定切）...")
            chunks = split_audio_on_silence_safe(audio)
            if not chunks:
                st.error("無法切出有效音訊段，請檢查音檔或調整參數。")
                st.stop()

            mode_label = "循序轉錄中（Prompt 引導）..." if use_prompting else f"平行轉錄中（{MAX_WORKERS} 執行緒）..."
            status.update(label=mode_label)
            all_text = transcribe_all(
                chunks,
                transcript_container,
                progress_bar,
                use_prompting=use_prompting,
                glossary=glossary_input if use_prompting else "",
                style_seed=style_seed if use_prompting else ""
            )
            raw_output = all_text.strip()

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

            pretty_lines = pretty_format_sentences(flat_sentences)
            refined_lines = refine_zh_tw_via_prompt(pretty_lines)
            paras = group_into_paragraphs(refined_lines, max_chars=280, max_sents=4)
            final_md = "\n\n".join(paras)
            transcript_container.markdown(final_md)
            if refined_lines == pretty_lines:
                st.caption("提示：可讀版潤飾/翻譯可能未生效（已回退原文顯示）。")

        st.success("轉錄完成！")
        st.download_button(
            "下載逐字稿（.txt）",
            data=final_md.encode("utf-8"),
            file_name=f"{os.path.splitext(f.name)[0]}_transcript.txt",
            mime="text/plain",
        )

        status.update(label="整併重點（內部計算）...")
        map_blocks_text = map_summarize_blocks(flat_sentences)

        status.update(label="生成最終會議摘要與內容解析...")
        final_minutes = reduce_finalize_json(map_blocks_text)
        final_md_summary = reduce_finalize_markdown(map_blocks_text)

        # 將使用者填寫的 metadata 合併進 JSON
        user_meta = final_minutes.get("metadata", {})
        if meta_title:
            user_meta["title"] = meta_title
        if meta_date:
            user_meta["date"] = str(meta_date)
        if meta_location:
            user_meta["location"] = meta_location
        if meta_participants.strip():
            user_meta["participants"] = [p.strip() for p in meta_participants.splitlines() if p.strip()]
        if use_diarize:
            user_meta["transcription_mode"] = "speaker_diarization"
        final_minutes["metadata"] = user_meta

        with tab2:
            st.markdown(final_md_summary)
            st.download_button(
                "下載會議記錄 JSON",
                data=json.dumps(final_minutes, ensure_ascii=False, indent=2),
                file_name="meeting_minutes.json",
                mime="application/json"
            )

        with tab3:
            render_topics_only(final_minutes, st)

        with tab4:
            st.markdown("#### 原始內容（轉錄原始輸出，未分句／未去重）")
            st.code(raw_output, language="text")

        status.update(label="全部完成！", state="complete", expanded=True)

# 清理暫存
try:
    os.remove(temp_input_path)
    if 'wav_path' in locals() and wav_path != temp_input_path:
        os.remove(wav_path)
except Exception:
    pass
