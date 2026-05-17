import os
import re
import json
import math
import time
import random
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
MODEL_MAP    = "gpt-5.4-mini"               # 分段摘要
MODEL_REDUCE = "gpt-5.4-mini"                 # 總整/潤飾

MAX_WORKERS = 4      # 平行轉錄最大執行緒數

# 上傳格式：MP3 mono 64kbps（符合 OpenAI 官方建議「use a compressed audio format」）
# 對比 16kHz mono WAV（32 KB/s），MP3 64kbps（8 KB/s）流量降 75%，更不易撞 25MB 上限與 rate limit。
UPLOAD_FORMAT = "mp3"
UPLOAD_MP3_BITRATE = "64k"
UPLOAD_SAMPLE_RATE = 16000

# OpenAI Transcription API 硬上限
API_MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB
SAFETY_MARGIN_BYTES = 1 * 1024 * 1024    # 留 1 MB 給 multipart overhead

# Diarize 分段策略：以 byte size 為主、duration 為輔（兩者取較嚴格者）
DIARIZE_MAX_SEC = 1400         # API 經驗值上限（秒）
DIARIZE_TARGET_SEGMENT_SEC = 1200  # 預設切點，留安全餘量

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

# Cache 目錄：優先放在 user-private 家目錄（避免敏感逐字稿洩漏給同主機其他使用者）。
# 順序：~/.cache/anya_stt（Linux/Mac 慣例）→ %LOCALAPPDATA%/anya_stt（Windows）→ tempdir（fallback）。
# 建立失敗時 _CACHE_ENABLED=False，所有讀寫都會優雅降級為 no-op。
def _pick_cache_dir() -> Tuple[str, bool]:
    """選擇 cache 目錄，回傳 (path, is_private)。私有目錄會設 0o700 權限。"""
    candidates: List[str] = []
    # 1. Linux/Mac：~/.cache/anya_stt（XDG_CACHE_HOME 慣例）
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        candidates.append(os.path.join(xdg_cache, "anya_stt"))
    home = os.path.expanduser("~")
    if home and home != "~":
        # Windows：優先用 LOCALAPPDATA（user-scoped）
        local_app = os.environ.get("LOCALAPPDATA")
        if local_app:
            candidates.append(os.path.join(local_app, "anya_stt", "cache"))
        # Linux/Mac fallback
        candidates.append(os.path.join(home, ".cache", "anya_stt"))
    # 最後 fallback：系統 temp（多 user 主機上會被 Codex 點名，但維持頁面可用）
    candidates.append(os.path.join(tempfile.gettempdir(), "anya_stt_cache"))

    for path in candidates:
        try:
            os.makedirs(path, exist_ok=True)
            # POSIX 系統設 0o700（owner only），Windows 上 chmod 是 no-op
            try:
                os.chmod(path, 0o700)
            except Exception:
                pass
            is_private = path != candidates[-1]  # 最後一個是 tempdir，非 private
            return path, is_private
        except Exception:
            continue
    return "", False  # 全部失敗

CACHE_DIR, _CACHE_IS_PRIVATE = _pick_cache_dir()
CACHE_MAX_FILES = 200   # 超過時刪除最舊的檔案
_CACHE_ENABLED = bool(CACHE_DIR)

def cache_cleanup():
    if not _CACHE_ENABLED:
        return
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
    if not _CACHE_ENABLED:
        return None
    path = os.path.join(CACHE_DIR, key + ".txt")
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception:
        return None
    return None

def cache_set_text(key: str, value: str):
    if not _CACHE_ENABLED:
        return
    path = os.path.join(CACHE_DIR, key + ".txt")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(value)
    except Exception:
        pass

def convert_to_wav(input_path: str, output_path: str, target_sr=16000):
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(target_sr).set_channels(1)
    audio.export(output_path, format="wav")
    return output_path


# MP3 encoder 可用性：第一次匯出失敗後 fallback WAV，並記住結果避免每次重試
_MP3_ENCODER_AVAILABLE: Optional[bool] = None  # None=未探測, True=可用, False=不可用


def export_for_upload(segment: AudioSegment) -> str:
    """將音訊段匯出為可上傳給 OpenAI Transcription API 的暫存檔。
    優先使用 MP3（64kbps mono 16kHz）—— 符合官方「use a compressed audio format」建議；
    若環境缺 libmp3lame（少見的精簡 ffmpeg build），自動 fallback 到 WAV。
    回傳：暫存檔路徑（呼叫端負責 os.remove）。
    """
    global _MP3_ENCODER_AVAILABLE
    seg = segment.set_frame_rate(UPLOAD_SAMPLE_RATE).set_channels(1)

    # MP3 路徑：先嘗試一次；失敗後鎖定為不可用，後續直接走 WAV
    if _MP3_ENCODER_AVAILABLE is not False:
        fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        try:
            seg.export(
                tmp_path,
                format="mp3",
                bitrate=UPLOAD_MP3_BITRATE,
                parameters=["-ac", "1", "-ar", str(UPLOAD_SAMPLE_RATE)],
            )
            _MP3_ENCODER_AVAILABLE = True
            return tmp_path
        except Exception:
            _MP3_ENCODER_AVAILABLE = False
            _safe_remove(tmp_path)
            # 落到 WAV fallback

    # WAV fallback：未壓縮，需注意 25MB 上限（呼叫端的 preflight size check 會擋）
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    seg.export(
        tmp_path,
        format="wav",
        parameters=["-ac", "1", "-ar", str(UPLOAD_SAMPLE_RATE)],
    )
    return tmp_path


def _safe_remove(path: Optional[str]) -> None:
    if not path:
        return
    try:
        os.remove(path)
    except Exception:
        pass

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

def _transcribe_chunk(chunk: AudioSegment, prompt: str, use_cache: bool = True) -> Tuple[str, Optional[str]]:
    """單段音訊轉錄，供循序與平行模式共用。
    use_cache=False 時略過讀寫 cache（prompt 引導模式，每次 prompt 不同）。
    回傳：(transcript_text, last_error_message or None)
    """
    cache_key = f"stt_{MODEL_STT}_{_hash_bytes(chunk.raw_data)}"
    if use_cache:
        cached = cache_get_text(cache_key)
        if cached:
            return cached, None

    full_text = ""
    last_error: Optional[str] = None
    tmp_path = None
    try:
        tmp_path = export_for_upload(chunk)
        # 30 秒 chunk 用 MP3 64kbps 約 240KB，遠低於 25MB；不額外檢查 size。
        with open(tmp_path, "rb") as audio_file:
            try:
                resp = client.audio.transcriptions.create(
                    model=MODEL_STT,
                    file=audio_file,
                    response_format="text",
                    prompt=prompt,
                )
                full_text = resp if isinstance(resp, str) else (getattr(resp, "text", None) or "")
            except Exception as e1:
                last_error = f"{type(e1).__name__}: {e1}"
                # 回退：去掉 prompt 再試一次（兼容部分舊端點）
                audio_file.seek(0)
                try:
                    resp = client.audio.transcriptions.create(
                        model=MODEL_STT,
                        file=audio_file,
                        response_format="text",
                        prompt=BASE_PROMPT,
                    )
                    full_text = resp if isinstance(resp, str) else (getattr(resp, "text", None) or "")
                    last_error = None  # 回退成功
                except Exception as e2:
                    last_error = f"{type(e2).__name__}: {e2}"
                    full_text = ""
    except Exception as e_outer:
        last_error = f"{type(e_outer).__name__}: {e_outer}"
    finally:
        _safe_remove(tmp_path)

    if use_cache and full_text.strip():  # 只快取成功的結果
        cache_set_text(cache_key, full_text.strip())
    return full_text, last_error


def _is_rate_limit_error(err_msg: Optional[str]) -> bool:
    """判斷錯誤是否為 rate limit / 429 / quota。"""
    if not err_msg:
        return False
    low = err_msg.lower()
    return ("ratelimit" in low or "rate_limit" in low or "rate limit" in low
            or "429" in low or "quota" in low or "too many requests" in low)


def _parse_retry_after_seconds(err_msg: Optional[str]) -> Optional[float]:
    """從錯誤訊息中嘗試解析 Retry-After / x-ratelimit-reset-* 提示秒數。
    OpenAI 的 RateLimitError 字串通常含 'Please try again in 6.123s' 或 'in 1m12s'。
    """
    if not err_msg:
        return None
    # 'try again in 6.123s' / 'in 6s'
    m = re.search(r"(?:try again in|in)\s+([0-9]+(?:\.[0-9]+)?)\s*s\b", err_msg, re.I)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    # 'in 1m12s' / 'in 1m 12s'
    m = re.search(r"in\s+(?:([0-9]+)\s*m)?\s*([0-9]+(?:\.[0-9]+)?)\s*s\b", err_msg, re.I)
    if m and (m.group(1) or m.group(2)):
        try:
            mins = float(m.group(1) or 0)
            secs = float(m.group(2) or 0)
            return mins * 60 + secs
        except ValueError:
            pass
    return None


def _backoff_with_jitter(base: float, attempt: int, cap: float = 60.0) -> float:
    """指數退避 + 等量隨機 jitter（Full Jitter 演算法，符合 OpenAI 官方建議）。
    base * 2^(attempt-1)，封頂後再隨機介於 [0, value]。
    """
    raw = min(cap, base * (2 ** (attempt - 1)))
    return random.uniform(0, raw)


def _retry_chunk_sequential(
    chunks: List[AudioSegment],
    indices: List[int],
    prompt: str,
    use_cache: bool,
    container,
    label: str = "重試",
    max_attempts: int = 3,
) -> Tuple[Dict[int, str], Dict[int, str]]:
    """循序重試指定 chunk index 清單，含指數退避 + jitter；rate limit 時優先讀取 Retry-After。
    回傳：(成功結果 dict, 失敗錯誤訊息 dict)
    """
    successes: Dict[int, str] = {}
    errors: Dict[int, str] = {}
    for idx in indices:
        last_err: Optional[str] = None
        for attempt in range(1, max_attempts + 1):
            # 等待策略：rate limit 優先讀 API 給的 Retry-After，否則用 jitter backoff
            wait_sec: float
            if _is_rate_limit_error(last_err):
                hinted = _parse_retry_after_seconds(last_err)
                if hinted is not None:
                    wait_sec = hinted + random.uniform(0.5, 2.0)  # 略大於官方提示
                else:
                    wait_sec = _backoff_with_jitter(base=8.0, attempt=attempt, cap=90.0)
            else:
                wait_sec = _backoff_with_jitter(base=3.0, attempt=attempt, cap=30.0)

            container.markdown(
                f"*{label}第 {idx+1} 段（第 {attempt}/{max_attempts} 次，等待 {wait_sec:.1f}s）...*"
            )
            time.sleep(wait_sec)
            text, err = _transcribe_chunk(chunks[idx], prompt, use_cache=use_cache)
            if text.strip():
                successes[idx] = text
                last_err = None
                break
            last_err = err
        if idx not in successes and last_err:
            errors[idx] = last_err
    return successes, errors


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
    # 關鍵設計：失敗段必須「就地重試」，否則 rolling_context 會用空字串前進，
    # 污染下游所有 chunk 的 prompt，違背 prompt 模式存在的目的（術語/風格一致性）。
    # 官方建議：「prompt the model with the transcript of the preceding segment」
    if use_prompting:
        all_text = ""
        rolling_context = ""
        results: Dict[int, str] = {}
        for i, chunk in enumerate(chunks):
            prompt_parts = [BASE_PROMPT]
            extra = build_prompt(rolling_context, glossary, style_seed, max_chars=800)
            if extra:
                prompt_parts.append(extra)
            this_prompt = "\n".join(prompt_parts)

            text, err = _transcribe_chunk(chunk, this_prompt, use_cache=False)

            # 就地重試「有 exception」的段（API 真的失敗）；
            # 空字串 + 無錯誤 = 該段無語音內容（合法），不重試、不終止。
            if not text.strip() and err:
                container.info(f"第 {i+1} 段轉錄失敗（{err[:80]}），立即重試（避免污染下游 context）...")
                succ, errs = _retry_chunk_sequential(
                    chunks, [i], this_prompt, use_cache=False,
                    container=container, label="就地重試"
                )
                text = succ.get(i, "")
                retry_err = errs.get(i)
                # 重試仍是「有 exception」才算失敗；空字串無錯誤視為合法
                if not text.strip() and retry_err:
                    container.error(
                        f"⛔ 第 {i+1} 段重試後仍失敗，已終止流程"
                        f"（避免下游 chunk 在錯誤 context 下繼續）。\n\n"
                        f"**錯誤訊息：** `{retry_err}`\n\n"
                        "可能原因：API rate limit、網路逾時、音訊內容問題。"
                    )
                    st.stop()

            results[i] = text
            all_text += text + "\n"
            rolling_context = (rolling_context + " " + text).strip()[-5000:]
            progress_bar.progress((i + 1) / total)
            container.markdown(f"*{i+1}/{total} 段完成...*\n\n" + all_text)

        return "\n".join(results.get(i, "") for i in range(total)).strip()

    # ── 無 prompt 引導：平行執行 ──
    results: Dict[int, str] = {}
    errors: Dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_transcribe_chunk, chunk, BASE_PROMPT, True): i
            for i, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            i = futures[future]
            try:
                text, err = future.result()
                results[i] = text or ""
                if not (text or "").strip() and err:
                    errors[i] = err
            except Exception as e:
                results[i] = ""
                errors[i] = f"{type(e).__name__}: {e}"
            done = len(results)
            progress_bar.progress(done / total)
            container.markdown(f"*{done}/{total} 段完成...*")

    # 只重試「有 exception」的段（errors 有記錄）；
    # 空字串 + 無錯誤 = API 成功但這段無語音（如靜音、暫停、無對話），合法狀態，不重試。
    error_indices = [i for i in range(total) if errors.get(i)]
    if error_indices:
        # 偵測是否多為 rate limit；若是，先等一個較長視窗讓 quota 回復
        rl_count = sum(1 for i in error_indices if _is_rate_limit_error(errors.get(i)))
        if rl_count >= max(2, len(error_indices) // 2):
            # 嘗試從錯誤訊息中找最大的 Retry-After 提示
            hinted_waits = [
                _parse_retry_after_seconds(errors.get(i))
                for i in error_indices if _is_rate_limit_error(errors.get(i))
            ]
            valid_hints = [w for w in hinted_waits if w is not None]
            cool_down = max(valid_hints) + random.uniform(1, 3) if valid_hints else 30.0
            container.warning(
                f"⏳ 偵測到 {rl_count} 段疑似 API rate limit，"
                f"等待 {cool_down:.1f} 秒讓配額回復"
                f"{'（依 API Retry-After 提示）' if valid_hints else ''}..."
            )
            time.sleep(cool_down)

        container.info(f"開始循序重試失敗段（共 {len(error_indices)} 段，含指數退避）...")
        succ, errs = _retry_chunk_sequential(
            chunks, error_indices, BASE_PROMPT, use_cache=True, container=container
        )
        for i, t in succ.items():
            results[i] = t
            errors.pop(i, None)
        for i, e in errs.items():
            errors[i] = e

    # 統計：只有 errors 仍有記錄才算真正失敗（空字串無錯誤 = 無語音段，視為成功）
    still_failed = [i for i in range(total) if errors.get(i)]
    silent_count = sum(
        1 for i in range(total)
        if not results.get(i, "").strip() and not errors.get(i)
    )

    if still_failed:
        sample_err = errors.get(still_failed[0], "未知錯誤")
        is_rate = sum(1 for i in still_failed if _is_rate_limit_error(errors.get(i)))
        hint = ""
        if is_rate:
            hint = (
                "\n\n**建議：** 偵測到 rate limit，請：\n"
                "1) 等幾分鐘後重試；2) 將 `MAX_WORKERS` 從 4 調降為 2 或 1；"
                "3) 檢查 OpenAI 用量是否達上限。"
            )
        container.error(
            f"⛔ 第 {', '.join(str(i+1) for i in still_failed)} 段轉錄失敗"
            f"（共 {len(still_failed)}/{total} 段）。\n\n"
            f"**首段錯誤訊息：** `{sample_err}`{hint}"
        )
        st.stop()

    if silent_count > 0:
        container.info(
            f"ℹ️ 偵測到 {silent_count}/{total} 段為無語音內容（靜音、暫停或無對話），"
            f"已自動略過。其餘 {total - silent_count} 段轉錄成功。"
        )

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


def _call_diarize_api(tmp_path: str) -> Tuple[str, str]:
    """對單一音訊檔（MP3 / WAV）呼叫 diarize API，回傳 (diarized_md, plain_text)。
    呼叫前應由呼叫端保證檔案大小 < 25MB（API 上限）。
    """
    with open(tmp_path, "rb") as audio_file:
        resp = client.audio.transcriptions.create(
            model=MODEL_DIARIZE,
            file=audio_file,
            response_format="diarized_json",
            chunking_strategy="auto",
        )
    return _parse_diarize_resp(resp)


def _call_diarize_api_with_retry(
    tmp_path: str,
    container,
    label: str = "說話人辨識",
    max_attempts: int = 3,
) -> Tuple[str, str, Optional[str]]:
    """有 retry 的 diarize 呼叫。重試策略對齊一般轉錄：
    - 指數退避 + Full Jitter
    - 429 / rate limit 優先讀 Retry-After 提示
    回傳：(diarized_md, plain_text, last_error or None)。成功時 last_error=None。
    """
    last_err: Optional[str] = None
    for attempt in range(1, max_attempts + 1):
        try:
            md, plain = _call_diarize_api(tmp_path)
            return md, plain, None
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt >= max_attempts:
                break
            # 等待策略：rate limit 優先讀 API 給的 Retry-After，否則用 jitter backoff
            if _is_rate_limit_error(last_err):
                hinted = _parse_retry_after_seconds(last_err)
                if hinted is not None:
                    wait_sec = hinted + random.uniform(0.5, 2.0)
                else:
                    wait_sec = _backoff_with_jitter(base=8.0, attempt=attempt, cap=90.0)
            else:
                wait_sec = _backoff_with_jitter(base=3.0, attempt=attempt, cap=30.0)
            container.markdown(
                f"*{label} 第 {attempt}/{max_attempts} 次失敗（{last_err[:80]}），"
                f"等待 {wait_sec:.1f}s 後重試...*"
            )
            time.sleep(wait_sec)
    return "", "", last_err


# ── Diarize 段落 cache：成功段落直接快取，下次同 audio 同切法可 resume ──
def _diarize_seg_cache_key(seg: AudioSegment) -> str:
    """以段落音訊內容的 hash 為 key（不同切法 → 不同 bytes → 不同 key，天然不衝突）。"""
    return f"diarize_{MODEL_DIARIZE}_{_hash_bytes(seg.raw_data)}"


def cache_get_diarize_segment(seg: AudioSegment) -> Optional[Tuple[str, str]]:
    raw = cache_get_text(_diarize_seg_cache_key(seg))
    if not raw:
        return None
    try:
        data = json.loads(raw)
        md = data.get("md", "")
        plain = data.get("plain", "")
        if md or plain:
            return md, plain
    except Exception:
        return None
    return None


def cache_set_diarize_segment(seg: AudioSegment, md: str, plain: str) -> None:
    try:
        cache_set_text(
            _diarize_seg_cache_key(seg),
            json.dumps({"md": md, "plain": plain}, ensure_ascii=False),
        )
    except Exception:
        pass


def _max_bytes_per_upload() -> int:
    """單次上傳大小上限（API 25MB 扣掉 multipart overhead 安全餘量）。"""
    return API_MAX_UPLOAD_BYTES - SAFETY_MARGIN_BYTES


def _estimated_bytes_per_sec() -> int:
    """估算每秒上傳位元組數。MP3 64kbps = 8000 B/s；WAV 16kHz mono 16-bit = 32000 B/s。"""
    if _MP3_ENCODER_AVAILABLE is False:
        return 32000  # WAV fallback
    return 8000  # MP3（預設或已確認可用）


def _plan_diarize_segments(duration_sec: float) -> List[Tuple[int, int]]:
    """計算 diarize 的分段（毫秒區間清單），以 byte size 為主、duration 為輔。
    保證每段：上傳大小 < 24MB 且 時長 ≤ DIARIZE_MAX_SEC。
    依目前 encoder（MP3 或 WAV fallback）自動調整段長。
    """
    max_bytes = _max_bytes_per_upload()
    bytes_per_sec = _estimated_bytes_per_sec()
    # 反推時長上限：byte 上限 / 每秒位元組數
    max_sec_by_bytes = max_bytes / bytes_per_sec
    # 取「byte 上限」「duration 上限」較小者，再打 0.85 折給編碼變動餘量
    seg_sec = min(max_sec_by_bytes, DIARIZE_MAX_SEC) * 0.85
    seg_sec = max(60, min(seg_sec, DIARIZE_TARGET_SEGMENT_SEC))  # 最少 60 秒、最多 1200 秒

    plan: List[Tuple[int, int]] = []
    seg_ms = int(seg_sec * 1000)
    total_ms = int(duration_sec * 1000)
    start = 0
    while start < total_ms:
        end = min(start + seg_ms, total_ms)
        plan.append((start, end))
        start = end
    return plan


def _namespace_speaker_labels(diarized_md: str, seg_idx: int) -> str:
    """長音檔分段時，將 `[說話者 A]` 改為 `[第 N 段-A]`，避免跨段同字母被誤認為同人。
    OpenAI 官方文件未提供跨段說話者自動拼接機制（`known_speaker_references[]` 需事先提供樣本），
    namespace 前綴是最誠實的處理：使用者一看就知道跨段標籤不互通。
    """
    # 同時處理「[說話者 A]」與「**[說話者 A]**」兩種格式（含 emoji 前綴）
    return re.sub(
        r"\[說話者 ([A-Z0-9]+)\]",
        rf"[第 {seg_idx + 1} 段-\1]",
        diarized_md,
    )


def transcribe_diarize(audio: AudioSegment, container, progress_bar) -> Tuple[str, str]:
    """
    說話人辨識轉錄。
    - 統一以 byte-size 為主、duration 為輔規劃分段（OpenAI API 25MB 硬上限）。
    - MP3 64kbps mono 上傳，使單段最長可達 ~50 分鐘（相較 WAV 的 13 分鐘）。
    - 跨段說話者標籤帶 segment 前綴（API 無自動拼接機制）。
    """
    duration_sec = len(audio) / 1000
    plan = _plan_diarize_segments(duration_sec)
    n_segs = len(plan)

    if n_segs > 1:
        seg_minutes = (plan[0][1] - plan[0][0]) / 60000
        container.info(
            f"音檔長度 {int(duration_sec // 60)} 分 {int(duration_sec % 60)} 秒，"
            f"超過單次上傳上限（每段 ~{seg_minutes:.0f} 分鐘 / 24MB），"
            f"自動分成 {n_segs} 段分別辨識。\n\n"
            f"⚠️ **說話者標籤會帶段落前綴**（如 `第 1 段-A`、`第 2 段-A`），"
            f"避免跨段同字母被誤認為同一人。OpenAI API 不支援跨段自動拼接說話者身份。"
        )

    all_diarized: List[str] = []
    all_plain: List[str] = []
    failed_segs: List[int] = []
    failure_errors: Dict[int, str] = {}
    resumed_count = 0  # 從快取讀取的段數，用於最後提示

    for seg_idx, (start_ms, end_ms) in enumerate(plan):
        seg = audio[start_ms:end_ms]
        start_min, end_min = int(start_ms / 60000), int(end_ms / 60000)

        # ── Resume：先查 cache，命中就跳過 API 呼叫 ──
        cached = cache_get_diarize_segment(seg)
        if cached is not None:
            seg_md, seg_plain = cached
            resumed_count += 1
            container.markdown(
                f"*第 {seg_idx + 1}/{n_segs} 段（{start_min}–{end_min} 分鐘）：✓ 從快取讀取（resume）*"
            )
        else:
            if n_segs > 1:
                container.markdown(f"*處理第 {seg_idx + 1}/{n_segs} 段（{start_min}–{end_min} 分鐘）...*")
            else:
                container.markdown("*正在送出整段音檔進行說話人辨識，請稍候...*")

            tmp_path = None
            seg_md, seg_plain = "", ""
            seg_err: Optional[str] = None
            try:
                tmp_path = export_for_upload(seg)
                # Preflight size check：MP3 仍超 25MB 時 fail-fast，避免 API 回 413
                file_size = os.path.getsize(tmp_path)
                if file_size > _max_bytes_per_upload():
                    raise RuntimeError(
                        f"段落上傳大小 {file_size / 1024 / 1024:.1f}MB 超過 API 上限 "
                        f"{API_MAX_UPLOAD_BYTES / 1024 / 1024:.0f}MB"
                    )

                seg_md, seg_plain, seg_err = _call_diarize_api_with_retry(
                    tmp_path,
                    container,
                    label=f"第 {seg_idx + 1}/{n_segs} 段說話人辨識",
                )
            except Exception as e:
                seg_err = f"{type(e).__name__}: {e}"
            finally:
                _safe_remove(tmp_path)

            # 成功才寫入 cache（resume 用）；失敗仍進入 failed_segs
            if not seg_err and (seg_md or seg_plain):
                cache_set_diarize_segment(seg, seg_md, seg_plain)
            else:
                failed_segs.append(seg_idx + 1)
                failure_errors[seg_idx + 1] = seg_err or "未知錯誤"

        # 顯示與累積（無論成功失敗都附段落標題，方便 resume 後對應）
        if n_segs > 1 and seg_md:
            seg_md = _namespace_speaker_labels(seg_md, seg_idx)
            all_diarized.append(
                f"### 第 {seg_idx + 1} 段（{start_min}–{end_min} 分鐘）\n\n{seg_md}"
            )
        elif n_segs > 1 and not seg_md:
            err_msg = failure_errors.get(seg_idx + 1, "未知錯誤")
            all_diarized.append(
                f"### 第 {seg_idx + 1} 段（{start_min}–{end_min} 分鐘）\n\n*轉錄失敗：{err_msg}*"
            )
        else:
            all_diarized.append(seg_md or f"*轉錄失敗：{failure_errors.get(seg_idx + 1, '未知錯誤')}*")
        all_plain.append(seg_plain)

        progress_bar.progress((seg_idx + 1) / n_segs)

    diarized_md = "\n\n---\n\n".join(all_diarized) if n_segs > 1 else (all_diarized[0] if all_diarized else "")
    plain_text = "\n".join(all_plain)
    container.markdown(diarized_md)

    if resumed_count > 0:
        container.success(
            f"✓ 已從快取讀取 {resumed_count}/{n_segs} 段（resume，省略重複 API 呼叫）"
        )

    # 任一段失敗時回傳空 plain_text，觸發呼叫端的 st.stop()
    if failed_segs:
        sample_err = failure_errors.get(failed_segs[0], "未知錯誤")
        success_count = n_segs - len(failed_segs)
        resume_hint = (
            f"\n\n💾 **已成功的 {success_count} 段已存入快取**，"
            f"重新執行時會自動跳過、只重跑失敗段，不會浪費已完成的工作。"
        ) if success_count > 0 else ""
        container.error(
            f"⛔ 第 {', '.join(map(str, failed_segs))} 段說話人辨識失敗"
            f"（共 {len(failed_segs)}/{n_segs} 段）。\n\n"
            f"**首段錯誤訊息：** `{sample_err}`\n\n"
            f"已重試 3 次仍無法完成。無法保證逐字稿完整性，已終止後續摘要生成。"
            f"{resume_hint}"
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

    # 按鈕與「啟用說話人辨識」toggle 同一行
    btn_col, toggle_col = st.columns([1, 2], vertical_alignment="left")
    with btn_col:
        start_btn = st.button("開始轉錄與摘要", type="primary")
    with toggle_col:
        use_diarize = st.toggle(
            "啟用說話人辨識",
            value=False,
            help=(
                "辨識不同說話者並標記 [說話者 A / B / C...]。"
                "使用 gpt-4o-transcribe-diarize 模型。"
            ),
        )

    if use_diarize:
        st.warning(
            "⚠️ **Diarize 模式適合多人會議**。對單人簡報、訪談、純講者錄音，"
            "轉錄品質會比一般模式差很多（diarize 模型不支援 prompt 引導，"
            "也無法用詞彙表修正專有名詞）。\n\n"
            "**建議：** 確認音檔是多人對話再啟用；若只有一位主要說話者，請保持關閉以獲得最佳品質。"
        )

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
            diarized_md, plain_text = transcribe_diarize(audio, transcript_container, progress_bar)
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
