# -*- coding: utf-8 -*-
# 會議錄音 → 直播逐字＋摘要（Safe UI Writer：所有 UI 僅主執行緒更新）
# 管線：STT Producer →（並行運算）RefineBatcher + MapBatcher → 主執行緒輪詢回填 → 最後 Reduce

import os, re, json, hashlib, tempfile, multiprocessing, time, concurrent.futures
from typing import List, Dict, Any, Optional

import streamlit as st
from openai import OpenAI
from pydub import AudioSegment, silence
from pydub.utils import which

# ========== 基本設定 ==========
st.set_page_config(page_title="會議錄音 → 直播逐字＋摘要", page_icon="📝", layout="wide")

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
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="pink-card header-pill"><span class="emoji">✍️</span> 安妮亞開會不漏接：逐字 × 摘要</div>', unsafe_allow_html=True)

# 檢查 FFmpeg
AudioSegment.converter = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")
if not AudioSegment.converter or not AudioSegment.ffprobe:
    st.error("找不到 ffmpeg/ffprobe，請先於系統安裝後再試。")
    st.stop()

# Key
OPENAI_KEY = st.secrets.get("OPENAI_KEY", os.getenv("OPENAI_API_KEY"))
if not OPENAI_KEY:
    st.error("找不到 API Key，請在 Streamlit Secrets 設定 OPENAI_KEY 或環境變數 OPENAI_API_KEY。")
    st.stop()
client = OpenAI(api_key=OPENAI_KEY)

# ========== 參數 ==========
MODEL_STT    = "gpt-4o-mini-transcribe"
MODEL_MAP    = "gpt-5-mini"
MODEL_REDUCE = "gpt-4.1"

# 切段參數
MIN_SILENCE_LEN_MS = 700
KEEP_SILENCE_MS    = 300
SILENCE_DB_OFFSET  = 16
OVERLAP_MS         = 1200
MAX_CHUNK_MS       = 30_000
MIN_CHUNK_MS       = 2_000
FALLBACK_WINDOW_MS = 20_000

# 並行運算池（只做運算，不碰 UI）
MAX_WORKERS_REFINE = min(2, multiprocessing.cpu_count())
MAX_WORKERS_MAP    = min(2, multiprocessing.cpu_count())

# 微批次（45~60 分鐘建議值，可在 UI 調）
REFINE_MAX_LINES = 80
REFINE_MAX_CHARS = 6000
REFINE_MAX_WAIT_S = 0.35

MAP_MAX_LINES = 30
MAP_MAX_CHARS = 4000
MAP_MAX_WAIT_S = 0.35

DEFAULT_MAP_CHUNK_SIZE = 40

CACHE_DIR = ".stt_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ========== 工具 ==========
def _hash_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

def cache_get_text(key: str) -> Optional[str]:
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
    gain = target_dbfs - audio.dBFS
    return audio.apply_gain(gain)

def trim_leading_silence(audio: AudioSegment, silence_threshold_db: float = -30.0, chunk_ms: int = 10) -> AudioSegment:
    trim_ms = 0
    while trim_ms < len(audio) and audio[trim_ms:trim_ms+chunk_ms].dBFS < silence_threshold_db:
        trim_ms += chunk_ms
    return audio[trim_ms:]

def apply_filters(audio: AudioSegment, use_high_pass=False, hp_hz=100, use_low_pass=False, lp_hz=9500) -> AudioSegment:
    out = audio
    if use_high_pass: out = out.high_pass_filter(hp_hz)
    if use_low_pass:  out = out.low_pass_filter(lp_hz)
    return out

def split_audio_on_silence_safe(audio: AudioSegment) -> List[AudioSegment]:
    silence_thresh = audio.dBFS - SILENCE_DB_OFFSET
    raw_chunks = silence.split_on_silence(audio, min_silence_len=MIN_SILENCE_LEN_MS,
                                          silence_thresh=silence_thresh, keep_silence=KEEP_SILENCE_MS)
    if not raw_chunks:
        chunks, i = [], 0
        while i < len(audio):
            end = min(i + FALLBACK_WINDOW_MS, len(audio))
            chunks.append(audio[i:end]); i = end
    else:
        filtered = []
        for c in raw_chunks:
            if len(c) < 250:
                if filtered: filtered[-1] = filtered[-1] + c
                else:        filtered.append(c)
            else:
                filtered.append(c)
        if not filtered: filtered = raw_chunks

        chunks = []
        for i, c in enumerate(filtered):
            if i == 0: chunks.append(c)
            else:
                prev = filtered[i-1]
                safe_overlap = min(OVERLAP_MS, len(prev))
                chunks.append((prev[-safe_overlap:] if safe_overlap>0 else AudioSegment.silent(duration=0)) + c)

    normalized = []
    for seg in chunks:
        if len(seg) <= MAX_CHUNK_MS:
            normalized.append(seg)
        else:
            start = 0
            while start < len(seg):
                end = min(start + MAX_CHUNK_MS, len(seg))
                normalized.append(seg[start:end]); start = end

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
    for i in range(0, len(parts)-1, 2):
        s = (parts[i] + parts[i+1]).strip()
        if s: result.append(s)
    if len(parts) % 2 != 0:
        tail = parts[-1].strip()
        if tail: result.append(tail)
    return result

# ====== 去重（相鄰視窗 + Jaccard trigram）======
def _norm_for_dedupe(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r'\s+', '', s)
    s = (s.replace('，', ',').replace('。', '.')
         .replace('！', '!').replace('？', '?')
         .replace('；', ';').replace('（', '(').replace('）', ')'))
    return s

def _jaccard_trigram(a: str, b: str) -> float:
    n = 3
    if len(a) < n or len(b) < n: return 1.0 if a == b else 0.0
    A = {a[i:i+n] for i in range(len(a)-n+1)}
    B = {b[i:i+n] for i in range(len(b)-n+1)}
    un = len(A | B)
    return (len(A & B) / un) if un else 0.0

def dedupe_against_prev_fast(curr: List[str], prev: List[str], threshold=0.88, max_prev=12) -> List[str]:
    if not curr: return []
    tail = prev[-max_prev:] if prev else []
    tail_norm = [_norm_for_dedupe(p) for p in tail]
    out = []
    for s in curr:
        ns = _norm_for_dedupe(s)
        if ns in tail_norm: continue
        similar = False
        for pn in tail_norm:
            if not pn: continue
            if abs(len(ns)-len(pn)) > int(max(len(ns), len(pn)) * 0.4): continue
            if _jaccard_trigram(ns, pn) >= threshold: similar = True; break
        if not similar: out.append(s)
    return out

def add_cjk_spacing(text: str) -> str:
    text = re.sub(r'([\u4e00-\u9fff])([A-Za-z0-9%#@&])', r'\1 \2', text)
    text = re.sub(r'([A-Za-z0-9%#@&])([\u4e00-\u9fff])', r'\1 \2', text)
    return text

def normalize_symbols(text: str) -> str:
    text = text.replace("％", "%").replace("＄", "$").replace("–", "-").replace("—", "-")
    text = text.replace("\u200b", "").replace("\u200c", "")
    return text

def pretty_format_sentences(sentences: List[str]) -> List[str]:
    return [normalize_symbols(add_cjk_spacing(s)) for s in sentences]

# ====== LLM 呼叫（潤飾、Map、Reduce）======
def refine_zh_tw_via_prompt(lines: List[str]) -> List[str]:
    if not lines: return lines
    SEP = "\u241E"; MAX_BATCH_CHARS = 9000; MAX_BATCH_LINES = 120
    def _refine_batch(batch: List[str]) -> List[str]:
        blob = SEP.join(batch)
        dev_msg = (
            "你將收到多行逐字稿，請逐行『潤飾＋必要時翻譯』為正體中文（台灣用語）。\n"
            "1) 不可捏造；2) 英文改繁體；3) 不合併/不拆分；4) 保留數字與符號；"
            "5) 用台灣用語；6) 使用 ␞ 分隔並維持行數。只輸出文本，不要解釋。"
        )
        try:
            resp = client.responses.create(
                model=MODEL_REDUCE,
                input=[
                    {"role":"developer","content":[{"type":"input_text","text":dev_msg}]},
                    {"role":"user","content":[{"type":"input_text","text":blob}]},
                ],
                text={"format":{"type":"text"}},
                tools=[],
            )
            out = (resp.output_text or "").rstrip("\n")
            out_lines = out.split(SEP) if SEP in out else out.split("\n")
            return out_lines if len(out_lines)==len(batch) else batch
        except Exception:
            return batch

    refined_all, batch, size = [], [], 0
    for s in lines:
        if (len(batch) >= MAX_BATCH_LINES) or (size + len(s) + 1 > MAX_BATCH_CHARS):
            refined_all.extend(_refine_batch(batch)); batch, size = [], 0
        batch.append(s); size += len(s) + 1
    if batch: refined_all.extend(_refine_batch(batch))
    return refined_all if refined_all else lines

def map_once_to_bullets(lines: List[str]) -> str:
    part = "\n".join(lines)
    dev_msg = (
        "你是一位會議記錄小幫手，請將下列逐字稿整理為條列式重點（繁體中文）。"
        "要求：具體、避免空泛；若有決策/風險/未決/行動項目請標示；只輸出條列。"
    )
    try:
        resp = client.responses.create(
            model=MODEL_MAP,
            input=[
                {"role":"developer","content":[{"type":"input_text","text":dev_msg}]},
                {"role":"user","content":[{"type":"input_text","text":part}]},
            ],
            text={"format":{"type":"text"}},
            tools=[],
        )
        return (resp.output_text or "").strip()
    except Exception as e:
        return f"- 【API 摘要失敗：{e}】"

def reduce_finalize_markdown(map_blocks: List[str]) -> str:
    dev_msg = (
        "你是會議記錄總整專家。請將多個分段摘要整併為『單一份最終會議紀錄（Markdown）』。\n"
        "1) 僅依來源；2) 開頭 3~6 句總結；3) 多個 ## 主題；4) 以『決策：』『未決：』『風險：』標示；"
        "5) 只輸出 Markdown。"
        "\n\n=== 分段摘要 ===\n" + "\n\n".join(f"[Part {i+1}]\n{blk}" for i, blk in enumerate(map_blocks))
    )
    try:
        resp = client.responses.create(
            model=MODEL_REDUCE,
            input=[{"role":"developer","content":[{"type":"input_text","text":dev_msg}]}],
            text={"format":{"type":"text"}},
            tools=[],
        )
        return (resp.output_text or "").strip()
    except Exception as e:
        return f"⚠️ 生成會議摘要失敗：{e}"

def reduce_finalize_json(map_blocks: List[str]) -> Dict[str, Any]:
    dev_msg = (
        "你是會議記錄總整專家。請將多個分段摘要合併成結構化 JSON，包含：\n"
        "- metadata, topics[], decisions[], risks[], open_questions[], action_items[], overall_summary\n"
        "要求：不可捏造，未知留空，合法 JSON。\n\n"
        "=== 分段摘要 ===\n" + "\n\n".join(f"[Part {i+1}]\n{blk}" for i, blk in enumerate(map_blocks))
    )
    try:
        resp = client.responses.create(
            model=MODEL_REDUCE,
            input=[{"role":"developer","content":[{"type":"input_text","text":dev_msg}]}],
            text={"format":{"type":"text"}},
            tools=[],
        )
        s = (resp.output_text or "").strip()
        start, end = s.find("{"), s.rfind("}")
        if start != -1 and end != -1: return json.loads(s[start:end+1])
    except Exception as e:
        return {"overall_summary": f"解析 JSON 失敗：{e}"}
    return {"overall_summary": "解析 JSON 失敗（未知原因）"}

# ====== 段落分組（渲染用）======
def group_into_paragraphs(sentences: List[str], max_chars=260, max_sents=4) -> List[str]:
    paras, cur, length = [], [], 0
    for s in sentences:
        s = s.strip()
        if not s: continue
        if cur and (len(cur) >= max_sents or length + len(s) > max_chars):
            paras.append(" ".join(cur)); cur, length = [s], len(s)
        else:
            cur.append(s); length += len(s)
    if cur: paras.append(" ".join(cur))
    return paras

# ====== 新增：topics 正規化，避免 'str' 沒有 get 的錯 ======
def normalize_topics(raw) -> List[Dict[str, Any]]:
    if not raw:
        return []
    # 統一成 list
    if isinstance(raw, (str, int, float)):
        raw = [raw]
    if isinstance(raw, dict):
        raw = [raw]

    out: List[Dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            out.append({
                "title": item.get("title") or item.get("topic") or item.get("name") or "主題",
                "key_points": item.get("key_points") or [],
                "decisions": item.get("decisions") or [],
                "risks": item.get("risks") or [],
                "open_questions": item.get("open_questions") or [],
            })
        else:
            out.append({
                "title": str(item),
                "key_points": [],
                "decisions": [],
                "risks": [],
                "open_questions": [],
            })
    return out

# ========== 批次器（只做運算，不碰 UI；主執行緒負責 poll 與 render）==========
class RefineBatcher:
    def __init__(self, executor: concurrent.futures.ThreadPoolExecutor,
                 max_lines=80, max_chars=6000, max_wait_s=0.35):
        self.exec = executor
        self.max_lines, self.max_chars, self.max_wait_s = max_lines, max_chars, max_wait_s
        self.grouped_tail: List[str] = []  # 相鄰視窗用
        self.batch: List[str] = []
        self.batch_chars = 0
        self.last_flush = time.monotonic()
        self.pending: Dict[int, concurrent.futures.Future] = {}
        self.buffer: Dict[int, List[str]] = {}
        self.next_emit = 0
        self.seq = 0
        self.refined_lines_all: List[str] = []  # 主執行緒收集後渲染
        self.raw_lines_all: List[str] = []      # 可作為摘要輸入
        self.done_chunks = 0
        self.total_chunks = 0

    def set_total(self, n: int):
        self.total_chunks = n

    def add_sentences(self, sents: List[str]):
        # 相鄰視窗去重 → pretty
        unique = sents if not self.grouped_tail else dedupe_against_prev_fast(
            sents, self.grouped_tail, threshold=0.88, max_prev=12
        )
        self.grouped_tail = unique
        pretty = pretty_format_sentences(unique)
        self.raw_lines_all.extend(pretty)

        now = time.monotonic()
        timeup = (now - self.last_flush) >= self.max_wait_s
        for s in pretty:
            if not s.strip(): continue
            if (len(self.batch) >= self.max_lines) or (self.batch_chars + len(s) > self.max_chars) or timeup:
                self._submit_current_batch()
                timeup = False
            self.batch.append(s); self.batch_chars += len(s)
        self.done_chunks += 1

    def _submit_current_batch(self):
        if not self.batch: return
        bid = self.seq; payload = self.batch[:]
        self.buffer[bid] = payload
        self.pending[bid] = self.exec.submit(refine_zh_tw_via_prompt, payload)
        self.seq += 1
        self.batch, self.batch_chars = [], 0
        self.last_flush = time.monotonic()

    def poll_emit(self) -> bool:
        # 主執行緒呼叫：把已完成的批次依序取回
        updated = False
        while self.next_emit in self.pending and self.pending[self.next_emit].done():
            fut = self.pending.pop(self.next_emit)
            try:
                refined = fut.result()
            except Exception:
                refined = self.buffer.get(self.next_emit, [])
            self.refined_lines_all.extend(refined)
            self.buffer.pop(self.next_emit, None)
            self.next_emit += 1
            updated = True
        return updated

    def flush(self):
        # STT 結束後：送出殘批
        self._submit_current_batch()

    def is_all_done(self) -> bool:
        return (not self.pending) and (not self.batch)

class MapBatcher:
    def __init__(self, executor: concurrent.futures.ThreadPoolExecutor,
                 max_lines=30, max_chars=4000, max_wait_s=0.35):
        self.exec = executor
        self.max_lines, self.max_chars, self.max_wait_s = max_lines, max_chars, max_wait_s
        self.batch: List[str] = []
        self.batch_chars = 0
        self.last_flush = time.monotonic()
        self.pending: Dict[int, concurrent.futures.Future] = {}
        self.buffer: Dict[int, List[str]] = {}
        self.blocks: List[str] = []
        self.next_emit = 0
        self.seq = 0
        self.done_chunks = 0
        self.total_chunks = 0

    def set_total(self, n: int):
        self.total_chunks = n

    def add_sentences(self, sents: List[str]):
        # Map 可直接用 pretty（不一定要去重，若要省成本可也餵去重後的）
        pretty = pretty_format_sentences(sents)
        now = time.monotonic()
        timeup = (now - self.last_flush) >= self.max_wait_s
        for s in pretty:
            if not s.strip(): continue
            if (len(self.batch) >= self.max_lines) or (self.batch_chars + len(s) > self.max_chars) or timeup:
                self._submit_current_batch()
                timeup = False
            self.batch.append(s); self.batch_chars += len(s)
        self.done_chunks += 1

    def _submit_current_batch(self):
        if not self.batch: return
        bid = self.seq; payload = self.batch[:]
        def _task(lines: List[str], idx: int) -> str:
            content = map_once_to_bullets(lines)
            return f"### 即時重點 Part {idx+1}\n\n{content}"
        self.buffer[bid] = payload
        self.pending[bid] = self.exec.submit(_task, payload, bid)
        self.seq += 1
        self.batch, self.batch_chars = [], 0
        self.last_flush = time.monotonic()

    def poll_emit(self) -> bool:
        updated = False
        while self.next_emit in self.pending and self.pending[self.next_emit].done():
            fut = self.pending.pop(self.next_emit)
            try:
                md = fut.result()
            except Exception:
                md = "（本批摘要回傳失敗）"
            self.blocks.append(md)
            self.buffer.pop(self.next_emit, None)
            self.next_emit += 1
            updated = True
        return updated

    def flush(self):
        self._submit_current_batch()

    def is_all_done(self) -> bool:
        return (not self.pending) and (not self.batch)

# ========== STT 主迴圈（主執行緒負責所有 UI）==========
def build_prompt(prev_text: str, glossary: str, style_seed: str, max_tokens: int = 220) -> str:
    parts = ["請全程使用正體中文（繁體，台灣用語）。"]
    if style_seed.strip(): parts.append(style_seed.strip())
    if glossary.strip():
        words = [w.strip() for w in glossary.splitlines() if w.strip()]
        if words: parts.append("Glossary: " + ", ".join(words))
    if prev_text.strip():
        tail = prev_text.strip()[-1200:] if len(prev_text) > 1200 else prev_text.strip()
        parts.append(tail)
    toks = " ".join(parts).split()
    if len(toks) > max_tokens: parts = [" ".join(toks[-max_tokens:])]
    return "\n".join(parts).strip()

def stream_transcribe_all(chunks: List[AudioSegment], container, progress_bar,
                          use_prompting=False, glossary="", style_seed="") -> str:
    all_text, rolling_context = "", ""
    last_flush = 0.0
    FLUSH_INTERVAL = 0.15
    for i, chunk in enumerate(chunks):
        chunk_hash = _hash_bytes(chunk.raw_data)
        cache_key = f"stt_{MODEL_STT}_{chunk_hash}"
        cached = cache_get_text(cache_key)
        if cached:
            all_text += cached + "\n"
            rolling_context = (rolling_context + " " + cached).strip()[-5000:]
            progress_bar.progress((i + 1) / len(chunks))
            container.markdown(all_text)
            yield cached  # 把這段文本回傳給上層邏輯
            continue

        full_text = ""
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp_path = tmp.name
                chunk.export(tmp_path, format="wav", parameters=["-ac","1","-ar","16000"])
            with open(tmp_path, "rb") as audio_file:
                extra = {}
                if use_prompting:
                    prompt_str = build_prompt(rolling_context, glossary, style_seed, max_tokens=220)
                    if prompt_str: extra["prompt"] = prompt_str
                try:
                    stream = client.audio.transcriptions.create(
                        model=MODEL_STT,
                        file=audio_file,
                        response_format="text",
                        prompt=("This audio contains a discussion or presentation. Preserve original language per sentence."),
                        stream=True,
                        **extra
                    )
                except Exception:
                    try:
                        stream = client.audio.transcriptions.create(
                            model=MODEL_STT,
                            file=audio_file,
                            response_format="text",
                            prompt=("This audio contains a discussion or presentation. Preserve original language per sentence."),
                            stream=True
                        )
                        container.warning("此轉錄端點不支援 prompt，引導已自動停用（本次）。")
                    except Exception as e2:
                        container.error(f"API 轉錄失敗：{e2}")
                        stream = None

                if stream is not None:
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
        finally:
            if tmp_path:
                try: os.remove(tmp_path)
                except Exception: pass

        txt = full_text.strip()
        cache_set_text(cache_key, txt)
        all_text += txt + "\n"
        rolling_context = (rolling_context + " " + txt).strip()[-5000:]
        progress_bar.progress((i + 1) / len(chunks))
        container.markdown(all_text)
        yield txt  # 回傳本段文本
    # 迴圈結束
    return

# ========== UI：上傳與進階調整 ==========
with st.expander("上傳會議錄音檔案", expanded=True):
    f = st.file_uploader("請上傳音檔（.wav, .mp3, .m4a, .mp4, .webm）", type=["wav","mp3","m4a","mp4","webm"])
    start_btn = st.button("開始 Streaming 轉錄與摘要")

with st.expander("進階調整（全部設定，可選）", expanded=False):
    st.caption("即時性↑：把等待時間/單批行數調小；省成本↑：反之。")
    cols = st.columns(2)
    with cols[0]:
        do_trim_leading = st.checkbox("去前導靜音（建議開）", True)
        do_normalize    = st.checkbox("音量正規化到 -20 dBFS（建議開）", True)
        use_high_pass   = st.checkbox("高通濾波（降低頻噪）", False)
        hp_hz           = st.slider("高通截止頻率 (Hz)", 60, 300, 100, 10, disabled=not use_high_pass)
        use_low_pass    = st.checkbox("低通濾波（降高頻噪）", False)
        lp_hz           = st.slider("低通截止頻率 (Hz)", 4000, 12000, 9500, 100, disabled=not use_low_pass)
    with cols[1]:
        use_prompting = st.checkbox("STT Prompt 引導（專有名詞拼寫更穩）", False)
        glossary_input = st.text_area("專有名詞拼寫（每行一個）", height=100, placeholder="Aimee\nShawn\nBBQ")
        style_seed     = st.text_area("風格示例（非指令）", height=60, placeholder="例：用詞精簡，一致標點。")

    st.markdown("###### 並行潤飾微批次（塞車怎麼調？）")
    REFINE_MAX_WAIT_S = st.slider("Refine 等待秒數", 0.10, 0.80, REFINE_MAX_WAIT_S, 0.05,
                                  help="越小越即時（API 次數↑）；塞車請先調小這個。")
    REFINE_MAX_LINES  = st.slider("Refine 單批最大行數", 20, 140, REFINE_MAX_LINES, 5,
                                  help="越小越即時；看到排隊久→調小。")
    REFINE_MAX_CHARS  = st.slider("Refine 單批最大字數", 2000, 12000, REFINE_MAX_CHARS, 500)
    MAX_WORKERS_REFINE = st.slider("Refine 工人數（1~2）", 1, 2, MAX_WORKERS_REFINE, 1,
                                   help="1 起步；塞車再開到 2（成本↑）。")

    st.markdown("###### 即時 Map 微批次")
    MAP_MAX_WAIT_S = st.slider("Map 等待秒數", 0.10, 0.80, MAP_MAX_WAIT_S, 0.05)
    MAP_MAX_LINES  = st.slider("Map 單批最大行數", 10, 80, MAP_MAX_LINES, 2)
    MAP_MAX_CHARS  = st.slider("Map 單批最大字數", 1000, 10000, MAP_MAX_CHARS, 250)
    MAX_WORKERS_MAP = st.slider("Map 工人數（1~2）", 1, 2, MAX_WORKERS_MAP, 1)

if not (f and start_btn):
    st.stop()

# ========== 主流程 ==========
raw_bytes = f.read()
st.audio(raw_bytes)
tab1, tab2, tab3, tab4 = st.tabs(["轉錄結果", "重點摘要", "內容解析", "原始內容"])

with tab2:
    st.markdown("#### 即時重點（Map streaming）")
    map_stream_container = st.empty()
    map_progress = st.progress(0.0)
    st.divider()
    final_summary_placeholder = st.empty()

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
            audio = apply_filters(audio, use_high_pass=use_high_pass, hp_hz=hp_hz,
                                  use_low_pass=use_low_pass, lp_hz=lp_hz)

        status.update(label="靜音切段（有保護；找不到靜音則回退固定切）...")
        chunks = split_audio_on_silence_safe(audio)
        if not chunks:
            st.error("無法切出有效音訊段，請檢查音檔或調整參數。")
            st.stop()

        st.markdown("#### 轉錄結果")
        stream_container = st.empty()
        progress_bar = st.progress(0.0)

        # 建立運算池（只做 API 呼叫與運算，不更新 UI）
        refine_pool = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS_REFINE)
        map_pool    = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS_MAP)
        refine = RefineBatcher(refine_pool, REFINE_MAX_LINES, REFINE_MAX_CHARS, REFINE_MAX_WAIT_S)
        mapper = MapBatcher(map_pool, MAP_MAX_LINES, MAP_MAX_CHARS, MAP_MAX_WAIT_S)
        refine.set_total(len(chunks)); mapper.set_total(len(chunks))

        status.update(label="逐段 Streaming 轉錄中（並行運算：潤飾＋即時摘要）...")
        all_text = ""
        # STT 主迴圈：每段結束→丟進兩個 batcher；每圈輪詢已完成批次→主執行緒更新 UI
        for txt in stream_transcribe_all(chunks, stream_container, progress_bar,
                                         use_prompting=use_prompting,
                                         glossary=glossary_input or "",
                                         style_seed=style_seed or ""):
            all_text += txt + "\n"
            sents = split_sentences(txt)
            refine.add_sentences(sents)
            mapper.add_sentences(sents)

            # 輪詢已完成批次並渲染（只在主執行緒）
            updated_refine = refine.poll_emit()
            if updated_refine:
                paras = group_into_paragraphs(refine.refined_lines_all, max_chars=280, max_sents=4)
                stream_container.markdown("\n\n".join(paras))
            if mapper.poll_emit():
                map_stream_container.markdown("\n\n".join(mapper.blocks))

            # 進度（用處理段數近似推進）
            if refine.total_chunks:
                st.session_state["__refine_prog"] = min(1.0, refine.done_chunks / refine.total_chunks)
            if mapper.total_chunks:
                st.session_state["__map_prog"] = min(1.0, mapper.done_chunks / mapper.total_chunks)
                map_progress.progress(st.session_state["__map_prog"])

        # 收尾：送出殘批並等待全部完成，期間持續輪詢與渲染
        refine.flush(); mapper.flush()
        t0 = time.monotonic()
        while not (refine.is_all_done() and mapper.is_all_done()):
            any_r = refine.poll_emit()
            any_m = mapper.poll_emit()
            if any_r:
                paras = group_into_paragraphs(refine.refined_lines_all, max_chars=280, max_sents=4)
                stream_container.markdown("\n\n".join(paras))
            if any_m:
                map_stream_container.markdown("\n\n".join(mapper.blocks))
            if not (any_r or any_m):
                time.sleep(0.05)
            # 簡單超時保護：避免極端阻塞（發生時保留已完成內容）
            if time.monotonic() - t0 > 120:
                st.warning("背景批次等待逾時，已顯示目前完成的內容。")
                break

        refine_pool.shutdown(wait=False)
        map_pool.shutdown(wait=False)

        # 最終可讀內容
        refined_lines = refine.refined_lines_all if refine.refined_lines_all else refine.raw_lines_all
        paras = group_into_paragraphs(refined_lines, max_chars=280, max_sents=4)
        stream_container.markdown("\n\n".join(paras))
        st.success("Transcription + Refine complete!")

        status.update(label="整併重點（Reduce 中）...")
        map_blocks_text = mapper.blocks[:] if mapper.blocks else [map_once_to_bullets(refine.raw_lines_all)]
        final_md_summary = reduce_finalize_markdown(map_blocks_text)
        final_summary_placeholder.markdown(final_md_summary)
        final_minutes = reduce_finalize_json(map_blocks_text)

        # 類型保護：確保 final_minutes 至少是 dict，避免後續 .get 爆炸
        if not isinstance(final_minutes, dict):
            final_minutes = {"overall_summary": str(final_minutes)}

        with tab3:
            st.markdown("#### 主題")
            # 先正規化 topics，避免 'str' 沒有 get 的錯
            topics_raw = (final_minutes or {}).get("topics", [])
            topics = normalize_topics(topics_raw)

            # 除錯開關：需要時可檢視原始 JSON 結構
            if st.checkbox("顯示 final_minutes（debug）", False):
                st.json(final_minutes)

            for t in topics:
                st.markdown(f"##### {t['title']}")
                if t["key_points"]:     st.markdown("\n".join(f"- {x}" for x in t["key_points"]))
                if t["decisions"]:      st.markdown("決策：\n" + "\n".join(f"- {x}" for x in t["decisions"]))
                if t["risks"]:          st.markdown("風險：\n" + "\n".join(f"- {x}" for x in t["risks"]))
                if t["open_questions"]: st.markdown("未決問題：\n" + "\n".join(f"- {x}" for x in t["open_questions"]))

        with tab4:
            st.markdown("#### 原始內容（最原始串流輸出，未分句／未去重）")
            st.code(all_text.strip(), language="text")

        status.update(label="全部完成！", state="complete", expanded=True)

# 清理暫存
try:
    os.remove(temp_input_path)
    if 'wav_path' in locals() and wav_path != temp_input_path:
        os.remove(wav_path)
except Exception:
    pass
