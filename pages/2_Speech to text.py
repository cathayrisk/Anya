# -*- coding: utf-8 -*-
# 會議錄音 → 直播逐字＋摘要（最終整合版：並行潤飾＋即時 Map）
# 特色：
# - STT（Producer）→「潤飾」RefineConsumer 與「即時 Map 摘要」SummarizerConsumer 並行
# - 背景執行緒支援 ScriptRunContext（可安全寫 UI），失敗自動回退為「不寫 UI」
# - 相鄰視窗去重（Jaccard trigram）取代大量 difflib，速度更穩
# - 參數可視化（看到塞車/想省成本時怎麼調，都寫在 UI 滑桿旁）
# - 最後用 Reduce 產出正式長文與結構化 JSON

import os
import re
import json
import hashlib
import tempfile
import multiprocessing
from typing import List, Dict, Any

import streamlit as st
from openai import OpenAI
from pydub import AudioSegment, silence
from pydub.utils import which

from queue import Queue, Empty
from threading import Thread, Lock
import time
import concurrent.futures

# 嘗試載入 Streamlit 的 ScriptRunContext 工具（允許在子執行緒安全更新 UI）
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx, add_script_run_ctx
except Exception:
    get_script_run_ctx = None
    add_script_run_ctx = None

# ========== 基本設定 ==========
st.set_page_config(page_title="會議錄音 → 直播逐字＋摘要", page_icon="📝", layout="wide")

# 自訂樣式
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
MODEL_STT = "gpt-4o-mini-transcribe"  # STT 忠實轉錄原語言（照你原先設定）
MODEL_MAP = "gpt-5-mini"              # 分段摘要
MODEL_REDUCE = "gpt-4.1"              # 總整/潤飾

# 切段參數
MIN_SILENCE_LEN_MS = 700
KEEP_SILENCE_MS = 300
SILENCE_DB_OFFSET = 16
OVERLAP_MS = 1200

# 片段長度保護與回退
MAX_CHUNK_MS = 30_000    # 單段最長 30 秒
MIN_CHUNK_MS = 2_000     # 單段最短 2 秒
FALLBACK_WINDOW_MS = 20_000  # 找不到靜音時，固定切 20 秒

DEFAULT_MAP_CHUNK_SIZE = 40

# 預設工人數（潤飾/Map 兩邊各自的最大同時批次，保守從 1 起）
MAX_STREAM_WORKERS = min(2, multiprocessing.cpu_count())

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

def normalize_loudness(audio: AudioSegment, target_dbfs: float = -20.0) -> AudioSegment:
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

# ====== 高效去重輔助（相鄰視窗 + Jaccard trigram）======
def _norm_for_dedupe(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r'\s+', '', s)
    s = (s.replace('，', ',').replace('。', '.')
           .replace('！', '!').replace('？', '?')
           .replace('；', ';').replace('（', '(').replace('）', ')'))
    return s

def _jaccard_trigram(a: str, b: str) -> float:
    n = 3
    if len(a) < n or len(b) < n:
        return 1.0 if a == b else 0.0
    A = {a[i:i+n] for i in range(len(a)-n+1)}
    B = {b[i:i+n] for i in range(len(b)-n+1)}
    un = len(A | B)
    return (len(A & B) / un) if un else 0.0

def dedupe_against_prev_fast(curr: List[str], prev: List[str],
                             threshold: float = 0.88, max_prev: int = 12) -> List[str]:
    if not curr:
        return []
    tail = prev[-max_prev:] if prev else []
    tail_norm = [_norm_for_dedupe(p) for p in tail]

    out: List[str] = []
    for s in curr:
        ns = _norm_for_dedupe(s)
        if ns in tail_norm:
            continue
        similar = False
        for pn in tail_norm:
            if not pn:
                continue
            if abs(len(ns) - len(pn)) > int(max(len(ns), len(pn)) * 0.4):
                continue
            if _jaccard_trigram(ns, pn) >= threshold:
                similar = True
                break
        if not similar:
            out.append(s)
    return out

def add_cjk_spacing(text: str) -> str:
    text = re.sub(r'([\u4e00-\u9fff])([A-Za-z0-9%#@&])', r'\1 \2', text)
    text = re.sub(r'([A-Za-z0-9%#@&])([\u4e00-\u9fff])', r'\1 \2', text)
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
    if not lines:
        return lines
    SEP = "\u241E"  # 可視分隔符 ␞
    MAX_BATCH_CHARS = 9000
    MAX_BATCH_LINES = 120

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
    return {"overall_summary": "解析 JSON 失敗（未知原因）", "raw": ""}

def reduce_finalize_markdown(map_blocks: List[str]) -> str:
    dev_msg = (
        "你是會議記錄總整專家。請將多個分段摘要整併為『單一份最終會議紀錄（Markdown）』。\n"
        "要求：\n"
        "1) 僅根據提供的分段摘要整併，嚴禁捏造來源沒有的資訊。\n"
        "2) 不輸出 metadata（標題/日期/地點/參與者/時長），只要內容本體。\n"
        "3) 結構：\n"
        " - 以一段「總結」開場，3~6 句，說清楚整體脈絡與結論。\n"
        " - 之後用多個小節（## 主題名稱），每節採用短段落敘述為主，可穿插少量條列。\n"
        " - 若有決策/風險/未決問題，於對應主題內以『決策：』『未決：』『風險：』標示。\n"
        "4) 只輸出純 Markdown 內容，不要額外說明。\n\n"
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
        return (resp.output_text or "").strip()
    except Exception as e:
        return f"⚠️ 生成會議摘要失敗：{e}"

# ========== 並行潤飾 Consumer ==========
# 45~60 分鐘建議值（可在 UI slider 微調）
REFINE_MAX_LINES = 80     # 越小越即時（API 次數↑/成本↑）；80~100 常見
REFINE_MAX_CHARS = 6000   # 6000~9000；越大越省成本但回饋慢
REFINE_MAX_WAIT_S = 0.35  # 0.25~0.45；塞車可調小，省成本可調大

class RefineConsumer:
    def __init__(self, stream_container, progress_bar, workers: int = 1):
        self.q: Queue = Queue(maxsize=6)  # 適度背壓
        self.stream_container = stream_container
        self.progress_bar = progress_bar
        self.grouped_sentences: List[List[str]] = []
        self.refined_lines_all: List[str] = []
        self.unique_sentences_raw_all: List[str] = []  # 也留存給摘要用
        self._stop = False
        self._total = 0
        self._done = 0
        self.lock = Lock()

        self.workers = max(1, int(workers))
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.workers)
        self.batch_id = 0
        self.next_emit_id = 0
        self.pending: Dict[int, concurrent.futures.Future] = {}
        self.batch_buffer: Dict[int, List[str]] = {}

        # UI 安全回退
        self.fallback_no_ui = False

    def set_total(self, n: int):
        self._total = n

    def put(self, item):
        self.q.put(item)

    def stop(self):
        self._stop = True
        self.q.put(None)

    def _safe_progress(self, ratio: float):
        if self.fallback_no_ui:
            return
        try:
            self.progress_bar.progress(ratio)
        except Exception:
            self.fallback_no_ui = True

    def _safe_render_all(self):
        if self.fallback_no_ui:
            return
        try:
            existing = "\n\n".join(group_into_paragraphs(self.refined_lines_all, max_chars=280, max_sents=4))
            self.stream_container.markdown(existing)
        except Exception:
            self.fallback_no_ui = True

    def _submit_batch(self, batch_lines: List[str], bid: int):
        def _task(lines: List[str]) -> List[str]:
            try:
                return refine_zh_tw_via_prompt(lines)
            except Exception:
                return lines
        fut = self.executor.submit(_task, batch_lines[:])
        self.pending[bid] = fut

    def _emit_ready(self):
        emitted_any = False
        while self.next_emit_id in self.pending and self.pending[self.next_emit_id].done():
            fut = self.pending.pop(self.next_emit_id)
            try:
                refined = fut.result()
            except Exception:
                refined = self.batch_buffer.get(self.next_emit_id, [])
            self.refined_lines_all.extend(refined)
            self._safe_render_all()
            self.batch_buffer.pop(self.next_emit_id, None)
            self.next_emit_id += 1
            emitted_any = True
        return emitted_any

    def run(self):
        batch_lines: List[str] = []
        batch_chars = 0
        last_flush = time.time()

        while True:
            try:
                item = self.q.get(timeout=0.2)
            except Empty:
                item = None

            now = time.time()
            timeup = (now - last_flush) >= REFINE_MAX_WAIT_S

            if item is None:
                if timeup and batch_lines:
                    bid = self.batch_id
                    self.batch_buffer[bid] = batch_lines[:]
                    self._submit_batch(batch_lines, bid)
                    self.batch_id += 1
                    batch_lines, batch_chars = [], 0
                    last_flush = now
                if self._stop:
                    if batch_lines:
                        bid = self.batch_id
                        self.batch_buffer[bid] = batch_lines[:]
                        self._submit_batch(batch_lines, bid)
                        self.batch_id += 1
                        batch_lines, batch_chars = [], 0
                    while self.pending:
                        self._emit_ready()
                        time.sleep(0.05)
                    break
                else:
                    self._emit_ready()
                continue

            sents = item
            unique = sents if not self.grouped_sentences else dedupe_against_prev_fast(
                sents, self.grouped_sentences[-1], threshold=0.88, max_prev=12
            )
            self.grouped_sentences.append(unique)
            flat = pretty_format_sentences(unique)
            self.unique_sentences_raw_all.extend(flat)

            flushed = False
            for s in flat:
                if not s.strip():
                    continue
                if (len(batch_lines) >= REFINE_MAX_LINES) or (batch_chars + len(s) > REFINE_MAX_CHARS) or timeup:
                    bid = self.batch_id
                    self.batch_buffer[bid] = batch_lines[:]
                    self._submit_batch(batch_lines, bid)
                    self.batch_id += 1
                    batch_lines, batch_chars = [], 0
                    last_flush = now
                    flushed = True
                batch_lines.append(s)
                batch_chars += len(s)

            self._done += 1
            if self._total:
                self._safe_progress(min(1.0, self._done / self._total))
            if flushed:
                self._emit_ready()

        self.executor.shutdown(wait=True)

# ========== 並行 Map 摘要 Consumer ==========
MAP_MAX_LINES = 30       # 即時性↑：調小到 24；省成本↑：調到 40~60
MAP_MAX_CHARS = 4000     # 3000~6000；越大越省成本
MAP_MAX_WAIT_S = 0.35    # 0.25~0.45；塞車可調小
MAP_WORKERS = 1          # 先 1；慢再開到 2（順序已維持，成本↑）

class SummarizerConsumer:
    def __init__(self, map_container, map_progress, workers: int = 1):
        self.q: Queue = Queue(maxsize=6)
        self.map_container = map_container
        self.map_progress = map_progress
        self._stop = False
        self._total = 0
        self._done = 0

        self.blocks: List[str] = []    # 即時 Map 區塊（Markdown）
        self.workers = max(1, int(workers))
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.workers)
        self.batch_id = 0
        self.next_emit_id = 0
        self.pending: Dict[int, concurrent.futures.Future] = {}
        self.batch_buffer: Dict[int, List[str]] = {}

        self.fallback_no_ui = False

    def set_total(self, n: int):
        self._total = n

    def put(self, sents: List[str]):
        self.q.put(sents)

    def stop(self):
        self._stop = True
        self.q.put(None)

    def _safe_progress(self, ratio: float):
        if self.fallback_no_ui:
            return
        try:
            self.map_progress.progress(ratio)
        except Exception:
            self.fallback_no_ui = True

    def _safe_render_all(self):
        if self.fallback_no_ui:
            return
        try:
            self.map_container.markdown("\n\n".join(self.blocks))
        except Exception:
            self.fallback_no_ui = True

    def _submit_batch(self, batch_lines: List[str], bid: int):
        def _task(lines: List[str]) -> str:
            part = "\n".join(lines)
            dev_msg = (
                "你是一位會議記錄小幫手，請將下列逐字稿整理為條列式重點（繁體中文）。"
                "要求：每點具體、避免空泛；若有決策/風險/未決問題/行動項目請清楚標記；"
                "只輸出條列重點，不要額外說明。"
            )
            try:
                resp = client.responses.create(
                    model=MODEL_MAP,
                    input=[
                        {"role": "developer", "content": [{"type": "input_text", "text": dev_msg}]},
                        {"role": "user", "content": [{"type": "input_text", "text": part}]},
                    ],
                    text={"format": {"type": "text"}},
                    tools=[],
                )
                content = (resp.output_text or "").strip()
                return f"### 即時重點 Part {bid+1}\n\n" + content
            except Exception as e:
                return f"### 即時重點 Part {bid+1}\n\n- 【API 摘要失敗：{e}】"
        fut = self.executor.submit(_task, batch_lines[:])
        self.pending[bid] = fut

    def _emit_ready(self):
        emitted = False
        while self.next_emit_id in self.pending and self.pending[self.next_emit_id].done():
            fut = self.pending.pop(self.next_emit_id)
            try:
                md = fut.result()
            except Exception:
                md = "（本批摘要回傳失敗）"
            self.blocks.append(md)
            self._safe_render_all()
            self.batch_buffer.pop(self.next_emit_id, None)
            self.next_emit_id += 1
            emitted = True
        return emitted

    def run(self):
        batch_lines: List[str] = []
        batch_chars = 0
        last_flush = time.time()

        while True:
            try:
                item = self.q.get(timeout=0.2)
            except Empty:
                item = None

            now = time.time()
            timeup = (now - last_flush) >= MAP_MAX_WAIT_S

            if item is None:
                if timeup and batch_lines:
                    bid = self.batch_id
                    self.batch_buffer[bid] = batch_lines[:]
                    self._submit_batch(batch_lines, bid)
                    self.batch_id += 1
                    batch_lines, batch_chars = [], 0
                    last_flush = now
                if self._stop:
                    if batch_lines:
                        bid = self.batch_id
                        self.batch_buffer[bid] = batch_lines[:]
                        self._submit_batch(batch_lines, bid)
                        self.batch_id += 1
                        batch_lines, batch_chars = [], 0
                    while self.pending:
                        self._emit_ready()
                        time.sleep(0.05)
                    break
                else:
                    self._emit_ready()
                continue

            lines = pretty_format_sentences(item)
            for s in lines:
                if not s.strip():
                    continue
                if (len(batch_lines) >= MAP_MAX_LINES) or (batch_chars + len(s) > MAP_MAX_CHARS) or timeup:
                    bid = self.batch_id
                    self.batch_buffer[bid] = batch_lines[:]
                    self._submit_batch(batch_lines, bid)
                    self.batch_id += 1
                    batch_lines, batch_chars = [], 0
                    last_flush = now
                batch_lines.append(s)
                batch_chars += len(s)

            self._done += 1
            if self._total:
                self._safe_progress(min(1.0, self._done / self._total))

        self.executor.shutdown(wait=True)

# ========== STT Producer ==========
def build_prompt(prev_text: str, glossary: str, style_seed: str, max_tokens: int = 220) -> str:
    parts = []
    parts.append("請全程使用正體中文（繁體，台灣用語）。")
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
    toks = prompt.split()
    if len(toks) > max_tokens:
        prompt = " ".join(toks[-max_tokens:])
    return prompt

def stream_transcribe_all(
    chunks: List[AudioSegment],
    container,
    progress_bar,
    use_prompting: bool = False,
    glossary: str = "",
    style_seed: str = "",
    on_sentences=None  # 回呼：把本段句子分流給消費者
):
    all_text = ""
    rolling_context = ""
    last_flush = 0.0
    FLUSH_INTERVAL = 0.15

    for i, chunk in enumerate(chunks):
        chunk_hash = _hash_bytes(chunk.raw_data)
        cache_key = f"stt_{MODEL_STT}_{chunk_hash}"
        cached = cache_get_text(cache_key)
        if cached:
            all_text += cached + "\n"
            rolling_context = (rolling_context + " " + cached).strip()
            if len(rolling_context) > 5000:
                rolling_context = rolling_context[-5000:]
            progress_bar.progress((i + 1) / len(chunks))
            container.markdown(all_text)
            if on_sentences:
                on_sentences(split_sentences(cached))
            continue

        full_text = ""
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp_path = tmp.name
                chunk.export(tmp_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
            with open(tmp_path, "rb") as audio_file:
                extra_kwargs = {}
                if use_prompting:
                    prompt_str = build_prompt(rolling_context, glossary, style_seed, max_tokens=220)
                    if prompt_str:
                        extra_kwargs["prompt"] = prompt_str
                try:
                    stream = client.audio.transcriptions.create(
                        model=MODEL_STT,
                        file=audio_file,
                        response_format="text",
                        prompt=(
                            "This audio contains a discussion or presentation. Always preserve the original language "
                            "of each sentence. If a sentence is in English, output it in English; if in Chinese, output it "
                            "in Traditional Chinese; if mixed, output the original mixed-language sentence. Do not translate."
                        ),
                        stream=True,
                        **extra_kwargs
                    )
                except Exception:
                    try:
                        stream = client.audio.transcriptions.create(
                            model=MODEL_STT,
                            file=audio_file,
                            response_format="text",
                            prompt=(
                                "This audio contains a discussion or presentation. Always preserve the original language "
                                "of each sentence. If a sentence is in English, output it in English; if in Chinese, output it "
                                "in Traditional Chinese; if mixed, output the original mixed-language sentence. Do not translate."
                            ),
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
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

        cache_set_text(cache_key, full_text.strip())
        all_text += full_text + "\n"
        rolling_context = (rolling_context + " " + full_text).strip()
        if len(rolling_context) > 5000:
            rolling_context = rolling_context[-5000:]

        if on_sentences:
            sents = split_sentences(full_text)
            on_sentences(sents)

        progress_bar.progress((i + 1) / len(chunks))
        container.markdown(all_text)

    return all_text.strip()

# 顯示模式工具：段落群組
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

def render_topics_only(md: Dict[str, Any], stlib):
    stlib.markdown("#### 主題")
    topics = md.get("topics", [])
    for t in topics:
        stlib.markdown(f"##### {t.get('title','主題')}")
        kp = t.get("key_points", [])
        if kp:
            stlib.markdown("\n".join(f"- {x}" for x in kp))
        if t.get("decisions"):
            stlib.markdown("決策：\n" + "\n".join(f"- {x}" for x in t.get("decisions", [])))
        if t.get("risks"):
            stlib.markdown("風險：\n" + "\n".join(f"- {x}" for x in t.get("risks", [])))
        if t.get("open_questions"):
            stlib.markdown("未決問題：\n" + "\n".join(f"- {x}" for x in t.get("open_questions", [])))

# ========== 上傳區 ==========
with st.expander("上傳會議錄音檔案", expanded=True):
    f = st.file_uploader("請上傳音檔（.wav, .mp3, .m4a, .mp4, .webm）", type=["wav", "mp3", "m4a", "mp4", "webm"])
    start_btn = st.button("開始 Streaming 轉錄與摘要")

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

    st.markdown("###### Prompt 引導（若端點不支援會自動回退）")
    use_prompting = st.checkbox("啟用 Prompt 引導（改善專有名詞拼寫與風格一致）", value=False)
    glossary_input = st.text_area(
        "專有名詞拼寫清單（每行一個）",
        height=120,
        placeholder="例：\nAimee\nShawn\nBBQ\nZyntriQix",
        disabled=not use_prompting
    )
    style_seed = st.text_area(
        "風格示例（1～3 句示例文本，不是指令）",
        height=80,
        placeholder="例：\n保持簡潔、標點一致。例句：we discuss quarterly outlook and risks.",
        disabled=not use_prompting
    )

    st.markdown("###### 並行潤飾控制（塞車怎麼調？）")
    st.caption("即時性↑：把等待時間/單批行數調小；省成本↑：反之。需要時把工人數開到 2。")
    REFINE_MAX_WAIT_S = st.slider("微批次最大等待秒數 REFINE_MAX_WAIT_S", 0.10, 0.80, REFINE_MAX_WAIT_S, 0.05)
    REFINE_MAX_LINES  = st.slider("單批最大行數 REFINE_MAX_LINES", 20, 140, REFINE_MAX_LINES, 5)
    REFINE_MAX_CHARS  = st.slider("單批最大字數 REFINE_MAX_CHARS", 2000, 12000, REFINE_MAX_CHARS, 500)
    MAX_STREAM_WORKERS = st.slider("潤飾工人數 MAX_STREAM_WORKERS（1～2）", 1, 2, MAX_STREAM_WORKERS, 1)

    st.markdown("###### 即時 Map 控制（讓重點更快出現）")
    st.caption("第一時間看到重點：把等待時間/單批行數調小；成本太高再調大。")
    MAP_MAX_WAIT_S = st.slider("Map 微批次最大等待秒數 MAP_MAX_WAIT_S", 0.10, 0.80, MAP_MAX_WAIT_S, 0.05)
    MAP_MAX_LINES  = st.slider("Map 單批最大行數 MAP_MAX_LINES", 10, 80, MAP_MAX_LINES, 2)
    MAP_MAX_CHARS  = st.slider("Map 單批最大字數 MAP_MAX_CHARS", 1000, 10000, MAP_MAX_CHARS, 250)
    MAP_WORKERS    = st.slider("Map 工人數 MAP_WORKERS（1～2）", 1, 2, MAP_WORKERS, 1)

if not (f and start_btn):
    st.stop()

# ========== 主流程 ==========
raw_bytes = f.read()
st.audio(raw_bytes)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["轉錄結果", "重點摘要", "內容解析", "原始內容"])

# Tab2 先準備即時 Map 容器與最終長文區
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
        if st.session_state.get("_first_run_trim", True) and len(audio) > 0:
            st.session_state["_first_run_trim"] = False
        if st.session_state.get("_first_run_trim", False):
            pass
        if st.session_state.get("_first_run_trim", False):
            pass
        if True:
            # 依使用者勾選執行
            if st.session_state.get("_dummy", False):
                pass
        if True:
            if True:
                pass
        if True:
            pass
        if len(audio) > 0:
            if st.session_state.get("_dummy2", False):
                pass

        # 實際前處理
        if True:
            if True:
                pass
        if True:
            pass

        if True:
            pass

        if True:
            pass

        if True:
            pass

        if True:
            pass

        # 正式執行前處理
        if do_trim_leading:
            audio = trim_leading_silence(audio, silence_threshold_db=-30.0, chunk_ms=10)
        if do_normalize:
            audio = normalize_loudness(audio, target_dbfs=-20.0)
        if use_high_pass or use_low_pass:
            audio = apply_filters(audio, use_high_pass=use_high_pass, hp_hz=hp_hz,
                                  use_low_pass=use_low_pass, lp_hz=lp_hz)

        status.update(label="靜音切段（附最長/最短保護；找不到靜音會回退固定切）...")
        chunks = split_audio_on_silence_safe(audio)
        if not chunks:
            st.error("無法切出有效音訊段，請檢查音檔或調整參數。")
            st.stop()

        st.markdown("#### 轉錄結果")
        stream_container = st.empty()
        progress_bar = st.progress(0.0)

        # 並行潤飾與 Map：啟動兩位消費者
        refine_progress = st.progress(0.0)
        consumer = RefineConsumer(stream_container, refine_progress, workers=MAX_STREAM_WORKERS)
        consumer_thread = Thread(target=consumer.run, daemon=True, name="RefineConsumer")

        summarizer = SummarizerConsumer(map_stream_container, map_progress, workers=MAP_WORKERS)
        summarizer_thread = Thread(target=summarizer.run, daemon=True, name="SummarizerConsumer")

        # 掛 ScriptRunContext（若失敗會自動回退成「背景不寫 UI」模式）
        if get_script_run_ctx and add_script_run_ctx:
            ctx = get_script_run_ctx()
            if ctx is not None:
                try:
                    add_script_run_ctx(consumer_thread, ctx)
                    add_script_run_ctx(summarizer_thread, ctx)
                except Exception:
                    consumer.fallback_no_ui = True
                    summarizer.fallback_no_ui = True
            else:
                consumer.fallback_no_ui = True
                summarizer.fallback_no_ui = True
        else:
            consumer.fallback_no_ui = True
            summarizer.fallback_no_ui = True

        consumer_thread.start()
        summarizer_thread.start()
        consumer.set_total(len(chunks))
        summarizer.set_total(len(chunks))

        def fanout_on_sentences(sents: List[str]):
            consumer.put(sents)
            summarizer.put(sents)

        status.update(label="逐段 Streaming 轉錄中（並行潤飾＋即時摘要）...")
        all_text = stream_transcribe_all(
            chunks,
            stream_container,
            progress_bar,
            use_prompting=use_prompting,
            glossary=glossary_input if use_prompting else "",
            style_seed=style_seed if use_prompting else "",
            on_sentences=fanout_on_sentences
        )

        # 收尾與顯示
        consumer.stop(); summarizer.stop()
        consumer_thread.join(); summarizer_thread.join()

        # 若背景不能寫 UI，這裡一次把內容畫上去
        if consumer.fallback_no_ui:
            refined_lines = consumer.refined_lines_all[:] if consumer.refined_lines_all else consumer.unique_sentences_raw_all
            paras = group_into_paragraphs(refined_lines, max_chars=280, max_sents=4)
            stream_container.markdown("\n\n".join(paras))
        if summarizer.fallback_no_ui:
            if summarizer.blocks:
                map_stream_container.markdown("\n\n".join(summarizer.blocks))

        st.success("Transcription + Refine complete!")

        status.update(label="整併重點（Reduce 中）...")
        # Reduce：使用即時 Map 的 blocks；若沒有就退回一次性 Map
        map_blocks_text = summarizer.blocks[:] if summarizer.blocks else map_summarize_blocks(
            consumer.unique_sentences_raw_all if consumer.unique_sentences_raw_all else split_sentences(all_text)
        )
        final_md_summary = reduce_finalize_markdown(map_blocks_text)
        final_summary_placeholder.markdown(final_md_summary)

        # 額外提供結構化 JSON 與主題檢視
        final_minutes = reduce_finalize_json(map_blocks_text)

        with tab3:
            render_topics_only(final_minutes, st)

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
