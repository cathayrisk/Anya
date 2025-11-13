# -*- coding: utf-8 -*-
# 會議錄音 → 直播逐字＋摘要（無 OpenCC 版：refine 一律輸出正體中文-台灣用語）
# 變更重點：
# - 不使用 OpenCC。改以「兩段式 LLM 保險」：先 refine，再強制繁體化（保持行數與分隔符）。
# - 強化 system/dev 提示：嚴禁簡體字，保留數字/網址/emoji/程式片段。
# - 行數一致性檢查與失敗回退，避免走位。
# - 保留四分頁：轉錄結果／重點摘要／內容解析／原始內容。
# - 未加入任何「即時重點（Map streaming）」UI 或流程。

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
MODEL_STT = "gpt-4o-mini-transcribe"  # STT 忠實轉錄原語言
MODEL_MAP = "gpt-5-mini"              # 分段摘要
MODEL_REDUCE = "gpt-4.1"              # 總整/潤飾（refine 與強制繁體化也用這顆）

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

# ========== 顯示層：逐行『潤飾＋必要時翻譯』為正體（台灣用語） ==========
def refine_zh_tw_via_prompt(lines: List[str]) -> List[str]:
    """
    將多行句子逐行『潤飾＋必要時翻譯』為正體中文（台灣用語）。
    - 批次處理＋分隔符防走位；單批失敗只回退該批，不影響其他批。
    - 二階段 LLM 保險：第一階段潤飾，第二階段強制繁體化（維持行數與分隔符 ␞）。
    """
    if not lines:
        return lines

    SEP = "\u241E"  # ␞ 極少見的可視分隔符
    MAX_BATCH_CHARS = 9000  # 單批最大字數（保守）
    MAX_BATCH_LINES = 120   # 單批最多行數（保守）

    sys_rule_stage1 = (
        "請務必以『正體中文（台灣用語）』輸出每一行；嚴禁出現任何簡體字或大陸用語。"
        "若輸入含英文或混語，請翻譯為正體中文；保留數字、單位、網址、emoji、程式片段。"
        "不可合併、刪除或新增行；輸入幾行就輸出幾行，並以 ␞ 分隔。"
        "常見用字：資訊/風險/預算/里程碑/環評/滯洪/抽水站（避免：信息/风险/预算/里程碑(簡字)/环评/滞洪(簡字)/抽水站(簡字)）。"
    )

    sys_rule_stage2 = (
        "將使用者提供的文本逐行『轉換為正體中文（台灣用語）字形與用詞』。"
        "嚴格規則：\n"
        "1) 只變換字形與用詞為台灣慣用，不得改動語意、數字、標點、網址、emoji。\n"
        "2) 不得增刪或合併任何一行；輸入幾行輸出幾行；以 ␞ 作為行分隔，務必保留相同數量的分隔符。\n"
        "3) 嚴禁輸出簡體字或大陸地區用語。"
    )

    def _stage1_refine(batch: List[str]) -> List[str]:
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
                    {"role": "system", "content": [{"type": "input_text", "text": sys_rule_stage1}]},
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

    def _stage2_force_tw(batch: List[str]) -> List[str]:
        """
        第二階：僅將字形/用語統一為台灣正體；不動行數、不動結構。
        """
        blob = SEP.join(batch)
        try:
            resp = client.responses.create(
                model=MODEL_REDUCE,
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": sys_rule_stage2}]},
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

    # 分批處理（兩階段）
    refined_all: List[str] = []
    batch: List[str] = []
    size = 0
    for s in lines:
        if (len(batch) >= MAX_BATCH_LINES) or (size + len(s) + 1 > MAX_BATCH_CHARS):
            first = _stage1_refine(batch)
            second = _stage2_force_tw(first)
            refined_all.extend(second)
            batch, size = [], 0
        batch.append(s)
        size += len(s) + 1
    if batch:
        first = _stage1_refine(batch)
        second = _stage2_force_tw(first)
        refined_all.extend(second)

    return refined_all if refined_all else lines

# Prompt（若端點支援就用、不支援自動回退）
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
    style_seed: str = ""
):
    import time
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
                            "This audio contains a discussion or presentation. "
                            "Always preserve the original language of each sentence. "
                            "If a sentence is in English, output it in English; "
                            "if in Chinese, output it in Traditional Chinese; "
                            "if mixed, output the original mixed-language sentence. "
                            "Do not translate or alter the language. "
                            "The audio may cover various topics such as updates, feedback, or informative lectures."
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
                                "This audio contains a discussion or presentation. "
                                "Always preserve the original language of each sentence. "
                                "If a sentence is in English, output it in English; "
                                "if in Chinese, output it in Traditional Chinese; "
                                "if mixed, output the original mixed-language sentence. "
                                "Do not translate or alter the language. "
                                "The audio may cover various topics such as updates, feedback, or informative lectures."
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

        progress_bar.progress((i + 1) / len(chunks))
        container.markdown(all_text)

    return all_text.strip()

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
        "你是會議記錄總整專家。請將多個分段摘要整併為『單一份最終會議記錄（Markdown）』。\n"
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
    start_btn = st.button("開始 Streaming 轉錄與摘要")

# ========== 單一整體收合的進階調整 ==========
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

if not (f and start_btn):
    st.stop()

# ========== 主流程 ==========
raw_bytes = f.read()
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

        status.update(label="靜音切段（附最長/最短保護；找不到靜音會回退固定切）...")
        chunks = split_audio_on_silence_safe(audio)
        if not chunks:
            st.error("無法切出有效音訊段，請檢查音檔或調整參數。")
            st.stop()

        st.markdown("#### 轉錄結果")
        stream_container = st.empty()
        progress_bar = st.progress(0.0)

        status.update(label="逐段 Streaming 轉錄中...")
        all_text = stream_transcribe_all(
            chunks,
            stream_container,
            progress_bar,
            use_prompting=use_prompting,
            glossary=glossary_input if use_prompting else "",
            style_seed=style_seed if use_prompting else ""
        )
        raw_stream_text = all_text.strip()

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

        # 可讀版：輕量整理 → 潤飾/翻譯為正體（台灣）→ 段落化 → Markdown 呈現（直接覆蓋直播容器，避免空窗）
        pretty_lines = pretty_format_sentences(flat_sentences)
        refined_lines = refine_zh_tw_via_prompt(pretty_lines)

        paras = group_into_paragraphs(refined_lines, max_chars=280, max_sents=4)
        final_md = "\n\n".join(paras)
        stream_container.markdown(final_md)

        st.success("Transcription complete!")

        status.update(label="整併重點（內部計算）...")
        map_blocks_text = map_summarize_blocks(flat_sentences)

        status.update(label="生成最終會議摘要與內容解析...")
        final_minutes = reduce_finalize_json(map_blocks_text)
        final_md_summary = reduce_finalize_markdown(map_blocks_text)

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
            st.markdown("#### 原始內容（最原始串流輸出，未分句／未去重）")
            st.code(raw_stream_text, language="text")

        status.update(label="全部完成！", state="complete", expanded=True)

# 清理暫存
try:
    os.remove(temp_input_path)
    if 'wav_path' in locals() and wav_path != temp_input_path:
        os.remove(wav_path)
except Exception:
    pass
