import asyncio
import base64
import hashlib
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import aiohttp
import streamlit as st
from pydub import AudioSegment
from pydub.utils import which


st.set_page_config(
    page_title="語音轉文字",
    page_icon="📝",
    layout="wide",
)

st.markdown(
    """
<style>
:root {
  --brand:#9c2b2f;
  --brand-soft:#fff6f6;
  --border:#f2d9d9;
}
.main .block-container{padding-top:2.2rem}
.anya-card{
  background:var(--brand-soft);
  border:1px solid var(--border);
  padding:16px 22px;
  border-radius:12px;
  margin-bottom:12px;
}
.anya-title{
  font-size:22px;
  font-weight:700;
  color:#2f2f2f;
  line-height:1.35;
}
.transcript-readable{
  font-size:1.02rem;
  line-height:1.9;
  letter-spacing:0.02em;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="anya-card anya-title">📝 語音轉文字：Realtime Whisper 逐字稿</div>',
    unsafe_allow_html=True,
)


OPENAI_KEY = st.secrets.get("OPENAI_KEY", os.getenv("OPENAI_API_KEY"))
REALTIME_STT_MODEL = "gpt-realtime-whisper"
REALTIME_URL = "wss://api.openai.com/v1/realtime?intent=transcription"
TARGET_SAMPLE_RATE = 24_000
PCM_BYTES_PER_SAMPLE = 2
CACHE_DIR = ".stt_cache"

os.makedirs(CACHE_DIR, exist_ok=True)


AudioSegment.converter = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")
if not AudioSegment.converter or not AudioSegment.ffprobe:
    st.error("找不到 ffmpeg / ffprobe，請先安裝後再試。")
    st.stop()

if not OPENAI_KEY:
    st.error("找不到 API Key，請在 Streamlit Secrets 設定 OPENAI_KEY，或設定環境變數 OPENAI_API_KEY。")
    st.stop()


def hash_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def cache_get_text(key: str) -> str | None:
    path = os.path.join(CACHE_DIR, f"{key}.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def cache_set_text(key: str, value: str) -> None:
    path = os.path.join(CACHE_DIR, f"{key}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(value)


def normalize_loudness(audio: AudioSegment, target_dbfs: float = -20.0) -> AudioSegment:
    if audio.dBFS == float("-inf"):
        return audio
    return audio.apply_gain(target_dbfs - audio.dBFS)


def trim_leading_silence(
    audio: AudioSegment,
    silence_threshold_db: float = -30.0,
    chunk_ms: int = 10,
) -> AudioSegment:
    trim_ms = 0
    while trim_ms < len(audio) and audio[trim_ms : trim_ms + chunk_ms].dBFS < silence_threshold_db:
        trim_ms += chunk_ms
    return audio[trim_ms:]


def apply_filters(
    audio: AudioSegment,
    use_high_pass: bool = False,
    hp_hz: int = 100,
    use_low_pass: bool = False,
    lp_hz: int = 9500,
) -> AudioSegment:
    out = audio
    if use_high_pass:
        out = out.high_pass_filter(hp_hz)
    if use_low_pass:
        out = out.low_pass_filter(lp_hz)
    return out


def load_audio_from_upload(uploaded_file) -> AudioSegment:
    suffix = os.path.splitext(uploaded_file.name)[1] or ".audio"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_input:
        temp_input.write(uploaded_file.getvalue())
        temp_input_path = temp_input.name

    try:
        return AudioSegment.from_file(temp_input_path)
    finally:
        try:
            os.remove(temp_input_path)
        except OSError:
            pass


def audio_to_pcm16(audio: AudioSegment) -> bytes:
    pcm_audio = (
        audio.set_channels(1)
        .set_frame_rate(TARGET_SAMPLE_RATE)
        .set_sample_width(PCM_BYTES_PER_SAMPLE)
    )
    return pcm_audio.raw_data


def chunk_pcm(pcm: bytes, chunk_ms: int) -> list[bytes]:
    bytes_per_ms = TARGET_SAMPLE_RATE * PCM_BYTES_PER_SAMPLE // 1000
    chunk_size = max(bytes_per_ms * chunk_ms, bytes_per_ms * 100)
    chunk_size -= chunk_size % PCM_BYTES_PER_SAMPLE
    return [pcm[i : i + chunk_size] for i in range(0, len(pcm), chunk_size)]


def update_live_text(container, final_parts: list[str], partial_parts: dict[str, str]) -> None:
    final_text = "\n\n".join(part.strip() for part in final_parts if part.strip())
    partial_text = "".join(partial_parts.values()).strip()
    combined = final_text
    if partial_text:
        combined = f"{combined}\n\n{partial_text}" if combined else partial_text
    container.markdown(
        f'<div class="transcript-readable">{combined or "等待轉錄結果..."}</div>',
        unsafe_allow_html=True,
    )


async def realtime_transcribe_pcm16(
    pcm: bytes,
    *,
    language_hint: str | None,
    chunk_ms: int,
    vad_silence_ms: int,
    throttle_to_audio_time: bool,
    transcript_container,
    progress_bar,
) -> str:
    headers = {
        "Authorization": f"Bearer {OPENAI_KEY}",
    }
    session_update: dict[str, Any] = {
        "type": "session.update",
        "session": {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": TARGET_SAMPLE_RATE,
                    },
                    "transcription": {
                        "model": REALTIME_STT_MODEL,
                    },
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": vad_silence_ms,
                    },
                }
            },
        },
    }
    if language_hint:
        session_update["session"]["audio"]["input"]["transcription"]["language"] = language_hint

    chunks = chunk_pcm(pcm, chunk_ms)
    final_by_item: dict[str, str] = {}
    item_order: list[str] = []
    partial_by_item: dict[str, str] = {}
    sent_all_audio = False
    last_event_at = time.monotonic()

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(
            REALTIME_URL,
            headers=headers,
            heartbeat=20,
            max_msg_size=10 * 1024 * 1024,
        ) as ws:
            await ws.send_str(json.dumps(session_update))

            async def send_audio() -> None:
                nonlocal sent_all_audio
                for i, chunk in enumerate(chunks):
                    await ws.send_str(
                        json.dumps(
                            {
                                "type": "input_audio_buffer.append",
                                "audio": base64.b64encode(chunk).decode("ascii"),
                            }
                        )
                    )
                    progress_bar.progress(min((i + 1) / max(len(chunks), 1), 1.0))
                    if throttle_to_audio_time:
                        await asyncio.sleep(chunk_ms / 1000)

                await asyncio.sleep(max(vad_silence_ms / 1000, 0.5))
                try:
                    await ws.send_str(json.dumps({"type": "input_audio_buffer.commit"}))
                except Exception:
                    pass
                sent_all_audio = True

            sender = asyncio.create_task(send_audio())

            try:
                while True:
                    if sent_all_audio and time.monotonic() - last_event_at > 4:
                        break

                    try:
                        msg = await asyncio.wait_for(ws.receive(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue

                    if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
                        break
                    if msg.type == aiohttp.WSMsgType.ERROR:
                        raise RuntimeError(f"Realtime WebSocket 錯誤：{ws.exception()}")
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue

                    last_event_at = time.monotonic()
                    event = json.loads(msg.data)
                    event_type = event.get("type")

                    if event_type == "error":
                        message = event.get("error", {}).get("message") or json.dumps(event, ensure_ascii=False)
                        raise RuntimeError(f"Realtime API 回傳錯誤：{message}")

                    if event_type == "conversation.item.input_audio_transcription.delta":
                        item_id = event.get("item_id", "current")
                        if item_id not in item_order:
                            item_order.append(item_id)
                        partial_by_item[item_id] = partial_by_item.get(item_id, "") + event.get("delta", "")
                        update_live_text(
                            transcript_container,
                            [final_by_item[item_id] for item_id in item_order if item_id in final_by_item],
                            {item_id: partial_by_item[item_id] for item_id in item_order if item_id in partial_by_item},
                        )

                    if event_type == "conversation.item.input_audio_transcription.completed":
                        item_id = event.get("item_id", f"item_{len(item_order)}")
                        if item_id not in item_order:
                            item_order.append(item_id)
                        transcript = (event.get("transcript") or partial_by_item.get(item_id, "")).strip()
                        if transcript:
                            final_by_item[item_id] = transcript
                        partial_by_item.pop(item_id, None)
                        update_live_text(
                            transcript_container,
                            [final_by_item[item_id] for item_id in item_order if item_id in final_by_item],
                            {item_id: partial_by_item[item_id] for item_id in item_order if item_id in partial_by_item},
                        )
            finally:
                await sender
                await ws.close()

    return "\n\n".join(final_by_item[item_id].strip() for item_id in item_order if final_by_item.get(item_id, "").strip())


def run_realtime_transcription(*args, **kwargs) -> str:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(realtime_transcribe_pcm16(*args, **kwargs))

    # Streamlit normally runs page code synchronously, but some deployments have
    # an active event loop in the main thread. Run the websocket client elsewhere.
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(lambda: asyncio.run(realtime_transcribe_pcm16(*args, **kwargs)))
        return future.result()


def build_download_markdown(transcript: str, filename: str) -> str:
    return f"# 語音逐字稿\n\n來源檔案：{filename}\n\n---\n\n{transcript.strip()}\n"


with st.expander("上傳音檔", expanded=True):
    uploaded = st.file_uploader(
        "請上傳音檔（wav, mp3, m4a, mp4, webm）",
        type=["wav", "mp3", "m4a", "mp4", "webm"],
    )

    cols = st.columns(3)
    with cols[0]:
        language_hint = st.selectbox(
            "語言提示",
            [
                ("自動偵測", ""),
                ("中文", "zh"),
                ("英文", "en"),
                ("日文", "ja"),
                ("韓文", "ko"),
                ("法文", "fr"),
                ("德文", "de"),
                ("西班牙文", "es"),
            ],
            format_func=lambda item: item[0],
        )[1]
    with cols[1]:
        chunk_ms = st.slider("送出音訊區塊（毫秒）", 100, 1000, 250, 50)
    with cols[2]:
        vad_silence_ms = st.slider("句尾靜音判定（毫秒）", 300, 2000, 700, 100)

    with st.expander("音訊前處理", expanded=False):
        preprocess_cols = st.columns(2)
        with preprocess_cols[0]:
            do_trim_leading = st.checkbox("去除前導靜音", value=True)
            do_normalize = st.checkbox("音量正規化到 -20 dBFS", value=True)
        with preprocess_cols[1]:
            use_high_pass = st.checkbox("高通濾波", value=False)
            hp_hz = st.slider("高通截止頻率 (Hz)", 60, 300, 100, 10, disabled=not use_high_pass)
            use_low_pass = st.checkbox("低通濾波", value=False)
            lp_hz = st.slider("低通截止頻率 (Hz)", 4000, 12000, 9500, 100, disabled=not use_low_pass)

    throttle_to_audio_time = st.checkbox(
        "依實際音訊速度送出（較慢，但更接近直播情境）",
        value=False,
    )
    use_cache = st.checkbox("啟用逐字稿快取", value=True)
    start_btn = st.button("開始轉文字", type="primary", disabled=uploaded is None)


if not uploaded or not start_btn:
    st.info("上傳音檔後按「開始轉文字」，會使用 gpt-realtime-whisper 產生即時逐字稿。")
    st.stop()


raw_bytes = uploaded.getvalue()
st.audio(raw_bytes)

transcript_tab, download_tab, debug_tab = st.tabs(["逐字稿", "下載", "處理資訊"])

with transcript_tab:
    status_box = st.status("準備音訊...", expanded=True)
    transcript_container = st.empty()
    progress_bar = st.progress(0.0)

    cache_key = f"realtime_whisper_{hash_bytes(raw_bytes)}_{language_hint or 'auto'}"
    cached_text = cache_get_text(cache_key) if use_cache else None

    if cached_text:
        transcript = cached_text
        transcript_container.markdown(
            f'<div class="transcript-readable">{transcript}</div>',
            unsafe_allow_html=True,
        )
        progress_bar.progress(1.0)
        status_box.update(label="已從快取載入逐字稿。", state="complete", expanded=False)
    else:
        try:
            status_box.update(label="載入與前處理音訊...")
            audio = load_audio_from_upload(uploaded)
            if do_trim_leading:
                audio = trim_leading_silence(audio)
            if do_normalize:
                audio = normalize_loudness(audio)
            if use_high_pass or use_low_pass:
                audio = apply_filters(
                    audio,
                    use_high_pass=use_high_pass,
                    hp_hz=hp_hz,
                    use_low_pass=use_low_pass,
                    lp_hz=lp_hz,
                )

            status_box.update(label="轉換為 24 kHz mono PCM16...")
            pcm = audio_to_pcm16(audio)

            status_box.update(label="連線到 Realtime transcription 並開始轉文字...")
            transcript = run_realtime_transcription(
                pcm,
                language_hint=language_hint or None,
                chunk_ms=chunk_ms,
                vad_silence_ms=vad_silence_ms,
                throttle_to_audio_time=throttle_to_audio_time,
                transcript_container=transcript_container,
                progress_bar=progress_bar,
            ).strip()

            if not transcript:
                st.warning("沒有收到逐字稿。請確認音檔是否有可辨識語音，或把「句尾靜音判定」調高後重試。")
            elif use_cache:
                cache_set_text(cache_key, transcript)

            status_box.update(label="轉文字完成。", state="complete", expanded=False)
        except Exception as exc:
            status_box.update(label="轉文字失敗。", state="error", expanded=True)
            st.error(str(exc))
            st.stop()


with download_tab:
    st.download_button(
        "下載純文字逐字稿",
        data=transcript.strip(),
        file_name="transcript.txt",
        mime="text/plain",
    )
    st.download_button(
        "下載 Markdown 逐字稿",
        data=build_download_markdown(transcript, uploaded.name),
        file_name="transcript.md",
        mime="text/markdown",
    )


with debug_tab:
    st.write(
        {
            "model": REALTIME_STT_MODEL,
            "sample_rate": TARGET_SAMPLE_RATE,
            "language_hint": language_hint or "auto",
            "chunk_ms": chunk_ms,
            "vad_silence_ms": vad_silence_ms,
            "throttle_to_audio_time": throttle_to_audio_time,
            "cache_enabled": use_cache,
        }
    )
