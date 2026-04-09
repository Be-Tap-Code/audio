import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import faster_whisper
import numpy as np
import torch

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from diarization import SortformerDiarizer
from helpers import (
    find_numeral_symbol_tokens,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_words_speaker_mapping,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("realtime-diarization")

# ============================================================
# Configuration
# ============================================================
SAMPLE_RATE = 16000
# Process every 3 seconds of audio (balance between latency and accuracy)
CHUNK_SECONDS = 3.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SECONDS)
# Minimum audio needed before first processing
MIN_INITIAL_SAMPLES = int(SAMPLE_RATE * 2.0)  # 2 seconds

# Maximum audio history to keep for Sortformer context (10 seconds)
MAX_CONTEXT_SAMPLES = SAMPLE_RATE * 10

# VAD threshold (RMS energy)
VAD_THRESHOLD = 0.01

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_MODEL_SIZE = "medium"
DIARIZER_NAME = "sortformer"
DEFAULT_LANGUAGE = "vi"

app = FastAPI(title="Realtime Whisper Diarization")
app.mount("/static", StaticFiles(directory="web"), name="static")


# ============================================================
# Model loading (global, singleton)
# ============================================================
def _choose_compute_type() -> str:
    if DEVICE == "cuda":
        return "float16"
    return "int8"


logger.info("Loading Whisper model (%s) on %s", WHISPER_MODEL_SIZE, DEVICE)
whisper_model = faster_whisper.WhisperModel(
    WHISPER_MODEL_SIZE,
    device=DEVICE,
    compute_type=_choose_compute_type(),
)
suppress_tokens = find_numeral_symbol_tokens(whisper_model.hf_tokenizer)

# Use batched pipeline for faster inference
whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)

logger.info("Loading %s diarizer on %s", DIARIZER_NAME, DEVICE)
diarizer = SortformerDiarizer(device=DEVICE)


# ============================================================
# Session state
# ============================================================
@dataclass
class SessionState:
    language: str = DEFAULT_LANGUAGE
    # Raw PCM16 bytes accumulated from client
    audio_buffer: bytearray = field(default_factory=bytearray)
    # Total samples processed (for offset calculation)
    processed_samples: int = 0
    # Previous diarization result for merging
    previous_utterances: list = field(default_factory=list)
    # Flag to indicate if we should use the full pipeline or fast mode
    first_chunk: bool = True
    # Cumulative transcript for the session
    all_transcripts: list = field(default_factory=list)


# ============================================================
# Audio helpers
# ============================================================
def _pcm16_to_float32(chunk: bytes) -> np.ndarray:
    """Convert PCM16 bytes to normalized float32 numpy array."""
    audio_i16 = np.frombuffer(chunk, dtype=np.int16)
    return audio_i16.astype(np.float32) / 32768.0


def _compute_rms(audio: np.ndarray) -> float:
    """Compute RMS energy of audio signal."""
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio ** 2)))


def _is_speech(audio: np.ndarray, threshold: float = VAD_THRESHOLD) -> bool:
    """Simple VAD check based on RMS energy."""
    return _compute_rms(audio) > threshold


# ============================================================
# Transcription + Diarization
# ============================================================
def _transcribe_chunk(audio: np.ndarray, language: str) -> list[dict]:
    """Run Whisper transcription on audio chunk with word timestamps."""
    segments, _info = whisper_pipeline.transcribe(
        audio,
        language=language,
        suppress_tokens=suppress_tokens,
        batch_size=8,
        word_timestamps=True,
    )

    word_timestamps = []
    for segment in segments:
        words = getattr(segment, "words", None) or []
        for word in words:
            if word.start is None or word.end is None:
                continue
            token_text = (word.word or "").strip()
            if not token_text:
                continue
            word_timestamps.append({
                "start": float(word.start),
                "end": float(word.end),
                "text": token_text,
            })

    return word_timestamps


def _diarize_chunk(audio: np.ndarray) -> list[tuple]:
    """Run Sortformer diarization on audio chunk."""
    speaker_ts = diarizer.diarize(torch.from_numpy(audio).unsqueeze(0))
    return speaker_ts


def _process_audio_chunk(
    audio: np.ndarray,
    offset_ms: int,
    language: str,
    use_diarization: bool = True,
) -> list[dict]:
    """Transcribe and optionally diarize an audio chunk."""
    # Transcription
    word_timestamps = _transcribe_chunk(audio, language)
    if not word_timestamps:
        return []

    if not use_diarization:
        # Fast mode: no diarization, just return text with timestamps
        items = []
        for w in word_timestamps:
            items.append({
                "speaker": "Speaker 0",
                "start_ms": int(w["start"] * 1000) + offset_ms,
                "end_ms": int(w["end"] * 1000) + offset_ms,
                "text": w["text"],
            })
        return items

    # Diarization
    speaker_ts = _diarize_chunk(audio)
    if not speaker_ts:
        # Fallback if diarizer returns nothing
        items = []
        for w in word_timestamps:
            items.append({
                "speaker": "Speaker 0",
                "start_ms": int(w["start"] * 1000) + offset_ms,
                "end_ms": int(w["end"] * 1000) + offset_ms,
                "text": w["text"],
            })
        return items

    # Merge word timestamps with speaker labels
    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, word_anchor_option="mid")

    # Convert to utterances (group consecutive words from same speaker)
    utterances = _words_to_utterances(wsm, offset_ms)
    return utterances


def _words_to_utterances(
    word_speaker_mapping: list[dict],
    offset_ms: int,
) -> list[dict]:
    """Group consecutive words from the same speaker into utterances."""
    if not word_speaker_mapping:
        return []

    utterances = []
    current = None

    for item in word_speaker_mapping:
        speaker_id = int(item["speaker"])
        speaker = f"Speaker {speaker_id}"
        start_ms = int(item["start_time"]) + offset_ms
        end_ms = int(item["end_time"]) + offset_ms
        word = item["word"].strip()
        if not word:
            continue

        if current is None or current["speaker"] != speaker:
            current = {
                "speaker": speaker,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "text": word,
            }
            utterances.append(current)
        else:
            current["end_ms"] = end_ms
            current["text"] = f'{current["text"]} {word}'.strip()

    return utterances


# ============================================================
# FastAPI routes
# ============================================================
@app.get("/")
def index() -> HTMLResponse:
    html_path = Path("web/index.html")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.websocket("/ws/realtime")
async def realtime_ws(websocket: WebSocket):
    await websocket.accept()
    state = SessionState()
    await websocket.send_text(json.dumps({
        "type": "ready",
        "sample_rate": SAMPLE_RATE,
        "chunk_seconds": CHUNK_SECONDS,
        "model": WHISPER_MODEL_SIZE,
    }))
    logger.info("Client connected")

    try:
        while True:
            message = await websocket.receive()

            # Handle binary audio data
            if "bytes" in message and message["bytes"] is not None:
                state.audio_buffer.extend(message["bytes"])

                # Process when we have enough samples
                while len(state.audio_buffer) >= CHUNK_SAMPLES * 2:
                    raw = bytes(state.audio_buffer[:CHUNK_SAMPLES * 2])
                    del state.audio_buffer[:CHUNK_SAMPLES * 2]

                    offset_ms = int(state.processed_samples * 1000 / SAMPLE_RATE)
                    state.processed_samples += CHUNK_SAMPLES

                    audio = _pcm16_to_float32(raw)

                    # Simple VAD: skip silent chunks
                    if not _is_speech(audio):
                        logger.debug("Skipping silent chunk (RMS=%.4f)", _compute_rms(audio))
                        continue

                    start_time = time.time()

                    # Run transcription + diarization
                    utterances = await asyncio.to_thread(
                        _process_audio_chunk,
                        audio,
                        offset_ms,
                        state.language,
                        use_diarization=True,
                    )

                    elapsed_ms = int((time.time() - start_time) * 1000)

                    if utterances:
                        # Store for history
                        state.all_transcripts.extend(utterances)

                        # Send to client
                        await websocket.send_text(json.dumps({
                            "type": "transcript",
                            "items": utterances,
                            "active_speaker": utterances[-1]["speaker"],
                            "processing_ms": elapsed_ms,
                            "realtime_factor": round(
                                (CHUNK_SECONDS * 1000) / max(elapsed_ms, 1), 2
                            ),
                        }, ensure_ascii=False))
                        logger.info(
                            "Processed chunk in %dms (RTF=%.2f): %s",
                            elapsed_ms,
                            (CHUNK_SECONDS * 1000) / max(elapsed_ms, 1),
                            utterances[-1]["text"][:50],
                        )
                    else:
                        # Send empty result to keep client alive
                        await websocket.send_text(json.dumps({
                            "type": "no_speech",
                            "offset_ms": offset_ms,
                        }))

            # Handle text messages (config, stop, etc.)
            elif "text" in message and message["text"] is not None:
                payload = json.loads(message["text"])

                if payload.get("type") == "config":
                    if payload.get("language"):
                        state.language = str(payload["language"]).lower()
                        logger.info("Language changed to: %s", state.language)
                    await websocket.send_text(json.dumps({
                        "type": "ack",
                        "message": f"Language set to {state.language}",
                    }))

                elif payload.get("type") == "stop":
                    await websocket.send_text(json.dumps({"type": "stopped"}))
                    break

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Realtime session error: %s", exc)
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": str(exc),
            }))
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
