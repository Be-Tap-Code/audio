import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import faster_whisper
import numpy as np
import torch

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from diarization import SortformerDiarizer
from helpers import (
    find_numeral_symbol_tokens,
    get_realigned_ws_mapping_with_punctuation,
    get_words_speaker_mapping,
)

# Optional: speaker identification
try:
    from speaker_identification import SpeakerIdentifier, DEFAULT_THRESHOLD

    SPEAKER_ID_AVAILABLE = True
except ImportError:
    SPEAKER_ID_AVAILABLE = False
    SpeakerIdentifier = None
    DEFAULT_THRESHOLD = 0.65

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("realtime-diarization")

# ============================================================
# Configuration
# ============================================================
SAMPLE_RATE = 16000
CHUNK_SECONDS = float(os.environ.get("REALTIME_CHUNK_SECONDS", "3.0"))
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SECONDS)
MAX_CONTEXT_SECONDS = float(os.environ.get("REALTIME_MAX_CONTEXT_SECONDS", "12.0"))
MAX_CONTEXT_SAMPLES = int(SAMPLE_RATE * MAX_CONTEXT_SECONDS)
VAD_THRESHOLD = float(os.environ.get("REALTIME_VAD_THRESHOLD", "0.01"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "large-v2")
DIARIZER_NAME = "sortformer"
DEFAULT_LANGUAGE = os.environ.get("DEFAULT_LANGUAGE", "vi")

# Speaker identification config
SPEAKER_ID_ENABLED = False
SPEAKER_DB_PATH = os.environ.get("SPEAKER_DB_PATH", "speaker_db.json")
SPEAKER_ID_THRESHOLD = float(os.environ.get("SPEAKER_ID_THRESHOLD", str(DEFAULT_THRESHOLD)))

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
whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)

logger.info("Loading %s diarizer on %s", DIARIZER_NAME, DEVICE)
diarizer = SortformerDiarizer(device=DEVICE)

speaker_identifier = None
if SPEAKER_ID_AVAILABLE and os.path.exists(SPEAKER_DB_PATH):
    try:
        speaker_identifier = SpeakerIdentifier(
            device=DEVICE,
            db_path=SPEAKER_DB_PATH,
            threshold=SPEAKER_ID_THRESHOLD,
        )
        if speaker_identifier.db.get_count() > 0:
            SPEAKER_ID_ENABLED = True
            logger.info(
                "Speaker identification enabled: %d speakers loaded",
                speaker_identifier.db.get_count(),
            )
        else:
            speaker_identifier = None
            logger.info("Speaker database empty - identification disabled")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to init speaker identifier: %s", exc)
        speaker_identifier = None
else:
    logger.info("Speaker identification not available (db: %s)", SPEAKER_DB_PATH)


# ============================================================
# Session state
# ============================================================
@dataclass
class SessionState:
    language: str = DEFAULT_LANGUAGE
    audio_buffer: bytearray = field(default_factory=bytearray)
    context_buffer: bytearray = field(default_factory=bytearray)
    processed_samples: int = 0
    last_emitted_ms: int = 0
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
    return float(np.sqrt(np.mean(audio**2)))


def _is_speech(audio: np.ndarray, threshold: float = VAD_THRESHOLD) -> bool:
    """Simple VAD check based on RMS energy."""
    return _compute_rms(audio) > threshold


def _append_context(state: SessionState, raw_chunk: bytes) -> tuple[np.ndarray, int]:
    """Append a chunk to the rolling context and return the current window and its offset."""
    state.context_buffer.extend(raw_chunk)
    max_context_bytes = MAX_CONTEXT_SAMPLES * 2
    if len(state.context_buffer) > max_context_bytes:
        del state.context_buffer[: len(state.context_buffer) - max_context_bytes]

    context_samples = len(state.context_buffer) // 2
    context_start_samples = max(0, state.processed_samples - context_samples)
    context_offset_ms = int(context_start_samples * 1000 / SAMPLE_RATE)
    return _pcm16_to_float32(bytes(state.context_buffer)), context_offset_ms


# ============================================================
# Transcription + Diarization
# ============================================================
def _transcribe_chunk(audio: np.ndarray, language: str) -> list[dict]:
    """Run Whisper transcription on audio window with word timestamps."""
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
            word_timestamps.append(
                {
                    "start": float(word.start),
                    "end": float(word.end),
                    "text": token_text,
                }
            )

    return word_timestamps


def _diarize_chunk(audio: np.ndarray) -> list[tuple]:
    """Run Sortformer diarization on audio window."""
    return diarizer.diarize(torch.from_numpy(audio).unsqueeze(0))


def _process_audio_chunk(
    audio: np.ndarray,
    offset_ms: int,
    language: str,
    emit_from_ms: int = 0,
    use_diarization: bool = True,
) -> tuple[list[dict], list[tuple]]:
    """Transcribe and optionally diarize an audio window."""
    word_timestamps = _transcribe_chunk(audio, language)
    if not word_timestamps:
        return [], []

    if not use_diarization:
        items = []
        for word in word_timestamps:
            end_ms = int(word["end"] * 1000) + offset_ms
            if end_ms <= emit_from_ms:
                continue
            items.append(
                {
                    "speaker": "Speaker 0",
                    "start_ms": int(word["start"] * 1000) + offset_ms,
                    "end_ms": end_ms,
                    "text": word["text"],
                }
            )
        return items, []

    speaker_ts = _diarize_chunk(audio)
    if not speaker_ts:
        items = []
        for word in word_timestamps:
            end_ms = int(word["end"] * 1000) + offset_ms
            if end_ms <= emit_from_ms:
                continue
            items.append(
                {
                    "speaker": "Speaker 0",
                    "start_ms": int(word["start"] * 1000) + offset_ms,
                    "end_ms": end_ms,
                    "text": word["text"],
                }
            )
        return items, []

    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, word_anchor_option="start")
    wsm = get_realigned_ws_mapping_with_punctuation(wsm)
    wsm = [item for item in wsm if int(item["end_time"]) + offset_ms > emit_from_ms]
    if not wsm:
        return [], speaker_ts

    utterances = _words_to_utterances(wsm, offset_ms)

    if SPEAKER_ID_ENABLED:
        utterances = _resolve_speaker_names(utterances, speaker_ts, audio)

    return utterances, speaker_ts


def _resolve_speaker_names(
    utterances: list[dict],
    speaker_ts: list[tuple],
    audio: np.ndarray,
) -> list[dict]:
    """Replace 'Speaker N' with actual names using speaker identification."""
    if not speaker_identifier or not speaker_ts:
        return utterances

    try:
        name_map = speaker_identifier.identify_speaker_names(
            speaker_ts,
            torch.from_numpy(audio).unsqueeze(0),
        )
        if name_map:
            for utt in utterances:
                spk_id = int(utt["speaker"].split()[-1])
                if spk_id in name_map:
                    utt["speaker"] = name_map[spk_id]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Speaker identification failed: %s", exc)

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

        should_split = current is None or current["speaker"] != speaker
        if current is not None and current["text"].rstrip().endswith((".", "?", "!")):
            should_split = True

        if should_split:
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


@app.get("/api/speakers")
def list_speakers():
    """List all registered speakers."""
    if not speaker_identifier:
        return {"enabled": False, "speakers": []}
    return {
        "enabled": True,
        "speakers": speaker_identifier.list_speakers(),
        "threshold": SPEAKER_ID_THRESHOLD,
    }


@app.post("/api/speakers/register")
async def register_speaker(name: str, audio_path: str):
    """Register a new speaker from audio file."""
    if not speaker_identifier:
        return {"error": "Speaker identification not enabled"}
    try:
        speaker_identifier.register_speaker(name, audio_path)
        return {"status": "ok", "message": f"Registered: {name}"}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


@app.delete("/api/speakers/{name}")
def remove_speaker(name: str):
    """Remove a speaker from database."""
    if not speaker_identifier:
        return {"error": "Speaker identification not enabled"}
    if speaker_identifier.remove_speaker(name):
        return {"status": "ok", "message": f"Removed: {name}"}
    return {"error": f"Speaker not found: {name}"}


@app.websocket("/ws/realtime")
async def realtime_ws(websocket: WebSocket):
    await websocket.accept()
    state = SessionState()
    await websocket.send_text(
        json.dumps(
            {
                "type": "ready",
                "sample_rate": SAMPLE_RATE,
                "chunk_seconds": CHUNK_SECONDS,
                "model": WHISPER_MODEL_SIZE,
                "speaker_id": {
                    "enabled": SPEAKER_ID_ENABLED,
                    "speaker_count": speaker_identifier.db.get_count() if speaker_identifier else 0,
                },
            }
        )
    )
    logger.info("Client connected")

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"] is not None:
                state.audio_buffer.extend(message["bytes"])

                while len(state.audio_buffer) >= CHUNK_SAMPLES * 2:
                    raw = bytes(state.audio_buffer[: CHUNK_SAMPLES * 2])
                    del state.audio_buffer[: CHUNK_SAMPLES * 2]

                    state.processed_samples += CHUNK_SAMPLES
                    chunk_audio = _pcm16_to_float32(raw)

                    if not _is_speech(chunk_audio):
                        logger.debug("Skipping silent chunk (RMS=%.4f)", _compute_rms(chunk_audio))
                        continue

                    context_audio, context_offset_ms = _append_context(state, raw)
                    start_time = time.time()

                    utterances, speaker_ts = await asyncio.to_thread(
                        _process_audio_chunk,
                        context_audio,
                        context_offset_ms,
                        state.language,
                        state.last_emitted_ms,
                        True,
                    )

                    elapsed_ms = int((time.time() - start_time) * 1000)

                    if utterances:
                        state.last_emitted_ms = max(
                            state.last_emitted_ms,
                            max(item["end_ms"] for item in utterances),
                        )
                        state.all_transcripts.extend(utterances)

                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "transcript",
                                    "items": utterances,
                                    "active_speaker": utterances[-1]["speaker"],
                                    "processing_ms": elapsed_ms,
                                    "realtime_factor": round(
                                        (CHUNK_SECONDS * 1000) / max(elapsed_ms, 1), 2
                                    ),
                                },
                                ensure_ascii=False,
                            )
                        )
                        logger.info(
                            "Processed chunk in %dms (RTF=%.2f): %s",
                            elapsed_ms,
                            (CHUNK_SECONDS * 1000) / max(elapsed_ms, 1),
                            utterances[-1]["text"][:50],
                        )
                    else:
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "no_speech",
                                    "offset_ms": context_offset_ms,
                                }
                            )
                        )

            elif "text" in message and message["text"] is not None:
                payload = json.loads(message["text"])

                if payload.get("type") == "config":
                    if payload.get("language"):
                        state.language = str(payload["language"]).lower()
                        logger.info("Language changed to: %s", state.language)
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "ack",
                                "message": f"Language set to {state.language}",
                            }
                        )
                    )

                elif payload.get("type") == "stop":
                    await websocket.send_text(json.dumps({"type": "stopped"}))
                    break

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Realtime session error: %s", exc)
        try:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "error",
                        "message": str(exc),
                    }
                )
            )
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
