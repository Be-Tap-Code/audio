#!/usr/bin/env python3
"""
Speaker Diarization Web UI - FastAPI Backend
Call Deepgram API for Vietnamese speaker diarization
"""

import os
import sys
import json
import uuid
import tempfile
import asyncio
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global task storage
tasks = {}

# Configuration
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Constraints
MAX_AUDIO_DURATION_SECONDS = 45 * 60  # 45 minutes
RECOMMENDED_MAX_SPEAKERS = 8  # Best performance with fewer speakers


def call_deepgram_api(
    audio_path: str,
    api_key: str,
    model: str = "nova-3",
    language: str = "vi",
    diarize: bool = True,
    punctuate: bool = True,
    utterances: bool = True,
) -> dict:
    """
    Call Deepgram API with optimal configuration for Vietnamese.
    """
    url = "https://api.deepgram.com/v1/listen"

    params = {
        "model": model,
        "language": language,
        "diarize": "true" if diarize else "false",
        "punctuate": "true" if punctuate else "false",
        "utterances": "true" if utterances else "false",
    }

    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "audio/mpeg" if audio_path.lower().endswith(".mp3") else "audio/wav",
    }

    try:
        with open(audio_path, "rb") as f:
            response = requests.post(url, params=params, headers=headers, data=f, timeout=300)

        response.raise_for_status()
        return response.json()

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {audio_path}")
    except requests.exceptions.RequestException as e:
        error_msg = f"API request failed: {e}"
        if hasattr(e, "response") and e.response is not None:
            error_msg += f"\nResponse: {e.response.text}"
        raise Exception(error_msg)


def format_utterances(data: dict, min_confidence: float = 0.0) -> list[str]:
    """
    Format utterances from JSON response into [Speaker:X] Text format.
    """
    output_lines = []

    try:
        results = data.get("results", {})
        utterances = results.get("utterances", [])

        if not utterances:
            channels = results.get("channels", [])
            if channels and channels[0].get("alternatives"):
                words = channels[0]["alternatives"][0].get("words", [])
                return format_from_words(words, min_confidence)

        for utt in utterances:
            confidence = utt.get("confidence", 0)
            if confidence < min_confidence:
                continue

            speaker = utt.get("speaker", "unknown")
            transcript = utt.get("transcript", "").strip()

            if transcript:
                output_lines.append(f"[Speaker:{speaker}] {transcript}")

    except (KeyError, TypeError, IndexError) as e:
        print(f"Warning: Error parsing response: {e}", file=sys.stderr)

    return output_lines


def format_from_words(words: list[dict], min_confidence: float = 0.0) -> list[str]:
    """
    Fallback: Format from words array when utterances are not available.
    """
    if not words:
        return []

    output_lines = []
    current_speaker = None
    current_sentence = []

    for word_data in words:
        confidence = word_data.get("confidence", 0)
        if confidence < min_confidence:
            continue

        speaker = word_data.get("speaker", "unknown")
        word = word_data.get("punctuated_word") or word_data.get("word", "")

        if speaker != current_speaker and current_sentence:
            text = "".join(current_sentence).strip()
            if text:
                output_lines.append(f"[Speaker:{current_speaker}] {text}")
            current_sentence = []

        current_speaker = speaker
        current_sentence.append(word)

    if current_sentence:
        text = "".join(current_sentence).strip()
        if text:
            output_lines.append(f"[Speaker:{current_speaker}] {text}")

    return output_lines


async def process_audio_task(task_id: str, file_path: str, model: str, language: str):
    """
    Background task to process audio file.
    """
    try:
        tasks[task_id]["status"] = "processing"
        tasks[task_id]["progress"] = 10

        # Check audio duration using mutagen (pure Python, no ffmpeg required)
        try:
            from mutagen.mp3 import MP3
            from mutagen.mp4 import MP4
            from mutagen.flac import FLAC
            from mutagen.oggvorbis import OggVorbis
            from mutagen.wave import WAVE

            audio = None
            file_ext = Path(file_path).suffix.lower()

            if file_ext == ".mp3":
                audio = MP3(file_path)
            elif file_ext == ".m4a":
                audio = MP4(file_path)
            elif file_ext == ".flac":
                audio = FLAC(file_path)
            elif file_ext == ".ogg":
                audio = OggVorbis(file_path)
            elif file_ext == ".wav":
                audio = WAVE(file_path)

            if audio is not None and hasattr(audio, 'info') and hasattr(audio.info, 'length'):
                duration = audio.info.length
                tasks[task_id]["duration_seconds"] = duration

                # Validate duration constraint
                if duration > MAX_AUDIO_DURATION_SECONDS:
                    raise Exception(
                        f"Audio duration ({duration/60:.1f} minutes) exceeds the maximum limit of "
                        f"{MAX_AUDIO_DURATION_SECONDS/60:.0f} minutes. Please use a shorter audio file."
                    )

                tasks[task_id]["duration_formatted"] = f"{duration/60:.1f} minutes"
        except Exception as e:
            # If duration check fails, log warning but continue processing
            if "exceeds the maximum limit" in str(e):
                raise e
            print(f"Warning: Could not check duration: {e}", file=sys.stderr)

        # Call Deepgram API
        response_data = call_deepgram_api(
            audio_path=file_path,
            api_key=DEEPGRAM_API_KEY,
            model=model,
            language=language,
        )

        tasks[task_id]["progress"] = 80

        # Format output
        lines = format_utterances(response_data, min_confidence=0.0)

        # Check speaker count for recommendation
        speakers = set()
        for line in lines:
            if line.startswith("[Speaker:"):
                speaker_id = line.split("]")[0].replace("[Speaker:", "")
                speakers.add(speaker_id)

        num_speakers = len(speakers)
        tasks[task_id]["num_speakers"] = num_speakers

        if num_speakers > RECOMMENDED_MAX_SPEAKERS:
            tasks[task_id]["warning"] = (
                f"Detected {num_speakers} speakers. For best accuracy, "
                f"we recommend audio with {RECOMMENDED_MAX_SPEAKERS} or fewer speakers."
            )

        tasks[task_id]["progress"] = 100

        # Save output text
        output_text = "\n".join(lines)
        output_file = Path(tempfile.gettempdir()) / f"{task_id}.txt"
        output_file.write_text(output_text, encoding="utf-8")

        # Save JSON response
        json_file = Path(tempfile.gettempdir()) / f"{task_id}.json"
        json_file.write_text(json.dumps(response_data, ensure_ascii=False, indent=2), encoding="utf-8")

        # Update task status
        tasks[task_id]["status"] = "done"
        tasks[task_id]["result"] = {
            "text_file": str(output_file),
            "json_file": str(json_file),
            "output_text": output_text,
            "duration": response_data.get("metadata", {}).get("duration", 0),
            "duration_formatted": tasks[task_id].get("duration_formatted"),
            "language": language,
            "num_speakers": num_speakers,
            "warning": tasks[task_id].get("warning"),
        }

    except Exception as e:
        tasks[task_id]["status"] = "error"
        tasks[task_id]["error"] = str(e)

    finally:
        # Clean up uploaded file
        try:
            Path(file_path).unlink(missing_ok=True)
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Cleanup on shutdown
    for task_data in tasks.values():
        if "file_path" in task_data:
            try:
                Path(task_data["file_path"]).unlink(missing_ok=True)
            except Exception:
                pass


app = FastAPI(title="Hệ thống ghi chép tự động", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    """Serve the main HTML page."""
    return FileResponse(Path(__file__).parent / "index.html")


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    model: str = "nova-3",
    language: str = "vi",
    background_tasks: BackgroundTasks = None,
):
    """
    Upload audio file and start processing.
    """
    if not DEEPGRAM_API_KEY:
        raise HTTPException(status_code=500, detail="DEEPGRAM_API_KEY not configured")

    # Validate file type
    allowed_extensions = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}")

    # Generate unique task ID
    task_id = str(uuid.uuid4())

    # Save uploaded file
    file_path = UPLOAD_DIR / f"{task_id}{file_ext}"
    try:
        content = await file.read()
        file_path.write_bytes(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Create task entry
    tasks[task_id] = {
        "status": "pending",
        "progress": 0,
        "filename": file.filename,
        "language": language,
        "file_path": str(file_path),
    }

    # Start background processing
    background_tasks.add_task(process_audio_task, task_id, str(file_path), model, language)

    return {
        "task_id": task_id,
        "status": "pending",
        "message": "File uploaded successfully. Processing started."
    }


@app.get("/api/status/{task_id}")
async def get_task_status(task_id: str):
    """
    Get task processing status.
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task_data = tasks[task_id]
    return {
        "task_id": task_id,
        "status": task_data["status"],
        "progress": task_data.get("progress", 0),
        "filename": task_data.get("filename"),
        "result": task_data.get("result"),
        "error": task_data.get("error"),
    }


@app.get("/api/download/{task_id}/{file_type}")
async def download_result(task_id: str, file_type: str):
    """
    Download processed result file.
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task_data = tasks[task_id]
    if task_data["status"] != "done":
        raise HTTPException(status_code=400, detail="Task not completed yet")

    if file_type == "txt":
        file_path = task_data["result"]["text_file"]
        return FileResponse(file_path, filename=f"{task_data['filename']}_output.txt", media_type="text/plain")
    elif file_type == "json":
        file_path = task_data["result"]["json_file"]
        return FileResponse(file_path, filename=f"{task_data['filename']}_response.json", media_type="application/json")
    else:
        raise HTTPException(status_code=400, detail="Invalid file type. Use 'txt' or 'json'")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "api_key_configured": bool(DEEPGRAM_API_KEY),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
