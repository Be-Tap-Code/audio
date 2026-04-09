# Whisper Diarization

Pipeline transcribe + forced alignment + speaker diarization built on Faster-Whisper, NeMo, Demucs, and FastAPI.

## Short answer

No. On a new machine, `pip install -r requirements.txt` alone is not enough.

You also need:
- `ffmpeg` installed at the OS level and available on `PATH`
- `torch` and `torchaudio` installed separately for the correct runtime (`cpu` or your CUDA version)
- Python `3.10+`
- Internet access on the first run so Whisper / NeMo / punctuation models can be downloaded

## Supported setup

This repo is prepared for Python `3.10` to `3.12`.

Recommended workflow on a fresh machine:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel cython uv
```

### 1. Install ffmpeg

Ubuntu / Debian:
```bash
sudo apt update
sudo apt install -y ffmpeg
```

macOS:
```bash
brew install ffmpeg
```

Windows:
```powershell
winget install ffmpeg
```

### 2. Install PyTorch for the target machine

CPU only:
```bash
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

CUDA example:
```bash
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Pick the PyTorch index that matches the actual machine / CUDA driver. If you are not using GPU, install the CPU build.

### 3. Install project dependencies

```bash
python -m pip install -c constraints.txt -r requirements.txt
python -m pip install -e .
```

`pip install -e .` is recommended so the package metadata and CLI entrypoint are registered locally.

## Verify environment

Run:

```bash
python scripts/check_environment.py
whisper-diarize --help
```

If both commands work, the environment is usually ready.

## Usage

Basic CLI:

```bash
python diarize.py -a path/to/audio.wav
```

Installed CLI:

```bash
whisper-diarize -a path/to/audio.wav
```

Useful options:
- `--device cpu` to force CPU mode
- `--no-stem` to skip Demucs source separation
- `--diarizer sortformer` to use the realtime-friendly diarizer
- `--speakers-dir speakers/` to map `Speaker N` to known speaker names

Example:

```bash
whisper-diarize -a meeting.mp3 --device cpu --diarizer sortformer --no-stem
```

## Realtime web demo

```bash
timeout 20s uvicorn web_realtime:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000`.

## Notes for new environments

- The first run downloads multiple models, so startup can be slow.
- `demucs` and audio decoding depend on `ffmpeg`; without it the pipeline is incomplete.
- `speaker_identification.py` also needs `torchaudio` at runtime.
- If you commit this repo, avoid committing generated audio outputs, `build/`, `speakers/`, caches, and `temp_outputs_*`.

## Quick checklist before commit / push

- Run `python scripts/check_environment.py`
- Run `python -m compileall diarize.py web_realtime.py speaker_identification.py helpers.py diarization scripts`
- Make sure no generated outputs or local speaker samples are staged
- Keep `requirements.txt`, `constraints.txt`, `pyproject.toml`, and `README.md` in sync
