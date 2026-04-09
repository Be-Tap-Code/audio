import importlib.util
import platform
import shutil
import sys

REQUIRED_MODULES = [
    "torch",
    "torchaudio",
    "faster_whisper",
    "ctc_forced_aligner",
    "deepmultilingualpunctuation",
    "nemo",
    "demucs",
    "soundfile",
    "fastapi",
    "uvicorn",
]


def check_module(name: str) -> tuple[bool, str]:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return False, "missing"
    return True, "ok"


def main() -> int:
    print("== whisper-diarization environment check ==")
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print()

    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 10):
        print("[FAIL] Python 3.10+ is required")
        return 1
    print("[ OK ] Python version is supported")

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        print(f"[ OK ] ffmpeg found at: {ffmpeg_path}")
    else:
        print("[FAIL] ffmpeg is not installed or not on PATH")

    missing = []
    for module_name in REQUIRED_MODULES:
        ok, status = check_module(module_name)
        label = "OK" if ok else "FAIL"
        print(f"[{label:>4}] import {module_name}: {status}")
        if not ok:
            missing.append(module_name)

    try:
        import torch

        print(f"[ OK ] torch version: {torch.__version__}")
        print(f"[ OK ] CUDA available: {torch.cuda.is_available()}")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] torch runtime check failed: {exc}")

    print()
    if ffmpeg_path and not missing:
        print("Environment looks ready.")
        print("Note: the first real run will still download Whisper / NeMo / punctuation models.")
        return 0

    print("Environment is incomplete.")
    print("Install the missing parts, then run this check again.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
