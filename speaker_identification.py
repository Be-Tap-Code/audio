"""
Speaker Identification Module

Workflow:
1. Put reference audios in a folder: speakers/An.mp3, speakers/Long.wav, ...
2. Run diarize.py with --speakers-dir speakers/
3. Output will show speaker names instead of "Speaker 0", "Speaker 1", ...

Reference speaker embeddings are cached on disk and only recomputed when
speaker files are added, modified, or removed.
"""

import json
import logging
from pathlib import Path

import faster_whisper
import numpy as np
import torch
import torchaudio

from nemo.collections.asr.models import EncDecSpeakerLabelModel

logger = logging.getLogger("speaker-identification")

EMBEDDING_MODEL = "titanet_large"
DEFAULT_THRESHOLD = 0.20
MIN_SEGMENT_SECONDS = 1.0
MAX_SEGMENTS_PER_SPEAKER = 8
CACHE_FILENAME = ".speaker_embeddings_cache.npz"
CACHE_VERSION = 1
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus"}


class SpeakerIdentifier:
    """Identify diarized speakers from a directory of named reference audios."""

    def __init__(
        self,
        speakers_dir: str,
        device: str = "cuda",
        threshold: float = DEFAULT_THRESHOLD,
        model_name: str = EMBEDDING_MODEL,
    ):
        self.device = device
        self.threshold = threshold
        self.model_name = model_name
        self.profiles: dict[str, np.ndarray] = {}
        self.cache_path: Path | None = None

        logger.info("Loading speaker embedding model: %s on %s", model_name, device)
        self.model = EncDecSpeakerLabelModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        self._load_speakers(speakers_dir)

    def _load_speakers(self, speakers_dir: str):
        """Load speaker reference audios from a directory, using a persistent cache."""
        speakers_path = Path(speakers_dir)
        if not speakers_path.exists():
            raise FileNotFoundError(f"Speakers directory not found: {speakers_dir}")
        if not speakers_path.is_dir():
            raise NotADirectoryError(f"Speakers path is not a directory: {speakers_dir}")

        self.cache_path = speakers_path / CACHE_FILENAME
        audio_files = [
            audio_file
            for audio_file in sorted(speakers_path.iterdir())
            if audio_file.is_file() and audio_file.suffix.lower() in AUDIO_EXTENSIONS
        ]
        if not audio_files:
            raise ValueError(f"No valid audio files found in {speakers_dir}")

        cache_meta, cache_embeddings = self._read_cache()
        current_profiles: dict[str, np.ndarray] = {}
        current_cache_meta: dict[str, dict] = {}
        cache_dirty = False

        for audio_file in audio_files:
            name = audio_file.stem.strip()
            if not name:
                logger.warning("Skipping unnamed reference file: %s", audio_file.name)
                continue

            file_key = audio_file.name
            file_meta = self._build_file_meta(audio_file)
            current_cache_meta[file_key] = file_meta

            cached_meta = cache_meta.get(file_key)
            cached_embedding = cache_embeddings.get(file_key)
            if cached_meta == file_meta and cached_embedding is not None:
                logger.info("Loaded cached embedding: %s", audio_file.name)
                current_profiles[name] = cached_embedding.astype(np.float32)
                continue

            logger.info("Computing embedding: %s", audio_file.name)
            try:
                embedding = self._extract_embedding_from_file(str(audio_file))
                current_profiles[name] = embedding.astype(np.float32)
                cache_embeddings[file_key] = current_profiles[name]
                cache_dirty = True
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to load %s: %s", audio_file.name, exc)
                current_cache_meta.pop(file_key, None)

        removed_files = set(cache_meta) - set(current_cache_meta)
        if removed_files:
            cache_dirty = True
            for file_key in sorted(removed_files):
                logger.info("Removing stale cache entry: %s", file_key)
                cache_embeddings.pop(file_key, None)

        if not current_profiles:
            raise ValueError(f"No valid audio files found in {speakers_dir}")

        self.profiles = current_profiles

        if cache_dirty or cache_meta != current_cache_meta:
            cache_subset = {
                file_key: cache_embeddings[file_key]
                for file_key in current_cache_meta
                if file_key in cache_embeddings
            }
            self._write_cache(current_cache_meta, cache_subset)
        else:
            logger.info("Speaker cache is up to date: %s", self.cache_path)

        logger.info("Loaded %d speaker profiles from %s", len(self.profiles), speakers_dir)

    def _build_file_meta(self, audio_file: Path) -> dict:
        stat = audio_file.stat()
        return {
            "name": audio_file.stem.strip(),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    def _read_cache(self) -> tuple[dict[str, dict], dict[str, np.ndarray]]:
        if self.cache_path is None or not self.cache_path.exists():
            return {}, {}

        try:
            cache = np.load(self.cache_path, allow_pickle=False)
            meta_blob = cache["__meta__"].item()
            meta = json.loads(meta_blob)
            if meta.get("version") != CACHE_VERSION or meta.get("model_name") != self.model_name:
                logger.info("Ignoring outdated speaker cache: %s", self.cache_path)
                return {}, {}

            files_meta = meta.get("files", {})
            embeddings = {
                file_key: cache[f"emb::{file_key}"].astype(np.float32)
                for file_key in files_meta
                if f"emb::{file_key}" in cache.files
            }
            logger.info("Loaded speaker cache: %s", self.cache_path)
            return files_meta, embeddings
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read speaker cache %s: %s", self.cache_path, exc)
            return {}, {}

    def _write_cache(self, files_meta: dict[str, dict], embeddings: dict[str, np.ndarray]):
        if self.cache_path is None:
            return

        payload = {
            "__meta__": np.array(
                json.dumps(
                    {
                        "version": CACHE_VERSION,
                        "model_name": self.model_name,
                        "files": files_meta,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        }
        for file_key, embedding in embeddings.items():
            payload[f"emb::{file_key}"] = np.asarray(embedding, dtype=np.float32)

        np.savez_compressed(self.cache_path, **payload)
        logger.info("Saved speaker cache: %s", self.cache_path)

    def identify(self, audio: torch.Tensor) -> str:
        """Identify a single speaker segment."""
        if not self.profiles:
            return "Unknown"
        query_embedding = self._extract_embedding(audio)
        best_name, best_score = self._best_match(query_embedding)
        return best_name if best_score >= self.threshold else "Unknown"

    def identify_batch(
        self,
        speaker_segments: list[tuple],
        full_audio: torch.Tensor,
        sample_rate: int = 16000,
    ) -> dict[int, str]:
        """Map each diarized speaker id to its highest-scoring known speaker name."""
        if not self.profiles or not speaker_segments:
            return {}

        speaker_embeddings = self._build_speaker_embeddings(speaker_segments, full_audio, sample_rate)
        if not speaker_embeddings:
            logger.warning("Could not build any diarized speaker embeddings.")
            return {}

        speaker_name_map: dict[int, str] = {}
        for spk_id, embedding in sorted(speaker_embeddings.items()):
            scored_candidates = []
            for name, profile_embedding in sorted(self.profiles.items()):
                score = self._cosine_similarity(embedding, profile_embedding)
                scored_candidates.append((score, name))
                logger.info("Speaker %d vs %s: score=%.3f", spk_id, name, score)

            best_score, best_name = max(scored_candidates, key=lambda item: item[0])
            if best_score >= self.threshold:
                speaker_name_map[spk_id] = best_name
                logger.info(
                    "Assigned Speaker %d -> %s (best score=%.3f, threshold=%.3f)",
                    spk_id,
                    best_name,
                    best_score,
                    self.threshold,
                )
            else:
                speaker_name_map[spk_id] = f"Unknown_{spk_id}"
                logger.info(
                    "Assigned Speaker %d -> %s (best score=%.3f below threshold=%.3f)",
                    spk_id,
                    speaker_name_map[spk_id],
                    best_score,
                    self.threshold,
                )

        return speaker_name_map

    def list_speakers(self) -> list[str]:
        return sorted(self.profiles.keys())

    def get_count(self) -> int:
        return len(self.profiles)

    def _build_speaker_embeddings(
        self,
        speaker_segments: list[tuple],
        full_audio: torch.Tensor,
        sample_rate: int,
    ) -> dict[int, np.ndarray]:
        """Average embeddings from the longest usable segments per diarized speaker."""
        speaker_embeddings: dict[int, np.ndarray] = {}
        unique_speakers = sorted({segment[2] for segment in speaker_segments})
        min_segment_samples = int(sample_rate * MIN_SEGMENT_SECONDS)

        for spk_id in unique_speakers:
            segments = [seg for seg in speaker_segments if seg[2] == spk_id]
            segments = sorted(segments, key=lambda seg: seg[1] - seg[0], reverse=True)
            embeddings = []

            for start_ms, end_ms, _ in segments[:MAX_SEGMENTS_PER_SPEAKER]:
                start_sample = max(0, int(start_ms * sample_rate / 1000))
                end_sample = min(full_audio.shape[-1], int(end_ms * sample_rate / 1000))
                if (end_sample - start_sample) < min_segment_samples:
                    continue

                segment_audio = full_audio[:, start_sample:end_sample]
                try:
                    embeddings.append(self._extract_embedding(segment_audio))
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "Failed to extract embedding for speaker %d segment %d-%d ms: %s",
                        spk_id,
                        start_ms,
                        end_ms,
                        exc,
                    )

            if not embeddings:
                logger.warning(
                    "Speaker %d has no usable segments >= %.1fs for identification.",
                    spk_id,
                    MIN_SEGMENT_SECONDS,
                )
                continue

            speaker_embeddings[spk_id] = np.mean(embeddings, axis=0).astype(np.float32)

        return speaker_embeddings

    def _extract_embedding_from_file(self, audio_path: str) -> np.ndarray:
        """Extract a speaker embedding from a reference audio file."""
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "torchaudio could not read %s (%s). Falling back to ffmpeg decoder.",
                audio_path,
                exc,
            )
            decoded = faster_whisper.decode_audio(audio_path)
            waveform = torch.from_numpy(decoded).unsqueeze(0)
            sample_rate = 16000

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

        return self._extract_embedding(waveform)

    def _extract_embedding(self, audio: torch.Tensor) -> np.ndarray:
        """Extract a speaker embedding from a mono 16kHz waveform."""
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        if audio.shape[0] != 1:
            audio = audio.mean(dim=0, keepdim=True)

        with torch.no_grad():
            device = next(self.model.parameters()).device
            audio = audio.to(device=device, dtype=torch.float32)
            audio_length = torch.tensor([audio.shape[-1]], device=device, dtype=torch.int64)
            _logits, embedding = self.model.forward(
                input_signal=audio,
                input_signal_length=audio_length,
            )
            if embedding is None:
                raise RuntimeError("Speaker model returned no embedding tensor.")
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.detach().cpu().numpy()
            if embedding.ndim > 1:
                embedding = embedding.mean(axis=0)

        return embedding.astype(np.float32)

    def _best_match(self, query_embedding: np.ndarray) -> tuple[str, float]:
        """Return the best matching known speaker and its cosine similarity."""
        best_score = -1.0
        best_name = "Unknown"

        for name, profile_embedding in self.profiles.items():
            score = self._cosine_similarity(query_embedding, profile_embedding)
            if score > best_score:
                best_score = score
                best_name = name

        return best_name, best_score

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
