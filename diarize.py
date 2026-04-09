import argparse
import logging
import os
import re

import faster_whisper
import torch

from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
from deepmultilingualpunctuation import PunctuationModel

from helpers import (
    cleanup,
    find_numeral_symbol_tokens,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    get_words_speaker_mapping,
    langs_to_iso,
    process_language_arg,
    punct_model_langs,
    whisper_langs,
    write_srt,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

MTYPES = {"cpu": "int8", "cuda": "float16"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--audio", help="name of the target audio file", required=True)
    parser.add_argument(
        "--no-stem",
        action="store_false",
        dest="stemming",
        default=True,
        help="Disables source separation.This helps with long files that don't contain a lot of music.",
    )
    parser.add_argument(
        "--suppress_numerals",
        action="store_true",
        dest="suppress_numerals",
        default=False,
        help="Suppresses Numerical Digits."
        "This helps the diarization accuracy but converts all digits into written text.",
    )
    parser.add_argument(
        "--whisper-model",
        dest="model_name",
        default="medium.en",
        help="name of the Whisper model to use",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        dest="batch_size",
        default=8,
        help="Batch size for batched inference, reduce if you run out of memory, "
        "set to 0 for original whisper longform inference",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        choices=whisper_langs,
        help="Language spoken in the audio, specify None to perform language detection",
    )
    parser.add_argument(
        "--device",
        dest="device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="if you have a GPU use 'cuda', otherwise 'cpu'",
    )
    parser.add_argument(
        "--diarizer",
        default="msdd",
        choices=["msdd", "sortformer"],
        help="Choose the diarization model to use",
    )
    parser.add_argument(
        "--no-speaker-realignment",
        action="store_true",
        default=False,
        help="Disable speaker realignment based on punctuation. "
        "Useful for languages without clear punctuation or alternating speakers.",
    )
    parser.add_argument(
        "--speakers-dir",
        type=str,
        default=None,
        help="Path to folder containing speaker reference audios (e.g. An.mp3, Long.wav). "
        "File names (without extension) become speaker labels.",
    )
    parser.add_argument(
        "--identify-threshold",
        type=float,
        default=0.20,
        help="Cosine similarity threshold for speaker identification (default: 0.20). "
        "Lower = more likely to match, higher = stricter.",
    )
    return parser


def _resolve_vocal_target(args: argparse.Namespace, temp_path: str) -> str:
    if not args.stemming:
        return args.audio

    return_code = os.system(
        f"python -m demucs.separate -n htdemucs --two-stems=vocals "
        f'"{args.audio}" -o "{temp_path}" --device "{args.device}"'
    )

    vocals_path = os.path.join(
        temp_path,
        "htdemucs",
        os.path.splitext(os.path.basename(args.audio))[0],
        "vocals.wav",
    )
    if return_code != 0 or not os.path.exists(vocals_path):
        logging.warning(
            "Source splitting failed, using original audio file. "
            "Use --no-stem argument to disable it."
        )
        return args.audio

    return vocals_path


def _load_diarizer(name: str, device: str):
    if name == "msdd":
        from diarization import MSDDDiarizer

        return MSDDDiarizer(device=device)

    from diarization import SortformerDiarizer

    return SortformerDiarizer(device=device)


def _restore_punctuation(word_speaker_mapping: list[dict], language: str) -> list[dict]:
    if language not in punct_model_langs:
        logging.warning(
            "Punctuation restoration is not available for %s language. Using the original punctuation.",
            language,
        )
        return word_speaker_mapping

    punct_model = PunctuationModel(model="kredor/punctuate-all")
    words_list = [item["word"] for item in word_speaker_mapping]
    labeled_words = punct_model.predict(words_list, chunk_size=230)

    ending_puncts = ".?!"
    model_puncts = ".,;:!?"
    is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

    for word_dict, labeled_tuple in zip(word_speaker_mapping, labeled_words):
        word = word_dict["word"]
        if (
            word
            and labeled_tuple[1] in ending_puncts
            and (word[-1] not in model_puncts or is_acronym(word))
        ):
            word += labeled_tuple[1]
            if word.endswith(".."):
                word = word.rstrip(".")
            word_dict["word"] = word

    return word_speaker_mapping


def _apply_speaker_identification(
    args: argparse.Namespace,
    sentence_speaker_mapping: list[dict],
    word_speaker_mapping: list[dict],
    speaker_ts: list[tuple],
    audio_waveform,
    vocal_target: str,
):
    if not args.speakers_dir:
        return

    from speaker_identification import SpeakerIdentifier

    if not os.path.isdir(args.speakers_dir):
        logging.warning("Speakers directory not found: %s", args.speakers_dir)
        logging.info("Skipping speaker identification.")
        return

    logging.info("=" * 60)
    logging.info("SPEAKER IDENTIFICATION")
    logging.info("Loading reference audios from: %s", args.speakers_dir)
    logging.info("=" * 60)

    speaker_identifier = SpeakerIdentifier(
        speakers_dir=args.speakers_dir,
        device=args.device,
        threshold=args.identify_threshold,
    )
    logging.info(
        "Identified %d speakers: %s",
        speaker_identifier.get_count(),
        ", ".join(speaker_identifier.list_speakers()),
    )

    identification_waveform = audio_waveform
    if os.path.abspath(vocal_target) != os.path.abspath(args.audio):
        logging.info("Using original audio for speaker identification: %s", args.audio)
        identification_waveform = faster_whisper.decode_audio(args.audio)

    speaker_name_map = speaker_identifier.identify_batch(
        speaker_ts,
        torch.from_numpy(identification_waveform).unsqueeze(0),
    )

    if not speaker_name_map:
        logging.warning("Could not identify any speakers from the audio.")
        del speaker_identifier
        torch.cuda.empty_cache()
        return

    logging.info("")
    logging.info("Speaker mapping results:")
    for spk_id, name in sorted(speaker_name_map.items()):
        logging.info("  Speaker %d -> %s", spk_id, name)
    logging.info("")

    for sentence in sentence_speaker_mapping:
        current = sentence["speaker"]
        try:
            spk_id = int(current.split()[-1])
        except (ValueError, IndexError):
            continue
        if spk_id in speaker_name_map:
            sentence["speaker"] = speaker_name_map[spk_id]

    for item in word_speaker_mapping:
        spk_id = item["speaker"]
        if spk_id in speaker_name_map:
            item["speaker"] = speaker_name_map[spk_id]

    del speaker_identifier
    torch.cuda.empty_cache()


def run(args: argparse.Namespace) -> None:
    language = process_language_arg(args.language, args.model_name)
    temp_path = os.path.join(os.getcwd(), f"temp_outputs_{os.getpid()}")
    os.makedirs(temp_path, exist_ok=True)

    try:
        vocal_target = _resolve_vocal_target(args, temp_path)

        whisper_model = faster_whisper.WhisperModel(
            args.model_name,
            device=args.device,
            compute_type=MTYPES[args.device],
        )
        whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)
        audio_waveform = faster_whisper.decode_audio(vocal_target)
        suppress_tokens = (
            find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
            if args.suppress_numerals
            else [-1]
        )

        if args.batch_size > 0:
            transcript_segments, info = whisper_pipeline.transcribe(
                audio_waveform,
                language,
                suppress_tokens=suppress_tokens,
                batch_size=args.batch_size,
            )
        else:
            transcript_segments, info = whisper_model.transcribe(
                audio_waveform,
                language,
                suppress_tokens=suppress_tokens,
                vad_filter=True,
            )

        full_transcript = "".join(segment.text for segment in transcript_segments)

        del whisper_model, whisper_pipeline
        torch.cuda.empty_cache()

        alignment_model, alignment_tokenizer = load_alignment_model(
            args.device,
            dtype=torch.float16 if args.device == "cuda" else torch.float32,
        )
        emissions, stride = generate_emissions(
            alignment_model,
            torch.from_numpy(audio_waveform).to(alignment_model.dtype).to(alignment_model.device),
            batch_size=args.batch_size,
        )

        del alignment_model
        torch.cuda.empty_cache()

        tokens_starred, text_starred = preprocess_text(
            full_transcript,
            romanize=True,
            language=langs_to_iso[info.language],
        )
        segments, scores, blank_token = get_alignments(
            emissions,
            tokens_starred,
            alignment_tokenizer,
        )
        spans = get_spans(tokens_starred, segments, blank_token)
        word_timestamps = postprocess_results(text_starred, spans, stride, scores)

        diarizer_model = _load_diarizer(args.diarizer, args.device)
        speaker_ts = diarizer_model.diarize(torch.from_numpy(audio_waveform).unsqueeze(0))
        del diarizer_model
        torch.cuda.empty_cache()

        word_speaker_mapping = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")
        word_speaker_mapping = _restore_punctuation(word_speaker_mapping, info.language)

        if args.no_speaker_realignment:
            logging.info("Speaker realignment is disabled.")
        else:
            word_speaker_mapping = get_realigned_ws_mapping_with_punctuation(word_speaker_mapping)

        sentence_speaker_mapping = get_sentences_speaker_mapping(word_speaker_mapping, speaker_ts)
        _apply_speaker_identification(
            args,
            sentence_speaker_mapping,
            word_speaker_mapping,
            speaker_ts,
            audio_waveform,
            vocal_target,
        )

        output_prefix = os.path.splitext(args.audio)[0]
        with open(f"{output_prefix}.txt", "w", encoding="utf-8-sig") as file_handle:
            get_speaker_aware_transcript(sentence_speaker_mapping, file_handle)

        with open(f"{output_prefix}.srt", "w", encoding="utf-8-sig") as srt_handle:
            write_srt(sentence_speaker_mapping, srt_handle)

        logging.info("Done! Output files:")
        logging.info("  - %s.txt", output_prefix)
        logging.info("  - %s.srt", output_prefix)
    finally:
        if os.path.exists(temp_path):
            cleanup(temp_path)


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
