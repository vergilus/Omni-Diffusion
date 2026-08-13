#!/usr/bin/env python3
"""Convert a LibriTTS-R train-clean split into Omni-Diffusion TTS JSONL.

Each ``*.wav`` is paired with the same-stem ``*.normalized.txt`` file.  The
normalized transcript, rather than ``*.original.txt``, is used as the TTS
prompt.  The emitted records follow the TTS schema documented in README.md.
"""

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path


TTS_PROMPT = "Convert the text to speech.\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/share/Audio/LibriTTS-R/extracted/LibriTTS_R"),
        help="LibriTTS_R root containing train-clean-* directories.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train-clean-*"],
        help="Glob(s), relative to --input-root, to scan (default: train-clean-*).",
    )
    parser.add_argument(
        "--relative-audio-paths",
        action="store_true",
        help="Store paths relative to --input-root instead of absolute paths.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file.")
    parser.add_argument("--limit", type=int, help="Write at most this many valid records.")
    return parser.parse_args()


def iter_wavs(root: Path, split_patterns: list[str]):
    """Yield wave files in a deterministic order without scanning a split twice."""
    seen = set()
    for pattern in split_patterns:
        for split_dir in sorted(root.glob(pattern)):
            if not split_dir.is_dir():
                continue
            for wav_path in sorted(split_dir.rglob("*.wav")):
                resolved = wav_path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    yield resolved


def make_record(wav_path: Path, input_root: Path, relative_audio_paths: bool) -> dict | None:
    transcript_path = wav_path.with_suffix(".normalized.txt")
    if not transcript_path.is_file():
        return None

    text = " ".join(transcript_path.read_text(encoding="utf-8").split()) # normalize whitespace
    if not text or len(text.split()) > 70:
        return None

    audio_path = str(wav_path.relative_to(input_root) if relative_audio_paths else wav_path)
    return {
        "messages": [
            {"role": "user", "content": f"{TTS_PROMPT}{text}"},
            {"role": "assistant", "content": "<|audio|>"},
        ],
        "audios": [audio_path],
    }


def main() -> int:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not input_root.is_dir():
        raise SystemExit(f"Input root does not exist: {input_root}")
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output}; use --overwrite to replace it.")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive.")

    wav_paths = iter_wavs(input_root, args.splits)
    stats: Counter[str] = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        try:
            for wav_path in wav_paths:
                stats["wav_seen"] += 1
                transcript_path = wav_path.with_suffix(".normalized.txt")
                if not transcript_path.is_file():
                    stats["missing_normalized_text"] += 1
                    continue

                record = make_record(wav_path, input_root, args.relative_audio_paths)
                if record is None:
                    stats["empty_normalized_text"] += 1
                    continue

                temporary_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["written"] += 1
                if args.limit is not None and stats["written"] >= args.limit:
                    break
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    temporary_path.replace(output)
    print(
        "Wrote {written} records to {output}; scanned {wav_seen} wavs, "
        "skipped {missing_normalized_text} missing and {empty_normalized_text} empty or oversized transcripts.".format(
            output=output,
            written=stats["written"],
            wav_seen=stats["wav_seen"],
            missing_normalized_text=stats["missing_normalized_text"],
            empty_normalized_text=stats["empty_normalized_text"],
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
