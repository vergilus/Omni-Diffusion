#!/usr/bin/env python3
"""Convert LibriSpeech Parquet conversations into Omni-Diffusion ASR JSONL."""

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    import pyarrow.parquet as pq
except ImportError as exc:
    raise SystemExit("Install pyarrow first: python -m pip install pyarrow") from exc


SOURCE_ASR_PROMPT = "请将以下音频转录为文字："
TARGET_ASR_PROMPT = "Convert the speech to text.\n<|audio|>"
ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory searched recursively for *.parquet.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument(
        "--audio-root",
        type=Path,
        required=True,
        help="Base directory prepended to relative paths in the audios column.",
    )
    parser.add_argument("--conversation-column", default="conversations")
    parser.add_argument("--audio-column", default="audios")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, help="Write at most this many records for a smoke test.")
    return parser.parse_args()


def convert_conversations(value: Any) -> list[dict[str, str]] | None:
    """Convert [{from: ..., value: ...}] to Omni-Diffusion messages."""
    if not isinstance(value, list):
        return None

    messages = []
    for turn in value:
        if not isinstance(turn, dict):
            return None
        role = ROLE_MAP.get(turn.get("from", turn.get("role")))
        content = turn.get("value", turn.get("content"))
        if role is None or not isinstance(content, str):
            return None
        if role == "user":
            content = content.replace(SOURCE_ASR_PROMPT, TARGET_ASR_PROMPT)
        messages.append({"role": role, "content": content})

    if not messages or not any(message["role"] == "assistant" for message in messages):
        return None
    return messages


def resolve_audio_paths(value: Any, audio_root: Path) -> list[str] | None:
    if not isinstance(value, list) or not value:
        return None
    resolved = []
    for item in value:
        if not isinstance(item, str) or not item:
            return None
        path = Path(item).expanduser()
        if not path.is_absolute():
            path = audio_root / path
        resolved.append(str(path.resolve(strict=False)))
    return resolved


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    audio_root = args.audio_root.expanduser().resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    if not audio_root.is_dir():
        raise SystemExit(f"Audio root does not exist: {audio_root}")
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output}; use --overwrite to replace it.")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive.")

    parquet_files = sorted(input_dir.rglob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"No *.parquet files found under {input_dir}")

    output.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        try:
            for parquet_path in parquet_files:
                parquet_file = pq.ParquetFile(parquet_path)
                required = {args.conversation_column, args.audio_column}
                missing = required - set(parquet_file.schema_arrow.names)
                if missing:
                    raise ValueError(f"{parquet_path} is missing columns: {sorted(missing)}")

                for batch in parquet_file.iter_batches(columns=[args.conversation_column, args.audio_column]):
                    for row in batch.to_pylist():
                        messages = convert_conversations(row[args.conversation_column])
                        audios = resolve_audio_paths(row[args.audio_column], audio_root)
                        if messages is None or audios is None:
                            skipped += 1
                            continue
                        temporary_file.write(json.dumps({"messages": messages, "audios": audios}, ensure_ascii=False) + "\n")
                        written += 1
                        if args.limit is not None and written >= args.limit:
                            break
                    if args.limit is not None and written >= args.limit:
                        break
                if args.limit is not None and written >= args.limit:
                    break
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    temporary_path.replace(output)
    print(f"Wrote {written} records to {output}; skipped {skipped} invalid rows.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
