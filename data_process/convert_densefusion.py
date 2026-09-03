#!/usr/bin/env python3
"""Convert FineVision DenseFusion Parquet data to Omni-Diffusion JSONL."""

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

try:
    import pyarrow.parquet as pq
except ImportError as exc:
    raise SystemExit("Install pyarrow first: python -m pip install pyarrow") from exc


SOURCE_IMAGE_TAG = "<image>"
OMNI_IMAGE_TAG = "<|image|>"

ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory searched recursively for *.parquet files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory; creates images/ and the JSONL file here.")
    parser.add_argument("--image-dir-name", default="images")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--conversation-column", default="conversations")
    parser.add_argument("--image-column", default="image")
    parser.add_argument(
        "--image-extension",
        choices=("auto", "jpg", "jpeg", "png", "webp"),
        default="auto",
        help="Suffix for materialized files; auto detects JPEG/PNG/WebP.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, help="Write at most this many valid records.")
    parser.add_argument("--batch-size", type=int, default=256, help="Parquet record-batch size (default: 256).")
    return parser.parse_args()


def _image_bytes(value: Any) -> bytes | None:
    """Extract one encoded image from DenseFusion's one-element image list."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        result = bytes(value)
    elif isinstance(value, (list, tuple)):
        if len(value) != 1:
            return None
        return _image_bytes(value[0])
    elif isinstance(value, dict):
        for key in ("bytes", "data", "image"):
            if key in value:
                return _image_bytes(value[key])
        return None
    else:
        return None
    return result if result else None


def _detected_extension(data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp"
    return ".bin"


def _convert_conversations(value: Any) -> list[dict[str, str]] | None:
    if not isinstance(value, list) or not value:
        return None
    messages: list[dict[str, str]] = []
    for turn in value:
        if not isinstance(turn, dict):
            return None
        role = ROLE_MAP.get(turn.get("from", turn.get("role")))
        content = turn.get("value", turn.get("content"))
        if role is None or not isinstance(content, str) or not content.strip():
            return None
        content = content.replace(SOURCE_IMAGE_TAG, OMNI_IMAGE_TAG)
        messages.append({"role": role, "content": content})
    if not any(message["role"] == "assistant" for message in messages):
        return None
    return messages


def _safe_sample_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    sample_id = value.strip()
    if not sample_id or sample_id in {".", ".."} or "/" in sample_id or "\\" in sample_id:
        return None
    return sample_id


def _iter_rows(parquet_files: Iterable[Path], args: argparse.Namespace):
    columns = [args.id_column, args.conversation_column, args.image_column]
    for parquet_path in parquet_files:
        parquet_file = pq.ParquetFile(parquet_path)
        available = set(parquet_file.schema_arrow.names)
        missing = set(columns) - available
        if missing:
            raise ValueError(f"{parquet_path} is missing columns: {sorted(missing)}")
        for batch in parquet_file.iter_batches(columns=columns, batch_size=args.batch_size):
            yield from zip(*(column.to_pylist() for column in batch.columns))


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    image_dir = output_dir / args.image_dir_name
    output = output_dir / "train.jsonl"
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output}; use --overwrite to replace it.")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive.")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")

    parquet_files = sorted(input_dir.rglob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"No *.parquet files found under {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    stats: Counter[str] = Counter()
    seen_ids: set[str] = set()
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_dir,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        try:
            for sample_id_value, conversations_value, image_value in _iter_rows(parquet_files, args):
                stats["seen"] += 1
                sample_id = _safe_sample_id(sample_id_value)
                messages = _convert_conversations(conversations_value)
                image_data = _image_bytes(image_value)
                if sample_id is None:
                    stats["invalid_id"] += 1
                    continue
                if sample_id in seen_ids:
                    stats["duplicate_id"] += 1
                    continue
                if messages is None:
                    stats["invalid_conversations"] += 1
                    continue
                if image_data is None:
                    stats["invalid_image"] += 1
                    continue
                if not any(OMNI_IMAGE_TAG in message["content"] for message in messages):
                    user_message = next((m for m in messages if m["role"] == "user"), None)
                    if user_message is None:
                        stats["missing_image_tag"] += 1
                        continue
                    user_message["content"] = OMNI_IMAGE_TAG + "\n" + user_message["content"]
                    stats["inserted_image_tag"] += 1
                suffix = f".{args.image_extension}" if args.image_extension != "auto" else _detected_extension(image_data)
                image_name = f"{sample_id}{suffix}"
                image_path = image_dir / image_name
                if image_path.exists() and not args.overwrite:
                    raise SystemExit(f"Image already exists: {image_path}; use --overwrite to replace it.")
                image_path.write_bytes(image_data)
                temporary_file.write(
                    json.dumps({"messages": messages, "images": [str(image_path)]}, ensure_ascii=False) + "\n"
                )
                seen_ids.add(sample_id)
                stats["written"] += 1
                if args.limit is not None and stats["written"] >= args.limit:
                    break
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
    temporary_path.replace(output)
    print(
        f"Wrote {stats['written']} records to {output}; materialized images under {image_dir}. "
        f"Seen {stats['seen']} rows; stats: {dict(stats)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

