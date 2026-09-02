"""Measure Fisher-Rao spacetime intervals on training-like data.

The script uses the same Qwen2Dataset preprocessing as training. For every
packed item it constructs masked states online, evaluates the model only at
supervised target positions, and accumulates adjacent-step Fisher-Rao
intervals. CPU memory is bounded by small scalar sums; only the current and
previous probability pair is retained on the model device, and complete time
trajectories are never retained.

For data-parallel execution, launch the same script with torchrun, for example
``torchrun --standalone --nproc_per_node=8 experiments/check_fisher.py ...``.
Each rank owns one full model copy and a strided training-data view; rank 0
merges only small per-task/per-interval scalar summaries after local Fisher
calculations finish.
"""

import argparse
import contextlib
import json
import logging
import os
import random
import sys
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
THIRD_PARTY_ROOT = os.path.join(REPO_ROOT, "third_party", "GLM-4-Voice")
for path in (
    REPO_ROOT,
    THIRD_PARTY_ROOT,
    os.path.join(THIRD_PARTY_ROOT, "cosyvoice"),
    os.path.join(THIRD_PARTY_ROOT, "third_party", "Matcha-TTS"),
):
    if path not in sys.path:
        sys.path.insert(0, path)

# Avoid a read-only default cache when the model uses Triton kernels.
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp")

import torch
from transformers import AutoTokenizer

import omni_diffusion.data.dataset_qwen2 as dataset_qwen2_module
from omni_diffusion.data.dataset_qwen2 import Qwen2Dataset
from omni_diffusion.data.processor.image_processor import ImageProcessor
from omni_diffusion.models.dream import DreamConfig, DreamModel
from omni_diffusion.tokenizer import get_audio_tokenizer, update_tokenizer


LOGGER = logging.getLogger("check_fisher")
MASK_TOKEN_ID = 151666
IGNORE_TOKEN_ID = -100
AUDIO_TOKEN_COUNT = 16384
IMAGE_TOKEN_COUNT = 8192
EPS = 1e-12


def distributed_context() -> Dict[str, int]:
    """Read torchrun metadata without initializing a process group."""
    return {
        "rank": int(os.environ.get("RANK", "0")),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
    }


def init_distributed(context: Dict[str, int]) -> None:
    """Initialize one process per GPU when launched through torchrun."""
    world_size = context["world_size"]
    if world_size <= 1:
        return
    if torch.distributed.is_initialized():
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    init_kwargs = {
        "backend": backend,
        "rank": context["rank"],
        "world_size": world_size,
        "timeout": timedelta(hours=2),
    }
    if backend == "nccl":
        # Tell NCCL the exact local device instead of making it infer one from
        # the global rank. This is important on multi-process, multi-node jobs.
        init_kwargs["device_id"] = torch.device(
            "cuda", context["local_rank"]
        )
    torch.distributed.init_process_group(**init_kwargs)


def is_main_process(context: Dict[str, int]) -> bool:
    return context["rank"] == 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _move_nested(value, device):
    """Move tensors in the audio structures produced by the collator."""
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, list):
        return [_move_nested(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_nested(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_nested(item, device) for key, item in value.items()}
    return value


@contextlib.contextmanager
def mask_all_supervised_positions():
    """Make preprocessing return the complete x_T target mask.

    Qwen2Dataset still performs normal text/audio preprocessing. Only its
    final random masking decision is replaced, so labels remain the untouched
    x_0 target sequence from the training path.
    """
    original_forward_process = dataset_qwen2_module.forward_process

    def mask_all(bsz, seq_len, device, labels, **kwargs):
        return labels.ne(IGNORE_TOKEN_ID), torch.ones((bsz, 1), device=device)

    dataset_qwen2_module.forward_process = mask_all
    try:
        yield
    finally:
        dataset_qwen2_module.forward_process = original_forward_process


def fisher_rao_distance(
    first: torch.Tensor, second: torch.Tensor, eps: float = EPS
) -> torch.Tensor:
    """Return Fisher-Rao angles for probability vectors [..., vocab]."""
    first = first.float().clamp_min(0)
    second = second.float().clamp_min(0)
    first = first / first.sum(dim=-1, keepdim=True).clamp_min(eps)
    second = second / second.sum(dim=-1, keepdim=True).clamp_min(eps)
    affinity = (first.sqrt() * second.sqrt()).sum(dim=-1).clamp(0.0, 1.0)
    # 4 asin(sqrt((1 - BC) / 2)) is 2 acos(BC), but is more stable near BC=1.
    return 4.0 * torch.asin(
        torch.sqrt(((1.0 - affinity).clamp_min(0.0)) / 2.0)
    )


def _new_task_interval_stats(num_intervals: int) -> Dict:
    """Create scalar accumulators for one task, indexed by time interval."""
    zeros = lambda: [0.0] * num_intervals
    counts = lambda: [0] * num_intervals
    return {
        "num_samples": 0,
        "target_position_count": 0,
        "sequence_interval_sum": zeros(),
        "sequence_interval_squared_sum": zeros(),
        "sequence_interval_negative_count": counts(),
        "sequence_distance_sum": zeros(),
        "sequence_distance_squared_sum": zeros(),
        "local_interval_sum": zeros(),
        "local_interval_squared_sum": zeros(),
        "local_token_count": counts(),
        "local_interval_negative_count": counts(),
        "sample_interval_mean_sum": 0.0,
        "sample_interval_mean_squared_sum": 0.0,
        "sample_path_length_sum": 0.0,
        "sample_path_length_squared_sum": 0.0,
    }


def _merge_task_interval_stats(destination: Dict, source: Dict) -> None:
    """Add one rank/batch accumulator into another in place."""
    if destination["num_samples"] == 0 and destination["target_position_count"] == 0:
        destination.update({
            key: (list(value) if isinstance(value, list) else value)
            for key, value in source.items()
        })
        return
    for key, value in source.items():
        if isinstance(value, list):
            if len(destination[key]) != len(value):
                raise ValueError(f"interval accumulator length mismatch for {key}")
            for index, item in enumerate(value):
                destination[key][index] += item
        elif key in destination:
            destination[key] += value
        else:
            destination[key] = value


def _merge_rank_summaries(destination: Dict, source: Dict) -> None:
    """Merge rank-local scalar summaries without loading sample trajectories."""
    destination["num_samples"] += int(source.get("num_samples", 0))
    for task, source_stats in source.get("by_task", {}).items():
        task_stats = destination["by_task"].setdefault(
            task, _new_task_interval_stats(len(source_stats["sequence_interval_sum"]))
        )
        _merge_task_interval_stats(task_stats, source_stats)


@torch.inference_mode()
def fisher_interval_statistics(
    previous: torch.Tensor,
    current: torch.Tensor,
    delta_time: float,
    c: float,
    row_chunk_size: int = 256,
) -> Dict[str, float]:
    """Compute one interval using bounded row temporaries.

    ``previous`` and ``current`` are the only trajectory tensors retained by
    the caller. Fisher angles are reduced chunk by chunk, so this function
    never materializes a full ``[target_positions]`` distance vector.
    """
    if previous.shape != current.shape or previous.ndim != 2:
        raise ValueError("Fisher interval inputs must have shape [targets, vocab]")
    if row_chunk_size < 1:
        raise ValueError("row_chunk_size must be positive")
    target_count = int(previous.shape[0])
    if target_count == 0:
        raise ValueError("Fisher interval requires at least one target")

    distance_squared_sum = 0.0
    local_interval_sum = 0.0
    local_interval_squared_sum = 0.0
    local_negative_count = 0
    temporal_per_token = (c * float(delta_time)) ** 2
    for start in range(0, target_count, row_chunk_size):
        stop = min(start + row_chunk_size, target_count)
        distances = fisher_rao_distance(previous[start:stop], current[start:stop])
        distances = distances.float()
        squared = distances.square()
        local_interval = temporal_per_token - squared
        distance_squared_sum += float(squared.sum())
        local_interval_sum += float(local_interval.sum())
        local_interval_squared_sum += float(local_interval.square().sum())
        local_negative_count += int((local_interval < 0).sum())
        del distances, squared, local_interval

    sequence_distance = distance_squared_sum**0.5
    sequence_interval = target_count * temporal_per_token - distance_squared_sum
    return {
        "target_count": target_count,
        "sequence_distance": sequence_distance,
        "sequence_interval": sequence_interval,
        "sequence_interval_squared": sequence_interval**2,
        "sequence_interval_negative": float(sequence_interval < 0),
        "local_interval_sum": local_interval_sum,
        "local_interval_squared_sum": local_interval_squared_sum,
        "local_negative_count": local_negative_count,
    }


def theta_from_alpha(alpha: torch.Tensor) -> torch.Tensor:
    """Fisher-angle clock for a masked channel with retention alpha."""
    return 2.0 * torch.acos(alpha.clamp(0.0, 1.0).sqrt())


def _as_audio_list(value) -> List[torch.Tensor]:
    if value is None:
        return []
    if torch.is_tensor(value):
        return list(value) if value.ndim > 1 else [value]
    return list(value)


def _as_audio_index_list(value) -> List[torch.Tensor]:
    if value is None:
        return []
    if torch.is_tensor(value):
        return list(value) if value.ndim >= 3 else [value]
    return list(value)


def collate_analysis_states(states: Sequence[Dict]) -> Dict:
    """Stack fixed-length packed states and offset their audio batch rows."""
    if not states:
        raise ValueError("cannot collate an empty analysis batch")

    def stack_field(name: str, default_factory):
        values = []
        for state in states:
            value = state.get(name)
            if value is None:
                value = default_factory(state["input_ids"])
            value = torch.as_tensor(value).view(-1)
            values.append(value)
        lengths = {value.numel() for value in values}
        if len(lengths) != 1:
            raise ValueError(f"analysis batch has inconsistent {name} lengths")
        return torch.stack(values, dim=0)

    input_ids = stack_field("input_ids", lambda value: value)
    attention_mask = stack_field(
        "attention_mask", lambda value: torch.ones_like(value)
    )
    position_ids = stack_field(
        "position_ids", lambda value: torch.arange(value.numel())
    )
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }

    audios = []
    audio_indices = []
    for sample_index, state in enumerate(states):
        sample_audios = _as_audio_list(state.get("audios"))
        sample_indices = _as_audio_index_list(state.get("audio_indices"))
        if len(sample_audios) != len(sample_indices):
            raise ValueError(
                "analysis batch audio and audio_indices lengths do not match"
            )
        audios.extend(sample_audios)
        for indices in sample_indices:
            indices = torch.as_tensor(indices).clone()
            if indices.ndim < 2 or indices.shape[0] != 2:
                raise ValueError("audio_indices must have leading shape [2, ...]")
            indices[0].fill_(sample_index)
            audio_indices.append(indices)
    if audios:
        batch["audios"] = audios
        batch["audio_indices"] = audio_indices
    return batch


def packed_block_attention_mask(
    attention_mask: torch.Tensor, position_ids: torch.Tensor
) -> torch.Tensor:
    """Build Dream's [batch, 1, length, length] packed attention mask."""
    if attention_mask.ndim != 2 or position_ids.ndim != 2:
        raise ValueError("packed mask inputs must have shape [batch, length]")
    is_new = position_ids == 0
    segment_id = torch.cumsum(is_new.long(), dim=1) - 1
    block_mask = (
        segment_id.unsqueeze(1) == segment_id.unsqueeze(2)
    ).long()
    return (block_mask * attention_mask.unsqueeze(-1)).to(torch.bool).unsqueeze(1)


class FisherAnalyzer:
    """Loaded Dream model plus the operations needed for this study."""

    def __init__(
        self,
        model_name_or_path: str,
        audio_tokenizer_path: Optional[str],
        audio_tokenizer_type: Optional[str],
        image_tokenizer_path: Optional[str],
        flow_path: Optional[str],
        device_map: str,
        torch_dtype: torch.dtype,
        load_image_tokenizer: bool = False,
        audio_tokenizer_rank: Optional[int] = None,
        image_tokenizer_rank: Optional[int] = None,
    ):
        config = DreamConfig.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        if audio_tokenizer_type is not None:
            tokenizer = update_tokenizer(tokenizer, audio_tokenizer_type)
        tokenizer.add_tokens(
            [f"<|image_{index}|>" for index in range(IMAGE_TOKEN_COUNT)],
            special_tokens=False,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = DreamModel.from_pretrained(
            model_name_or_path,
            config=config,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation="sdpa",
        ).eval()
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(
                len(tokenizer),
                pad_to_multiple_of=8,
                mean_resizing=False,
            )

        self.model = model
        self.tokenizer = tokenizer
        self.model_device = next(model.parameters()).device
        self.audio_tokenizer = get_audio_tokenizer(
            audio_tokenizer_path,
            audio_tokenizer_type,
            flow_path=flow_path,
            rank=audio_tokenizer_rank,
        )

        # Qwen2Dataset expects this processor object even for ASR/TTS data.
        self.image_processor = ImageProcessor(
            image_tokenizer_path,
            "dynamic",
            image_size=512,
            normalize_type="imagenet",
            min_patch_grid=1,
            max_patch_grid=12,
        )
        self.image_processor.image_tokenizer.rank = (
            image_tokenizer_rank if torch.cuda.is_available() else None
        )
        if load_image_tokenizer:
            self.image_processor.load_model()
        else:
            # Current ASR/TTS records have no image fields. Avoid loading the
            # unrelated MAGVIT weights unless an image-containing YAML is used.
            self.image_processor.image_tokenizer = None
        self.audio_token_offset = self.tokenizer.convert_tokens_to_ids("<|audio_0|>")
        LOGGER.info("model_device=%s vocab_size=%d", self.model_device, len(tokenizer))

    def forward_kwargs(self, state: Dict) -> Dict:
        input_ids = state["input_ids"]
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.model_device)

        attention_mask = state.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        elif attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
        attention_mask = attention_mask.to(self.model_device)

        position_ids = state.get("position_ids")
        if position_ids is None:
            position_ids = torch.arange(
                input_ids.shape[1], device=self.model_device
            ).unsqueeze(0)
        elif position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        position_ids = position_ids.to(self.model_device)

        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "audios": _move_nested(state.get("audios"), self.model_device),
            "audio_indices": _move_nested(
                state.get("audio_indices"), self.model_device
            ),
            "use_cache": False,
            "return_dict": True,
        }
        return {key: value for key, value in kwargs.items() if value is not None}

    @torch.inference_mode()
    def target_probabilities_batch(
        self,
        state: Dict,
        prediction_positions: Sequence[torch.Tensor],
        chunk_size: int,
    ) -> List[torch.Tensor]:
        """Return target-row distributions on the model device.

        Keeping this one step's tensor on-device avoids a large CPU transfer;
        :meth:`analyze_batch` releases it as soon as the adjacent interval is
        reduced.
        """
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if not prediction_positions:
            raise ValueError("prediction_positions cannot be empty")

        positions_by_sample = [
            torch.as_tensor(positions).view(-1).long()
            for positions in prediction_positions
        ]
        if any(positions.numel() == 0 for positions in positions_by_sample):
            raise ValueError("every batch sample must have prediction positions")

        kwargs = self.forward_kwargs(state)
        input_ids = kwargs["input_ids"]
        if input_ids.shape[0] != len(positions_by_sample):
            raise ValueError("prediction position batch size does not match input_ids")
        sequence_length = input_ids.shape[1]
        for positions in positions_by_sample:
            if positions.min() < 0 or positions.max() >= sequence_length:
                raise IndexError("prediction position is outside the input sequence")

        counts = [positions.numel() for positions in positions_by_sample]
        flat_batch = torch.cat(
            [
                torch.full((count,), sample_index, dtype=torch.long)
                for sample_index, count in enumerate(counts)
            ]
        )
        flat_positions = torch.cat(positions_by_sample)
        # Preserve fisher_chunk_size as a per-sample setting. A batch of B
        # samples projects up to B times as many target rows per lm_head call.
        effective_chunk_size = chunk_size * len(positions_by_sample)

        base_model = getattr(self.model, "model", None)
        if (
            base_model is not None
            and hasattr(base_model, "forward")
            and hasattr(self.model, "get_output_embeddings")
        ):
            attention_mask = kwargs["attention_mask"]
            if attention_mask.ndim == 2:
                kwargs["attention_mask"] = packed_block_attention_mask(
                    attention_mask, kwargs["position_ids"]
                )
            elif attention_mask.ndim != 4:
                raise ValueError("attention_mask must have shape [B,L] or [B,1,L,L]")
            encoding = base_model(**kwargs)
            hidden_states = encoding.last_hidden_state
            lm_head = self.model.get_output_embeddings()
            lm_head_device = lm_head.weight.device
            row_batch = flat_batch.to(hidden_states.device)
            row_positions = flat_positions.to(hidden_states.device)
            rows = hidden_states[row_batch, row_positions].to(lm_head_device)
            vocab_size = int(lm_head.weight.shape[0])
            flat_probabilities = torch.empty(
                rows.shape[0],
                vocab_size,
                dtype=torch.float32,
                device=lm_head_device,
            )
            for start in range(0, rows.shape[0], effective_chunk_size):
                stop = min(start + effective_chunk_size, rows.shape[0])
                flat_probabilities[start:stop].copy_(
                    torch.softmax(lm_head(rows[start:stop]).float(), dim=-1)
                )
            del encoding, hidden_states, rows
        else:
            output = self.model(**kwargs)
            logits = output.logits
            row_batch = flat_batch.to(logits.device)
            row_positions = flat_positions.to(logits.device)
            rows = logits[row_batch, row_positions]
            flat_probabilities = torch.empty(
                rows.shape[0],
                rows.shape[-1],
                dtype=torch.float32,
                device=rows.device,
            )
            for start in range(0, rows.shape[0], effective_chunk_size):
                stop = min(start + effective_chunk_size, rows.shape[0])
                flat_probabilities[start:stop].copy_(
                    torch.softmax(rows[start:stop].float(), dim=-1)
                )
            del output, logits, rows

        probabilities = []
        offset = 0
        for count in counts:
            probabilities.append(flat_probabilities[offset : offset + count])
            offset += count
        return probabilities

    def classify_target(self, batch: Dict, labels: torch.Tensor) -> str:
        target_ids = labels[labels.ne(IGNORE_TOKEN_ID)]
        audio_begin = self.audio_token_offset
        audio_end = audio_begin + AUDIO_TOKEN_COUNT
        audio_count = int(
            ((target_ids >= audio_begin) & (target_ids < audio_end)).sum()
        )
        if audio_count:
            return "TTS"
        if batch.get("audios"):
            return "ASR"
        return "text"

    def prepare_item(
        self,
        batch: Dict,
        num_time_steps: int,
        time_grid: str,
        mask_seed: int,
    ) -> Dict:
        """Prepare the clean sequence and deterministic mask trajectory.

        States are constructed by :meth:`state_at` just before each forward;
        keeping all ``num_time_steps`` copies here would needlessly consume
        CPU memory for long trajectories.
        """
        input_ids = batch["input_ids"].detach().cpu().long()
        labels = batch["labels"].detach().cpu().long()
        if input_ids.ndim != 1 or labels.ndim != 1:
            raise ValueError("expected one packed dataset item")
        target_mask = labels.ne(IGNORE_TOKEN_ID)
        target_positions = target_mask.nonzero(as_tuple=True)[0]
        # Dream keeps the training shift h[i] -> x[i+1].  In a packed
        # sequence, position 0 starts a new attention block and has no
        # predecessor in that block, even when its physical index is > 0.
        position_ids = batch.get("position_ids")
        if position_ids is not None:
            position_ids = torch.as_tensor(position_ids).view(-1)
            valid_prediction = position_ids[target_positions] > 0
        else:
            valid_prediction = target_positions > 0
        target_positions = target_positions[valid_prediction]
        if target_positions.numel() == 0:
            raise ValueError("no supervised target with a valid prediction row")

        # The dataset item is x_T. Restore all supervised labels to obtain x_0.
        clean_input = input_ids.clone()
        clean_input[target_mask] = labels[target_mask]
        sequence_length = input_ids.numel()
        generator = torch.Generator(device="cpu").manual_seed(mask_seed)
        mask_scores = torch.rand(sequence_length, generator=generator)

        if num_time_steps < 2:
            raise ValueError("num_time_steps must be at least 2")
        if time_grid == "theta":
            theta = torch.linspace(0.0, torch.pi, num_time_steps)
            alpha = torch.cos(theta / 2.0).square()
            # Make the two requested endpoint states exact in float32.
            alpha[0] = 1.0
            alpha[-1] = 0.0
        else:
            alpha = torch.linspace(1.0, 0.0, num_time_steps)
            theta = theta_from_alpha(alpha)

        base_state = {
            key: value
            for key, value in batch.items()
            if key not in {"input_ids", "labels", "images", "image_indices"}
        }
        return {
            "batch": batch,
            "labels": labels,
            "target_mask": target_mask,
            "clean_input": clean_input,
            "mask_scores": mask_scores,
            "target_positions": target_positions,
            "target_ids": labels[target_positions],
            "alpha": alpha,
            "theta": theta,
            "base_state": base_state,
            "prediction_positions": target_positions - 1,
            "mask_seed": mask_seed,
            "task": self.classify_target(batch, labels),
        }

    @staticmethod
    def state_at(prepared: Dict, step: int) -> torch.Tensor:
        """Materialize one masked state on CPU for the next model forward."""
        alpha_value = prepared["alpha"][step]
        state_mask = prepared["target_mask"] & (
            prepared["mask_scores"] >= alpha_value
        )
        state_ids = prepared["clean_input"].clone()
        state_ids[state_mask] = MASK_TOKEN_ID
        return state_ids

    def analyze_batch(
        self,
        batches: Sequence[Dict],
        sample_indices: Sequence[int],
        output_dir: str,
        num_time_steps: int,
        time_grid: str,
        fisher_chunk_size: int,
        c: float,
        mask_seeds: Sequence[int],
        save_distributions: bool,
    ) -> Dict:
        """Analyze a batch while retaining only adjacent-step probabilities.

        The returned interval accumulators are small (``O(num_time_steps)``)
        and are merged by task. No historical probability trajectory is kept.
        """
        if not (
            len(batches) == len(sample_indices) == len(mask_seeds)
        ):
            raise ValueError("analysis batch metadata lengths do not match")
        if not batches:
            return {"num_samples": 0, "samples": [], "by_task": {}}

        prepared_items = [
            self.prepare_item(batch, num_time_steps, time_grid, mask_seed)
            for batch, mask_seed in zip(batches, mask_seeds)
        ]
        # Audio features and packed metadata do not change with diffusion
        # time. Move them once per analysis batch instead of copying them at
        # every step; dynamic input_ids remain inexpensive CPU tensors.
        for item in prepared_items:
            item["base_state"] = {
                key: _move_nested(value, self.model_device)
                for key, value in item["base_state"].items()
            }
        num_intervals = num_time_steps - 1
        by_task = {
            task: _new_task_interval_stats(num_intervals)
            for task in {item["task"] for item in prepared_items}
        }
        sample_interval_sums = [[0.0] * num_intervals for _ in prepared_items]
        sample_path_lengths = [0.0] * len(prepared_items)
        sample_negative_counts = [[0] * num_intervals for _ in prepared_items]
        first_probabilities = None
        previous_probabilities = None
        static_batch = collate_analysis_states(
            [
                dict(item["base_state"], input_ids=item["clean_input"])
                for item in prepared_items
            ]
        )
        static_batch["attention_mask"] = packed_block_attention_mask(
            static_batch["attention_mask"], static_batch["position_ids"]
        )
        for step in range(num_time_steps):
            state_batch = dict(static_batch)
            state_batch["input_ids"] = torch.stack(
                [self.state_at(item, step) for item in prepared_items], dim=0
            )
            step_probabilities = self.target_probabilities_batch(
                state_batch,
                [item["prediction_positions"] for item in prepared_items],
                fisher_chunk_size,
            )
            if step == 0:
                previous_probabilities = step_probabilities
                if save_distributions:
                    first_probabilities = list(step_probabilities)
                del state_batch
                continue

            delta_theta = float(
                prepared_items[0]["theta"][step]
                - prepared_items[0]["theta"][step - 1]
            )
            for item_index, (item, previous, current) in enumerate(
                zip(prepared_items, previous_probabilities, step_probabilities)
            ):
                stats = fisher_interval_statistics(
                    previous,
                    current,
                    delta_theta,
                    c,
                    # Probability softmax chunks control projection peak
                    # memory; larger Fisher row chunks avoid Python-loop
                    # overhead while keeping reduction temporaries bounded.
                    row_chunk_size=max(fisher_chunk_size, 256),
                )
                task_stats = by_task[item["task"]]
                interval_index = step - 1
                task_stats["sequence_interval_sum"][interval_index] += stats[
                    "sequence_interval"
                ]
                task_stats["sequence_interval_squared_sum"][interval_index] += stats[
                    "sequence_interval_squared"
                ]
                task_stats["sequence_interval_negative_count"][interval_index] += int(
                    stats["sequence_interval_negative"]
                )
                task_stats["sequence_distance_sum"][interval_index] += stats[
                    "sequence_distance"
                ]
                task_stats["sequence_distance_squared_sum"][interval_index] += (
                    stats["sequence_distance"] ** 2
                )
                task_stats["local_interval_sum"][interval_index] += stats[
                    "local_interval_sum"
                ]
                task_stats["local_interval_squared_sum"][interval_index] += stats[
                    "local_interval_squared_sum"
                ]
                task_stats["local_token_count"][interval_index] += stats[
                    "target_count"
                ]
                task_stats["local_interval_negative_count"][interval_index] += stats[
                    "local_negative_count"
                ]
                sample_interval_sums[item_index][interval_index] = stats[
                    "sequence_interval"
                ]
                sample_negative_counts[item_index][interval_index] = int(
                    stats["sequence_interval_negative"]
                )
                sample_path_lengths[item_index] += stats["sequence_distance"]

            old_previous = previous_probabilities
            previous_probabilities = step_probabilities
            del old_previous, state_batch

        samples = []
        for item_index, (item, sample_index) in enumerate(
            zip(prepared_items, sample_indices)
        ):
            task_stats = by_task[item["task"]]
            interval_mean = sum(sample_interval_sums[item_index]) / num_intervals
            negative_fraction = sum(sample_negative_counts[item_index]) / num_intervals
            path_length = sample_path_lengths[item_index]
            task_stats["num_samples"] += 1
            task_stats["target_position_count"] += int(
                item["target_positions"].numel()
            )
            task_stats["sample_interval_mean_sum"] += interval_mean
            task_stats["sample_interval_mean_squared_sum"] += interval_mean**2
            task_stats["sample_path_length_sum"] += path_length
            task_stats["sample_path_length_squared_sum"] += path_length**2
            result = {
                "sample_index": int(sample_index),
                "task": item["task"],
                "num_target_positions": int(item["target_positions"].numel()),
                "time_grid": time_grid,
                "num_time_steps": num_time_steps,
                "mask_seed": int(item["mask_seed"]),
                "sequence_interval_mean": interval_mean,
                "sequence_interval_negative_fraction": negative_fraction,
                "fisher_rao_path_length": path_length,
            }
            if save_distributions:
                if first_probabilities is None or previous_probabilities is None:
                    raise RuntimeError("endpoint probabilities were not produced")
                distribution_path = Path(output_dir) / f"fisher_{sample_index:06d}.pt"
                torch.save(
                    {
                        "prob_0": first_probabilities[item_index].cpu(),
                        "prob_T": previous_probabilities[item_index].cpu(),
                        "x_0": self.state_at(item, 0),
                        "x_T": self.state_at(item, num_time_steps - 1),
                        "alpha": item["alpha"],
                        "theta": item["theta"],
                        "target_positions": item["target_positions"],
                        "target_token_ids": item["target_ids"],
                    },
                    distribution_path,
                )
                result["distribution_path"] = str(distribution_path)
            samples.append(result)

        del previous_probabilities, first_probabilities, static_batch
        return {"num_samples": len(samples), "samples": samples, "by_task": by_task}


def build_training_dataset(analyzer: FisherAnalyzer, args) -> Qwen2Dataset:
    dataset_output_dir = getattr(args, "dataset_output_dir", args.output_dir)
    dataset = Qwen2Dataset(
        args.dataset_name,
        analyzer.tokenizer,
        image_size=args.image_size,
        image_token_length=args.image_token_length,
        max_padding_length=args.model_max_length,
        variable_length=False,
        output_dir=dataset_output_dir,
        shift_token=False,
        create_position_ids=True,
        create_attention_mask=True,
        create_attention_mask_2d=False,
        create_loss_mask=False,
        max_num_frame=args.max_num_frame,
        max_fps=args.max_fps,
        reset_position_ids=getattr(args, "reset_position_ids", True),
        reset_attention_mask=getattr(args, "reset_attention_mask", True),
        min_patch_grid=args.min_patch_grid,
        max_patch_grid=args.max_patch_grid,
        process_type="dynamic",
        normalize_type="imagenet",
        seed=args.seed,
        cross_dataset_joint=False,
        dataset_joint=True,
        audio_tokenizer_type=args.audio_tokenizer_type,
        audio_tokenizer_path=args.audio_tokenizer_path,
        image_tokenizer_path=args.image_tokenizer_path,
        use_megatron=False,
    )
    dataset.processor["audio"].audio_tokenizer = analyzer.audio_tokenizer
    dataset.processor["image"] = analyzer.image_processor
    return dataset


def _safe_mean(sum_value: float, count: int) -> float:
    return float(sum_value) / max(int(count), 1)


def _safe_std(sum_value: float, squared_sum: float, count: int) -> float:
    count = int(count)
    if count < 1:
        return 0.0
    mean = float(sum_value) / count
    return max(float(squared_sum) / count - mean * mean, 0.0) ** 0.5


def finalize_rank_summary(
    rank_summary: Dict,
    num_time_steps: int,
    time_grid: str,
    c: float,
    requested_num_samples: int,
) -> Dict:
    """Convert raw sums into compact, task-separated interval statistics."""
    if time_grid == "theta":
        theta = torch.linspace(0.0, torch.pi, num_time_steps)
        alpha = torch.cos(theta / 2.0).square()
        alpha[0], alpha[-1] = 1.0, 0.0
    else:
        alpha = torch.linspace(1.0, 0.0, num_time_steps)
        theta = theta_from_alpha(alpha)
    intervals_meta = [
        {
            "interval_index": index,
            "step": index + 1,
            "alpha_start": float(alpha[index]),
            "alpha_end": float(alpha[index + 1]),
            "theta_start": float(theta[index]),
            "theta_end": float(theta[index + 1]),
            "delta_theta": float(theta[index + 1] - theta[index]),
            "time_metric_per_target": c**2
            * float(theta[index + 1] - theta[index]) ** 2,
        }
        for index in range(num_time_steps - 1)
    ]
    output = {
        "num_samples": min(int(rank_summary.get("num_samples", 0)), requested_num_samples),
        "requested_num_samples": requested_num_samples,
        "num_time_steps": num_time_steps,
        "num_intervals": num_time_steps - 1,
        "time_grid": time_grid,
        "interval_c": c,
        "interval_definition": (
            "sum_j[(c*delta_theta)^2 - dF_j^2], with j over supervised "
            "prediction positions only"
        ),
        "by_task": {},
    }
    for task, stats in sorted(rank_summary.get("by_task", {}).items()):
        num_samples = int(stats["num_samples"])
        intervals = []
        for index, metadata in enumerate(intervals_meta):
            count = num_samples
            token_count = int(stats["local_token_count"][index])
            sequence_mean = _safe_mean(
                stats["sequence_interval_sum"][index], count
            )
            distance_mean = _safe_mean(
                stats["sequence_distance_sum"][index], count
            )
            local_mean = _safe_mean(
                stats["local_interval_sum"][index], token_count
            )
            interval = dict(metadata)
            interval.update(
                {
                    "sample_count": count,
                    "target_position_count": token_count,
                    "sequence_interval_mean": sequence_mean,
                    "sequence_interval_std": _safe_std(
                        stats["sequence_interval_sum"][index],
                        stats["sequence_interval_squared_sum"][index],
                        count,
                    ),
                    "sequence_interval_negative_fraction": _safe_mean(
                        stats["sequence_interval_negative_count"][index], count
                    ),
                    "sequence_fisher_distance_mean": distance_mean,
                    "sequence_fisher_distance_std": _safe_std(
                        stats["sequence_distance_sum"][index],
                        stats["sequence_distance_squared_sum"][index],
                        count,
                    ),
                    "local_interval_mean": local_mean,
                    "local_interval_std": _safe_std(
                        stats["local_interval_sum"][index],
                        stats["local_interval_squared_sum"][index],
                        token_count,
                    ),
                    "local_interval_negative_fraction": _safe_mean(
                        stats["local_interval_negative_count"][index], token_count
                    ),
                }
            )
            intervals.append(interval)
        output["by_task"][task] = {
            "num_samples": num_samples,
            "target_position_count": int(stats["target_position_count"]),
            "target_positions_mean": _safe_mean(
                stats["target_position_count"], num_samples
            ),
            "sequence_interval_mean": _safe_mean(
                sum(stats["sequence_interval_sum"]), num_samples * (num_time_steps - 1)
            ),
            "sequence_interval_negative_fraction": _safe_mean(
                sum(stats["sequence_interval_negative_count"]),
                num_samples * (num_time_steps - 1),
            ),
            "sample_sequence_interval_mean": _safe_mean(
                stats["sample_interval_mean_sum"], num_samples
            ),
            "sample_sequence_interval_std": _safe_std(
                stats["sample_interval_mean_sum"],
                stats["sample_interval_mean_squared_sum"],
                num_samples,
            ),
            "fisher_rao_path_length_mean": _safe_mean(
                stats["sample_path_length_sum"], num_samples
            ),
            "fisher_rao_path_length_std": _safe_std(
                stats["sample_path_length_sum"],
                stats["sample_path_length_squared_sum"],
                num_samples,
            ),
            "intervals": intervals,
        }
    return output


def _process_rank_results(
    analyzer: FisherAnalyzer,
    args,
    context: Dict[str, int],
    local_output_dir: Path,
) -> Dict:
    """Process this rank's strided records and persist scalar accumulators."""
    dataset = build_training_dataset(analyzer, args)
    rank = context["rank"]
    world_size = context["world_size"]
    raw_item_limit = args.max_dataset_items
    if raw_item_limit is None:
        max_items = len(dataset)
    else:
        # Qwen2Dataset emits a packed item only after its source-local buffer
        # overflows. Under torchrun, a global limit of N would give each rank
        # only N/world_size records and can leave every buffer unfinished.
        # Treat an explicit limit as a per-rank scan budget, while still
        # clamping the actual global raw range to the dataset length.
        if world_size > 1:
            raw_item_limit *= world_size
        max_items = min(raw_item_limit, len(dataset))
    analysis_batch_size = max(
        int(getattr(args, "analysis_batch_size", 1)), 1
    )
    can_batch = analysis_batch_size > 1 and hasattr(
        analyzer, "analyze_batch"
    )
    # The requested size is an upper bound. If a real batch does not fit, the
    # rank remembers the smaller size for subsequent groups instead of paying
    # for the same failed forward on every iteration.
    effective_batch_size = analysis_batch_size
    LOGGER.info(
        "[rank %d] analysis_batch_size=%d",
        rank,
        analysis_batch_size,
    )
    # Start with an even quota. If malformed/empty records make the global
    # count too small, all ranks take another quota-sized pass below.
    local_quota = args.num_samples // world_size + int(
        rank < (args.num_samples % world_size)
    )
    rank_summary = {"rank": rank, "num_samples": 0, "by_task": {}}
    next_index = rank
    next_sample_index = 0

    def log_result(result: Dict) -> None:
        LOGGER.info(
            "[rank %d] sample=%d task=%s targets=%d path=%f interval_mean=%f",
            rank,
            result["sample_index"],
            result["task"],
            result["num_target_positions"],
            result["fisher_rao_path_length"],
            result["sequence_interval_mean"],
        )

    def analyze_pending(pending: Sequence[tuple[int, Dict]]) -> None:
        nonlocal effective_batch_size, next_sample_index
        if not pending:
            return
        def analyze_group(group: Sequence[tuple[int, Dict]]) -> None:
            """Analyze a group, bisecting it when batching is not viable."""
            nonlocal effective_batch_size, next_sample_index
            if not group:
                return

            if can_batch and len(group) > 1:
                data_indices = [data_index for data_index, _ in group]
                try:
                    batch_result = analyzer.analyze_batch(
                        batches=[batch for _, batch in group],
                        sample_indices=list(range(next_sample_index, next_sample_index + len(group))),
                        output_dir=str(local_output_dir),
                        num_time_steps=args.num_time_steps,
                        time_grid=args.time_grid,
                        fisher_chunk_size=args.fisher_chunk_size,
                        c=args.c,
                        mask_seeds=[args.seed + index for index in data_indices],
                        save_distributions=args.save_distributions,
                    )
                    if len(batch_result["samples"]) != len(group):
                        raise ValueError(
                            "analyze_batch returned a different number of results"
                        )
                    _merge_rank_summaries(rank_summary, batch_result)
                    for result in batch_result["samples"]:
                        log_result(result)
                    next_sample_index += len(group)
                    return
                except (RuntimeError, ValueError, IndexError) as error:
                    # A batch can exceed available memory or contain records
                    # with incompatible metadata. Bisect it so compatible
                    # samples still run together, and remember OOM-derived
                    # capacity for later groups.
                    LOGGER.warning(
                        "[rank %d] Batch size %d failed (%s); splitting the "
                        "group",
                        rank,
                        len(group),
                        error,
                    )
                    if isinstance(error, RuntimeError) and "out of memory" in str(error).lower():
                        effective_batch_size = max(
                            1, min(effective_batch_size, len(group) // 2)
                        )
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    LOGGER.warning(
                        "[rank %d] Effective analysis batch size is now %d",
                        rank,
                        effective_batch_size,
                    )
                    midpoint = len(group) // 2
                    analyze_group(group[:midpoint])
                    analyze_group(group[midpoint:])
                    return

            for data_index, batch in group:
                sample_index = next_sample_index
                try:
                    batch_result = analyzer.analyze_batch(
                        batches=[batch],
                        sample_indices=[sample_index],
                        output_dir=str(local_output_dir),
                        num_time_steps=args.num_time_steps,
                        time_grid=args.time_grid,
                        fisher_chunk_size=args.fisher_chunk_size,
                        c=args.c,
                        mask_seeds=[args.seed + data_index],
                        save_distributions=args.save_distributions,
                    )
                    result = batch_result["samples"][0]
                except (RuntimeError, ValueError, IndexError) as error:
                    LOGGER.warning(
                        "[rank %d] Skipping raw dataset item %d: %s",
                        rank,
                        data_index,
                        error,
                    )
                    if isinstance(error, RuntimeError) and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                _merge_rank_summaries(rank_summary, batch_result)
                next_sample_index += 1
                log_result(result)

        for start in range(0, len(pending), max(effective_batch_size, 1)):
            analyze_group(pending[start : start + max(effective_batch_size, 1)])

    with mask_all_supervised_positions():
        while True:
            pending = []
            while next_index < max_items and rank_summary["num_samples"] < local_quota:
                data_index = next_index
                next_index += world_size
                batch = dataset[data_index]
                # Packed datasets may return {} while their source-local buffer
                # is filling; continue consuming raw records until a packed
                # item is available.
                if "input_ids" not in batch or "labels" not in batch:
                    continue
                pending.append((data_index, batch))
                if (
                    len(pending) >= analysis_batch_size
                    or rank_summary["num_samples"] + len(pending) >= local_quota
                    or next_index >= max_items
                ):
                    analyze_pending(pending)
                    pending = []

            if pending:
                analyze_pending(pending)

            exhausted = next_index >= max_items
            if world_size <= 1 or not torch.distributed.is_initialized():
                break

            counts = [None] * world_size
            exhausted_flags = [None] * world_size
            torch.distributed.all_gather_object(counts, rank_summary["num_samples"])
            torch.distributed.all_gather_object(exhausted_flags, exhausted)
            total_count = sum(int(count) for count in counts)
            if total_count >= args.num_samples or all(exhausted_flags):
                break
            remaining = args.num_samples - total_count
            active_ranks = [
                index for index, exhausted_flag in enumerate(exhausted_flags)
                if not exhausted_flag
            ]
            if remaining <= 0 or rank not in active_ranks:
                local_quota = rank_summary["num_samples"]
            else:
                active_position = active_ranks.index(rank)
                local_quota += remaining // len(active_ranks) + int(
                    active_position < (remaining % len(active_ranks))
                )

    rank_summary["rank"] = rank
    with (local_output_dir / "rank_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(rank_summary, handle, ensure_ascii=False)
    LOGGER.info(
        "[rank %d] Wrote scalar summary for %d samples to %s",
        rank,
        rank_summary["num_samples"],
        local_output_dir,
    )
    return rank_summary


def run_analysis(analyzer: FisherAnalyzer, args) -> Dict:
    """Run local analysis and merge only scalar rank summaries on rank 0."""
    context = {
        "rank": int(getattr(args, "rank", 0)),
        "world_size": int(getattr(args, "world_size", 1)),
        "local_rank": int(getattr(args, "local_rank", 0)),
    }
    output_dir = Path(args.output_dir)
    local_output_dir = Path(
        getattr(
            args,
            "local_output_dir",
            output_dir
            / f"rank_{context['rank']:05d}"
            if context["world_size"] > 1
            else output_dir,
        )
    )
    args.dataset_output_dir = str(local_output_dir)
    local_summary = _process_rank_results(
        analyzer, args, context, local_output_dir
    )

    if context["world_size"] <= 1 or not torch.distributed.is_initialized():
        merged_summary = local_summary
    else:
        # Probability tensors remain on the rank that produced them. Only the
        # small per-interval scalar files are read by rank 0 after the barrier.
        torch.distributed.barrier()
        if not is_main_process(context):
            return local_summary
        merged_summary = {
            "num_samples": 0,
            "by_task": {},
        }
        for rank in range(context["world_size"]):
            rank_dir = output_dir / f"rank_{rank:05d}"
            summary_path = rank_dir / "rank_summary.json"
            if not summary_path.exists():
                raise RuntimeError(f"Missing rank summary file: {summary_path}")
            with summary_path.open("r", encoding="utf-8") as handle:
                _merge_rank_summaries(merged_summary, json.load(handle))

    if not merged_summary.get("num_samples"):
        raise RuntimeError(
            "No complete training item was produced; increase --max_dataset_items "
            "or inspect the dataset error log."
        )
    summary = finalize_rank_summary(
        merged_summary,
        num_time_steps=args.num_time_steps,
        time_grid=args.time_grid,
        c=args.c,
        requested_num_samples=args.num_samples,
    )
    with (output_dir / "fisher_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    LOGGER.info("Wrote scalar Fisher summary for %d samples to %s", summary["num_samples"], output_dir)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fisher-Rao spacetime diagnostics on training-like TTS/ASR data."
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--dataset_name",
        default="/share/users/zouwei/data/finetune.yaml",
        help="Training YAML consumed by Qwen2Dataset.",
    )
    parser.add_argument(
        "--audio_tokenizer_path",
        default="/share/users/zouwei/models/THUDM/glm-4-voice-tokenizer",
    )
    parser.add_argument(
        "--audio_tokenizer_type", default="sensevoice_glm4voice"
    )
    parser.add_argument("--image_tokenizer_path", default="showlab/magvitv2")
    parser.add_argument(
        "--load_image_tokenizer",
        action="store_true",
        help="Load MAGVIT weights when the training YAML contains images.",
    )
    parser.add_argument(
        "--flow_path",
        default=None,
        help="Optional TTS decoder; omitted because Fisher analysis only encodes audio.",
    )
    parser.add_argument("--device_map", default="cuda:0")
    parser.add_argument(
        "--torch_dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=128,
        help="Global number of valid sample packs across all ranks.",
    )
    parser.add_argument(
        "--max_dataset_items",
        type=int,
        default=None,
        help=(
            "Raw scan budget. Single-GPU uses a global index limit; under "
            "torchrun this many records are assigned to each rank so packed "
            "buffers can fill."
        ),
    )
    parser.add_argument("--model_max_length", type=int, default=3072)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--image_token_length", type=int, default=1025)
    parser.add_argument("--max_num_frame", type=int, default=16)
    parser.add_argument("--max_fps", type=int, default=1)
    parser.add_argument("--min_patch_grid", type=int, default=1)
    parser.add_argument("--max_patch_grid", type=int, default=12)
    parser.add_argument(
        "--reset_position_ids",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match the training launcher: restart positions for packed records.",
    )
    parser.add_argument(
        "--reset_attention_mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match the training launcher: block attention across packed records.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num_time_steps",
        type=int,
        default=100,
        help="Number of online masked states, including alpha=1 and alpha=0.",
    )
    parser.add_argument(
        "--time_grid",
        choices=("theta", "alpha"),
        default="theta",
        help="Uniform Fisher-angle or uniform retention-rate sampling.",
    )
    parser.add_argument(
        "--fisher_chunk_size",
        type=int,
        default=256,
        help="Target rows per probability softmax chunk.",
    )
    parser.add_argument(
        "--analysis_batch_size",
        type=int,
        default=8,
        help=(
            "Number of packed samples processed together by each rank. "
            "Larger values increase per-GPU utilization and memory; reduce "
            "this value if a batch runs out of memory."
        ),
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="Reference causal speed for interval=c^2 dt^2-ds^2.",
    )
    parser.add_argument(
        "--save_distributions",
        action="store_true",
        help=(
            "Save endpoint [target_position,vocab] float32 prob_0/prob_T "
            "tensors; this intentionally retains one extra endpoint copy."
        ),
    )
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num_samples must be positive")
    if args.max_dataset_items is not None and args.max_dataset_items < 1:
        parser.error("--max_dataset_items must be positive")
    if args.model_max_length < 2:
        parser.error("--model_max_length must be at least 2")
    if args.num_time_steps < 2:
        parser.error("--num_time_steps must be at least 2")
    if args.fisher_chunk_size < 1:
        parser.error("--fisher_chunk_size must be positive")
    if args.analysis_batch_size < 1:
        parser.error("--analysis_batch_size must be positive")
    if args.c <= 0:
        parser.error("--c must be positive")
    return args


def main() -> None:
    args = parse_args()
    # export LD_LIBRARY_PATH=/share/users/zouwei/miniconda3/envs/dev/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    context = distributed_context()
    if context["world_size"] < 1:
        raise RuntimeError("WORLD_SIZE must be positive")
    if context["world_size"] > 1 and torch.cuda.is_available():
        if context["local_rank"] >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={context['local_rank']} is outside the visible "
                f"CUDA device range 0..{torch.cuda.device_count() - 1}"
            )
        torch.cuda.set_device(context["local_rank"])
    init_distributed(context)
    args.rank = context["rank"]
    args.local_rank = context["local_rank"]
    args.world_size = context["world_size"]
    output_dir = Path(args.output_dir)
    if context["world_size"] > 1:
        args.local_output_dir = str(
            output_dir / f"rank_{context['rank']:05d}"
        )
    else:
        args.local_output_dir = str(output_dir)
    if str(args.device_map).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not visible to this process, but --device_map requests "
            f"{args.device_map}. Run on a GPU allocation (for example an H100) "
            "or pass --device_map cpu for a small test model."
        )
    if context["world_size"] > 1 and torch.cuda.is_available():
        # A per-rank full model is data parallel. "auto" would otherwise let
        # Accelerate shard every rank's model across all visible GPUs.
        args.device_map = f"cuda:{context['local_rank']}"
    set_seed(args.seed)
    os.makedirs(args.local_output_dir, exist_ok=True)
    if context["rank"] == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    if args.torch_dtype == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    else:
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[args.torch_dtype]
    analyzer = FisherAnalyzer(
        model_name_or_path=args.model_name_or_path,
        audio_tokenizer_path=args.audio_tokenizer_path,
        audio_tokenizer_type=args.audio_tokenizer_type,
        image_tokenizer_path=args.image_tokenizer_path,
        flow_path=args.flow_path,
        device_map=args.device_map,
        torch_dtype=dtype,
        load_image_tokenizer=args.load_image_tokenizer,
        audio_tokenizer_rank=(
            context["local_rank"] if torch.cuda.is_available() else None
        ),
        image_tokenizer_rank=(
            context["local_rank"] if torch.cuda.is_available() else None
        ),
    )
    try:
        run_analysis(analyzer, args)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
