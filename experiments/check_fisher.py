"""Measure Fisher-Rao spacetime diagnostics on the training data.

The script uses the same Qwen2Dataset preprocessing as training. For every
packed item it constructs a nested family of masked states, evaluates the
model only at supervised target positions, and records endpoint/intermediate
Fisher-Rao distances, theta-clock velocities, per-position-summed Minkowski
intervals, path length, geodesicity, and a numerical triangle-inequality check.

This is an analysis script, not a generation script. The endpoint distance is
only a boundary measurement; the intermediate path makes the spacetime
diagnostics falsifiable.
"""

import argparse
import contextlib
import json
import logging
import os
import random
import sys
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
EPS = 1e-12


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


def sequence_fisher_distance(token_distances: torch.Tensor) -> torch.Tensor:
    """Fisher distance on the product (mean-field) simplex."""
    return torch.sqrt(token_distances.float().square().sum()).clamp_min(0.0)


def aggregate_interval(
    token_distances: torch.Tensor,
    delta_time: float,
    c: float,
) -> torch.Tensor:
    """Sum one 1+1 interval per prediction position.

    The sequence spatial term is ``sum_j ds_j^2``. Its matching temporal
    term is ``sum_j c^2 dt^2``, not one shared ``c^2 dt^2`` term.
    """
    token_distances = token_distances.float()
    return token_distances.numel() * (c * delta_time) ** 2 - token_distances.square().sum()


def factorized_joint_distance(
    first: torch.Tensor, second: torch.Tensor, eps: float = EPS
) -> torch.Tensor:
    """Distance of the factorized joint distribution, reported for reference."""
    first = first.float().clamp_min(0)
    second = second.float().clamp_min(0)
    first = first / first.sum(dim=-1, keepdim=True).clamp_min(eps)
    second = second / second.sum(dim=-1, keepdim=True).clamp_min(eps)
    affinity = (first.sqrt() * second.sqrt()).sum(dim=-1).clamp_min(eps)
    joint_affinity = torch.exp(torch.log(affinity).sum()).clamp(0.0, 1.0)
    return 2.0 * torch.acos(joint_affinity)


def theta_from_alpha(alpha: torch.Tensor) -> torch.Tensor:
    """Fisher-angle clock for a masked channel with retention alpha."""
    return 2.0 * torch.acos(alpha.clamp(0.0, 1.0).sqrt())


def _as_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


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
    ):
        config = DreamConfig.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        if audio_tokenizer_type is not None:
            tokenizer = update_tokenizer(tokenizer, audio_tokenizer_type)
        tokenizer.add_tokens(
            [f"<|image_{index}|>" for index in range(8192)],
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
        self.image_processor.image_tokenizer.rank = 0 if torch.cuda.is_available() else None
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
    def target_probabilities(
        self,
        state: Dict,
        prediction_positions: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        """Run one encoding forward and return distributions at target rows.

        Dream's public forward materializes a vocabulary projection for every
        sequence row. The analysis only needs the causal rows immediately
        before supervised labels, so use the base transformer output and apply
        ``lm_head`` to those rows in chunks. The fallback keeps this helper
        usable with small test doubles that expose only ``forward``.
        """
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        prediction_positions = prediction_positions.long()
        kwargs = self.forward_kwargs(state)
        if prediction_positions.numel() == 0:
            raise ValueError("no prediction positions")
        if (
            prediction_positions.min() < 0
            or prediction_positions.max() >= kwargs["input_ids"].shape[1]
        ):
            raise IndexError("prediction position is outside the input sequence")

        base_model = getattr(self.model, "model", None)
        if (
            base_model is not None
            and hasattr(base_model, "forward")
            and hasattr(self.model, "get_output_embeddings")
        ):
            # Match DreamModel.forward's packed-sequence block mask before
            # calling DreamBaseModel directly.
            attention_mask = kwargs["attention_mask"]
            position_ids = kwargs["position_ids"]
            is_new = position_ids == 0
            segment_id = torch.cumsum(is_new.long(), dim=1) - 1
            block_mask = (
                segment_id.unsqueeze(1) == segment_id.unsqueeze(2)
            ).long()
            kwargs["attention_mask"] = block_mask * attention_mask.unsqueeze(-1)
            encoding = base_model(**kwargs)
            hidden_states = encoding.last_hidden_state[0]
            lm_head = self.model.get_output_embeddings()
            lm_head_device = lm_head.weight.device
            probabilities = []
            for start in range(0, prediction_positions.numel(), chunk_size):
                positions = prediction_positions[start : start + chunk_size].to(
                    hidden_states.device
                )
                rows = hidden_states.index_select(0, positions).to(lm_head_device)
                probabilities.append(
                    torch.softmax(lm_head(rows).float(), dim=-1).cpu()
                )
            del encoding, hidden_states
            return torch.cat(probabilities, dim=0)

        # Small fake models used by static/CPU tests may not expose the
        # DreamBaseModel split. Their public logits path is still correct.
        output = self.model(**kwargs)
        logits = output.logits[0]
        chunks = []
        for start in range(0, prediction_positions.numel(), chunk_size):
            positions = prediction_positions[start : start + chunk_size].to(
                logits.device
            )
            rows = logits.index_select(0, positions).float()
            chunks.append(torch.softmax(rows, dim=-1).cpu())
        del output, logits
        return torch.cat(chunks, dim=0)

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

    def analyze_item(
        self,
        batch: Dict,
        sample_index: int,
        output_dir: str,
        num_time_steps: int,
        time_grid: str,
        fisher_chunk_size: int,
        c: float,
        mask_seed: int,
        save_distributions: bool,
    ) -> Dict:
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

        masks = []
        states = []
        for alpha_value in alpha:
            state_mask = target_mask & (mask_scores >= alpha_value)
            state_ids = clean_input.clone()
            state_ids[state_mask] = MASK_TOKEN_ID
            masks.append(state_mask)
            states.append(state_ids)

        base_state = {
            key: value
            for key, value in batch.items()
            if key not in {"input_ids", "labels", "images", "image_indices"}
        }
        prediction_positions = target_positions - 1
        probabilities = []
        state_nll = []
        target_ids = labels[target_positions]
        for state_ids in states:
            probability = self.target_probabilities(
                dict(base_state, input_ids=state_ids),
                prediction_positions,
                fisher_chunk_size,
            )
            probabilities.append(probability)
            target_probability = probability.gather(1, target_ids[:, None]).squeeze(1)
            state_nll.append(float((-target_probability.clamp_min(EPS).log()).mean()))

        interval_rows = []
        token_distances_by_step = []
        for step in range(1, num_time_steps):
            token_distances = fisher_rao_distance(
                probabilities[step - 1], probabilities[step]
            )
            token_distances_by_step.append(token_distances)
            delta_theta = float(theta[step] - theta[step - 1])
            active_target = (masks[step - 1] | masks[step])[target_positions]
            active_distances = (
                token_distances[active_target]
                if active_target.any()
                else token_distances
            )
            ds_sequence = sequence_fisher_distance(token_distances)
            ds_active = sequence_fisher_distance(active_distances)
            target_count = token_distances.numel()
            active_count = int(active_target.sum())
            active_metric_count = active_distances.numel()
            active_rms_count = max(active_metric_count, 1)
            sequence_distance_rms = _as_float(ds_sequence) / max(
                target_count, 1
            ) ** 0.5
            active_distance_rms = _as_float(ds_active) / active_rms_count**0.5
            local_speed = token_distances / max(delta_theta, EPS)
            sequence_speed = _as_float(ds_sequence) / max(delta_theta, EPS)
            local_interval = c**2 * delta_theta**2 - token_distances.square()
            sequence_interval = aggregate_interval(
                token_distances, delta_theta, c
            )
            active_interval = aggregate_interval(
                active_distances, delta_theta, c
            )
            sequence_interval_value = _as_float(sequence_interval)
            active_interval_value = _as_float(active_interval)
            interval_rows.append(
                {
                    "step": step,
                    "alpha_start": float(alpha[step - 1]),
                    "alpha_end": float(alpha[step]),
                    "theta_start": float(theta[step - 1]),
                    "theta_end": float(theta[step]),
                    "delta_theta": delta_theta,
                    "masked_target_count": int(masks[step].sum()),
                    "active_target_count": int(active_target.sum()),
                    "fisher_distance_sequence": _as_float(ds_sequence),
                    "fisher_distance_active": _as_float(ds_active),
                    # RMS values are length diagnostics only. The raw
                    # sequence distance above remains the geometric quantity.
                    "fisher_distance_sequence_rms": sequence_distance_rms,
                    "fisher_distance_active_rms": active_distance_rms,
                    "local_speed_max": _as_float(local_speed.max()),
                    "sequence_speed": sequence_speed,
                    "sequence_speed_rms": sequence_distance_rms
                    / max(delta_theta, EPS),
                    "local_interval_min": _as_float(
                        local_interval.min()
                    ),
                    "local_interval_negative_fraction": float(
                        (local_interval < 0).float().mean()
                    ),
                    # The sequence interval is the sum of per-position
                    # intervals, so its temporal term scales with N.
                    "sequence_time_metric": target_count
                    * (c * delta_theta) ** 2,
                    "sequence_interval": sequence_interval_value,
                    "sequence_interval_rms": sequence_interval_value
                    / max(target_count, 1),
                    "active_time_metric": active_metric_count
                    * (c * delta_theta) ** 2,
                    "active_metric_count": active_metric_count,
                    "active_interval": active_interval_value,
                    "active_interval_rms": active_interval_value
                    / max(active_metric_count, 1),
                }
            )

        endpoint_token_distances = fisher_rao_distance(
            probabilities[0], probabilities[-1]
        )
        endpoint_distance = sequence_fisher_distance(endpoint_token_distances)
        endpoint_joint_distance = factorized_joint_distance(
            probabilities[0], probabilities[-1]
        )
        path_length = sum(row["fisher_distance_sequence"] for row in interval_rows)
        endpoint_distance_value = _as_float(endpoint_distance)
        target_count = endpoint_token_distances.numel()
        endpoint_interval = aggregate_interval(
            endpoint_token_distances,
            float(theta[-1] - theta[0]),
            c,
        )
        endpoint_interval_value = _as_float(endpoint_interval)

        triangle_violations = []
        for step in range(num_time_steps - 2):
            d02 = sequence_fisher_distance(
                fisher_rao_distance(probabilities[step], probabilities[step + 2])
            )
            d01 = sequence_fisher_distance(token_distances_by_step[step])
            d12 = sequence_fisher_distance(token_distances_by_step[step + 1])
            triangle_violations.append(_as_float(d02 - d01 - d12))

        sequence_intervals = [row["sequence_interval"] for row in interval_rows]
        sequence_interval_rms_values = [
            row["sequence_interval_rms"] for row in interval_rows
        ]
        active_intervals = [row["active_interval"] for row in interval_rows]
        active_interval_rms_values = [
            row["active_interval_rms"] for row in interval_rows
        ]
        local_speeds = [row["local_speed_max"] for row in interval_rows]
        required_c = max(
            (row["sequence_speed_rms"] for row in interval_rows),
            default=0.0,
        )
        required_c_raw = max(
            (row["sequence_speed"] for row in interval_rows),
            default=0.0,
        )
        sequence_speed_rms_max = max(
            (row["sequence_speed_rms"] for row in interval_rows),
            default=0.0,
        )
        result = {
            "sample_index": sample_index,
            "task": self.classify_target(batch, labels),
            "num_target_positions": int(target_positions.numel()),
            "target_positions": target_positions.tolist(),
            "target_token_ids": target_ids.tolist(),
            "time_grid": time_grid,
            "num_time_steps": num_time_steps,
            "mask_seed": mask_seed,
            "fisher_clock_endpoint": float(theta[-1] - theta[0]),
            "fisher_rao_distance": endpoint_distance_value,
            "fisher_rao_sequence_distance": endpoint_distance_value,
            "fisher_rao_sequence_distance_rms": endpoint_distance_value
            / max(target_positions.numel(), 1) ** 0.5,
            "fisher_rao_joint_distance": _as_float(endpoint_joint_distance),
            "endpoint_time_metric": target_count
            * (c * float(theta[-1] - theta[0])) ** 2,
            "endpoint_interval": endpoint_interval_value,
            "endpoint_interval_rms": endpoint_interval_value
            / max(target_count, 1),
            "fisher_rao_path_length": path_length,
            "geodesicity_ratio": path_length / max(endpoint_distance_value, EPS),
            "interval_c": c,
            "sequence_interval_min": min(sequence_intervals),
            "sequence_interval_rms_min": min(sequence_interval_rms_values),
            "sequence_interval_negative_fraction": sum(
                value < 0 for value in sequence_intervals
            )
            / len(sequence_intervals),
            "required_sequence_c_max": required_c,
            "required_sequence_c_raw_max": required_c_raw,
            "sequence_speed_rms_max": sequence_speed_rms_max,
            "local_speed_max": max(local_speeds),
            "active_interval_min": min(active_intervals),
            "active_interval_rms_min": min(active_interval_rms_values),
            "local_interval_min": min(
                row["local_interval_min"] for row in interval_rows
            ),
            "local_interval_negative_fraction_mean": sum(
                row["local_interval_negative_fraction"]
                for row in interval_rows
            )
            / len(interval_rows),
            "triangle_violation_max": max(triangle_violations, default=0.0),
            "triangle_violation_count": sum(
                value > 1e-5 for value in triangle_violations
            ),
            "state_nll": state_nll,
            "intervals": interval_rows,
        }

        if save_distributions:
            distribution_path = Path(output_dir) / f"fisher_{sample_index:06d}.pt"
            torch.save(
                {
                    # Only endpoint distributions are persisted. Intermediate
                    # states are used for local interval/path diagnostics but
                    # are intentionally not duplicated on disk.
                    "prob_0": probabilities[0],
                    "prob_T": probabilities[-1],
                    "x_0": states[0],
                    "x_T": states[-1],
                    "alpha": alpha,
                    "theta": theta,
                    "target_positions": target_positions,
                    "target_token_ids": target_ids,
                },
                distribution_path,
            )
            result["distribution_path"] = str(distribution_path)
        return result


def build_training_dataset(analyzer: FisherAnalyzer, args) -> Qwen2Dataset:
    dataset = Qwen2Dataset(
        args.dataset_name,
        analyzer.tokenizer,
        image_size=args.image_size,
        image_token_length=args.image_token_length,
        max_padding_length=args.model_max_length,
        variable_length=False,
        output_dir=args.output_dir,
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


def summarize_results(results: Sequence[Dict]) -> Dict:
    summary = {"num_samples": len(results), "by_task": {}}
    for task in sorted({result["task"] for result in results}):
        subset = [result for result in results if result["task"] == task]

        def mean(key: str) -> float:
            return sum(float(item[key]) for item in subset) / len(subset)

        summary["by_task"][task] = {
            "num_samples": len(subset),
            "fisher_rao_distance_mean": mean("fisher_rao_distance"),
            "fisher_rao_sequence_distance_rms_mean": mean(
                "fisher_rao_sequence_distance_rms"
            ),
            "fisher_rao_joint_distance_mean": mean("fisher_rao_joint_distance"),
            "endpoint_interval_mean": mean("endpoint_interval"),
            "endpoint_interval_rms_mean": mean("endpoint_interval_rms"),
            "fisher_rao_path_length_mean": mean("fisher_rao_path_length"),
            "geodesicity_ratio_mean": mean("geodesicity_ratio"),
            "sequence_interval_min_mean": mean("sequence_interval_min"),
            "sequence_interval_rms_min_mean": mean(
                "sequence_interval_rms_min"
            ),
            "sequence_interval_negative_fraction_mean": mean(
                "sequence_interval_negative_fraction"
            ),
            "required_sequence_c_max_mean": mean("required_sequence_c_max"),
            "required_sequence_c_raw_max_mean": mean(
                "required_sequence_c_raw_max"
            ),
            "sequence_speed_rms_max_mean": mean("sequence_speed_rms_max"),
            "local_speed_max_mean": mean("local_speed_max"),
            "active_interval_min_mean": mean("active_interval_min"),
            "active_interval_rms_min_mean": mean("active_interval_rms_min"),
            "local_interval_min_mean": mean("local_interval_min"),
            "local_interval_negative_fraction_mean": mean(
                "local_interval_negative_fraction_mean"
            ),
            "triangle_violation_count": sum(
                item["triangle_violation_count"] for item in subset
            ),
        }
    return summary


def run_analysis(analyzer: FisherAnalyzer, args) -> List[Dict]:
    dataset = build_training_dataset(analyzer, args)
    max_items = args.max_dataset_items or len(dataset)
    results = []
    with mask_all_supervised_positions():
        for data_index in range(max_items):
            batch = dataset[data_index]
            # Packed datasets may return {} while their source-local buffer is
            # filling; continue consuming raw records until a packed item is
            # available.
            if "input_ids" not in batch or "labels" not in batch:
                continue
            try:
                result = analyzer.analyze_item(
                    batch,
                    sample_index=len(results),
                    output_dir=args.output_dir,
                    num_time_steps=args.num_time_steps,
                    time_grid=args.time_grid,
                    fisher_chunk_size=args.fisher_chunk_size,
                    c=args.c,
                    mask_seed=args.seed + len(results),
                    save_distributions=args.save_distributions,
                )
            except (RuntimeError, ValueError, IndexError) as error:
                LOGGER.warning("Skipping raw dataset item %d: %s", data_index, error)
                continue
            results.append(result)
            LOGGER.info(
                "sample=%d task=%s targets=%d endpoint=%f path=%f interval_min=%f",
                result["sample_index"],
                result["task"],
                result["num_target_positions"],
                result["fisher_rao_distance"],
                result["fisher_rao_path_length"],
                result["sequence_interval_min"],
            )
            if len(results) >= args.num_samples:
                break

    if not results:
        raise RuntimeError(
            "No complete training item was produced; increase --max_dataset_items "
            "or inspect the dataset error log."
        )
    output_dir = Path(args.output_dir)
    with (output_dir / "fisher_results.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    summary = summarize_results(results)
    with (output_dir / "fisher_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    LOGGER.info("Wrote %d samples to %s", len(results), output_dir)
    return results


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
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--max_dataset_items", type=int, default=None)
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
        default=5,
        help="Nested masked states, including alpha=1 and alpha=0.",
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
        default=16,
        help="Target rows per probability softmax chunk.",
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
        help="Save endpoint [target_position,vocab] float32 prob_0/prob_T tensors.",
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
    if args.c <= 0:
        parser.error("--c must be positive")
    return args


def main() -> None:
    args = parse_args()
    if str(args.device_map).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not visible to this process, but --device_map requests "
            f"{args.device_map}. Run on a GPU allocation (for example an H100) "
            "or pass --device_map cpu for a small test model."
        )
    set_seed(args.seed)
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
        audio_tokenizer_rank=0 if torch.cuda.is_available() else None,
    )
    run_analysis(analyzer, args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
