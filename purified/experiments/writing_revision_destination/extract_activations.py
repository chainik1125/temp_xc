"""Extract final-H Llama layer-10 residual-stream windows for KLiCKe.

The input cohort is the exact-token artifact built by ``token_audit.py``.
Extraction is resumable at deterministic shard boundaries. Existing shards are
validated against their exact cohort rows before being skipped, and the cache
is marked complete only after every shard passes checksum and shape checks.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file, save_file

from .klicke import sha256_file
from .token_audit import (
    DEFAULT_HISTORY_TOKENS,
    DEFAULT_MAX_MODEL_TOKENS,
    DEFAULT_SUBJECT_TOKENIZER,
    PROTOCOL_VERSION as TOKEN_PROTOCOL_VERSION,
)


EXTRACTION_PROTOCOL_VERSION = "klicke-deletion-resid-post-l10-v1"
DEFAULT_LAYER = 10
HASH_BYTES = 32


@dataclass(frozen=True)
class ExtractionConfig:
    protocol_version: str = EXTRACTION_PROTOCOL_VERSION
    model: str = DEFAULT_SUBJECT_TOKENIZER
    revision: str | None = None
    layer: int = DEFAULT_LAYER
    window_tokens: int = DEFAULT_HISTORY_TOKENS
    max_model_tokens: int = DEFAULT_MAX_MODEL_TOKENS
    batch_size: int = 8
    shard_size: int = 256
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    attention: str = "sdpa"


REQUIRED_COHORT_COLUMNS = {
    "event_hash",
    "writer_hash",
    "window_hash",
    "input_ids",
    "window_token_ids",
    "token_distance",
    "capped_token_label",
    "lexical_deleted_words",
    "lexical_label",
    "prefix_token_count",
    "special_tokens_added",
    "remove_actions",
    "single_character_backspaces",
}
FORBIDDEN_COHORT_COLUMNS = {
    "writer_id",
    "preburst_text",
    "postburst_text",
    "words",
    "text",
}
REQUIRED_SHARD_KEYS = {
    "activations",
    "row_index",
    "event_hash",
    "writer_hash",
    "window_hash",
    "window_token_ids",
    "token_distance",
    "capped_token_label",
    "lexical_deleted_words",
    "lexical_label",
    "prefix_token_count",
    "special_tokens_added",
    "remove_actions",
    "single_character_backspaces",
}


def _atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_safetensors(
    tensors: dict[str, torch.Tensor],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(
        {name: value.contiguous() for name, value in tensors.items()},
        str(temporary),
    )
    os.replace(temporary, path)


def _as_ids(value: object) -> tuple[int, ...]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError("token-ID column must contain a list-like value")
    return tuple(int(item) for item in value)


def _window_hash(token_ids: Sequence[int]) -> str:
    payload = ",".join(str(int(value)) for value in token_ids)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hash_tensor(values: Sequence[str]) -> torch.Tensor:
    rows = []
    for value in values:
        raw = bytes.fromhex(str(value))
        if len(raw) != HASH_BYTES:
            raise ValueError("expected a full SHA-256 hexadecimal hash")
        rows.append(list(raw))
    return torch.tensor(rows, dtype=torch.uint8)


def _frame_semantic_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame.sort_values("event_hash").itertuples(index=False):
        digest.update(
            (
                f"{row.event_hash}\x1f{row.writer_hash}\x1f"
                f"{row.window_hash}\x1f{int(row.token_distance)}\x1f"
                f"{int(row.lexical_label)}\n"
            ).encode("ascii")
        )
    return digest.hexdigest()


def validate_token_cohort(
    cohort_path: str | Path,
    manifest_path: str | Path,
    config: ExtractionConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load a token cohort and prove its manifest and row invariants."""

    cohort_path = Path(cohort_path)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = manifest.get("protocol", {})
    if protocol.get("version") != TOKEN_PROTOCOL_VERSION:
        raise ValueError("token cohort protocol version drifted")
    cohort_record = manifest.get("cohort")
    if not isinstance(cohort_record, dict):
        raise ValueError("token manifest does not identify a cohort artifact")
    if cohort_record.get("sha256") != sha256_file(cohort_path):
        raise ValueError("token cohort Parquet checksum failed")
    if protocol.get("tokenizer") != config.model:
        raise ValueError("cohort tokenizer and extraction model disagree")
    if int(protocol.get("history_tokens", -1)) != config.window_tokens:
        raise ValueError("cohort and extraction window lengths disagree")
    if int(protocol.get("max_model_tokens", -1)) != config.max_model_tokens:
        raise ValueError("cohort and extraction context limits disagree")

    frame = pd.read_parquet(cohort_path)
    missing = REQUIRED_COHORT_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"token cohort lacks columns: {sorted(missing)}")
    forbidden = FORBIDDEN_COHORT_COLUMNS.intersection(frame.columns)
    if forbidden:
        raise ValueError(f"token cohort exposes private columns: {sorted(forbidden)}")
    if len(frame) != int(manifest["counts"]["retained_events"]):
        raise ValueError("token cohort row count disagrees with its manifest")
    if frame["event_hash"].duplicated().any():
        raise ValueError("token cohort contains duplicate event hashes")
    if frame["window_hash"].duplicated().any():
        raise ValueError("token cohort contains duplicate exact windows")
    if _frame_semantic_sha256(frame) != manifest.get("cohort_sha256"):
        raise ValueError("token cohort semantic fingerprint failed")

    token_cap = int(protocol["token_cap"])
    normalized_input_ids = []
    normalized_windows = []
    for row in frame.itertuples(index=False):
        input_ids = _as_ids(row.input_ids)
        window = _as_ids(row.window_token_ids)
        if not 1 <= len(input_ids) <= config.max_model_tokens:
            raise ValueError("cohort prefix length lies outside model limits")
        if len(window) != config.window_tokens:
            raise ValueError("cohort window has the wrong width")
        if input_ids[-config.window_tokens :] != window:
            raise ValueError("cohort window is not the final prefix window")
        if int(row.prefix_token_count) != len(input_ids):
            raise ValueError("cohort prefix-token count is stale")
        if not 0 <= int(row.special_tokens_added) < len(input_ids):
            raise ValueError("cohort special-token count is invalid")
        if str(row.window_hash) != _window_hash(window):
            raise ValueError("cohort token-window hash is stale")
        if int(row.token_distance) < 1:
            raise ValueError("cohort contains a nonpositive token distance")
        if int(row.capped_token_label) != min(
            int(row.token_distance),
            token_cap,
        ):
            raise ValueError("cohort capped-token target is stale")
        normalized_input_ids.append(input_ids)
        normalized_windows.append(window)
    frame = frame.copy()
    frame["input_ids"] = normalized_input_ids
    frame["window_token_ids"] = normalized_windows
    return frame, manifest


def slice_final_windows(
    hidden: torch.Tensor,
    lengths: Sequence[int],
    *,
    window_tokens: int,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Copy the final real-token block output from a right-padded batch."""

    if hidden.ndim != 3 or hidden.shape[0] != len(lengths):
        raise ValueError("hidden batch and prefix lengths disagree")
    result = torch.empty(
        (len(lengths), window_tokens, int(hidden.shape[-1])),
        dtype=output_dtype,
        device="cpu",
    )
    for row, length in enumerate(lengths):
        if not window_tokens <= int(length) <= hidden.shape[1]:
            raise ValueError("prefix length cannot supply the final window")
        values = hidden[row, int(length) - window_tokens : int(length)]
        result[row] = values.to(dtype=output_dtype, device="cpu")
    return result


def _load_subject(config: ExtractionConfig, cohort_manifest: dict[str, object]):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model,
        revision=config.revision,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"

    if config.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif config.dtype == "float16":
        dtype = torch.float16
    else:
        raise ValueError("dtype must be bfloat16 or float16")
    model = AutoModel.from_pretrained(
        config.model,
        revision=config.revision,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map={"": config.device},
        attn_implementation=config.attention,
    )
    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id
    backbone = model.model if hasattr(model, "model") else model
    if not hasattr(backbone, "layers"):
        raise TypeError("cannot locate subject-model transformer layers")
    if not 0 <= config.layer < len(backbone.layers):
        raise ValueError("requested layer lies outside the subject model")

    expected_revision = cohort_manifest["protocol"].get(
        "tokenizer_revision_resolved"
    )
    observed_tokenizer_revision = (
        getattr(tokenizer, "_commit_hash", None)
        or getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    )
    if (
        expected_revision
        and observed_tokenizer_revision
        and expected_revision != observed_tokenizer_revision
    ):
        raise ValueError("runtime tokenizer revision differs from token cohort")
    observed_model_revision = getattr(model.config, "_commit_hash", None)
    if (
        expected_revision
        and observed_model_revision
        and expected_revision != observed_model_revision
    ):
        raise ValueError("runtime model revision differs from token cohort")
    expected_vocab = cohort_manifest["protocol"].get("tokenizer_vocab_size")
    if expected_vocab is not None and int(expected_vocab) != len(tokenizer):
        raise ValueError("runtime tokenizer vocabulary differs from token cohort")
    if int(model.config.vocab_size) != len(tokenizer):
        raise ValueError("runtime model and tokenizer vocabularies disagree")
    if str(model.config.model_type) != "llama":
        raise ValueError("deletion activation protocol requires a Llama model")
    return model, tokenizer, backbone.layers[config.layer]


def build_padded_batch(
    input_rows: Sequence[Sequence[int]],
    *,
    pad_token_id: int,
    device: str,
    forced_width: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Right-pad IDs and construct explicit real-token position IDs."""

    if not input_rows:
        raise ValueError("cannot build an empty model batch")
    lengths = [len(values) for values in input_rows]
    width = max(lengths) if forced_width is None else int(forced_width)
    if width < max(lengths):
        raise ValueError("forced batch width truncates an input prefix")
    input_ids = torch.full(
        (len(input_rows), width),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, values in enumerate(input_rows):
        tensor = torch.tensor(values, dtype=torch.long, device=device)
        input_ids[row, : len(tensor)] = tensor
        attention_mask[row, : len(tensor)] = 1
    position_ids = attention_mask.cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    for row, length in enumerate(lengths):
        expected = torch.arange(length, device=device)
        if not torch.equal(position_ids[row, :length], expected):
            raise AssertionError("real-token position IDs are not zero based")
    return input_ids, attention_mask, position_ids, lengths


def _forward_batch(
    model,
    layer_module,
    input_rows: Sequence[Sequence[int]],
    config: ExtractionConfig,
    *,
    pad_token_id: int,
    forced_width: int | None = None,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    input_ids, attention_mask, position_ids, lengths = build_padded_batch(
        input_rows,
        pad_token_id=pad_token_id,
        device=config.device,
        forced_width=forced_width,
    )

    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hidden.detach()

    handle = layer_module.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
    finally:
        handle.remove()
    if "hidden" not in captured:
        raise RuntimeError("layer-10 residual-stream hook did not run")
    return slice_final_windows(
        captured["hidden"],
        lengths,
        window_tokens=config.window_tokens,
        output_dtype=output_dtype,
    )


def _comparison_metrics(
    observed: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, object]:
    if observed.shape != reference.shape or observed.ndim != 2:
        raise ValueError("diagnostic activation windows must have shape [T, d]")
    observed = observed.float()
    reference = reference.float()
    delta = observed - reference
    denominator = torch.linalg.vector_norm(reference).clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        observed.reshape(1, -1),
        reference.reshape(1, -1),
    )
    per_offset = []
    for offset in range(len(observed)):
        offset_delta = delta[offset]
        offset_reference = reference[offset]
        offset_denominator = torch.linalg.vector_norm(
            offset_reference
        ).clamp_min(1e-12)
        offset_cosine = torch.nn.functional.cosine_similarity(
            observed[offset].reshape(1, -1),
            offset_reference.reshape(1, -1),
        )
        per_offset.append(
            {
                "offset_oldest_zero": offset,
                "relative_l2": float(
                    torch.linalg.vector_norm(offset_delta)
                    / offset_denominator
                ),
                "max_abs": float(offset_delta.abs().max()),
                "cosine": float(offset_cosine.item()),
            }
        )
    return {
        "relative_l2": float(
            torch.linalg.vector_norm(delta) / denominator
        ),
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "cosine": float(cosine.item()),
        "per_offset": per_offset,
    }


def padding_invariance_diagnostics(
    model,
    layer_module,
    input_rows: Sequence[Sequence[int]],
    config: ExtractionConfig,
    *,
    pad_token_id: int,
    relative_tolerance: float = 0.01,
    cosine_floor: float = 0.999,
) -> dict[str, object]:
    """Decompose padding-path and batch-dimension numerical drift."""

    candidates = list(input_rows)
    if not candidates:
        raise ValueError("cannot validate padding on an empty cohort")
    shortest = min(
        range(len(candidates)),
        key=lambda index: len(candidates[index]),
    )
    longest = max(
        range(len(candidates)),
        key=lambda index: len(candidates[index]),
    )
    if shortest == longest and len(candidates) > 1:
        indices = [0, 1]
    else:
        indices = [shortest] if shortest == longest else [shortest, longest]
    rows = [candidates[index] for index in indices]
    common_width = max(len(row) for row in rows)
    batched = _forward_batch(
        model,
        layer_module,
        rows,
        config,
        pad_token_id=pad_token_id,
        forced_width=common_width,
        output_dtype=torch.float32,
    )
    padded_singles = torch.cat(
        [
            _forward_batch(
                model,
                layer_module,
                [row],
                config,
                pad_token_id=pad_token_id,
                forced_width=common_width,
                output_dtype=torch.float32,
            )
            for row in rows
        ],
        dim=0,
    )
    unpadded_singles = torch.cat(
        [
            _forward_batch(
                model,
                layer_module,
                [row],
                config,
                pad_token_id=pad_token_id,
                output_dtype=torch.float32,
            )
            for row in rows
        ],
        dim=0,
    )
    row_records = []
    for local_index, cohort_index in enumerate(indices):
        total = _comparison_metrics(
            batched[local_index],
            unpadded_singles[local_index],
        )
        padding = _comparison_metrics(
            padded_singles[local_index],
            unpadded_singles[local_index],
        )
        batching = _comparison_metrics(
            batched[local_index],
            padded_singles[local_index],
        )
        passed = (
            float(total["relative_l2"]) <= relative_tolerance
            and float(total["cosine"]) >= cosine_floor
        )
        row_records.append(
            {
                "cohort_row": int(cohort_index),
                "prefix_tokens": len(rows[local_index]),
                "common_padded_width": common_width,
                "total_batched_vs_unpadded": total,
                "padding_path_padded_single_vs_unpadded": padding,
                "batch_dimension_batched_vs_padded_single": batching,
                "dominant_component": (
                    "padding_path"
                    if float(padding["relative_l2"])
                    >= float(batching["relative_l2"])
                    else "batch_dimension"
                ),
                "passed": passed,
            }
        )
    return {
        "status": (
            "passed"
            if all(bool(record["passed"]) for record in row_records)
            else "failed"
        ),
        "mode": "batched_padding_invariance",
        "relative_l2_tolerance": relative_tolerance,
        "cosine_floor": cosine_floor,
        "attention_implementation": config.attention,
        "model_compute_dtype": config.dtype,
        "extraction_batch_size": config.batch_size,
        "explicit_position_ids": True,
        "position_rule": (
            "attention_mask.cumsum(-1)-1 on real tokens; padded positions zero"
        ),
        "comparison_output_dtype": "float32 before cache fp16 conversion",
        "rows": row_records,
    }


def singleton_repeatability_diagnostics(
    model,
    layer_module,
    input_rows: Sequence[Sequence[int]],
    config: ExtractionConfig,
    *,
    pad_token_id: int,
) -> dict[str, object]:
    """Require identical repeated unpadded singleton forwards."""

    if config.batch_size != 1:
        raise ValueError("singleton repeatability requires extraction batch size 1")
    candidates = list(input_rows)
    if not candidates:
        raise ValueError("cannot validate repeatability on an empty cohort")
    shortest = min(
        range(len(candidates)),
        key=lambda index: len(candidates[index]),
    )
    longest = max(
        range(len(candidates)),
        key=lambda index: len(candidates[index]),
    )
    if shortest == longest and len(candidates) > 1:
        indices = [0, 1]
    else:
        indices = [shortest] if shortest == longest else [shortest, longest]

    row_records = []
    for cohort_index in indices:
        row = candidates[cohort_index]
        first = _forward_batch(
            model,
            layer_module,
            [row],
            config,
            pad_token_id=pad_token_id,
            output_dtype=torch.float32,
        )
        second = _forward_batch(
            model,
            layer_module,
            [row],
            config,
            pad_token_id=pad_token_id,
            output_dtype=torch.float32,
        )
        metrics = _comparison_metrics(second[0], first[0])
        exact_equal = bool(torch.equal(first, second))
        row_records.append(
            {
                "cohort_row": int(cohort_index),
                "prefix_tokens": len(row),
                "model_width": len(row),
                "padding_tokens": 0,
                "repeat_second_vs_first": metrics,
                "exact_equal": exact_equal,
                "passed": exact_equal,
            }
        )
    return {
        "status": (
            "passed"
            if all(bool(record["passed"]) for record in row_records)
            else "failed"
        ),
        "mode": "singleton_repeatability",
        "requirement": (
            "bitwise-identical layer output for repeated forwards with identical "
            "token IDs, masks, positions, shape, and model state"
        ),
        "attention_implementation": config.attention,
        "model_compute_dtype": config.dtype,
        "extraction_batch_size": config.batch_size,
        "explicit_position_ids": True,
        "position_rule": "zero-based arange over the unpadded singleton prefix",
        "comparison_output_dtype": "float32 before cache fp16 conversion",
        "rows": row_records,
    }


def extraction_invariance_diagnostics(
    model,
    layer_module,
    input_rows: Sequence[Sequence[int]],
    config: ExtractionConfig,
    *,
    pad_token_id: int,
) -> dict[str, object]:
    """Select the fail-closed check that matches extraction execution."""

    if config.batch_size == 1:
        return singleton_repeatability_diagnostics(
            model,
            layer_module,
            input_rows,
            config,
            pad_token_id=pad_token_id,
        )
    return padding_invariance_diagnostics(
        model,
        layer_module,
        input_rows,
        config,
        pad_token_id=pad_token_id,
    )


def shard_tensors(
    frame: pd.DataFrame,
    activations: torch.Tensor,
    *,
    start: int,
    window_tokens: int,
) -> dict[str, torch.Tensor]:
    if activations.shape[:2] != (len(frame), window_tokens):
        raise ValueError("activation shard shape disagrees with cohort rows")
    windows = np.asarray(
        [list(values) for values in frame["window_token_ids"]],
        dtype=np.int32,
    )
    return {
        "activations": activations.to(dtype=torch.float16, device="cpu"),
        "row_index": torch.arange(
            start,
            start + len(frame),
            dtype=torch.int64,
        ),
        "event_hash": _hash_tensor(frame["event_hash"].astype(str).tolist()),
        "writer_hash": _hash_tensor(frame["writer_hash"].astype(str).tolist()),
        "window_hash": _hash_tensor(frame["window_hash"].astype(str).tolist()),
        "window_token_ids": torch.from_numpy(windows),
        "token_distance": torch.tensor(
            frame["token_distance"].to_numpy(dtype=np.int32)
        ),
        "capped_token_label": torch.tensor(
            frame["capped_token_label"].to_numpy(dtype=np.int16)
        ),
        "lexical_deleted_words": torch.tensor(
            frame["lexical_deleted_words"].to_numpy(dtype=np.int16)
        ),
        "lexical_label": torch.tensor(
            frame["lexical_label"].to_numpy(dtype=np.int8)
        ),
        "prefix_token_count": torch.tensor(
            frame["prefix_token_count"].to_numpy(dtype=np.int32)
        ),
        "special_tokens_added": torch.tensor(
            frame["special_tokens_added"].to_numpy(dtype=np.int8)
        ),
        "remove_actions": torch.tensor(
            frame["remove_actions"].to_numpy(dtype=np.int32)
        ),
        "single_character_backspaces": torch.tensor(
            frame["single_character_backspaces"].to_numpy(dtype=bool)
        ),
    }


def validate_shard_tensors(
    tensors: dict[str, torch.Tensor],
    frame: pd.DataFrame,
    *,
    start: int,
    window_tokens: int,
    hidden_size: int | None,
) -> int:
    missing = REQUIRED_SHARD_KEYS.difference(tensors)
    if missing:
        raise ValueError(f"activation shard lacks tensors: {sorted(missing)}")
    activations = tensors["activations"]
    if activations.dtype != torch.float16:
        raise ValueError("activation cache must be float16")
    if activations.ndim != 3 or activations.shape[:2] != (
        len(frame),
        window_tokens,
    ):
        raise ValueError("activation tensor shape is stale")
    observed_hidden = int(activations.shape[-1])
    if hidden_size is not None and observed_hidden != hidden_size:
        raise ValueError("activation hidden size changed across shards")
    if not torch.isfinite(activations).all():
        raise ValueError("activation shard contains non-finite values")

    expected = shard_tensors(
        frame,
        torch.zeros_like(activations),
        start=start,
        window_tokens=window_tokens,
    )
    for name in REQUIRED_SHARD_KEYS.difference({"activations"}):
        if not torch.equal(tensors[name].cpu(), expected[name]):
            raise ValueError(f"activation shard metadata drifted: {name}")
    return observed_hidden


def _runtime_record(model, tokenizer, config: ExtractionConfig) -> dict[str, object]:
    import transformers

    model_revision = getattr(model.config, "_commit_hash", None)
    tokenizer_revision = (
        getattr(tokenizer, "_commit_hash", None)
        or getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    )
    device = torch.device(config.device)
    return {
        "model_revision_observed": model_revision,
        "tokenizer_revision_observed": tokenizer_revision,
        "model_type": str(model.config.model_type),
        "hidden_size": int(model.config.hidden_size),
        "num_hidden_layers": int(model.config.num_hidden_layers),
        "vocab_size": int(model.config.vocab_size),
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_vocab_size": int(len(tokenizer)),
        "hookpoint": f"model.layers[{config.layer}] block output[0]",
        "hook_semantics": "resid_post",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
    }


def _expected_ranges(rows: int, shard_size: int) -> list[tuple[int, int]]:
    return [
        (start, min(start + shard_size, rows))
        for start in range(0, rows, shard_size)
    ]


def _event_hash_digest(frame: pd.DataFrame) -> str:
    payload = "\n".join(frame["event_hash"].astype(str)) + "\n"
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _ensure_shard_sidecar(
    tensor_path: Path,
    frame: pd.DataFrame,
    *,
    start: int,
    end: int,
    request_sha256: str,
    shape: Sequence[int],
) -> tuple[Path, dict[str, object]]:
    sidecar_path = tensor_path.with_suffix(".manifest.json")
    record = {
        "tensor": tensor_path.name,
        "start": start,
        "end": end,
        "rows": end - start,
        "shape": list(shape),
        "dtype": "float16",
        "event_hashes_sha256": _event_hash_digest(frame),
        "request_sha256": request_sha256,
        "tensor_sha256": sha256_file(tensor_path),
    }
    if sidecar_path.exists():
        if json.loads(sidecar_path.read_text(encoding="utf-8")) != record:
            raise ValueError(f"activation shard sidecar drifted: {tensor_path.name}")
    else:
        _atomic_json(record, sidecar_path)
    return sidecar_path, record


def run(
    *,
    cohort_path: str | Path,
    cohort_manifest_path: str | Path,
    output_dir: str | Path,
    config: ExtractionConfig,
    limit: int | None = None,
) -> dict[str, object]:
    cohort_path = Path(cohort_path)
    cohort_manifest_path = Path(cohort_manifest_path)
    output_dir = Path(output_dir)
    cohort, cohort_manifest = validate_token_cohort(
        cohort_path,
        cohort_manifest_path,
        config,
    )
    if config.batch_size < 1:
        raise ValueError("batch size must be positive")
    if config.shard_size < 1:
        raise ValueError("shard size must be positive")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        cohort = cohort.iloc[:limit].reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(exist_ok=True)

    request = {
        "protocol_version": EXTRACTION_PROTOCOL_VERSION,
        "config": asdict(config),
        "cohort_path": str(cohort_path.resolve()),
        "cohort_sha256": sha256_file(cohort_path),
        "cohort_semantic_sha256": _frame_semantic_sha256(cohort),
        "cohort_manifest_path": str(cohort_manifest_path.resolve()),
        "cohort_manifest_sha256": sha256_file(cohort_manifest_path),
        "rows": len(cohort),
        "limit": limit,
        "hookpoint": f"model.layers[{config.layer}] block output[0]",
        "hook_semantics": "resid_post",
        "padding": (
            "none: unpadded singleton forwards; final real-token window only"
            if config.batch_size == 1
            else (
                "deterministic within-shard length buckets with right padding; "
                "final real-token window only"
            )
        ),
        "raw_text_in_cache": False,
        "full_prefix_activations_in_cache": False,
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    request_path = output_dir / "request.json"
    if request_path.exists():
        if json.loads(request_path.read_text(encoding="utf-8")) != request:
            raise ValueError("existing extraction request differs from this run")
    else:
        _atomic_json(request, request_path)

    runtime_path = output_dir / "runtime.json"
    runtime = (
        json.loads(runtime_path.read_text(encoding="utf-8"))
        if runtime_path.exists()
        else None
    )
    hidden_size = int(runtime["hidden_size"]) if runtime is not None else None
    request_sha256 = sha256_file(request_path)
    ranges = _expected_ranges(len(cohort), config.shard_size)
    pending = []
    for start, end in ranges:
        path = shard_dir / f"rows_{start:06d}_{end:06d}.safetensors"
        sidecar_path = path.with_suffix(".manifest.json")
        frame = cohort.iloc[start:end]
        if path.exists():
            if runtime is None:
                raise ValueError("activation shard exists without runtime metadata")
            tensors = load_file(str(path), device="cpu")
            observed = validate_shard_tensors(
                tensors,
                frame,
                start=start,
                window_tokens=config.window_tokens,
                hidden_size=hidden_size,
            )
            if observed != hidden_size:
                raise AssertionError("validated shard hidden size drifted")
            _ensure_shard_sidecar(
                path,
                frame,
                start=start,
                end=end,
                request_sha256=request_sha256,
                shape=tensors["activations"].shape,
            )
        elif sidecar_path.exists():
            raise ValueError(
                f"activation sidecar exists without tensor: {path.name}"
            )
        else:
            pending.append((start, end, path))

    print(
        f"[deletion-extract] rows={len(cohort)} "
        f"complete_shards={len(ranges) - len(pending)} "
        f"pending_shards={len(pending)}",
        flush=True,
    )
    if pending:
        model, tokenizer, layer_module = _load_subject(
            config,
            cohort_manifest,
        )
        maximum_token_id = max(
            max(int(value) for value in input_ids)
            for input_ids in cohort["input_ids"]
        )
        if maximum_token_id >= int(model.config.vocab_size):
            raise ValueError("cohort contains a token outside the model vocabulary")
        for row in cohort.itertuples(index=False):
            input_ids = list(row.input_ids)
            special_tokens = int(row.special_tokens_added)
            text_ids = input_ids[special_tokens:]
            rebuilt = tokenizer.build_inputs_with_special_tokens(text_ids)
            if [int(value) for value in rebuilt] != input_ids:
                raise ValueError(
                    "cohort model inputs disagree with runtime special tokens"
                )
        diagnostic_path = output_dir / "padding_diagnostic.json"
        if runtime is None or not diagnostic_path.exists():
            padding_diagnostic = extraction_invariance_diagnostics(
                model,
                layer_module,
                cohort["input_ids"].tolist(),
                config,
                pad_token_id=int(tokenizer.pad_token_id),
            )
            _atomic_json(padding_diagnostic, diagnostic_path)
        else:
            padding_diagnostic = json.loads(
                diagnostic_path.read_text(encoding="utf-8")
            )
        if padding_diagnostic.get("status") != "passed":
            if padding_diagnostic.get("mode") == "singleton_repeatability":
                failures = [
                    {
                        "cohort_row": record["cohort_row"],
                        "prefix_tokens": record["prefix_tokens"],
                        "exact_equal": record["exact_equal"],
                        "relative_l2": record[
                            "repeat_second_vs_first"
                        ]["relative_l2"],
                        "max_abs": record[
                            "repeat_second_vs_first"
                        ]["max_abs"],
                        "cosine": record[
                            "repeat_second_vs_first"
                        ]["cosine"],
                    }
                    for record in padding_diagnostic["rows"]
                    if not record["passed"]
                ]
            else:
                failures = [
                    {
                        "cohort_row": record["cohort_row"],
                        "prefix_tokens": record["prefix_tokens"],
                        "dominant_component": record["dominant_component"],
                        "relative_l2": record[
                            "total_batched_vs_unpadded"
                        ]["relative_l2"],
                        "max_abs": record[
                            "total_batched_vs_unpadded"
                        ]["max_abs"],
                        "cosine": record[
                            "total_batched_vs_unpadded"
                        ]["cosine"],
                    }
                    for record in padding_diagnostic["rows"]
                    if not record["passed"]
                ]
            raise ValueError(
                "activation invariance failed closed; "
                f"diagnostics={diagnostic_path}; failures={failures}"
            )
        observed_runtime = _runtime_record(model, tokenizer, config)
        observed_runtime["activation_invariance_check"] = "passed"
        observed_runtime["activation_invariance_mode"] = padding_diagnostic[
            "mode"
        ]
        observed_runtime["activation_diagnostic_sha256"] = sha256_file(
            diagnostic_path
        )
        if runtime is None:
            runtime = observed_runtime
            _atomic_json(runtime, runtime_path)
            hidden_size = int(runtime["hidden_size"])
        elif runtime != observed_runtime:
            raise ValueError("runtime provenance changed during resumed extraction")

        started = time.time()
        for shard_number, (start, end, path) in enumerate(pending, start=1):
            frame = cohort.iloc[start:end]
            order = sorted(
                range(len(frame)),
                key=lambda index: (
                    len(frame.iloc[index]["input_ids"]),
                    str(frame.iloc[index]["event_hash"]),
                ),
            )
            activations = torch.empty(
                (
                    len(frame),
                    config.window_tokens,
                    int(hidden_size),
                ),
                dtype=torch.float16,
                device="cpu",
            )
            for batch_start in range(0, len(order), config.batch_size):
                local_indices = order[
                    batch_start : batch_start + config.batch_size
                ]
                batch = frame.iloc[local_indices]
                values = _forward_batch(
                    model,
                    layer_module,
                    batch["input_ids"].tolist(),
                    config,
                    pad_token_id=int(tokenizer.pad_token_id),
                )
                activations[local_indices] = values
            tensors = shard_tensors(
                frame,
                activations,
                start=start,
                window_tokens=config.window_tokens,
            )
            validate_shard_tensors(
                tensors,
                frame,
                start=start,
                window_tokens=config.window_tokens,
                hidden_size=hidden_size,
            )
            _atomic_safetensors(tensors, path)
            _ensure_shard_sidecar(
                path,
                frame,
                start=start,
                end=end,
                request_sha256=request_sha256,
                shape=tensors["activations"].shape,
            )
            print(
                f"[deletion-extract] shard={shard_number}/{len(pending)} "
                f"rows={start}:{end} elapsed_s={time.time() - started:.1f}",
                flush=True,
            )
            del activations, tensors
            gc.collect()
            torch.cuda.empty_cache()

    if runtime is None:
        raise AssertionError("complete extraction lacks runtime provenance")
    shard_records = []
    coverage = []
    for start, end in ranges:
        path = shard_dir / f"rows_{start:06d}_{end:06d}.safetensors"
        tensors = load_file(str(path), device="cpu")
        validate_shard_tensors(
            tensors,
            cohort.iloc[start:end],
            start=start,
            window_tokens=config.window_tokens,
            hidden_size=int(runtime["hidden_size"]),
        )
        sidecar_path, sidecar = _ensure_shard_sidecar(
            path,
            cohort.iloc[start:end],
            start=start,
            end=end,
            request_sha256=request_sha256,
            shape=tensors["activations"].shape,
        )
        coverage.extend(tensors["row_index"].tolist())
        shard_records.append(
            {
                "path": str(path.relative_to(output_dir)),
                "sidecar": str(sidecar_path.relative_to(output_dir)),
                "start": start,
                "end": end,
                "rows": end - start,
                "shape": list(tensors["activations"].shape),
                "sha256": sidecar["tensor_sha256"],
                "sidecar_sha256": sha256_file(sidecar_path),
            }
        )
    if coverage != list(range(len(cohort))):
        raise ValueError("activation shards do not exactly cover cohort rows")
    complete = {
        "status": "complete",
        "protocol_version": EXTRACTION_PROTOCOL_VERSION,
        "rows": len(cohort),
        "window_tokens": config.window_tokens,
        "hidden_size": int(runtime["hidden_size"]),
        "dtype": "float16",
        "request_sha256": request_sha256,
        "runtime_sha256": sha256_file(runtime_path),
        "shards": shard_records,
    }
    complete_path = output_dir / "complete.json"
    if complete_path.exists():
        if json.loads(complete_path.read_text(encoding="utf-8")) != complete:
            raise ValueError("existing completion manifest is stale")
    else:
        _atomic_json(complete, complete_path)
    return complete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_SUBJECT_TOKENIZER)
    parser.add_argument("--revision")
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument(
        "--window-tokens",
        type=int,
        default=DEFAULT_HISTORY_TOKENS,
    )
    parser.add_argument(
        "--max-model-tokens",
        type=int,
        default=DEFAULT_MAX_MODEL_TOKENS,
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--attention", default="sdpa")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExtractionConfig(
        model=args.model,
        revision=args.revision,
        layer=args.layer,
        window_tokens=args.window_tokens,
        max_model_tokens=args.max_model_tokens,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        device=args.device,
        dtype=args.dtype,
        attention=args.attention,
    )
    result = run(
        cohort_path=args.cohort,
        cohort_manifest_path=args.cohort_manifest,
        output_dir=args.output_dir,
        config=config,
        limit=args.limit,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
