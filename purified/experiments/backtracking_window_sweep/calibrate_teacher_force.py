"""Aggregate-only calibration against one official Ward event.

This diagnostic never writes activations, token IDs, text, or event keys.  It
teacher-forces trace 0 in BF16 and compares the first source-eligible event
against the official six-offset tail across:

* transformer block outputs 8..12;
* boundary shifts -3..3;
* BOS policies on/off.

Five layers are captured in one forward per BOS policy.  SDPA is primary.
Eager is attempted only when SDPA has neither a bit-exact nor a preregistered
close candidate.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .extract_wide_teacher_force import (
    MODEL_DTYPE,
    MODEL_ID,
    MODEL_REVISION,
    OUTPUT_DTYPE,
    TOKENIZER_ID,
    TOKENIZER_REVISION,
    _load_inputs,
    token_containing_char,
)
from .reconstruct_wide_artifact import OFFICIAL_OFFSETS


CALIBRATION_PROTOCOL = "ward-c7-teacher-force-calibration.v1"
CANDIDATE_LAYERS = tuple(range(8, 13))
BOUNDARY_SHIFTS = tuple(range(-3, 4))
BOS_POLICIES = (True, False)
CLOSE_RMSE_MAX = 1e-3
CLOSE_COSINE_MIN = 0.999999


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def candidate_metrics(
    candidate: np.ndarray,
    official: np.ndarray,
) -> dict[str, float | int | bool]:
    """Return aggregate comparisons without retaining activation values."""

    candidate = np.asarray(candidate, dtype=np.float32)
    official = np.asarray(official, dtype=np.float32)
    if candidate.shape != official.shape:
        raise ValueError(
            f"candidate shape {candidate.shape} != official {official.shape}"
        )
    difference = candidate.astype(np.float64) - official.astype(np.float64)
    flat_candidate = candidate.astype(np.float64, copy=False).reshape(-1)
    flat_official = official.astype(np.float64, copy=False).reshape(-1)
    denominator = float(
        np.linalg.norm(flat_candidate) * np.linalg.norm(flat_official)
    )
    cosine = (
        float(np.dot(flat_candidate, flat_official) / denominator)
        if denominator > 0
        else float("nan")
    )
    exact_count = int(np.count_nonzero(candidate == official))
    total = int(candidate.size)
    rmse = float(np.sqrt(np.mean(np.square(difference))))
    return {
        "total_values": total,
        "exact_count": exact_count,
        "mismatch_count": total - exact_count,
        "exact_fraction": exact_count / total,
        "max_abs": float(np.max(np.abs(difference))),
        "mean_abs": float(np.mean(np.abs(difference))),
        "rmse": rmse,
        "cosine": cosine,
        "bit_exact": exact_count == total,
        "close": (
            rmse <= CLOSE_RMSE_MAX
            and math.isfinite(cosine)
            and cosine >= CLOSE_COSINE_MIN
        ),
    }


def candidate_rank(record: Mapping[str, Any]) -> tuple:
    """Put exactness first, then numerical agreement."""

    return (
        -int(record["bit_exact"]),
        -int(record["exact_count"]),
        float(record["rmse"]),
        float(record["max_abs"]),
        -float(record["cosine"]),
        int(record["layer"]),
        abs(int(record["boundary_shift"])),
        int(record["boundary_shift"]),
        not bool(record["add_special_tokens"]),
    )


def _tokenize(tokenizer, text: str, *, add_special_tokens: bool) -> dict:
    encoded = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        return_offsets_mapping=True,
        return_attention_mask=False,
    )
    input_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [
        tuple(int(value) for value in pair)
        for pair in encoded["offset_mapping"]
    ]
    if len(input_ids) != len(offsets):
        raise ValueError("token IDs and offsets have different lengths")
    nonempty = [(start, end) for start, end in offsets if end > start]
    if not nonempty or max(end for _, end in nonempty) != len(text):
        raise ValueError("tokenizer offsets do not cover full_response")
    if tokenizer.decode(input_ids, skip_special_tokens=True) != text:
        raise ValueError("tokenizer does not round-trip full_response")
    return {"input_ids": input_ids, "offsets": offsets}


def _load_tokenizer():
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        TOKENIZER_ID,
        revision=TOKENIZER_REVISION,
        trust_remote_code=False,
    )
    if not tokenizer.is_fast:
        raise ValueError("pinned tokenizer did not load as fast")
    return tokenizer


def _load_model(device: str, attention_implementation: str):
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation=attention_implementation,
    )
    model.eval()
    backbone = model.model if hasattr(model, "model") else model
    if not hasattr(backbone, "layers"):
        raise TypeError("cannot locate transformer blocks")
    if len(backbone.layers) <= max(CANDIDATE_LAYERS):
        raise ValueError("pinned model lacks a candidate layer")
    if int(model.config.hidden_size) != 4096:
        raise ValueError("pinned model hidden size is not 4096")
    observed_revision = getattr(model.config, "_commit_hash", None)
    if observed_revision not in {None, MODEL_REVISION}:
        raise ValueError("observed model revision differs from pin")
    runtime = {
        "model_revision_requested": MODEL_REVISION,
        "model_revision_observed": observed_revision,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(torch.device(device)),
        "attention_implementation_requested": attention_implementation,
        "attention_implementation_observed": getattr(
            model.config,
            "_attn_implementation",
            None,
        ),
    }
    return model, backbone, runtime


def _capture_positions(
    model,
    backbone,
    *,
    input_ids: Sequence[int],
    positions: Sequence[int],
    device: str,
) -> dict[int, np.ndarray]:
    """Capture only the candidate positions from five blocks in one forward."""

    import torch

    if min(positions) < 0 or max(positions) >= len(input_ids):
        raise ValueError("candidate calibration positions are out of bounds")
    values = torch.tensor(input_ids, dtype=torch.long, device=device)[None]
    attention_mask = torch.ones_like(values)
    position_index = torch.tensor(
        list(positions),
        dtype=torch.long,
        device=device,
    )
    captured: dict[int, np.ndarray] = {}
    handles = []

    for layer_index in CANDIDATE_LAYERS:
        def hook(_module, _inputs, output, *, layer=layer_index):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer] = (
                hidden[0]
                .index_select(0, position_index)
                .to(device="cpu", dtype=torch.float32)
                .numpy()
            )

        handles.append(
            backbone.layers[layer_index].register_forward_hook(hook)
        )
    try:
        with torch.inference_mode():
            output = model(
                input_ids=values,
                attention_mask=attention_mask,
                use_cache=False,
            )
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(CANDIDATE_LAYERS):
        raise RuntimeError("not every candidate layer hook fired")
    del output, values, attention_mask, position_index
    return captured


def _evaluate_backend(
    *,
    tokenizer,
    trace,
    event,
    official_tail: np.ndarray,
    device: str,
    attention_implementation: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    model, backbone, runtime = _load_model(
        device,
        attention_implementation,
    )
    records: list[dict[str, Any]] = []
    for add_special_tokens in BOS_POLICIES:
        tokenized = _tokenize(
            tokenizer,
            trace.full_response,
            add_special_tokens=add_special_tokens,
        )
        if len(tokenized["input_ids"]) > int(
            model.config.max_position_embeddings
        ):
            raise ValueError("trace exceeds model context")
        boundary = token_containing_char(
            tokenized["offsets"],
            event.target_char,
        )
        candidate_positions = sorted(
            {
                boundary + shift + offset
                for shift in BOUNDARY_SHIFTS
                for offset in OFFICIAL_OFFSETS
            }
        )
        captured = _capture_positions(
            model,
            backbone,
            input_ids=tokenized["input_ids"],
            positions=candidate_positions,
            device=device,
        )
        position_to_row = {
            position: row
            for row, position in enumerate(candidate_positions)
        }
        for layer in CANDIDATE_LAYERS:
            for shift in BOUNDARY_SHIFTS:
                rows = [
                    position_to_row[boundary + shift + offset]
                    for offset in OFFICIAL_OFFSETS
                ]
                metrics = candidate_metrics(
                    captured[layer][rows],
                    official_tail,
                )
                records.append(
                    {
                        "attention_implementation": (
                            attention_implementation
                        ),
                        "add_special_tokens": add_special_tokens,
                        "layer": layer,
                        "boundary_shift": shift,
                        **metrics,
                    }
                )
        del captured
    del model, backbone
    gc.collect()
    torch.cuda.empty_cache()
    return records, runtime


def run(
    *,
    prompts_path: Path,
    labels_path: Path,
    traces_path: Path,
    official_path: Path,
    traces_sha256: str,
    source_path: str,
    source_commit: str,
    output_path: Path,
    device: str,
) -> dict[str, Any]:
    (
        traces,
        official_x,
        _official_labels,
        official_keys,
        source_pin,
        _provenance,
        validation,
    ) = _load_inputs(
        prompts_path=prompts_path,
        labels_path=labels_path,
        traces_path=traces_path,
        official_path=official_path,
        traces_sha256=traces_sha256,
        source_path=source_path,
        source_commit=source_commit,
    )
    trace = traces[0]
    if not trace.events:
        raise ValueError("trace 0 has no source-eligible official event")
    event = trace.events[0]
    official_index = {
        key: index for index, key in enumerate(official_keys.tolist())
    }
    row_index = official_index[event.key]
    official_tail = np.asarray(
        official_x[row_index],
        dtype=OUTPUT_DTYPE,
    )
    tokenizer = _load_tokenizer()
    tokenizer_backend_sha256 = _sha256_text(
        tokenizer.backend_tokenizer.to_str()
    )

    records, runtime = _evaluate_backend(
        tokenizer=tokenizer,
        trace=trace,
        event=event,
        official_tail=official_tail,
        device=device,
        attention_implementation="sdpa",
    )
    sdpa_best = min(records, key=candidate_rank)
    eager_attempted = not (
        bool(sdpa_best["bit_exact"]) or bool(sdpa_best["close"])
    )
    runtimes = {"sdpa": runtime}
    if eager_attempted:
        eager_records, eager_runtime = _evaluate_backend(
            tokenizer=tokenizer,
            trace=trace,
            event=event,
            official_tail=official_tail,
            device=device,
            attention_implementation="eager",
        )
        records.extend(eager_records)
        runtimes["eager"] = eager_runtime
    records.sort(key=candidate_rank)
    best = records[0]
    result = {
        "protocol_version": CALIBRATION_PROTOCOL,
        "status": "complete",
        "writes_activation_shards": False,
        "source": {
            "path": source_pin.path,
            "commit": source_pin.commit,
            "sha256": source_pin.sha256,
        },
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "dtype": MODEL_DTYPE,
        },
        "tokenizer": {
            "id": TOKENIZER_ID,
            "revision": TOKENIZER_REVISION,
            "backend_sha256": tokenizer_backend_sha256,
        },
        "calibration_event": {
            "trace_idx": 0,
            "event_key_sha256": _sha256_text(event.key),
            "full_response_sha256": trace.full_response_sha256,
            "official_row": row_index,
            "official_tail_sha256": hashlib.sha256(
                official_tail.tobytes()
            ).hexdigest(),
        },
        "candidate_grid": {
            "layers": list(CANDIDATE_LAYERS),
            "boundary_shifts": list(BOUNDARY_SHIFTS),
            "add_special_tokens": list(BOS_POLICIES),
            "primary_attention_implementation": "sdpa",
            "close_rmse_max": CLOSE_RMSE_MAX,
            "close_cosine_min": CLOSE_COSINE_MIN,
        },
        "source_cohort": {
            "eligible_rows": validation["source_eligible_rows"],
            "eligible_sha256": validation["source_eligible_sha256"],
            "exclusions_sha256": validation[
                "source_exclusions_sha256"
            ],
        },
        "eager_attempted": eager_attempted,
        "runtime": runtimes,
        "candidate_count": len(records),
        "best_candidate": best,
        "bit_exact_candidates": [
            {
                key: record[key]
                for key in (
                    "attention_implementation",
                    "add_special_tokens",
                    "layer",
                    "boundary_shift",
                )
            }
            for record in records
            if record["bit_exact"]
        ],
        "candidates": records,
    }
    _atomic_json(result, output_path)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--traces-sha256", required=True)
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    result = run(
        prompts_path=args.prompts,
        labels_path=args.labels,
        traces_path=args.traces,
        official_path=args.official,
        traces_sha256=args.traces_sha256,
        source_path=args.source_path,
        source_commit=args.source_commit,
        output_path=args.output,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "candidate_count": result["candidate_count"],
                "eager_attempted": result["eager_attempted"],
                "best_candidate": result["best_candidate"],
                "bit_exact_candidates": result["bit_exact_candidates"],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
