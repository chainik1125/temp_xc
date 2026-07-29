"""Run the SAE-free Ward feature-formation calibration.

This consumes the fixed 32-dimensional Rademacher projection cached by the
earlier weak-label run.  The projection was chosen without event labels, so
the present analysis adds no learned representation before the held-out
linear readout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from experiments.temporal_screen_1.feature_formation.estimators import (
    curve_to_dict,
    positionwise_curve,
    summarize_curve,
    transported_curve,
)


DEFAULT_CACHE = Path(
    "/tmp/ward_deepseek8b_layer10_rademacher32_seed20260729.pt"
)
HERE = Path(__file__).resolve().parent
DEFAULT_RESULT = HERE / "results" / "ward_sae_free.json"


def _stable_side(qid: str) -> int:
    digest = hashlib.sha256(qid.encode("utf-8")).digest()
    return 1 if digest[0] % 2 == 0 else -1


def build_paired_panel(
    cache: dict,
    offsets: np.ndarray,
    *,
    exclusion_radius: int = 96,
    preferred_separation: int = 160,
) -> tuple[np.ndarray, list[dict]]:
    """Pair the first usable event with a distant neutral in the same rollout."""

    sequences = cache["sequences"]
    records = []
    panels = []
    min_offset = int(np.min(offsets))
    max_offset = int(np.max(offsets))
    for sequence, raw_events, qid, category in zip(
        sequences,
        cache["event_positions"],
        cache["qids"],
        cache["categories"],
        strict=True,
    ):
        values = np.asarray(sequence, dtype=np.float32)
        events = np.asarray(raw_events, dtype=np.int64)
        events = events[
            (events + min_offset >= 0)
            & (events + max_offset < len(values))
        ]
        if not len(events):
            continue
        event = int(events[0])

        lo = -min_offset
        hi = len(values) - max_offset
        candidates = np.arange(lo, hi, dtype=np.int64)
        if len(raw_events):
            distance = np.min(
                np.abs(candidates[:, None] - np.asarray(raw_events)[None, :]),
                axis=1,
            )
            candidates = candidates[distance >= exclusion_radius]
        if not len(candidates):
            continue
        side = _stable_side(str(qid))
        preferred = event + side * preferred_separation
        preferred_side = candidates[(candidates - event) * side > 0]
        pool = preferred_side if len(preferred_side) else candidates
        neutral = int(pool[np.argmin(np.abs(pool - preferred))])

        event_values = values[event + offsets]
        neutral_values = values[neutral + offsets]
        panels.append(np.stack([event_values, neutral_values], axis=0))
        records.append(
            {
                "qid": str(qid),
                "category": str(category),
                "event_position": event,
                "neutral_position": neutral,
                "neutral_minus_event": neutral - event,
                "n_events": int(len(raw_events)),
                "sequence_tokens": int(len(values)),
            }
        )
    if not panels:
        raise RuntimeError("no usable event/neutral pairs")
    return np.stack(panels).astype(np.float32), records


def run(cache_path: Path = DEFAULT_CACHE) -> dict:
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    offsets = np.asarray(
        list(range(-64, -32, 4)) + list(range(-32, 17)),
        dtype=np.int64,
    )
    panel, pair_records = build_paired_panel(cache, offsets)
    curve_kwargs = {
        "n_splits": 5,
        "regularization": 0.05,
        "seed": 20260729,
    }
    local = positionwise_curve(
        panel,
        offsets,
        width=1,
        **curve_kwargs,
    )
    window6 = positionwise_curve(
        panel,
        offsets,
        width=6,
        **curve_kwargs,
    )
    transported_local = transported_curve(
        panel,
        offsets,
        discovery_band=(-13, -8),
        width=1,
        **curve_kwargs,
    )
    transported_window6 = transported_curve(
        panel,
        offsets,
        discovery_band=(-13, -8),
        width=6,
        **curve_kwargs,
    )
    separations = np.asarray(
        [row["neutral_minus_event"] for row in pair_records]
    )
    return {
        "method": "SAE-free fixed-random-projection formation calibration",
        "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "layer": 10,
        "representation": {
            "kind": "fixed Rademacher projection",
            "dimensions": int(panel.shape[-1]),
            "projection_sha256": cache["projection_hash"],
        },
        "pairing": {
            "n_pairs": int(len(panel)),
            "same_rollout_neutral": True,
            "event": "first usable genuine-backtracking sentence",
            "neutral_exclusion_radius_tokens": 96,
            "preferred_separation_tokens": 160,
            "neutral_after_fraction": float(np.mean(separations > 0)),
            "median_absolute_separation": float(np.median(np.abs(separations))),
            "records": pair_records,
        },
        "offsets": offsets.tolist(),
        "discovery_band": [-13, -8],
        "curves": {
            "positionwise_local": curve_to_dict(local),
            "positionwise_window6": curve_to_dict(window6),
            "transported_local": curve_to_dict(transported_local),
            "transported_window6": curve_to_dict(transported_window6),
        },
        "summaries": {
            "positionwise_local": asdict(summarize_curve(local)),
            "positionwise_window6": asdict(summarize_curve(window6)),
            "transported_local": asdict(summarize_curve(transported_local)),
            "transported_window6": asdict(
                summarize_curve(transported_window6)
            ),
        },
        "caveat": (
            "Event-aligned observational calibration; a causal formation "
            "claim requires checkpoint branching and attention intervention."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    args = parser.parse_args()
    payload = run(args.cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    headline = {
        "n_pairs": payload["pairing"]["n_pairs"],
        "summaries": payload["summaries"],
        "output": str(args.output),
    }
    print(json.dumps(headline, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

