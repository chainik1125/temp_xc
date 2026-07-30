"""Track the known Ward direction and its nearest conventional SAE latent.

This is the corrected feature-formation calibration.  It does not search for
a new readout at every token.  Instead it:

1. projects event/neutral residuals onto the already-causal Ward DoM vectors;
2. builds a direction-source-offset by observation-offset matrix; and
3. tracks the conventional SAE latent with the closest decoder direction.

The full residual panel and SAE checkpoint are reused from the earlier Modal
run, so this analysis is cheap.

Run from the repository root:

    modal run \
      experiments/temporal_screen_1/feature_formation/modal_known_ward_feature.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2] if len(HERE.parents) > 2 else HERE
DATA = REPO / "results" / "ward_backtracking"
RESULT = HERE / "results" / "ward_known_feature_formation.json"

app = modal.App("temporal-screen-known-ward-feature")
cache_volume = modal.Volume.from_name(
    "temporal-screen-ward-weak-label-cache",
    create_if_missing=False,
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.7.1",
        "numpy==2.2.6",
        "scikit-learn==1.7.1",
    )
    .add_local_dir(
        str(HERE),
        "/work/experiments/temporal_screen_1/feature_formation",
    )
    .add_local_file(
        str(DATA / "dom_vectors.pt"),
        "/work/data/dom_vectors.pt",
    )
)


def _curve_at(curve: list[dict], offset: int) -> dict:
    rows = [row for row in curve if int(row["offset"]) == int(offset)]
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one curve row at offset {offset}")
    return rows[0]


@app.function(
    image=image,
    gpu="A10G",
    cpu=8,
    memory=32_768,
    timeout=45 * 60,
    volumes={"/cache": cache_volume},
)
def run() -> dict:
    import sys
    import time

    import numpy as np
    import torch

    sys.path.insert(0, "/work")
    from experiments.temporal_screen_1.feature_formation.known_feature import (
        curve_summary,
        curve_to_dict,
        paired_scalar_curve,
        project_direction_panel,
    )

    started = time.time()
    activation_cache_path = Path(
        "/cache/ward_feature_formation_layer10_full_panel_v1.pt"
    )
    sae_cache_path = Path(
        "/cache/ward_feature_formation_layer10_topk8192_k32_v1.pt"
    )
    sae_result_cache_path = Path(
        "/cache/ward_feature_formation_layer10_topk8192_k32_result_v1.json"
    )
    activation_cache = torch.load(
        activation_cache_path,
        map_location="cpu",
        weights_only=False,
    )
    sae_checkpoint = torch.load(
        sae_cache_path,
        map_location="cpu",
        weights_only=False,
    )
    dom = torch.load(
        "/work/data/dom_vectors.pt",
        map_location="cpu",
        weights_only=False,
    )
    panel = activation_cache["paired_activations"].float().numpy()
    offsets = np.asarray(activation_cache["offsets"], dtype=np.int64)

    direction_curves = {}
    direction_summaries = {}
    direction_meta = {}
    cross_offset = {}
    for arm in ("base", "reasoning"):
        vectors = dom[arm]["vectors"].float().numpy()
        source_offsets = [int(value) for value in dom[arm]["offsets"]]
        vector_names = []
        arm_curves = []
        for source_offset, vector in zip(
            source_offsets,
            vectors,
            strict=True,
        ):
            name = f"{arm}_off{source_offset:+d}"
            points = project_direction_panel(panel, vector, offsets)
            curve = curve_to_dict(points)
            direction_curves[name] = curve
            direction_summaries[name] = curve_summary(points)
            direction_meta[name] = {
                "source_model": arm,
                "derivation_offset": source_offset,
                "norm": float(np.linalg.norm(vector)),
                "is_union": False,
            }
            vector_names.append(name)
            arm_curves.append(curve)

        union_name = f"{arm}_union"
        union = dom[arm]["union"].float().numpy()
        union_points = project_direction_panel(panel, union, offsets)
        direction_curves[union_name] = curve_to_dict(union_points)
        direction_summaries[union_name] = curve_summary(union_points)
        direction_meta[union_name] = {
            "source_model": arm,
            "derivation_offsets": [
                int(value) for value in dom[arm]["union_offsets"]
            ],
            "norm": float(np.linalg.norm(union)),
            "is_union": True,
        }
        cross_offset[arm] = {
            "direction_names": vector_names,
            "direction_source_offsets": source_offsets,
            "observation_offsets": offsets.tolist(),
            "auc": [
                [float(row["auc"]) for row in curve]
                for curve in arm_curves
            ],
            "paired_difference": [
                [float(row["paired_difference"]) for row in curve]
                for curve in arm_curves
            ],
            "paired_effect_dz": [
                [float(row["paired_effect_dz"]) for row in curve]
                for curve in arm_curves
            ],
            "matched_offset": [
                {
                    "direction_name": name,
                    "offset": source_offset,
                    **_curve_at(curve, source_offset),
                }
                for name, source_offset, curve in zip(
                    vector_names,
                    source_offsets,
                    arm_curves,
                    strict=True,
                )
            ],
        }

    state = sae_checkpoint["state_dict"]
    decoder = state["W_dec"].float()
    decoder = decoder / decoder.norm(dim=0, keepdim=True).clamp_min(1e-12)
    base_union = dom["base"]["union"].float()
    base_unit = base_union / base_union.norm().clamp_min(1e-12)
    cosine = decoder.T @ base_unit
    positive_order = torch.argsort(cosine, descending=True)
    absolute_order = torch.argsort(torch.abs(cosine), descending=True)
    selected = torch.unique(
        torch.cat([positive_order[:16], absolute_order[:16]])
    ).tolist()
    best_positive = int(positive_order[0])
    best_absolute = int(absolute_order[0])

    w_enc = state["W_enc"].float().to("cuda")
    b_enc = state["b_enc"].float().to("cuda")
    b_dec = state["b_dec"].float().to("cuda")
    flat = torch.from_numpy(panel).reshape(-1, panel.shape[-1])
    selected_values = torch.zeros(
        (len(flat), len(selected)),
        dtype=torch.float32,
    )
    distributed_projection = torch.zeros(
        len(flat),
        dtype=torch.float32,
    )
    selected_lookup = {
        int(feature): column for column, feature in enumerate(selected)
    }
    k = int(sae_checkpoint["config"]["k"])
    cosine_cuda = cosine.to("cuda")
    with torch.inference_mode():
        for start in range(0, len(flat), 256):
            end = min(start + 256, len(flat))
            batch = flat[start:end].to("cuda", dtype=torch.float32)
            pre = (batch - b_dec) @ w_enc.T + b_enc
            values, indices = torch.topk(pre, k=k, dim=-1)
            values = torch.relu(values)
            distributed_projection[start:end] = torch.sum(
                values * cosine_cuda[indices],
                dim=-1,
            ).cpu()
            for feature, column in selected_lookup.items():
                selected_values[start:end, column] = torch.sum(
                    values * (indices == feature),
                    dim=-1,
                ).cpu()
    selected_values = selected_values.reshape(
        panel.shape[0],
        panel.shape[1],
        panel.shape[2],
        len(selected),
    ).numpy()
    distributed_projection = distributed_projection.reshape(
        panel.shape[0],
        panel.shape[1],
        panel.shape[2],
    ).numpy()

    sae_curves = {}
    sae_summaries = {}
    for feature in selected:
        column = selected_lookup[int(feature)]
        points = paired_scalar_curve(
            selected_values[:, 0, :, column],
            selected_values[:, 1, :, column],
            offsets,
        )
        name = f"f{int(feature)}"
        sae_curves[name] = curve_to_dict(points)
        sae_summaries[name] = curve_summary(points)

    cosine_rows = [
        {
            "feature": int(feature),
            "decoder_cosine_to_base_union": float(cosine[feature]),
            "decoder_cosine_absolute": float(abs(cosine[feature])),
        }
        for feature in absolute_order[:64].tolist()
    ]
    distributed_points = paired_scalar_curve(
        distributed_projection[:, 0],
        distributed_projection[:, 1],
        offsets,
    )
    prior_sae_result = (
        json.loads(sae_result_cache_path.read_text())
        if sae_result_cache_path.exists()
        else None
    )

    base_vectors = dom["base"]["vectors"].float()
    reasoning_vectors = dom["reasoning"]["vectors"].float()
    direction_cosines = torch.nn.functional.cosine_similarity(
        base_vectors[:, None, :],
        reasoning_vectors[None, :, :],
        dim=-1,
    )
    payload = {
        "method": (
            "prespecified causal Ward-direction formation plus "
            "decoder-aligned conventional SAE feature"
        ),
        "model": activation_cache["model"],
        "layer": int(activation_cache["layer"]),
        "n_pairs": int(panel.shape[0]),
        "offsets": offsets.tolist(),
        "pairing": {
            "event": "first usable genuine-backtracking sentence",
            "neutral": "distant same-rollout position",
            "records": activation_cache["pair_records"],
        },
        "known_directions": {
            "primary": "base_union",
            "reason": (
                "The base-derived union is prespecified and causally steers "
                "the reasoning model, while avoiding a direction fit in the "
                "reasoning-model representation used for this panel."
            ),
            "curves": direction_curves,
            "summaries": direction_summaries,
            "meta": direction_meta,
            "cross_offset": cross_offset,
            "base_reasoning_offset_cosine": {
                "base_offsets": [
                    int(value) for value in dom["base"]["offsets"]
                ],
                "reasoning_offsets": [
                    int(value) for value in dom["reasoning"]["offsets"]
                ],
                "matrix": direction_cosines.tolist(),
            },
        },
        "conventional_sae": {
            "checkpoint": str(sae_cache_path),
            "architecture": "single-token TopK SAE",
            "config": sae_checkpoint["config"],
            "best_positive_feature": best_positive,
            "best_positive_cosine": float(cosine[best_positive]),
            "best_absolute_feature": best_absolute,
            "best_absolute_cosine": float(cosine[best_absolute]),
            "top_alignment": cosine_rows,
            "tracked_features": [int(value) for value in selected],
            "curves": sae_curves,
            "summaries": sae_summaries,
            "distributed_projection": {
                "definition": (
                    "sum_j z_j cosine(decoder_j, base_union), over the "
                    "token's active TopK features; equivalently the "
                    "base-union projection of the SAE reconstruction after "
                    "dropping the constant decoder bias"
                ),
                "curve": curve_to_dict(distributed_points),
                "summary": curve_summary(distributed_points),
            },
            "prior_reconstruction": (
                None
                if prior_sae_result is None
                else prior_sae_result["sae"]["reconstruction"]
            ),
        },
        "interpretation_guardrails": [
            (
                "The DoM directions were derived from labeled sentence "
                "activations, so same-model curves are descriptive rather "
                "than held-out discovery."
            ),
            (
                "The base-derived direction evaluated in the reasoning model "
                "is the cleaner primary transfer curve and has an independent "
                "causal steering result."
            ),
            (
                "A transient peak followed by decay is compatible with a "
                "feature being consumed or transformed by later layers/tokens."
            ),
            (
                "The SAE checkpoint has weak reconstruction quality; failure "
                "of one aligned latent is evidence about this SAE, not proof "
                "that no sparse causal feature exists."
            ),
        ],
        "runtime": {
            "gpu": torch.cuda.get_device_name(0),
            "wall_seconds": time.time() - started,
        },
    }
    cache_path = Path("/cache/ward_known_feature_formation_v1.json")
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    cache_volume.commit()
    print(
        json.dumps(
            {
                "n_pairs": payload["n_pairs"],
                "base_union": direction_summaries["base_union"],
                "reasoning_union": direction_summaries[
                    "reasoning_union"
                ],
                "best_positive_sae_feature": best_positive,
                "best_positive_sae_cosine": float(
                    cosine[best_positive]
                ),
                "best_positive_sae_summary": sae_summaries[
                    f"f{best_positive}"
                ],
                "wall_seconds": payload["runtime"]["wall_seconds"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return payload


@app.local_entrypoint()
def main():
    payload = run.remote()
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[saved] {RESULT}")
