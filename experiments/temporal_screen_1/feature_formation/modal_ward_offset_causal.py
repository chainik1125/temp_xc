"""Causal check for Ward directions derived at different token offsets.

Generation delegates to the repository's matched Ward implementation in
``experiments.ward_backtracking.steer_eval`` so prompt wrapping, hook
placement, decoding, and the keyword metric remain identical to Stage A.

To keep this calibration inexpensive, the run evaluates one prespecified
effective magnitude (+12) on the held-out 20-prompt split.  The exact
magnitude-zero baseline is reused from the completed Stage A artifact.

Run from the repository root:

    modal run \
      experiments/temporal_screen_1/feature_formation/modal_ward_offset_causal.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2] if len(HERE.parents) > 2 else HERE
WARD = REPO / "experiments" / "ward_backtracking"
DATA = REPO / "results" / "ward_backtracking"
RESULT = HERE / "results" / "ward_offset_causal_efficacy.json"

app = modal.App("temporal-screen-ward-offset-causal")
hf_secret = modal.Secret.from_name("hf-token")
cache_volume = modal.Volume.from_name(
    "temporal-screen-ward-weak-label-cache",
    create_if_missing=False,
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.7.1",
        "transformers==4.53.3",
        "accelerate==1.9.0",
        "huggingface-hub>=0.33",
        "hf-xet>=1.1",
        "numpy==2.2.6",
        "pyyaml==6.0.2",
    )
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_dir(
        str(WARD),
        "/work/experiments/ward_backtracking",
    )
    .add_local_file(
        str(DATA / "dom_vectors.pt"),
        "/work/data/dom_vectors.pt",
    )
    .add_local_file(
        str(DATA / "prompts.json"),
        "/work/data/prompts.json",
    )
    .add_local_file(
        str(DATA / "steering_results.json"),
        "/work/data/steering_results.json",
    )
)


def _aggregate(rows: list[dict]) -> dict:
    import numpy as np

    grouped = {}
    for row in rows:
        grouped.setdefault(row["source"], []).append(row)
    return {
        source: {
            "n_prompts": len(source_rows),
            "mean_keyword_rate": float(
                np.mean([row["keyword_rate"] for row in source_rows])
            ),
            "any_keyword_fraction": float(
                np.mean([row["wait_count"] > 0 for row in source_rows])
            ),
            "mean_keyword_count": float(
                np.mean([row["wait_count"] for row in source_rows])
            ),
            "mean_words": float(
                np.mean([row["n_words"] for row in source_rows])
            ),
        }
        for source, source_rows in grouped.items()
    }


@app.function(
    image=image,
    gpu="A100-40GB",
    cpu=12,
    memory=65_536,
    timeout=6 * 60 * 60,
    secrets=[hf_secret],
    volumes={"/cache": cache_volume},
)
def run() -> dict:
    import sys
    import time

    import torch

    sys.path.insert(0, "/work")
    from experiments.ward_backtracking.steer_eval import (
        _load_eval_prompts,
        _run_target,
    )

    started = time.time()
    dom = torch.load(
        "/work/data/dom_vectors.pt",
        map_location="cpu",
        weights_only=False,
    )
    base_union = dom["base"]["union"].float()
    union_norm = base_union.norm()
    sources = {}
    source_meta = {}
    for source_offset, vector in zip(
        dom["base"]["offsets"],
        dom["base"]["vectors"],
        strict=True,
    ):
        name = f"base_derived_off{int(source_offset):+d}_raw"
        sources[name] = vector.float()
        source_meta[name] = {
            "kind": "Ward DoM",
            "derivation_offset": int(source_offset),
            "norm": float(vector.norm()),
            "scale": "raw Stage-A vector",
        }
    sources["base_derived_union_raw"] = base_union
    source_meta["base_derived_union_raw"] = {
        "kind": "Ward DoM union",
        "derivation_offsets": [
            int(value) for value in dom["base"]["union_offsets"]
        ],
        "norm": float(union_norm),
        "scale": "raw Stage-A vector",
    }

    off_zero_index = list(dom["base"]["offsets"]).index(0)
    off_zero = dom["base"]["vectors"][off_zero_index].float()
    sources["base_derived_off+0_normmatched"] = (
        off_zero * union_norm / off_zero.norm().clamp_min(1e-12)
    )
    source_meta["base_derived_off+0_normmatched"] = {
        "kind": "Ward DoM norm-matched control",
        "derivation_offset": 0,
        "original_norm": float(off_zero.norm()),
        "norm": float(union_norm),
        "scale": "matched to base union norm",
    }

    sae_checkpoint = torch.load(
        "/cache/ward_feature_formation_layer10_topk8192_k32_v1.pt",
        map_location="cpu",
        weights_only=False,
    )
    decoder = sae_checkpoint["state_dict"]["W_dec"].float()
    decoder = decoder / decoder.norm(dim=0, keepdim=True).clamp_min(1e-12)
    base_unit = base_union / union_norm.clamp_min(1e-12)
    cosine = decoder.T @ base_unit
    sae_feature = int(torch.argmax(cosine))
    sae_vector = decoder[:, sae_feature] * union_norm
    sae_name = f"sae_f{sae_feature}_positive_aligned_normmatched"
    sources[sae_name] = sae_vector
    source_meta[sae_name] = {
        "kind": "single conventional TopK SAE decoder direction",
        "feature": sae_feature,
        "decoder_cosine_to_base_union": float(cosine[sae_feature]),
        "norm": float(sae_vector.norm()),
        "scale": "matched to base union norm",
    }

    prompts = _load_eval_prompts(
        Path("/work/data/prompts.json"),
        n=20,
        seed=42,
    )
    rows = _run_target(
        target_tag="reasoning",
        hf_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        layer=10,
        sources=sources,
        magnitudes=[12.0],
        prompts=prompts,
        max_new_tokens=1_500,
        dtype="bfloat16",
        device="cuda",
    )

    previous = json.loads(
        Path("/work/data/steering_results.json").read_text()
    )
    baseline_rows = [
        row
        for row in previous["rows"]
        if row["target"] == "reasoning"
        and row["source"] == "base_derived_union"
        and float(row["magnitude"]) == 0.0
    ]
    if {row["prompt_id"] for row in baseline_rows} != {
        row["id"] for row in prompts
    }:
        raise RuntimeError("reused Stage-A baseline prompt set does not match")
    for row in baseline_rows:
        row = dict(row)
        row["source"] = "stage_a_magnitude_zero_baseline"
        rows.append(row)
    source_meta["stage_a_magnitude_zero_baseline"] = {
        "kind": "exact reused Stage-A baseline",
        "magnitude": 0.0,
    }

    payload = {
        "method": (
            "held-out causal generation with repository Ward reference "
            "implementation"
        ),
        "target": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "layer": 10,
        "magnitude": 12.0,
        "n_eval_prompts": len(prompts),
        "max_new_tokens": 1_500,
        "sources": source_meta,
        "aggregates": _aggregate(rows),
        "rows": rows,
        "runtime": {
            "gpu": torch.cuda.get_device_name(0),
            "wall_seconds": time.time() - started,
        },
        "guardrails": [
            (
                "This is a single-magnitude efficacy slice, not a full "
                "dose-response curve."
            ),
            (
                "Raw band-vector norms are similar; offset zero is also "
                "reported after norm matching because its raw norm is larger."
            ),
            (
                "The keyword metric is the Ward screen metric but is noisier "
                "than a semantic judge of genuine backtracking."
            ),
        ],
    }
    cache_path = Path("/cache/ward_offset_causal_efficacy_v1.json")
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    cache_volume.commit()
    print(
        json.dumps(
            {
                "aggregates": payload["aggregates"],
                "runtime": payload["runtime"],
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
