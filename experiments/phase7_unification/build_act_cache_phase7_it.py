"""Build the Gemma-2-2b-IT activation cache for Phase 7.

Forked thin-wrapper from `build_act_cache_phase7.py`. Imports paths
from `_paths_it` so the cache lands in
`data/cached_activations/gemma-2-2b-it/fineweb/` and uses the IT
subject model + L11..L15 MLC layer set + L13 anchor.

Reuses the BASE `cache_single_layer` machinery — only the paths and
the layer_specs metadata differ. Same single-layer-per-invocation +
resumable design (MooseFS-friendly).

Run once per layer in {11, 12, 13, 14, 15}:

    .venv/bin/python -m experiments.phase7_unification.build_act_cache_phase7_it --layer 13
    .venv/bin/python -m experiments.phase7_unification.build_act_cache_phase7_it --layer 11
    ... etc.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("TQDM_DISABLE", "1")

from experiments.phase7_unification._paths_it import (
    CACHE_DIR, MLC_LAYERS, ANCHOR_LAYER, SUBJECT_MODEL, banner,
)
from experiments.phase7_unification.build_act_cache_phase7 import (
    cache_single_layer,
)


def write_layer_specs_it(cache_dir: Path) -> None:
    specs_path = cache_dir / "layer_specs.json"
    if specs_path.exists():
        specs = json.loads(specs_path.read_text())
    else:
        specs = {
            "model": SUBJECT_MODEL.split("/")[-1],   # gemma-2-2b-it
            "hf_path": SUBJECT_MODEL,
            "d_model": 2304,
            "layer_specs": {},
            "mode": "forward",
            "anchor_layer_0idx": ANCHOR_LAYER,
            "mlc_layers_0idx": list(MLC_LAYERS),
        }
    for layer in MLC_LAYERS:
        if (cache_dir / f"resid_L{layer}.npy").exists():
            key = f"resid_L{layer}"
            specs["layer_specs"][key] = {
                "layer": layer, "component": "resid",
                "d_act": 2304, "label": key, "family": "gemma",
            }
    specs_path.write_text(json.dumps(specs, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True,
                    choices=list(MLC_LAYERS),
                    help="0-indexed layer in {11..15}")
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--flush-every", type=int, default=1_000)
    args = ap.parse_args()
    banner(__file__)
    cache_single_layer(
        layer=args.layer, cache_dir=args.cache_dir,
        model_name=SUBJECT_MODEL,
        batch_size=args.batch_size, flush_every=args.flush_every,
    )
    write_layer_specs_it(args.cache_dir)


if __name__ == "__main__":
    main()
