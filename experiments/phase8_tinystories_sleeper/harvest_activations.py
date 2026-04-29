"""Harvest residual-stream activations from the sleeper TinyStories model.

Runs once up front: loads the paired clean/deployment dataset, forwards through
the sleeper model with hooks on all 5 residual-stream points, caches the
activations + tokens to disk. Downstream scripts read these caches and never
touch the HF dataset again.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))  # experiment dir — sleeper_utils / sae_models live here

from sleeper_utils import (  # noqa: E402
    MLC_HOOK_NAMES,
    cache_activations,
    load_paired_dataset,
    load_sleeper_model,
    prompt_mask_from_markers,
)


def pick_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", type=int, default=10_000)
    parser.add_argument("--n_val", type=int, default=200)
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--output_dir", default=str(ROOT / "outputs" / "data"))
    parser.add_argument(
        "--hook_names",
        nargs="+",
        default=None,
        help="Residual-stream hookpoints to cache. Defaults to the 5-point stack "
             "used by the main experiment (hook_names).",
    )
    args = parser.parse_args()

    hook_names = args.hook_names if args.hook_names else hook_names

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)

    print(f"[harvest] device={device}")
    print(f"[harvest] loading sleeper model…")
    model = load_sleeper_model(device=device)
    print(
        f"[harvest] loaded: n_layers={model.cfg.n_layers} d_model={model.cfg.d_model} "
        f"n_ctx={model.cfg.n_ctx}"
    )

    print(f"[harvest] loading paired dataset (train={args.n_train}, val={args.n_val}, test={args.n_test})…")
    splits = load_paired_dataset(
        tokenizer=model.tokenizer,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        seq_len=args.seq_len,
        seed=args.seed,
    )
    for split_name, pt in splits.items():
        dep_frac = pt.is_deployment.float().mean().item()
        print(
            f"[harvest]   {split_name}: N={pt.tokens.shape[0]} "
            f"dep_frac={dep_frac:.2f} marker_mean={pt.story_marker_pos.float().mean():.1f}"
        )

    print(f"[harvest] caching activations on {hook_names}…")
    split_acts: dict[str, torch.Tensor] = {}
    for split_name, pt in splits.items():
        print(f"[harvest]   {split_name} ({pt.tokens.shape[0]} sequences)")
        acts = cache_activations(
            model=model,
            tokens=pt.tokens,
            hook_names=hook_names,
            chunk_size=args.chunk_size,
        )
        print(f"[harvest]     acts.shape={tuple(acts.shape)} dtype={acts.dtype}")
        split_acts[split_name] = acts

    meta = {
        "base_model": "roneneldan/TinyStories-Instruct-33M",
        "sleeper_model": "mars-jason-25/tiny-stories-33M-TSdata-sleeper",
        "dataset": "mars-jason-25/tiny_stories_instruct_sleeper_data",
        "n_train": args.n_train,
        "n_val": args.n_val,
        "n_test": args.n_test,
        "seq_len": args.seq_len,
        "seed": args.seed,
        "hook_names": hook_names,
        "d_model": int(model.cfg.d_model),
        "n_layers": int(model.cfg.n_layers),
    }

    torch.save(
        {
            "splits": {
                name: {
                    "tokens": pt.tokens,
                    "is_deployment": pt.is_deployment,
                    "story_marker_pos": pt.story_marker_pos,
                }
                for name, pt in splits.items()
            },
            "meta": meta,
        },
        out_dir / "tokens_cache.pt",
    )
    torch.save(
        {name: split_acts[name] for name in splits},
        out_dir / "activations_cache.pt",
    )
    (out_dir / "harvest_meta.json").write_text(json.dumps(meta, indent=2))

    # Sanity-check prompt mask distribution
    for name, pt in splits.items():
        pm = prompt_mask_from_markers(args.seq_len, pt.story_marker_pos)
        print(
            f"[harvest]   {name}: prompt_len mean={pm.float().sum(1).mean().item():.1f} "
            f"min={pm.float().sum(1).min().item():.0f} "
            f"max={pm.float().sum(1).max().item():.0f}"
        )

    print(f"[harvest] wrote {out_dir}/tokens_cache.pt + activations_cache.pt")


if __name__ == "__main__":
    main()
