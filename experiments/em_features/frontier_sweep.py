"""Frontier sweep: bundles top-k features into a steering direction, runs the
em-features longform-steering loop at multiple alphas, scores with the OpenAI
judge, and writes incremental JSON.

Replaces the previous ``feature_ablation/frontier_sweep.py`` that was lost
when the GPU host was reimaged. Wraps
``open_source_em_features.pipeline.longform_steering`` directly so the eval
math matches what produced the existing JSON results.

    uv run python -m experiments.em_features.frontier_sweep \\
        --steerer txc --model qwen --layer 15 \\
        --features_json .../top_200_features_layer_15.json \\
        --txc_ckpt /root/em_features/checkpoints/qwen_l15_txc_brickenauxk_a8_step10000.pt \\
        --k 10 --alpha_grid -10 -8 -6 -5 -4 -3 -2 -1.5 -1 1 2 5 \\
        --n_rollouts 8 --out_path /root/em_features/results/.../frontier.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_SRC = REPO_ROOT / "experiments" / "separation_scaling" / "vendor" / "src"
EM_FEATURES = Path(os.environ.get("EM_FEATURES_REPO", "/root/em_features"))
for p in (str(VENDOR_SRC), str(REPO_ROOT), str(EM_FEATURES)):
    if p not in sys.path:
        sys.path.insert(0, p)

from sae_day.sae import TemporalCrosscoder, MultiLayerCrosscoder, TopKSAE  # noqa: E402
from experiments.em_features.architectures.txc_bare_multidistance_contrastive_antidead import (  # noqa: E402
    TXCBareMultiDistanceContrastiveAntidead,
)

from open_source_em_features.pipeline.longform_steering import (  # noqa: E402
    load_em_dataset,
    generate_longform_completions,
    evaluate_generations_with_openai,
)
from open_source_em_features.utils.model_loading import load_model_and_tokenizer  # noqa: E402
from experiments.em_features.gemini_judge import evaluate_generations_with_gemini  # noqa: E402


MODEL_REGISTRY = {
    "qwen": {
        "subject": "andyrdt/Qwen2.5-7B-Instruct_bad-medical",
        "base": "Qwen/Qwen2.5-7B-Instruct",
    },
    "llama": {
        "subject": "andyrdt/Llama-3.2-3B-Instruct_bad-medical",
        "base": "meta-llama/Llama-3.2-3B-Instruct",
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steerer", choices=["sae", "txc", "mlc", "custom_sae", "han", "vec"], required=True)
    p.add_argument("--model", choices=list(MODEL_REGISTRY), default="qwen")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--features_json", type=Path, required=True,
                   help="JSON with top features (schema from run_find_misalignment_features*)")
    p.add_argument("--txc_ckpt", type=Path, default=None)
    p.add_argument("--mlc_ckpt", type=Path, default=None)
    p.add_argument("--custom_sae_ckpt", type=Path, default=None)
    p.add_argument("--han_ckpt", type=Path, default=None,
                   help="For --steerer han: TXCBareMultiDistanceContrastiveAntidead checkpoint")
    p.add_argument("--directions_path", type=Path, default=None,
                   help="For --steerer vec: a .pt file with a (n_features, d_in) tensor.")
    p.add_argument("--k", type=int, default=10, help="Number of top features to bundle")
    p.add_argument("--alpha_grid", type=float, nargs="+", required=True)
    p.add_argument("--n_rollouts", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--out_path", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--judge", choices=["openai", "gemini"], default="gemini",
                   help="Which judge backend to use (default: gemini, bypasses gpt-4o-mini RPD)")
    p.add_argument("--judge_model", default="gemini-3.1-flash-lite-preview",
                   help="Judge model name (gemini-3.1-flash-lite-preview, gemini-3.1-pro-preview, etc.)")
    p.add_argument("--seed", type=int, default=42,
                   help="Master seed; reset before each α so per-α results are reproducible across runs.")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def seed_all(seed: int) -> None:
    """Reset all RNGs to a known state for reproducible generation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_top_feature_ids(features_json: Path, k: int) -> list[int]:
    """Read top features from the JSON written by run_find_misalignment_features*.

    Schema (TXC/MLC/SAE finder): {"layer": L, "features": [{"feature_id": int, "score": float}, ...]}
    or                            {"top_features": [int, int, ...], "scores": [...]}.
    Both seen in prior code — handle both.
    """
    d = json.loads(features_json.read_text())
    if "features" in d:
        return [int(f["feature_id"]) for f in d["features"][:k]]
    if "top_features" in d:
        return [int(i) for i in d["top_features"][:k]]
    raise KeyError(f"Unrecognized feature-json schema in {features_json}: keys={list(d)}")


def get_directions(args, layer: int, device: str) -> torch.Tensor:
    """Returns (n_features, d_in) tensor of per-feature decoder directions
    at the steered layer. Indexed by feature_id."""
    if args.steerer == "txc":
        ckpt = torch.load(args.txc_ckpt, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        m = TemporalCrosscoder(d_in=cfg["d_in"], d_sae=cfg["d_sae"], T=cfg["T"], k_total=cfg["k_total"]).to(device)
        m.load_state_dict(ckpt["state_dict"])
        # last-position decoder column is the canonical TXC steering direction
        return m.W_dec[-1].detach()  # (d_sae, d_in)
    if args.steerer == "mlc":
        ckpt = torch.load(args.mlc_ckpt, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        m = MultiLayerCrosscoder(d_in=cfg["d_in"], d_sae=cfg["d_sae"], L=cfg["L"], k_total=cfg["k_total"]).to(device)
        m.load_state_dict(ckpt["state_dict"])
        layers = list(cfg["layers"])
        if layer not in layers:
            raise ValueError(f"--layer {layer} not in MLC ckpt layers {layers}")
        slot = layers.index(layer)
        return m.W_dec[slot].detach()  # (d_sae, d_in)
    if args.steerer == "custom_sae":
        ckpt = torch.load(args.custom_sae_ckpt, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        m = TopKSAE(d_in=cfg["d_in"], d_sae=cfg["d_sae"], k=cfg["k"]).to(device)
        m.load_state_dict(ckpt["state_dict"])
        return m.W_dec.detach()  # (d_sae, d_in)
    if args.steerer == "han":
        ckpt = torch.load(args.han_ckpt, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        m = TXCBareMultiDistanceContrastiveAntidead(
            d_in=cfg["d_in"], d_sae=cfg["d_sae"], T=cfg["T"], k=cfg["k"],
            shifts=tuple(cfg.get("shifts", (1, 2))),
            matryoshka_h_size=cfg.get("matryoshka_h_size", cfg["d_sae"] // 5),
            alpha=cfg.get("alpha_contrastive", 1.0),
            aux_k=cfg.get("aux_k", 512),
            dead_threshold_tokens=cfg.get("dead_threshold_tokens", 640_000),
            auxk_alpha=cfg.get("auxk_alpha", 1.0 / 32.0),
        ).to(device)
        m.load_state_dict(ckpt["state_dict"])
        # W_dec shape is (d_sae, T, d_in); last temporal slot is the canonical steering dir.
        return m.W_dec[:, -1, :].detach()  # (d_sae, d_in)
    if args.steerer == "vec":
        return torch.load(args.directions_path, map_location=device).to(device)
    raise NotImplementedError(f"steerer={args.steerer} not implemented (need 'sae' loader)")


def bundle(directions: torch.Tensor, ids: list[int]) -> torch.Tensor:
    """Sum the rows for given feature ids; return as (d_in,) float on same device.

    NOTE: do NOT unit-normalize here. ``generate_longform_completions``
    internally renormalizes (``normalized_direction = steering_direction /
    (norm + 1e-6)`` then passes ``coefficients=magnitude``), so passing a
    pre-normalized vector silently scales α by 1/(sum-norm) — for k=10
    unit-norm decoder rows the sum has norm ≈ √10, so unit-norming
    weakens steering ~3× at the same α value.
    """
    rows = directions[ids]  # (k, d_in)
    bundled = rows.sum(dim=0).float()
    return bundled


def main():
    args = parse_args()
    args.out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[frontier] steerer={args.steerer} model={args.model} layer={args.layer}", flush=True)
    print(f"[frontier] features_json={args.features_json}  k={args.k}", flush=True)

    feat_ids = load_top_feature_ids(args.features_json, args.k)
    print(f"[frontier] top-{args.k} feature_ids: {feat_ids}", flush=True)

    directions = get_directions(args, args.layer, args.device)
    bundled = bundle(directions, feat_ids).to(torch.bfloat16)  # match model dtype downstream
    print(f"[frontier] bundled direction shape={tuple(bundled.shape)}  norm={float(bundled.norm()):.4f}", flush=True)

    print(f"[frontier] loading subject model {MODEL_REGISTRY[args.model]['subject']}", flush=True)
    model, tok = load_model_and_tokenizer(
        MODEL_REGISTRY[args.model]["subject"],
        base_model_id=MODEL_REGISTRY[args.model]["base"],
        torch_dtype=torch.bfloat16,
        device=args.device,
    )
    # ActivationSteerer's _POSSIBLE_LAYER_ATTRS only knows "model.layers", but
    # PeftModelForCausalLM nests them at base_model.model.model.layers. Merging
    # the adapter unwraps the PEFT layer and exposes the plain HF model.
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
        print("[frontier] merged PEFT adapter (model is now plain HF)", flush=True)
    model.eval()

    em = load_em_dataset()
    questions = [d["messages"][0]["content"] for d in em]

    rows: list[dict] = []
    out = {"meta": {"steerer": args.steerer, "layer": args.layer, "k": args.k,
                    "feature_ids": feat_ids, "n_rollouts": args.n_rollouts,
                    "alpha_grid": list(args.alpha_grid)}, "rows": rows}

    for alpha in args.alpha_grid:
        tag = f"{args.steerer}_k{args.k}_alpha{alpha:+.2f}"
        print(f"\n=== {tag} ===", flush=True)
        # Seed deterministically per-α so the same generations are produced across runs
        # (e.g., α=0 across different ckpts → identical questions × rollouts).
        seed_all(args.seed)
        gens = generate_longform_completions(
            model=model, tokenizer=tok, questions=questions,
            steering_direction=bundled, magnitude=float(alpha),
            layer_idx=int(args.layer), n_generations=int(args.n_rollouts),
            max_new_tokens=int(args.max_new_tokens), temperature=1.0,
        )
        print(f"  Completed {tag}: {len(gens)} total responses generated", flush=True)
        if args.judge == "gemini":
            align_scores, coh_scores = asyncio.run(
                evaluate_generations_with_gemini(gens, model_name=args.judge_model)
            )
        else:
            align_scores, coh_scores = asyncio.run(evaluate_generations_with_openai(gens))
        # Filter NaN / None (CODE/REFUSAL) per em-features convention
        a_clean = [float(s) for s in align_scores if s is not None and not (isinstance(s, float) and (s != s))]
        c_clean = [float(s) for s in coh_scores if s is not None and not (isinstance(s, float) and (s != s))]
        mean_a = sum(a_clean) / len(a_clean) if a_clean else float("nan")
        mean_c = sum(c_clean) / len(c_clean) if c_clean else float("nan")
        print(f"  α={alpha:+.2f}  align={mean_a}  coh={mean_c}", flush=True)
        rows.append({
            "alpha": float(alpha),
            "mean_alignment": mean_a,
            "mean_coherence": mean_c,
            "n_alignment": len(a_clean),
            "n_coherence": len(c_clean),
            "n_total": len(gens),
        })
        # Incremental write so a crash in mid-sweep still leaves usable rows
        args.out_path.write_text(json.dumps(out, indent=2))

    print(f"\nwrote {args.out_path}", flush=True)


if __name__ == "__main__":
    main()
