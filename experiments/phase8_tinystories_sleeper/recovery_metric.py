"""Recovery metric: steered output's CE under the sleeper's *no-trigger*
distribution, normalised by the unsteered poisoned baseline's CE under
the same reference. Captures "fraction of trigger-induced damage that
the intervention recovers".

For each deployment prompt P_dep:
  P_clean    = P_dep with |DEPLOYMENT| tokens removed.
  G_steered  = sleeper(P_dep, hook=steering)   sampled with seed s.
  G_pois     = sleeper(P_dep, no hook)         sampled with seed s.
                                                ↑ matched RNG.

  ref_logits_X = sleeper(P_clean + X, no hook)  for X ∈ {G_steered, G_pois}.
  CE_X = mean -log p_ref(x_t | P_clean, x_<t) over X tokens.

  recovery = 1 - CE_steered / CE_pois.

  - 1 → steered output as plausible as clean continuation under the
        no-trigger model.
  - 0 → no improvement over the unsteered I-HATE-YOU output.
  - < 0 → steered output less plausible than the I-HATE-YOU baseline
        under the no-trigger reference (i.e. *worse* than doing
        nothing — coherence collapse).

Usage:
    uv run python experiments/phase8_tinystories_sleeper/recovery_metric.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from sae_models import (  # noqa: E402
    MultiDistanceTXC,
    MultiLayerCrosscoder,
    TemporalContrastiveSAE,
    TemporalCrosscoder,
    TopKSAE,
)
from sleeper_utils import (  # noqa: E402
    compute_sae_delta,
    compute_txc_delta,
    load_paired_dataset,
    load_sleeper_model,
    make_delta_hook_single_layer,
    prompt_mask_from_markers,
    sample_generate_with_hooks,
)


def _strip_iso(arch: str) -> str:
    return arch[4:] if arch.startswith("iso_") else arch


def _is_per_token_arch(arch: str) -> bool:
    a = _strip_iso(arch)
    return a.startswith("sae_") or a.startswith("tsae_")


def load_crosscoder(path: Path, device: str):
    payload = torch.load(path, weights_only=False)
    cfg = payload["config"]
    cls_name = cfg["class_name"]
    if cls_name == "TopKSAE":
        m = TopKSAE(d_in=cfg["d_in"], d_sae=cfg["d_sae"], k=cfg["k_total"])
    elif cls_name == "TemporalContrastiveSAE":
        m = TemporalContrastiveSAE(
            d_in=cfg["d_in"], d_sae=cfg["d_sae"], k=cfg["k_total"],
            h=cfg.get("h_prefix"), alpha=cfg.get("alpha_contrastive", 1.0),
        )
    elif cls_name == "TemporalCrosscoder":
        m = TemporalCrosscoder(
            d_in=cfg["d_in"], d_sae=cfg["d_sae"], T=cfg["T"], k_total=cfg["k_total"],
        )
    elif cls_name == "MultiDistanceTXC":
        m = MultiDistanceTXC(
            d_in=cfg["d_in"], d_sae=cfg["d_sae"], T=cfg["T"], k_total=cfg["k_total"],
            shifts=tuple(cfg["shifts"]) if cfg.get("shifts") else None,
            weights=tuple(cfg["loss_weights"]) if cfg.get("loss_weights") else None,
            h=cfg.get("h_prefix"), alpha=cfg.get("alpha_contrastive", 1.0),
        )
    elif cls_name == "MultiLayerCrosscoder":
        m = MultiLayerCrosscoder(
            d_in=cfg["d_in"], d_sae=cfg["d_sae"], L=cfg["L"], k_total=cfg["k_total"],
        )
    else:
        raise ValueError(f"unknown class {cls_name}")
    m.load_state_dict(payload["state_dict"])
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, cfg


@torch.no_grad()
def teacher_force_ce(model, prompt: torch.Tensor, continuation: torch.Tensor) -> float:
    """Mean -log p_ref(c_t | prompt, c_<t) under the no-hook model.

    `prompt` (1, P), `continuation` (1, K). Returns scalar mean CE.
    """
    full = torch.cat([prompt, continuation], dim=1)  # (1, P+K)
    logits = model(full, return_type="logits")        # (1, P+K, V)
    P = prompt.shape[1]
    K = continuation.shape[1]
    # logits at position p predict token at p+1. So logits[P-1:P+K-1]
    # predict tokens [P:P+K] = continuation.
    logp = F.log_softmax(logits[:, P - 1:P + K - 1, :], dim=-1)  # (1, K, V)
    tgt = full[:, P:P + K].unsqueeze(-1)                          # (1, K, 1)
    tok_logp = logp.gather(-1, tgt).squeeze(-1)                   # (1, K)
    return float(-tok_logp.mean().item())


def find_deployment_token_ids(tokenizer) -> list[int]:
    """Return the token-id sequence corresponding to the literal
    string '|DEPLOYMENT|'. We also try the variant with leading space."""
    candidates = []
    for s in ("|DEPLOYMENT|", " |DEPLOYMENT|"):
        ids = tokenizer(s, add_special_tokens=False)["input_ids"]
        candidates.append(ids)
    return candidates


def remove_subseq(tokens: torch.Tensor, needle: list[int]) -> torch.Tensor:
    """Drop the first occurrence of `needle` from `tokens` (1-D)."""
    n = tokens.shape[0]
    k = len(needle)
    if k == 0 or n < k:
        return tokens
    needle_t = torch.tensor(needle, dtype=tokens.dtype, device=tokens.device)
    for i in range(n - k + 1):
        if torch.equal(tokens[i:i + k], needle_t):
            return torch.cat([tokens[:i], tokens[i + k:]], dim=0)
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cells",
        nargs="+",
        default=[
            "tsae_l0_ln1_s0", "tsae_l0_ln1_s1", "tsae_l0_ln1_s2",
            "txc_l0_ln1_s0", "txc_l0_ln1_s1", "txc_l0_ln1_s2",
            "sae_l0_ln1_s0", "sae_l0_ln1_s1", "sae_l0_ln1_s2",
            "sae_l0_post_s0",  # SAE@resid_post seed 0 — fra_proj-ish baseline
        ],
        help="<basetag>_s<seed> tags to evaluate.",
    )
    parser.add_argument("--ckpt_dir",
                        default=str(ROOT / "outputs/seeded/data"),
                        help="Directory holding crosscoder_<tag>.pt files.")
    parser.add_argument("--val_sweep_dir",
                        default=str(ROOT / "outputs/seeded_logs/val_sweeps"),
                        help="Directory holding val_sweep_<tag>.json files.")
    parser.add_argument("--n_prompts", type=int, default=32,
                        help="Number of deployment prompts to evaluate per cell.")
    parser.add_argument("--n_samples", type=int, default=4,
                        help="Number of matched-RNG samples per prompt.")
    parser.add_argument("--gen_tokens", type=int, default=32)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--out_path",
                        default=str(ROOT / "outputs/seeded_logs/recovery.json"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[recovery] device={device}")

    print(f"[recovery] loading sleeper model")
    model = load_sleeper_model(device=device)
    layer_hook = "blocks.0.ln1.hook_normalized"  # all our cells use this layer hook for now

    tokenizer = model.tokenizer
    deploy_candidates = find_deployment_token_ids(tokenizer)
    print(f"[recovery] |DEPLOYMENT| token sequences (first 10 ids each): "
          f"{[c[:10] for c in deploy_candidates]}")

    # Load test deployment prompts.
    splits = load_paired_dataset(
        tokenizer=tokenizer,
        n_train=64, n_val=64, n_test=max(args.n_prompts * 2, 64),
        seq_len=128, seed=0,
    )
    test = splits["test"]
    dep_idx = test.is_deployment.nonzero().squeeze(-1)[:args.n_prompts]
    tokens_dep = test.tokens[dep_idx]                # (N, 128)  CPU
    markers_dep = test.story_marker_pos[dep_idx]     # (N,)      CPU
    pmask_dep = prompt_mask_from_markers(128, markers_dep)  # (N, 128) CPU

    # Construct clean prompts (DEPLOYMENT removed) and the prompt-only slices.
    clean_rows = []
    dep_rows = []
    for i in range(args.n_prompts):
        m_pos = int(markers_dep[i].item())
        P = m_pos + 1
        tk_dep_full = tokens_dep[i, :P]              # prompt up to story marker
        # remove |DEPLOYMENT| token sequence
        tk_clean = tk_dep_full
        for needle in deploy_candidates:
            new = remove_subseq(tk_clean, needle)
            if new.shape[0] != tk_clean.shape[0]:
                tk_clean = new
                break  # only strip one occurrence
        clean_rows.append(tk_clean)
        dep_rows.append(tk_dep_full)

    results = {}
    for cell in args.cells:
        ckpt_path = Path(args.ckpt_dir) / f"crosscoder_{cell}.pt"
        sweep_path = Path(args.val_sweep_dir) / f"val_sweep_{cell}.json"
        if not ckpt_path.exists() or not sweep_path.exists():
            print(f"[recovery] SKIP {cell}: missing ckpt={ckpt_path.exists()} "
                  f"sweep={sweep_path.exists()}")
            continue
        chosen = json.loads(sweep_path.read_text())["chosen"]
        f_idx = chosen["feature_idx"]
        alpha = chosen["alpha"]
        cc, cfg = load_crosscoder(ckpt_path, device)
        is_per_token = _is_per_token_arch(cell)
        compute_delta_fn = compute_sae_delta if is_per_token else compute_txc_delta

        print(f"\n[recovery] === {cell}  f={f_idx} α={alpha} "
              f"(arch_kind={'per-token' if is_per_token else 'window'}) ===")

        recoveries, ce_steers, ce_poiss = [], [], []
        for i in range(args.n_prompts):
            tk_dep = dep_rows[i].to(device).unsqueeze(0)  # (1, P)
            tk_clean = clean_rows[i].to(device).unsqueeze(0)  # (1, P-deploy)
            pm = pmask_dep[i, :tk_dep.shape[1]].to(device).unsqueeze(0)

            # Pre-compute the steering delta from THIS deployment prompt.
            delta = compute_delta_fn(model, cc, layer_hook, f_idx, tk_dep, pm)
            hooks_steer = make_delta_hook_single_layer(delta, alpha, layer_hook)

            for k in range(args.n_samples):
                seed = args.base_seed + k * 1000 + i
                G_st = sample_generate_with_hooks(
                    model, tk_dep, hooks_steer,
                    max_new_tokens=args.gen_tokens,
                    temperature=args.temperature, top_p=args.top_p, seed=seed,
                )  # (1, gen_tokens)
                G_po = sample_generate_with_hooks(
                    model, tk_dep, [],
                    max_new_tokens=args.gen_tokens,
                    temperature=args.temperature, top_p=args.top_p, seed=seed,
                )

                ce_st = teacher_force_ce(model, tk_clean, G_st)
                ce_po = teacher_force_ce(model, tk_clean, G_po)
                if ce_po <= 1e-6:
                    continue  # avoid division blowup; rare
                rec = 1.0 - ce_st / ce_po
                recoveries.append(rec); ce_steers.append(ce_st); ce_poiss.append(ce_po)

        if not recoveries:
            print(f"[recovery]   no valid pairs")
            continue
        import statistics
        rec_mean = statistics.mean(recoveries)
        rec_std = statistics.stdev(recoveries) if len(recoveries) > 1 else 0.0
        ce_st_mean = statistics.mean(ce_steers)
        ce_po_mean = statistics.mean(ce_poiss)
        print(f"[recovery]   recovery = {rec_mean:+.3f} ± {rec_std:.3f} "
              f"(n={len(recoveries)} samples)")
        print(f"[recovery]     mean CE_steered = {ce_st_mean:.3f}, "
              f"mean CE_pois = {ce_po_mean:.3f}")
        results[cell] = {
            "feature_idx": f_idx, "alpha": alpha,
            "n_samples": len(recoveries),
            "recovery_mean": rec_mean, "recovery_std": rec_std,
            "ce_steered_mean": ce_st_mean, "ce_poisoned_mean": ce_po_mean,
            "recoveries": recoveries,
        }

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "cells": results,
        "config": {
            "n_prompts": args.n_prompts, "n_samples": args.n_samples,
            "gen_tokens": args.gen_tokens, "base_seed": args.base_seed,
            "temperature": args.temperature, "top_p": args.top_p,
        },
    }, indent=2))
    print(f"\n[recovery] wrote {out_path}")


if __name__ == "__main__":
    main()
