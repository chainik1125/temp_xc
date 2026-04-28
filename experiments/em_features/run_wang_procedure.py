"""Wang et al. 2026 (arXiv:2506.19823) — full causal feature-attribution procedure.

Stages:
  1. (already done by run_find_features_encoder) — top-200 features by Δz̄
  2. causal screen: steer each top-N feature at small α, check if it moves
     mean alignment vs the unsteered baseline. Keep top-K survivors.
  3. per-survivor coherence-aware sweep: for each survivor, sweep small α
     until coherence drops by >coherence_drop_threshold vs baseline; record
     the strongest steering that stays coherent.
  4. final headline: full 27-α frontier on top-N_FINAL survivors (single-feature steering).

Outputs JSON suitable for plotting alongside the cosine/encoder grids.

  uv run python -m experiments.em_features.run_wang_procedure \\
      --ckpt v2_qwen_l15_sae_arditi_k128_step100000.pt --arch sae \\
      --features_json .../top_200_features.json \\
      --layer 15 --out /root/em_features/results/wang_sae_step100000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_SRC = REPO_ROOT / "experiments" / "separation_scaling" / "vendor" / "src"
for p in (str(VENDOR_SRC), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--arch", choices=["sae", "han"], required=True)
    p.add_argument("--features_json", type=Path, required=True,
                   help="Output of run_find_features_encoder (top-200 by Δz̄).")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)

    # Stage 2 (screen)
    p.add_argument("--screen_top_n", type=int, default=100,
                   help="How many features from top_200 to screen.")
    p.add_argument("--screen_alpha", type=float, default=1.0,
                   help="Small steering coefficient for the screen (Wang: 0.4 in activation-norm units).")
    p.add_argument("--screen_rollouts", type=int, default=2,
                   help="Rollouts per question per screened feature.")

    # Stage 3 (per-survivor coherence-aware sweep)
    p.add_argument("--n_survivors", type=int, default=20,
                   help="Top-K from the screen to keep for stage 3.")
    p.add_argument("--coh_drop_threshold", type=float, default=0.10,
                   help="Stop increasing α when mean coherence drops by this fraction below baseline.")
    p.add_argument("--strength_alpha_grid", type=str, default="-10,-6,-4,-2,-1,1,2,4,6,10",
                   help="Per-feature α grid for stage 3.")
    p.add_argument("--strength_rollouts", type=int, default=4)

    # Stage 4 (final headline)
    p.add_argument("--n_final", type=int, default=3,
                   help="Top-K from stage 3 to do full 27-α frontier on.")
    p.add_argument("--final_alpha_grid", type=str,
                   default="-100,-10,-8,-6,-5,-4,-3,-2,-1.75,-1.5,-1.25,-1,0,1,1.25,1.5,1.75,2,3,4,5,6,7,8,9,10,100")
    p.add_argument("--final_rollouts", type=int, default=8)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--judge_model", default="gemini-3.1-flash-lite-preview")
    p.add_argument("--judge_temperature", type=float, default=0.5)
    p.add_argument("--max_concurrent", type=int, default=12)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip_done", action="store_true",
                   help="Skip stages whose output JSON already exists.")
    return p.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enable_determinism() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_steerer_decoder_row(arch: str, ckpt_path: Path, feature_id: int, device: str) -> torch.Tensor:
    from sae_day.sae import TopKSAE
    from experiments.em_features.architectures.txc_bare_multidistance_contrastive_antidead import (
        TXCBareMultiDistanceContrastiveAntidead,
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    if arch == "sae":
        sae = TopKSAE(d_in=cfg["d_in"], d_sae=cfg["d_sae"], k=cfg["k"]).to(device)
        sae.load_state_dict(ckpt["state_dict"])
        sae.eval()
        # W_dec convention: (d_sae, d_in)
        return sae.W_dec[feature_id].detach().clone()
    else:
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
        m.eval()
        # W_dec for Han: (d_sae, T, d_in) — take the last temporal slot to match frontier_sweep
        return m.W_dec[feature_id, -1, :].detach().clone()


def run_alpha_for_feature(
    *, generate_fn, model, tokenizer, questions, layer: int, direction: torch.Tensor,
    alpha: float, n_rollouts: int, max_new_tokens: int, seed: int,
) -> list[dict]:
    """Generate longform completions at a fixed (feature, α)."""
    seed_all(seed)
    return generate_fn(
        model=model, tokenizer=tokenizer, questions=questions,
        steering_direction=direction, magnitude=float(alpha),
        layer_idx=int(layer), n_generations=int(n_rollouts),
        max_new_tokens=int(max_new_tokens), temperature=1.0,
    )


def main():
    args = parse_args()
    enable_determinism()
    args.out.mkdir(parents=True, exist_ok=True)

    # Load shared infrastructure: bad-medical Qwen + EM questions + judge
    import asyncio
    from open_source_em_features.pipeline.longform_steering import (
        generate_longform_completions, load_em_dataset,
    )
    from open_source_em_features.utils.model_loading import load_model_and_tokenizer
    from experiments.em_features.gemini_judge import evaluate_generations_with_gemini

    # Subject + base model IDs (Qwen-7B bad-medical with PEFT adapter)
    SUBJECT = "andyrdt/Qwen2.5-7B-Instruct_bad-medical"
    BASE = "Qwen/Qwen2.5-7B-Instruct"

    print(f"[wang] arch={args.arch} ckpt={args.ckpt.name} feats={args.features_json}", flush=True)
    print(f"[wang] screen_top_n={args.screen_top_n}  n_survivors={args.n_survivors}  n_final={args.n_final}", flush=True)

    # Load encoder-ranked top features
    feats_d = json.loads(args.features_json.read_text())
    ranked = feats_d["features"][:args.screen_top_n]
    print(f"[wang] loaded {len(ranked)} features (top {args.screen_top_n} by Δz̄)", flush=True)

    # Load subject model + tokenizer once
    print("[wang] loading subject model (bad-medical Qwen)...", flush=True)
    model, tok = load_model_and_tokenizer(
        SUBJECT, base_model_id=BASE, torch_dtype=torch.bfloat16, device=args.device,
    )
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
    model.eval()
    em = load_em_dataset()
    questions = [d["messages"][0]["content"] for d in em]

    # ===== Stage 2: causal screen =====
    screen_path = args.out / "stage2_screen.json"
    if args.skip_done and screen_path.exists():
        print(f"[stage2] skipping (already at {screen_path})", flush=True)
        screen = json.loads(screen_path.read_text())
    else:
        print(f"\n=== STAGE 2: causal screen of {len(ranked)} features at α=±{args.screen_alpha} ===", flush=True)
        screen_rows = []
        for i, f in enumerate(ranked):
            fid = f["feature_id"]
            t0 = time.time()
            direction = load_steerer_decoder_row(args.arch, args.ckpt, fid, args.device)
            results = {}
            for alpha in (-args.screen_alpha, args.screen_alpha):
                gens = run_alpha_for_feature(
                generate_fn=generate_longform_completions,
                    model=model, tokenizer=tok, questions=questions, layer=args.layer,
                    direction=direction, alpha=alpha, n_rollouts=args.screen_rollouts,
                    max_new_tokens=args.max_new_tokens, seed=args.seed,
                )
                a, c = asyncio.run(evaluate_generations_with_gemini(
                    gens, model_name=args.judge_model, max_concurrent=args.max_concurrent,
                    temperature=args.judge_temperature,
                ))
                a_v = [x for x in a if x is not None]
                c_v = [x for x in c if x is not None]
                results[alpha] = {
                    "mean_align": float(np.mean(a_v)) if a_v else None,
                    "mean_coh": float(np.mean(c_v)) if c_v else None,
                    "n_align": len(a_v), "n_total": len(gens),
                }
            # Wang-style screen score: bigger drop in alignment when steered POSITIVE
            # (= activating the misalignment feature) means stronger signal
            ma_pos = results[args.screen_alpha]["mean_align"]
            ma_neg = results[-args.screen_alpha]["mean_align"]
            screen_score = (ma_neg or 0) - (ma_pos or 0)  # positive = useful: + steers toward bad
            row = {
                "rank_z": f["rank"], "feature_id": fid,
                "delta_z": f.get("delta_z"),
                "pos": results[args.screen_alpha], "neg": results[-args.screen_alpha],
                "screen_score": screen_score,
            }
            screen_rows.append(row)
            print(f"  [{i+1:>3d}/{len(ranked)}] feat {fid:>6d} Δz={f.get('delta_z',0):+.3f}  "
                  f"α=+{args.screen_alpha}: align={ma_pos}  α=-{args.screen_alpha}: align={ma_neg}  "
                  f"score={screen_score:+.2f}  ({time.time()-t0:.1f}s)", flush=True)
        screen_rows.sort(key=lambda r: r["screen_score"] or -999, reverse=True)
        screen = {"meta": {"screen_alpha": args.screen_alpha, "n_screened": len(screen_rows),
                           "n_rollouts": args.screen_rollouts}, "rows": screen_rows}
        screen_path.write_text(json.dumps(screen, indent=2))
        print(f"[stage2] wrote {screen_path}", flush=True)

    survivors = screen["rows"][:args.n_survivors]
    print(f"\n[stage2] top-{args.n_survivors} survivors (by screen_score):", flush=True)
    for r in survivors[:10]:
        print(f"  feat {r['feature_id']:>6d}  score={r['screen_score']:+.2f}  Δz={r['delta_z']:+.3f}", flush=True)

    # ===== Stage 3: per-survivor coherence-aware sweep =====
    strength_alphas = [float(x) for x in args.strength_alpha_grid.split(",")]
    strength_path = args.out / "stage3_strength.json"
    if args.skip_done and strength_path.exists():
        print(f"[stage3] skipping (already at {strength_path})", flush=True)
        strength = json.loads(strength_path.read_text())
    else:
        print(f"\n=== STAGE 3: coherence-aware strength sweep on {len(survivors)} survivors ===", flush=True)
        # Baseline coherence at α=0 (compute once)
        gens0 = run_alpha_for_feature(
            generate_fn=generate_longform_completions,
            model=model, tokenizer=tok, questions=questions, layer=args.layer,
            direction=torch.zeros_like(load_steerer_decoder_row(args.arch, args.ckpt, survivors[0]["feature_id"], args.device)),
            alpha=0.0, n_rollouts=args.strength_rollouts,
            max_new_tokens=args.max_new_tokens, seed=args.seed,
        )
        a0, c0 = asyncio.run(evaluate_generations_with_gemini(
            gens0, model_name=args.judge_model, max_concurrent=args.max_concurrent,
            temperature=args.judge_temperature,
        ))
        baseline_coh = float(np.mean([x for x in c0 if x is not None]))
        baseline_align = float(np.mean([x for x in a0 if x is not None]))
        coh_floor = baseline_coh * (1.0 - args.coh_drop_threshold)
        print(f"[stage3] baseline α=0: align={baseline_align:.2f}  coh={baseline_coh:.2f}  coh_floor={coh_floor:.2f}", flush=True)

        # Resume any partial stage3 from earlier run
        partial_path = args.out / "stage3_strength.partial.json"
        strength_rows: list[dict] = []
        completed_ids: set[int] = set()
        if partial_path.exists():
            try:
                strength_rows = json.loads(partial_path.read_text()).get("rows", [])
                completed_ids = {r["feature_id"] for r in strength_rows}
                print(f"[stage3] resuming with {len(strength_rows)} features already done", flush=True)
            except Exception:
                strength_rows = []

        def _save_partial():
            partial_path.write_text(json.dumps(
                {"meta": {"baseline_align": baseline_align, "baseline_coh": baseline_coh,
                          "coh_floor": coh_floor, "alpha_grid": strength_alphas,
                          "completed": len(strength_rows), "total": len(survivors)},
                 "rows": strength_rows}, indent=2))

        for s_idx, s in enumerate(survivors):
            fid = s["feature_id"]
            if fid in completed_ids:
                continue
            direction = load_steerer_decoder_row(args.arch, args.ckpt, fid, args.device)
            curve = []
            best_strong = None
            for alpha in strength_alphas:
                gens = run_alpha_for_feature(
                    generate_fn=generate_longform_completions,
                    model=model, tokenizer=tok, questions=questions, layer=args.layer,
                    direction=direction, alpha=alpha, n_rollouts=args.strength_rollouts,
                    max_new_tokens=args.max_new_tokens, seed=args.seed,
                )
                a, c = asyncio.run(evaluate_generations_with_gemini(
                    gens, model_name=args.judge_model, max_concurrent=args.max_concurrent,
                    temperature=args.judge_temperature,
                ))
                a_v = [x for x in a if x is not None]
                c_v = [x for x in c if x is not None]
                pt = {
                    "alpha": alpha,
                    "mean_align": float(np.mean(a_v)) if a_v else None,
                    "mean_coh": float(np.mean(c_v)) if c_v else None,
                }
                curve.append(pt)
                if pt["mean_coh"] is not None and pt["mean_coh"] >= coh_floor:
                    if best_strong is None or abs(alpha) > abs(best_strong["alpha"]):
                        best_strong = pt
            strength_rows.append({"feature_id": fid, "screen_score": s["screen_score"],
                                  "delta_z": s["delta_z"], "curve": curve,
                                  "best_strong": best_strong})
            _save_partial()  # checkpoint after each feature
            tag = (f"feat {fid:>6d}  best_strong α="
                   f"{best_strong['alpha']:+.2f} align={best_strong['mean_align']:.2f} coh={best_strong['mean_coh']:.2f}"
                   if best_strong else f"feat {fid:>6d}  no coherent α")
            print(f"  [{s_idx+1:>3d}/{len(survivors)}] {tag}", flush=True)

        # Rank survivors by alignment shift away from baseline (None-safe)
        for r in strength_rows:
            bs = r.get("best_strong")
            if bs and bs.get("mean_align") is not None:
                r["align_shift"] = abs(bs["mean_align"] - baseline_align)
            else:
                r["align_shift"] = 0.0
        strength_rows.sort(key=lambda r: r["align_shift"], reverse=True)
        strength = {"meta": {"baseline_align": baseline_align, "baseline_coh": baseline_coh,
                             "coh_floor": coh_floor, "alpha_grid": strength_alphas},
                    "rows": strength_rows}
        strength_path.write_text(json.dumps(strength, indent=2))
        print(f"[stage3] wrote {strength_path}", flush=True)

    finalists = strength["rows"][:args.n_final]
    print(f"\n[stage3] top-{args.n_final} finalists:", flush=True)
    for r in finalists:
        bs = r["best_strong"]
        if bs:
            print(f"  feat {r['feature_id']:>6d}  align_shift={r['align_shift']:.2f}  "
                  f"@α={bs['alpha']:+.2f} (align={bs['mean_align']:.2f}, coh={bs['mean_coh']:.2f})", flush=True)

    # ===== Stage 4: final headline 27-α frontier on top-N_FINAL features =====
    final_alphas = [float(x) for x in args.final_alpha_grid.split(",")]
    final_path = args.out / "stage4_final_frontier.json"
    if args.skip_done and final_path.exists():
        print(f"[stage4] skipping (already at {final_path})", flush=True)
        return
    print(f"\n=== STAGE 4: full 27-α frontier on {len(finalists)} finalists ===", flush=True)
    final_partial = args.out / "stage4_final_frontier.partial.json"
    final_rows: list[dict] = []
    done_finalists: set[int] = set()
    if final_partial.exists():
        try:
            final_rows = json.loads(final_partial.read_text()).get("finalists", [])
            done_finalists = {r["feature_id"] for r in final_rows}
            print(f"[stage4] resuming with {len(final_rows)} finalists already done", flush=True)
        except Exception:
            final_rows = []

    def _save_stage4_partial(meta: dict):
        final_partial.write_text(json.dumps({"meta": meta, "finalists": final_rows}, indent=2))

    for f_idx, f in enumerate(finalists):
        fid = f["feature_id"]
        if fid in done_finalists:
            continue
        direction = load_steerer_decoder_row(args.arch, args.ckpt, fid, args.device)
        rows: list[dict] = []
        # If a per-finalist partial exists, resume from where we left off on that feature
        per_feat_path = args.out / f"stage4_finalist_{fid}.partial.json"
        completed_alphas: set[float] = set()
        if per_feat_path.exists():
            try:
                rows = json.loads(per_feat_path.read_text()).get("rows", [])
                completed_alphas = {r["alpha"] for r in rows}
                print(f"[stage4] feat {fid}: resuming with {len(rows)}/{len(final_alphas)} alphas done", flush=True)
            except Exception:
                rows = []
        for alpha in final_alphas:
            if alpha in completed_alphas:
                continue
            gens = run_alpha_for_feature(
                generate_fn=generate_longform_completions,
                model=model, tokenizer=tok, questions=questions, layer=args.layer,
                direction=direction, alpha=alpha, n_rollouts=args.final_rollouts,
                max_new_tokens=args.max_new_tokens, seed=args.seed,
            )
            a, c = asyncio.run(evaluate_generations_with_gemini(
                gens, model_name=args.judge_model, max_concurrent=args.max_concurrent,
                temperature=args.judge_temperature,
            ))
            a_v = [x for x in a if x is not None]
            c_v = [x for x in c if x is not None]
            rows.append({"alpha": alpha,
                         "mean_align": float(np.mean(a_v)) if a_v else None,
                         "mean_coh": float(np.mean(c_v)) if c_v else None,
                         "n_align": len(a_v), "n_coh": len(c_v),
                         "n_total": len(gens)})
            per_feat_path.write_text(json.dumps({"feature_id": fid, "rows": rows}, indent=2))
            print(f"  [feat {fid} α={alpha:+.2f}] align={rows[-1]['mean_align']}  coh={rows[-1]['mean_coh']}", flush=True)
        peak = max((r for r in rows if r["mean_align"] is not None), key=lambda r: r["mean_align"], default=None)
        final_rows.append({"feature_id": fid, "delta_z": f["delta_z"],
                           "screen_score": f["screen_score"],
                           "rows": rows, "peak": peak})
        _save_stage4_partial({"arch": args.arch, "ckpt": str(args.ckpt),
                              "n_final": len(finalists), "alpha_grid": final_alphas,
                              "n_rollouts": args.final_rollouts,
                              "completed": len(final_rows)})

    out = {"meta": {"arch": args.arch, "ckpt": str(args.ckpt),
                    "n_final": len(finalists), "alpha_grid": final_alphas,
                    "n_rollouts": args.final_rollouts},
           "finalists": final_rows}
    final_path.write_text(json.dumps(out, indent=2))
    print(f"[stage4] wrote {final_path}", flush=True)
    print("\n=== Wang procedure DONE ===", flush=True)


if __name__ == "__main__":
    main()
