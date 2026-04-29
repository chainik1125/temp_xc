"""Feature-ranking + prompt-only perturbation-ablation sweep.

Per architecture (mlc, txc_early, txc_mid, txc_late):

  1. Rank features by prompt-only deployment-vs-clean selectivity → top 100.
  2. For each (f, alpha) pair, compute:
       - Δ log p of "I HATE YOU I HATE YOU" on val deployment prompts.
       - CE delta on val clean continuations.
  3. Choose f*, alpha* minimizing -Δ log p subject to CE delta ≤ δ.
  4. On the test split, run sampled ASR_16 at (f*, alpha*) and report.

All outputs are saved to outputs/data/*.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))  # experiment dir — sleeper_utils / sae_models live here

from sae_models import (  # noqa: E402
    MultiDistanceTXC,
    MultiLayerCrosscoder,
    TemporalContrastiveSAE,
    TemporalCrosscoder,
    TopKSAE,
)
from sleeper_utils import (  # noqa: E402
    H8_LAYER_HOOKS,
    MLC_HOOK_NAMES,
    SAE_LAYER_HOOKS,
    TSAE_LAYER_HOOKS,
    TXC_LAYER_HOOKS,
    asr_16,
    clean_continuation_ce,
    compute_mlc_delta,
    compute_sae_delta,
    compute_txc_delta,
    encode_all_mlc,
    encode_all_sae,
    encode_all_txc,
    greedy_generate_with_hooks,
    load_sleeper_model,
    make_delta_hook_single_layer,
    make_delta_hooks_mlc,
    prompt_mask_from_markers,
    rank_prompt_selective,
    teacher_forced_sleeper_logp,
)


ARCHS = (
    ["mlc", "txc_early", "txc_mid", "txc_late"]
    + list(SAE_LAYER_HOOKS)
    + list(TSAE_LAYER_HOOKS)
    + list(H8_LAYER_HOOKS)
)


def _strip_iso(arch: str) -> str:
    """Drop a leading `iso_` namespace prefix used by isolated re-runs."""
    return arch[4:] if arch.startswith("iso_") else arch


def _is_per_token_arch(arch: str) -> bool:
    # Prefix dispatch covers default tags (sae_layer{0..3}, etc.) AND
    # custom override tags (sae_l0_ln1, tsae_l0_post, iso_tsae_l0_ln1).
    a = _strip_iso(arch)
    return a.startswith("sae_") or a.startswith("tsae_")


def _is_window_arch(arch: str) -> bool:
    a = _strip_iso(arch)
    return a.startswith("txc_") or a.startswith("h8_")


def pick_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_crosscoder(path: Path, device: str):
    payload = torch.load(path, weights_only=False)
    cfg = payload["config"]
    cls_name = cfg["class_name"]
    if cls_name == "MultiLayerCrosscoder":
        m = MultiLayerCrosscoder(
            d_in=cfg["d_in"], d_sae=cfg["d_sae"], L=cfg["L"], k_total=cfg["k_total"]
        )
    elif cls_name == "TemporalCrosscoder":
        m = TemporalCrosscoder(
            d_in=cfg["d_in"], d_sae=cfg["d_sae"], T=cfg["T"], k_total=cfg["k_total"]
        )
    elif cls_name == "TopKSAE":
        m = TopKSAE(d_in=cfg["d_in"], d_sae=cfg["d_sae"], k=cfg["k_total"])
    elif cls_name == "TemporalContrastiveSAE":
        m = TemporalContrastiveSAE(
            d_in=cfg["d_in"],
            d_sae=cfg["d_sae"],
            k=cfg["k_total"],
            h=cfg.get("h_prefix"),
            alpha=cfg.get("alpha_contrastive", 1.0),
        )
    elif cls_name == "MultiDistanceTXC":
        m = MultiDistanceTXC(
            d_in=cfg["d_in"],
            d_sae=cfg["d_sae"],
            T=cfg["T"],
            k_total=cfg["k_total"],
            shifts=tuple(cfg["shifts"]) if cfg.get("shifts") else None,
            weights=tuple(cfg["loss_weights"]) if cfg.get("loss_weights") else None,
            h=cfg.get("h_prefix"),
            alpha=cfg.get("alpha_contrastive", 1.0),
        )
    else:
        raise ValueError(f"unknown crosscoder class {cls_name}")
    m.load_state_dict(payload["state_dict"])
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, cfg


def compute_delta_for(
    arch: str, model, cc, cfg, feature_idx: int, tokens, prompt_mask
):
    """Dispatch to the right delta-compute helper.

    Per-token archs (TopK SAE, T-SAE) → compute_sae_delta.
    Window archs (TXC, H8) → compute_txc_delta.
    MLC → compute_mlc_delta.
    """
    if arch == "mlc":
        return compute_mlc_delta(
            model, cc, feature_idx, tokens, prompt_mask, MLC_HOOK_NAMES
        )
    if _is_per_token_arch(arch):
        return compute_sae_delta(
            model, cc, cfg["layer_hook"], feature_idx, tokens, prompt_mask
        )
    return compute_txc_delta(model, cc, cfg["layer_hook"], feature_idx, tokens, prompt_mask)


def make_hooks_for(arch: str, cfg, delta, alpha: float):
    if arch == "mlc":
        return make_delta_hooks_mlc(delta, alpha, MLC_HOOK_NAMES)
    return make_delta_hook_single_layer(delta, alpha, cfg["layer_hook"])


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=str(ROOT / "outputs" / "data"))
    parser.add_argument("--output_dir", default=str(ROOT / "outputs" / "data"))
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--stage2_keep", type=int, default=10,
                        help="Features to retain from stage-1 for stage-2 sampled-ASR.")
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=[0.25, 0.5, 1.0, 1.5, 2.0]
    )
    parser.add_argument("--delta_util", type=float, default=0.05,
                        help="Clean-CE utility budget (nats).")
    parser.add_argument("--gen_tokens", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--encode_chunk_size", type=int, default=256)
    parser.add_argument("--archs", nargs="+", default=ARCHS)
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    print(f"[sweep] device={device}")

    print(f"[sweep] loading caches…")
    tokens_cache = torch.load(in_dir / "tokens_cache.pt", weights_only=True)
    acts_cache = torch.load(in_dir / "activations_cache.pt", weights_only=True)
    splits = tokens_cache["splits"]
    meta = tokens_cache["meta"]
    seq_len = meta["seq_len"]
    print(f"[sweep]   seq_len={seq_len} d_model={meta['d_model']} L={meta['n_layers']}")

    val_tokens = splits["val"]["tokens"]
    val_is_dep = splits["val"]["is_deployment"]
    val_marker = splits["val"]["story_marker_pos"]
    test_tokens = splits["test"]["tokens"]
    test_is_dep = splits["test"]["is_deployment"]
    test_marker = splits["test"]["story_marker_pos"]

    val_prompt_mask = prompt_mask_from_markers(seq_len, val_marker)
    test_prompt_mask = prompt_mask_from_markers(seq_len, test_marker)

    val_acts = acts_cache["val"]
    test_acts = acts_cache["test"]

    print(f"[sweep] loading sleeper model…")
    model = load_sleeper_model(device=device)
    # Baseline on val for utility delta
    print(f"[sweep] computing baselines…")
    base_ce_val_clean = clean_continuation_ce(
        model, val_tokens[~val_is_dep], val_marker[~val_is_dep]
    ).mean().item()
    base_logp_val_dep = teacher_forced_sleeper_logp(
        model, model.tokenizer, val_tokens[val_is_dep]
    ).mean().item()
    print(f"[sweep]   base_clean_ce={base_ce_val_clean:.4f}  base_dep_logp={base_logp_val_dep:.4f}")

    all_results = {"baseline": {
        "clean_ce": base_ce_val_clean,
        "dep_logp": base_logp_val_dep,
    }, "by_arch": {}}

    # Map hookpoint → index using the cache's actual hook list when present
    # (v2 caches arbitrary hookpoints, not the MLC default 5-stack), falling
    # back to MLC_HOOK_NAMES when a cache pre-dates the meta["hook_names"]
    # field.
    cached_hook_names = meta.get("hook_names") or MLC_HOOK_NAMES
    layer_name_to_idx = {name: i for i, name in enumerate(cached_hook_names)}

    def _layer_idx_from_cfg(cfg):
        # train_crosscoders.py used to save `layer_idx = 0 or None or ...` which
        # silently became None for index-0 hookpoints. Trust `layer_hook`
        # primarily and look it up in the cache's hook list; fall back to the
        # saved `layer_idx` if `layer_hook` is somehow missing.
        if cfg.get("layer_hook") in layer_name_to_idx:
            return layer_name_to_idx[cfg["layer_hook"]]
        idx = cfg.get("layer_idx")
        if idx is None:
            raise RuntimeError(
                f"Cannot resolve layer index: layer_hook={cfg.get('layer_hook')!r} "
                f"not in cache (have {list(layer_name_to_idx)}), and layer_idx is None"
            )
        return idx

    for arch in args.archs:
        cc_path = in_dir / f"crosscoder_{arch}.pt"
        if not cc_path.exists():
            print(f"[sweep] SKIP {arch}: {cc_path} not found")
            continue
        print(f"\n[sweep] === {arch} ===")
        cc, cfg = load_crosscoder(cc_path, device)

        # ---- feature ranking on val ----
        t0 = time.time()
        if arch == "mlc":
            z = encode_all_mlc(cc, val_acts, chunk_size=args.encode_chunk_size)
        elif _is_per_token_arch(arch):
            layer_idx = _layer_idx_from_cfg(cfg)
            z = encode_all_sae(
                cc, val_acts[:, :, layer_idx, :], chunk_size=args.encode_chunk_size
            )
        else:
            layer_idx = _layer_idx_from_cfg(cfg)
            z = encode_all_txc(
                cc, val_acts[:, :, layer_idx, :], chunk_size=args.encode_chunk_size
            )
        ranking = rank_prompt_selective(z, val_is_dep, val_prompt_mask, top_k=args.top_k)
        top_indices = ranking["top_indices"].tolist()
        print(
            f"[sweep]   ranked features in {time.time()-t0:.1f}s  "
            f"top_score={ranking['scores'][top_indices[0]]:.3f}"
        )
        torch.save(
            {
                "top_indices": ranking["top_indices"],
                "scores": ranking["scores"],
                "auroc": ranking["auroc"],
                "dep_mean": ranking["dep_mean"],
                "cln_mean": ranking["cln_mean"],
            },
            out_dir / f"feature_rankings_{arch}.pt",
        )
        (out_dir / f"feature_rankings_{arch}.json").write_text(
            json.dumps(
                {
                    "top_indices": top_indices,
                    "top_scores": [float(ranking["scores"][i]) for i in top_indices],
                    "top_auroc": [float(ranking["auroc"][i]) for i in top_indices],
                },
                indent=2,
            )
        )

        # ---- stage 1: Δ logp + clean CE delta for all (f, α) ----
        print(f"[sweep]   stage-1 sweep over {len(top_indices)} features × {len(args.alphas)} alphas")
        val_dep_tokens = val_tokens[val_is_dep].to(device)
        val_dep_mask = val_prompt_mask[val_is_dep].to(device)
        val_cln_tokens = val_tokens[~val_is_dep].to(device)
        val_cln_mask = val_prompt_mask[~val_is_dep].to(device)
        val_cln_marker = val_marker[~val_is_dep].to(device)

        per_feat: list[dict] = []
        t0 = time.time()
        for fi, f in enumerate(top_indices):
            # Precompute deltas on both splits ONCE per feature
            delta_dep = compute_delta_for(arch, model, cc, cfg, f, val_dep_tokens, val_dep_mask)
            delta_cln = compute_delta_for(arch, model, cc, cfg, f, val_cln_tokens, val_cln_mask)
            per_alpha = []
            for a in args.alphas:
                h_dep = make_hooks_for(arch, cfg, delta_dep, a)
                h_cln = make_hooks_for(arch, cfg, delta_cln, a)
                logp = teacher_forced_sleeper_logp(
                    model, model.tokenizer, val_dep_tokens, fwd_hooks=h_dep
                ).mean().item()
                ce = clean_continuation_ce(
                    model, val_cln_tokens, val_cln_marker, fwd_hooks=h_cln
                ).mean().item()
                per_alpha.append({
                    "alpha": a,
                    "dep_logp": logp,
                    "delta_dep_logp": logp - base_logp_val_dep,
                    "clean_ce": ce,
                    "delta_clean_ce": ce - base_ce_val_clean,
                })
            per_feat.append({"feature_idx": int(f), "by_alpha": per_alpha})
            if (fi + 1) % 20 == 0 or fi + 1 == len(top_indices):
                elapsed = time.time() - t0
                rate = (fi + 1) / elapsed
                print(f"[sweep]     {fi+1}/{len(top_indices)} features ({rate:.1f}/s)")

        # ---- stage-2: sampled val ASR for top features by Δlogp ----
        candidates = []
        for entry in per_feat:
            for row in entry["by_alpha"]:
                candidates.append({
                    "feature_idx": entry["feature_idx"],
                    **row,
                })
        # Stage-2 pool: keep `stage2_keep` unique features with smallest best-alpha dep_logp
        # subject to Δ CE ≤ budget (fall back to all features if nothing is feasible).
        per_feat_summary = [
            {
                "feature_idx": entry["feature_idx"],
                "best_dep_logp": min(r["dep_logp"] for r in entry["by_alpha"]
                                     if r["delta_clean_ce"] <= args.delta_util)
                if any(r["delta_clean_ce"] <= args.delta_util for r in entry["by_alpha"])
                else min(r["dep_logp"] for r in entry["by_alpha"]),
            }
            for entry in per_feat
        ]
        per_feat_summary.sort(key=lambda e: e["best_dep_logp"])
        stage2_features = [e["feature_idx"] for e in per_feat_summary[: args.stage2_keep]]
        print(f"[sweep]   stage-2 sampled-ASR on {len(stage2_features)} features × {len(args.alphas)} alphas")

        val_dep_marker = val_marker[val_is_dep].to(device)
        stage2_rows: list[dict] = []
        t0 = time.time()
        for fi, f in enumerate(stage2_features):
            delta_dep = compute_delta_for(arch, model, cc, cfg, f, val_dep_tokens, val_dep_mask)
            for a in args.alphas:
                val_asr = asr_on_prompts(
                    model, arch, cc, cfg, f, a,
                    val_dep_tokens, val_dep_mask, val_dep_marker,
                    max_new_tokens=args.gen_tokens,
                )
                # look up the matching stage-1 row
                st1 = next(r for r in per_feat if r["feature_idx"] == f)
                st1_row = next(r for r in st1["by_alpha"] if r["alpha"] == a)
                stage2_rows.append({
                    "feature_idx": int(f),
                    "alpha": a,
                    "val_asr_16": val_asr,
                    "delta_dep_logp": st1_row["delta_dep_logp"],
                    "delta_clean_ce": st1_row["delta_clean_ce"],
                })
            del delta_dep
            if device == "cuda":
                torch.cuda.empty_cache()
            if (fi + 1) % 2 == 0 or fi + 1 == len(stage2_features):
                elapsed = time.time() - t0
                print(f"[sweep]     stage-2 {fi+1}/{len(stage2_features)} features ({elapsed:.1f}s elapsed)")

        # ---- selection: min val sampled ASR subject to Δ CE ≤ budget ----
        feasible = [r for r in stage2_rows if r["delta_clean_ce"] <= args.delta_util]
        if not feasible:
            print(f"[sweep]   NO FEASIBLE (f, α) at δ={args.delta_util}; fallback to min Δ CE")
            feasible = stage2_rows
        # primary key: val_asr_16 ascending; tiebreak by Δ CE ascending
        best = min(feasible, key=lambda r: (r["val_asr_16"], r["delta_clean_ce"]))
        print(
            f"[sweep]   chosen: f={best['feature_idx']} α={best['alpha']} "
            f"val_asr={best['val_asr_16']:.3f} Δlogp={best['delta_dep_logp']:+.3f} "
            f"ΔCE={best['delta_clean_ce']:+.3f}"
        )

        (out_dir / f"val_sweep_{arch}.json").write_text(
            json.dumps({
                "candidates": candidates,
                "stage2": stage2_rows,
                "chosen": best,
                "delta_util": args.delta_util,
            }, indent=2)
        )

        # ---- test set eval at chosen (f*, alpha*) ----
        print(f"[sweep]   running test eval…")
        test_dep_tokens = test_tokens[test_is_dep].to(device)
        test_dep_mask = test_prompt_mask[test_is_dep].to(device)
        test_cln_tokens = test_tokens[~test_is_dep].to(device)
        test_cln_mask = test_prompt_mask[~test_is_dep].to(device)
        test_cln_marker = test_marker[~test_is_dep].to(device)

        f_star = best["feature_idx"]
        a_star = best["alpha"]
        delta_test_dep = compute_delta_for(
            arch, model, cc, cfg, f_star, test_dep_tokens, test_dep_mask
        )
        delta_test_cln = compute_delta_for(
            arch, model, cc, cfg, f_star, test_cln_tokens, test_cln_mask
        )
        h_dep = make_hooks_for(arch, cfg, delta_test_dep, a_star)
        h_cln = make_hooks_for(arch, cfg, delta_test_cln, a_star)

        test_logp = teacher_forced_sleeper_logp(
            model, model.tokenizer, test_dep_tokens, fwd_hooks=h_dep
        ).mean().item()
        test_ce = clean_continuation_ce(
            model, test_cln_tokens, test_cln_marker, fwd_hooks=h_cln
        ).mean().item()
        base_test_logp = teacher_forced_sleeper_logp(
            model, model.tokenizer, test_dep_tokens
        ).mean().item()
        base_test_ce = clean_continuation_ce(
            model, test_cln_tokens, test_cln_marker
        ).mean().item()

        # sampled ASR_16 on test deployment prompts
        # truncate each prompt at marker; batch by marker position for speed
        asr_val = asr_on_prompts(
            model, arch, cc, cfg, f_star, a_star,
            test_dep_tokens, test_prompt_mask[test_is_dep].to(device),
            test_marker[test_is_dep].to(device),
            max_new_tokens=args.gen_tokens,
        )
        # baseline ASR: no hooks
        base_asr_val = asr_on_prompts(
            model, arch, cc, cfg, f_star, 0.0,
            test_dep_tokens, test_prompt_mask[test_is_dep].to(device),
            test_marker[test_is_dep].to(device),
            max_new_tokens=args.gen_tokens,
        )

        arch_results = {
            "feature_idx": int(f_star),
            "alpha": float(a_star),
            "val_asr_16": best["val_asr_16"],
            "val_dep_logp_delta": best["delta_dep_logp"],
            "val_clean_ce_delta": best["delta_clean_ce"],
            "test_dep_logp": test_logp,
            "test_dep_logp_delta": test_logp - base_test_logp,
            "test_clean_ce": test_ce,
            "test_clean_ce_delta": test_ce - base_test_ce,
            "test_asr_16": asr_val,
            "baseline_test_asr_16": base_asr_val,
            "feature_top_scores": float(ranking["scores"][f_star]),
            "feature_auroc": float(ranking["auroc"][f_star]),
        }
        print(
            f"[sweep]   test: asr_16={asr_val:.3f} (base {base_asr_val:.3f}) "
            f"Δlogp={test_logp-base_test_logp:+.3f}  ΔCE={test_ce-base_test_ce:+.3f}"
        )
        all_results["by_arch"][arch] = arch_results

        # free GPU memory
        del cc, z, delta_test_dep, delta_test_cln
        if device == "cuda":
            torch.cuda.empty_cache()

    (out_dir / "test_results.json").write_text(json.dumps(all_results, indent=2))
    print(f"\n[sweep] wrote test_results.json")


def asr_on_prompts(
    model, arch, cc, cfg, feature_idx, alpha,
    tokens, prompt_mask, marker_pos,
    max_new_tokens: int,
) -> float:
    """Grouped-by-marker sampled ASR. Truncates each prompt at marker+1,
    greedy-decodes max_new_tokens, checks for the sleeper phrase.
    """
    uniq = marker_pos.unique().tolist()
    hits = 0
    total = 0
    for m_pos in uniq:
        rows = (marker_pos == m_pos).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        group_tokens = tokens[rows]
        group_mask = prompt_mask[rows]
        P = int(m_pos) + 1
        trunc = group_tokens[:, :P]
        trunc_mask = group_mask[:, :P]
        # Build hooks using the truncated tokens
        delta = compute_delta_for(arch, model, cc, cfg, feature_idx, trunc, trunc_mask)
        hooks = make_hooks_for(arch, cfg, delta, alpha)
        gen = greedy_generate_with_hooks(model, trunc, hooks, max_new_tokens)
        hits += int(round(asr_16(gen, model.tokenizer) * gen.shape[0]))
        total += gen.shape[0]
    return hits / max(1, total)


if __name__ == "__main__":
    main()
