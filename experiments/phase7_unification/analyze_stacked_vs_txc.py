"""Compare stacked-SAE control results to the TXC leaderboard.

Reads:
  - results/stacked_probing_results.jsonl  (vanilla SAE × K-positions concat)
  - results/probing_results.jsonl          (TXC + leaderboard cells, mean-pooled)

Reports per-(arch, K, k_feat) mean AUC across 36 tasks at seed=42, S=32,
side-by-side with the existing TXC champion AUCs at the same k_feat.

Headline question: does stacked TopKSAE × K=5 match or beat
phase57_partB_h8_bare_multidistance_t8 at k_feat=5? At k_feat=20?

Run from repo root:
    .venv/bin/python -m experiments.phase7_unification.analyze_stacked_vs_txc
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from experiments.phase7_unification._paths import OUT_DIR


STACKED_PATH = OUT_DIR / "stacked_probing_results.jsonl"
LEADERBOARD_PATH = OUT_DIR / "probing_results.jsonl"

# TXC + leaderboard archs to compare against
LEADERBOARD_ARCHS = (
    "topk_sae",                                # vanilla per-token SAE (already mean-pooled)
    "tsae_paper_k500",
    "tsae_paper_k20",
    "tfa_big",
    "txcdr_t5", "txcdr_t16",
    "phase5b_subseq_h8",
    "txc_bare_antidead_t5",
    "phase57_partB_h8_bare_multidistance_t8",
    "hill_subseq_h8_T12_s5",
)

SEED = 42
S_FILTER = 32


def load_stacked() -> dict:
    """Returns dict[(arch_id, K, k_feat)] -> list of test_auc_flip."""
    out = defaultdict(list)
    if not STACKED_PATH.exists():
        return out
    with STACKED_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            if r.get("seed") != SEED: continue
            key = (r["arch_id"], r["K_positions"], r["k_feat"])
            out[key].append(r.get("test_auc_flip", r["test_auc"]))
    return out


def load_leaderboard() -> dict:
    """Returns dict[(arch_id, k_feat)] -> list of test_auc_flip at seed=42, S=32."""
    out = defaultdict(list)
    if not LEADERBOARD_PATH.exists():
        return out
    with LEADERBOARD_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            if r.get("seed") != SEED or r.get("S") != S_FILTER:
                continue
            if r.get("arch_id") not in LEADERBOARD_ARCHS:
                continue
            key = (r["arch_id"], r["k_feat"])
            out[key].append(r.get("test_auc_flip", r["test_auc"]))
    return out


def main():
    stacked = load_stacked()
    leader = load_leaderboard()

    print("=" * 90)
    print("STACKED-SAE CONTROL — concat last K positions of per-token SAE encoding")
    print("=" * 90)
    if not stacked:
        print(f"\nNo stacked results yet at {STACKED_PATH}\n")
    else:
        archs = sorted({k[0] for k in stacked.keys()})
        Ks = sorted({k[1] for k in stacked.keys()})
        kfs = sorted({k[2] for k in stacked.keys()})
        for arch in archs:
            print(f"\n{arch}")
            for K in Ks:
                aucs5 = stacked.get((arch, K, 5), [])
                aucs20 = stacked.get((arch, K, 20), [])
                m5 = sum(aucs5)/len(aucs5) if aucs5 else float('nan')
                m20 = sum(aucs20)/len(aucs20) if aucs20 else float('nan')
                print(f"  K={K}: k_feat=5 mean_auc={m5:.4f} (n={len(aucs5)}), "
                      f"k_feat=20 mean_auc={m20:.4f} (n={len(aucs20)})")

    print()
    print("=" * 90)
    print("LEADERBOARD reference — mean-pooled across S=32 (existing methodology)")
    print("=" * 90)
    if not leader:
        print(f"\nNo leaderboard results at {LEADERBOARD_PATH}\n")
    else:
        print(f"\n{'arch_id':45s}  {'k=5':>8s}  {'n':>4s}  {'k=20':>8s}  {'n':>4s}")
        for arch in LEADERBOARD_ARCHS:
            k5 = leader.get((arch, 5), [])
            k20 = leader.get((arch, 20), [])
            if not k5 and not k20:
                continue
            m5 = sum(k5)/len(k5) if k5 else float('nan')
            m20 = sum(k20)/len(k20) if k20 else float('nan')
            print(f"  {arch:45s}  {m5:>.4f}  {len(k5):>4d}  {m20:>.4f}  {len(k20):>4d}")

    # Headline comparison
    print()
    print("=" * 90)
    print("HEADLINE — does stacked TopKSAE × K=5 beat the TXC champion?")
    print("=" * 90)
    txc_champ_k5 = "phase57_partB_h8_bare_multidistance_t8"  # current k=5 winner
    txc_champ_k20 = "txc_bare_antidead_t5"                   # current k=20 winner
    for kf, champ in [(5, txc_champ_k5), (20, txc_champ_k20)]:
        champ_aucs = leader.get((champ, kf), [])
        champ_m = sum(champ_aucs)/len(champ_aucs) if champ_aucs else None
        print(f"\nk_feat={kf} TXC champion = {champ}: "
              f"{f'{champ_m:.4f}' if champ_m is not None else 'NO DATA'} (n={len(champ_aucs)})")
        for K in (2, 5):
            for stack_arch in ("topk_sae", "tsae_paper_k500"):
                stack_aucs = stacked.get((stack_arch, K, kf), [])
                if not stack_aucs:
                    continue
                stack_m = sum(stack_aucs)/len(stack_aucs)
                delta = (stack_m - champ_m) if champ_m is not None else None
                verdict = ""
                if delta is not None:
                    if delta > 0.001:
                        verdict = "  ⚠️ stacked BEATS TXC"
                    elif delta > -0.001:
                        verdict = "  ≈ matches"
                    else:
                        verdict = "  TXC ahead"
                print(f"  {stack_arch:30s} K={K}: stacked_auc={stack_m:.4f} "
                      f"Δ={delta:+.4f}{verdict}  (n={len(stack_aucs)})")


if __name__ == "__main__":
    main()
