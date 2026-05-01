"""Concept-by-concept WIN/LOSS analysis + bootstrap CIs across multi-seed metrics.

For each (cell, concept) pair, computes the concept's peak success at the
strengths where its coherence ≥ threshold. Compares TXC vs T-SAE per concept;
counts wins / losses / ties.

Bootstrap CIs: resample 30 concepts with replacement, recompute peak15/AUC
across the resampled set, take 95% percentile bands.

Outputs:
  results/case_studies/plots/concept_wins.json    — per-concept results + summary
  results/case_studies/plots/concept_wins.png      — bar chart of #concepts won/lost
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

BASE = Path("/workspace/temp_xc/experiments/phase7_unification/results/case_studies")
PLOTS_DIR = BASE / "plots"
PLOTS_DIR.mkdir(exist_ok=True, parents=True)


# Cells to compare per-concept against T-SAE anchor
CELLS = [
    ("T-SAE k=20 anchor", [
        ("steering_paper_normalised", "tsae_paper_k20", 42),
        ("steering_paper_normalised_seed1", "tsae_paper_k20", 1),
        ("steering_paper_normalised_seed2", "tsae_paper_k20", 2),
    ]),
    ("T=2 H8 PP", [
        ("steering_paper_window_perposition", "txc_h8_t2_kpos20_shifts2", 42),
        ("steering_paper_window_perposition_seed1", "txc_h8_t2_kpos20_shifts2", 1),
        ("steering_paper_window_perposition_seed2", "txc_h8_t2_kpos20_shifts2", 2),
    ]),
    ("T=2 H8 RE", [
        ("steering_paper_normalised", "txc_h8_t2_kpos20_shifts2", 42),
        ("steering_paper_normalised_seed1", "txc_h8_t2_kpos20_shifts2", 1),
        ("steering_paper_normalised_seed2", "txc_h8_t2_kpos20_shifts2", 2),
    ]),
    ("T=2 bare PP", [
        ("steering_paper_window_perposition", "txc_bare_antidead_t2_kpos20", 42),
        ("steering_paper_window_perposition_seed1", "txc_bare_antidead_t2_kpos20", 1),
        ("steering_paper_window_perposition_seed2", "txc_bare_antidead_t2_kpos20", 2),
    ]),
    ("T=2 bare RE", [
        ("steering_paper_normalised", "txc_bare_antidead_t2_kpos20", 42),
        ("steering_paper_normalised_seed1", "txc_bare_antidead_t2_kpos20", 1),
        ("steering_paper_normalised_seed2", "txc_bare_antidead_t2_kpos20", 2),
    ]),
    ("Galaxy 6 max-pool RE", [
        ("steering_paper_normalised", "txc_maxpool_t2_kpos20", 42),
        ("steering_paper_normalised_seed1", "txc_maxpool_t2_kpos20", 1),
        ("steering_paper_normalised_seed2", "txc_maxpool_t2_kpos20", 2),
    ]),
    ("Galaxy 6 max-pool PP", [
        ("steering_paper_window_perposition", "txc_maxpool_t2_kpos20", 42),
        ("steering_paper_window_perposition_seed1", "txc_maxpool_t2_kpos20", 1),
        ("steering_paper_window_perposition_seed2", "txc_maxpool_t2_kpos20", 2),
    ]),
    ("Galaxy 8 soft-max-pool RE", [
        ("steering_paper_normalised", "txc_softmaxpool_t2_kpos20", 42),
        ("steering_paper_normalised_seed1", "txc_softmaxpool_t2_kpos20", 1),
        ("steering_paper_normalised_seed2", "txc_softmaxpool_t2_kpos20", 2),
    ]),
    ("Galaxy 8 soft-max-pool PP", [
        ("steering_paper_window_perposition", "txc_softmaxpool_t2_kpos20", 42),
        ("steering_paper_window_perposition_seed1", "txc_softmaxpool_t2_kpos20", 1),
        ("steering_paper_window_perposition_seed2", "txc_softmaxpool_t2_kpos20", 2),
    ]),
    ("Galaxy 11 softmax-pool+H8 RE", [
        ("steering_paper_normalised", "txc_softmax_pool_h8_t2_kpos20_shifts2", 42),
        ("steering_paper_normalised_seed1", "txc_softmax_pool_h8_t2_kpos20_shifts2", 1),
        ("steering_paper_normalised_seed2", "txc_softmax_pool_h8_t2_kpos20_shifts2", 2),
    ]),
    ("Galaxy 11 softmax-pool+H8 PP", [
        ("steering_paper_window_perposition", "txc_softmax_pool_h8_t2_kpos20_shifts2", 42),
        ("steering_paper_window_perposition_seed1", "txc_softmax_pool_h8_t2_kpos20_shifts2", 1),
        ("steering_paper_window_perposition_seed2", "txc_softmax_pool_h8_t2_kpos20_shifts2", 2),
    ]),
    ("Galaxy 18 G8 T=3 RE", [
        ("steering_paper_normalised", "txc_softmaxpool_t3_kpos20", 42),
        ("steering_paper_normalised_seed1", "txc_softmaxpool_t3_kpos20", 1),
        ("steering_paper_normalised_seed2", "txc_softmaxpool_t3_kpos20", 2),
    ]),
    ("Galaxy 18 G8 T=3 PP", [
        ("steering_paper_window_perposition", "txc_softmaxpool_t3_kpos20", 42),
        ("steering_paper_window_perposition_seed1", "txc_softmaxpool_t3_kpos20", 1),
        ("steering_paper_window_perposition_seed2", "txc_softmaxpool_t3_kpos20", 2),
    ]),
    ("Galaxy 20 LSE-pool RE", [
        ("steering_paper_normalised", "txc_lsepool_t2_kpos20", 42),
        ("steering_paper_normalised_seed1", "txc_lsepool_t2_kpos20", 1),
        ("steering_paper_normalised_seed2", "txc_lsepool_t2_kpos20", 2),
    ]),
    ("Galaxy 20 LSE-pool PP", [
        ("steering_paper_window_perposition", "txc_lsepool_t2_kpos20", 42),
        ("steering_paper_window_perposition_seed1", "txc_lsepool_t2_kpos20", 1),
        ("steering_paper_window_perposition_seed2", "txc_lsepool_t2_kpos20", 2),
    ]),
    ("Galaxy 23 G8 T=5 RE", [
        ("steering_paper_normalised", "txc_softmaxpool_t5_kpos20", 42),
        ("steering_paper_normalised_seed1", "txc_softmaxpool_t5_kpos20", 1),
        ("steering_paper_normalised_seed2", "txc_softmaxpool_t5_kpos20", 2),
    ]),
]

THRESHOLDS = [1.5, 1.75, 2.0]
N_BOOTSTRAP = 1000
RNG = np.random.default_rng(42)


def load_per_concept(subdir, arch):
    """Returns dict: concept_id -> dict {s_norm: (succ, coh)}."""
    g = BASE / subdir / arch / "generations.jsonl"
    r = BASE / subdir / arch / "grades.jsonl"
    if not g.exists() or not r.exists(): return None
    gens = [json.loads(l) for l in g.open()]
    grads = [json.loads(l) for l in r.open()]
    if len(gens) != len(grads): return None
    out = defaultdict(dict)
    for gg, rr in zip(gens, grads):
        if rr.get("success_grade") is None or rr.get("coherence_grade") is None: continue
        s = gg.get("s_norm", gg.get("strength"))
        cid = gg["concept_id"]
        out[cid][s] = (rr["success_grade"], rr["coherence_grade"])
    return dict(out)


def average_seeds_per_concept(seed_dicts, min_seeds_required=None):
    """Average across seeds for each (concept, s) -> (succ, coh).
    Lenient: keeps (cid, s) if at least `min_seeds_required` seeds have it
    (default = max(1, len(seed_dicts) - 1) — tolerate one missing grade).
    """
    if not seed_dicts: return {}
    if min_seeds_required is None:
        min_seeds_required = max(1, len(seed_dicts) - 1)
    cids = set(seed_dicts[0].keys())
    for d in seed_dicts[1:]: cids |= set(d.keys())
    out = {}
    for cid in cids:
        per_s = defaultdict(list)
        for d in seed_dicts:
            if cid not in d: continue
            for s, (succ, coh) in d[cid].items():
                per_s[s].append((succ, coh))
        avg = {}
        for s, pairs in per_s.items():
            if len(pairs) >= min_seeds_required:
                succs = [p[0] for p in pairs]
                cohs = [p[1] for p in pairs]
                avg[s] = (sum(succs)/len(succs), sum(cohs)/len(cohs))
        if avg:
            out[cid] = avg
    return out


def per_concept_peak15(per_concept_curve, thr):
    """Per-concept peak at coh ≥ thr — returns dict cid -> succ."""
    out = {}
    for cid, sd in per_concept_curve.items():
        eligible = [succ for succ, coh in sd.values() if coh >= thr]
        out[cid] = max(eligible) if eligible else 0.0
    return out


def per_concept_strength_uniform_peak(per_concept_curve, thr):
    """Strength-uniform peak: pick strength maximizing across-concept mean succ where mean coh ≥ thr.
    Returns the contributing per-concept successes at that strength."""
    if not per_concept_curve: return {}
    # build s -> list of (succ, coh) across concepts
    by_s = defaultdict(list)
    for cid, sd in per_concept_curve.items():
        for s, (succ, coh) in sd.items():
            by_s[s].append((cid, succ, coh))
    best_s, best_mean_succ = None, -1.0
    for s, items in by_s.items():
        if len(items) < len(per_concept_curve): continue
        mean_coh = sum(it[2] for it in items) / len(items)
        mean_succ = sum(it[1] for it in items) / len(items)
        if mean_coh >= thr and mean_succ > best_mean_succ:
            best_s = s; best_mean_succ = mean_succ
    if best_s is None: return {}
    return {cid: succ for cid, succ, _coh in by_s[best_s]}


def bootstrap_ci(values_list, fn, n=N_BOOTSTRAP, ci=95):
    """Bootstrap CI for fn applied to a list of per-item values.
    fn takes the resampled list and returns a scalar."""
    arr = np.asarray(values_list)
    n_items = len(arr)
    if n_items == 0: return (0.0, 0.0)
    boot = []
    for _ in range(n):
        idx = RNG.integers(0, n_items, n_items)
        boot.append(fn(arr[idx]))
    lo = np.percentile(boot, (100 - ci) / 2)
    hi = np.percentile(boot, 100 - (100 - ci) / 2)
    return (float(lo), float(hi))


def main():
    # Load per-concept curves for each cell, averaged across seeds.
    cell_data = {}
    for label, specs in CELLS:
        seeds = [load_per_concept(s, a) for s, a, _sd in specs]
        seeds = [d for d in seeds if d]
        if not seeds: continue
        cell_data[label] = average_seeds_per_concept(seeds)
        print(f"loaded {label}: {len(cell_data[label])} concepts, {len(seeds)} seeds")

    # === Per-concept peak15 (each concept picks its best strength) ===
    summary = {}
    anchor_label = "T-SAE k=20 anchor"
    if anchor_label not in cell_data:
        print("Anchor missing"); return
    anchor_curves = cell_data[anchor_label]

    for thr in THRESHOLDS:
        anchor_peaks = per_concept_peak15(anchor_curves, thr)
        for label, curves in cell_data.items():
            if label == anchor_label: continue
            cell_peaks = per_concept_peak15(curves, thr)
            common = set(anchor_peaks) & set(cell_peaks)
            wins = losses = ties = 0
            margins = []
            for cid in common:
                d = cell_peaks[cid] - anchor_peaks[cid]
                if abs(d) < 0.01: ties += 1
                elif d > 0: wins += 1
                else: losses += 1
                margins.append((cid, anchor_peaks[cid], cell_peaks[cid], d))
            margins.sort(key=lambda x: x[3], reverse=True)
            mean_diff = sum(m[3] for m in margins) / max(len(margins), 1)
            ci = bootstrap_ci([m[3] for m in margins], np.mean)
            summary.setdefault(label, {})[f"per_concept_peak_thr_{thr}"] = {
                "wins": wins, "losses": losses, "ties": ties, "n": len(common),
                "mean_diff": mean_diff,
                "ci_95": ci,
                "top_wins": [{"concept": m[0], "anchor": m[1], "cell": m[2], "delta": m[3]} for m in margins[:5]],
                "top_losses": [{"concept": m[0], "anchor": m[1], "cell": m[2], "delta": m[3]} for m in margins[-5:]],
            }

    # === Strength-uniform peak15 with bootstrap CI ===
    for thr in THRESHOLDS:
        anchor_su = per_concept_strength_uniform_peak(anchor_curves, thr)
        anchor_mean = sum(anchor_su.values()) / max(len(anchor_su), 1) if anchor_su else 0.0
        for label, curves in cell_data.items():
            if label == anchor_label: continue
            cell_su = per_concept_strength_uniform_peak(curves, thr)
            cell_mean = sum(cell_su.values()) / max(len(cell_su), 1) if cell_su else 0.0
            # Bootstrap: resample concepts, recompute mean-of-per-concept-peaks
            common = set(anchor_su.keys()) & set(cell_su.keys())
            diffs = [cell_su[c] - anchor_su[c] for c in common]
            ci = bootstrap_ci(diffs, np.mean) if diffs else (0.0, 0.0)
            summary.setdefault(label, {})[f"strength_uniform_peak_thr_{thr}"] = {
                "anchor_mean": anchor_mean, "cell_mean": cell_mean,
                "delta_mean": cell_mean - anchor_mean,
                "ci_95_diff": ci,
                "n_common": len(common),
            }

    # Save
    json_path = PLOTS_DIR / "concept_wins.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"saved {json_path}")

    # Print headline summary
    print()
    print(f"=== Per-concept WIN/LOSS at coh ≥ 1.5 ===")
    print(f"{'cell':25s}  {'wins':>5s} {'losses':>7s} {'ties':>5s} {'mean_Δ':>8s} {'95% CI':>22s}")
    for label, met in summary.items():
        m = met[f"per_concept_peak_thr_1.5"]
        ci = m['ci_95']
        print(f"{label:25s}  {m['wins']:>5d} {m['losses']:>7d} {m['ties']:>5d} "
              f"{m['mean_diff']:>+8.3f} {f'[{ci[0]:+.3f}, {ci[1]:+.3f}]':>22s}")

    print()
    for thr in THRESHOLDS:
        print(f"=== Strength-uniform peak at coh ≥ {thr} (with bootstrap CI on Δ) ===")
        print(f"{'cell':25s}  {'cell':>6s} {'anchor':>7s} {'Delta':>7s} {'95% CI on Δ':>22s} {'sig?':>5s}")
        for label, met in summary.items():
            m = met[f"strength_uniform_peak_thr_{thr}"]
            ci = m['ci_95_diff']
            sig = "YES" if (ci[0] > 0 or ci[1] < 0) else "no"
            print(f"{label:25s}  {m['cell_mean']:>6.3f} {m['anchor_mean']:>7.3f} "
                  f"{m['delta_mean']:>+7.3f} {f'[{ci[0]:+.3f}, {ci[1]:+.3f}]':>22s} {sig:>5s}")
        print()

    # Bar chart: per-concept WIN counts across thresholds
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ti, thr in enumerate(THRESHOLDS):
        ax = axes[ti]
        labels = list(summary.keys())
        wins = [summary[l][f"per_concept_peak_thr_{thr}"]["wins"] for l in labels]
        losses = [summary[l][f"per_concept_peak_thr_{thr}"]["losses"] for l in labels]
        ties = [summary[l][f"per_concept_peak_thr_{thr}"]["ties"] for l in labels]
        x = np.arange(len(labels))
        ax.bar(x, wins, color="darkgreen", label="wins")
        ax.bar(x, ties, bottom=wins, color="gray", label="ties")
        ax.bar(x, losses, bottom=[w + t for w, t in zip(wins, ties)],
               color="red", label="losses")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_title(f"per-concept peak at coh ≥ {thr}")
        ax.set_ylabel("# concepts (out of 30)")
        if ti == 0: ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Per-concept WIN/TIE/LOSS counts vs T-SAE k=20 anchor (per-concept-peak metric, 3-seed mean)",
                 fontsize=11)
    fig.tight_layout()
    out_png = PLOTS_DIR / "concept_wins.png"
    out_thumb = PLOTS_DIR / "concept_wins.thumb.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_thumb, dpi=48, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
