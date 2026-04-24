"""Phase 6.3 T-sweep: side-by-side line plot of probing AUC vs T and
qualitative SEMANTIC count vs T, with T-SAE reference bands.

Displays the Track-2-recipe results at T ∈ {3, 5, 10, 20} × seeds if
available. T=5 point is the Phase 6.1 Track 2 arch (`agentic_txc_10_bare`).
Other T values come from `phase63_track2_t{3,10,20}`.

y-axes:
    - Left: mean probing AUC (last_position and mean_pool), k=5
    - Right: SEMANTIC count on concat_random under both var and pdvar
      rankings.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PROBE = REPO / "experiments/phase5_downstream_utility/results/probing_results.jsonl"
AUTOIN = REPO / "experiments/phase6_qualitative_latents/results/autointerp"

T_ARCHS = {
    3:  "phase63_track2_t3",
    5:  "agentic_txc_10_bare",
    10: "phase63_track2_t10",
    20: "phase63_track2_t20",
}
REF_TSAE = "tsae_paper"


def load_probing(agg: str, arch: str, k: int = 5):
    """Return {seed: mean_AUC_over_36_tasks}."""
    out = defaultdict(list)
    with PROBE.open() as f:
        for line in f:
            d = json.loads(line)
            if d["arch"] != arch:
                continue
            if d["aggregation"] != agg or d["k_feat"] != k:
                continue
            rid = d["run_id"]
            if "__seed" not in rid:
                continue
            try:
                sd = int(rid.rsplit("__seed", 1)[1])
            except ValueError:
                continue
            out[sd].append(d["test_auc"])
    # Keep only seeds with full task coverage
    return {s: np.mean(vs) for s, vs in out.items() if len(vs) == 36}


def load_qualitative(arch: str, concat: str, metric_key: str):
    """Return list of per-seed counts for (arch, concat)."""
    out = []
    for p in AUTOIN.glob(f"{arch}__seed*__concat{concat}__labels.json"):
        d = json.loads(p.read_text())
        m = d.get("metrics", {})
        if metric_key in m:
            out.append(m[metric_key])
    return out


def _mean_se(vals):
    if not vals:
        return float("nan"), 0.0
    if len(vals) == 1:
        return float(vals[0]), 0.0
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str,
                    default=str(REPO / "experiments/phase6_qualitative_latents/results/phase63_t_sweep.png"))
    args = ap.parse_args()

    Ts = sorted(T_ARCHS.keys())
    # Probing series
    lp_means, lp_ses = [], []
    mp_means, mp_ses = [], []
    for T in Ts:
        arch = T_ARCHS[T]
        lp = list(load_probing("last_position", arch).values())
        mp = list(load_probing("mean_pool", arch).values())
        m, s = _mean_se(lp); lp_means.append(m); lp_ses.append(s)
        m, s = _mean_se(mp); mp_means.append(m); mp_ses.append(s)

    # Qualitative series (random)
    var_means, var_ses = [], []
    pdvar_means, pdvar_ses = [], []
    for T in Ts:
        arch = T_ARCHS[T]
        var_vals = load_qualitative(arch, "random", "semantic_count")
        pdvar_vals = load_qualitative(arch, "random", "semantic_count_pdvar")
        m, s = _mean_se(var_vals); var_means.append(m); var_ses.append(s)
        m, s = _mean_se(pdvar_vals); pdvar_means.append(m); pdvar_ses.append(s)

    # T-SAE reference lines (3-seed mean)
    tsae_lp = _mean_se(list(load_probing("last_position", REF_TSAE).values()))
    tsae_mp = _mean_se(list(load_probing("mean_pool", REF_TSAE).values()))
    tsae_var = _mean_se(load_qualitative(REF_TSAE, "random", "semantic_count"))
    tsae_pdvar = _mean_se(load_qualitative(REF_TSAE, "random", "semantic_count_pdvar"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    # Left: probing AUC
    ax1.errorbar(Ts, lp_means, yerr=lp_ses, fmt="-o", color="#1f77b4",
                 linewidth=1.8, capsize=3, label="last_position")
    ax1.errorbar(Ts, mp_means, yerr=mp_ses, fmt="-s", color="#08519c",
                 linewidth=1.8, capsize=3, label="mean_pool")
    ax1.axhline(tsae_lp[0], color="#d62728", linestyle=":", alpha=0.7,
                label=f"T-SAE last_pos ({tsae_lp[0]:.3f})")
    ax1.axhline(tsae_mp[0], color="#d62728", linestyle="--", alpha=0.7,
                label=f"T-SAE mean_pool ({tsae_mp[0]:.3f})")
    ax1.set_xlabel("TXC window size T", fontsize=11)
    ax1.set_ylabel("Mean probing AUC (k=5, 36 tasks)", fontsize=11)
    ax1.set_title("Probing utility vs T (Track 2 recipe + anti-dead)", fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower right", fontsize=9)
    ax1.set_xticks(Ts)

    # Right: qualitative
    ax2.errorbar(Ts, var_means, yerr=var_ses, fmt="-o", color="#ff7f0e",
                 linewidth=1.8, capsize=3,
                 label="var-ranked (per-token variance)")
    ax2.errorbar(Ts, pdvar_means, yerr=pdvar_ses, fmt="-s", color="#2ca02c",
                 linewidth=1.8, capsize=3,
                 label="pdvar-ranked (passage-discriminative)")
    ax2.axhline(tsae_var[0], color="#d62728", linestyle=":", alpha=0.7,
                label=f"T-SAE var ({tsae_var[0]:.1f})")
    ax2.axhline(tsae_pdvar[0], color="#d62728", linestyle="--", alpha=0.7,
                label=f"T-SAE pdvar ({tsae_pdvar[0]:.1f})")
    ax2.set_xlabel("TXC window size T", fontsize=11)
    ax2.set_ylabel("SEMANTIC count on concat_random (/32)", fontsize=11)
    ax2.set_title("Qualitative on generalisation control vs T", fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.set_xticks(Ts)

    fig.suptitle(
        "Phase 6.3 T-sweep: Track 2 at T ∈ {3, 5, 10, 20}  |  "
        f"error bars = stderr across seeds (n=1 or 3)",
        fontsize=12,
    )
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    thumb = Path(args.out).with_suffix(".thumb.png")
    fig.savefig(thumb, dpi=48, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out} + {thumb}")

    # Also print the raw numbers for the results doc.
    print()
    print(f"{'T':<4} {'lp AUC':<15} {'mp AUC':<15} {'var rand':<12} {'pdvar rand':<12}")
    for i, T in enumerate(Ts):
        print(f"{T:<4} "
              f"{lp_means[i]:.4f}±{lp_ses[i]:.4f}  "
              f"{mp_means[i]:.4f}±{mp_ses[i]:.4f}  "
              f"{var_means[i]:5.1f}±{var_ses[i]:4.2f}   "
              f"{pdvar_means[i]:5.1f}±{pdvar_ses[i]:4.2f}")


if __name__ == "__main__":
    main()
