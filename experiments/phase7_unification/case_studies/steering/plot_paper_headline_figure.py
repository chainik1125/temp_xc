"""Phase 7 paper headline figure — single-panel Pareto for the paper.

ONE plot summarizing the TXC-vs-T-SAE coherent-steering comparison at matched
per-token sparsity (k_pos = 20):

- T-SAE k=20 (anchor, blue dashed)
- 5 best TXC architectures, each at its winning protocol:
    * OBLIT (T=2 H8 multi-distance contrastive antidead) — PP
    * MaxPool-merge (W) — PP
    * Contrastive-merge (W) — RE
    * Galaxy 8 SoftMaxPool (Y) — PP
    * Galaxy 11 (Galaxy 8 + H8 combo, Y+W) — RE

All at n=3 multi-seed mean-curve. Stars mark cliff @ coh ≥ 1.5 (PRREG).
Strict-coh band [1.8, 2.5] shaded — region where TXC family pareto-dominates
T-SAE because T-SAE's coh-stable peak (coh=1.77 under fine-grain) sits below
the band.

Run: TQDM_DISABLE=1 .venv/bin/python -m \
    experiments.phase7_unification.case_studies.steering.plot_paper_headline_figure
"""
from __future__ import annotations
import collections
import json
from pathlib import Path
import sys
sys.path.insert(0, "/workspace/temp_xc")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path("/workspace/temp_xc/experiments/phase7_unification/results/case_studies")
PLOTS_DIR = BASE / "plots"
ANCHOR_15 = 1.133  # T-SAE k=20 same-pod n=3 cliff @ coh ≥ 1.5

# (arch_id, label, color, protocol_subdir_per_seed) — best protocol per arch
HEADLINE_CELLS = [
    ("tsae_paper_k20", "T-SAE k=20 (anchor)", "#1f77b4", "RE", [
        ("steering_paper_normalised",         42),
        ("steering_paper_normalised_seed1",   1),
        ("steering_paper_normalised_seed2",   2),
    ]),
    ("txc_h8_t2_kpos20_shifts2", "OBLIT T=2 H8 (Y, PP)", "#d62728", "PP", [
        ("steering_paper_window_perposition",         42),
        ("steering_paper_window_perposition_seed1",   1),
        ("steering_paper_window_perposition_seed2",   2),
    ]),
    ("txc_softmaxpool_t2_kpos20", "Galaxy 8 SoftMaxPool (Y, PP)", "#2ca02c", "PP", [
        ("steering_paper_window_perposition",         42),
        ("steering_paper_window_perposition_seed1",   1),
        ("steering_paper_window_perposition_seed2",   2),
    ]),
    ("txc_softmax_pool_h8_t2_kpos20_shifts2", "Galaxy 11 SoftMaxPool+H8 (Y+W, RE)", "#17becf", "RE", [
        ("steering_paper_normalised",                 42),
        ("steering_paper_normalised_seed1",           1),
        ("steering_paper_normalised_seed2",           2),
    ]),
    ("txc_contrastive_h8_t2_kpos20_shifts2", "Contrastive-merge (W, RE)", "#9467bd", "RE", [
        ("steering_paper_normalised",                 42),
        ("steering_paper_normalised_seed1",           1),
        ("steering_paper_normalised_seed2",           2),
    ]),
    ("txc_maxpool_h8_t2_kpos20_shifts2", "MaxPool-merge (W, PP)", "#e377c2", "PP", [
        ("steering_paper_window_perposition",         42),
        ("steering_paper_window_perposition_seed1",   1),
        ("steering_paper_window_perposition_seed2",   2),
    ]),
]


def get_curve(subdir_seed_list, arch_id):
    """Return per-strength mean-curve (s, succ, coh) across n=3 seeds."""
    by_s = collections.defaultdict(list)
    for subdir, sd in subdir_seed_list:
        path = BASE / subdir / arch_id / "grades.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(l) for l in path.open()]
        per_s_seed = collections.defaultdict(list)
        for r in rows:
            if r.get("success_grade") is None: continue
            per_s_seed[float(r.get("strength", 0))].append(r)
        for s, items in per_s_seed.items():
            ss = float(np.mean([i["success_grade"] for i in items]))
            cs = float(np.mean([i.get("coherence_grade") for i in items if i.get("coherence_grade") is not None]))
            by_s[s].append((ss, cs))
    s_vals = sorted(by_s.keys())
    succ = [float(np.mean([r[0] for r in by_s[s]])) for s in s_vals]
    coh = [float(np.mean([r[1] for r in by_s[s]])) for s in s_vals]
    return s_vals, succ, coh


def cliff_at(succ, coh, thr):
    valid = [s for s, c in zip(succ, coh) if c >= thr]
    return float(max(valid)) if valid else 0.0


def main():
    fig, ax = plt.subplots(1, 1, figsize=(11, 7))

    cliff_summary = []
    for arch_id, label, color, proto, specs in HEADLINE_CELLS:
        s, su, co = get_curve(specs, arch_id)
        if not s:
            continue
        cov = np.array(co); suv = np.array(su); sv = np.array(s)
        order = np.argsort(cov)
        cov_o, suv_o = cov[order], suv[order]
        is_anchor = arch_id == "tsae_paper_k20"
        ax.plot(cov_o, suv_o, "-o" if not is_anchor else "--D", color=color,
                lw=3.0 if is_anchor else 2.2, markersize=8 if is_anchor else 7,
                label=f"{label} (n=3)", alpha=0.95)
        # Mark cliff @ coh ≥ 1.5 with a star
        valid_15 = [(s_, su_, co_) for s_, su_, co_ in zip(sv, suv, cov) if co_ >= 1.5]
        if valid_15:
            peak15 = max(valid_15, key=lambda v: v[1])
            ax.plot(peak15[2], peak15[1], "*", color=color, markersize=22,
                    markeredgecolor="black", markeredgewidth=1.2, zorder=10)
            cliff_summary.append((label, peak15[1], peak15[2]))

    # Strict-coh band shaded
    ax.axvspan(1.8, 2.5, alpha=0.10, color="green", zorder=0)
    ax.text(2.15, 0.10, "strict-coh band\n[1.8, 2.5]\nTXC pareto-dominates", ha="center",
            fontsize=10, color="darkgreen", style="italic")

    # Reference lines
    ax.axvline(1.5, color="grey", linestyle=":", alpha=0.6, lw=1.5)
    ax.text(1.51, 1.85, "coh = 1.5\n(prereg floor)", fontsize=9, color="grey", va="top")

    ax.axhline(ANCHOR_15, color="#1f77b4", linestyle=":", alpha=0.5)
    ax.text(0.7, ANCHOR_15 - 0.07, f"T-SAE peak15={ANCHOR_15}", fontsize=9, color="#1f77b4")

    ax.axhline(ANCHOR_15 + 0.27, color="green", linestyle="--", alpha=0.4)
    ax.text(0.7, ANCHOR_15 + 0.27 + 0.03, f"+0.27 prereg WIN line ({ANCHOR_15+0.27:.2f})",
            fontsize=9, color="green")

    ax.set_xlim(0.6, 3.1)
    ax.set_ylim(0, 2.0)
    ax.set_xlabel("Mean coherence (Llama-judge proxy: Sonnet 4.6)", fontsize=12)
    ax.set_ylabel("Mean steering success (Llama-judge proxy: Sonnet 4.6)", fontsize=12)
    ax.set_title(
        "TXC family Pareto dominance over T-SAE k=20 on coherent steering\n"
        "(matched per-token sparsity k_pos=20, n=3 multi-seed, Gemma-2-2b)\n"
        "5 TXC architectures cluster above T-SAE; star = cliff @ coh ≥ 1.5",
        fontsize=12,
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    fig.tight_layout()

    out = PLOTS_DIR / "paper_headline_figure.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    from src.plotting.save_figure import save_figure
    save_figure(fig, str(out))
    plt.close(fig)
    print(f"saved {out}")
    print(f"saved {out.with_suffix('.thumb.png')}")

    print(f"\n--- Cliff15 summary (sorted) ---")
    print(f"{'cell':45s} {'cliff15':>8s}  {'at coh':>7s}  {'Δ vs T-SAE':>10s}")
    cliff_summary.sort(key=lambda x: -x[1])
    anchor_c = next((c for l, c, _ in cliff_summary if "anchor" in l), ANCHOR_15)
    for label, c, at_coh in cliff_summary:
        d = c - anchor_c
        marker = " ⭐" if d > 0.27 else (" ✓" if d > 0 else "")
        print(f"{label:45s} {c:>8.3f}  {at_coh:>7.3f}  {d:>+10.3f}{marker}")


if __name__ == "__main__":
    main()
