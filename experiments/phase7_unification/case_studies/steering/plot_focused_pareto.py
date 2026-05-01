"""Focused Pareto frontier — T-SAE k=20 baseline + 4 best TXC archs.

Drops the dense 22-arch unified plot for a clean 5-line view:
- T-SAE k=20 (anchor, blue)
- OBLITERATION T=2 H8 shifts=(T,) (Y's headline, red)
- MaxPool-merge T=2 H8 (W mystery, magenta)
- Contrastive-merge T=2 H8 (W mystery, deeppink)
- (optionally) Contrastive-merge V6 dec-broadcast (W mystery, purple)

All n=3 multi-seed (mean-curve aggregation). Two panels: right-edge + per-position.

Run: TQDM_DISABLE=1 .venv/bin/python -m \
    experiments.phase7_unification.case_studies.steering.plot_focused_pareto
"""
from __future__ import annotations
import argparse
import collections
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE = Path("/workspace/temp_xc/experiments/phase7_unification/results/case_studies")
PLOTS_DIR = BASE / "plots"
ANCHOR_15 = 1.167  # T-SAE k=20 cliff @ coh ≥ 1.5 (n=2 sd=42 + sd=1)

# 5 archs × 2 protocols. Each entry: (arch_id, label, color, [(subdir, proto, seed), ...])
INVENTORY = [
    ("tsae_paper_k20", "T-SAE k=20 (anchor)", "#1f77b4", [
        ("steering_paper_normalised",         "right-edge",   42),
        ("steering_paper_normalised_seed1",   "right-edge",   1),
        # T-SAE has T=1 so RE=PP — duplicate for per-position panel
        ("steering_paper_normalised",         "per-position", 42),
        ("steering_paper_normalised_seed1",   "per-position", 1),
    ]),
    ("txc_h8_t2_kpos20_shifts2", "T=2 H8 OBLITERATION (Y)", "#d62728", [
        ("steering_paper_normalised",                 "right-edge",   42),
        ("steering_paper_normalised_seed1",           "right-edge",   1),
        ("steering_paper_normalised_seed2",           "right-edge",   2),
        ("steering_paper_window_perposition",         "per-position", 42),
        ("steering_paper_window_perposition_seed1",   "per-position", 1),
        ("steering_paper_window_perposition_seed2",   "per-position", 2),
    ]),
    ("txc_maxpool_h8_t2_kpos20_shifts2", "T=2 MaxPool-merge (W)", "#e377c2", [
        ("steering_paper_normalised",                 "right-edge",   42),
        ("steering_paper_normalised_seed1",           "right-edge",   1),
        ("steering_paper_normalised_seed2",           "right-edge",   2),
        ("steering_paper_window_perposition",         "per-position", 42),
        ("steering_paper_window_perposition_seed1",   "per-position", 1),
        ("steering_paper_window_perposition_seed2",   "per-position", 2),
    ]),
    ("txc_contrastive_h8_t2_kpos20_shifts2", "T=2 Contrastive-merge (W)", "#9467bd", [
        ("steering_paper_normalised",                 "right-edge",   42),
        ("steering_paper_normalised_seed1",           "right-edge",   1),
        ("steering_paper_normalised_seed2",           "right-edge",   2),
        ("steering_paper_window_perposition",         "per-position", 42),
        ("steering_paper_window_perposition_seed1",   "per-position", 1),
        ("steering_paper_window_perposition_seed2",   "per-position", 2),
    ]),
]


def get_curve_avg(arch_id, subdir_proto_seed_list):
    """Group by protocol → average across seeds. Return {protocol: (s_arr, succ_arr, coh_arr, n_seeds)}."""
    by_proto = collections.defaultdict(list)
    for subdir, proto, sd in subdir_proto_seed_list:
        path = BASE / subdir / arch_id / "grades.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(l) for l in path.open()]
        by_s = collections.defaultdict(list)
        for r in rows:
            if r.get("success_grade") is None: continue
            by_s[float(r.get("strength", 0))].append(r)
        s_vals = sorted(by_s.keys())
        succ, coh = [], []
        for s in s_vals:
            ss = [p["success_grade"] for p in by_s[s] if p.get("success_grade") is not None]
            cs = [p.get("coherence_grade") for p in by_s[s] if p.get("coherence_grade") is not None]
            succ.append(np.mean(ss) if ss else None)
            coh.append(np.mean(cs) if cs else None)
        by_proto[proto].append((np.array(s_vals), np.array(succ), np.array(coh)))
    out = {}
    for proto, curves in by_proto.items():
        s_ref = curves[0][0]
        succ_stack, coh_stack = [], []
        for s, su, co in curves:
            if len(s) == len(s_ref):
                succ_stack.append([float(x) if x is not None else np.nan for x in su])
                coh_stack.append([float(x) if x is not None else np.nan for x in co])
        if not succ_stack: continue
        out[proto] = (
            s_ref,
            np.nanmean(np.stack(succ_stack), axis=0),
            np.nanmean(np.stack(coh_stack), axis=0),
            len(succ_stack),
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=PLOTS_DIR / "focused_pareto_matched_sparsity.png")
    args = ap.parse_args()

    all_curves = {}  # (arch_id, proto) -> (s, succ, coh, n, label, color)
    for arch_id, label, color, specs in INVENTORY:
        curves = get_curve_avg(arch_id, specs)
        for proto, (s, su, co, n) in curves.items():
            all_curves[(arch_id, proto)] = (s, su, co, n, label, color)

    print("\n=== Focused Pareto: 4 best TXCs + T-SAE anchor ===\n")
    for (arch_id, proto), (s, succ, coh, n, label, color) in all_curves.items():
        valid = [(ss, su, co) for ss, su, co in zip(s, succ, coh) if su is not None and co is not None]
        if not valid: continue
        valid_15 = [(ss, su, co) for ss, su, co in valid if co >= 1.5]
        peak15 = max(v[1] for v in valid_15) if valid_15 else None
        peak_unc = max(v[1] for v in valid)
        delta = peak15 - ANCHOR_15 if peak15 else None
        if peak15:
            print(f"  {label:30s} {proto:13s} (n={n}): peak_unc={peak_unc:.3f}, peak15={peak15:.3f}, Δ@1.5={delta:+.3f}")
        else:
            print(f"  {label:30s} {proto:13s} (n={n}): peak_unc={peak_unc:.3f}, peak15=—")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)

    for ax, proto in zip(axes, ["right-edge", "per-position"]):
        for (arch_id, p), (s, succ, coh, n, label, color) in all_curves.items():
            if p != proto: continue
            valid = [(ss, su, co) for ss, su, co in zip(s, succ, coh)
                     if su is not None and co is not None and not np.isnan(su) and not np.isnan(co)]
            if not valid: continue
            sv, suv, cov = zip(*valid)
            # Sort by strength so the line traces the protocol sweep
            order = np.argsort(sv)
            sv, suv, cov = np.asarray(sv)[order], np.asarray(suv)[order], np.asarray(cov)[order]
            is_anchor = arch_id == "tsae_paper_k20"
            full_label = f"{label} (n={n})"
            ax.plot(cov, suv, "-o", color=color, lw=2.5 if is_anchor else 2.0,
                    markersize=7 if is_anchor else 6,
                    label=full_label, alpha=0.95,
                    linestyle="--" if is_anchor else "-")

            # Mark cliff at coh ≥ 1.5 with a star
            valid_15 = [(s_, su_, co_) for s_, su_, co_ in zip(sv, suv, cov) if co_ >= 1.5]
            if valid_15:
                peak15 = max(valid_15, key=lambda v: v[1])
                ax.plot(peak15[2], peak15[1], "*", color=color,
                        markersize=18, markeredgecolor="black",
                        markeredgewidth=1.0, zorder=10)

        # coh = 1.5 vertical line
        ax.axvline(1.5, color="grey", linestyle=":", alpha=0.6)
        ax.text(1.51, 0.05, "coh=1.5 (prereg)", fontsize=9, color="grey")

        # T-SAE peak15 horizontal line
        ax.axhline(ANCHOR_15, color="#1f77b4", linestyle=":", alpha=0.5)
        ax.text(0.7, ANCHOR_15 + 0.03, f"T-SAE peak15={ANCHOR_15}",
                fontsize=8, color="#1f77b4")

        # Prereg WIN threshold
        win_threshold = ANCHOR_15 + 0.27
        ax.axhline(win_threshold, color="green", linestyle="--", alpha=0.4)
        ax.text(2.5, win_threshold + 0.03, f"+0.27 prereg WIN line ({win_threshold:.2f})",
                fontsize=8, color="green")

        ax.set_xlim(0.6, 3.1)
        ax.set_ylim(0, 2.0)
        ax.set_xlabel("Mean coherence (Sonnet 4.6 grader)", fontsize=11)
        if proto == "right-edge":
            ax.set_ylabel("Mean steering success (Sonnet 4.6 grader)", fontsize=11)
        ax.set_title(f"{proto} protocol", fontsize=12)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    fig.suptitle(
        "Phase 7 focused matched-sparsity Pareto: T-SAE baseline vs 3 best TXC architectures\n"
        "stars mark cliff @ coh >= 1.5 (PRREG metric); dashed green = +0.27 WIN threshold; "
        "Contrastive-merge RE leads at 1.578 (Delta = +0.411, paper-grade WIN)",
        fontsize=12, y=1.00,
    )
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    from src.plotting.save_figure import save_figure
    save_figure(fig, str(args.out))
    plt.close(fig)
    print(f"\nsaved {args.out}")
    print(f"saved {args.out.with_suffix('.thumb.png')}")


if __name__ == "__main__":
    main()
