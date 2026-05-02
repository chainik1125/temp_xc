"""Unified Pareto frontier plot — all matched-sparsity archs (Y's + W's).

For each arch + protocol, computes:
  - Multi-seed mean of (success, coh) per s_norm, averaged across available seeds
  - Pareto-optimal points across all archs at each strength

Produces:
  - Pareto plot (success vs coh) with all archs' (success, coh) points; upper envelope highlighted
  - Bar plot of peak success at coh ≥ 1.5 vs anchor 1.10
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

BASE = Path("/workspace/temp_xc/experiments/phase7_unification/results/case_studies")
ANCHOR_15 = 1.133  # T-SAE k=20 peak success at coh ≥ 1.5 (same-pod n=3 sd=42+sd=1+sd=2; W retrained 2026-05-01)


# Inventory: (arch_id, protocol_subdirs_per_seed, label, color)
INVENTORY = [
    # Anchor — single seed
    ("tsae_paper_k20", "T-SAE k=20 (anchor)", "blue", [
        ("steering_paper_normalised",         "right-edge", 42),
        ("steering_paper_normalised_seed1",   "right-edge", 1),
        ("steering_paper_normalised_seed2",   "right-edge", 2),
    ]),
    # Y — bare antidead (multi-seed)
    ("txc_bare_antidead_t2_kpos20", "T=2 bare", "orange", [
        ("steering_paper_normalised",                       "right-edge",   42),
        ("steering_paper_normalised_seed1",                 "right-edge",   1),
        ("steering_paper_normalised_seed2",                 "right-edge",   2),
        ("steering_paper_window_perposition",               "per-position", 42),
        ("steering_paper_window_perposition_seed1",         "per-position", 1),
        ("steering_paper_window_perposition_seed2",         "per-position", 2),        ("steering_paper_window_tiled_broadcast",          "tiled-broadcast", 42),
        ("steering_paper_window_tiled_broadcast_seed1",    "tiled-broadcast", 1),
        ("steering_paper_window_tiled_broadcast_seed2",    "tiled-broadcast", 2),

    ]),
    ("txc_bare_antidead_t5_kpos20", "T=5 bare", "green", [
        ("steering_paper_normalised",                       "right-edge",   42),
        ("steering_paper_normalised_seed1",                 "right-edge",   1),
        ("steering_paper_window_perposition",               "per-position", 42),
        ("steering_paper_window_perposition_seed1",         "per-position", 1),
    ]),
    ("txc_bare_antidead_t5_kwin20", "T=5 bare k_win=20", "darkgreen", [
        ("steering_paper_normalised",                       "right-edge",   42),
        ("steering_paper_window_perposition",               "per-position", 42),
    ]),
    # Y — H8 multidistance shifts=(T,)
    ("txc_h8_t2_kpos20_shifts2", "T=2 H8 shifts=(T,)", "red", [
        ("steering_paper_normalised",                       "right-edge",   42),
        ("steering_paper_normalised_seed1",                 "right-edge",   1),
        ("steering_paper_normalised_seed2",                 "right-edge",   2),
        ("steering_paper_window_perposition",               "per-position", 42),
        ("steering_paper_window_perposition_seed1",         "per-position", 1),
        ("steering_paper_window_perposition_seed2",         "per-position", 2),        ("steering_paper_window_tiled_broadcast",          "tiled-broadcast", 42),
        ("steering_paper_window_tiled_broadcast_seed1",    "tiled-broadcast", 1),
        ("steering_paper_window_tiled_broadcast_seed2",    "tiled-broadcast", 2),

    ]),
    ("txc_h8_t3_kpos20_shifts3", "T=3 H8 shifts=(T,)", "salmon", [
        ("steering_paper_normalised",                       "right-edge",   42),
        ("steering_paper_window_perposition",               "per-position", 42),
    ]),
    ("txc_h8_t5_kpos20_shifts5", "T=5 H8 shifts=(T,)", "darkred", [
        ("steering_paper_normalised",                       "right-edge",   42),
        ("steering_paper_normalised_seed1",                 "right-edge",   1),
        ("steering_paper_window_perposition",               "per-position", 42),
        ("steering_paper_window_perposition_seed1",         "per-position", 1),
    ]),
    # Y — grown chain
    ("txc_bare_antidead_t3_kpos20_grownFromT2sd42", "T=3 grown", "purple", [
        ("steering_paper_normalised",                       "right-edge",   42),
        ("steering_paper_window_perposition",               "per-position", 42),        ("steering_paper_window_tiled_broadcast",          "tiled-broadcast", 42),
        ("steering_paper_window_tiled_broadcast_seed1",    "tiled-broadcast", 1),
        ("steering_paper_window_tiled_broadcast_seed2",    "tiled-broadcast", 2),

    ]),
    ("txc_bare_antidead_t5_kpos20_grownFromT2sd42", "T=5 grown direct", "violet", [
        ("steering_paper_normalised",                       "right-edge",   42),
        ("steering_paper_window_perposition",               "per-position", 42),
    ]),
    ("txc_bare_antidead_t4_kpos20_grownChainFromT3", "T=4 grown chain", "indigo", [
        ("steering_paper_normalised",                       "right-edge",   42),
        ("steering_paper_window_perposition",               "per-position", 42),
    ]),
    ("txc_bare_antidead_t5_kpos20_grownChainFromT4", "T=5 grown chain", "navy", [
        ("steering_paper_normalised",                       "right-edge",   42),
        ("steering_paper_window_perposition",               "per-position", 42),
    ]),
    # Y — T-SAE warm-start
    ("txc_bare_antidead_t2_kpos20_ws_tsae_encoder", "T=2 T-SAE warm-start", "gold", [
        ("steering_paper_normalised",                       "right-edge",   42),
        ("steering_paper_window_perposition",               "per-position", 42),        ("steering_paper_window_tiled_broadcast",          "tiled-broadcast", 42),
        ("steering_paper_window_tiled_broadcast_seed1",    "tiled-broadcast", 1),
        ("steering_paper_window_tiled_broadcast_seed2",    "tiled-broadcast", 2),

    ]),
    # W's cells
    ("txc_bare_antidead_t3_kpos20", "T=3 bare (W's cell C)", "cyan", [
        ("steering_paper_normalised",                       "right-edge",   42),
        ("steering_paper_window_perposition",               "per-position", 42),
    ]),
    ("agentic_txc_02_kpos20", "T=5 matryoshka (W's cell E)", "teal", [
        ("steering_paper_normalised",                       "right-edge",   42),
        ("steering_paper_window_perposition",               "per-position", 42),
    ]),
    # W's MYSTERY archs (multi-seed verified)
    ("txc_maxpool_h8_t2_kpos20_shifts2", "T=2 MaxPool (W mystery)", "magenta", [
        ("steering_paper_normalised",                       "right-edge",   42),
        ("steering_paper_normalised_seed1",                 "right-edge",   1),
        ("steering_paper_normalised_seed2",                 "right-edge",   2),
        ("steering_paper_window_perposition",               "per-position", 42),
        ("steering_paper_window_perposition_seed1",         "per-position", 1),
        ("steering_paper_window_perposition_seed2",         "per-position", 2),        ("steering_paper_window_tiled_broadcast",          "tiled-broadcast", 42),
        ("steering_paper_window_tiled_broadcast_seed1",    "tiled-broadcast", 1),
        ("steering_paper_window_tiled_broadcast_seed2",    "tiled-broadcast", 2),

    ]),
    ("txc_contrastive_h8_t2_kpos20_shifts2", "T=2 Contrastive-merge (W mystery)", "deeppink", [
        ("steering_paper_normalised",                       "right-edge",   42),
        ("steering_paper_normalised_seed1",                 "right-edge",   1),
        ("steering_paper_normalised_seed2",                 "right-edge",   2),
        ("steering_paper_window_perposition",               "per-position", 42),
        ("steering_paper_window_perposition_seed1",         "per-position", 1),
        ("steering_paper_window_perposition_seed2",         "per-position", 2),        ("steering_paper_window_tiled_broadcast",          "tiled-broadcast", 42),
        ("steering_paper_window_tiled_broadcast_seed1",    "tiled-broadcast", 1),
        ("steering_paper_window_tiled_broadcast_seed2",    "tiled-broadcast", 2),

    ]),
    # Y's Galaxy archs (multi-seed verified)
    ("txc_maxpool_t2_kpos20", "T=2 Galaxy 6 max-pool (Y)", "#ff7f0e", [
        ("steering_paper_normalised",                       "right-edge",   42),
        ("steering_paper_normalised_seed1",                 "right-edge",   1),
        ("steering_paper_normalised_seed2",                 "right-edge",   2),
        ("steering_paper_window_perposition",               "per-position", 42),
        ("steering_paper_window_perposition_seed1",         "per-position", 1),
        ("steering_paper_window_perposition_seed2",         "per-position", 2),        ("steering_paper_window_tiled_broadcast",          "tiled-broadcast", 42),
        ("steering_paper_window_tiled_broadcast_seed1",    "tiled-broadcast", 1),
        ("steering_paper_window_tiled_broadcast_seed2",    "tiled-broadcast", 2),

    ]),
    ("txc_softmaxpool_t2_kpos20", "T=2 Galaxy 8 SoftMaxPool (Y)", "#2ca02c", [
        ("steering_paper_normalised",                       "right-edge",   42),
        ("steering_paper_normalised_seed1",                 "right-edge",   1),
        ("steering_paper_normalised_seed2",                 "right-edge",   2),
        ("steering_paper_window_perposition",               "per-position", 42),
        ("steering_paper_window_perposition_seed1",         "per-position", 1),
        ("steering_paper_window_perposition_seed2",         "per-position", 2),        ("steering_paper_window_tiled_broadcast",          "tiled-broadcast", 42),
        ("steering_paper_window_tiled_broadcast_seed1",    "tiled-broadcast", 1),
        ("steering_paper_window_tiled_broadcast_seed2",    "tiled-broadcast", 2),

    ]),
    ("txc_softmax_pool_h8_t2_kpos20_shifts2", "T=2 Galaxy 11 SoftMaxPool+H8 (Y)", "#17becf", [
        ("steering_paper_normalised",                       "right-edge",   42),
        ("steering_paper_normalised_seed1",                 "right-edge",   1),
        ("steering_paper_normalised_seed2",                 "right-edge",   2),
        ("steering_paper_window_perposition",               "per-position", 42),
        ("steering_paper_window_perposition_seed1",         "per-position", 1),
        ("steering_paper_window_perposition_seed2",         "per-position", 2),        ("steering_paper_window_tiled_broadcast",          "tiled-broadcast", 42),
        ("steering_paper_window_tiled_broadcast_seed1",    "tiled-broadcast", 1),
        ("steering_paper_window_tiled_broadcast_seed2",    "tiled-broadcast", 2),

    ]),
    ("txc_softmaxpool_t3_kpos20", "T=3 Galaxy 18 SoftMaxPool (Y)", "#bcbd22", [
        ("steering_paper_normalised",                       "right-edge",   42),
        ("steering_paper_normalised_seed1",                 "right-edge",   1),
        ("steering_paper_normalised_seed2",                 "right-edge",   2),
        ("steering_paper_window_perposition",               "per-position", 42),
        ("steering_paper_window_perposition_seed1",         "per-position", 1),
        ("steering_paper_window_perposition_seed2",         "per-position", 2),
        ("steering_paper_window_tiled_broadcast",           "tiled-broadcast", 42),
        ("steering_paper_window_tiled_broadcast_seed1",     "tiled-broadcast", 1),
        ("steering_paper_window_tiled_broadcast_seed2",     "tiled-broadcast", 2),
    ]),
    ("txc_softmaxpool_t5_kpos20", "T=5 Galaxy 23 SoftMaxPool (Y)", "#e377c2", [
        ("steering_paper_normalised",                       "right-edge",   42),
        ("steering_paper_normalised_seed1",                 "right-edge",   1),
        ("steering_paper_normalised_seed2",                 "right-edge",   2),
        ("steering_paper_window_tiled_broadcast",           "tiled-broadcast", 42),
        ("steering_paper_window_tiled_broadcast_seed1",     "tiled-broadcast", 1),
        ("steering_paper_window_tiled_broadcast_seed2",     "tiled-broadcast", 2),
    ]),
    ("txc_lsepool_t2_kpos20", "T=2 Galaxy 20 LSE-pool (Y)", "#7f7f7f", [
        ("steering_paper_normalised",                       "right-edge",   42),
        ("steering_paper_normalised_seed1",                 "right-edge",   1),
        ("steering_paper_normalised_seed2",                 "right-edge",   2),
        ("steering_paper_window_perposition",               "per-position", 42),
        ("steering_paper_window_perposition_seed1",         "per-position", 1),
        ("steering_paper_window_perposition_seed2",         "per-position", 2),
        ("steering_paper_window_tiled_broadcast",           "tiled-broadcast", 42),
        ("steering_paper_window_tiled_broadcast_seed1",     "tiled-broadcast", 1),
        ("steering_paper_window_tiled_broadcast_seed2",     "tiled-broadcast", 2),
    ]),
    ("txc_galaxy4_t2_kw10_kp10", "T=2 Galaxy 4 hierarchical (Y)", "#1f77b4", [
        ("steering_paper_normalised",                       "right-edge",   42),
        ("steering_paper_normalised_seed1",                 "right-edge",   1),
        ("steering_paper_normalised_seed2",                 "right-edge",   2),
        ("steering_paper_window_perposition",               "per-position", 42),
        ("steering_paper_window_perposition_seed1",         "per-position", 1),
        ("steering_paper_window_perposition_seed2",         "per-position", 2),
        ("steering_paper_window_tiled_broadcast",           "tiled-broadcast", 42),
        ("steering_paper_window_tiled_broadcast_seed1",     "tiled-broadcast", 1),
        ("steering_paper_window_tiled_broadcast_seed2",     "tiled-broadcast", 2),
    ]),
    # W's T=10 deadzone-escape chain (single seed; n=3 verification TODO)
    ("txc_h8_t10_kpos20_shifts10", "T=10 OBLIT shifts=10 (W deadzone)", "#999999", [
        ("steering_paper_normalised",                       "right-edge",      42),
        ("steering_paper_window_tiled_broadcast",           "tiled-broadcast", 42),
    ]),
    ("txc_h8_t10_kpos20_shifts2", "T=10 H8 shifts=2 (W deadzone)", "#bbbbbb", [
        ("steering_paper_normalised",                       "right-edge",      42),
        ("steering_paper_window_tiled_broadcast",           "tiled-broadcast", 42),
    ]),
    ("subseq_h8_tmax10_tsamp5_kpos20_shifts2_ctg",
     "T=10 subseq contig (W deadzone)", "#5dade2", [
        ("steering_paper_normalised",                       "right-edge",      42),
        ("steering_paper_window_tiled_broadcast",           "tiled-broadcast", 42),
    ]),
    ("subseq_h8_tmax10_tsamp5_kpos20_shifts2_gauss_s1.5_3.0_g2",
     "T=10 subseq Gaussian (W) ⭐", "#ff8c00", [
        ("steering_paper_normalised",                       "right-edge",      42),
        ("steering_paper_window_tiled_broadcast",           "tiled-broadcast", 42),
    ]),
    ("spatial_matry_h8_t10_kpos20_shifts2_pref3686_9216_18432_sub1_5_10_indep_uniform_contr",
     "T=10 sp.M indep+unif (W)", "#aed6f1", [
        ("steering_paper_normalised",                       "right-edge",      42),
        ("steering_paper_window_tiled_broadcast",           "tiled-broadcast", 42),
    ]),
    ("spatial_matry_h8_t10_kpos20_shifts2_pref3686_9216_18432_sub1_5_10_nested_uniform_contr",
     "T=10 sp.M nest+unif (W)", "#85c1e9", [
        ("steering_paper_normalised",                       "right-edge",      42),
        ("steering_paper_window_tiled_broadcast",           "tiled-broadcast", 42),
    ]),
    ("spatial_matry_h8_t10_kpos20_shifts2_pref3686_9216_18432_sub1_5_10_indep_gauss_s1.5_3.0_g2_contr",
     "T=10 sp.M indep+Gauss (W)", "#5499c7", [
        ("steering_paper_normalised",                       "right-edge",      42),
        ("steering_paper_window_tiled_broadcast",           "tiled-broadcast", 42),
    ]),
    ("spatial_matry_h8_t10_kpos20_shifts2_pref3686_9216_18432_sub1_5_10_nested_gauss_s1.5_3.0_g2_contr",
     "T=10 sp.M nest+Gauss (W)", "#2874a6", [
        ("steering_paper_normalised",                       "right-edge",      42),
        ("steering_paper_window_tiled_broadcast",           "tiled-broadcast", 42),
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
        succ = []
        coh = []
        for s in s_vals:
            pairs = by_s[s]
            ss = [p["success_grade"] for p in pairs if p.get("success_grade") is not None]
            cs = [p.get("coherence_grade") for p in pairs if p.get("coherence_grade") is not None]
            succ.append(np.mean(ss) if ss else None)
            coh.append(np.mean(cs) if cs else None)
        by_proto[proto].append((np.array(s_vals), np.array(succ), np.array(coh)))
    out = {}
    for proto, curves in by_proto.items():
        # Align to first curve's s_vals
        s_ref = curves[0][0]
        succ_stack = []
        coh_stack = []
        for s, su, co in curves:
            if len(s) == len(s_ref):
                succ_stack.append(su)
                coh_stack.append(co)
        if not succ_stack: continue
        out[proto] = (
            s_ref,
            np.mean(np.stack(succ_stack), axis=0),
            np.mean(np.stack(coh_stack), axis=0),
            len(succ_stack),
        )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path,
                   default=Path("/workspace/temp_xc/experiments/phase7_unification/results/case_studies/plots"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Gather all (arch, protocol) curves
    print(f"\n=== Gathering all matched-sparsity archs ===\n")
    all_curves = {}  # (arch_id, proto) → (s, succ, coh, n_seeds, label, color)
    for arch_id, label, color, subdir_list in INVENTORY:
        curves = get_curve_avg(arch_id, subdir_list)
        for proto, (s, succ, coh, n) in curves.items():
            all_curves[(arch_id, proto)] = (s, succ, coh, n, label, color)
            valid = [(ss, su, co) for ss, su, co in zip(s, succ, coh) if su is not None and co is not None]
            valid_15 = [v for v in valid if v[2] >= 1.5]
            peak15 = max(v[1] for v in valid_15) if valid_15 else None
            peak_unc = max(v[1] for v in valid)
            p15s = f"{peak15:.3f}" if peak15 is not None else "—"
            print(f"  {label:30s} {proto:13s} (n={n}): peak_unc={peak_unc:.3f}, peak15={p15s}")

    # ──────────────────── Pareto plot: all (success, coh) points across archs+strengths
    # 3-row vertical stack so each panel gets a dedicated row with its legend
    # parked outside on the right — no overlap with neighboring panels.
    fig, axes = plt.subplots(3, 1, figsize=(16, 18), constrained_layout=True)

    for proto_idx, proto_filter in enumerate(["right-edge", "per-position", "tiled-broadcast"]):
        ax = axes[proto_idx]
        # Plot each arch's curve
        all_pts = []
        for (arch_id, proto), (s, succ, coh, n, label, color) in all_curves.items():
            # T-SAE k=20 has T=1: right-edge == per-position (trivially); show on both panels
            if proto != proto_filter and arch_id != "tsae_paper_k20":
                continue
            # T-SAE shown on RE panel only (tiled-broadcast/per-position don't apply to T=1)
            if arch_id == "tsae_paper_k20" and proto_filter != "right-edge":
                if proto_filter == "per-position":
                    pass  # show on PP panel too (T=1 trivially equivalent)
                else:
                    continue
            marker = {"right-edge": "o", "per-position": "^", "tiled-broadcast": "s"}.get(proto, "x")
            display_label = label
            if arch_id == "tsae_paper_k20" and proto_filter == "per-position":
                display_label = f"{label} (T=1, RE=PP)"
            # Highlight T-SAE k=20 anchor: thicker dashed line, brighter alpha,
            # larger markers, top zorder so it draws on top of everything else.
            is_anchor = arch_id == "tsae_paper_k20"
            lw = 3.0 if is_anchor else 1.0
            alpha = 1.0 if is_anchor else 0.65
            ms = 9 if is_anchor else 4
            ls = "--" if is_anchor else "-"
            zorder = 20 if is_anchor else 3
            anchor_label = f"⭐ {display_label} (n={n}) [ANCHOR]" if is_anchor else f"{display_label} (n={n})"
            ax.plot(coh, succ, marker=marker, markersize=ms, color=color,
                    alpha=alpha, linewidth=lw, linestyle=ls, zorder=zorder,
                    label=anchor_label,
                    markeredgecolor="black" if is_anchor else None,
                    markeredgewidth=1.2 if is_anchor else 0)
            for ss, su, co in zip(s, succ, coh):
                if su is not None and co is not None:
                    all_pts.append((co, su, label))
        # Pareto frontier (upper envelope: for each coh bin, max success)
        # Bin coh into 0.1 bins
        bins = np.arange(0.5, 3.2, 0.1)
        bin_max = {}
        for co, su, lab in all_pts:
            b = round(co, 1)
            if b not in bin_max or su > bin_max[b][0]:
                bin_max[b] = (su, lab)
        # Sort by coh, draw envelope
        bins_sorted = sorted(bin_max.keys())
        envelope_su = [bin_max[b][0] for b in bins_sorted]
        ax.plot(bins_sorted, envelope_su, color="black", linewidth=2, linestyle="--",
                label="Pareto envelope", alpha=0.6, zorder=15)
        # Mark coh=1.5 threshold
        ax.axvline(1.5, color="grey", linestyle=":", linewidth=0.8)
        ax.text(1.51, 0.05, "coh=1.5", fontsize=8, color="grey")
        # Mark anchor's peak15
        ax.axhline(ANCHOR_15, color="blue", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.text(0.6, ANCHOR_15+0.02, f"T-SAE k=20 peak15={ANCHOR_15}", fontsize=8, color="blue")
        ax.axhline(ANCHOR_15+0.27, color="green", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.text(0.6, ANCHOR_15+0.27+0.02, f"WIN threshold ({ANCHOR_15+0.27})", fontsize=8, color="green")
        ax.set_xlabel("mean coherence")
        ax.set_ylabel("mean success")
        ax.set_title(f"{proto_filter} protocol")
        # Legend OUTSIDE the plot to the right of each panel
        ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  framealpha=0.92, ncol=1, borderaxespad=0.0)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.6, 3.1)
        ax.set_ylim(-0.05, 2.0)

    fig.suptitle("Phase 7 Y+W matched-sparsity Pareto: success vs coherence (multi-seed averaged)\n"
                 "All archs at k_pos=20 (or k_win=20 / k_pos=10 / k_win=200 wild variants). "
                 "T-SAE k=20 ANCHOR ⭐ highlighted in bold blue dashed.")
    out = args.out_dir / "unified_pareto_matched_sparsity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(args.out_dir / "unified_pareto_matched_sparsity.thumb.png", dpi=48, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")

    # ──────────────────── Summary bar plot: peak15 per arch+proto vs anchor
    fig, ax = plt.subplots(figsize=(14, 6))
    rows = []
    for (arch_id, proto), (s, succ, coh, n, label, color) in all_curves.items():
        valid = [(ss, su, co) for ss, su, co in zip(s, succ, coh) if su is not None and co is not None]
        valid_15 = [v for v in valid if v[2] >= 1.5]
        peak15 = max(v[1] for v in valid_15) if valid_15 else 0.0
        rows.append((label, proto, peak15, n, color))
    # Sort by peak15 descending
    rows.sort(key=lambda r: r[2], reverse=True)
    labels = [f"{r[0]} ({r[1]}, n={r[3]})" for r in rows]
    peaks = [r[2] for r in rows]
    colors = [r[4] for r in rows]
    edgecolors = ["black" if r[1] == "per-position" else "grey" for r in rows]
    bars = ax.barh(range(len(rows)), peaks, color=colors, edgecolor=edgecolors, linewidth=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("peak success at coh ≥ 1.5")
    ax.axvline(ANCHOR_15, color="blue", linestyle="--", linewidth=1, label=f"T-SAE k=20 anchor ({ANCHOR_15})")
    ax.axvline(ANCHOR_15+0.27, color="green", linestyle="--", linewidth=1, label=f"WIN threshold ({ANCHOR_15+0.27})")
    ax.axvline(ANCHOR_15-0.27, color="red", linestyle="--", linewidth=1, label=f"LOSS threshold ({ANCHOR_15-0.27})")
    for i, (label, proto, peak, n, _) in enumerate(rows):
        ax.text(peak + 0.02, i, f"{peak:.3f}", fontsize=7, va="center")
    ax.set_title("Phase 7 Y+W matched-sparsity ranking — peak success at coh ≥ 1.5\n"
                 "(black edges = per-position protocol; grey edges = right-edge)")
    ax.legend(loc="lower right", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(peaks) * 1.15 + 0.1)
    plt.tight_layout()
    out = args.out_dir / "unified_ranking_matched_sparsity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(args.out_dir / "unified_ranking_matched_sparsity.thumb.png", dpi=48, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # ──────────────────── Per-position-only ranking (cleaner)
    fig, ax = plt.subplots(figsize=(13, 7))
    rows = [(label, proto, peak15, n, color) for (arch_id, proto), (s, succ, coh, n, label, color) in all_curves.items()
            for valid in [[(ss, su, co) for ss, su, co in zip(s, succ, coh) if su is not None and co is not None]]
            for valid_15 in [[v for v in valid if v[2] >= 1.5]]
            for peak15 in [max(v[1] for v in valid_15) if valid_15 else 0.0]
            if proto == "per-position" or arch_id == "tsae_paper_k20"]
    rows.sort(key=lambda r: r[2], reverse=True)
    labels = [f"{r[0]} (n={r[3]})" for r in rows]
    peaks = [r[2] for r in rows]
    colors = [r[4] for r in rows]
    bars = ax.barh(range(len(rows)), peaks, color=colors, edgecolor="black", linewidth=0.8)
    # Highlight WIN cell
    for i, peak in enumerate(peaks):
        if peak >= ANCHOR_15 + 0.27:
            bars[i].set_edgecolor("gold")
            bars[i].set_linewidth(3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("peak success at coh ≥ 1.5  (per-position protocol; T=1 anchor: RE=PP)", fontsize=10)
    ax.axvline(ANCHOR_15, color="blue", linestyle="--", linewidth=1.2, label=f"T-SAE k=20 anchor ({ANCHOR_15})")
    ax.axvline(ANCHOR_15 + 0.27, color="green", linestyle="--", linewidth=1.5, label=f"WIN threshold (+0.27 = {ANCHOR_15 + 0.27})")
    ax.axvline(ANCHOR_15 - 0.27, color="red", linestyle="--", linewidth=1.2, label=f"LOSS threshold (-0.27)")
    for i, (label, proto, peak, n, _) in enumerate(rows):
        delta = peak - ANCHOR_15
        ax.text(peak + 0.015, i, f"{peak:.3f} (Δ={delta:+.2f})", fontsize=8, va="center")
    ax.set_title("Phase 7 Y+W matched-sparsity ranking — PER-POSITION protocol only\n"
                 "(gold edge = STRICT WIN; multi-seed anchor 1.167; T=2 H8 PP 3-seed at 1.400 Δ=+0.23 — TIE band)")
    ax.legend(loc="lower right", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(peaks) * 1.2 + 0.1)
    plt.tight_layout()
    out = args.out_dir / "unified_ranking_per_position.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(args.out_dir / "unified_ranking_per_position.thumb.png", dpi=48, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # ──────────────────── Sequential growth trajectory plot
    fig, ax = plt.subplots(figsize=(10, 5))
    # Build T → peak15 mapping from our cells
    growth_chain = []  # (T, peak15, label, color)
    for arch_id, label, color, _ in INVENTORY:
        if arch_id in all_curves and (arch_id, "per-position") in all_curves:
            s, succ, coh, n, _, _ = all_curves[(arch_id, "per-position")]
            valid_15 = [(ss, su, co) for ss, su, co in zip(s, succ, coh) if su is not None and co is not None and co >= 1.5]
            if valid_15:
                peak15 = max(v[1] for v in valid_15)
                growth_chain.append((arch_id, peak15, label, color))

    # Group by family
    families = {
        "Sequential growth chain": [
            ("txc_bare_antidead_t2_kpos20",                    2),
            ("txc_bare_antidead_t3_kpos20_grownFromT2sd42",    3),
            ("txc_bare_antidead_t4_kpos20_grownChainFromT3",   4),
            ("txc_bare_antidead_t5_kpos20_grownChainFromT4",   5),
        ],
        "Bare random-init": [
            ("txc_bare_antidead_t2_kpos20", 2),
            ("txc_bare_antidead_t3_kpos20", 3),
            ("txc_bare_antidead_t5_kpos20", 5),
        ],
        "H8 multidist + shifts=(T,)": [
            ("txc_h8_t2_kpos20_shifts2", 2),
            ("txc_h8_t3_kpos20_shifts3", 3),
            ("txc_h8_t5_kpos20_shifts5", 5),
        ],
    }
    family_colors = {"Sequential growth chain": "indigo", "Bare random-init": "orange", "H8 multidist + shifts=(T,)": "red"}
    for fam_name, cells in families.items():
        xs, ys = [], []
        for arch_id, T in cells:
            if (arch_id, "per-position") in all_curves:
                s, succ, coh, n, _, _ = all_curves[(arch_id, "per-position")]
                valid_15 = [(ss, su, co) for ss, su, co in zip(s, succ, coh) if su is not None and co is not None and co >= 1.5]
                if valid_15:
                    peak15 = max(v[1] for v in valid_15)
                    xs.append(T)
                    ys.append(peak15)
        ax.plot(xs, ys, marker="o", linewidth=2, markersize=10, color=family_colors[fam_name], label=fam_name)
        for x, y in zip(xs, ys):
            ax.text(x + 0.05, y + 0.01, f"{y:.2f}", fontsize=8)

    # Anchor line
    ax.axhline(ANCHOR_15, color="blue", linestyle="--", linewidth=1.2, label=f"T-SAE k=20 anchor ({ANCHOR_15})")
    ax.axhline(ANCHOR_15 + 0.27, color="green", linestyle="--", linewidth=1.5, label=f"WIN threshold ({ANCHOR_15+0.27})")
    ax.set_xlabel("T (window length)", fontsize=11)
    ax.set_ylabel("peak success at coh ≥ 1.5 (per-position)", fontsize=11)
    ax.set_title("Phase 7 Y+W matched-sparsity by T — per-position protocol\n"
                 "Sequential growth chain (T=2→T=5) preserves anchor; H8 shifts=(T,) lifts T=2 above WIN threshold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1.5, 5.5)
    ax.set_ylim(0.4, 1.6)
    plt.tight_layout()
    out = args.out_dir / "unified_growth_trajectory.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(args.out_dir / "unified_growth_trajectory.thumb.png", dpi=48, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # JSON dump
    summary = []
    for (arch_id, proto), (s, succ, coh, n, label, color) in all_curves.items():
        valid = [(ss, su, co) for ss, su, co in zip(s, succ, coh) if su is not None and co is not None]
        valid_15 = [v for v in valid if v[2] >= 1.5]
        peak15 = max(v[1] for v in valid_15) if valid_15 else None
        peak_unc = max(v[1] for v in valid)
        summary.append({
            "arch_id": arch_id, "label": label, "protocol": proto, "n_seeds": n,
            "peak_unc": float(peak_unc), "peak_coh_ge_1_5": float(peak15) if peak15 else None,
            "delta_vs_anchor": float(peak15 - ANCHOR_15) if peak15 else None,
        })
    out = args.out_dir / "unified_pareto_summary.json"
    out.write_text(json.dumps(sorted(summary, key=lambda x: -(x["peak_coh_ge_1_5"] or 0)), indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
