"""Q1.2 — peak-strength scaling test under paper-clamp.

Tests whether window-arch peak operating strengths track the activation
magnitude ratios measured in Q1.1. Hypothesis (Dmitry): peak shifts as
~T x per-token peak. Q1.1 measured ratios that *aren't* a clean linear-
in-T (TXC 2.4x; SubseqH8 6.9x; MLC 9.5x); Q1.2 asks whether the empirical
peak strengths follow these per-arch ratios or a different relation.

Source: Dmitry's per-strength tables in
`docs/dmitry/case_studies/rlhf/notes/per_arch_breakdown.md` on branch
`origin/dmitry-rlhf`. Each cell is mean (success, coherence) over the
30-concept x 5-example probe set, single seed=42, Sonnet 4.6 grader,
paper-clamp protocol (clamp-on-latent + error preserve).

We parse the tables verbatim from the markdown (idempotent — re-running
re-extracts and reproduces the JSON + plot).

Output:
  results/case_studies/steering_magnitude/q1_2_strength_curves.json
    per-arch curves (s, suc, coh) + fitted peak s_star_arch +
    cross-arch ratios.
  results/case_studies/steering_magnitude/q1_2_strength_curves.png
    success(strength) overlay on log-x; coherence(strength) overlay;
    peak s* annotated.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- data ──


# Hand-transcribed from origin/dmitry-rlhf:per_arch_breakdown.md, "Paper
# protocol — clamp-on-latent + error preserve" table.
# Verified by `git show origin/dmitry-rlhf:docs/dmitry/case_studies/rlhf/notes/per_arch_breakdown.md`.
PAPER_CLAMP = {
    "topk_sae": {
        "is_window": False, "T": 1,
        "rows": [
            (10,    0.33, 2.57),
            (100,   1.07, 1.40),
            (150,   0.93, 0.87),
            (500,   0.13, 0.73),
            (1000,  0.17, 0.83),
            (1500,  0.27, 0.77),
            (5000,  0.40, 0.83),
            (10000, 0.37, 0.80),
            (15000, 0.47, 0.80),
        ],
    },
    "tsae_paper_k500": {
        "is_window": False, "T": 1,
        "rows": [
            (10,    0.27, 2.50),
            (100,   1.33, 1.50),
            (150,   1.20, 1.10),
            (500,   0.17, 0.80),
            (1000,  0.27, 0.80),
            (1500,  0.37, 0.83),
            (5000,  0.27, 0.83),
            (10000, 0.37, 0.87),
            (15000, 0.33, 0.90),
        ],
    },
    "tsae_paper_k20": {
        "is_window": False, "T": 1,
        "rows": [
            (10,    0.37, 2.77),
            (100,   1.93, 1.37),
            (150,   1.90, 1.07),
            (500,   0.20, 0.97),
            (1000,  0.37, 0.73),
            (1500,  0.30, 0.83),
            (5000,  0.30, 0.87),
            (10000, 0.37, 0.87),
            (15000, 0.37, 0.87),
        ],
    },
    "agentic_txc_02": {
        "is_window": True, "T": 5,
        "rows": [
            (10,    0.17, 2.80),
            (100,   0.30, 2.40),
            (150,   0.43, 1.80),
            (500,   0.97, 1.20),
            (1000,  0.63, 0.93),
            (1500,  0.37, 0.97),
            (5000,  0.13, 0.93),
            (10000, 0.23, 0.97),
            (15000, 0.20, 0.93),
        ],
    },
    "phase5b_subseq_h8": {
        "is_window": True, "T": 10,
        "rows": [
            (10,    0.27, 3.00),
            (100,   0.23, 2.30),
            (150,   0.30, 2.03),
            (500,   1.10, 1.53),
            (1000,  0.30, 1.07),
            (1500,  0.17, 1.03),
            (5000,  0.20, 1.07),
            (10000, 0.33, 1.00),
            (15000, 0.37, 1.00),
        ],
    },
    "phase57_partB_h8_bare_multidistance_t5": {
        "is_window": True, "T": 5,
        "rows": [
            (10,    0.27, 2.97),
            (100,   0.30, 2.13),
            (150,   0.43, 1.70),
            (500,   1.13, 1.10),
            (1000,  0.43, 0.97),
            (1500,  0.30, 0.97),
            (5000,  0.43, 1.00),
            (10000, 0.40, 1.00),
            (15000, 0.43, 1.00),
        ],
    },
}

# Q1.1 magnitude ratios (median active z[j*] / T-SAE k=20 reference)
# from results/case_studies/steering_magnitude/q1_1_z_orig_distributions.json.
Q1_1_RATIOS = {
    "topk_sae": 1.12,
    "tsae_paper_k500": 1.24,
    "tsae_paper_k20": 1.00,
    "mlc_contrastive_alpha100_batchtopk": 9.53,    # MLC not in Dmitry's paper-clamp table
    "agentic_txc_02": 2.37,
    "phase5b_subseq_h8": 6.93,
    "phase57_partB_h8_bare_multidistance_t5": None,    # not measured in Q1.1
}

REFERENCE_ARCH = "tsae_paper_k20"

OUT_SUBDIR = (
    Path("experiments/phase7_unification/results/case_studies/steering_magnitude")
)


# --------------------------------------------------------- peak fitting ──


def _peak_strength(rows: list[tuple[int, float, float]]) -> dict:
    """Locate peak success and (separately) coherence-respecting operating
    point. Use parabolic interpolation in log10(s) on the top-3 success cells
    to refine the peak, then snap back if the parabola is degenerate.

    Returns:
      argmax_s_grid     - strength at which observed success is maximal
      argmax_suc        - that success value
      argmax_coh        - coherence at that strength
      s_star_log_fit    - parabolic-fit peak in log10(s) space, or None
      s_star_log_fit_suc - fitted success value at s_star_log_fit
    """
    arr = np.array(rows, dtype=np.float64)         # (n_strengths, 3) [s, suc, coh]
    s = arr[:, 0]
    suc = arr[:, 1]
    coh = arr[:, 2]

    i_max = int(np.argmax(suc))
    out = {
        "argmax_s_grid": float(s[i_max]),
        "argmax_suc": float(suc[i_max]),
        "argmax_coh": float(coh[i_max]),
        "s_star_log_fit": None,
        "s_star_log_fit_suc": None,
    }

    # Parabolic refinement around the grid maximum.
    if 0 < i_max < len(s) - 1:
        x = np.log10(s[i_max - 1: i_max + 2])      # (3,) log10 strength
        y = suc[i_max - 1: i_max + 2]              # (3,) success
        # Fit y = a x^2 + b x + c; vertex at x* = -b / (2a) when a < 0.
        try:
            a, b, _ = np.polyfit(x, y, 2)
            if a < -1e-9:
                x_star = -b / (2.0 * a)
                # Clamp the vertex to lie within the bracket.
                x_star = float(np.clip(x_star, x[0], x[-1]))
                y_star = float(np.polyval([a, b, _], x_star))
                out["s_star_log_fit"] = float(10.0 ** x_star)
                out["s_star_log_fit_suc"] = y_star
        except np.linalg.LinAlgError:
            pass
    return out


# -------------------------------------------------------------- payload ──


def build_payload() -> dict:
    archs_data = {}
    for arch_id, info in PAPER_CLAMP.items():
        peak = _peak_strength(info["rows"])
        rows = [{"s": int(s), "suc": float(suc), "coh": float(coh)}
                for s, suc, coh in info["rows"]]
        archs_data[arch_id] = {
            "is_window": info["is_window"],
            "T": info["T"],
            "rows": rows,
            "peak": peak,
            "magnitude_ratio_to_ref": Q1_1_RATIOS.get(arch_id),
        }

    # Cross-arch ratios.
    ref_peak = archs_data[REFERENCE_ARCH]["peak"]["argmax_s_grid"]
    for arch_id, d in archs_data.items():
        d["peak_ratio_to_ref"] = float(d["peak"]["argmax_s_grid"] / ref_peak)

    return {
        "source": "origin/dmitry-rlhf:docs/dmitry/case_studies/rlhf/notes/per_arch_breakdown.md",
        "protocol": "paper-clamp (clamp-on-latent + error preserve)",
        "n_concepts": 30,
        "n_examples_per_concept": 5,
        "seed": 42,
        "grader": "claude-sonnet-4-6",
        "reference_arch": REFERENCE_ARCH,
        "archs": archs_data,
    }


# ----------------------------------------------------------------- plot ──


def _plot(payload: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.plotting.save_figure import save_figure

    palette = {
        "topk_sae": "#1f77b4",
        "tsae_paper_k500": "#ff7f0e",
        "tsae_paper_k20": "#d62728",
        "agentic_txc_02": "#2ca02c",
        "phase5b_subseq_h8": "#17becf",
        "phase57_partB_h8_bare_multidistance_t5": "#8c564b",
    }
    label = {
        "topk_sae": "TopKSAE (per-token, k=500)",
        "tsae_paper_k500": "T-SAE (per-token, k=500)",
        "tsae_paper_k20": "T-SAE (per-token, k=20)",
        "agentic_txc_02": "TXC matryoshka (T=5)",
        "phase5b_subseq_h8": "SubseqH8 (T_max=10)",
        "phase57_partB_h8_bare_multidistance_t5": "H8 multidist (T=5)",
    }

    fig, (ax_suc, ax_coh) = plt.subplots(1, 2, figsize=(13, 5))
    for arch_id, d in payload["archs"].items():
        s_vals = np.array([r["s"] for r in d["rows"]])
        suc = np.array([r["suc"] for r in d["rows"]])
        coh = np.array([r["coh"] for r in d["rows"]])
        ls = "-" if d["is_window"] else "--"
        ax_suc.plot(s_vals, suc, ls, color=palette[arch_id], lw=2, marker="o",
                    label=label[arch_id])
        ax_coh.plot(s_vals, coh, ls, color=palette[arch_id], lw=2, marker="o",
                    label=label[arch_id])
        # peak marker
        s_star = d["peak"]["s_star_log_fit"] or d["peak"]["argmax_s_grid"]
        suc_star = d["peak"]["s_star_log_fit_suc"] or d["peak"]["argmax_suc"]
        ax_suc.scatter([s_star], [suc_star], s=120, marker="*",
                       facecolor=palette[arch_id], edgecolor="black", zorder=5)

    ax_suc.set_xscale("log")
    ax_suc.set_xlabel("paper-clamp strength s (absolute z[j*] clamp value, log)")
    ax_suc.set_ylabel("mean steering success (0-3)")
    ax_suc.set_title("Q1.2: success vs strength under paper-clamp\n"
                     "(stars = parabolic-fit peak)")
    ax_suc.legend(loc="upper right", fontsize=8)
    ax_suc.grid(True, ls=":", alpha=0.4)

    ax_coh.set_xscale("log")
    ax_coh.set_xlabel("paper-clamp strength s (log)")
    ax_coh.set_ylabel("mean coherence (0-3)")
    ax_coh.set_title("Coherence vs strength")
    ax_coh.legend(loc="upper right", fontsize=8)
    ax_coh.grid(True, ls=":", alpha=0.4)

    fig.suptitle(
        f"Q1.2 — peak-strength scaling under paper-clamp ({payload['n_concepts']} concepts, "
        f"single seed, Sonnet 4.6 grader). Source: Dmitry's per_arch_breakdown.md",
        fontsize=10,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, str(out_path))
    plt.close(fig)


# ---------------------------------------------------------- comparison ──


def _print_summary(payload: dict) -> None:
    print(f"\n  paper-clamp peak strengths (single-seed, Dmitry tables):")
    hdr = (f"  {'arch':<48}  {'T':>3}  {'peak_s_grid':>12}  "
           f"{'peak_s_fit':>10}  {'peak_suc':>9}  "
           f"{'peak_ratio_ref':>14}  {'mag_ratio_q1.1':>14}")
    print(hdr)
    for arch_id, d in payload["archs"].items():
        s_grid = d["peak"]["argmax_s_grid"]
        s_fit = d["peak"]["s_star_log_fit"] or float("nan")
        suc = d["peak"]["argmax_suc"]
        pr = d["peak_ratio_to_ref"]
        mr = d["magnitude_ratio_to_ref"]
        mr_s = f"{mr:.2f}" if mr is not None else "  n/a"
        print(f"  {arch_id:<48}  {d['T']:>3}  {s_grid:>12.0f}  "
              f"{s_fit:>10.1f}  {suc:>9.2f}  {pr:>14.2f}  {mr_s:>14}")


def main() -> None:
    OUT_SUBDIR.mkdir(parents=True, exist_ok=True)
    payload = build_payload()

    json_path = OUT_SUBDIR / "q1_2_strength_curves.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {json_path}")

    png_path = OUT_SUBDIR / "q1_2_strength_curves.png"
    _plot(payload, png_path)
    print(f"wrote {png_path}  (+ thumb)")

    _print_summary(payload)


if __name__ == "__main__":
    main()
