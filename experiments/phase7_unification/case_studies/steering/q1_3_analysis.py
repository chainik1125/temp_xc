"""Q1.3 — analyse the finer-grid paper-clamp re-run.

Reads grades.jsonl from
`results/case_studies/steering_paper_normalised/<arch>/grades.jsonl` for each
arch, aggregates (mean success, mean coherence) per (arch, strength), fits a
parabolic peak in log10(s), and compares against:

  - Q1.2's coarser-grid peaks (Dmitry's tables).
  - Q1.1 magnitude ratios — the family-normalised single-point hypothesis
    predicts window archs peak at s = 100 × magnitude_ratio_arch.
  - The "universal 5x" hypothesis predicts window archs peak at ~500
    regardless of arch.

Outputs:
  results/case_studies/steering_magnitude/q1_3_finer_grid_curves.json
  results/case_studies/steering_magnitude/q1_3_finer_grid_curves.png
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.phase7_unification.case_studies._paths import CASE_STUDIES_DIR


ARCHS = [
    ("topk_sae", False, 1),
    ("tsae_paper_k500", False, 1),
    ("tsae_paper_k20", False, 1),
    ("agentic_txc_02", True, 5),
    ("phase5b_subseq_h8", True, 10),
]
REFERENCE_ARCH = "tsae_paper_k20"
SUBDIR = "steering_paper_normalised"
OUT_SUBDIR = (
    Path("experiments/phase7_unification/results/case_studies/steering_magnitude")
)

# Q1.1 magnitude ratios (median active z[j*] / T-SAE k=20 reference).
Q1_1_RATIOS = {
    "topk_sae": 1.12,
    "tsae_paper_k500": 1.24,
    "tsae_paper_k20": 1.00,
    "agentic_txc_02": 2.37,
    "phase5b_subseq_h8": 6.93,
}


def _aggregate(arch_id: str) -> dict[int, tuple[float, float, int]]:
    """Returns {strength: (mean_success, mean_coherence, n_concepts)}."""
    path = CASE_STUDIES_DIR / SUBDIR / arch_id / "grades.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    by_s: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for line in path.open():
        r = json.loads(line)
        if r.get("success_grade") is None or r.get("coherence_grade") is None:
            continue
        s = int(r["strength"])
        by_s[s].append((float(r["success_grade"]), float(r["coherence_grade"])))
    out = {}
    for s, pairs in sorted(by_s.items()):
        sucs = [p[0] for p in pairs]
        cohs = [p[1] for p in pairs]
        out[s] = (float(np.mean(sucs)), float(np.mean(cohs)), len(pairs))
    return out


def _peak_strength(rows: list[tuple[int, float, float]]) -> dict:
    """Same fit as Q1.2 for cross-comparability."""
    arr = np.array(rows, dtype=np.float64)
    s = arr[:, 0]; suc = arr[:, 1]; coh = arr[:, 2]
    i_max = int(np.argmax(suc))
    out = {
        "argmax_s_grid": float(s[i_max]),
        "argmax_suc": float(suc[i_max]),
        "argmax_coh": float(coh[i_max]),
        "s_star_log_fit": None,
        "s_star_log_fit_suc": None,
    }
    if 0 < i_max < len(s) - 1:
        x = np.log10(s[i_max - 1: i_max + 2])
        y = suc[i_max - 1: i_max + 2]
        try:
            a, b, c = np.polyfit(x, y, 2)
            if a < -1e-9:
                x_star = float(np.clip(-b / (2.0 * a), x[0], x[-1]))
                y_star = float(np.polyval([a, b, c], x_star))
                out["s_star_log_fit"] = float(10.0 ** x_star)
                out["s_star_log_fit_suc"] = y_star
        except np.linalg.LinAlgError:
            pass
    return out


def _build_payload() -> dict:
    archs_data = {}
    for arch_id, is_window, T in ARCHS:
        agg = _aggregate(arch_id)
        rows = [(s, suc, coh) for s, (suc, coh, n) in agg.items()]
        peak = _peak_strength(rows)
        archs_data[arch_id] = {
            "is_window": is_window,
            "T": T,
            "rows": [{"s": int(s), "suc": float(suc), "coh": float(coh), "n": int(n)}
                     for s, (suc, coh, n) in agg.items()],
            "peak": peak,
            "magnitude_ratio_to_ref": Q1_1_RATIOS.get(arch_id),
        }
    ref_peak = archs_data[REFERENCE_ARCH]["peak"]["argmax_s_grid"]
    for arch_id, d in archs_data.items():
        d["peak_ratio_to_ref"] = float(d["peak"]["argmax_s_grid"] / ref_peak)
    return {
        "source": "this run; finer grid {50,100,150,200,300,400,500,700,1000}",
        "protocol": "paper-clamp (clamp-on-latent + error preserve)",
        "n_concepts": 30,
        "n_examples_per_concept": 5,
        "seed": 42,
        "grader": "claude-sonnet-4-6",
        "reference_arch": REFERENCE_ARCH,
        "archs": archs_data,
    }


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
    }
    label = {
        "topk_sae": "TopKSAE (per-token, k=500)",
        "tsae_paper_k500": "T-SAE (per-token, k=500)",
        "tsae_paper_k20": "T-SAE (per-token, k=20)",
        "agentic_txc_02": "TXC matryoshka (T=5)",
        "phase5b_subseq_h8": "SubseqH8 (T_max=10)",
    }

    fig, (ax_suc, ax_coh) = plt.subplots(1, 2, figsize=(13, 5))
    for arch_id, d in payload["archs"].items():
        s_vals = np.array([r["s"] for r in d["rows"]])
        suc = np.array([r["suc"] for r in d["rows"]])
        coh = np.array([r["coh"] for r in d["rows"]])
        ls = "-" if d["is_window"] else "--"
        c = palette[arch_id]
        ax_suc.plot(s_vals, suc, ls, color=c, lw=2, marker="o", label=label[arch_id])
        ax_coh.plot(s_vals, coh, ls, color=c, lw=2, marker="o", label=label[arch_id])
        s_star = d["peak"]["s_star_log_fit"] or d["peak"]["argmax_s_grid"]
        suc_star = d["peak"]["s_star_log_fit_suc"] or d["peak"]["argmax_suc"]
        ax_suc.scatter([s_star], [suc_star], s=140, marker="*",
                       facecolor=c, edgecolor="black", zorder=5)
        # magnitude-normalised prediction marker (only for window archs).
        if d["is_window"] and d["magnitude_ratio_to_ref"]:
            s_pred_mag = 100.0 * d["magnitude_ratio_to_ref"]
            ax_suc.axvline(s_pred_mag, color=c, ls=":", lw=1, alpha=0.5)

    # Universal-5x prediction (~500) for window archs.
    ax_suc.axvline(500, color="gray", ls="-.", lw=1.0, alpha=0.4,
                   label="universal-5x prediction (s=500)")

    ax_suc.set_xscale("log")
    ax_suc.set_xlabel("paper-clamp strength s (absolute z[j*] clamp value, log)")
    ax_suc.set_ylabel("mean steering success (0-3)")
    ax_suc.set_title("Q1.3: success vs strength, finer grid\n"
                     "(stars = log-parabolic peak; "
                     "dotted = magnitude-normalised prediction per arch)")
    ax_suc.legend(loc="upper right", fontsize=8)
    ax_suc.grid(True, ls=":", alpha=0.4)

    ax_coh.set_xscale("log")
    ax_coh.set_xlabel("paper-clamp strength s (log)")
    ax_coh.set_ylabel("mean coherence (0-3)")
    ax_coh.set_title("Coherence vs strength")
    ax_coh.legend(loc="upper right", fontsize=8)
    ax_coh.grid(True, ls=":", alpha=0.4)

    fig.suptitle(
        f"Q1.3 — finer-grid paper-clamp ({payload['n_concepts']} concepts, "
        f"single seed, Sonnet 4.6 grader, prompt-cached system rubric)",
        fontsize=10,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, str(out_path))
    plt.close(fig)


def _print_summary(payload: dict) -> None:
    print(f"\nQ1.3 finer-grid paper-clamp peaks:")
    hdr = (f"  {'arch':<32}  {'T':>3}  {'peak_s_grid':>11}  {'peak_s_fit':>10}  "
           f"{'peak_suc':>9}  {'mag_pred_s':>11}  {'mag_ratio':>9}")
    print(hdr)
    for arch_id, d in payload["archs"].items():
        s_grid = d["peak"]["argmax_s_grid"]
        s_fit = d["peak"]["s_star_log_fit"] or float("nan")
        suc = d["peak"]["argmax_suc"]
        mr = d["magnitude_ratio_to_ref"]
        mag_pred = (100.0 * mr) if mr else float("nan")
        mr_s = f"{mr:.2f}" if mr is not None else "  n/a"
        print(f"  {arch_id:<32}  {d['T']:>3}  {s_grid:>11.0f}  {s_fit:>10.1f}  "
              f"{suc:>9.2f}  {mag_pred:>11.1f}  {mr_s:>9}")


def main() -> None:
    OUT_SUBDIR.mkdir(parents=True, exist_ok=True)
    payload = _build_payload()
    json_path = OUT_SUBDIR / "q1_3_finer_grid_curves.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {json_path}")
    png_path = OUT_SUBDIR / "q1_3_finer_grid_curves.png"
    _plot(payload, png_path)
    print(f"wrote {png_path}  (+ thumb)")
    _print_summary(payload)


if __name__ == "__main__":
    main()
