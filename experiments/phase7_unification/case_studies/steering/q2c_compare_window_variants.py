"""Q2.C — compare right-edge clamp (Q1.3) vs full-window clamp on T=5/T=10.

Tests Q2 candidate (C): "decompose the window into per-position contributions".
This script aggregates grades from steering_paper_normalised/ (right-edge,
Q1.3) and steering_paper_pos_full/ (full-window, this run) for the two
window archs and produces a side-by-side success(strength) comparison.

If full-window beats right-edge by >=0.3 peak success on either arch,
candidate (C) becomes a real protocol candidate. If not, it joins (B)
as rejected by Q1+Q2 evidence.

Output:
  results/case_studies/steering_magnitude/q2c_window_variant_comparison.json
  results/case_studies/steering_magnitude/q2c_window_variant_comparison.png
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.phase7_unification.case_studies._paths import CASE_STUDIES_DIR


WINDOW_ARCHS = [
    ("agentic_txc_02", 5),
    ("phase5b_subseq_h8", 10),
]
VARIANTS = [
    ("right_edge", "steering_paper_normalised", "right-edge clamp"),
    ("full_window", "steering_paper_pos_full", "full-window clamp"),
]
OUT_SUBDIR = (
    Path("experiments/phase7_unification/results/case_studies/steering_magnitude")
)


def _aggregate(arch_id: str, subdir: str) -> dict[int, tuple[float, float, int]]:
    path = CASE_STUDIES_DIR / subdir / arch_id / "grades.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    by_s: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for line in path.open():
        r = json.loads(line)
        if r.get("success_grade") is None or r.get("coherence_grade") is None:
            continue
        s = int(r["strength"])
        by_s[s].append((float(r["success_grade"]), float(r["coherence_grade"])))
    return {s: (float(np.mean([p[0] for p in pairs])),
                float(np.mean([p[1] for p in pairs])),
                len(pairs))
            for s, pairs in sorted(by_s.items())}


def _peak(rows: list[tuple[int, float, float]]) -> dict:
    arr = np.array(rows, dtype=np.float64)
    s, suc, coh = arr[:, 0], arr[:, 1], arr[:, 2]
    i = int(np.argmax(suc))
    out = {"argmax_s": float(s[i]), "argmax_suc": float(suc[i]),
           "argmax_coh": float(coh[i]), "s_star": None, "s_star_suc": None}
    if 0 < i < len(s) - 1:
        x = np.log10(s[i - 1: i + 2]); y = suc[i - 1: i + 2]
        try:
            a, b, c = np.polyfit(x, y, 2)
            if a < -1e-9:
                xs = float(np.clip(-b / (2 * a), x[0], x[-1]))
                out["s_star"] = float(10 ** xs)
                out["s_star_suc"] = float(np.polyval([a, b, c], xs))
        except np.linalg.LinAlgError:
            pass
    return out


def _build_payload() -> dict:
    payload = {"archs": {}}
    for arch_id, T in WINDOW_ARCHS:
        payload["archs"][arch_id] = {"T": T, "variants": {}}
        for vkey, subdir, label in VARIANTS:
            agg = _aggregate(arch_id, subdir)
            rows = [(s, suc, coh) for s, (suc, coh, n) in agg.items()]
            peak = _peak(rows)
            payload["archs"][arch_id]["variants"][vkey] = {
                "label": label, "subdir": subdir,
                "rows": [{"s": int(s), "suc": float(suc), "coh": float(coh), "n": int(n)}
                         for s, (suc, coh, n) in agg.items()],
                "peak": peak,
            }
    return payload


def _plot(payload: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.plotting.save_figure import save_figure

    palette = {
        ("agentic_txc_02", "right_edge"): "#2ca02c",
        ("agentic_txc_02", "full_window"): "#9eca7c",
        ("phase5b_subseq_h8", "right_edge"): "#17becf",
        ("phase5b_subseq_h8", "full_window"): "#7fcfe0",
    }
    arch_label = {
        "agentic_txc_02": "TXC matryoshka (T=5)",
        "phase5b_subseq_h8": "SubseqH8 (T_max=10)",
    }
    var_marker = {"right_edge": "o", "full_window": "s"}

    fig, (ax_suc, ax_coh) = plt.subplots(1, 2, figsize=(13, 5))
    for arch_id, d in payload["archs"].items():
        for vkey, info in d["variants"].items():
            s_vals = np.array([r["s"] for r in info["rows"]])
            suc = np.array([r["suc"] for r in info["rows"]])
            coh = np.array([r["coh"] for r in info["rows"]])
            ls = "-" if vkey == "right_edge" else "--"
            color = palette[(arch_id, vkey)]
            ax_suc.plot(s_vals, suc, ls, color=color, lw=2,
                        marker=var_marker[vkey],
                        label=f"{arch_label[arch_id]} -- {info['label']}")
            ax_coh.plot(s_vals, coh, ls, color=color, lw=2,
                        marker=var_marker[vkey])
            s_star = info["peak"]["s_star"] or info["peak"]["argmax_s"]
            suc_star = info["peak"]["s_star_suc"] or info["peak"]["argmax_suc"]
            ax_suc.scatter([s_star], [suc_star], s=140, marker="*",
                           facecolor=color, edgecolor="black", zorder=5)

    for ax in (ax_suc, ax_coh):
        ax.set_xscale("log")
        ax.grid(True, ls=":", alpha=0.4)
    ax_suc.set_xlabel("paper-clamp strength s (log)")
    ax_coh.set_xlabel("paper-clamp strength s (log)")
    ax_suc.set_ylabel("mean steering success (0-3)")
    ax_coh.set_ylabel("mean coherence (0-3)")
    ax_suc.set_title("Q2.C: success vs strength, right-edge vs full-window")
    ax_coh.set_title("Coherence vs strength")
    ax_suc.legend(loc="upper right", fontsize=8)
    fig.suptitle("Q2.C — does full-window injection help over right-edge?",
                 fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, str(out_path))
    plt.close(fig)


def _print_summary(payload: dict) -> None:
    print(f"\n  Q2.C: per-position window clamp comparison (paper-clamp protocol):")
    print(f"  {'arch':<26}  {'T':>3}  {'variant':<14}  {'peak_s':>7}  "
          f"{'peak_s_fit':>10}  {'peak_suc':>9}")
    for arch_id, d in payload["archs"].items():
        for vkey, info in d["variants"].items():
            peak = info["peak"]
            s_fit = peak["s_star"] or float("nan")
            print(f"  {arch_id:<26}  {d['T']:>3}  {vkey:<14}  "
                  f"{peak['argmax_s']:>7.0f}  {s_fit:>10.1f}  "
                  f"{peak['argmax_suc']:>9.2f}")


def main() -> None:
    OUT_SUBDIR.mkdir(parents=True, exist_ok=True)
    payload = _build_payload()
    json_path = OUT_SUBDIR / "q2c_window_variant_comparison.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {json_path}")
    png_path = OUT_SUBDIR / "q2c_window_variant_comparison.png"
    _plot(payload, png_path)
    print(f"wrote {png_path}  (+ thumb)")
    _print_summary(payload)


if __name__ == "__main__":
    main()
