"""Plot: val Δ log p(sleeper) vs clean CE delta per architecture.

Low Δ log p = more suppression. Low CE delta = less damage.
Each architecture gets one color; all (f, α) candidates scatter; the chosen
(f*, α*) is marked with a star. Vertical line at the utility budget δ.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent


ARCH_COLORS = {
    # fra_proj's original 8-arch comparison (kept for compatibility).
    "mlc":         "#1f77b4",
    "txc_early":   "#ff7f0e",
    "txc_mid":     "#2ca02c",
    "txc_late":    "#d62728",
    "sae_layer0":  "#9467bd",
    "sae_layer1":  "#8c564b",
    "sae_layer2":  "#e377c2",
    "sae_layer3":  "#7f7f7f",
    "tsae_layer0": "#bcbd22",
    "tsae_layer1": "#17becf",
    "tsae_layer2": "#aec7e8",
    "tsae_layer3": "#ffbb78",
    "h8_early":    "#98df8a",
    "h8_mid":      "#c5b0d5",
    "h8_late":     "#f7b6d2",
    # Phase 8 v2 layer-0 deep dive: 5 hookpoints × 3 arch families. Colors
    # cluster by hookpoint (5 hues), brightness by arch family.
    "sae_l0_ln1":  "#9467bd",  # purple
    "sae_l0_pre":  "#1f77b4",  # blue
    "sae_l0_mid":  "#2ca02c",  # green
    "sae_l0_post": "#d62728",  # red
    "sae_l1_ln1":  "#ff7f0e",  # orange
    "tsae_l0_ln1": "#c5b0d5",  # light purple
    "tsae_l0_pre": "#aec7e8",  # light blue
    "tsae_l0_mid": "#98df8a",  # light green
    "tsae_l0_post":"#ff9896",  # light red
    "tsae_l1_ln1": "#ffbb78",  # light orange
    "txc_l0_ln1":  "#5e3d8a",  # dark purple
    "txc_l0_pre":  "#0d3a5c",  # dark blue
    "txc_l0_mid":  "#1f5e1f",  # dark green
    "txc_l0_post": "#7a1818",  # dark red
    "txc_l1_ln1":  "#a64500",  # dark orange
    # Isolated single-arch follow-up runs.
    "tsae_attn_out_l0": "#e377c2",  # T-SAE at attn_out_l0
}
ARCH_LABELS = {
    "mlc":         "MLC (L=5)",
    "txc_early":   "TXC layer 0 (T=30)",
    "txc_mid":     "TXC layer 2 (T=30)",
    "txc_late":    "TXC layer 3 (T=30)",
    "sae_layer0":  "SAE layer 0",
    "sae_layer1":  "SAE layer 1",
    "sae_layer2":  "SAE layer 2",
    "sae_layer3":  "SAE layer 3",
    "tsae_layer0": "T-SAE layer 0",
    "tsae_layer1": "T-SAE layer 1",
    "tsae_layer2": "T-SAE layer 2",
    "tsae_layer3": "T-SAE layer 3",
    "h8_early":    "H8 layer 0 (T=30)",
    "h8_mid":      "H8 layer 2 (T=30)",
    "h8_late":     "H8 layer 3 (T=30)",
    "sae_l0_ln1":  "SAE  ln1.0",
    "sae_l0_pre":  "SAE  resid_pre.0",
    "sae_l0_mid":  "SAE  resid_mid.0",
    "sae_l0_post": "SAE  resid_post.0",
    "sae_l1_ln1":  "SAE  ln1.1",
    "tsae_l0_ln1": "TSAE ln1.0",
    "tsae_l0_pre": "TSAE resid_pre.0",
    "tsae_l0_mid": "TSAE resid_mid.0",
    "tsae_l0_post":"TSAE resid_post.0",
    "tsae_l1_ln1": "TSAE ln1.1",
    "txc_l0_ln1":  "TXC  ln1.0 (T=30)",
    "txc_l0_pre":  "TXC  resid_pre.0 (T=30)",
    "txc_l0_mid":  "TXC  resid_mid.0 (T=30)",
    "txc_l0_post": "TXC  resid_post.0 (T=30)",
    "txc_l1_ln1":  "TXC  ln1.1 (T=30)",
    "tsae_attn_out_l0": "TSAE attn_out.0",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=str(ROOT / "outputs" / "data"))
    parser.add_argument("--output_dir", default=str(ROOT / "outputs" / "plots"))
    parser.add_argument("--results_md_dir", default=str(ROOT),
                        help="Directory to write RESULTS.md into.")
    parser.add_argument("--delta_util", type=float, default=0.05)
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    plotted = []
    for arch in ARCH_COLORS:
        path = in_dir / f"val_sweep_{arch}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        stage2 = data.get("stage2", [])
        if stage2:
            xs = [r["delta_clean_ce"] for r in stage2]
            ys = [r["val_asr_16"] for r in stage2]
            ax.scatter(xs, ys, s=30, alpha=0.45, color=ARCH_COLORS[arch], label=ARCH_LABELS[arch])
        ch = data["chosen"]
        ax.scatter(
            ch["delta_clean_ce"], ch["val_asr_16"],
            s=260, marker="*", edgecolor="black", linewidths=1.0,
            color=ARCH_COLORS[arch], zorder=5,
        )
        plotted.append(arch)

    ax.axvline(args.delta_util, linestyle="--", color="grey", alpha=0.6, label=f"δ = {args.delta_util}")
    ax.set_xlabel("Δ clean-continuation CE (nats)  —  lower = less damage")
    ax.set_ylabel("val sampled ASR_16  —  lower = more suppression")
    ax.set_title("TinyStories sleeper: best single-feature ablation per architecture")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)

    footer = (
        f"data: val_sweep_{{arch}}.json (stage-2 rows) + test_results.json  •  "
        f"params: δ={args.delta_util}, top100 by Δlogp → top10 by val ASR over α∈{{0.25, 0.5, 1, 1.5, 2}}  •  "
        f"func: plot_pareto.py"
    )
    fig.text(0.01, 0.005, footer, ha="left", va="bottom", fontsize=6, color="grey")
    fig.tight_layout(rect=(0, 0.02, 1, 1))

    out_path = out_dir / "pareto_asr_vs_utility.png"
    fig.savefig(out_path, dpi=150)
    print(f"[plot] wrote {out_path}  (archs: {plotted})")

    # Render markdown table into README if test_results.json exists.
    results_path = in_dir / "test_results.json"
    if results_path.exists():
        res = json.loads(results_path.read_text())
        rows = []
        for arch in ARCH_COLORS:
            ar = res["by_arch"].get(arch)
            if ar is None:
                continue
            rows.append({
                "arch": ARCH_LABELS[arch],
                **ar,
            })
        if rows:
            table = ["| Architecture | f | α* | val ASR_16 | test ASR_16 | baseline ASR_16 | Δ test log p | Δ test CE |",
                     "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for r in rows:
                table.append(
                    f"| {r['arch']} | {r['feature_idx']} | {r['alpha']} | "
                    f"{r.get('val_asr_16', float('nan')):.2f} | "
                    f"{r['test_asr_16']:.2f} | {r['baseline_test_asr_16']:.2f} | "
                    f"{r['test_dep_logp_delta']:+.2f} | {r['test_clean_ce_delta']:+.3f} |"
                )
            results_md_path = Path(args.results_md_dir) / "RESULTS.md"
            results_md_path.parent.mkdir(parents=True, exist_ok=True)
            results_md_path.write_text("\n".join(table) + "\n")
            print(f"[plot] wrote {results_md_path}")


if __name__ == "__main__":
    main()
