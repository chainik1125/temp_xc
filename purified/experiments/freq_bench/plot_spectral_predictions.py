"""Predicted vs measured NTPS for the (T, S, B) spectral ablation sweep."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LEADERBOARD = ROOT / "results" / "leaderboard.jsonl"
OUT = ROOT / "results" / "freq_bench" / "v2_sweep"
PRED_PATH = OUT / "spectral_predictions.json"
PROTO = "1.2.0"


def load_measured():
    by_label = {}
    for line in open(LEADERBOARD):
        r = json.loads(line)
        if (r.get("experiment") == "freq_bench"
                and r.get("evaluator_protocol_version") == PROTO):
            ec = r.get("eval_cfg", {})
            label = ec.get("label", "")
            if label.startswith("band_"):
                by_label[label] = r["metrics"]
    return by_label


def main():
    preds = json.load(open(PRED_PATH))
    measured = load_measured()
    rows = []
    for c in preds["cells"]:
        m = measured.get(c["label"])
        if m is None:
            continue
        rows.append({"label": c["label"], "T": c["T"], "S": c["S"],
                     "bands": c["bands"], "pred": c["predicted_NTPS"],
                     "ntps": m["NTPS"], "gap": m["order_gap"],
                     "rev_drop": m["reverse_drop"], "ff": m.get("freqfrac")})

    # ── (a) predicted vs measured scatter
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    xs = np.array([r["pred"] for r in rows])
    ys = np.array([r["ntps"] for r in rows])
    colors = []
    for r in rows:
        if "DC" in r["label"]:                colors.append("#888888")
        elif "BAC" in r["label"]:             colors.append("#d62728")
        elif "_B1" in r["label"] or "_B2" in r["label"]: colors.append("#9467bd")
        elif "BDC1" in r["label"]:            colors.append("#ff7f0e")
        elif r["S"] == 1 and r["bands"] == "all": colors.append("#1f77b4")
        elif r["S"] != 1:                     colors.append("#2ca02c")
        elif r["T"] == 16:                    colors.append("#17becf")
        else:                                  colors.append("#000000")
    ax.scatter(xs, ys, c=colors, s=60, edgecolor="k", zorder=3)
    for r in rows:
        ax.annotate(r["label"].replace("band_", ""), (r["pred"], r["ntps"]),
                    fontsize=7, xytext=(4, 2), textcoords="offset points")
    lim = [-0.05, 0.95]
    ax.plot(lim, lim, "k--", alpha=.5, lw=1, label="predicted = measured")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("predicted NTPS (pre-registered)")
    ax.set_ylabel("measured NTPS")
    ax.set_title("Predicted vs measured\n(11 cells, pre-registered)")
    ax.grid(alpha=.3); ax.legend(fontsize=9, loc="upper left")

    # ── (b) bands and strides
    ax = axes[1]
    by_kind = {"band": [], "stride": [], "repro": []}
    for r in rows:
        if r["S"] != 1 and r["bands"] == "all":
            by_kind["stride"].append(r)
        elif r["bands"] != "all":
            by_kind["band"].append(r)
        else:
            by_kind["repro"].append(r)

    xs = np.arange(len(rows))
    labels = []
    bars = []
    for kind, color in [("repro", "#1f77b4"),
                        ("band", "#9467bd"),
                        ("stride", "#2ca02c")]:
        for r in sorted(by_kind[kind], key=lambda r: r["ntps"]):
            bars.append((color, r))

    for i, (color, r) in enumerate(bars):
        ax.bar(i, r["ntps"], 0.7, color=color, edgecolor="k")
        ax.text(i, r["ntps"] + 0.015, f"{r['ntps']:.2f}",
                ha="center", fontsize=8)
        labels.append(r["label"].replace("band_", ""))
    ax.set_xticks(np.arange(len(bars)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("measured NTPS")
    ax.set_title("All cells, grouped: reproductions / band / stride ablations")
    ax.axhline(0, color="k", lw=.8)
    ax.grid(alpha=.3, axis="y")
    handles = [plt.Rectangle((0, 0), 1, 1, fc="#1f77b4"),
               plt.Rectangle((0, 0), 1, 1, fc="#9467bd"),
               plt.Rectangle((0, 0), 1, 1, fc="#2ca02c")]
    ax.legend(handles, ["reproductions", "band ablations", "stride ablations"],
              fontsize=8, loc="upper left")

    plt.suptitle("Spectral (T, S, B) ablation sweep — Dmitry's spectral request",
                 fontsize=12)
    plt.tight_layout()
    p = OUT / "spectral_predicted_vs_measured.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("saved", p)

    json.dump({"rows": rows}, open(OUT / "spectral_results.json", "w"),
              indent=2, default=float)
    print("saved", OUT / "spectral_results.json")

    print("\n=== predicted vs measured ===")
    for r in rows:
        err = r["ntps"] - r["pred"]
        marker = "✓" if abs(err) < 0.15 else "✗"
        print(f"  {marker} {r['label']:25s} pred={r['pred']:+.2f} "
              f"meas={r['ntps']:+.3f} err={err:+.3f} ff={r['ff']:.3f}")


if __name__ == "__main__":
    main()
