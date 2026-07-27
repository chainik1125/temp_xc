"""ACTMIX RLHF writeup figure (CARD § 7 A1; directive 059a66239).

`figs_writeup/fig_rlhf_shuffle_tsweep.{png,pdf}` in the Aniket
template: x = T (log2), ordered solid + shuffled dashed, faint
per-seed lines, seed-mean ± sd, "T=16 − T=1: +X" annotation.
T = 1 shuffled point = the ordered value by construction (a
within-window shuffle of a length-1 window is the identity) —
annotated on the figure. Seed coverage per T is auto-disclosed in
the corner note (ragged during the interim render).

Usage:
  .venv/bin/python -m experiments.explorations.actmix_rlhf.render_writeup_fig \
      --tag interim   # or: --tag final
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.explorations.actmix_rlhf.cells import (  # noqa: E402
    DATASOURCE, TXC_ARCH)

ROOT = Path(__file__).resolve().parents[3]
LEADERBOARD = ROOT / "results" / "leaderboard.jsonl"
OUT_DIR = ROOT / "figs_writeup"
SEEDS = (42, 1, 2)  # explicit whitelist — stray seed-0 smoke rows excluded
TXC = "#D55E00"  # house Okabe-Ito; hue follows the entity, linestyle
                 # carries the order condition (CVD-safe split)


def load_points():
    """(T, seed) -> {ordered, shuffled} from the latest matching row."""
    best = {}
    for line in LEADERBOARD.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if (r.get("evaluator_name") != "rlhf"
                or r.get("arch") != TXC_ARCH
                or r.get("datasource") != DATASOURCE
                or int(r.get("seed", -1)) not in SEEDS):
            continue
        if not (r.get("training_cfg") or {}).get("n_steps"):
            continue  # untrained twins are not in this figure
        m = r["metrics"]
        T = int(m["T"])
        key = (T, int(r["seed"]))
        if key in best and best[key]["ts"] >= r["ts"]:
            continue
        ordered = m["preference_auc_k20"]
        shuffled = (ordered if T == 1  # identity by construction
                    else m.get("shuffled_preference_auc_k20"))
        best[key] = {"ts": r["ts"], "ordered": ordered, "shuffled": shuffled}
    return best


def series(points, seed, field):
    Ts = sorted(T for (T, s) in points if s == seed)
    return Ts, [points[(T, seed)][field] for T in Ts]


def mean_sd(points, field):
    Ts = sorted({T for (T, _) in points})
    mu, sd, n = [], [], []
    for T in Ts:
        vals = [v[field] for (t, _), v in points.items()
                if t == T and v[field] is not None]
        n.append(len(vals))
        mu.append(sum(vals) / len(vals))
        m = mu[-1]
        sd.append((sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5
                  if len(vals) > 1 else None)
    return Ts, mu, sd, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", choices=("interim", "final"), required=True)
    args = ap.parse_args()

    points = load_points()
    if not points:
        raise SystemExit("no matching leaderboard rows")

    fig, ax = plt.subplots(figsize=(5.4, 3.7))

    for seed in SEEDS:  # faint per-seed lines, both conditions
        for field, ls in (("ordered", "-"), ("shuffled", "--")):
            Ts, ys = series(points, seed, field)
            if Ts:
                ax.plot(Ts, ys, ls, color=TXC, alpha=0.25, lw=1, zorder=1)

    for field, ls, mk, mfc, label in (
            ("ordered", "-", "o", TXC, "ordered"),
            ("shuffled", "--", "s", "white", "within-window shuffled")):
        Ts, mu, sd, n = mean_sd(points, field)
        ax.plot(Ts, mu, ls, color=TXC, lw=2, marker=mk, ms=6,
                mfc=mfc, mec=TXC, label=label, zorder=3)
        for T, m, s in zip(Ts, mu, sd):
            if s is not None:
                ax.errorbar(T, m, yerr=s, color=TXC, capsize=3,
                            lw=1.2, zorder=2)

    Ts, mu, _, n = mean_sd(points, "ordered")
    if 16 in Ts and 1 in Ts:
        delta = mu[Ts.index(16)] - mu[Ts.index(1)]
        ax.annotate(f"T=16 − T=1: {delta:+.3f}", xy=(0.03, 0.95),
                    xycoords="axes fraction", ha="left", va="top",
                    fontsize=9)
    ax.annotate("T=1: shuffle ≡ identity (by construction)",
                xy=(0.03, 0.88), xycoords="axes fraction",
                ha="left", va="top", fontsize=8, color="#555555")

    cov = " ".join(f"T{T}:n={k}" for T, k in zip(Ts, n))
    tag_note = ("INTERIM — remaining seeds in flight" if args.tag == "interim"
                else "FINAL — seeds {42, 1, 2}")
    ax.annotate(f"{tag_note} · {cov}", xy=(0.99, 0.02),
                xycoords="axes fraction", ha="right", va="bottom",
                fontsize=6.5, color="#777777")

    ax.set_xscale("log", base=2)
    ax.set_xticks(Ts)
    ax.set_xticklabels([str(T) for T in Ts])
    ax.minorticks_off()
    ax.set_xlabel("T (window length)")
    ax.set_ylabel("preference AUC (k = 20)")
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=8, loc="lower right",
              bbox_to_anchor=(1.0, 0.08))
    fig.tight_layout()

    OUT_DIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig_rlhf_shuffle_tsweep.{ext}",
                    dpi=300 if ext == "png" else None)
    print(f"[render_writeup_fig] {args.tag}: {len(points)} points; "
          f"coverage {cov}")


if __name__ == "__main__":
    main()
