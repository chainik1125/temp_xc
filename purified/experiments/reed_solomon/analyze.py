"""Analyse the Reed-Solomon degree-ladder + (T,S,B) sweep → plots.

Reads results/leaderboard.jsonl, keeps the latest row per
(degree, label, kind), and renders into results/reed_solomon/:

  ntps_by_degree.png      class-NTPS vs degree, one line per ladder arch
  ntps_sign_by_degree.png sign-NTPS vs degree, one line per ladder arch
  nmse_by_degree.png      full-message regression NMSE (msg + leading) vs degree
  tsb_where_txc_lands.png (T,S,B) cells per degree; the standard TXC point
                          (T5_S1_Ball) marked — "where does TXC land?" (Q3)

Run:  .venv/bin/python experiments/reed_solomon/analyze.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]   # purified/
LB = ROOT / "results" / "leaderboard.jsonl"
OUT = ROOT / "results" / "reed_solomon"
OUT.mkdir(parents=True, exist_ok=True)

LADDER = ["regular_sae", "txcdr_t5", "txc_base_TW", "tfa"]
DEGREES = [1, 2, 3]
STD_TXC = "tsb_T5_S1_Ball"   # the standard TXC point in the (T,S,B) family


def load() -> dict:
    """Return {(degree, label, kind): metrics} keeping the LATEST row."""
    by = {}
    for line in open(LB):
        r = json.loads(line)
        if r.get("experiment") != "freq_bench":
            continue
        ec = r.get("eval_cfg") or {}
        if "degree" not in ec:
            continue
        key = (int(ec["degree"]), ec.get("label"), ec.get("kind"))
        by[key] = r["metrics"]           # later rows overwrite earlier
    return by


def _line_plot(by, metric, title, ylabel, fname, hline=None):
    plt.figure(figsize=(7, 4.5))
    for arch in LADDER:
        ys = [by.get((D, arch, "ladder"), {}).get(metric, np.nan) for D in DEGREES]
        plt.plot(DEGREES, ys, marker="o", label=arch)
    if hline is not None:
        plt.axhline(hline[0], ls="--", c="gray", lw=1, label=hline[1])
    plt.xticks(DEGREES)
    plt.xlabel("polynomial degree D")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / fname, dpi=130)
    plt.close()
    print(f"wrote {OUT/fname}")


def tsb_plot(by):
    tsb_labels = sorted({k[1] for k in by if k[2] == "tsb"})
    plt.figure(figsize=(11, 5))
    x = np.arange(len(tsb_labels))
    width = 0.25
    for i, D in enumerate(DEGREES):
        ys = [by.get((D, lab, "tsb"), {}).get("NTPS", np.nan) for lab in tsb_labels]
        bars = plt.bar(x + (i - 1) * width, ys, width, label=f"D={D}")
    # mark the standard TXC point
    if STD_TXC in tsb_labels:
        xi = tsb_labels.index(STD_TXC)
        plt.axvspan(xi - 0.5, xi + 0.5, color="gold", alpha=0.25,
                    label="standard TXC (T5,S1,Ball)")
    plt.xticks(x, tsb_labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("class NTPS")
    plt.title("(T,S,B) family on the RS ladder — where does standard TXC land?")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(OUT / "tsb_where_txc_lands.png", dpi=130)
    plt.close()
    print(f"wrote {OUT/'tsb_where_txc_lands.png'}")


def capacity_plot(by):
    """D2/D3: NTPS at d_sae=1024 (ladder) vs 4096 (capacity probe)."""
    caps = sorted({k for k in by if k[2] == "capacity"})
    if not caps:
        print("no capacity cells found; skipping capacity panel")
        return
    cap_degs = sorted({k[0] for k in caps})
    archs = sorted({k[1].removesuffix("_cap") for k in caps})
    plt.figure(figsize=(8, 4.5))
    x = np.arange(len(archs))
    width = 0.18
    pos = 0
    for D in cap_degs:
        base = [by.get((D, a, "ladder"), {}).get("NTPS", np.nan) for a in archs]
        cap = [by.get((D, f"{a}_cap", "capacity"), {}).get("NTPS", np.nan) for a in archs]
        plt.bar(x + (pos - 1.5) * width, base, width, label=f"D={D} d_sae=1024")
        plt.bar(x + (pos - 0.5) * width, cap, width, label=f"D={D} d_sae=4096+15k", hatch="//")
        pos += 2
    plt.axhline(0, c="gray", lw=1)
    plt.xticks(x, archs, fontsize=9)
    plt.ylabel("class NTPS")
    plt.title("RS capacity probe — does 4× dict + 3× steps rescue D≥2?")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(OUT / "capacity_probe.png", dpi=130)
    plt.close()
    print(f"wrote {OUT/'capacity_probe.png'}")


def main():
    by = load()
    n = len({k for k in by})
    print(f"loaded {n} RS cells "
          f"(ladder={sum(1 for k in by if k[2]=='ladder')}, "
          f"tsb={sum(1 for k in by if k[2]=='tsb')})")

    _line_plot(by, "NTPS", "RS class target — NTPS vs degree",
               "class NTPS", "ntps_by_degree.png", hline=(0.0, "chance"))
    _line_plot(by, "NTPS_sign", "RS sign target — NTPS_sign vs degree",
               "sign NTPS", "ntps_sign_by_degree.png", hline=(0.0, "chance"))
    _line_plot(by, "nmse_msg", "RS full-message regression — NMSE vs degree "
               "(lower better)", "message NMSE", "nmse_by_degree.png",
               hline=(1.0, "predict-the-mean"))
    tsb_plot(by)
    capacity_plot(by)

    # machine-readable headline table
    METRICS = ("NTPS", "NTPS_sign", "nmse_msg", "nmse_lead", "freqfrac")
    tbl = {}
    for D in DEGREES:
        tbl[f"D{D}"] = {}
        for arch in LADDER:
            m = by.get((D, arch, "ladder"), {})
            tbl[f"D{D}"][arch] = {k: round(float(m.get(k, float("nan"))), 4)
                                  for k in METRICS}
    json.dump(tbl, open(OUT / "summary.json", "w"), indent=2)
    print(f"wrote {OUT/'summary.json'}")
    # print the ladder table
    print("\n=== RS arch ladder (class NTPS / sign NTPS / msg NMSE) ===")
    for D in DEGREES:
        print(f"D={D}: " + "  ".join(
            f"{a}={by.get((D,a,'ladder'),{}).get('NTPS',float('nan')):+.2f}/"
            f"{by.get((D,a,'ladder'),{}).get('NTPS_sign',float('nan')):+.2f}/"
            f"{by.get((D,a,'ladder'),{}).get('nmse_msg',float('nan')):.2f}" for a in LADDER))


if __name__ == "__main__":
    main()
