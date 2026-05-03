"""Build the NeurIPS hygiene table + per-arch training curves.

Reads results/ward_backtracking_txc/logs/<cell>__train.jsonl for each
cell registered in the headline arch list. Compiles FVU, L0, step,
elapsed-wall-time per arch into a CSV. Renders FVU-vs-step + L0-vs-step
PNGs per arch.

Usage:
  python -m experiments.ward_backtracking_txc.build_hygiene_table \
      --out results/ward_backtracking_txc/hygiene
"""
from __future__ import annotations
import argparse
import csv
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ward_txc.hygiene")


# (label, cell_id) — same set as the headline plot, plus appendix-only TXC-H8.
ARCH_CELLS = [
    ("TXC",        "txc__resid_L10__k16__s42"),
    ("TXC-H8",     "txc_h8__resid_L10__k16__s42"),  # appendix-only
    ("SAE",        "topk_sae__ln1_L10__k64__s42"),
    ("TSAE-paper", "tsae__resid_L10__k32__s42"),
    ("TFA",        "tfa__resid_L10__k32__s42"),
    ("MLC",        "mlc__resid_L10__k32__s42"),
]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def render_curves(label: str, rows: list[dict], out_dir: Path):
    if not rows:
        return
    steps = np.array([r["step"] for r in rows])
    fvu_eval = np.array([r.get("fvu_eval", np.nan) for r in rows])
    l0 = np.array([r.get("window_l0", np.nan) for r in rows])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5), sharex=True)
    ax1.plot(steps, fvu_eval, "-", lw=1.5, color="#1f4e79")
    ax1.set_xlabel("step"); ax1.set_ylabel("held-out FVU")
    ax1.set_title(f"{label} — FVU vs step")
    ax1.grid(alpha=0.3)
    ax2.plot(steps, l0, "-", lw=1.5, color="#7f3f98")
    ax2.set_xlabel("step"); ax2.set_ylabel("window L0 (mean active features per window)")
    ax2.set_title(f"{label} — L0 vs step")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fname = label.lower().replace(" ", "_").replace("-", "_") + ".png"
    fig.savefig(out_dir / fname, dpi=150)
    log.info("[saved] %s", out_dir / fname)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--logs-dir", type=Path,
                   default=Path("results/ward_backtracking_txc/logs"))
    p.add_argument("--out", type=Path,
                   default=Path("results/ward_backtracking_txc/hygiene"))
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    curves_dir = args.out / "training_curves"
    curves_dir.mkdir(exist_ok=True)

    table_rows = []
    for label, cell in ARCH_CELLS:
        log_path = args.logs_dir / f"{cell}__train.jsonl"
        rows = load_jsonl(log_path)
        if not rows:
            log.warning("[skip] no log for %s at %s", label, log_path)
            continue
        first, last = rows[0], rows[-1]
        n_steps = int(last["step"]) - int(first["step"]) + 1
        # Compute mean L0 over last 20% of the run (steady-state)
        late_l0 = np.mean([r.get("window_l0", np.nan)
                           for r in rows[int(0.8*len(rows)):]])
        # Frac variance explained = 1 - FVU
        fve = 1.0 - last.get("fvu_eval", np.nan)
        # Steady-state n_dead near end of training
        late_dead = int(np.mean([r.get("n_dead", 0)
                                 for r in rows[int(0.8*len(rows)):]]))
        table_rows.append({
            "label": label,
            "cell_id": cell,
            "n_steps_logged": int(last["step"]) + 1,
            "final_fvu_eval": float(last.get("fvu_eval", np.nan)),
            "final_fvu_train": float(last.get("fvu", np.nan)),
            "final_window_L0": float(last.get("window_l0", np.nan)),
            "mean_L0_late": float(late_l0),
            "n_dead_late": late_dead,
            "fve_held_out": float(fve),
            "stopped_early": last.get("step", 0) < 14999,
        })
        render_curves(label, rows, curves_dir)
        log.info("[%-12s] step=%5d  fvu_eval=%.4f  L0=%6.1f  fve=%.3f  early=%s",
                 label, table_rows[-1]["n_steps_logged"],
                 table_rows[-1]["final_fvu_eval"],
                 table_rows[-1]["final_window_L0"],
                 table_rows[-1]["fve_held_out"],
                 table_rows[-1]["stopped_early"])

    csv_path = args.out / "reconstruction_table.csv"
    fieldnames = ["label", "cell_id", "n_steps_logged",
                  "final_fvu_eval", "final_fvu_train",
                  "final_window_L0", "mean_L0_late",
                  "n_dead_late", "fve_held_out", "stopped_early"]
    with csv_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in table_rows:
            w.writerow(r)
    log.info("[saved] %s", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
