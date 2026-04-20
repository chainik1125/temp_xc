"""Plot training loss curves from the per-run JSON sidecars.

Reads every `results/training_logs/*.json` and writes:

    results/plots/training_curves.png       linear y
    results/plots/training_curves_loglog.png log-log

One line per architecture; color by family.

Usage:
    PYTHONPATH=/workspace/temp_xc \
      .venv/bin/python experiments/phase5_downstream_utility/plots/plot_training_curves.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.plotting.save_figure import save_figure


import os
REPO = Path(os.environ.get("PHASE5_REPO", Path(__file__).resolve().parents[3]))
LOGS_DIR = REPO / "experiments/phase5_downstream_utility/results/training_logs"
PLOTS_DIR = REPO / "experiments/phase5_downstream_utility/results/plots"


COLORS = {
    "topk_sae": "#1f77b4",       # blue
    "mlc": "#2ca02c",            # green
    "txcdr_t5": "#d62728",       # red
    "txcdr_t20": "#9467bd",      # purple
    "stacked_t5": "#ff7f0e",     # orange
    "stacked_t20": "#e377c2",    # pink
    "matryoshka_t5": "#17becf",  # cyan
    "shared_perpos_t5": "#7f7f7f", # gray
    "tfa": "#8c564b",            # brown
    "tfa_pos": "#bcbd22",        # olive
}


def _load_all_logs():
    out = {}
    for p in sorted(LOGS_DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text())
            out[data.get("run_id", p.stem)] = data
        except Exception as e:
            print(f"  SKIP {p.name}: {e}")
    return out


def plot_curves(logs: dict, log_x: bool = False, title_suffix: str = ""):
    fig, ax = plt.subplots(figsize=(10, 6))
    for run_id, data in sorted(logs.items()):
        arch = data.get("arch", "?")
        loss = data.get("loss", [])
        steps = data.get("steps_logged", list(range(len(loss))))
        if not loss:
            continue
        conv = data.get("converged", False)
        label = f"{run_id}"
        if conv:
            label += " (conv)"
        else:
            label += " (cap)"
        ax.plot(
            steps, loss,
            color=COLORS.get(arch, None), label=label, lw=1.2,
        )
    ax.set_xlabel("step")
    ax.set_ylabel("training loss")
    ax.set_title(f"Phase 5.1 training curves{title_suffix}")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    if log_x:
        ax.set_xscale("log")
        ax.set_yscale("log")
    fig.tight_layout()
    return fig


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    logs = _load_all_logs()
    if not logs:
        print("No training logs found.")
        return

    print(f"Loaded {len(logs)} training logs.")

    for run_id, data in sorted(logs.items()):
        arch = data.get("arch", "?")
        loss = data.get("loss", [])
        if loss:
            print(
                f"  {run_id:30s} {arch:20s} "
                f"final_loss={loss[-1]:.4f} steps={data.get('final_step')}"
                f" conv={data.get('converged')}"
            )

    fig = plot_curves(logs, log_x=False, title_suffix=" (linear)")
    save_figure(fig, str(PLOTS_DIR / "training_curves.png"))
    plt.close(fig)

    fig = plot_curves(logs, log_x=True, title_suffix=" (log-log)")
    save_figure(fig, str(PLOTS_DIR / "training_curves_loglog.png"))
    plt.close(fig)

    print(f"Wrote plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
