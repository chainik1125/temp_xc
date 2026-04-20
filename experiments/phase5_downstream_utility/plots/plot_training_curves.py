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


REPO = Path("/workspace/temp_xc")
LOGS_DIR = REPO / "experiments/phase5_downstream_utility/results/training_logs"
PLOTS_DIR = REPO / "experiments/phase5_downstream_utility/results/plots"


COLORS = {
    "topk_sae": "#1f77b4",
    "mlc": "#2ca02c",
    "txcdr_t5": "#d62728",
    "txcdr_t20": "#9467bd",
    "stacked_t5": "#ff7f0e",
    "stacked_t20": "#e377c2",
    "matryoshka_t5": "#17becf",
    "txcdr_shared_dec_t5": "#393b79",
    "txcdr_shared_enc_t5": "#637939",
    "txcdr_tied_t5": "#8c6d31",
    "txcdr_pos_t5": "#843c39",
    "txcdr_causal_t5": "#7b4173",
    "txcdr_block_sparse_t5": "#3182bd",
    "txcdr_lowrank_dec_t5": "#31a354",
    "txcdr_rank_k_dec_t5": "#756bb1",
    "temporal_contrastive": "#de9ed6",
    "time_layer_crosscoder_t5": "#ce6dbd",
    "tfa_small": "#8c564b",
    "tfa_pos_small": "#bcbd22",
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
