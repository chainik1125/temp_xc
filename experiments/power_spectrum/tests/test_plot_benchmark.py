from __future__ import annotations

import pandas as pd

from experiments.power_spectrum.code import plot_benchmark


def test_benchmark_plots_accept_complete_summary(tmp_path) -> None:
    rows = []
    for task_index, task in enumerate(plot_benchmark.TASKS):
        for model_index, model in enumerate(plot_benchmark.MODELS):
            rows.append(
                {
                    "task": task,
                    "model": model,
                    "n": 3,
                    "mean": 0.1 * task_index + 0.01 * model_index,
                    "std": 0.01,
                    "delta_vs_txc_pre": 0.01 * model_index,
                }
            )
    frame = plot_benchmark._ordered(pd.DataFrame(rows))
    primary = tmp_path / "primary.png"
    deltas = tmp_path / "deltas.png"
    plot_benchmark.plot_primary(frame, primary)
    plot_benchmark.plot_deltas(frame, deltas)
    assert primary.stat().st_size > 0
    assert deltas.stat().st_size > 0
