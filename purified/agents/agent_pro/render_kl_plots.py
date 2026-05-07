"""Render the 4 mandatory plots for Setup K + Setup L (agent_pro mission).

Mirrors the briefing's render snippet; idempotent — re-runs as more
shards complete.

Plots written:
  experiments/c2_synthetic_coupled/plots/c2_setup_k_*.png
  experiments/c2_synthetic_coupled/plots/c2_setup_l_*.png
"""
from __future__ import annotations

import os
os.environ.setdefault("TQDM_DISABLE", "1")

from experiments.c2_synthetic_coupled.plot_headline import (
    NOISY_PLOT_DIR,
    render_setup,
)

# Phase tags written by fill_baselines.py:
#   txc_base × T cells get hunt_phase="tsweep"
#   all other (arch, T) cells get hunt_phase="fill"
LINE_PHASES = ("zoom", "fill")
SCATTER_PHASES = ("zoom", "tsweep", "fill")
TSWEEP_PHASES = ("tsweep",)


def _render_one(setup_name: str, datasource: str, title_root: str):
    def line_filter(d, _ds=datasource, _phases=LINE_PHASES):
        return (
            d.get("datasource") == _ds
            and (d.get("eval_cfg") or {}).get("hunt_phase") in _phases
        )

    def scatter_filter(d, _ds=datasource, _phases=SCATTER_PHASES):
        return (
            d.get("datasource") == _ds
            and (d.get("eval_cfg") or {}).get("hunt_phase") in _phases
        )

    def tsweep_filter(d, _ds=datasource, _phases=TSWEEP_PHASES):
        return (
            d.get("datasource") == _ds
            and (d.get("eval_cfg") or {}).get("hunt_phase") in _phases
        )

    render_setup(
        setup_name=setup_name,
        plot_dir=NOISY_PLOT_DIR,
        line_filter_fn=line_filter,
        scatter_filter_fn=scatter_filter,
        tsweep_filter_fn=tsweep_filter,
        title_root=title_root,
        fixed_k_for_tsweep=1,
    )


def main():
    setups = [
        ("k", "toy_anticorrelated_Kg10_Kl30_d256",
         "Setup K (one-hot anti-correlated globals)"),
        ("l", "toy_magmod_Kg10_Kl30_d256_alpha1",
         "Setup L (magnitude-modulated locals)"),
        # PHALANX + OBELISK render targets (idempotent — emit empty if
        # leaderboard rows missing).
        ("phalanx", "toy_phalanx_Kg10_Kl30_d256_period8",
         "Setup PHALANX (period-locked global pulses)"),
        ("obelisk", "toy_obelisk_Kg10_Kl30_d256_alpha5",
         "Setup OBELISK (rare amplified magnitude-mod)"),
    ]
    for name, ds, title in setups:
        print(f"\n=== rendering Setup {name.upper()} ({ds}) ===")
        _render_one(name, ds, title)


if __name__ == "__main__":
    main()
