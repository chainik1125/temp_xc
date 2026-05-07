"""Audit C2 setups for completeness: 5 baselines + 4 plots + T-sweep.

Per Han 2026-05-07: 'when you finish everything check (i) they used all
baselines (ii) they have all plots (iii) they have the proper T-sweep.'

The 5 required baselines: TopK-SAE, Stacked T=2, Stacked T=5, T-SAE
(tsae_paper), TFA-pos.
The 4 required plots: gauc_vs_k, eauc_vs_k, scatter, tsweep.
The T-sweep should include txc_base T ∈ {2, 4, 5, 6, 8, 10, 12}.

Run via: .venv/bin/python -m experiments.c2_synthetic_coupled.audit_setups
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

LB_PATH = Path("results/leaderboard.jsonl")

# Setup → list of canonical datasources
SETUPS = {
    "A":         ["toy_coupled_K10_M20_d256"],
    "D-np5":     ["toy_coupled_noisy_K10_M20_d256_pB05_np5"],
    "D-np10":    ["toy_coupled_noisy_K10_M20_d256_pB05_np10"],
    "E":         ["toy_hierarchical_Kg10_Kl30_d256"],
    "F-σ0.5":    ["toy_coupled_obs_noise_K10_M20_d256_sigma0p5"],
    "F-σ1":      ["toy_coupled_obs_noise_K10_M20_d256_sigma1p0"],
    "F-σ2":      ["toy_coupled_obs_noise_K10_M20_d256_sigma2p0"],
    "G-σ1":      ["toy_hierarchical_Kg10_Kl30_d256_sigma1p0"],
    "G-σ2":      ["toy_hierarchical_Kg10_Kl30_d256_sigma2p0"],
    "J":         ["toy_hierarchical_Kg10_Kl50_d256"],
    "M":         ["toy_hetero_rho_Kg10_Kl30_d256_5slow_5fast"],
    "whisper":   ["toy_hierarchical_Kg10_Kl30_d256_sparse"],
    "polaris":   ["toy_hetero_rho_Kg10_Kl30_d256_5ultraslow_5slow"],
    "lighthouse":["toy_hetero_rho_Kg10_Kl30_d256_1slow_9fast"],
    "dewdrop":   ["toy_dewdrop_Kg10_Kl30_d256_p16"],
    "chord":     ["toy_chord_Kg10_Kl30_d256_2groups"],
    "aurora":    ["toy_aurora_K10_M20_d256_sigma1_alpha09"],
}

# Setup → list of plot file paths (relative to purified/)
def expected_plots(setup_short: str) -> list[Path]:
    """Return the 4 mandatory plot paths for the given setup."""
    setup_short = setup_short.lower()
    # Hierarchical-flavoured setups go in c2_hierarchical/plots/
    if setup_short in {"e", "g-σ1", "g-σ2", "j", "m", "whisper",
                       "polaris", "lighthouse", "dewdrop", "chord"}:
        plot_dir = "experiments/c2_hierarchical/plots"
    else:
        plot_dir = "experiments/c2_synthetic_coupled/plots"
    # Normalise the setup name for filenames
    if setup_short in {"d-np5", "d-np10"}:
        name = setup_short.replace("-", "_")
    elif setup_short in {"f-σ1", "g-σ1"}:
        name = setup_short.split("-")[0]  # canonical at σ=1.0
    elif setup_short in {"f-σ0.5", "f-σ2", "g-σ2"}:
        return []   # non-canonical σ; skip
    else:
        name = setup_short
    return [
        Path(f"{plot_dir}/c2_setup_{name}_gauc_vs_k.png"),
        Path(f"{plot_dir}/c2_setup_{name}_eauc_vs_k.png"),
        Path(f"{plot_dir}/c2_setup_{name}_scatter.png"),
        Path(f"{plot_dir}/c2_setup_{name}_tsweep.png"),
    ]


REQUIRED_BASELINES = ["topk_sae", "stacked_sae T=2", "stacked_sae T=5",
                      "tsae_paper", "tfa_pos"]
REQUIRED_TXC_T_VALUES = [2, 4, 5, 6, 8, 10, 12]


def main():
    # Pass 1: dedupe + bucket
    latest = {}
    for line in LB_PATH.open():
        line = line.strip()
        if not line: continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("component") != "c2": continue
        ec = d.get("eval_cfg") or {}
        if ec.get("smoke"): continue
        latest[d["eval_key"]] = d

    by_setup = defaultdict(list)
    ds_to_setup = {ds: s for s, dses in SETUPS.items() for ds in dses}
    for d in latest.values():
        s = ds_to_setup.get(d.get("datasource"))
        if s:
            by_setup[s].append(d)

    print("=" * 100)
    print("C2 SETUP AUDIT  (5 baselines × 4 plots × T-sweep)")
    print("=" * 100)
    print()

    for setup_short in SETUPS.keys():
        cells = by_setup.get(setup_short, [])
        # Detect arch_id presence
        archs_present = set()
        txc_T_present = set()
        for d in cells:
            ec = d.get("eval_cfg") or {}
            t = ec.get("t_label", "default")
            if d["arch"] == "stacked_sae":
                if t == "T=2": archs_present.add("stacked_sae T=2")
                elif t in ("T=5", "default"): archs_present.add("stacked_sae T=5")
            elif d["arch"] == "txc_base":
                # collect all T values
                if t == "default": txc_T_present.add(5)
                elif t.startswith("T="):
                    try: txc_T_present.add(int(t.split("=")[1]))
                    except: pass
            elif d["arch"] in ("topk_sae", "tsae_paper", "tfa_pos"):
                archs_present.add(d["arch"])

        # Plots
        plots = expected_plots(setup_short)
        plots_status = [(p, p.exists()) for p in plots]

        # Print row
        print(f"\n--- Setup {setup_short} ({SETUPS[setup_short][0]}) "
              f"--- ({len(cells)} cells)")
        # Baselines
        for r in REQUIRED_BASELINES:
            mark = "✅" if r in archs_present else "❌"
            print(f"  baseline  {mark} {r}")
        # T-sweep
        missing_T = [T for T in REQUIRED_TXC_T_VALUES if T not in txc_T_present]
        if not missing_T:
            print(f"  T-sweep   ✅ all 7 T-values present")
        else:
            print(f"  T-sweep   ❌ missing T = {missing_T}")
        # Plots
        if not plots:
            print(f"  plots     -- (non-canonical σ; supplementary only)")
        else:
            for p, ok in plots_status:
                mark = "✅" if ok else "❌"
                print(f"  plot      {mark} {p.name}")


if __name__ == "__main__":
    main()
