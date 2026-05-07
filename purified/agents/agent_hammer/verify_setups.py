"""Verify every C2 setup has (i) 5 baselines, (ii) 4 plots, (iii) T-sweep.

Per Han's autonomous-work directive 2026-05-07:
  > ensure all setups have all baselines TopK Stacked T=2 Stacked T=5
  > TFA-pos TSAE
  > CHECK (i) they used all baselines (ii) they have all plots
  > (iii) they have the proper T-sweep

Reports per-setup gaps. Run after baseline backfills + plot renders:

    .venv/bin/python -m agents.agent_hammer.verify_setups
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

LEADERBOARD = Path("results/leaderboard.jsonl")
PLOT_DIRS = [
    Path("experiments/c2_synthetic_coupled/plots"),
    Path("experiments/c2_hierarchical/plots"),
]
EXPECTED_BASELINES = [
    ("topk_sae", "default"),
    ("tfa_pos", "default"),
    ("tsae_paper", "default"),
    ("stacked_sae", "T=2"),
    ("stacked_sae", "T=5"),
]
EXPECTED_T_SWEEP = ["T=2", "T=4", "T=5", "T=6", "T=8", "T=10", "T=12"]
EXPECTED_PLOT_SUFFIXES = [
    "_gauc_vs_k.png",
    "_eauc_vs_k.png",
    "_scatter.png",
    "_tsweep.png",
]


def datasource_to_setup(ds: str) -> str | None:
    if ds == "toy_coupled_K10_M20_d256":
        return "A"
    if ds.startswith("toy_coupled_K10_M20_d256_rho"):
        return "C"
    if ds == "toy_markov_n20_d40_noisy":
        return "B"
    if ds.startswith("toy_coupled_noisy_K10_M20_d256_pB") and "rho" in ds:
        return "H"
    if ds.startswith("toy_coupled_noisy_K10_M20_d256_pB"):
        return "D"
    if ds.startswith("toy_hierarchical_Kg10_Kl30_d256") and "sigma" in ds:
        return "G"
    if ds.startswith("toy_hierarchical_Kg10_Kl30_d256"):
        return "E"
    if ds.startswith("toy_hierarchical_Kg10_Kl50_d256"):
        return "J"
    if ds.startswith("toy_coupled_obs_noise_"):
        return "F"
    return None


def collect_cells():
    """Return: setup → datasource → set of (arch, t_label)."""
    have = defaultdict(lambda: defaultdict(set))
    for line in LEADERBOARD.open():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("eval_cfg", {}).get("smoke"):
            continue
        ds = r.get("datasource", "")
        s = datasource_to_setup(ds)
        if s is None:
            continue
        arch = r["arch"]
        t = r.get("eval_cfg", {}).get("t_label", "default")
        # Stacked default → T=5 alias
        if arch == "stacked_sae" and t == "default":
            t = "T=5"
        have[s][ds].add((arch, t))
    return have


def list_plots():
    """Return: list of plot filenames (basenames) across PLOT_DIRS."""
    files = []
    for d in PLOT_DIRS:
        if not d.exists():
            continue
        files.extend(p.name for p in d.glob("c2_setup_*.png") if not p.name.endswith(".thumb.png"))
    return sorted(set(files))


def main() -> None:
    have = collect_cells()
    plots = list_plots()

    print("=" * 70)
    print("(i) Baselines per (setup, datasource)")
    print("=" * 70)
    print(f"{'setup':6s} {'datasource':50s} {' '.join(b[0][:4] for b in EXPECTED_BASELINES):28s}")
    print("-" * 90)
    incomplete_setups = set()
    for s in sorted(have.keys()):
        for ds in sorted(have[s].keys()):
            arches = have[s][ds]
            cols = []
            for (a, t) in EXPECTED_BASELINES:
                ok = ((a, t) in arches) or (a == "stacked_sae" and t == "T=5" and ("stacked_sae", "default") in arches)
                cols.append("✓" if ok else "✗")
                if not ok:
                    incomplete_setups.add(s)
            print(f"{s:6s} {ds[:50]:50s} {'  '.join(cols)}")

    print()
    print("=" * 70)
    print("(ii) Plots per setup (4-plot standard)")
    print("=" * 70)
    plot_map = defaultdict(set)
    for p in plots:
        # parse "c2_setup_<X>_<suffix>.png"
        if not p.startswith("c2_setup_"):
            continue
        rest = p[len("c2_setup_"):]
        for suffix in ("_gauc_vs_k.png", "_eauc_vs_k.png", "_scatter.png", "_tsweep.png"):
            if rest.endswith(suffix):
                name = rest[: -len(suffix)]
                plot_map[name].add(suffix)
                break
    print(f"{'setup_name':40s} gauc_vs_k eauc_vs_k scatter tsweep")
    print("-" * 80)
    for name in sorted(plot_map.keys()):
        suffixes = plot_map[name]
        cols = ["✓" if s in suffixes else "✗" for s in EXPECTED_PLOT_SUFFIXES]
        print(f"{name:40s}     {cols[0]}        {cols[1]}      {cols[2]}      {cols[3]}")

    print()
    print("=" * 70)
    print("(iii) T-sweep cells per (setup, datasource)")
    print("=" * 70)
    print(f"{'setup':6s} {'datasource':50s} {'T=':28s}")
    print("-" * 90)
    for s in sorted(have.keys()):
        for ds in sorted(have[s].keys()):
            arches = have[s][ds]
            t_present = sorted({t for (a, t) in arches if a == "txc_base"})
            ts = ",".join(t.split("=")[1] if "=" in t else t for t in t_present)
            cols = ""
            for et in EXPECTED_T_SWEEP:
                cols += "✓" if (("txc_base", et) in arches or (et == "T=5" and ("txc_base", "default") in arches)) else "✗"
            print(f"{s:6s} {ds[:50]:50s} {cols}  ({ts})")

    print()
    print("=" * 70)
    print(f"INCOMPLETE setups (missing baseline): {sorted(incomplete_setups)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
