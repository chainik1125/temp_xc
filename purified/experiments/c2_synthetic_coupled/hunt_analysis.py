"""Phase 1 HUNT analysis — pick the (p_B, n_parents) cell where TXC wins biggest.

Reads ``results/leaderboard.jsonl``, filters for hunt-phase rows on the
toy_coupled_noisy_* datasources, computes mean gAUC over seeds for
each (datasource, k_pos, arch), and reports:

  - per-cell gap = gAUC[txc_base] - gAUC[topk_sae]
  - max gap per datasource (across k_pos)
  - winning (datasource, k_pos)

Output: prints a markdown table + writes ``hunt_summary.json`` with
the full per-cell breakdown.

Usage (from purified/):
    .venv/bin/python -m experiments.c2_synthetic_coupled.hunt_analysis
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

LEADERBOARD = Path("results/leaderboard.jsonl")
OUT_DIR = Path("experiments/c2_synthetic_coupled")
OUT_FILE = OUT_DIR / "hunt_summary.json"

HUNT_DATASOURCE_PREFIX = "toy_coupled_noisy_K10_M20_d256_"
ARCHS = ("topk_sae", "txc_base")


def main():
    # Dedupe by eval_key — duplicate rows can arise if two subprocesses
    # compute the same (arch, seed, k_pos) on different GPUs. Keep latest.
    by_eval_key: dict[str, dict] = {}
    with LEADERBOARD.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d["component"] != "c2":
                continue
            if d["arch"] not in ARCHS:
                continue
            ds = d.get("datasource", "")
            if not ds.startswith(HUNT_DATASOURCE_PREFIX):
                continue
            ec = d.get("eval_cfg", {}) or {}
            if ec.get("hunt_phase") != "hunt":
                continue
            if ec.get("smoke") is True:
                continue
            t_label = ec.get("t_label", "default")
            # txc_base only at default T=5; topk_sae has no T.
            if d["arch"] == "txc_base" and t_label != "default":
                continue
            if d["arch"] == "topk_sae" and t_label != "default":
                continue
            gauc = d["metrics"].get("gauc")
            eauc = d["metrics"].get("eauc")
            if gauc is None or eauc is None:
                continue
            by_eval_key[d["eval_key"]] = {
                "datasource": ds,
                "arch": d["arch"],
                "seed": d["seed"],
                "k_pos": int(ec.get("k_pos")),
                "p_B": float(ec.get("p_B", 1.0)),
                "n_parents": int(ec.get("n_parents", 0)),
                "rho": float(ec.get("rho", 0.0)),
                "gauc": float(gauc),
                "eauc": float(eauc),
            }
    rows = list(by_eval_key.values())

    if not rows:
        print("[hunt_analysis] No hunt rows found yet — has the sweep finished?")
        return

    # Group: (ds, k_pos, arch) -> list of gauc/eauc over seeds
    grouped: dict[tuple, dict] = defaultdict(lambda: {"gauc": [], "eauc": []})
    meta: dict[str, dict] = {}
    for r in rows:
        key = (r["datasource"], r["k_pos"], r["arch"])
        grouped[key]["gauc"].append(r["gauc"])
        grouped[key]["eauc"].append(r["eauc"])
        meta[r["datasource"]] = {"p_B": r["p_B"], "n_parents": r["n_parents"], "rho": r["rho"]}

    # Aggregate per (ds, k_pos): mean over seeds for each arch.
    per_cell: dict[tuple, dict] = {}
    for (ds, k_pos, arch), v in grouped.items():
        cell_key = (ds, k_pos)
        per_cell.setdefault(cell_key, {})
        per_cell[cell_key][arch] = {
            "gauc_mean": mean(v["gauc"]),
            "gauc_std": (max(v["gauc"]) - min(v["gauc"])) / 2 if len(v["gauc"]) > 1 else 0.0,
            "eauc_mean": mean(v["eauc"]),
            "n_seeds": len(v["gauc"]),
        }

    # Compute per-cell gap and per-datasource max gap.
    per_ds_max_gap: dict[str, dict] = {}
    cell_records = []
    for (ds, k_pos), v in sorted(per_cell.items()):
        sae = v.get("topk_sae")
        txc = v.get("txc_base")
        if sae is None or txc is None:
            continue
        gap_gauc = txc["gauc_mean"] - sae["gauc_mean"]
        gap_eauc = txc["eauc_mean"] - sae["eauc_mean"]
        cell_records.append({
            "datasource": ds,
            "k_pos": k_pos,
            "p_B": meta[ds]["p_B"],
            "n_parents": meta[ds]["n_parents"],
            "rho": meta[ds]["rho"],
            "topk_sae_gauc": sae["gauc_mean"],
            "txc_base_gauc": txc["gauc_mean"],
            "gauc_gap": gap_gauc,
            "topk_sae_eauc": sae["eauc_mean"],
            "txc_base_eauc": txc["eauc_mean"],
            "eauc_gap": gap_eauc,
            "n_seeds_sae": sae["n_seeds"],
            "n_seeds_txc": txc["n_seeds"],
        })
        if ds not in per_ds_max_gap or gap_gauc > per_ds_max_gap[ds]["gauc_gap"]:
            per_ds_max_gap[ds] = {
                "k_pos": k_pos,
                "gauc_gap": gap_gauc,
                "eauc_gap": gap_eauc,
                "txc_gauc": txc["gauc_mean"],
                "sae_gauc": sae["gauc_mean"],
                **meta[ds],
            }

    # Print per-cell table.
    print()
    print("# Phase 1 HUNT — per-cell gAUC gap (txc_base − topk_sae)")
    print()
    print("| datasource | p_B | n_par | k_pos | SAE gAUC | TXC gAUC | gap | SAE eAUC | TXC eAUC |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in cell_records:
        ds_short = r["datasource"].replace(HUNT_DATASOURCE_PREFIX, "")
        print(
            f"| {ds_short} | {r['p_B']:.1f} | {r['n_parents']:d} | {r['k_pos']:2d} | "
            f"{r['topk_sae_gauc']:.3f} | {r['txc_base_gauc']:.3f} | "
            f"**{r['gauc_gap']:+.3f}** | {r['topk_sae_eauc']:.3f} | "
            f"{r['txc_base_eauc']:.3f} |"
        )

    # Winner table.
    print()
    print("# Per-datasource winner (max gauc_gap across k_pos)")
    print()
    print("| datasource | p_B | n_par | best k_pos | SAE gAUC | TXC gAUC | gap |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    sorted_winners = sorted(per_ds_max_gap.items(), key=lambda kv: -kv[1]["gauc_gap"])
    for ds, w in sorted_winners:
        ds_short = ds.replace(HUNT_DATASOURCE_PREFIX, "")
        print(
            f"| {ds_short} | {w['p_B']:.1f} | {w['n_parents']:d} | "
            f"{w['k_pos']:2d} | {w['sae_gauc']:.3f} | {w['txc_gauc']:.3f} | "
            f"**{w['gauc_gap']:+.3f}** |"
        )

    if sorted_winners:
        winner_ds, winner_info = sorted_winners[0]
        print()
        print(f"**OVERALL WINNER**: {winner_ds}")
        print(
            f"  → p_B={winner_info['p_B']:.2f}  n_parents={winner_info['n_parents']}  "
            f"k_pos={winner_info['k_pos']}  gauc_gap={winner_info['gauc_gap']:+.3f}  "
            f"(TXC {winner_info['txc_gauc']:.3f} vs SAE {winner_info['sae_gauc']:.3f})"
        )

    out = {
        "n_rows": len(rows),
        "n_cells": len(cell_records),
        "cell_records": cell_records,
        "per_datasource_winner": {ds: w for ds, w in per_ds_max_gap.items()},
        "overall_winner_datasource": sorted_winners[0][0] if sorted_winners else None,
        "overall_winner_info": sorted_winners[0][1] if sorted_winners else None,
    }
    OUT_FILE.write_text(json.dumps(out, indent=2))
    print()
    print(f"Wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
