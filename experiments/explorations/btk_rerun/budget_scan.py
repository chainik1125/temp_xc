"""Budget-scan analysis — can tuned no-ReLU arms beat the incumbents?

Scans rows produced with explicit ``k_win`` overrides, d_sae wings, or
non-canonical batch sizes (excluded from the headline analysis), and
scores every (arm, budget-config) against two incumbents at matched T on
the coupled bench:

- ``composite@paper``: txc_base at the paper budget (k_win = k_pos*T),
  best over k_pos ∈ {1,2,5} per T;
- ``perwinraw@paper``: same for the ReLU-deleted twin.

Outputs ``budget_scan.md`` (ranked table per T + overall verdict) and
``budget_scan.json`` under --out-dir.

Usage: python -m experiments.explorations.btk_rerun.budget_scan
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BENCH = "toy_coupled_K10_M20_d256"
EVAL_L = 40
INCUMBENT_ARCHS = {"txc_base": "composite@paper",
                   "txc_base_perwinraw": "perwinraw@paper"}
SCAN_ARCHS = {"txc_base_btkonly": "btkonly",
              "txc_base_perwinraw": "perwinraw"}


def _ov(r):
    return (r.get("training_cfg") or {}).get("arch_hparams_override") or {}


def load(leaderboard: Path):
    incumbents = defaultdict(list)   # (T, name, d_sae, k_pos) -> gauc
    scan = defaultdict(list)                              # config key -> rows
    for line in leaderboard.open():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (r.get("experiment") != "synthetic"
                or r.get("datasource") != BENCH):
            continue
        ec = r.get("eval_cfg") or {}
        if ec.get("smoke") or ec.get("eval_window_L") != EVAL_L:
            continue
        tc = r.get("training_cfg") or {}
        ov = _ov(r)
        T = int(ov.get("T", 5))
        g = (r.get("metrics") or {}).get("gauc")
        if g is None:
            continue
        is_scan = ("k_win" in ov or tc.get("batch_size", 1024) != 1024
                   or ov.get("d_sae") not in (None, 20))
        paper_rule = ("k_win" not in ov
                      and tc.get("batch_size", 1024) == 1024
                      and tc.get("n_steps") == 6000)
        if paper_rule and r["arch"] in INCUMBENT_ARCHS:
            dsae_i = ov.get("d_sae", 20)
            incumbents[(T, INCUMBENT_ARCHS[r["arch"]], dsae_i,
                        ov.get("k_pos"))].append(float(g))
        if not is_scan:
            continue
        if is_scan and r["arch"] in SCAN_ARCHS:
            cfg = (SCAN_ARCHS[r["arch"]],
                   ov.get("k_win", f"k{ov.get('k_pos')}*T"),
                   ov.get("d_sae", 20),
                   tc.get("batch_size", 1024))
            scan[(cfg, T)].append(float(g))
    return incumbents, scan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--leaderboard", type=Path,
                   default=Path("results/leaderboard.jsonl"))
    p.add_argument("--out-dir", type=Path, default=Path("plots/btk_rerun"))
    args = p.parse_args()
    inc, scan = load(args.leaderboard)

    # Incumbent best-per-(T, d_sae): per-(k_pos) cell means from the
    # leaderboard, max over k_pos at each (T, name, d_sae).
    inc_best = {}
    for (T, name, dsae_i, _k), vals in inc.items():
        m = float(np.mean(vals))
        cur = inc_best.get((T, name, dsae_i))
        if cur is None or m > cur:
            inc_best[(T, name, dsae_i)] = m

    lines = ["# Budget scan — coupled gauc vs incumbents\n"]
    results = {}
    Ts = sorted({t for (_, t) in scan})
    for T in Ts:
        lines.append(f"\n## T = {T}\n")
        comp = inc_best.get((T, "composite@paper", 20))
        pwin = inc_best.get((T, "perwinraw@paper", 20))
        lines.append(f"incumbents (paper d20): composite = "
                     f"{comp:.3f} | perwinraw = {pwin:.3f}\n"
                     if comp and pwin else "(d20 incumbents missing)\n")
        lines.append("| arm | k_win | d_sae | batch | gauc (mean±std, n) | vs comp d20 | vs pwin d20 | vs comp same-d |")
        lines.append("|---|---|---|---|---|---|---|---|")
        rows_T = sorted(((cfg, vals) for (cfg, t), vals in scan.items()
                         if t == T),
                        key=lambda kv: -np.mean(kv[1]))
        for (arm, kw, ds, b), vals in rows_T:
            m, s, n = float(np.mean(vals)), float(np.std(vals)), len(vals)
            dc = m - comp if comp else float("nan")
            dp = m - pwin if pwin else float("nan")
            comp_d = inc_best.get((T, "composite@paper", ds))
            dcd = m - comp_d if comp_d is not None else float("nan")
            flag = (" **BEATS ALL**" if dc > 0 and dp > 0
                    and (comp_d is None or dcd > 0)
                    else (" beats-d20" if dc > 0 and dp > 0 else ""))
            cd_txt = f"{dcd:+.3f}" if comp_d is not None else "—"
            lines.append(f"| {arm} | {kw} | {ds} | {b} | "
                         f"{m:.3f}±{s:.3f} n={n} | {dc:+.3f} | {dp:+.3f} | "
                         f"{cd_txt}{flag} |")
            results[f"T{T}|{arm}|kw{kw}|d{ds}|b{b}"] = {
                "gauc": m, "std": s, "n": n, "vs_composite_d20": dc,
                "vs_perwinraw_d20": dp, "vs_composite_same_d": dcd}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "budget_scan.md").write_text("\n".join(lines) + "\n")
    (args.out_dir / "budget_scan.json").write_text(
        json.dumps({"incumbents": {f"T{t}|{n}": v for (t, n), v
                                   in inc_best.items()},
                    "scan": results}, indent=1))
    print(f"[budget_scan] {len(scan)} scan configs -> "
          f"{args.out_dir}/budget_scan.md")
    import math
    beats = [k for k, v in results.items()
             if v["vs_composite_d20"] > 0 and v["vs_perwinraw_d20"] > 0
             and (math.isnan(v["vs_composite_same_d"])
                  or v["vs_composite_same_d"] > 0)]
    print("configs beating d20 incumbents AND same-d composite:",
          beats if beats else "none")


if __name__ == "__main__":
    main()
