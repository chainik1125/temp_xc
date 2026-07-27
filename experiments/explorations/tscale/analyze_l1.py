"""tscale L1 verdict table — reads results/l1_rows.jsonl, prints curves +
CARD_SPLIT § 3 gate checks for each tag vs the matched-steps baseline twin.

Run: .venv/bin/python -m experiments.explorations.tscale.analyze_l1
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results" / "l1_rows.jsonl"
BASE_TAG = "baseline-4k"


def load() -> dict[str, dict[int, dict]]:
    by_tag: dict[str, dict[int, dict]] = defaultdict(dict)
    with RESULTS.open() as fh:
        for line in fh:
            r = json.loads(line)
            by_tag[r["tag"]][int(r["cfg"]["T_eff"])] = r
    return by_tag


def main() -> None:
    by_tag = load()
    base = by_tag.get(BASE_TAG, {})

    for tag, cells in by_tag.items():
        Ts = sorted(cells)
        print(f"\n== {tag} ==  (dev-8 s42; k20 primary)")
        for k in ("k20", "k5"):
            missing = [T for T in Ts if k not in cells[T]["eval"]]
            if missing:
                continue
            row = "  ".join(
                f"T{T}:{cells[T]['eval'][k]['dev_mean_auc']:.4f}" for T in Ts
            )
            print(f"  {k:>3}: {row}")
        for T in Ts:
            e = cells[T]["eval"]["k20"]
            ti = cells[T].get("train_info", {})
            gap = e["dev_mean_auc"] - e["dev_mean_auc_shuf"]
            print(f"    T{T:>2}: l0_eval {e['realized_l0']:7.1f}  order-gap {gap:+.4f}"
                  f"  active-frac {ti.get('frac_latents_active_batch', float('nan')):.4f}"
                  if isinstance(ti.get("frac_latents_active_batch"), float)
                  else f"    T{T:>2}: l0_eval {e['realized_l0']:7.1f}  order-gap {gap:+.4f}")
        if tag == BASE_TAG or not {1, 16} <= set(Ts) or not {1, 16} <= set(base):
            continue
        # Gate check vs baseline twin (k20)
        c1 = cells[1]["eval"]["k20"]["dev_mean_auc"]
        c16 = cells[16]["eval"]["k20"]["dev_mean_auc"]
        b1 = base[1]["eval"]["k20"]["dev_mean_auc"]
        b16 = base[16]["eval"]["k20"]["dev_mean_auc"]
        d_c, d_b = c16 - c1, b16 - b1
        slope_ok = (d_c >= d_b + 0.008) or (d_c >= -0.005)
        level_ok = (c16 >= b16) and (c1 >= b1 - 0.010)
        print(f"  Δ16 cand {d_c:+.4f} vs base {d_b:+.4f} → slope "
              f"{'PASS' if slope_ok else 'FAIL'}; level (T16 {c16:.4f} vs {b16:.4f}, "
              f"T1 {c1:.4f} vs floor {b1 - 0.010:.4f}) → "
              f"{'PASS' if level_ok else 'FAIL'}")
        print(f"  L1→L2 PROMOTE: {'YES' if (slope_ok and level_ok) else 'NO'}")


if __name__ == "__main__":
    main()
