"""Read `results/frontier.json` and report the item-6 verdict.

Answers the hub's pre-registered question (`aa0272633`, amended
`567d6818e`) and NOT a looser one: *does a temporal crosscoder buy
anything over pooling per-token SAE features across the same window at
comparable sparsity?* — not "does TXC beat a per-token probe", which is
the comparison that establishes nothing and is what the original claim
was fairly challenged for.

Verdict rule, applied mechanically so it cannot drift after seeing the
numbers: for each T, compare TXC's (budget, recovery) point against the
pooled and stacked FRONTIERS — the best recovery each SAE arm achieves
at budget <= TXC's. TXC "wins" at that T only if it beats BOTH at
no greater budget. Pre-registered: if TXC does not sit above, item 6 is
a NEGATIVE and is reported as one.

⚑ Units. `realized_l0_per_window` is PER WINDOW here; `probing.py`'s
`realized_l0` is PER TOKEN. Never cross-compare the two. Within this
file, `l0_unit` differs BY ARM by construction and is printed per row:
TXC + pooled count nonzeros in a d_sae code (one slot per feature);
stacked sums over T*d_sae slots. Stacked also gets T x the probe input,
so a stacked win is partly probe capacity, not architecture — reported
alongside, never netted out.

    .venv/bin/python -m experiments.explorations.task_hunt.sycgen.report_frontier
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

HERE = Path(__file__).resolve().parent
SRC = HERE / "results" / "frontier.json"


def main():
    if not SRC.exists():
        raise SystemExit(f"no frontier.json at {SRC} — run frontier.py first")
    rows = json.loads(SRC.read_text())
    print(f"[report] {len(rows)} rows\n")

    # (arm, T, k) -> aggregate over seeds
    agg: dict[tuple, list] = defaultdict(list)
    for r in rows:
        agg[(r["arm"], r["T"], r.get("k_tok"))].append(r)

    def cell(a, T, k):
        rs = agg.get((a, T, k), [])
        if not rs:
            return None
        return {
            "r": mean(x["recovery"] for x in rs),
            "sd": pstdev([x["recovery"] for x in rs]) if len(rs) > 1 else 0.0,
            "l0": mean(x["realized_l0_per_window"] for x in rs),
            "n": len(rs),
            "unit": rs[0].get("l0_unit", "?"),
        }

    Ts = sorted({r["T"] for r in rows})
    ks = sorted({r["k_tok"] for r in rows if r.get("k_tok") is not None})

    print("=== per-arm frontier (mean over seeds; l0 = PER WINDOW) ===")
    verdict_rows = []
    for T in Ts:
        txc = cell("txc", T, None)
        if txc is None:
            print(f"T={T}: no TXC row"); continue
        print(f"\nT={T}  TXC r={txc['r']:.4f}±{txc['sd']:.4f} "
              f"l0/win={txc['l0']:.2f} [{txc['unit']}] n={txc['n']}")
        best = {}
        for arm in ("pooled", "stacked"):
            pts = [(k, cell(arm, T, k)) for k in ks]
            pts = [(k, c) for k, c in pts if c]
            for k, c in pts:
                print(f"    {arm:8s} k={k:<3} r={c['r']:.4f}±{c['sd']:.4f} "
                      f"l0/win={c['l0']:.2f} [{c['unit']}]")
            # frontier point: best recovery at budget <= TXC's
            elig = [(k, c) for k, c in pts if c["l0"] <= txc["l0"] + 1e-9]
            if elig:
                k, c = max(elig, key=lambda kc: kc[1]["r"])
                best[arm] = (k, c, "<=TXC budget")
            elif pts:
                k, c = min(pts, key=lambda kc: kc[1]["l0"])
                best[arm] = (k, c, "CHEAPEST AVAILABLE — still costlier than TXC")
        line = {"T": T, "txc": txc}
        for arm, (k, c, how) in best.items():
            wins = txc["r"] > c["r"]
            print(f"  -> {arm:8s} best@budget: k={k} r={c['r']:.4f} "
                  f"l0/win={c['l0']:.2f} ({how})  TXC {'ABOVE' if wins else 'NOT above'}")
            line[arm] = {"k": k, "r": c["r"], "l0": c["l0"], "txc_above": wins,
                         "how": how}
        verdict_rows.append(line)

    print("\n=== VERDICT (pre-registered) ===")
    both = [v for v in verdict_rows
            if v.get("pooled", {}).get("txc_above")
            and v.get("stacked", {}).get("txc_above")]
    pooled_only = [v for v in verdict_rows
                   if v.get("pooled", {}).get("txc_above")
                   and not v.get("stacked", {}).get("txc_above")]
    for v in verdict_rows:
        p = v.get("pooled", {}); s = v.get("stacked", {})
        print(f"  T={v['T']:<3} TXC r={v['txc']['r']:.4f}  "
              f"vs pooled {'ABOVE' if p.get('txc_above') else 'below'}"
              f"  vs stacked {'ABOVE' if s.get('txc_above') else 'below'}")
    print(f"\n  TXC above BOTH arms at {len(both)}/{len(verdict_rows)} T values"
          f"; above pooled only at {len(pooled_only)}.")
    print("  Reminder: stacked carries T x the probe input, so a stacked")
    print("  loss is partly a probe-capacity effect and must be said so.")
    print("  If TXC is not above, item 6 is a NEGATIVE — report it as one.")


if __name__ == "__main__":
    main()
