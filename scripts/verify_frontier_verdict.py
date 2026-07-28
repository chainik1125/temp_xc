"""Independent hub check of the item-6 frontier verdict.

Does NOT duplicate `sycgen/report_frontier.py` — that file is mac-d's and
produces the verdict. This one re-derives it from the same
`frontier.json` and adds the piece the pre-registration requires and the
report script does not yet implement: **outcome (d), UNDERPOWERED, as a
state distinct from a win or a loss.**

Why it exists (hub review, LOG ~23:0x 07-28): `report_frontier.py` decides
with `wins = txc.r > pooled.r`, a bare inequality on two means, with the
seed `sd` computed, printed, and unused. At n=3 with training variance
dominating sampling variance, `0.5001 > 0.5000` reports ABOVE. A rule
that cannot say "we do not know" will always return something.

The test here is deliberately CRUDE and disclosed as crude. It is NOT a
significance test: n=3 does not support one, and mac-c (`575958b0d`)
explicitly forbade importing their measured 1.83-3.99x sigma
understatement onto these cells — it argues (d) is live, it does not
size (d). So the rule is simply:

    |r_txc - r_arm| <= max(sd_txc, sd_arm)   =>   INDISTINGUISHABLE

...which asks only "is the gap smaller than the spread we can see?"

⚑ Budget axis: `realized_l0_per_window` ONLY. Never `l0_per_token` —
it is not measured, it is `l0_per_window / T` (`synthetic_recovery.py:201`),
and reading a trend on that axis manufactures a free lunch
(mac-c `c1a9f98ad`, ratified).

    .venv/bin/python scripts/verify_frontier_verdict.py [path/to/frontier.json]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "experiments/explorations/task_hunt/sycgen/results/frontier.json"


def agg(rows, arm, T, k):
    rs = [r for r in rows
          if r.get("arm") == arm and r.get("T") == T and r.get("k_tok") == k]
    if not rs:
        return None
    vals = [r["recovery"] for r in rs]
    return {"r": mean(vals),
            "sd": pstdev(vals) if len(vals) > 1 else 0.0,
            "l0": mean(r["realized_l0_per_window"] for r in rs),
            "n": len(vals),
            "unit": rs[0].get("l0_unit", "?")}


def main(path: Path) -> int:
    if not path.exists():
        print(f"no frontier.json at {path} — nothing to verify yet")
        return 0
    rows = json.loads(path.read_text())
    Ts = sorted({r["T"] for r in rows})
    ks = sorted({r["k_tok"] for r in rows if r.get("k_tok") is not None})
    tally = {"above": 0, "below": 0, "indistinguishable": 0}

    print("Budget axis: realized_l0_per_window (per-window). "
          "l0_per_token is NOT measured — never use it.\n")
    for T in Ts:
        txc = agg(rows, "txc", T, None)
        if not txc:
            continue
        print(f"T={T}  TXC r={txc['r']:.4f} sd={txc['sd']:.4f} "
              f"l0/win={txc['l0']:.2f} n={txc['n']}")
        for arm in ("pooled", "stacked"):
            pts = [(k, agg(rows, arm, T, k)) for k in ks]
            pts = [(k, c) for k, c in pts if c]
            elig = [(k, c) for k, c in pts if c["l0"] <= txc["l0"] + 1e-9]
            if not elig:
                # "No eligible point" splits into TWO very different cases and
                # collapsing them loses real information (found when this
                # disagreed with mac-d at T=16, and mac-d was right):
                #   - every SAE point COSTLIER and TXC still wins  -> STRONGER
                #     than matched budget: TXC is better AND cheaper.
                #   - every SAE point cheaper                      -> genuinely
                #     not comparable at TXC's cost.
                ch_k, ch = min(pts, key=lambda kc: kc[1]["l0"])
                if ch["l0"] > txc["l0"] and txc["r"] > ch["r"]:
                    tally["above"] += 1
                    print(f"   {arm:8s}: cheapest k={ch_k} r={ch['r']:.4f} "
                          f"l0/win={ch['l0']:.2f} costs {ch['l0']/txc['l0']:.2f}x "
                          f"TXC | TXC beats it by {txc['r']-ch['r']:+.4f} "
                          f"-> TXC above AND CHEAPER (stronger than matched)")
                else:
                    print(f"   {arm:8s}: no point at budget <= TXC's and the "
                          f"cheapest ({ch['l0']:.2f}) does not resolve it — "
                          f"NOT COMPARABLE")
                continue
            k, c = max(elig, key=lambda kc: kc[1]["r"])
            d = txc["r"] - c["r"]
            spread = max(txc["sd"], c["sd"])
            if abs(d) <= spread:
                state, key = "INDISTINGUISHABLE (n=3)", "indistinguishable"
            elif d > 0:
                state, key = "TXC above", "above"
            else:
                state, key = "TXC below", "below"
            tally[key] += 1
            print(f"   {arm:8s}: best@k={k} r={c['r']:.4f} sd={c['sd']:.4f} "
                  f"l0/win={c['l0']:.2f} | delta={d:+.4f} spread={spread:.4f}"
                  f" -> {state}")

    print(f"\nTALLY  above={tally['above']}  below={tally['below']}  "
          f"indistinguishable={tally['indistinguishable']}")
    print("\nThe indistinguishable count is NOT folded into either side.")
    print("If it dominates, the honest headline is 'we cannot tell at n=3'")
    print("— outcome (d), which is distinct from a loss. Sizing (d) properly")
    print("needs a 5-seed treatment on item 6's own cells, not a factor")
    print("imported from another leg (mac-c 575958b0d).")
    print("\nStacked carries T x the probe input: a stacked result is partly")
    print("probe capacity, not architecture. Reported, never netted out.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT))
