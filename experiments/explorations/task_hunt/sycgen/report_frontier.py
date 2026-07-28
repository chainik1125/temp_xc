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

⚑⚑ THE RULE IN THE PARAGRAPH ABOVE IS BIASED TOWARD TXC — hub, LOG
`73f8ea388`, 00:41 07-29. "Best recovery at budget <= TXC's" is
conservative in words only. k is swept on a COARSE grid
(1,2,4,8,16,32) whose consecutive points differ by 40-75% in budget,
so no SAE point lands at TXC's budget and this rule silently selects a
MUCH CHEAPER baseline. At T=2 it compares TXC @5.66 against pooled
@3.51 -- 38% LESS BUDGET -- and returns "TXC above"; the cheapest
pooled point ABOVE TXC's budget (5.97, +5%) scores 0.4876 vs 0.4989,
inside the seed spread.

CORRECTED HEADLINE vs pooled: **above 2/4 (T=8, T=16),
INDISTINGUISHABLE 2/4 (T=2, T=4), never below** -- was above 3/4. The
authoritative surface is `figs_writeup/tab_sycgen_budget_matched.md`
(via `scripts/gen_sycgen_budget_table.py`), which BRACKETS TXC's budget
and interpolates to its exact l0, printing rules A/B/C side by side.

THIS FILE IS DELIBERATELY LEFT IMPLEMENTING RULE A. It is not a bug to
fix here: it is the rule-A reference, kept so the correction stays
auditable and so the two verdict implementations do not converge into
one premise wearing two coats -- which is exactly how this survived a
ratification and a cross-check. `scripts/verify_frontier_verdict.py`
carries a SELECTION-BIAS GUARD (fires when the chosen comparator spends
<0.90x TXC's budget) rather than a copy of the bracket rule.

Standing check before any comparator verdict ships: print the BUDGET
RATIO of the selected comparator to the model. If it is not ~1.0, the
word "matched" has not been earned.

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
            # THREE states, not two (hub review 3291d7287). A bare
            # `txc.r > c.r` makes 0.5001 vs 0.5000 an "ABOVE" that counts
            # toward the headline — the forbidden move, executed
            # mechanically. Outcome (d) "underpowered" is a DISTINCT
            # pre-registered outcome and must never be folded into a win
            # or a loss.
            #
            # ⚑ Deliberately NOT a significance test: n=3 does not
            # support one, and mac-c (72cf1334f) forbade importing the
            # 1.83-3.9x variance inflation onto these cells. This is a
            # CRUDE guard — |delta| against the larger of the two seed
            # SDs — and is labelled crude wherever it prints.
            delta = txc["r"] - c["r"]
            noise = max(txc["sd"], c["sd"])
            if abs(delta) <= noise:
                state = "INDISTINGUISHABLE"
            else:
                state = "ABOVE" if delta > 0 else "BELOW"
            print(f"  -> {arm:8s} best@budget: k={k} r={c['r']:.4f}±{c['sd']:.4f} "
                  f"l0/win={c['l0']:.2f} ({how})")
            print(f"       TXC {state:17s} |delta|={abs(delta):.4f} "
                  f"vs max(sd)={noise:.4f}  [crude, n=3]")
            line[arm] = {"k": k, "r": c["r"], "l0": c["l0"], "state": state,
                         "delta": delta, "noise": noise, "how": how}
        verdict_rows.append(line)

    print("\n=== VERDICT (pre-registered, THREE states) ===")
    for v in verdict_rows:
        p = v.get("pooled", {}); s = v.get("stacked", {})
        print(f"  T={v['T']:<3} TXC r={v['txc']['r']:.4f}±{v['txc']['sd']:.4f}"
              f"   vs pooled {p.get('state','-'):17s}"
              f" vs stacked {s.get('state','-')}")

    def tally(arm):
        c = {"ABOVE": 0, "BELOW": 0, "INDISTINGUISHABLE": 0}
        for v in verdict_rows:
            st = v.get(arm, {}).get("state")
            if st:
                c[st] += 1
        return c

    n = len(verdict_rows)
    for arm in ("pooled", "stacked"):
        c = tally(arm)
        print(f"\n  vs {arm}: ABOVE {c['ABOVE']}/{n}, BELOW {c['BELOW']}/{n}, "
              f"INDISTINGUISHABLE {c['INDISTINGUISHABLE']}/{n}")
    both = sum(1 for v in verdict_rows
               if v.get("pooled", {}).get("state") == "ABOVE"
               and v.get("stacked", {}).get("state") == "ABOVE")
    print(f"\n  TXC ABOVE BOTH arms at {both}/{n} T values.")
    print("  INDISTINGUISHABLE is a THIRD outcome — it is NOT a win and")
    print("  NOT a loss, and is never folded into either. The honest")
    print("  headline may be 'we cannot tell at n=3'; that is a result.")
    print("  The threshold is CRUDE (|delta| vs max seed sd), not a")
    print("  significance test — n=3 does not support one.")
    print("  Reminder: stacked carries T x the probe input, so a stacked")
    print("  loss is partly a probe-capacity effect and must be said so.")
    print("  If TXC is not ABOVE, item 6 is a NEGATIVE — report it as one.")


if __name__ == "__main__":
    main()
