"""Which k values does the SAE sweep actually need to bracket TXC's budget?

Hub pre-spend planning for the sparsity-matched shuffle run
(`briefings/sycgen-shuffle-sparsity-matched.md` §2b), which says "sweep k
finely enough to bracket tightly". That is a direction, not a grid. This
turns it into a grid, at $0, from `frontier.json` — which already
measures pooled's realized `l0_per_window` at every k.

WHY IT MATTERS: the delivered budget table reported "TXC above 3/4"
because the comparator rule picked the best pooled point at
`l0 <= TXC's l0` on a COARSE grid, which at T=2 meant a baseline given
38% less budget (LOG `73f8ea388`). The fix is a tighter bracket. This
script says exactly where a tighter bracket is possible, and — the part
that is not obvious — where it is NOT.

METHOD: pooled's realized l0 is well described by a power law in k over a
bracketing pair, so fit `l0 = a * k^b` on the two points straddling TXC's
budget and solve for the k that lands on it. Integer k only.

    .venv/bin/python scripts/plan_bracket_grid.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "experiments/explorations/task_hunt/sycgen/results/frontier.json"


def main() -> int:
    rows = json.loads(SRC.read_text())
    Ts = sorted({r["T"] for r in rows})
    ks = sorted({r["k_tok"] for r in rows if r.get("k_tok") is not None})

    def l0(arm, T, k):
        rs = [r for r in rows if r["arm"] == arm and r["T"] == T
              and r.get("k_tok") == k]
        return mean(r["realized_l0_per_window"] for r in rs) if rs else None

    print("Target: a pooled point landing AT TXC's measured l0/window.")
    print("Budget axis is realized_l0_per_window — never the derived "
          "per-token axis.\n")

    add, structural = {}, []
    for T in Ts:
        txc = mean(r["realized_l0_per_window"] for r in rows
                   if r["arm"] == "txc" and r["T"] == T)
        pts = [(k, l0("pooled", T, k)) for k in ks]
        pts = [(k, v) for k, v in pts if v is not None]
        print(f"T={T:2d}  TXC l0/win={txc:.2f}   pooled: "
              + " ".join(f"k{k}={v:.2f}" for k, v in pts))

        lo = [(k, v) for k, v in pts if v <= txc]
        hi = [(k, v) for k, v in pts if v > txc]
        if not lo:
            ck, cv = min(pts, key=lambda p: p[1])
            print(f"      ** pooled's CHEAPEST point (k={ck}, {cv:.2f}) "
                  f"already costs {cv/txc:.2f}x TXC. k=1 is the FLOOR.")
            print(f"      => T={T} is STRUCTURALLY Pareto: no finer grid can "
                  f"reach TXC's budget, so this is NOT a grid artifact and "
                  f"cannot be swept away.\n")
            structural.append(T)
            continue

        (kl, vl) = max(lo, key=lambda p: p[1])
        (kh, vh) = min(hi, key=lambda p: p[1])
        frac = (txc - vl) / (vh - vl)
        b = (math.log(vh) - math.log(vl)) / (math.log(kh) - math.log(kl))
        k_star = kl * math.exp((math.log(txc) - math.log(vl)) / b)
        print(f"      bracket k={kl}({vl:.2f}) .. k={kh}({vh:.2f})  "
              f"width={vh/vl:.2f}x | TXC sits {frac*100:.0f}% across it")
        # ⚑ Only recommend new cells if BOTH existing ends are far from
        # TXC's budget. An earlier version recommended k=3 at T=2 and T=4
        # purely because ceil(k*) was a new integer — ignoring that an
        # existing end was already within 5%. Recommending cells that
        # cannot move a verdict is the same failure as the coarse-grid
        # bias, just pointed at the budget instead of the result.
        near = min(abs(vl / txc - 1), abs(vh / txc - 1))
        cand = sorted({round(k_star), math.floor(k_star), math.ceil(k_star)})
        cand = [c for c in cand if c >= 1 and c not in ks]
        if near <= 0.10:
            end = "below" if abs(vl / txc - 1) < abs(vh / txc - 1) else "above"
            print(f"      k* = {k_star:.2f}; nearest existing end is "
                  f"{near*100:.1f}% {end} TXC's budget -> ALREADY TIGHT, "
                  f"add nothing. A new cell here cannot move the verdict.\n")
        elif cand:
            add[T] = cand
            print(f"      k* = {k_star:.2f}; nearest end is {near*100:.1f}% "
                  f"off -> ADD k in {cand}\n")
        else:
            print(f"      k* = {k_star:.2f} -> falls on an EXISTING grid "
                  f"point; the integer grid is already as tight as it gets "
                  f"here.\n")

    print("=" * 68)
    print(f"ADD THESE CELLS ONLY: {add if add else '(none)'}")
    print(f"STRUCTURALLY PARETO (k=1 floor exceeds TXC's budget): T={structural}")
    print("\nReading: a finer k sweep is worth running ONLY where a bracket")
    print("end is far from TXC's budget. Where k=1 already costs more than")
    print("TXC, refinement is impossible and the Pareto result is the only")
    print("comparison available — in our favour, and legitimately so.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
