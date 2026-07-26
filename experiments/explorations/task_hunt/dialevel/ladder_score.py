"""Read-only scorer for the R11 order-mechanism ladder (LADDER_CARD.md
§ 4, frozen and committed BEFORE any result exists).

Prints, per model: identity gates (base vs committed screen, L0 seed-0
vs committed R11 cost, T16 null + L4 replicas), per-arm 3-seed mean
costs with spreads and null bands, entropy disclosures, and evaluates
the frozen five-outcome verdict rule on the screened pair
(gpt2 + llama31_8b; gemma2_2b = coverage).

Run: .venv/bin/python -m experiments.explorations.task_hunt.dialevel.ladder_score
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
PAIR = ("gpt2", "llama31_8b")          # the briefing's 2/2
COVERAGE = "gemma2_2b"
ARMS = ("L0", "L1", "L2", "L3f", "L3n")
N_SEEDS = 3


def costs_for(key: str):
    lad = json.loads((RES / f"ladder_{key}.json").read_text())["cells"]
    scr = json.loads((RES / f"screen_{key}.json").read_text())["cells"]
    cap = json.loads((RES / "capacity_check.json").read_text())["cells"]
    out = {}
    for T in (16, 32):
        base = lad[f"T{T}/base"]["auc"]
        d = {"base": base,
             "base_committed": scr[f"wd/T{T}/win_linear"]["auc"],
             "null_dev": abs(lad[f"T{T}/null_label"]["auc"] - 0.5),
             "L4_auc": lad[f"T{T}/L4"]["auc"],
             "L4_committed": cap[f"{key}/T{T}/win_foreign_linear"]["auc"],
             "cost_committed": (scr[f"wd/T{T}/win_linear"]["auc"]
                                - scr[f"wd/T{T}/win_shuf_linear"]["auc"])}
        for arm in ARMS:
            cs = [base - lad[f"T{T}/{arm}/s{s}"]["auc"]
                  for s in range(N_SEEDS)]
            d[arm] = {"mean": sum(cs) / len(cs), "seeds": cs,
                      "spread": max(cs) - min(cs),
                      "moved_frac": lad[f"T{T}/{arm}/s0"].get("moved_frac")}
        d["L0_s0"] = base - lad[f"T{T}/L0/s0"]["auc"]
        out[T] = d
    if (RES / f"ladder_{key}.json").exists():
        out["null_T16_committed"] = scr["wd/T16/null_win_linear"]["auc"]
        out["null_T16_ladder"] = json.loads(
            (RES / f"ladder_{key}.json").read_text())["cells"][
                "T16/null_label"]["auc"]
    return out


def gates(key: str, c) -> list[str]:
    fails = []
    for T in (16, 32):
        if abs(c[T]["base"] - c[T]["base_committed"]) > 0.010:
            fails.append(f"gate3 T{T}: base {c[T]['base']:.4f} vs committed "
                         f"{c[T]['base_committed']:.4f}")
        if abs(c[T]["L4_auc"] - c[T]["L4_committed"]) > 0.010:
            fails.append(f"gate4/L4 T{T}: {c[T]['L4_auc']:.4f} vs committed "
                         f"{c[T]['L4_committed']:.4f}")
    if abs(c[32]["L0_s0"] - c[32]["cost_committed"]) > 0.015:
        fails.append(f"gate4 T32: L0 s0 cost {c[32]['L0_s0']:+.4f} vs "
                     f"committed R11 {c[32]['cost_committed']:+.4f}")
    if abs(c["null_T16_ladder"] - c["null_T16_committed"]) > 0.010:
        fails.append(f"gate4/null T16: {c['null_T16_ladder']:.4f} vs "
                     f"committed {c['null_T16_committed']:.4f}")
    return fails


def verdict(cc: dict) -> str:
    """cc: model -> costs_for output, for the screened PAIR only."""

    def both(f):
        return all(f(cc[m]) for m in PAIR)

    L0 = {m: cc[m][32]["L0"]["mean"] for m in PAIR}
    c = {m: {a: cc[m][32][a]["mean"] for a in ARMS} for m in PAIR}
    if both(lambda d: d[32]["L2"]["mean"] >= 0.5 * d[32]["L0"]["mean"]) and \
       both(lambda d: d[32]["L1"]["mean"] < (1 / 3) * d[32]["L0"]["mean"]):
        out = "TURN-STRUCTURE"
    elif both(lambda d: d[32]["L1"]["mean"] >= 0.5 * d[32]["L0"]["mean"]) and \
            both(lambda d: d[32]["L2"]["mean"] < (1 / 3) * d[32]["L0"]["mean"]):
        out = "WITHIN-TURN"
    elif both(lambda d: d[32]["L1"]["mean"] >= (1 / 3) * d[32]["L0"]["mean"]) \
            and both(lambda d: d[32]["L2"]["mean"]
                     >= (1 / 3) * d[32]["L0"]["mean"]):
        out = "MIXED"
    elif both(lambda d: d[32]["L3n"]["mean"] >= 2 * d[32]["L3f"]["mean"]
              and d[32]["L3n"]["mean"] - d[32]["L3f"]["mean"] >= 0.02
              and d[32]["L3n"]["mean"] >= (1 / 3) * d[32]["L0"]["mean"]):
        out = "RECENCY-RESIDUAL"
    else:
        return f"UNRESOLVED  (T32 L0={L0}, arms={c})"

    # T16 sign robustness on the defining arms
    need = {"TURN-STRUCTURE": ["L2"], "WITHIN-TURN": ["L1"],
            "MIXED": ["L1", "L2"], "RECENCY-RESIDUAL": ["L3n"]}[out]
    ok = all(all(cc[m][16][a]["mean"] > 0 for a in need) for m in PAIR)
    if out == "RECENCY-RESIDUAL":
        ok = ok and all(cc[m][16]["L3n"]["mean"] > cc[m][16]["L3f"]["mean"]
                        for m in PAIR)
    if not ok:
        return f"UNRESOLVED (T16 disagreement; T32 point estimate = {out})"
    return out


def main():
    cc = {}
    for key in PAIR + (COVERAGE,):
        if not (RES / f"ladder_{key}.json").exists():
            print(f"[{key}] no ladder results yet")
            continue
        c = costs_for(key)
        cc[key] = c
        g = gates(key, c)
        print(f"=== {key}" + ("  [COVERAGE]" if key == COVERAGE else ""))
        if g:
            print("  !! GATE FAILURES: " + "; ".join(g))
        for T in (16, 32):
            d = c[T]
            print(f"  T{T}: base={d['base']:.4f} "
                  f"(committed {d['base_committed']:.4f})  "
                  f"null_dev={d['null_dev']:.4f}  "
                  f"L4={d['L4_auc']:.4f} (committed {d['L4_committed']:.4f})")
            for arm in ARMS:
                a = d[arm]
                print(f"    {arm}: cost={a['mean']:+.4f} "
                      f"(spread {a['spread']:.4f}, "
                      f"moved {a['moved_frac']:.2f})"
                      + (f"  [s0={d['L0_s0']:+.4f} vs committed R11 "
                         f"{d['cost_committed']:+.4f}]" if arm == "L0"
                         else ""))
    if all(m in cc for m in PAIR):
        pair_gates = [f for m in PAIR for f in gates(m, cc[m])]
        if pair_gates:
            print("\nVERDICT: REPRODUCTION FAILURE — " + "; ".join(pair_gates))
        else:
            v = verdict(cc)
            print(f"\nVERDICT (2/2 pair, card § 4): {v}")
            if COVERAGE in cc and not gates(COVERAGE, cc[COVERAGE]):
                print(f"  coverage {COVERAGE}: T32 costs " + ", ".join(
                    f"{a}={cc[COVERAGE][32][a]['mean']:+.4f}"
                    for a in ARMS))


if __name__ == "__main__":
    main()
