"""rdens/verdict.py — mechanical § 3 scoring of the rdens factory
screen (RDENS_CARD.md; committed BEFORE the deciding result exists —
the screen launched this same push window, bed236f1d house practice).

Operationalization of the card § 3 clauses (factory conventions of
record; margins are the house +0.02):
  per model/layer, at each T:
    g          = real flat AUC − real tok AUC        (window gain)
    g_agg      = real mean AUC − real tok AUC        (pooling gain)
    order      = g − g_agg  (= flat − mean)
    null_arm_gap = real flat AUC − NULL-ARM flat AUC (σ_null clause)
    floor_bar  = max(slope_auc, rate_auc) at T from the COMMITTED
                 § 2 evidence lines (labels/rdens_stats.json)
  CLAIMING zone: T ≤ 16 (§ 2 pre-registration). T32 reported only.
  KEEP iff at SOME claiming T: g > 0 ∧ (g − g_agg) ≥ 0.02
       ∧ null_arm_gap ≥ 0.02 ∧ real flat AUC > floor_bar.
  KILL iff ANY of: g_agg ≥ g at every T ≥ 8 (the chaz clause);
       null_arm_gap < 0.02 at every claiming T;
       real flat AUC ≤ floor_bar at every claiming T.
  Else WEAK. Venue limits (hs11/distill absent) quoted, not scored.

Run: .venv/bin/python -m experiments.explorations.task_hunt.rdens.verdict
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABELS = HERE.parent / "labels"
CLAIM_TS = (2, 4, 8, 16)
ALL_TS = (2, 4, 8, 16, 32)
MARGIN = 0.02


def main():
    res = json.loads(
        (HERE / "results" / "rdens_main_screen.json").read_text())
    stats = json.loads((LABELS / "rdens_stats.json").read_text())
    ev = stats["visible_floor_auc_by_T"]

    def floor_bar(T):
        d = ev.get(str(T))
        if d is None:                      # T ∈ {2,4}: below the § 2
            d = ev["8"]                    # table — nearest committed
        return max(d["slope_auc"], d["rate_auc"])   # line, conservative

    cells = res["cells"]
    suffix = "/real/tok"
    combos = sorted({k[: -len(suffix)] for k in cells
                     if k.endswith(suffix)})
    out = {"card": "rdens/CARD.md § 3 (mechanical)",
           "status": "PENDING TEAM REVIEW", "combos": {}}
    for combo in combos:                   # e.g. "base/hs13"
        real_tok = cells.get(f"{combo}/real/tok")
        if real_tok is None:
            continue
        rows, keeps = {}, []
        kill_chaz, any_null_ok, any_floor_ok = True, False, False
        for T in ALL_TS:
            rc = cells.get(f"{combo}/real/T{T}")
            nc = cells.get(f"{combo}/null/T{T}")
            if rc is None:
                continue
            flat = rc["flat"]["auc"]
            row = {"g": round(rc["g"], 4), "g_agg": round(rc["g_agg"], 4),
                   "order": round(rc["g_order"], 4),
                   "shuffle_gap": round(rc["shuffle_gap"], 4),
                   "flat_auc": round(flat, 4),
                   "floor_bar": round(floor_bar(T), 4),
                   "claiming": T in CLAIM_TS}
            if nc is not None:
                row["null_arm_gap"] = round(flat - nc["flat"]["auc"], 4)
            if T >= 8 and rc["g_agg"] < rc["g"]:
                kill_chaz = False
            if T in CLAIM_TS:
                null_ok = row.get("null_arm_gap", -1) >= MARGIN
                floor_ok = flat > floor_bar(T)
                any_null_ok |= null_ok
                any_floor_ok |= floor_ok
                row["KEEP_at_T"] = bool(
                    rc["g"] > 0 and (rc["g"] - rc["g_agg"]) >= MARGIN
                    and null_ok and floor_ok)
                if row["KEEP_at_T"]:
                    keeps.append(T)
            rows[str(T)] = row
        verdict = ("KEEP" if keeps else
                   "KILL" if (kill_chaz or not any_null_ok
                              or not any_floor_ok) else "WEAK")
        out["combos"][combo] = {
            "per_T": rows, "KEEP_Ts": keeps,
            "kill_chaz_clause_gagg_swallows": bool(kill_chaz),
            "verdict": verdict,
            "venue_limits": "base/hs13 primary; hs11+distill expected "
                            "absent on ward volume (card § 3)"}
    dst = HERE / "results" / "rdens_verdict.json"
    dst.write_text(json.dumps(out, indent=1))
    print(json.dumps({c: {"verdict": v["verdict"], "KEEP_Ts": v["KEEP_Ts"]}
                      for c, v in out["combos"].items()}, indent=1))
    print(f"[verdict] wrote {dst}")


if __name__ == "__main__":
    main()
