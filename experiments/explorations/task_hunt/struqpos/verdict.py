"""STRUQPOS verdict — hunt4 §4 existential form, binary-doc adapted
(executes STRUQPOS_SCREEN_CARD §5). Reads the 3 per-leg screen JSONs and
bundles a KEEP/KILL by majority.

Per leg, the contextual arm KEEP-qualifies iff ALL of:
  gain          ctx.best - tok.best   >= +0.05
  order-carried ctx.best - shuf.best  >= +0.02
  above-floor   ctx.best  >  local_floor.best
KILL clauses (any ⇒ leg KILL):
  C1  tok.best >= 0.60 OR local_floor.best >= 0.60   (local leak)
  C2  ctx.best - shuf.best < +0.02                    (content/proximity, not position)
  C3  ctx.best - tok.best  < +0.05                    (no signal over bag)
Null-integrity gate (PIN 1): shuf_labelperm_auc must be ~0.50
(|.-0.50| <= 0.05); else the leg is DISCLOSED-not-scored.

Bundle = majority of the 3 legs: KEEP iff >=2 legs KEEP and no leg fires
C1. Run: .venv/bin/python -m experiments.explorations.task_hunt.struqpos.verdict
"""
from __future__ import annotations
import json
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
LEGS = ("gpt2", "gemma2_2b", "llama31_8b")
GAIN, ORDER, C1_BAR, PERM_TOL = 0.05, 0.02, 0.60, 0.05


def score_leg(s: dict) -> dict:
    a = s["arms"]
    tok, ctx = a["tok"]["best"], a["ctx"]["best"]
    shuf, floor = a["shuf"]["best"], a["local_floor"]["best"]
    lp = s.get("shuf_labelperm_auc", 0.5)
    null_ok = abs(lp - 0.5) <= PERM_TOL
    gain, order = ctx - tok, ctx - shuf
    c1 = tok >= C1_BAR or floor >= C1_BAR
    c2 = order < ORDER
    c3 = gain < GAIN
    keep = (gain >= GAIN) and (order >= ORDER) and (ctx > floor) and not c1
    kills = [k for k, v in (("C1", c1), ("C2", c2), ("C3", c3)) if v]
    if not null_ok:
        v = "DISCLOSED"           # null contaminated; not scored
    elif keep:
        v = "KEEP"
    else:
        v = "KILL"
    return {"leg": s["leg"], "verdict": v, "tok": tok, "ctx": ctx,
            "shuf": shuf, "local_floor": floor, "gain": round(gain, 4),
            "order_carried": round(order, 4), "shuf_labelperm": lp,
            "null_ok": null_ok, "kill_clauses": kills}


def main():
    legs = {}
    for leg in LEGS:
        p = RES / f"screen_struqpos_{leg}.json"
        if p.exists():
            legs[leg] = score_leg(json.loads(p.read_text()))
    keeps = [v for v in legs.values() if v["verdict"] == "KEEP"]
    any_c1 = any("C1" in v["kill_clauses"] for v in legs.values())
    scored = [v for v in legs.values() if v["verdict"] in ("KEEP", "KILL")]
    bundle = ("KEEP" if len(keeps) >= 2 and not any_c1
              else "KILL" if len(scored) >= 2 else "INCONCLUSIVE")
    out = {"face": "struqpos", "bundle_verdict": bundle,
           "n_keep": len(keeps), "n_scored": len(scored),
           "any_C1": any_c1, "legs": legs}
    (RES / "verdict.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(f"\nBUNDLE: {bundle} ({len(keeps)}/3 legs KEEP)")


if __name__ == "__main__":
    main()
