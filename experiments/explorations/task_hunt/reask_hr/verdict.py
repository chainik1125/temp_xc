"""reask_hr/verdict.py — mechanical § 4 scoring of the reask_hr
screens (REASK_HR_SCREEN_CARD.md; the hunt4 § 4 rules verbatim —
existential form, SKIP handling per the approved 6b03b1b06 patch;
`hunt4.verdict.score_model` imported unmodified).

Bundle = majority over the three models (all legs run from the
freeze). Order routing: wd win−shuf >= +0.03 at any T in
{4,8,16,32} where the wd gain is positive. PENDING TEAM REVIEW.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.explorations.task_hunt.hunt4.verdict import score_model

HERE = Path(__file__).resolve().parent
MODELS = ("gpt2", "gemma2_2b", "llama31_8b")
FACE = "reask_hr"


def main():
    out = {"card": "REASK_HR_SCREEN_CARD.md § 4 (mechanical, hunt4 rules)",
           "status": "PENDING TEAM REVIEW",
           "models": {}, "bundle": {}}
    present = []
    for m in MODELS:
        p = HERE / "results" / f"screen_wildchat_{m}.json"
        if not p.exists():
            out["models"][m] = "MISSING"
            continue
        present.append(m)
        c = json.loads(p.read_text())["cells"]
        out["models"][m] = {FACE: score_model(c, FACE)}
    vs_all = {m: out["models"][m][FACE]["verdict"] for m in present}
    vs = [v for v in vs_all.values() if v != "SKIP"]
    keep_n, kill_n = vs.count("KEEP"), vs.count("KILL")
    if not vs:
        bundle = "SKIP-INFEASIBLE"
    elif len(vs) >= 3:
        bundle = ("KEEP" if keep_n >= 2 else
                  "KILL" if kill_n >= 2 else "WEAK")
    elif len(vs) == 2:
        bundle = vs[0] if vs[0] == vs[1] else "WEAK-SPLIT (2 legs)"
    else:
        bundle = f"single-model {vs[0]} (others SKIP/absent)"
    orders = [out["models"][m][FACE].get("order_pass_wd", False)
              for m in present]
    out["bundle"][FACE] = {
        "verdicts": vs_all, "bundle_verdict": bundle,
        "order_pass_models": int(sum(orders)),
        "table": ("panel-gate candidate" if bundle == "KEEP"
                  and sum(orders) >= 2 else
                  "breadth" if bundle == "KEEP" else "—")}
    dst = HERE / "results" / "verdict.json"
    dst.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["bundle"], indent=1))
    print(f"[verdict] wrote {dst}")


if __name__ == "__main__":
    main()
