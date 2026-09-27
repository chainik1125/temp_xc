"""sycgen/verdict.py — mechanical § 4 scoring of the sycgen screens
(SCREEN_CARD.md; hunt4 § 4 rules verbatim — `hunt4.verdict.score_model`
imported unmodified; GO condition 4).

Bundle = majority over the three models. GO condition 3: the
within-domain vocab numbers are carried BESIDE the verdict, per model,
lifted from each screen json's row stats. PENDING TEAM REVIEW.

Run: .venv/bin/python -m experiments.explorations.task_hunt.sycgen.verdict
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.explorations.task_hunt.hunt4.verdict import score_model

HERE = Path(__file__).resolve().parent
MODELS = ("gpt2", "gemma2_2b", "llama31_8b")
FACE = "sycgen_age"


def main():
    out = {"card": "sycgen/SCREEN_CARD.md § 4 (mechanical, hunt4 rules)",
           "status": "PENDING TEAM REVIEW",
           "go_conditions": "dc3cb8fd9",
           "models": {}, "vocab_beside_verdict": {}, "bundle": {}}
    present = []
    for m in MODELS:
        p = HERE / "results" / f"screen_sycgen_{m}.json"
        if not p.exists():
            out["models"][m] = "MISSING"
            continue
        present.append(m)
        blob = json.loads(p.read_text())
        out["models"][m] = {FACE: score_model(blob["cells"], FACE)}
        out["vocab_beside_verdict"][m] = \
            blob["meta"]["rows"].get("within_domain_vocab", {})
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
        bundle = f"SINGLE-LEG ({vs[0]})"
    out["bundle"] = {"verdicts": vs_all, "bundle": bundle,
                     "keep": keep_n, "kill": kill_n}
    p = HERE / "results" / "verdict_sycgen.json"
    p.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out["bundle"], indent=1))
    print(f"-> {p}")


if __name__ == "__main__":
    main()
