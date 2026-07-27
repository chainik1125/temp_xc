"""hunt4w2/verdict.py — mechanical § 4 scoring of the wave-2 screens
(HUNT4W2_SCREEN_CARD.md; the hunt4 § 4 rules verbatim — existential
form, SKIP handling per the approved 6b03b1b06 patch).

Bundle = per (corpus, face) over screened models: 2/2 agreement
stands, splits PENDING-THIRD-LEG (llama31 stream regenerable by the
committed gen4c builder — a third leg needs its npz + floors built
first; the card prices it as conditional). Order routing: wd
win−shuf ≥ +0.03 at any T ∈ {4,8,16,32} where the wd gain is
positive. PENDING TEAM REVIEW.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.explorations.task_hunt.hunt4.verdict import score_model
from experiments.explorations.task_hunt.hunt4w2.screen import CORPUS_FACES

HERE = Path(__file__).resolve().parent
MODELS = ("gpt2", "gemma2_2b", "llama31_8b")


def main():
    out = {"card": "HUNT4W2_SCREEN_CARD.md § 4 (mechanical, hunt4 rules)",
           "status": "PENDING TEAM REVIEW", "corpora": {}}
    for corpus, faces in CORPUS_FACES.items():
        node = {"models": {}, "bundle": {}}
        present = []
        for m in MODELS:
            p = HERE / "results" / f"screen_{corpus}_{m}.json"
            if not p.exists():
                node["models"][m] = "MISSING"
                continue
            present.append(m)
            c = json.loads(p.read_text())["cells"]
            node["models"][m] = {f: score_model(c, f) for f in faces}
        for f in faces:
            vs_all = {m: node["models"][m][f]["verdict"] for m in present}
            vs = [v for v in vs_all.values() if v != "SKIP"]
            keep_n, kill_n = vs.count("KEEP"), vs.count("KILL")
            if not vs:
                bundle = "SKIP-INFEASIBLE"
            elif len(vs) >= 3:
                bundle = ("KEEP" if keep_n >= 2 else
                          "KILL" if kill_n >= 2 else "WEAK")
            elif len(vs) == 2:
                bundle = vs[0] if vs[0] == vs[1] else "PENDING-THIRD-LEG"
            else:
                bundle = f"single-model {vs[0]} (others SKIP/absent)"
            orders = [node["models"][m][f].get("order_pass_wd", False)
                      for m in present]
            node["bundle"][f] = {
                "verdicts": vs_all, "bundle_verdict": bundle,
                "order_pass_models": int(sum(orders)),
                "table": ("panel-gate candidate" if bundle == "KEEP"
                          and sum(orders) >= 2 else
                          "breadth" if bundle == "KEEP" else "—")}
        out["corpora"][corpus] = node
    dst = HERE / "results" / "verdict.json"
    dst.write_text(json.dumps(out, indent=1))
    print(json.dumps({c: n["bundle"] for c, n in out["corpora"].items()},
                     indent=1))
    print(f"[verdict] wrote {dst}")


if __name__ == "__main__":
    main()
