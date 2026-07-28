"""evalage/verdict.py — mechanical § 4 scoring of the evalage screens
(SCREEN_CARD.md § 3.5; hunt4 § 4 rules verbatim —
`hunt4.verdict.score_model` imported unmodified).

Bundle = majority over the three models. The topic-vocabulary numbers
ride BESIDE the verdict (card § 3.3), per model, lifted from each
screen json's row stats — both legs (events/conv, tokens/conv) plus the
per-topic within-topic unigram AUC. PENDING TEAM REVIEW.

Run: .venv/bin/python -m experiments.explorations.task_hunt.evalage.verdict
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.explorations.task_hunt.hunt4.verdict import score_model

HERE = Path(__file__).resolve().parent
MODELS = ("gpt2", "gemma2_2b", "llama31_8b")
FACE = "evalage_age"


def _vocab_summary(rows: dict) -> dict:
    """Compress the per-topic band to what a reader needs: the WORST
    topic on each leg, which is where a leak would show first."""
    v = rows.get("topic_vocab", {})
    if not v:
        return {}
    aucs = [(k, r["unigram_auc_within"]) for k, r in v.items()
            if r.get("unigram_auc_within") is not None]
    ev = [r["events_per_conv_cv"] for r in v.values()
          if r.get("events_per_conv_cv") is not None]
    tk = [r["tokens_per_conv_cv"] for r in v.values()
          if r.get("tokens_per_conv_cv") is not None]
    out = {"n_topics": len(v),
           "n_topics_scored": len(aucs),
           "n_topics_thin": sum(1 for r in v.values() if r.get("skipped"))}
    if aucs:
        worst = max(aucs, key=lambda kv: kv[1])
        out["worst_topic_unigram_auc"] = {"topic": worst[0],
                                          "auc": float(worst[1])}
        out["median_topic_unigram_auc"] = float(
            np.median([a for _, a in aucs]))
    if ev:
        out["events_per_conv_cv"] = {"max": float(max(ev)),
                                     "median": float(np.median(ev))}
    if tk:
        out["tokens_per_conv_cv"] = {"max": float(max(tk)),
                                     "median": float(np.median(tk))}
    out["two_leg_note"] = (
        "both legs reported on purpose — my vocabulary_control_check "
        "collapsed them into events-per-token, so evalage passed the "
        "LENGTH channel by luck (uniform max_new), not by design")
    return out


def main():
    out = {"card": "evalage/SCREEN_CARD.md § 3.5 (mechanical, hunt4 rules)",
           "status": "PENDING TEAM REVIEW",
           "frame": "GLOBAL terciles (card § 3.1)",
           "models": {}, "vocab_beside_verdict": {}, "frames": {},
           "bundle": {}}
    present = []
    for m in MODELS:
        p = HERE / "results" / f"screen_evalage_{m}.json"
        if not p.exists():
            out["models"][m] = "MISSING"
            continue
        present.append(m)
        blob = json.loads(p.read_text())
        rows = blob["meta"]["rows"]
        out["models"][m] = {FACE: score_model(blob["cells"], FACE)}
        out["vocab_beside_verdict"][m] = _vocab_summary(rows)
        out["frames"][m] = {
            "tercile_edges": rows.get("tercile_edges"),
            "edges_match_premeasure": rows.get("edges_match_premeasure"),
            "position_floor": blob["cells"].get(
                f"{FACE}/position_floor", {}).get("acc_test"),
            "wd_rows": rows.get(f"{FACE}_wd/test", {})}
    if not present:
        print("[verdict] no screen results present yet")
        return
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
                     "keep": keep_n, "kill": kill_n,
                     "models_present": present}
    p = HERE / "results" / "verdict_evalage.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out["bundle"], indent=1))
    print(f"-> {p}")


if __name__ == "__main__":
    main()
