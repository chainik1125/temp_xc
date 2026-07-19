"""Cycle-3 Stage 2 — blind selection over the 4 new interaction/equality cards.

Four of Cycle 3's six calibration slots are FIXED by the reviewed briefing
(two C2 re-freezes + the two still-frozen C1 cards) — no scoring can move
them. The two open slots take one new categorical interaction/equality card
per domain, chosen by the same separated design as always: an independent
think-judge scores labelability × novelty × predicted-temporalness (the
generator never scores its own hypotheses; no data has been touched), then
selection is deterministic code: top score product per domain, ties by name.

Writes `results/selection_cycle3.json` + `selection_cycle3.md`.

    .venv/bin/python -m experiments.explorations.synthetic.expansion.select_c3
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from explorations.synthetic.expansion.client import Judge, Meter

HERE = Path(__file__).resolve().parent

MANDATED = [
    {"name": "list-item-parallelism-r2", "domain": "text-corpus",
     "temporal_class": "interaction/equality",
     "why": "briefing-mandated re-freeze (C2 gate-8 tolerance mis-scaled; "
            "strongest C2 signal, likely first text PROCEED)"},
    {"name": "computation-verification-r2", "domain": "reasoning-trace",
     "temporal_class": "periodic",
     "why": "briefing-mandated re-freeze (real spectral peak; periodic_hawkes "
            "hybrid mirror replaces the periodic-only mirror gate-8 killed)"},
    {"name": "enumeration-cadence", "domain": "text-corpus",
     "temporal_class": "periodic",
     "why": "frozen C1 card, fills periodic × text (ledger under-coverage)"},
    {"name": "goal-restatement-recurrence", "domain": "reasoning-trace",
     "temporal_class": "long-memory",
     "why": "frozen C1 card, fills long-memory × reasoning (ledger under-coverage)"},
]

SYSTEM = """\
You are the selection scorer for a measure->mirror benchmark-expansion loop
(Cycle 3). You will see 4 frozen candidate prereg cards — all categorical
per-sentence content labels whose measured statistic is the equality-adjacency
[c_t == c_{t-1}] (categorical self-match ACF). Score each on three axes, 1-5
integers:

- labelability: will a Claude Haiku judge, given ONE sentence and ZERO context,
  reliably assign one of the k content classes? Penalize ambiguous class
  boundaries, classes needing document context, and marginals likely to violate
  the card's own floors (one class >75%, or fewer than 3 classes above ~3%) —
  in a general web-corpus sample, instruction-specific classes may starve.
- novelty: new axis coverage for a program holding: backtracking (reasoning,
  bursty, SPEC), assumption-consequence (reasoning, AC-order, SPEC),
  hedging-drift (reasoning, DC-drift, SPEC*), self-reference-echo (reasoning,
  bursty, SPEC* — a FAILED interaction/equality attempt that measured as
  self-excitation). Reward mechanisms whose multi-class run/segment structure is
  genuinely new; penalize anything that will plausibly measure as one dominant
  class + binary clustering (the re-filing trap).
- predicted_temporalness: will the self-match ACF(1) beat the N1 within-doc
  permutation and N2 trend nulls beyond sampling + labeler noise? A clean ABORT
  is fine; a near-constant or context-starved label stream wastes the slot.

Be a skeptic: you allocate 2 expensive calibration slots (one per domain), not 4.
Respond with ONLY a JSON array (no prose, no fence) of:
{"name": "<candidate name>", "labelability": 1-5, "novelty": 1-5,
 "predicted_temporalness": 1-5, "reason": "<one crisp sentence per axis>"}"""


def main():
    meter = Meter()
    judge = Judge(meter)
    cands = json.loads((HERE / "results/candidates_cycle3.json").read_text())["candidates"]

    user = ("The 4 frozen candidate cards:\n\n" + json.dumps(cands, indent=1)
            + "\n\nScore all 4. JSON array only.")
    text = judge.call("think", SYSTEM, user, max_tokens=4000, tag="select_c3")
    scores = {s["name"]: s for s in json.loads(re.search(r"\[.*\]", text, re.S).group(0))}
    assert set(scores) == {c["name"] for c in cands}, "scorer must cover all candidates"

    rows = []
    for c in cands:
        s = scores[c["name"]]
        rows.append({
            "name": c["name"], "domain": c["domain"],
            "temporal_class": c["temporal_class"],
            "labelability": s["labelability"], "novelty": s["novelty"],
            "predicted_temporalness": s["predicted_temporalness"],
            "score": s["labelability"] * s["novelty"] * s["predicted_temporalness"],
            "reason": s["reason"],
        })

    selected_new = []
    for dom in ("reasoning-trace", "text-corpus"):
        pool = sorted([r for r in rows if r["domain"] == dom],
                      key=lambda r: (-r["score"], r["name"]))
        selected_new.append(pool[0])

    slate = [r["name"] for r in selected_new] + [m["name"] for m in MANDATED]
    out = {"n_select": len(slate),
           "rule": "4 slots briefing-mandated (re-freezes + frozen C1 cards); "
                   "2 open slots = top labelability*novelty*predicted_temporalness "
                   "per domain among the new interaction/equality cards; "
                   "scorer=claude-opus-4-8 (blind: no data, no arch scores)",
           "selected_new": [r["name"] for r in selected_new],
           "mandated": MANDATED, "scores": rows, "slate": slate}
    (HERE / "results/selection_cycle3.json").write_text(json.dumps(out, indent=2))

    lines = ["# Cycle-3 selection (Stage 2, blind)", "",
             f"Rule: {out['rule']}", "",
             "## Open slots — new interaction/equality cards", "",
             "| candidate | domain | lab | nov | temp | score | picked |",
             "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: -r["score"]):
        pick = "**✓**" if r["name"] in out["selected_new"] else ""
        lines.append(f"| {r['name']} | {r['domain']} | {r['labelability']} | "
                     f"{r['novelty']} | {r['predicted_temporalness']} | "
                     f"{r['score']} | {pick} |")
    lines += ["", "## Scorer reasons", ""]
    lines += [f"- **{r['name']}** — {r['reason']}" for r in rows]
    lines += ["", "## Mandated slots (briefing, not scored)", ""]
    lines += [f"- **{m['name']}** ({m['domain']} × {m['temporal_class']}) — {m['why']}"
              for m in MANDATED]
    lines += ["", f"Full slate ({len(slate)}, 3 per domain): "
              + ", ".join(f"`{n}`" for n in slate), ""]
    (HERE / "selection_cycle3.md").write_text("\n".join(lines) + "\n")

    print(json.dumps(out["slate"], indent=1))
    print(f"[spend] ${meter.spent:.3f} of ${meter.cap:.0f}")


if __name__ == "__main__":
    main()
