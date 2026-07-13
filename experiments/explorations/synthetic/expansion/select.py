"""Stage 2 — blind selection: top-N by labelability × novelty × temporalness.

Scores come from a SEPARATE think-judge call (never the generator scoring its
own hypotheses; no data has been touched; no architecture scores exist).
Selection itself is deterministic code: per-domain floor (≥ N/2 from each
domain) + under-coverage bias (a candidate in an empty or abort-only ledger
cell outranks one in a cell that already has a PROCEED), then the score
product. Writes `results/selection.json` + a human-readable `selection.md`.

    .venv/bin/python -m experiments.explorations.synthetic.expansion.select
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from explorations.synthetic.expansion.client import Judge, Meter

HERE = Path(__file__).resolve().parent
N_SELECT = 4  # Cycle 1: 2 per domain — prove the loop, don't scale yet

# Ledger cell status at selection time (domain, temporal-class) -> status.
# Anything not listed is empty. Empty/abort-only = under-covered (tier 1);
# a cell holding a PROCEED = tier 0 (deprioritized).
CELL_STATUS = {
    ("reasoning-trace", "bursty/self-exciting"): "PROCEED",  # backtracking SPEC
    ("text-corpus", "DC-slow-drift"): "ABORT-only",          # topic_switching
}

SYSTEM = """\
You are the selection scorer for a measure->mirror benchmark-expansion loop. You will
see 10 frozen candidate prereg cards. Score each on three axes, 1-5 integers:

- labelability: will a Claude Haiku judge, given ONE sentence + ~3 sentences of
  preceding context, produce a reliable label? Penalize ambiguity, subtle judgments,
  extreme base rates (<2% or >80% positives make temporal statistics starved or
  saturated), and labels needing whole-document understanding.
- novelty: how much new axis coverage vs a program that already has: backtracking
  (reasoning, bursty, PROCEED), topic-switching (text, DC-drift, ABORT — labeler
  inadequate), and synthetic benches for periodic tones, changepoint modes, signed
  motion. Consider whether the labeler/mechanism differs, not just the cell name.
- predicted_temporalness: a-priori, will the ORDERED statistic beat a within-document
  permutation (N1) and a position-trend-preserving null (N2) beyond sampling + labeler
  noise? This is triage, not a reward: a clean ABORT is a fine outcome, but a label
  that is near-constant within documents (pure composition) or iid wastes the slot.
  Score what the within-document ORDER structure plausibly is.

Be a skeptic: your job is to allocate 4 expensive calibration slots well, not to be
agreeable. Respond with ONLY a JSON array (no prose, no fence) of:
{"name": "<candidate name>", "labelability": 1-5, "novelty": 1-5,
 "predicted_temporalness": 1-5, "reason": "<one crisp sentence per axis>"}"""


def main():
    meter = Meter()
    judge = Judge(meter)
    cands = json.loads((HERE / "results/candidates.json").read_text())["candidates"]

    user = ("The 10 frozen candidate cards:\n\n" + json.dumps(cands, indent=1)
            + "\n\nScore all 10. JSON array only.")
    text = judge.call("think", SYSTEM, user, max_tokens=4000, tag="select")
    scores = {s["name"]: s for s in json.loads(re.search(r"\[.*\]", text, re.S).group(0))}
    assert set(scores) == {c["name"] for c in cands}, "scorer must cover all candidates"

    rows = []
    for c in cands:
        s = scores[c["name"]]
        cell = (c["domain"], c["temporal_class"])
        tier = 0 if CELL_STATUS.get(cell) == "PROCEED" else 1
        rows.append({
            "name": c["name"], "domain": c["domain"],
            "temporal_class": c["temporal_class"],
            "cell_status": CELL_STATUS.get(cell, "empty"), "tier": tier,
            "labelability": s["labelability"], "novelty": s["novelty"],
            "predicted_temporalness": s["predicted_temporalness"],
            "score": s["labelability"] * s["novelty"] * s["predicted_temporalness"],
            "reason": s["reason"],
        })

    selected = []
    for dom in ("reasoning-trace", "text-corpus"):     # per-domain floor: 2 each
        pool = sorted([r for r in rows if r["domain"] == dom],
                      key=lambda r: (-r["tier"], -r["score"], r["name"]))
        # under-coverage bias also within the pick: never two picks in one cell
        picked, cells = [], set()
        for r in pool:
            if (r["domain"], r["temporal_class"]) in cells:
                continue
            picked.append(r)
            cells.add((r["domain"], r["temporal_class"]))
            if len(picked) == N_SELECT // 2:
                break
        selected += picked

    out = {"n_select": N_SELECT, "rule": "tier(under-coverage) desc, then "
           "labelability*novelty*predicted_temporalness desc; per-domain floor "
           "N/2; max one pick per ledger cell; scorer=claude-opus-4-8 (blind: "
           "no data, no arch scores)",
           "selected": [r["name"] for r in selected], "scores": rows}
    (HERE / "results/selection.json").write_text(json.dumps(out, indent=2))

    lines = ["# Cycle-1 selection (Stage 2, blind)", "",
             f"Rule: {out['rule']}", "",
             "| candidate | domain | class | cell | lab | nov | temp | score | picked |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (-r["tier"], -r["score"])):
        pick = "**✓**" if r["name"] in out["selected"] else ""
        lines.append(f"| {r['name']} | {r['domain']} | {r['temporal_class']} | "
                     f"{r['cell_status']} | {r['labelability']} | {r['novelty']} | "
                     f"{r['predicted_temporalness']} | {r['score']} | {pick} |")
    lines += ["", "## Scorer reasons", ""]
    lines += [f"- **{r['name']}** — {r['reason']}" for r in rows]
    (HERE / "selection.md").write_text("\n".join(lines) + "\n")

    print(json.dumps(out["selected"], indent=1))
    print(f"[spend] ${meter.spent:.3f} of ${meter.cap:.0f}")


if __name__ == "__main__":
    main()
