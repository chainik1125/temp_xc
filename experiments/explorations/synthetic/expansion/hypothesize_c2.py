"""Cycle-2 Stage 1 — hypothesize the interaction/equality cards (Opus).

The one temporal class with ZERO proposals after Cycle 1, both domains — the
grounded analogue of the abstract changepoint bench's equality-pattern axis.
Exactly 4 new cards (2 per domain), all `interaction/equality`, designed
under the two gates the Cycle-1 review promoted to design-time:

- **Gate 7 (no-leakage labeler):** the per-sentence label must be assignable
  from the sentence's OWN content; the interaction/equality character must
  live in the STATISTIC computed across positions (match / echo /
  alternation of own-content labels), never in the label definition.
- **Gate 8 (non-fitted-moment mirror):** each card names ≥1 statistic its
  mirror is NOT fit to + an abs tolerance, before any fit.

Renders cards to `prereg/` + appends to `results/candidates_cycle2.json`.
COMMIT (freeze) before any labeling.

    .venv/bin/python -m experiments.explorations.synthetic.expansion.hypothesize_c2
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from experiments.explorations.synthetic.expansion.hypothesize import DOMAINS
from explorations.synthetic.expansion.client import Judge, Meter

HERE = Path(__file__).resolve().parent
N_NEW = 4

SYSTEM = """\
You are the hypothesis generator for a measure->mirror benchmark-expansion loop
(Cycle 2). PRIME DIRECTIVE: a sound verdict, never a win — an ABORT is a success;
never build the temporal statistic into the label.

Your task: propose candidates for the ONE uncovered temporal class,
**interaction/equality** — properties where the temporal structure is a
CROSS-POSITION comparison pattern: positions that MATCH each other (same discourse
role, same structural template, same referent class), echo at a lag, or alternate.
This is the grounded analogue of an abstract 'equality-pattern latent' benchmark
(where only architectures that can compare positions expose a boundary/matching
structure). The interesting signature is: P(label_t == label_{t+k}) structure
(match-runs, echo peaks, or ANTI-match alternation), MI between positions, or a
directed pair transition — all computed FROM the label stream.

TWO HARD DESIGN GATES (cards violating them are dead on arrival):
- Gate 7 — no-leakage labeler: the per-sentence label MUST be assignable from the
  sentence's OWN content alone (its wording, structure, markers). NEVER define a
  label by the sentence's relation to neighbours ('answers the preceding', 'same
  topic as before', 'repeats an earlier point'). The cross-position
  interaction/equality character lives ONLY in the statistic computed across the
  label stream, never in the label definition. Labels are judged with ZERO context
  sentences.
- Gate 8 — non-fitted-moment mirror: name >=1 statistic your chosen mirror is NOT
  fit to, plus an absolute tolerance it must meet on held-out draws.

Labeler constraints (same as Cycle 1): per-sentence binary 0/1 or small ordinal
0..4, decidable by a Claude Haiku judge from the single sentence; an independent
10-line heuristic (keywords/regex/punctuation) must exist as cross-check; avoid
labels with base rate <2% or >80% (starved/saturated statistics).

The measurement battery your primary statistic must come from:
- "acf1"  — lag-1 autocorrelation (binary indicator ACF or categorical self-match ACF;
            NEGATIVE acf1 = alternation, also a valid order signature — state the sign)
- "asym"  — directed pair transition asymmetry (fwd vs time-reversed, needs src,dst)
- "mi1"   — mutual information at lag 1
- "fano"  — Fano factor (binary only)
Mirror menu: logistic_ar (binary self-exciting), markov (k-state), semi_markov
(dwell + jump chain), ar1 (scalar), periodic_rate (binary cyclic).

Respond with ONLY a JSON array (no prose, no fence) of exactly 4 candidate objects
— 2 with domain "reasoning-trace", 2 with domain "text-corpus", all
temporal_class "interaction/equality" — with EXACTLY these fields:
{
 "name": "<kebab-case-short-name>",
 "domain": "reasoning-trace" | "text-corpus",
 "temporal_class": "interaction/equality",
 "property": "<one line>",
 "why_not_covered": "<one line>",
 "hypothesis_character": "<2-3 sentences: the cross-position pattern + WHY>",
 "abort_condition": "<the null result that would make it ABORT>",
 "label_signal": "<per-sentence signal, one line — own-content only>",
 "label_kind": "binary" | "ordinal",
 "n_values": 2 | 3 | 4 | 5,
 "judge_instruction": "<EXACT system-prompt text for the Haiku judge; must say to judge ONLY from the sentence's own wording; 2-3 inline examples>",
 "heuristic_crosscheck": "<concrete 10-line keyword/regex check>",
 "composition_risk": "<low|medium|high + one line>",
 "primary_statistic": "acf1" | "asym" | "mi1" | "fano",
 "primary_sign": "positive" | "negative",
 "pair_src_dst": [src, dst] or null,
 "ordered_statistics": ["..."],
 "expected_signature": "<vs the N1/N2/N3 nulls>",
 "chance_oracle": "<chance / oracle for the eventual probe>",
 "arch_predictions": {"per_token_sae": "...", "window_families": "..."},
 "mirror_process": "logistic_ar" | "markov" | "semi_markov" | "ar1" | "periodic_rate",
 "mirror_matched": "<fitted param(s)>",
 "mirror_not_matched": "<deliberately not matched>",
 "gate8_moment": "acf" | "mi" | "fano" | "dwell_cv" | "gap_cv",
 "gate8_idx": 0 or null,
 "gate8_tol_abs": <float>,
 "gate8_rationale": "<why this moment is not set by the fitted params>"
}"""


def build_user() -> str:
    ledger = (HERE / "LEDGER.md").read_text()
    domains = "\n".join(f"- {k}: {v}" for k, v in DOMAINS.items())
    return f"""\
## The two pinned data domains (unchanged from Cycle 1)

{domains}

## Current coverage ledger (Cycle-1 verdicts included — do not re-propose covered
## properties; your 4 cards all target the empty interaction/equality row)

{ledger}

## Null battery (unchanged)
N1 within-doc permutation (kills order, keeps marginal — kills composition);
N2 position-conditional iid (keeps trend); N3 global iid. Ordered statistic must
beat N1 AND N2 beyond sampling noise AND the inter-judge noise floor.

## Task
Exactly 4 cards: 2 reasoning-trace + 2 text-corpus, all interaction/equality,
distinct mechanisms (not four rewordings of one idea). JSON array only."""


def render_card(c: dict, frozen_date: str) -> str:
    from experiments.explorations.synthetic.expansion.hypothesize import render_card as base
    md = base(c, frozen_date)
    pair = f", pair src→dst = {tuple(c['pair_src_dst'])}" if c.get("pair_src_dst") else ""
    gate_block = f"""## 8. Cycle-2 design gates (preregistered in this card)
- **Gate 7 (no-leakage):** the judge instruction above is strictly per-sentence
  (own wording only); labeling runs with **zero context sentences** (ctx=0).
- **Primary statistic:** `{c["primary_statistic"]}` (expected sign:
  {c["primary_sign"]}{pair}).
- **Gate 8 (non-fitted moment):** the `{c["mirror_process"]}` mirror must also
  reproduce `{c["gate8_moment"]}`{f" (lag {c['gate8_idx'] + 1})" if c.get("gate8_idx") is not None else ""}
  on held-out draws within **±{c["gate8_tol_abs"]} absolute** —
  {c["gate8_rationale"]}. Fail ⇒ mirror invalid ⇒ ABORT.

---
_Frozen-by: claude-opus-4-8 via `expansion.hypothesize_c2` (runpod agent, Cycle 2).
Amendments (dated, transparent): none._
"""
    # replace the base card's trailing frozen-by block with the gated one
    md = md[: md.rindex("---")] + gate_block
    return md


def main():
    import datetime

    meter = Meter()
    judge = Judge(meter)
    text = judge.call("think", SYSTEM, build_user(), max_tokens=16000, tag="hypothesize_c2")
    (HERE / "results" / "hypothesize_c2_raw.txt").write_text(text)
    cands = json.loads(re.search(r"\[.*\]", text, re.S).group(0))
    assert len(cands) == N_NEW
    for c in cands:  # this cycle targets exactly one class; tolerate omission
        c.setdefault("temporal_class", "interaction/equality")
    assert all(c["temporal_class"] == "interaction/equality" for c in cands)
    doms = [c["domain"] for c in cands]
    assert doms.count("reasoning-trace") == 2 and doms.count("text-corpus") == 2
    for c in cands:
        assert c["primary_statistic"] in ("acf1", "asym", "mi1", "fano"), c["name"]
        assert c["mirror_process"] in ("logistic_ar", "markov", "semi_markov", "ar1",
                                       "periodic_rate"), c["name"]

    date = datetime.date.today().isoformat()
    for c in cands:
        (HERE / "prereg" / f"{c['name']}.md").write_text(render_card(c, date))
        print(f"[hypothesize_c2] {c['domain']:16} {c['name']:36} primary={c['primary_statistic']}"
              f"({c['primary_sign']}) mirror={c['mirror_process']}")
    (HERE / "results" / "candidates_cycle2.json").write_text(json.dumps(
        {"frozen": date, "generator": "claude-opus-4-8", "cycle": 2,
         "candidates": cands}, indent=2))
    print(f"[spend] ${meter.spent:.3f} of ${meter.cap:.0f}")
    print("[NEXT] run amend_cards_c2, then COMMIT (freeze) before select/calibrate.")


if __name__ == "__main__":
    main()
