"""Cycle-3 Stage 1 — hypothesize categorical interaction/equality cards (Opus).

interaction/equality is empty again after the C2 review re-filed
`self-reference-echo` to bursty/self-exciting: BINARY "refers-back" labels
keep measuring as self-excitation. This cycle uses the review's gate-7-clean
recipe (now in the README): a **categorical per-sentence content label**
(which sub-goal / operation / claim-topic the sentence is about — assignable
from the sentence alone), with the **equality-adjacency `[c_t = c_{t-1}]`**
as the measured statistic — the categorical self-match ACF, exactly how the
synthetic changepoint mode grounds equality-pattern latents.

Constraints baked into the prompt (not left to the generator):
- label_kind = categorical, 3–6 classes, every class decidable from the single
  sentence (ctx=0, gate 7);
- primary statistic FIXED = self-match `acf1`, sign positive (match-runs);
- mirror ∈ {markov, semi_markov} (the categorical menu rows);
- gate-8 tolerance FIXED by the uniform C3 rule (±20% relative, see
  `amend_cards_c3.py`) — the generator names only the moment + rationale.

Renders cards to `prereg/` + `results/candidates_cycle3.json`.
COMMIT (freeze) before any labeling.

    .venv/bin/python -m experiments.explorations.synthetic.expansion.hypothesize_c3
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from experiments.explorations.synthetic.expansion.hypothesize import DOMAINS
from explorations.synthetic.expansion.client import Judge, Meter

HERE = Path(__file__).resolve().parent
N_NEW = 4
TOL_REL = 0.20

SYSTEM = """\
You are the hypothesis generator for a measure->mirror benchmark-expansion loop
(Cycle 3). PRIME DIRECTIVE: a sound verdict, never a win — an ABORT is a success;
never build the temporal statistic into the label.

Your task: candidates for the ONE still-empty temporal class, **interaction/equality**
— the grounded analogue of an abstract changepoint bench whose latents are
equality-patterns over positions. TWO prior binary attempts both failed to land in
this class: a binary "refers-back/echo" label measured as pure self-excitation
(re-filed to bursty), and a binary alternation card's preregistered negative sign
was falsified (real events cluster). The review's diagnosis: binary event labels
collapse cross-position structure into event clustering.

THE MANDATED RECIPE (gate-7-clean, mirrors how the synthetic changepoint mode
works): a CATEGORICAL per-sentence CONTENT label c_t with 3-6 classes — which
sub-goal / operation-type / discourse-move / claim-topic-type the sentence
enacts, assignable from the sentence's OWN wording alone — and the equality lives
ONLY in the measured statistic: the equality-adjacency [c_t == c_{t-1}], i.e. the
categorical self-match ACF at lag 1 (segments of constant label = the changepoint
structure; matching is BETWEEN positions, the label itself never references
neighbours). The interesting real-text hypothesis is that sentences organize into
same-label RUNS (phases/segments) whose boundaries only position-comparing
architectures can expose.

HARD DESIGN GATES (cards violating them are dead on arrival):
- Gate 7 — no-leakage labeler: every class definition MUST be decidable from the
  single sentence (its wording, structure, markers). NEVER 'same topic as before',
  'continues the previous step', 'answers the preceding'. Labels are judged with
  ZERO context sentences.
- Gate 8 — non-fitted-moment mirror: name >=1 statistic your chosen mirror is NOT
  fit to. The tolerance is FIXED program-wide this cycle (±20% of the held-out real
  magnitude) — you provide only the moment + a rationale for why it is not set by
  the fitted parameters.

Labeler constraints: categorical, n_values 3-6 INCLUDING a background/other class
(class 0); every class decidable by a Claude Haiku judge from the single sentence;
an independent ~10-line heuristic (keywords/regex) must map sentences to the SAME
class ids as cross-check; avoid marginals where one class exceeds ~75% or where
fewer than 3 classes reach ~3% (a near-constant stream has no adjacency structure
to measure and wastes the calibration slot).

FIXED measurement design (not yours to choose):
- primary_statistic = "acf1" — the categorical self-match ACF(1), the equality-
  adjacency [c_t == c_{t-1}] excess over the iid match rate; expected sign positive.
- Nulls: N1 within-doc permutation / N2 position-conditional iid / N3 global iid.
Mirror menu for categorical streams: "markov" (k-state chain — matches the
adjacency/transition structure directly) or "semi_markov" (empirical dwell + jump
chain — matches run-length structure). gate8_moment options: "acf" with gate8_idx 3
(self-match ACF at lag 4 — set by the chain's mixing, not directly fit), "mi" with
gate8_idx 1 (MI at lag 2), or "dwell_cv" with gate8_idx null (run-length dispersion;
NOT valid for semi_markov, which fits dwell directly).

Respond with ONLY a JSON array (no prose, no fence) of exactly 4 candidate objects
— 2 with domain "reasoning-trace", 2 with domain "text-corpus", all
temporal_class "interaction/equality", DISTINCT mechanisms — with EXACTLY these fields:
{
 "name": "<kebab-case-short-name>",
 "domain": "reasoning-trace" | "text-corpus",
 "temporal_class": "interaction/equality",
 "property": "<one line>",
 "why_not_covered": "<one line — incl. how it differs from the two failed binary attempts>",
 "hypothesis_character": "<2-3 sentences: the run/segment structure + WHY>",
 "abort_condition": "<the null result that would make it ABORT>",
 "label_signal": "<the categorical content label, one line — own-content only>",
 "label_kind": "categorical",
 "n_values": 3 | 4 | 5 | 6,
 "class_names": ["<class 0 = background/other>", "..."],
 "judge_instruction": "<EXACT system-prompt text for the Haiku judge; must say to judge ONLY from the sentence's own wording; define every class id with 1-2 inline examples each>",
 "heuristic_crosscheck": "<concrete ~10-line keyword/regex mapping to the same class ids>",
 "composition_risk": "<low|medium|high + one line>",
 "primary_statistic": "acf1",
 "primary_sign": "positive",
 "pair_src_dst": null,
 "ordered_statistics": ["self-match ACF(1) (equality-adjacency)", "..."],
 "expected_signature": "<vs the N1/N2/N3 nulls>",
 "chance_oracle": "<chance / oracle for the eventual probe>",
 "arch_predictions": {"per_token_sae": "...", "window_families": "..."},
 "mirror_process": "markov" | "semi_markov",
 "mirror_matched": "<fitted param(s)>",
 "mirror_not_matched": "<deliberately not matched>",
 "gate8_moment": "acf" | "mi" | "dwell_cv",
 "gate8_idx": 3 | 1 | null,
 "gate8_rationale": "<why this moment is not set by the fitted params>"
}"""


def build_user() -> str:
    ledger = (HERE / "LEDGER.md").read_text()
    domains = "\n".join(f"- {k}: {v}" for k, v in DOMAINS.items())
    return f"""\
## The two pinned data domains (unchanged since Cycle 1)

{domains}

## Current coverage ledger (post-C2-review; note the two failed binary
## interaction/equality attempts and the measured-class re-filing rule — a
## candidate that MEASURES as plain self-excitation will be re-filed out of
## this class, so design for genuine multi-class segment/run structure)

{ledger}

## Task
Exactly 4 cards: 2 reasoning-trace + 2 text-corpus, all interaction/equality via
the categorical-content-label recipe, distinct mechanisms (not four rewordings).
JSON array only."""


def render_card(c: dict, frozen_date: str) -> str:
    from experiments.explorations.synthetic.expansion.hypothesize import render_card as base
    md = base(c, frozen_date)
    classes = "; ".join(f"`{i}` = {n}" for i, n in enumerate(c["class_names"]))
    gate_block = f"""## 8. Cycle-3 design gates (preregistered in this card)
- **Gate 7 (no-leakage):** categorical per-sentence CONTENT label, every class
  decidable from the sentence's own wording; labeling runs with **zero context
  sentences** (ctx=0). Classes: {classes}.
- **Primary statistic (fixed by the C3 recipe):** the equality-adjacency
  `[c_t = c_{{t-1}}]` — categorical self-match `acf1`, expected sign positive.
  The equality lives in the statistic, never in the label.
- **Gate 8 (non-fitted moment):** the `{c["mirror_process"]}` mirror must also
  reproduce `{c["gate8_moment"]}`{f" (lag {c['gate8_idx'] + 1})" if c.get("gate8_idx") is not None else ""}
  on held-out draws within the **uniform C3 tolerance: ±{int(TOL_REL * 100)}% of the
  held-out real magnitude** (`amend_cards_c3.py` rule; floors apply) —
  {c["gate8_rationale"]}. Fail ⇒ mirror invalid ⇒ ABORT.
- **Measured-class filing:** if calibration measures this stream as something
  other than multi-class run/segment equality structure (e.g. one class
  dominates and the signature is binary-like self-excitation), the ledger cell
  is assigned by the MEASURED class, not this card's proposal.

---
_Frozen-by: claude-opus-4-8 via `expansion.hypothesize_c3` (runpod agent, Cycle 3).
Amendments (dated, transparent): none._
"""
    md = md[: md.rindex("---")] + gate_block
    return md


def main():
    import datetime

    meter = Meter()
    judge = Judge(meter)
    text = judge.call("think", SYSTEM, build_user(), max_tokens=20000, tag="hypothesize_c3")
    (HERE / "results" / "hypothesize_c3_raw.txt").write_text(text)
    cands = json.loads(re.search(r"\[.*\]", text, re.S).group(0))
    assert len(cands) == N_NEW
    doms = [c["domain"] for c in cands]
    assert doms.count("reasoning-trace") == 2 and doms.count("text-corpus") == 2
    for c in cands:
        assert c["temporal_class"] == "interaction/equality", c["name"]
        assert c["label_kind"] == "categorical" and 3 <= c["n_values"] <= 6, c["name"]
        assert len(c["class_names"]) == c["n_values"], c["name"]
        assert c["primary_statistic"] == "acf1" and c["primary_sign"] == "positive"
        assert c["mirror_process"] in ("markov", "semi_markov"), c["name"]
        assert c["gate8_moment"] in ("acf", "mi", "dwell_cv"), c["name"]
        assert not (c["mirror_process"] == "semi_markov"
                    and c["gate8_moment"] == "dwell_cv"), c["name"]

    date = datetime.date.today().isoformat()
    for c in cands:
        (HERE / "prereg" / f"{c['name']}.md").write_text(render_card(c, date))
        print(f"[hypothesize_c3] {c['domain']:16} {c['name']:36} k={c['n_values']} "
              f"mirror={c['mirror_process']} g8={c['gate8_moment']}")
    (HERE / "results" / "candidates_cycle3.json").write_text(json.dumps(
        {"frozen": date, "generator": "claude-opus-4-8", "cycle": 3,
         "candidates": cands}, indent=2))
    print(f"[spend] ${meter.spent:.3f} of ${meter.cap:.0f}")
    print("[NEXT] run select_c3, then COMMIT (freeze) before calibrate.")


if __name__ == "__main__":
    main()
