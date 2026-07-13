"""Stage 1 — hypothesize (Opus): candidate prereg cards, frozen before data.

Asks the `think` judge (Opus) for N_CANDIDATES candidate temporal properties,
balanced across the LEDGER's under-covered `domain × temporal-class` cells,
as structured JSON. Renders each into a prereg card
(`prereg/<name>.md`, following `prereg_template.md`) plus a machine-readable
`results/candidates.json`. The cards are then COMMITTED (frozen) before any
labeling or measurement — later stages may only abort, never revise.

Selection scoring is deliberately NOT done here (a separate call in
`select.py`), so the generator never scores its own hypotheses.

    .venv/bin/python -m experiments.explorations.synthetic.expansion.hypothesize
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from explorations.synthetic.expansion.client import Judge, Meter

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
N_CANDIDATES = 10  # 5 per domain

DOMAINS = {
    "reasoning-trace": (
        "300 chain-of-thought reasoning traces from DeepSeek-R1-Distill-Llama-8B "
        "over math/logic prompts (10 categories x 30), already sentence-segmented: "
        "25,528 sentences, mean 85 sentences/trace (median 91, max 169). The unit of "
        "time is the sentence index within a trace. Pinned at "
        "results/c7_backtracking/stage_a/sentence_labels.json."),
    "text-corpus": (
        "400 English web documents sampled from HuggingFaceFW/fineweb (sample-10BT, "
        "streamed, seed=0), filtered to 60-200 sentences per document, "
        "sentence-segmented with a pinned regex splitter. The unit of time is the "
        "sentence index within a document. Pinned at "
        "experiments/explorations/synthetic/expansion/data/fineweb_sample.json."),
}

TEMPORAL_CLASSES = {
    "DC-slow-drift": "a state that persists and drifts slowly (heavy dwell, slow decay)",
    "AC-order-sensitive": "depends on the order of recent history, not the marginal",
    "periodic": "rhythmic / cyclic recurrence at a preferred period",
    "bursty/self-exciting": "events cluster; one event raises the near-future rate",
    "interaction/equality": "a cross-position comparison (same/different, matching)",
    "long-memory": "renewal / heavy-tailed recurrence far beyond short lags",
}

SYSTEM = """\
You are the hypothesis generator for a measure->mirror benchmark-expansion loop in an
interpretability research program. The program measures whether temporal properties of
real LM-relevant text carry genuine ORDER structure (vs mere composition), and — only
if they do — freezes them as benchmark specs.

PRIME DIRECTIVE: a sound verdict, never a win. An ABORT (property turns out
non-temporal) is a success. You are NOT rewarded for properties that will PROCEED; you
are rewarded for properties whose verdict, either way, is informative and cleanly
measurable. Never propose a property whose labeler would leak the temporal statistic
(e.g. a label defined by "is this a repeat of an earlier sentence" trivially builds in
order-dependence — that is circularity, not discovery).

The cautionary tale: topic_switching ABORTED because sentence-topic autocorrelation was
82% per-document COMPOSITION (some docs are about X, so neighboring sentences match)
rather than within-document order. Favor labels whose marginal is roughly homogeneous
across documents but whose ORDERING within a document carries the structure; or at
least flag the composition risk and pick statistics that survive it (the N1
within-document permutation null exactly kills composition, so the gate handles it —
but a label that is ~constant within each document will trivially fail and waste a
calibration slot; note when that risk is high).

Each candidate must be:
- labelable per-SENTENCE by a small LLM judge (Claude Haiku) given the sentence plus
  ~3 sentences of preceding context, as a binary 0/1 or small ordinal 0..4 judgment.
  Simple, decidable judgments only; no global-document reasoning.
- checkable by an independent cheap heuristic (keywords / lexicon / punctuation /
  regex) for the cross-validation gate.
- plausibly ORDER-structured within a trace/document (not just doc-level composition).
- distinct from what is already covered (see the coverage ledger).

Respond with ONLY a JSON array (no prose, no code fence) of candidate objects with
EXACTLY these fields:
{
 "name": "<kebab-case-short-name>",
 "domain": "reasoning-trace" | "text-corpus",
 "temporal_class": one of ["DC-slow-drift","AC-order-sensitive","periodic","bursty/self-exciting","interaction/equality","long-memory"],
 "property": "<one line: the temporal phenomenon>",
 "why_not_covered": "<the axis it probes that existing coverage doesn't>",
 "hypothesis_character": "<hypothesised temporal character + WHY, 2-3 sentences>",
 "abort_condition": "<the specific null result that would make it ABORT>",
 "label_signal": "<per-sentence signal, one line>",
 "label_kind": "binary" | "ordinal",
 "n_values": 2 | 3 | 4 | 5,
 "judge_instruction": "<the EXACT system-prompt text for the Haiku judge: define the label crisply with 2-3 inline examples; it must be decidable from one sentence + short context>",
 "heuristic_crosscheck": "<the independent keyword/lexicon/regex check, concrete enough to implement in 10 lines>",
 "composition_risk": "<low|medium|high + one line why>",
 "ordered_statistics": ["<statistic 1>", "..."],
 "expected_signature": "<what the hypothesised class predicts those statistics look like vs the N1/N2/N3 nulls>",
 "chance_oracle": "<chance / oracle for the eventual latent-recovery probe, incl. any provable floor>",
 "arch_predictions": {"per_token_sae": "<prediction + reason>", "window_families": "<prediction + reason>"},
 "mirror_process": one of ["logistic_ar","markov","semi_markov","ar1","periodic_rate"],
 "mirror_matched": "<which parameter(s) the mirror matches>",
 "mirror_not_matched": "<which structure is deliberately NOT matched>"
}"""


def build_user() -> str:
    ledger = (HERE / "LEDGER.md").read_text()
    classes = "\n".join(f"- {k}: {v}" for k, v in TEMPORAL_CLASSES.items())
    domains = "\n".join(f"- {k}: {v}" for k, v in DOMAINS.items())
    return f"""\
## The two data domains (pinned; these are the ONLY data available this cycle)

{domains}

## Temporal classes (the ledger's rows)

{classes}

## Current coverage ledger (prioritize empty / abort-only cells; never propose the
## already-PROCEEDed backtracking property again)

{ledger}

## Null battery every candidate will face
- N1: within-trace/document permutation (kills all order, keeps each doc's marginal —
  kills composition confounds too).
- N2: position-conditional iid resample (keeps any position trend, kills clustering).
- N3: iid from the global marginal.
A property is temporal only if the ordered statistic beats the nulls beyond sampling
noise AND the labeler noise floor (measured by inter-judge agreement).

## Task
Propose exactly {N_CANDIDATES} candidates: 5 with domain "reasoning-trace" and 5 with
domain "text-corpus", together covering at least 4 distinct temporal classes per
domain, prioritizing the ledger's empty cells. Balance ambition with labelability —
a Haiku judge with 3 sentences of context must produce a usable label. Output the
JSON array only."""


def render_card(c: dict, frozen_date: str) -> str:
    return f"""# Preregistration — `{c["name"]}`  (FROZEN {frozen_date}, before any data)

> Cycle-1 card, generated by the `think` judge (claude-opus-4-8) and frozen by
> commit before any labeling or measurement. Later stages may only *abort*,
> never revise it (dated amendments only). One card = one ledger cell.

## 0. Identity
- **Property:** {c["property"]}
- **Ledger cell:** domain = `{c["domain"]}` · temporal-class = `{c["temporal_class"]}`
- **Why it's not already covered:** {c["why_not_covered"]}

## 1. Hypothesis (frozen)
- **Hypothesised temporal character + WHY:** {c["hypothesis_character"]}
- **What would make it ABORT:** {c["abort_condition"]}
- **Composition risk:** {c["composition_risk"]}

## 2. Labeler
- **Signal:** per-sentence `{c["label_kind"]}` (n_values={c["n_values"]}): {c["label_signal"]}
- **Labeler + version:** Claude judge — bulk `claude-haiku-4-5-20251001`,
  adjudication `claude-sonnet-5`. Exact judge instruction (frozen):

```
{c["judge_instruction"]}
```

- **Validation plan:** held-out inter-judge agreement (Sonnet relabels a doc
  sample; raw agreement + Cohen's κ → symmetric-flip noise floor ε̂) + an
  independent heuristic cross-check: {c["heuristic_crosscheck"]}
  The stage-3 effect must survive label flips at ε̂ (the backtracking pattern).

## 3. Data (version-pinned)
- **Source:** {DOMAINS[c["domain"]]}
- **Unit of time:** sentence index within a trace/document. Held-out split at
  the document level for signature stability + mirror validation.

## 4. Statistics + order-destroying null(s)
- **Ordered statistic(s):** {"; ".join(c["ordered_statistics"])}
- **Expected signature:** {c["expected_signature"]}
- **Nulls:** N1 within-doc permutation; N2 position-conditional iid (trend-
  preserving); N3 iid global marginal. **Gate:** ordered must exceed the null
  beyond sampling noise AND the labeler noise floor ε̂, else ABORT.

## 5. Baselines
- {c["chance_oracle"]}

## 6. Predictions per architecture (blind — no arch is run this cycle)
- per-token SAE: {c["arch_predictions"]["per_token_sae"]}
- window families (TXC-pre/-post/Stacked/Spectral): {c["arch_predictions"]["window_families"]}

## 7. Mirror (Appendix B), to be fit only if the gate PASSES
- **Process:** `{c["mirror_process"]}`. **Matched param(s):** {c["mirror_matched"]}.
  **Deliberately NOT matched:** {c["mirror_not_matched"]}

---
_Frozen-by: claude-opus-4-8 via `expansion.hypothesize` (runpod agent).
Amendments (dated, transparent): none._
"""


def main():
    import datetime

    meter = Meter()
    judge = Judge(meter)
    user = build_user()
    print(f"[hypothesize] prompting think-judge ({len(SYSTEM) + len(user)} chars)…")
    text = judge.call("think", SYSTEM, user, max_tokens=16000, tag="hypothesize")
    m = re.search(r"\[.*\]", text, re.S)
    cands = json.loads(m.group(0))
    assert len(cands) == N_CANDIDATES, f"expected {N_CANDIDATES}, got {len(cands)}"
    by_dom = {}
    for c in cands:
        by_dom.setdefault(c["domain"], []).append(c["name"])
    print(f"[hypothesize] {len(cands)} candidates: {json.dumps(by_dom, indent=1)}")
    assert all(len(v) == N_CANDIDATES // 2 for v in by_dom.values()), "domain imbalance"

    date = datetime.date.today().isoformat()
    (HERE / "prereg").mkdir(exist_ok=True)
    for c in cands:
        (HERE / "prereg" / f"{c['name']}.md").write_text(render_card(c, date))
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "candidates.json").write_text(json.dumps(
        {"frozen": date, "generator": "claude-opus-4-8", "candidates": cands}, indent=2))
    print(f"[hypothesize] wrote {len(cands)} cards -> prereg/ + results/candidates.json")
    print(f"[spend] ${meter.spent:.3f} of ${meter.cap:.0f}")
    print("[NEXT] COMMIT these cards (freeze) before running select/calibrate.")


if __name__ == "__main__":
    main()
