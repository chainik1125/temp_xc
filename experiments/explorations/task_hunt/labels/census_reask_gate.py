"""$0 reask GATE CENSUS (mac-c second-source items 3a/3b, entry
20:25; pre-card input — decides the PRIMARY variant before any
freeze).

(a) Which refmark gating substrings fire on the deflected assistant
turn of each reask event, split HARD-REFUSAL vs APOLOGY/PERSONA —
if apology-only events dominate, the hard-refusal-gated variant
pre-registers as primary (the face must measure refusal
persistence, not conversational repair).
(b) The Jaccard length-ratio gate, disclosed: triples passing the
deflection gate + 3-word floors but failing ONLY on J ≥ 0.3 —
count + their content-word length ratios (starvation transparency,
the tretd precedent).

Run: .venv/bin/python -m experiments.explorations.task_hunt.labels.census_reask_gate
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import pull_refmark2k as pull
from . import refmark_lib as rl
from . import wave3_lib as w3

HERE = Path(__file__).resolve().parent

HARD_REFUSAL = ("I cannot", "I can't", "I am unable to", "I'm unable to",
                "I am not able to", "I'm not able to")
APOLOGY_PERSONA = ("I'm sorry", "I am sorry", "I apologize", "As an AI",
                   "As a language model", "As an assistant")
assert set(HARD_REFUSAL) | set(APOLOGY_PERSONA) == set(rl.REFUSAL_SUBSTRINGS)


def gate_hits(text: str):
    low = text.lower()
    hard = tuple(s for s in HARD_REFUSAL if s.lower() in low)
    apol = tuple(s for s in APOLOGY_PERSONA if s.lower() in low)
    return hard, apol


def main():
    convs, _ = pull.load()
    per_string = {s: 0 for s in rl.REFUSAL_SUBSTRINGS}
    n_events = n_hard = n_apology_only = 0
    n_gate_pass = n_jaccard_only_fail = 0
    ratios = []
    j_near_miss = 0
    for msgs in convs:
        ev = w3.reask_events(msgs)
        for i in np.flatnonzero(ev):
            n_events += 1
            hard, apol = gate_hits(msgs[i - 1][1])
            for s in hard + apol:
                per_string[s] += 1
            if hard:
                n_hard += 1
            else:
                n_apology_only += 1
        # (b): triples passing (i)+(ii)+3-word floors, scored on Jaccard
        for i, (role, content) in enumerate(msgs):
            if role == "assistant" or i < 2:
                continue
            r1, c1 = msgs[i - 1]
            r2, c2 = msgs[i - 2]
            if r1 != "assistant" or not rl.is_marker_turn(c1):
                continue
            if r2 == "assistant":
                continue
            w_now, w_orig = w3.content_words(content), w3.content_words(c2)
            if (len(w_now) < w3.REASK_MIN_CONTENT_WORDS
                    or len(w_orig) < w3.REASK_MIN_CONTENT_WORDS):
                continue
            n_gate_pass += 1
            j = w3.jaccard(w_now, w_orig)
            if j < w3.REASK_JACCARD:
                n_jaccard_only_fail += 1
                ratios.append(min(len(w_now), len(w_orig))
                              / max(len(w_now), len(w_orig)))
                if j >= 0.15:
                    j_near_miss += 1
    r = np.array(ratios) if ratios else np.array([0.0])
    out = {
        "events_total": n_events,
        "gate_split": {
            "hard_refusal_events": n_hard,
            "apology_or_persona_only_events": n_apology_only,
            "apology_only_frac": n_apology_only / max(n_events, 1),
            "per_string_event_counts": per_string,
        },
        "jaccard_length_ratio_gate": {
            "triples_passing_deflection_and_word_floors": n_gate_pass,
            "failing_only_on_jaccard": n_jaccard_only_fail,
            "jaccard_only_fail_frac": n_jaccard_only_fail
            / max(n_gate_pass, 1),
            "near_miss_j_in_015_030": j_near_miss,
            "length_ratio_among_jaccard_fails": {
                "median": float(np.median(r)),
                "p25": float(np.quantile(r, 0.25)),
                "p75": float(np.quantile(r, 0.75)),
                "frac_beyond_10_to_3": float((r < 0.3).mean()),
            },
            "note": "J <= min/max ratio, so J >= 0.3 needs counts "
                    "within 10:3 — the disclosed selection rule",
        },
    }
    p = HERE / "reask_gate_census.json"
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(f"-> {p}")


if __name__ == "__main__":
    main()
