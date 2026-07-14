"""Cycle-2 Stage-1 amendments (dated, transparent, committed BEFORE any data).

Two kinds, both mandated by the Cycle-1 review:

1. **Gate-8 preregistration for the four frozen-but-uncalibrated Cycle-1
   cards** (periodic + long-memory). Their § 7 mirror sections predate the
   non-fitted-moment gate; this appends the required moment + abs tolerance,
   chosen design-time (before any labels exist for these candidates):
   - `periodic_rate` mirrors → **Fano(w=10)** within **±0.30** (the fit
     targets the cyclic rate profile, not the dispersion);
   - `semi_markov` mirrors → **indicator ACF(1)** within **±0.05** (the fit
     targets dwell + jump chain, not the autocorrelation directly).

2. **The gate-7 re-exam instruction for `assumption-then-consequence`**
   (currently SPEC* provisional): a STRICTLY per-sentence judge instruction —
   own connectives only, no context clause, labeled with ctx=0 — run as the
   separate record `assumption-consequence-g7`. Survives ⇒ SPEC; collapses ⇒
   ABORT (either is a good outcome).

Also writes `results/amendments_cycle2.json` (machine-readable side) that
`calibrate.py` reads for gate-8 checks and the re-exam labeler.

    .venv/bin/python -m experiments.explorations.synthetic.expansion.amend_cards_c2
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATE = "2026-07-14"

GATE8 = {
    # candidate -> (non-fitted moment key, curve index or None, abs tolerance, rationale)
    "computation-verification-alternation": ("fano", None, 0.30,
        "periodic_rate fits the cyclic rate profile (period + harmonics), not dispersion"),
    "enumeration-cadence": ("fano", None, 0.30,
        "periodic_rate fits the cyclic rate profile (period + harmonics), not dispersion"),
    "goal-restatement-recurrence": ("acf", 0, 0.05,
        "semi_markov fits dwell distributions + jump chain, not the indicator ACF"),
    "pronoun-referent-recurrence": ("acf", 0, 0.05,
        "semi_markov fits dwell distributions + jump chain, not the indicator ACF"),
}

G7_REEXAM_NAME = "assumption-consequence-g7"
G7_JUDGE_INSTRUCTION = (
    "You label one sentence from a math/logic reasoning trace. Judge ONLY from the "
    "sentence's own wording — do NOT use any surrounding context. "
    "0 = neither. "
    "1 = ASSUMPTION/CASE: the sentence itself introduces a supposition, case split, "
    "or premise with its own marker ('suppose', 'assume', 'let', 'if', 'consider the "
    "case', 'WLOG'). "
    "2 = CONSEQUENCE: the sentence itself asserts a derived result with its own "
    "inference marker ('then', 'therefore', 'thus', 'hence', 'it follows', 'this "
    "implies', 'so'). "
    "Examples: 'Suppose x is even.' -> 1. 'Consider the case where n=0.' -> 1. "
    "'Then x squared is also even.' -> 2. 'It follows that the sum is divisible by "
    "4.' -> 2. 'The problem gives three numbers.' -> 0. "
    "If both marker types appear, label by the sentence's MAIN clause.")


def amend_gate8():
    for name, (key, idx, tol, why) in GATE8.items():
        p = HERE / "prereg" / f"{name}.md"
        txt = p.read_text()
        stat = f"`{key}`" + (f" (lag {idx + 1})" if idx is not None else "")
        amendment = (
            f"\n\n## Amendment {DATE} — gate-8 preregistration (Cycle-2 Stage 1, "
            f"before any labeling of this candidate)\n\n"
            f"Per the Cycle-1 review's **non-fitted-moment mirror gate** (README "
            f"guardrail 8), added before calibration: the fitted mirror must also "
            f"reproduce {stat} on held-out real vs synthetic draws within "
            f"**±{tol} absolute** — {why}. Failing this gate ⇒ the mirror is "
            f"invalid ⇒ ABORT (no skeptic pass can rescue it).\n")
        assert "Amendment 2026-07-14 — gate-8" not in txt, f"{name} already amended"
        p.write_text(txt + amendment)
        print(f"[amend] gate-8 prereg -> {name}")


def amend_g7_reexam():
    p = HERE / "prereg" / "assumption-then-consequence.md"
    txt = p.read_text()
    amendment = (
        f"\n\n## Amendment {DATE} — gate-7 re-examination (Cycle-2 rider, "
        f"before re-labeling)\n\n"
        f"The Cycle-1 review downgraded this card's verdict to **SPEC* "
        f"provisional**: the original judge instruction contains a relational "
        f"clause (\"derives a result that FOLLOWS from prior statements\" / "
        f"\"context tells you whether it opens a premise or discharges one\") — "
        f"soft leakage under the new **no-leakage labeler gate** (README "
        f"guardrail 7). Re-examination protocol, frozen before any re-labeling:\n\n"
        f"- Re-label ALL 300 traces under the record name "
        f"`{G7_REEXAM_NAME}` with the STRICTLY per-sentence instruction below, "
        f"and **zero context sentences** (ctx=0) in the chunking.\n"
        f"- Same statistic, same nulls, same gate as the frozen card; mirror "
        f"gate-8 moment preregistered as self-match `acf` (lag 1) within "
        f"**±0.05 absolute**.\n"
        f"- Directed asymmetry **survives** the gate ⇒ upgrade `SPEC*`→`SPEC`; "
        f"**collapses** ⇒ the Cycle-1 signal was labeler leakage ⇒ `ABORT` "
        f"(either outcome is a success — prime directive).\n\n"
        f"Frozen re-exam judge instruction:\n\n```\n{G7_JUDGE_INSTRUCTION}\n```\n")
    assert "gate-7 re-examination" not in txt, "already amended"
    p.write_text(txt + amendment)
    print("[amend] gate-7 re-exam -> assumption-then-consequence")


def main():
    amend_gate8()
    amend_g7_reexam()
    blob = {
        "date": DATE,
        "gate8": {n: {"moment": k, "idx": i, "tol_abs": t, "rationale": w}
                  for n, (k, i, t, w) in GATE8.items()},
        "g7_reexam": {"name": G7_REEXAM_NAME,
                      "base_card": "assumption-then-consequence",
                      "judge_instruction": G7_JUDGE_INSTRUCTION,
                      "ctx": 0,
                      "gate8": {"moment": "acf", "idx": 0, "tol_abs": 0.05}},
    }
    (HERE / "results" / "amendments_cycle2.json").write_text(json.dumps(blob, indent=2))
    print("[amend] -> results/amendments_cycle2.json")


if __name__ == "__main__":
    main()
