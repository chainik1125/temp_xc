"""C7 close-branch skeptic — the expansion rubric over the C7 card +
battery + close summary (frozen card § fork: the skeptic fires on
whichever branch resolves; the SPEC branch uses the in-pipeline r4
skeptic, THIS runner covers the close branch, including an r4 FAIL).

Judgment on `claude-fable-5` (ROLES["think"]); raw verdict persisted
pre-parse under ``records/proof-operation-phase-runs-c7-close/`` and
NEVER re-rolled — the runner refuses to run if a raw verdict exists.
Committed before first execution.

    export ANTHROPIC_API_KEY=$(cat /workspace/.tokens/anthropic_key)
    .venv/bin/python -m experiments.explorations.synthetic.expansion.skeptic_c7_close
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from explorations.synthetic.expansion.client import Judge, Meter
from experiments.explorations.synthetic.expansion.calibrate import skeptic_pass

HERE = Path(__file__).resolve().parent
NAME = "proof-operation-phase-runs-c7-close"


def main():
    raw = HERE / "records" / NAME / "skeptic_raw.txt"
    if raw.exists():
        sys.exit(f"[skeptic-c7] {raw} exists — verdicts are never re-rolled; "
                 "read the cached verdict instead.")
    battery = json.loads(
        (HERE / "results" / "estimator_battery_c7.json").read_text())
    if battery["fork"]["branch"] != "close":
        sys.exit("[skeptic-c7] fork branch is not 'close' — the r4 pipeline "
                 "skeptic covers the SPEC branch; refusing to run.")
    card = (HERE / "prereg" /
            "estimator-card-c7-monotone-extraction.md").read_text()
    summary = {
        "what_is_judged": (
            "The C7 CLOSE of the reasoning int/eq cell (NEGATIVE at this "
            "corpus resolution) under the frozen card's pre-specified fork "
            "— not a PROCEED. Kill items apply to the close's soundness: "
            "was any gate, tolerance, or rule bent post-hoc; does the "
            "evidence actually support 'structure real but unextractable "
            "at 287 docs x ~85 sentences'; is the monotonicity pre-check "
            "result trustworthy."),
        "battery": battery,
        "c6_context": {
            "verdict": "empty passing set (reviewed & APPROVED)",
            "findings": [
                "shrink-then-deconvolve non-monotone (deconvolution "
                "re-amplifies; null-clean extraction window empty on real "
                "material)",
                "quantile deflation leaks through tails",
                "both candidates cancel weak-regime winner's curse "
                "(raw +34% -> <=3%)"],
            "record": "results/estimator_battery_c6.md"},
        "structure_evidence_unchallenged": (
            "real-vs-permuted ACF(4) gap 0.056 (C5, model-independent); "
            "the close does NOT claim the structure is absent"),
    }
    meter = Meter()
    judge = Judge(meter)
    out = skeptic_pass(judge, NAME, card, summary)
    (HERE / "records" / NAME / "skeptic.json").write_text(
        json.dumps(out, indent=1, default=float))
    kills = [k for k, v in out.items() if isinstance(v, dict) and v.get("kill")]
    print(f"[skeptic-c7] verdict parsed; kills: {kills or 'NONE'}  "
          f"spend=${meter.spent:.2f}")


if __name__ == "__main__":
    main()
