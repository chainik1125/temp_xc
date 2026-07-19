"""Cycle-3 Stage-1 amendments (dated, transparent, committed BEFORE any data).

All mandated by the Cycle-2 review (briefing `grounded-benchmark-expansion-cycle3`):

1. **The uniform C3 gate-8 tolerance rule.** C2 preregistered raw-absolute
   tolerances and mis-scaled one (list-item-parallelism died at 4% relative
   error). Cycle 3 preregisters ONE uniform rule for every gate-8 check it
   freezes: **±20% of the held-out real magnitude**, with tiny per-moment-type
   absolute floors guarding degenerate near-zero magnitudes
   (`tol_eff = max(0.20·|real|, floor)`). Chosen before any C3 fit or label.

2. **Two re-freezes — real signals killed only on mirror methodology:**
   - `list-item-parallelism` → record `list-item-parallelism-r2`: mirror
     unchanged (`logistic_ar`), gate-8 moment unchanged (Fano), tolerance
     re-preregistered under the uniform rule. C2's cached labels + labeler
     validation are REUSED (the labeler is unchanged — relabeling would spend
     with zero information gain); the C2 ABORT stands as the record of the C2
     gate.
   - `computation-verification-alternation` → record
     `computation-verification-r2`: mirror swapped `periodic_rate` →
     **`periodic_hawkes`** (the new Appendix-B hybrid: C2 measured the events
     as rhythmic AND bursty — spec_peak 3.84 real, Fano 2.29 vs the pure
     periodic mirror's 0.87). Gate-8 moment stays Fano (window-count
     dispersion is derived from, never directly fit by, the logistic
     parameters), uniform tolerance. Cached labels reused.

3. **Tolerance conversion for the two still-frozen C1 cards**
   (`enumeration-cadence`, `goal-restatement-recurrence`) — their C2 gate-8
   amendments used raw absolutes; converted to the uniform relative rule.
   Still blind: neither card has ever been labeled.

4. **The hedging mirror re-fit prereg (rider).** After ar1+trend and the
   preregistered semi-Markov attempt both failed ACF(2) (real long-memory
   plateau ~0.13 through lag 8), the re-fit uses the new **`hier_ar1`** menu
   extension (per-sequence latent level + pooled trend + within-seq AR(1)).
   Preregistered BEFORE fitting: gate-8 = ACF(2) AND ACF(4) (both non-fitted:
   the fit sees lag-1 pairs + per-sequence means only), each within the
   uniform ±20% relative tolerance; matched-moment sanity check ACF(1) within
   ±0.05 abs. PASS ⇒ `SPEC*`→`SPEC` via a dated spec amendment; FAIL ⇒ stays
   `SPEC*`, mirror INVALID.

Writes `results/amendments_cycle3.json` (machine side read by `calibrate.py`
and `mirror_upgrade_hedging_c3.py`).

    .venv/bin/python -m experiments.explorations.synthetic.expansion.amend_cards_c3
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATE = "2026-07-19"

TOL_REL = 0.20
FLOORS = {"acf": 0.01, "mi": 0.003, "fano": 0.05, "dwell_cv": 0.05,
          "gap_cv": 0.05, "spec_peak": 0.05}

RULE_TEXT = (
    f"the uniform Cycle-3 rule: within **±{int(TOL_REL * 100)}% of the "
    f"held-out real magnitude** (`tol_eff = max({TOL_REL}·|real|, floor)`, "
    f"floors: acf 0.01, mi 0.003, dispersion moments 0.05) — preregistered "
    f"before any Cycle-3 labeling or fitting, per the C2-review lesson that "
    f"raw-absolute tolerances mis-scale when the statistic's magnitude is "
    f"unknown")


def _append(card: str, text: str, guard: str):
    p = HERE / "prereg" / f"{card}.md"
    txt = p.read_text()
    assert guard not in txt, f"{card} already amended for C3"
    p.write_text(txt + text)
    print(f"[amend-c3] -> {card}")


def amend_list_item():
    _append("list-item-parallelism", f"""

## Amendment {DATE} — Cycle-3 re-freeze (before any C3 calibration)

The C2 calibration ABORTED on gate-8 alone: Fano |err| 0.163 vs the frozen
±0.15 **absolute** tolerance — a 4.2% relative miss on a statistic of
magnitude ≈3.9, while every other held-out moment matched and the primary
signal was C2's strongest (ACF(1)=0.52 ≫ N1 hi 0.21, κ=0.64). The C2 review
ruled the tolerance mis-scaled, not the mirror wrong, and mandated a
re-freeze. Preregistered now, before any C3 run:

- **Record name:** `list-item-parallelism-r2`. The C2 ABORT record stands
  untouched as the verdict under the C2-frozen tolerance.
- **Mirror unchanged:** `logistic_ar` (K=8). **Gate-8 moment unchanged:**
  Fano(w=10). **Tolerance re-preregistered** under {RULE_TEXT}.
- **Labels + labeler validation REUSED from the C2 record** (identical frozen
  judge instruction, identical pinned corpus — relabeling would re-spend for
  zero information). Everything downstream (signature, nulls, mirror fit,
  skeptic) runs fresh.
""", "Cycle-3 re-freeze")


def amend_comp_verif():
    _append("computation-verification-alternation", f"""

## Amendment {DATE} — Cycle-3 re-freeze with hybrid mirror (before any C3 calibration)

The C2 calibration confirmed the primary signal is REAL (spec_peak 3.84 ≫
null ≤1.18, survives the noise floor) but ABORTED on gate-8: the events are
ALSO bursty (held-out Fano 2.29) and the pure `periodic_rate` mirror can only
generate near-Poisson dispersion (Fano 0.87). The C2 review's systemic
finding: the menu lacked a process for phenomena that are rhythmic AND
clustered at once, and mandated this re-freeze. Preregistered now:

- **Record name:** `computation-verification-r2`. The C2 ABORT stands as the
  verdict for the periodic-only mirror.
- **Mirror swapped** `periodic_rate` → **`periodic_hawkes`** (Appendix-B
  Cycle-3 extension: logit P(event) = periodic base (cyclogram period + one
  harmonic) + K=8-lag self-excitation kernel, fit jointly by logistic
  regression). **Matched:** period, phase profile, excitation kernel.
  **Deliberately NOT matched:** window-count dispersion (Fano), inter-event
  gap distribution, and the content of what gets verified.
- **Gate-8 moment:** Fano(w=10) — derived overdispersion, never directly fit —
  within {RULE_TEXT}.
- **Labels + labeler validation REUSED from the C2 record** (identical frozen
  judge instruction, ctx=3, identical pinned traces). Signature, nulls,
  mirror fit and skeptic run fresh.
""", "Cycle-3 re-freeze")


def amend_c1_frozen_tols():
    for card, stat in [("enumeration-cadence", "Fano(w=10)"),
                       ("goal-restatement-recurrence", "indicator ACF(1)")]:
        _append(card, f"""

## Amendment {DATE} — Cycle-3 tolerance conversion (still before any labeling)

This card's {stat} gate-8 tolerance was preregistered in C2 as a raw
absolute — exactly the mis-scaling the C2 review flagged (a fixed absolute is
meaningless until the statistic's magnitude is known). Converted, still blind
(this candidate has never been labeled), to {RULE_TEXT}. Moment and mirror
unchanged.
""", "Cycle-3 tolerance conversion")


def amend_hedging():
    _append("uncertainty-hedging-drift", f"""

## Amendment {DATE} — second mirror re-fit prereg: `hier_ar1` (Cycle-3 rider, before fitting)

Gate-8 history: the C1-blessed `ar1+trend` mirror FAILED the retroactive C2
check (ACF(2) |err| 0.071 > 0.05 abs), and the C2-preregistered `semi_markov`
attempt FAILED the same moment identically (0.071) — the real stream's ACF is
a long-memory **plateau** (≈0.13–0.15 through lag 8) no single-timescale menu
process can hold up. The C2 review mandated a hierarchical extension.
Preregistered now, before any fit:

- **Process:** `hier_ar1` (Appendix-B Cycle-3 extension): pooled position
  trend + one empirical latent level per document + within-document AR(1).
  **Matched:** the trend, the per-document level distribution, lag-1
  persistence. **Deliberately NOT matched:** any content coupling, and every
  ACF lag ≥ 2 (the plateau must EMERGE from the level variance).
- **Gate-8 (non-fitted moments):** held-out real vs synthetic **ACF(2) AND
  ACF(4)**, each within {RULE_TEXT}. Both must pass.
- **Matched-moment sanity check (not a gate-8 substitute):** ACF(1) within
  ±0.05 absolute.
- **Fresh 70/30 document split** (seed 2000; the semi-Markov attempt used
  1000). Cached C1 labels; no API calls.
- **Outcome rule:** PASS ⇒ upgrade `SPEC*`→`SPEC` via a dated amendment to
  `synthetic/hedging_drift/bench_spec.md` swapping the canonical mirror;
  FAIL ⇒ stays `SPEC*`, mirror recorded INVALID (a bespoke process would then
  need the README's written justification in a future cycle).
""", "second mirror re-fit prereg")


def main():
    amend_list_item()
    amend_comp_verif()
    amend_c1_frozen_tols()
    amend_hedging()
    blob = {
        "date": DATE,
        "tol_rule": {"rel": TOL_REL, "floors": FLOORS,
                     "eff": "tol_eff = max(rel*|real_heldout|, floor[moment])"},
        "refreeze": {
            "list-item-parallelism-r2": {
                "base_card": "list-item-parallelism",
                "labels_from": "list-item-parallelism",
                "mirror": "logistic_ar",
                "gate8": {"moment": "fano", "idx": None, "tol_rel": TOL_REL}},
            "computation-verification-r2": {
                "base_card": "computation-verification-alternation",
                "labels_from": "computation-verification-alternation",
                "mirror": "periodic_hawkes",
                "gate8": {"moment": "fano", "idx": None, "tol_rel": TOL_REL}},
        },
        "c1_frozen_tol_conversion": {
            "enumeration-cadence": {"moment": "fano", "idx": None, "tol_rel": TOL_REL},
            "goal-restatement-recurrence": {"moment": "acf", "idx": 0, "tol_rel": TOL_REL},
        },
        "hedging_refit": {
            "record": "uncertainty-hedging-drift",
            "mirror": "hier_ar1", "split_seed": 2000,
            "gate8": [{"moment": "acf", "idx": 1, "tol_rel": TOL_REL},
                      {"moment": "acf", "idx": 3, "tol_rel": TOL_REL}],
            "matched_check": {"moment": "acf", "idx": 0, "tol_abs": 0.05}},
    }
    (HERE / "results" / "amendments_cycle3.json").write_text(json.dumps(blob, indent=2))
    print("[amend-c3] -> results/amendments_cycle3.json")


if __name__ == "__main__":
    main()
