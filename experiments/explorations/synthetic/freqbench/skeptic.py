"""FreqBench proof-skeptic — the LOOP.md fixed kill-rubric on Fable 5.

Five items, ANY kill ⇒ ABORT (LOOP.md "The skeptic"): a_proof_circularity ·
b_triviality · c_relevance · d_redundancy · e_substrate. Run per card AFTER
T1/T2 pass, BEFORE § 8-gated grids are spent.

Ops rules (the C4 lesson): the RAW model response is persisted to
``results/skeptic_raw_<card>.txt`` BEFORE any parsing; the parsed verdict is
rubric-key-validated; a parse failure is repaired from the raw text, never
re-rolled. Spend goes to the freqbench meter (``results/spend.json`` +
``spend_log.jsonl``) — NOT the expansion meter.

    .venv/bin/python -m experiments.explorations.synthetic.freqbench.skeptic FB-2 FB-3
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from explorations.synthetic.expansion.client import Judge, Meter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RES = HERE / "results"

RUBRIC = ["a_proof_circularity", "b_triviality", "c_relevance",
          "d_redundancy", "e_substrate"]

SYSTEM = """You are the FreqBench proof-skeptic — the adversarial gate of a
theorem-first benchmark-generator loop. Your ONLY job is to find reasons to
KILL the card. The prime directive of the program is "a sound verdict, never
a win": an ABORT is a success; a benchmark that survives you must deserve it.

You receive: the frozen axis-point card, the T1/§8 gating results, the T2
non-triviality battery results, and any documented gate-check amendments made
after freeze. Scrutinize the amendments especially hard: they are the exact
place where an agent could have tuned a gate to manufacture a PROCEED.

Evaluate the five kill-items:

- a_proof_circularity: does any "proof" assume its conclusion, or prove a
  DIFFERENT task/parameterization than the one actually built? Are the
  numerical discharges on the actual generator at the frozen parameters?
- b_triviality: does a symmetry, relabeling, bag-of-symbols, or memorization
  route survive the T2 battery? Could the claimed regime-3 structure be read
  by a trivial route the controls missed?
- c_relevance: does any real phenomenon occupy the target coordinates (cite
  the program's own measurements where given), OR is the card honestly marked
  `spanning` with a defensible research reason? A vague appeal is a kill.
- d_redundancy: does an existing bench in the registry already discriminate
  architectures at these coordinates?
- e_substrate: any hidden deviation from the shared panel/conventions
  (capacity anchoring, k_pos matching, eval windowing, canonical runner)?
  Any gate amendment that functions as tolerance-shopping?

Return STRICT JSON, nothing else, in this schema:
{"a_proof_circularity": {"verdict": "pass"|"kill", "reason": "..."},
 "b_triviality": {...}, "c_relevance": {...}, "d_redundancy": {...},
 "e_substrate": {...},
 "overall": "PROCEED"|"ABORT",
 "notes": "anything the record should carry"}
"overall" must be ABORT iff any item is kill."""


def build_user(card: str) -> str:
    card_text = (HERE / "cards" / f"{card}.md").read_text()
    if card == "FB-1":
        gate = (ROOT / "experiments/explorations/synthetic/phasepair/results/"
                "phasepair_gating_stats.json").read_text()
        t2 = (ROOT / "experiments/explorations/synthetic/phasepair/results/"
              "phasepair_t2_stats.json").read_text()
        amendments = """DOCUMENTED POST-FREEZE GATE AMENDMENTS (scrutinize):
1. The raw-linear floor check was made ONE-SIDED: the T∈{4,8} raw-window
   6-class linear probes scored 0.112-0.115 — BELOW chance 0.167 — which is
   a degenerate-multiclass-probe artifact, not linear access (access pushes
   ABOVE chance). Below-chance values are recorded, not gated. The SIGN
   floors (the primary latent) were immaculate two-sided (max dev 0.012).
   Question: honest precision-fix or tolerance-shopping?
ALSO NOTE (T2 finding, recorded in the battery): the reflection a↦-a plus
an orthogonal column-flip of R exchanges the sign classes exactly — the
sign latent is chirality w.r.t. the realized embedding (well-defined per
seed, not poolable across seeds). Judge whether this undermines the task
(b_triviality) or is a benign per-seed convention like the phase nuisance."""
    elif card == "FB-2":
        gate = (ROOT / "experiments/explorations/synthetic/multilane/results/"
                "multilane_gating_stats.json").read_text()
        t2 = (ROOT / "experiments/explorations/synthetic/multilane/results/"
              "multilane_t2_stats.json").read_text()
        amendments = """DOCUMENTED POST-FREEZE GATE AMENDMENTS (scrutinize):
1. The § 8 info-presence check originally required a generic MLP(256) on raw
   ordered tiles to clear chance+0.20; it read 0.173 while the periodogram
   ORACLE on the same tiles read 0.906. The check was re-keyed to the oracle
   witness (README equality-variant: "nonlinear/oracle readout"), with the
   MLP number kept as a recorded datum. Question: is that the correct
   reading of information-presence, or tolerance-shopping?"""
    elif card == "FB-4":
        gate = (ROOT / "experiments/explorations/synthetic/rotated_multilane/"
                "results/rotated_multilane_gating_stats.json").read_text()
        t2 = (ROOT / "experiments/explorations/synthetic/rotated_multilane/"
              "results/rotated_multilane_t2_stats.json").read_text()
        amendments = """DOCUMENTED POST-FREEZE GATE AMENDMENTS (scrutinize):
1. The T1 window-concat linear floor first ran against FB-2's ABSOLUTE bar
   (chance+0.02) and FAILED at 0.115-0.137 (first-pass stats preserved in
   commit d9e00a5b). Diagnosis: the numerically IDENTICAL values appear on
   the unrotated base data under the same probe (a linear probe is exactly
   invariant under an orthogonal feature map) — a substrate-level variance
   leak this probe's sample size surfaces equally on FB-2 (P2 bounds
   class-conditional MEANS only; FB-2's own gating probe read ~=chance).
   The check was re-keyed to the card's actual obligation — rotation-
   INVARIANCE of the floor (paired identical-probe gap <= 0.005) — with the
   absolutes recorded on both sides. Question: honest re-key or tolerance-
   shopping? Note the datum it leaves for the program: the FB-2 raw-window-
   linear "at chance" reading is probe-protocol-conditional at the margins.
SPECIAL SITUATION (judge BOTH directions): this card was frozen with an
explicit § 3 absorption obligation predicting the rotation knob is INERT
(the base embedding is Haar-random and re-drawn per data seed, so Q·P is
distributionally identical to P). The builder's own pre-registered T2
decision rule therefore expects verdict ABORT_T2_SYMMETRY — an ABORT here
is the process working, not a loss. Your job: (a) if you find the
absorption reasoning or its empirical arms UNSOUND, kill items accordingly
and say the card should instead PROCEED to grids; (b) if the absorption
holds, confirm the kill (b_triviality/d_redundancy) so the ABORT is
double-witnessed. Also assess whether the salvage note (the live knob for
the intended basis-alignment question is TEMPORAL, a candidate FB-5 left
un-frozen for review) is the right disposition."""
    else:
        gate = (ROOT / "experiments/explorations/synthetic/colored_sources/"
                "results/colored_gating_stats.json").read_text()
        t2 = (ROOT / "experiments/explorations/synthetic/colored_sources/"
              "results/colored_t2_stats.json").read_text()
        amendments = """DOCUMENTED POST-FREEZE GATE AMENDMENTS (scrutinize):
1. Floor checks originally compared eigen-estimators to an iid-GAUSSIAN
   candidate null; eigenbases are orthonormal and score higher by geometry
   alone (0.181 vs 0.170). Re-keyed to the orthonormal null; an absolute
   floor bar |rec_adj| <= 0.05 was set (vs the 0.96 ceiling), and a
   SYSTEMATIC stream-leakage of +0.011 rec_sq (8 seeds) was measured and
   recorded (CS-1 strictly assumes iid draws; the correlated stream tilts
   C0's finite-sample eigenvectors).
2. The card's § 3 bag-of-symbols line predicted pooled codes at the floor;
   the gating math shows pooling/shuffling only DILUTES C_D (pooled eig
   rec_adj +0.69) — the bench's true null is window TRUNCATION (W <= D).
   Recorded as a dated precision-amendment; the CS-1 floor is untouched.
3. The T2 untrained-floor check was made one-sided (positive artifacts
   only): untrained spectral scores NEGATIVE adj (−0.07..−0.09) because
   band-limited kernels have correlated time-slices (effective candidate
   count < d_sae*T) — conservative AGAINST spectral, recorded as a metric
   property. Question for each: honest precision-fix or gate-tuning?"""
    reg = (ROOT / "experiments/explorations/synthetic/registry.py").read_text()
    reg_slice = reg[reg.index("BENCHES:"):reg.index("# ── Architectures")]
    return (f"=== FROZEN CARD {card} ===\n{card_text}\n\n"
            f"=== T1/§8 GATING RESULTS ===\n{gate}\n\n"
            f"=== T2 BATTERY RESULTS ===\n{t2}\n\n"
            f"=== {amendments}\n\n"
            f"=== REGISTRY (current benches, for d_redundancy) ===\n"
            f"{reg_slice}\n")


def parse_verdict(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise ValueError("no JSON object found in raw response")
    v = json.loads(m.group(0))
    for k in RUBRIC:
        assert k in v and v[k].get("verdict") in ("pass", "kill"), f"bad item {k}"
    assert v.get("overall") in ("PROCEED", "ABORT")
    kills = [k for k in RUBRIC if v[k]["verdict"] == "kill"]
    assert (v["overall"] == "ABORT") == bool(kills), "overall inconsistent"
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cards", nargs="+", choices=["FB-1", "FB-2", "FB-3", "FB-4"])
    args = ap.parse_args()

    meter = Meter(path=RES / "spend.json")
    judge = Judge(meter)
    for card in args.cards:
        user = build_user(card)
        raw = judge.call("think", SYSTEM, user, max_tokens=8000,
                         tag=f"skeptic_{card}")
        raw_path = RES / f"skeptic_raw_{card}.txt"
        raw_path.write_text(raw)                      # persist BEFORE parsing
        print(f"[{card}] raw persisted -> {raw_path} "
              f"(spend ${meter.spent:.2f})", flush=True)
        v = parse_verdict(raw)
        out_path = RES / f"skeptic_verdict_{card}.json"
        out_path.write_text(json.dumps(v, indent=1))
        kills = [k for k in RUBRIC if v[k]["verdict"] == "kill"]
        print(f"[{card}] {v['overall']}"
              + (f"  kills: {kills}" if kills else ""), flush=True)
        for k in RUBRIC:
            print(f"  {k}: {v[k]['verdict']} — {v[k]['reason'][:160]}",
                  flush=True)


if __name__ == "__main__":
    main()
