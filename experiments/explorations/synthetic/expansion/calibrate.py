"""Stage 3 — calibrate one selected candidate (the `measure.py` template, run
by the factory harness).

Pipeline per candidate (all gates preregistered in its frozen card):

1. **Bulk-label** every document of its domain with the Haiku judge using the
   card's frozen `judge_instruction` (chunked, order-preserving; a doc with a
   dead chunk is dropped whole, coverage reported).
2. **Validate the labeler**: Sonnet relabels a held-out doc sample →
   agreement, Cohen's κ, noise floor ε̂; PLUS the card's independent heuristic
   cross-check. κ < 0.30 ⇒ ABORT (labeler inadequate — the topic_switching
   precedent).
3. **Measure the signature** (kind per card) + the N1/N2/N3 null battery +
   bootstrap CIs + a split-half stability check.
4. **Gate**: the card's PRIMARY ordered statistic must exceed BOTH the N1 and
   N2 97.5% null bands, and still do so after ε̂-noise perturbation of the
   labels. Otherwise ABORT.
5. **PROCEED only**: fit the card's Appendix-B mirror on a 70% doc split,
   validate on the held-out 30%; then the adversarial SKEPTIC pass (Opus,
   fixed 5-item kill-rubric) — any kill ⇒ verdict demoted to ABORT.
6. Write `records/<name>/` : labels, stats JSON, figure, and the
   `calibration.md` record (written for ABORTs too — an abort is a success).

    .venv/bin/python -m experiments.explorations.synthetic.expansion.calibrate <name>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

from explorations.synthetic.expansion import labeler as lab
from explorations.synthetic.expansion import mirrors
from explorations.synthetic.expansion import signature as sig
from explorations.synthetic.expansion.client import ROLES, Judge, Meter
from explorations.synthetic.expansion.corpus import load_reasoning_traces

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SEED = 0
KAPPA_FLOOR = 0.30          # labeler adequacy (below ⇒ ABORT, topic_switching-style)
N_NULL, N_BOOT = 200, 300
INTERJUDGE_DOCS = 12

# ── the card's independent heuristics (implemented verbatim from prereg/) ──

_HEDGE = ("maybe", "perhaps", "might", "i think", "not sure", "let me try",
          "possibly", "seems")
_COMMIT = ("therefore", "thus", "clearly", "definitely", "must be",
           "the answer is", "hence", "obviously")


def _heur_hedging(sents):
    out = []
    for s in sents:
        t = s.lower()
        score = sum(w in t for w in _COMMIT) - sum(w in t for w in _HEDGE)
        out.append(2 if score > 0 else (0 if score < 0 else 1))
    return np.array(out, dtype=np.int8)


_ASSUME = re.compile(r"\b(suppose|assume|let|if|consider the case|in case)\b", re.I)
_CONSEQ = re.compile(r"\b(then|therefore|so|thus|hence|it follows|this implies|which means)\b", re.I)


def _heur_assumption(sents):
    return np.array([1 if _ASSUME.search(s) else (2 if _CONSEQ.search(s) else 0)
                     for s in sents], dtype=np.int8)


_QSTART = ("what", "why", "how", "when", "who", "is", "are", "do", "does", "can")
_ASTART = ("yes", "no", "the answer", "this is", "because", "it is")


def _heur_qa(sents):
    out = []
    prev_q = False
    for s in sents:
        t = s.strip().lower()
        if s.strip().endswith("?") or any(t.startswith(w + " ") for w in _QSTART):
            out.append(1)
            prev_q = True
        elif prev_q and any(t.startswith(w) for w in _ASTART):
            out.append(2)
            prev_q = False
        else:
            out.append(0)
            prev_q = False
    return np.array(out, dtype=np.int8)


_QUOTE_RE = re.compile(r"(said|says|stated|according to|told|remarked|noted that|claimed)", re.I)


def _heur_quote(sents):
    def has_pair(s):
        return s.count('"') >= 2 or ("“" in s and "”" in s)
    return np.array([1 if (has_pair(s) or _QUOTE_RE.search(s)) else 0
                     for s in sents], dtype=np.int8)


# — Cycle-2 heuristics (verbatim from the frozen cards) —

_GOAL = re.compile(r"\b(we (are|were) asked|the (problem|question) (asks|gives|states|wants)"
                   r"|need to (find|compute|determine)|asked to (find|compute)"
                   r"|find the (value|area|probability|number)|what is asked|the goal is)\b", re.I)
_GIVEN = re.compile(r"\b(we are given|given that|the given|it is given)\b", re.I)


def _heur_selfref(sents):
    return np.array([1 if (_GOAL.search(s) or _GIVEN.search(s)) else 0
                     for s in sents], dtype=np.int8)


_OP = re.compile(r"\b(multiply|divide|add|subtract|expand|factor|substitute|simplify"
                 r"|differentiate|integrate|square|combine)\b", re.I)
_RESULT = re.compile(r"\b(therefore|so the|thus the|hence|the answer is|this means"
                     r"|we (get|obtain|conclude)|equals?)\b", re.I)
_ARITH = re.compile(r"[0-9]\s*[+\-*/=]\s*[0-9]")


def _heur_opalt(sents):
    out = []
    for s in sents:
        if _RESULT.search(s) and not _OP.search(s):
            out.append(0)
        elif _OP.search(s) or _ARITH.search(s):
            out.append(1)
        else:
            out.append(0)
    return np.array(out, dtype=np.int8)


_ADDR = re.compile(r"\b(you|your|yourself)\b", re.I)
_CTA = re.compile(r"\b(contact us|subscribe|sign up|click here|reach out|feel free"
                  r"|thank you for|welcome to|follow us)\b", re.I)


def _heur_address(sents):
    return np.array([1 if (_CTA.search(s) or _ADDR.search(s)) else 0
                     for s in sents], dtype=np.int8)


_ENUM = re.compile(r"^\s*(\d+[.)]|[-*•]|first|second|third|fourth|next|then|finally|lastly)\b", re.I)
_IMP = re.compile(r"^\s*[A-Z][a-z]+\s+(the|a|an|your|two|one|these)\b")


def _heur_listitem(sents):
    return np.array([1 if (_ENUM.match(s) or (_IMP.match(s) and len(s.split()) < 12)) else 0
                     for s in sents], dtype=np.int8)


_VERIF = re.compile(r"(check|verify|confirm|plug(ging)? back|substitute back|sanity"
                    r"|make sure|does this (match|work)|let('?s| us) verify)", re.I)


def _heur_verif(sents):
    return np.array([1 if _VERIF.search(s) else 0 for s in sents], dtype=np.int8)


# — Cycle-3 heuristics (verbatim from the frozen C1 cards, first calibrated now) —

_ENUM_C1 = re.compile(r"^(first(ly)?|second(ly)?|third(ly)?|next|then|finally"
                      r"|step \d+|\d+[\.\)]|lastly)\b", re.I)


def _heur_enum(sents):
    return np.array([1 if _ENUM_C1.match(s.strip()) else 0 for s in sents],
                    dtype=np.int8)


_GOALRE = re.compile(r"(we need to|we want to|the problem asks|recall that|goal is"
                     r"|find the|solve for|remember that"
                     r"|going back to the (problem|question))", re.I)
_ARITH_AFTER = re.compile(r"[0-9)\]]\s*[+\-*/^=]")


def _heur_goal(sents):
    return np.array([1 if (_GOALRE.search(s) and not _ARITH_AFTER.search(s)) else 0
                     for s in sents], dtype=np.int8)


# — Cycle-3 categorical heuristics (verbatim priority order from the frozen
#   interaction/equality cards; first match wins) —

_PO_VERIF = re.compile(r"verify|check|confirm|plug.*back|satisfies|makes sense")
_PO_CASE = re.compile(r"case |suppose |if we assume|either|branch|scenario")
_PO_SETUP = re.compile(r"we need|we want|let .* denote|define|find the|the goal")
_PO_ALG = re.compile(r"=|plus|minus|times|divide|multiply|simplif|solve|substitut")


def _heur_proofop(sents):
    out = []
    for s in sents:
        t = s.lower()
        if _PO_VERIF.search(t):
            out.append(3)
        elif _PO_CASE.search(t):
            out.append(2)
        elif _PO_SETUP.search(t):
            out.append(4)
        elif _PO_ALG.search(t):
            out.append(1)
        else:
            out.append(0)
    return np.array(out, dtype=np.int8)


_RC_STEP = re.compile(r"^(preheat|mix|add|stir|tighten|cut|pour|place|remove"
                      r"|combine|heat|set)\b|^[a-z]+ the ")
_RC_INGR = re.compile(r"cups?|tbsp|tsp|ingredients|you will need|grams|ounces"
                      r"|materials:|[0-9]+ ?(cup|tablespoon|gram)")
_RC_TIP = re.compile(r"be careful|note:|tip:|for best results|make sure|avoid"
                     r"|warning|optionally")
_RC_CTX = re.compile(r"originates|history|traditionally|because|this is|known as"
                     r"|dates back")


def _heur_recipe(sents):
    out = []
    for s in sents:
        t = s.strip().lower()
        if _RC_STEP.match(t):
            out.append(3)
        elif _RC_INGR.search(t):
            out.append(2)
        elif _RC_TIP.search(t):
            out.append(4)
        elif _RC_CTX.search(t):
            out.append(1)
        else:
            out.append(0)
    return np.array(out, dtype=np.int8)


_PRONOUN_START = re.compile(r"^\s*(he|she|it|they|this|that|these|those)\b", re.I)
_PROPER = re.compile(r"(?<!^)(?<![.!?]\s)\b[A-Z][a-z]+(\s+[A-Z][a-z]+)+\b")


def _heur_named(sents):
    out = []
    for s in sents:
        named = (_PROPER.search(s) or re.match(r"^\s*[A-Z][a-z]+\s+[A-Z][a-z]+", s))
        out.append(1 if (named and not _PRONOUN_START.match(s)) else 0)
    return np.array(out, dtype=np.int8)


# ── per-candidate calibration config (statistic kinds + primary gate stat) ──

# Per-candidate config. Cycle-2 fields: `sign` ("+"/"-") — the preregistered
# direction of the primary effect (negative = alternation, gated against the
# null band's LO side); `ctx` — context sentences shown to the judge (0 for
# gate-7 strict per-sentence cards); `gate8` — (moment, curve-idx, abs tol)
# the mirror must reproduce held-out (fail ⇒ mirror invalid ⇒ ABORT).
CFG = {
    # — Cycle 1 —
    "uncertainty-hedging-drift": dict(
        domain="reasoning-trace", kind="scalar", pair=None,
        primary=("acf", 0), heuristic=_heur_hedging,
        mirror="ar1", mirror_kw={"position": True}, mirror_kind="scalar"),
    "assumption-then-consequence": dict(
        domain="reasoning-trace", kind="categorical", pair=(1, 2),
        primary=("asym", None), heuristic=_heur_assumption,
        mirror="markov", mirror_kw={}, mirror_kind="categorical"),
    "question-answer-adjacency": dict(
        domain="text-corpus", kind="categorical", pair=(1, 2),
        primary=("asym", None), heuristic=_heur_qa,
        mirror="markov", mirror_kw={}, mirror_kind="categorical"),
    "quotation-burst": dict(
        domain="text-corpus", kind="binary", pair=None,
        primary=("acf", 0), heuristic=_heur_quote,
        mirror="logistic_ar", mirror_kw={"K": 8}, mirror_kind="binary"),
    # — Cycle 2: new interaction/equality cards (gate-7: ctx=0) —
    "self-reference-echo": dict(
        domain="reasoning-trace", kind="binary", pair=None,
        primary=("acf", 0), sign="+", ctx=0, heuristic=_heur_selfref,
        mirror="logistic_ar", mirror_kw={"K": 8}, mirror_kind="binary",
        gate8=("mi", 0, 0.015)),
    "operator-alternation": dict(
        domain="reasoning-trace", kind="binary", pair=None,
        primary=("acf", 0), sign="-", ctx=0, heuristic=_heur_opalt,
        mirror="markov", mirror_kw={}, mirror_kind="binary",
        gate8=("dwell_cv", None, 0.12)),
    "greeting-signoff-mirror": dict(
        domain="text-corpus", kind="binary", pair=None,
        primary=("mi", 0), sign="+", ctx=0, heuristic=_heur_address,
        mirror="periodic_rate", mirror_kw={}, mirror_kind="binary",
        gate8=("mi", 0, 0.02)),
    "list-item-parallelism": dict(
        domain="text-corpus", kind="binary", pair=None,
        primary=("acf", 0), sign="+", ctx=0, heuristic=_heur_listitem,
        mirror="logistic_ar", mirror_kw={"K": 8}, mirror_kind="binary",
        gate8=("fano", None, 0.15)),
    # — Cycle 2: frozen Cycle-1 cards (kept at ctx=3, their frozen convention;
    #   gate8 from the dated amendments) —
    "computation-verification-alternation": dict(
        domain="reasoning-trace", kind="binary", pair=None,
        primary=("spec_peak", None), sign="+", ctx=3, heuristic=_heur_verif,
        mirror="periodic_rate", mirror_kw={}, mirror_kind="binary",
        gate8=("fano", None, 0.30)),
    "pronoun-referent-recurrence": dict(
        domain="text-corpus", kind="binary", pair=None,
        primary=("gap_cv", None), sign="+", ctx=3, heuristic=_heur_named,
        mirror="semi_markov", mirror_kw={}, mirror_kind="categorical",
        gate8=("acf", 0, 0.05)),
    # — Cycle 2 rider: the gate-7 re-exam (strict per-sentence relabel of the
    #   provisional SPEC*; judge instruction from the dated card amendment) —
    "assumption-consequence-g7": dict(
        domain="reasoning-trace", kind="categorical", pair=(1, 2),
        primary=("asym", None), sign="+", ctx=0, heuristic=_heur_assumption,
        mirror="markov", mirror_kw={}, mirror_kind="categorical",
        gate8=("acf", 0, 0.05), base_card="assumption-then-consequence"),
    # — Cycle 3: re-freezes (dated card amendments; cached C2 labels reused;
    #   gate-8 under the uniform relative rule of amend_cards_c3) —
    "list-item-parallelism-r2": dict(
        domain="text-corpus", kind="binary", pair=None,
        primary=("acf", 0), sign="+", ctx=0, heuristic=_heur_listitem,
        mirror="logistic_ar", mirror_kw={"K": 8}, mirror_kind="binary",
        gate8=("fano", None, 0.20, "rel"), base_card="list-item-parallelism",
        labels_from="list-item-parallelism"),
    "computation-verification-r2": dict(
        domain="reasoning-trace", kind="binary", pair=None,
        primary=("spec_peak", None), sign="+", ctx=3, heuristic=_heur_verif,
        mirror="periodic_hawkes", mirror_kw={"K": 8}, mirror_kind="binary",
        gate8=("fano", None, 0.20, "rel"),
        base_card="computation-verification-alternation",
        labels_from="computation-verification-alternation"),
    # — Cycle 3: the two still-frozen C1 cards, first calibration (tolerances
    #   converted to the uniform relative rule, still blind) —
    "enumeration-cadence": dict(
        domain="text-corpus", kind="binary", pair=None,
        primary=("spec_peak", None), sign="+", ctx=3, heuristic=_heur_enum,
        mirror="periodic_rate", mirror_kw={}, mirror_kind="binary",
        gate8=("fano", None, 0.20, "rel")),
    "goal-restatement-recurrence": dict(
        domain="reasoning-trace", kind="binary", pair=None,
        primary=("gap_cv", None), sign="+", ctx=3, heuristic=_heur_goal,
        mirror="semi_markov", mirror_kw={}, mirror_kind="categorical",
        gate8=("acf", 0, 0.20, "rel")),
    # — Cycle 3: the selected new categorical interaction/equality cards
    #   (gate-7 recipe: content classes, ctx=0; primary = the equality-
    #   adjacency [c_t=c_{t-1}] = categorical self-match ACF(1)) —
    "proof-operation-phase-runs": dict(
        domain="reasoning-trace", kind="categorical", pair=None,
        primary=("acf", 0), sign="+", ctx=0, heuristic=_heur_proofop,
        mirror="semi_markov", mirror_kw={}, mirror_kind="categorical",
        gate8=("mi", 1, 0.20, "rel")),
    "recipe-instruction-phase-runs": dict(
        domain="text-corpus", kind="categorical", pair=None,
        primary=("acf", 0), sign="+", ctx=0, heuristic=_heur_recipe,
        mirror="semi_markov", mirror_kw={}, mirror_kind="categorical",
        gate8=("acf", 3, 0.20, "rel")),
    # — Cycle 4: re-freezes of the two C3 real-signal int/eq aborts under the
    #   hier_categorical menu extension (dated card amendments; cached C3
    #   labels + validation reused; gate-8 HARDENED to ≥2 non-fitted moments,
    #   each including its C3 killer) —
    "proof-operation-phase-runs-r2": dict(
        domain="reasoning-trace", kind="categorical", pair=None,
        primary=("acf", 0), sign="+", ctx=0, heuristic=_heur_proofop,
        mirror="hier_categorical", mirror_kw={}, mirror_kind="categorical",
        gate8=(("mi", 1, 0.20, "rel"), ("acf", 3, 0.20, "rel")),
        base_card="proof-operation-phase-runs",
        labels_from="proof-operation-phase-runs"),
    "recipe-instruction-phase-runs-r2": dict(
        domain="text-corpus", kind="categorical", pair=None,
        primary=("acf", 0), sign="+", ctx=0, heuristic=_heur_recipe,
        mirror="hier_categorical", mirror_kw={}, mirror_kind="categorical",
        gate8=(("acf", 3, 0.20, "rel"), ("mi", 1, 0.20, "rel")),
        base_card="recipe-instruction-phase-runs",
        labels_from="recipe-instruction-phase-runs"),
    # — Cycle 5: re-freeze of the reasoning int/eq abort under the
    #   seg_hier_categorical menu extension (dated card amendment; C3 labels +
    #   validation reused; hardened two-moment gate-8 kept, BOTH moments in
    #   the lag-2–8 region that killed r2; PLUS the preregistered INSERTION
    #   CONTROL — the mirror re-fit on run-permuted streams must not
    #   hallucinate either moment beyond the real-data tolerance) —
    "proof-operation-phase-runs-r3": dict(
        domain="reasoning-trace", kind="categorical", pair=None,
        primary=("acf", 0), sign="+", ctx=0, heuristic=_heur_proofop,
        mirror="seg_hier_categorical", mirror_kw={}, mirror_kind="categorical",
        gate8=(("mi", 1, 0.20, "rel"), ("acf", 3, 0.20, "rel")),
        base_card="proof-operation-phase-runs",
        labels_from="proof-operation-phase-runs"),
    # — Cycle 7: r4 under the monotone calibrated estimator (dated C7 card
    #   amendment; runs ONLY if the C7 battery's pre-specified fork says so
    #   — pre-check + gates 1–3 pass and λ*_real ≤ 0.85; gate-8 + insertion
    #   control + label reuse verbatim from the r3/r4 amendments) —
    "proof-operation-phase-runs-r4": dict(
        domain="reasoning-trace", kind="categorical", pair=None,
        primary=("acf", 0), sign="+", ctx=0, heuristic=_heur_proofop,
        mirror="seg_hier_categorical_mono", mirror_kw={},
        mirror_kind="categorical",
        gate8=(("mi", 1, 0.20, "rel"), ("acf", 3, 0.20, "rel")),
        base_card="proof-operation-phase-runs",
        labels_from="proof-operation-phase-runs"),
}

# Uniform C3 relative-tolerance floors (amend_cards_c3.py preregistration).
GATE8_FLOORS = {"acf": 0.01, "mi": 0.003, "fano": 0.05, "dwell_cv": 0.05,
                "gap_cv": 0.05, "spec_peak": 0.05}


def load_domain(domain: str):
    if domain == "reasoning-trace":
        return load_reasoning_traces(REPO / "results/c7_backtracking/stage_a")
    return json.loads((HERE / "data/fineweb_sample.json").read_text())


def _moment(seqs, moment: str, idx, kind: str) -> float:
    """Evaluate a gate-8 moment on raw sequences (independent of headline)."""
    if moment == "acf":
        v = sig.selfmatch_acf(seqs) if kind == "categorical" else sig.acf(seqs)
        return float(v[idx or 0])
    if moment == "mi":
        if kind == "scalar":
            return float(sig.mi_vs_lag(sig.quantile_bin(seqs), 12, 8)[idx or 0])
        n_sym = int(max(int(np.concatenate(seqs).max()) + 1, 2))
        return float(sig.mi_vs_lag(seqs, 12, n_sym)[idx or 0])
    if moment == "fano":
        return float(sig.fano(seqs))
    if moment == "gap_cv":
        return float(sig.inter_event_cv(seqs)["cv"])
    if moment == "dwell_cv":
        return float(sig.dwell_stats(seqs)["cv"])
    if moment == "spec_peak":
        return float(sig.spec_peak(seqs))
    raise ValueError(f"unknown gate-8 moment {moment!r}")


def primary_value(h: dict, primary) -> float:
    key, idx = primary
    v = np.asarray(h[key], dtype=float)
    return float(v.ravel()[idx]) if idx is not None else float(v)


def band_value(band: dict, primary, which: str) -> float:
    key, idx = primary
    v = np.asarray(band[key][which], dtype=float)
    return float(v.ravel()[idx]) if idx is not None else float(v)


# ── skeptic pass (Opus, fixed kill-rubric — briefing guardrail #5) ──────────

SKEPTIC_SYSTEM = """\
You are the adversarial SKEPTIC in a measure->mirror benchmark loop. A candidate
temporal property has provisionally PASSED its null-battery gate and its mirror was
fit. Your job is to KILL it if it does not deserve to be frozen as a benchmark spec.
You will see the frozen prereg card, the labeler-validation numbers, the measured
statistics vs the nulls, and the mirror validation.

Fill this fixed kill-rubric. For each item answer kill=true only with concrete
evidence from the numbers given (kill=false needs a one-line justification too):
 a. noise_floor: is the ordered-vs-shuffled gap within the labeler noise floor
    (i.e. could label noise alone produce it)?
 b. leakage: could the labeler be leaking the target (the label definition itself
    builds in the temporal statistic, e.g. an 'answer' label that requires a prior
    question makes question->answer ordering circular)?
 c. composition: is the effect per-document composition / marginal, not within-
    document order (the topic_switching trap — check the N1 comparison)?
 d. circularity: does the mirror match the statistic by construction in a way that
    makes the validation vacuous (validating on the same quantity that was fit is
    expected — kill only if the SPEC would test nothing beyond what was inserted)?
 e. segmentation: is the effect plausibly an artifact of sentence segmentation or
    windowing choices (e.g. the splitter creating alternation by construction)?

Respond with ONLY a JSON object, no prose, no fence:
{"a_noise_floor": {"kill": bool, "evidence": "..."},
 "b_leakage": {"kill": bool, "evidence": "..."},
 "c_composition": {"kill": bool, "evidence": "..."},
 "d_circularity": {"kill": bool, "evidence": "..."},
 "e_segmentation": {"kill": bool, "evidence": "..."},
 "overall_note": "..."}"""


def _parse_json_object(text: str) -> dict:
    """Best-effort extraction of one JSON object from judge output.

    Tries, in order: a ```json fenced block, raw_decode from each '{' (longest
    valid parse wins), then the legacy greedy-regex parse. Raises ValueError
    if nothing parses — the caller decides whether to repair.
    """
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    dec = json.JSONDecoder()
    best = None
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, end = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and (best is None or end - i > best[1]):
            best = (obj, end - i)
    if best:
        return best[0]
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        return json.loads(m.group(0))   # may raise — nothing parsed
    raise ValueError("no JSON object found in judge output")


def skeptic_pass(judge: Judge, name: str, card_md: str, summary: dict) -> dict:
    user = (f"## Frozen prereg card\n\n{card_md}\n\n## Calibration numbers\n\n"
            + json.dumps(summary, indent=1, default=float)
            + "\n\nFill the kill-rubric. JSON only.")
    text = judge.call("think", SKEPTIC_SYSTEM, user, max_tokens=4000,
                      tag=f"skeptic:{name}")
    # Persist the raw verdict BEFORE parsing: a parse crash must never lose an
    # adversarial verdict (no-re-roll rule — the verdict exists once written).
    raw_path = HERE / "records" / name / "skeptic_raw.txt"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(text)
    rubric = ("a_noise_floor", "b_leakage", "c_composition", "d_circularity",
              "e_segmentation")
    try:
        out = _parse_json_object(text)
        if not all(k in out for k in rubric):
            raise ValueError("parsed object missing rubric items")
        out["_judge_model"] = ROLES["think"]
        return out
    except (ValueError, json.JSONDecodeError):
        # Deterministic cheap repair: ask the bulk model to fix syntax ONLY
        # (e.g. a string truncated at max_tokens). Verdicts must not change.
        fixed = judge.call(
            "bulk",
            "Return the following content as ONE syntactically valid JSON "
            "object. Fix syntax only (quoting, escaping, commas, unterminated "
            "strings). Do NOT change any keys, values, wording, or verdicts. "
            "JSON only.",
            text, max_tokens=4000, tag=f"skeptic-jsonfix:{name}")
        out = _parse_json_object(fixed)
        if not all(k in out for k in rubric):
            raise ValueError("skeptic verdict unrecoverable — rubric items missing")
        out["_judge_model"] = ROLES["think"]
        return out


# ── main per-candidate pipeline ────────────────────────────────────────────

def load_candidate(name: str, cfg: dict) -> dict:
    """Look up the frozen card across cycle registries (+ g7-re-exam override)."""
    pool = json.loads((HERE / "results/candidates.json").read_text())["candidates"]
    for extra in ("candidates_cycle2.json", "candidates_cycle3.json"):
        p = HERE / "results" / extra
        if p.exists():
            pool += json.loads(p.read_text())["candidates"]
    lookup = cfg.get("base_card", name)
    cand = dict(next(c for c in pool if c["name"] == lookup))
    if name == "assumption-consequence-g7":
        amend = json.loads((HERE / "results/amendments_cycle2.json").read_text())["g7_reexam"]
        cand["judge_instruction"] = amend["judge_instruction"]
        cand["name"] = name
    return cand


def run(name: str):
    cfg = CFG[name]
    cand = load_candidate(name, cfg)
    spec = {"name": name, "kind": cand["label_kind"], "n_values": cand["n_values"],
            "judge_instruction": cand["judge_instruction"]}
    out_dir = HERE / "records" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    meter = Meter()
    judge = Judge(meter)
    rng = np.random.default_rng(SEED)

    data = load_domain(cfg["domain"])
    docs = [d["sentences"] for d in data["docs"]]
    doc_ids = [d["id"] for d in data["docs"]]
    print(f"[{name}] domain={cfg['domain']}  docs={len(docs)}  "
          f"sents={sum(map(len, docs))}  spend=${meter.spent:.2f}")

    # 1 ── bulk label (cache to disk so a crashed run never re-spends).
    # A re-freeze record (labels_from) reuses its base record's cached labels +
    # labeler validation verbatim: the labeler is unchanged, only the mirror
    # gate was re-preregistered — relabeling would re-spend for zero
    # information (preregistered in the C3 card amendments).
    labels_path = out_dir / "labels.json"
    if cfg.get("labels_from") and not labels_path.exists():
        import shutil
        src_dir = HERE / "records" / cfg["labels_from"]
        shutil.copy(src_dir / "labels.json", labels_path)
        shutil.copy(src_dir / "labeler_validation.json",
                    out_dir / "labeler_validation.json")
        print(f"[{name}] labels + validation reused from records/{cfg['labels_from']}")
    if labels_path.exists():
        blob = json.loads(labels_path.read_text())
        seqs = [np.array(x, dtype=np.int8) if x is not None else None
                for x in blob["labels"]]
        coverage = blob["coverage"]
        print(f"[{name}] labels loaded from cache ({coverage['doc_coverage']:.3f} coverage)")
    else:
        seqs, coverage = lab.label_stream(judge, docs, spec, role="bulk",
                                          chunk=50, ctx=cfg.get("ctx", 3),
                                          workers=8, tag=f"bulk:{name}")
        labels_path.write_text(json.dumps(
            {"doc_ids": doc_ids, "coverage": coverage,
             "labels": [s.tolist() if s is not None else None for s in seqs]}))
        print(f"[{name}] labeled: {coverage}  spend=${meter.spent:.2f}")

    # 2 ── labeler validation: inter-judge + heuristic cross-check
    val_path = out_dir / "labeler_validation.json"
    if val_path.exists():
        val = json.loads(val_path.read_text())
    else:
        inter = lab.validate_interjudge(judge, docs, seqs, spec,
                                        sample_docs=INTERJUDGE_DOCS, seed=SEED,
                                        ctx=cfg.get("ctx", 3),
                                        tag=f"interjudge:{name}")
        heur = [cfg["heuristic"](d) for d in docs]
        xc = (lab.crosscheck_binary(seqs, heur) if cand["label_kind"] == "binary"
              else lab.crosscheck_categorical(seqs, heur, cand["n_values"]))
        d = inter["disagreement"]
        eps = (lab.noise_floor_from_disagreement(d) if cand["label_kind"] == "binary"
               else lab.noise_floor_categorical(d))
        val = {"interjudge": inter, "heuristic_crosscheck": xc, "noise_floor_eps": eps}
        val_path.write_text(json.dumps(val, indent=2, default=float))
    eps = val["noise_floor_eps"]
    kappa = val["interjudge"]["kappa"]
    print(f"[{name}] interjudge κ={kappa:.3f} agree={val['interjudge']['agreement']:.3f} "
          f"ε̂={eps:.3f}  spend=${meter.spent:.2f}")

    ok = [s for s in seqs if s is not None]
    if cfg["kind"] == "scalar":
        ok = [s.astype(float) for s in ok]

    # 3 ── signature + nulls + stability
    stats = sig.measure(ok, cfg["kind"], seed=SEED, n_null=N_NULL, n_boot=N_BOOT,
                        pair=cfg["pair"],
                        noise_eps=(eps,) if cfg["kind"] == "binary" and eps > 0 else ())
    half = rng.permutation(len(ok))
    h1 = [ok[i] for i in half[: len(ok) // 2]]
    h2 = [ok[i] for i in half[len(ok) // 2:]]
    hkw = dict(pair=cfg["pair"])
    stability = {"half1": primary_value(sig.headline(h1, cfg["kind"], **hkw), cfg["primary"]),
                 "half2": primary_value(sig.headline(h2, cfg["kind"], **hkw), cfg["primary"])}

    # 4 ── noise-perturbed primary (effect must survive the noise floor)
    if cfg["kind"] == "binary":
        pert = sig.flip_labels(ok, eps, rng)
    else:
        pert = sig.perturb_categorical([s.astype(np.int8) for s in ok] if cfg["kind"] == "scalar" else ok,
                                       eps, rng)
        if cfg["kind"] == "scalar":
            pert = [p.astype(float) for p in pert]
    pert_h = sig.headline(pert, cfg["kind"], **hkw)

    real_p = primary_value(stats["real"], cfg["primary"])
    n1_hi = band_value(stats["nulls"]["N1_permute"], cfg["primary"], "hi")
    n2_hi = band_value(stats["nulls"]["N2_trend"], cfg["primary"], "hi")
    n1_lo = band_value(stats["nulls"]["N1_permute"], cfg["primary"], "lo")
    n2_lo = band_value(stats["nulls"]["N2_trend"], cfg["primary"], "lo")
    pert_p = primary_value(pert_h, cfg["primary"])

    sign = cfg.get("sign", "+")
    if sign == "+":
        clears_sampling = real_p > n1_hi and real_p > n2_hi
        clears_noise = pert_p > n1_hi and pert_p > n2_hi
    else:  # preregistered NEGATIVE effect (alternation): beat the LO side
        clears_sampling = real_p < n1_lo and real_p < n2_lo
        clears_noise = pert_p < n1_lo and pert_p < n2_lo

    gate = {
        "primary_stat": f"{cfg['primary'][0]}"
                        + (f"[lag{cfg['primary'][1] + 1}]" if cfg["primary"][1] is not None else ""),
        "sign": sign,
        "real": real_p, "noise_perturbed": pert_p,
        "N1_hi": n1_hi, "N2_hi": n2_hi, "N1_lo": n1_lo, "N2_lo": n2_lo,
        "labeler_kappa": kappa, "kappa_floor": KAPPA_FLOOR,
        "noise_floor_eps": eps, "stability": stability,
        "clears_sampling": bool(clears_sampling),
        "clears_noise": bool(clears_noise),
        "labeler_ok": bool(kappa >= KAPPA_FLOOR),
    }
    verdict = "PROCEED" if all(
        (gate["clears_sampling"], gate["clears_noise"], gate["labeler_ok"])) else "ABORT"
    gate["verdict_pre_skeptic"] = verdict
    print(f"[{name}] GATE: real={real_p:.4f} pert={pert_p:.4f} "
          f"N1hi={n1_hi:.4f} N2hi={n2_hi:.4f} κ={kappa:.2f} -> {verdict}")

    # 5 ── PROCEED only: mirror fit + held-out validation + skeptic
    # The skeptic verdict is CACHED across reruns (like the labels): a rerun —
    # e.g. after an infrastructure fix to the validation sampling — must never
    # re-roll the adversarial pass hoping for a different answer.
    prev_path = out_dir / "calibration_stats.json"
    prev_skeptic = None
    if prev_path.exists():
        prev_skeptic = json.loads(prev_path.read_text()).get("skeptic")
    mirror_blob = skeptic = None
    if verdict == "PROCEED":
        idx = rng.permutation(len(ok))
        cut = int(0.7 * len(ok))
        train = [ok[i] for i in idx[:cut]]
        ev = [ok[i] for i in idx[cut:]]
        fit_fn, gen_fn = mirrors.MENU[cfg["mirror"]]
        m_train = train if cfg["mirror_kind"] != "categorical" else [s.astype(np.int8) for s in train]
        params = fit_fn(m_train, **cfg["mirror_kw"])
        syn = gen_fn(params, [s.size for s in ev], rng)
        if cfg["kind"] == "scalar":
            syn = [np.asarray(s, dtype=float) for s in syn]
        mv = mirrors.validate_mirror(ev, syn, cfg["kind"], maxlag=12)
        if cfg["pair"] is not None:
            mv["real_directed"] = sig.directed_transition(ev, *cfg["pair"])
            mv["syn_directed"] = sig.directed_transition(
                [np.asarray(s, dtype=np.int8) for s in syn], *cfg["pair"])
        mirror_blob = {"params": params, "n_train": len(train), "n_eval": len(ev),
                       "validation": mv}
        print(f"[{name}] mirror fit+validated (n_train={len(train)})")

        # gate 8 (preregistered): the mirror must reproduce its named
        # NON-FITTED moment(s) on held-out real vs synthetic within tolerance.
        # Tolerances are absolute (C1/C2 3-tuples) or, from Cycle 3 on,
        # relative to the held-out real magnitude with a per-moment floor
        # (4-tuples ending "rel"; the amend_cards_c3 uniform rule). From
        # Cycle 4 on, `gate8` may be a tuple OF such tuples — the ≥2
        # non-fitted-moment hardening (LEDGER C4 target): ALL must pass.
        if cfg.get("gate8"):
            specs = cfg["gate8"]
            if isinstance(specs[0], str):   # single pre-C4 spec
                specs = (specs,)
            syn_cat = [np.asarray(s, dtype=np.int8) for s in syn] \
                if cfg["kind"] != "scalar" else syn
            ev_cat = [np.asarray(s, dtype=np.int8) for s in ev] \
                if cfg["kind"] != "scalar" else ev
            g8s = []
            for moment, midx, tol, *mode in specs:
                rv = _moment(ev_cat, moment, midx, cfg["kind"])
                sv = _moment(syn_cat, moment, midx, cfg["kind"])
                if mode and mode[0] == "rel":
                    tol_eff = max(tol * abs(rv), GATE8_FLOORS[moment])
                    tol_note = f"±{tol:.0%} rel of |real| (floor {GATE8_FLOORS[moment]})"
                else:
                    tol_eff = tol
                    tol_note = "abs (pre-C3 preregistration)"
                g8 = {"moment": moment + (f"[lag{midx + 1}]" if midx is not None else ""),
                      "real_heldout": rv, "synthetic": sv,
                      "abs_err": abs(rv - sv), "tol_abs": tol_eff, "tol_note": tol_note,
                      "pass": bool(abs(rv - sv) <= tol_eff)}
                g8s.append(g8)
                print(f"[{name}] gate8 {g8['moment']}: real={rv:.4f} syn={sv:.4f} "
                      f"|err|={g8['abs_err']:.4f} tol={tol_eff:.4f} ({tol_note}) -> "
                      f"{'PASS' if g8['pass'] else 'FAIL'}")
            mirror_blob["gate8"] = g8s[0] if len(g8s) == 1 else g8s
            if not all(g["pass"] for g in g8s):
                verdict = "ABORT"
                gate["gate8_fail"] = True
                gate["verdict"] = verdict
                print(f"[{name}] MIRROR GATE-8 FAIL -> ABORT (skeptic skipped)")

            # INSERTION CONTROL (preregistered, Cycle 5; seg mirror only).
            # The segment mirror's DP + raw compositions + deconvolution can
            # hallucinate lag structure on data with no segment layer (the
            # winner's curse — every automatic shrinkage variant tried on the
            # harness toys either drowned real signal or leaked). So the
            # control measures the estimator's hallucination ON THIS DATA:
            # re-fit the same mirror on run-permuted train streams (the
            # no-adjacent-repeat shuffle preserves doc composition, run
            # lengths, and no-self-jump while destroying segment structure)
            # and require, for EVERY gate-8 moment, that the null fit's
            # generated value stays within the real-data effective tolerance
            # of the permuted streams' own value — hallucination must be
            # subdominant to the structure the gate certifies. Runs even
            # after a gate-8 fail (cheap, informative either way).
            if cfg["mirror"].startswith("seg_hier_categorical"):
                perm_train = mirrors.run_permuted_streams(m_train)
                params0 = fit_fn(perm_train, **cfg["mirror_kw"])
                syn0 = gen_fn(params0, [s.size for s in ev], rng)
                syn0_cat = [np.asarray(s, dtype=np.int8) for s in syn0]
                perm_ev = mirrors.run_permuted_streams(ev_cat)
                ctrls = []
                for g8, (moment, midx, *_rest) in zip(g8s, specs):
                    pv = _moment(perm_ev, moment, midx, cfg["kind"])
                    sv0 = _moment(syn0_cat, moment, midx, cfg["kind"])
                    ctrl = {"moment": g8["moment"], "perm_heldout": pv,
                            "syn_nullfit": sv0, "insertion": abs(sv0 - pv),
                            "tol_abs": g8["tol_abs"],
                            "pass": bool(abs(sv0 - pv) <= g8["tol_abs"])}
                    ctrls.append(ctrl)
                    print(f"[{name}] insertion-control {ctrl['moment']}: "
                          f"perm={pv:.4f} nullfit={sv0:.4f} "
                          f"ins={ctrl['insertion']:.4f} tol={g8['tol_abs']:.4f} "
                          f"-> {'PASS' if ctrl['pass'] else 'FAIL'}")
                mirror_blob["insertion_control"] = ctrls
                if not all(c["pass"] for c in ctrls):
                    verdict = "ABORT"
                    gate["insertion_control_fail"] = True
                    gate["verdict"] = verdict
                    print(f"[{name}] INSERTION CONTROL FAIL -> ABORT "
                          "(mirror over-expressive on this data)")

        if verdict == "PROCEED":  # gate 8 may already have demoted it
            if prev_skeptic is not None:
                skeptic = dict(prev_skeptic, reused_from_prior_run=True)
                print(f"[{name}] skeptic verdict reused from prior run (never re-rolled)")
            else:
                card_name = cfg.get("base_card", name)
                card_md = (HERE / "prereg" / f"{card_name}.md").read_text()
                summary = {"gate": gate, "labeler_validation": val,
                           "null_bands_primary": {
                               k: {w: band_value(stats["nulls"][k], cfg["primary"], w)
                                   for w in ("mean", "lo", "hi")} for k in stats["nulls"]},
                           "mirror": mirror_blob, "coverage": coverage}
                skeptic = skeptic_pass(judge, name, card_md, summary)
            kills = [k for k, v in skeptic.items()
                     if isinstance(v, dict) and v.get("kill")]
            if kills:
                verdict = "ABORT"
                gate["killed_by_skeptic"] = kills
                print(f"[{name}] SKEPTIC KILLED: {kills}")
            else:
                print(f"[{name}] skeptic pass: survived all 5 items")

    gate["verdict"] = verdict

    blob = {"name": name, "config": {k: str(v) if callable(v) else v
                                     for k, v in cfg.items()},
            "coverage": coverage, "labeler_validation": val, "signature": stats,
            "noise_perturbed_headline": {k: (np.asarray(v).tolist() if np.ndim(v) else float(v))
                                         for k, v in pert_h.items()},
            "gate": gate, "mirror": mirror_blob, "skeptic": skeptic,
            "spend_usd_after": meter.spent}
    (out_dir / "calibration_stats.json").write_text(
        json.dumps(blob, indent=2, default=float))
    _figure(name, cfg, stats, gate, out_dir)
    print(f"[{name}] VERDICT: {verdict}   spend=${meter.spent:.2f} of ${meter.cap:.0f}")
    return blob


def _figure(name, cfg, stats, gate, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lags = np.arange(1, len(stats["real"]["acf"]) + 1)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(lags, stats["real"]["acf"], "o-", color="#1f77b4", label="real", lw=2)
    for nm, col in [("N1_permute", "#999"), ("N2_trend", "#d62728"), ("N3_iid", "#2ca02c")]:
        m = np.array(stats["nulls"][nm]["acf"]["mean"])
        lo = np.array(stats["nulls"][nm]["acf"]["lo"])
        hi = np.array(stats["nulls"][nm]["acf"]["hi"])
        ax[0].plot(lags, m, "--", color=col, label=nm, lw=1)
        ax[0].fill_between(lags, lo, hi, color=col, alpha=0.15)
    ax[0].axhline(0, color="k", lw=0.6, alpha=0.4)
    ax[0].set_xlabel("lag (sentences)")
    ax[0].set_ylabel("autocorrelation")
    ax[0].set_title("ACF: real vs nulls")
    ax[0].legend(fontsize=8)

    pr = stats["position_profile"]
    ax[1].plot(np.linspace(0, 1, len(pr)), pr, "o-", color="#1f77b4")
    ax[1].set_xlabel("normalized position")
    ax[1].set_ylabel("mean label")
    ax[1].set_title("Position profile")

    g = gate
    bars = ["real", "noise_perturbed", "N1_hi", "N2_hi"]
    ax[2].bar(bars, [g[b] for b in bars],
              color=["#1f77b4", "#9467bd", "#999", "#d62728"])
    ax[2].set_title(f"Gate on {g['primary_stat']} → {g['verdict']}")
    ax[2].tick_params(axis="x", rotation=20)
    for a in ax:
        a.grid(True, alpha=0.25)
    fig.suptitle(f"{name} — temporal signature (n={stats['n_seqs']} docs, "
                 f"{stats['n_spans']} sentences)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext, dpi in [("pdf", None), ("png", 120), ("thumb.png", 55)]:
        fig.savefig(out_dir / f"signature.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run(sys.argv[1])
