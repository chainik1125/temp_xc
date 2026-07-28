"""`retryesc_gen` scaffold — repeat-failure escalation, generated.

Executes `retryesc_gen/GENERATION_CARD.md` (freeze `3f6ba0d3d`, § 2.2a
corrigendum applied). Rebuild of `retryesc`, which died label-side on
`unigram_auc` 0.689-0.716 vs a 0.60 bar because **task difficulty
genuinely drove failure rate**, so task nouns predicted event age.

THE THREE THINGS THIS SCAFFOLD MUST GET RIGHT
---------------------------------------------
1. **s1.2 shape.** EVENT = an environment turn reporting failure of a
   strategy ALREADY ATTEMPTED earlier in the episode. Whether an
   attempt is a repeat depends on what happened hundreds of tokens ago
   -- out-of-window by construction -- while the kernel support stays
   inside T. Seeing "I'll try <strategy>" inside the window does NOT
   reveal repeat-status; you need the earlier mention, which is
   outside it.

2. **Density-safe (card s3, BINDING).** Environment failure text is
   drawn from `FAILURE_TEXTS` **independently of repeat-status**, so a
   repeat failure and a first-time failure are TEXTUALLY
   INDISTINGUISHABLE. This is what lets the corpus be dense enough to
   clear the floor-excess target without handing the label to a
   visible cue. If this property is ever broken the candidate is dead.

3. **Vocabulary ⊥ difficulty (card s4).** The outcome schedule is drawn
   FIRST, from `plan()`, independent of the task. The task is drawn
   afterwards and never consults the schedule. So difficulty is
   *assigned*, not intrinsic: the same task appears under easy and hard
   schedules across documents. Strategies come from a GLOBAL pool
   shared by every task, so strategy tokens carry no task identity
   either.

CLOCK (card s2.2a): target gap median g ~ 385 tok (range 297-499) so
that `floor_excess == f == P(event inside T=64)` lands in [+0.15,+0.25].
`g` is the ONLY knob permitted to move after freeze; the target does not.

RESUME IS EXACT here, unlike `run_evalage`: every per-turn choice is
drawn from the plan UP FRONT, so `rng` is not consumed inside the
generation loop and a resumed run is bit-identical.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ── outcome codes (the plan's alphabet) ─────────────────────────────
OK = 0            # environment reports success
FAIL_NEW = 1      # failure of a strategy not tried before
FAIL_REPEAT = 2   # failure of a strategy already tried  <-- THE EVENT

# ── clock targets ───────────────────────────────────────────────────
#
# ⚑ THE BINDING TARGET IS `f`, AND ONLY `f`. The card said so from the
# start ("the bar is written on the MEASURED quantity, not on g"), which
# is what makes the following correction cheap instead of fatal.
FLOOR_EXCESS_BAND = (0.15, 0.25)

# ⚠ SUPERSEDED — the card's gap-median target of 385 tok (range
# 297-499) was derived assuming probe positions are UNIFORM over the
# stream. They are not: eligible positions are assistant tokens, offset
# from the preceding event by the masked environment turn, and a long
# assistant turn pushes its own tail past T=64. So `f` is far lower at
# a given gap than the uniform model predicts.
#
# Measured (dry_run.py, stub prose): the in-band setting realises a gap
# median of ~180 tok, less than HALF the "target", and lands f = 0.185.
# The gap figure is retained only as a descriptive receipt.
GAP_MEDIAN_SUPERSEDED = 385
GAP_RANGE_SUPERSEDED = (297, 499)
GAP_MEDIAN_MEASURED_IN_BAND = 180

# Pair counts per episode -> ~3.5k-6k tokens/doc (card s2.3), ~25-35x
# the corpus-clock bar that killed `dharm` at 155.6 tok/doc.
PAIRS_MIN, PAIRS_MAX = 30, 46

# P(this pair's environment turn is a repeat-failure) -- THE CLOCK KNOB,
# and the only quantity the card permits to move after freeze.
#
# Tuned at plan time ($0) against the s2.2a target: at ~94 tok/pair a
# realised event rate of ~0.26 puts an event every ~3.9 pairs => gap
# median ~385 tok. Scan that produced it (300 docs, seed 0):
#     P_REPEAT  0.33 -> gap 301   0.28 -> 351   0.24 -> 412   0.20 -> 485
# The projection assumes realised assistant length ~0.8x cap; THE PILOT
# IS THE ARBITER and re-tunes this against measured `claim_zone`.
P_REPEAT = 0.26
P_FAIL_NEW = 0.34            # remainder is OK

# ⚑ ASSISTANT TURN LENGTH IS THE DOMINANT CLOCK KNOB — measured, and
# it is NOT what the frozen card first assumed.
#
# `f` is the fraction of ELIGIBLE (assistant) tokens within T=64 of an
# event. Eligible tokens are not uniform over the stream: they sit in
# assistant turns, offset from the preceding event by the masked
# environment turn. So a LONG assistant turn pushes most of its own
# tokens past T=64 even when the inter-event gap is short.
#
# Measured in `retryesc_gen/dry_run.py` (40 docs, stub prose, seed 0):
#     len band     P_REPEAT 0.26      P_REPEAT 0.32
#     (60,120)     f 0.1119           f 0.1230
#     (45, 90)     f 0.1485           f 0.1629
#     (35, 70)     f 0.1896   <--     f 0.2079
#     (28, 56)     f 0.2378           f 0.2609
#
# Turn length moves `f` by ~0.13 across that range; P_REPEAT moves it by
# ~0.02. So the clock is bought with the TOKEN CLOCK, not by making the
# agent fail more often — which is what card § 2.3 argued on principle
# and what this measurement now supports on evidence.
LEN_LO, LEN_HI = 35, 70

# ── the GLOBAL strategy pool (shared by every task, deliberately) ───
# Generic remediation moves. Shared across tasks so strategy tokens
# cannot identify the task, and task tokens cannot identify difficulty.
#
# ⚑ POOL SIZE IS A POSITION CONTROL, not a cosmetic choice. With a
# small pool the strategies EXHAUST partway through an episode, after
# which every failure is necessarily a repeat -- which would make
# event-status a function of position and hand the probe a positional
# confound (the trap that killed `reask_hr`). At 24 strategies against
# ~15.6 expected fresh draws per episode, exhaustion is rare; the
# plan-time receipt `exhaustion_check` measures it rather than assuming.
STRATEGIES = [
    "increase the timeout and retry",
    "clear the cache directory and rebuild",
    "pin the dependency to an earlier version",
    "run the step with verbose logging enabled",
    "reset the local state and start from a clean checkout",
    "reduce the batch size",
    "re-resolve the lockfile from scratch",
    "disable the parallel execution path",
    "re-authenticate and reissue the request",
    "fall back to the single-threaded code path",
    "bypass the proxy and connect directly",
    "regenerate the configuration from the template",
    "widen the retry budget for the failing stage",
    "switch the serialisation format for this step",
    "drop the optional post-processing pass",
    "recreate the working directory from the manifest",
    "force a full reindex rather than an incremental one",
    "raise the file-descriptor limit for the process",
    "roll the schema forward one revision and retry",
    "isolate the step in its own subprocess",
    "normalise the input encoding before the call",
    "replace the streaming reader with a buffered one",
    "invalidate the upstream response cache",
    "serialise the two conflicting writers",
]

# ── task pool (vocabulary carrier; NEVER consulted by the schedule) ─
# 16 tasks, matching evalage's topic count. Each carries distinct
# surface vocabulary. Because difficulty is ASSIGNED by `plan_schedule`
# before the task is drawn, no task is systematically harder than
# another -- which is precisely the property `retryesc` lacked, where
# 80 real tasks had genuinely different difficulty and their nouns
# leaked it (unigram 0.689-0.716).
TASKS = [
    "the image thumbnailer emits corrupted output for CMYK inputs",
    "the nightly export job writes an empty CSV on the first of the month",
    "the websocket client drops its connection after roughly ten minutes",
    "unit tests pass locally but fail in the container with a locale error",
    "the migration script leaves two rows with duplicate primary keys",
    "the PDF renderer omits embedded fonts on the second page onward",
    "the rate limiter admits bursts that exceed the configured ceiling",
    "the search index misses documents whose titles contain diacritics",
    "the audio transcoder desynchronises video by a few frames",
    "the cache warmer evicts entries it has just written",
    "the geocoder returns coordinates swapped in the southern hemisphere",
    "the log shipper truncates messages longer than a few kilobytes",
    "the spreadsheet importer misreads dates in day-first locales",
    "the retry wrapper multiplies its backoff instead of adding to it",
    "the archive extractor mishandles symlinks that point outside the root",
    "the metrics collector reports a counter that decreases over time",
]

# ── environment turn templates ──────────────────────────────────────
# ⚑ BINDING: this pool is drawn from WITHOUT consulting repeat-status.
# A repeat failure and a first-time failure are textually identical in
# distribution. Breaking this breaks the candidate (card s3).
FAILURE_TEXTS = [
    "Command exited with status 1. The operation did not complete.",
    "The previous command failed. No changes were applied.",
    "Error: the step terminated before finishing.",
    "The previous command timed out and was terminated.",
    "Exit code 2. The requested operation was rejected.",
    "The operation failed. State is unchanged.",
]
SUCCESS_TEXTS = [
    "The command completed successfully.",
    "Done. The step finished without errors.",
    "Exit code 0. The operation completed.",
]

TASK_TEMPLATE = (
    "You are working in a terminal environment on the following task.\n\n"
    "TASK: {task}\n\n"
    "Work step by step. After each attempt you will be shown the result."
)
# The scaffold ASSIGNS the strategy; the model writes the agent's
# reasoning for it. This keeps repeat-status grounded in the text
# (the strategy is named) while leaving it unreadable inside a window.
DIRECTIVE_TEMPLATE = "Next, attempt this approach: {strategy}"


@dataclass
class Pair:
    """One environment/agent exchange, fully determined at plan time."""
    outcome: int
    strategy: str
    env_text: str
    max_new: int

    @property
    def is_event(self) -> bool:
        return self.outcome == FAIL_REPEAT


@dataclass
class EpisodePlan:
    pairs: list[Pair]
    task: str                     # drawn AFTER the schedule, never before
    meta: dict


def pairs_as_dicts(pairs: list[Pair]) -> list[dict]:
    """JSON-serializable form of a pair list.

    ⚑ Required because checkpointing is BLOCKING for every generation
    card and `save_ckpt` writes plain JSON — a dataclass in the doc plan
    crashes the first checkpoint. Caught in `dry_run.py` at $0, which is
    where it should be caught."""
    return [{"outcome": p.outcome, "strategy": p.strategy,
             "env_text": p.env_text, "max_new": p.max_new} for p in pairs]


def is_event(pair: dict) -> bool:
    """Event iff the environment reports failure of an ALREADY-TRIED
    strategy. Reads the plan, never the text — the text is deliberately
    uninformative about repeat-status (card § 3)."""
    return pair["outcome"] == FAIL_REPEAT


def plan_schedule(rng: np.random.Generator, n_pairs: int) -> list[dict]:
    """Draw the outcome schedule ALONE — no task, no vocabulary.

    This function must not see the task. That independence is what
    makes `unigram ⊥ label` a property of the generator rather than a
    band to squeak past (card s4)."""
    used: list[int] = []
    out = []
    for _ in range(n_pairs):
        u = rng.random()
        if u < P_REPEAT and used:
            outcome = FAIL_REPEAT
            sid = int(rng.choice(used))            # an ALREADY-TRIED one
        elif u < P_REPEAT + P_FAIL_NEW or not used:
            outcome = FAIL_NEW
            fresh = [i for i in range(len(STRATEGIES)) if i not in used]
            if not fresh:                          # pool exhausted -> repeat
                outcome, sid = FAIL_REPEAT, int(rng.choice(used))
            else:
                sid = int(rng.choice(fresh))
                used.append(sid)
        else:
            outcome = OK
            if used:
                sid = int(rng.choice(used))
            else:
                sid = int(rng.integers(len(STRATEGIES)))
                used.append(sid)
        out.append({"outcome": outcome, "sid": sid})
    return out


def plan(rng: np.random.Generator, tasks: list[str], n_docs: int,
         len_lo: int = LEN_LO, len_hi: int = LEN_HI) -> list[EpisodePlan]:
    """Full per-document plan. DRAW ORDER IS LOAD-BEARING:
    schedule FIRST (difficulty), task SECOND (vocabulary)."""
    plans = []
    for _ in range(n_docs):
        n_pairs = int(rng.integers(PAIRS_MIN, PAIRS_MAX + 1))
        sched = plan_schedule(rng, n_pairs)          # <-- no task in scope
        task = tasks[int(rng.integers(len(tasks)))]  # <-- never consults sched
        pairs = []
        for s in sched:
            # ⚑ env text drawn WITHOUT consulting repeat-status
            if s["outcome"] == OK:
                env = SUCCESS_TEXTS[int(rng.integers(len(SUCCESS_TEXTS)))]
            else:
                env = FAILURE_TEXTS[int(rng.integers(len(FAILURE_TEXTS)))]
            pairs.append(Pair(outcome=s["outcome"],
                              strategy=STRATEGIES[s["sid"]],
                              env_text=env,
                              max_new=int(rng.integers(len_lo, len_hi + 1))))
        n_ev = sum(p.is_event for p in pairs)
        plans.append(EpisodePlan(
            pairs=pairs, task=task,
            meta={"n_pairs": n_pairs, "n_events": n_ev,
                  "event_rate": n_ev / n_pairs}))
    return plans


def exhaustion_check(plans: list[EpisodePlan]) -> dict:
    """POSITION control receipt. If the strategy pool exhausts mid-episode
    every later failure is forced to be a repeat, so event-status becomes
    a function of position — the confound that killed `reask_hr`.

    Also reports the event rate in the first vs last third of an episode.
    Some rise is STRUCTURAL and unavoidable (the first attempt can never
    be a repeat); what must not happen is the rate being *pinned* by
    exhaustion."""
    exhausted, first_third, last_third = [], [], []
    for p in plans:
        n = len(p.pairs)
        exhausted.append(len({q.strategy for q in p.pairs})
                         >= len(STRATEGIES))
        k = max(1, n // 3)
        first_third.append(np.mean([q.is_event for q in p.pairs[:k]]))
        last_third.append(np.mean([q.is_event for q in p.pairs[-k:]]))
    return {
        "frac_docs_exhausting_pool": float(np.mean(exhausted)),
        "pool_size": len(STRATEGIES),
        "event_rate_first_third": float(np.mean(first_third)),
        "event_rate_last_third": float(np.mean(last_third)),
        "rise": float(np.mean(last_third) - np.mean(first_third)),
        "note": "some rise is structural (attempt 1 cannot be a repeat); "
                "the screen's balanced manifest stratifies on position, "
                "and position_auc is band 3 (<= 0.95)",
    }


def schedule_independence_check(plans: list[EpisodePlan]) -> dict:
    """PLAN-TIME receipt for card s4: is the event rate independent of
    the task? A task whose event rate differs from the pool mean is a
    difficulty signal that vocabulary could read.

    Returns the spread and a `stop` flag. This is the check `retryesc`
    never had, and its absence is what killed it."""
    by_task: dict[str, list[float]] = {}
    for p in plans:
        by_task.setdefault(p.task, []).append(p.meta["event_rate"])
    rates = {t: float(np.mean(v)) for t, v in by_task.items()}
    if not rates:
        return {"n_tasks": 0}
    vals = np.asarray(list(rates.values()), dtype=float)
    cv = float(vals.std() / vals.mean()) if vals.mean() else float("nan")
    return {
        "n_tasks": len(rates),
        "per_task_event_rate": rates,
        "mean": float(vals.mean()),
        "spread_max_minus_min": float(vals.max() - vals.min()),
        "cv": cv,
        # Same PROPOSED-not-ratified status as elicit_lib.VOCAB_CV_BAR.
        "cv_bar_proposed": 0.35,
        "stop": bool(cv > 0.35),
    }
