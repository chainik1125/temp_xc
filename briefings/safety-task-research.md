# SAFETY-RELEVANT TASK RESEARCH — mac-c briefing (status: active)

**Source: 2026-07-27 team meeting (Dmitry) + Han's directive. Owner:
mac-c. Venue: mac-local CPU + the `clew` skill (Han's curated
registry, ~1000 safety/interp works) — READ-ONLY, $0 GPU.**

## The ask

Dmitry's ruling: the task hunt must find **SAFETY-RELEVANT** tasks.
Backtracking, refusal, and emergent misalignment are safety-relevant;
question-mark distance and turn-length trend are TOYS (ttrend →
appendix, out of the rebuttal). New tasks can enter the rebuttal
thread until **Aug 3**, and the arXiv/ICLR version after that.

Your deliverable: a **candidate task menu** (10–20 entries, ranked)
of safety-relevant trailing quantities the TXC could recover, each
entry carrying:

1. **The trailing quantity** — a per-token-silent state that
   accumulates/decays over context (the trailing-functional recipe:
   offset-weighted functionals of sparse silent events; surface-quiet
   preferred; order-carriage plausible).
2. **Safety motivation with citations** — which papers ground it
   (clew hits; arXiv IDs; alignment-blog posts count and are IN clew
   but NOT in S2).
3. **Label construction sketch** — corpus, event definition,
   kernel/functional, ground-truth source (rule-based >> LLM-judge;
   note API-budget if judge-gated).
4. **Expected traps** — visible-cue floor (what surface markers leak
   it), identity trap, position trap; kill-precedent lookups (our § 8
   record: density-family dies to g_agg on Ward; adoption-family dies
   to its floor at T32; etc.).
5. **Feasibility class** — screenable-this-week (existing corpora /
   caches) vs needs-pipeline (elicitation, judge).

Seed directions (grow/replace freely): refusal-pressure buildup
across a conversation; jailbreak-progression state (multi-turn
attack escalation); sycophancy drift under user pushback;
deception/consistency debt (claims made earlier constraining now);
persona/instruction drift from system prompt over distance;
situational-awareness accumulation; scheming/sandbagging indicators
over long context; EM-adjacent harmful-advice drift in specialist
domains; CoT-faithfulness divergence trailing state; reward-hacking
onset in agentic traces.

## Tools

**Primary: the `clew` skill** (`~/.claude/skills/clew`) — keyword +
SPECTER2 semantic search + citation graphs + cached full text over
Han's Zotero collection. Use it first; it indexes the alignment-blog
material S2 lacks.

**Fallback: direct Semantic Scholar** (Han's key, ON LOAN — treat
accordingly). Load from the macOS keychain, env-only:

    export S2_API_KEY="$(security find-generic-password -s s2-api-key -w)"
    [ -n "$S2_API_KEY" ] || { echo "keychain read failed — STOP, tell Han"; }

NON-NEGOTIABLE hygiene: never echo/print the value; never write it
into files, logs, reports, or scripts; never pass it as a
command-line argument (process listings leak). Header-inject from the
env var only (`-H "x-api-key: $S2_API_KEY"`). Rate limit is
1 req/s CUMULATIVE across all users of the key incl. Han's clew
syncs — space ≥ 1.1 s, use `fields=`, prefer `POST /paper/batch`
(≤500 ids) over loops, paginate citations/references via `offset`
until no `next`, honor Retry-After, cap retries ~3, give up loudly.
Do NOT run a direct S2 workload while a `clew sync` is running.

## Discipline

- Research inventory, NOT a freeze: screens/cards/verdicts stay with
  the hunt executor (runpod-a) under the standard discipline.
- Every candidate cites its sources; no numbers invented; label-side
  pre-measures are the NEXT step, not this one.
- Deliverable: `experiments/explorations/task_hunt/SAFETY_TASK_MENU.md`
  + a LOG entry (PTR, mac-local reviews on push).
- Secondary item (bounded, after the menu): **txc_pro recovery dig**
  — the locked hparams live at
  `origin/final-aniket:purified/configs/locked_archs.yaml`
  (registry tag `phase5b_subseq_h8`: d_sae 18432, T_max 10,
  t_sample 5, n_matryoshka 8, contrastive_shifts [1,2] +
  inverse-distance weighting, auxk_alpha 0.03125), but
  `txc_pro.py` itself was LOST in purification. Recover the class
  implementation + any real probing T-scaling evidence from the
  pre-purification trees (phase-5b era; your census/HF-snapshot
  infrastructure applies). Report provenance-grade: what it was,
  where it lived, what evidence existed (A12-aware: distrust any
  shipped T-ordering).

*Delete this briefing when the menu + dig are delivered.*
