# Mini-card — candidate 3: forbidden-word violation onset (Stage-1 screen)

**Status: FROZEN at commit (commit-then-run — no rollout has been
generated and no activation probed at this commit).** Agent: runpod-d.
Briefing: `briefings/task-hunt.md` candidate 3 (Han 2026-07-24).
Provenance: grounded. **SILOED** — design, priors and verdict here are
independent of Aniket's parallel forbidden-word work; none of his
results are inputs, and this card was written without consulting them.

## Data + labels (exact, zero judge cost)

Rollouts: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` on the public
CoT-Control **keyword-suppression** split (github.com/YuehHanChen/
CoTControl @ `5d78aeff`, `CoT-Control-QA/datasets/{gpqa,hle,
mmlu_pro_mini}_w_keyword.csv`; 1214 questions, keywords pre-validated
upstream against 24 baseline traces per question). Prompt + keyword
choice replicate that commit verbatim (`generate.py` docstring lists
the exact lines); temperature 0.6 / top_p 0.95 (R1 card), max 2048
tokens, seed 0, one rollout per question, no adversarial pressure.

**Label = offset to the FIRST occurrence of the keyword in the think
region**, matched by case-insensitive whole-word regex (upstream's
`_count_word_occurrences` rule), keyword only — synonyms are the
grader's compliance instrument, not a position label. For horizon
D ∈ {4, 8, 16} (frozen):

- **positives:** tokens whose distance to the first keyword token lies
  in [D, 2D) — the anticipation band at that horizon;
- **negatives:** think-region tokens of the same rollouts at distance
  > 64 tokens before the first occurrence, or from non-violating
  rollouts, capped 5 : 1 (the `bt_freq`/conversion-depth negative
  convention);
- everything at or after the first occurrence is **excluded** (the
  keyword itself is trivially readable; this screen is anticipation
  only).

R1-family controllability is ≈ 0.1 % upstream ⇒ violations are the
common case and labels are dense. **Feasibility gate (checked before
the screen, not tuneable):** if the measured violation rate is < 30 %
or fewer than 200 rollouts violate, the screen is under-powered —
record that and either widen to a second rollout per question at the
same frozen settings (once, disclosed) or KILL as infeasible.

## Why non-ambience is CONTESTED (the card's crux)

Pre-violation the model "circles" the concept: semantically neighboring
tokens are each individually informative about the forbidden word, so a
single token's marginal may already carry the imminent-violation
pressure. That makes **per-token ≈ window at ALL horizons the
pre-registered kill**, not a surprise. Surviving it — a window margin
that grows with T *and* a positive order component — is the
non-ambient anticipation story, and it is the only outcome that counts
as a hunt result here.

## Screen protocol (frozen)

Cache one mid-depth layer (hs13 = resid_post L12, the g(ℓ) precedent)
of R1-Distill over its own rollouts (generator = reader here, unlike
the Ward substrate). Frozen `problib` stack, class-weighted, AUC
primary: per-token linear, window-flatten linear, window-MEAN linear
(g_agg/g_order), within-window-SHUFFLED linear, MLP-512 presence at
T = 16; permutation null seed 99. T ∈ {2, 4, 8, 16, 32}; split BY
ROLLOUT 80/20 (rng 7); rows capped per rollout (rng 13/14).

## Frozen predictions (STORY.md § 7 taxonomy)

- **P1 (threshold ladder):** recovery at horizon D turns on as T
  crosses D — g(T) at horizon D is ≤ 3 σ_null for T < D and clears it
  for T ≥ D, giving three staggered curves (D = 4 turns on earliest).
  This is the *second threshold ladder* the briefing predicts.
- **P2 (saturation):** each horizon's g(T) saturates once T exceeds the
  typical circling span; with no measured span at freeze time the
  prediction is saturation by T = 32 for D = 4 and D = 8, and possibly
  not yet reached for D = 16.
- **P3 (order):** g_order > 0 at the turn-on T for at least one horizon
  (the circling is a trajectory, not a static bag). If g = g_agg at
  every cell, the result is order-free evidence pooling — reportable
  but NOT the hunt's target.
- **P4 (ambience risk, the pre-registered kill):** per-token AUC within
  0.02 of window AUC at every T and every horizon ⇒ the circling is
  ambient ⇒ KILL.

## Falsifier / kill rule (pre-registered)

KILL if ANY of: (1) P4 fires; (2) g(T) ≤ 3 σ_null at every (T, D);
(3) no horizon shows turn-on ordering (g(T) unchanged across
D ∈ {4, 8, 16} — the "anticipation" is horizon-independent, i.e. a
rollout-level property leaking into every position, the EM trap);
(4) the feasibility gate fails and a second rollout pass does not fix
it. Verdict → one paragraph in `../LOG.md`.
