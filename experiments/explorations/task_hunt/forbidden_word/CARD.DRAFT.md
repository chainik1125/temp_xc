# DRAFT mini-card — forbidden-word violation onset (NOT frozen)

**Candidate 3, task-hunt arm A** (`briefings/task-hunt.md`). Drafted by
`runpod-b` (prep deliverable 5) so the running agent never waits on a
spec. **The running agent (`runpod-d`) freezes its own card** — edit
freely, then rename to `CARD.md` in the freezing commit. **SILOED per
the briefing: design, priors, screen, and verdict must not condition on
(or wait for) Aniket's parallel forbidden-word work.**

## Latent + label spec (exact string match, zero judge cost)

Data: R1-Distill-Llama-8B rollouts on the public CoT-Control
keyword-suppression split (`docs/papers/reasoning_models_struggle.md`;
github.com/YuehHanChen/CoTControl — keywords pre-validated per
question). The model is instructed not to use the keyword in its CoT and
usually fails (R1-family controllability ≈ 0.1 % per the briefing ⇒
violations are the common case; measure the realized violation rate on
the generated rollouts and record it).

Per violating rollout: **first occurrence** of the keyword in the think
region by case-insensitive word-boundary match on the decoded text
(include the keyword's surface inflections only if CoT-Control's own
checker does — mirror THEIR match rule exactly and record it); char span
→ token index via offsets (the `build_ward_stream` convention). Every
earlier position gets the countdown d = t_first − p.

Targets (the onset-anticipation gate, em_onset D+ design):

- **withinD** for D ∈ {4, 8, 16} (binary, PRIMARY LADDER): positive =
  d ∈ (0, D]; negative = d > 64 (guard band (D, 64] excluded),
  within-rollout negatives primary (controls prompt/topic);
  never-violating rollouts as a separate, labeled negative pool only.
- Anchor eligibility, identity/position matching, doc-level splits,
  caps: reuse the `replag/CARD.md` control battery (the strict
  per-position-baseline lesson) — anchor-identity matching matters
  doubly here because pre-violation text is topically loaded.
- Exclude the instruction text and any prompt echo of the keyword from
  anchors (the hint-presence trap: a window spanning the PROMPT reads
  the keyword off the input — anchors must sit in generated think
  tokens only, and windows must not reach into the prompt).

## Ambience is CONTESTED (the card's crux — say it plainly)

The model "circles" the concept before violating: semantic-neighborhood
tokens are individually informative, so per-token probes have a real
route to the label. **Per-token ≈ window at ALL horizons is the
pre-registered kill** (circling is ambient). Surviving requires the
window advantage to carry an order component (g_order > 0): imminence
as a rising trajectory, not bag-of-neighborhood. A window win with
shuffled ≈ ordered is aggregation, not anticipation — record it as such
(it is not the KEEP story).

## Predicted T-pattern (STORY.md § 7: threshold ladder)

Recovery at horizon D turns on as T crosses D (window must span the
anticipation evidence at that range), onsets ordered D4 < D8 < D16,
saturating at the typical circling span (measure that span from the
labels — e.g. run length of neighborhood tokens before t_first — and
report it beside the ladder).

## Falsifier / KEEP-KILL (draft)

- **KEEP** iff ≥ 2 adjacent horizons show the ladder (gap ≈ 0 below
  coverage, ≥ 0.05 AUC at full coverage, ordered onsets) AND g_order > 0
  on the surviving cells.
- **KILL — ambient circling** if per-token ≈ window everywhere.
- **KILL — aggregation only** if the gap exists but shuffled ≈ ordered
  at every horizon.
- Queue discipline per the briefing: generation may run in the
  background, but screen only after candidates 1–2 have committed
  verdicts.

## Label-builder notes (for the freezing agent)

`labels/lib.py` already provides the manifest/split/balance machinery
(`balanced_manifest`, `doc_split`) and the countdown pattern mirrors
`dist_next_kw` in `conversion_depth/build_ward_stream.py` — build the
extractor as `labels/build_forbidden.py` with its own sanity tests,
committed before outputs, once rollouts exist on your volume.
