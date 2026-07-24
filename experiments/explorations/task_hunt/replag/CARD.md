# Mini-card (FROZEN pre-screen) — repetition-lag Δ

**Candidate 1, task-hunt arm B** (`briefings/task-hunt-b.md`).
Agent: `runpod-e`. Frozen by commit BEFORE any screen run; the screen
(`screen.py`) executes exactly this card. Governing protocol:
`briefings/task-hunt.md` Stage 1; probe stack: `conversion_depth/problib.py`
(frozen — no per-target retuning).

## Latent + labels (exact, zero-judge)

Corpus: the committed pinned fineweb sample
(`experiments/explorations/synthetic/expansion/data/fineweb_sample.json`,
400 docs). Per model tokenizer: join each doc's sentences with a space,
tokenize, cut into non-overlapping model-visible sequences of length 128
(BOS + 127 content tokens for gemma/llama; 128 content tokens for gpt2,
no BOS). Labels computed on exactly the token grid the model sees,
**within-sequence only** (the model's visible context).

For content position p: **Δ(p) = p − q** where q is the largest content
position < p with `tok[q] == tok[p]` (n = 1); ∞ if none in-sequence.
Builder also emits n = 2 (adjacent-bigram) labels — NOT screened here;
reserved as a pre-named robustness addendum if the candidate survives.

Buckets: **B4 = Δ∈[2,4], B8 = Δ∈[5,8], B16 = Δ∈[9,16], B32 = Δ∈[17,32]**.
Δ = 1 excluded (previous-token heads convert it trivially at every scale
— uninformative for the ladder).

Anchor eligibility: ≥ 64 content tokens of in-sequence history
(H(p) ≥ 64), so every T ≤ 32 window fits and the negative definition
below is meaningful at every anchor.

## Probe targets (5)

- **det4 / det8 / det16 / det32** (binary, PRIMARY LADDER): positive =
  anchor with Δ ∈ bucket; negative = anchor with **Δ > 64 or ∞**
  (novel-in-context under the same eligibility).
- **lag4** (4-class, ORDER READOUT): among repeated anchors, class =
  bucket ∈ {B4, B8, B16, B32}.

**Confound controls (the strict per-position-baseline lesson):**

1. **Anchor-identity matching** — within each task and each split, the
   anchor-token-id histogram is made EXACTLY equal across classes
   (per (token_id, position-bucket) cell, count = min over classes,
   seeded subsample). Kills "rare tokens are novel, function words
   repeat" — the ambient token-identity route to the label.
2. **Position matching** — matching cells are (token_id, H-bucket) with
   H-buckets {[64,80), [80,96), [96,112), [112,128)}. Fallback to
   token-id-only matching if a task yields < 300 rows/class/split under
   joint matching (recorded if used).
3. Split by DOCUMENT 80/20, rng(7); matching runs within each split.
   Caps (seeded): 4000 train / 1500 test rows per class.
4. Any (model, task) below 300 matched rows/class/split after fallback
   ⇒ recorded "insufficient matched rows", task skipped (a data
   limitation, not a verdict).

Builder null (label sanity): within-sequence token shuffle ⇒ Δ
distribution under exchangeability, reported next to the real one
(real text must be far heavier at small Δ).

## Why non-ambient (regime-2/3-shaped)

Δ is a TWO-position property: no single token knows its own lag — the
label couples position p to position p−Δ. A window spanning both
positions has physical access; a single-position reader has access only
insofar as the model already CONVERTED repetition structure into
per-token features (induction heads: "I am a repeat", match strength /
recency). Equality-matching between two positions is quadratic in the
activations, so we predict the window's advantage is largely
**nonlinear** (regime 3): the linear window probe may stay near the
per-token ceiling while the window MLP shows the ladder. That is why the
frozen stack's MLP presence checks are load-bearing here, and why the
Stage-2 archs that can express cross-position matching (TXC encoder over
the window) are predicted to separate from per-token-decoded T-SAE.

## Model × layer (frozen screen cells)

Screen across SCALE, one mid-depth layer each (g(ℓ) precedent):

| model | layers | screen layer | cached alternates |
|---|---|---|---|
| gpt2 (124M) | 12 | resid_post L6 = hs7 | hs4, hs10 |
| google/gemma-2-2b (base) | 26 | resid_post L13 = hs14 (phase-5 convention) | hs8, hs20 |
| NousResearch/Meta-Llama-3.1-8B (base) | 32 | resid_post L13 = hs14 (measured g(ℓ) peak) | hs8, hs22 |

Alternates are cached (cheap, same forward) but NOT screened; they may
be probed only in a labeled post-hoc addendum.

## Frozen probe grid (per model × target)

- per-token linear + per-token MLP(512) — T-independent, once.
- T ∈ {2, 4, 8, 16, 32}, right-edge windows (p−T+1 … p):
  window linear (flatten), window-MEAN linear (order-free decomp),
  window SHUFFLED linear (context slots permuted per-row with seeded
  rng, **anchor slot fixed at the right edge** — the order-free ceiling
  that keeps anchor identity).
- window MLP + shuffled-window MLP at T ∈ {8, 32} (presence checks).
  Pre-authorized escalation: if the linear pair is blind (no ladder)
  while the T∈{8,32} MLPs separate, extend MLPs to all T for the money
  plot — noted in the LOG before running, no other change.
- Permutation nulls (NULL_SEED 99) on the linear pair at T = 16 only.
- Binary metric: rank-AUC (class_weight=True); lag4 metric: acc_test
  (balanced by construction) + per_class.

Coverage bookkeeping: cov(B, T) = P(Δ ≤ T−1 | Δ ∈ B) computed from the
realized label mass and reported beside every cell (the ladder's x-axis
is coverage, its onset the T that first covers the bucket).

## Frozen predictions (STORY.md § 7: threshold-ladder family)

P1 **Ladder**: for detection bucket B, the window−token gap is ≈ 0
   where cov(B,T) = 0 and turns on as T crosses the bucket (gap grows
   with coverage); onset T is ordered B4 < B8 < B16 < B32. The gap may
   live in the MLP pair rather than the linear pair (regime-3
   expectation above); either pair counts, same pair across buckets.

P2 **Per-token flat in T** (definitional control) and **per-token AUC
   ordered by scale**: llama ≥ gemma ≥ gpt2 on each detection target
   (conversion grows with size).

P3 **Scale ordering of the unconverted gap** (the briefing's frozen
   prior): at full coverage (T = 32), window−token gap LARGER in
   smaller models: gpt2 ≥ gemma ≥ llama.

P4 **Detection is aggregation-shaped**: shuffled-window ≈ ordered
   window on det* (bag membership suffices) — g_order ≈ 0.
   **lag4 is order-shaped**: shuffled-window collapses toward the
   per-token ceiling on lag4 — g_order > 0. This dissociation is the
   order receipt; Stage 2's shuffle ablation uses the lag readout.

P5 window-MEAN < shuffled window on det* (the mean dilutes the anchor;
   shuffled keeps it) — a decomposition detail, not a keep/kill input.

## Falsifier / KEEP-KILL (frozen)

- **KEEP** iff in ≥ 1 model, ≥ 2 adjacent buckets show the P1 ladder:
  gap ≤ 0.02 AUC at cov = 0, ≥ 0.05 AUC at full coverage, onsets
  ordered — in a consistent probe pair — AND the P2 flatness control
  holds.
- **KILL** if no model shows the ladder in any pair (including "window
  MLP shows nothing anywhere": recorded as *no raw-access ladder at the
  frozen probe budget* — the honest kill).
- P3/P4 outcomes are recorded findings either way and do not gate
  KEEP/KILL.
- Stage 2 (if KEEP): the single best (model) cell, canonical runner,
  T ∈ {2, 4, 8, 16}, seeds {1, 2, 42}, per-arch predictions frozen
  before training.
