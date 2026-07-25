# RECORD — relational (regime-3) task hunt

Agent `runpod`, 2026-07-25. Exploration `experiments/explorations/relational/`.
Prime directive: **a sound verdict, never a win.**

**The question.** Every real-world margin in the TempBench paper is small because
every real task in it sits in regime 1 or 2 of the synthetic program's coordinate
system: the latent is either per-token-readable (nobody separates) or
linear-in-window (every window arch ties). Regime 3 — latents needing
cross-position comparison — is the only regime that separates window
architectures from each other, and the paper's own TXC (post-squash,
`u = σ(Σ_τ W^(τ) x_{t+τ})`) is the unique family that can read it. Is there a
**real-activation** task in regime 3?

**The instrument.** A balanced-marginal equality label. `stimuli.py` builds a
2×2 over two constituents so `label = [A == B]` has flat marginals at both
positions. Then, by the additive-code theorem
(`synthetic/changepoint/bench_record.md` § 3), *every* additive-over-position
code — per-token SAE, T-SAE, Stacked, MLC, TXC-pre, and any pooled per-token
code — is at a provable chance floor, and only a position-mixing nonlinearity can
read the label.

---

## § 1 — Methodological corrections made during the run (all disclosed)

1. **v1 stimuli discarded for memorisation.** The first generator emitted 2,400
   rows from **80 distinct texts** (60× duplication). Every arm scored AUC 1.000,
   *including windows that could not see constituent A*. That is the
   `signed_motion` lesson: a `T·d`-dimensional probe memorises a handful of
   distinct test points. The generator now asserts `distinct_texts == n_items`;
   v2 has 5,760 (agreement) and 4,800 (contradiction) all-distinct items across
   24 and 20 lexical groups, split **by group** so no lexical item is shared
   between train and test. The v1 numbers were discarded, not reinterpreted.

2. **The IN/OUT control is what caught it.** Because the generator records the
   exact character offset of each constituent, every row has a known token
   distance, so the same label at the same probe position can be scored
   separately for windows that *reach* constituent A and windows that do not.
   This is a causal control on binding and it is immune to the
   shuffle-gap-grows-with-`T` confound that bit task_hunt candidate 2
   (`task_hunt/RECORD.md` § 2). It is now a standing arm.

3. **A design error in the gate itself, corrected mid-run.** A linear probe on
   the *flattened* window **is** an additive-over-position code, so it can never
   certify headroom that only a coincidence code could exploit. The gate now runs
   MLP arms as well, and the headline statistic is

   > `nonlinear_residual = window_MLP − max(window_linear, per_token)`

   i.e. the part of the label that is present in the window but **not** linearly
   available from any per-position decomposition. That is the quantity the
   theorem actually speaks to.

4. **The documented tokenizer trap fired.** `AutoTokenizer` returned the slow
   `LlamaTokenizer` while reporting `is_fast=True`, with overlapping offsets
   (`(11,14),(13,14)`), and every row was silently rejected — exactly
   `task_hunt/RECORD.md` § 4 note 5. `is_fast` is not a valid guard; the gate now
   validates that offsets are monotone, non-overlapping, and recover the exact
   span, and uses `PreTrainedTokenizerFast`.

5. **Model substitution.** `gemma-2-2b-it` (the paper's § 5.1 probing model) is a
   gated repo this account cannot access, so the gate runs the paper's § 5.2
   model, `DeepSeek-R1-Distill-Llama-8B`. Paper-compatibility is preserved;
   probing-comparability awaits a licence acceptance.

6. **Freeze order.** The candidate-4 card is committed (`74df8f7f`) **before any
   contradiction cell existed** — git-provable. The candidate-5 pilot ran before
   that commit, so its freeze is only artifact-timestamped; its predictions are
   quoted and scored below, and the weaker provenance is stated rather than
   glossed.

---

## § 2 — Candidate 5: agreement attraction — **KILL** (both rules fired)

Label: `[number(head noun) == number(verb)]`, e.g. *"The inspector noted that the
keys beside the doors past the checkpoint is broken."* Probe row at the verb
token. 5,760 distinct items, 24 lexical groups, head→verb distance 4–12 tokens
(median 7).

**Label-side triage — PASS.** AUC from the head number 0.500, from the verb
number 0.500, from length 0.503, from the inter-constituent gap 0.503; cells
exactly equal; zero duplicate texts. The label is provably unreadable from any
single position's *content*.

**Result** (R1-Distill-Llama-8B, stratum `all`, bootstrap CIs, 3σ = 3× the SD of
a 4-draw label-permutation null):

| layer | T | per-token [95% CI] | win-linear | win-mean | win-MLP | g | nonlinear residual | 3σ null |
|---|---|---|---|---|---|---|---|---|
| 0 | 4 | 0.495 [0.464, 0.533] | 0.506 | 0.462 | 0.486 | +0.012 | −0.021 | 0.065 |
| 0 | 8 | 0.495 [0.464, 0.533] | 0.503 | 0.508 | **0.749** | +0.009 | **+0.246** | 0.082 |
| 2 | 4 | 0.983 [0.978, 0.988] | 0.969 | 0.934 | 0.989 | −0.014 | +0.006 | 0.064 |
| 2 | 8 | 0.983 [0.978, 0.988] | 0.948 | 0.858 | 0.990 | −0.035 | +0.007 | 0.033 |
| 4 | 8 | 1.000 [1.000, 1.000] | 1.000 | 0.998 | 1.000 | −0.000 | −0.000 | 0.029 |
| 8 | 8 | 1.000 [1.000, 1.000] | 1.000 | 0.999 | 1.000 | −0.000 | −0.000 | 0.008 |
| 16 | 8 | 1.000 [0.999, 1.000] | 0.998 | 0.992 | 0.999 | −0.001 | −0.001 | 0.028 |
| 24 | 8 | 0.999 [0.999, 1.000] | 0.997 | 0.992 | 0.994 | −0.003 | −0.005 | 0.016 |

**Verdict: KILL.** K1 fires (per-token is within 0.02 of the best window arm at
every layer ≥ 2) and K2 fires (`nonlinear_residual ≤ 3σ_null` in every cell from
layer 2 onward). **Agreement equality is converted.** The model computes the
relation itself — using cross-token information, since it is absent at the
embeddings — and deposits the result at the current position, so the additive
ceiling already contains everything a window code could offer.

**Conversion is fast.** Per-token goes 0.495 → 0.983 → 1.000 over layers 0/2/4
and holds to layer 24. For contrast, the round-1 forbidden-word candidate took
≈ 20 layers to climb +0.13 (`task_hunt/LOG.md`). This is the
*built-and-immediately-linearised* g(ℓ) shape, and this is its first **relational**
instance — previously it was recorded only for a scalar pressure signal.

### § 2b — The positive control (why this kill is informative, not blind)

At **layer 0**, `T = 8`:

| stratum | rows | per-token | win-linear | win-MLP [95% CI] | nonlinear residual | 3σ |
|---|---|---|---|---|---|---|
| all | 5,760 | 0.495 | 0.503 | 0.749 [0.721, 0.776] | **+0.246** | 0.082 |
| **A inside window** | 3,407 | 0.503 | 0.487 | **0.772 [0.736, 0.805]** | **+0.269** | 0.077 |
| **A outside window** | 2,353 | 0.511 | 0.515 | 0.498 [0.451, 0.547] | −0.017 | 0.056 |

Three things at once, on real activations:

1. **The additive-code theorem holds empirically.** Both constituents are present
   in the window, yet every *linear* readout of a per-position decomposition sits
   at chance (0.487–0.515). This is the changepoint bench's provable result,
   reproduced outside the synthetic setting.
2. **A cross-position nonlinearity reads it** — 0.772, a **+0.27** margin over the
   additive ceiling, with non-overlapping CIs.
3. **It is binding, not capacity.** The same MLP on the same rows collapses to
   0.498 when the window cannot reach constituent A, and at `T = 4` (where no row
   reaches it) the residual is −0.021. A dose–response in window size, with the
   control at chance.

So the instrument **demonstrably fires when regime-3 headroom exists**. Its zero
from layer 2 onward is therefore a statement about the model, not about the
probe — which is precisely what a kill needs in order to be worth reporting.

### § 2c — Frozen predictions, scored

- Prior `P(violent) = 0.30`, with "expect 0.75 vs 0.92 rather than chance vs 0.9"
  → **FALSIFIED**, in the *more converted* direction: 1.000 vs 1.000.
- "Per-token is actively misled by the distractor" → **FALSIFIED**. The distractor
  costs the model nothing at these depths; attraction does not show up as
  degraded linear decodability of the relation.
- The design-level premise ("the constituents are present but not combined") →
  **CONFIRMED, but only at layer 0.** True where the model has not yet computed
  the relation; false everywhere a dictionary would actually be trained.

---

## § 3 — Candidate 4: contradiction / fact-consistency — running

Card frozen at `74df8f7f`. Predictions P1–P5 and kill rules K1/K2 are in
[`cards/contradiction_xor.md`](cards/contradiction_xor.md). Mention distance is
17–37 tokens (median 26), so `T = 32/64` are the informative cells and layer 0
is included as a now-known positive control.

---

## § 4 — Resources

Peak VRAM **7.87 GB** across all probe cells (guard 60 GB); model caching peaked
at 16.9 GB. **Zero OOM events.** Free disk held at 12.0–12.4 GB against a 12 GB
abort floor — no further model pulls were made, and activations are held in RAM
rather than written to the volume. A disk scan found nothing belonging to this
exploration that could be freed; the large items on the volume
(`/workspace/ocean` 76 GB, a 15 GB Qwen coder model, 16.6 GB of role-probes
venvs) belong to other work and were left untouched.

---

## § 5 — What this means for the paper (so far)

The candidate-5 kill is **not** a null result for the reviewers' question; it is
the beginning of the atlas the paper's Limitations section says is missing:

- **Reviewer bbby** asks whether cross-position weight sharing is responsible for
  the gains. § 2b gives the sharp form of the answer *as a measurement*: on a
  label whose readout requires a cross-position conjunction, every additive code
  is at chance while a position-mixing nonlinearity reaches 0.77. The
  architecture class matters — where the model has not already done the work.
- **Reviewer 4z15** asks to isolate the temporal contribution from generic
  crosscoder capacity. The IN/OUT control is that isolation, and it is stronger
  than an ablation: identical rows, identical probe, identical budget, and the
  effect is present only when the window spans the two constituents.
- **Reviewer EAxU** calls the results preliminary because TXC's case rests on
  backtracking. The honest reading of § 2 is that this is *expected*: relational
  latents that a model needs are linearised within a few layers, so temporal
  architectures can only earn their keep on latents the model declines to
  maintain. That is a predictive criterion, which is what was missing.

The scope limit is stated plainly: one model, one hookpoint family, English
templates, one label. This kills *this* label on *this* model. It does not
establish that no equality latent survives conversion anywhere — that is what the
remaining candidates test.
