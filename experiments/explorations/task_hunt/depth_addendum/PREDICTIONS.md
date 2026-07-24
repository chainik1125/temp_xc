# BLIND directional predictions — early-layer addendum (task-hunt r2-e § 2)

**Status: FROZEN at commit, BEFORE any addendum cell has run (the
cand-3 depth-sweep precedent: script + predictions committed first;
git order is the evidence).** Agent: runpod-e. Script: `run_depth.py`
(this directory, same commit). Zero new data — cached activations +
frozen round-1 manifests and probe stack only. POST-HOC diagnostic by
construction: nothing here reopens any frozen round-1 verdict.

**The question:** does the temporal signal GROW at pre-conversion
depths? Two arms, two different g(ℓ) readouts:

- **Replag arm** — lag4 (4-class repetition-lag value, chance 0.25) on
  the three replag models, at every cached depth (gpt2 hs {4, 7, 10};
  gemma2-2b hs {8, 14, 20}; llama31-8b hs {8, 14, 22}; screen layers
  were hs7/hs14/hs14), T ∈ {4, 8}: tok-linear, window-linear,
  window-MEAN-linear, context-shuffled-linear — g_order(ℓ) =
  win − mean, shuffle drop = win − shuf.
- **Slope8 arm** — the hedging-trend target on both Ward readers
  (base, distill) at all 17 capture points (hs 0, 1, 3, …, 31),
  tok-linear + window-MEAN-linear at T = 64 — g_agg(ℓ) = mean64 − tok.

Round-1 anchors (screen layers, cited for calibration): lag4 gpt2 hs7
T4 tok 0.515 / win 0.500 / mean 0.430 (g_order +0.07); gemma hs14 tok
0.462, llama hs14 tok 0.430, both g_order ≈ +0.02–0.04 band; slope8
hs15 T64 mean 0.565 (distill) / 0.511 (base) vs tok 0.468 (both).

## Predictions — replag / lag4 (order signal at pre-conversion depth)

- **A1 (conversion runs forward):** per-token lag4 accuracy is LOWER
  at each model's early alternate (gpt2 hs4, gemma hs8, llama hs8)
  than at its screen layer — the per-position lag feature is built by
  mid-depth attention and does not exist yet.
- **A2 (the headline):** g_order(ℓ) = win − mean at T = 4 is LARGER at
  the early alternate than at the screen layer in ALL THREE models —
  before conversion, locating which context slot matches the anchor is
  order information a mean destroys, so the ordered window holds an
  advantage the model has not yet converted away. Magnitude for gpt2:
  early g_order ≥ +0.10 (screen +0.07); directional only for the
  other two.
- **A3 (the scale gap narrows early):** the cross-model spread of
  g_order NARROWS at the early layers — gemma/llama early g_order
  moves toward gpt2's, because the round-1 scale ordering (gpt2 ≫
  2B/8B) was a property of mid-depth conversion strength, not of the
  models' inputs. (The alternative — the spread persists early — would
  say big models never carry the order signal anywhere, an
  architecture-independent kill of the order route.)
- **A4 (conversion stays converted):** at the late alternates (gpt2
  hs10, gemma hs20, llama hs22) g_order stays at or below its
  screen-layer value (≈ 0 band for gemma/llama).
- **A5 (receipt coherence):** wherever g_order is large, the shuffle
  drop (win − shuf) is large with the same sign and ordering (the two
  order readouts agree).

## Predictions — slope8 / g_agg(ℓ) (aggregation signal vs depth)

- **B1 (aggregation is everywhere, not late):** g_agg(ℓ) > 0 across
  essentially the whole depth range on distill — pooled lexical
  hedging evidence exists from the earliest layers (the hedge state is
  lexically stamped), so the mean-probe's advantage is NOT a late
  phenomenon. Early (hs1/hs3) g_agg is within a factor of ~2 of the
  hs15 value.
- **B2 (no depth converts the trend — the WHY of the Stage-2 bet):**
  per-token slope8 stays LOW at ALL depths on both readers — tok(ℓ)
  never approaches the mean-probe level (no layer holds a
  per-position trend summary). This is the anti-conversion signature
  that makes hedging-LEVEL a genuine window task; if tok climbs
  anywhere deep, the Stage-2 layer choice was lucky and the record
  must say so.
- **B3 (readers agree early, diverge late):** |g_agg(distill) −
  g_agg(base)| is smaller at hs0–3 than at hs15+ — the two readers
  share early features (same base weights, fine-tune divergence grows
  with depth); the generator's advantage is a mid/late-depth
  phenomenon.
- **B4 (shape, weakly held):** distill g_agg(ℓ) is flat-to-rising into
  a mid-depth region (hs13–17) rather than peaked at hs0.

## Null calibration (pre-registered)

One permutation-null pair per arm at a NEW layer (lag4 gpt2 hs4 T4;
slope8 distill hs1 T64), NULL_SEED 99 — checking the frozen screens'
noise floors transfer to the new depths. All other cells are paired
comparisons on identical rows, so cell-to-cell differences share the
round-1 σ_null scale (replag 3σ ≈ 0.03 on acc; confidence nulls at
chance).

## Scoring

Each prediction is scored CONFIRMED / FALSIFIED in the LOG entry, with
falsifications reported as findings (round-1 discipline: the data has
refused an agent's favored story before, and that was the result).
