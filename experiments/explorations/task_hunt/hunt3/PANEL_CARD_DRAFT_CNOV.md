# DRAFT — cnov panel card (NOT FROZEN; team picks at 17:00)

**Status: DRAFT ONLY.** Produced per overnight § 1 ("the best 1–2
with DRAFT panel cards — not frozen — team picks"). Nothing here is
pre-registered yet; if the team selects cnov, this draft goes through
freeze-review (mac-local) and THEN becomes binding. Screen basis:
HUNT3_SCREEN_CARD.md § 4 KEEP 3/3 + panel-gate routing
(order-sensitivity 2/3 models at T32: llama +0.031, gemma +0.039,
gpt2 +0.026 — `hunt3/results/verdict.json`).

## 1. The task

`cnov` — kernel-weighted trailing rate (support 64 tok, HL 16) of
FIRST-IN-CONVERSATION token types on DailyDialog. Out-of-window
structural guarantee: whether a type occurred before the window
start is uncomputable from any T-window — the strongest form of the
txcwin novelty definition, transplanted to the one substrate with
measured order-carriage (WRITEUP § 7).

## 2. Proposed panel design (R29-lane shape)

- Datasource: NEW `dial_real_cnov_gpt2_l7` — the EXISTING dialevel
  gpt2 stream + the committed cnov face
  (`labels/hunt3_dailydialog_gpt2.npz`) wired through the λ-recovery
  evaluator exactly as ttrend/dqgap were (zero new forward passes;
  registration = data.yaml entry + the face key).
- Arms: `txc_batchtopk_post` @ T ∈ {8, 16, 32} + `batchtopk_sae` +
  `tsae` @ T1, k_pos = 8, d_sae = 2048, n_steps 8000, buffer 524288,
  V2 eval_extra verbatim — the exact ttrend-panel constants (warmest
  infra; every convention already ratified there). relu-mix arm
  labels (R30: composition is a no-op at these widths).
- Seeds: 3 fresh (proposal: {9, 10, 11} — never used on this
  substrate).
- Claiming zone (from the screen): T ∈ {16, 32} primary; T8
  reported non-claiming (screen qualifying arm was T32; T16 next).

## 3. Proposed bars (S1–S5 family, subject to freeze-review)

- S1: post − {sae, tsae} margin ≥ +0.05 AND paired t 95% CI LB > 0
  at T ∈ {16, 32} (both baselines, both Ts).
- S2: untrained ≤ 0.5× trained at claiming Ts.
- S3: T8→32 trend, exact within-seed permutation, reported not
  gating.
- S4 (KILL, the evidence line): recovery must beat the panel-side
  visible floor — the screen measured the first-in-window floor at
  0.437–0.527 (T16) / 0.496–0.527 (T32) probe-accuracy units; the
  panel re-measures it in λ units on the panel rows BEFORE any cell
  (mac-b's machinery on request, briefing § 2).
- S5: grouped v2 > 0 at claiming Ts.
- Pre-named traps carried from the screen: position (inverted-0.86)
  and conversation-identity (doc-mean 0.86) — the panel inherits the
  position-matched manifest convention, and the wd-style control
  becomes an explicit panel arm question at freeze-review.

## 4. Cost

≈ 30–36 cells ≈ $4–6 (ttrend-panel scaling), H100 main + L4 tsae.

## 5. Why this and not nvtrend

nvtrend also KEEPs 3/3 but routes to the BREADTH table by the frozen
order rule (0/3 models at +0.03; margins ≤ +0.017) — its gain is
pooling-matchable aggregation, the class that went 0-for-2 at
panels. It stays on the breadth table with its numbers; no panel
slot proposed.
