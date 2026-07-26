# PANEL CARD DRAFT — diafaces mini-panel (gated; DO NOT LAUNCH from this draft)

**Status: DRAFT, prepared while the Stage-1 screen runs (day-2 W2
briefing § "gated mini-panel"). This document becomes a FROZEN card
only in a later commit that (a) fills the [AT-FREEZE] fields from the
screen verdict and (b) quotes mac-local's written LOG approval of all
five gate clauses (shared doc § panel gate, as PINNED pre-results in
mac-local's 2026-07-26 freeze-review entry: (i) KEEP on ≥ 2 of 3
models under the card's majority rule; (ii) wd order arm
sc = win_linear − win_shuf_linear ≥ +0.03 at T ∈ {16,32} on ≥ 2 of 3
models INCLUDING at least one of {gpt2, llama31}; (iii) launch by
14:30 London; (iv) ledger ≤ $250 at launch; (v) mac-local written
approval naming (i)–(iv)). If the gate does not fire, this draft +
the registered datasource plugin IS the deliverable: panel-ready with
an order-carried screen prior, first day-one launch of the
post-deadline queue.**

## 1. Question

Stage-1 KEEPs answer "is a trailing dialogue state linearly present
in raw activations beyond its floors?" The mini-panel asks the
program's Stage-2 question: **do position-mixing dictionary
architectures RECOVER that state from the stream better than
order-free ones**, on the one substrate with measured order-carriage
(R11) — the λ̂ Stage-2 pattern, reduced for the clock.

## 2. Datasource (plugin committed NOW; YAML entry only at freeze)

`src/explorations/task_hunt/real_dialogue.py::dialogue_face_real` —
the `real_lambda` SyntheticData pattern: dialevel caches + committed
diafaces label grid; reference-basis `emission_features` (spanning
sanity only, never feature recovery); **conversation-identity
`trace_ids`** so the v2 probe splits by dialogue (doc-identity route
is 0.76/0.85 label-side — the panel's disclosed hazard); labels NaN
at BOS/boundary/undefined positions; global-RMS normalization;
licence note in `extra`. DS name at freeze:
`dial_real_[FACE]_[MODEL]_l[HS]` (winning face; the STRONGER screen
model; that model's screen layer — gpt2 hs7 / gemma2 hs14 / llama31
hs14).

## 3. Grid (the λ̂ Stage-2 shape, REDUCED: one model, T ≤ 16)

- Archs (5, the λ̂ panel set): `batchtopk_sae` (T = 1 order-free
  baseline), `tsae` (T = 1), `txc_batchtopk_pre`,
  `txc_batchtopk_post` (**k_pos = 8·T from cell one** — the
  post-matched lesson), `stacked_batchtopk`; T ∈ {2, 4, 8, 16}.
- Seeds {1, 2, 42}; trained (n_steps 8000) + untrained (0);
  d_sae 2048, k_pos 8, eval_window_L 32 — all UNCHANGED from the λ̂
  panel; **buffer_tokens 524288 UNCHANGED** (fill argument: the
  dialevel stream is 0.81–0.88 M tokens ≥ buffer — complete fill, no
  wrap; comparability with every prior panel is the point).
- ≈ 84 rows, all through `temp_bench.core.runner.run_experiment`,
  merged into the canonical leaderboard locally (0 dup keys), panel
  file `results/stage2_dial_real_[FACE]_[MODEL]_l[HS].json`.
- Runner: frozen clone of the λ̂ Stage-2 runner
  (`lambda_intensity/run_stage2.py` pattern; container partitioning
  per arch-block like `--only-seed`), committed at freeze.

## 4. Receipts in the panel (frozen shape)

- **Paired v1 + v2 probe columns** per `PROBE_V2_SPEC` § 2 on every
  row (claim on v1, v2 reported — METHODS DECISION 2026-07-25);
  variance receipts via `support_stats/stage2_variance`
  (`--row-layout auto`, `--post-k-rule times-T`).
- **Realized-l0 band** stated per arm; under-band cells disclosed
  (the R22 lesson).
- **Evidence-line analog per T**: the screen's visible floor
  (tt: kernel-WLS on visible complete turns; dq: "?" count) evaluated
  on the panel's eval rows at T ∈ {2,4,8,16} — the "what counting
  affords" line under every recovery curve.
- **Doc-identity floor + demeaned receipt**: per-conversation mean
  predictor (label-side floor) AND a within-conversation demeaned
  probe variant reported next to the headline (the wd-BINDING rule
  carried to Stage 2).

## 5. Pre-registered predictions [bars filled AT FREEZE from screen numbers]

- **P1**: trained pre/stacked beat `batchtopk_sae` at some T ≤ 16 on
  v1 recovery (margin bar set at freeze).
- **P2**: the margin GROWS with T over {2,4,8,16} (the screen's
  reach argument at panel windows).
- **P3**: tsae (T = 1) sits between sae and the best T > 1 arm
  (the λ̂-panel ordering; R22's bound is the precedent).
- **P4**: untrained arms recover ≤ [AT-FREEZE] of trained (the
  training-does-something receipt).
- **P5**: recovery survives the demeaned receipt in DIRECTION
  (within-conversation r > 0 for the winning arm) — else the panel
  headline is conversation identity and says so.

**KEEP/KILL [formulas frozen now, numbers at freeze]:** KEEP iff P1
holds on v1 with the variance receipt's CI clear of 0 AND P5 holds.
KILL if the best trained arm ≤ sae + noise everywhere, or P5 fails.
Else WEAK. No max-over-arms; arms quoted per identity; nothing
quotable pre-ratification (process ruling 2026-07-26).

## 6. Venue, economics, discipline

**A100-40 ONLY for panel cells** (shared-doc GPU rule; reason stated
in the ledger line), est ≈ 2 h wall: **tsae cells FIRST, one per
container** (CPU-buffer-bound, 62–77 min precedent), main pool after;
est ≤ $60 total against mac-a's $120 day-cap and the ≤ $250-at-launch
gate clause. Commit-then-run (runner + YAML entry + this card FROZEN
and pushed; container pinned via rev-parse + `_assert_pinned()`);
`--detach`; payload persistence to Volume; containers never push;
ledger read-before/append-after; repatriate-merge-locally with dup
check. Deliverable: LOG verdict `mac-a (executor)` PENDING TEAM
REVIEW + leaderboard rows + panel JSON + receipts proposal;
checkpoints to Volume (HF mirror = Han follow-up rule).
