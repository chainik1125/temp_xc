# SALVAGE_CARD — ttrend TXC-post fresh-seed confirmation panel (salvage W1)

**Status: FROZEN pre-registration** (commit sha = the freeze; driver pin
must match). Briefing: `briefings/salvage-mac-a.md` (Han, 2026-07-26
evening; shared ops `briefings/salvage-shared.md`). Executor: mac-a.
Verdict will be **PENDING TEAM REVIEW**; nothing quotable before
mac-local ratification. mac-local freeze-review lands in parallel with
the launch (standing salvage discipline).

## 1. What this is and why fresh seeds

The day-2 tt panel (`PANEL_CARD.md`, re-freeze `db677a4b8`) returned
KEEP on its frozen pre/stacked claiming set only via P1∧P5, with **P4
FAILED central** — pre/stacked flunked the untrained control. The
**post arm was the panel's cleanest profile but was NOT in the frozen
claiming set and therefore could not claim**: trained v1 0.1421 (T16)
/ 0.2968 (T32) vs untrained −0.0084 / +0.0037 (seed means, panel
JSON). This card pre-registers a NEW confirmation panel for the post
arm on **fresh seeds {3, 4, 5}**.

**First-look hazard, stated per the briefing:** the post observation
was made on seeds {1, 2, 42} — quoting it directly would be claiming
on the same draw that generated the hypothesis. The hazard is
neutralized here by confirming on seeds the observation never
touched: every bar below is evaluated ONLY on {3, 4, 5} rows. If
fresh seeds kill the post observation, that is exactly what fresh
seeds are for, and the negative is reported at full prominence.

**Why this face is surface-quiet:** "is this dialogue's turn length
trending up or down" has no surface-count reading at any T — a trend
requires comparing levels at different distances. The committed
visible-cue evidence line (`results/panel_evidence_line_tt.json`,
measured label-side pre-freeze, day-2) is DEGENERATE at T ≤ 8
(r = 0.000 / 0.000 / 0.005 at T = 2/4/8), r = 0.0148 at T16
(floor-active 1.83% of rows), r = 0.1142 at T32 (27.9%).

## 2. The k resolution (DISCLOSED DEVIATION — flagged for freeze-review)

The briefing specifies "k = 8·T". The panel's realized-l0 receipts
show the OBSERVED post config was `k_pos = 8` **per window** (not per
token): realized l0_per_window across trained post cells, seeds
{1,2,42} —

| T | l0_per_window (3 seeds) | l0_per_token |
|---|---|---|
| 2 | 5.562–5.573 | 2.78 |
| 4 | 6.194–6.567 | 1.55–1.64 |
| 8 | 6.827–7.053 | 0.85–0.88 |
| 16 | 7.524–7.587 | 0.47 |
| 32 | 8.021–8.062 | 0.25 |

A confirmation must touch the observed config, so:

- **PRIMARY (claiming) arm = `k_pos = 8`, panel-identical.** This is
  also budget-CONSERVATIVE against the per-token baselines (k = 8 per
  token): at T32 the post arm claims with a 32× smaller active budget
  per token, so an S1 pass cannot be a capacity artifact.
- **SECONDARY (non-claiming, robustness) arm = `k_pos = 8·T`** — the
  briefing's letter, i.e. the postmatched code-rate convention
  (`lambda_intensity/card_stage2_postmatched.md` § 2; the variance
  harness's `--post-k-rule times-T` selects exactly these rows). It
  answers "does the result survive at matched per-token budget?" but
  it is NOT the observed config, so it cannot claim regardless of
  outcome. Its results are reported at full prominence either way.

**No max-over-arms:** the claiming arm is fixed HERE, pre-results, to
the primary. The secondary can neither rescue a primary failure nor
be quoted in place of the primary.

## 3. Frozen cells (72 total; enumeration = `run_salvage.py`, asserted)

Datasource `dial_real_ttrend_gpt2_l7` (gpt2 hs7, d_in 768; committed
stream + labels unchanged from the panel). d_sae 2048, n_steps 8000
(trained) / 0 (untrained), eval_window_L 32, buffer_tokens 524288
UNCHANGED (tt stream 526,208 tokens ≥ buffer — complete fill, no
wrap; the panel disclosure carries). Seeds {3, 4, 5} everywhere.

| block | cells |
|---|---|
| PRIMARY post | `txc_batchtopk_post`, T ∈ {2,4,8,16,32}, k_pos 8, trained + untrained × 3 seeds = 30 |
| SECONDARY post | same T ladder, k_pos = 8·T ∈ {16,32,64,128,256}, trained + untrained × 3 seeds = 30 |
| baselines | `batchtopk_sae` + `tsae` @ T = 1, k_pos 8, trained + untrained × 3 seeds = 12 |

Untrained control per arch and per arm (k_pos matches its arm) — the
standing discipline. **Paired v1+v2 probe columns on every row**: the
PROBE_V2_SPEC § 2 block (`lambda_probe_v2: true`, ridge, 13 alphas
logspace(−2,4), 8192 windows, conversation-grouped `trace` split) is
attached as `eval_extra` to every cell, and the enumeration HARD-FAILS
if any cell lacks it (the day-2 v2-defect lesson, now a pre-run
assert). v1 (`lambda_recovery`) is canonical for every bar except S5;
v2 reported alongside everywhere.

**Realized-l0 bands (numeric, pre-committed):** primary trained cells
must land l0_per_window ∈ [4.5, 9.5] (the observed band above ±
margin); secondary trained cells must land l0_per_window ∈
[0.5, 1.05] × 8·T. A cell outside its band is disclosed in the
verdict and is non-claiming pending investigation — it does not
silently count toward any bar.

## 4. Pre-registered bars (claiming T set = {16, 32}; all on PRIMARY arm, seeds {3,4,5} only)

- **S1 (margin, CI-bounded):** for EACH baseline b ∈ {batchtopk_sae,
  tsae} and EACH T ∈ {16, 32}: paired-by-seed margins
  Δ_s = post_trained(T, s) − b_trained(s), s ∈ {3,4,5}. Require
  mean(Δ) ≥ +0.05 AND the paired t 95% CI, mean ± 4.302653·sd/√3,
  has lower bound > 0. All four (2 baselines × 2 T) must pass.
  Cross-check lane: mac-b's `support_stats/stage2_variance.py` with
  `--seeds 3,4,5` (primary arm = default `--post-k-rule fixed`).
- **S2 (untrained control):** untrained seed-mean ≤ 0.5 × trained
  seed-mean at BOTH claiming T (prior data ~0.01×; expect ≪ 0.5).
- **S3 (T-scaling, reported NOT gating):** statistic = seed-mean OLS
  slope of trained-primary v1 on log2 T over T ∈ {8, 16, 32}; exact
  within-seed permutation of T-labels (6³ = 216 assignments),
  one-sided p = #(perm ≥ observed)/216 (floor 1/216). Reported with
  the verdict.
- **S4 (evidence line, KILL clause):** trained-primary seed-mean v1
  must EXCEED the committed evidence line at both claiming T —
  0.0148 at T16, 0.1142 at T32 (values above, from
  `panel_evidence_line_tt.json`; trivially cleared at T16 if S1
  holds — stated anyway per the briefing). Fail ⇒ KILL.
- **S5 (grouped v2):** trained-primary seed-mean `lambda_recovery_v2`
  > 0 at both claiming T (conversation-grouped split — the identity
  receipt).

**KEEP iff S1 ∧ S2 ∧ S4 ∧ S5.** S3's exact p is reported either way.
The secondary arm gets the same S1/S2/S4/S5 computations reported at
full prominence, gating nothing. Scorer: `score_salvage.py`
(self-contained; every formula above implemented there).

## 5. Ops (standing discipline, verbatim from the day-2 cards)

Commit-then-run: this card + `run_salvage.py` + `score_salvage.py` +
`merge_salvage_payload.py` frozen and pushed before any cell; driver
`scripts/modal_diafaces_salvage.py` pins the freeze via `git
rev-parse` (never hand-typed) and `_assert_pinned()` in-container.
Detach at launch; containers NEVER push; payloads persist to Volume
`temp-xc-replag-caches:/workspace/diafaces_salvage/` in `finally`;
repatriate-merge-locally via `merge_salvage_payload.py` (pin assert +
paired-columns assert per row + dup-eval_key skip + dirty counts
disclosed under the pool leaderboard-growth convention).

Venue (Han's day-2 GPU amendments): H100 main block (69 cells,
workers 6 — gpt2 d768, the tt panel ran 99 cells at 6 workers in ~1 h
clean) + 3× high-CPU L4 64 GB for trained tsae (one per seed,
tsae-first scheduling lesson). Workers/scheduling are NOT frozen
config (batch-halving-class pre-authorization); `--only-cells
arch:T:seed:kind:k_pos` re-pass selector is selection-only. Est
≤ $15 of mac-a's $100 salvage cap; ledger read-before/append-after.

## 6. Deliverables

LOG verdict (PENDING TEAM REVIEW) + receipts proposal + leaderboard
rows (0 dups) + — if KEEP — `figs_writeup/fig4_ttrend_post_confirmation.*`
(post + baselines + untrained + evidence line vs T) with a proposed
caption block for mac-local to integrate.
