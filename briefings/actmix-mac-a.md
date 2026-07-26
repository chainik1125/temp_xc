---
status: active
created: 2026-07-26 ~20:30 London
for: mac-a (executor) — ACTMIX W1: relu_mode implementation + calibration + KEEP-set survives-the-fix
read-first: briefings/actmix-shared.md
---

# ACTMIX W1 — the fix, the calibration, and whether our KEEPs survive it

**Stage 1 — implement `btk-only` (tonight, first).** Add a
relu-free sparsity path to the v2 BatchTopK family
(`txc_batchtopk_{pre,post}`, `batchtopk_sae`, `tsae`,
`stacked_batchtopk`) as plugin-compliant variants with
arch_version bumps (shared briefing rule; executor's choice of
mechanism — suffixed registry names or a YAML-threaded hparam —
provided old rows stay reproducible and the registry validates).
Mind the JumpReLU-threshold inference path: with no ReLU, training
BatchTopK may select negative values when positives run out —
decide and DOCUMENT the convention (recommend: selection over raw
pre-acts, threshold gating unchanged at eval; state the negative-
selection count as a logged diagnostic). Unit tests green before
any cell.

Post your Stage-1 convention note (registry names, hparam,
negative-selection handling, threshold path) to the LOG **as soon
as it exists** — runpod-1/2 consume it verbatim (shared-briefing
single-source rule); they are waiting on you, not the reverse.

**Stage 2 — calibration mini-grid (freeze card first; the
DECISIVE input to Dmitry's re-run gate, wanted before 9am PT).**
COST/SPEED NOTE: the relu-mix side of the grid may be REUSED from
existing leaderboard rows where the config matches exactly (the
salvage/topup panels already hold post T∈{4,16,32} and sae/tsae T1
at seeds {3,4} relu-mix, same datasource + probe conventions) —
dup-key discipline, cite the reused eval_keys in the card. Only the
btk-only cells need compute (≈ half the grid, est drops to ~$4).
Substrate: `dial_real_ttrend_gpt2_l7` (warmest infra, 19-min
turnaround proven). Arms: {batchtopk_sae, tsae} @ T=1 and
txc_batchtopk_post @ T ∈ {4, 16, 32}, each × {relu-mix, btk-only}
× seeds {3,4} trained + untrained. (≈ 2·(2+3)·2·2 = 40 cells,
d768 — cheap; your call to trim untrained to 1 seed.) Report per
arm: realized l0 vs nominal, recovery, and for post the T-slope
d(recovery)/d log T under both arms. Pre-registered reading (from
the shared briefing, restate in card): sae improves most under
btk-only; tsae least; post low-T cells improve; slope may soften.
Est ≤ $8. Deliverable: CALIB card + score JSON + one figure
(relu-mix vs btk-only, per arm) + LOG verdict PTR.

**Stage 3 — KEEP-set survives-the-fix (gated on Stage 2 landing
sane).** Pre-register a card: under `btk-only`, re-run exactly the
claiming cells of the hunt's KEEPs and score against the SAME bars:
1. ttrend TXC-post: post T ∈ {16,32} + both baselines, seeds
   {6,7,8}, trained + untrained (the R29 pooling-free lane,
   re-scored S1/S2/S4/S5).
2. λ̂ backtracking intensity: the R22 top-up comparison cells
   (pre/T8 + tsae trio, seeds as in the R22 lane).
3. dq: pre/T8 lead cells + tsae (the R27 margin, 3 seeds).
Verdicts per task: SURVIVES / SURVIVES-WITH-MOVED-MARGINS (numbers
side by side) / DOES-NOT-SURVIVE — at full prominence either way;
the fix arm CANNOT claim anything new (different composition = new
pre-registration), it only stress-tests existing claims. Est
≤ $15–20.

**Cap for W1 total: $40.** Modal, detached, ledgered. Venue: H100
for trained pools if queue-free, else L40S; tsae-first scheduling.
