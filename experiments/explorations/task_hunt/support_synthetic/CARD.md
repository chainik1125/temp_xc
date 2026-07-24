# Support-synthetic card — mechanism receipts for the Stage-2 λ̂ result

**Frozen pre-build / pre-run.** Mandate: `briefings/hunt-support-synthetic.md`
(discharges the mac-local review notes on the T=16-dip interpretation and the
T-SAE-fairness rejoinder — `task_hunt/LOG.md` review entry, 2026-07-24).
Substrate: `toy_backtracking_selfexcite_d64` (the λ̂ mirror; provable per-token
DPI floor corr ≈ 0.41; shared `lambda_recovery` per-tile leading-edge evaluator,
protocol 1.3.0). Bench canon throughout: F = 20, N_STEPS = 30 000, seeds
{1, 2, 42}, eval_window_L = 32, canonical runner, window mode. Every trained
cell through `temp_bench.core.runner.run_experiment`; identical-config cells
already on the leaderboard are **cache hits** (runner returns the existing row
by `eval_key`; nothing re-appended) — reuse is disclosed below, 0 dup keys by
construction. Agent: runpod. Verdicts are the mechanical output of
`analyze_dilution.py` / `analyze_tsae.py` (committed pre-run) against the rules
frozen here.

---

## Item 1 — budget-dilution receipt (the RECORD § 3b dip sentence)

### 1.1 What "fixed budget" means (resolved from the real panel, not assumed)

The real Stage-2 panel (`lambda_intensity/run_stage2.py`) fixed **k_pos = 8 per
token for every arch at every T** ("fairness rides on equal k_pos") and a single
dictionary `d_sae = 2048`; realized `l0_per_token` for TXC-pre stayed 6.9–7.8
across T = 2..16. Since TXC-pre's BatchTopK budget is per-token (pool B·T), its
per-window **atom** count already grew ∝ T in the real panel. The only per-window
code resource held fixed there was the **d_sae-dim shared window code**. The
RECORD's clause "extra positions dilute a fixed code budget" is therefore
operationalized as: **fixed budget = fixed d_sae at fixed per-token k_pos**
(the real convention, mirrored exactly), and **budget-scaled = d_sae grown ∝ T**
at the same fixed per-token k_pos, so the shared code's capacity tracks the
content it must carry. Realized `l0_per_token` **and** `l0_per_window` are
measured and reported for every cell (the l0 trace disambiguates
atom-starvation from dimension-crowding: if arm A's realized per-window support
saturates against the dictionary wall while arm B's grows ∝ T, that is the
dilution mechanism made visible in the data).

### 1.2 Cells (all TXC-pre = `txc_batchtopk_pre`, k_pos = 1, 3 seeds + untrained)

k_pos = 1 is the mirror's canonical headline slice (bench_record; u_bt is
recoverable at k_pos = 1; the bench showed λ̂ recovery k-independent at
k ∈ {1, 2}).

| line | d_sae | T ladder | new cells (rest are cache hits) |
|---|---|---|---|
| **A1** — canonical fixed budget | 20 (= F) | {2, 4, 8, 16} | T = 16 |
| **A2** — fixed budget, ladder-complete | 40 (= 2F, locked capacity sweep) | {2, 4, 8, 16, 32} | T = 16, 32 |
| **B** — budget-scaled | **5·T** ∈ {10, 20, 40, 80, 160} | {2, 4, 8, 16, 32} | (16, 80), (32, 160) |

- (T = 32, d_sae = 20) is **dict-infeasible** for the pooled family
  (k_pos·T = 32 > 20; BatchTopK would be a no-op) — A1's ladder ends at 16;
  A2 (2F) is the smallest locked capacity that carries the briefing's full
  ladder at fixed budget. Disclosed, not silent.
- Arm B's scaling pins **dims-per-position at 5 = 20/4**, exactly the
  per-position allotment the canonical line has at its measured peak (T = 4,
  d = 20 → 0.952) — a per-position budget already proven sufficient. If
  recovery still declines while that allotment is held, "fixed code budget
  spread over more positions" is not the explanation.
- B coincides with A1 at (4, 20) and with A2 at (8, 40) by construction
  (shared cells, run once).
- Untrained controls (n_steps = 0, k_pos = 1, 3 seeds) at every (T, d_sae)
  line-point; existing ones reused. Untrained shapes are commentary (see P4),
  never verdict inputs.
- Existing cells reused from the leaderboard (bench grid, seeds {1,2,42}):
  A1 T ∈ {2,4,8} (0.870 / 0.952 / 0.949 seed-means), A2 T ∈ {2,4,8}, B
  T ∈ {2,4,8} (= (2,10), (4,20), (8,40)), plus untrained at (T ∈ {2,4,8}, 20).
  New trained cells: 3 line-points × … = (16,20), (16,40), (32,40), (16,80),
  (32,160) × 3 seeds = 15; new untrained: same five points minus none existing
  at d ≠ 20 ⇒ (16,20), (16,40), (32,40), (16,80), (32,160) × 3 = 15.

### 1.3 Probe-capacity disclosure (frozen before any data)

`lambda_recovery` fits an **unregularized** LinearRegression on single-tile
codes with 1024 train windows × (32/T) tiles, so examples-per-probe shrink with
T while arm B's feature dim grows: p/n reaches 160/1024 ≈ 0.16 at B's (32, 160)
— enough to depress held-out r mechanically when true recovery is mid-range.
Rule, frozen: **all Item-1 verdicts are computed on T ≤ 16** (every line;
symmetric — p/n ≤ 0.08 everywhere there). The T = 32 points (A2, B) are
reported as the descriptive ladder extension; B's (32, 160) point is read
against its **own matched untrained control** (identical probe conditions),
never against other-T cells. The T = 32 extension can qualify wording, never
flip a verdict.

### 1.4 Frozen predictions

- **P1 (arm A dips):** A1 declines from its peak by T = 16:
  paired D = mean(peak) − mean(16) ≥ bar. (Peak from cached cells is T = 4,
  0.952; rise T=2→peak, 0.082, already exceeds bar.)
- **P2 (arm B relieved):** B shows no decline ≥ bar anywhere in T ≤ 16.
- **P3 (dose-response, commentary):** at T = 16, decline severity orders
  A1 > A2 > B (dims-per-position 1.25 < 2.5 < 5); A2's decline is intermediate
  or arrives only at T = 32.
- **P4 (untrained, commentary):** untrained recovery declines with T even at
  init (capacity effect present without training; existing untrained means
  0.71 / 0.78 / 0.66 at T = 2/4/8 already hint at it).

### 1.5 Mechanical decision rules (analyze_dilution.py implements these verbatim)

Per line ℓ: cell stat = mean over seeds; T_peak(ℓ) = argmax_T mean.
Paired-by-seed decline to any T' > T_peak, T' ≤ 16:
D(ℓ, T') = mean_s[rec_s(T_peak) − rec_s(T')], SE = std(ddof=1)/√3,
**bar = max(2·SE, 0.05)**. DIP(ℓ) fires iff D(ℓ, T') ≥ bar for some such T'.
Guard: all three seeds present in every compared cell, else the cell pair is
excluded and logged.

- **BACKED** — DIP(A1) ∧ ¬DIP(B): the mirror reproduces the dip under the
  real panel's fixed-budget convention and scaling the window-code budget
  removes it. LOG paragraph backs the RECORD sentence ("predicted by the
  budget model, reproduced in the mirror").
- **RETRACT** — DIP(B): the dip persists with the per-position code budget
  held at the proven-sufficient level ⇒ the dilution interpretation is wrong;
  LOG paragraph states the RECORD § 3b interpretation clause must be retracted
  (a finding; the RECORD amendment itself is a review action).
- **NO-MIRROR-DIP** — ¬DIP(A1) ∧ ¬DIP(A2): the mirror does not reproduce the
  real dip under the real convention ⇒ the real dip needs a different
  explanation (a finding — said plainly in the LOG).
- **AMBIGUOUS** — remaining patterns (e.g. DIP(A2) without DIP(A1), which
  inverts the dose-response ordering): reported honestly as such, with the
  pattern spelled out; no BACKED claim.

Deliverable: two-arm figure (recovery vs T; A1/A2 solid, B dashed, untrained
dotted; realized l0_per_window annotated per point; per-token DPI floor 0.41
reference) + the LOG paragraph.

---

## Item 2 — T-SAE fairness receipt (its own temporal knob)

### 2.1 The knob and the variant

The registered T-SAE (`tsae`, class `TSAEPaper`) hardcodes its temporal
hyperparameter: the contrastive pair is consecutive tokens
(`t_offset = randint(0, T_seq−1)`; pair `(x[:,t], x[:,t+1])`). Per hard rule 3
the knob is exposed by a **plugin variant arch file** (never editing the
registered class): `TSAEDelta(TSAEPaper)` in `src/temp_bench/archs/tsae_delta.py`
with hparam `pair_delta` = Δ, generalizing exactly two lines
(`randint(0, T_seq−Δ)`; pair `(x[:,t], x[:,t+Δ])`) — train_step otherwise a
verbatim port. At Δ = 1 the RNG stream and arithmetic are identical to
`TSAEPaper`. YAML entries (one class, many entries — the dissection precedent):
`tsae_d1, tsae_d2, tsae_d4, tsae_d8` (Δ ∈ {1, 2, 4, 8}), plus auxiliary
`tsae_a0` = registered `TSAEPaper` at `contrastive_alpha = 0` (no new class;
tests whether the contrastive term matters at all on the mirror).

**Contract tests (committed with the build, green before any grid):**
(i) TSAEDelta(Δ=1) ≡ TSAEPaper bit-wise (loss and params) over ≥ 5 train
steps at a shared seed; (ii) pair-offset bounds and pair identity at Δ > 1;
(iii) T ≠ 1 still rejected; (iv) registry loads all five entries.

### 2.2 Cells

Canonical mirror budget: d_sae = 20 (per-section default), k_pos = 1 (headline
slice; the bench grid showed T-SAE's λ̂ recovery independent of k across
{1..16} and of d_sae across {10, 20, 40} — band 0.38–0.44, pinned at the DPI
floor). 5 entries × 3 seeds trained (sequence mode, ~115 s/cell) + untrained
(n_steps = 0) for **all five entries** × 3 seeds.

- **Untrained guard:** Δ and α touch only train_step, so the five entries'
  untrained metrics must be exactly equal per seed. Any inequality = pipeline
  bug; stop and fix before reading trained cells.

### 2.3 Frozen prediction and rules (analyze_tsae.py implements verbatim)

- **P5 (FLAT):** λ̂ recovery is flat in Δ within seed noise, with every setting
  in the per-token DPI-floor band (~0.41): for each Δ ∈ {2, 4, 8}, paired
  D(Δ) = mean_s[rec_s(Δ) − rec_s(Δ=1)], bar = max(2·SE, 0.05), |D(Δ)| < bar.
  Mechanism: the decode is per-token at eval; by the DPI no training-time
  temporal pressure can lift a single-token readout past the floor.
- **RISE (the falsifier & flag):** D(Δ) ≥ +bar for any Δ ⇒ **immediately**:
  (a) LOG flag stating the real panel's T-SAE cell may underestimate the
  baseline; (b) note addressed to runpod-d (round-2 T-SAE cell needs a re-run
  at the best Δ before any rebuttal figure ships); then (c) the standard
  skeptic on the rise claim (`claude-fable-5`, raw persisted pre-parse, never
  re-rolled, Meter → expansion spend.json, session cap $5; cumulative
  $11.52/$25). Flag first, skeptic second — the deadline outranks polish.
- **DECLINE** (D ≤ −bar): not the fairness threat; reported, no flag.
- `tsae_a0` aux: same paired test vs `tsae_d1`; commentary only, no verdict
  weight.

Deliverable: recovery-vs-Δ figure (floor band + registered-tsae reference) +
the LOG paragraph closing the rejoinder.

---

## Process rails (both items)

Commit order: this card (freeze) → build (variant arch + YAML + tests + run /
analyze / render scripts; tests green; `run.py validate` clean) → grids →
mechanical verdicts → figure + LOG paragraphs → STATUS rewrite → push → stop
for mac-local review (briefing stays). Results & records under
`experiments/explorations/task_hunt/support_synthetic/`; figures under
`figs/`. No reviewer/meeting quotes in tracked files. Deadline: results by
Saturday morning PT.

_Frozen-by: claude-fable-5 (runpod agent), 2026-07-24, before any Item-1/Item-2
build or run commit._
