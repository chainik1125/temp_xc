# Probe-truth card — which λ readout reports TRUE recovery?

**Status: FROZEN at commit (commit-then-run; no cell of this campaign has
been executed when this card is committed — git order is the evidence).**
Agent: runpod-b. Briefing: `briefings/mirror-probe-truth.md`.
Substrate: `toy_backtracking_selfexcite_d64` (the λ̂ mirror), bench canon
throughout: F = 20, N_STEPS = 30 000, seeds {1, 2, 42}, `eval_window_L` = 32,
window mode, canonical runner (`temp_bench.core.runner.run_experiment`) for
every leaderboard cell. Verdicts are the mechanical output of
`analyze_probe_truth.py` (committed before any analysis) against the rules
frozen in § 6–7.

**This campaign produces a receipt, not a verdict.** The λ-readout methods
DECISION is mac-local's, taken against the 4-branch rule pre-registered in
`task_hunt/LOG.md` before this evidence existed. Nothing here adopts,
retires, or amends the canonical readout; no committed panel number moves.

---

## 1. The question, and why the mirror can answer it

On the real panels, v2 (`lambda_recovery_v2`: RidgeCV + 8192 windows) lifts
dense-code cells by +0.18…+0.23 over v1 (`lambda_recovery`: OLS + 1024
windows) and reverses the T-decline. **A bigger number is not evidence of a
better probe.** On a real task the true recoverable λ is unknown, so "v2
reports more" is compatible with three different worlds:

- v1 is capacity-limited and **sags below** truth on dense codes; v2 tracks
  truth → v2 is the better readout;
- both probes track truth and v2's real-panel lift comes from something else
  (e.g. a genuinely different eval-window population at nw = 8192) → v1 is
  fine, and the lift needs another explanation;
- v2 **reports above** truth — regularization + more rows buying an
  optimistic held-out correlation → v2 is worse for headline use.

On the mirror λ is a *generative* quantity, so the third variable in that
comparison — truth — is available. This card fixes how truth is obtained
(§ 3), what is measured against it (§ 4), and what pattern licenses which
reading (§ 6–7), **before any cell runs**.

### 1.1 The design trap this card must avoid (stated because it inverts the answer)

v1's probe budget is hardcoded: `lambda_recovery_metrics` forwards no
`n_windows`, so every committed cell fits on `n = 1024·(32/T)` tile-rows
— 16384 / 8192 / 4096 / **2048** at T = 2 / 4 / 8 / 16. The real λ̂ panel
runs `d_sae = 2048`, so at T = 16 it sits at **p/n = 1.00** — precisely the
regime where unregularized OLS interpolates. The mirror's *committed*
budget (`CARD.md`: d_sae ∈ {20, 40, 5·T}) sits at **p/n ≈ 0.001–0.08**.

A probe-truth campaign run at the mirror's canonical budget would find both
probes agreeing everywhere — not because the probes are equivalent, but
because the mirror never entered the regime the real panel occupies. That
result would read as branch 2 (DECLINE) and would be **wrong for a reason
invisible in its own numbers.**

**Consequence, disclosed:** this campaign keeps the mirror's canonical line
as its low-p/n control (line M, § 2) and *extends* the ladder in `d_sae`
and `k_pos` until it spans the real panel's regime (p/n up to 1.0, and
4.0 for Stacked). That is a deliberate, stated deviation from the briefing's
"reuse the committed mirror config exactly": the committed config is
reused, and is not sufficient on its own. **p/n — not T alone — is this
campaign's x-axis.**

### 1.2 What the mirror cannot answer (the transfer limit, stated up front)

The mirror licenses statements about **the probe as a function of (p/n,
code density, true recovery level)**. It does not license a statement that
any particular real-panel cell's v2 number is or is not truthful: the real
cells are matched to mirror cells by p/n and realized l0, not by task. Any
reading transferred to the real panels rides on that matching being the
operative variable — which is exactly the claim the probe-capacity
diagnostics make, and which this campaign tests on the mirror rather than
assumes.

---

## 2. Substrate, arms and ladder (frozen)

Datasource `toy_backtracking_selfexcite_d64` — λ_i = σ(a + Σ_{l=1..K} w_l
b_{i−l}), **K = 2**, τ = 2, α = 3.06, base rate 0.12, d_in = 64, F = 20.
Documented ceilings: per-token DPI floor **corr ≈ 0.41** (provable); window
ceiling **≈ 0.91 at T = 2** (only b_{i−1} inside the tile), **≈ 0.99 at
T ≥ 4** (both driver lags inside the tile).

### 2.1 Stage 1 — constructed-code calibration (exact known truth, no training)

`probe_truth_calib.py`, **off the leaderboard** (the `probe_capacity.py`
precedent: nothing appended to `results/leaderboard.jsonl`). The encoder is
replaced by an *analytic* one whose λ-information content is set by
construction; everything below it — window sampling at v1's seeds, the
train/eval sequence split, the tiling, the leading-edge target, the
shuffled-target chance floor — is the **committed** code path
(`_train_lambda_probe` and `_train_lambda_probe_v2` are imported and
called, not re-implemented).

Constructed code = `S` signal dims ⊕ `p − S` noise dims:

- **signal dims** read the event stream off the tile by projecting on the
  known emission direction `u_bt` (`data.extra["emission_features"][0]`),
  thresholded at half the firing magnitude → exactly the binary `b` at
  chosen tile positions;
- **noise dims** are a fixed random sparse read-out of the tile's *content*
  subspace with `u_bt` projected out (random Gaussian map → ReLU → top-8
  per row). Content directions are drawn independently of `b`, so these
  dims carry **zero** λ information by construction while being sparse,
  nonneg and mutually correlated like a real SAE code.

Three arms, each with an independently documented truth:

| arm | signal dims | known truth ρ\* |
|---|---|---|
| `full` | b at tile positions T−2, T−3 (the K = 2 driver, where inside the tile) | the window ceiling: ≈ 0.91 at T = 2, ≈ 0.99 at T ≥ 4 |
| `token` | b at tile position T−1 (the leading-edge event itself) | the per-token DPI floor ≈ 0.41 |
| `null` | none | **0** |

Reproducing 0.41 and 0.91/0.99 from the construction is a **validity gate**
on the calibration itself (§ 6, G1): the arm truths are known from the
bench's own analysis, independently of anything measured here.

Ladder: `p ∈ {8, 32, 128, 512, 2048, 4096}` × `n_rows ∈ {2048, 4096, 8192,
16384}` (realized as T = 16, `n_windows ∈ {1024, 2048, 4096, 8192}`, so
n = 2048 **is** v1's committed setting and n = 16384 **is** v2's) × seeds
{1, 2, 42}. One extra Stacked-like corner `p = 8192` at n = 2048 (p/n = 4).
p/n spans **0.0005 … 4.0**.

ρ\* per (arm, seed) is computed as the held-out correlation of the
population-optimal linear predictor — OLS on the ≤ 2 signal dims alone at
the largest n (p ≤ 2 vs n = 16384: estimation error ~1e-3), evaluated on
the same eval rows the probes are scored on. Adding uninformative dims
cannot move the population optimum, so ρ\*(p) is a horizontal line and the
whole measurement is each probe's deviation from it as p/n grows.

### 2.2 Stage 2 — paired v1/v2 on the surviving mirror checkpoints

`run_probe_truth.py --stage existing`. Every mirror checkpoint still on
disk gets both readouts on the same windows through the canonical runner
(training is a cache hit; the v2 flags hash into `eval_key`, so these are
new rows and no existing row is touched).

**Coverage, stated honestly:** the mirror has 843 leaderboard rows over 843
distinct `train_key`s, but **only 22 checkpoints survive on disk**
(`checkpoints/manifest.jsonl` carries 9878 rows with **zero** HF refs, so
there is no restore path for the rest). The survivors are all d_sae = 20,
n_steps = 30 000, T ∈ {1, 4, 8}, k_pos ∈ {2, 8, 16}: `batchtopk_sae` T1,
`tsae` T1, `txc_batchtopk_pre` T4×3 + T8, `txc_batchtopk_post` T4×3 + T8,
`stacked_batchtopk` T4×3 + T8, `spectral_txc` T4×3 + T8. Every one of them
sits at **p/n ≤ 0.04**. So Stage 2 is a real paired sample and a genuine
low-p/n control (P3), and it **cannot** speak to the regime the question
lives in. The briefing's "this alone may answer the question by breakfast"
does not survive contact with the checkpoint prune: **this campaign is
training-bound.**

### 2.3 Stage 3 — the trained ladder (the overnight body)

`run_probe_truth.py --stage train`. All cells: canonical runner, seeds
{1, 2, 42}, N_STEPS = 30 000, L = 32, v2 flags in `eval_extra` so each row
carries its own paired v1 column.

| line | arch | k_pos | d_sae | T | p (probe features) | p/n at T = 16 |
|---|---|---|---|---|---|---|
| **M** — mirror-canonical control | `txc_batchtopk_pre` | 1 | 20 | 2, 4, 8, 16 | 20 | 0.010 |
| **C** — capacity ladder (core) | `txc_batchtopk_pre` | 8 | 256, 1024, 2048 | 2, 4, 8, 16 | d_sae | 0.125 / 0.500 / **1.000** |
| **P** — matched post (briefing § 3a) | `txc_batchtopk_post` | **8·T** | 2048 | 2, 4, 8, 16 | 2048 | 1.000 |
| **S** — Stacked (p > n exception) | `stacked_batchtopk` | 8 | 512 | 4, 16 | **T·d_sae** | 4.000 |

- Line **M** re-runs the dilution receipt's A1 line verbatim (`CARD.md`
  § 1.2). Its checkpoints were pruned, so it retrains; training is
  deterministic given the seed, so its v1 column must reproduce A1's
  committed seed-means (0.870 / 0.952 / 0.949 at T = 2 / 4 / 8) — a free
  integrity check, reported either way (§ 6, G2).
- Line **C** adopts the real λ̂ panel's convention (fixed per-token
  k_pos = 8, single `d_sae`) so that at (T = 16, d_sae = 2048) the mirror
  cell is matched to the real panel's dense cell in **both** p/n (1.00) and
  realized window density (k_win = k_pos·T = 128 vs the panel's measured
  l0_per_window ≈ 125).
- Line **P** uses runpod-d's code-rate convention (`card_stage2_postmatched.md`
  § 2): `txc_batchtopk_post` pools its BatchTopK budget **per window**, so
  nominal k = 8·T is what holds l0_per_token at 8. This is the confound
  that qualified the real λ̂ panel; the mirror can now test its λ readout
  against truth.
- Line **S** is the disclosed `p > n` case (`PROBE_V2_SPEC.md` § 1 knob 2):
  at T = 16, p = 8192 exceeds even v2's row count. Its truth anchor is
  **not licensed** (§ 3) — S is reported as commentary, never as a verdict
  input.
- **Untrained controls** (`n_steps = 0`, 3 seeds) at *every* line point
  above — 22 points. An untrained code is a random projection: true
  recovery is low but nonzero and p/n is unchanged, which makes it the
  cleanest trained-code test of probe *optimism*. Per the support-synthetic
  precedent untrained shapes are commentary; here they additionally feed
  the P4 optimism test, which is stated as a separate rule (§ 6).

**Row decomposition (leaderboard hygiene).** 66 trained cells (12 M + 36 C +
12 P + 6 S) + 66 untrained = **132 new rows**, one per cell, all carrying
`lambda_probe_v2: true` in `eval_cfg` → all 132 `eval_key`s are new by
construction, 0 duplicate keys, 0 existing rows rewritten. Cells whose
`train_key` already has a live checkpoint are training cache-hits; the rest
retrain (the prune, § 2.2).

### 2.4 Sequencing (so a short night still ships)

Card (this file) → build → **launch Stage 3 training first** (it is the long
pole) → Stage 1 calibration and Stage 2 while it runs → anchors → analysis →
figure. Every stage is a separate commit + LOG line. Priority if the night
runs short: **C ≻ P ≻ M ≻ S**, and within C, T = 16 before T = 2. A partial
ladder with an explicit coverage statement beats a rushed full one; the
coverage table in `probe_truth.json` is emitted from the rows that actually
landed, never from the planned grid.

---

## 3. How truth is obtained for a *trained* code (the anchor, frozen)

Stage 1 has exact truth by construction. Trained codes do not, so truth is
estimated — and the estimator is fixed here, before any number exists:

> **TRUTH ANCHOR.** For a cell with p probe features, refit the same probe
> family on `n_rows ≥ 32·p` (floor 16384, cap 65536) drawn by the same
> committed sampler from the same train/eval split, with **both** OLS and
> RidgeCV. The anchor is the mean of the two.
>
> **Licence (all three required, else the cell's truth is UNKNOWN and the
> cell is excluded from every verdict rule and counted as coverage loss):**
> (a) realized `n_rows ≥ 16·p`; (b) |anchor_ols − anchor_ridge| ≤ 0.02;
> (c) the cell's v1 replication check reproduces its committed
> `lambda_recovery` to ≤ 1e-6.

The rationale for (b) is the whole reason the anchor is credible: as n/p
grows the two probe families **must** converge to the same population
value; where they still disagree, n/p is not yet large enough and no truth
is claimed. Condition (c) is `probe_capacity.py`'s licence, kept verbatim:
the anchor script re-implements nothing — it imports the committed helpers,
so an exact v1 replication proves it is reading the same code on the same
rows.

Stage 1 **validates the anchor procedure**: on the constructed codes the
anchor is computed alongside the exactly-known ρ\*, and G1 (§ 6) requires
it to recover ρ\* to ≤ 0.02. If it does not, the anchor is not licensed on
trained codes either and the campaign reports branch 4.

The anchor is computed off the leaderboard (`probe_truth_anchor.py`): it
uses settings that are not a frozen convention and must not become one.

---

## 4. What is measured (frozen definitions)

Per cell (arch, T, d_sae, k_pos, kind) and seed:

- **v1** — the committed convention: OLS, `n_windows = 1024`, `n // 2`
  split. Taken from the leaderboard row's `lambda_recovery`.
- **v2** — the frozen candidate (`PROBE_V2_SPEC.md` § 1): RidgeCV over
  `np.logspace(-2, 4, 13)` selected inside the train half, `n_windows =
  8192`, trace split (degenerates to `n // 2` on synthetic data). Taken
  from the same row's `lambda_recovery_v2`.
- **truth** — § 3 (anchor) or § 2.1 (ρ\*, Stage 1).
- **Δ1 = v1 − truth**, **Δ2 = v2 − truth**.
- **the 2×2** — `{nw 1024, 8192} × {ols, ridge}` from the anchor script, to
  attribute any gap to the row count or to the penalty separately. (nw 1024
  + ols reproduces v1 exactly — contract-tested.)

Aggregation, per the support-synthetic convention: cell statistic = mean
over the 3 seeds; SE = std(ddof = 1)/√3; **bar = max(2·SE, 0.05)**. A cell
missing any of its 3 seeds is excluded from paired rules and logged.

---

## 5. Pre-registered predictions

Stated before any cell has run. Each names the rows it is evaluated on.

- **P1 — v1 sags where p/n is high.** Over cells with p/n ≥ 0.5 and a
  licensed anchor: mean Δ1 ≤ −bar (v1 reports **below** truth).
- **P2 — v2 tracks truth there.** Over the same cells: |mean Δ2| < bar.
- **P3 — both probes are right where there is room** (the machinery gate).
  Over cells with p/n ≤ 0.05: |mean Δ1| < bar **and** |mean Δ2| < bar.
- **P4 — no optimism.** Nowhere on the ladder does mean Δ2 ≥ +bar; and on
  the Stage-1 `null` arm (ρ\* = 0) v2's reported recovery stays inside its
  own shuffled-target chance band at every p/n.
- **P5 — the T-decline is at least partly a probe artifact.** On line C at
  d_sae = 2048, v1's reported recovery declines from its peak to T = 16 by
  ≥ bar while the anchor's decline over the same T range is < bar.

**The falsifier.** P1 is the campaign's load-bearing claim; **P1 failing
(v1 within bar of truth everywhere the anchor is licensed, including at
p/n ≥ 0.5) is a clean result that argues against adopting v2**, and is
reported as such, first, in the scorecard. P4 failing is a stronger
anti-v2 result and outranks everything else in the reading order.

---

## 6. Validity gates (checked first; a failed gate caps the reading)

- **G1 — the calibration reproduces the bench's known constants.** Stage 1
  ρ\* must land within 0.02 of 0.41 (`token` arm) and of 0.91 / 0.99
  (`full` arm at T = 2 / T ≥ 4), and the anchor procedure must recover ρ\*
  to ≤ 0.02 at every p/n where it is licensed. **Failure ⇒ no truth claim
  is licensed anywhere; the receipt reports branch 4 and says why.**
- **G2 — line M reproduces the committed dilution numbers.** M's v1 column
  vs `CARD.md` § 1.2's A1 seed-means (0.870 / 0.952 / 0.949 at T = 2/4/8)
  to ≤ 0.01. Failure does not by itself void the campaign (it would
  indicate a training-nondeterminism or environment difference) but is
  reported prominently and caps any claim that mirrors a committed number.
- **G3 — chance floors behave.** Every cell's `lambda_chance` and
  `lambda_chance_v2` must sit within ±0.05 of 0; cells that fail are
  excluded and logged (a nonzero chance floor means the split or the
  target is degenerate for that cell).
- **G4 — coverage.** The verdict rules require ≥ 3 licensed cells at
  p/n ≥ 0.5 with 3 seeds each. Below that the receipt reports branch 4
  (incomplete) whatever the partial pattern shows.

---

## 7. Reading rules → the pre-registered branches (mechanical)

`analyze_probe_truth.py` emits exactly one of these into `probe_truth.json`
as `branch_evidence`, together with the per-prediction table. **The label
names which branch of mac-local's rule the mirror evidence is consistent
with; it does not take the decision.**

- **`ADOPT-consistent` (branch 1)** — G1–G4 pass ∧ P1 ∧ P2 ∧ P3 ∧ P4.
  v2 tracks truth across the ladder where v1 sags.
- **`DECLINE-consistent` (branch 2)** — G1–G4 pass ∧ P3 ∧ P2 ∧ ¬P1. Both
  probes track truth even at p/n ≥ 0.5 ⇒ the mirror does not reproduce the
  capacity failure, and the real-panel lift needs a different explanation.
- **`REJECT-consistent` (branch 3)** — ¬P4: v2 reports above truth by ≥ bar
  on any licensed cell, or inflates on the `null` arm. Reported **first**
  in the scorecard regardless of what else holds.
- **`AMBIGUOUS` (branch 4)** — any gate fails, or coverage is short (G4),
  or a mixed pattern (e.g. P1 ∧ ¬P2 — v1 sags and v2 misses truth in the
  other direction). The pattern is spelled out; no adoption claim either
  way.

P5 is **commentary** on the T-decline reading, never a branch input: it
speaks to what the § 3b "peaks rather than saturates" sentence is allowed
to say, not to which probe is canonical.

---

## 8. Deliverables and process rails

`results/probe_truth.json` (every cell, both probes, the anchor, the 2×2,
the gate and prediction table, the coverage table, `branch_evidence`);
`figs/probe_truth.png` (reported recovery vs T for BOTH probes with the
TRUE level marked, per arm — plus the p/n panel that carries the actual
x-axis, § 1.1); the one-paragraph scorecard in `task_hunt/LOG.md`.

Commit order: **this card (freeze) → builds (calibration / grid / anchor /
analysis / render, each committed before it is run) → results → figure →
LOG scorecard → STATUS rewrite → push → stop for mac-local review** (the
briefing stays until reviewed). All numbers script-derived; no reviewer or
meeting quotes in tracked files. Deadline: results by Saturday morning PT.

_Frozen-by: runpod-b, 2026-07-24 22:00 UTC, before any Stage-1/2/3 build or
run commit._

---

## 9. AMENDMENT APPLIED (2026-07-25, post-freeze, disclosed — not a silent edit)

`briefings/mirror-probe-truth.md` gained a **binding amendment from
mac-local (2026-07-25)** after this card froze, superseding the briefing's
§ 1 and re-scoping the branch input. Its operative items:

1. **p/n is the campaign's x-axis, not T.** The four branches fire only on
   evidence swept through p/n ≈ 1.0. **A mirror result at p/n ≪ 0.1 fires
   NO branch** and must not be reported as though it did.
2. **The direct known-truth probe is the priority branch input** —
   constructed codes at set L0 with exact truth, plus the null code
   (truth 0), swept through p/n = 1.0, both probes, including the branch-3
   check "does v2 ever report ABOVE truth".
3. Coverage honesty accepted in advance (22/843 checkpoints is a real
   paired sample).
4. Deadline moved to **Saturday midday PT**; if nothing has fired by then,
   branch 4 applies and that is a good outcome.

**What this changes in the analysis, exactly.** The card's § 5 scoped
P1/P2 to trained cells "with a licensed anchor" and § 7 built the branch
map on them; the Stage-1 exact-truth deltas were reported but marked
`branch_input: False`. The amendment inverts that for the p/n ≥ 0.5
rows. `analyze_probe_truth.py` therefore emits:

- **`branch_evidence` (primary)** — computed by `branch_amended` on
  `amended_branch_input`: constructed cells at p/n ≥ 0.5 (v1's regime,
  § 1.1 arithmetic), T = 16, seed-means with the § 4 bar unchanged.
  A-P1 = v1 sags on ≥ 2/3 of signal cells (truth ≥ 0.1; the null arm
  cannot sag from zero); A-P2 = v2 within bar on ≥ 2/3 of all cells;
  REJECT keeps priority via § 5's P4 (unchanged) plus any constructed
  cell with d2 ≥ bar; G1's constants check still gates; the anchor
  licence does not (truth is exact by construction here). Mix arms join
  this input automatically as they land — same statistic, more truth
  levels at the same p/n.
- **`branch_evidence_frozen_card_scope`** — the § 5–§ 7 pipeline verbatim,
  retained so the re-scoping is visible and cannot quietly buy an
  outcome. Trained-ladder rows at p/n ≪ 0.1 remain reported as evidence
  that fires no branch (amendment item 1) — this was already this card's
  § 1.1 position and is unchanged.

Nothing else in §§ 2–6 changed: no arm, rung, seed, statistic, bar, gate
or anchor rule was touched. The amendment is scope, not measurement.

_Amendment-applied-by: runpod-b, 2026-07-25, after Stage 1 + Stage 2
completed and before any branch label was pushed; the frozen sections
above are untouched._
