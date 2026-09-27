# Cross-ratification mini-card — txcwin trailing-novelty (mac-b, salvage W2)

Frozen BEFORE any gap-fill computation (commit-then-run; the pin for
every result produced under this card is this file's commit SHA from
`git rev-parse`). Scope: `briefings/salvage-mac-b.md`. Nothing under
`txcwin/` outside `crossratify/` is modified; disagreements with the
thread's own conclusions are flagged side-by-side, never overridden.
Everything here is PENDING TEAM REVIEW **and pending Andrii's review**.

## 1. What the audit already established (read-only, no compute)

Recomputed from committed artifacts (`focus_novresid.json`,
`focus_nov_8b.json`, `rawgate_gpt2_L6.json`, their `audit.py` +
my independent recompute):

- Controls PRESENT in their design: doc-level 80/20 split (seed 7,
  identical rows across arms); 3 trained seeds (1,2,42); untrained
  control per (arch,T) (single seed); measured code-budget calibration
  with the TXC-pre incalibrability honestly excluded (c4); raw-probe
  floor gate on gpt2 at T∈{4,16} (novelty_resid = CANDIDATE, gate
  retracted r1 when switch_clock failed it); label-build triage of the
  token-identity channel (unigram tercile-AUC ≈ 0.56) and the position
  channel (≈ chance on the detrended label).
- gpt2 @ T=8 (the claims' pinned T): c1 +0.248 (15σ), c2 +0.270
  (21.9σ), c3 +0.262 (11.3σ); worst-winner-seed > best-comparator-seed
  on all three. REPRODUCED from artifacts.
- 8B @ T=8: c1 2.6σ, c2 2.7σ (pass); **c3 1.9σ, non-strict — fails
  their own W3/W8** (TXC-post s1 = 0.198 outlier; its bootstrap CI
  [0.155, 0.446] is anomalously wide). 8B @ T=16 is the robust cell:
  post 0.507 (min 0.472) vs stacked 0.210, per-token 0.129 (8.8–12.4σ,
  strict) — and T=16 is what the report's "0.507 vs 0.129" quotes,
  while claims.jsonl pins T=8 and does not name the model. FLAG, not
  override.

## 2. Gaps this card fills

**GAP-A (CPU, local, $0): window-surface visible-cue baseline** — the
task-hunt standard control absent from their design. The label is the
kernel-smoothed (half-life 16, support 64, lags 1..64) trailing rate of
first-in-DOCUMENT token types, position-detrended; a T-window sees only
31.2% (T=8) / 53.3% (T=16) of the kernel mass and cannot compute
first-in-document from window tokens alone. Measured floor for "the
window sees repeats":

- Rows/split/probe: their `task_rows` (seed 11, max_rows 8000) and
  `score_task` (doc split seed 7, ridge lam=1, skill = Pearson r on
  held-out docs) VERBATIM, features in place of codes. Models: gpt2 +
  llama31 tokenization (the 8B label pack), T ∈ {4, 8, 16},
  label `nov_resid` (primary), `nov_rate` (disclosure).
- Arms (features from `token_ids` in the window only, plus disclosed
  train-doc statistics; exact spec = `visible_cue.py` at this pin):
  - **V-pos**: document-position features (log2(pos) + position-bin
    one-hots). Prediction: ≈ 0 on nov_resid (validates the detrend);
    large on nov_rate (Heaps trend).
  - **V-rep**: window repetition surface — kernel-weighted
    new-in-window indicator (the label's own construction restricted
    to the window), distinct-type fraction, last-token new-in-window
    flag, repeated-token fraction, max repeat count / T, mean log-gap
    to previous in-window occurrence.
  - **V-uni**: token-identity prior (their `type_mean_scores`
    estimator, train docs only): last-token type-mean + window mean of
    type-means. The known leak channel, in comparable r units.
  - **V-all**: all of the above jointly.
- Pre-stated reading, per model at the claim's T (claims c1–c3 compare
  dictionaries head-to-head, so they survive in FORM regardless; this
  gates the "surface-quiet case study" designation):
  - V-all ≥ best dictionary seed-mean at that T → surface-quiet
    designation FAILS (dq-style demotion).
  - best-per-token-dict ≤ V-all < best dictionary → "partially
    window-visible", disclosed prominently.
  - V-all < best per-token dict (gpt2 0.215 / 8B 0.129 at T=8) →
    surface-quiet at window scale CONFIRMED.
  - Prediction (pre-stated): V-rep lands well below the per-token
    dictionary; V-uni is the strongest surface arm but stays ≪ best
    dictionary, consistent with their unigram triage.

**GAP-B (Modal, est ≤ $5 of the $60 cap): raw gate at the claims' T
and on the 8B.** Their gate ran only gpt2 T∈{4,16}; it never ran at
the pinned T=8 on either model and never on the replication model at
all. Fill: `raw_arms` (their code verbatim) on novelty_resid +
novelty_rate; gpt2 L6 at T=8; 8B (DeepSeek-R1-Distill-Llama-8B) L12 at
T∈{4,8,16}. Caches rebuilt in-container by their `build_cache`
(stream-SHA-keyed), persisted to the Volume. Gate criterion verbatim
theirs: CANDIDATE iff max(gap_window, gap_mean) > 0.03.

- Pre-stated outcomes: both models pass at T=8 → gate gap closed,
  replication licence intact. 8B fails at all T → the 8B replication
  loses its temporal-structure licence (flagged for Andrii; c1–c3
  become gpt2-only under our controls). gpt2 fails at exactly T=8
  while passing 4,16 → boundary disclosure + re-pin discussion.
- Also reported: raw_last / raw_mean at T=8 so every dictionary skill
  can be quoted as a fraction of the raw ceiling (on gpt2 T=16 data,
  raw_last 0.572 exceeds every trained dictionary — this goes in the
  memo openly whatever the T=8 numbers say).

## 3. Deliberately NOT run (proposed to Andrii instead)

- 8B T=8 seed top-up (3→6 seeds; would resolve c3@T8 directly) — that
  is re-running their science, not adding a control.
- claims.jsonl amendment: name model + T per claim (or re-pin the 8B
  claims at T=16 where they are robust).
- Untrained controls at 3 seeds instead of 1 (minor; W4 margins are
  ~10× the plausible init spread).

## 4. Discipline

Ledger read-before/append-after the Modal launch; detach; containers
never push; results repatriated into `crossratify/results/` and merged
locally; verdict vocabulary in the memo: SUPPORTED /
SUPPORTED-WITH-GAPS (named) / NOT-REPRODUCED, per claim.
