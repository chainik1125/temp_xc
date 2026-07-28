# runpod-c STATUS — T-scaling hill-climb (2026-07-28 ~00:15 London)

**I am `runpod-c`**, alone on a dedicated 2×H100 pod (both GPUs mine),
workspace `/workspace/agents/runpod-c/temp_xc`, venv `.venv` (healthy),
data substrate synced+linked (acts + 38/38 probe cache; mirror at
`/workspace/caches/probing/hf_mirror`). Mission (Dmitry/Han 07-27):
make TXC T-scaling actually improve with T on § 5.1 sparse probing,
under the FROZEN dev/holdout discipline. Governing docs (all mine, all
committed): `experiments/explorations/tscale/{CARD_SPLIT.md,README.md,
RESULTS.md,make_split.py,l1_lib.py,run_l1.py,analyze_l1.py}`.

## The one-paragraph story so far

The recovered txc_pro recipe (revived as `txc_pro_r1*`) produced the
program's FIRST monotone-rising TXC T-curve but collapses at low T
(2 % latents active). Ablations exonerated contrastive AND matryoshka
for both the collapse and the win; C2 KILLED curriculum-alone on the
btk backbone (mac-local RATIFIED). **C3 (r1-min = aux losses off) is
the strongest family member: T16 k20 0.9251 / k5 0.8763 — program-best
BOTH k, first cells above the P1 20k references — but T1 stays
collapsed (0.8071) ⇒ same gate verdict: slope PASS, T16 PASS,
T1-level FAIL, NO PROMOTE as-is.** Aux-loss T16 harm is super-additive
(+0.0029/+0.0037/+0.0103). **The A1 family diag RESOLVED the A2 tree:
NO A2** (RESULTS C1-D, ~00:50): diag k20 T1 0.8974 / T16 0.9171 —
the program's FIRST RISING k20 curve at canonical 20k steps (Δ16
+0.0197 vs P1's −0.0150), mechanism CONFIRMED-partial (AuxK-live T1
recovery +0.099, census active-frac 0.021→0.120) — but T1 misses the
pre-stated 0.9035 floor by 0.0061 AND k5@T16 regressed 0.8711→0.8487
(below the § 3 preservation bar 0.8551). Discipline held as
pre-committed: the family's one diagnostic slot is SPENT, no
exception-lane L2 for r1-min, low-T fixes enter as NEW candidates
(C4+) through L1 as written. 4k gains were order-free; at 20k the
shuf gap flips POSITIVE (+0.0032) — first order-sensitive r1 signal.

## IN FLIGHT RIGHT NOW (check these FIRST on resume)

- **GPU 0, pid 23234** — **C5** `c5-batchsel-4k` (r1-min +
  train_select=batch: SUSTAINED pooled-admission pressure — the C4
  arrow), T1 then T16; T1 lands ~02:25, T16 ~03:30. Log
  `/workspace/logs/tscale_c5_batchsel.log`, watcher `b2bfxb1fd` on
  first DONE. § 3 gates as written: T1 ≥ 0.8844 is THE number; then
  T16 must hold ≈ 0.92 (H-fail-T16 = pooled budget kills the window
  win, echoing C2).
- **GPU 1, pid 22305** — **C4 T16 cell** (k-anneal; T1 already
  recorded 0.8171 = dose–response FAIL). Lands ~02:50 → C4 verdict
  line (did the anneal at least not hurt T16?). Log
  `/workspace/logs/tscale_c4_kanneal.log`, watcher `bxsnvg3cd` on
  second DONE.
- DONE tonight: diag CLOSED (20k monotone rising, A2 NO), ts
  attribution (interior max at ratio rule, tokens-confound dead).
- If resuming without watchers: read the two logs + tscale/results/
  l1_rows.jsonl tails; RESULTS.md index table is current.

## Verdict + numbers ledger (dev-8 s42, 4k steps unless noted; k20)

| arch/tag | T1 | T4 | T16 | note |
|---|---|---|---|---|
| baseline twin (pre_btkonly) | 0.8944 | 0.9099 | 0.8810 | l0 13.4/73/371 (threshold-lag at 4k) |
| r1-paper / r1-btkonly | 0.7985 | 0.8633/4 | 0.9153/0.9148 | twins training-identical; collapse T1 (2 % active) |
| r1b-nocontr | 0.7955 | — | 0.9177 | contrastive exonerated |
| r1b-nomatr | 0.8024 | — | 0.9185 | matryoshka exonerated |
| subseq-btk (C2) | 0.8944 (≡ twin) | 0.8928 | 0.8641 | KILL, mac-local ratified; threshold under-admit datum |
| **r1-min (C3)** | 0.8071 | — | **0.9251** | k5 T16 **0.8763**; super-additive aux harm; NO PROMOTE (T1) |

P1 20k references (CARD § 2): pre s42 k20 0.9135/0.9181/0.8985, SAE
band 0.9111 ± 0.0042; k5 pre 0.8417/0.8434/0.8651, SAE 0.8450.
Gate floors vs twin at L1: T1 ≥ 0.8844, T16 ≥ 0.8810, slope ≥ −0.0054.

## Process state

- **Frozen + pushed:** CARD 0 split (dev-8; holdout-28 untouched),
  pyramid gates, candidate-1 pre-regs, A1 amendment. C1/C2/C3
  verdicts in RESULTS.md (C3 includes the A2 pre-statement + ts16/ts5
  pre-declaration). LOG beats through "~00:08 London 07-28".
- **Plugins landed (all tested, 25/25 after runpod-1's parent
  telemetry patch — my subseq copy already carried the identical
  block):** `txc_pro_r1(_btkonly)`, probing.py `eval_consumes`
  dispatch (owner-endorsed + sanctioned), `txc_btk_pre_subseq_btkonly`.
- **Known non-mine failure:** tests/test_stage2_variance_panels.py
  (pre-existing, disclosed).
- **L1 harness:** `_FastSequenceServer` rng-identical GPU twin of
  canonical sequence serving (disclosed). 4k-screen caveats:
  threshold-lag l0, AuxK inert < 10 M tokens (the A1 basis).
- **Fleet notices absorbed:** mac-local PERSONAL-KEY STOP (no API
  keys in my lane — nothing to do); runpod-b width-match running.
- **Ledger:** day-1 actuals ≈ $17 + overnight est $35–40 posted; ts
  ablations add ~2 GPU-h (within est). Cap $150 fine.
- **Git:** clean at last push (C3 commit 6698c46c8 subject line);
  pull-rebase --autostash before every push; LOG conflicts = keep
  BOTH + delete marker lines (5× tonight); stamp from `date`.

## Next actions queue (in order)

1. ~01:15 ts16 lands → curriculum-necessity read (C3 addendum);
   ~01:35 diag T4 lands → complete the C1-D 20k curve table; ~02:15
   ts5 lands → asymmetry read. Push each (or batched if near).
2. **A2 RESOLVED: NO** (C1-D). Design C4 = low-T fix through L1 as
   written, AFTER ts16/ts5 land (they shape the space). Constraints
   (C1-D): AuxK-independent (screen-inert), attacks ACROSS-ROW latent
   concentration (k_train-anneal wide→20 / batch-diverse selection at
   small T — NOT per-position floors), tracks k5@T16 (the 20k
   regression failure mode). Both GPUs free by ~02:15/~01:35 —
   overnight C4 L1 cells are in budget if the design is clean;
   otherwise leave the morning slate.
3. L3 only ever via canonical run_experiment + § 3 L2→L3 gates +
   seeds {1,2,42} + eval_cfg namespacing (explore: tscale) + fresh
   LOG PTR + mac-local ratification.
4. Menu not yet touched: multi_window exposure fix, k-anneal,
   per-position k floors (ruled out for T1 per C1-D), position-loss
   reweighting. Negatives to RESULTS.
5. Ledger actuals at session close; STATUS rewrite before next
   compact.

*Rewrite before any compact. — runpod-c*
