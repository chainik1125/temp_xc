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
(+0.0029/+0.0037/+0.0103). A1 (iii) BARS a second family diagnostic —
the slot is consumed by the running full-recipe 20k diag. The A2
decision tree is PRE-STATED in RESULTS C3 (written BEFORE any diag
cell landed): diag T{16,1} passes L2 slope+level → propose card
amendment A2 = family's one L2 slot runs FULL L2 on r1-min
(append-then-run, loud PTR); T1 no-recover → no A2, low-T fixes enter
at L1; T16 no-hold → lane dies. All r1 T16 gains remain ORDER-FREE
(pooled composition — binding claim caveat, fcf62963b regime).

## IN FLIGHT RIGHT NOW (check these FIRST on resume)

- **GPU 1, pid 13644** — A1 family diagnostic `r1b-L2diag-20k` (FULL
  r1 btkonly recipe, 20k steps, T order 16→1→4): T16 lands ~00:40,
  T1 ~01:15 (**THE collapse-with-AuxK-live answer**), T4 then drain
  ~02:30. Log `/workspace/logs/tscale_l2diag.log`. Watcher
  `b0bjfqzbb` fires on first `[l1] DONE`. On T1 landing, walk the A2
  tree in RESULTS C3 — the trigger numbers are T16 ≥ 0.8985 (with
  slope) AND T1 ≥ 0.9035 vs the P1 s42 row (CARD § 2).
- **GPU 0, pid 20672** — t_sample attribution chain (L1, 4k, T16,
  r1-min backbone): `r1min-ts16-4k` (t_sample=16 = NO subsampling; if
  ≈ 0.9251 the win is per-sample-window-TopK + sequence serving, NOT
  the curriculum) lands ~01:10, then `r1min-ts5-4k` (locked absolute
  instance, asymmetry 3.2) ~02:15. Log
  `/workspace/logs/tscale_l1_tsample.log`. Watcher `bzauur1hp` fires
  on first DONE. Confound on record in C3: tokens/step scales with
  t_sample at matched steps.
- If resuming without watchers: read the two logs + tscale/results/
  l1_rows.jsonl tails.

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

1. ~00:40 diag T16 lands → record; ~01:10 ts16 lands → the
   curriculum-necessity read; ~01:15 diag T1 lands → **walk the A2
   tree** (RESULTS C3, verbatim triggers). Each: RESULTS addendum
   (C1 for diag cells, C3 addendum for ts cells) + LOG beat if
   decision-grade + push.
2. If A2 triggers: append A2 to CARD_SPLIT (append-only, timestamped,
   BEFORE launch), launch full L2 on r1-min (20k, dev {1,2,4,8,16},
   both k — GPU 0 frees ~02:15, GPU 1 ~02:30; ~5 h wall), loud PTR.
   If not: low-T fixes (per-position k floors, k-anneal, k_train
   floor at small T) enter as C4+ through L1 as written.
3. L3 only ever via canonical run_experiment + § 3 L2→L3 gates +
   seeds {1,2,42} + eval_cfg namespacing (explore: tscale) + fresh
   LOG PTR + mac-local ratification.
4. Menu not yet touched: multi_window exposure fix, k-anneal,
   per-position k floors, position-loss reweighting. Negatives to
   RESULTS.
5. Ledger actuals at session close; STATUS rewrite before next
   compact.

*Rewrite before any compact. — runpod-c*
