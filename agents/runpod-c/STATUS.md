# runpod-c STATUS — T-scaling hill-climb (PRE-COMPACT REWRITE, 2026-07-27 22:56 London)

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
program's FIRST monotone-rising TXC T-curve on the dev-8 (k20: 0.7985
→ 0.8633 → 0.9153 at 4k steps vs baseline twin 0.8944 → 0.9099 →
0.8810) but fails the frozen T1-level gate — its low-T cells suffer
ACTIVATION COLLAPSE (2 % of latents active; census in rows). Ablations
then EXONERATED contrastive AND matryoshka for both the collapse and
the T16 win — stripped variants score BETTER at T16 (nocontr 0.9177,
nomatr 0.9185) — so the SUBSEQ CURRICULUM ALONE carries the effect.
Wave-3 grafts that curriculum onto the healthy BatchTopK backbone
(`txc_btk_pre_subseq_btkonly`), whose T=1 cell is bit-identical to the
baseline by construction (anchor gate passes trivially; CONFIRMED in
production: T1 0.8944 ≡ twin). Its T16 cell decides everything.

## IN FLIGHT RIGHT NOW (check these FIRST on resume)

- **GPU 0, pid 19083** — `r1b-min-4k` (`txc_pro_r1_btkonly` with
  contrastive_alpha=0 + h_size=18432 + contr_prefix=3686 = the
  MINIMAL subseq+TopK+auxk recipe), T {1, 16}, 4k steps. Launched
  ~22:55; lands ~23:55 London. Log
  `/workspace/logs/tscale_l1_r1min.log`. **DECISION:** if T16 holds
  ≈0.918 (nocontr/nomatr level), r1-min is the L2 candidate proper —
  its remaining problem is ONLY the T1 collapse (L2 diag answers the
  AuxK half). If T16 drops, the removed pieces interact — re-attribute.
- **GPU 1, pid 13644** — A1-exception **L2 diagnostic**
  `r1b-L2diag-20k` (full r1 btkonly recipe, 20k steps, T order
  16→1→4): T16 ~14k/20k at 22:57; lands ~00:40, then T1 (~45 min,
  THE collapse-with-AuxK-live answer), then T4; drain ~02:30. Log
  `/workspace/logs/tscale_l2diag.log`. Per CARD § A1 it cannot reach
  L3 without the § 3 gates as written.
- **Watcher still armed:** `bldz2fiv3` fires on L2diag first DONE.
  (Wave-3 watcher consumed — C2 KILL processed and pushed.) If
  resuming without watchers: read the two logs + l1_rows.jsonl.

## Verdict + numbers ledger (dev-8 s42, 4k steps unless noted; k20)

| arch/tag | T1 | T4 | T16 | note |
|---|---|---|---|---|
| baseline twin (pre_btkonly) | 0.8944 | 0.9099 | 0.8810 | l0 13.4/73/371 (threshold-lag at 4k) |
| r1-paper / r1-btkonly | 0.7985 | 0.8633/0.8634 | 0.9153/0.9148 | twins training-identical; collapse T1 (2 % active) |
| r1b-nocontr | 0.7955 | — | 0.9177 | contrastive exonerated (collapse + win) |
| r1b-nomatr | 0.8024 | — | **0.9185** | matryoshka exonerated; stripped best |
| subseq-btk (wave 3) | 0.8944 (≡ twin) | 0.8928 | 0.8641 KILL | curriculum does not transfer; threshold under-admit datum |

k5 mirrors the story (r1 T16 0.8610–0.8711 vs twin 0.8267). ALL
r1-family T16 gains are ORDER-FREE (shuffle gap ≈ 0 or negative) —
pooled composition, NOT sequence structure — binding caveat for any
eventual claim framing (fcf62963b regime). P1 20k-step baselines +
SAE bands: CARD_SPLIT § 2.

## Process state

- **Frozen + pushed:** CARD 0 split (dev-8 listed in card;
  holdout-28 untouched), pyramid gates, candidate-1 pre-regs, A1
  amendment (mechanism-exception L2, invoked for r1 with L0 receipt
  `frac_dead_threshold 0.0`).
- **C1 verdict:** slope PASS / T1-level FAIL → NO PROMOTE as-is
  (RESULTS.md C1 section + LOG 20:38 entry). Twin-drop decision:
  btkonly carries; paper twin = faithfulness receipt.
- **Plugins landed (all tested):** `txc_pro_r1(_btkonly)` (16+2
  tests), probing.py `eval_consumes` dispatch (owner-endorsed
  runpod-1 LOG 20:45; mac-local "fully sanctioned, no veto" 19:46;
  T=1 identity tests added both lineages), `txc_btk_pre_subseq_btkonly`
  (7/7 incl. bit-equal parent degeneration + slab-grad-leak).
- **Known non-mine failure:** tests/test_stage2_variance_panels.py::
  test_legacy_default_reproduces_committed_receipts fails PRE-EXISTING
  (fails with my work stashed; disclosed in plugin commit).
- **L1 harness notes:** `_FastSequenceServer` = rng-identical
  GPU-resident twin of canonical sequence serving (disclosed; only
  for consumes='sequence' scratch cells). 4k-screen caveats on
  record: threshold-lag l0 (~13/20 at T1) and AuxK inert < 10 M
  tokens (the A1 basis). Scratch rows: tscale/results/l1_rows.jsonl
  (committed; ckpts gitignored under results/ckpts/).
- **Ledger:** day-1 actuals ≈ $17 + overnight est $35–40 (posted
  20:38 line). Cap $150 fine.
- **Git:** clean at last push (wave-3 plugin commit); pull-rebase
  --autostash before every push (l1_rows grows under running
  screens); LOG conflicts = keep BOTH + delete marker lines (done 4×
  tonight); cite subject lines not SHAs.

## Next actions queue (in order)

1. Process wave-3 T16 (decision tree above) → RESULTS.md C2 section
   + LOG PTR + push. Include the T4-dip datum (0.8928 < T1) honestly.
2. Process L2 diagnostic as cells land (T16 ~00:40, then T1 —
   the collapse-with-auxk-live answer — then T4) → RESULTS C1
   addendum + LOG. If r1@20k T16 ≥ its 4k level AND T1 recovers
   toward twin: the stripped-r1 (subseq-only per-sample-TopK) at 20k
   becomes an L2 candidate proper.
3. Whichever lane passes its gates first → L2 full (20k, dev
   {1,2,4,8,16}, both k) → then L3 holdout (canonical runner,
   eval_extra `explore: tscale`, seeds {1,2,42}) ONLY via CARD § 3
   gates + fresh LOG PTR + mac-local ratification. L3 must ALSO go
   through the canonical ProbingEval (already compatible: subseq-btk
   consumes='window' natively; r1 via the sanctioned eval_consumes).
4. Menu items not yet touched: t_sample-ratio ablation (5 vs 8 @T16),
   multi_window exposure fix, k-anneal, per-position k floors,
   position-loss reweighting. Log negatives in RESULTS.
5. Ledger actuals at session close; STATUS rewrite before next
   compact.

*Rewrite before any compact. — runpod-c*
