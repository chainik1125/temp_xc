# runpod-2 — working state (PRE-COMPACT REWRITE 2026-07-28 ~01:57 UTC / ~02:57 London)

**Am:** executor on the ACTMIX shared 3×H100 pod, GPU 2 ONLY
(`source scripts/set_agent_env.sh runpod-2` EVERY shell — unsourced
relaunch once nearly allocated on runpod-1's GPU 0). Clone
`/workspace/agents/runpod-2/temp_xc`, own venv. Autonomous loop:
git pull → read LOG tail → act → push (pull-rebase, keep-both LOG
resolver, push checked by EXIT CODE not grep — '! [rejected]'
contains 'arxiv -> arxiv'). Re-arm origin listener (150 s bash
fetch loop, run_in_background=true — NEVER disown, untracked
listeners can't wake me) after EVERY wake. Stamps: run `date`
FIRST, then write (repeated slips). Beware `&` precedence in
compound launches — one command per launch, verify child env via
/proc/<pid>/environ.

## DELIVERED tonight (all ratified unless PTR-marked)

1. **RLHF equivalence certificate 829f05070 (RATIFIED 89370c68a):**
   3/3 twins tensor-IDENTICAL through T16 (sae k500 / txc T5 / txc
   T16, s42; Δauc exactly 0; torch.equal all 7 shared tensors each;
   extra key threshold_set bookkeeping-only). Pre-registered T16
   divergence REFUTED + disclosed. Mechanism: boundary_min_pre ≥
   2.21/2.47 every logged step; dead_frac 0.654@T16 present but
   non-contesting; bit-identity RETRO-PROVES zero boundary contact
   at ALL 25k steps (addendum fd3e4ff16, lemma-pair w/ runpod-1's
   census divergence: identity ⇒ no contact ever; divergence ⇒
   rare contact — one mechanism, two venues; quote PER-TASK only).
   RLHF_EQUIVALENCE.{md,json} committed; receipts complete.
2. **A5b (RATIFIED 06fa6cee7): rmx_a CANCELLED-WITH-CERTIFICATE**
   (monotone selection floor + alias hygiene). **AUTO-RE-OPEN
   binding: any DIVERGENT rmx_b per-cell check re-cards rmx_a.**
3. **Cross-pod sha protocol (acked):** twin checks = file sha256
   (sha-equal ⟺ tensor-equal at same code version); my btkonly T8
   trio shas posted in LOG 00:16 entry; **T10 trio DUE at x10
   drain** (compute + post). runpod-b posts rmx_b relumix shas per
   landing (their pace ~100 min/cell, drain ~10:30 UTC);
   MISMATCH ⇒ auto-re-open + immediate report.
4. **Durability COMPLIANT (eb143b62a):** 26/26 trained RLHF ckpts
   on `temp-bench-data/ckpts/<train_key>/` (ratified path), T16
   spot-check HF-LFS sha MATCH; receipts in
   `actmix_rlhf/results/hf_durability_receipts.jsonl`; uploader
   `actmix_rlhf/hf_durability_push.py`. **x6/x10 + pf ckpts push
   at lane completion (standing cadence rule).**
5. **⚑ PAPER-FAITHFUL FREEZE 0c9605f1f (CARD § 8; sprint
   4ce0369de/606e4587d; PTR):** plugin `agentic_txc_02_v1t`
   (src/temp_bench/archs/agentic_txc02.py — vendored 94119bc08
   verbatim, upstream param names, min(3,T) scales, batch
   1024/512(T10)/256(T16), in-plugin plateau mirror: post-plateau
   zero-graph loss ⇒ Adam no-op, proven in tests). 11 contract
   tests green (tests/test_agentic_txc02.py). Lanes in cells.py:
   pf_pilot (T2/s42 gate) / pf_lo T{1,2,4}×3 / pf_mid T{6,8}×3 /
   pf_hi T{10,16}×3 / pf_anchor (T5 evals ×3 seeds). Anchors =
   txcdr-base agentic_txc_02__seed{42,1,2}.pt — ALL 3 SEEDS,
   staged at /workspace/caches/rlhf/agentic_anchors/, sha receipts
   in /workspace/logs/pf_staging.log; stage_anchors.py mints their
   v2 train_keys + provenance side-manifest (phase_b precedent).
   gemma-2-2b-IT pre-downloaded. Datasource = PAPER stream
   gemma_2_2b_it_l13_fineweb_24k128 (data_key 48d2d17ff88598d4,
   ANCHOR-FORCED — not phase7-l12); cache ABSENT here → rebuild at
   x-drain. Gates in card: G1 pilot-vs-upstream-log
   (agentic_txc_02_t2__seed42.json: 5800 steps, l0≈197/200), G2
   anchor-eval placement, G3 l0<k_win ReLU fingerprint. Est
   $31-51 expected / $105 no-plateau bound.

## RUNNING NOW (GPU 2)

x6 ‖ x10 (btk T{6,10}×3, A2, pin 829f05070, fracs 0.35/0.50,
launched ~00:15 UTC): logs
/workspace/logs/actmix_rlhf_lane_x{6,10}.log (nohup buffered —
flush on cell/exit; check runs jsonl + nvidia-smi, silence ≠
stall), jsonl actmix_rlhf_runs_x{6,10}.jsonl. MEASURED pace:
T6 cell 118 min (cell 1 auc .6350 l0 617), T10 cell 192 min
(cell 1 auc .6218 l0 1041) ⇒ **x6 drain ~06:10, x10 drain
~09:45 UTC (revised from 08:00-08:30)**. Plan at x6-drain:
start pf l13 substrate CO-RESIDENT in x6's vacated 0.35 frac
(established co-residency practice) so gates/pilot pull left;
grid-vs-relief projection at G1 per the acked rule (21b874820).
Watchers: x6 bhqw8y3zk / x10 b4mz7sims (grep '[lane xN]
DONE|FAIL', 240 s).

## THEN (frozen order)

1. **x6 drains (~05:30-06:30):** commit rows; x10 continues.
2. **x10 drains (~08:00-08:30):** commit rows → **btkonly T10 trio
   sha256 post** (LOG, trained-only n_steps=25000, pair by
   train_key) → HF-push x6/x10 ckpts (durability cadence) →
   ledger actuals (~$24 est).
3. **Paper-faithful sequence on GPU 2** (CARD § 8, sourced shell,
   pins via rev-parse ≥ 0c9605f1f):
   a. l13 activation cache build (~50 min GPU;
      build_activation_cache for gemma_2_2b_it_l13_fineweb_24k128).
   b. hh-rlhf@l13-IT eval cache (~25 min; same builder family as
      /workspace/caches/rlhf/cached_hh_rlhf which is l12-BASE —
      record fresh integrity stats).
   c. `.venv/bin/python -m experiments.explorations.actmix_rlhf.stage_anchors`
      → pf_anchor lane (3 evals) → **G2 gate**.
   d. pf_pilot (T2/s42) → **G1 gate** (STOP+report on fail).
   e. Grid: pf_lo ‖ pf_mid co-resident; pf_hi after OR sharded to
      free pod GPUs at the pin (offer standing; probing shards
      precedent: runpod-a shard E, runpod-c shard D).
   Ledger est/actuals per launch; per-cell HF ckpt push.
4. **10:15 London CHECKPOINT RENDER = deliverable of record**
   (b5c25b0f5 plan, my ack 21a5d352d: landed trios only, T4
   set-question default = data-driven; in-flight caption line;
   supersede at same paths if x10 DONE ≤10:40). PRE-RENDER CHECK:
   verify row selection excludes untrained-twin/control rows
   (runpod-1 band-pollution precedent 5a699a5d4). Original
   af7d0869b hard point superseded by the checkpoint plan: `.venv/bin/python -m
   experiments.explorations.actmix_rlhf.render_writeup_fig --tag
   final` (mono; renderer auto-carries the BINDING agentic_txc_02
   arch-disclosure footnote a9e9fc213/859fed058, bbox tight) +
   table + LOG beat. T4 btk (runpod-a's x4, DONE:
   0.6185/0.6108/0.6295) is NOT in the 7-point hard render;
   8-point exhibit render {1,2,4,5,6,8,10,16} follows grid
   completion. v2 columns NEVER labeled "paper base" (8fefb409d).
5. **rmx_b sha checks as runpod-b posts** (equal ⇒ certificate
   extension; unequal ⇒ AUTO-RE-OPEN, report immediately).

## Fleet refs (for reading wakes)

Paper-faithful probing = runpod-1's card d9235755b, shards on
runpod-a (E) + runpod-c (D). rmx_b = runpod-b GPU 1 (launched
00:05 pin 829f05070). Hunt = mac-c/mac-d (sycgen screen frozen,
evalage generating; NOT my lane). EM frozen (NOTHING relaunches).
Backtracking = Aniket-only. Framing guard on shuffle quotes;
T16 = 3-instrument adjacency ceiling. Churn-stop etiquette:
frozen cards beat racing directives; 15-min ack discipline when
the hub maps name me.

## Ledger

07-27 actuals ≈ $40 (A1 $27 + Ward $2 + eq $11). 07-28 queued:
x6/x10 ~$24 + pf substrate ~$4 + pilot ~$2 + grid $25-45
(pilot-gated). Caps intact ($150/day). Tokens at
/workspace/.tokens/* — values NEVER in git/logs.

## PTR open items

Certificate exemption readings (T5/T16 certificate-covered on
exhibit); § 8 l13-stream reading + anchor staging + plateau
mirror + G1-G3; stage2_variance_panels legacy test failure =
PRE-EXISTING, λ̂ lane's (flagged 01:48 LOG).

## Queued: morning manifest owner pass

mac-d flag 7af84fb80: 336 same-train_key content conflicts
pod↔local in checkpoints/manifest.jsonl (mirror-status rewrites
vs as-launched) — hub-coordinated owner pass in the morning;
runpod-1 queued too (8d74622e1). I attend as RLHF-ckpt owner.
My live manifest lines are APPEND-ONLY new train_keys
(stash-protected around pulls) — commit only my appends at
drain, keep-both on collision, NO unilateral conflict fixes.

## If resuming after compact

Read this + LOG tail from my 01:48 entry + CARD § 8. Check:
`nvidia-smi` (GPU 2 = x6/x10 or free), `tail -n3
/workspace/logs/actmix_rlhf_lane_x{6,10}.log`, `tail -n2
/workspace/logs/actmix_rlhf_runs_x{6,10}.jsonl`, `git log
--oneline -5` + pull. Execute THEN list in order. Background
tasks notify by ID — match against Watchers above; re-arm the
listener after every wake.
