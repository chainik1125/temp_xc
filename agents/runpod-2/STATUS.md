# runpod-2 — working state

**Am:** EM executor on the ACTMIX shared 3×H100 pod, GPU 2 only
(`source scripts/set_agent_env.sh runpod-2`). Clone
`/workspace/agents/runpod-2/temp_xc`, own venv (peft 0.19.1 added —
not in pyproject; needed for the LoRA-merge builders).

**Task:** `briefings/actmix-runpod-2.md` — EM shuffle control +
T-window sweep, both arms (btk-only now; paper-match blocked on
mac-c). Deadline: rebuttal-grade numbers before 17:00 London
2026-07-27.

## Setup pinned (recon done, 2026-07-26 ~21:15 London)

- **Task = § 5.3 medical organism detection** (Qwen2.5-7B-Instruct +
  `andyrdt/Qwen2.5-7B-Instruct_bad-medical`), paper layer **L15**
  (resid_post, hs16). Eval = `temp_bench.evals.em` (detection 3.0.0
  port, ALREADY IMPLEMENTED — `experiments/em/run.py`'s "stub"
  docstring is stale). Primary `pr_auc_S16`; within-window shuffle +
  realized-l0 are built into the eval.
- **Training stream** = `medical_em_prompts` BASE-forward
  (paper/origin-final convention: train on base, detect on organism
  — TRACKING.md § 1), datasource `qwen_2_5_7b_instruct_medical_l15`
  (configs/data.yaml). Runner never auto-builds caches → out-of-band
  builder writes `results/data_cache/<data_key>/acts.npy`.
- **Cohort eval cache** = stage-4 judge outputs recovered from
  `origin/final` git history; rebuilt via
  `phase4_em_depth.py cache` → `/workspace/conv_depth_caches/
  em_medical/` (all 29 hs). Cohort REPRODUCED EXACTLY: 1728
  rollouts, misaligned frac 0.323 (matches TRACKING.md).
- **Grid design** (card = `experiments/explorations/actmix_em/CARD.md`):
  TXC-post btk-only retrained per T ∈ {1,2,4,8,16}; SAE + TSAE
  btk-only trained once (per-token; tsae REJECTS T≠1 — bands);
  untrained twins = n_steps 0. Seeds {42,1} paired (paper em
  convention), s2 stretch. d_sae 32768, 20 atoms/token nominal
  parity (em-redo Part II convention; k_pos = 20·T per window for
  post), n_steps 25k, lr 3e-4, warmup 1k, bf16, batch 1024
  (tsae 32 seqs). Endpoint-first dispatch (T16+T1 s42 first).
- **Aniket convention** (read-only, origin/neurips-aniket): shuffle
  semantics ALIGNED (per-row within-window input permutation,
  deterministic seed — same as the em eval port). Their extra
  controls (reversal, circular shift, positional-stack SAE) are NOT
  in EM's protocol 3.0.0 — divergence stated with reason in card.

## Flags raised (for mac-local/mac-c — in CARD.md too)

- F1: `experiments/em/run.py` default datasource is the 14B FINANCE
  l24 anchor, but eval/cohort/published-negative are 7B MEDICAL L15.
  Executing medical-L15; finance not covered this phase.
- F2: substrate = base-trained/organism-detected per origin/final;
  dmitry-em-repl (actual paper-numbers source) may differ — Phase B
  waits on mac-c's pin; Phase A is labeled btk-only, never
  paper-match.
- F3: k-budget + registry names for btk-only come from mac-a's note
  (single-source). Card carries my default (20/token parity) marked
  PENDING; adopt mac-a's on arrival, before any cell launches.

## In flight (updated ~21:40 London)

- [DONE] cohort cache: 29 hs × 1728 rollouts at
  `/workspace/conv_depth_caches/em_medical/` — integrity reproduced
  (1728 / 0.323 / d_model 3584 vs TRACKING.md).
- [DONE] train caches: BASE-forward L15 (`56a61e3776062439`, paper
  convention) + organism L{9,13,15} (Phase-B insurance,
  `2d0a9b6176e91bad` at L15).
- [DONE] mac-a convention landed 92db86c41, mac-local APPROVED
  9e634bed9 (pods GO) — `*_btkonly` names consumed verbatim.
- [DONE] pre-freeze smokes: validate OK; all 3 btkonly archs green
  through em runner; T16/k320 override green; n_steps=0 green.
- [armed] origin listener (archs/, task_hunt LOG, COMPOSITION_AUDIT,
  briefings/) every 150 s — session-local.
- FREEZE pushed 9f6350372 (card BEFORE cells ✓). Launch history
  (all blind — zero cells had completed at each amendment):
  1. 3-lane a/b/c launch → OOM (T16‖T8‖T1: step peak ∝ T·B·d_sae;
     T16 tried 43+7 GB; FAILs preserved in wall logs a/b).
  2. Amendment 1 (own-history sha; superseded): 2-lane h/l regroup.
     Relaunch showed uncapped T16 caches ~73 GB → sibling starves.
  3. Amendment 2 PIN3=79c13e3f1: driver gains env-driven
     torch.cuda.set_per_process_memory_fraction guard
     (TEMP_BENCH_GPU_FRACTION; launch mechanics, train_key
     untouched); lane h = ALL T≥4 serialized @ 0.68, lane l =
     ≤T2 + per-token @ 0.22; PYTHONUNBUFFERED=1 for live logs.
- [RUNNING since ~22:20 London] lanes h (PID 9977) + l (PID 9978),
  pin 79c13e3f1, logs /workspace/logs/actmix_em_lane_{h,l}.log,
  wall-jsonl actmix_em_runs_{h,l}.jsonl. Runner cache-hits make
  relaunches resumable; a quiet lane is UNKNOWN, not dead — check
  `ps` + log mtime before re-dispatching.
- AMENDMENT 4 (~23:15, blind): measured 0.21 s/step at T1 2-way,
  ∝T scaling, fp32 trainer path (the path every v2 row used) ⇒
  T16-trained ≈ 15 h solo — physically past the deadline. DESCOPED
  T16-trained (both seeds; untrained T16 twin KEPT), s1 window
  cells dropped (= ladder rung 1). Trained curve = T ∈ {1,2,4,8}.
  t16 waiter DISARMED. At the T8s42 boundary (~10:20 ETA): kill
  lane h → relaunch lane f = [untT16, untT8, untT4, T4s42].
- RE-TASKED (Han 387268df0): EM grid completes as DMITRY-SUPPORT
  input (card/bars unchanged); EM Phase B CANCELLED; next = RLHF
  ablation (briefing appended to actmix-runpod-2.md; audit § 6 is
  the pin). RLHF prep IN FLIGHT: 4 shipped seed-42 ckpts +
  training_logs + phase-7 L12 train cache (txcdr-base-data)
  downloading to /workspace/caches/rlhf/; gemma-2-2b downloading;
  hh-rlhf npz NOT mirrored → rebuild via 023d52c24's
  build_hh_rlhf_cache.py recipe (short GPU job, slip into a lane
  gap). Paper-match arm = eval-only case-study artifacts (OUTSIDE
  leaderboard, probe_codes precedent); btk-only arm = canonical
  runner via a ported evals/rlhf.py (protocol 2.0.0; em.py port
  precedent). agentic_txc_02 is matryoshka-contrastive — no v2
  twin; btk-only TXC arm = txc_batchtopk_post_btkonly at paper
  SHAPES (d_sae 18432, k_win=100·T), difference stated in card.
  mac-a's identity note applies: k=500/token is DEEP selection —
  arms may genuinely diverge here (smoke + neg_frac gate first).
- AMENDMENT 5 (~00:10, blind): measured T8 = 3.6 s/step 2-way ⇒
  ~25 h — lane h KILLED (~1.5 h partial discarded, ledgered); T8
  joins T16 as physics-dropped. Final EM trained curve T ∈ {1,2,4}
  (T4 via lane g SOLO after lane l drains); untrained twins ALL of
  {1,2,4,8,16}; s1 = tokens only. Lane l solo at 95% SM.
- RLHF: FREEZE pushed 72b0ca729; cache rebuilt, integrity gate
  PASSED at phase-7's own recorded t-test (36.232/28.573/9.76e-10
  to the digit; gate initially mis-anchored to Ye's App B.1 —
  fixed + documented). **Paper-match arm COMPLETE + pushed**:
  topk_sae 0.613 | tsae_k500 0.631 (l0 547) | tsae_k20 0.610
  (l0 17.5) | agentic_txc_02 0.610→0.598 shuffled, gap +0.012.
  R-K1 PASS, R-K3 PASS (the paper's 3 length-spurious features
  reproduce EXACTLY), R-E1 HOLDS (order-insensitive, < 0.02).
- OVERNIGHT PLAN: lane l drains s42 lights + untrained + s1 tokens
  (per-cell wakes); at T2-s42-done launch EM lane g (T4 solo
  priority) + RLHF btk-only lane (frac 0.22: sae_k500 smoke w/
  neg_frac gate → T5 → T1 → sae_k100 → tsae shapes → twins; driver
  to write). ~13:00 hard stop for stretch cells; then analyze.py,
  table+figs both tasks, LOG verdicts PTR, ledger actuals, STATUS.
- **EM FULL STOP (Han, dca32ce6b ~00:15)**: lanes killed (T2 died
  mid-train), 3 landed cells NON-QUOTABLE-pending-Dmitry (txc@T1
  0.4597 l0 1056 | sae 0.5115 l0 233 | tsae 0.3923 l0 37 — E4 MISS,
  arch-dependent threshold-transfer flagged to mac-a), close-out
  LOG entry + ledger actuals-to-stop (−$40 corr) written ~00:30.
  Wall logs + caches + ckpts preserved as Dmitry handover. NOTHING
  EM relaunches. Priority test: task hunt / RLHF / sparse probing.
- **RLHF = the lane now.** Machinery ALL BUILT + pushed: evaluator
  port (protocol 2.0.0, shared impl in evals/rlhf.py; identity
  check vs papermatch = 6-decimal match), datasource
  gemma_2_2b_base_l12_phase7 (hardlinked phase-7 stream, data_key
  44b72320bc3a56e2), frozen cells (core-first, sae_k500 = smoke/
  neg_frac gate), pinned driver. Paper-match arm COMPLETE + pushed.
  Lane r RUNNING solo on GPU 2 (pin 55cb658, frac cap 0.22 —
  memory-only; full SM solo; if the T16 stretch is ever reached,
  relaunch that cell uncapped). ETA full core ≈ 07:00–10:00.
- Lane r was collateral in mac-local's ~00:15 kill sweep (GPU 2
  was empty after; runpod-1's probing shards are on GPUs 0/1) —
  RELAUNCHED ~00:50 uncapped, PID 22407, pin e6643da08, log
  actmix_rlhf_lane_r2.log; 4 done cells cache-hit instantly;
  rs-waiter (PID 22408) chains the stretch lane (T8 → untT8 →
  untT16 → T16) when r drains. Gemma cells are FAST solo (token
  10-16 min, T5 78 min) — full T-grid incl. T16 is feasible again.
- **RLHF btk-only landed so far (s42): sae_k500 0.625 (l0 535) |
  txc_T5 0.6229 gap +0.0033 (l0 517) | txc_T1 0.5777 (l0 108.5) |
  sae_k100 0.6130 (l0 108.3).** Early reads vs card: R-K1 PASS;
  R-E3 direction ✓ (harmonized T5 0.623 ≥ shipped 0.610); T5
  shuffle gap even smaller than shipped (+0.003 vs +0.012); R-E4
  narrow MISS (|0.578−0.613| = 0.035 > 0.03) at MATCHED realized
  l0 — arch-head difference, not sparsity. Scored properly at
  verdict time.
- [next] on wakes: triage cells; at r+rs drain (~04:00-09:00):
  seed-1 core set decision (token cells cheap), then analysis —
  both-arm Dmitry table + T-sweep fig + LOG verdict PTR + ledger
  actuals. 13:00 hard line for any remaining stretch. STATUS
  rewrite before any compact.

## Descope ladder (pre-stated, applied in order if time runs short)

1. drop seed 1 window cells (keep s42 curve + s1 token cells);
2. drop T=2 (keep {1,4,8,16});
3. drop T=8 (keep {1,4,16});
4. untrained twins s42 only (already default), tsae s42 only.
Never dropped: T=1 + T=16 endpoints s42, sae cell, shuffle eval
(in-eval), realized-l0 disclosure, honest side-by-side with the
paper's published negative.

## Git

Branch arxiv @ fd4cc10f9 (pulled 2026-07-26 ~21:00 London). Nothing
committed by me yet. Pull-rebase before every push; pod never
pushes without it.
