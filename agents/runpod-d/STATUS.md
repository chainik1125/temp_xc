# Working state — agent `runpod-d`

**Last rewrite:** 2026-07-24 (session 1, mid-flight — GPU chains running).

## Who / where
GPU RunPod pod (H100 80 GB, 224 cores, 2 TB RAM), Linux at
`/workspace/temp_xc`, `/workspace/.agent_id` = `runpod-d`. Role:
**task-hunt arm A** (`briefings/task-hunt.md`). Branch `arxiv`, shared
with 4 other agents — **always `git pull --rebase` before pushing**
(credentials: `GIT_ASKPASS=/workspace/.tokens/git-askpass.sh`, already
set as `core.askPass`; identity configured).

## Environment (built this session — reuse, don't rebuild)
- `.venv` (torch 2.8+cu128) = the probe/training venv;
  **`/workspace/vllm_venv`** = separate vLLM 0.25.1 venv (candidate 3).
- `HF_HOME=/workspace/hf_cache`; tokens at `/workspace/.tokens/`.
- **Caches on the volume** (~1 h to rebuild if lost):
  `/workspace/conv_depth_caches/{ward_stream,base,distill}` (17 capture
  points each, fp16), `/workspace/task_hunt_labels/lambda_intensity/`
  (λ̂ grids incl. the dense Stage-2 variant),
  `/workspace/task_hunt_labels/forbidden_word/cotcontrol_data/`.
- `results/c7_backtracking/stage_a/traces.json` was missing from the
  branch and re-ported from `origin/aniket-ward-stage-b` per that
  directory's own ATTRIBUTION recipe. It stays untracked.

## Done (committed + pushed)
1. **Substrate rebuilt** and verified — reproduces
   `conversion_depth/RECORD.md` § 3 exactly (base L10 ant_kw
   0.843/0.886; distill L10 0.844/0.895).
2. **Candidate 1 (backtracking λ̂): screened, verdict KEEP (qualified)**
   — verdict paragraph in `task_hunt/LOG.md`, methods in
   `task_hunt/RECORD.md` § 1. T-scaling is real (g rises to +0.054 at
   T=32, no saturation, all 4 model×layer cells); **the order story is
   negative** (g_order ≤ 0 in 17/20 cells, shuffle costs ≤ 0.022).
   Three frozen predictions falsified and scored; one renderer
   rule-scoring correction disclosed in the LOG.
3. **Cards frozen pre-run** for candidates 2 and 3; Stage-2 plumbing
   (plugin datasource + two `configs/data.yaml` entries) committed.
4. **Leaderboard hygiene checked**: 7121 rows, 0 dup `eval_key`s.

## RUNNING RIGHT NOW (two chained background scripts — do not fan out)
**GPU serialization is mandatory** — running two raw-activation screens
plus the Stage-2 pool concurrently caused CUDA OOM in both (a T=64
flatten probe needs ≈28 GB for standardization alone). Hence:

- `/workspace/logs/chain.sh` → log `/workspace/logs/chain.log`:
  shuffle-receipt (finishing) → **proof-op screen** (candidate 2,
  resumes from its JSON) → **Stage 2** (84 cells, 2 workers,
  `run_stage2.py`, log `/workspace/logs/stage2_base.log`).
- `/workspace/logs/chain2.sh` → log `/workspace/logs/chain2.log`:
  waits for chain 1, then **candidate 3** vLLM generation
  (`fw_generate.log`) → cache + screen (`fw_screen.log`).

## Next actions (in order)
1. When the proof-op screen finishes: score it against
   `proofops/card.md` and write the KEEP/KILL paragraph into
   `LOG.md` + the ladder table into `RECORD.md` § 2. Partial result so
   far: `tir` per-token 0.614, g +0.028 (T=8) → **+0.049 (T=16)** →
   +0.032 (T=32), g_order turning **positive at T=16 (+0.011)** — a
   peak-then-decline shape (localized latent, STORY § 7) and the order
   component candidate 1 lacked. Needs T=64 + the `op` ambient anchor
   before any verdict (the card's real claim is g_tir − g_op rising).
2. Finish the shuffle-receipt writeup: `render` already emits
   `figs/shuffle_receipt.*`; the table is in `RECORD.md` § 3.
3. When Stage 2 finishes: build **the T-scaling figure** (recovery vs
   T, one line per arch — the money plot) and write the record. Only
   `buffer_tokens=524288` rows are headline (see RECORD § 4.5).
4. Candidate 3 verdict after chain 2; check its feasibility gate
   (violation rate ≥ 30 %) BEFORE reading the screen.
5. Update `experiments/explorations/synthetic/STATUS.md` with the arm-A
   outcome; leave `briefings/task-hunt.md` in place until mac-local
   reviews.

## Deadline
Results wanted by **2026-07-26 morning PT**. Ample wall-clock remains;
the constraint is GPU serialization, not the calendar.
