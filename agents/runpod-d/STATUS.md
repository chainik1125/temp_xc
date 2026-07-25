# Working state — agent `runpod-d`

**Last rewrite:** 2026-07-25 ~19:15 UTC, POST-PANEL. **stage2-oprate
is EXECUTED AND CLOSED on my side: 84/84 cells, 0 failures, verdict
NEGATIVE (pre-registered branch) written to LOG + RECORD § 3d, all
artifacts pushed.** Briefing `briefings/stage2-oprate.md` stays until
mac-local review. I am otherwise UNASSIGNED; force-majeure pause list
still binds (no em-redo, no factory builds, no mirror Stage-3).

## Box (interim A40 pod — `briefings/a40-bootstrap.md` is authority)

Clone `/workspace/agents/runpod-d/temp_xc`; `source
/workspace/agents/runpod-d/env.sh` every shell (GPUs 0,1,2 mine;
3,4,5 runpod-e's). EPHEMERAL disk: anything not pushed does not
exist. Funding clock: started ≈ 11:40 UTC 2026-07-25, ~12 h.
- **RAM is a 301 GB cgroup CAP** shared with runpod-e —
  `/sys/fs/cgroup/memory/memory.oom_control` oom_kill counter is the
  ground truth (dmesg silent). Long-lived grid workers accumulate one
  ~8.5 GB datasource copy PER SEED → use
  `grid.run_pool(..., max_tasks_per_child=1)` (committed, additive).
  Window-arch workers peak ~30–41 GB each (T16 worst); tsae ~13 GB.
- Caches rebuilt this session (receipt byte-identical):
  `/workspace/conv_depth_caches/{ward_stream,base}`. distill NOT
  built. `traces.json` is gitignored — re-port per
  `results/c7_backtracking/stage_a/ATTRIBUTION.md` if lost.
- tsae trained cell = 5 h 01 m on this box (3 in parallel), CPU-bound
  (SequenceBuffer clones ~2.1 GB per step; buffer_tokens irrelevant
  to it). Non-tsae panel side ≈ 4 h at 6–7 fresh-process workers.

## What I delivered this session (all pushed to origin/arxiv)

1. Ward stream + base cache rebuilt from committed builders;
   `ward_stream_stats.json` byte-identical receipt.
2. `stage2-oprate` CLAIM + card FROZEN pre-run (`5b35f671`):
   `oprate/CARD_STAGE2.md` — anchor base/hs13, 84 cells, post at
   k = 8·T matched from cell one, buffer 524288 uniform, paired
   v1+v2 columns, claim on v1.
3. Datasource plugin `src/explorations/task_hunt/real_oprate.py` +
   `ward_real_oprate_{case,ver}_base_l12` registered (34 datasources,
   validate OK; 320 tests green).
4. Panel COMPLETE: 84/84 ok, 0 dup eval_keys, 0 null metrics
   (leaderboard 8,913 rows). Transcripts, figures
   (`oprate/figs/stage2_oprate_tscaling[_v2].*`), summaries with
   machine-readable band bookkeeping, evidence line, variance
   receipts (`support_stats/stage2_variance_oprate_case[_v2].*`).
5. **VERDICT (LOG + RECORD § 3d): NEGATIVE, pre-registered branch.**
   v1 flat (trend p=0.54); v2 rises (pre 0.158→0.261, T8 margins
   bounded >0 at n=3) but EVERY window cell < the label-side count
   OLS at matched T (0.198/0.226/0.270/0.360); untrained window codes
   already read 0.05–0.09 under v2. No latent-state language. P1,P2
   (on v1), P4 falsified; P3,P5 held; 3 batchtopk band mismatches
   recorded (baseline under-spent → negative is conservative).
6. `rate_ver` NOT started (briefing's own rule — tsae 5 h ≫ remaining
   window). Datasource is registered; one command runs it on a future
   box: `run_stage2 <workers> ward_real_oprate_ver_base_l12 [sel]`
   (+ `evidence_line.py ver` first).

## DO THIS NEXT (if I get context back before the pod dies)

1. **Nothing is claimed by me.** Check `briefings/` for new
   `for: runpod-d` files after pull.
2. Per bootstrap flex order if GPUs idle and >2 h remain: offer GPUs
   0–2 to runpod-e's replication cells; then B8 `slen` screen — but
   B8 needs fineweb caches (runpod-e's side), NOT mine.
3. When nothing useful remains: TELL THE OPERATOR to stop the pod
   (that point is effectively now on my side — panel done, receipts
   pushed; the flex items belong to runpod-e's cache territory).

## Traps (keep — all measured this session or inherited)
- cgroup OOM kills are silent (no dmesg): watch
  memory.oom_control oom_kill counter; fresh-process workers
  (max_tasks_per_child=1) are the fix; SIGKILL workers directly
  (SIGTERM unreliable), verify with ps + nvidia-smi.
- NO stash/checkout/rebase of tracked jsonl while grid pools run
  (mmap SIGBUS documented in grid.py); mid-run pushes go to the
  `arxiv-runpod-d-wip` side ref (force-push HEAD), real arxiv push at
  pool boundaries only.
- 5 grid workers OOM a 44 GB A40 (v2 eval ~12.7 GB GPU per worker);
  3 workers/GPU is the safe ceiling at d_in=4096.
- run_pool OVERWRITES its results file per invocation → receipts from
  leaderboard.jsonl (canonical, deduped); shard transcripts get
  distinct filenames (sel suffix).
- pgrep -f self-match; NaN→null leaderboard poisoning (0-null check
  after runs); n_steps=0 rows explain ~1e-3 replication gaps; force
  PreTrainedTokenizerFast for R1-Distill; OMP env from env.sh (16)
  overrides grid's setdefault(2) — cap total ≤ ~24 cores.
