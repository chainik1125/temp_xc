# Working state — agent `runpod-d`

**Last rewrite:** 2026-07-25 ~13:05 UTC, MID-PANEL (stage2-oprate
executing). Force-majeure A40 pod (`briefings/a40-bootstrap.md` is the
box-facts authority — old H100 box facts below in git history are
DEAD). ~12-h funding clock started ≈ 11:40 UTC 2026-07-25.

## Who / where / setup (THIS pod — interim 6×A40)

Clone `/workspace/agents/runpod-d/temp_xc`; `source
/workspace/agents/runpod-d/env.sh` in EVERY shell (pins
CUDA_VISIBLE_DEVICES=0,1,2 — GPUs 3–5 are runpod-e's; verify
device_count()==3). Ephemeral disk: ANYTHING NOT PUSHED DOES NOT
EXIST. `git pull --rebase --autostash` before every push (LOG
append-only union; leaderboard/manifest have merge=union). Durability
side-channel: `origin/arxiv-runpod-d-wip` (force-pushed HEAD when
arxiv rebase is inconvenient mid-run).

- 320 tests green on this box. `run.py validate` OK (34 datasources).
- Models pulled to shared `HF_HOME=/workspace/hf_cache`: base
  Llama-3.1-8B + R1-Distill-8B.
- **Caches REBUILT this session** (receipt: `ward_stream_stats.json`
  byte-identical under git): `/workspace/conv_depth_caches/ward_stream`
  + `/workspace/conv_depth_caches/base` (17 capture points, 72 GB).
  `traces.json` re-ported per stage_a/ATTRIBUTION.md (gitignored, local
  only — re-port again if lost). Distill cache NOT built (not needed:
  anchor is base/hs13).

## ASSIGNMENT IN FLIGHT — `briefings/stage2-oprate.md` (CASE STUDY #2)

CLAIMED (LOG 2026-07-25 claim line, pushed `3a8b1f21`). Card FROZEN
pre-run + pushed `5b35f671`: `oprate/CARD_STAGE2.md` — READ IT; it
binds everything (anchor base/hs13; 84 cells = 42 trained + 42
untrained; post at nominal k=8·T matched; buffer_tokens 524288
uniform; V2 paired columns on every row per PROBE_V2_SPEC §2; claim on
v1 — the taken methods decision; realized-l0 band [5.0, 8.25];
untrained post must realize exactly 8.00 ±0.02 or post arm VOID;
evidence-line analog is the latent-state bar).

**Running now (Pool A ≈ 12:47 UTC; Pool B relaunched as shards
≈ 13:20 UTC):**
- Pool A (GPU 1): `run_stage2 4 ward_real_oprate_case_base_l12
  only-tsae` — 6 cells, 3 trained tsae are multi-hour CPU-bound
  (SequenceBuffer clones ~2.1 GB per step — structural; buffer_tokens
  does NOT touch it; scheduled first per addendum; panel reportable as
  partial-with-tsae-pending if needed).
- Pool B = TWO shards after a 44 GB OOM at 5 workers (v2 eval peaks
  ≈ 12.7 GB/worker; one stacked/T2/s1 trained cell failed and simply
  reruns): `skip-tsae:0/2` 3 workers GPU 0 + `skip-tsae:1/2` 3 workers
  GPU 2, both with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
  Scheduling-only amendment, committed; cell content byte-identical.
  Old aborted transcript kept as `...__skip-tsae.json.aborted`.
- Results append to `oprate/results/stage2_..__{only,skip}-tsae.json`
  (separate files so pools can't clobber; leaderboard.jsonl is
  canonical). A Monitor polls counts/failures every 2 min.
- First smoke cell verified: untrained batchtopk v1=0.016 v2=0.021
  l0t=8.000 exactly (chance + exactness both as predicted).

**Receipts already on the record:**
- Evidence-line regression analog (card §3): r = 0.183 / 0.198 /
  0.226 / 0.270 / **0.360** at T = 1/2/4/8/16; drops 16%/23%
  (train/eval pools) per T — `oprate/results/evidence_line_case.json`.
  Window cells must beat this at matched T for latent-state language.
- Renderer ready: `oprate/render_stage2.py` (v1 + paired-v2 figures,
  evidence line drawn, band bookkeeping machine-readable).

## THE QUEUE AFTER CELLS LAND (card §6; stop early at any gate is fine)

1. Push per completed batch (leaderboard + manifest + pool JSONs;
   autostash rebase; wip ref as fallback).
2. Merge pool JSONs → `stage2_ward_real_oprate_case_base_l12.json`
   (render_stage2.merge_pools does it), run renderer.
3. Variance receipts EXACTLY per
   `support_stats/PANEL_RECIPES.md § runpod-d` (--post-k-rule times-T,
   --row-layout auto default, v1 then v2 out-prefixes
   stage2_variance_oprate_case[_v2]).
4. LOG verdict + scorecard (P1–P5 from card §4 each scored
   held/falsified; band mismatches listed; window-drop counts per T;
   0 dup eval_keys / 0 null metrics check), RECORD section, figure,
   STATUS rewrite. Leaderboard hygiene check:
   `python -c "...jsonl dup/null scan..."` before verdict.
5. `rate_ver` panel ONLY if 1–4 done AND >2 h left on funding clock
   (datasource `ward_real_oprate_ver_base_l12` already registered;
   same runner with ds arg; evidence_line.py ver first).
6. When nothing useful remains: TELL OPERATOR to stop the pod.

## HARD GUARDS (unchanged + force-majeure)
- v1 canonical, v2 paired, never v2-as-canonical. Do NOT relitigate
  probe capacity; do NOT run anything for it.
- PAUSED program-wide: em-redo, factory builds/screens, mirror Stage-3.
- No panel re-runs for probe questions; no max-over-arms; canonical
  runner only; card bookkeeping discharged in the verdict.
- runpod-e owns GPUs 3–5 + ~24 CPU cores; keep my pools ≤ ~24 cores.

## Traps already hit (keep)
- **NEW (cost a pool): `git pull --rebase --autostash` (or any
  stash/checkout that rewrites tracked jsonl) WHILE grid pools run
  SIGBUS-kills workers mid-cell** (the mmap trap grid.py documents for
  its own results file) → BrokenProcessPool, pool dies, training work
  lost. Mid-run git discipline: add/commit new files + push to the
  `arxiv-runpod-d-wip` side ref ONLY; real arxiv rebase+push at pool
  boundaries. Also: SIGTERM does not reliably kill grid workers —
  SIGKILL the worker PIDs and verify with `ps`/`nvidia-smi`.
- run_pool OVERWRITES its results file on re-run → receipts from
  leaderboard.jsonl only.
- Killing a ProcessPoolExecutor parent orphans workers → kill worker
  PIDs, verify nvidia-smi.
- pgrep -f self-match; NaN→null leaderboard poisoning (check 0 null
  metrics); n_steps=0 rows explain ~1e-3 replication gaps; force
  PreTrainedTokenizerFast for R1-Distill; OMP_NUM_THREADS: env.sh=16
  overrides grid's setdefault(2) — workers can spike to ~16 threads in
  BLAS sections, watch total ≤24 cores.
