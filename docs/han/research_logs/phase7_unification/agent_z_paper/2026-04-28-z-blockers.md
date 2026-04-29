---
author: Han
date: 2026-04-28
tags:
  - design
  - in-progress
---

## Agent Z blockers — Phase 7 hill-climb on local 5090

### TL;DR

Brief assumed Z could pull base activation cache + probe_cache_S32 from HF
or shared filesystem. Neither is available. Z must build them locally
before any hill-climb training or probing can run. ~1.5–2 hr one-time
cost. Path A (build locally) selected; Path B (wait for X to upload) not
guaranteed reachable from a 5090 without HF assistance.

### Local environment (verified)

- GPU: RTX 5090, 32 GB VRAM, 27 GB free at start, driver 591.86, cc=12.0.
- System RAM: 54 GB (50 GB free).
- Disk: 517 GB free on `/dev/sdd` (1 TB total, 46% used).
- `.venv/`: torch 2.9.1+cu128, CUDA available, gemma-2-2b auto-detected.
- HF auth: token present (37 chars), `whoami=han1823123123`. Verified
  read access on the gated `google/gemma-2-2b` (config.json downloaded).
- HF cache locally: gemma-2-2b config only (model weights not yet
  downloaded; gemma-2-2b-it weights ARE present from prior IT-side
  work). FineWeb + probe-task datasets (ag_news, amazon_reviews,
  hh-rlhf, super_glue, winogrande, etc.) ARE cached.

### Blockers

**B1. No base L12 activation cache locally.** The Phase 7 anchor cache
at `data/cached_activations/gemma-2-2b/fineweb/resid_L12.npy` does not
exist. Only the IT-side cache is local
(`data/cached_activations/gemma-2-2b-it/fineweb/`, ~70 GB).

**B2. No probe cache locally.** Neither `experiments/phase7_unification/results/probe_cache/` (S=128 right-padded, the
input to the S=32 rebuild) nor `experiments/phase7_unification/results/probe_cache_S32/` (the active cache used by `run_probing_phase7.py`)
exists. Phase 5's IT-side probe cache IS local (66 GB at
`experiments/phase5_downstream_utility/results/probe_cache/`) — but
that's IT activations, not base.

**B3. HF data repos do not have what the brief expected.**

- `han1823123123/txcdr-base-data`: only `.gitattributes` + `README.md`
  (empty placeholder). The brief said "Z can use the existing
  base-side probe cache from `txcdr-base-data` if it still has the
  crosstoken tasks." That fallback is not reachable.
- `han1823123123/txcdr-data` (the IT-side data repo): has the IT
  activation cache and Phase 5 IT-side probe cache, but nothing
  base-side and nothing for Phase 7 S=32.
- `han1823123123/txcdr-base` (model repo): has 103 ckpts (incl.
  `hill_subseq_h8_T12_s5__seed42.pt` from Agent A) and 103 training
  logs, but no activation or probe caches.

**B4. V1 ckpt path in `training_index.jsonl` points at RunPod.** The
existing row reads `ckpt: /workspace/temp_xc/...` (Agent A's H200
path). That filesystem is gone with the dead pod. The HF copy is the
only source of truth.

### Decision: Path A — build base + probe caches locally

Time and disk costs (estimates):

| step | runtime | disk | notes |
|---|---|---|---|
| download gemma-2-2b base weights | ~3 min | ~5 GB | gated, token verified |
| build base L12 activation cache | ~30–45 min | 14 GB | single layer; reuse `token_ids.npy` from IT-side cache (same tokenizer) |
| build probe cache (anchor only) | ~30 min | ~25 GB | only L12 needed for hill-climb archs; skip MLC-tail tasks |
| rebuild_probe_cache_s32 | ~5 min | ~6 GB | per-example slice 128→32 |
| pull V1 ckpt from HF | ~1 min | ~150 MB | one-time |
| **probe V1 (first signal)** | ~5 min | — | end of pre-flight |

Total to first probing result: ~75–90 min. Total disk footprint added:
~50 GB. Disk budget is fine (517 GB free).

Per Han's permission, IT-side caches (~70 GB activations + 66 GB Phase
5 probe cache + DeepSeek cache) can be deleted to recover space if
needed. Holding off until I know whether `build_probe_cache_phase7.py`
reads from the Phase 5 cache as a tokenized-dataset source — see B6
below.

### Open questions / sub-blockers

**B5. Hill-climb max_steps deviation.** `round1_subseq_t_sweep.py` and
`round2_long_shifts.py` both hardcode `cfg = TrainCfg(seed=seed, max_steps=8000)`. Paper-canonical training_constants in
`paper_archs.json` use `max_steps=25_000` with `plateau_threshold=0.02`,
`min_steps=3000`. Agent A's V1 trained with `max_steps=8000` and
`final_step=3000` (plateau hit at the floor). Whether paper leaderboard
cells terminated at similar step counts is the question — if yes, 8000
is fine; if no, V1 likely under-trained vs the leaderboard targets and
the comparison is biased downward against Z. Will spot-check a few
existing leaderboard `training_logs/*.json` for typical `final_step`.

**B6. Does `build_probe_cache_phase7.py` need raw datasets or can it
re-use the Phase 5 cache?** Will read the script before deciding
whether to delete the Phase 5 IT probe cache.

### Action plan (in order)

1. Spot-check leaderboard cell `final_step` distribution to settle B5
   (decide max_steps for hill-climb cells).
2. Build base L12 activation cache (5090, single layer only).
3. Build probe cache + rebuild S=32 (anchor-only; FLIP tasks
   `winogrande_correct_completion` + `wsc_coreference` must be
   present).
4. Pull V1 ckpt from HF, rewrite local `training_index.jsonl` row
   path, run `run_probing_phase7.py --run_ids hill_subseq_h8_T12_s5__seed42 --S 32 --k_feat 5 20`. **Compare to current #1
   k_feat=5 0.8989 and k_feat=20 0.9358.**
5. (Then) hill-climb V2/V3 + round2 per the brief.

### What this changes for X / Y

- Z's probe cache is local-only initially. Once built and validated, Z
  could push it to `txcdr-base-data` so that future agents (X-H200,
  later Z sessions) don't re-build. Ask Han first — uploading 30 GB
  to a public HF dataset is a non-trivial action.
- If X completes the FLIP-included probe-cache rebuild on the A40 in
  parallel, the canonical version is X's. Z's local copy is a fork
  used only to make probing self-contained on the 5090.
