---
author: Han
date: 2026-04-29
tags:
  - design
  - in-progress
---

## Agent X handover — IT-side results + missing MLC sparse cells

> Pre-compact handover briefing. Next agent's mission: (1) probe the
> Gemma-2-2b-IT subject model under Phase 7 current methodology to
> close the IT-side leaderboard gap; (2) train + probe `mlc_sparse`
> (mlc, k_win=100) and `ag_mlc_08_sparse` (agentic_mlc_08, k_win=100)
> if feasible on A40.
>
> Read this entire doc plus `agent_x_brief.md` and the latest
> `agent_x_paper/README.md` before starting.

### TL;DR

- **IT-side (Gemma-2-2b-it L13)**: 12 paper cells × 1 subject model
  entirely missing from `probing_results.jsonl` under Phase 7
  methodology. Need: build IT activation cache (~2 hr), build IT
  probe cache (~30 min), train 8 leaderboard archs × ≥2 seeds (~10 hr
  if all A40_ok), probe (~1 hr) on the **PAPER** task set first;
  expand to FULL only if budget allows.
- **`mlc_sparse` / `ag_mlc_08_sparse` (k_win=100)**: tagged H200_required
  in `paper_archs.json` because MLC's 5-layer cache exceeds A40 VRAM
  (70 GB > 46 GB) and the cgroup RAM cap (46 GB) at b=4096. May be
  feasible on A40 with disk-mmap streaming refactor (untested) or with
  `n_seqs` reduced to 6000. Try and document.
- **Source-of-truth task set**: `experiments/phase7_unification/task_sets.py::PAPER`
  (16 tasks, finalised 2026-04-29). Iterate on PAPER; re-run on FULL
  only at the end if time permits. See `2026-04-29-paper-task-set.md`
  for rationale.
- **Branch**: `han-phase7-unification` (push to GitHub via the inline-URL
  pattern with the PAT at `/workspace/.tokens/gh_token`).
- **Key code/files**:
  - `experiments/phase7_unification/task_sets.py` — FULL, PAPER, HEADLINE
  - `experiments/phase7_unification/paper_archs.json` — locked arch spec
    (now includes a top-level `"task_sets"` block)
  - `experiments/phase7_unification/build_leaderboard_2seed.py` — leaderboard builder, defaults to PAPER
  - `experiments/phase7_unification/build_tsweep_2seed.py` — T-sweep builder
  - `experiments/phase7_unification/results/probing_results.jsonl` — append-only AUCs (X owns)
  - `experiments/phase7_unification/results/training_index.jsonl` — ckpt registry
  - `docs/han/research_logs/phase7_unification/agent_x_paper/` — paper deliverables bundle

### Pod environment expected

- A40 (46 GB VRAM, 46 GB cgroup RAM cap, 900 GB persistent volume).
- `/workspace/temp_xc` is a checkout of the repo; `.venv` from `uv sync`
  is the active environment.
- HF cache at `/workspace/hf_cache`; tokens at `/workspace/.tokens/{gh_token,hf_token}`.
- Always `export HF_HOME=/workspace/hf_cache UV_LINK_MODE=copy` before
  `uv sync` (MooseFS/UV-linkmode bug on this volume).
- Always `TQDM_DISABLE=1`. Never run multiple GPU python processes
  concurrently — the pod has been OOM-killed twice that way.

### What's already done (so don't repeat)

#### Source-of-truth files

- `experiments/phase7_unification/paper_archs.json` — 12-cell paper
  arch spec, with a new top-level `"task_sets"` block defining FULL
  (36) and PAPER (16) and policy `default_used_by_paper_artefacts: "PAPER"`.
- `experiments/phase7_unification/task_sets.py` — FULL, PAPER, HEADLINE
  frozensets. All builders import `HEADLINE as PAPER`. Backwards-compat
  aliases `FULL_36` / `PAPER_16` defined for legacy reads.
- `experiments/phase7_unification/results/probing_results.jsonl` — ~6500
  rows, deduped. Covers seed ∈ {1, 2, 42} for all base-side leaderboard
  + T-sweep cells. **All BASE-side / no IT-side rows under current
  methodology yet.**
- `experiments/phase7_unification/results/training_index.jsonl` — append-only
  ckpt registry. Add an entry for every new training (X did this for
  `phase5b_subseq_h8__seed2` after probing it without a registry entry).
  See the existing entries for the exact JSON schema (run_id, arch_id, T,
  T_max, t_sample, k_win, k_pos, n_layers, src_class, seed, batch_size,
  ckpt path, etc.).
- `experiments/phase7_unification/results/results_manifest.json` — coverage
  map; rebuild with `build_results_manifest.py` after any probing batch.

#### BASE-side coverage (Gemma-2-2b L12)

10 of 12 paper cells covered at 3 seeds × 16 PAPER tasks (or 2 seeds
where seed=2 ckpt isn't on HF):

| paper_id | arch_id | k_win | seeds covered |
|---|---|---|---|
| tfa | tfa_big | 500 | {1, 42} |
| tsae_k20 | tsae_paper_k20 | 20 | {1, 2, 42} |
| tsae_k500 | tsae_paper_k500 | 500 | {1, 2, 42} |
| mlc | mlc | 500 | {1, 2, 42} |
| **mlc_sparse** | **mlc** | **100** | **MISSING** ← need to add |
| ag_mlc_08 | agentic_mlc_08 | 500 | {1, 2, 42} |
| **ag_mlc_08_sparse** | **agentic_mlc_08** | **100** | **MISSING** ← need to add |
| txc_t5 | txcdr_t5 | 500 | {1, 2, 42} |
| txc_t16 | txcdr_t16 | 500 | {1, 42} |
| good_txc_p5 | phase5b_subseq_h8 | 500 | {1, 2, 42} |
| good_txc_p7_k20 | txc_bare_antidead_t5 | 500 | {1, 2, 42} |
| good_txc_p7_k5 | phase57_partB_h8_bare_multidistance_t8 | 500 | {1, 2, 42} |

#### Headline numbers under PAPER (3-seed) — for sanity-check after you finish

If you re-run on the BASE side after closing the gap, you should get
roughly these numbers (top entries):

- **k_feat=20 winner: `txc_bare_antidead_t5`** at 0.9127 (σ_seeds 0.0012),
  Δ vs `topk_sae` mean-pool = +0.0036 (~6× σ_seeds, decisive).
- **k_feat=5 top-6 within 0.0035 AUC** — no single defensible champion;
  `mlc` 0.8707, `topk_sae` 0.8695, `txc_bare_antidead_t5` 0.8683 etc.

### Mission #1 — IT-side leaderboard

Goal: probe Gemma-2-2b-IT (anchor L13) leaderboard archs at PAPER set.

**Anchor / layers (different from BASE!):**
- subject model: `google/gemma-2-2b-it` (instruction-tuned)
- anchor layer: L13 (0-indexed) — *not* L12 like BASE
- MLC layers: L11..L15 — *not* L10..L14 like BASE

These are different from what `_paths.py` defaults to (which is base
side). You'll need to either set env-var overrides or branch the path
config. See `experiments/phase5_downstream_utility/build_multilayer_cache.py`
for the IT-side cache build script (Phase 5 was IT-side).

**Plan:**

1. **Build IT activation cache** (~2 hr A40, ≤ 14 GB VRAM headroom):
   - Token IDs: reuse from BASE (`data/cached_activations/gemma-2-2b/fineweb/token_ids.npy`)
     since base + IT share the same tokenizer. The
     `build_act_cache_phase7` script has logic to copy from a sibling
     gemma-2-2b-it/ dir or vice-versa — adapt as needed.
   - Run `build_act_cache_phase7.py --layer L --subject_model google/gemma-2-2b-it`
     for L ∈ {11, 12, 13, 14, 15}. **One layer at a time** (single-layer
     hook + memmap is the MooseFS-safe path; multi-layer concurrency
     was found to cause silent corruption).
   - At the end, write/update `layer_specs.json` to record the IT layers.

2. **Build IT probe cache** (~30 min A40):
   - Run `build_probe_cache_phase7.py --include-crosstoken` with
     subject model overridden to `gemma-2-2b-it` and anchor=L13. The
     existing script hardcodes `SUBJECT_MODEL` from `_paths.py` —
     either override via env var or fork to a `_paths_it.py`. Recommend
     forking to avoid breaking BASE-side reproducibility.
   - Then `rebuild_probe_cache_s32.py` to produce S=32 left-aligned cache
     at `experiments/phase7_unification/results/probe_cache_S32_it/`
     (different dir from BASE).

3. **Train 8 A40_ok leaderboard archs × 2 seeds** (~10 hr A40 at b=4096):
   - tfa_big, tsae_paper_k20, tsae_paper_k500, txcdr_t5, txcdr_t16,
     phase5b_subseq_h8, txc_bare_antidead_t5, phase57_partB_h8_bare_multidistance_t8
   - Seeds 42 + 1 (drop seed=2 if budget tight; PAPER only needs 2-seed
     for σ).
   - **Strict apples-to-apples**: b=4096, max_steps=25000, plateau early
     stop. If a cell OOMs at b=4096, defer to H200 (do NOT lower batch).
     See `paper_archs.json::training_constants` for the locked values.
   - Push trained ckpts to `han1823123123/txcdr-base` (or maybe better:
     create `han1823123123/txcdr-it` for clean separation — Han's call).

4. **Probe on PAPER set** (~1 hr A40):
   - `run_probing_phase7.py --headline` filtered to PAPER tasks
     (the 16 listed in `task_sets.py::PAPER`). The probing script
     currently iterates ALL tasks in `probe_cache_S32/` — for headline
     you only need PAPER's 16 tasks. Either temporarily limit
     `probe_cache_S32_it/` to those 16 task dirs, or pass
     `--task_names` with the 16 explicit names.

5. **Append rows to `probing_results.jsonl`** with `subject_model:
   "google/gemma-2-2b-it"` and `anchor_layer: 13` baked into the meta
   so the leaderboard builder can filter by subject_model.

6. **Update `build_leaderboard_2seed.py`** to support per-subject-model
   leaderboards. Currently it doesn't filter by subject_model — add
   a `--subject base|it` flag and re-render two leaderboards.

7. **Re-render plots** for IT (suptitle "Gemma-2-2b-it L13, PAPER task
   set, multi-seed mean ± σ_seeds") and side-by-side base-vs-IT comparison.

**A40 feasibility check before starting**: not all 8 leaderboard archs
fit on A40 at b=4096 with the full PRELOAD_SEQS=24000 anchor cache (14
GB) plus model weights + Adam state. The MLC family (mlc, agentic_mlc_08)
needs the MLC tail cache (70 GB) which doesn't fit at all — these are
the H200_required cells. The other 6 archs (tfa_big, tsae_paper_k20/k500,
txcdr_t5/t16, phase5b_subseq_h8, txc_bare_antidead_t5,
phase57_partB_h8_bare_multidistance_t8) are A40_ok.

If MLC IT-side cells are H200_required: train the 6 A40_ok cells now,
mark mlc + agentic_mlc_08 (and their _sparse variants) as deferred to
H200, document the gap clearly in the IT-side leaderboard writeup.

### Mission #2 — `mlc_sparse` and `ag_mlc_08_sparse`

Goal: train + probe the two missing k_win=100 MLC cells under paper
constants (b=4096) on whichever pod can fit them.

**Why they're flagged H200_required**: MLC needs the multi-layer cache
(5 layers × 14.2 GB = 71 GB) preloaded on GPU OR streamed from disk.
On A40 (46 GB VRAM), 71 GB doesn't fit at all.

**Workaround that might work on A40**:
- **Reduce PRELOAD_SEQS from 24000 → 6000**: cuts cache from 71 GB →
  17.7 GB. Fits with model + Adam state in ~30 GB total. Trades
  preloaded coverage for fit. See `train_t20_s8.py` (which X used for
  the SubseqH8 cell) for the pattern: pass `n_seqs=6000` to
  `preload_multilayer()`. Check `_train_utils.py::preload_multilayer`
  for the API.
- This is a deviation from the b=4096 / PRELOAD=24000 paper-canonical
  constants. Document it explicitly. The MLC sparse cells with reduced
  preload may not be apples-to-apples with the dense MLC cells trained
  on H200 with PRELOAD=24000; flag this in the writeup.

If the workaround fails (still OOM), defer to H200 cleanly and document.

**Note on legacy IT-side k_win=100 ckpts**: `han1823123123/txcdr` has
old Phase 5 IT-side `mlc__k100` and similar at b=1024. They DO NOT
count per `paper_archs.json` ("If a cell can't fit on A40 at b=4096,
DO NOT downsize batch — DEFER to H200"). Don't confuse these with the
needed paper-canonical b=4096 ckpts.

### Common pitfalls / gotchas (X learned the hard way)

1. **OOM-kill on concurrent python procs**: never run two python
   processes that load gemma. The 46 GB cgroup RAM cap kills both
   silently. Always `pgrep -f` to ensure no other python is running
   before starting a GPU job.
2. **Stuck bash waiters with `pgrep -f` patterns matching their own
   command line**: if you write `until ! pgrep -f run_probing_phase7;
   do sleep N; done` as a wait loop, it never exits because the bash
   itself contains "run_probing_phase7" in its command line and is
   visible to its own pgrep. Use a PID-based check instead: capture
   the PID at launch, then `until ! kill -0 $PID 2>/dev/null; do
   sleep N; done`.
3. **`git stash push` while a process is appending to a tracked file**:
   destroys some appended rows on stash-pop because the rebase
   silently truncates the file mid-write. If you need to commit while
   probing is running, pause the probing first or commit the file
   *before* starting the run.
4. **Per-task script outputs go to stdout but bash heredoc-captured
   processes can lose output**: always `PYTHONUNBUFFERED=1` and direct
   to a log file via `> /workspace/temp_xc/logs/X.log 2>&1` rather than
   relying on `nohup`'s default behaviour.
5. **MLC tail data in probe_cache_S32 is only 11 GB**: when probing
   MLC archs at PAPER, the per-task `acts_mlc_tail.npz` is 11 GB on
   disk but only ~1.4 GB needs to be in CPU RAM at a time (one task
   at a time, then explicit `del` + `gc.collect()`). The
   `run_probing_phase7.py` already does this correctly.
6. **HF push may silently fail** when `~/.cache/huggingface` is mid-write.
   After every training, do `huggingface-cli upload --repo-type model
   han1823123123/txcdr-base ckpts/X.pt ckpts/X.pt` and verify the SHA
   matches before deleting local.

### Files to read before starting

In order of importance:

1. `agent_x_brief.md` — original X mission spec
2. `agent_x_paper/README.md` — current state of paper deliverables
3. `agent_x_paper/2026-04-29-paper-task-set.md` — why PAPER is the
   headline task set
4. `agent_x_paper/2026-04-29-leaderboard-multiseed.md` — current BASE
   leaderboard
5. `experiments/phase7_unification/paper_archs.json` — locked arch spec
6. `experiments/phase7_unification/_paths.py` — path config (fork or
   override for IT)
7. `experiments/phase7_unification/build_act_cache_phase7.py` — base
   activation cache build
8. `experiments/phase7_unification/build_probe_cache_phase7.py` — probe
   cache build
9. `experiments/phase5_downstream_utility/build_multilayer_cache.py` —
   IT-side activation cache build (Phase 5 reference)
10. `experiments/phase7_unification/run_probing_phase7.py` — probing driver

### Definition of done

For Mission #1 (IT-side):
- 6 IT A40_ok leaderboard archs trained at b=4096, ≥ 2 seeds each,
  appended to `training_index.jsonl`.
- Probed on PAPER's 16 tasks at S=32, k_feat ∈ {5, 20}, FLIP applied,
  rows appended to `probing_results.jsonl` with `subject_model:
  google/gemma-2-2b-it, anchor_layer: 13`.
- `build_leaderboard_2seed.py` extended to filter by subject_model;
  IT-side leaderboard rendered at `results/plots/phase7_leaderboard_it_multiseed.png`
  + thumbnail; copied to `agent_x_paper/plots/`.
- `agent_x_paper/2026-04-29-leaderboard-multiseed.md` updated with an
  IT-side section (or new sibling md `2026-04-29-leaderboard-it.md`)
  documenting the IT numbers and base-vs-IT comparison.
- README updated with IT-side coverage status.
- All committed + pushed.

For Mission #2 (MLC sparse):
- Either `mlc_sparse` and `ag_mlc_08_sparse` trained at b=4096 (with
  PRELOAD_SEQS reduced if necessary, documented), probed on PAPER, rows
  appended; OR a clean writeup explaining why neither A40 nor the
  reduced-preload workaround fits, deferring to H200.
- `paper_archs.json::leaderboard_archs` cells for `mlc_sparse` and
  `ag_mlc_08_sparse` updated to show training status.
- README "Gaps" section updated.

### What NOT to do

- Don't promote new arches to `paper_archs.json` without Han's go-ahead.
- Don't change the task selection (FULL or PAPER) without Han's
  go-ahead — these are locked.
- Don't lower batch_size below 4096 to fit on A40 unless explicitly
  documented as a deliberate deviation; defer to H200 instead.
- Don't run anything in parallel on the same GPU.
- Don't probe with old methodology (Phase 5/5B aggregation modes) —
  PAPER methodology is `phase7_S32_first_real_meanpool` with FLIP only.
- Don't replace the BASE-side rows in `probing_results.jsonl`; only
  append IT rows.
