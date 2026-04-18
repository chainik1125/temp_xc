---
author: Aniket Deshpande
date: 2026-04-18
tags:
  - reference
  - bench-harness
---

## Bug-ledger defense check

Walks every entry in `eval_infra_lessons.md` (19 bugs, B1-B19) and
states how the new benchmarking harness (`src/bench/run_eval.py` +
extensions to existing modules) prevents or catches each one.

Categories:

- **R** — captured as an in-harness assertion in `src/bench/regressions.py`,
  fails fast at `run_eval` startup.
- **F** — fix already landed as a code change in earlier commits (the
  patches from the overnight SAEBench run); tracked by a regression check.
- **P** — process / plumbing issue; can't reasonably be tested inside
  Python. Documented in `eval_infra_lessons.md` § "Codebase TODOs"
  and covered by launch checklist.
- **V** — VRAM-profile mode assertion; gated on a `--profile-vram` flag
  because it requires GPU and isn't free.

### B1. MLC mapped to `crosscoder` via isinstance chain — **R + F**

Fixed: `_arch_registry_key` in `src/bench/sweep.py` now checks
`LayerCrosscoderSpec` before `CrosscoderSpec` (commit `a7b5c6f`).

Regression: `tests/regressions/test_registry_keys.py` asserts every
ArchSpec in `REGISTRY` maps back to its canonical name. Runs in
`run_eval` startup.

### B2. `evaluate_model` didn't handle `data_format="multi_layer"` — **R + F**

Fixed: `src/bench/eval.py:119` has the `multi_layer` branch + new
`_eval_multi_layer` helper (commit `56d1043`).

Regression: unit test dispatches a dummy MLC spec through
`evaluate_model` and asserts no `Unknown data_format` is raised.

### B3. `run_mlc_probing` return dict missing `elapsed_sec` — **R + F**

Fixed: `src/bench/saebench/mlc_probing.py:313` now includes
`elapsed_sec` in the return dict (commit `5c0d4f5`).

Regression: assertion in `run_probing` wrapper that MLC's return
dict has all expected keys (`run_id`, `n_records_written`,
`elapsed_sec`, optionally `skipped`).

### B4. Protocol B `tempxc_k_at` scaled window_k quadratically — **R + F**

Fixed: `tempxc_k_at` now returns `max(1, tempxc_k_base // t)` for
Protocol B (commit `07d18df`).

Regression: unit test prints effective window_k = protocol_k * T for
every (arch, protocol, T) in the sweep grid and asserts they fall in
`[1, D_SAE]` range. B's window_k must be constant across T.

### B5. `_load_arch_and_model` ran with `k=0` placeholder — **R + F**

Fixed: `_load_arch_and_model` now resolves k via
`protocol_k(arch, protocol, t)` inside the loader, not as a caller arg
(commit `1edcb06`).

Regression: after `load_state_dict`, assert `model.k > 0`. Additional
"live encoder" check: pass a random batch of activations through
`encode()` and assert output has non-zero features (this catches the
class of bug where k is loaded correctly but activations end up zero
for some other reason).

### B6. `eval_hidden` was 3D for MLC — **R + F**

Fixed: `build_multi_layer_activations_pipeline` keeps eval_hidden 4D
`(n_eval, seq_len, n_layers, d_model)` (commit `56d1043`).

Regression: after `build_data_pipeline`, assert
`eval_hidden.dim() == expected_rank(spec.data_format)`. Mapping:
`flat` → 2D, `seq` → 3D, `window` → 3D, `multi_layer` → 4D.

### B7. `_want_temporal_metrics` triggered on MLC — **R + F**

Fixed: `_want_temporal_metrics` returns False for
`data_format="multi_layer"` (commit `56d1043`).

Regression: unit test with MLC spec + 4D dummy tensor — assert return
is False.

### B8. TempXC encode OOM at SAEBench's B=125-230 — **F + V**

Fixed: chunked over B in `_encode_tempxc` with target peak ~1.5 GB
(commit `fae061a`).

V: optional VRAM profile mode that logs peak allocation per chunk
and fails if it exceeds a configurable threshold. Not default because
it requires GPU. Invoked by `run_eval --profile-vram`.

### B9. Full-window aggregation tensor size — **F + V**

Fixed: `cfg.sae_batch_size` lowered to 16 (commit `c91ed48`).

V: same profile mode as B8. Additionally, a pre-launch calculation
asserts `expected_peak_bytes(sae_batch_size, L, T, d_sae) < gpu_free`
when GPU is available.

### B10. Preflight 500-step SAE ckpt shared path with real run — **F + R**

Fixed: preflight self-cleans its ckpt + sweep dir at the end
(commit `276b0c3`).

Regression: after preflight completes, `run_eval` asserts the
preflight-written checkpoint path is absent before starting the real
sweep. Catches preflight-cleanup regressions.

### B11. `force_rerun=False` + placeholder `k=0` = stale cache — **F + R**

Fixed: `run_probing`'s default is now `force_rerun=True`
(commit `4b6e3ff`).

Regression: assert `inspect.signature(run_probing).parameters['force_rerun'].default is True`.

### B12. Branch checkout deleted working-tree files — **P**

Process issue. Can't regression-test inside Python. Captured in
`eval_infra_lessons.md` § L4 ("use git for pod→laptop data transfer")
and in the launch checklist ("before any `git checkout` on the pod,
`git push` first").

### B13. `sae_bench`'s train_probe_on_activations multi-class API — **R + F**

Fixed: our MLC path now inlines sklearn LogisticRegression directly
(commit `edb8289`).

Regression: ensure no code path calls SAEBench's
`probe_training.train_probe_on_activations` from within `mlc_probing`.
Import-graph check.

### B14. `SparseProbingEvalConfig` doesn't carry `prepend_bos` — **R + F**

Fixed: hardcoded in `mlc_probing` (commit `1805a82`).

Regression: assert `mlc_probing` doesn't reference
`cfg.prepend_bos` or `cfg.context_length`. grep-based check in
`src/bench/regressions.py`.

### B15. O(N_classes²) encoding in mlc_probing — **R + F**

Fixed: encode each unique text once, per-class loop binarizes
(commit `7735968`).

Regression: smoke test with 3 synthetic classes × 5 examples each —
assert total text-encode operations = 15, not 45 (N_classes × N_total).
Uses a call-counter wrapper.

### B16. `cfg.sae_batch_size` default 125 too large — **F + R**

Fixed: set `cfg.sae_batch_size = 16` in `probing_runner` (commit `c91ed48`).

Regression: assert `cfg.sae_batch_size <= 16` after config
construction in `run_probing`.

### B17. RunPod SSH gateway requires `-tt` — **P**

Process / shell script issue. All pod-side scripts use `ssh -tt`.
Documented in `eval_infra_lessons.md` § L4 and B17.

### B18. `.bashrc` polluted non-interactive SSH stdout — **P**

Process issue. TODO in `eval_infra_lessons.md`: patch
`scripts/runpod_activate.sh` to be silent for non-interactive shells.
Once that lands, rsync transfers work too.

### B19. GitHub PAT: classic vs fine-grained — **P**

Process issue. Documented in `eval_infra_lessons.md` § L4 with the
recommendation: "always start with a classic PAT with `repo` scope".

---

### Item 8 (new): Persistence-layer sanity check — **R**

After each probing cell completes, `run_probing` re-computes aggregate
accuracies from the persisted per-example prediction JSONL and asserts
they match the probing loop's reported aggregates to machine precision.
Catches silent persistence-layer drift — the class of bug where
predictions look plausible but have been misaligned with the labels
(e.g. due to dataset-ordering assumptions breaking).

Implemented as `_sanity_check_persistence()` in the probing runner;
fails the run with a clear error message if mismatched.

---

### Summary

| Bug range | # | Coverage |
|-----------|---|----------|
| B1–B7 (dispatch/config/data-shape) | 7 | All R+F (assertion + fix landed) |
| B8–B9 (VRAM) | 2 | F (fix landed) + V (optional profile mode) |
| B10–B11 (cache/state) | 2 | R+F |
| B12 (git process) | 1 | P (documented) |
| B13–B16 (3rd-party/algo/config) | 4 | R+F |
| B17–B19 (infra plumbing) | 3 | P (documented) |
| Item 8 (persistence drift) | 1 | R (new) |
| **Total** | **20** | **14 R, 16 F, 3 V, 6 P** (overlapping categories) |

14 bugs become impossible-or-loudly-caught at `run_eval` startup.
The 6 process bugs can't be regression-tested inside Python; they're
the launch-checklist's job. That checklist lives in `eval_infra_lessons.md`.
