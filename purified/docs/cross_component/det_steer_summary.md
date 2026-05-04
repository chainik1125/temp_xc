---
author: aniket
date: 2026-05-04
tags:
  - results
  - in-progress
---

## det_steer summary — what landed, what each agent must do

This document is the **handoff** for Han to route to agents. It
captures (i) what shared infrastructure landed on `det-steer`,
(ii) what the methodology validation showed, (iii) the per-component
integration TODOs, and (iv) what is explicitly out-of-scope for me
(only Han or the named agents may land it).

## Branch + filesystem

- Branch: **`det-steer`**, branched off `origin/final-aniket`. All
  commits are **purely additive new files** — no edits to anything
  existing on `final-aniket` or `final`. Han pulls into
  `final-aniket` (or directly into `final`) as he sees fit.
- Worktree layout while building this: `temp_xc-detsteer/` (det-steer)
  and `temp_xc-final/` (read-only `final` for testing). Tests run via
  `PYTHONPATH=...detsteer/.../src final/.../venv/bin/python -m pytest`.
- I (the agent doing det_steer) am NOT a named agent in PROTOCOL.md.
  All my output is shared-infra files + cross-component docs; I do
  not touch `agents/<name>/`, `docs/components/cN.md` AUTO-RESULTS,
  `docs/paper/`, `decisions.md`, `agents/README.md`, or
  `configs/locked_archs.yaml` (per CLAUDE.md Hard Rule #7).

## What landed on `det-steer`

### Shared infrastructure

- `src/temp_bench/utils/shuffles.py` — `shuffle_within_window` (torch
  + numpy variants), the paired ablation primitive that determines
  whether a TXC's "temporal" detection is genuinely temporal or just
  window-density.
- `src/temp_bench/eval/detection.py` — `detect_case_study(arch, X, y,
  group_ids, ...)`, the cross-case-study probe + shuffle ablation.
  Internals match `case_studies.backtracking.compute_pr_auc_at_S`
  intentionally; case studies import this and stop re-implementing.
  Also `encode_and_pool` (the TXC-aware encode → max-pool primitive)
  and `detection_table` (markdown rendering).
- `src/temp_bench/eval/steering_hooks.py` — `TXCSteeringHook` with
  `mode ∈ {v0, v1, v2, v4}`, √T energy correction, per-row
  `magnitudes`. Plus diagnostics: `position_variance(W_dec)`,
  `encoder_preimage(arch, fid)`,
  `encoder_decoder_divergence(arch, fid)`, and the
  `build_hook(arch, feature_id, mode, ref_norm, ...)` convenience.
- `src/temp_bench/eval/steering_protocols.py` —
  `latent_space_steer(arch, residual_window, feature_id, magnitude,
  ref_norm)` for V3 (the "encode → perturb in z → decode → overwrite"
  primitive, separate from forward hooks).

### Tests

- `tests/test_shuffles.py` — 9 tests.
- `tests/test_detection.py` — 13 tests including a
  temporal-vs-density cohort assertion that the shuffle gap is
  positive-only on truly-temporal signal.
- `tests/test_steering_hooks.py` — 25 tests including V0 ≡ V4 at tied
  init, V1 cycling, V2 trailing fill, √T energy match, per-row
  magnitudes.
- `tests/test_steering_protocols.py` — 4 tests verifying
  `latent_space_steer` returns `α · W_dec[f]` on a linear TXC.

**51 tests pass** under
`PYTHONPATH=det-steer/src final-venv -m pytest`.

### Experiment scripts

- `experiments/det_steer/validate_protocols.py` — methodology
  validation on synthetic data + a tiny TXC trained ad-hoc. Runs in
  ~2 min on one H100. Already executed; outputs in
  `experiments/det_steer/results/validate/`.
- `experiments/det_steer/run_c7_locked.py` — production detection +
  shuffle gap + position-variance histogram on the locked C7
  checkpoints + sentence_acts (`sentence_acts_L10.npz`). Requires HF
  auth to pull from `han1823123123/temp-bench-{models,data}`.
  agent_back runs this inside their existing C7 workspace.
- `experiments/det_steer/run_steering_ab.py` — V0/V1/V2/V4 steering
  A/B on one chosen feature, full 25-magnitude grid. Re-uses
  `case_studies.backtracking`'s cohort, `phase1_unsteered` cache,
  generation panels, Sonnet judge, and Δgc compute. Designed to be
  invoked by agent_back inside the existing C7 sweep workspace
  (judge_outputs.jsonl is shared).

### Cross-component docs (this dir)

- [[det_steer_detection]] — protocol spec for C5 / C6 / C7 detection.
- [[det_steer_steering]] — TXC steering audit + V0 / V1 / V2 / V3 / V4
  trade-off + ablation plan.
- [[det_steer_summary]] — this doc.
- `README.md` — directory overview + index.

## Methodology validation results

Run command:
```bash
TQDM_DISABLE=1 PYTHONPATH=det-steer/src CUDA_VISIBLE_DEVICES=0 \
  final-venv/python experiments/det_steer/validate_protocols.py
```

### 1. Encoder–decoder divergence is 0 at tied init, drifts after train

| state | rel_residual mean | cos_sim mean |
|---|---:|---:|
| tied init (per `txc_base.py` init order) | 1.9e-8 | 1.0000 |
| after 600 SGD steps on random Gaussian inputs | 0.064 | 0.998 |

→ V4 vs V0 is meaningfully different post-training. Quantifies the
gap V4 captures over V0.

See `results/validate/position_variance_and_divergence.png`.

### 2. Detection PR-AUC + shuffle ablation distinguishes temporal vs density

Same drifted TXC, two synthetic cohorts:

**Temporal cohort** (positives carry dim-0 spike at t=0,
anti-spike at t=T-1):

| S | PR-AUC unshuffled | PR-AUC shuffled | gap |
|---:|---:|---:|---:|
| 1 | 0.562 | 0.502 | +0.060 |
| 2 | 0.608 | 0.531 | +0.077 |
| 4 | 0.677 | 0.541 | **+0.136** |
| 8 | 0.779 | 0.540 | **+0.239** |
| 16 | 0.786 | 0.527 | **+0.259** |

→ Positive shuffle gap, growing with S. Unshuffled PR-AUC is well
above shuffled (which is at chance). Protocol correctly identifies
that the signal is temporal.

**Density cohort** (positives carry dim-0 spike at every t):

| S | PR-AUC unshuffled | PR-AUC shuffled | gap |
|---:|---:|---:|---:|
| 1 | 0.695 | 0.761 | -0.066 |
| 2 | 0.850 | 0.857 | -0.007 |
| 4 | 0.922 | 0.943 | -0.021 |
| 8 | 0.960 | 0.969 | -0.008 |
| 16 | 0.969 | 0.965 | +0.004 |

→ Shuffle gap ≈ 0 across S. Unshuffled and shuffled PR-AUC overlap.
Protocol correctly identifies that the signal is window-density, not
temporal.

See `results/validate/pr_auc_shuffle_gap.png`.

### 3. Hook math: V0 / V1 / V2 / V4 deltas behave as documented

For the picked feature (highest rel_residual after train, fid=18 on
this run), with `ref_norm=1.0`, `magnitudes=1.0`, batch length `2T=10`:

| mode | per-position L2 norm | total energy |
|---|---|---:|
| V0 | constant ≈ 1.0 across all 10 positions | 10.0 |
| V1 | cycles W[t mod 5], √T-corrected (per-pos ≈ 1/√5) | 2.0 |
| V2 | zero for first 5, then W[T-1-j] for trailing 5 | 1.0 |
| V4 | constant ≈ 1.0 across all 10 positions | 10.0 |

→ V0 and V4 inject same energy (constant vector); V1 and V2 inject
1/T as much under √T correction (so per-step magnitude is comparable
to V0). Each mode's per-position structure matches the spec in
[[det_steer_steering]].

See `results/validate/hook_modes_delta.png`.

## Per-component integration TODO

The shared infra lands on `det-steer`. Each named agent integrates by
importing the new modules in their own runner / analysis files. None
of these edits are mine to land (PROTOCOL.md § 3 — files under
`agents/<name>/` and `docs/components/cN.md` AUTO-RESULTS are owned
by the named agent). They are written here for Han to route.

### agent_back (C7 — DETECTION)

- [ ] Confirm `extract_labeled_sentence_acts` returns the expected
      `(n_sent, T=6, d_in)` (verified on `sentence_acts_L10.npz`).
      The encode-side `amax(dim=1)` in `run_arch_evaluation` is
      already correct.
- [ ] Run `experiments/det_steer/run_c7_locked.py` from C7 workspace
      with the locked TXC-base / TXC-pro / TopK-SAE / T-SAE
      checkpoints. Output drops into
      `experiments/det_steer/results/c7_locked/`. Write the resulting
      PR-AUC + shuffle gap table into `c7.md`'s AUTO-RESULTS via
      `experiments/c7_backtracking/analysis.py` +
      `temp_bench.report.render(component="c7")`.
- [ ] Add a `position_variance_<arch>.png` figure to the c7
      writeup for each TXC arch (already produced by
      `run_c7_locked.py`).

### agent_back (C7 — STEERING)

- [ ] Run the V0 reproducibility check on locked TXC-base / TXC-pro:
      confirm peak Δgc reproduces the wasteland's +1.574 within seed
      σ. If not, debug V0 first; the A/B is moot otherwise.
- [ ] Run `experiments/det_steer/run_steering_ab.py` for one chosen
      feature on each TXC arch, sweeping
      `--modes v0,v1,v2,v4 --cycle_phases 0,1,2,3,4`. Pick the best
      V1 phase; report all four protocols' peak Δgc + peak mag +
      stability + Δ vs V0. Land in `c7.md` AUTO-RESULTS as a new
      "Steering protocol A/B" subsection.
- [ ] (If V1/V2/V4 don't beat V0): run the V3 spike — one feature × 5
      magnitudes × 20 cohort qids via the latent_space_steer driver.
      Reports the upper-bound on TXC's causal capacity; if it
      substantially beats V0, the trajectory protocols are an
      under-approximation.

### agent_steer (C5 — DETECTION)

- [ ] Build the encode-and-pool feature matrix per
      (arch, prompt × magnitude × seed) continuation. Cohort element
      = one continuation; label = success at coh ≥ 2.0; group =
      prompt_id.
- [ ] Call `temp_bench.eval.detection.detect_case_study(arch,
      sentence_acts, labels, prompt_ids, S_grid=(1,2,4,8,16,32),
      shuffle_seed=42)` per arch. Add the resulting PR-AUC + shuffle
      gap table to `c5.md`'s AUTO-RESULTS via
      `experiments/c5_steering/analysis.py`.

### agent_steer (C5 — STEERING)

- [ ] Coordinate with agent_back: if the C7 A/B picks V1 or V2,
      pre-test on TXC-pro for C5 before locking. TXC-pro's subseq +
      multi-distance-contrastive may have a steeper position-variance
      profile and benefit from V2 more than TXC-base does.
- [ ] If adopted: replace the V7 hook in
      `case_studies.steering` with `TXCSteeringHook(mode="v?",
      sqrt_t_correction=True)` for TXC archs. Per-token archs
      (TopK-SAE, T-SAE) keep the existing V7 hook (V0-equivalent
      and correct for their math).

### agent_em (C6 — DETECTION)

- [ ] Build encode-and-pool feature matrix per (arch, prompt × rollout
      × seed) rollout. Cohort element = one rollout; label =
      `align ≤ 50`; group = `prompt_id`. Hookpoint = L24 resid_post
      (matches existing C6 datasource).
- [ ] Call `detect_case_study(...)` per arch. Add table + shuffle
      gap to `c6.md` AUTO-RESULTS via
      `experiments/c6_em/analysis.py`. Compare against the existing
      bundle-null / gap-close numbers — a TXC that ties / beats
      SAE-arditi on detection but loses on steering would strengthen
      the C6 dissociation framing.

### agent_paper (cross-territory)

- [ ] Decide whether `temp_bench.eval.detection`,
      `temp_bench.eval.steering_hooks`, and
      `temp_bench.eval.steering_protocols` should be exported from
      `temp_bench.eval.__init__` (they're not, currently — the eval
      package on `final-aniket` only exports `CaseStudy`). Suggested:
      yes, alongside the existing `qualitative, probing, steering,
      synthetic` re-exports already present on `final`.
- [ ] Decide whether to surface `position_variance` /
      `encoder_decoder_divergence` as part of the headline
      "architecture analysis" section of `docs/paper/architecture.md`
      (one sentence + one figure per TXC arch — quantifies the V0
      audit's premise on the locked checkpoints).

## Out-of-scope for me

These are explicitly outside my territory and require Han or the
named agent to land:

- Editing `docs/components/cN.md` AUTO-RESULTS (PROTOCOL.md § 7
  Hard Rule #10 — analysis.py owns those).
- Editing `agents/agent_*/briefing.md` (PROTOCOL.md § 3 +
  Hard Rule #7).
- Editing `decisions.md`, `agents/README.md`,
  `configs/locked_archs.yaml`, `pyproject.toml`, `uv.lock`
  (CLAUDE.md Hard Rule #7).
- Pushing to `final` (Han's branch — I work on `det-steer`,
  PROTOCOL.md § 1).
- Running anything that needs Han's private HF repos
  (`han1823123123/temp-bench-{models,data}`) — my HF token doesn't
  have access. The `run_c7_locked.py` / `run_steering_ab.py` scripts
  are written and tested against the case_studies API, but final
  execution is gated on the agent that owns those checkpoints.

## Reproduction

```bash
# 1. Branches:
#    final          — Han's authoritative paper code (read-only for me)
#    final-aniket   — Han's staging branch where my work merges
#    det-steer      — branched off final-aniket; pure additions
git worktree add -b det-steer ../temp_xc-detsteer origin/final-aniket
git worktree add ../temp_xc-final origin/final

# 2. Dev (PYTHONPATH override; final has the venv):
cd /workspace/aniket/temp_xc-final/purified
TQDM_DISABLE=1 \
  PYTHONPATH=/workspace/aniket/temp_xc-detsteer/purified/src \
  .venv/bin/python -m pytest \
    /workspace/aniket/temp_xc-detsteer/purified/tests/test_shuffles.py \
    /workspace/aniket/temp_xc-detsteer/purified/tests/test_detection.py \
    /workspace/aniket/temp_xc-detsteer/purified/tests/test_steering_hooks.py \
    /workspace/aniket/temp_xc-detsteer/purified/tests/test_steering_protocols.py \
    -q
# expect: 51 passed

# 3. Methodology validation (any GPU):
TQDM_DISABLE=1 \
  PYTHONPATH=/workspace/aniket/temp_xc-detsteer/purified/src \
  CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python \
    /workspace/aniket/temp_xc-detsteer/purified/experiments/det_steer/validate_protocols.py
# expect: ~2 min, plots + summary.json under
#   experiments/det_steer/results/validate/

# 4. Locked-checkpoint runs (require HF auth to Han's private repos —
#    intended to be run by agent_back inside their existing C7 workspace):
TQDM_DISABLE=1 .venv/bin/python -m experiments.det_steer.run_c7_locked \
    --archs txc_base,txc_pro,topk_sae,tsae_paper
TQDM_DISABLE=1 .venv/bin/python -m experiments.det_steer.run_steering_ab \
    --arch txc_pro --modes v0,v1,v2,v4 \
    --workspace results/runs/<eval_key>
```

## Files added on `det-steer`

```
purified/src/temp_bench/utils/shuffles.py
purified/src/temp_bench/eval/detection.py
purified/src/temp_bench/eval/steering_hooks.py
purified/src/temp_bench/eval/steering_protocols.py
purified/tests/test_shuffles.py
purified/tests/test_detection.py
purified/tests/test_steering_hooks.py
purified/tests/test_steering_protocols.py
purified/experiments/det_steer/__init__.py
purified/experiments/det_steer/README.md
purified/experiments/det_steer/validate_protocols.py
purified/experiments/det_steer/run_c7_locked.py
purified/experiments/det_steer/run_steering_ab.py
purified/experiments/det_steer/results/validate/{summary.json,*.png,*.thumb.png}
purified/docs/cross_component/README.md
purified/docs/cross_component/det_steer_detection.md
purified/docs/cross_component/det_steer_steering.md
purified/docs/cross_component/det_steer_summary.md
```

## References

- [[det_steer_detection]]
- [[det_steer_steering]]
- `experiments/det_steer/README.md`
- PROTOCOL.md § 1 (branch model), § 3 (filesystem ownership), § 7
  (component writeup template — AUTO-RESULTS), § 11 (framework
  discipline).
