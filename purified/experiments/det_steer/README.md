# det_steer — detection + steering protocol validation

Cross-component infrastructure for measuring **detection** (sparse linear
probe PR-AUC) and steering **TXCs correctly** (position-aware hooks).
Lives in `experiments/det_steer/` rather than under any single `cN_*/`
folder because the same code runs across C5, C6, C7.

## Pieces

- `validate_protocols.py` — methodology validation on tiny synthetic TXCs.
  Verifies the four hook modes do what their docstrings claim, the
  shuffle ablation correctly distinguishes temporal-vs-window-density
  signal, and the position-variance / encoder-decoder-divergence
  diagnostics behave as documented. Runs in ~5 minutes on one H100.
- `run_c7_locked.py` — production run on the **locked C7 checkpoints**:
  pulls TXC-base + TXC-pro from `han1823123123/temp-bench-models`,
  pulls C7 sentence_acts from `han1823123123/temp-bench-data`, runs
  detection PR-AUC + shuffle gap, position-variance histograms,
  encoder-decoder divergence per-feature. Skips the steering A/B
  (it requires the R1-Distill-Llama generation pipeline + Sonnet
  judge — that loop lives in `experiments/c7_backtracking/run.py`
  and agent_back will adopt the new hooks there).
- `run_steering_ab.py` — the V0/V1/V2/V4 steering A/B. Designed to be
  invoked by agent_back from inside their existing C7 sweep:
  same cohort + judge, just substitutes `TXCSteeringHook` for the
  legacy single-vector `SteeringHook`. Sweeps the phase parameter for
  V1 and reports the best phase per arch.

## Code paths used

- `temp_bench.eval.detection.detect_case_study` — encode → max-pool →
  GroupKFold-by-qid sparse-probe PR-AUC + within-window shuffle gap.
- `temp_bench.eval.steering_hooks.TXCSteeringHook` — V0 / V1 / V2 / V4
  hook with √T energy correction.
- `temp_bench.eval.steering_hooks.position_variance` /
  `encoder_decoder_divergence` / `encoder_preimage` — per-feature
  diagnostics.
- `temp_bench.eval.steering_protocols.latent_space_steer` — V3
  diagnostic (latent-space `z' = z + α e_f` decode-and-overwrite).
- `temp_bench.utils.shuffles.shuffle_within_window` — paired ablation
  helper.

## Running

```bash
# 1. Methodology validation — synthetic data, no HF deps. Fast.
TQDM_DISABLE=1 .venv/bin/python -m experiments.det_steer.validate_protocols

# 2. Real data: run on the locked C7 checkpoints + sentence_acts
# (HF auth required — Han's private repos).
TQDM_DISABLE=1 .venv/bin/python -m experiments.det_steer.run_c7_locked \
    --archs txc_base,txc_pro,topk_sae,tsae_paper

# 3. Steering A/B (V0/V1/V2/V4 on one chosen feature). Agent_back
# integrates this inside their sweep so the cohort + judge stay shared.
TQDM_DISABLE=1 .venv/bin/python -m experiments.det_steer.run_steering_ab \
    --arch txc_pro --feature_id <selected> --modes v0,v1,v2,v4
```

All scripts respect `TQDM_DISABLE=1` (CLAUDE.md rule 1) and seed via
`temp_bench.utils.seed.set_seed`. Outputs land under
`experiments/det_steer/results/` (per the experiment-folder convention)
with paired `.png` / `.thumb.png` figures via `save_figure()`.

## Findings docs (companion writeups)

- `docs/cross_component/det_steer_detection.md` — protocol spec for C5,
  C6, C7 detection cells.
- `docs/cross_component/det_steer_steering.md` — TXC steering audit + V0
  vs V1/V2/V3/V4 trade-off.
- `docs/cross_component/det_steer_summary.md` — methodology-validation
  results, integration TODOs, what each case-study agent must adopt.

## Discipline notes

- This sub-tree is **additive only**. It does NOT touch
  `experiments/c5_steering/`, `experiments/c6_em/`,
  `experiments/c7_backtracking/`, or any agent's briefing / component
  doc. Agents adopt the new infra by importing from
  `temp_bench.eval.{detection,steering_hooks,steering_protocols}`
  inside their own runners.
- Per PROTOCOL.md § 7 (results live in state), aggregate numbers from
  these scripts are **not** hand-typed into `cN.md` AUTO-RESULTS
  blocks. The relevant component agent re-runs the protocol inside
  their sweep and renders into their own AUTO-RESULTS via
  `temp_bench.report.render(component="cN")`.
