# Phase 8 RUNLOG — autonomous

Append-only ticks. Each entry is a snapshot of state, decision, and next ETA.

## 2026-04-28 ~setup — track0.bootstrap
Worktree at `temp_xc-phase8` on branch `dmitry-phase8` (off `dmitry-rlhf`).
SSH alias resolved: user said "a40_tinystories_1", actual config is `a40_tiny_1`
(46 GB A40, fresh /root, 100 GB free). uv installed; `uv sync` complete on remote.
fra_proj scripts copied: harvest_activations, train_crosscoders, run_ablation_sweep,
plot_pareto, sae_models, sleeper_utils, README, SUMMARY, RESULTS.

Code changes (not yet pushed):
- `sae_models.py` +TemporalContrastiveSAE, +MultiDistanceTXC
- `sleeper_utils.py` +TSAE_LAYER_HOOKS, +H8_LAYER_HOOKS
- `train_crosscoders.py` +arch building + multi-shape gather + compute_loss dispatch
- `run_ablation_sweep.py` +class_name branches + per-token-vs-window helpers
- `plot_pareto.py` +T-SAE + H8 entries in colour / label maps
- `docs/han/research_logs/phase8_tinystories_sleeper/brief.md` written

Next: commit + push, pull on remote, smoke test (small d_sae, few steps).

## 2026-04-28 ~smoke.launch — track1.smoke
Pushed peft dep + smoke runner. Remote uv sync added peft==0.19.1.
Smoke pipeline running on a40_tiny_1, pid=2587, log at
`experiments/phase8_tinystories_sleeper/results/smoke.log`. Args:
n_train=512, n_val=64, n_test=64, d_sae=128, k_total=16, T=10, n_steps=200,
batch_size=512. Subset of archs: sae_layer0, tsae_layer0, h8_early,
txc_early, mlc — one of each architecture family.

Two earlier smoke iterations failed:
1. `peft` not installed — fixed by adding to pyproject.toml.
2. (also caught) Earlier launch piped each python invocation through
   `tail -40`, masking failures from set -e. Replaced with run_smoke.sh
   that exits on first failing stage.

Next: wake at +4.5m to read smoke.log; if SMOKE DONE → kick off full
pipeline.

## 2026-04-28 ~smoke.fail-1 — track1.smoke
First smoke iteration crashed in train at H8 multi-distance gather:
`pos_idx + T + max_shift - 1` walked off the right edge of `acts_padded`.
Root cause: original padding reserved `right = T-1-left` slots, but H8's
shift schedule needs an additional `max_shift = T // 2` of right-pad.

Fix (commit c475b9d): pad by `right + T//2` whenever H8 is in scope.
Verified all four arch families instantiate correctly before crash:
  mlc 987k params, txc_early 1.97M, sae_layer0 198k, tsae_layer0 198k,
  h8_early 1.97M.

Re-launched smoke; waking at +4.5m.


