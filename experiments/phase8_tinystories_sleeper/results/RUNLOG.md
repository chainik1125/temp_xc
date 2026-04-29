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
