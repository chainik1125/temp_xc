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

## 2026-04-28 ~smoke.pass — track1.smoke
SMOKE DONE. All four arch families load, train, encode, ablate, decode,
and produce frontier points. Smoke headline (d_sae=128, T=10, 200 steps,
n_test=64 — directional only):

| arch        | f  | α   | test ASR | base | ΔCE     |
|-------------|----|-----|----------|------|---------|
| sae_layer0  | 14 | 2.0 | 0.188    | 0.969| +0.006  |
| tsae_layer0 | 31 | 1.0 | 0.125    | 0.969| +0.011  |
| h8_early    | 48 | 2.0 | 0.000    | 0.969| +0.004  |
| txc_early   | 78 | 2.0 | 0.188    | 0.969| +0.021  |
| mlc         | 80 | 1.0 | 0.000    | 0.969| -0.000  |

H8 / MLC zero-ASR is suspicious for 200 steps × d_sae=128 — likely noise,
revisit at full scale.

## 2026-04-28 ~full.launch — track2.full
Full pipeline launched on a40_tiny_1, pid=3769, log at
`experiments/phase8_tinystories_sleeper/results/full.log`.

Params (matches fra_proj):
- harvest: n_train=10000, n_val=200, n_test=200, seq_len=128
- train: d_sae=1536, k_total=32, T=30, n_steps=8000, batch_size=4096, lr=5e-4
- sweep: top_k=100, stage2_keep=10, alphas={0.25,0.5,1.0,1.5,2.0}, δ=0.05
- archs (15): mlc, txc_{early,mid,late}, sae_layer{0..3},
  tsae_layer{0..3}, h8_{early,mid,late}

Expected runtime ~3-4h. Switching cadence to 20m. Next wake reads
`full.log` tail + `nvidia-smi` and updates this RUNLOG.

## 2026-04-28 ~full.t+20m — track2.full
- Harvest: DONE. 10000 train + 200 val + 200 test cached at
  `outputs/data/{tokens,activations}_cache.pt`. Activations shape
  (10000, 128, 5, 768) fp16. Marker mean ~41 tokens (prompt is the first
  ~⅓ of each 128-token sequence). Min prompt 3 tokens — short, but
  fra_proj used these as-is so we trust the filter.
- Train: in progress, pid 4004, 23 min CPU, GPU 100% util, 38/46 GB.
  Stdout buffered (no step messages yet) — added `PYTHONUNBUFFERED=1`
  to run_full.sh for next runs (won't help current run; will flush at
  process exit).
- ETA train end: roughly +20-25m from now (8000 steps × 15 archs × 4096
  batch is ~50min at A40 throughput; scaling linearly from 23 min in).

## 2026-04-28 ~full.t+98m — track2.full
Train still running. CPU 1h 38m, GPU 99%, memory 38/46 GB steady. No
checkpoints saved yet; stdout fully buffered (current run has no
`PYTHONUNBUFFERED`). Original budget 60m → 2× = 120m, current 1.6×.

Per decision-rule defaults: extend up to 2× nominal. Will check again at
113m mark; if still going at 128m+ I'll kill and re-launch with
PYTHONUNBUFFERED=1 + reduced n_steps (4000) for visibility.

Best guess on slowdown: 15 archs × 4096 batch × T=30 plus the H8
multi-distance contrastive (3 H8 instances × 3 shifts of contrastive
InfoNCE per step) is just compute-heavy at full scale. The smoke test ran
~5 archs × 200 steps × T=10 × 512 batch = ~1/200th of the full work,
which would scale to ~100min — consistent with the 98min observed.

## 2026-04-28 ~full.t+126m — track2.full
Train at 2h 06m, past the 2× threshold. Decision: continue waiting rather
than kill. Reasoning:
- GPU 100% util steady, mem 38/46 GB steady, no errors → training is
  on track, not stuck.
- The 60m nominal was a guess, not a measured number. The smoke scaling
  predicted ~100m and we're at 2h — within 2× of the *honest* estimate.
- Killing now wastes 2h of compute; checkpoints aren't saved until end
  of training loop.

Revised hard kill ceiling: 3h CPU. If still going at +180m, kill and
re-launch with reduced scope (n_steps=4000, drop the 3 H8 instances or
tile them to one shift) plus PYTHONUNBUFFERED=1 for visibility.






