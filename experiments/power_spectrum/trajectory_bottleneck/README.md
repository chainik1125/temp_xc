## Shared-SAE trajectory bottleneck

This is the deliberately missing two-stage baseline for the C7 backtracking
result:

- Train one shared, per-token TopK SAE for 300k optimizer steps on exactly the
  same `B x T` exposures as the published `T=5`, seed-42 TXC.
- Freeze its dictionary and precompute sparse token codes.
- Learn a one-code bottleneck over each five-token code trajectory. The code
  has the same nominal `L0 <= 100` budget as the TXC.
- Evaluate with the frozen question-grouped C7 sparse-probe protocol, including
  shuffle, reverse, and circular order controls.

Two second stages are run:

- `rank0`: learned per-feature temporal pooling and temporal decoding, without
  cross-feature mixing.
- `rank256`: the same model plus a learned rank-256 cross-feature temporal
  residual.

The experiment is self-contained in this folder, while intentionally importing
the frozen C7 schedule and SAE trainer from Aniket's `origin/neurips-aniket`
protocol at runtime.

### Durable RunPod execution

`supervisor.sh` is launched under `setsid` and `nohup` on a persistent RunPod
volume. Training checkpoints are atomic and resumable. The supervisor restarts
ordinary process failures, applies a hard wall-clock deadline, and invokes a
pod-stop helper after flushing state. An SSH or client connection drop does not
affect the worker.
