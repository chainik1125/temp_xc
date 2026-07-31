## RunPod overnight run

### Dispatch

- Pod: `trajectory-bottleneck-c7-300k`
- Pod ID: `k804r98a0a6w6b`
- GPU: one secure-cloud NVIDIA H100 SXM 80 GB
- Price: `$2.99/hour`
- Created: `2026-07-31T05:51:17Z`
- Hard stop: `2026-07-31T13:51:17Z`
- Maximum session cost: `$23.92`
- Source branch: `codex/spectral-screen-overnight-20260729`
- Source commit at launch: `5b60c3eac`
- Latest source commit: `aefd15eb2`

### Validated launch state

- Both public artifacts were downloaded from the pinned
  `han1823123123/temp-bench-data` revision
  `6ef9b1debf863dedcef9555cad3a4903fb9e8c43`.
- Event-artifact SHA-256:
  `1656f6be2cd85fb85c8b246b9b27933f73ef40cfaac84078169dfd3bbbe27810`.
- Training-cache SHA-256:
  `dc34dfb117f77abddef4b4396d0d00afc707c39876d0ee36015de1e7b8406914`.
- The rank-0 and rank-256 model paths passed a CUDA forward/backward
  finite-gradient smoke test.
- The trainer uses the verified RAM-local copy
  `/dev/shm/trajectory_resid_post_L10.npy`; persistent checkpoints remain on
  `/workspace`.
- The supervisor has `PPID=1` and its own session, so it is independent of
  SSH. It invokes the RunPod stop API on completion, ordinary terminal failure,
  or the hard deadline.

### Exact active cell

The first stage is a shared per-token SAE with:

- `T=5`, seed `42`
- `d_in=4096`, `d_sae=32768`
- per-token TopK `k=20`
- batch `1024` windows, exposing the same five raw tokens as the TXC
- `300000` optimizer steps
- learning rate `3e-4`, warmup `1000`
- deterministic schedule seed `911200`
- checkpoint interval `5000`

After that finishes, the same supervisor precomputes sparse codes, trains the
rank-0 and rank-256 one-code trajectory bottlenecks to 300k steps, and runs the
question-grouped C7 probe.

### Persistent paths

- Root: `/workspace/trajectory_bottleneck_c7`
- Checkpoints: `/workspace/trajectory_bottleneck_c7/checkpoints`
- Results and heartbeat: `/workspace/trajectory_bottleneck_c7/results`
- Logs: `/workspace/trajectory_bottleneck_c7/logs`

The run is resumable from the last complete checkpoint if the eight-hour
session ends before the full pipeline.
