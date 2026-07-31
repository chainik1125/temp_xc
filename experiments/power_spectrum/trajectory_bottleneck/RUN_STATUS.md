## Cancelled RunPod dispatch

**Status:** stopped before the first checkpoint after the user flagged likely
duplication.

### Dispatch

- Pod: `trajectory-bottleneck-c7-300k`
- Pod ID: `k804r98a0a6w6b`
- GPU: one secure-cloud NVIDIA H100 SXM 80 GB
- Price: `$2.99/hour`
- Created: `2026-07-31T05:51:17Z`
- Stopped: `2026-07-31T06:34:03Z`
- Hard stop: `2026-07-31T13:51:17Z`
- Maximum session cost: `$23.92`
- Actual runtime: approximately 43 minutes
- Estimated actual cost: approximately `$2.13`
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

### State at cancellation

The process was still training the base SAE. The persistent run directory
contained only `checkpoints/base_sae/config.json`; it had not produced a model
checkpoint. No trained state was discarded.

### Cell that had been dispatched

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

### Duplication audit

Several close controls have already been run:

- Aniket's 20k backtracking reviewer sweep compares TXC against positional,
  invariant, and last-token SAE trajectories across
  `T in {1,2,4,6,10}`. At `T=6`, increasing positional-SAE probe support to
  192--256 features matches and then beats TXC.
- A complete 20k C7 seven-architecture panel already includes a trained
  Stacked SAE (`0.328` peak delta-gc versus `0.426` for TXC-base).
- The 300k C7 follow-up includes a Stacked SAE (`0.246` peak delta-gc versus
  `0.541` for TXC-base).
- The sycgen reviewer addendum evaluates a frozen per-token SAE with fixed
  mean pooling and a trained parameter-matched Stacked SAE; under the matched
  ridge readout both controls match or exceed TXC at every tested window.

None of these is the exact proposed *learned* shared-SAE
trajectory-to-one-code adapter (`rank0`/`rank256`). The distinction is real,
but the broader scientific question—whether SAE trajectories already contain
the temporal signal and whether generic aggregation can recover it—has
substantial existing coverage. Do not relaunch this pipeline without first
justifying the learned adapter as a narrower, preregistered residual question
and reusing an existing base SAE checkpoint.
