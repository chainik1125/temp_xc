---
author: aniket
date: 2026-05-04
tags:
  - design
  - in-progress
---

## Extended-training handoff for 4× H100 RunPod (Han's 300K-step ask)

This document is a complete handoff for a fresh Claude Code session
spinning up on a 4× H100 RunPod pod to run Han's "save-checkpoint-every-100-iters,
4 TXCs × {bs=256,1024} × 300K steps" matrix on the C7 backtracking case study,
plus continue the extended undertrain-check pass already in flight on a
single-H100 pod.

## TL;DR for the next CC instance

- We're in **final mode**, NeurIPS abstract submitted, full deadline ≤70 hr from
  2026-05-04 ~14:00 UTC. C7 = Ward Stage B backtracking case study is the chosen
  vehicle for the undertraining check (see Slack thread further down).
- A **20K + 100K chain** is in flight on a different pod (single H100, originating
  pod). Do **not** duplicate that work on the new 4× H100. The 4× H100 pod's job is
  Han's matrix: `{txc_base, txc_pro, txc_base@T=20, txc_pro@T=20}` × `{bs=256, bs=1024}`
  × 300K steps × snapshot every 100 iters, on C7 backtracking.
- Plumbing in `experiments/c7_backtracking/run.py` already accepts `--n-steps`. The
  remaining wiring (`--batch-size`, `--snapshot-every`, T=20 yaml entries) is
  **not yet landed** — it's described below as the first task for the new agent.
- Branch policy: **never push to `final` directly**. Land changes on `final-aniket`
  or a fresh branch off `final-aniket`; Han pulls into `final` himself.
- Tokens: copy the originating pod's `.env` to the new pod (or paste the same keys
  into `/workspace/.tokens/{hf_token, anthropic_key, gh_token}`).

## Branch / worktree map (read this first)

| Branch | Tip @ 2026-05-04 | What it owns |
|---|---|---|
| `origin/final` | `1ede1997` | Canonical paper code. Has full c7 backtracking infra: `purified/experiments/c7_backtracking/{run,analysis,smoke}.py`, `purified/src/temp_bench/case_studies/backtracking.py` (1646 LOC), all 9 archs (`txc_base`, `txc_pro`, `tfa`, `tsae_paper`, `mlc`, `topk_sae`, `stacked_sae`, `tfa_pos`, `sae_arditi`), single canonical trainer at `purified/src/temp_bench/training/sae_trainer.py`. **Read-only for us; Han owns it.** |
| `origin/final-aniket` | `f99c0c39` | Same SHA as `det-steer`. Branched off `final` but **deletes** the case-study and arch implementations and replaces them with cross-component det_steer infra. Cannot run training on its own. **This is the staging branch for our work** — Han pulls from here into `final`. |
| `origin/det-steer` | `f99c0c39` | Identical to `final-aniket`. Purely additive: `purified/src/temp_bench/eval/{detection, steering_hooks, steering_protocols}.py`, `purified/src/temp_bench/utils/{shuffles, gpu_locks}.py`, `purified/experiments/det_steer/{validate_protocols, run_c7_locked, run_steering_ab}.py`, 51 tests. Documented in `purified/docs/cross_component/det_steer_summary.md`. |
| `origin/aniket-phase7-y` | `aec59bb7` | This handoff doc lives here. Older legacy structure (no `purified/`). Don't try to run c7 from this branch. |

**Worktree layout we used on the originating pod** (the fresh pod will likely
mirror this — recommended):

```
/workspace/aniket/temp_xc           → aniket-phase7-y       (legacy, holds this doc)
/workspace/aniket/temp_xc-final     → detached @ origin/final tip (where c7 runs)
/workspace/aniket/temp_xc-detsteer  → det-steer             (det-steer additions)
```

The new pod should:
1. Create the same three worktrees (or just `temp_xc-final` if no det-steer needed yet).
2. Apply the uncommitted edits to `run.py` listed below.
3. Run from `temp_xc-final/purified/`.

## Slack thread context (verbatim from Han)

The originating instruction from Aniket's user message that started this session:

> we're in final mode, we've submitted the abstract to neurips and need to run
> some final things. specifically, longer training on the backtracking study to
> make sure TXC isnt undertrained.

Slack thread highlights:

- **Han 7:41 AM**: "all of our TXCs are just undertrained in terms of total
  number of tokens seen during training? In the case studies we've been using
  `batch_size=256` and `<20k` training iterations, which works out to `<10M`
  tokens trained on, whereas the TFA and T-SAE papers both trained on 1B+ tokens."
- **Dmitry 12:15 PM**: "we should just get a GPU going which re-runs everything
  at 500k steps in order of priority. Ideally we'd be able to restart training,
  but getting the GPU handover from a saved run is a bit tricky." — i.e.
  checkpoint-resume is non-trivial; preferred path is fresh runs to N steps.
- **Han 1:09 PM**: "@Dmitry how much $$$ do we have left? If we have room to
  spare, I can launch an H200/B200 to do this extended training for the TXCs.
  Currently using $8.50/hr. Alternatively, I can do an 8×A40 setup and train 8
  things in parallel."
- **Aniket 2:04 PM**: "I have a pod with 2× H100s — can let it run on
  backtracking to see if things change with confirmed plateauing." (This handoff
  is the result. The new RunPod is 4× H100.)
- **Han 5:47 PM (the new ask, the reason for this handoff)**: "can someone train
  four TXCs: txc_base, txc_pro and txc_base T=20 and txc_pro T=20 at batch sizes
  {256, 1024} and save checkpoints every 100 iterations until we hit 300K
  iterations? On any task with a clear eval metric."

Aniket's verdict in this session: stay consistent with Han's `bs=1024, n_steps=20000`
sprint defaults for the 20K baseline, then run the same archs to 100K to see if
the result differs (= empirical answer to "is 20K undertrained?"). The 100K chain
is in flight on the originating pod. The 300K matrix Han just asked for is what
the **new 4× H100 pod is for**.

## C7 backtracking — what the case study is

(Full reference: `purified/docs/components/c7.md` and
`purified/src/temp_bench/case_studies/backtracking.py` on `final`.)

- **Setup**: subject model `meta-llama/Llama-3.1-8B` BASE (or
  `NousResearch/Meta-Llama-3.1-8B` mirror — confirmed bit-identical, commit
  `0e09c867`). Steering vectors derived from BASE, applied at inference on
  reasoning model `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`.
- **Hookpoint**: `resid_post` layer 10 (paper-justified, Ward et al. 2025
  Appendix B.1).
- **d_in = 4096**, d_sae = 32768 (8× expansion).
- **Cohort**: 31 truly-wrong + 30 originally-correct MATH-500 questions = 61
  panels × 25 magnitudes (`-16, -12, -10, -8, -7, -6, -5, -4, -3, -2, -1, -0.5,
  0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16`).
- **cut25 protocol**: cut Stage A trace at 25 % of unsteered length, then
  steer-and-continue.
- **Headline metric**: **peak Δgc** (gain in Sonnet-4.6-judged genuine
  backtracking events, baselined per `(arch, qid)` to mag=0).
- **Detection metric**: PR-AUC at S ∈ {1, 2, 4, 8, 16, 32} (NOT F1, ~12 %
  positive class).
- **Reference (wasteland, hill-climbed TXC)**: peak Δgc = **+1.574** at mag=−12.
  Locked archs at `bs=256, n_steps=30k`: txc_base 0.393, topk_sae 0.361,
  tsae_paper 0.246. Locked archs at `bs=1024, n_steps=20k`: not yet landed (in
  flight on originating pod).

## Token math: why undertraining is plausible

- `bs=256 × 5-token-window × 20k steps` = 25.6 M tokens.
- `bs=1024 × 5 × 20k` (current sprint default) = 102 M tokens.
- T-SAE paper §4.1 / TFA App B.1: ~500 M – 1 B tokens.
- GemmaScope: 2 B – 8 B tokens.
- → 102 M is ~10× under the literature floor. Direct loss-curve evidence on c7
  is unavailable (the c7 train_fn discards `result["log"]` — only c3/c4 added
  log persistence in commit `033a3eb6`).
- Han's own `decisions.md § 12` originally framed `bs=256, n_steps=30k` as
  "severely under-trained" and pushed everyone to `bs=1024, n_steps=25k`, then
  cut to 20k for the deadline.

## Originating pod state at 22:50 UTC 2026-05-04 (when this doc was written)

**Worktrees:**

```
/workspace/aniket/temp_xc           → aniket-phase7-y         (untouched)
/workspace/aniket/temp_xc-final     → detached @ 1ede1997     (active sweep)
/workspace/aniket/temp_xc-detsteer  → det-steer @ f99c0c39    (purely additive)
```

`temp_xc-final/purified/.venv` had a broken uv-python symlink on first probe;
fixed by `cd temp_xc-final/purified && rm -rf .venv && uv sync` (Python 3.12.13,
torch 2.8.0+cu128). The new pod will need to do the same in any worktree it
intends to run from.

**Tokens** in `/workspace/aniket/temp_xc/.env` (HF_TOKEN, ANTHROPIC_API_KEY,
GH_TOKEN, plus HF cache pointers). Source via `set -a; source .env; set +a` in
the launch shell.

**Manual fix applied** (matters because the new pod will hit the same bug if
sentence_acts cache is rebuilt from scratch):

- `np.savez_compressed(path)` auto-appends `.npz` if `path` doesn't already end
  in `.npz`. `case_studies/backtracking.py:1219` writes to
  `sentence_acts_L10.npz.tmp` then renames to `.npz` — but numpy actually wrote
  to `.npz.tmp.npz` so the rename fails.
- **Fix applied**: renamed `.npz.tmp.npz → .npz` (1.13 GB,
  `(25204, 6, 4096) fp32`, 12.6 % positive). The cache is correct; just the
  filename was wrong.
- **Permanent fix not yet committed**: change line 1219 to write the tmp file
  with name ending `.npz.tmp.npz` upfront so the rename target is correct. Land
  on `final-aniket`, not `final`.

**Uncommitted edits in `temp_xc-final/purified/experiments/c7_backtracking/run.py`:**

```diff
-def main(*, archs=None, seeds=DEFAULT_SEEDS, build_cache_only: bool = False,
-         force_train: bool = False, force_eval: bool = False):
+def main(*, archs=None, seeds=DEFAULT_SEEDS, build_cache_only: bool = False,
+         force_train: bool = False, force_eval: bool = False,
+         n_steps: int = 20_000):
     archs = archs or DEFAULT_ARCHS
-    log.info("[c7.run] datasource=%s", DATASOURCE)
+    log.info("[c7.run] datasource=%s n_steps=%d", DATASOURCE, n_steps)
```

```diff
-                    training_cfg=TrainingConfig(n_steps=20_000),
+                    training_cfg=TrainingConfig(n_steps=n_steps),
```

```diff
     ap.add_argument("--force-eval", action="store_true")
+    ap.add_argument("--n-steps", type=int, default=20_000,
+                    help="Training steps per cell (default 20000 — sprint baseline; "
+                         "use 100000+ for the undertrain-check extended pass).")
     args = ap.parse_args()
```

**Live processes on the originating pod** (do not duplicate on the new pod):

- PID `8456` — current 20K sweep, all 7 archs ordered
  `txc_base, txc_pro, tfa, topk_sae, stacked_sae, tsae_paper, mlc`, seed 42.
  Started 19:57:03 UTC. txc_base TRAINED but cell failed mid-eval on the savez
  bug (now fixed by rename); txc_pro mid-training as of doc-write time.
- PID `9609` — chain watcher (`/tmp/c7_chain.sh` on the originating pod). Polls
  PID 8456; when it exits, runs:
  1. `python -m experiments.c7_backtracking.run --archs txc_base --seeds 42 --n-steps 20000`
     to recover the missing 20K eval row (training cached → just eval, ~30 min).
  2. `python -m experiments.c7_backtracking.run --archs txc_base txc_pro tfa topk_sae stacked_sae tsae_paper mlc --seeds 42 --n-steps 100000`
     for the 100K full sweep.

This means the originating pod will produce, for seed=42, two leaderboard rows
per arch: one at `n_steps=20000` and one at `n_steps=100000`. Different
`train_keys` (training_cfg is hashed in), so no collision.

## Han's 300K matrix (the new 4× H100 pod's job)

8 cells:

| arch | T | bs | n_steps | snapshot_every | est. tokens |
|---|---|---|---|---|---|
| txc_base | 5 | 256 | 300 000 | 100 | 384 M |
| txc_base | 5 | 1024 | 300 000 | 100 | 1.54 B |
| txc_pro | 10 | 256 | 300 000 | 100 | 384 M (t_sample=5) |
| txc_pro | 10 | 1024 | 300 000 | 100 | 1.54 B |
| txc_base_t20 | 20 | 256 | 300 000 | 100 | 1.54 B |
| txc_base_t20 | 20 | 1024 | 300 000 | 100 | 6.14 B |
| txc_pro_t20 | 20 | 256 | 300 000 | 100 | 1.54 B |
| txc_pro_t20 | 20 | 1024 | 300 000 | 100 | 6.14 B |

### Disk reality check (read before launching)

- 3000 snapshots per cell × 8 cells = 24 000 snapshots.
- Per snapshot (state_dict only, bf16):
  - `txc_base`: 1.34 B params × 2 bytes ≈ **2.7 GB**.
  - `txc_pro`: 2.68 B params × 2 bytes ≈ **5.4 GB**.
  - `txc_base_t20`: ~2.7 GB × (T=20/T=5) on the encoder/decoder slabs ≈
    **~10 GB** (rough; depends on whether b_dec scales with T and on how the
    init scales the per-position W slabs).
  - `txc_pro_t20`: similarly **~20 GB**.
- 8 cells × 3000 ckpts ≈ **20–40 TB raw**. RunPod persistent volumes max out at
  ~1 TB. **A literal "every 100 iters" snapshot policy is not feasible.**

**Recommended snapshot policy** (you should confirm with Han):
- Every 1000 iters for steps 0–10K (10 snapshots).
- Every 5000 iters for steps 10K–100K (18 snapshots).
- Every 25 000 iters for steps 100K–300K (8 snapshots).
- = 36 snapshots per cell × 8 cells = 288 ckpts. At 5 GB avg = ~1.5 TB total.
  Fits on a 1 TB volume only with bf16 + selective archs.
- Alternative: every 100 iters for the first 5K, every 1000 thereafter — captures
  the early plateau Dmitry's after, doesn't blow the disk.

### GPU budget

Single H100 observation: txc_base @ 20K = ~38 min training, txc_pro @ 20K = ~100
min training (still in flight at doc-write time, expected ~70-100 min train
+ ~30 min eval).

Linearly extrapolated to 300K, sequential on a single H100:
- txc_base: ~9.5 hr / cell.
- txc_pro: ~25 hr / cell.
- T=20 variants will be ~2-4× slower than T=5/10 due to T-axis params + AuxK
  + (for txc_pro) matryoshka recon over more positions.

Rough total: **~150-300 GPU-hours** sequential.

On 4× H100 (the new pod) running 4 cells in parallel: 1 cell per GPU. Roughly
~75 hr wallclock for the heaviest cell, ~10-20 hr for the lightest. The
deadline is ≤70 hr from 2026-05-04 14:00 UTC — i.e. ~2026-05-07 12:00 UTC. The
heaviest cell may not finish before the deadline. Confirm scope with Han before
launching all 8.

### Wiring the new pod must do (not yet landed)

1. **Add T=20 yaml entries** in `purified/configs/locked_archs.yaml`. The
   simplest pattern is two new arches that inherit from txc_base / txc_pro
   with T overridden:

   ```yaml
   txc_base_t20:
     class_path: temp_bench.architectures.txc_base:TXCBase
     arch_version: "1.0.0"
     category: txc
     hparams:
       d_sae: 18432
       T: 20
       k_pos: 20
       auxk_alpha: 0.03125
       dead_threshold_tokens: 10_000_000
       bdec_geom_median_init: true
       decoder_unit_norm: true
       decoder_grad_orthogonalize: true
     per_component_hparams:
       c7:
         d_sae: 32768
     notes: "TXC-base with T=20 (Han 2026-05-04 PM 4×H100 ask)."

   txc_pro_t20:
     class_path: temp_bench.architectures.txc_pro:TXCPro
     arch_version: "1.0.0"
     category: txc
     hparams:
       d_sae: 18432
       T_max: 20
       t_sample: 5  # keep the same training-time subset; T_max is what changes
       k_pos: 20
       n_matryoshka: 8
       contrastive_shifts: [1, 2]
       contrastive_inverse_distance_weight: true
       auxk_alpha: 0.03125
       dead_threshold_tokens: 10_000_000
       bdec_geom_median_init: true
       decoder_unit_norm: true
       decoder_grad_orthogonalize: true
     per_component_hparams:
       c7:
         d_sae: 32768
     notes: "TXC-pro with T_max=20 (Han 2026-05-04 PM 4×H100 ask)."
   ```

   Open question for Han: should `t_sample` also change for the T=20 variant?
   Default Phase 5b H8 was T_max=10 / t_sample=5 (50% subset). Mirror that to
   T_max=20 / t_sample=10? Or hold t_sample=5 to match the original? **Default
   to t_sample=5** unless Han says otherwise — keeps the contrastive batch math
   and TopK budget identical to the headline arch.

2. **Activation cache for T=20**: `purified/configs/datasources.yaml` has
   `seq_len=128` for `llama_3_1_8b_base_l10_ward_nousmirror` — plenty of room
   for T=20 windows. The `_build_batch_iter` in `run.py:100-132` samples
   T-token sliding windows from `(N=4044, L=128, d=4096)`, so as long as
   `T ≤ L`, we're fine. **No new act cache needed.**

3. **Add `--batch-size N` and `--snapshot-every N` flags to `run.py`**:

   ```python
   ap.add_argument("--batch-size", type=int, default=1024,
                   help="Per-step batch size (default 1024 — sprint baseline).")
   ap.add_argument("--snapshot-every", type=int, default=0,
                   help="If >0, save intermediate state_dict to "
                        "<train_key>/snapshots/step_<N>.safetensors every N steps. "
                        "Use 100 for the Han 2026-05-04 PM 300K matrix.")
   ```

   Plumb both into `main()` and the `runner.run_cell` call:

   ```python
   training_cfg=TrainingConfig(n_steps=n_steps, batch_size=batch_size),
   ```

   For snapshot_every: `train_sae` already accepts `snapshot_every` and
   `snapshot_fn` parameters (see
   `purified/src/temp_bench/training/sae_trainer.py`). The c7 `my_train_fn`
   adapter doesn't currently pass them. Add:

   ```python
   def my_train_fn(*, arch_name, arch_hparams, seed, training_cfg, act_cache_key,
                   component, snapshot_every: int = 0):
       ...
       def _snapshot_fn(step: int, payload: dict[str, Any]) -> None:
           # Persist state_dict + log to disk under a deterministic path.
           # Use the train_key from compute_train_key(); see runner.py:run_cell
           # for how the canonical path is built.
           from safetensors.torch import save_file
           from temp_bench.config import compute_train_key, checkpoints_dir
           from temp_bench.config import load_arch
           spec = load_arch(arch_name, component=component)
           train_key = compute_train_key(arch=spec, seed=seed,
                                         training_cfg=training_cfg,
                                         act_cache_key=act_cache_key)
           snap_dir = checkpoints_dir() / train_key / "snapshots"
           snap_dir.mkdir(parents=True, exist_ok=True)
           save_file(payload["state_dict"], snap_dir / f"step_{step}.safetensors")

       result = train_sae(model, batch_iter, training_cfg,
                          snapshot_every=snapshot_every,
                          snapshot_fn=_snapshot_fn if snapshot_every > 0 else None)
       return result["state_dict"]
   ```

   Then thread `snapshot_every` from CLI → main() → runner.run_cell's
   `train_fn` (the protocol allows extra kwargs via partial).

4. **Persist the train log too** while you're at it — c7 currently throws away
   `result["log"]`. Mirror agent_nlp's commit `033a3eb6`:

   ```python
   log_path = workspace / "train_log.json"
   workspace.mkdir(parents=True, exist_ok=True)
   log_path.write_text(json.dumps(result["log"]))
   ```

   This unlocks the loss-plateau check Dmitry was asking for.

## Eval cadence

Han said "on any task with a clear eval metric." C7 has Δgc + PR-AUC. But
running the **full** eval (cohort generation + Sonnet judge dispatch) on every
snapshot is impractical (~30 min × 3000 snapshots × 8 cells = months of judge
calls + thousands of dollars in Anthropic charges).

**Recommended eval policy**:
- Run `experiments/c7_backtracking/run.py`'s eval phase only on **selected
  milestones** — e.g. 10K, 30K, 100K, 200K, 300K snapshots (5 evals per cell,
  40 evals total).
- Train + snapshot continuously; eval async after the training cell completes.
- Build a small `evaluate_snapshot.py` wrapper that:
  1. Loads `snapshots/step_<N>.safetensors` into the arch instance.
  2. Calls `run_arch_evaluation(arch, ...)` from
     `case_studies/backtracking.py`.
  3. Persists a `metrics_step_<N>.json` next to the snapshot.

This gives a **Δgc-vs-step curve per arch** which is the actual product Han is
after.

## Det-steer integration (DEFERRED — do not block on this)

Per `purified/docs/cross_component/det_steer_summary.md`, det-steer adds:

- `temp_bench.eval.steering_hooks.TXCSteeringHook` — V0 (mean-decoder
  constant — current C7 default; equivalent to TopK-SAE steering), V1
  (position-cycled), V2 (trailing-window), V4 (encoder pre-image). With
  `√T` energy correction.
- `temp_bench.eval.detection.detect_case_study` — sparse-probe PR-AUC + paired
  within-window shuffle ablation.
- Diagnostics: `position_variance(W_dec)`, `encoder_decoder_divergence(arch, fid)`.

Per the audit doc: current C7 steering is **TopK-SAE-equivalent** for TXC
(throws away the temporal trajectory via `W_dec.mean(dim=1)`). The audit
suggests V1/V2/V4 may unlock TXC's actual steering ceiling.

**Layer det-steer in only after the 300K matrix produces snapshots.** Pattern:

```bash
# From temp_xc-final/purified, with det-steer src on PYTHONPATH:
TQDM_DISABLE=1 PYTHONPATH=/workspace/aniket/temp_xc-detsteer/purified/src \
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m experiments.det_steer.run_steering_ab \
    --arch txc_pro --feature_id <selected> --modes v0,v1,v2,v4
```

This shares cohort + judge with `case_studies.backtracking`, so
`judge_outputs.jsonl` cache hits make the V0 baseline cheap.

## Concrete launch checklist for the new 4× H100 pod

1. **Get the repo**:
   ```bash
   cd /workspace/aniket
   git clone https://github.com/chainik1125/temp_xc.git
   cd temp_xc
   git fetch --all
   git worktree add -b extended-300k ../temp_xc-final origin/final  # or final-aniket if you want det-steer added
   git worktree add ../temp_xc-detsteer origin/det-steer
   ```

2. **Sync env in `temp_xc-final/purified`**:
   ```bash
   cd /workspace/aniket/temp_xc-final/purified
   uv sync   # builds .venv with Python 3.12 + torch 2.8.0+cu128
   ```

3. **Tokens** — copy `/workspace/aniket/temp_xc/.env` from the originating pod or
   recreate with the same keys (HF_TOKEN, ANTHROPIC_API_KEY, GH_TOKEN). Source
   before launching:
   ```bash
   set -a; source /workspace/aniket/temp_xc/.env; set +a
   ```

4. **Apply the run.py edits** (already on the originating pod's worktree —
   either pull from a branch we push, or replicate the diff above).

5. **Add yaml entries** for `txc_base_t20` and `txc_pro_t20`.

6. **Land the savez fix** in `case_studies/backtracking.py:1219` — write the
   tmp file with name ending in `.npz.tmp.npz` upfront, OR sync the
   `sentence_acts_L10.npz` file from the originating pod to skip the extraction.

7. **Plumb `--batch-size` and `--snapshot-every`** as described above.

8. **Decide snapshot cadence with Han** (every 100 iters is disk-prohibitive;
   propose the staircase above).

9. **Launch the matrix**, one cell per GPU. The runner's caching means
   crashed/restarted cells re-use the trained checkpoint:
   ```bash
   # GPU 0 — txc_base × bs=256
   CUDA_VISIBLE_DEVICES=0 TQDM_DISABLE=1 AGENT_NAME=4xh100_extended \
     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
     .venv/bin/python -m experiments.c7_backtracking.run \
       --archs txc_base --seeds 42 \
       --n-steps 300000 --batch-size 256 --snapshot-every 1000 \
     > logs/c7_300k_txc_base_bs256.log 2>&1 &

   # GPU 1 — txc_base × bs=1024
   CUDA_VISIBLE_DEVICES=1 ... --batch-size 1024 ...

   # GPU 2 — txc_pro × bs=256
   CUDA_VISIBLE_DEVICES=2 ... --archs txc_pro --batch-size 256 ...

   # GPU 3 — txc_pro × bs=1024
   CUDA_VISIBLE_DEVICES=3 ... --archs txc_pro --batch-size 1024 ...

   # ... then T=20 variants in a second wave once the first wave finishes.
   ```

10. **Monitor**: tail logs, watch
    `purified/results/leaderboard.jsonl` for new rows, `purified/checkpoints/`
    for snapshots.

## Reference numbers (for sanity-check after 300K runs land)

| arch | regime | peak Δgc | source |
|---|---|---:|---|
| TXC (k=16, T=6, L0=96, hill-climbed) | wasteland | **+1.574** | `origin/aniket-ward-stage-b @ a62175ee` (NOT a paper claim) |
| txc_base | bs=256, n_steps=30k | +0.393 | `purified/results/leaderboard.jsonl` |
| topk_sae | bs=256, n_steps=30k | +0.361 | same |
| tsae_paper | bs=256, n_steps=30k | +0.246 | same |
| topk_sae | bs=256, n_steps=30k (older) | +0.131 | same |
| txc_pro | bs=1024, n_steps=20k | TBD (in flight on originating pod) | leaderboard, will appear |
| txc_base | bs=1024, n_steps=100k | TBD (chain) | leaderboard, will appear |
| txc_base / txc_pro / T=20 variants | bs={256,1024}, n_steps=300k | TBD (this pod) | leaderboard, NEW |

Watch for the "is 20K severely undertrained?" answer in the diff between the
20K and 100K rows on the originating pod. If peak Δgc jumps significantly going
20K → 100K (say > 2σ across seeds, or > 2× the 20K value), that confirms 20K
is binding. The 300K matrix on the new pod then quantifies how far the curve
keeps climbing AND tests whether longer windows (T=20) extract more signal than
T=5/10 once given enough compute.

## Open questions for Han (surface, do not assume)

1. **Snapshot cadence**: every-100-iters policy is disk-prohibitive at 300K
   steps × 8 cells. Confirm the staircase proposed above (or alternative).
2. **txc_pro_t20.t_sample**: keep at 5 (= T=5/10 default), match T (= 20),
   or sweep? Default to 5 unless Han says otherwise.
3. **Eval cadence**: full Δgc + PR-AUC eval on every snapshot is unaffordable
   in Anthropic API spend. Confirm the 5-milestone evaluation schedule (10K,
   30K, 100K, 200K, 300K) per cell.
4. **Snapshot persistence**: keep all snapshots, or only the final + 5
   eval-targets? At ~5 GB each, the full set chews through a 1 TB volume.
5. **Is this still "any task with a clear eval metric" = C7 only?** Or should
   the matrix run on C3 (sparse probing, AUC) or C6 (EM, Wang) too? Multi-task
   would multiply GPU-hours by 3×.

## Files modified or created during this session

- `/workspace/aniket/temp_xc-final/purified/experiments/c7_backtracking/run.py`
  — added `n_steps` arg + `--n-steps` CLI flag (uncommitted).
- `/workspace/aniket/temp_xc-final/purified/.venv` — rebuilt via `uv sync`
  after broken uv-python symlink.
- `/workspace/aniket/temp_xc-final/purified/results/c7_backtracking/stage_a/sentence_acts_L10.npz`
  — renamed from `.npz.tmp.npz` after the savez bug bit; file content correct.
- `/tmp/c7_chain.sh` (originating pod only) — chain watcher script.
- `/workspace/aniket/temp_xc-final/purified/logs/c7_extended_smoke.log` —
  current 20K sweep log.
- `/workspace/aniket/temp_xc-final/purified/logs/chain_watcher.log` — chain
  watcher progress.
- `/workspace/aniket/temp_xc/docs/aniket/2026-05-04-extended-training-handoff.md`
  — this file.

## What I didn't do (to avoid stepping on Han's territory)

- Never pushed to `final`. All edits sit on local worktrees pending Han's review.
- Did not commit the run.py edits — the user's policy is "we write things to
  `final-aniket`, Han pulls into `final`." When ready, commit on `final-aniket`
  or a topic branch off it.
- Did not modify `purified/configs/locked_archs.yaml` (out of agent_back's
  scope per the original briefing); the T=20 entries above are recommended
  text but should land on `final-aniket` only.
- Did not commit the savez bug fix; it's a `final` file.

## How to find this doc later

Path: `docs/aniket/2026-05-04-extended-training-handoff.md` on
`origin/aniket-phase7-y` (or wherever the user pushes it). Suggested commit:

```bash
cd /workspace/aniket/temp_xc
git add docs/aniket/2026-05-04-extended-training-handoff.md
git commit -m "Y handoff: extended-training pod brief for Han's 300K matrix"
git push origin aniket-phase7-y
```

The new pod's CC instance starts by reading this doc, then begins the
checklist above.
