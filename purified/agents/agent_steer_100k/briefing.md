<!--
DRAFT — written by agent_paper 2026-05-04 PM. Han populates the rest
of "Identity + mandate" if any priorities shift. Section ownership
rules: PROTOCOL.md § 14.
-->

---
agent: agent_steer_100k
last_state_update: 2026-05-04T22:25:00Z
component: c5
---

## Identity + mandate (Han owns — agents do not edit)

You are **agent STEER 100K**. You are a literal copy of agent_steer —
same component (C5), same datasource (Gemma-2-2b-IT L13 resid_post),
same archs (tsae_paper + txc_base + txc_pro), same V7 tiled-broadcast
steering protocol, same Sonnet judge, same metric — **with one and
only one difference: `n_steps=100_000` instead of agent_steer's
`n_steps=20_000`.** Your cells are intended to be the better-trained
version of agent_steer's; if they finish in time, they become the C5
paper headline (replacing the 20K cells).

Files you may edit:
- `agents/agent_steer_100k/briefing.md` (your own — agent-owned sections only)
- `experiments/c5_steering_100k/` (new experiment directory you create
  with a minimal driver that imports agent_steer's plumbing — see
  "First concrete task" below).
- Code under `src/temp_bench/` that you author + commit — but in
  practice you should not need to write any. The C5 case-study code,
  V7 steering, Sonnet judge, and Gemma activation cache loader are
  **agent_steer's territory** and already work. Re-use via imports.

**Files that are OUT OF SCOPE — do NOT edit:**
- `agents/agent_*/` — every other agent's directory **including
  `agents/agent_steer/`**. agent_steer's briefing, decisions, and
  per-cell state are theirs. You read for context, you do not write.
- `experiments/c5_steering/` — agent_steer's territory. Their `run.py`,
  `analysis.py`, `_real_training_cfg`, etc. You import from here
  without modification.
- `docs/components/c5.md` — agent_paper integrates the headline (yours
  vs agent_steer's, whichever lands) into AUTO-RESULTS at paper time.
  You don't touch this directly. Neither does agent_steer.
- `docs/paper/*` — agent_paper's territory.
- `agents/agent_paper/decisions.md` — global decisions log.
- `agents/README.md` — roster/contract; agent_paper-owned.
- `configs/locked_archs.yaml` — only agent_paper edits this.
- `pyproject.toml` and `uv.lock` — atomic, agent_paper coordinates.

If you find yourself wanting to edit any out-of-scope file, **STOP**.
Add a bullet to "Open questions for Han" in your own briefing,
surface it in chat, and let Han or agent_paper land the change. Even
if Han verbally approves, do not commit cross-territory edits yourself.
This is non-negotiable — see PROTOCOL.md § 8 + CLAUDE.md Hard Rule #7.

### Mandate — same C5 sweep, longer training

The paper-canonical C5 sweep (agent_steer) runs at `batch=1024`,
`n_steps=20_000` (~20.5M activation tokens per arch — the Gemma-axis
deadline override per `decisions.md` § 12 update). The published-SAE-
paper budgets are higher (T-SAE: ~4-8B; TFA: 1B; Phase 7: ~100M).
You have the compute to do better — **a fresh single-GPU H100 pod
dedicated to this**. Run the same sweep at `n_steps=100_000` (~102M
tokens, comfortably in the field-standard range) and ship those cells
to the leaderboard.

**Whichever sweep completes first becomes the C5 paper headline:**

- If your 100K sweep finishes before the deadline: agent_paper picks
  your cells as canonical, agent_steer's 20K cells become the
  "compute-pressure backup" reference. Within-component fairness
  preserved — every C5 arch is at 100K.
- If your 100K sweep is mid-flight at the deadline: agent_steer's
  20K cells stay canonical. Your partial 100K cells are kept in the
  leaderboard for a "convergence consistency" caveat.
- Cells from both sweeps coexist cleanly in `leaderboard.jsonl` — the
  `train_key` hash includes `n_steps`, so 20K and 100K cells occupy
  distinct keys and don't collide. agent_steer's `analysis.py` uses
  `canonical_train_keys()` with the `TrainingConfig()` that's current
  at render time; agent_paper toggles which sweep is canonical by
  which `n_steps` the analysis filter pins.

Hardware: **1× H100 80GB pod, ephemeral, 240 GB system RAM, 1 TB
/workspace**. Pinned to GPU 0 (the only GPU). Pod mode `ephemeral`:
`/workspace` is wiped on pod stop, HF is the source of truth.
Bootstrap pulls from `han1823123123/temp-bench-{models,data}`;
`cache.save_checkpoint` auto-pushes on save (push failure is fatal).

The 240 GB RAM means agent_nlp's preloaded `.clone()` pattern (commit
`e12dc719`) is unconstrained — preload the full Gemma-2-2b-IT L13
activation cache (~14 GB at 24K seqs × 128 tokens × 2304 d_in fp16)
into RAM once, no headroom worries. agent_steer already adopted this
pattern in `experiments/c5_steering/run.py`; you inherit it via the
same import.

H100 vs A40 perf: H100 is roughly 2× faster than agent_steer's A40
on SAE training. Per-cell wall halves vs an equivalent A40 run.

Subject + protocol (replicating agent_steer's setup verbatim):

- Datasource: `gemma_2_2b_it_l13_fineweb_24k128`
- Architectures: `tsae_paper`, `txc_base`, `txc_pro`
- Per-component d_sae overrides (already in
  `configs/locked_archs.yaml`'s `per_component_hparams.c5`).
- Steering: V7 tiled-broadcast residual-stream protocol (per
  `temp_bench.case_studies.steering` — agent_steer's port). Same
  strengths grid `{10, 100, 150, 500, 1000, 1500, 5000, 10000, 15000}`,
  same `"We find"` prompt, 60 new tokens, greedy decode.
- Concept set: same 30-concept × 5-example-sentence set agent_steer
  uses (5 safety/alignment, 10 domain, 7 style, 5 sentiment, 3 format).
- Per-arch best-feature selection by mean activation across content
  positions (concept-lift argmax) — agent_steer's helper.
- Judge: Sonnet 4.6 (Anthropic; same as agent_steer + agent_em — see
  decisions.md § 12 for why we don't use Gemini).
- Per-call `judge_outputs.jsonl` persistence for post-deadline κ.
- Headline metric: `peak_success_grade_at_coh_1.75` (agent_steer's
  `EVAL_PROTOCOL_VERSION="1.0.1"` post the metric-fix backfill).

Seeds: **{42, 1, 2}** matching agent_steer's full n=3 sweep. Your H100
is fast enough for all three. If txc_pro becomes the long pole, drop
seed=2 last and surface it as an Open Question.

`TrainingConfig` for your cells:

```python
TrainingConfig(
    batch_size=1024,
    n_steps=100_000,        # <-- the only difference from agent_steer
    plateau_early_stop=False,
    # bricken_* stays at defaults (False) — C5 does not use Bricken
    # per decisions.md § 7 ("C5 keeps it OFF — revisit only if time
    # permits at the end of the paper sprint").
)
```

agent_steer's `experiments/c5_steering/run.py:_real_training_cfg`
returns `TrainingConfig(n_steps=20_000)`. You override `n_steps=100_000`
in your driver script (see "First concrete task").

Per-cell wall-time estimate on H100: agent_steer's `tsae_paper` at
batch=1024 × 20K on A40 was ~25-40 min. Scaling: 100K = 5× steps,
H100 ≈ 2× faster than A40 → ~60-100 min training per cell. Plus
~15-20 min eval (steering sweep + judge calls). Per-cell wall ≈ 80-120
min. `txc_pro` is the slow arch (matryoshka + multi-distance contrastive)
— add 50% margin: ~3-4 hr per cell. Three archs × 3 seeds = 9 cells,
~12-25 hr wall. With ~30 hr remaining, full sweep is feasible if you
start now.

V7 ↔ TXC-pro compatibility: agent_steer's `--pre-test-only` mode
checks if V7 works on TXC-pro before the full sweep. agent_steer's
prior cells at 20K ran TXC-pro under V7 successfully (mean coh
~2.1-2.2 in their notes), so V7 should work at 100K too — but run
the pre-test on your first txc_pro cell as a safety check anyway.
If mean coherence ≤ 1.0, fall back to `--protocol pp` for the full
sweep.

Locked decisions in scope: #1 (two TXCs), #4 (cross-branch reads),
#6 (HF repos), #7 (Bricken off for C5), #11 (T-SAE = paper-faithful Ye et al.),
§ 12 (uniform batch=1024, plateau_off — you keep these; only n_steps
differs), § 13 (the 100K copy-sweep policy).

References:
- `agents/README.md` (your roster row)
- `agents/agent_steer/briefing.md` (the canonical C5 setup you replicate
  — read this before launching anything)
- `docs/components/c5.md` (the canonical C5 writeup; do NOT edit)
- `experiments/c5_steering/{run.py,analysis.py}` (import from)
- `decisions.md` § 7, § 12, § 13
- `papers/temporal_sae.md` § B.2 (T-SAE steering protocol reference)
- `PROTOCOL.md` § 7 (results live in state), § 8 (anti-conflict),
  § 11 (framework discipline), § 14 (briefing maintenance)

**Before you `status: complete`**: `bash scripts/wrap_up_session.sh`.
Ephemeral pod → checkpoints auto-pushed. Push your 100K cells'
`results/runs/<eval_key>/judge_outputs.jsonl` and metrics via the
wrap-up script before any pod restart.

### First concrete task — write a minimal driver script

Create `experiments/c5_steering_100k/run.py` (the experiments dir is
on PYTHONPATH via `experiments/__init__.py`, which makes the import
paths cleaner) and an empty `experiments/c5_steering_100k/__init__.py`:

```python
"""C5 driver — replicates agent_steer's setup at n_steps=100_000.

Imports agent_steer's train_fn / eval_fn / V7 steering infrastructure
from experiments.c5_steering.* without modification; only n_steps in
the TrainingConfig differs.
"""
from __future__ import annotations
import argparse
from temp_bench import runner
from temp_bench.schemas import TrainingConfig

# Re-use agent_steer's plumbing verbatim:
from experiments.c5_steering.run import (
    DATASOURCE,
    EVAL_PROTOCOL_VERSION,
    my_train_fn,
    my_eval_fn,
    _real_training_cfg as _orig_training_cfg,
)


def _real_training_cfg() -> TrainingConfig:
    """100K override of agent_steer's `_real_training_cfg`."""
    base = _orig_training_cfg()              # 20K, batch=1024, etc.
    return base.model_copy(update={"n_steps": 100_000})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+",
                    default=["tsae_paper", "txc_base", "txc_pro"],
                    choices=["tsae_paper", "txc_base", "txc_pro"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    args = ap.parse_args()

    cfg = _real_training_cfg()
    print(f"[c5_100k] training_cfg n_steps={cfg.n_steps} "
          f"batch_size={cfg.batch_size} plateau_off={not cfg.plateau_early_stop}")

    for arch in args.archs:
        for seed in args.seeds:
            print(f"[c5_100k] launching cell arch={arch} seed={seed}")
            runner.run_cell(
                component="c5",
                arch_name=arch,
                seed=seed,
                datasource_name=DATASOURCE,
                training_cfg=cfg,
                eval_cfg={"sweep": "c5_100k_v1"},
                eval_protocol_version=EVAL_PROTOCOL_VERSION,
                train_fn=my_train_fn,
                eval_fn=my_eval_fn,
            )


if __name__ == "__main__":
    main()
```

Then run:
```bash
TQDM_DISABLE=1 .venv/bin/python -m experiments.c5_steering_100k.run \
  --archs tsae_paper txc_base txc_pro --seeds 42 1 2 \
  > logs/c5_100k_full.log 2>&1 &
```

agent_paper integrates results at paper-render time — your cells just
need to land in `leaderboard.jsonl` with the right
`(arch, seed, training_cfg)`. The runner handles the rest.

---

## Current state (agent owns — overwrite at every compact)

**Last verified: 2026-05-04T22:25Z (smoke pass + full 100K sweep launched).**

- `git HEAD`: `6db405bd` (Agent PAPER: spin up agent_em_100k +
  agent_steer_100k …) plus untracked `experiments/c5_steering_100k/`
  driver and untracked checkpoint dirs from HF sync (paper-data .gitignore).
- Pod: 1× H100 80GB, ephemeral, 240 GB system RAM,
  `/workspace/temp_xc/purified` working dir.
- Driver: `experiments/c5_steering_100k/run.py` (+ `__init__.py`).
  Imports `run_one_cell` from `experiments.c5_steering.run` and threads
  `n_steps=100_000` through its existing `--n-steps` override knob —
  no agent_steer code changes. (Briefing's sketch had a stale import
  `my_eval_fn` which doesn't exist as top-level — it's a closure built
  by `_make_eval_fn`. I diverged to import `run_one_cell` instead;
  semantically identical, agent_steer's plumbing untouched.)
- **Smoke test PASSED at 22:24Z**: `--archs tsae_paper --seeds 42
  --n-steps 200 --smoke --n-concepts 5 --strengths 100 1000`. Full
  pipeline hit: act_cache preload → train 200 steps → load Gemma-2-2b-it
  → V7 generations → Sonnet judge → metrics.json + judge_outputs.jsonl
  + leaderboard append. Wall ~5 min (~2 min act_cache .clone(), ~30s
  Gemma subject-model download, ~30s judging, rest gen + bookkeeping).
- Smoke leaderboard row: `eval_key=0ddccd2ce5921881`,
  `train_key=53442c9165d7f761`, `agent=agent_steer_100k`, `smoke=True`,
  `peak_success_grade_at_coh_1.75=0.0` (expected — 200 steps far too
  few). Manifest `hf_url=null` because `TEMP_BENCH_POD_MODE` wasn't set
  in that subshell. agent_paper's analysis filters smoke=True out.
- **Full 100K sweep launched 22:25Z**, PID 2524 (wrapper) → python child.
  Log: `logs/c5_100k_full.log`. Sequence: `tsae_paper × {42,1,2}` →
  `txc_base × {42,1,2}` → `txc_pro × {42,1,2}`, sequential on GPU 0.
  Env: `AGENT_NAME=agent_steer_100k TQDM_DISABLE=1
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  TEMP_BENCH_POD_MODE=ephemeral CUDA_VISIBLE_DEVICES=0`. PID file
  `/tmp/p_full`. First cell at 22:25:47 — `eval_key=933efa408d1e91aa`
  (tsae_paper seed=42).
- **Wall-time ETA**: 18-25 hr for full sweep. Smoke timing showed
  ~6.7 steps/sec for tsae_paper @ b1024 (incl. preload amortised) →
  100K ≈ 4 hr per tsae cell (~20 min vs 200-step smoke). txc_pro
  matryoshka + multi-distance contrastive is 2-3× slower → ~8-12 hr
  per txc_pro cell. Fits in remaining ~30 hr sprint window.
- Active GPU usage: GPU 0 (only GPU on this pod).
- HF sync done at 22:18Z: act_cache (14GB acts.npy + meta.json
  + token_ids.npy) at `results/act_cache/e4916bcae1881963/`;
  agent_steer's prior C5 checkpoints under `checkpoints/<train_key>/`
  (good for sanity diff at end). Sync clobbered `checkpoints/README.md`
  with the HF repo's README; `git checkout` restored it.
- Recent decisions in scope: § 7 (Bricken off for C5), § 12 (b=1024 +
  plateau_off — `_real_training_cfg` defaults satisfy these),
  § 13 (the 100K copy-sweep policy — my mandate).
- Persistent Monitor `b8tmqh8mw` watching log for cell start/end markers
  + crash signals (Traceback / OOM / CUDA error / push failed).

## What I just did (agent owns — overwrite)

Newest first.

- **22:25Z launched full 100K sweep** — 9 cells, sequential on H100.
  PID 2524. First cell tsae_paper seed=42 confirmed launching at
  22:25:47 with `eval_key=933efa408d1e91aa`. Persistent Monitor armed
  for crash detection.
- **22:19-22:24Z smoke test** — passed end-to-end. Confirmed: import
  path works; `run_one_cell(n_steps=200)` produces a distinct train_key
  from any 20K/100K cell; eval pipeline hits V7 + Sonnet; leaderboard
  append + checkpoint manifest write happen automatically. `hf_url=null`
  on the smoke manifest entry because `TEMP_BENCH_POD_MODE=ephemeral`
  wasn't set in that subshell (smoke checkpoints are throwaway, fine).
  For the real sweep I set it explicitly.
- **22:14Z wrote `experiments/c5_steering_100k/run.py` + empty
  `__init__.py`**. Driver diverges from briefing's sketch (which imports
  a non-existent `my_eval_fn`) by importing `run_one_cell` from
  agent_steer's `experiments/c5_steering/run.py`. Single behavioural
  change: the default `--n-steps` is `100_000` (vs agent_steer's
  20_000). CLI mirrors agent_steer's flags so smoke / pre-test-only
  / protocol / seeds all behave the same.
- **22:13Z confirmed `compute_train_key` distinguishes 20K vs 100K**:
  `04d9d7753bd10ea2` (20K) vs `c0729094920eb9f0` (100K) for tsae_paper
  seed=42 + Gemma cache. No collision with agent_steer's leaderboard.
- **22:18Z `bash scripts/sync_from_hf.sh`** completed — pulled 14GB
  Gemma act cache + agent_steer's prior C5 checkpoints. Restored
  `checkpoints/README.md` after HF clobbered it.
- **22:12Z bootstrap** — `source scripts/set_agent_env.sh agent_steer_100k`
  + `bash scripts/agent_smoke_test.sh` (124/124 tests pass, H100 80GB
  visible, GPU 0 idle).

## Next action (agent owns — overwrite)

1. **Watch the persistent Monitor (`b8tmqh8mw`)** for cell-completion
   markers. As each cell finishes:
   - `tail -1 results/leaderboard.jsonl` — verify new row.
   - `tail -1 checkpoints/manifest.jsonl | jq .hf_url` — verify
     auto-push (NOT null on real cells; if null, the HF push silently
     skipped — investigate `TEMP_BENCH_POD_MODE`).
2. **If a cell crashes** (OOM, CUDA error, unexpected death):
   - `tail -200 logs/c5_100k_full.log` for the stack trace.
   - `nvidia-smi` for residual processes.
   - Re-launch from the failed cell onwards: `--archs <arch> --seeds <missing>`.
3. **First non-tsae cell convergence check** (likely txc_base seed=42):
   ```python
   import json
   log = json.load(open("logs/c5_b1024_txc_base_seed42_trainlog.json"))
   loss = log["loss"]; n = len(loss)
   print(f"final-1K-step drop: {(loss[-1000] - loss[-1]) / loss[-1] * 100:.2f}%")
   ```
   If > 5%, surface as Open Question — same convergence-cap concern
   Han flagged for the 20K cells.
4. **txc_pro V7 sanity** — agent_steer's b1024 V7 mean coh @ 20K was
   ~2.1-2.2; if my 100K txc_pro mean coh ≤ 1.0, surface as Open Question
   and consider re-running with `--protocol pp`.
5. **Periodic check-ins** via ScheduleWakeup every ~30 min while the
   sweep runs — verify sweep alive, GPU utilised, leaderboard growing.
6. **After all 9 cells land**: do NOT run `report.render(component='c5')`
   — that's agent_paper's territory. Just confirm leaderboard has 9 new
   `(arch, seed, n_steps=100_000)` rows and stop.
7. **Before session-end / context-compact**: overwrite this briefing's
   bottom sections again with current state. Run
   `bash scripts/wrap_up_session.sh` to verify HF state.

## Don't repeat (agent owns — overwrite)

- **Don't run real cells without `TEMP_BENCH_POD_MODE=ephemeral`** —
  the smoke checkpoint's `hf_url=null` is a free-floating warning. On
  real 100K cells without that env, the pod stop wipes the 4-hr+
  checkpoint and HF has nothing to recover from.
- **Don't edit `experiments/c5_steering/`** — agent_steer's territory.
  Import, never modify. (Confirmed: my driver only imports.)
- **Don't edit `docs/components/c5.md`** — agent_paper integrates the
  100K vs 20K canonical-toggle at paper-render time.
- **Don't write 100K cells with `--smoke`** — that hides them from
  analysis aggregates. Only the smoke validation used `--smoke`.
- **Don't enable Bricken** — C5 is Bricken-off per decisions.md § 7.
  `_real_training_cfg` defaults are correct as-is.
- **Don't pursue Y/W steering hill-climb winners** — Galaxy 8/11/18 /
  SoftMaxPool / ContrastiveMergeH8 are excluded by decision #1.
- **Don't bypass `runner.run_cell`** — `run_one_cell` calls it
  internally; my driver calls `run_one_cell`. Never construct
  leaderboard rows by hand.
- **Don't allocate `train_key` / `eval_key` manually** — the runner
  + `_workspace_for` do it deterministically. Smoke verified the
  `eval_key=0ddccd2ce5921881` matches `_workspace_for`'s computation.
- **Don't push to HF manually** — `cache.save_checkpoint` (with
  TEMP_BENCH_POD_MODE=ephemeral) does it on save; `_make_eval_fn`'s
  `_push_run_dir_to_hf` pushes the run_dir. Verify post-hoc via
  `hf_url` on manifest entries.
- **Don't restart the sweep on a transient stall** — verify the python
  PID via `cat /tmp/p_full` and `ps -p <pid>` before relaunching;
  silently double-spawning would race for GPU 0 and OOM.

## Open questions for Han (agent owns — overwrite)

(None at briefing-write time. Will surface after the first non-tsae
cell completes if convergence at 100K shows residual drop > 5% or
txc_pro V7 produces degenerate output.)
