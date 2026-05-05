<!--
DRAFT — written by agent_paper 2026-05-04 PM, REWRITTEN 2026-05-05
to reflect the C5 MW pivot (the original 100K convergence-test
mission was abandoned — see "Mission pivot" below).
Section ownership: PROTOCOL.md § 14.
-->

---
agent: agent_steer_100k
last_state_update: 2026-05-05T11:00:00Z
component: c5 (multi-window deployment)
---

## Identity + mandate (Han owns — agents do not edit)

You are **agent STEER 100K** (legacy name; mission has pivoted —
see below). You own the **C5 multi-window deployment** only. Files
you may edit:

- `agents/agent_steer_100k/briefing.md` (your own — agent-owned sections only)
- `experiments/c5_steering_mw/` (new experiment directory you create
  with a minimal driver — see "First concrete task" below).
- Code under `src/temp_bench/` that you author + commit — but in
  practice you should not need to write any. The C5 case-study code,
  V7 steering, Sonnet judge, Gemma activation cache loader, and the
  concept-lift baseline fix are **agent_steer's territory** and
  already work. Re-use via imports.

**Files that are OUT OF SCOPE — do NOT edit:**
- `agents/agent_*/` — every other agent's directory **including
  `agents/agent_steer/`**. agent_steer's briefing, decisions, and
  per-cell state are theirs. You read for context, you do not write.
- `experiments/c5_steering/` — agent_steer's territory. Their `run.py`,
  `analysis.py`, `_real_training_cfg`, `select_best_features`, etc.
  Import from here without modification.
- `experiments/c5_steering_100k/` — your OLD driver from the abandoned
  100K mission. Leave it as-is; agent_paper may delete or repurpose it
  later.
- `docs/components/c5.md` — agent_steer's territory.
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

### ⚠️ Mission pivot 2026-05-05 — abandon 100K, deploy MW at canonical 20K

**Old mission (abandoned)**: replicate agent_steer's C5 sweep at
`n_steps=100_000` to test convergence sensitivity. **You reported the
ETA as 108 hours** for the full 3-arch × 3-seed sweep — does not fit
in the remaining sprint window, and agent_steer caught a critical bug
that invalidated all v1.0.0 eval cells anyway (concept-lift feature
selection, fixed in commit `ef33f822`, EVAL_PROTOCOL_VERSION
1.0.0 → 1.1.0).

Combined with agent_paper landing the multi-window TXC archs
(`txc_base_mw`, `txc_pro_mw` in `configs/locked_archs.yaml` —
decisions.md § 14), the 100K compute budget is far better spent on
deploying the multi-window fix at the canonical schedule.

**New mission**: run **`txc_base_mw + txc_pro_mw` × 3 seeds at C5's
canonical 20K schedule** with the v1.1.0 concept-lift fix.

The hypothesis (Han 2026-05-05): **MW at 20K achieves the
training-data-volume effect of non-MW at ~100K, in ~1/20 the
wall-time**. If true, this is the cleanest case for the multi-window
deployment — a side-by-side comparison vs agent_steer's existing
non-MW 20K v1.1.0 cells.

Token-volume math justifying the hypothesis:

| Configuration | Tokens/step | n_steps | Total tokens |
|---|---:|---:|---:|
| Non-MW 20K (agent_steer canonical) | ~5K | 20K | ~100M |
| Non-MW 100K (the abandoned mission) | ~5K | 100K | ~500M |
| **MW 20K (this mission)** | **~50K** | **20K** | **~1B** |

So MW 20K actually exceeds the non-MW 100K token budget — at ~5 hr
wall vs the abandoned 108 hr. If the convergence story holds, this
is the right C5 result for the paper headline (and supersedes
agent_steer's non-MW 20K v1.1.0 cells in AUTO-RESULTS once both
sweeps are in the leaderboard, with within-component fairness
preserved because the headline filter pins to one canonical config).

**Existing 100K work — what survives, what's discarded**:

- **100K training checkpoints** (whatever you've completed): kept on
  HF `han1823123123/temp-bench-models`. The framework's training
  cache works — if anyone ever wants to do the convergence-sensitivity
  test at v1.1.0, they re-eval those checkpoints (~1 hr, training
  cached).
- **100K v1.0.0 eval cells**: stay in `leaderboard.jsonl`. They're
  buggy (concept-lift bug) and stale; v1.1.0 protocol filter excludes
  them automatically.
- **`experiments/c5_steering_100k/run.py`**: leave it for now.
  agent_paper may repurpose later if we ever want the convergence
  test.
- **In-flight 100K processes**: KILL them on this session start. Free
  up the H100 for the MW sweep.

### Mandate — same C5 sweep, multi-window archs, canonical schedule

Hardware: **1× H100 80GB pod, ephemeral, 240 GB system RAM, 1 TB
/workspace**. Pinned to GPU 0 (the only GPU). Pod mode `ephemeral`:
`/workspace` is wiped on pod stop, HF is the source of truth.

VRAM check (per agent_paper's analysis 2026-05-05): MW at C5's
Gemma-scale dims (d_in=2304, d_sae=18432) gives peak activation memory
~25-50 GB on the encoder/decoder pass — well within the 80 GB H100
budget. No mitigations needed; you don't need bf16 forcing or batch
reductions like agent_back does at A40-scale.

Subject + protocol (replicating agent_steer's setup verbatim, with
the multi-window arch swap):

- Datasource: `gemma_2_2b_it_l13_fineweb_24k128`
- Architectures: **`txc_base_mw` + `txc_pro_mw`** (2 archs total).
  These are YAML aliases of TXCBase / TXCPro with `multi_window: true`
  in hparams (decisions.md § 14). The Python classes are identical
  to `txc_base` / `txc_pro`; only the per-step sampling differs.
  **NOT `tsae_paper`** — agent_steer's existing v1.1.0 tsae_paper
  cells are the canonical T-SAE comparison; running tsae_paper a
  third time is wasted compute (no MW-equivalent for non-TXC archs).
- Per-component d_sae overrides: locked_archs.yaml's
  `per_component_hparams.c5` for both `txc_base_mw` and `txc_pro_mw`
  applies automatically (mirrored from `txc_base` / `txc_pro`).
- Steering: V7 tiled-broadcast residual-stream protocol (per
  `temp_bench.case_studies.steering` — agent_steer's port). Same
  strengths grid `{10, 100, 150, 500, 1000, 1500, 5000, 10000, 15000}`,
  same `"We find"` prompt, 60 new tokens, greedy decode.
- Concept set: same 30-concept × 5-example-sentence set agent_steer
  uses (5 safety/alignment, 10 domain, 7 style, 5 sentiment, 3 format).
- **Per-arch best-feature selection**: USE THE FIXED v1.1.0
  concept-lift baseline (commit `ef33f822`, agent_steer's
  `select_best_features` in `temp_bench.case_studies.steering`). Your
  import path automatically picks this up. Verify by inspecting the
  function before launching:
  `grep "baseline = activation_matrix.mean" src/temp_bench/case_studies/steering.py`.
- Judge: Sonnet 4.6 (Anthropic; same as agent_steer + agent_em — see
  decisions.md § 12 for why we don't use Gemini).
- Per-call `judge_outputs.jsonl` persistence for post-deadline κ.
- Headline metric: `peak_success_grade_at_coh_1.75`.
- `EVAL_PROTOCOL_VERSION`: **"1.1.0"** (the post-fix version).

Seeds: **{42, 1, 2}** (full n=3). 6 cells total.

`TrainingConfig` for your cells (canonical schedule — NOT 100K):

```python
TrainingConfig(
    batch_size=1024,
    n_steps=20_000,         # canonical, NOT 100_000
    plateau_early_stop=False,
    # bricken_* stays at defaults (False) — C5 does not use Bricken
    # per decisions.md § 7.
)
```

Per-cell wall-time estimate on H100: agent_paper's local 5090
benchmark of TXCPro multi-window vs non-MW (commit `d724241c`)
showed wall-time only 1.26× longer at 5× more data per step on a
synthetic task. At C5's actual scale + 20K steps, expect:

- `txc_base_mw`: ~30-50 min per cell (5K-step extrapolation × 4)
- `txc_pro_mw`: ~50-90 min per cell (matryoshka + multi-distance
  contrastive + multi-window all together)

Total sweep at H100 single-GPU serial: 6 cells × ~60 min average =
**~6 hours wall**. Comfortably within the remaining sprint window
even with margin for surprises.

V7 ↔ TXC-pro compatibility (sanity check before launch): agent_steer's
non-MW txc_pro at 20K showed mean coh ~2.1-2.2 at v1.0.0 (pre-bug-fix
metric was misleading but the steering itself worked). MW shouldn't
change V7 compatibility — encode/decode shapes are unchanged at
inference, only train_step's data sampling differs. Run `--pre-test-only`
on your first txc_pro_mw cell anyway as a safety net; if mean coh ≤ 1.0,
fall back to `--protocol pp` for the full sweep.

Locked decisions in scope: #1 (two TXCs — `txc_base_mw` and
`txc_pro_mw` are the canonical paper TXCs going forward per § 1's
2026-05-05 amendment), #4 (cross-branch reads), #6 (HF repos),
#7 (Bricken off for C5), #11 (T-SAE = paper-faithful Ye et al.),
§ 12 (uniform batch=1024, plateau_off, n_steps=20K for C5),
§ 14 (multi-window deployment).

References:
- `agents/README.md` (your roster row)
- `agents/agent_steer/briefing.md` (the canonical C5 setup you replicate
  — read this BEFORE launching, especially their § "Han decisions
  2026-05-05 — 🐛 c5 concept-lift fix" once they document it)
- `docs/components/c5.md` (the canonical C5 writeup; do NOT edit)
- `experiments/c5_steering/{run.py,analysis.py}` (import from)
- `decisions.md` § 7, § 12, § 14
- `papers/temporal_sae.md` § B.2 (T-SAE steering protocol reference)
- `PROTOCOL.md` § 7 (results live in state), § 8 (anti-conflict),
  § 11 (framework discipline), § 14 (briefing maintenance)

**Before you `status: complete`**: `bash scripts/wrap_up_session.sh`.
Ephemeral pod → checkpoints auto-pushed. Push your MW cells'
`results/runs/<eval_key>/judge_outputs.jsonl` and metrics via the
wrap-up script before any pod restart.

### First concrete task — kill 100K, write the MW driver, launch

Step 0 — **kill the in-flight 100K processes** before doing anything
else (free the GPU + RAM):

```bash
ps -ef | grep "experiments.c5_steering_100k" | grep -v grep
# → kill the PIDs from your /tmp/p_* files OR ps output above
kill $(cat /tmp/p_gpu1 /tmp/p_gpu3 2>/dev/null) 2>/dev/null
nvidia-smi --query-gpu=memory.used --format=csv
# → expect <500 MB used; if not, force-kill stragglers via `kill -9`
```

Step 1 — `git pull --rebase origin final`, then verify the v1.1.0 fix
is present:

```bash
grep "baseline = activation_matrix.mean" src/temp_bench/case_studies/steering.py
# → should match (added in agent_steer's commit ef33f822)
.venv/bin/python -c "from experiments.c5_steering.run import EVAL_PROTOCOL_VERSION; print(EVAL_PROTOCOL_VERSION)"
# → expect "1.1.0"
.venv/bin/python -c "from temp_bench.config import load_arch; print(load_arch('txc_base_mw').hparams)"
# → expect a dict containing multi_window=True
```

Step 2 — write `experiments/c5_steering_mw/__init__.py` (empty) and
`experiments/c5_steering_mw/run.py`:

```python
"""C5 multi-window deployment driver.

Replicates agent_steer's setup verbatim, swapping the TXC archs from
`txc_base` / `txc_pro` to the multi-window aliases `txc_base_mw` /
`txc_pro_mw`. Same canonical schedule (batch=1024, n_steps=20_000,
plateau_off), same V7 steering protocol, same Sonnet judge, same
v1.1.0 EVAL_PROTOCOL_VERSION (concept-lift baseline fix).
"""
from __future__ import annotations
import argparse
from temp_bench import runner
from temp_bench.schemas import TrainingConfig
from experiments.c5_steering.run import (
    DATASOURCE,
    EVAL_PROTOCOL_VERSION,        # "1.1.0" after agent_steer's ef33f822
    my_train_fn,
    my_eval_fn,
    _real_training_cfg as _orig_training_cfg,
)


def _real_training_cfg() -> TrainingConfig:
    """Canonical C5 schedule — NOT a 100K override. Inherits from
    agent_steer's _real_training_cfg() to stay automatically in sync
    if they tweak it."""
    return _orig_training_cfg()    # batch=1024, n_steps=20_000, plateau_off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+",
                    default=["txc_base_mw", "txc_pro_mw"],
                    choices=["txc_base_mw", "txc_pro_mw"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    args = ap.parse_args()

    cfg = _real_training_cfg()
    print(f"[c5_mw] training_cfg n_steps={cfg.n_steps} "
          f"batch_size={cfg.batch_size} plateau_off={not cfg.plateau_early_stop}")

    for arch in args.archs:
        for seed in args.seeds:
            print(f"[c5_mw] launching cell arch={arch} seed={seed} "
                  f"eval_protocol_version={EVAL_PROTOCOL_VERSION}")
            runner.run_cell(
                component="c5",
                arch_name=arch,
                seed=seed,
                datasource_name=DATASOURCE,
                training_cfg=cfg,
                eval_cfg={"sweep": "c5_mw_v1"},
                eval_protocol_version=EVAL_PROTOCOL_VERSION,
                train_fn=my_train_fn,
                eval_fn=my_eval_fn,
            )


if __name__ == "__main__":
    main()
```

Step 3 — smoke-test with 1 cell at a tiny n_steps override, BEFORE
launching the full 6-cell sweep, to verify:
- `txc_base_mw` instantiates correctly (multi_window=True flows through).
- `train_step` doesn't OOM at C5's d_in=2304 / d_sae=18432.
- The eval side picks up the v1.1.0 fix.

```bash
# Smoke: 200 steps × 1 cell × 1 seed (~5 min)
TQDM_DISABLE=1 .venv/bin/python -c "
from temp_bench.schemas import TrainingConfig
from temp_bench import runner
from experiments.c5_steering.run import DATASOURCE, EVAL_PROTOCOL_VERSION, my_train_fn, my_eval_fn
result = runner.run_cell(
    component='c5', arch_name='txc_base_mw', seed=42,
    datasource_name=DATASOURCE,
    training_cfg=TrainingConfig(n_steps=200, batch_size=1024, plateau_early_stop=False),
    eval_cfg={'sweep': 'c5_mw_smoke'},
    eval_protocol_version=EVAL_PROTOCOL_VERSION,
    train_fn=my_train_fn, eval_fn=my_eval_fn,
)
print('smoke result:', result.train_key, result.eval_key, result.cached)
"
```

If smoke passes, proceed to step 4. If it OOMs or fails, surface the
error in Open questions and ping Han before the full sweep.

Step 4 — launch the full 6-cell sweep:

```bash
TQDM_DISABLE=1 AGENT_NAME=agent_steer_100k \
  .venv/bin/python -m experiments.c5_steering_mw.run \
  --archs txc_base_mw txc_pro_mw --seeds 42 1 2 \
  > logs/c5_mw_full.log 2>&1 &
echo $! > /tmp/p_c5_mw
```

Step 5 — monitor via `Monitor` tool or periodic `tail`. Each cell's
completion appends a row to `results/leaderboard.jsonl` with
`arch=txc_base_mw` or `txc_pro_mw`, `eval_protocol_version=1.1.0`,
and the headline `peak_success_grade_at_coh_1.75`.

agent_paper integrates results at paper-render time — your cells just
need to land in `leaderboard.jsonl` with the right
`(arch, seed, training_cfg, eval_protocol_version)`. The runner
handles the rest.

---

## Current state (agent owns — overwrite at every compact)

**Last verified: 2026-05-05T11:00Z (mission pivot — directive received from
Han / agent_paper). 100K mission abandoned (108 hr ETA). New mission:
deploy multi-window TXCs at canonical 20K schedule.**

- `git HEAD`: at or after `cad94382` (decisions.md § 14 Bricken caveat).
  Pull on session start to pick up agent_steer's `ef33f822` v1.1.0 fix.
- Pod: 1× H100, ephemeral, 240 GB RAM. `/workspace/temp_xc/` clone.
- In flight (TO BE KILLED on session start): 100K cells from the
  abandoned mission. Identify via `ps -ef | grep c5_steering_100k`
  and the `/tmp/p_*` PID files.
- 100K checkpoints already trained: kept on HF; not used in this mission
  but available for post-paper convergence-test re-eval.
- Last leaderboard append: from the 100K mission (v1.0.0 buggy cells —
  filter excludes).
- Recent decisions in scope: `decisions.md` § 1 (canonical TXCs are
  `txc_base_mw` / `txc_pro_mw` going forward), § 12 (canonical training
  cfg), § 14 (multi-window deployment), § 7 (Bricken off for C5).

## What I just did (agent owns — overwrite)

(Pivot — overwrite this section with your own actions when you start.)

- 2026-05-05T11:00Z: agent_paper rewrote this briefing per Han's
  pivot directive. 100K mission abandoned; new mission is C5 MW at
  canonical 20K schedule (6 cells, ~6 hr wall).

## Next action (agent owns — overwrite)

1. `cd $(git rev-parse --show-toplevel)/purified`
2. `source scripts/set_agent_env.sh agent_steer_100k`
3. `bash scripts/agent_smoke_test.sh` — expect 131/131 + preflight green.
4. `git pull --rebase origin final` — picks up agent_steer's v1.1.0
   concept-lift fix + agent_paper's `txc_base_mw` / `txc_pro_mw` YAML
   aliases.
5. **Kill in-flight 100K processes** per "First concrete task" Step 0.
6. Verify v1.1.0 fix + MW arch registration per Step 1.
7. Write `experiments/c5_steering_mw/run.py` per Step 2.
8. Smoke-test at n_steps=200 per Step 3.
9. Launch the full 6-cell sweep per Step 4.
10. Monitor + verify leaderboard rows land at `eval_protocol_version=1.1.0`.

## Don't repeat (agent owns — overwrite)

- **Don't run anything at `n_steps=100_000`** for this mission. The
  100K convergence test is abandoned per Han 2026-05-05; canonical
  schedule (n_steps=20_000) is the only target.
- **Don't edit `experiments/c5_steering/`** — agent_steer's territory.
  Import, don't modify. The v1.1.0 concept-lift fix in their
  `select_best_features` is the version you inherit.
- **Don't include `tsae_paper` in your archs list** — agent_steer's
  existing v1.1.0 tsae_paper cells are the canonical T-SAE comparison;
  running it again here is wasted compute (no MW variant exists for
  non-TXC archs).
- **Don't edit `docs/components/c5.md`** — agent_paper integrates at
  paper-render time.
- **Don't bypass `runner.run_cell`** — the call goes through the
  canonical pathway (which appends to `leaderboard.jsonl`).
- **Don't allocate `train_key` / `eval_key` manually** — the runner
  computes them from your inputs deterministically.
- **Don't enable Bricken** — C5 is Bricken-off per decisions.md § 7.
- **Don't pursue the Y/W steering hill-climb winners** — Galaxy 8/11/18
  / SoftMaxPool / ContrastiveMergeH8 are excluded by decision #1.
- **Don't push to HF manually** — `cache.save_checkpoint` does it on
  ephemeral pods.

## Open questions for Han (agent owns — overwrite)

(None at briefing-rewrite time. Surface anything that comes up during
the kill-100K step or smoke test.)
