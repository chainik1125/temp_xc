<!--
DRAFT — written by agent_paper 2026-05-04 PM. Han populates the rest
of "Identity + mandate" if any priorities shift. Section ownership
rules: PROTOCOL.md § 14.
-->

---
agent: agent_steer_100k
last_state_update: 2026-05-04T16:30:00Z
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

**Last verified: <fill in on first session>**

- `git HEAD`: <sha>
- Pod: 1× H100, ephemeral, 240 GB RAM. `/workspace/temp_xc/` clone.
- Last leaderboard append: `(none yet)`.
- Last checkpoint saved: `(none yet)`.
- Active GPU usage: GPU 0 (only GPU on this pod).
- Recent decisions in scope: `decisions.md` § 7 (Bricken off for C5),
  § 12 (canonical training cfg), § 13 (100K copy-sweep policy).
- In flight: `(nothing yet — first session)`.

## What I just did (agent owns — overwrite)

(none yet — first session)

## Next action (agent owns — overwrite)

1. `cd $(git rev-parse --show-toplevel)/purified`
2. `source scripts/set_agent_env.sh agent_steer_100k`
3. `bash scripts/agent_smoke_test.sh` — expect 124/124 + preflight
   green.
4. `bash scripts/sync_from_hf.sh` — pulls Gemma activation cache
   (act_cache_key for `gemma_2_2b_it_l13_fineweb_24k128`) +
   agent_steer's 20K checkpoints (good for sanity diff at the end).
5. `git pull --rebase origin final` — stay current with agent_steer's
   recent commits.
6. **Write `experiments/c5_steering_100k/run.py` + empty `__init__.py`**
   per the sketch above.
7. **Smoke-test the driver** with `--archs tsae_paper --seeds 42`
   first (smallest model, fastest) and a `n_steps=200` patch before
   committing the 100K cells. Verify the cell hits agent_steer's
   `my_train_fn`, `my_eval_fn`, V7 steering, Sonnet judge.
8. Launch the real 100K cells (full sweep):
   ```bash
   TQDM_DISABLE=1 .venv/bin/python -m experiments.c5_steering_100k.run \
     --archs tsae_paper txc_base txc_pro --seeds 42 1 2 \
     > logs/c5_100k_full.log 2>&1 &
   ```
9. Monitor via `Monitor` tool or periodic `tail` on the log.
10. As cells complete: confirm new leaderboard rows landed
    (`tail -1 results/leaderboard.jsonl`). Don't render anything to
    c5.md yourself — agent_paper handles that at paper-render time.
11. If txc_pro becomes the long pole, drop seed=2 last and surface
    in Open Questions.

## Don't repeat (agent owns — overwrite)

- **Don't edit `experiments/c5_steering/`** — agent_steer's territory.
  Import, don't modify.
- **Don't edit `docs/components/c5.md`** — agent_paper integrates at
  paper-render time.
- **Don't bypass `runner.run_cell`** — even though you're calling it
  from a custom driver, the call itself goes through the canonical
  pathway (which appends to `leaderboard.jsonl`).
- **Don't allocate `train_key` / `eval_key` manually** — the runner
  computes them from your inputs deterministically.
- **Don't enable Bricken** — C5 is Bricken-off per decisions.md § 7.
  Trust agent_steer's `_real_training_cfg()` defaults.
- **Don't pursue the Y/W steering hill-climb winners** — Galaxy 8/11/18
  / SoftMaxPool / ContrastiveMergeH8 are excluded by decision #1.
  Stick to the locked TXC-base + TXC-pro.
- **Don't push to HF manually** — `cache.save_checkpoint` does it on
  ephemeral pods. Verify via the URL in the manifest after each cell.

## Open questions for Han (agent owns — overwrite)

(None at briefing-write time. Surface anything that comes up during
the first cell's run.)
