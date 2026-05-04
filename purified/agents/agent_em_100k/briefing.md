<!--
DRAFT — written by agent_paper 2026-05-04 PM. Han populates the rest
of "Identity + mandate" if any priorities shift. Section ownership
rules: PROTOCOL.md § 14.
-->

---
agent: agent_em_100k
last_state_update: 2026-05-04T16:30:00Z
component: c6
---

## Identity + mandate (Han owns — agents do not edit)

You are **agent EM 100K**. You are a literal copy of agent_em — same
component (C6), same datasource (Qwen-2.5-14B-Instruct + finance LoRA,
L24 resid_post), same archs (sae_arditi + txc_base with the
brickenauxk_a8 recipe), same Wang full-protocol, same Sonnet judge,
same metric — **with one and only one difference: `n_steps=100_000`
instead of agent_em's `n_steps=25_000`.** Your cells are intended to
be the better-trained version of agent_em's; if they finish in time,
they become the C6 paper headline (replacing the 25K cells).

Files you may edit:
- `agents/agent_em_100k/briefing.md` (your own — agent-owned sections only)
- `experiments/c6_em_100k/` (new experiment directory you create with
  a minimal driver that imports agent_em's plumbing — see "First
  concrete task" below).
- Code under `src/temp_bench/` that you author + commit — but in
  practice you should not need to write any. The C6 case-study code,
  Wang procedure, Bricken trainer, Qwen activation cache loader, and
  per-arch overrides are **agent_em's territory** and already work.
  Re-use them via imports; do not modify them.

**Files that are OUT OF SCOPE — do NOT edit:**
- `agents/agent_*/` — every other agent's directory **including
  `agents/agent_em/`**. agent_em's briefing, decisions, and per-cell
  state are theirs. You read for context, you do not write.
- `experiments/c6_em/` — agent_em's territory. Their `train.py`,
  `run.py`, `analysis.py`, `make_training_cfg`, etc. You import from
  here without modification.
- `docs/components/c6.md` — agent_paper integrates the headline (yours
  vs agent_em's, whichever lands) into AUTO-RESULTS at paper time.
  You don't touch this directly. Neither does agent_em.
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

### Mandate — same C6 sweep, longer training

The paper-canonical C6 sweep (agent_em) runs at `batch=1024`,
`n_steps=25_000` (~25.6M activation tokens per arch) — chosen because
that's what fit in the 72-hour sprint window. The published-SAE-paper
budgets are higher (T-SAE: ~4-8B; TFA: 1B; Phase 7: ~100M tokens).
You have the compute to do better — **a fresh single-GPU H100 pod
dedicated to this**. Run the same sweep at `n_steps=100_000` (~102M
tokens, comfortably in the field-standard range) and ship those cells
to the leaderboard.

**Whichever sweep completes first becomes the C6 paper headline:**

- If your 100K sweep finishes before the deadline: agent_paper picks
  your cells as canonical, agent_em's 25K cells become the
  "compute-pressure backup" reference. Within-component fairness is
  preserved — every C6 arch is at 100K.
- If your 100K sweep is mid-flight at the deadline: agent_em's 25K
  cells stay canonical. Your partial 100K cells are kept in the
  leaderboard for a "convergence consistency" caveat.
- Cells from both sweeps coexist cleanly in `leaderboard.jsonl` — the
  `train_key` hash includes `n_steps`, so the 25K and 100K cells
  occupy distinct keys and don't collide. agent_em's `analysis.py`
  uses `canonical_train_keys()` with the `TrainingConfig()` that's
  current at render time; agent_paper toggles which sweep is canonical
  by which `n_steps` the analysis filter pins.

Hardware: **1× H100 80GB pod, ephemeral, 240 GB system RAM, 1 TB
/workspace**. Pinned to GPU 0 (the only GPU). Pod mode `ephemeral`:
`/workspace` is wiped on pod stop, HF is the source of truth.
Bootstrap pulls from `han1823123123/temp-bench-{models,data}`;
`cache.save_checkpoint` auto-pushes on save (push failure is fatal).

The 240 GB RAM means agent_nlp's preloaded `.clone()` pattern (commit
`e12dc719`) is unconstrained — preload the full Qwen-14B finance
activation cache (~31 GB at 24K seqs × 128 tokens × 5120 d_in fp16)
into RAM once, no headroom worries. agent_em already adopted this
pattern in `experiments/c6_em/train.py` (commit `6269f4d2`); you
inherit it via the same import.

Subject + organism (replicating agent_em's setup verbatim):

- Datasource: `qwen_2_5_14b_instruct_finance_l24_resid_post`
- Architectures: `sae_arditi` (no Bricken) + `txc_base` with the
  brickenauxk_a8 recipe (`bricken_enabled=True`, `auxk_alpha=1/8`,
  `dead_threshold_tokens=128_000`). agent_em wires the brickenauxk_a8
  override via `experiments/c6_em/train.py:_instantiate_with_overrides`
  + `make_training_cfg` — you import + reuse, don't reimplement.
- Per-component `d_sae=32768, k_pos=25` for txc_base (already in
  `configs/locked_archs.yaml`'s `per_component_hparams.c6`).
- Wang procedure: full 4-stage protocol (`temp_bench.case_studies.em`
  — agent_em's port). Same 100-feature Δz̄ ranking, same causal screen
  at α=±1, same per-survivor strength sweep, same per-feat α frontier.
- Judge: Sonnet 4.6 (Anthropic; same as agent_em — see decisions.md
  § 12 for why we don't use Gemini).
- Per-cell `judge_outputs.jsonl` persistence for post-deadline κ.

Seeds: **{42, 1}** matching agent_em's reduced n=2 sweep
(decisions.md § 12 update — seed=2 dropped under compute pressure).
If you finish seed=42 with margin, run seed=1; if both finish with
margin, restore seed=2 for the dropped agent_em entry.

`TrainingConfig` for your cells:

```python
TrainingConfig(
    batch_size=1024,
    n_steps=100_000,        # <-- the only difference from agent_em
    plateau_early_stop=False,
    bricken_enabled=True,   # for txc_base; sae_arditi sets this False
    bricken_resample_every=500,
    bricken_min_fires=1,
    bricken_n_check=2048,
    bricken_max_resample_fraction=0.5,
    ema_auxk_alpha=0.125,             # 1/8 per a8 recipe
    dead_threshold_tokens=128_000,    # 128k tokens per a8 recipe
)
```

agent_em's `experiments/c6_em/train.py:make_training_cfg` already
builds this dict for you; you just override `n_steps=100_000` in the
driver script (see "First concrete task").

Per-cell wall-time estimate: SAE training (no Bricken) at 100K =
~56 min on H100 (4× the 14 min agent_em saw at 25K). TXC + Bricken
similar, possibly +30% from the resample overhead. Wang full =
~2.85 hr/cell unchanged. Per-cell wall ≈ 4 hr. Both archs × 1 seed
= 2 cells × 4 hr ≈ **8 hr total**. Both archs × 2 seeds = ~16 hr.
With ~30 hr remaining in the sprint, two-seed coverage is feasible
if you start now.

Locked decisions in scope: #1 (two TXCs — TXC-base only here, no
TXC-pro in C6 per decisions.md § 7), #6 (HF repos), #7 (Bricken on by
default for C6), #11 (T-SAE = paper-faithful Ye et al. — but C6 uses
sae_arditi as the SAE baseline, not T-SAE), § 12 (uniform batch=1024,
plateau_off — you keep these; only n_steps differs), § 13 (the 100K
copy-sweep policy).

References:
- `agents/README.md` (your roster row)
- `agents/agent_em/briefing.md` (the canonical C6 setup you replicate
  — read this before launching anything)
- `docs/components/c6.md` (the canonical C6 writeup; do NOT edit)
- `experiments/c6_em/{train.py,run.py,analysis.py}` (import from)
- `decisions.md` § 7, § 12, § 13
- `papers/temporal_sae.md` (T-SAE reference for context)
- `PROTOCOL.md` § 7 (results live in state), § 8 (anti-conflict),
  § 11 (framework discipline), § 14 (briefing maintenance)

**Before you `status: complete`**: `bash scripts/wrap_up_session.sh`.
Ephemeral pod → checkpoints auto-pushed. Push your 100K cells'
`results/runs/<eval_key>/` artifacts via the wrap-up script before
any pod restart.

### First concrete task — write a minimal driver script

Create `experiments/c6_em_100k/run.py` (the experiments dir is on
PYTHONPATH via `experiments/__init__.py`, which makes the import
paths cleaner) and an empty `experiments/c6_em_100k/__init__.py`:

```python
"""C6 driver — replicates agent_em's setup at n_steps=100_000.

Imports agent_em's train_fn / eval_fn / Wang infrastructure from
experiments.c6_em.* without modification; only n_steps in the
TrainingConfig differs.
"""
from __future__ import annotations
import argparse
from temp_bench import runner
from temp_bench.schemas import TrainingConfig

# Re-use agent_em's plumbing verbatim:
from experiments.c6_em.train import make_training_cfg
from experiments.c6_em.run import (
    DATASOURCE,
    EVAL_PROTOCOL_VERSION,
    my_train_fn,
    my_eval_fn,
)


def _real_training_cfg(arch_name: str) -> TrainingConfig:
    """100K override of agent_em's `make_training_cfg` for this arch."""
    base = make_training_cfg(arch_name)         # 25K, batch=1024, etc.
    return base.model_copy(update={"n_steps": 100_000})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+",
                    default=["sae_arditi", "txc_base"],
                    choices=["sae_arditi", "txc_base"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    args = ap.parse_args()

    for arch in args.archs:
        for seed in args.seeds:
            cfg = _real_training_cfg(arch)
            print(f"[c6_100k] launching cell arch={arch} seed={seed} "
                  f"n_steps={cfg.n_steps}")
            runner.run_cell(
                component="c6",
                arch_name=arch,
                seed=seed,
                datasource_name=DATASOURCE,
                training_cfg=cfg,
                eval_cfg={"sweep": "c6_100k_v1"},
                eval_protocol_version=EVAL_PROTOCOL_VERSION,
                train_fn=my_train_fn,
                eval_fn=my_eval_fn,
            )


if __name__ == "__main__":
    main()
```

Then run:
```bash
TQDM_DISABLE=1 .venv/bin/python -m experiments.c6_em_100k.run \
  --archs sae_arditi txc_base --seeds 42 \
  > logs/c6_100k_seed42.log 2>&1 &
```

When seed=42 cells finish, kick off seed=1 immediately (don't wait
for seed=2 stretch unless margin is comfortable).

agent_paper integrates results at paper-render time — your cells
just need to land in `leaderboard.jsonl` with the right
`(arch, seed, training_cfg)`. The runner handles the rest.

---

## Current state (agent owns — overwrite at every compact)

**Last verified: <fill in on first session>**

- `git HEAD`: <sha>
- Pod: 1× H100, ephemeral, 240 GB RAM. `/workspace/temp_xc/` clone.
- Last leaderboard append: `(none yet)`.
- Last checkpoint saved: `(none yet)`.
- Active GPU usage: GPU 0 (only GPU on this pod).
- Recent decisions in scope: `decisions.md` § 7 (Bricken-on for C6),
  § 12 (canonical training cfg), § 13 (100K copy-sweep policy).
- In flight: `(nothing yet — first session)`.

## What I just did (agent owns — overwrite)

(none yet — first session)

## Next action (agent owns — overwrite)

1. `cd $(git rev-parse --show-toplevel)/purified`
2. `source scripts/set_agent_env.sh agent_em_100k`
3. `bash scripts/agent_smoke_test.sh` — expect 124/124 + preflight
   green.
4. `bash scripts/sync_from_hf.sh` — pulls Qwen-14B activation cache
   (act_cache_key for `qwen_2_5_14b_instruct_finance_l24_resid_post`)
   + agent_em's 25K checkpoints (good for sanity diff at the end).
5. `git pull --rebase origin final` — stay current with agent_em's
   recent commits (especially the .clone() preload at `6269f4d2`).
6. **Write `experiments/c6_em_100k/run.py` + empty `__init__.py`** per
   the sketch above.
7. **Smoke-test the driver** with a tiny override (e.g., `--archs
   sae_arditi --seeds 42` and a `n_steps=200` patch) before launching
   the real 100K cells. Verify the cell hits agent_em's `my_train_fn`,
   `my_eval_fn`, Wang stages, judge calls.
8. Launch the real 100K cells:
   ```bash
   TQDM_DISABLE=1 .venv/bin/python -m experiments.c6_em_100k.run \
     --archs sae_arditi txc_base --seeds 42 \
     > logs/c6_100k_seed42.log 2>&1 &
   ```
9. Monitor via `Monitor` tool or periodic `tail` on the log.
10. As cells complete: confirm new leaderboard rows landed
    (`tail -1 results/leaderboard.jsonl`). Don't render anything to
    c6.md yourself — agent_paper handles that at paper-render time.
11. When seed=42 cells done: kick off seed=1 immediately. If both
    finish with margin, attempt seed=2.

## Don't repeat (agent owns — overwrite)

- **Don't edit `experiments/c6_em/`** — agent_em's territory. Import,
  don't modify.
- **Don't edit `docs/components/c6.md`** — agent_paper integrates at
  paper-render time.
- **Don't bypass `runner.run_cell`** — even though you're calling it
  from a custom driver, the call itself goes through the canonical
  pathway (which appends to `leaderboard.jsonl`).
- **Don't allocate `train_key` / `eval_key` manually** — the runner
  computes them from your inputs deterministically.
- **Don't change `bricken_*` defaults from the brickenauxk_a8 recipe**
  for txc_base — that's the recipe agent_em is using and we want a
  literal copy at higher n_steps, not a recipe change.
- **Don't run sae_arditi with `bricken_enabled=True`** — agent_em's
  `make_training_cfg("sae_arditi")` returns `bricken_enabled=False`
  for that arch. Trust the helper, don't override.
- **Don't push to HF manually** — `cache.save_checkpoint` does it on
  ephemeral pods. Verify via the URL in the manifest after each cell.

## Open questions for Han (agent owns — overwrite)

(None at briefing-write time. Surface anything that comes up during
the first cell's run.)
