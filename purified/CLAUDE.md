# CLAUDE.md — purified/ subtree

You are an AI agent working on the paper-ready `final` branch. **All work
happens inside `purified/`.** The root-level `temp_xc/{src,experiments,docs}`
is the "wasteland" — read it for context, never import or modify it.

## First actions on every session

1. Read `purified/PROTOCOL.md` (operational rules).
2. Read your agent dir's `briefing.md` (your specific mandate).
3. Read `docs/components/c{N}.md` for any component you'll touch.
4. Skim the last 5 entries of `results/leaderboard.jsonl` to see what
   other agents have just produced.

## Hard rules

1. **Never import from `temp_xc/{src,experiments,docs}`.** If you need
   reference code, copy it into `purified/src/temp_bench/` (duplication
   is fine; coupling is not).
2. **Never edit `purified/agents/<other_agent>/`** — those are owned by
   other agents.
3. **Use `purified/.venv`** built from `purified/pyproject.toml`. Run
   `uv sync` from inside `purified/`. Do not use the root `temp_xc/.venv`.
4. **Always set `TQDM_DISABLE=1`** before any Python invocation.
5. **The two TXC architectures are locked**: `TXC-base` =
   `txc_bare_antidead_t5`; `TXC-pro` = `phase5b_subseq_h8`. Do not
   introduce a third TXC variant. Sparsity (k) and dictionary size
   (d_sae) are the only free parameters per component.
6. **Writeups go in `docs/components/c{N}.md`**, not in agent dirs.
   Agent dirs hold your *briefing* and *log* only — ephemeral state.

## Hardware quotas

| Pod | Agents | Note |
|---|---|---|
| Local 5090 (32GB) | Agent PAPER | Orchestrator + C1 + C2 + paper drafting |
| 2× H100 RunPod | Agent NLP, Agent EM | NLP=C3+C4 (shared cache); EM=C6 (Qwen-14B) |
| 3× A40 RunPod | Agent STEER, Agent BACK, optional 3rd | STEER=C5; BACK=C7 |
| H200 | reserve | Only for EM if R32 organism blows H100 mem |

## How to record results

**Use `temp_bench.runner.run_cell`. Don't write your own caching or
leaderboard logic.** See `docs/paper/framework.md` for the design.

```python
from temp_bench import runner
from temp_bench.schemas import TrainingConfig

result = runner.run_cell(
    component="c3",
    arch_name="txc_base",
    seed=42,
    datasource_name="gemma_2_2b_it_l13_fineweb_24k128",
    training_cfg=TrainingConfig(),       # or override fields
    eval_cfg={"k_feat": 20, "S": 32},
    eval_protocol_version="1.0.0",       # bump on metric change
    train_fn=my_train_fn,                 # in temp_bench.training
    eval_fn=my_eval_fn,                   # in temp_bench.eval.probing
)
# result.eval_key, result.train_key, result.cached
```

The runner:
- Computes deterministic ``train_key`` and ``eval_key`` from inputs.
- Skips training if a checkpoint with that ``train_key`` exists.
- Skips evaluation if that ``eval_key`` is in ``leaderboard.jsonl``.
- Saves the trained checkpoint, run-dir metrics, and one validated row
  in the leaderboard. Schema-rejected rows abort the cell.

A run-dir is created at ``results/runs/<eval_key>/`` containing
``metrics.json`` and any plots (``plots/*.png`` + ``*.thumb.png``,
saved via ``temp_bench.plotting.save_figure``).

## How to record checkpoints

The runner already saves the trained checkpoint to
``checkpoints/<train_key>/`` and appends a validated row to
``checkpoints/manifest.jsonl``. Your only extra step is HF backup at
session end:

```python
from huggingface_hub import HfApi
from temp_bench.config import checkpoint_dir

train_key = result.train_key
HfApi().upload_folder(
    folder_path=str(checkpoint_dir(train_key)),
    path_in_repo=train_key,
    repo_id="han1823123123/temp-bench-models",
    repo_type="model",
)
```

Both repos are private:
- `han1823123123/temp-bench-models` — checkpoints, keyed by ``<train_key>``
- `han1823123123/temp-bench-data` — activation caches, judge transcripts

See `checkpoints/README.md` for the full helper recipe.

## Markdown style

Same as the root CLAUDE.md: ATX headings, no H1, dash bullets, fenced code
blocks with language, YAML frontmatter (author/date/tags) on `docs/`.

## Quick reference

```bash
# bootstrap (RunPod, idempotent)
cd /workspace/temp_xc/purified && bash scripts/bootstrap_runpod.sh

# verify environment + run tests + preflight
cd purified && bash scripts/agent_smoke_test.sh

# run any component (idempotent — safe to re-run)
cd purified && TQDM_DISABLE=1 .venv/bin/python -m experiments.c1_synthetic_topk.run

# add a new architecture
# 1. drop a class in src/temp_bench/architectures/<name>.py
# 2. add an entry to configs/locked_archs.yaml
# 3. re-run any component — only the new arch's cells compute
```
