<!--
DRAFT — written by agent_paper 2026-05-04 PM. Han populates the rest
of "Identity + mandate" if any priorities shift. Section ownership
rules: PROTOCOL.md § 14.
-->

---
agent: agent_em_100k
last_state_update: 2026-05-05T11:00:00Z
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

**Last verified: 2026-05-05T11:00Z. seed=42 cell 1 LANDED
(peak_align=82.11, +3.78 vs agent_em 25K headline 78.33). Cell 2
TXC Wang stage 4 in flight, ETA cell 2 done ~11:20Z. Sweep window
will not fit full seed=1 — TXC training rate is 5.9× slower than
SAE on this pod, see OQ #2.**

- `git HEAD`: `68ece86c` (`final`, pushed to origin).
- Pod: 1× H100 80GB, ephemeral. Actual RAM 2 TB (briefing said 240 GB).
- Active GPU usage: GPU 0, ~30-100% util (TXC Wang stage 4 generations).

### Cells landed in leaderboard

| arch | seed | train_key | eval_key | peak_align | peak_coh | α* |
|---|---|---|---|---|---|---|
| sae_arditi | 42 | `e5de419224108f98` | `397c345995d1acf2` | **82.11** | 90.88 | -10.0 |
| (txc_base seed=42 — pending) | 42 | `0884a29eabb0030d` | `155998b1fa5cee39` | … | … | … |

Headline reference (agent_em's 25K cells):
- sae_arditi 25K: peak_align=78.33 (`9778d10381696f58`)
- txc_base 25K:  peak_align=81.70 (`754166d1711923c1`)

### Training durations on this H100 pod

| arch | n_steps | wall-clock | steps/sec |
|---|---|---|---|
| sae_arditi (seed=42) | 100k | 52 min | ~32 |
| txc_base + Bricken (seed=42) | 100k | **5h 8min** | ~5.4 |

TXC was **5.9× slower per step** than SAE (vs briefing's "1.3×" guess).
Bricken fired 199 times, last n_resampled=16384 (50% cap hit).

### Per-stage Wang timing (cell 1 SAE)

- stage 1 (Δz̄ rank): 42 sec
- stage 2 (causal screen 100 features): 67.8 min
- stage 3 (sweep 20 survivors × 10 αs): 87.1 min
- stage 4 (frontier 3 finalists × 27 αs): 45.2 min
- Total Wang: ~3.3 hr; 14816 judge calls all 200 OK.

### In flight (PID 5260, re-launched 23:14:41Z 2026-05-04)

`.venv/bin/python -m experiments.c6_em_100k.run --archs sae_arditi
txc_base --seeds 42` → `logs/c6_100k_seed42_v2.log`. TXC Wang stages
1–3 done (12.3 min + 77.5 min + 91.5 min). Stage 4 started 10:36Z
(2026-05-05), ETA done ~11:21Z. Persistent monitor `b0gvkc8ij`.

### Activation cache + smoke artifacts

- Act cache `results/act_cache/e052801ef8e6d22b/` (7.86 GB fp16).
  Built fresh on this pod in 48 sec + auto-pushed to HF data repo.
- Smoke row in leaderboard: `train_key=29d23894a05bfc12`,
  `eval_key=55daa62002321413` (n_steps=200, peak_align=0.0). Benign
  noise — filtered by canonical_train_keys.

### Decisions in scope

- `decisions.md` § 7 (Bricken-on for C6 txc_base via brickenauxk_a8).
- § 12 (canonical training cfg: batch=1024, plateau_off — kept).
- § 13 (100K copy-sweep policy — agent_paper toggles canonical at
  paper-render time).

## What I just did (agent owns — overwrite)

1. Set up env, smoke test (124/124), built activation cache, wrote
   `experiments/c6_em_100k/{run.py,__init__.py}` driver, smoke test
   on sae_arditi/200-steps/skip-eval. Launched real seed=42 sweep.
   Commit `fe9bbe29` pushed.
2. **peft incident** (23:13Z 2026-05-04): SAE training completed
   cleanly (52 min, loss=126.5, ckpt e5de419224108f98 saved + pushed).
   Wang stage 1 then crashed on `from peft import PeftModel` —
   `peft` not in pyproject.toml. Workaround: `uv pip install peft`
   (peft 0.19.1) on this pod, no lockfile edit. Surfaced as **OQ #1**.
   Re-launched (PID 5260); runner cache-hit the SAE ckpt, went
   straight to Wang. Commit `70d89b9b` pushed.
3. **SAE seed=42 100K Wang complete** (02:35Z 2026-05-05, 3.3 hr):
   peak_align=82.11, peak_coh=90.88 at feat 26501 α=-10.0.
   eval_key=`397c345995d1acf2`. 14816 judge calls all 200 OK.
4. **TXC seed=42 100K trained** (07:44Z, 5h 8min) — train_key=
   `0884a29eabb0030d`, ckpt 6.3 GB pushed to HF. Bricken fired 199×
   (last n_resampled=16384). Final loss=293.9. Wang stages 1-3 done
   in 12.3+77.5+91.5 min; stage 4 started 10:36Z.
5. **Resolved 2-way merge conflict in origin/final** (Han pushed
   conflict markers in 6ea931e2's leaderboard.jsonl + manifest.jsonl
   from a HEAD vs 9044d487 merge). Cleaned via dedup-by-key (eval_key
   / train_key); 114 leaderboard + 57 manifest unique rows preserved
   from origin + my 1 row each. Commit `68ece86c` pushed.
6. Audited the API balance=0 user concern: 14816 judge calls across
   cell 1 Wang all returned 200 OK; no errors in logs. Cell 1 result
   sound. Training (no API calls) unaffected.
7. Verified agent_steer's C5 fix (`ef33f822`) is C5-only —
   touches `case_studies/steering.py` + `experiments/c5_steering/`,
   does NOT affect C6 Wang procedure (`case_studies/em.py`).

## Next action (agent owns — overwrite)

1. **Wait for cell 2 (TXC seed=42 100K Wang stage 4) — ETA ~11:20Z.**
   Persistent monitor `b0gvkc8ij` watches for stage-4 done +
   CELL DONE markers. Verify
   `grep "agent_em_100k" results/leaderboard.jsonl | tail -1 | jq`
   shows `eval_key: 155998b1fa5cee39`, `train_key: 0884a29eabb0030d`,
   `peak_align > 70` (expected — agent_em's 25K txc_base headline
   was 81.70).
2. **After cell 2 lands**: commit + push the new leaderboard row.
   Then update briefing's Cells-landed table with txc_base headline.
3. **Decide seed=1 strategy** — TXC training takes 5h+ on this pod
   (vs briefing's 1.3 hr estimate). Sprint window ends ~22:30Z May 5.
   - **Recommend: SAE seed=1 only** (~52 min train + ~3.3 hr Wang =
     ~4 hr; finish ~15:30Z if launched right after cell 2 lands).
     Gives partial within-arch n=2 for sae_arditi at 100K.
   - **Skip TXC seed=1** (would push to 24:00Z, past deadline).
   - Launch with:
     ```bash
     TQDM_DISABLE=1 .venv/bin/python -m experiments.c6_em_100k.run \
       --archs sae_arditi --seeds 1 \
       > logs/c6_100k_seed1_sae.log 2>&1 &
     ```
   - Expected train_key (sae_arditi seed=1 100K): compute via
     `temp_bench.config.compute_train_key` before launch.
4. **When all done** (or before pod restart / `status: complete`):
   `bash scripts/wrap_up_session.sh` — adds metrics.json, judge
   transcripts, manifest tail, leaderboard tail; commits + pushes;
   confirms HF state for ephemeral mode.
5. **Don't render anything to docs/components/c6.md yourself** —
   agent_paper integrates at paper-render time.

## Don't repeat (agent owns — overwrite)

### Territory rules
- **Don't edit `experiments/c6_em/`** — agent_em's territory. Import,
  don't modify.
- **Don't edit `docs/components/c6.md`** — agent_paper integrates at
  paper-render time.
- **Don't render anything to `docs/components/c6.md` yourself** — that's
  agent_paper's. Just confirm leaderboard rows land.

### Driver internals
- **Don't bypass `runner.run_cell`** — even from a custom driver, go
  through the canonical pathway (which appends to leaderboard.jsonl).
- **Don't allocate `train_key` / `eval_key` manually** — the runner
  computes them from your inputs deterministically.
- **Don't change `bricken_*` defaults from the brickenauxk_a8 recipe**
  for txc_base — literal copy of agent_em's recipe at higher n_steps.
- **Don't run sae_arditi with `bricken_enabled=True`** — agent_em's
  `make_training_cfg("sae_arditi")` returns `bricken_enabled=False`.
- **Don't push to HF manually** — `cache.save_checkpoint` does it on
  ephemeral pods. Verify via the URL in manifest after each cell.

### Briefing sketch import drifts (corrected in driver)
- **`make_training_cfg` lives in `experiments.c6_em.run`, NOT `.train`.**
  `train.py` only exports `my_train_fn`. Importing from `.train`
  raises ImportError.
- **`my_eval_fn` is a factory `make_eval_fn(datasource_name)`**, not
  a top-level symbol.
- **`DATASOURCE` is named `DEFAULT_DATASOURCE`** in run.py.

### Pod-specific gotchas
- **`peft` not in pyproject.toml** — Wang stage 1 will crash without
  it. Always run `uv pip install peft` after env setup until OQ #1
  lands. See OQ #1.
- **No `qwen_2_5_14b_instruct_finance_l24_resid_post` cache on HF**
  as of 2026-05-04. Driver's `ensure_activation_cache` builds it in
  ~48 sec on H100 + auto-pushes. Future runs hit cache.
- **TXC training is ~6× slower than SAE on this pod** — 5h 8min for
  100K vs SAE's 52 min. Briefing said TXC ~1.3× SAE. Plan time
  accordingly. See OQ #2.

### Append-only file conflicts
- **Origin/final's leaderboard.jsonl + manifest.jsonl had committed
  conflict markers** at HEAD `6ea931e2` (Han's merge of 9044d487
  was published with `<<<<<<< HEAD` / `=======` / `>>>>>>>` lines
  intact). When you rebase, expect git to surface these as fresh
  conflicts. Resolve by **dedup-by-key**: load both sides via JSON,
  union by `eval_key` (leaderboard) or `train_key` (manifest), keep
  insertion order. See `c3` commit `68ece86c` for the recipe.
- **Smoke leaderboard row at `train_key=29d23894a05bfc12`** is
  intentional noise (n_steps=200, peak_align=0.0). Don't remove —
  it's append-only. Filtered out by canonical_train_keys.

## Open questions for Han (agent owns — overwrite)

### OQ #1 (URGENT, 2026-05-04T23:14Z): `peft` not in pyproject.toml

`src/temp_bench/case_studies/em.py:119` does `from peft import PeftModel`,
required by Wang Stage 1 to load the LoRA adapter. **`peft` is not in
`pyproject.toml` or `uv.lock`.** agent_em must have it installed via
`pip install peft` outside the lockfile on their pod.

My fresh H100 pod failed with `ModuleNotFoundError: No module named
'peft'` *after* the SAE seed=42 100K training had already completed
(52 min wasted compute before failure was detected).

**Workaround applied**: `uv pip install peft` (peft 0.19.1) on this
pod, no pyproject.toml or uv.lock change (cross-territory rule
respected). Survives only this pod's lifetime.

**Permanent fix (agent_paper)**: atomic pyproject.toml + uv.lock
commit adding `"peft>=0.15"`. Without it, every fresh ephemeral pod
running C6 Wang loses 50+ min before crashing.

### OQ #2 (2026-05-05T08:00Z): TXC training is 5.9× slower than SAE on this pod

agent_em's briefing said TXC + Bricken at 100K should take ~73 min
(1.3× SAE's ~56 min, extrapolating from agent_em's 25K timing of
~25 min). On my H100 pod, **TXC took 5h 8min** (308 min) for 100K
steps — vs SAE's 52 min on the same hardware.

Confirmed not a stall — process active, GPU at 30-100% util,
1 CPU core saturated. Bricken fired 199× (last n_resampled=16384,
50% cap). Final loss=293.9 (reasonable; lower than smoke 200-step
loss=169776).

**Hypothesis**: the per-step Python `batch_iter` for-loop in
`experiments/c6_em/train.py:_build_batch_iter` does a Python
iteration over batch=1024 to slice T=5 windows from the preloaded
acts tensor. For T=5 (vs T=1 for SAE), each iteration moves 5×
more data per slice + dtype convert. Combined with Bricken's
per-500-step `n_check=2048` recompute and AuxK overhead, per-step
cost is much higher than SAE.

agent_em's 25K cell at ~25 min would be ~16.7 steps/sec. My 100K
cell ran at ~5.4 steps/sec. So I'm 3× slower than agent_em even
within TXC. Could be CPU-frequency difference between pods (hard
to verify without their `cat /proc/cpuinfo`).

**Impact on sweep**: seed=1 won't fit if both archs run. Plan: SAE
seed=1 only (~4 hr) for partial n=2 coverage on sae_arditi. Skip
TXC seed=1 stretch.

**Possible permanent fixes** (out of scope for me — would touch
agent_em's territory):
1. Vectorize `_build_batch_iter`: replace the Python for-loop with
   `acts[seq_idx[:, None], pos_idx[:, None] + arange(T)]` advanced
   indexing. Should give 10-100× speedup on the data path.
2. Pin `torch.set_num_threads(N)` in trainer to avoid CPU
   over-subscription on multi-core pods.
