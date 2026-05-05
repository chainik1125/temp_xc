<!--
DRAFT — written by agent_paper 2026-05-03 for Han's review.
Han: rewrite the "Identity + mandate" section as you wish before
spinning up agent_em; agents will not touch it after.
Section ownership rules: PROTOCOL.md § 14.
-->

---
agent: agent_em
last_state_update: 2026-05-03T22:00:00Z
component: c6
---

## Identity + mandate (Han owns — agents do not edit)

You are **agent EM**. You own C6 only. Files you may edit:
- `agents/agent_em/briefing.md` (your own — agent-owned sections only)
- `docs/components/c6.md`
- `experiments/c6_em/`
- Code under `src/temp_bench/` that you author + commit (the Wang
  procedure runner under `temp_bench.case_studies.em`, Bricken
  trainer logic under `temp_bench.training.bricken`)
- `configs/datasources.yaml` — adding new C6 datasources is fine.

**Files that are OUT OF SCOPE — do NOT edit even if it seems harmless:**
- `agents/agent_*/` — every other agent's directory.
- `docs/paper/*` — agent_paper's territory.
- `agents/agent_paper/decisions.md` — global decisions log.
- `agents/README.md` — roster/contract; agent_paper-owned.
- `configs/locked_archs.yaml` — only agent_paper edits this.
- `pyproject.toml` and `uv.lock` — dependency changes affect every
  agent's venv; pyproject + lockfile must be committed atomically,
  and only agent_paper coordinates that. If you need a new dep,
  surface in Open questions.

If you find yourself wanting to edit any out-of-scope file, **STOP**.
Add a bullet to "Open questions for Han" in your own briefing,
surface it in chat, and let Han or agent_paper land the change. Even
if Han verbally approves, do not commit cross-territory edits yourself.

### ⚠️⚠️⚠️ STAND DOWN — MW pivot RESCINDED 2026-05-05 PM ⚠️⚠️⚠️

**Han + agent_paper diagnosed that the MW pivot was solving a misframed
problem.** SAEBench (papers/are_saes_useful.md, App. B) shows canonical
SAE training is buffer-based, batch=2048 TOKENS/step, ~500M tokens
total — per-step token throughput on the order of 10³, not 10⁵.

Our two patterns at this paper:

| Component | Pattern                             | per-token tokens/step |
|---|---|---:|
| C3, C4, C5 | sequence-based (B sentences × all 128 positions) | **131,072** — 5× over SAEBench's 2K canonical |
| C6, C7    | window-based (B sentences × T positions, T=1 for SAE) | **1,024** — close to SAEBench canonical |

C3/C5's 131K is OVER-batched per-step; C6/C7's 1K is near-canonical.
The earlier "TXC has 25× FLOPs disadvantage" framing was directionally
true ONLY at C3/C5 (per-token archs over-batched). At C6 + C7 the
per-token baselines were *already* at literature scale (T=1 window-
based), so MW would actually OVER-correct them.

**Han's call (2026-05-05 PM)**: ABORT all 4 MW pivots. C3 + C5
per-token baselines re-train at T=1; C6 + C7 are unchanged.

**Your specific situation** (do these in order):

1. **Your canonical 8/8 sweep stands as the C6 paper headline.** The
   final cell `bqp1ssnty` (TXC seed=1 7B-medical) should have landed
   ~13:14 UTC; verify in `leaderboard.jsonl` and re-render c6.md
   AUTO-RESULTS as planned. Mean gaps remain: -3.25 14B-finance,
   -6.14+ 7B-medical (TXC > SAE).
2. **Do NOT launch the C6 MW 4-cell sweep.** The MW driver
   (`experiments/c6_em_mw/run.py`, commit `03facd49`) and YAML alias
   `txc_base_mw` stay registered as inert reserves for post-paper
   revisitation. Don't delete; just don't launch.
3. **Status (after 8/8 lands + re-render + commit + push)**: canonical
   mission COMPLETE; idle. No further compute on this pod is needed
   for the paper sprint. Use remaining session time for paper-writing
   contributions to c6.md (caveats, methodology bullet on per-step
   training-FLOPs asymmetry — now resolved by literature alignment
   rather than MW). Then `bash scripts/wrap_up_session.sh` and HF
   backup the 8 canonical checkpoints.
4. **What replaces MW for C3+C5 fairness** is a per-token baseline
   re-train at T=1 window-based, handled by agent_em_100k (C3) and
   agent_filler (C5). Your C6 work is unaffected because C6 was
   already T=1 window-based (correct by accident — SAE-arditi's eval
   protocol used `_build_batch_iter` with T=1 from day one). The C6
   numbers stay as-is.

**The C6 MW deployment directive below this line is RESCINDED.** Do
not read it as actionable. Left in place for git provenance only.

---

### ⚠️ Han decisions 2026-05-05 — C6 MW deployment (post-canonical mission) [RESCINDED 2026-05-05 PM — see STAND DOWN above]

After your final canonical cell (`bqp1ssnty` — txc_base seed=1
7B-medical) lands ~13:14 UTC, your canonical mission is COMPLETE.
Your next mission: **deploy multi-window TXC at C6**.

**Background**: agent_paper landed `txc_base_mw` as a YAML alias of
TXCBase with `multi_window: true` baked into hparams (decisions.md
§ 14, commit `ecc4c661`). Same Python class; only the per-step
sampling differs (stride-T tiling instead of 1-random-window-per-row).
This eliminates the ~25× per-step FLOPs disadvantage that biased the
canonical comparison against TXC. Your existing canonical cells stay
in `leaderboard.jsonl` as the historical pre-fix baseline; new MW
cells land at fresh `train_keys` (the `multi_window: true` hparam
flows into `compute_train_key`'s hash).

**Mission scope** (n=2 paired with your existing canonical cells):

- 2 archs × 2 seeds × 2 organisms = **4 cells**:
  - txc_base_mw + brickenauxk_a8 × {seed=42, seed=1} × 14B-finance
  - txc_base_mw + brickenauxk_a8 × {seed=42, seed=1} × 7B-medical
- TXC-pro is NOT in the C6 mandate (per § 7, C6 uses txc_base + Bricken
  only). No `txc_pro_mw` cells for C6.
- SAE-arditi is per-token (no MW variant); your existing canonical
  SAE cells are the comparison baseline. **Do not re-run SAE.**

**Per-cell `TrainingConfig`**:

```python
TrainingConfig(
    batch_size=1024,
    n_steps=25_000,                 # canonical, NOT 100K
    plateau_early_stop=False,
    bricken_enabled=True,           # for txc_base_mw; per § 7
    bricken_resample_every=5000,    # ⚠️ 10× the non-MW default of 500.
                                    #   Han's call (decisions.md § 14
                                    #   "Bricken resample-rate caveat"):
                                    #   keep rate-equivalent to non-MW.
    bricken_min_fires=1,
    bricken_n_check=2048,
    bricken_max_resample_fraction=0.5,
    ema_auxk_alpha=0.125,           # 1/8 per a8 recipe
    dead_threshold_tokens=128_000,  # 128k tokens per a8 recipe
)
```

**The Bricken rate decision**: under MW, each step processes N=10
windows × t_sample=5 tokens = 50 tokens per row vs 5 in non-MW.
Bricken_every=500 in non-MW = check every 500×5=2500 tokens.
Bricken_every=5000 in MW = check every 5000×50=250000 tokens —
not the same as non-MW! Let me reconsider.

Actually Han's choice (5000) is per § 14 — picking the simpler
"step-count is 10× larger" interpretation of rate-equivalence vs
the "tokens-between-checks" interpretation. **Use 5000 as
specified**; document the choice in c6.md caveats. Fine-grained
tuning is out of scope for the sprint.

**Per-cell wall-time estimate on your pod (2× H100, fast per step
relative to agent_em_100k's pod)**:

- Training: agent_em's canonical TXC + Bricken at 25K = ~80 min.
  MW adds ~5× per-step compute (matryoshka not present in
  txc_base_mw, but the encoder shape is `(B*N=10240, T=5, d_in=5120)
  @ (T, d, s=32768)` = ~22 TFLOPs vs ~0.86 TFLOPs non-MW). H100
  does this in ~22 ms/step × 25K = ~9 minutes... actually compute
  isn't the bottleneck, the data path is. Realistically: ~2-3× slower
  per step than non-MW, so ~3-4 hr training per cell.
- Wang full: ~3 hr (unchanged — Wang is generation-bound, not
  training).
- Per cell: ~6-7 hr.
- 4 cells serial on 1 GPU: ~24-28 hr.
- 4 cells parallel on 2 GPUs (you have 2 H100s on this pod, agent_nlp
  may have completed their topk_sae sweep by now — check before
  launching): ~12-14 hr wall.

**Important**: agent_em's `_build_batch_iter` returns shape
`(B, T, d)` — pre-windowed. The MW `train_step` expects
`(B, seq_len, d)` and tiles internally. **You need a NEW batch_iter
for MW that returns full sequences.** Sketch in
"First concrete task — write the MW driver" below.

VRAM check (per agent_paper's analysis): C6 at Qwen-14B scale
(d_in=5120, d_sae=32768) MW peaks ~50-75 GB activations. Within H100
80GB but tight. Use `precision="bf16"` (already default for H100).
If OOM, reduce batch_size to 512 (effective B*N=5120 still > non-MW
B=1024, still gives MW benefit).

References:
- `agents/README.md` (your roster row)
- `docs/components/c6.md` (your existing C6 writeup; do NOT edit
  the AUTO-RESULTS block — agent_paper integrates at paper-render
  time)
- `experiments/c6_em/{train.py,run.py,analysis.py}` — your existing
  plumbing; you write a NEW `experiments/c6_em_mw/run.py` that
  imports from these and wires in a full-sequence batch_iter.
- `decisions.md` § 7, § 12, § 14 (especially the Bricken rate caveat)
- `papers/temporal_sae.md` (sae_arditi is the SAE comparison; T-SAE
  reference for context)

### First concrete task — write the C6 MW driver, smoke, launch

Step 1 — verify your final canonical cell completed (the
`bqp1ssnty` Bash task should have fired with txc_base seed=1 7B
~13:14 UTC). Check the leaderboard:

```bash
.venv/bin/python -c "
from temp_bench.cache import _read_jsonl, leaderboard_path
rows = [r for r in _read_jsonl(leaderboard_path()) if r['component']=='c6']
for r in rows[-4:]:
    print(r['arch'], r['seed'], r['datasource'], r['metrics'].get('peak_align'))
"
```

Step 2 — pull and verify the MW arch is registered:

```bash
git pull --rebase origin final
.venv/bin/python -c "from temp_bench.config import load_arch; print(load_arch('txc_base_mw').hparams)"
# → expect dict containing multi_window=True
```

Step 3 — write `experiments/c6_em_mw/__init__.py` (empty) and
`experiments/c6_em_mw/run.py`. Key design: import as much as
possible from your existing `experiments/c6_em/{train.py,run.py}`,
override only:

(a) the batch_iter (full-sequence instead of pre-windowed),
(b) the TrainingConfig override (`bricken_resample_every=5000`).

Sketch:

```python
"""C6 multi-window deployment driver.

Replicates agent_em's setup with two changes:
1. arch_name: txc_base → txc_base_mw (multi_window=True in hparams)
2. batch_iter: returns FULL sequences (B, seq_len, d) instead of
   pre-windowed (B, T, d). MW's train_step does the tiling.
3. TrainingConfig: bricken_resample_every=5000 (§ 14 rate-eq under MW).
"""
from __future__ import annotations
import argparse
import json

import numpy as np
import torch

from temp_bench import runner
from temp_bench.schemas import TrainingConfig
from temp_bench.config import act_cache_dir, compute_act_cache_key, load_datasource
from experiments.c6_em.run import EVAL_PROTOCOL_VERSION, my_eval_fn
from experiments.c6_em.train import (
    _instantiate_with_overrides,
    make_training_cfg as _make_canonical_cfg,
)
from temp_bench.training.sae_trainer import train_sae

# Module-global cache for full-sequence preload — shared across cells in
# the same process.
_PRELOADED_C6_FULL: dict[str, torch.Tensor] = {}


def _build_full_seq_batch_iter(act_cache_key: str, *, seed: int = 42):
    """MW data path: returns (B, seq_len, d_in) full sequences. The
    arch's train_step tiles into (B*N, T, d_in) windows internally.
    """
    cache_dir = act_cache_dir(act_cache_key)
    specs = json.loads((cache_dir / "layer_specs.json").read_text())
    cache_path = str(cache_dir / f"{specs['key']}.npy")
    if cache_path not in _PRELOADED_C6_FULL:
        mmapped = np.load(cache_path, mmap_mode="r")
        _PRELOADED_C6_FULL[cache_path] = (
            torch.from_numpy(np.ascontiguousarray(mmapped)).clone()
        )
    acts = _PRELOADED_C6_FULL[cache_path]
    rng = np.random.default_rng(seed)
    def batch_iter(n: int) -> torch.Tensor:
        idx = rng.integers(0, acts.shape[0], size=n)
        return acts[idx].to(torch.float32)
    return batch_iter


def _make_mw_training_cfg(arch_name: str) -> TrainingConfig:
    """C6 MW training cfg — start from your canonical make_training_cfg
    then bump bricken_resample_every from 500 → 5000 per § 14."""
    base = _make_canonical_cfg(arch_name)
    return base.model_copy(update={"bricken_resample_every": 5000})


def my_train_fn_mw(*, arch_name, arch_hparams, seed, training_cfg,
                   act_cache_key, component):
    """C6 MW train_fn — analogous to agent_em's my_train_fn but with
    the full-sequence batch_iter."""
    ds_name = arch_hparams.pop("__datasource_name__", None) or arch_hparams.get("datasource")
    # … you adapt your existing c6_em/run.py:my_train_fn here, swapping
    # _build_batch_iter → _build_full_seq_batch_iter. Keep the
    # _instantiate_with_overrides call (brickenauxk_a8 recipe).
    ...


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+",
                    default=["txc_base_mw"], choices=["txc_base_mw"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1])
    ap.add_argument("--organisms", nargs="+",
                    default=["qwen_2_5_14b_instruct_finance_l24_resid_post",
                             "qwen_2_5_7b_instruct_medical_l15_resid_post"])
    args = ap.parse_args()

    for ds in args.organisms:
        for arch in args.archs:
            for seed in args.seeds:
                cfg = _make_mw_training_cfg(arch)
                runner.run_cell(
                    component="c6", arch_name=arch, seed=seed,
                    datasource_name=ds,
                    training_cfg=cfg,
                    eval_cfg={"sweep": "c6_mw_v1"},
                    eval_protocol_version=EVAL_PROTOCOL_VERSION,
                    train_fn=my_train_fn_mw,
                    eval_fn=my_eval_fn,
                )


if __name__ == "__main__":
    main()
```

(The exact `my_train_fn_mw` shape will depend on your existing
`my_train_fn` in `c6_em/run.py` — adapt it to use
`_build_full_seq_batch_iter`. Keep the brickenauxk_a8 override path
via `_instantiate_with_overrides`. Don't rewrite the trainer; call
`temp_bench.training.sae_trainer.train_sae` like your existing
my_train_fn does.)

Step 4 — smoke ONE cell at `n_steps=200`:

```bash
TQDM_DISABLE=1 .venv/bin/python -c "
from experiments.c6_em_mw.run import _make_mw_training_cfg, my_train_fn_mw
from experiments.c6_em.run import EVAL_PROTOCOL_VERSION, my_eval_fn
from temp_bench import runner
from temp_bench.schemas import TrainingConfig
cfg = _make_mw_training_cfg('txc_base_mw').model_copy(update={'n_steps': 200})
result = runner.run_cell(
    component='c6', arch_name='txc_base_mw', seed=42,
    datasource_name='qwen_2_5_14b_instruct_finance_l24_resid_post',
    training_cfg=cfg, eval_cfg={'sweep': 'c6_mw_smoke'},
    eval_protocol_version=EVAL_PROTOCOL_VERSION,
    train_fn=my_train_fn_mw, eval_fn=my_eval_fn,
)
print('smoke result:', result.train_key, result.eval_key, result.cached)
"
```

Step 5 — launch the full 4-cell sweep. If both H100s are yours
(agent_nlp wrapped up), parallelize across GPU 0+1; otherwise serial
on GPU 1:

```bash
# Parallel (if agent_nlp is idle):
TQDM_DISABLE=1 AGENT_NAME=agent_em \
  bash scripts/run_on_gpu.sh 0 -- \
  .venv/bin/python -m experiments.c6_em_mw.run \
  --seeds 42 \
  > logs/c6_mw_gpu0.log 2>&1 &

TQDM_DISABLE=1 AGENT_NAME=agent_em \
  bash scripts/run_on_gpu.sh 1 -- \
  .venv/bin/python -m experiments.c6_em_mw.run \
  --seeds 1 \
  > logs/c6_mw_gpu1.log 2>&1 &

# Serial (if you only have GPU 1):
TQDM_DISABLE=1 AGENT_NAME=agent_em \
  .venv/bin/python -m experiments.c6_em_mw.run \
  > logs/c6_mw_serial.log 2>&1 &
```

Step 6 — monitor + verify rows land at
`arch=txc_base_mw, eval_protocol_version=2.0.0` (your existing
canonical eval protocol).

agent_paper integrates results at paper-render time. Your existing
canonical cells (txc_base @ 25K) stay as the historical baseline;
new `txc_base_mw` cells become the canonical headline at paper time.

### Han decisions 2026-05-04 PM (preloaded batch_iter — apply .clone() locally)

agent_nlp profiled the data path and found the trainer was bottlenecked
on numpy fancy indexing over an mmap'd `.npy` (~150K page-table walks
per step at batch=1024). They landed a shared opt-in helper at
`temp_bench.data.nlp.cache.preloaded_batch_iter_from_act_cache` (commit
`e12dc719`) that pre-materializes the cache into a CPU torch tensor via
`.clone()`. Empirical: ~1.4× end-to-end trainer speedup; ~3.4× on the
data path. **You can benefit, but it's not a drop-in.**

**Why it's not a drop-in for C6**: your `_build_batch_iter` in
`experiments/c6_em/train.py:46-79` samples T-token sliding windows
(`arr[seq_idx[i], pos_idx[i]:pos_idx[i] + T]`), while agent_nlp's
shared helper returns whole sequences `(batch, seq_len, d_in)`. Same
mmap bottleneck, different access pattern.

**Apply the `.clone()` pattern locally** in your existing `_build_batch_iter`.
Sketch:

```python
# At module scope in experiments/c6_em/train.py:
_PRELOADED_C6_ACTS: dict[str, torch.Tensor] = {}

def _build_batch_iter(act_cache_key: str, *, T: int = 5, seed: int = 42):
    from temp_bench.config import act_cache_dir as _acd
    cache_dir = _acd(act_cache_key)
    specs = json.loads((cache_dir / "layer_specs.json").read_text())
    hp_key = specs["key"]
    cache_path = str(cache_dir / f"{hp_key}.npy")

    # Preload once per (process, cache_path); subsequent cells share.
    if cache_path not in _PRELOADED_C6_ACTS:
        mmapped = np.load(cache_path, mmap_mode="r")
        # .clone() is load-bearing — without it, torch.from_numpy
        # zero-copy wraps the mmap and page-faults persist.
        _PRELOADED_C6_ACTS[cache_path] = (
            torch.from_numpy(np.ascontiguousarray(mmapped)).clone()
        )
    acts = _PRELOADED_C6_ACTS[cache_path]
    N, L, d = acts.shape
    if L < T:
        raise RuntimeError(f"Cache seq_len={L} < T={T}")
    rng = np.random.default_rng(seed)

    def batch_iter(n: int) -> torch.Tensor:
        seq_idx = rng.integers(0, N, size=n)
        pos_idx = rng.integers(0, L - T + 1, size=n)
        out = torch.empty((n, T, d), dtype=torch.float32)
        for i in range(n):
            out[i] = acts[seq_idx[i], pos_idx[i]:pos_idx[i] + T].to(torch.float32)
        return out

    return batch_iter
```

The for-loop stays but now operates on a torch tensor in RAM (not an
mmap), so each iteration is RAM-rate instead of page-fault-rate.

**Determinism**: same `np.random.default_rng(seed)` for indices, same
fp32 contract — `train_keys` are unchanged, checkpoints are bit-identical.
**No fairness implication; adopt mid-sweep is safe** (the .clone() path
and the mmap path produce the same checkpoint bytes for the same
`(act_cache_key, seed)` pair; cells from either path coexist cleanly in
the leaderboard).

**RAM cost on H100 pod**: one `acts.npy.size`-bytes copy per process per
distinct cache. Qwen-14B finance cache is ~31 GB at typical 24K-seq
configurations; 7B-medical is ~22 GB. With both organisms loaded by one
process, ~53 GB. With your two H100 GPUs each running their own process
(both organisms each), worst case ~106 GB. H100 pod system RAM is
typically 150-200 GB — **safe with margin**, but verify via
`free -g` before launching the second process if you ever run all four
(2 organisms × 2 processes) concurrently.

If the .clone() pattern fits cleanly into a single shared helper for
C6 (e.g., a `preloaded_qwen_em_batch_iter` in
`temp_bench.data.nlp.qwen_em` with the T-window slicing built in),
that's worth landing as a follow-up, but apply it locally first to
unblock your remaining cells.

### Han decisions 2026-05-04 PM (CRITICAL — TrainingConfig re-issued per Phase 5)

A "batch=256 → 2048" cross-agent directive was issued earlier today
(commit a9200560) and reverted (commit 0beae2bf). It treated only the
contrastive archs (T-SAE, TXC-pro) as needing higher batch — Han caught
that as unfair to non-contrastive baselines. **The new directive is
Phase-5-faithful, applied uniformly across all archs and all pods.**
Read `decisions.md` § 12 in full before running anything. Gist:

- **`TrainingConfig` defaults are now**: `batch_size=1024`, `n_steps=25_000`,
  `plateau_early_stop=False` (disabled — see below). Just default-construct
  `TrainingConfig()` in your runner — no per-component overrides for these
  knobs (`bricken_*` overrides per § 7 stay).
- **Uniform across pods**: H100 + A40 both run at batch=1024. Every arch
  trains under identical conditions, matching the SAE-comparison-paper
  standard (T-SAE §4.1, TFA App. B.1, GemmaScope).
- **Plateau-stop is OFF; 25K cap is binding for every cell.** The schema's
  plateau detection (absolute max-min over 5K window) is cross-arch unfair
  because archs at different loss scales would trigger at different points.
  SAE literature uses fixed step counts. Every cell trains to the 25K cap;
  fairness mechanism is "exactly 25.6M tokens per arch."
- **If you observe loss still descending steeply at step 25K** (e.g.,
  final-1K-step drop > 5% of loss value), surface that as a comment on
  your run — the cap may need to be bumped uniformly across all archs.
- **Cache hygiene**: `batch_size` + `n_steps` are in the `train_key` hash
  (`src/temp_bench/config.py:181-193`). New cells get fresh keys
  automatically. Old batch=256 cells stay in `results/leaderboard.jsonl`
  for diff comparison — **when rendering AUTO-RESULTS, filter for new
  rows only** (e.g., `training_cfg.batch_size == 1024`).

**Your specific action — ABORT in-flight calibration cells.** The
calibration cells running on H100s right now are batch=256 — they will
produce undertrained checkpoints unusable as paper headlines. Stop them
ASAP (Han approved the ~12 H100-hour sunk cost), reload with
default-constructed `TrainingConfig()` (which now defaults to the new
values), and restart. Existing batch=256 calibration rows stay in the
leaderboard for reference; new rows write fresh.

### Han decisions 2026-05-04 (resolves prior session's open questions)

1. **`per_component_hparams[c6]` for txc_base + txc_pro: LANDED.**
   `configs/locked_archs.yaml` now has `c6: { d_sae: 32768, k_pos: 25 }`
   for `txc_base` and `c6: { d_sae: 32768 }` for `txc_pro`. The prior
   75.88 peak_align cell stays in the leaderboard with its own
   train_key as a "small TXC" reference.
2. **Judge: stick with Anthropic. NO Gemini.** Dmitry's Gemini numbers
   are wasteland reference, not a paper claim we need to match exactly.
   Judge variance σ ≈ 6 align points dwarfs Haiku-vs-Gemini divergence
   on Wang grading. Document the deviation in c6.md caveats;
   judge_outputs.jsonl persistence lets us validate κ post-deadline.

### Han decisions 2026-05-04 (NEW — Wang abbreviation oversight + 7B re-run)

**The "abbreviated Wang" you ran (skipping stages 2 + 3) was a
methodological oversight.** It wasn't in c6.md's Setup spec (which
explicitly says "Wang procedure (4 stages): Δz̄ encoder rank → causal
screen at α = ±1 → strength sweep → final per-feat α frontier") and
wasn't in `decisions.md`. The +3.79 align gap is reported as the C6
headline but is suspect — features ranked top-3 by Δz̄ may not be the
ones full Wang's causal screen would surface, and a 6-α grid may
miss TXC's actual peak. **The result needs to be re-derived with
the FULL Wang protocol.**

3. **Run FULL Wang on ALL C6 cells.** All 4 stages: Δz̄ rank → causal
   screen at α=±1 (filter features that don't causally shift align)
   → per-survivor strength sweep (~10 α values per surviving feature)
   → final per-feat α frontier. Drop the top-3 cutoff and the 6-α
   grid abbreviation. Same protocol on every cell — both 14B-finance
   AND 7B-medical (see #4). Existing abbreviated-Wang cells stay in
   the leaderboard for diff comparison; full-Wang cells get fresh
   eval_keys (new `eval_protocol_version` bump if you change the
   eval_cfg shape).

4. **Add 7B-medical re-run.** The C6 paper framing ("step-efficiency
   on 7B + Mixed on 14B = tradeoff") currently rests on Dmitry's
   wasteland-published 7B numbers, which used a different judge
   (Gemini), different prompts (full Wang), and a different TXC
   variant. That's a cross-paper-citation pattern reviewers will
   challenge. Pair the 14B numbers with our OWN 7B-medical numbers.
   - Add a new datasource `qwen_2_5_7b_instruct_medical_l24_resid_post`
     to `configs/datasources.yaml` (your territory — `# C6 ...` is
     fine to add, follow the C6 14B entry's format).
     `subject_model: Qwen/Qwen2.5-7B-Instruct`,
     `lora_adapter: <Dmitry's medical organism — find on origin/em-nanda
     and pin the source commit hash in the notes>`. d_model=3584.
   - Run 3 seeds × 2 archs (sae_arditi-7B + txc_base+brickenauxk_a8-7B)
     with FULL Wang on the 7B-medical cohort. Same protocol as
     14B-finance.
   - Expected outcome: TXC much closer to / matching SAE-arditi
     (per Dmitry's wasteland reference: TXC brickenauxk 30k @
     resid_mid = 53.87 ties T-SAE 100k @ resid_post = 52.39, ~3.5
     below SAE arditi 57.42). If our re-derived numbers confirm
     that pattern, the "step-efficiency win" half of the paper's
     tradeoff framing becomes a single-paper apples-to-apples
     comparison.

5. **Use both H100s when agent_nlp is idle.** GPU sharing is a
   convention now — the `claim_gpu` lockfile system was deleted
   2026-05-04 (PROTOCOL.md § 13 *GPU sharing convention*). To borrow
   agent_nlp's GPU 0:

   - **Verify they're idle**: read `agents/agent_nlp/briefing.md`
     "Current state" — does it say `status: complete` or "idle"? If
     they're mid-cell with an ETA, wait or use only GPU 1.
   - **Verify with `nvidia-smi`**: GPU 0 should show <1 GB used and
     no long-running python process.
   - **Update YOUR briefing's "Current state"** with
     `"Borrowing GPU 0 until ETA HH:MM UTC for C6 7B-medical seed=N
     — agent_nlp is status: complete."` BEFORE you launch.
   - **Launch via the wrapper** (sets `CUDA_VISIBLE_DEVICES=0` for
     the subprocess only; your own python process stays pinned to
     GPU 1):

     ```bash
     bash scripts/run_on_gpu.sh 0 -- python -m experiments.c6_em.run --seeds 1
     ```

     Or in Python if you'd rather drive in-process:

     ```python
     import os, subprocess
     env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "AGENT_NAME": "agent_em"}
     subprocess.run(["python", "-m", "experiments.c6_em.run", "--seeds", "1"], env=env)
     ```

6. **Failure mode**: if you and agent_nlp accidentally launch on the
   same GPU simultaneously, both crash with CUDA OOM. Recoverable in
   ~5 min — restart the cell on the other GPU or wait for peer's run
   to finish. No state corruption (each cell is independent and
   deterministic via `train_key`).

7. **Time budget**: full Wang on 12 cells (3 seeds × 2 archs × 2
   organisms) is ~25–50 H100-hr serial; ~12–25 hr wall time if you
   parallelize across both H100s. agent_nlp's pod is yours to borrow
   as long as they're idle — re-verify before each long borrow.

You are agent EM, lead on **C6: emergent misalignment** on
`Qwen/Qwen2.5-14B-Instruct` + finance LoRA organism (R1 + R32). The
component is in **status: pending-retest** in `docs/components/c6.md`.

Hardware: pod `2× H100`, pinned to **GPU 1**. Pod mode `persistent`.
agent_nlp shares the pod on GPU 0; you will not collide because
pinning is enforced. **Fallback**: if R32 OOMs the H100 (14B model +
LoRA at fp16 ≈ 28 GB so it should fit, but R32 may stress it), spin
up `agent_em_h200` (provisioned dormant — see `agents/README.md`).

Why the re-test: Dmitry's published Qwen-14B finance numbers
(`em_nanda_results_paper.md`) were plain TXC k=100, no Bricken,
no anti-dead — not a fair comparison vs SAE arditi which has 100k
training steps and dead-feature handling. With the brickenauxk_a8
recipe (Bricken + EMA-AuxK α=1/8 + dead-threshold 128k tokens),
TXC may close the +3.91 gap on R1 and the +12.58 gap on R32.

Decision tree (after R1 30k mid-α first re-run):
- gap ≤ 3 align → **Tied** — headline win
- gap 3–9 align → **Mixed** — note step-efficiency win on Qwen-7B medical
- gap > 9 align → **Honest negative** — back to original framing

Coordinate with **Dmitry on `origin/em-nanda`** — he is still active
on this component. Read `EM_NANDA_BRIEF.md` for his latest state
before launching. Don't merge his branch into `final`; read via
`git show` (decision #4).

Salvageable contributions (independent of headline outcome):
- **Bundle null is architecture-general**: both arches' k=30 bundles
  peak at align ≈ 41.3 on R32, falling 13–23 align points below
  single-feat champions. Falsifies "distributed misalignment by sum."
- **Bundle precision is architecture-specific**: SAE has k=30 < k=3 <
  single-feat (precision helps); TXC inverts (top-3 anti-correlate).

Locked decisions in scope: #2 (C6 reframe + bundle-null result), #4
(cross-branch reads), #6 (HF repos), #7 (Bricken opt-in — **C6 turns
it on by default**, you don't need an A/B; the recipe is justified by
Dmitry's Qwen-7B medical evidence).

References:
- `agents/README.md` (your roster row)
- `docs/components/c6.md` (full setup, decision tree, Wang 4-stage)
- `docs/paper/architecture.md` *Per-experiment training knobs* (Bricken)
- `decisions.md` (esp. #2, #7)
- `origin/em-nanda:docs/dmitry/results/em_features/EM_NANDA_BRIEF.md` (latest)
- `PROTOCOL.md` § 11 (framework), § 12 (GPU pinning),
  § 9 *Session wrap-up*

**Before you `status: complete`**: `bash scripts/wrap_up_session.sh`.
Persistent pod → manual `hf upload` recipe printed by the script for
every checkpoint not yet on HF. Don't let Han stop the pod until
that loop completes (judge_outputs.jsonl + .safetensors live ONLY
on /workspace until you push them).

---

## Current state (agent owns — overwrite at every compact)

**Last verified: 2026-05-05T14:30Z. C6 CANONICAL MISSION COMPLETE.
8/8 canonical cells landed. c6.md status=`complete`. MW pivot
RESCINDED 2026-05-05 PM by Han (literature-alignment showed C6's
T=1 window-based sampling at ~1024 tokens/step already matches
SAEBench-canonical ~2048 tokens/step; MW would over-correct).
HF backup of 8 canonical checkpoints in flight (bash `biqgmy3x3`).**

**Final canonical headline (full Wang, batch=1024 / 25K, n=2 paired
seeds):**

| organism      | seed | SAE peak | TXC peak | gap (SAE-TXC) |
|---|---:|---:|---:|---:|
| 14B-finance   | 42   | 78.33    | 81.70    | -3.37 |
| 14B-finance   | 1    | 76.88    | 80.00    | -3.12 |
| 7B-medical    | 42   | 68.47    | 74.61    | -6.14 |
| 7B-medical    | 1    | 68.91    | 72.64    | -3.73 |

Mean gaps: 14B-finance -3.25 align (just outside Tied → "Mixed");
7B-medical -4.94 align ("Mixed", larger TXC margin).

**Both organisms: TXC + brickenauxk_a8 strictly beats SAE-arditi.**
Per c6.md decision tree: both in "Mixed" band. Paper framing: TXC
wins on both organisms; the 7B win is larger (smaller subject model
amplifies the misalignment-encoding advantage).

**Headline (full Wang, eval_protocol_version=2.0.0, n=2 paired seeds):**

| organism      | seed | SAE peak | TXC peak | gap (SAE-TXC) | TXC α at peak |
|---|---:|---:|---:|---:|---:|
| 14B-finance   | 42   | 78.33    | 81.70    | -3.37         | +100 |
| 14B-finance   | 1    | 76.88    | 80.00    | -3.12         | -100 |
| 7B-medical    | 42   | 68.47    | 74.61    | -6.14         | -100 |
| 7B-medical    | 1    | 68.91    | (running)| —             | — |

14B-finance mean gap: -3.25 align (TXC narrowly beats SAE; "Tied"
band per c6.md decision tree, just outside the ≤3 cutoff).

7B-medical seed=42 only: -6.14 align (TXC clearly beats SAE; "Mixed"
band 3-9). Need TXC seed=1 7B to confirm the mean.

**Methodological-flaw validation:** the 2026-05-03 abbreviated headline
showed SAE > TXC by +3.79 — this commit's full-Wang flips the sign,
because TXC's coherent peaks live at extreme α (±100), outside the
abbreviated 6-α grid {-30,-10,-3,+1,+3,+10}. agent_paper's flag was
correct.

**Per-cell timing (calibrated on real data):**

- 14B-finance cells: ~14 min train (SAE) / ~80 min train (TXC,
  T=5×FLOPs + Bricken). Wang: 56 min s2 + 70 min s3 + 40 min s4 =
  ~2.85 hr. Per cell: SAE ~3 hr, TXC ~4.2 hr.
- 7B-medical cells: ~10 min train (SAE) / ~60 min train (TXC).
  Wang: 33 min s2 + 40 min s3 + 22 min s4 = ~95 min. Per cell:
  SAE ~1.7 hr, TXC ~2.5 hr.

**Bash IDs of in-flight / completed cells:**

- ✅ `bu54cn30l` — sae_arditi seed=42 14B-finance
- ✅ `blv4w2awj` — txc_base   seed=42 14B-finance
- ✅ `b1uftr5k7` — sae_arditi seed=1  14B-finance
- ✅ `bwo6rgkjm` — txc_base   seed=1  14B-finance
- ✅ `bus6fiwpz` — 7B-medical activation cache build (3 min)
- ❌ `befthewic` — sae_arditi seed=42 7B (initial; killed by jsonl
  conflict markers in cache crash)
- ✅ `bb1zm7vem` — sae_arditi seed=42 7B (retry; train cache hit)
- ❌ `bnfkxelug` — txc_base   seed=42 7B (initial; killed by API
  credit-balance outage 06:36-07:38 UTC; train cache survived)
- ✅ `bowx2wejr` — txc_base   seed=42 7B (retry)
- ✅ `b82e26734` — sae_arditi seed=1  7B
- 🔄 `bqp1ssnty` — txc_base   seed=1  7B (final cell, ETA ~13:14)

**Incidents handled:**

1. TXC seed=42 14B silent-died on GPU 0 at 14:09 (sharing GPU with
   agent_nlp). Restarted on GPU 1 — succeeded.
2. SAE seed=42 7B initial cell crashed at runner.cache eval_in_leaderboard
   read because earlier rebase resolutions left conflict markers
   ('<<<<<<< HEAD', '=======', '>>>>>>> ...') in 3 leaderboard.jsonl
   lines. Cleaned via dedupe + JSON-validate; commit 934445c0.
3. TXC seed=42 7B initial Wang ran 06:36-07:38 with all judge calls
   returning HTTP 400 ("credit balance too low"). Process died at
   stage 3 baseline (defensive RuntimeError check fired — no garbage
   row). Han topped up, retried with train cache hit; succeeded.

**Code/infra adopted from peer agents:**

- `.clone()` preload patch in experiments/c6_em/train.py:_build_batch_iter
  (commit 48023e5a) — eliminates ~150K page-faults/step at batch=1024.
  Determinism preserved; train_keys bit-identical. ~1.4× speedup.
- `canonical_train_keys` filter wired into c6 analysis.py
  (commit f34e84db). Two calls (sae_arditi defaults; txc_base
  defaults + brickenauxk_a8 override) unioned to 8 expected keys.
  Pre-2026-05-04 batch=256 cells dropped from headline.

Other state:

- Pre-flaw 9 c6 leaderboard rows at `eval_protocol_version=1.0.0`
  (batch=256) remain in the leaderboard for diff-only comparison.
  analysis.py filters via two complementary mechanisms:
  1. **`canonical_train_keys` filter** (decisions.md § 12, agent_paper
     commit `9a39137a`): drops any row whose `train_key` doesn't match
     the current canonical TrainingConfig. C6 calls the helper twice
     (sae_arditi: defaults; txc_base: defaults + brickenauxk_a8 override
     per § 7), unions the keys.
  2. **`eval_protocol_version=="2.0.0"` filter**: independent backstop
     that drops any 1.0.0 abbreviated row even if a future schema
     change happened to give them matching train_keys.
- Activation cache at `results/act_cache/e052801ef8e6d22b/` (Qwen-14B
  finance, 6000 prompts × 128 tokens × 5120 d_in fp16 ≈ 7.86 GB).
- 7B-medical activation cache BUILT at `results/act_cache/e3c427abcd6dc30f/`
  (Qwen-7B medical, 6000 × 128 × 3584 fp16 ≈ 5.51 GB). Built 2026-05-04
  20:09 in parallel with SAE seed=1 14B training (no GPU contention).
- GPU lock system was deleted by agent_paper (commit `6e6efcbd`);
  use the GPU-sharing convention via `bash scripts/run_on_gpu.sh
  <idx> -- <cmd>` (PROTOCOL.md § 13). C6 has stayed sequential on
  GPU 1 since the 14:09 collision incident — agent_nlp's c3 retrain
  has held GPU 0 throughout.
- 124/124 pytest green (added 4 from agent_nlp's preloaded_batch_iter
  test suite, commit `e12dc719`).

## Historical context — abbreviated-Wang flaw + fix (2026-05-03 → 04)

**Status: RESOLVED.** The 9 abbreviated-Wang rows from 2026-05-03 are
preserved at `eval_protocol_version="1.0.0"` for diff comparison only;
analysis.py drops them via the `canonical_train_keys` filter (which
matches batch=1024) and the `eval_protocol_version=="2.0.0"` check.
Full Wang ported in commit `09866e53` (2026-05-04 morning), refined
in `144a3e84`.

What was wrong: my first runner (`run_wang_minimal`) implemented only
stages 1 + 4 of c6.md's 4-stage spec (Δz̄ rank → top-3 → 6-α frontier).
Skipped stage 2 (causal screen at α=±1) + stage 3 (per-survivor
coh-aware sweep). Headline reported SAE > TXC by +3.79 align.

What full Wang showed: TXC's coherent peaks live at extreme α=±100
(outside the abbreviated 6-α grid {-30,-10,-3,+1,+3,+10}). Stage 4's
27-α grid `-100,-10,-8,-6,-5,-4,-3,-2,-1.75,-1.5,-1.25,-1,0,1,1.25,
1.5,1.75,2,3,4,5,6,7,8,9,10,100` (Dmitry's exact grid) surfaces them.
Result: TXC > SAE on every cell (sign flip vs abbreviated). See
"Headline" table above.

Caveats from c6.md (still apply):

1. **Judge**: Anthropic Claude Haiku 4.5 (Han decision §2). All
   per-rollout transcripts in `judge_outputs.jsonl` per cell for
   post-deadline κ validation.
2. **Corpus stand-ins**: 14B uses `cfierro/personality-qs-risky-
   financial-advice`; 7B uses `cfierro/personality-qs-bad-medical-
   advice`. Both are HF mirrors of Turner et al.'s organism-specific
   probe sets (the original local files are not on HF).
3. **Hparams**: txc_base uses the c6 per-component override
   (d_sae=32768, k_pos=25). sae_arditi uses the locked default
   (d_sae=32768, k_pos=128). Both paper-correct.

Bricken trajectory during 14B TXC training: ~50 fires over 25 k steps
(every 500), peak n_resampled near max_resample_fraction=0.5 cap
(~16384). Consistent with Dmitry's ~75%-dead-by-step-40k trajectory.

## What I just did (agent owns — overwrite)

Two missions stacked: (A) canonical full-Wang sweep — 7/8 cells done;
(B) MW driver scaffold — landed, ready to launch after canonical.

(A) Canonical sweep (2026-05-04 → 2026-05-05):

- **Wang full port** (`09866e53`, `144a3e84`): stages 2 + 3 + 27-α
  stage 4. EVAL_PROTOCOL_VERSION bumped to "2.0.0". `run_wang_minimal`
  preserved for diff-against-abbreviated.
- **TrainingConfig refresh** (`bdb88829`): default-construct +
  brickenauxk_a8 override only (decisions § 12).
- **7B-medical datasource added** (commit `144a3e84`):
  `qwen_2_5_7b_instruct_medical_l15_resid_post` (Qwen-7B + andyrdt
  bad-medical LoRA + cfierro medical mirror; L15 relative-depth
  match for L24 on 14B).
- **`canonical_train_keys` filter** wired into c6 analysis.py
  (`f34e84db`). 8 keys (2 archs × 2 seeds × 2 organisms).
- **`.clone()` preload patch** (`48023e5a`): ~1.4× trainer speedup.
- **7 of 8 cells run** at full Wang batch=1024 / 25K:
  - 4 × 14B-finance (SAE + TXC × seeds 42, 1) — all DONE.
  - 4 × 7B-medical (SAE seeds 42, 1 + TXC seed=42) — DONE.
  - TXC seed=1 7B-medical: in flight (bash `bqp1ssnty`),
    stage 4 feat 1/3 α 4/27 at 12:43Z, ETA ~13:00.
- **2 incidents recovered**: jsonl conflict-marker crash (cleanup
  in `934445c0`); Anthropic API credit-balance outage 06:36-07:38
  (defensive RuntimeError caught it; train cache hit on retry,
  commit `541556fa`).

(B) MW deployment scaffold (2026-05-05 PM, post-canonical mission):

- **MW driver landed** (commit `03facd49`):
  - `experiments/c6_em_mw/run.py` (~290 lines): full-sequence
    `_build_full_seq_batch_iter`, `make_mw_training_cfg` (with
    `bricken_resample_every=5000` per § 14 rate-eq), `my_train_fn_mw`
    (asserts `_multi_window=True`), `main()` with --datasource +
    --seeds CLI.
  - `experiments/c6_em/train.py:_instantiate_with_overrides`
    extended to apply brickenauxk_a8 overrides for `txc_base_mw`
    too (one-line change).
  - Eval pipeline UNCHANGED — reuses `make_eval_fn` from c6_em/run.py
    at protocol "2.0.0".

(C) c6.md hand-curated refresh (commit `043b6f87`):

- Hypothesis trimmed (was forward-looking on R1 30k; now points at
  AUTO-RESULTS).
- "Existing evidence" + "Salvageable contributions" renamed to
  "Reference numbers (wasteland — for context only)" per PROTOCOL §7.
- Setup completely rewritten: drop R32 + drop txc_pro + add 7B-medical
  + add txc_base_mw + add MW Bricken-rate caveat + add the n=2 cut
  rationale + Hardware section with GPU 1 pin.
- Caveats: drop the obsolete "Mixed is judge-agnostic" pre-judgment;
  add bullets on per-step FLOPs asymmetry (motivating MW), n=2 cost,
  abbreviated-Wang flaw resolution.
- Reproduction: TBD → real bash recipe with both canonical + MW
  drivers.
- Provenance: split into external (em-nanda + papers) and internal
  (final-branch commits).
- AUTO-RESULTS rendered for 7/8 cells via direct
  `experiments.c6_em.analysis.run_analysis()` + `_replace_auto_results`.
  Bypassed the framework's `report.render(component='c6')` because
  3 c6_* dirs trip the "one dir per component" check (see Open
  questions below).
- analysis.py:_decision fixed to use `abs(gap)` for band classification
  (negative gap = TXC wins). Decision now correctly classifies
  -6.14 as "Mixed" not "Tied".

(D) Tests: 131/131 green (124 + 7 from agent_paper's MW addition).

## Next action (agent owns — overwrite)

**Mission COMPLETE per Han's STAND DOWN (2026-05-05 PM).** Following
the explicit STAND DOWN action items 1-4 from the new mandate
section above.

1. ✅ **Final canonical cell landed** (TXC seed=1 7B-medical,
   peak_align=72.64; bash `bqp1ssnty` exit 0 at 13:03 UTC). c6.md
   AUTO-RESULTS re-rendered with 8/8 cells.
2. ✅ **MW sweep killed** at 14:18Z (had launched at 13:15Z; ran ~1 hr
   before kill). MW driver `experiments/c6_em_mw/run.py` stays
   in-tree as inert reserve per Han's directive ("don't delete; just
   don't launch").
3. ✅ **c6.md status: complete** (commit `b59834eb`). Hypothesis locked
   to data-driven outcome ("TXC strictly beats SAE on both organisms,
   both Mixed band"). FLOPs-asymmetry caveat rewritten to reflect the
   literature-alignment resolution.
4. ✅ **HF backup COMPLETE** (commit `62f58262`):
   - 17 agent_em qwen checkpoints pushed to
     `han1823123123/temp-bench-models`. Verified via
     `api.list_repo_files`.
   - 8 canonical (paper headline): `9778d10381696f58`,
     `754166d1711923c1`, `5e4e188045d5d3c8`, `672dbf61896f7843`,
     `c0da3ed8794554a1`, `88a4ddf6819d8057`, `9b011dfeea88f8af`,
     `2016074933c41e7f`.
   - 9 historical (abbreviated-Wang era + MW smoke + a partial
     MW cell that landed before the kill).
   - Manifest hf_url field backfilled for all 17 rows.

**C6 mission: COMPLETE.** No further compute needed. The pod is
safe to stop at any time — all 17 of my checkpoints are on HF; all
results in the leaderboard; AUTO-RESULTS rendered; status locked
to `complete` in c6.md.

Use any remaining session time for paper-writing contributions to
c6.md if Han wants more methodology detail on the per-step training
-FLOPs literature-alignment caveat. Otherwise, idle.

## Don't repeat (agent owns — overwrite)

- **DON'T cite the 1.0.0 abbreviated rows as headline.** Headlines
  are 2.0.0 only. analysis.py's `canonical_train_keys` + protocol
  filter handles it; if you bypass via direct leaderboard query,
  filter manually.
- **DON'T edit `pyproject.toml` / `uv.lock` / `configs/locked_archs.yaml`
  / `agents/README.md` / `docs/paper/*` / other agents' dirs.**
  Cross-territory. Han's 2026-05-04 paper-agent authorisation lets
  any agent port a blocking arch — narrow exception. The c6 hparam
  override has already landed.
- **DON'T merge `em-nanda` into `final`** — decision #4 forbids it.
  Cross-branch reads only (`git show origin/em-nanda:<path>`).
- **DON'T bypass `runner.run_cell`.** Single canonical pathway.
- **DON'T forget `TQDM_DISABLE=1`.** Hard Rule #8.
- **DON'T forget the GPU sharing convention** (`scripts/run_on_gpu.sh`,
  PROTOCOL.md §13). Verify peer is idle in their briefing +
  nvidia-smi before launching; update your Current state with the
  borrow window. Lesson from 14:09 incident: peer's idle status
  ages — they may start mid-borrow. Recheck at long-borrow start
  + treat the borrow as preemptible.
- **DON'T re-resolve jsonl conflicts manually.** Always re-run the
  dedupe + JSON-validate Python snippet (see
  `What I just did` § "incidents recovered" for the recipe). Stale
  conflict markers crashed SAE seed=42 7B mid-cell.
- **DON'T launch a Wang cell when the Anthropic API is suspect.**
  Defensive `RuntimeError` at stage-3 baseline catches the all-None
  case but you lose the time to that point. If the credit balance
  alert fires, hold launches until Han confirms top-up.

## Open questions for Han (agent owns — overwrite)

1. **Framework `_experiment_dir` multi-dir error** (NEW 2026-05-05):
   `temp_bench.report.render(component='c6')` raises RuntimeError
   "Multiple experiment dirs match c6_*" because three sub-dirs now
   exist:
   - `experiments/c6_em/` (mine — canonical analysis.py).
   - `experiments/c6_em_100k/` (agent_em_100k's — they pivoted to
     C3 MW per commit `4217f4ba`, so this dir may be deprecated).
   - `experiments/c6_em_mw/` (mine — MW driver).
   Convention says "one dir per component". I'm bypassing via
   `from experiments.c6_em.analysis import run_analysis` +
   `_replace_auto_results` (see Next action #2). Either:
   (a) framework picks a canonical analysis.py per component;
   (b) the multi-dir check relaxes;
   (c) I consolidate into one dir (break agent_em_100k's existing
   dir and migrate MW into c6_em/ as `run_mw.py`).
   Not blocking — bypass works — but agent_paper should pick a
   convention. **Surfaced; awaiting decision.**

## Other precision notes for the next instance

- `experiments/c6_em/run.py` uses `--seed N` (singular). The earlier
  Han-decision §5 example wrote `--seeds N`; that's a typo.
  `--n-steps` was REMOVED in commit `bdb88829` per decisions § 12 —
  don't try to override n_steps from the CLI; use TrainingConfig().
- `sae_arditi` and `txc_base` checkpoints under the new
  TrainingConfig defaults (batch=1024 / 25K) are bit-identical
  whether you build them with the mmap or the `.clone()` preload —
  determinism preserved by `np.random.default_rng(seed)` indices +
  fp32 contract.
- The `analysis.py` filter logic depends on TWO matches: (1)
  `canonical_train_keys` (drops batch=256 cells); (2)
  `eval_protocol_version=="2.0.0"` (drops 1.0.0 abbreviated). Both
  needed because in principle a 1.0.0 row could exist with a 1024
  batch (didn't happen here, but defensive).
- Bash IDs of the 8 paper-headline cells (commit citations + headline
  numbers) are in the "Headline" + "Bash IDs" tables in Current state.
- All 4 14B-finance cells use train_key cache hits if you re-run
  them under the canonical TrainingConfig. The 4 7B-medical cells
  also have train_key cache hits — only the Wang stages re-run.
- Stage-2 + stage-3 + stage-4 + stage-3-baseline all persist
  judge_outputs.jsonl with a `stage` tag (literal "2", "3",
  "3-baseline", "4"). Post-deadline κ validation can scan by stage.
- `EVAL_PROTOCOL_VERSION = "2.0.0"` lives at `experiments/c6_em/run.py`
  near the top, NOT in the schema. The schema's
  `eval_protocol_version` is just a string field.
- 9 legacy 1.0.0 leaderboard rows (batch=256, abbreviated Wang) STAY
  in `results/leaderboard.jsonl` for diff comparison. Don't delete
  them — they're our own audit trail of the methodological-flaw fix.
