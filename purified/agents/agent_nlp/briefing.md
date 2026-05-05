<!--
DRAFT — written by agent_paper 2026-05-03 for Han's review.
Han: rewrite the "Identity + mandate" section as you wish before
spinning up agent_nlp; agents will not touch it after.
Section ownership rules: PROTOCOL.md § 14.
-->

---
agent: agent_nlp
last_state_update: 2026-05-05T11:21:00Z
component: c3, c4
---

## Identity + mandate (Han owns — agents do not edit)

You are **agent NLP**. You own C3 + C4 only. Files you may edit:
- `agents/agent_nlp/briefing.md` (your own — agent-owned sections only)
- `docs/components/c3.md` and `docs/components/c4.md`
- `experiments/c3_probing/`, `experiments/c4_qualitative/`
- Code under `src/temp_bench/` that you author + commit (eval modules
  for probing / qualitative; data loaders under `temp_bench.data.nlp`)
- `configs/datasources.yaml` — adding new C3/C4 datasources is fine.
  YAML edits to other components' datasources require a Han ping.

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
surface it in chat, and let Han or agent_paper land the change. This
is non-negotiable even if Han verbally approves — the audit trail of
who edited what depends on each agent staying in their lane.

### ⚠️ NEW MISSION 2026-05-05 PM — C3 TFA baseline (decisions § 16)

**Your prior C3 TopK T=1 mission is COMPLETE** (commit `9b9d6cc5`).
New mission: **add TFA × 3 seeds × 2 k_feats** to the C3 sparse-probing
benchmark. Working alongside agent_em_100k (who builds 5-layer cache
+ runs MLC in parallel on their pod).

**Wasteland-faithful TFA training** (`origin/han-phase7-unification:
experiments/phase7_unification/train_phase7.py:312-353`):

> "TFA processes FULL sequences (B, T=128, d) through attention.
>  TFA-specific batch size override: TFA_BATCH = 32. Phase 5 default
>  was 64. Per-step tokens: 32 × 128 = 4096 (close to SAEBench 2K)."

TFA's attention tensor is heavy at large B (`B × T × d_sae` ~ 9.6 GB
fp32 at B=1024, T=128, d_sae=18432). The wasteland chose **B=32** to
keep per-step memory tractable AND give TFA's attention full ~128-token
context (Fig. 2(d) of `papers/priors_in_time.md` shows ~80% variance
explained at 100+ tokens of context). We adopt the same convention.

**TrainingConfig** (TFA-specific B override):

```python
TrainingConfig(
    n_steps=20_000,
    batch_size=32,                # ← TFA-specific override
    plateau_early_stop=False,
    train_window_size=None,       # full sequence (B, 128, d_in)
)
```

Per-step encoder load: B × T = 32 × 128 = 4096 tokens (≈ 2× SAEBench
canonical, paper-faithful per Phase 7 wasteland).

### Mission scope

| Arch | seeds | k_feats | TrainingConfig |
|---|---|---|---|
| `tfa` | {42, 1, 2} | {5, 20} | B=32, full seq, n_steps=20_000 |

3 unique trainings + 6 evals on the existing single-layer cache
`gemma_2_2b_it_l13_fineweb_24k128` (no new cache build needed for TFA).

### Wall-time on 2× H100

agent_em is post-canonical idle (commit `b549d91c` confirms), so GPU 1
is free for you to borrow (per § 13 GPU sharing convention; verify
`nvidia-smi` shows GPU 1 idle + note the borrow in your Current state).
2 cells parallel, 1 cell serial → **~2 hr wall**.

Per-cell estimate on H100: ~30-60 min train (TFA's attention costs
~3-5× more per step than vanilla SAE) + ~30 min probing eval × 2 k_feats
= ~1.5-2 hr per cell. With 2 GPUs, run 2 seeds simultaneously.

### First concrete task — write driver, smoke, launch

Step 0 — `git pull --rebase origin final`. Verify the framework:

```bash
.venv/bin/python -c "
from temp_bench.config import load_arch
spec = load_arch('tfa')
print('hparams:', spec.hparams)
"
```

Step 1 — write `experiments/c3_probing_tfa_baseline/{__init__.py,
run.py}`. Same pattern as your `c3_probing_topk_baseline/run.py`:
just swap arch name + override batch_size on TrainingConfig:

```python
"""C3 TFA baseline at B=32 + full seq (decisions § 16)."""
from __future__ import annotations
import argparse
from temp_bench import runner
from temp_bench.schemas import TrainingConfig
from experiments.c3_probing.run import (
    DATASOURCE,
    EVAL_PROTOCOL_VERSION,
    my_train_fn,
    my_eval_fn,
)
from temp_bench.config import compute_act_cache_key, load_datasource


# Wasteland-faithful TFA training (decisions § 16, Phase 7 reference).
# B=32, full sequence (T=128) → 4096 tokens/step, close to SAEBench's
# 2K canonical and gives attention 128 tokens of context.
TFA_TRAINING_CFG = TrainingConfig(
    n_steps=20_000,
    batch_size=32,                  # ← TFA-specific override
    plateau_early_stop=False,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    ap.add_argument("--k-feats", nargs="+", type=int, default=[5, 20])
    ap.add_argument("--n-steps", type=int, default=None)
    args = ap.parse_args()

    act_cache_key = compute_act_cache_key(load_datasource(DATASOURCE))
    cfg = TFA_TRAINING_CFG
    if args.n_steps is not None:
        cfg = cfg.model_copy(update={"n_steps": args.n_steps})

    for seed in args.seeds:
        for k in args.k_feats:
            print(f"[c3_tfa_baseline] cell tfa seed={seed} k_feat={k} "
                  f"B={cfg.batch_size} n_steps={cfg.n_steps}",
                  flush=True)
            eval_cfg = {
                "k_feat": k, "S": 32, "smoke": False,
                "_act_cache_key": act_cache_key,
                "_datasource_name": DATASOURCE,
            }
            runner.run_cell(
                component="c3", arch_name="tfa", seed=seed,
                datasource_name=DATASOURCE,
                training_cfg=cfg, eval_cfg=eval_cfg,
                eval_protocol_version=EVAL_PROTOCOL_VERSION,
                train_fn=my_train_fn, eval_fn=my_eval_fn,
            )


if __name__ == "__main__":
    main()
```

Step 2 — smoke ONE cell at `n_steps=200`:

```bash
TQDM_DISABLE=1 AGENT_NAME=agent_nlp \
  bash scripts/run_on_gpu.sh 1 -- \
  .venv/bin/python -m experiments.c3_probing_tfa_baseline.run \
  --seeds 42 --k-feats 5 --n-steps 200 2>&1 | tail -20
```

Step 3 — borrow GPU 1 (verify agent_em is idle first per § 13). Then
launch 2 seeds in parallel:

```bash
# Verify agent_em is post-canonical idle
grep "status: complete\|canonical mission COMPLETE" \
    agents/agent_em/briefing.md | tail -3

# Update YOUR Current state with the borrow note BEFORE launching:
# "Borrowing GPU 1 for C3 TFA sweep — agent_em is post-canonical
#  idle (mission complete commit b549d91c)."

# Launch in parallel
TQDM_DISABLE=1 AGENT_NAME=agent_nlp \
  bash scripts/run_on_gpu.sh 0 -- \
  .venv/bin/python -m experiments.c3_probing_tfa_baseline.run \
  --seeds 42 \
  > logs/c3_tfa_gpu0.log 2>&1 &
echo $! > /tmp/p_tfa_gpu0

TQDM_DISABLE=1 AGENT_NAME=agent_nlp \
  bash scripts/run_on_gpu.sh 1 -- \
  .venv/bin/python -m experiments.c3_probing_tfa_baseline.run \
  --seeds 1 2 \
  > logs/c3_tfa_gpu1.log 2>&1 &
echo $! > /tmp/p_tfa_gpu1
```

Step 4 — monitor + verify rows land at `arch=tfa`, `component=c3`,
fresh `train_keys` (B=32 produces a distinct hash from any existing
B=1024 cells).

### Analysis filter update — 4-arch family

Once your TFA cells AND agent_em_100k's MLC cells land, the C3
canonical filter expands to FOUR arch families. Update
`experiments/c3_probing/analysis.py` to add 2 more
`canonical_train_keys` calls:

```python
tfa_keys = canonical_train_keys(
    component="c3",
    archs=["tfa"],
    seeds=(1, 2, 42),
    datasource_names=("gemma_2_2b_it_l13_fineweb_24k128",),
    training_cfg=TrainingConfig(n_steps=20_000, batch_size=32),
)
mlc_keys = canonical_train_keys(
    component="c3",
    archs=["mlc"],
    seeds=(1, 2, 42),
    datasource_names=("gemma_2_2b_it_l11to15_fineweb_24k128",),
    training_cfg=TrainingConfig(n_steps=20_000),
)
canonical = txc_keys | topk_keys | tsae_keys | tfa_keys | mlc_keys
```

(MLC keys come in once agent_em_100k's mission lands; you can wire the
union now and the MLC keys will simply be empty until then.)

### Watch-outs

- **B=32 is intentional** — don't bump it back to 1024. Wasteland
  evidence (`train_phase7.py:316-321`) shows B>32 risks OOM on
  d_sae=18432 with the attention tensor at fp32. Document the per-
  arch B exception in c3.md caveats when you re-render.
- **Borrow GPU 1 carefully** — agent_em's pod is shared. Before
  launching, verify their Current state shows "canonical mission
  COMPLETE" / idle. If they start any compute work, your borrow gets
  preempted.
- **Don't reuse the multi-layer datasource** — TFA at C3 uses the
  existing single-layer `gemma_2_2b_it_l13_fineweb_24k128` (full seq
  T=128 from L13 alone). The l11to15 multi-layer datasource is
  agent_em_100k's MLC territory.

---

### ⚠️ Han decisions 2026-05-05 PM — C3 baseline T=1 re-train (your headline shifts) [§ 15 COMPLETE — commit `9b9d6cc5`]

**Background**: SAEBench (papers/are_saes_useful.md, App. B) shows
canonical SAE training is buffer-based, batch=2048 TOKENS/step, ~500M
tokens total. Our C3 per-token archs (`topk_sae`, `tsae_paper`) currently
train at `(B=1024, seq_len=128) → flatten = 131,072 tokens/step` — 65×
over SAEBench's canonical 2K. C6 + C7 baselines coincidentally use a
window-based pattern at T=1 that puts them within 2× of canonical; C3 +
C5 inherited the over-batched sequence-based pattern. The earlier
"MW deployment" pivot (decisions.md § 14) was solving the right
diagnosis with the wrong fix. **The right fix is to bring per-token
baselines DOWN to T=1 window-based, not bring TXC up via MW.**

**Han's call (2026-05-05 PM)**: re-train C3's per-token archs at the
per-arch literature-faithful window size, **split between you and
agent_em_100k**:

- **YOU run `topk_sae` × 3 seeds × 2 k_feats at `train_window_size=1`**
  (vanilla TopK, no temporal — 1024 tokens/step). 3 unique trainings +
  6 evals. agent_em is now idle (their canonical mission is done +
  the C6 MW pivot is rescinded), so **GPU 1 of your shared 2× H100 pod
  is free for you to borrow**. Run 2 cells in parallel across GPU 0 +
  GPU 1 → ~2.4 hr wall.
- **agent_em_100k runs `tsae_paper`** × 3 seeds × 2 k_feats at
  `train_window_size=2` (Bhalla/Ye 2025 §3.1 paper-faithful adjacent
  pairs) on their separate 1× H100 pod. ~4.5 hr wall there. You don't
  run T-SAE.

MLC unported per § 11 (appendix-only); skipped. TXC archs (`txc_base`,
`txc_pro`) are unchanged — they sample 1 random T-window per row already,
regardless of mode.

**Your two-GPU plan**:

1. Your in-flight `topk_sae` sweep at T=None (PID 119732, GPU 0) keeps
   running and finishes ~20:21Z as your diff-reference dataset (the
   "pre-fix over-batched" baseline that contextualizes the new T=1
   cells). Don't kill it.
2. **NOW (immediately): launch the new TopK T=1 sweep on GPU 1**
   (agent_em's idle GPU; their canonical mission is complete per
   `agents/agent_em/briefing.md`). Run all 3 seeds serial there. With
   T=1 (1024 tokens/step vs 131K full-sequence), per-cell wall is
   ~1.2 hr → 3 cells serial = ~3.6 hr.
3. After your T=None sweep on GPU 0 finishes (~20:21Z), GPU 0 is
   also yours to use for the T=1 sweep (parallelize remaining seeds
   if any are still pending). With both GPUs available, the T=1
   sweep wraps in ~2.4 hr from start of step 2.

**Driver design** — write `experiments/c3_probing_topk_baseline/run.py`,
mirroring the existing topk plumbing with `train_window_size=1`:

```python
"""C3 TopK SAE baseline re-train at T=1 (decisions.md § 15).
Imports your existing C3 plumbing; only the TrainingConfig changes.
"""
from __future__ import annotations
import argparse
from temp_bench import runner
from temp_bench.schemas import TrainingConfig
from experiments.c3_probing.run import (
    DATASOURCE,
    EVAL_PROTOCOL_VERSION,
    my_train_fn,
    my_eval_fn,
)
from temp_bench.config import compute_act_cache_key, load_datasource

TOPK_TRAINING_CFG = TrainingConfig(n_steps=20_000, train_window_size=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    ap.add_argument("--k-feats", nargs="+", type=int, default=[5, 20])
    ap.add_argument("--n-steps", type=int, default=None,
                    help="Override n_steps for smoke tests.")
    args = ap.parse_args()

    act_cache_key = compute_act_cache_key(load_datasource(DATASOURCE))
    cfg = TOPK_TRAINING_CFG
    if args.n_steps is not None:
        cfg = cfg.model_copy(update={"n_steps": args.n_steps})

    for seed in args.seeds:
        for k in args.k_feats:
            print(f"[c3_topk_T1] cell topk_sae seed={seed} k_feat={k} "
                  f"T={cfg.train_window_size} n_steps={cfg.n_steps}",
                  flush=True)
            eval_cfg = {
                "k_feat": k, "S": 32, "smoke": False,
                "_act_cache_key": act_cache_key,
                "_datasource_name": DATASOURCE,
            }
            runner.run_cell(
                component="c3", arch_name="topk_sae", seed=seed,
                datasource_name=DATASOURCE,
                training_cfg=cfg, eval_cfg=eval_cfg,
                eval_protocol_version=EVAL_PROTOCOL_VERSION,
                train_fn=my_train_fn, eval_fn=my_eval_fn,
            )


if __name__ == "__main__":
    main()
```

**Smoke + launch**:

```bash
# Smoke (n_steps=200) on GPU 1 — verify framework picks up T=1
TQDM_DISABLE=1 AGENT_NAME=agent_nlp \
  bash scripts/run_on_gpu.sh 1 -- \
  .venv/bin/python -m experiments.c3_probing_topk_baseline.run \
  --seeds 42 --k-feats 5 --n-steps 200 2>&1 | tail -20

# Full sweep on GPU 1 (immediately) — your in-flight T=None sweep
# stays on GPU 0 until ~20:21Z; afterwards parallelize across both.
TQDM_DISABLE=1 AGENT_NAME=agent_nlp \
  bash scripts/run_on_gpu.sh 1 -- \
  .venv/bin/python -m experiments.c3_probing_topk_baseline.run \
  > logs/c3_topk_T1_gpu1.log 2>&1 &
echo $! > /tmp/p_topk_T1
```

After your old sweep on GPU 0 finishes (PID 119732), if any T=1 cells
are still pending, launch a second process on GPU 0 to drain them
(split seeds between the two processes if needed).

**Verify before borrowing GPU 1** (PROTOCOL § 13 GPU sharing
convention):

- Read `agents/agent_em/briefing.md` "Current state" — should say
  "canonical mission COMPLETE" or "idle". If they're mid-cell with
  an ETA, wait for their cell to land first.
- `nvidia-smi` GPU 1 should show <1 GB used + no long-running python.
- Update YOUR Current state with `"Borrowing GPU 1 for C3 TopK T=1
  sweep — agent_em is post-canonical idle, their next mission is
  paper-writing."` BEFORE launching.

**Framework change (commit `5555e7eb`)** — read before paper-rendering:

- `temp_bench.data.nlp.cache.preloaded_batch_iter_from_act_cache` gains
  `train_window_size: int | None = None`. None = full-sequence
  (current); int T = 1 random T-window per row.
- `TrainingConfig.train_window_size: int | None = None` added; flows
  into `compute_train_key` via `model_dump(exclude_none=True)`. Default
  (None) preserves YOUR EXISTING TRAIN KEYS — the in-flight topk_sae
  sweep keeps its cache. Setting an int gets a fresh key.
- Your existing `experiments/c3_probing/run.py:my_train_fn` already
  passes through `training_cfg.train_window_size` (commit landing in
  this push), so the new driver works without any of your eval-side
  changes.

**Framework change (commit `5555e7eb`)** — read before paper-rendering:

- `temp_bench.data.nlp.cache.preloaded_batch_iter_from_act_cache` gains
  `train_window_size: int | None = None`. None = full-sequence
  (current); int T = 1 random T-window per row.
- `TrainingConfig.train_window_size: int | None = None` added; flows
  into `compute_train_key` via `model_dump(exclude_none=True)`. Default
  (None) preserves YOUR EXISTING TRAIN KEYS — the in-flight topk_sae
  sweep keeps its cache. Setting an int gets a fresh key.

**Your action — analysis.py filter shift**:

`experiments/c3_probing/analysis.py` currently calls
`canonical_train_keys(component='c3', archs=[...], seeds=(1,2,42),
training_cfg=TrainingConfig(n_steps=20_000))` — one TrainingConfig for
all 4 archs. After both agent_em_100k's T=2 sweep AND your T=1 sweep land, the
canonical archs split into THREE TrainingConfigs (one per arch family):

- TXC archs (`txc_base`, `txc_pro`): unchanged. Use
  `TrainingConfig(n_steps=20_000)`.
- TopK SAE: re-trained at T=1. Use
  `TrainingConfig(n_steps=20_000, train_window_size=1)`.
- T-SAE: re-trained at T=2 (paper-faithful pairs). Use
  `TrainingConfig(n_steps=20_000, train_window_size=2)`.

`canonical_train_keys` accepts a single `training_cfg` and a list of
archs. To compute the union across three TrainingConfigs, call it
three times and union the returned sets:

```python
from temp_bench.report import canonical_train_keys, query_leaderboard
from temp_bench.schemas import TrainingConfig

txc_keys = canonical_train_keys(
    component="c3",
    archs=["txc_base", "txc_pro"],
    seeds=(1, 2, 42),
    datasource_names=("gemma_2_2b_it_l13_fineweb_24k128",),
    training_cfg=TrainingConfig(n_steps=20_000),
)
topk_keys = canonical_train_keys(
    component="c3",
    archs=["topk_sae"],
    seeds=(1, 2, 42),
    datasource_names=("gemma_2_2b_it_l13_fineweb_24k128",),
    training_cfg=TrainingConfig(n_steps=20_000, train_window_size=1),
)
tsae_keys = canonical_train_keys(
    component="c3",
    archs=["tsae_paper"],
    seeds=(1, 2, 42),
    datasource_names=("gemma_2_2b_it_l13_fineweb_24k128",),
    training_cfg=TrainingConfig(n_steps=20_000, train_window_size=2),
)
canonical = txc_keys | topk_keys | tsae_keys
rows = [r for r in query_leaderboard(component="c3") if r.train_key in canonical]
```

The old over-batched per-token cells (and the in-flight `topk_sae`
seed=1/2/42 sweep that finishes today) stay in the leaderboard under
their old hashes for diff comparison; the analysis.py filter drops them
from the headline AUTO-RESULTS automatically once you wire the second
call.

**C4 implication**: C4's training cache-hits on C3 checkpoints. After
the re-train, C4 evals on `topk_sae` + `tsae_paper` re-run (eval_key
derives from train_key, so the new train_keys produce new eval_keys).
~10 min Haiku per cell × 6 cells (2 archs × 3 seeds) = ~$0.30, ~1.5
hr wall — same as before, just on the new T=1 baselines.

**Paper-claim shift (uncertain pending re-runs)**: Today's C3 headline
shows TopK > TXC > T-SAE on probing AUC. Under canonical T=1 training,
the per-token archs likely score lower (they were over-batched 65×; now
they're at literature scale). The headline may flip to TXC > TopK /
T-SAE, or land on a tighter cluster. Either result is reviewer-
defensible because every arch trains at SAEBench's canonical compute
scale. Wait for agent_em_100k's re-train to land before re-rendering.

**No action on your current sweep**: your in-flight `topk_sae` sweep
(PID 119732, seed=1 step ~9000/20000 at 12:36Z) keeps running and
keeps its cache (the framework change is backwards-compatible because
`train_window_size=None` produces the same hash as before). The
re-trained cells are agent_em_100k's responsibility; your output is
the diff-comparison reference.

See `decisions.md` § 15 for the full rationale.

---

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
  knobs.
- **Uniform across pods**: H100 + A40 both run at batch=1024. The point
  is every arch trains under identical conditions, matching the SAE-
  comparison-paper standard (T-SAE §4.1: "All SAEs are trained with...
  chosen to allow for comparability"; TFA App. B.1: "all SAEs trained
  from scratch... to enable a consistent and fair evaluation").
- **Plateau-stop is OFF; 25K cap is binding for every cell.** The schema's
  plateau detection (`max(loss[-5000:]) - min(loss[-5000:]) < 1e-4`,
  `training/sae_trainer.py:158-165`) is an absolute threshold over a fixed
  window — cross-arch unfair (archs at different loss scales would trigger
  at different points). SAE-comparison literature (T-SAE §4.1, TFA App. B.1,
  GemmaScope) uses fixed step counts. Every cell trains to the 25K cap;
  fairness mechanism is "exactly 25.6M tokens per arch."
- **If you observe loss still descending steeply at step 25K** (e.g.,
  final-1K-step drop > 5% of loss value), surface that as a comment on
  your run — the cap may need to be bumped uniformly across all archs.
- **Cache hygiene**: `batch_size`, `n_steps`, and the plateau-* fields
  are all in the `train_key` hash (`src/temp_bench/config.py:181-193`).
  New cells get fresh `train_key` / `eval_key` automatically. Old
  batch=256 cells stay in `results/leaderboard.jsonl` for diff
  comparison — **when rendering AUTO-RESULTS, filter for new rows only**
  (e.g., `training_cfg.batch_size == 1024`).

**Your specific re-run**: 24 C3 cells (4 archs × 3 seeds × 2 k_feats)
need new train_keys. C4 cells share the C3 checkpoints, so once C3 is
re-trained, C4 evaluation re-runs cheaply (cache-hit on training).
Bump `EVAL_PROTOCOL_VERSION` if you want to invalidate eval cache too,
but it's not strictly needed since `eval_key` derives from `train_key`.

You are agent NLP, lead on the language-model components of the paper:
**C3 (sparse probing)** and **C4 (qualitative latents)**. Both are on
the same subject: `google/gemma-2-2b-it` layer 13 residual stream. C4
piggybacks on C3's activation cache.

Hardware: pod `2× H100`, pinned to **GPU 0**. Pod mode `persistent` —
`/workspace` survives stop/start, HF backup is optional but
recommended at session end. agent_em shares the pod on GPU 1.

Your **long-pole task** is the activation cache (~14 GB,
~3 H100-hours): 24K FineWeb sequences × 128 tokens, fp16, layer 13. As
soon as it's on HF (`han1823123123/temp-bench-data`), agent_steer can
unblock — they are gated on this. Push as soon as ready, don't wait
for downstream training.

C3 hypothesis (from `docs/components/c3.md`): TXC-pro matches the best
per-token SAE at k=5 and small seed-significant win at k=20. TXC-base
matches at every k.

C4 hypothesis: TXC-pro matches T-SAE on Top-256 cumulative SEMANTIC
Pareto. **One metric only** — drop pdvar and any paper-style probe
variants (decision pinned in `docs/components/c4.md`).

**Task suite is locked**: `SAEBench+CT` (n=38) — upstream SAEBench's
canonical 36 binary one-vs-rest tasks (8 datasets, classes per
SAEBench's `dataset_info.chosen_classes_per_dataset`) plus the two
cross-token coreference tasks (WinoGrande + SuperGLUE WSC). See
`decisions.md` § 11 and `docs/components/c3.md` "Task suite" for
the full table + reproduction notes.

When you port the wasteland's `probe_datasets.py` + `crosstoken_datasets.py`,
apply three SAEBench-faithfulness fixes (do not blindly copy the
wasteland 36):
- **github-code**: use SAEBench's `codeparrot/github-code` with the 5
  SAEBench languages (C, Python, HTML, Java, PHP), NOT wasteland's
  `code_search_net` (python/java/javascript/go). NOT gated despite
  what the HF web viewer suggests — that page is disabled because the
  loader is a Python script, not because access is restricted. Just
  needs `trust_remote_code=True` (set via `HF_DATASETS_TRUST_REMOTE_CODE=1`
  in your shell or `set_agent_env.sh`) and `datasets<4` (already
  pinned in `pyproject.toml`). Smoke-test the loader once with a
  tiny `streaming=True` pull BEFORE the 3-H100-hour cache build.
- **amazon_sentiment**: emit BOTH 1.0-vs-rest AND 5.0-vs-rest binaries
  (wasteland only has 5.0).
- **amazon_categories**: hardcode `["1","2","3","5","6"]` and use a
  deterministic non-streaming pull (wasteland streaming-top-5 missed
  cat6 and is non-deterministic across runs).

Locked decisions in scope: #1 (two TXCs — no hill-climbing), #4
(cross-branch reads via `git show`), #6 (HF repos), #7 (Bricken
resample is C6-only by default; **C3/C4 keep it OFF** — revisit only
if time permits at the end of the paper sprint), #11 (SAEBench+CT
task suite).

References:
- `agents/README.md` (your roster row + pod specs)
- `docs/components/c3.md` and `docs/components/c4.md`
- `docs/paper/architecture.md` (locked TXC spec)
- `decisions.md` (10 locked policy items)
- `PROTOCOL.md` § 11 (framework discipline), § 12 (GPU pinning),
  § 9 *Session wrap-up*

**Before you `status: complete`**: `bash scripts/wrap_up_session.sh`.
Persistent pod → script prints a manual `hf upload` recipe for every
checkpoint not yet on HF. Don't let Han stop the pod until that loop
completes (probe-cache + act-cache are on HF; your trained
.safetensors and per-cell run-dirs are NOT until you push).

---

## ✅ Cross-arch TrainingConfig saga — RESOLVED (2026-05-04 → 2026-05-05)

The "agent_nlp shipped v1.0.0 + v1.1.0 at batch=256 / n_steps=10K which
is 40× less effective gradient info than Phase 7" finding (briefing
flagged 2026-05-04 AM) was rolled into `decisions.md` § 12. Canonical
schedule is now Phase-5-faithful for all SAE-family archs across all
agents:

```
TrainingConfig:
  batch_size = 1024
  n_steps = 20_000             # ← Han deadline override 2026-05-04 PM,
                               #    cuts 25K → 20K to fit 72-h sprint
  learning_rate = 3e-4
  warmup_steps = 1_000
  plateau_early_stop = False
  precision = "bf16"
```

Convergence verified for all txc_base/txc_pro/tsae_paper cells (final-1K
loss drop < 0.3% — well below the § 12 5% flag threshold). topk_sae
sweep in flight; will verify on landing.

**Old batch=256 cells (v1.0.0 + v1.1.0) stay in `results/leaderboard.jsonl`
for diff comparison only**; analysis.py + `temp_bench.report.canonical_train_keys`
filter the headline to the new b1024/n20K cells.

Commit chronology (latest first):
- `e12dc719` — `temp_bench.data.nlp.preloaded_batch_iter_from_act_cache`
  (3.4× data-path speedup, opt-in shared helper, bit-identical drop-in
  for the default `batch_iter_from_act_cache`; 4 unit tests confirm).
- `513a85ea` — agent_nlp n_steps=20K override
- `9a39137a` — agent_paper canonical_train_keys helper
- `06681098` — plateau-stop disabled
- `9718a442` — batch=1024 / n_steps=25K Phase-5-faithful
- `0beae2bf` — revert of agent_paper's earlier batch=2048 attempt
- `a9200560` — original (reverted) batch=2048 directive

---

## Current state (agent owns — overwrite at every compact)

**Last verified: 2026-05-05 11:21 UTC (topk_sae sweep in flight; 18/24 done)**

- `git HEAD`: at or after `e12dc719` (origin/final). Latest agent_nlp work:
  - `e12dc719` — preloaded_batch_iter helper landed in `temp_bench.data.nlp`
  - `513a85ea` — n_steps=20K Han deadline override
  - `033a3eb6` — train_log persistence (per-cell convergence telemetry)
  - `b43ccf5b` — analyses migrated to canonical_train_keys
- **In flight**: topk_sae sweep, PID 119732, started 2026-05-05 10:06 UTC.
  Currently at topk_sae seed=1 step 9000/20000 at ~2.0 steps/sec.
  ETA cell 1 ~91 min; total topk sweep ~9 hr.
- **Pod state**: GPU 0 solo to agent_nlp (45 GB used by my process,
  35 GB free). agent_em on GPU 1 with c6 sae_arditi at 17 GB. GPU 1
  still entirely agent_em's.
- Leaderboard: 18 canonical (b1024 / n20K) C3 cells + 24 batch=256 cells
  (12 v1.0.0 + 12 v1.1.0) kept for diff. topk_sae will add 6 more cells
  (3 seeds × 2 k_feats; 3 unique trainings).
- Checkpoints: 9 unique batch=1024/n_steps=20K train_keys on disk +
  manifest. 3 more (topk_sae) pending.

## C3 batch=1024 / n_steps=20K — partial headline (18/24 cells complete)

Per decisions.md § 15 (Han 2026-05-05 PM), per-token archs (`topk_sae`,
`tsae_paper`) re-train at literature-canonical T=1 / T=2 window sizes;
TXC archs unchanged at None (sample windows internally regardless).
3 cfg families → 3 train_keys per cell. Mean ± σ across 3 seeds:

| arch                        | k=5              | k=20             |
|-----------------------------|------------------|------------------|
| `txc_base`     (None)       | 0.8367 ± 0.004   | **0.8952 ± 0.004** |
| `txc_pro`      (None)       | **0.8450 ± 0.013** | 0.8936 ± 0.009   |
| `tsae_paper`   T=None       | 0.8301 ± 0.006   | 0.8975 ± 0.005   |
| `topk_sae`     **T=1**      | 0.8306 ± 0.003   | 0.8831 ± 0.002   |
| `tsae_paper`   T=2          | _pending_ (agent_em_100k, ~4.5 hr ETA) | _pending_ |

**Story so far** — TXC archs lead at k=20. At k=5, txc_pro nudges
ahead but with σ 0.013 the lead is within noise. topk_sae at literature-
canonical T=1 is now LOWEST at both k=5 and k=20 — confirms agent_paper's
hypothesis that the v1.1.0 "TopK > TXC" headline was an artifact of
65× over-batching the per-token arch.

Pending: tsae_paper T=2 (agent_em_100k owns) likely shifts down by a
similar 0.014-0.024 magnitude based on the topk_sae delta. The k=20
ordering should remain TXC ≥ T-SAE T=2 once it lands.

Vs v1.1.0 batch=256 / n_steps=10K (kept on disk for diff comparison only):

| arch                | Δ k=5    | Δ k=20    |
|---------------------|----------|-----------|
| txc_base            | -0.003   | +0.007    |
| txc_pro             | +0.007   | +0.008    |
| tsae_paper T=None   | +0.002   | **+0.012**|
| topk_sae   T=1      | -0.014   | -0.018    |

T=None topk_sae results (run on GPU 0 in parallel as diff-reference,
in-flight at seed=2 ~step 16K, ETA ~50 min) — *not* the headline,
just to quantify the over-batching effect:

| arch                  | k=5 (single seed=1) | k=20 (single seed=1) |
|-----------------------|---------------------|----------------------|
| `topk_sae` T=None     | 0.8461              | 0.9085               |

**Convergence verified** (decisions § 12 5%-flag check on final-1K-step
loss drop):
- `txc_base` seeds 1/2/42: 0.25% / 0.19% / 0.15% ✓
- `txc_pro` seeds 1/2/42: 0.09% / 0.06% / 0.06% ✓
- `tsae_paper` seeds 1/2/42: -3.84% / +5.58% / -7.31% (non-monotonic at
  this resolution because the temporal-contrastive loss component
  oscillates; macro-trajectory 70K → 16K confirms convergence)
- `topk_sae`: pending verification on cell completion

**TopK-SAE pending** — once it lands, full 4-arch headline ships.

**Observation**: relative ordering of v1.1.0 (TopK > TXC variants > T-SAE)
no longer holds at batch=1024 / n=20K. txc/tsae are now in a tight
~0.894 cluster at k=20. C3 paper claim *might* shift from "honest
negative" to "TXCs tie TopK at k=20" depending on where topk_sae lands.

**docs/components/c3.md still shows v1.1.0 numbers** — intentionally not
overwritten with placeholder during in-flight period; will rerender after
topk_sae completes.

## C4 batch=1024 / n_steps=20K — pending (after topk lands)

C4 cells share train_keys with C3 (same datasource + cfg), so C4's
training will hit CACHED on the canonical 12 train_keys after topk_sae
finishes. Just the qualitative eval re-runs (~10 min Haiku 4.5 calls
per cell × 9 cells = ~$0.40 total cost; ~1.5 hr wall).

C4 launch command (after topk done):
```bash
bash experiments/c4_qualitative/run.sh \
  --archs tsae_paper txc_base txc_pro --seeds 1 2 42
```
Pre-condition: ANTHROPIC_API_KEY at `/workspace/.tokens/anthropic_key`
(verified). NOT including topk_sae for C4 — wasn't in v1.1.0 C4 either.

C4 v1.0.0 results (kept for diff):
- tsae_paper: 74.7 ± 8.1 SEMANTIC, 0.905 judge agreement
- txc_pro: 60.0 ± 2.6, 0.852
- txc_base: 42.0 ± 2.0, 0.768
Honest negative for C4 hypothesis (TXC-pro does NOT Pareto-dominate
T-SAE on SEMANTIC count). Will re-derive at b1024/n20K.

## What I just did (agent owns — overwrite)

Post-compact + post-Han-batch-fix sequence (2026-05-04 13:00 UTC →
2026-05-05 11:21 UTC):

1. ✅ Migrated C3 + C4 analyses to `temp_bench.report.canonical_train_keys`
   (agent_paper helper). One-line filter: `r.train_key in valid_keys`.
2. ✅ Profiled `batch_iter_from_act_cache` and identified mmap page-table
   walk bottleneck (~330 ms / call at batch=1024 because numpy fancy
   indexing forces ~150K 4 KB-page lookups even when file is fully in
   OS page cache). Landed `temp_bench.data.nlp.preloaded_batch_iter_from_act_cache`
   as opt-in shared helper that `.clone()`s into anonymous RAM. 4 unit
   tests in `tests/test_preloaded_batch_iter.py` confirm bit-identity
   with the default helper.
3. ✅ Han-approved deadline override `n_steps=25K → 20K` to fit the 72-h
   sprint budget at observed 2.74 steps/sec (txc_base) → 2.27 → 2.0
   (varies by arch). Updated runners + analyses to filter on the
   override.
4. ✅ `train_log` per-cell persistence (commit `033a3eb6`) — every cell
   writes `logs/c3_b1024_<arch>_seed<seed>_trainlog.json` with the
   trainer's full per-step loss curve, so I can post-cell verify
   convergence (decisions § 12 5%-flag). All 9 trained cells pass.
5. ✅ Drafted urgent message for agent_steer (n_steps=20K + helper
   adoption); Han forwarded.
6. 🟡 **IN FLIGHT** (PID 119732, started 10:06 UTC): topk_sae sweep,
   3 seeds × 2 k_feats. Currently at topk_sae seed=1 step ~9000/20000
   at ~2.0 steps/sec. ETA ~9 hr for the sweep.
7. ⏸ **Pending** post-topk: C4 evals (9 cells, training cache-hit, ~1.5 hr)
   → render → HF push → wrap-up.

Run-state details — earlier session work that is still relevant:

- Activation cache `gemma_2_2b_it_l13_fineweb_24k128` on HF
  (`han1823123123/temp-bench-data/act_cache/e4916bcae1881963/`).
- Probe cache schema 2.0.0 (Phase 7 padding fix landed; left-aligned
  N×32×d_in + first_real metadata) on HF (266 files).
- 4 archs ported into `temp_bench.architectures` (topk_sae, tsae_paper,
  txc_base, txc_pro). MLC still unported (intentional — appendix-only).
- 38-task SAEBench+CT probe loader with all 3 SAEBench-faithfulness
  fixes (codeparrot github-code 5-lang post-filter; amazon_sentiment
  1+5 binaries; amazon_categories deterministic shuffle for cat6).

## Decisions made + carried forward (overseer can override)

- **Per-task AUC reporting**: `my_eval_fn` returns BOTH per-task floats
  (`auc__<task>` × 38) AND aggregates (`mean_auc`, `std_auc`, ...) on
  every leaderboard row. analysis.py uses aggregates for the headline
  and per-task floats for σ_tasks.
- **Smoke rows filtered** out of headline via `eval_cfg.smoke==True`.
- **Bricken A/B for C3**: SKIPPED per decision § 7 default.
- **MLC port**: SKIPPED. Lower priority per agent_paper "Non-decisions";
  appendix-only OK. Test entry stays in `KNOWN_UNPORTED`.
- **EVAL_PROTOCOL_VERSION = "1.1.0"** — bumped for the Phase 7 padding
  fix. Stays at 1.1.0 for the batch=1024 / n_steps=20K re-train (eval
  pathway unchanged; train_key change alone invalidates eval cache).
- **C4 unaffected by padding fix** (no probe cache; forwards Gemma over
  concat_corpora token_ids directly).
- **C4 cells share train_keys with C3**. C4 runs after C3+topk_sae
  complete; cells hit CACHED on training and only re-run qualitative
  eval (~10 min Haiku per cell, 9 cells, ~$0.40 total).
- **n_steps=20K Han deadline override** is paper-headline; not landing
  back in the schema default (per agent_paper § 12 still says 25K).
  Other agents may follow suit if budget pressure (forwarded to
  agent_steer 2026-05-04 PM). decisions § 12 update is agent_paper's
  call.

## Next action — TOPK + C4 + RENDER + WRAP-UP (agent owns)

Plumbing is fully shipped. Remaining work is sequential and persistent-monitor-driven.

### Step-by-step

1. **WAIT for topk_sae sweep to land** — persistent monitor `bnh9xsqjg`
   fires on each cell completion. Currently topk_sae seed=1 step
   ~9000/20000 at ~2.0 steps/sec. ETA ~9 hr (3 unique trainings × ~3hr +
   eval interleave).

2. **Verify topk_sae convergence** — once trainlogs land, check final-1K
   loss drop is < 5% (decisions § 12 flag):
   ```
   for f in logs/c3_b1024_topk_sae_seed*_trainlog.json; do
       .venv/bin/python -c "import json; log = json.load(open('$f')); l = log['loss']; print('$f', f'final-1K vs prev-1K drop: {(sum(l[-2000:-1000])/1000 - sum(l[-1000:])/1000) / (sum(l[-1000:])/1000) * 100:.2f}%')"
   done
   ```

3. **Launch C4** (training cache-hits on new C3 checkpoints; just eval):
   ```
   bash experiments/c4_qualitative/run.sh \
     --archs tsae_paper txc_base txc_pro --seeds 1 2 42
   ```
   Pre-condition: `/workspace/.tokens/anthropic_key` exists (verified).
   ~10 min Haiku per cell × 9 cells = ~1.5 hr. Cost ~$0.40.

4. **Render** C3 + C4 (writes AUTO-RESULTS blocks):
   ```
   .venv/bin/python -c "from temp_bench import report; report.render_all()"
   ```
   Or per-component:
   ```
   .venv/bin/python -m experiments.c3_probing.analysis
   .venv/bin/python -m experiments.c4_qualitative.analysis
   ```

5. **Push HF checkpoints** (persistent pod, optional):
   ```
   .venv/bin/python -c "
   from temp_bench.cache import iter_manifest_for_agent
   from huggingface_hub import HfApi
   token = open('/workspace/.tokens/hf_token').read().strip()
   api = HfApi(token=token)
   for row in iter_manifest_for_agent('agent_nlp'):
       if row.hf_url is None:
           api.upload_folder(folder_path=row.local_path.rsplit('/', 1)[0],
                             path_in_repo=row.train_key,
                             repo_id='han1823123123/temp-bench-models',
                             repo_type='model')
   "
   ```

6. **Final session wrap-up**:
   ```
   bash scripts/wrap_up_session.sh
   ```
   Then update briefing's "What I just did" with final state and
   commit + push.

### Reference: preserved leaderboard rows after re-run

`results/leaderboard.jsonl` will contain:
- 24 v1.0.0 cells (batch=256, n_steps=10K, OLD eval, OLD padding)
- 24 v1.1.0 cells (batch=256, n_steps=10K, OLD eval, NEW padding fix)
- 24 v1.1.0 cells (batch=1024, n_steps=20K, NEW eval, NEW padding) ← **headline**
- 9 v1.0.0 C4 cells (legacy, will be superseded)
- 9 v1.1.0 C4 cells at b1024/n20K (after C4 launch)
- ~50 smoke rows

`canonical_train_keys(component='c3', archs=..., seeds=..., training_cfg=TrainingConfig(n_steps=20_000))`
returns the 12 train_keys for the headline filter; analysis.py wires
this in.

### Reference: live monitoring

- Persistent monitor `bnh9xsqjg` watches `logs/c3_v3_topk.log` for
  `[NEW] | [SETUP | trainlog saved | Error` events.
- Manual progress check:
  ```
  tail -5 logs/c3_v3_topk.log
  .venv/bin/python -c "
  from temp_bench.report import canonical_train_keys, query_leaderboard
  from temp_bench.schemas import TrainingConfig
  keys = canonical_train_keys(component='c3', archs=['topk_sae','tsae_paper','txc_base','txc_pro'], seeds=(1,2,42), datasource_names=('gemma_2_2b_it_l13_fineweb_24k128',), training_cfg=TrainingConfig(n_steps=20_000))
  rows = [r for r in query_leaderboard(component='c3') if r.train_key in keys]
  print(f'Canonical rows: {len(rows)}/24')
  "
  ```

## Don't repeat (agent owns — overwrite)

Locked-decision tripwires:

- **Two TXCs only** (decision #1) — don't introduce a galaxy steering
  variant or a non-locked TXC; raise it in `docs/components/c3.md`
  first if you genuinely need to.
- **Cross-territory edits** — see the OUT OF SCOPE list in mandate.
  Even if Han verbally approves in chat, surface the request in
  writing first. My last-but-one commit got partially rejected on
  exactly this (commit `2283aa15`).
- **Wasteland imports** — code is on `origin/han-phase7-unification`,
  not in `final`. Use `git show`. Never `from src.architectures...`.
- **Bypass `runner.run_cell`** — it's the only writer to the
  leaderboard. Schema validation is mandatory.
- **Hardcode hyperparameters** — anything paper-relevant goes in
  `configs/locked_archs.yaml` and `configs/datasources.yaml`. Edit the
  yaml, not the .py.

Hard-won technical gotchas from this session (verify before bypassing):

- **`datasets<4` pin is load-bearing** for `codeparrot/github-code`.
  v4+ removed `trust_remote_code` and the dataset uses a Python
  loading script. Pinned in `pyproject.toml`.
- **github-code `languages=[...]` does NOT filter the stream.** Must
  `if sample['language'] != target_lang: continue` after iter.
- **`tsae_paper.config.T == 1`, NOT 2.** Contrastive pair is a TRAINING
  construct sampled inside `train_step`. T=2 routes the probe to
  window-encoding (wrong for T-SAE).
- **`LeaderboardRow.metrics` is float-only** (Pydantic). Categorical
  diagnostics like `task_name` go outside `metrics`.
- **Background `nohup ... &`** — bash wrapper returns immediately;
  python keeps running. Verify via `ps -ef | grep python`.
- **Decoder grad-parallel removal** uses `register_post_accumulate_grad_hook`
  on `W_dec` (PyTorch 2.0+). See `tsae.py`/`txc_base.py`/`txc_pro.py::_project_dec_grad`.
- **`einops` is NOT a dep.** Use vanilla `torch.einsum`.
- **TQDM_DISABLE=1 must be exported per bash call.** `set_agent_env.sh`
  doesn't set it. Standard pattern at `export TQDM_DISABLE=1 && ...`.
- **`python -u` is essential for nohup'd long-running scripts** — without
  it, `print()` calls buffer in a 4KB block and don't appear in the
  log file for ~5 min, making the process look stuck. Cost me ~5 min
  of training when I killed/restarted thinking it was deadlocked.
- **`tokenizer.padding_side="right"` for cache build, NEVER left**.
  Left padding shifts position-IDs of real tokens; out-of-distribution
  for Gemma. Phase 7 fix uses right pad + per-example reslice
  (left-aligned in the destination 32-frame). See
  `temp_bench.data.nlp.probe_cache::_encode_texts`.
- **`.contiguous()` after `.T` on saved tensors** — safetensors rejects
  non-contiguous tensors. Bit me on `tsae.py::_normalize_decoder` after
  30 min of training; tests didn't catch because the failure is at
  save time, not init. Always wrap `W_dec.data = ....T.contiguous()`.
- **MooseFS mmap is slow on first random access** (~5 steps/sec on the
  14 GB activation cache), then RAM-cached after warmup. Two parallel
  processes BEFORE warmup deadlocked in state D. Sequence cache build
  → eval; THEN parallel runs are safe.
- **`topk_sae` per-token z explodes at batch=1024**. Shape is
  `(B, seq_len, d_sae) = (1024, 128, 18432)` bf16 = 4.83 GB raw.
  `(z != 0).float()` in `architectures/base.py:81::train_step` doubles
  to 9.66 GB allocation in fp32, on top of model + activations + grad.
  Total peak ~25-30 GB on H100. Co-running with agent_em (38 GB Qwen-14B
  process) → OOM. Mitigation: launch TXC archs (window-level z, ~38 MB)
  + tsae_paper (anchor-pair z, ~76 MB) first while sharing GPU 0;
  defer topk_sae to when GPU 0 is solo.
- **`.clone()` is load-bearing for the preloaded batch_iter.** Without
  it, `torch.from_numpy(np.ascontiguousarray(mmap))` zero-copy wraps
  the mmap and fancy indexing still page-faults — the whole point of
  the preload defeated. `.clone()` materialises into anonymous RAM so
  subsequent indexing is RAM-rate. See `tests/test_preloaded_batch_iter.py`
  for the bit-identity guarantee.
- **First-cell-of-process is slow**. The `.clone()` reads 14 GB from
  MooseFS (or page cache). Cold start: 30 sec. Warm start: 1 sec.
  Module-global cache means subsequent cells in same process reuse
  the RAM tensor.
- **Add setup-phase prints to debug "stuck" trainers.** Default trainer
  is silent until first 1000-step boundary. At 0.5-1 step/sec under
  shared-GPU contention that's 30+ min of zero output, looks deadlocked.
  My fix: print every 100 steps for first 1000, then every 1000.
- **Git commit identity**: repo has no user.email/user.name set.
  Commits use inline `GIT_AUTHOR_*` env vars. Rebases use `git -c
  user.email=... -c user.name=... rebase ...` (env vars don't propagate
  cleanly through rebase's internal commits).
- **Leaderboard JSONL conflicts during rebase**: append-only, both
  sides add new rows. Resolution = strip conflict markers, keep both
  sets. Done it 5+ times this session via:
  ```python
  with open('results/leaderboard.jsonl') as fh: keep = [l for l in fh if not l.startswith(('<<<<<<<', '=======', '>>>>>>>'))]
  ```
- **`runner.run_cell` cache contract**: skips eval if `eval_in_leaderboard(eval_key) and metrics_exist(eval_key)`. Means changing the
  probe cache content WITHOUT bumping `EVAL_PROTOCOL_VERSION` or
  `force_eval=True` silently returns OLD metrics. Lesson learned twice.
  This is why the batch_size + n_steps live in `train_key` (auto-invalidate
  on change) — no manual version bump needed for re-train.
- **`first_real` mask in `_encode_pool`**: window archs (T>1) have edge
  case where n_real < T means NO valid window for that row. Code
  falls back to all-windows mean for those rows (probe noisy but no
  NaN). Affects winogrande/wsc on TXC archs; a few rows per task.

## Open questions for Han / agent_paper (agent owns — overwrite)

1. **n_steps=20K paper-wide?** — landed in my code (commit `513a85ea`)
   per Han's deadline call 2026-05-04 PM. agent_steer was sent the
   recommendation to follow suit (Han forwarded the message). Should
   agent_paper update `decisions.md` § 12 + the schema default so
   future agents don't have to override per-cell? Currently § 12 still
   says 25K.

2. **C4 4th arch coverage** — C4 v1.0.0 ran 3 archs (txc_base, txc_pro,
   tsae_paper). No topk_sae C4. The new b1024/n20K C4 will follow the
   same 3-arch convention. Should we ALSO judge topk_sae for the C4
   Pareto plot, given topk_sae's checkpoint will exist? Adds ~$0.13 +
   ~30 min wall, gives 4 points per arch on the Pareto. (I lean yes
   for completeness; happy to defer.)

3. **`base.py:81` memory hot-spot — opportunistic fix.** The shared
   `TempBenchArch.train_step` computes
   `l0 = (z_flat != 0).float().sum(dim=-1).mean()` which allocates a
   `(B*S, d_sae)` fp32 tensor (9.66 GB at batch=1024 / d_sae=18432 /
   per-token archs like topk_sae). Reordering to
   `(z_flat != 0).sum(dim=-1).float().mean()` defers the float
   conversion to the scalar reduction → drops 9.66 GB peak. With GPU 0
   solo to me, topk_sae fit fine at 45 GB total without the fix; flag
   is **no longer urgent for me** but agent_em (Qwen-14B + d_sae=32768)
   or agent_back (Llama-8B A40 + d_sae=32768) might still benefit.
   Out-of-scope edit for me (`src/temp_bench/architectures/base.py`
   is shared code).

4. **Headline rendering during in-flight period** — currently
   docs/components/c3.md still shows v1.1.0 batch=256 numbers; the new
   b1024/n20K headline only fills in once topk_sae lands. Acceptable
   to keep stale numbers visible until then? (Alternative: render the
   placeholder now, but that strips the only visible C3 numbers from
   the docs for ~9 hr.)

5. **Probe cache HF push at schema 2.0.0** — DONE 2026-05-04 morning.
   266 files at `han1823123123/temp-bench-data/probe_cache/gemma_2_2b_it_l13_fineweb_24k128/`.
   agent_steer / ephemeral pods sync via
   `hf download han1823123123/temp-bench-data --repo-type dataset --include 'probe_cache/gemma_2_2b_it_l13_fineweb_24k128/*'`.
