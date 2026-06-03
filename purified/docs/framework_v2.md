# Framework v2 — TempBench, post-submission cleanup

**Branch**: `arxiv` (off `origin/final-aniket`).
**Status**: spec frozen; v2 migration landed.
**Audience**: anyone (especially research agents) reading the repo for the first
time. **Read this end-to-end before touching code.**

## Mission

Provide the *minimal, uniform* infrastructure needed to reproduce the
paper's main-body findings and to extend the framework with new
architectures, evaluations, and synthetic benchmarks.

Two principles, in order:

1. **One canonical pathway for everything**. All training goes through one
   trainer. All results land in one leaderboard with one schema. Synthetic
   and real-LM experiments share `architectures/`, `core/`, and the result
   contract — they diverge only at the data + eval seams, where the paper
   itself diverges.

2. **Plugin extension, not core rewriting**. Adding a new arch / eval /
   datasource / experiment is a single-file drop-in plus a YAML entry.
   Agents never edit `temp_bench/core/`.

## Paper-section mapping (replaces the cN convention)

The paper's main body has only two experimental sections:

| Paper § | Was called | What it is |
|---|---|---|
| § 4 — Synthetic Setting | C1 + C2 | Toy data, ground-truth feature recovery (eAUC, gAUC, NMSE) |
| § 5.1 — Sparse Probing | C3 | Gemma-2-2B-IT, 36-task SAEBench |
| § 5.2 — Backtracking | C7 | DeepSeek-R1-Distill, detection + inducement |
| § 5.3 — Emergent Misalignment | C6 | Qwen-2.5, Wang procedure |
| § 5.4 — HH-RLHF Decomposition | C5 | HH-RLHF preference data |

C4 (qualitative latents) is appendix-only and out of scope for the arxiv
framework.

## Directory layout

```
purified/
├── run.py                              # single CLI dispatcher
├── pyproject.toml                      # deps incl. upstream baseline pkgs
│
├── temp_bench/                         # the library (locked core)
│   ├── core/
│   │   ├── runner.py                   # run_experiment(spec), run_sweep(grid)
│   │   ├── cache.py                    # leaderboard.jsonl + manifest.jsonl I/O
│   │   ├── config.py                   # YAML loaders + deterministic cache-keys
│   │   ├── code_version.py             # commit_sha + dirty + diff_sha256 capture
│   │   ├── schemas.py                  # Pydantic models (LeaderboardRow, …)
│   │   └── trainer.py                  # ONE SAE trainer, consumes BatchIter
│   ├── data/
│   │   ├── activation_buffer.py        # token-level shuffle buffer
│   │   ├── window_buffer.py            # (B, T, d_in) buffer for TXC archs
│   │   ├── real_lm.py                  # subject-model hookpoint + cache build
│   │   └── synthetic.py                # markov + coupled_hmm + helpers
│   ├── interfaces/
│   │   ├── architecture.py             # TempBenchArch ABC
│   │   ├── batch_iter.py               # BatchIter protocol
│   │   └── evaluator.py                # Evaluator ABC
│   ├── archs/                          # REGISTRY (drop-in)
│   │   ├── txc_base.py                 # ours
│   │   ├── txc_pro.py                  # ours
│   │   ├── topk_sae.py                 # ours (vanilla TopK reference)
│   │   ├── stacked_sae.py              # ours
│   │   ├── mlc.py                      # ours (multi-layer crosscoder)
│   │   ├── sae_arditi.py               # ours (SAE-arditi for § 5.3)
│   │   ├── tsae.py                     # ADAPTER wrapping AI4LIFE-GROUP/temporal-saes
│   │   └── tfa.py                      # ADAPTER wrapping TFA reference impl
│   ├── evals/                          # REGISTRY (drop-in)
│   │   ├── synthetic_recovery.py       # § 4
│   │   ├── probing.py                  # § 5.1
│   │   ├── backtracking.py             # § 5.2
│   │   ├── em.py                       # § 5.3
│   │   └── rlhf.py                     # § 5.4
│   └── utils/                          # seed, plotting, judge_client, …
│
├── experiments/                        # paper-section entry points
│   ├── synthetic/run.py
│   ├── probing/run.py
│   ├── backtracking/run.py
│   ├── em/run.py
│   ├── rlhf/run.py
│   ├── render_paper_figures.py
│   └── TEMPLATE/                       # copy-this-to-extend
│
├── configs/
│   ├── archs.yaml                      # arch registry: class_path, version
│   ├── data.yaml                       # datasource registry
│   ├── experiments.yaml                # canonical paper-section sweeps
│   └── sweeps/                         # agent-defined sweep configs (any)
│
├── checkpoints/<train_key>/            # trained models + manifest.jsonl
├── results/                            # leaderboard.jsonl + per-run dirs
├── tests/                              # pytest contract tests
└── docs/
    ├── framework_v2.md                 # this file
    └── figs/                           # rendered paper figures (fig2_*.{pdf,png})
```

## Core interfaces

Three minimal ABCs / protocols anchor everything.

### `TempBenchArch` (`temp_bench/interfaces/architecture.py`)

Every architecture in `temp_bench/archs/` subclasses this:

```python
class TempBenchArch(nn.Module, ABC):
    """One SAE/crosscoder/TXC architecture."""

    @abstractmethod
    def encode(self, x: Tensor) -> Tensor:
        """x: (B, T, d_in) → z: (B, T, d_sae) or (B, d_sae) per arch."""

    @abstractmethod
    def decode(self, z: Tensor) -> Tensor:
        """z → x_hat: (B, T, d_in)."""

    @abstractmethod
    def train_step(self, x: Tensor) -> dict[str, Tensor]:
        """One optimizer step. Returns metrics dict (incl. 'loss' for the trainer)."""

    @property
    def d_sae(self) -> int: ...

    @property
    def consumes(self) -> Literal["token", "window"]:
        """Which buffer this arch trains from. Determines BatchIter contract."""

    def post_step(self) -> None:
        """Called by trainer AFTER optimizer.step(). Default no-op.
        Used by T-SAE for decoder unit-norm projection, by TXC for grad-parallel removal."""
```

### `BatchIter` (`temp_bench/interfaces/batch_iter.py`)

```python
class BatchIter(Protocol):
    """Yields training batches.
    
    - If arch.consumes == "token": __call__(B) → (B, d_in)
    - If arch.consumes == "window": __call__(B) → (B, T, d_in)
    
    Two implementations:
    - ActivationBuffer (token-level shuffle buffer; literature standard)
    - WindowBuffer (window-level shuffle buffer; for TXC archs)
    Both refilled from cached sequences or streamed from subject model.
    """
    def __call__(self, batch_size: int) -> torch.Tensor: ...
```

### `Evaluator` (`temp_bench/interfaces/evaluator.py`)

```python
class Evaluator(ABC):
    """Evaluates a trained arch on a paper-section-specific task."""

    @abstractmethod
    def eval(self, model: TempBenchArch, spec: EvalSpec) -> dict[str, float]:
        """Returns a flat dict of float metrics for the leaderboard row.
        All non-float diagnostics (per-task arrays, judge transcripts) go to
        a per-run dir; this dict is the row that lands in leaderboard.jsonl."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier (e.g. 'probing', 'backtracking'). Used in eval_key."""

    @property
    def protocol_version(self) -> str:
        """Bump to invalidate cached eval rows."""
        return "1.0.0"
```

## Code-version tracking

Every result row (leaderboard + manifest) carries:

```python
class CodeVersion(BaseModel):
    commit_sha: str          # full SHA of HEAD at run time
    dirty: bool              # True if working tree had uncommitted changes
    diff_sha256: str | None  # sha256 of `git diff HEAD` output when dirty, else None
```

**The runner refuses to launch with `dirty=True` unless `--allow-dirty` is set.**
This forces discipline: commit-or-stash before running. The escape hatch lets
in-progress experiments still execute (with audit trail intact: `(commit_sha,
diff_sha256)` reconstructs the exact code state).

Code version is **recorded but NOT in the cache key**. Arch invalidation
remains the explicit `arch_version` bump (in `configs/archs.yaml`). Code
version is for **audit**, not for cache invalidation.

## Cache keys (unchanged from v1)

Deterministic SHA256 of canonical JSON:

```
train_key = sha256({
    arch_class: <class_path>,
    arch_version: <semver>,
    hparams: <merged hparams>,
    seed: <int>,
    training_cfg: <TrainingConfig dict>,
    data_key: <hash of data spec>,
})[:16]

eval_key = sha256({
    train_key: <train_key>,
    evaluator_name: <str>,
    eval_protocol_version: <semver>,
    eval_cfg: <dict>,
})[:16]
```

Identical inputs → identical keys → cache hit (skip).

## Data path: token shuffle buffer

The literature-standard SAE training data pipeline:

```
              source sequences (synthetic gen OR real-LM act cache)
                          │
                          ▼
              ┌─────────────────────┐
              │  ActivationBuffer   │   token-level RingBuffer (~2M tokens)
              │  - refill_threshold │   refilled when half-empty by sampling
              │  - sample(B)        │   new sequences and flattening their tokens
              └────────┬────────────┘
                       │
              ┌────────┴────────────┐
              │                     │
              ▼                     ▼
        arch.consumes              arch.consumes
        == "token"                 == "window"
              │                     │
              ▼                     ▼
       BatchIter(B)→           WindowBuffer wraps the
       (B, d_in)               buffer + yields (B, T, d_in)
                               by sampling T-consecutive tokens
                               from buffered sequences
```

**Why**: literature SAEs train on i.i.d. shuffled tokens (Anthropic, SAEBench).
Our v1 trained on `(B, seq_len, d_in)` whole sequences with strong within-
sequence correlation, plus a later opt-in for 1 random T-window per sequence.
Neither matched the standard. v2 makes the buffer the default.

**Window archs**: TXC-base, TXC-pro, T-SAE, MLC, Stacked-SAE need
contiguous T tokens. The `WindowBuffer` samples a random window of length T
from each buffered sequence; effectively i.i.d. windows.

## Architectures: ours vs. adapter-wrapped

All archs subclass `TempBenchArch` and live in `temp_bench/archs/`. From
the registry's perspective, they look identical. Internally:

**Ours** (`txc_base.py`, `txc_pro.py`, `topk_sae.py`, `stacked_sae.py`,
`mlc.py`, `sae_arditi.py`): implementations written by us, with all logic
inside.

**Adapter-wrapped** (`tsae.py`, `tfa.py`): wraps an upstream package and
forwards calls. The wrapper handles shape conventions only; the math runs
in the upstream library.

```python
# temp_bench/archs/tsae.py
"""T-SAE — wraps Ye et al. 2025 (AI4LIFE-GROUP/temporal-saes).

The upstream package implements the loss, decoder unit-norm, threshold
inference, and BatchTopK matryoshka structure. This module ADAPTS the
batch shape from our framework's (B, T, d_in) → upstream's expected
consecutive-token-pair (B, 2, d_in) and exposes the upstream model
through the TempBenchArch interface.
"""

from temporal_saes.trainers.temporal_sequence_top_k import TemporalSequenceTopK

class TSAE(TempBenchArch):
    consumes = "window"
    
    def __init__(self, *, d_in, d_sae, k_pos, T_pair=2, **upstream_kwargs):
        super().__init__()
        self._upstream = TemporalSequenceTopK(
            d=d_in, n_features=d_sae, k=k_pos, **upstream_kwargs
        )

    def encode(self, x):
        return self._upstream.encode(x[:, 0])  # encode first token of pair

    def decode(self, z):
        return self._upstream.decode(z)

    def train_step(self, x_window):
        # Adapt our (B, T, d_in) → upstream's (B, 2, d_in) consecutive pair
        x_pair = x_window[:, :2]   # first 2 tokens of each window
        loss = self._upstream.loss(x_pair)
        return {"loss": loss}
```

Registry entry:

```yaml
# configs/archs.yaml
tsae:
  class_path: temp_bench.archs.tsae:TSAE
  version: "upstream-2025-11-05"   # date of upstream pin
  upstream: "AI4LIFE-GROUP/temporal-saes@<commit>"
  hparams:
    d_sae: 16384
    k_pos: 20
```

The `version` field uses `"upstream-<date>"` for adapter archs (vs. semver
for ours) so the audit trail makes the distinction clear.

## Evaluators

One module per paper section in `temp_bench/evals/`. Each subclasses
`Evaluator`. The synthetic evaluator is simple:

```python
class SyntheticRecovery(Evaluator):
    name = "synthetic_recovery"
    protocol_version = "1.0.0"

    def eval(self, model, spec) -> dict[str, float]:
        # spec contains the synthetic generator + its ground-truth features
        decoder = model.W_dec.detach()                # (d_sae, d_in)
        eauc = feature_recovery_auc(decoder, spec.emission_features)
        gauc = feature_recovery_auc(decoder, spec.hidden_features)
        nmse = reconstruction_nmse(model, spec.eval_batch_iter)
        return {"eauc": eauc, "gauc": gauc, "nmse": nmse}
```

Real-LM evaluators have more orchestration (probe cache, judge calls,
detection labels) but produce the same flat metric dict.

## Dispatcher CLI (`run.py`)

```bash
# single cell
python run.py <experiment> --arch <name> --seed <int> [task-specific args]
python run.py synthetic --arch txc_base --seed 42 --setup coupled --k-pos 5

# sweep
python run.py sweep <sweep_config.yaml>
python run.py sweep configs/sweeps/probing_canonical.yaml

# reproduce a full paper section
python run.py reproduce <section>
python run.py reproduce synthetic      # runs canonical sweep for § 4
python run.py reproduce probing        # runs canonical sweep for § 5.1
python run.py reproduce all            # all 5 sections

# render paper figures from current leaderboard
python run.py render-figures
```

`--smoke` flag is honored everywhere: tiny dims, n_steps=10, single seed,
no real model loading. Used for sanity validation. Smoke cells are
written with `eval_cfg.smoke=True` and filtered out of paper headlines.

`--allow-dirty` overrides the dirty-tree gate; the diff is captured in
the row.

## Sweep config schema

```yaml
# configs/sweeps/example.yaml
experiment: probing                   # which entry point under experiments/
arch: [txc_base, txc_pro, tsae]      # list = grid axis
seed: [1, 2, 42]
k_feat: [5, 10, 20, 40, 80, 160, 320, 640]

# Optional knobs
n_parallel: 4                         # claim up to N GPUs from pool
skip_cached: true                     # default; idempotent re-runs
on_failure: continue                  # continue / abort
smoke: false                          # if true, all cells run smoke mode

# Optional task-specific overrides
training_cfg:
  n_steps: 20000
  batch_size: 4096
eval_cfg:
  S: 16
```

The dispatcher cross-products the lists and submits the grid.

## Reproducing the paper from scratch

```bash
# 1. fresh checkout
git clone <repo> && cd temp_xc/purified
uv sync

# 2. (optional) pull pre-cached checkpoints + caches from HF
# saves training time; reproduction still possible without
python run.py sync-from-hf

# 3. reproduce all 5 paper sections
python run.py reproduce all
# - § 4 synthetic: ~10 min on 5090 (no LM forward passes)
# - § 5.1 probing: ~3 hr on H100 (incl. activation cache build if absent)
# - § 5.2 backtracking: ~2 hr on H100
# - § 5.3 em: ~6 hr on H100 (judge calls cost ~$3)
# - § 5.4 rlhf: ~2 hr on H100

# 4. render figures
python run.py render-figures
# Writes Figs 2-6 PDFs to docs/figs/
```

## Extension recipes

### Add a new architecture

```bash
# 1. drop one file
cat > temp_bench/archs/my_arch.py <<'PY'
from temp_bench.interfaces.architecture import TempBenchArch
class MyArch(TempBenchArch):
    consumes = "token"
    def __init__(self, *, d_in, d_sae, k_pos, **kw): ...
    def encode(self, x): ...
    def decode(self, z): ...
    def train_step(self, x): ...
PY

# 2. register
cat >> configs/archs.yaml <<'YML'
my_arch:
  class_path: temp_bench.archs.my_arch:MyArch
  version: "1.0.0"
  hparams: { d_sae: 18432, k_pos: 20 }
YML

# 3. run a smoke cell
python run.py synthetic --arch my_arch --seed 42 --smoke
```

### Add a new evaluation

```bash
# 1. drop one file
cat > temp_bench/evals/my_eval.py <<'PY'
from temp_bench.interfaces.evaluator import Evaluator
class MyEval(Evaluator):
    name = "my_eval"
    def eval(self, model, spec) -> dict[str, float]: ...
PY

# 2. add an experiment entry point
mkdir -p experiments/my_experiment
cp experiments/TEMPLATE/run.py experiments/my_experiment/run.py
# edit run.py: set ARCH list, DATA spec, EVAL = MyEval
```

### Add a new synthetic benchmark

```bash
# 1. add generator + datasource
# edit temp_bench/data/synthetic.py: implement gen_my_bench(...)
# edit configs/data.yaml: add datasource entry pointing to gen_my_bench

# 2. run
python run.py synthetic --setup my_bench --arch txc_base --seed 42 --smoke
```

## Result schema

```python
class LeaderboardRow(BaseModel):
    schema_version: str = "2.0.0"
    eval_key: str
    train_key: str
    data_key: str
    
    experiment: str              # "synthetic" / "probing" / ...
    arch: str                    # registry key
    arch_version: str
    seed: int
    
    training_cfg: TrainingConfig
    eval_cfg: dict
    evaluator_name: str
    evaluator_protocol_version: str
    
    metrics: dict[str, float]    # the actual numbers
    primary_metric: str          # which key in metrics is the headline
    
    code_version: CodeVersion    # commit_sha + dirty + diff_sha256
    ts: str                      # ISO 8601
```

## Tests + smoke validation

`tests/` exercises:
- Interface contracts (every registered arch implements the ABC)
- Cache key determinism (same inputs → same key, across processes)
- Code-version capture (dirty detection, diff hash stability)
- Dispatcher routing (each subcommand lands at the right experiment)
- Synthetic end-to-end (smoke run produces a real result row)

Run before any commit: `cd purified && .venv/bin/python -m pytest tests/ -q`.

## Things v2 deliberately does NOT do

- **No per-agent briefings.** Agent orchestration lives in the agent
  framework (Claude Code's task tracker, harness logs), not in this repo.
- **No `agents/` directory.** Single source of truth lives in
  `FINDINGS.md` at repo root for agent-shared notes; otherwise this repo
  is just code.
- **No experiment-specific config hardcoding.** Everything in YAML.
- **No multiple eval pathways.** One Evaluator interface, one leaderboard.
- **No per-component subprocess launchers** (run_on_gpu.sh, sharded_launch
  etc.). The sweep dispatcher handles parallelism; ad-hoc launchers are
  the agent's territory if needed.

## Provenance

- v1 framework + cache contract: see `git show origin/final:purified/docs/paper/framework.md`
- Architectural inspiration for agent extensibility: 
  https://github.com/safety-research/automated-w2s-research
- Per-arch literature compliance: `decisions.md` §§ 15-18 on `origin/final`

## Status

- Framework spec: this document, version `2.0.0`, frozen 2026-05-27.
- Results: see `purified/docs/reproduction_report.md` (§ 4 synthetic) and
  `purified/docs/ac_signed_motion_bench.md` (AC / order-sensitive bench).
