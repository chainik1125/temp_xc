# code_benchmark — TempXC vs MLC on Gemma-2B-it Python activations

Sibling to `experiments/separation_scaling/`. Trains TopK SAE, temporal
crosscoder (TXC) and multi-layer crosscoder (MLC) on Gemma-2-2B-it residual
activations over Python functions, then runs a four-pass evaluation ladder
(coarse dashboard → attribution spread → program-state probes → LM KL
fidelity).

See the plan note at `docs/dmitry/theory/txc_code_benchmark_plan_v2.md` for
the scientific rationale.

## Install

```bash
cd <repo root>
uv sync --extra code-benchmark
```

## Reproduce

```bash
cd experiments/code_benchmark
PYTHONPATH=../separation_scaling/vendor/src \
    uv run python run_training.py --config config.yaml

# Pass 1 — coarse dashboard (add --with-lm for loss-recovered)
PYTHONPATH=../separation_scaling/vendor/src \
    uv run python run_eval_coarse.py --config config.yaml --with-lm

# Pass 2 — attribution spread
PYTHONPATH=../separation_scaling/vendor/src \
    uv run python run_eval_phase1_spread.py --config config.yaml

# Pass 3 — program-state probes
PYTHONPATH=../separation_scaling/vendor/src \
    uv run python run_eval_phase2_state.py --config config.yaml

# Pass 4 — LM KL fidelity stratified by surprisal-delta
PYTHONPATH=../separation_scaling/vendor/src \
    uv run python run_eval_phase3_kl.py --config config.yaml
```

## Layout

```
experiments/code_benchmark/
├── README.md                 (this file)
├── config.yaml               (dataset / model / arch / eval config)
├── code_pipeline/
│   ├── python_code.py        (activation extraction + on-disk cache)
│   ├── python_state_labeler.py  (per-token program-state labels)
│   ├── training.py           (unified training loop + window builders)
│   └── eval_utils.py         (shared eval helpers)
├── run_training.py           (trains SAE / TXC / MLC)
├── run_eval_coarse.py        (Pass 1 dashboard)
├── run_eval_phase1_spread.py (Pass 2 attribution spread)
├── run_eval_phase2_state.py  (Pass 3 program-state probes)
├── run_eval_phase3_kl.py     (Pass 4 LM KL fidelity)
├── cache/                    (activation cache, built on first run)
├── checkpoints/              (per-arch .pt files)
├── results/                  (per-arch + joined JSON outputs)
└── plots/                    (all PNG figures)
```

## Multi-host deployment

The activation cache is the heaviest artifact (~12 GB at default scale). Build
it once on one host; rsync to others; train one architecture per host:

```bash
# on extract-host
cd experiments/code_benchmark
PYTHONPATH=../separation_scaling/vendor/src \
    uv run python run_training.py --config config.yaml --only topk_sae

# once cache is built, mirror to other hosts:
rsync -av cache/ other-host:~/.../experiments/code_benchmark/cache/

# then on each host, train one architecture:
#   --only topk_sae   (on host A)
#   --only txc_t5     (on host B)
#   --only mlc_l5     (on host C)
```

Evaluations read checkpoints + cache from disk, so they can run on any host
after the checkpoints are rsynced to one place.
