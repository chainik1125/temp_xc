## Adding a new architecture to the bench harness

Target audience: Bill / Han / anyone with a new architecture idea.
Goal: go from "I have an `nn.Module` class" to "Aniket's harness runs
probing on it + writes all the standard outputs" without touching any
harness code.

### What the harness calls

The harness (`src.bench.run_eval`) expects every architecture to register
an `ArchSpec` subclass in `src/bench/architectures/__init__.py:REGISTRY`.
`ArchSpec` is the plug-in contract — it hides whether your arch is a
plain SAE, a crosscoder, or something stranger.

Required methods on the `ArchSpec` subclass:

| method | purpose |
|--------|---------|
| `create(d_in, d_sae, k, device) → nn.Module` | instantiate your model |
| `train(model, gen_fn, total_steps, batch_size, lr, device) → log` | training loop (or pass) |
| `eval_forward(model, x) → EvalOutput` | reconstruction loss + L0 |
| `encode(model, x) → (B, L, d_sae)` | probing features (ordered input) |
| `decoder_directions(model, pos=None) → (d_in, d_sae)` | decoder weights for UMAP |

Plus one class attribute: `data_format ∈ {"flat", "seq", "window", "multi_layer"}`.

Shuffle control is done FOR you via `ArchSpec.encode_for_probing(model, x,
shuffle_seed=None)` — the default implementation handles "shuffle input,
then call encode()". Override only if your architecture can't cope with a
shuffled input (MLC is an example — it takes 4D multi-layer input and has
its own shuffle logic in `mlc_probing.py`).

### Worked example: a vanilla matryoshka-penalty TXC

Step 1 — write the model + spec in `src/bench/architectures/matryoshka_txc.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.bench.architectures.base import ArchSpec, EvalOutput
from src.bench.architectures.crosscoder import TemporalCrosscoder


class MatryoshkaTXC(TemporalCrosscoder):
    """TempXC with a matryoshka penalty on the decoder.

    Same shape contracts as TemporalCrosscoder; the penalty is applied
    during training only — forward / encode / decode are unchanged.
    """
    def __init__(self, d_in, d_sae, T, k, matryoshka_lambda: float = 0.1):
        super().__init__(d_in, d_sae, T, k)
        self.matryoshka_lambda = matryoshka_lambda

    def extra_loss(self, z):
        # Encourage early features to reconstruct with any prefix — the
        # matryoshka prior. Only active in train().
        pass


class MatryoshkaTXCSpec(ArchSpec):
    name = "MatryoshkaTXC"
    data_format = "window"

    def __init__(self, T: int, matryoshka_lambda: float = 0.1):
        self.T = T
        self._lam = matryoshka_lambda

    @property
    def n_decoder_positions(self):
        return self.T

    def create(self, d_in, d_sae, k, device):
        return MatryoshkaTXC(d_in, d_sae, self.T, k, self._lam).to(device)

    def train(self, model, gen_fn, total_steps, batch_size, lr, device,
              log_every=500, grad_clip=1.0):
        # Reuse CrosscoderSpec's training loop; it supports extra_loss
        # if defined. Or copy-paste and customize.
        from src.bench.architectures.crosscoder import CrosscoderSpec
        base = CrosscoderSpec(T=self.T)
        return base.train(model, gen_fn, total_steps, batch_size, lr,
                          device, log_every=log_every, grad_clip=grad_clip)

    def eval_forward(self, model, x):
        loss, x_hat, z = model(x)
        return EvalOutput(
            sum_se=(x_hat - x).pow(2).sum().item(),
            sum_signal=x.pow(2).sum().item(),
            sum_l0=(z > 0).float().sum(dim=-1).sum().item(),
            n_tokens=x.shape[0],
        )

    def decoder_directions(self, model, pos=None):
        if pos is None:
            return model.decoder_dirs_averaged
        return model.decoder_directions_at(pos)

    def encode(self, model, x):
        # Shape contract: (B, L, d_in) → (B, L, d_sae). Inherit from
        # CrosscoderSpec if the windowing logic is identical.
        from src.bench.architectures.crosscoder import CrosscoderSpec
        return CrosscoderSpec(T=self.T).encode(model, x)
```

Step 2 — register in `src/bench/architectures/__init__.py`:

```python
from src.bench.architectures.matryoshka_txc import MatryoshkaTXCSpec

REGISTRY["matryoshka_txc"] = MatryoshkaTXCSpec
```

Step 3 — train it (reuses existing sweep):

```bash
# On pod, extends the existing scripts/runpod_saebench_train.sh pattern
python -m src.bench.sweep \
    --dataset-type cached_activations --cached-layer-key resid_L12 \
    --model-name gemma-2-2b --cached-dataset fineweb \
    --models matryoshka_txc \
    --k 100 --T 5 --steps 10000 \
    --results-dir results/saebench/sweeps/matryoshka_txc_protA_T5
```

Step 4 — probe it (ordered + shuffled pair in one command):

```bash
python -m src.bench.run_eval \
    --architecture matryoshka_txc \
    --protocol A --t 5 --aggregation full_window \
    --ckpt results/saebench/ckpts/matryoshka_txc__gemma-2-2b__l12__k100__protA__seed42.pt \
    --output results/saebench/results/matryoshka_txc_protA_T5.jsonl \
    --both-ordered-shuffled
```

Step 5 — the JSONL now has records with `"architecture": "matryoshka_txc"`.
Downstream analysis (`scripts/analyze_saebench.py`, the confusion-matrix
script) picks it up automatically; no changes needed.

### If your arch needs a new `data_format`

`data_format` drives which data generator (`flat` / `seq` / `window_T` /
`multi_layer`) the sweep feeds you, and which eval path `evaluate_model`
dispatches to. If your arch needs something genuinely new (e.g. a
cross-position + cross-layer hybrid), add the new string to:

1. `src/bench/data.py` — add a `build_<your_format>_pipeline` function
2. `src/bench/eval.py:evaluate_model` — add the dispatch branch (see B2
   in `eval_infra_lessons.md` for what happens if you skip this)

The regressions module (`src/bench/regressions.py`) should get a new
check that verifies your spec dispatches correctly. Don't skip it — we
have 19 prior bugs documenting why.

### Testing your arch before the overnight run

```bash
# Dry-run the regressions + the arch construction:
python -m src.bench.regressions   # must pass

# 500-step smoke test:
python -m src.bench.sweep \
    ... --models <your_key> --steps 500 --results-dir /tmp/smoke

# Probe it (bias_in_bios only, one aggregation, k=5):
python -m src.bench.run_eval \
    --architecture <your_key> ... --k-values 5 \
    --output /tmp/smoke/probe.jsonl
```

If that round-trips, your arch is ready for the overnight queue.

### Checklist

- [ ] Spec class inherits `ArchSpec`
- [ ] `data_format` attribute set correctly
- [ ] Registered in `REGISTRY`
- [ ] `encode()` returns `(B, L, d_sae)` shape
- [ ] Smoke test passes (500 steps + 1 probe cell)
- [ ] `python -m src.bench.regressions` passes with your arch in the REGISTRY

Once those are green, queue it via the morning submission channel and
`run_eval.py` picks it up overnight.
