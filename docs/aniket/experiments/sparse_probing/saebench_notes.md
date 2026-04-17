---
author: Aniket
date: 2026-04-17
tags:
  - reference
  - in-progress
---

## SAEBench Exploration Notes

Source: [EleutherAI/SAEBench](https://github.com/adamkarvonen/SAEBench),
specifically `sae_bench/evals/sparse_probing/` and
`sae_bench/custom_saes/`. Read in full before writing the wrapper in
`src/bench/saebench/saebench_wrapper.py`; everything below is what
constrains that wrapper's design.

## 1. Entry point and data flow

**CLI:** `python -m sae_bench.evals.sparse_probing.main`. Args that
matter: `--model_name`, `--sae_regex_pattern`, `--sae_block_pattern`,
`--output_folder`, `--force_rerun`, `--random_seed`,
`--llm_batch_size`, `--llm_dtype`, `--sae_batch_size`,
`--lower_vram_usage`, `--artifacts_path`.

**For custom (non-sae_lens) SAEs:** skip the CLI and call
`run_eval(config, selected_saes, device, output_path, ...)` directly,
where `selected_saes=[(name_str, sae_obj)]`. This is the path our
wrapper takes.

**Data flow per run:**

1. `run_eval_single_dataset` loads text + labels via
   `dataset_utils.get_multi_label_train_test_data`.
2. `activation_collection.get_all_llm_activations` runs Gemma forward
   (SAEBench owns the subject model — loads via
   `HookedTransformer.from_pretrained_no_processing`), caches LLM
   activations to `{artifacts_path}/sparse_probing/{model}/{hook_name}/{dataset}_activations.pt`
   as a dict `{class_name: tensor (B, L, d_model)}`.
3. `get_sae_meaned_activations` calls `sae.encode(acts: (B, L, d_in)) →
   (B, L, d_sae)`, then mean-pools across non-BOS/pad token positions
   to `(B, d_sae)`.
4. `probe_training.train_probe_on_activations` does class-separation
   feature selection (top-k) and sklearn logistic-regression probe
   fitting.

**Important:** SAEBench runs the subject LLM itself from
classification-labeled text datasets (bias-in-bios, ag_news,
amazon_reviews, github-code). It does **not** consume pre-cached
activation files. Our existing
`data/cached_activations/gemma-2-2b/fineweb/` caches are unusable for
the probing tasks because the probing text is entirely different.

## 2. SAE interface — what we must implement

Base class: `sae_bench.custom_saes.base_sae.BaseSAE`, an `nn.Module` +
ABC. Required abstract methods:

```python
def encode(self, x: torch.Tensor) -> torch.Tensor: ...
def decode(self, feature_acts: torch.Tensor) -> torch.Tensor: ...
def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

**Critical shape contract for `encode`:** input `(B, L, d_model)`,
output `(B, L, d_sae)`. SAE-lens shape, per-position. Our TempXC's
native encode is `(B, T, d_in) → (B, d_sae)` (shared-z), so the
wrapper must handle the L > T case via windowing + aggregation
(see plan § 5).

Required attributes (enforced by `test_sae` + read by non-probing
evals like SCR/TPP/absorption):

- `W_enc: (d_in, d_sae)` — may need to expose a derived view for
  crosscoders whose native W_enc is `(T, d_in, d_sae)`.
- `W_dec: (d_sae, d_in)` unit-norm rows.
- `b_enc, b_dec`.
- `dtype`, `device` (read directly, not via `next(self.parameters())`).
- `cfg: CustomSAEConfig` — dataclass with `d_in, d_sae, hook_layer,
  hook_name, model_name, architecture, dtype, context_size,
  apply_b_dec_to_input, finetuning_scaling_factor, activation_fn_str,
  prepend_bos, normalize_activations, ...`.

For the sparse-probing eval alone only `encode` is strictly required;
the attributes get read by adjacent evals we may run later.

## 3. Probing tasks — the default suite

Eight datasets, mean-pool context, per-example labels:

| dataset | classes | notes |
|---|---|---|
| LabHC/bias_in_bios_class_set1 | 5 profession binaries | |
| LabHC/bias_in_bios_class_set2 | 5 more | |
| LabHC/bias_in_bios_class_set3 | 5 more | |
| canrager/amazon_reviews_mcauley_1and5 | 5 product categories | |
| canrager/amazon_reviews_mcauley_1and5_sentiment | 2 sentiment | |
| codeparrot/github-code | ~5 programming languages | |
| fancyzhx/ag_news | 4 topic classes | |
| Helsinki-NLP/europarl | ~5 language IDs | |

Total: ~30 binary probing tasks after unrolling into per-class
one-vs-rest. Splits: `probe_train_set_size=4000`,
`probe_test_set_size=1000`. Sequence length: `context_length=128`.

**Default `k_values = [1, 2, 5]`**. Output schema pre-declares slots
for `k ∈ {1, 2, 5, 10, 20, 50, 100}`, so extending to include `20` to
match our pre-registered design is a config change, not a code
change.

## 4. Gemma 2B support

First-class. `activation_collection.LLM_NAME_TO_BATCH_SIZE["gemma-2-2b"]
= 32`, `LLM_NAME_TO_DTYPE["gemma-2-2b"] = "bfloat16"`.
`run_all_evals_custom_saes.py` hard-codes `layers=[12], d_model=2304`
as the canonical Gemma-2-2B SAEBench layer. Hook name:
`blocks.12.hook_resid_post`. Matches our checkpoint layer exactly.

## 5. Output schema

Written to `{output_folder}/{sae_release}_{sae_id}_eval_results.json`,
serialized from `SparseProbingEvalOutput` dataclass:

```json
{
  "eval_config": { ... },
  "eval_id": "...",
  "datetime_epoch_millis": 0,
  "eval_result_metrics": {
    "llm": { "llm_test_accuracy": 0.87, "llm_top_1_test_accuracy": 0.6, ... },
    "sae":  { "sae_test_accuracy": 0.83, "sae_top_5_test_accuracy": 0.78, ... }
  },
  "eval_result_details": [
    { "dataset_name": "ag_news", "llm_top_5_test_accuracy": 0.69, "sae_top_5_test_accuracy": 0.72, ... }
  ],
  "eval_result_unstructured": { ... per-binary-probe accuracies ... },
  "sae_cfg_dict": { ... },
  "eval_type_id": "sparse_probing"
}
```

Both aggregate and per-dataset breakdowns. Metric is **accuracy only**
— no F1, no calibration. `probing_runner.py` will re-emit these into
our own JSONL schema (one record per arch × T × protocol × aggregation
× task × k) for easier aggregation across runs.

## 6. Minimal working example

From `main.py` docstring (L544-586):

```python
import sae_bench.evals.sparse_probing.main as sp
from sae_bench.evals.sparse_probing.eval_config import SparseProbingEvalConfig
from sae_bench.sae_bench_utils import general_utils, activation_collection

device = general_utils.setup_environment()
sae = my_custom_adapter(checkpoint, arch, aggregation)   # our SAEBenchAdapter
selected_saes = [("txcdr_layer12_step1", sae)]
cfg = SparseProbingEvalConfig(random_seed=42, model_name="gemma-2-2b")
cfg.llm_batch_size = activation_collection.LLM_NAME_TO_BATCH_SIZE[cfg.model_name]
cfg.llm_dtype = activation_collection.LLM_NAME_TO_DTYPE[cfg.model_name]
sp.run_eval(cfg, selected_saes, device, "eval_results/sparse_probing",
            force_rerun=True, clean_up_activations=False, save_activations=True)
```

This is the call `probing_runner.py` wraps.

## 7. Gotchas

- **Dependency conflicts.** SAEBench pins `numpy < 2`, `datasets < 4`,
  `sae_lens ^6.22.2`, `python ^3.10`. Our pyproject.toml has
  `python >= 3.11`, `sae-lens >= 4.0` — compatible-ish, but `numpy < 2`
  is a potential issue. Plan: install SAEBench into a separate uv-managed
  environment (not the main `temp-xc` venv) and invoke via subprocess
  OR via a sidecar script that does its own `uv run --with saebench ...`.
  We do not vendor SAEBench into our repo.
- **Aggregation happens AFTER `encode()`.** SAEBench mean-pools across
  the L axis regardless of our architecture. Our four aggregation
  strategies (last / mean / max / full-window) have to be implemented
  at the encode level — the wrapper returns `(B, L, d_sae)` with per-
  position values set according to the strategy, and SAEBench's
  subsequent mean-pool becomes a second stage we accept as-is.
  Full-window aggregation returns `(B, L, T × d_sae)` which changes
  effective `d_sae`, so the wrapper must expose a different `cfg.d_sae`
  for that case.
- **BOS/pad masking** runs before `encode()` — SAEBench zeroes BOS/pad
  positions in the LLM acts before handing them to us. Don't assume
  all positions have non-zero input.
- **`sae.dtype` attribute** is read directly, not via
  `next(self.parameters()).dtype`. Must be a class attribute or
  property.
- **VRAM.** Gemma-2-2B forward peak ~22 GB at `batch=32, seq_len=128`.
  On an H100 80GB that leaves ~58 GB for our SAE forward + activation
  buffers — comfortable at T=5, and likely fine to T=40 if the adapter
  doesn't materialize full `(B, L, T, d_sae)` tensors at once.

## 8. Key source files

- `sae_bench/evals/sparse_probing/main.py` — entry point, `run_eval`.
- `sae_bench/evals/sparse_probing/eval_config.py` — `SparseProbingEvalConfig` dataclass.
- `sae_bench/evals/sparse_probing/eval_output.py` — `SparseProbingEvalOutput` dataclass.
- `sae_bench/evals/sparse_probing/probe_training.py` — feature selection + LR probe.
- `sae_bench/custom_saes/base_sae.py` — `BaseSAE` ABC.
- `sae_bench/custom_saes/custom_sae_config.py` — `CustomSAEConfig` dataclass.
- `sae_bench/custom_saes/run_all_evals_custom_saes.py` — canonical Gemma-2-2B entry.
- `sae_bench/sae_bench_utils/activation_collection.py` — LLM forward + hook dispatch.

## 9. Implications for our wrapper design

1. `SAEBenchAdapter(BaseSAE)` — subclass to inherit the abstract
   contract, implement `encode/decode/forward`, expose required
   attrs.
2. Aggregation lives inside `encode()` — input `(B, 128, d_in)`,
   output `(B, 128, d_sae or T*d_sae)` depending on strategy.
3. Dependency isolation via sidecar `uv` env to avoid `numpy<2` and
   `datasets<4` breaking our main env.
4. JSONL output schema defined in our `probing_runner.py` for
   cross-run aggregation; SAEBench's per-run JSON is parsed and
   unrolled into one record per (arch × T × protocol × aggregation ×
   task × k) combination.

Related: [[experiments/sparse_probing/plan|sparse probing plan]].
