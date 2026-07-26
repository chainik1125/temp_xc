# KLiCKe writing-revision destination gate

This CPU-only screen reconstructs conservative prefixes from the public KLiCKe
keystroke archive. Immediately before a consecutive trailing deletion burst,
it predicts whether the writer will erase two, three, four, or at least five
complete words from the final ordered word window.

Run the frozen full-corpus gate from the repository root:

```bash
uv run python -m purified.experiments.writing_revision_destination.klicke \
  --archive /private/tmp/KLiCKe-WritingTask.zip \
  --output purified/results/task_screening/klicke_trailing_deletion_destination_text_gate.json \
  --bootstrap-samples 2000
```

Render the fixed-cohort history-length sweep:

```bash
MPLCONFIGDIR=/private/tmp/txc-mpl python3 \
  -m purified.experiments.writing_revision_destination.report \
  --input purified/results/task_screening/klicke_trailing_deletion_destination_text_gate.json \
  --output purified/results/task_screening/klicke_trailing_deletion_destination_window_sweep.png
```

The JSON is aggregate-only. It records source and cohort hashes, extraction and
deduplication diagnostics, grouped metrics, equal-writer bootstrap contrasts,
and the fixed-cohort sweep; it never stores essay text or lexical windows.

## Exact subject-model token audit

Before extracting model activations, build the fixed final-10-token cohort with
the paper's Llama-3.1-8B tokenizer:

```bash
uv run python -m purified.experiments.writing_revision_destination.token_audit \
  --archive /private/tmp/KLiCKe-WritingTask.zip \
  --cohort-output purified/results/neurips_rebuttal/writing_revision_destination/token_cohort.parquet \
  --manifest-output purified/results/neurips_rebuttal/writing_revision_destination/token_audit.json \
  --history-tokens 10 \
  --token-cap 6 \
  --bootstrap-samples 2000
```

The audit tokenizes the complete document immediately before and after the
trailing deletion burst. It retains an event only when the post-deletion token
IDs are an exact prefix of the pre-deletion IDs, then globally deduplicates the
final ten-token window and drops conflicts. It reports the full observed token
distance distribution, evaluates a configurable capped token-distance target,
and repeats the same writer-grouped sweep for the original 2/3/4/5+ lexical
target. The Parquet contains token IDs and cryptographic writer/event hashes,
but no raw text or unhashed writer identifier. Its saved model inputs include
the tokenizer's paper-compatible BOS token; the exact-prefix target and final
ten-token window remain defined on text tokens.

## Layer-10 activation extraction

First run the unpadded singleton smoke cache. A different `--limit`, model
revision, or configuration must use a different output directory because the
resumable request manifest is immutable:

```bash
python -m purified.experiments.writing_revision_destination.extract_activations \
  --cohort purified/results/neurips_rebuttal/writing_revision_destination/token_cohort.parquet \
  --cohort-manifest purified/results/neurips_rebuttal/writing_revision_destination/token_audit.json \
  --output-dir purified/results/neurips_rebuttal/writing_revision_destination/activation_cache_smoke_v3_singleton \
  --device cuda:0 \
  --limit 64 \
  --batch-size 1 \
  --shard-size 32 \
  --attention sdpa
```

Before writing a shard, the smoke run records `padding_diagnostic.json`. With
batch size one, every model call has exactly the prefix length and therefore no
right padding. The diagnostic repeats the shortest and longest singleton
forwards with identical token IDs, masks, positions, and shapes, and requires
bitwise-identical layer output. Batched configurations instead retain the
two-row diagnostic that separates padding-path from batch-dimension numerical
drift.

After one attention implementation passes, the full one-A40 extraction can be
launched in a named tmux session:

```bash
export TXC_RUNPOD_ROOT=/workspace/temporal-crosscoders
export KLICKE_EXTRACT_SESSION=klicke-deletion-l10-full-v3-singleton
export KLICKE_EXTRACT_GPU=1
export KLICKE_MODEL_REVISION=1f47e50cdbe801ad8a5174156ec3a0655108fb9f
export KLICKE_EXTRACT_BATCH_SIZE=1
export KLICKE_EXTRACT_SHARD_SIZE=256
export KLICKE_EXTRACT_ATTENTION=sdpa
export KLICKE_ACTIVATION_CACHE=/workspace/temporal-crosscoders/purified/results/neurips_rebuttal/writing_revision_destination/activation_cache_v3_singleton_full
export KLICKE_EXTRACT_LOG=/workspace/logs/writing_revision_destination/full_v3_singleton.log
export KLICKE_PYTHON_BIN=/workspace/txc-venv/bin/python
bash purified/experiments/writing_revision_destination/launch_extract_tmux.sh
```

The extractor uses plain Hugging Face Transformers and hooks block 10's output,
the paper's layer-10 `resid_post`. It right-pads full saved prefixes for model
execution but stores only the final ten activations in float16, along with
token IDs, labels, and full SHA-256 hashes. Each deterministic shard is checked
against its cohort rows before a resumed run skips it. `complete.json` is
created only after every row and shard checksum validates.

## Raw activation gate

After extraction completes, run the primary token-distance sweep on CPU:

```bash
python -m purified.experiments.writing_revision_destination.evaluate_activations \
  --cohort purified/results/neurips_rebuttal/writing_revision_destination/token_cohort.parquet \
  --cohort-manifest purified/results/neurips_rebuttal/writing_revision_destination/token_audit.json \
  --cache-dir purified/results/neurips_rebuttal/writing_revision_destination/activation_cache \
  --output purified/results/neurips_rebuttal/writing_revision_destination/raw_token_distance.json \
  --target capped_token_label \
  --window-sizes 1 2 3 4 5 6 7 8 9 10
```

Repeat with `--target lexical_label` for the original 2/3/4/5+ sensitivity.
Every outer split is grouped by writer. Hidden-coordinate selection is fitted
only on each outer training fold and shared across temporal positions. The
reported controls are the endpoint, an inner-CV best single offset,
order-invariant mean/std/max, first and second differences, a linear-trajectory
residual, ordered history, fixed-fit reverse and shuffle perturbations, and a
probe retrained on deterministically shuffled histories. Multiclass log loss is
the primary metric.

Render the primary and lexical-sensitivity publication packages together:

```bash
MPLCONFIGDIR=/private/tmp/txc-mpl \
python -m experiments.writing_revision_destination.report \
  --input purified/results/neurips_rebuttal/writing_revision_destination/raw_token_distance_singleton_v1.json \
  --input purified/results/neurips_rebuttal/writing_revision_destination/raw_lexical_destination_singleton_v1.json \
  --output-dir purified/results/neurips_rebuttal/writing_revision_destination/publication_singleton_v1
```

Each target emits a 300-DPI PNG, PDF, CSV, aggregate JSON, and inline Markdown.
The left panel reports absolute held-out log loss; the right reports
equal-writer paired control-minus-ordered gaps with 95% bootstrap intervals.
