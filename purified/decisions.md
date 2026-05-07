---
status: locked
---

# Locked decisions

These decisions govern the paper's experimental design. They are
policy; re-opening any requires a new round of discussion.

## 1. Two TXC architectures (locked)

- **TXC-base** = `txc_bare_antidead_t5` — vanilla TopK + tsae_paper anti-dead stack.
- **TXC-pro** = `phase5b_subseq_h8` — subseq encoder + matryoshka H8 + multi-distance contrastive.

**Rationale**: these are the only two TXCs with consistent top-3 finishes
across both Phase 5 and Phase 7 probing leaderboards. Steering hill-climb
winners were considered and rejected because they lose 0.005–0.020
probing AUC vs canonical.

**Trade-off accepted**: C5 steering becomes a "matches T-SAE at high coh"
result rather than a "beats T-SAE" result. C3+C4 wins are stronger.

The "two TXCs only" rule holds at the paper-claim level. Per-component
hyperparameter overrides flow through `per_component_hparams.cN` in
`configs/locked_archs.yaml`; never via forked class files.

## 2. C6 EM is reframed as honest negative

**Finding**: prior EM evidence (origin/case-em-nanda) shows SAE arditi beats
TXC k=100 at every (steps × organism × α-regime) cell. Arch gap *widens*
to +12.58 align in R32 ext-α regime.

**Paper framing**: report the SAE win honestly. Salvage the
**bundle-null architecture-generality** result — both arches' k=30
bundles peak at align ≈ 41.3, falling 13–23 below their single-feat
champions. This falsifies the "distributed misalignment" hypothesis in
both dictionaries: an interpretive contribution despite the probing loss.

## 3. Branch model

- `final` is the only paper branch.
- Wasteland code lives only on `origin/wasteland-canonical`. Wasteland
  docs (`docs/`, `papers/`) were stripped from `final` for the
  anonymous release.
- Sibling branches are read-only context: `origin/case-em-nanda`,
  `origin/case-backtracking`, `origin/case-synthetic`. We never merge
  them into `final`.

## 4. Cross-branch reads

- **Do not merge** sibling branches into `final`. They are still being
  updated; merging would freeze stale state and create conflict surface.
- Read directly from origin: `git show origin/<branch>:<path>`.
- If a frozen snapshot of code is needed, copy it once into
  `purified/src/temp_bench/` with the source commit hash in a header
  comment, and stop tracking origin from then on.

## 5. CLAUDE.md scoping

Subdirectory `CLAUDE.md` files **auto-load on demand** when a file
under that directory is read. An agent launched at the repo root sees
the wasteland CLAUDE.md initially; `purified/CLAUDE.md` loads
automatically the moment it touches a paper file.

## 6. HuggingFace repos

Two private repos hold all artifacts (referenced via `${TEMP_BENCH_HF_ORG}`):

- **`${TEMP_BENCH_HF_ORG}/temp-bench-models`** — checkpoints (locked
  archs + baselines), keyed by `<train_key>`.
- **`${TEMP_BENCH_HF_ORG}/temp-bench-data`** — activation caches, judge
  transcripts, pre-tokenised tasks, synthetic data.

Set `TEMP_BENCH_HF_ORG` in the environment before running any
HF-touching script.

## 7. Bricken resample is opt-in per component, NOT a locked architecture default

Bricken resample co-tunes six knobs (resample_every=500, min_fires=1,
n_check=2048, max_resample_fraction=0.5, EMA-AuxK α=1/8,
dead_threshold=128k tokens). All jointly tuned for the Qwen-7B medical
organism in prior work.

**Decision**:

- The locked architectures TXC-base and TXC-pro **do not** include
  Bricken resample. They include only what's listed in
  `docs/paper/architecture.md` proper.
- Bricken resample is exposed as an opt-in `BrickenConfig` knob in
  `src/temp_bench/training/bricken.py`. Components turn it on
  themselves and disclose the choice in their writeup.
- **C6 only by default** (prior evidence directly supports it on the
  Qwen-7B medical organism).
- C1/C2 keep it off (no dead-feature pressure at d_sae=40).
- C3/C4/C5/C7 keep it off (the cost of an A/B test for each component
  outweighed the maybe-marginal effect within the sprint window).

## 8. Wasteland code deleted; wasteland docs kept

- **Code wasteland deleted** from `final` (`src/`, `experiments/`,
  `references/`, `tests/`, `scripts/` at root, root `pyproject.toml`,
  `uv.lock`, `Dockerfile`, etc.). Lives only on
  `origin/wasteland-canonical`.
- For the anonymous release, the in-repo `docs/` wasteland tree was
  also stripped — only `purified/docs/` remains.

**Why asymmetric originally**: docs were read often (passively, for
context — every component writeup cited ~5 wasteland research logs).
Code is read once per port (actively, for transcription, ~10 ports
total). Delete what's read once; keep what's read often.

**Benefit**: an accidental `from src.architectures.tfa import …`
raises `ModuleNotFoundError` immediately rather than silently picking
up wasteland code. The "no wasteland imports" rule becomes git-level
enforcement, not policy.

## 9. Consolidate purified/ docs from 3 → 2 files

`purified/` ships two top-level docs only:

- **CLAUDE.md** (~250 lines): operating manual + brief overview.
  Auto-loaded when any file under `purified/` is read.
- **PROTOCOL.md** (~330 lines): cache-key contract, two-TXC discipline,
  baselines, framework discipline.

`purified/README.md` was dropped; its unique content was merged into
`CLAUDE.md`.

## 10. C3 task suite is `SAEBench+CT` (n=38)

C3 evaluates on **SAEBench+CT**, defined as the canonical upstream
SAEBench sparse-probing suite (Karvonen et al., 36 binary one-vs-rest
tasks across 8 datasets) augmented with two cross-token coreference
probing tasks (WinoGrande, SuperGLUE WSC). **Total: 38 tasks.**

The SAEBench composition is fixed by `chosen_classes_per_dataset` in
upstream `sae_bench/sae_bench_utils/dataset_info.py`:

```
bias_in_bios_class_set1: ["0","1","2","6","9"]      → 5
bias_in_bios_class_set2: ["11","13","14","18","19"] → 5
bias_in_bios_class_set3: ["20","21","22","25","26"] → 5
amazon_reviews_mcauley_1and5: ["1","2","3","5","6"] → 5
amazon_reviews_mcauley_1and5_sentiment: ["1.0","5.0"] → 2
codeparrot/github-code: ["C","Python","HTML","Java","PHP"] → 5
ag_news: ["0","1","2","3"] → 4
europarl: ["en","fr","de","es","nl"] → 5
                                                       ──
                                                       36
+ winogrande_correct_completion + wsc_coreference    → 38
```

**Three implementation deltas** vs the prior internal "FULL-36" loader:

1. **github-code provider**: SAEBench's `codeparrot/github-code` with
   the 5 SAEBench languages `["C","Python","HTML","Java","PHP"]`.
   Loader requires `trust_remote_code=True` and `datasets<4`.
2. **amazon_sentiment**: include the 1.0-vs-rest binary alongside the
   5.0-vs-rest variant.
3. **amazon_categories**: hardcode the class list to
   `["1","2","3","5","6"]` and use a non-streaming pull large enough
   to populate all 5 classes deterministically.

**Why SAEBench-faithful + the 2 coref additions**:

- SAEBench is the recognised standard. Saying "we evaluated on
  SAEBench" defends against the "you cherry-picked tasks that favor
  TXC" review on the headline benchmark axis.
- WinoGrande + WSC retained because they are the cleanest single-task
  evidence for TXC's cross-token inductive bias. Reported transparently
  as a "+CT" extension, not folded silently into "SAEBench".

**Naming convention for the paper**: refer to the suite as
**SAEBench+CT** in tables and figure captions. First mention in prose:
"the standard SAEBench sparse-probing benchmark (Karvonen et al., 36
tasks across 8 datasets) augmented with two cross-token coreference
probing tasks (WinoGrande, SuperGLUE WSC; n=38 binary one-vs-rest
tasks total)."

## 11. TrainingConfig: batch=1024, n_steps=25_000, fixed schedule

All C3/C4/C5/C6/C7 cells use a **fixed-step protocol** applied uniformly
across all archs and all pods.

| Knob | Value | Source |
|---|---|---|
| `batch_size` | **1024** uniform | empirically validated on prior internal sweeps |
| `n_steps` | **25_000** (binding cap) | empirically validated convergence point |
| `plateau_early_stop` | **False** (disabled) | SAE-comparison literature standard |

**Why plateau-stop is off**: an absolute-threshold plateau detector
(`max(loss[-5000:]) - min(loss[-5000:]) < 1e-4`) causes cross-arch
unfairness because the same window-range means very different things
for archs whose losses naturally land at different scales. The
SAE-comparison literature avoids this by using fixed step counts:

- **T-SAE paper (Bhalla/Ye 2025) §4.1**: fixed schedule, no early stopping.
- **TFA paper (Lubana et al.) App. B.1**: "trained from scratch on 1B
  precached activations" — fixed token count.
- **GemmaScope / Anthropic monosemanticity**: fixed token budgets.

The code path for plateau-stop stays (gated off by
`plateau_early_stop=False`) for opt-in use; production cells run the
full 25K.

**Fairness mechanism**: every arch trains for exactly 25K steps × 1024
batch = 25.6M activation tokens. Cross-arch comparisons are symmetric
by construction.

**Cross-arch fairness**: every arch in every component re-trains under
the exact same `TrainingConfig`. C6 retains its `bricken_enabled=True`
override per § 7 (a published per-component knob). The (batch_size,
n_steps, plateau_*) knobs are identical.

**Cache hygiene**: `batch_size`, `n_steps`, and the plateau-* fields
all flow through `compute_train_key`. Old cells stay in the leaderboard
under their old keys for diff comparison. Analyses in
`experiments/cN_*/analysis.py` should filter via the
`temp_bench.report.canonical_train_keys` helper:

```python
from temp_bench.report import canonical_train_keys, query_leaderboard

valid = canonical_train_keys(
    component="c5",
    archs=["txc_base", "txc_pro", "tsae_paper"],
    seeds=(1, 2, 42),
    datasource_names=["gemma_2_2b_it_l13_fineweb_24k128"],
)
rows = [r for r in query_leaderboard(component="c5") if r.train_key in valid]
```

## 12. 100K-iter long-schedule sweeps for C5 and C6

Two additional sweeps at `n_steps=100_000` (~102M tokens) were run for
C5 and C6 alongside the canonical 20K/25K sweeps. The
literature-aligned compute scale gives reviewers a "compute headroom"
defence.

**Cells coexist cleanly in the leaderboard**: `n_steps` is in the
`train_key` hash, so 20K/25K and 100K cells occupy distinct keys.

**Whichever sweep completes first becomes the paper headline**: the
toggle is mechanical via the `training_cfg=` argument passed to
`canonical_train_keys()` in `c5_steering/analysis.py` and
`c6_em/analysis.py`.

**Within-component fairness invariant**: under no circumstances do we
mix 20K and 100K cells in the same C5/C6 AUTO-RESULTS table. The
short-schedule and long-schedule sweeps are separate canonical
universes.

**Cross-component independence**: only C5 and C6 get the long-schedule
copy sweep. C3/C4/C7 stay at their respective short schedules.

## 13. Literature-aligned T=1 baseline re-train (C3 + C5)

Closer reading of the SAE-comparison literature shows the per-token
baselines were *over*-batched against the field standard at C3 + C5:

- **SAEBench**, App. B: "We use a batch size of 2048 tokens... we
  train each SAE on approximately 500M tokens of activations."
  Buffer-based, batch in TOKENS.
- **T-SAE paper**, §4.1: "All SAEs are trained with the BatchTopK
  activation... batch size 4096 tokens... 500M activation tokens."
- **TFA paper**, App. B.1: "1B precached activations, batch size 1024
  tokens."

C3/C4/C5's sequence-based pattern (B=1024 × seq_len=128 = 131,072
tokens/step) over-batches per-token archs by ~65× vs SAEBench's 2K
canonical. C6/C7's window-based pattern at T=1 (1024 tokens/step) is
within 2× of canonical — "right by accident".

**Decision**: re-train the C3 + C5 per-token baselines at the per-arch
literature-faithful window size:

| Arch          | `train_window_size` | tokens/step | Reference |
|---------------|---:|---:|---|
| `topk_sae`    | **1** | 1024 | vanilla TopK, no temporal — matches C6's sae_arditi at T=1 |
| `tsae_paper`  | **2** | 2048 | Bhalla/Ye 2025 §3.1 "load activations in pairs $(x_t, x_{t-1})$" — exact paper match |

Both within 2× of SAEBench's 2K canonical scale. C6 baselines
(`sae_arditi` at T=1) are unchanged. **C7 T-SAE keeps T=5** — both
need to be supported. So `tsae_paper` at C3/C4/C5 trains at T=2,
while `tsae_paper` at C7 stays at T=5; two different `train_keys`,
both live in the leaderboard, component-specific. TXC archs at every
component are unchanged (they sample 1 random T-window per row
already).

**Framework change**:

- `temp_bench.data.nlp.cache.preloaded_batch_iter_from_act_cache` gains
  `train_window_size: int | None = None`. ``None`` preserves current
  behavior (full sequences); ``int T`` returns 1 random T-window per
  row, shape `(batch_size, T, d_in)`.
- `TrainingConfig.train_window_size: int | None = None` field, plumbed
  into `compute_train_key` via `model_dump(exclude_none=True)`. Default
  (None) preserves all existing train_keys; setting an int gets a
  fresh key.

**Within-component fairness invariant**: under no circumstances do we
mix sequence-based and T=1-window cells in the same C3 / C5
AUTO-RESULTS table. The re-trained per-token cells become the
canonical baselines; old over-batched cells become the diff-only
"pre-fix" reference.

## 14. MLC + TFA as paper-faithful baselines at C3, C5, C6

The two baselines' paper-faithful training conventions differ:

**MLC** (Multi-Layer Crosscoder, layer-axis analog of TXC):

- Native data format: `(B, L, d_in)` where L = number of adjacent
  layers (paper default L=5).
- Datasource `gemma_2_2b_it_l11to15_fineweb_24k128`
  (`layers: [11,12,13,14,15]`).
- `build_activation_cache` detects `layers: list[int]` and registers
  N forward hooks in one model pass → captures (N, L=5, seq_len, d_in)
  into a single `.npy`.
- `preloaded_batch_iter_from_multilayer_cache(act_cache_key, seed)`
  samples 1 random (seq, token) per row → returns `(B, L=5, d_in)`.
- TrainingConfig: `B=1024, n_steps=20_000, train_window_size=None`
  (the L axis lives outside the train_window_size system).
- Per-step encoder load: B × L = 5120 tokens (similar to TXC at T=5).

**TFA** (Temporal Feature Analysis):

- Native data format: `(B, T_seq, d_in)` where T_seq is the full
  sequence length. TFA's attention attends over preceding context
  tokens; T_seq=128 is the design intent.
- Paper-faithful training: `TFA_BATCH=32`, full-sequence sample,
  per-step tokens = 32 × 128 = 4096.
- TrainingConfig per-arch override: `B=32, n_steps=20_000,
  train_window_size=None`. The `B=32` is the only cross-arch B
  exception (TFA's per-step memory pressure precludes B=1024 at the
  d_sae we use).

**Cross-arch B uniformity note**: TFA is the single arch with a B=32
override (paper-faithful per-arch convention); all other archs continue
at B=1024. Documented in `docs/components/c{3,5,6}.md` caveats.

**Within-component fairness invariant**: every arch within a component
trains for ``n_steps=20_000``. Per-step token throughput VARIES per
arch (TopK 1024, T-SAE 2048, TFA 4096, TXC 5120, MLC 5120) — each at
its source paper's intended setup. Reviewer-defensible because every
arch's training matches its primary reference.

## Open items (not yet locked)

- **MLC scope** — competitive with TXC-base at C3 k=5. Include as
  related work / appendix? Decide before the paper goes to draft.
- **Bumping the 25K cap if loss is clearly still descending** — if
  multiple archs end at 25K with loss still falling steeply (e.g.,
  final-1K-step drop > 5% of the loss value), revisit the cap. Any
  bump must be uniform across all archs (fairness).
