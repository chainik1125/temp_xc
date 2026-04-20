---
author: Aniket Deshpande
date: 2026-04-18
tags:
  - design
  - in-progress
  - venhoff-eval
---

## Venhoff reasoning-eval integration plan

**Status**: pre-registration. Written before any integration code.

### Decisions locked

- **Q1 (2026-04-18) integration path**: Path 1 for SAE; Path 3 for TempXC; Path MLC for MLC. Approved.
- **Q2 (2026-04-18) layer**: SAE/TempXC/MLC training anchor layer **6**. Steering layer **12** (Llama-8B base; Venhoff's default).
- **Q3 (2026-04-18) aggregation**: All 4 TempXC aggregations run; **`full_window` is the headline**.
- **Q4 (2026-04-20 updated)**: Smoke **100 MATH500 problems** (SAE-only P0 gate); full hybrid run **500 MATH500 problems** (whole split).
- **Q5 (2026-04-18) taxonomy side-channel**: Haiku 4.5 judge + GPT-4o bridge kept from original plan; side-channel only now.
- **Q6 (2026-04-20) dataset**: **MATH500** — Dmitry's redirect. Venhoff reports 3.5% Gap Recovery on the Llama-8B×MATH500 cell (Table 2). That's the bar.
- **Q7 (2026-04-20) model pair**: **Llama-3.1-8B base ↔ DeepSeek-R1-Distill-Llama-8B thinking**.
- **Q8 (2026-04-20) Phase 2/3 code approach**: **vendor Venhoff's `train-vectors/` + `hybrid/` scripts, drive via subprocess with our ckpts exported in their format**. Saves porting 250k lines.

**Still open**: **P2 vs P3 as the NeurIPS-abstract bar.** Dmitry to confirm whether a P2-strength result (5-15% Gap Recovery — above 3.5% but not overwhelming) clears the abstract, or whether we need P3 (>15%). Phase 1a smoke launches before this resolves.

**Paper**: Venhoff et al., *"Base Models Know How to Reason, Thinking
Models Learn When"* ([arXiv:2510.07364](https://arxiv.org/abs/2510.07364)).
**Repo**: `cvenhoff/thinking-llms-interp` (not `cot-interp` — renamed).

## 1. What Venhoff's pipeline actually does (per our audit)

Five stages, 4 dirs:

| stage | entry | in → out |
|---|---|---|
| generate-responses | `run.sh` | MMLU-Pro test split → reasoning traces JSON |
| train-saes | `run.sh` | traces + activations → trained SAE + cluster labels + GPT-4o titles |
| annotate thinking | `run_annotation.sh` | trained SAE → per-sentence cluster labels on held-out traces |
| train-vectors | `run_llama_8b.sh` | annotated sentences → per-category steering vectors |
| hybrid | `run_llama_8b.sh` | steering vectors + base model → reasoning-like base-model outputs |

Phase 1 scope stops at stage 3. Steering + hybrid are Phase 2+ (only if Phase 1
produces signal).

## 2. The audit surfaced four major contract mismatches

The web-claude prompt implied a straightforward SAE-swap. The actual Venhoff
contract is not what we're currently training for. Read these carefully.

### Mismatch A: per-sentence-mean, not per-token

Their SAE is trained on **per-sentence-mean residual-stream activations**,
not per-token. Specifically `utils/utils.py::process_saved_responses`:

1. Re-tokenize the cached CoT trace
2. Regex-split into sentences
3. For each sentence, slice activations `[:, start-1:end, :]` and `mean(dim=1)`
4. Center + L2-normalize by a dataset-global mean, pickled alongside the SAE

This is the SAEBench hazard at a coarser granularity: the mean-pool across
tokens destroys temporal structure before the SAE ever sees it. Our TXC's
whole value-add is that it encodes multi-position patterns — feeding it
pre-averaged vectors erases exactly the information it should detect.

### Mismatch B: tiny dictionaries (50 latents), not 18432

Venhoff sweeps `n_clusters ∈ {5, 10, 15, 20, 25, 30, 35, 40, 45, 50}`. The "SAE"
here is a dictionary-learning discovery tool: `num_latents = n_clusters`, `k=3`
top-k activations. Closer to a tunable k-means than the wide SAEs we've been
training (18432 latents, k=100).

Our trained TempXC checkpoints are ~370× wider and ~33× denser. They're not
drop-in compatible with `sae.encoder(x) → argmax → cluster_label`. Either:
- Train a new **small-k**, **small-width** crosscoder matching their contract, or
- Add a post-hoc reducer (cluster our 18432 latents into 50 super-clusters)

### Mismatch C: argmax-single-latent labeling

`annotate_thinking.py` uses `argmax(encoder(sentence))` to pick one cluster per
sentence. Top-k=100 TempXC has no meaningful single argmax. Our `encode_for_probing`
returns (B, L, d_sae); we'd need to define what "the cluster" is under multi-active
features.

### Mismatch D: layer 6 for Llama-8B (not 12)

Their `run_annotation.sh` uses **layer 6** of DeepSeek-R1-Distill-Llama-8B
(~19% depth, not the ~37% we had in memory). The sweep in `train-saes/run.sh`
covers layers `{6, 10, 14, 18, 22, 26}` and they picked 6 for 8B. Steering uses
layer 12 for a different purpose.

If we want a direct head-to-head with their SAE, layer 6 is the anchor.

## 3. Axis-collapse decision (load-bearing)

Three coherent integration paths, ordered by integration cost and by how much
TXC's temporal advantage is preserved. **Pre-registering path before code.**

### Path 1 — Retrain small-k crosscoders to Venhoff's contract

Train a new TempXC and MLC with **num_latents ∈ {5..50}**, **k=3**, on
per-sentence-mean reasoning activations from MMLU-Pro traces. Plug into
their code unchanged.

- **Pros**: cleanest head-to-head vs their SAE. Zero changes to their annotator.
- **Cons**: at this scale, TempXC's temporal advantage is largely erased by the
  sentence-mean input. We're testing "can a 50-latent TempXC cluster reasoning
  steps" — a weaker question than "does TempXC's feature space match reasoning."
  High risk of reproducing the SAEBench null.
- **Cost**: ~3 h compute for the cross-cluster-size sweep per arch.

### Path 2 — Post-hoc reduction of our existing wide TempXC

Take our 18432-latent TempXC from SAEBench. Encode MMLU-Pro sentence activations.
Cluster the 18432 latents into 50 super-clusters (k-means on mean-activation
vectors). Use cluster-assignment as the "cluster label" per sentence.

- **Pros**: reuses existing ckpts. No retraining.
- **Cons**: two-step reduction changes what we're measuring. The actual
  "clustering" is our post-hoc k-means, not the learned latents. Venhoff's
  taxonomy-quality metric scores the clusters we synthesized, not TempXC's
  native features.
- **Cost**: minutes per arch; LLM-judge cost same as Path 1.

### Path 3 — Patch Venhoff's annotator to respect temporal axis [RECOMMENDED]

Replace `process_saved_responses`'s per-sentence-mean step with:

1. For each sentence of length L_s tokens, take the T-token window centered
   on the sentence (or the whole sentence if L_s ≤ T, pad otherwise).
2. Encode through our TempXC to get (T, d_sae). Apply **aggregation strategy**
   (`mean` / `max` / `full_window` / `last` — the same four we built for SAEBench).
3. Hand the aggregated vector to their clustering + annotation code.
4. For our wide-TempXC checkpoints, add the post-hoc reducer step from Path 2
   so their `argmax → label` pattern still works.

- **Pros**: the only path where TempXC's temporal axis actually participates in
  the taxonomy. Directly tests the thesis.
- **Cons**: most code changes. `process_saved_responses` + `annotate_thinking.py`
  + `train_clustering.py` all need adapted versions. Attribution complexity in
  `VENHOFF_PROVENANCE.md`.
- **Cost**: 1-2 days integration + same compute as Path 1.

### Decision (locked 2026-04-18)

**Path 1 for the SAE + MLC baseline, Path 3 for TempXC.** SAE and MLC
don't have a meaningful temporal axis to preserve, so Venhoff's contract
is the natural one. TempXC's temporal axis is the whole thesis, so
Path 3 is needed for a fair shot. Path 2 stays as a fallback for
TempXC if Path 3 proves too invasive.

**Aggregation ablation (TempXC Path 3)**: run all four (`last`, `mean`,
`max`, `full_window`) as a pre-registered ablation. **`full_window` is
the headline** — it's the one reported in the main figure and whose
delta vs SAE determines the P1-P4 prediction call. The other three
aggregations are reported in the supplement; if one of them beats
`full_window` we note that in the discussion but do not re-crown the
headline post-hoc (that would be the multiple-hypothesis hazard).

## 4. File-port list (once path is confirmed)

Copy from Venhoff's repo into `src/bench/venhoff/`, never modify their repo:

| Venhoff file | Our destination | Delta |
|---|---|---|
| `generate-responses/generate_responses.py` | `src/bench/venhoff/generate_traces.py` | Replace model-loading with our `model_registry.py`. Keep vLLM/nnsight dispatch. |
| `utils/utils.py::process_saved_responses` | `src/bench/venhoff/activation_collection.py` | For SAE path: port unchanged. For TempXC (Path 3): replace per-sentence-mean with per-window-encode. |
| `utils/utils.py::split_into_sentences` + `char_to_token` | `src/bench/venhoff/tokenization.py` | Port verbatim; add assertions for token-span alignment (the `-1` offset is fragile). |
| `utils/sae.py::SAE` | `src/bench/venhoff/sae_shim.py` | Light adapter so our TopKSAE / TempXC expose Venhoff's duck-typed contract (`.encoder`, `.W_dec`, `.b_dec`, `.activation_mean`). |
| `utils/clustering_methods.py::clustering_sae_topk` | `src/bench/venhoff/train_small_sae.py` | Port for Path 1 (small-k training). Invoked per cluster-size. |
| `utils/autograder_prompts.py` | `src/bench/venhoff/autograder_prompts.py` | Port verbatim. The 5 reasoning-category examples are benchmark-invariant across arches — must NOT change when swapping archs. |
| `train-saes/generate_titles_trained_clustering.py` | `src/bench/venhoff/taxonomy/label.py` | GPT-4o batch labeling; port unchanged. |
| `train-saes/evaluate_trained_clustering.py` | `src/bench/venhoff/taxonomy/score.py` | Accuracy + completeness + semantic-orthogonality scoring; port unchanged. |
| `generate-responses/annotate_thinking.py` | `src/bench/venhoff/annotate.py` | Path 3 only: replace the `sae.encoder((x - sae.b_dec).unsqueeze(0))` call with our adapter's aggregation-aware encode. |

Everything in `train-vectors/` and `hybrid/` is Phase 2+; stub only for now.

`VENHOFF_PROVENANCE.md` captures: original path, our dest path, commit hash
of Venhoff source, what we changed, why.

## 5. Configuration choices

| thing | value | why |
|---|---|---|
| dataset | **MMLU-Pro** (test split, TIGER-Lab/MMLU-Pro) | What Venhoff's released code ships with. Flag in plan.md that memory had GSM8K — that was earlier planning, not in the released pipeline. |
| model | **DeepSeek-R1-Distill-Llama-8B** | Already in our `model_registry.py`. Venhoff's default for the 8B slot. |
| layer | **6** (for Llama-8B) | Venhoff's choice per `run_annotation.sh`. Earlier than our intuition; do not re-derive. |
| n_traces (smoke) | **1000** | Venhoff's validation scale. Fits on H100 in ~1h for 8B. |
| n_traces (full) | **5000-10000** | Grow after smoke passes. |
| cluster sizes | `{5, 10, 15, 20, 25, 30, 35, 40, 45, 50}` | Full sweep matches their protocol. Smoke tests one size. |
| k (topk) | **3** | Their default. Our small-k retrain (Path 1) uses this. |
| judge model | **claude-haiku-4-5-20251001** (Haiku 4.5) | **Deviation from Venhoff's GPT-4o default**. Rationale: cost + we already have Anthropic credits; Venhoff's code exposes `anthropic` as an officially supported fallback. Flagged as pre-registered invariant violation in `VENHOFF_PROVENANCE.md`. Cross-judge bridge run on N=100 sentences from smoke test to quantify GPT-4o↔Haiku drift before committing. |
| max training steps | **10 000** (all fits — small-k Path 1 and wide Path 3) with **plateau-based early stop** (same infra as commit `2fae76c`) | Hard cap on training budget. Justification: our SAEBench runs showed NMSE plateaus well before 10k steps on fine-grained data, and reasoning-trace activations are comparable. Early stop triggers if validation NMSE improves < 0.5% over a 1000-step window. Whichever fires first (plateau or 10k cap) ends the fit. Applies to SAE, TempXC, and MLC — no per-arch exception. |

## 6. Compute + disk sizing (for the pod spec)

Assuming 1× H100 80GB, full Phase 1 run (SAE + TempXC-T5 + MLC, 10 cluster sizes,
all fits capped at **10k steps with plateau early-stop** per § 5):

- **Trace generation**: 1.1 h / 1000 traces / 8B model. Full: ~10 h for 10k traces.
- **Activation dump**: ~0.5 h per layer per dataset (layer 6 only per § 5; 1 pass). ~5 GB disk.
- **SAE / TempXC / MLC training (small-k, Path 1)**: <5 min per (arch, cluster-size) at the 10k cap. 3 arches × 10 sizes = 30 tiny trainings = ~2.5 h on one GPU (most plateau-stop well under 10k steps).
- **Path 3 training** (wide TempXC on reasoning activations, T=5 only, 10k-step cap): ~3-4 h (down from the earlier ~6 h un-capped estimate; plateau typically trips ~6-8k steps on reasoning activations).
- **Haiku 4.5 labeling** (substituted for GPT-4o): ~$5-15 in judge fees per arch across full cluster-size sweep (Haiku 4.5 is ~4-10× cheaper than GPT-4o). Wall time: ~2-6 h with Anthropic message-batches API; sub-hour with direct calls at low QPS.
- **Taxonomy scoring**: negligible compute, same batch-API wall.

**Recommended pod**: 1× H100 SXM 80GB, **40 GB root**, **300 GB volume**.
Volume sizing breakdown at 5k traces:
- DeepSeek-R1-Distill-Llama-8B bf16 weights (HF cache): ~16 GB
- Path 1 activation pickle (`N × d_model` float32, N≈150k, d=4096): ~2.5 GB
- Path 3 activation pickle (`N × T × d_model`, T=5): ~12 GB
- Path MLC activation pickle (`N × n_layers × d_model`, n_layers=5): ~12 GB
- vLLM KV cache working set at inference: ~20 GB peak
- Traces JSON + assignments/labels/scores JSON: <1 GB
- pip cache overflow + logs + vendor clone: ~20 GB
- Headroom for retries / reruns: ~200 GB

(Bumped from 250 GB after adding path_mlc on 2026-04-20 — the
multi-layer pickle is the same shape as path3, so we now carry two
12 GB pickles instead of one.)

Total compute: ~**20-25 H100 hours across 1-2 days** for full Phase 1 (revised
down from ~40 h after the T=5-only collapse + 10k-step cap). Smoke test
(one arch, one cluster size, 1000 traces): ~2-3 H100 hours.

## 7. Hazards audit

From the reconnaissance report (transcribed into eval_infra_lessons.md B20+
once we hit any of these):

- `load_sae` asserts `activation_mean` embedded in checkpoint + matches sidecar
  `.pkl`. Our TempXC / SAE ckpts have neither. Fix in the shim.
- `process_saved_responses` uses `token_start-1`. Off-by-one with our
  `HookedTransformer` tokenizer offsets needs verification before the full run.
- `split_into_sentences` has ad-hoc regex escapes for `3.14`, `E. coli`, `k!`
  — reasoning traces may have numeric / LaTeX patterns that still split weirdly.
  Sanity-check: average sentences-per-trace should be 20-40.
- Their `SAE` class has `k=checkpoint.get('topk', 3)` — wide-TempXC ckpts don't
  save a `topk` field. If we save one (`k=100`), annotate path silently does
  topk=100 labels per sentence, meaningless. Path 3 needs its own encoder
  wrapper that bypasses Venhoff's `SAE.encoder`.
- Their pipeline uses the `chat-limiter` GitHub fork; uv sync may fail on
  certain networks. Fallback: pip install `openai`, use blocking API (slower).
- They pin `numpy<2.0`, our env has newer numpy. May require a sidecar
  Python env (SAEBench-style), or version pinning in `pyproject.toml`.

## 8. Phase gating

**Phase 1a (smoke, ~1 day):**
- Clone Venhoff (read-only), run their pipeline unchanged on 100 traces to
  confirm their stack works in our env
- Port SAE baseline path (smallest change)
- Generate 1000 traces, train 1 SAE at cluster_size=15, layer=6, verify the
  taxonomy pipeline end-to-end produces a score

**Phase 1b (comparative, ~3 days):**
- Port the chosen TempXC integration (Path 3)
- Run full cluster-size sweep for SAE + TempXC at T=5, layer 6
- Generate the headline plot

**Phase 1c (extension, ~2 days if signal):**
- T-sweep for TempXC (add T=10, 20 — only if T=5 shows positive signal)
- Aggregation ablation deeper analysis (per-category breakdown)
- Shuffled-activation control (harness-native)

If Phase 1b shows TempXC ≤ SAE across all aggregations + cluster sizes, **we
stop**. Don't launch Phase 2 (steering vectors) against a null signal.

## 9. Success criteria for "TXC works on reasoning"

**Weak**: TempXC beats SAE on any of the three taxonomy-quality metrics
(accuracy / completeness / orthogonality) at any cluster size, for any
aggregation strategy. Publishable as a nuance finding.

**Medium**: TempXC beats SAE on the composite `avg_final_score` at cluster_size=15
(their chosen default). Composite delta ≥ 1 point on 0-10 scale.

**Strong**: TempXC-`full_window` beats SAE on composite score AND a human
spot-check on 20 sampled (sentence, cluster_label) pairs rates TempXC labels
as more temporally-coherent. This is the headline paper result.

## 10. Dmitry pre-alignment — outcome

Items Q1-Q5 locked at the top of this doc (2026-04-18 Slack review).
The single remaining open question is **P2 vs P3 as the NeurIPS
abstract bar**: whether a prediction-P2-strength result (TempXC wins
at small cluster sizes only, flattens at larger sizes) clears the
NeurIPS abstract, or whether P3 (monotonic win across cluster sizes)
is required. This does not gate Phase 1a — the smoke test runs the
same pipeline either way — so launch on everything else.
