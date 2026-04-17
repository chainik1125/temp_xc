# RunPod handoff — TXCDR vs TFA feature comparison

## What state is this in

The previous session trained a partial NLP sweep and, more importantly,
built a cross-arch autointerp pipeline with results for one layer
(resid_L25) of Gemma-2-2B-IT. The sweep script exists but the focus has
shifted away from long training runs toward qualitative feature-level
comparison of TFA vs TXCDR vs Stacked SAE.

**Branch:** `han-runpod` (NOT `han`).

**What is committed:**

- Trained checkpoints on disk only (1.5–3 GB each, `.pt` files
  gitignored): `results/nlp_sweep/gemma/ckpts/`
  - 9 files total: `{stacked_sae, crosscoder, tfa_pos} × k∈{50,100}` on
    resid_L25 unshuffled (6), plus `k=50` shuffled for each arch (3).
- Sweep NMSE / L0 summary JSONs (committed):
  `results/nlp_sweep/gemma/results_*.json`
- Autointerp artifacts (all committed):
  `results/nlp_sweep/gemma/scans/` — raw scans, LLM labels, temporal
  spread per feature, novel/pred decomposition, decoder-similarity
  summary, high-span comparison, plus `high_span__*.json`.
- Figures (full-res + 1400 px doc-size + 480 px thumb):
  `results/nlp_sweep/gemma/figures/`
- Research log with all findings:
  `docs/han/research_logs/2026-04-17-autointerp-initial.md` —
  **read this first.**

**Headline scientific finding (one-liner).** TFA-pos splits its 18 432
latents into two disjoint libraries: a sparse `novel_codes` set that is
passage-local and mostly padding-fire at this training scale, and a
semi-dense `pred_codes` set that mirrors stacked_sae / crosscoder's
document-general semantic profile. On the "does arch X extract
high-span features that detect corpus-general content?" axis, TXCDR
wins cleanly: 14 / 15 of its top high-span features are content-bearing
vs 1 / 15 for TFA novel.

**Known sore spots (why this isn't publishable yet).**

- TFA-pos is under-trained: NMSE = 0.12 at k=50 vs stacked 0.06,
  crosscoder 0.08. At k=100 TFA-pos diverged to NMSE=4.54 on the
  unshuffled run and to 1.80 shuffled, even with the v2 NaN-prevention
  patch (`proj_scale.clamp(-1, 1)` + grad-finiteness check).
- Single layer (`resid_L25`), single k=50 for most qualitative
  analyses, single seed, single model (Gemma-2-2B-IT), single dataset
  (FineWeb). No cross-layer or cross-model replication.
- LLM-labeled feature set is small (top-50 per arch × 4 arches = 200
  labels by `claude-haiku-4-5-20251001`, with Claude API key at
  `/workspace/.anthropic-key`).

## Your job

**Not** to run a full sweep. **Not** to chase down more numbers for
their own sake. The qualitative research question is still:

> Do TXCDR and TFA extract substantially different temporal features
> from LLM activations, and if so, does each arch surface patterns the
> others miss?

We have a plausible first-pass answer for Gemma resid_L25 k=50 (see
the research log). Your job is to (i) make sure that answer is
actually correct on the data we have, (ii) decide whether those
findings are strong enough to carry a NeurIPS-grade comparison, and
(iii) if not, extend just enough to close the gap — qualitatively,
with short targeted experiments, not with a fresh 20-hour sweep.

### Phase 1 — Sanity-check the existing results (few hours)

Goal: make sure the claims in the research log survive independent
verification.

1. **Reload and forward-verify the checkpoints.** For each of the 9
   ckpts, confirm the state dict has no NaN parameters and that a
   forward pass on real cached activations produces finite outputs.
   One-liner is enough; the previous session did this ad-hoc, not in
   a script.
2. **Replay the headline numbers.** The log claims:
   - Crosscoder uses 16% of its 18 432 latents; stacked_sae 86%;
     tfa_pos 85% novel, 42% pred.
   - TFA novel median chain-diversity = 1 (55/100 all-same-chain);
     TFA pred = 10 (0/100 all-same-chain).
   - 14/15 Crosscoder high-span features are content-bearing; 1/15
     for TFA novel; 0 high-span for TFA pred.
   Recompute each from the committed JSONs under
   `results/nlp_sweep/gemma/scans/`. If any number is off, fix the
   log.
3. **Spot-check high-span exemplars.** For ~5 Crosscoder features
   and the one content-bearing TFA novel feature (`feat 6333`), look
   at the top-20 activating windows (not just top-10) to confirm the
   label the log assigns actually fits. You can use
   `temporal_crosscoders/NLP/scan_features.py --top-k 20` with
   `--top-features 10 --features <specific_ids>`.
4. **Confirm TFA pred ≠ TFA novel is a real partition.** The log
   says top-50 by pred_mass and top-50 by novel_mass have zero
   overlap. Re-verify on the larger top-500 cut.
5. **Re-run the LLM labeler on a held-out set of features** (feats
   ranked 51-100 by mass per arch). The current labels only cover
   top-50; a held-out sample tells you whether the "78% unclear"
   rate for TFA novel generalizes beyond the very top.

Output: updated research log if anything changes; otherwise a short
"sanity-check PASSED" note appended to the bottom of the log with the
verification commands.

### Phase 2 — Evaluate NeurIPS-grade readiness (half a day of thinking)

The qualitative findings exist. Are they strong enough to anchor a
paper? Things a reviewer will poke at, in rough order of severity:

1. **Single everything.** One layer, one k for the qualitative
   comparison, one seed. Reviewer: "How do you know any of this
   generalizes?" You need at least one of {second layer, second k,
   second seed} to claim the TFA novel vs TFA pred partition is a
   property of the arch, not a training accident.
2. **TFA under-training.** NMSE 0.12 vs 0.06 for stacked at the same
   capacity. Reviewer: "You're comparing a well-trained SAE to a
   half-trained TFA and claiming TFA loses." You need to either
   (a) retrain TFA with a more stable recipe (smaller LR, larger
   batch, or a revised `lam` normalization) until NMSE parity, or
   (b) control for reconstruction by reporting the findings at
   *matched* NMSE — e.g. only compare Crosscoder features at NMSE
   0.12 (underfit it deliberately) against TFA at 0.12.
3. **Decoder cosine as a matching metric is weak.** The log found
   "TFA orthogonal to stacked and crosscoder" by decoder cosine, but
   also showed stacked at pos 0 vs stacked at pos 1 only shares
   ~2400 / 18 432 features. Same-arch within-position shift already
   breaks cosine matching; cross-arch cosine might be an artifact.
   A content-based matching metric (same top-K activating inputs
   regardless of weights) would be a stronger claim. The scan JSONs
   have `(chain_idx, window_start)` tuples — you can compute
   set-overlap on those between pairs of features.
4. **"Tokenization-boundary feature family in TFA novel" is based
   on 11 labeled features.** Need to scale to maybe 100+ labels for
   this family to be a real claim rather than an anecdote.
5. **No downstream utility demonstration.** Does "TXCDR wins on
   high-span" actually matter for any end task (feature steering,
   probing, circuit analysis)? Paper needs at least one such hook.

Output: a markdown file `docs/han/research_logs/2026-04-XX-neurips-gap.md`
listing which of the above blockers you think are deal-breakers,
which are fixable with a few hours of targeted work, and which you
propose to defer.

### Phase 3 — Targeted extensions (ranked by value/effort)

Pick the highest-value extension(s) that fit in a session. **The
goal is to answer qualitative questions, not to run long sweeps.**

1. **(Cheapest) Second-layer replication on Gemma.** Activations for
   `resid_L13` are *already cached* (see
   `data/cached_activations/gemma-2-2b-it/fineweb/resid_L13.npy`).
   Training 3 archs at k=50 on that cache is ~4-6h on the A40 —
   bump it up if you want k=100 too. Then rerun
   `scan_features.py`, `temporal_spread.py`,
   `tfa_pred_novel_split.py`, `high_span_comparison.py`,
   `explain_features.py`, `plot_autointerp_summary.py` pointing at
   `--layer-key resid_L13`. This is the single most valuable
   replication and costs half a day.
2. **DeepSeek-R1-Distill-Llama-8B** (bigger model, different
   architecture family). Activation caching is ~3h; training a
   single `(arch, k)` is ~30-90 min on A40 (d_model=4096,
   use `--expansion-factor 4` per the existing sweep script).
   **Recommend: cache only one layer** (e.g. resid_L12), train
   *only unshuffled at k=50* for all three arches. That's ~5-6h
   total. The question to answer: does the TFA novel vs TFA pred
   partition generalize to DeepSeek? Does crosscoder still win on
   high-span?
3. **Second dataset on the existing Gemma model.** Math (GSM8K) or
   code (HumanEval) instead of FineWeb, same layer/k. Tests
   whether the high-span feature types differ by domain. Caching
   math is cheap (a few K sequences), training is the same as
   layer 1. Value: if crosscoder's high-span features in code
   look nothing like in FineWeb, that's a good qualitative
   discovery; if they look the same, "general-purpose feature
   library" is a supportable claim.
4. **Better TFA training.** Specific hypotheses to test (pick one,
   not all):
   - Drop `lam` from `1/(4*d_in)` to something larger like
     `1/(d_in)` so novel_codes aren't squashed by ~4x. Activations
     are currently magnitude 0.03 — lifting them to ~0.12 might
     change which features get topk-selected.
   - Warm up more slowly (5k steps warmup over 10k total instead
     of 200 over 10k).
   - Reduce LR from 3e-4 to 1e-4 for TFA only.
   Each experiment: 2h train + 30 min analysis. Don't retrain all
   archs — just TFA. Compare new TFA novel exemplars vs the old
   log's "mostly padding" finding.
5. **Content-based feature matching** (code change, not a retrain).
   For each pair of archs, compute per-feature set of
   `(chain_idx, window_start)` tuples from the top-K exemplars,
   then Jaccard-match features across archs. This is the
   alternative to decoder cosine that the reviewer will ask for.
   Pure analysis, no GPU needed beyond what's already run.

### Explicit non-goals

- Don't re-run the full 16-hour `scripts/run_nlp_sweep_16h.sh`.
  Invocation 4 (`resid_L13` shuffled) is the only combination not
  already trained, and the scientific question doesn't need all 4
  variants × 2 layers × 2 k.
- Don't train a new subject LM from scratch. Gemma-2-2B-IT and
  DeepSeek-R1-8B are both in the registry (`src/bench/model_registry`);
  pick one of those.
- Don't fiddle with the NMSE comparison if you can't close the
  TFA training gap. Either fix TFA or move the comparison to a
  non-NMSE axis (content-based feature matching, coverage of
  known linguistic categories, steering efficacy on a fixed task).

## Environment setup (unchanged from previous session)

The uv venv at `/workspace/temp_xc/.venv` and HF cache at
`/workspace/hf_cache` are both on RunPod's persistent `/workspace`
volume. First session creates them, later sessions reuse.

```bash
cd /workspace
[ -d temp_xc ] || git clone https://github.com/chainik1125/temp_xc.git
cd temp_xc
git fetch origin && git checkout han-runpod && git pull

# Install uv if not present
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# Persist env vars
export HF_HOME=/workspace/hf_cache
export UV_LINK_MODE=copy
for line in 'export HF_HOME=/workspace/hf_cache' 'export UV_LINK_MODE=copy'; do
    grep -qF "$line" ~/.bashrc || echo "$line" >> ~/.bashrc
done

# Resolve deps (safe to re-run)
uv sync
source .venv/bin/activate

# Verify
PYTHONPATH=/workspace/temp_xc python -c "
import torch
print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB')
from src.bench.model_registry import list_models; print('Models:', list_models())
"

# Only needed the first time — token persists at $HF_HOME
huggingface-cli whoami >/dev/null 2>&1 || huggingface-cli login
```

### Reconnecting to an existing pod

```bash
cd /workspace/temp_xc
source .venv/bin/activate
git checkout han-runpod && git pull
```

### Claude API key (for the LLM labeler)

Stored at `/workspace/.anthropic-key` (persistent volume, gitignored).
Every session:

```bash
export ANTHROPIC_API_KEY=$(cat /workspace/.anthropic-key)
```

**Rotate this key when done with the session** — it has been in
prior session transcripts.

### GitHub push access

Token at `/workspace/.github-token`; repo-local credential helper
already wired up in this clone. `git push origin han-runpod` should
just work. If it doesn't:

```bash
# One-time per repo clone
cd /workspace/temp_xc
git config --local credential.helper \
    '!f() { echo username=x-access-token; printf "password=%s\n" "$(cat /workspace/.github-token)"; }; f'
```

Use a **classic** PAT with `repo` scope. Fine-grained PATs silently
403 even when the user is a collaborator.

## Key scripts you'll likely need

All under `temporal_crosscoders/NLP/`:

| Script | Purpose | Typical runtime |
|---|---|---|
| `scan_features.py` | Top-K activating windows per feature, all 3 archs | 3-10 min / arch on A40 |
| `tfa_pred_novel_split.py` | Measure pred vs novel mass per TFA feature | 2 min |
| `temporal_spread.py` | Per-feature concentration across T=5 positions | 2-5 min / arch |
| `feature_match.py` | Cross-arch decoder cosine similarity | 2-5 min |
| `high_span_comparison.py` | The TXCDR-vs-TFA headline analysis + figure | 3 min |
| `explain_features.py` | Claude-Haiku labels for top-K features (needs API key) | ~90s / arch (rate-limited) |
| `plot_autointerp_summary.py` | Regenerate all 6 standard figures | 2 min |
| `bench_adapters.py` | Library: loads src.bench checkpoints with the right interface | (module) |
| `scripts/run_nlp_sweep_16h.sh` | The original 16h sweep — **do not re-run as a whole** | 16+ h |

All scripts take `--subject-model`, `--cached-dataset`, `--layer-key`,
`--k`, `--T` so you can retarget them at a new layer/model without
editing code.

## Reading order for a new agent

1. `docs/han/research_logs/2026-04-17-autointerp-initial.md` — all
   findings, all figures, all caveats. **Start here.**
2. This file — what to do next.
3. `CLAUDE.md` — repo conventions (uv, figure saving via
   `save_figure`, markdown/log conventions).
4. `temporal_crosscoders/README.md` — the original synthetic-data
   sweep that preceded the NLP work.
5. `temporal_crosscoders/NLP/bench_adapters.py` — the glue between
   the training-time `src.bench.architectures.*` classes and the
   autointerp pipeline. Read this before touching any script that
   loads a checkpoint.
