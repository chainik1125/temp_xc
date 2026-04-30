---
author: Aniket Deshpande
date: 2026-04-30
tags:
  - design
  - in-progress
  - ward-backtracking
---

## Stage B paper-budget agent brief — pod-Claude on 2× H100

> Read [[plan|ward_backtracking/plan]] (Stage A scope + Stage B framing),
> then [[results|ward_backtracking/results]] (Stage A baseline numbers
> you need to beat / match), then [[results_b|ward_backtracking/results_b]]
> (Stage B sprint findings: TXC f1444 was 1.64× DoM at +16 *but* with
> "Wait Wait Wait..." collapse). Then this brief. Then start working.

### TL;DR

Run the full Stage B paper-budget pipeline end-to-end on a fresh 2× H100
pod, then write `results_b.md` with the new numbers + figures.

```bash
# After source scripts/runpod_activate.sh + nvidia-smi -L showing 2 GPUs:
bash experiments/ward_backtracking_txc/run_paper_pipeline.sh
```

This runs Phase 1 (sweep) → Phase 1.5 (rank winner) → Phase 2 (greedy
hill-climb from winner) → Phase 6 (plots), with held-out FVU early
stopping on every TXC training so wasted compute is bounded. **Total
expected wall: 8–12 hours on 2× H100.** Idempotent — every phase script
skips already-completed cells, so resuming from a crash is just
re-running the same command.

When you finish, update `docs/aniket/experiments/ward_backtracking/results_b.md`
with the new headline number, the hill-climb history, and a
coherence-aware comparison vs DoM. Then commit + push.

### What the experiment is

Stage A reproduced Ward 2025: a Difference-of-Means vector derived from
**base** Llama-3.1-8B activations, when added to the residual stream of
**DeepSeek-R1-Distill-Llama-8B**, induces backtracking ("Wait" / "Hmm"
in the chain-of-thought). The DoM vector lifted keyword rate from 0.7%
baseline to 4.6% at mag=+16.

Stage B asks: can a **base-only TemporalCrosscoder feature**, used as
the steering direction, match or beat that DoM vector? If yes,
TXC-as-feature-extractor recovers the same direction DoM finds in the
base model — a structural-prior win for the temporal axis. If no, DoM
is the right primitive at this scale.

The sprint version of Stage B (3k training steps, 2 archs, 2
hookpoints, 1 seed) gave a noisy positive: TXC `resid_L10_f1444_pos0`
hit 5.8% at mag=+16, **1.64× DoM(base)**'s 3.5%, BUT the +16 cell
collapsed into "Wait Wait Wait..." loops (max-same-word-run = 16).
The honest fair-coherence comparison was at mag=+12, where TXC ≈ DoM.

This paper-budget rerun has four jobs:

1. **Train to convergence** (15k step cap with held-out FVU early
   stopping) so the dictionary isn't undertrained.
2. **Sweep all 4 dictionary architectures** (TXC, TopK SAE, Stacked
   SAE, Han's TSAE) at all 3 hookpoints (resid, ln1, attn) at
   matched sparsity. Place TXC's result in a population of dictionary
   methods rather than a single comparison vs DoM.
3. **Hill-climb from the sweep winner** along atomic perturbation axes
   (arch swap, hookpoint swap, k_per_position scale) until no neighbor
   improves the metric, with a 4-iteration cap.
4. **Use a coherence-aware metric** so we don't reward degenerate
   collapse like the sprint did at +16.

### Pipeline architecture

```
Phase 1 (sweep, 6h on 2× H100)
    cache_activations.py        cache base Llama-3.1-8B activations at all 3 hookpoints
    train_txc.py                train 4 archs × 3 hookpoints = 12 cells, sharded across 2 GPUs
    mine_features.py            top-32 features per cell, ranked by D+/D- selectivity
    b1_steer_eval.py            steer reasoning model with each cell's top-K features × {pos0,union}
    b2_cross_model.py           per-offset firing curves on reasoning traces (cross-model diff)

Phase 1.5 (rank, 1 min CPU)
    rank_phase1.py              compute hill-climb metric per cell, write rank_phase1.json

Phase 2 (hill-climb, 4–6h on 2× H100)
    hill_climb.py               iterate: enumerate neighbors → evaluate_cell × N → pick best
    evaluate_cell.py            single-cell pipeline: train + mine + B1-per-cell + metric

Phase 2.5 (multi-seed verify, ~1h on 2× H100)
    verify_seeds.py             train hill-climb winner at 2 additional seeds, compute σ.
                                Single-seed numbers are not paper-grade; reviewers will
                                ask. We always verify before declaring a winner.

Phase 6 (plots, 2 min CPU)
    plot/*.py                   refresh all figures from final state
```

### Hill-climb spec

State: `results/ward_backtracking_txc/hillclimb_state.json`
- `current_best`: {cell_id, metric}
- `iteration`: 0..MAX_ITER (default 4)
- `history`: list of accepted moves
- `evaluated`: {cell_id: metric} for every cell touched (sweep + hill-climb)

**Perturbation axes** (single-axis atomic moves from current cell):

| axis | step | clamp |
|---|---|---|
| arch | swap to any other arch in `cfg.txc.arch_list` | — |
| hookpoint | swap to any other enabled hookpoint | — |
| k_per_position | scale ×0.5 or ×2 | clamped to [4, 256] |

For a winner like `tsae__ln1_L10__k32__s42`, the neighbors are:
- arch swaps: txc / topk_sae / stacked_sae × ln1_L10 × k32 × s42
- hookpoint swaps: tsae × {resid_L10, attn_L10} × k32 × s42
- k swaps: tsae × ln1_L10 × {k16, k64} × s42

= 7 neighbors per iteration.

**Stop conditions** (in order):
1. Hit `--max-iter` (default 4).
2. No neighbor improves `primary_kw_at_coh` by > `--improvement-threshold`
   (default 5% relative).
3. All neighbors already evaluated (true local maximum on the integer
   grid).
4. Subprocess failure on every neighbor in an iteration.

**Acceptance rule**: accept the best successfully-evaluated neighbor
if its score exceeds current × (1 + threshold). Tie-break by lower
arch+hookpoint+k tuple lex order so the iteration is deterministic.

### Metric (the hill-climb objective)

```
primary_kw_at_coh = max over magnitudes m, sources s of |kw_rate(s, m) - 0.007|
                    subject to: max_consecutive_same_word_run ≤ 2
                                across every prompt in cell (s, m)
```

Implemented in `metrics.py:cell_metric`. Reasoning:
- `|kw_rate - baseline|` rather than `kw_rate` directly — both positive
  and negative steering count (Dmitry's neg-steering ask). The sign of
  the best magnitude is recorded as "direction".
- Coherence floor `max-run ≤ 2` excludes "Wait Wait Wait..."
  degeneration (the lesson from the sprint).
- Peak across magnitudes (not at fixed mag) so cells with different
  ⟨|z|⟩ scales are comparable after norm-rescaling.

Also computed and saved per cell:
- `peak_kw_no_coh`: the unfiltered peak (the sprint's headline number)
- `frac_coherent`: fraction of (mag × source) cells that pass the
  coherence floor — diagnostic for "is this cell's whole curve usable?"
- `best_magnitude`, `best_source`, `direction`

### Knobs locked in `config.yaml`

```yaml
txc:
  T: 6                             # Ward window
  d_sae: 16384                     # 4× expansion of d_model=4096
  k_per_position: 32               # window L0 = 192
  train_steps: 15000               # cap; early-stop fires earlier in practice
  early_stop:
    enabled: true
    patience: 10                   # 10 × log_interval(=100) = 1000 steps no improvement
    min_rel_improvement: 0.005     # ≥ 0.5% relative drop in held-out FVU
    warmup_steps: 1000
  arch_list: [txc, topk_sae, stacked_sae, tsae]
  arch_kwargs.tsae: {n_heads: 8, bottleneck_factor: 64, sae_diff_type: topk}

cache:
  num_sequences: 3300
  seq_length: 256
  stride: 128                      # 50% overlap → 2× more windows from Stage A traces

mining:
  top_k_features: 32
  top_k_for_steering: 4

steering:
  magnitudes: [-16, -12, -8, -4, 0, 4, 8, 12, 16]   # symmetric per Dmitry
  max_new_tokens: 1500
  n_eval_prompts: 20
  gen_batch_size: 8                # H100 has the headroom
```

If you're tight on time and need to scope down, in priority order:
- Reduce `train_steps` to 10,000 (early stopping saves you anyway).
- Drop `tsae` from `arch_list` (it's the heaviest; TXC vs StackedSAE vs
  TopKSAE is still a clean comparison).
- Reduce `mining.top_k_for_steering` from 4 to 2 (cuts Phase 2's per-cell
  B1 in half).
- Cap `hill_climb --max-iter` to 2 instead of 4.

### Multi-GPU dispatch

`run_grid_2gpu.sh` (Phase 1) and `hill_climb.py` (Phase 2) both detect
the GPU count via `nvidia-smi -L | wc -l` and fan cells across them
with `CUDA_VISIBLE_DEVICES=k` per subprocess. The pool size is the GPU
count; the in-flight jobs never exceed that. Stage B's code uses bare
`"cuda"` device strings, so per-process GPU pinning Just Works.

To force GPU count: `NUM_GPUS=2 bash …` or `--num-gpus 2`.

### What to do when something fails

| symptom | likely cause | fix |
|---|---|---|
| `Loading weights:` hangs | HF model download slow/blocked | wait 10 min; if no movement, check `HF_TOKEN` and `nvidia-smi` GPU 0 mem usage |
| `[fatal] no GPUs detected` | nvidia-smi missing or pod not GPU-attached | reboot pod; verify `nvidia-smi` works as root |
| `OOM` during TSAE training | TSAE attention layer at d_sae=16k is heavier | drop `d_sae` to 8192 in cfg or `arch_kwargs.tsae.bottleneck_factor` to 128 |
| `tokenizer decode produces Ġ literally` | transformers byte-level decode bug | already worked around via `_fix_byte_decode` in `b1_steer_eval.py`; verify the fix is still being called |
| Hill-climb picks the same cell as winner repeatedly | metric is too noisy, threshold too low | increase `--improvement-threshold` from 0.05 to 0.10 |
| Per-cell B1 takes forever | DoM is being re-evaluated per cell | confirm `evaluate_cell.py` calls B1 with `--no-dom` (default) |
| Some cells fail with "no ckpt" | training subprocess crashed; check `/tmp/hillclimb_<id>_gpu*.log` | re-run `bash run_paper_pipeline.sh` — it's idempotent |

### Coordination / safety

- **Branch**: `aniket-ward-stage-b`. Commit directly. Push with the GH
  PAT in `.env` (rotate after the run).
- **No interaction with `aniket-phase7-y`** — different agent, different
  experiment, do not push there.
- **HF_TOKEN required** for Llama-3.1-8B (gated). DeepSeek-R1-Distill is
  not gated.
- **ANTHROPIC_API_KEY optional** — only used if you want to re-run Stage
  A's `validate.py`. Stage B's metric is local (keyword + coherence).

### Writeup mandate (when pipeline finishes)

Update `docs/aniket/experiments/ward_backtracking/results_b.md` (or
write a fresh `results_b_paper.md` and link it) with:

1. **TL;DR table** — replace sprint headline with paper headline. Use
   coherent magnitudes only (frac_coherent ≥ 0.8).
2. **Phase 1 leaderboard table** — top 5 cells from `rank_phase1.json`,
   columns: cell_id / arch / hookpoint / k / primary_kw_at_coh /
   peak_kw_no_coh / best_magnitude / direction / frac_coherent / n_sources.
3. **Hill-climb history** — for each accepted move, show
   from→to / Δ score / which axis moved.
4. **Final winner cell** — its training curves (FVU plateau? early-stop
   step?), its B1 curve vs DoM, its B2 per-offset firing.
5. **Per-architecture comparison** — for the winning hookpoint,
   side-by-side: TXC / TopK SAE / Stacked SAE / Han's TSAE vs DoM.
   This is the "TXC vs SAE/T-SAE baselines" plot Dmitry asked for.
6. **Coherence audit** — the same `coherence.png` from the sprint, but
   now showing all top cells, with the magnitude where each cell's curve
   leaves the coherent regime annotated.
7. **Outcome verdict** vs the pre-registered B1 outcomes (Positive /
   Mixed / Negative) — be honest; if the sprint's 1.64× claim doesn't
   hold at coherent magnitudes after full training, say so explicitly.

Push:
```bash
cd /workspace/aniket/temp_xc
git add docs/aniket/experiments/ward_backtracking/{images_b,results_b.md,agent_brief.md}
git -c user.name="aniket-desh" -c user.email="aniket4@illinois.edu" \
    commit -m "Stage B paper-budget run results"
GH=$(grep ^export\ GH_TOKEN .env | cut -d'"' -f2)
git push "https://chainik1125:${GH}@github.com/chainik1125/temp_xc.git" \
    HEAD:refs/heads/aniket-ward-stage-b
```

### Provenance / reading order

- This brief: `docs/aniket/experiments/ward_backtracking/agent_brief.md`
- Plan: `docs/aniket/experiments/ward_backtracking/plan.md`
- Stage A results: `docs/aniket/experiments/ward_backtracking/results.md`
- Stage B sprint results (read for context, esp. coherence lesson):
  `docs/aniket/experiments/ward_backtracking/results_b.md`
- Code: `experiments/ward_backtracking_txc/`
- Vendored TSAE: `temporal_crosscoders/han_tsae/saeTemporal.py`
  (from `references/TemporalFeatureAnalysis/sae/saeTemporal.py` on
  Han's `han-phase7-unification` branch, attention-based predicted+novel
  codes)
