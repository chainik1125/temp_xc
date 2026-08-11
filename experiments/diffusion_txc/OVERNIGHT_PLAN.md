---
author: Claude (with Dmitry)
date: 2026-08-10
tags:
  - proposal
  - in-progress
---

## Overnight plan: O1 scale-up + SAEBench proper; O2 behavioural evals (backtracking + EM)

Two pipelines, both on Modal (detach + volume, per-stage commits), both
resumable from any stage. Everything below is preregistered intent;
deviations get logged in the results docs.

### O0 (immediate): publish current checkpoints to HuggingFace

Push the four Gemma-2-2B signs-of-life checkpoints (`recon`/`dsm` ×
seeds 0/1) plus training JSONLs and eval JSONs to a new HF model repo
(default: `dmanningcoe/diffusion-topk-saes-gemma2-2b`, private first),
with a model card stating recipe, cache, and the SUMMARY.md table. Uses
the existing `hf-write-dmc` secret. ~10 minutes, runs first so artifacts
are off-volume regardless of what happens overnight.

### O1 — scale-up + real SAEBench harness (Gemma-2-2B L12)

Goal: same comparison at 10× budget with the dead-feature fix arm, scored
by the actual SAEBench harness at the matched sparsity, against their
published 16k TopK baseline numbers.

1. **Cache extension**: grow the layer-12 cache 10M → 100M tokens
   (~460 GB bf16 in the `diffusion-txc` volume; ~3.5 h A10G; deleted
   after the run to stop storage billing).
2. **Training, six jobs** (A10G, ~2.5 h each, parallel): H=16,384, k=40,
   AuxK, lr 3e-4, one epoch over 100M tokens (24.4k steps):
   - `recon` × 2 seeds (control, unchanged recipe);
   - `dsm` × 2 seeds (unchanged σ ~ LogU(0.05, 1.0)·RMS — the clean
     scale test);
   - `dsm_anneal` × 2 seeds (σ_max annealed 1.0 → 0.3 over training —
     the preregistered dead-feature fix; prediction: dead fraction ≤
     recon + 5pts with absorption/fragility gains retained).
   Same in-train logging as the signs-of-life run (NMSE/FLOPs, chunked
   ΔCE, dead fraction).
3. **SAEBench proper**: `pip install sae-bench`, wrap our TopKSAE in
   their custom-SAE interface, run their harness — core metrics, sparse
   probing, absorption, SCR, TPP (autointerp/RAVEL if the judge keys and
   runtime cooperate) — for all six checkpoints, and place them on the
   published SAEBench 16k-width Pareto plots (their TopK k≈40 baseline is
   the anchor). One Modal job per checkpoint; their eval activation cache
   (~100 GB) built once into the volume.
   - Known integration risks, pre-accepted: package pins, custom-SAE API
     drift, judge-key names. Fallback if the harness fights back
     overnight: run our own eval suite at 10× budget and leave harness
     integration for the morning.

Budget estimate: cache ~$4 + training 6 × ~$3 + evals ~$10–15 ≈ **$35–45**.

### O2 — behavioural evals on the paper's models (backtracking, then EM)

Structural note, stated plainly: the signs-of-life seeds are Gemma
dictionaries; the paper's pipelines are Llama-3.1-8B (backtracking c7)
and Qwen (EM). Evaluating "recon vs dsm on the paper's behavioural evals"
therefore means training matched pairs on the paper's own models and
hookpoints, with the paper's recipes — which is also the cleaner
experiment.

**O2a — backtracking (Llama-3.1-8B, the c7 pipeline).**

1. Cache per the paper's c7 recipe (FineWeb 128-token windows, base
   Llama-3.1-8B via the ungated mirror, ln1_L10 hookpoint — matching the
   stage-B `topk_sae` we audited), ~30M tokens, A100.
2. Train `recon` and `dsm` × 2 seeds at the stage-B SAE config
   (d_sae=16,384, k=64/token), same DSM ladder scaled to this
   hookpoint's measured RMS.
3. Run the paper's **detection** protocol exactly (their scripts, as the
   FreqBench addendum did): max-pooled |activations| over the D+ window
   on the reasoning-trace cache, ℓ1-logistic, 5-fold GroupKFold, top-S
   features by train-fold t-statistic; report PR-AUC/ROC at S ∈ {8, 32}
   for all four dictionaries, alongside the stage-B originals.
4. Also report dead-on-traces fractions (the audit showed 54–68% for all
   window arms — prediction: dsm lower after the cleanup mechanism, but
   unresolved direction given the Gemma dead-fraction surprise).
   Steering is a stretch goal, not overnight-committed.

**O2b — EM (Qwen, the fragile one — gated pipeline).**

Target: recon-vs-dsm SAE pairs at the paper's EM hookpoint (L24 ln1 on
the EM model), evaluated by the paper's steering protocol. Because EM is
fragile, the pipeline is gated with the lessons-learned diagnostic
sequence before any expensive generation:

1. Train `recon`/`dsm` pairs on the EM model's L24 ln1 activations
   (cache ~20M tokens, A100/H100).
2. **Gate 1 — baseline agreement**: unsteered generations through our
   harness must reproduce the reference baseline align/coh numbers.
3. **Gate 2 — hook no-op**: α=0 steering must be bit-identical to no
   hook. Only then:
4. **Feature selection** per dictionary by the paper's own ranking
   protocol (same selection rule for both objectives — no per-arm
   tuning), then the **α-sweep**: dense grid (≥12 α values spanning past
   coherence collapse), delegating generation to the reference
   `generate_*` function (chat template stays theirs), per-prompt seed
   lists (base+i, never a single batch seed), paper judges via
   `em-sprint-judges`, and the full metric set: peak / baseline /
   min-at-coherence-floor alongside Δalign|coh≥70 — never the headline Δ
   alone.
5. Report per (objective × seed): the α-response curves with the paper's
   colour conventions.

O2 budget: caches ~$8 + training 8 jobs ~$25 + detection ~$3 + EM sweep
(~3–4k generations + judging) ~$20–30 ≈ **$60–75**. EM lands only if
Gates 1–2 pass; a gate failure stops that branch with a diagnostic
report instead of burning the sweep.

### Decisions (Dmitry, 2026-08-10 evening)

1. **EM target: Qwen2.5-14B finance-alignment / extreme-sports cells** —
   the 14B results are the interesting ones because the universal feature
   **F603** exists there as the comparison anchor (pin its exact
   definition from the c6 sweep outputs — `dmitry/pre_purified/
   c6_em_overnight/sweep_outputs/*/wang_full_extended.json` — during O2b
   implementation).
2. **Cost-minimized compute: try PSC Bridges-2 first** for the heavy
   stages (caches + training; GPU-shared, ~1 SU/GPU-hr against
   cis240096p), Modal as automatic fallback per stage if PSC
   auth/queueing fails. Requires Dmitry to authenticate the SSH control
   socket. Modal-only worst case ≈ $60–80.
3. **O2a backtracking dictionaries train to 20k steps** (the full-length
   version, not the stage-B early-stopped 3.6k–9.2k).
4. **HF repo: `dmanningcoe/diffusion-topk-saes`, PUBLIC** (storage
   headroom), subfolders per base model.

### Night log and standing orders (2026-08-10/11, times EDT)

State as of 23:55: Modal Gemma-6 (seeds 0/1) at ~5k/24k steps, ETA ~01:10;
Modal Llama-4 (seeds 2/3) at 250/20k, ETA ~02:45; PSC re-seeded gemma 2/3
(43323676-83, 5:30 caps) + llama 0/1 + qwen 0/1 all pending; spend
tracking ~$42 of $75 Modal cap; SUs untouched. Two fatal PSC bugs fixed
before they could bite (python3.6 venv bootstrap; ln1 hook layer-index
parse). Signs-of-life artifacts published to HF (public,
dmanningcoe/diffusion-topk-saes).

Standing orders (user asleep; no blocking requests):

- Stall in a Modal training job → accept-truncated at the last 5k-step
  checkpoint, record reduced budget in provenance.budget; restart from
  zero only if it died before 5k AND >2h of night remains.
- Dead-fraction climb in gemma arms → observe only; it is the phenomenon
  the dsm_anneal arm tests.
- Eval round fires on whatever gemma checkpoints exist by ~01:30.
- PSC jobs: leave queued; whatever backfills is bonus seeds.
- No new spend beyond the $75 Modal cap; no PSC beyond the queued jobs;
  Qwen/EM pipeline strictly gated until morning.
- Morning deliverable: consolidated report ~08:00 (per-job outcomes,
  eval table with provenance, spend, incidents), plus this log updated.

### Orchestration

Each pipeline is one detached Modal app with staged entrypoints (resume
at any stage). A monitor agent per pipeline with a background watcher
(the pattern from the signs-of-life run), reporting milestones and
stopping on gate failures. All artifacts volume-committed; morning
deliverable is two results tables + curves regardless of partial
failures.
