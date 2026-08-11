---
author: Claude (with Dmitry)
date: 2026-08-10
tags:
  - design
  - in-progress
---

## topk-vs-topkdiff: per-token TopK SAE, reconstruction vs DSM, on Gemma-2-2B

Signs-of-life comparison of a standard TopK SAE against the same
architecture trained with denoising score matching, tracking the training
evals of Gao et al.'s TopK paper
([Scaling and evaluating sparse autoencoders](https://openreview.net/forum?id=tcsZt9ZNKD))
scaled down. One known-good config, 2 seeds per arm, matched everything
except the loss:

```text
recon:    L = || f(x) - x ||^2
topkdiff: L = || f(x + sigma*RMS*eps) - x ||^2,  sigma ~ LogUniform(0.05, 1.0)
```

### Setup

- Model/site: Gemma-2-2B, layer-12 residual stream (post-block; HF
  `hidden_states[13]`), d = 2304 — SAEBench's standard site.
- Data: `monology/pile-uncopyrighted`, ctx 128, ~10M token cache (bf16
  shards + token ids in the Modal volume `diffusion-txc`), one held-out
  eval shard + 64 full sequences for patch-in evals. SAEBench baselines
  use 500M tokens; this is a deliberate 2% signs-of-life budget.
- SAE: width H = 16384, TopK k = 40, AuxK (k_aux = 512, coef 1/32, dead
  window 800 steps) for both arms (Gao-faithful; their plain-TopK dead
  pathology is known from the c7 replication). Pre-bias (b_dec), unit-norm
  decoder columns. Adam lr 3e-4, batch 4096 tokens, 6000 steps (~2.5
  reshuffled epochs).
- Arms: `recon` and `dsm`, seeds {0, 1} each — 4 training jobs from the
  same cache.

### Training evals (logged to JSONL per run)

1. **NMSE vs steps and vs cumulative FLOPs** — eval NMSE is always
   computed on *clean* held-out activations for both arms (the DSM arm's
   train loss is on corrupted inputs and is not comparable). SAE training
   FLOPs/token ≈ 12·d·H (two matmuls, fwd + backward ≈ 3× fwd); identical
   across arms at equal steps by construction — the axis is there so later
   budget-mismatched runs stay comparable. Cache-generation FLOPs reported
   once, amortized across arms.
2. **Patch-in ΔCE every 500 steps** — 64 sequences × 128 ctx: clean CE,
   CE with layer-12 residual replaced by the SAE reconstruction, and
   mean-ablation CE; report ΔCE = CE_patched − CE_clean and
   loss-recovered = (CE_abl − CE_patched)/(CE_abl − CE_clean).
3. **Dead-latent fraction** every eval (no fire in the last 800 steps).
4. **Autointerp inputs** at end of training: for 128 random alive latents,
   the top-12 activating contexts (decoded strings) dumped to JSON; the
   LLM-judge scoring pass runs as a separate follow-up job.

### Preregistered expectations (2026-08-10)

- Clean NMSE: `recon` better (by design — this is the metric that
  historically punished denoising; do not headline it).
- ΔCE / loss-recovered: genuine open question — the guardrail metric. If
  the DSM arm's loss-recovered collapses, its codes describe a denoised
  fiction rather than the model, and the LLM program needs a σ floor.
- Dead latents: exploratory prediction — DSM lower (corruption keeps
  marginal latents firing).
- Autointerp (when judged): DSM ≥ recon via junk-tail cleanup (B3
  mechanism). Detection-style evals (sparse probing, absorption,
  fragility) are the follow-up where we expect the real separation; this
  experiment is the training-dynamics gate before them.

### Results (2026-08-10 run; step 6000, ~25 min/job on A10G)

| arm | seed | NMSE clean | ΔCE | loss recovered | dead frac |
| --- | --- | --- | --- | --- | --- |
| recon | 0 | 0.2988 | 0.468 | 0.907 | 7.7% |
| recon | 1 | 0.2979 | 0.459 | 0.909 | 7.5% |
| dsm | 0 | 0.3311 | 0.746 | 0.852 | 28.3% |
| dsm | 1 | 0.3306 | 0.849 | 0.831 | 25.5% |

Shared baselines: CE clean 5.679, CE mean-ablate 10.712. All curves healthy
and monotone; nothing converged (NMSE still falling, loss-recovered still
rising at 6000 for both arms — numbers are lower bounds).

**Scoring vs the preregistration:**

- NMSE: recon better, as preregistered (gap ≈ 35× seed spread). Not the
  headline metric, by prior commitment.
- Loss-recovered guardrail: **passed with a visible tax** — DSM 0.84 vs
  recon 0.91, far from collapse; DSM codes still describe the model. DSM
  is also noisier across seeds on the CE metrics specifically (ΔCE spread
  0.10 vs recon's 0.01).
- Dead latents: **exploratory prediction falsified, inverted** — DSM
  25–28% dead (3.6× recon's plateaued 7.5%) and still rising at 6000.
  Two rival readings, decidable from the saved artifacts: (a) *junk-tail
  pruning* — the B2/B3 mechanism (noise atoms are worthless to a
  denoiser) expressed as death rather than cleanup, in which case DSM's
  dead latents are disproportionately ones recon would have kept as
  noise-reconstructors; or (b) *capacity starvation* — high-σ batches
  concentrate top-k on robust features and fine-feature latents starve.
  (a) is benign-to-good; (b) argues for a lighter σ upper bound or
  σ-annealing. The autointerp dumps + firing stats for all four
  checkpoints can distinguish them.
- **None of this is the detection claim.** These are training-fidelity
  evals, where the whole program predicts recon looks better; the payoff
  metrics (sparse probing, absorption, fragility, judged autointerp) run
  next on these four checkpoints (`*_final.pt` in the volume).

### Eval round (PREREGISTERED 2026-08-10, before any eval ran)

SAEBench-style but scaled (documented deviations): sparse probing
(ag_news / amazon_polarity / dbpedia subsets; latent means per sequence;
top-k by class-mean-diff, k ∈ {1, 5, 20, all}), absorption-lite
(first-letter task on the eval shard; main-latents-inactive-while-probe-
correct rate), LLM-judged autointerp (detection protocol on the saved
top-context dumps), and the fragility eval SAEBench lacks (support
Jaccard + probe-prediction flip rate under ε·RMS input perturbations).

Predictions: DSM ≥ recon on sparse probing at small k; absorption lower
(corruption breaks co-occurrence economics); fragility better (it is the
training objective); judged autointerp ≥ recon via the junk-tail
mechanism — all despite recon's better NMSE/loss-recovered. If DSM loses
detection too, the LLM transfer of the synthetic story fails and we say
so.

### Eval results (2026-08-10; 2 seeds/arm, arm means)

| eval | recon | dsm | prereg verdict |
| --- | --- | --- | --- |
| sparse probing k=1 (3 datasets) | 0.494 / 0.603 / 0.594 | 0.470 / 0.581 / 0.631 | **not confirmed** — mixed, leaning recon |
| sparse probing k=5 | 0.570 / 0.670 / 0.933 | 0.674 / 0.625 / 0.909 | mixed (DSM +0.10 on ag_news, else recon) |
| sparse probing k=all | 0.886 / 0.884 / 0.982 | 0.866 / 0.872 / 0.986 | recon slightly ahead (tracks NMSE) |
| absorption rate (↓) | 0.306 | **0.179** | **confirmed, large** (−42% rel., probe acc matched 0.93/0.92) |
| fragility: support Jaccard @ ε=0.5 (↑) | 0.627 | **0.762** | **confirmed** (better at every ε, gap grows with ε; probe-flip sub-metric at floor for both) |
| judged autointerp (balanced acc) | 0.652 | 0.672 | directionally confirmed, ~1 SE |

**Verdict.** The structural predictions transferred: DSM features absorb
far less, their supports are much more perturbation-stable, and they are
slightly more explainable — at matched first-letter decodability. The
raw detection advantage did **not** transfer at this budget: small-k
sparse probing is mixed-to-recon-favoring, unlike the synthetic settings
where probe accuracy was DSM's headline win. Candidate explanations, all
testable: (i) 2% training budget + DSM's 25–28% dead latents cost exactly
the capacity small-k probing needs; (ii) topic/sentiment concepts are
dense, easy signals where recon codes already suffice — the synthetic
wins were on *temporally-bound* structure, which per-token evals cannot
see by construction; (iii) two seeds, and DSM's seed spread on probing
cells is large. Next steps in order: dead-latent mitigation for the DSM
arm (σ-annealing or AuxK-on-corrupted), a 5–10× budget run, then the
actual target — the temporal (TXC) version evaluated on behavioural
detection, where the theory says the advantage lives.

### Files

- `cache_activations.py` — one-off cache job (model forwards on Modal).
- `sae.py` — TopK+AuxK SAE (shared by both arms).
- `train_arms.py` — training loop + in-train evals.
- `modal_run.py` — cache job + 4 detached training jobs, volume-committed
  JSONL/weights (the usual dropout-proof pattern).
