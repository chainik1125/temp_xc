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

### Scale-up results (100M tokens, stream-trained overnight 2026-08-10/11)

Training (2 seeds/arm, Modal L40S, single-pass fresh tokens): recon NMSE
0.279/0.293 dead 31%; dsm 0.303/0.316 dead 57%; **dsm_anneal 0.291/0.304
dead 41%** — the annealed arm recovers most of the NMSE gap and a third of
the death gap, as preregistered. Llama-3.1-8B ln1_L10 pairs (82M tokens):
recon 0.313 dead 0.0%; dsm 0.346 dead 4.6% — hookpoint geometry dominates
death. Dictionary death grows with budget for every objective (recon
7.5% → 31% from 10M → 100M).

Evals (2 seeds/arm; autointerp skipped — stream-trained checkpoints have
no context dumps):

| metric | recon | dsm | dsm_anneal |
| --- | --- | --- | --- |
| absorption (↓) | 0.494 | **0.349** | 0.410 |
| fragility support-Jaccard @ ε=0.5 (↑) | 0.598 | **0.743** | 0.656 |
| first-letter decodability | 0.910 | 0.912 | 0.916 |
| sparse probing (12 cells) | mixed | mixed | mixed |

Verdicts at 10× budget:

- **Structural gains replicate and sharpen**: absorption *grows with
  training* for reconstruction (0.306 → 0.494) far faster than for DSM
  (0.179 → 0.349) — reconstruction training progressively absorbs
  features; denoising resists it. Fragility likewise (recon 0.627 → 0.598,
  dsm 0.762 → 0.743).
- **dsm_anneal lands between the parents on every axis** (predicted):
  most of recon's NMSE, most of dsm's structure, a third less death.
- **Sparse probing stays a wash for all arms** — and notably dsm_anneal's
  reduced death did NOT unlock a probing win, weakening the "dead-capacity
  tax" explanation and strengthening the alternatives: per-token
  topic/sentiment concepts are too dense/easy to discriminate objectives,
  and the detection advantage should live in temporal/behavioural
  structure (next: backtracking detection on the Llama pairs).

### Post-hoc gate swap (σ→0-slice ablation, 2026-08-11)

Same trained 100M checkpoints, TopK replaced at eval by rate-calibrated
per-latent threshold gates (mean L0 = 40 matched; per-token std ≈ 10).
Result: **weight-borne properties don't move** (absorption identical to
three decimals; fragility equal or slightly lower) while **readout-borne
properties improve for every arm** — k=5 sparse probing +3.2 to +5.3
points (recon 0.789 → 0.835, dsm 0.779 → 0.811, dsm_anneal 0.787 →
0.840), k=1 flat. Interpretation per the MMSE note: absorption and
robustness live in the dictionary; the gate shape governs readout, and
the Bayes-limit gate (variable L0, no winner-take-all) reads better than
TopK regardless of training objective. Free eval-time improvement;
strengthens the case for training the full σ-conditioned `bayes_gate`
family. The arm-vs-arm probing wash persists under both gates.

### Trained bayes_gate arms at signs-of-life scale (2026-08-11 eve)

Two σ-conditioned `bayes_gate` SAEs (per-latent rate-KL sparsity, DSM
objective, 1 seed, ~24M streamed tokens vs the TopK arms' 10M cache —
budget bias favours bayes) evaluated with the identical suite and eval
shard, gate conditioned at σ=0 (JumpReLU limit), hard threshold 0.5,
z = m·1[g>0.5]. Checkpoints `bayes_gate/bg6_sol` (L0 ≈ 68, NMSE 0.291)
and `bg7_sol` (L0 ≈ 49, NMSE 0.344); results
`logs_bayes_evals/evals_bayes_*.json`.

| arm | L0 | absorb ↓ | 1L-acc ↑ | probe k=5 ↑ | probe k=all ↑ | frag ε=.5 ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| recon (2s mean) | 40 | 0.306 | 0.928 | 0.724 | 0.918 | 0.627 |
| dsm (2s mean) | 40 | 0.180 | 0.918 | 0.737 | 0.908 | 0.762 |
| bg6_sol | 68 | **0.137** | **0.943** | **0.835** | **0.941** | 0.523 |
| bg7_sol | 49 | 0.217 | 0.935 | 0.768 | 0.939 | 0.515 |

Split verdict against the pre-registered readings:

- **Interpretability wins are real**: bg7 (nearest matched L0) cuts
  absorption 29% below recon; bg6 is the lowest-absorption arm we have
  trained, below even TopK-dsm. k=5 sparse probing jumps +7–11 points
  over every TopK arm — the post-hoc gate swap's +3–5 readout gain
  roughly doubles when the gate is trained in, and it breaks the probing
  wash for the first time. k=all probing best of all arms (no dead
  capacity). First-letter probe acc best of all arms.
- **Robustness regression — dictionary-borne, not a readout artifact**
  (settled by `modal_frag_sigma.py`, results
  `logs_bayes_evals/fragility_sigma_matched.json`): support-Jaccard is
  the worst of any arm (0.52 vs dsm's 0.76 at ε=0.5), and neither
  σ-matched conditioning (encode the perturbed input with u=ε², i.e.
  the model used as trained: 0.54–0.55) nor a rank/top-k readout
  (TopK-style relative support: 0.55–0.56) recovers it. The bayes
  encoder directions are genuinely less noise-stable than the
  TopK-dsm dictionary's. Working interpretation: the adaptive gate
  *externalizes* noise-handling into the conditioning channel, freeing
  the dictionary to specialise — which is plausibly the same property
  that buys the absorption and probing wins. Stable-causal-handle
  robustness (motivation 1a) belongs to DSM-TopK, not to bayes_gate;
  the two benefits currently do not co-occur in one arm.
- Caveats: 1 seed per bayes arm, token budget not matched (favours
  bayes), L0 not exactly matched (68/49 vs 40; within the bayes family
  higher L0 tracks lower absorption, so part of bg6's absorption edge
  may be L0-driven — bg7 is the honest comparison point).

Program call: the arm survives — first single change to move probing —
but its steering claims now rest on the wave-2 `w6_bayes`/future runs,
not on this table.

### Files

- `cache_activations.py` — one-off cache job (model forwards on Modal).
- `sae.py` — TopK+AuxK SAE (shared by both arms).
- `train_arms.py` — training loop + in-train evals.
- `modal_run.py` — cache job + 4 detached training jobs, volume-committed
  JSONL/weights (the usual dropout-proof pattern).
