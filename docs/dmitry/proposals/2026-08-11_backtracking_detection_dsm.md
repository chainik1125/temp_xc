---
author: Claude (backtracking-eval agent)
date: 2026-08-11
tags:
  - results
---

## Backtracking detection on the recon/dsm Llama dictionaries

**TL;DR.** On the paper's backtracking-detection protocol, the four newly
trained per-token dictionaries (reconstruction- and denoising-trained
alike) sit at the raw-activation floor: no per-token dictionary of either
objective adds detection signal over raw `ln1_L10` activations, and the
two objectives do not separate (PR-AUC@S=8 ≈ 0.18–0.19 vs raw 0.19, base
rate 0.126). The paper's own ordering reproduces — the temporal (TXC)
arms lead the table (0.215–0.243) — consistent with the program's thesis
that behavioural detection signal is temporal, not per-token. The
post-hoc gate swap, which bought +3–5 probing points on Gemma, transfers
**nothing for per-token dictionaries** (all ≤ ±0.01, signs flip across
seeds); the one suggestive exception is the stage-B **window** TXC
(+0.014, the table's largest move — one arm, one seed). One strong asymmetry did appear: on the reasoning-trace
distribution the dsm dictionaries are ~50% dead vs ~10% for recon —
distribution shift punishes denoising-trained ln1 dictionaries far
harder than reconstruction-trained ones.

### What was run

Four per-token TopK SAEs trained overnight on base Llama-3.1-8B `ln1_L10`
activations (FineWeb, `d_in` = 4096, *H* = 16,384, *k* = 64/token, 20k steps
≈ 82M tokens) were put through the TXC paper's c7 *backtracking-detection*
protocol, alongside the Stage B originals, in one identical pipeline.

- Arms under test: `recon_s2`, `recon_s3`, `dsm_s2`, `dsm_s3` from
  `dmanningcoe/diffusion-topk-saes`, subfolder `llama31-8b-ln1L10-20k/`.
  The `recon` arms minimise plain reconstruction; the `dsm` arms minimise
  reconstruction of the *clean* activation from an input corrupted with
  Gaussian noise at sigma ~ LogUniform(0.05, 1.0) x RMS (denoising score
  matching).
- Baseline at the same hookpoint: `topk_sae__ln1_L10__k64__s42` from
  `aniketdesh/ward-stage-b-dictionaries` — the paper's own SAE arm, same
  width, same *k*, same layer.
- Orientation arms at a *different* hookpoint (`resid_L10`):
  `txc__resid_L10__k16__s42` and `txc_h13__resid_L10__k16__s42`. These are
  included only to show the paper's SAE-vs-TXC ordering reproduces inside
  this harness; they are not a matched comparison for the new dictionaries.
- Floor: the raw `ln1_L10` activations themselves, six window positions
  stacked into a 24,576-dimensional feature vector.

Every dictionary is scored under **two inference gates**: its native TopK, and
the post-hoc rate-calibrated threshold gate (the "gate swap"). On Gemma
concept-probing that swap left absorption and robustness untouched but bought
+3 to +5 points of small-*k* sparse-probing accuracy for every training
objective (see `docs/dmitry/proposals/2026-08-11_gate_swap_note.md`); the
question here is whether that readout gain transfers to a behavioural
detection task, where top-*S* feature selection *is* the readout.

### Protocol provenance

The instruction was to reuse the paper's own code rather than reinvent the
protocol. Concretely, what was reused and what was written fresh:

- **Labels — reused, untouched.** D+/D- comes from
  `results/ward_backtracking/sentence_labels.json`, the artifact produced by
  `experiments/ward_backtracking/label_sentences.py`: an Anthropic Haiku-4.5
  judge classifies every sentence of every trace as `backtracking` or
  `other`, per Ward 2025 section 2.2 / Appendix C. 300 traces, 25,528 judged
  sentences, 3,175 positive. No relabelling, no keyword rule of my own.
- **Window and anchor alignment — reused.** The `[-13, -8]` offset window is
  `mining.offset_window` from `experiments/ward_backtracking_txc/config.yaml`
  (identical to `WIN_OFF` in the sprint's `c7_detect.py`). The
  sentence-`char_start` to token-position mapping is copied from
  `mine_features.py::_capture_windows`, including its `<think>` offset
  handling and its `cs <= target_char < ce or cs >= target_char` scan.
- **Probe and metrics — copied from `c7_detect.py`.** `prauc`, `rocauc` and
  `probe_cv` (ell-1 logistic, `C=1`, `liblinear`, 5-fold `GroupKFold` grouped
  by trace, top-*S* features chosen per training fold by Welch t-statistic,
  features scaled by training-fold standard deviation). The only change is
  loop order: folds outer, *S* inner, so each fold's training matrix and
  t-statistic are built once instead of once per *S*. Same folds, same
  t-statistic, same selection, same fit — the numbers are unchanged, the run
  is about twice as fast.
- **Second negative definition — copied from `c7_detect.py`.** See below.
- **Stage B checkpoint loading — reused.** Architectures are rebuilt by
  `ward_backtracking_txc/architectures.py::build_arch` from the config stored
  inside each `.pt`, the same way `sae_deadlatent_audit/modal_app.py` does it,
  so no config file can drift out from under the checkpoint.
- **Gate swap — reused unmodified.** `calibrate` and `GateSAE` are imported
  from `topk_vs_topkdiff/posthoc_gate_evals.py`, including its sort-and-gather
  fix for the per-column quantile (`torch.quantile` only accepts a shared
  `q`). Latents dead under TopK keep `theta = +inf` and stay off, so this
  evaluates the trained dictionary rather than resurrecting it. Those helpers
  assume the `(b_dec, W_enc: (d, H), b_enc, k)` layout, so two thin adapters
  present the other dictionaries in it: the Stage B per-token SAE only needs
  `W_enc` transposed, and the TXC family works exactly because
  `einsum('btd,tds->bs', x, W_enc)` *is*
  `x.reshape(B, T*d) @ W_enc.reshape(T*d, H)` with no pre-bias — an identity,
  not an approximation. The window arms therefore calibrate at their own
  window-level budget (*k* = 16/position x 6 = 96).
- **Written fresh:** only the glue — the activation-capture pass over the 300
  traces at two hookpoints, the example-set assembly, the adapters just
  described, and the Modal harness (patterned on
  `diffusion_txc/topk_vs_topkdiff/modal_posthoc.py`).

The sprint's `bt_data/traces.jsonl` and `labels.jsonl` (the pre-tokenised form
`c7_detect.py` reads) were **not** found anywhere in the repo or on the Modal
volumes; they were built ad hoc on a pod and never committed. Rather than
reconstruct that intermediate by guesswork, the pipeline reads the committed
Stage A artifacts directly and re-derives token anchors with the paper's own
alignment code. This is the substantive methodological difference from the
sprint's addendum run.

### Two negative sets, reported separately

PR-AUC is only interpretable against the positive base rate, and the paper's
D- and the sprint's D- are not the same set. Both are reported:

- **`sentence`** — the paper's own D+/D- split, and the one
  `mine_features.py` mines selectivity on: every judged sentence is an
  example, positive iff `is_backtracking`. 25,528 examples, 3,175 positive,
  base rate 0.124.
- **`far`** — `c7_detect.py::collect_examples`: positives are the
  backtracking anchors; negatives are token positions more than 25 tokens
  from any backtracking anchor, sampled 5 per positive at rng seed 11.
  18,912 examples, 3,175 positive, base rate 0.168.

The `far` set is the one comparable in construction to the sprint addendum's
numbers; the `sentence` set is the harder and more faithful-to-the-paper task,
since its negatives are real sentence boundaries rather than arbitrary
mid-sentence positions.

### Gate calibration data

Thresholds are calibrated on the **tail 50,000 activations of the trace
cache** — the eval distribution itself, as instructed, and by construction the
last traces in the corpus, so the calibration slice is trace-disjoint as well
as position-disjoint from most of the evaluation windows. Calibration is
label-free: it only matches each latent's preactivation quantile to that
latent's native TopK firing rate on the same slice, and never sees D+/D-.
Fold leakage is therefore not a concern, but the choice is stated here
explicitly because the slice is not disjoint from every probe fold.

Rate-matching is verified rather than assumed — realized L0 under the gate is
reported per dictionary in the table below and lands within about 1 unit of
the native budget, with the per-token variance that the gate is supposed to
introduce.

### Hookpoint check

The new dictionaries were trained through a forward hook on
`model.layers[10].input_layernorm` (`psc_train_sae.py::make_hook`), while the
paper's Stage B cache uses a forward *pre*-hook on `layers[10].self_attn`
(`cache_activations.py`). In a Llama decoder layer the output of
`input_layernorm` is exactly the tensor handed to `self_attn`, so these should
be the same activations. The run asserts it rather than assuming it: both
hooks are registered simultaneously and `torch.equal` on the first trace
returned **True**. The new dictionaries and the Stage B SAE baseline are
therefore reading the identical tensor, and the ln1 comparison is internal.

### Results

PR-AUC, native TopK gate (`sentence` / `far`; base rates 0.126 / 0.166):

| arm | sent S=8 | sent S=32 | far S=8 | far S=32 | dead on traces |
| --- | --- | --- | --- | --- | --- |
| raw ln1 stacked (floor) | 0.190 | 0.216 | 0.259 | 0.313 | — |
| new recon s2 / s3 | 0.190 / 0.191 | 0.220 / 0.222 | 0.270 / 0.268 | 0.319 / 0.323 | 9.7% |
| new dsm s2 / s3 | 0.176 / 0.191 | 0.213 / 0.221 | 0.278 / 0.274 | 0.313 / 0.313 | **50.4 / 50.7%** |
| stage-B topk_sae (paper SAE arm) | 0.202 | 0.238 | 0.302 | 0.330 | 42.5% |
| stage-B txc (resid, orientation) | 0.215 | 0.239 | 0.298 | 0.331 | — |
| stage-B txc_h13 (resid, orientation) | 0.215 | **0.243** | **0.307** | **0.351** | — |

Gate swap (same dictionaries, threshold gate; `sentence` S=8, TopK →
gate): recon 0.1897 → 0.1890 / 0.1908 → 0.1916, dsm 0.1760 → 0.1751 /
0.1910 → 0.1814, stage-B SAE 0.2021 → 0.2028 — all within ±0.01 with
sign flips across seeds: a clean null for **per-token** dictionaries;
the Gemma probing gain does not transfer. The one cell outside ±0.01 is
the **window** dictionary: stage-B TXC-base 0.2149 → **0.2288**
(+0.0139), the largest move in the table and an improvement (TXC-h13:
−0.002). One arm, one seed — suggestive, not a finding — but it points
where to look if the gate swap is revisited: window dictionaries, not
per-token ones. Rate-matching verified throughout (gate L0 60–64 vs
native 64; window arms 87–90 vs 96).

Readings:

- **Per-token dictionaries add nothing over raw activations here**, for
  either objective — while the temporal arms lead the table. The
  program's central prediction (behavioural detection advantage lives in
  temporal structure) is *supported by the failure*: the discriminating
  experiment is TXC+DSM, which does not exist yet on this model.
- **No recon-vs-dsm separation** on a behavioural detection task at the
  per-token level, matching the concept-probing wash.
- **The gate-swap boundary**: readout gains from variable-L0 gating
  appear in mean-pooled multi-class probing but not in max-pooled
  binary detection with t-statistic feature selection. The dissociation
  note's claim should be narrowed accordingly.
- **Distribution-shift death is objective-dependent at this hookpoint**:
  dsm keeps only half its latents alive on distill traces (recon: 90%)
  despite near-zero death for both on FineWeb. Denoising against
  in-distribution corruption apparently specialises latents more tightly
  to the training distribution — an important consideration for the
  planned trace-domain/mixed-corpus training.

### Caveats

- **Read PR-AUC against the base rate, not against zero.** The positive base
  rate is 0.124 (`sentence`) and 0.168 (`far`), so those are the
  no-information floors. Several arms land within a few points of their
  floor. This is a property of the protocol, not of this run: the paper's own
  published SAE number (0.175) and the sprint addendum's reproduction (0.164)
  sit at essentially the same distance from a 1-in-6 base rate. Differences
  between arms are the signal here; absolute PR-AUC is close to meaningless
  without the floor beside it.
- **The `far` set has a sentence-boundary confound.** Its positives are all
  sentence-initial anchors while its negatives are arbitrary token positions,
  and only about 4% of token positions are sentence-initial (25,528 sentences
  over roughly 600k tokens). Any dictionary carrying a "start of sentence"
  feature gets credit on `far` that has nothing to do with backtracking. The
  `sentence` set does not have this confound — both classes are sentence
  anchors — which is why it is the primary column. A dedicated
  sentence-start control was not run and would be the obvious next check.
- **Train/eval distribution mismatch, deliberately preserved.** The traces are
  DeepSeek-R1-Distill-Llama-8B generations; the activations are base
  Llama-3.1-8B; the dictionaries were trained on base-Llama FineWeb. This is
  the paper's own setup and was replicated rather than corrected, but it does
  mean every dictionary here is being read out of distribution, and the
  dead-latent numbers below should be understood in that light.
- **The orientation arms are not a matched comparison.** `txc` and `txc_h13`
  read `resid_L10`, a different hookpoint, at *k* = 16/position and a
  different training corpus and budget. They are in the table to show the
  harness behaves sensibly, not to rank architectures.
- **Two seeds per arm, one hookpoint, one layer.** Seed spread is reported
  per arm; where two seeds of one objective straddle another objective's
  value, no ordering should be claimed.
- **Labels are LLM-judge labels.** D+ is whatever Haiku 4.5 called
  backtracking. That is the paper's definition and it is reused unmodified,
  but it is a noisy target, and 3,175 of 25,528 sentences is a fairly
  permissive positive class.
- **The sprint's `bt_data` intermediates could not be found**, so these
  numbers are not directly comparable to the sprint addendum's table even on
  the `far` set: anchors here are re-derived from `sentence_labels.json`
  rather than from that run's committed event positions.
- **The gate swap is rate-matched, not budget-matched per token.** Mean L0
  tracks the native *k*, but L0 varies token to token by design, so a gate
  arm can spend more latents on some windows than TopK ever would. That is
  the intended difference between the gates, not a confound, but it does mean
  the two columns are matched on average rather than pointwise.

### Reproducing

```bash
# 20-trace end-to-end check first (exercises every checkpoint and both gates)
uvx modal run experiments/backtracking_detection_dsm/modal_detect.py::smoke \
    --n-traces 20 --modes topk,gate
# native TopK table, then the gate-swap table
uvx modal run --detach experiments/backtracking_detection_dsm/modal_detect.py::detect
uvx modal run --detach experiments/backtracking_detection_dsm/modal_detect.py::detect_gate
uvx modal volume get diffusion-txc backtracking_eval/detection_results.json .
uvx modal volume get diffusion-txc backtracking_eval/detection_results_gate.json .
```

Code: `experiments/backtracking_detection_dsm/` (`detect_core.py`,
`run_detection.py`, `modal_detect.py`). Results JSON is committed to the
`diffusion-txc` Modal volume under `backtracking_eval/`.

## w6 trio: first LLM DSM-TXC detection (2026-08-11 late)

The window-6 trio trained tonight (Llama resid_L10, k=96, H=16384,
~16.5k steps FineWeb/pile stream; `txc_w6/` on the volume) evaluated under
the identical protocol (`detection_results_w6.json`; smoke-gated window
flatten order, time-major NMSE 0.86 vs dim-major 2.39). Sentence-set
PR-AUC, same-protocol references from `detection_results.json`:

| arm | train dist | sent S8 | sent S32 | far S8 | dead-on-traces |
| --- | --- | --- | --- | --- | --- |
| raw stacked | — | 0.190 | 0.216 | 0.259 | — |
| per-token recon (2s) | FineWeb | 0.190 | 0.221 | 0.269 | ~0.10 |
| stage-B TopK SAE k64 | traces | 0.202 | 0.238 | 0.302 | — |
| stage-B TXC / h13 k16 | traces | 0.215 | 0.239–0.243 | 0.298–0.307 | — |
| w6_recon | FineWeb | 0.196 | 0.215 | 0.275 | 0.076 |
| w6_dsm | FineWeb | 0.181 | 0.209 | 0.277 | **0.962** |
| w6_bayes | FineWeb | 0.194 | 0.216 | 0.281 | 0.424 |

Three findings:

- **Temporal architecture alone does not reproduce the stage-B edge.**
  The FineWeb-trained window dicts sit at the raw floor (0.18–0.20 S8),
  not at stage-B TXC's 0.215. The 2×2 across
  {per-token, temporal} × {FineWeb, trace-trained} now decomposes the
  stage-B advantage: temporal-on-FineWeb ≈ +0.006, trace-domain
  per-token ≈ +0.012, both together ≈ +0.025 — the temporal gain
  mostly manifests *on top of* domain match. Trace-domain (mixed-corpus)
  training is the decisive next run, now with a factorial argument
  rather than a hunch.
- **Objective does not separate at matched recipe** (recon 0.196 vs
  dsm 0.181 vs bayes 0.194 at S8; fold spreads ±0.03 overlap): no DSM
  detection win, and no bayes win, on off-domain-trained window dicts.
- **DSM near-parity from 3.7% capacity.** w6_dsm reaches within ~0.015
  of recon while 96.2% dead on trace windows — ~600 live latents match
  ~15k. Per-latent, the surviving DSM features are far more informative
  (consistent with low-band concentration), but the OOD collapse caps
  the arm; it cannot win while dead. Same story as steering: coverage
  of the deployment distribution is the binding constraint on DSM
  everywhere.

## Mixed-corpus pair: the objective×domain interaction (2026-08-12 morning)

Same recipe as the w6 trio but trained on the 72/28 trace/FineWeb mixed
corpus (8000 steps, A100-80GB; `txc_w6_mix/`, results
`detection_results_w6mix.json`). Same protocol, sentence-set PR-AUC with
the trio and stage-B rows for comparison:

| arm | corpus | sent S8 | sent S32 | far S32 | dead-on-traces |
| --- | --- | --- | --- | --- | --- |
| w6_recon | FineWeb | 0.196 | 0.215 | 0.326 | 0.076 |
| w6_dsm | FineWeb | 0.181 | 0.209 | 0.308 | 0.962 |
| w6mix_recon | mixed | 0.190 | 0.228 | 0.332 | 0.068 |
| **w6mix_dsm** | mixed | **0.208** | **0.242** | **0.352** | **0.948** |
| stage-B TXC / h13 | traces | 0.215 | 0.239–0.243 | 0.331–0.351 | — |

Three findings, one pre-registration verdict:

- **First DSM detection win, and it is an interaction, not a main
  effect.** Domain training barely moved recon (S8 0.196→0.190, S32
  0.215→0.228) but flipped dsm from worst to best-of-ours (S8
  0.181→0.208, S32 0.209→0.242, far S32 0.352), reaching stage-B TXC
  level on every column (folds overlap; "matches", not "beats"). The
  density-model theory predicts exactly this asymmetry: DSM learns the
  training density, so deployment-density match unlocks it; recon was
  never density-limited and gains little.
- **Pre-registered mechanism prediction FAILED while the outcome
  prediction succeeded**: we predicted domain match would revive the
  dead pool (96% → toward 8%). It did not — w6mix_dsm is still 94.8%
  dead on traces. The gain came from the ~850 live latents becoming the
  *right* latents, not from more latents. Extreme pool concentration
  appears intrinsic to DSM; what domain training changes is what the
  concentrated pool encodes.
- **Site dissociation in one morning**: the same w6mix_dsm still fails
  the steering-site gate (distill-model activations: NMSE 0.807, 5.2%
  live vs 0.795/3.7% FineWeb-trained). Text-domain match fixed the
  detection site (base-Llama activations over trace text — the trained
  distribution) and did nothing for the cross-model site. Model
  identity, not text domain, is the binding axis for the steering
  transfer failure; distill-captured training is the implied next cell.

Caveats: 1 seed per cell; S8 fold spreads (±0.03) overlap between
w6mix_dsm and both w6mix_recon and stage-B — the direction is consistent
across all four columns but no single comparison is individually
significant; the mixed corpus overlaps the eval traces (label-free, same
status as stage-B training); w6mix ordering-gate NMSE on trace windows
halved vs FineWeb-trained (0.448 vs 0.861), confirming on-domain fit
improved even though the live pool did not widen.

## Instrument-sensitivity controls (2026-08-12, arc-review experiment 1)

Three controls on the w6mix pair decide whether the detection table can
distinguish dictionary quality at all, and whether "DSM's survivors are
disproportionately informative" is licensed
(`backtracking_eval/detection_controls.json`; same capture, example sets,
probe and folds — only the dictionary/feature source varies):

| arm | latents | sent S8 | sent S32 | far S32 |
| --- | --- | --- | --- | --- |
| w6mix_dsm | 849 | 0.203 | 0.237 | 0.346 |
| w6mix_recon (full) | 15,274 | 0.201 | 0.226 | 0.321 |
| recon subsampled to 849 (3 draws) | 849 | 0.123–0.146 | 0.158–0.174 | 0.216–0.252 |
| random (untrained) dictionary | 16,382 | 0.128 | 0.160 | 0.221 |
| label shuffle (chance) | — | 0.142–0.146 | ≈0.156 | ≈0.167 |

- **The per-latent-informativeness claim is licensed.** Matched-capacity
  recon collapses to near the untrained-dictionary floor; dsm's 849
  survivors carry ≈ the total signal of recon's full 15,274-latent pool
  (18× the per-latent informativeness of a random recon subset). The
  near-equality dsm@849 ≈ recon@15274 is itself suggestive: both
  dictionaries appear to capture similar total domain signal, with DSM
  allocating it into a minimal template set.
- **The instrument has dynamic range** (random 0.128 vs trained 0.20+),
  so the compression among top arms is signal, not floor artifact — but
  differences of ~0.01–0.02 between top arms remain within fold noise
  (within this run, dsm-vs-recon at sentence S8 is a wash; the far-set
  and capacity-normalized comparisons are the robust ones).
- Reference values shift slightly between runs (dsm 0.208→0.203, recon
  0.190→0.201 at S8) from probe-refit variance — within-run comparisons
  share all pipeline randomness and are the ones quoted above.

## Survivor-interpretation battery (2026-08-13) — SUPERSEDES the controls reading

Full battery on the w6mix pair (`backtracking_eval/survivor_interp.json`;
single-latent probes, capacity curves, pairwise correlations,
mass-matched selected controls, judged autointerp; pre-registered
readings embedded in the artifact):

- **The 18× per-latent claim dissolves under selected controls.**
  Yesterday's subsampled-recon control used RANDOM 849-latent subsets
  (0.12–0.15) — the wrong null. Recon's top-248-by-mass slice scores
  **0.223 > dsm's full 849 survivors (0.208)**; count-matched
  top-849-by-rate recon 0.214. Both dictionaries have a concentrated,
  label-free-discoverable informative core; DSM training deletes the
  tail rather than creating concentration. Surviving claim: equal total
  signal in an 18×-smaller pool (deployment/audit virtue), not
  information creation.
- **Not an interaction code**: smooth capacity curve (0.133→0.208, no
  cliff), bulk pairwise |corr| median 0.007. **Not super-features
  either**: best single survivor 0.179 ≈ recon's best 0.184; signal is
  additive over ~256+ weak units in both dictionaries. Pathology found:
  ~1% of survivor pairs are near-duplicate atoms (|corr| ≈ 1.0) —
  effective distinct-template count < 849.
- **Autointerp (judged, Haiku)**: dsm survivors 0.77 ± 0.05 > recon
  top-usage 0.64 ± 0.04 (rate-matched comparison — DSM's high-rate core
  is cleaner), but recon random-live 0.84 ± 0.03 is highest —
  interpretability tracks rarity, and DSM deleted the rare monosemantic
  tail.
- **Synthesis**: DSM ≈ label-free pruning of a shared core. Value =
  compactness and a cleaner high-rate unit set; cost = the crisp rare
  tail; information beyond recon = none demonstrated on this task.
