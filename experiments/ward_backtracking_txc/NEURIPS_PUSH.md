# NeurIPS 2026 — Backtracking Case Study Final Push

Owner: Aniket
Branch: `aniket-ward-stage-b` (do NOT touch any non-`aniket-*` branches; read-only fine)
Last meeting: 2026-05-02
Experiment freeze: Sunday 2026-05-04 EOD
Writing pivot: Monday 2026-05-05
Abstract due: Monday 2026-05-05
Paper due: Thursday 2026-05-08 ~05:00 local

## STATUS — 2026-05-02 evening

**Section 2 (headline steering plot) — code complete, retrain in flight, sweep pending.**

Shipped:
- `config.yaml`: 25-magnitude grid + `tsae.kval_topk=20`.
- `architectures.py`: `tsae` arch now respects `kwargs["kval_topk"]` (was hardcoded to `k*T`).
- `b3_variants.py`: `--include-correct N` + `--out` flags. `before_correct` now in rescue rows. Regression-rate-by-magnitude added to summary.json.
- `experiments/ward_backtracking_txc/build_flip_matrix.py`: 2×2 confusion + McNemar test per arch at per-arch best magnitude.
- `experiments/ward_backtracking_txc/calibrate_magnitudes.py`: 95th-percentile feature-activation calibration per (arch, feature).
- `experiments/ward_backtracking_txc/run_b3_multi_arch.py`: per-arch sweep dispatcher writing per-cell subdirs + meta.json.
- `experiments/ward_backtracking_txc/plot/headline_steering.py`: 3-panel headline (net rescues, rescue rate, regression rate), calibrated + raw versions.
- `experiments/ward_backtracking_txc/run_headline_pipeline.sh`: A→E orchestrator. Sweep is GPU-parallelized (2 arches per GPU).
- `notes/tsae_paper_param_audit.md`: Bhalla 2026 vs codebase architecture mismatch documented.

In flight (background):
- TSAE k=20 retrain on `resid_L10` (GPU 0) and `ln1_L10` (GPU 1). Logs at `logs/tsae_k20_*.log` + JSONL at `results/.../logs/tsae__<hp>__k32__s42__train.jsonl`. ETA: full 15k steps in ~25min total. Both well within the 2× TXC FVU acceptance gate.

To do — sequential:
1. Wait for both TSAE retrains to finish (or hit early-stop).
2. Re-mine TSAE features: `uv run python -m experiments.ward_backtracking_txc.mine_features --cell tsae__resid_L10__k32__s42` (and same for ln1_L10 if desired).
3. Run the full pipeline: `bash experiments/ward_backtracking_txc/run_headline_pipeline.sh` (this also re-mines TSAE in step A, idempotent if already done).

Headline arch picks (locked in `run_b3_multi_arch.py:HEADLINE_ARCHES`):
- TXC:        `txc__resid_L10__k16__s42`     f14621 pos0 (already canonical).
- TXC-H8:     `txc_h8__resid_L10__k16__s42`  f344   pos0 (top by B1 keyword rate).
- SAE:        `topk_sae__ln1_L10__k64__s42`  f5263  pos0 (top per `rank_global_sonnet`).
- TSAE-paper: `tsae__resid_L10__k32__s42`    f???   pos0 (resolved post-retrain).

**Skipped from headline:** TXC-H13, stacked_sae, tsae_paper (Bhalla ReLU+L1), TFA. All move to appendix.

---

---

## 0. Paper narrative (anchor for every decision below)

The paper now claims three things. Every experiment in this push must support one of them; if it doesn't, it goes to the appendix.

1. **Detection**: TXC is SOTA for feature *detection* of multi-token / multi-sentence behaviors (backtracking is a textbook case).
2. **Steering**: TXC ties other temporal architectures and conventional SAEs for *causal intervention*, but with broader steering robustness across magnitudes (no narrow spike). Steering protocols for TXCs remain an open problem.
3. **Hybrid protocol**: TXC for detection → SAE for steering, since steering is more reliable on conventional SAEs.

For the backtracking case study specifically, claims (1) and (2) are the deliverables. (3) is Andre's case study to drive home.

---

## 1. Standardized architecture config

For every plot, table, and metric in the backtracking case study, use exactly this set:

| Label | Architecture (registry name) | k | T | Activation / Loss | Notes |
|---|---|---|---|---|---|
| **TXC-T5** | `txc` | 32/pos (window L0=160) | 5 | TopK | Headline TXC. Simple, defensible default. |
| **TXC-H8** | `txc_h8` | 32/pos | 5 | TopK + InfoNCE multi-distance + Matryoshka (alpha=1.0, h_size=0.2·d_sae) | Sparse-probing benchmark winner. |
| **SAE** | `topk_sae` | k=32·5=160 (per-token) | 1 | TopK | Conventional baseline at the same compute / sparsity budget. |
| **TSAE-paper** | `tsae` (Han's TemporalSAE @ TopK) | `kval_topk=20` | 5 | TopK + attention (predicted+novel) | See §1.1 for the architectural mismatch with Bhalla 2026 — we ship the closest practical analog and document the deviation. |

### 1.1 TSAE paper params — confirmed from the source paper, plus a codebase mismatch

Dmitry quoted "K=20, contrastive_loss=0.1" in the meeting. The user pointed at the source paper (Bhalla et al., ICLR 2026, "Temporal Sparse Autoencoders", `https://openreview.net/pdf?id=bojVI4l9Kn`).

Direct quotes from the paper, page 5 (Hyperparameters):

> "All SAEs are trained with the **BatchTopK activation (k=20), 16k features**, and the auxiliary loss from (Bussmann et al., 2025; Gao et al., 2024)."
> "Temporal and Matryoshka SAEs are trained with **20%-80% feature splits**, where for Temporal SAEs the 20% are the high-level features."
> "We use **a regularization parameter of 1.0 on the temporal loss** for all Temporal SAEs."

So the paper-faithful spec is:

- **k = 20** (BatchTopK, per-token sparsity, with 16k features). ✅ Dmitry's k=20 is correct.
- **Temporal contrastive coefficient = 1.0**, NOT 0.1. ❌ Dmitry mis-remembered.
- **20% high-level / 80% low-level feature split**; contrastive loss applied only to the 20% high-level features, between adjacent tokens t and t−1.
- 16k total features.
- BatchTopK auxiliary loss (Bussmann 2025 / Gao 2024 style) for dead-feature revival.

**Codebase mismatch (must be acknowledged in the paper):**

This repo's `tsae_paper` registry entry is mis-labeled. Both `tsae` and `tsae_paper` in `experiments/ward_backtracking_txc/architectures.py` use the same Han `TemporalSAE` class from `temporal_crosscoders/han_tsae/`, which is an **attention-based predicted+novel-code architecture** — fundamentally different from Bhalla 2026, which uses a **standard SAE with a 20/80 high/low split and a contrastive loss between adjacent tokens**. `tsae_paper` only swaps the activation to ReLU+L1; it does NOT implement Bhalla's split or adjacent-token contrastive loss.

**Decision** (recommended, sized to fit pre-Sunday-EOD freeze):

Option A (pragmatic, ~4h): Use `tsae` (Han's TopK TemporalSAE) with `kval_topk=20`, treat as "Han 2025 attention TSAE". Re-label clearly throughout: this is *not* the Bhalla paper. Document the architectural difference in the paper's "Baselines" subsection. The Bhalla TSAE then only appears via `tsae_paper` as-currently-implemented (ReLU+L1 attention TSAE) in the appendix with the same caveat.

Option B (faithful, ~1–1.5 days): Implement a true Bhalla TSAE (standard SAE class + high/low split mask + adjacent-token contrastive). This is the right thing technically but eats most of Sunday. **Do not pick B unless explicitly approved by Dmitry on Slack.**

Default to A. Persist the decision and the audit in `notes/tsae_paper_param_audit.md`, including:

- Direct quotes from Bhalla 2026 page 5.
- The architectural mismatch between our `tsae_paper` and Bhalla.
- Which arch we're labeling "TSAE-paper" in figures (Han attention TSAE w/ k=20) and why.
- Note that we also retain `tsae_paper` as-currently-implemented in the appendix.

**One-line config change for Option A:** in `config.yaml`'s `tsae` block, set `kval_topk: 20` explicitly (currently inferred from `k_per_position * T = 32 * 5 = 160`). Other params (n_heads=8, bottleneck_factor=64, sae_diff_type='topk') stay as-is.

### 1.2 What's dropped

- **TXC-H13**: out of headline figures. Move to appendix per "standardize on simple variants" agreement.
- **TFA**: this codebase has **no TFA implementation**. Han's TFA lives on his branch which we don't touch. Decision: drop TFA from the backtracking headline. Document in the case-study limitations section that "we do not include TFA in the backtracking comparison; see the RLHF case study for TFA results." Andre and Han own the TFA-inclusive comparisons in their case studies.
- **`tsae_paper` (Bhalla)** and **`stacked_sae`**: keep checkpoints, leave existing data points, but drop from headline figure to keep the legend at 4 lines. Move to appendix.

---

## 2. Priority 1 — Headline steering plot

The current canonical Stage B steering result lives at `results/ward_backtracking_txc/b3_math500_cut25/`. Cut25 is the winning protocol (cut at 25% of unsteered trace, then steer-and-continue) — keep this fixed for the rest of the push.

Four changes before the plot is paper-ready.

### 2.1 Re-run TSAE at paper params (Option A from §1.1)

- Per §1.1, change `config.yaml:tsae` block to set `kval_topk: 20`. Leave `n_heads=8`, `bottleneck_factor=64`, `sae_diff_type='topk'`, `tied_weights=true` as-is.
- Document the architectural caveat (we're using Han's attention TSAE as the closest analog of Bhalla 2026 with the agreed paper k=20) in `notes/tsae_paper_param_audit.md`.
- Retrain on all three hookpoints (`resid_L10`, `attn_L10`, `ln1_L10`) for at least as many steps as the current strongest TXC (check `results/.../logs/txc_resid_L10_train.jsonl` for step count and loss convergence).
- **Acceptance gate**: the new TSAE's reconstruction loss (FVU on the eval batch in the JSONL log) must be within 2× of TXC-T5's FVU. If not, the TSAE is undertrained or misconfigured — keep training or escalate.
- After training, re-run `b3_math500_rescue.py` at the same magnitude grid as the other arches (see §2.2).
- DO NOT spend time porting the Bhalla 2026 contrastive term unless Dmitry approves it explicitly — that's Option B in §1.1 and likely eats Sunday.

### 2.2 Higher-resolution magnitude sweep

Current magnitude grid (`config.yaml:steering_magnitudes` line ~125): `[-16, -12, -8, -4, 0, 4, 8, 12, 16]`. Too coarse to characterize the SAE peak.

**New grid for all four headline arches (TXC-T5, TXC-H8, SAE, TSAE-paper):**

```yaml
steering_magnitudes:
  - -16
  - -12
  - -10
  - -8
  - -7
  - -6
  - -5
  - -4
  - -3
  - -2
  - -1
  - -0.5
  - 0
  - 0.5
  - 1
  - 2
  - 3
  - 4
  - 5
  - 6
  - 7
  - 8
  - 10
  - 12
  - 16
```

(25 magnitudes; densified ±0.5 to ±8 where the SAE peak lives. Re-densify further if the actual peak lands outside this grid after first run.)

- B1's parallel-across-magnitudes harness (per commit `95a27989`) makes this ~9× faster than serial. Use it.
- Run `b3_math500_rescue.py` (or `b3_variants.py` with `cut_fraction=0.25`) for each (arch × magnitude × question). Output: `results/ward_backtracking_txc/b3_math500_cut25/phase2_rescue_v2.json` (or extend existing).
- For each arch, the steering feature is the same one already mined — don't re-mine. Use `rank_phase1.json` / `rank_global_sonnet.json` to look up the canonical feature per (arch, hookpoint).

### 2.3 Add the flip matrix

For each (arch × magnitude), compute the 2×2 confusion of correctness from unsteered → steered:

```
              after-steering
              correct   incorrect
before correct  n_cc     n_ci
       incorr.  n_ic     n_ii
```

The data is already implicit in `phase2_rescue.json` — each row has `unsteered_correct` (from `phase1_unsteered.json`) and `steered_correct` (from rescue output). Just compute and persist as `flip_matrix.parquet` with columns `(arch, hookpoint, magnitude, question_id, before, after)`.

Plot two panels:

- **Net rescues**: `n_ic - n_ci` vs magnitude per arch. This is the judge-free supplement to coherence; a positive value means the architecture is genuinely correcting wrong reasoning.
- **Regressions**: `n_ci` (correct → incorrect) vs magnitude per arch. Cost of steering. Useful for showing TXC has lower regression rate at the magnitudes where it has comparable rescue rate.

**Statistical test**: McNemar's test on the discordant cells (`n_ic` vs `n_ci`) per arch at the per-arch best magnitude. Report `chi^2`, `p`, `n` in the plot caption / supplementary table.

### 2.4 Steering coefficient calibration

Cross-arch magnitude comparison is unitless until normalized. Use the activation-percentile calibration Andre suggested and Dmitry agreed to:

For each arch, on the eval-set forward passes (use the same activations cached in `results/ward_backtracking_txc/activations/`):

1. For the steered feature `f`, collect `f`'s activation values over all (token × example) positions where the feature fires (>0).
2. Compute the 95th percentile of those values: `s_arch = quantile(f_acts, 0.95)`.
3. Define "calibrated magnitude 1.0" per arch as `1 × s_arch`.
4. Re-plot the headline with x-axis = `magnitude_raw / s_arch` per arch.

Persist `calibration.json` with `{arch: {feature_id: s_arch}}` so plots can be regenerated without recomputing.

**Both versions go in the paper:** calibrated as headline (main text), raw as appendix figure.

### 2.5 Plot architecture cleanup

- Headline figure has 4 lines: TXC-T5, TXC-H8, SAE, TSAE-paper.
- TXC-H13, `stacked_sae`, `tsae_paper` (Bhalla) → appendix figure with all 7 lines.
- Use the standard project palette (TXC blues, SAE green/orange, TSAE pink). Match the palette already used in `experiments/ward_backtracking_txc/plot/steering_comparison_bars.py` if defined; otherwise pick once and stick to it.
- Subplots: (a) coherence vs calibrated magnitude, (b) net rescues vs calibrated magnitude, (c) keyword rate vs calibrated magnitude (existing). All with shared x-axis grid.

### 2.6 Acceptance criteria for §2

- [ ] All four headline arches run on the new 25-magnitude grid at cut25.
- [ ] TSAE retrained at paper params; reconstruction loss within 2× of TXC-T5; verification note in `notes/tsae_paper_param_audit.md`.
- [ ] `flip_matrix.parquet` written per `(arch, hookpoint, magnitude, question_id)`.
- [ ] McNemar test results in `results/ward_backtracking_txc/b3_math500_cut25/mcnemar_table.csv`.
- [ ] `calibration.json` written; both calibrated and raw headline plots saved.
- [ ] Coherence, keyword rate, productive-generation, net-rescue plots all use identical x-axis grids.
- [ ] Old TXC-H13 line removed from headline; appendix version preserved.

---

## 3. Priority 2 — Backtracking detection (THE most important new contribution)

This is the half of the narrative the meeting flagged as missing. Without it the case study supports only claim (2). With it, the case study supports both (1) and (2) and is the strongest in the paper.

The codebase has SAEBench probing code at `src/bench/saebench/probe_fit.py` and `src/bench/saebench/probing_runner.py` — reuse the probe-fitting routine but build the sentence-level dataset and orchestration ourselves.

### 3.1 Sentence-level labels

Source: existing reasoning traces in `results/ward_backtracking/traces.json` (Stage A output) and the steering-eval transcripts.

Pipeline (new script: `experiments/ward_backtracking_txc/detection/build_sentence_labels.py`):

1. Load all reasoning traces (unsteered, since we want the natural distribution of backtracking).
2. Sentence-tokenize each trace (use `nltk.sent_tokenize` or simpler regex on `\.|!|\?` boundaries; backtracking traces have idiosyncratic punctuation so keep the splitter forgiving).
3. For each sentence, label `is_backtracking ∈ {0, 1}` using the existing Sonnet judge in `grade_backtracking.py`. The judge already has a clean "GENUINE backtracking" rubric — wrap it for sentence-level inputs.
4. Persist as `results/ward_backtracking_txc/detection/sentence_labels.parquet` with columns `(question_id, trace_id, sentence_idx, sentence_text, is_backtracking, judge_raw)`.

**Target**: ≥1000 labeled sentences total. Aim for class balance roughly 30/70 backtracking/non — if it's worse than 10/90, oversample from traces known to backtrack heavily.

### 3.2 Sparse probing setup

For each architecture (TXC-T5, TXC-H8, SAE, TSAE-paper, plus **raw-residual** baseline — see §3.4):

1. Extract feature activations for each labeled sentence. Aggregation modes per §3.3.
2. **Feature selection**: rank features by mean-difference between `is_backtracking=1` and `=0` classes (or by univariate AUC). Take top-`|S|`.
3. **Probe**: fit a sparse linear `LogisticRegression(penalty='l1', C=1.0, ...)` or a plain `LogisticRegression` on those `|S|` features.
4. Sweep `|S| ∈ {1, 2, 4, 8, 16, 32}`.
5. Report **AUC** (primary) and **F1 at threshold=0.5** (secondary) at each `|S|`.
6. **5-fold CV across `question_id`** (NOT across sentences — sentences from the same question are too correlated; that's leakage). Stratify CV by question-level backtracking presence if class imbalance is severe.

### 3.3 Fair-comparison protocol — non-temporal architectures

This is Dmitry's caveat from ~38:00 in the transcript. TXC features inherently aggregate across the T-token window; non-temporal arches see one token at a time. Without aggregating their features fairly, we'd be sandbagging the baselines and the comparison would be invalid.

For SAE, TSAE-paper, raw-resid (any arch with T=1 or that operates per-token), run the probe on the following aggregations of features within the same T=5 window TXC sees:

- `last`: feature value at the rightmost token (closest analog to standard SAE probing).
- `mean`: mean across the T tokens.
- `max`: max across the T tokens.
- `full_window`: concatenate all T feature vectors. Dimension blows up (`T × d_sae`), feasible only for small `|S|` after selection.

Report all four for non-temporal arches. **The headline number for each non-temporal arch is `max(across_aggregations)`** — we must claim TXC wins against the strongest fair baseline, not against a deliberately weakened one.

For TXC-T5 / TXC-H8: features are already per-window, no aggregation needed. Report a single "native" curve per TXC variant.

### 3.4 Linear probe baseline on raw residuals

Train a linear probe directly on the residual stream activations at the same hookpoint (no sparse dictionary involved), with the same train/test splits and aggregations as §3.3.

This baseline must be in the paper. **If TXC doesn't beat raw-residual probing, we don't have a detection claim** — and we should know that before pivoting to writing.

### 3.5 Statistical test

Paired Wilcoxon signed-rank across 5 CV folds, TXC-T5 vs each baseline, AUC at `|S|=8` (or whichever `|S|` is used in the headline). Report W statistic, p-value, n. Apply Holm-Bonferroni correction across the 4 comparisons (TXC-T5 vs SAE, TSAE, raw-resid, TXC-H8 — or skip the TXC-H8 comparison since they're both ours).

### 3.6 Acceptance criteria for §3

- [ ] `sentence_labels.parquet` exists with ≥1000 sentences and the class balance noted.
- [ ] AUC and F1 reported for `{TXC-T5, TXC-H8, SAE, TSAE-paper, raw-resid} × {last, mean, max, full_window} × |S| ∈ {1,2,4,8,16,32}` and persisted as `probe_results.parquet`.
- [ ] Paired Wilcoxon (with HB correction) reported for TXC-T5 vs each baseline at headline `|S|`.
- [ ] Single headline figure: AUC vs `|S|` for the 5 architectures (using the strongest aggregation per non-temporal arch). Saved to `results/ward_backtracking_txc/detection/detection_headline.png`.
- [ ] **Decision gate**: if TXC-T5 does not beat raw-resid by a material margin (`ΔAUC > 0.02` at any `|S|`) with `p<0.05`, escalate to Dmitry on Slack before writing the case study.

---

## 4. Priority 3 — LLM judge validation

Dmitry was emphatic: every metric in §2 (coherence) and §3.1 (sentence labels) is suspect until we sanity-check the judge against blind human scoring on a small sample.

### 4.1 Blind agreement test (mandatory)

1. Sample 20 transcripts uniformly at random from the cut25 eval set, stratified roughly evenly across the steering magnitudes that hit the SAE peak vs off-peak.
2. **Aniket scores them himself first**, blind to the LLM judge: coherence (0–3 scale), backtracking-present (0/1), looping-present (0/1). Do this in a CSV without ever loading the judge output. (CC agent: prep the CSV, leave the human-score columns blank, hand to user.)
3. Then load LLM judge scores for the same 20.
4. Compute Cohen's `κ` (per task) and raw agreement rate.
5. **Targets**: ≥80% raw agreement and `κ ≥ 0.6`. Below either, refine the judge prompt (one iteration, max), re-run, re-test. If still below, note as a paper limitation and back off any specific claim that depends on a single judge-driven point.
6. The 20 transcripts must be a held-out split not used to develop the judge prompt.

Persist as `results/ward_backtracking_txc/judge_validation/blind_pairs.csv` and `kappa_report.txt`.

### 4.2 SAE peak forensics

Independent of §4.1. Pull ~20 transcripts at the SAE's best magnitude (after §2.2 densification reveals its actual location) and ~20 at slightly off-peak (e.g., ±0.5 from peak). Aniket manually inspects.

Hypotheses to test in the inspection:

- A specific sentence template the judge happens to like.
- Looping that the patched judge still misses (residual exploit).
- Genuine high-quality backtracking that's narrowly tuned (the paper-favorable interpretation).

Document findings in `notes/sae_peak_forensics.md`. This goes in the case-study discussion section even if the conclusion is uninteresting — the meeting transcript indicates Dmitry will want to cite it.

### 4.3 Loop / repetition rate as objective auxiliary metric

Compute per generation:

- Sentence n-gram overlap: max Jaccard token similarity among consecutive sentence pairs.
- Fraction of sentences that are near-duplicates (`token_jaccard ≥ 0.7` OR `cosine(sentence_encoder) ≥ 0.9` if a sentence encoder is already in the env; if not, just Jaccard).

Plot **repetition rate vs (calibrated) magnitude** for all 4 headline arches.

**Hypothesis under test**: SAE's narrow peak is bordered by sharp transitions to looping; TXC degrades more gracefully. If true, this is the mechanism for the broader-robustness claim and is worth a sentence or two in the case-study writeup.

### 4.4 Acceptance criteria for §4

- [ ] Blind `κ` and raw agreement reported in `kappa_report.txt`; if below threshold, prompt-iterate once and re-test.
- [ ] SAE peak forensic notes committed to `notes/sae_peak_forensics.md`.
- [ ] Repetition-rate-vs-magnitude plot for all 4 headline arches saved alongside the headline figure.

---

## 5. Priority 4 — NeurIPS hygiene (non-negotiable)

For every architecture shown anywhere in the paper, log and surface:

- [ ] Reconstruction loss / FVU (final + full training curve).
- [ ] L0 (mean active features per token / per window — note which) (final + curve).
- [ ] Fraction variance explained on a held-out activation batch (final).
- [ ] Training hyperparameters (yaml dumped to artifact dir).
- [ ] Number of training tokens / steps (explicit in the table).
- [ ] Wall-clock GPU-hours (for the appendix compute statement).

Most of this is already in `results/ward_backtracking_txc/logs/<arch>_<hookpoint>_train.jsonl`. Build one supplementary table from those logs:

| Architecture | k | T | Final FVU | Final L0 | FVE | Steps | Tokens | GPU-h |

Persist as `results/ward_backtracking_txc/hygiene/reconstruction_table.csv`. Render the training curves (per arch, FVU and L0 vs steps) as PNGs in `results/ward_backtracking_txc/hygiene/training_curves/`.

If TSAE is undertrained per §2.1's gate, retrain it for at least as many steps as the strongest TXC and re-fill the row.

**Reviewers will reject without this. It's not optional.**

---

## 6. Priority 5 — Statistical reporting standard

For every cross-architecture comparison the paper makes:

- **Steering** (per question, per magnitude): paired Wilcoxon signed-rank, TXC vs each baseline, on the per-question coherence and per-question correctness deltas.
- **Flip matrix**: McNemar's test on the discordant cells (`n_ic` vs `n_ci`).
- **Detection**: paired Wilcoxon across 5 CV folds on AUC, TXC vs each baseline.
- Always report: test statistic, p-value, n.
- **Multiple-comparisons correction**: Holm-Bonferroni across architectures within a single test family. (Bonferroni is acceptable but conservative for 4 arches.)

No bar charts without an associated test in the caption.

---

## 7. Time-boxed schedule

| Day | Focus | Deliverable |
|---|---|---|
| Sat May 02 (today) | TSAE param audit + retrain start; magnitude grid finalize and dispatch sweep | TSAE training underway; sweep grid running for the other 3 arches |
| Sun May 03 | §2 finalize (flip matrix + calibration + plots); start §3 detection probes; §4.1 blind test prep | Headline steering plot v1; probe pipeline running; blind CSV ready for Aniket |
| Mon May 05 | §5 hygiene table; abstract; finalize §3 detection plot; case-study writeup begins | Supplementary table; abstract submitted; detection plot v1 |
| Tue–Wed May 06–07 | Writing only. Iteration on plots based on narrative needs. | Paper case-study section drafted |
| Thu May 08 ~05:00 | Submit. | NeurIPS submission |

**Sunday EOD is the experiment freeze.** Anything not done by then goes to the appendix or doesn't ship.

**Priority order if anything has to be cut**: §2 (steering headline) > §3 (detection) > §5 (hygiene) > §4 (judge validation) > §6 (stats). Detection is the most important *new* contribution; if §3 doesn't ship, the case study reverts to a steering-only result and the paper relies on Andre's case study to make the detection point.

---

## 8. Open questions / decisions to surface to Dmitry

If any of these block work for >2h, post in Slack and proceed with the safer default rather than waiting.

1. **TSAE arch / faithfulness** (§1.1): the Bhalla 2026 paper params are `BatchTopK k=20`, contrastive coef = 1.0 (NOT 0.1 — Dmitry mis-remembered), with a 20/80 high/low feature split and adjacent-token contrastive loss applied to the 20%. Our codebase's `tsae_paper` does NOT implement this; both `tsae` and `tsae_paper` use Han's attention-based TemporalSAE. Default plan: ship `tsae` w/ `kval_topk=20` as "TSAE-paper" with a documented architectural caveat (Option A). Confirm Dmitry's OK on Slack before accepting Option B (port the Bhalla architecture, ~1–1.5 days).
2. **Calibration choice** (§2.4): 95th-percentile of feature activations is what Andre proposed and Dmitry agreed. Default: use it. Alternative: L2 of decoder direction. Stick with 95th unless Dmitry says otherwise.
3. **Multi-comparison correction**: Holm-Bonferroni assumed. Confirm.
4. **Detection panel scope**: do we want the backtracking detection figure in the same panel as Andre's harmful-prompt detection, or separate per-case-study? Default: separate; the joint panel is Dmitry's call across case studies.
5. **TFA in the backtracking case study**: confirmed dropped per §1.2. If Dmitry pushes back, point at "no TFA implementation on this branch and Han owns it."

---

## 9. File / artifact layout

```
results/ward_backtracking_txc/
  b3_math500_cut25/
    phase2_rescue_v2.json              # densified mag sweep, all 4 arches (existing schema)
    flip_matrix.parquet                # (arch, hp, mag, qid, before, after)
    calibration.json                   # {arch: {feature_id: 95th_pctile}}
    headline_calibrated.png            # main paper figure
    headline_raw.png                   # appendix
    repetition_rate.png                # judge-free auxiliary
    mcnemar_table.csv
    wilcoxon_steering_table.csv
  detection/
    sentence_labels.parquet            # (qid, trace_id, sent_idx, text, is_backtracking)
    probe_results.parquet              # (arch, agg, |S|, fold, auc, f1)
    detection_headline.png             # main paper figure
    wilcoxon_detection_table.csv
  judge_validation/
    blind_pairs.csv                    # human scores blind, then judge merged
    kappa_report.txt
    judge_prompt_v_final.md
  hygiene/
    reconstruction_table.csv
    training_curves/                   # per-arch png
    hyperparams/                       # per-arch yaml dump
notes/
  tsae_paper_param_audit.md
  sae_peak_forensics.md
experiments/ward_backtracking_txc/
  detection/
    build_sentence_labels.py           # NEW
    fit_probes.py                      # NEW (wraps src/bench/saebench/probe_fit.py)
    extract_features.py                # NEW (per-arch feature extraction with aggregations)
```

---

## 10. Out of scope (do not work on these)

- New TXC architecture variants beyond H8 and T=5 vanilla. Hill-climbing is appendix-only and Bill / Han's territory.
- T-sweep beyond T=5 for the headline. T=10 / T=20 results are appendix material if they already exist; do not generate new ones.
- Andre's harmful-prompt detection (his ownership).
- Han's RLHF case study (his ownership).
- Any model other than DeepSeek-R1-Distill-Llama-8B for this case study.
- Mess3 / synthetic results (Dmitry's section).
- Any branch that is not `aniket-*`. Read-only fine; no edits, no commits, no pushes.

---

## 11. Hand-off notes for the CC agent

- Re-cache feature activations only when necessary; the existing `activations/` cache is large (~7 GB per hookpoint) and has been validated.
- The B1 magnitude-parallelism harness (`b1_steer_eval.py` post-commit `95a27989`) is the correct execution model — do NOT regress to serial.
- All Sonnet judge calls cost real money (~$0.002/row); the existing grader is resumable. When extending the magnitude grid, re-grade only the new (magnitude, question) pairs, not the entire sweep.
- When persisting parquet, use pyarrow with snappy compression for consistency with the existing `results/` artifacts.
- Do not start a writeup file in `docs/` — that will pull in tag/frontmatter requirements and slow things down. Working notes go in `notes/` (plain markdown, no frontmatter).
- After §2 and §3 complete, post a Slack-ready 5-line summary to `results/ward_backtracking_txc/STATUS.md` so Aniket can copy-paste into the team channel.
