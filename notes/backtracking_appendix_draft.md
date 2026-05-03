# Backtracking case-study — main vs. appendix manifest + appendix draft

Owner: Aniket
Status: in-progress, picks up qualitative material as the pipeline produces it.
Audience: this is rough. Polish later. The point is to not lose context to the sweep.

---

## A. Figure / table manifest

### A.1 Main text (Section 4 / Reasoning case study)

| Slot | Artifact | Source path | Status |
|---|---|---|---|
| Fig 4a | Headline 3-panel: net rescues / rescue rate / regression rate vs *calibrated* magnitude. 5 lines: TXC, SAE, TSAE-paper, TFA, MLC. | `results/ward_backtracking_txc/b3_math500_cut25/headline_calibrated.png` | ✅ shipped |
| Fig 4b | Detection AUC vs |S| (and F1) for the same 5 archs. 5-fold grouped CV (group=question). | `results/ward_backtracking_txc/detection/detection_headline.png` | ✅ shipped |
| Tab 4a | Hygiene: per-arch FVU, L0, FVE, training steps. | `results/ward_backtracking_txc/hygiene/reconstruction_table.csv` | ✅ shipped |
| Caption stat | McNemar χ², p, n at per-arch best magnitude. Embedded in Fig 4a caption. | `results/ward_backtracking_txc/b3_math500_cut25/mcnemar_table.csv` | ✅ shipped |
| Caption stat | Paired Wilcoxon (Holm-Bonferroni) for detection AUC vs each baseline at |S|=8. | `results/ward_backtracking_txc/detection/wilcoxon_detection_table.csv` | ✅ shipped (none cross HB-corrected p<0.05; smallest p_holm=0.31) |

### A.2 Appendix (Section X — Backtracking case study)

| Slot | Artifact | Source path | Status |
|---|---|---|---|
| Fig X.1 | Headline plot **raw** magnitudes (uncalibrated). | `headline_raw.png` | ✅ shipped |
| Fig X.2 | Appendix variant of Fig 4a with TXC-H8 line included. 6 lines instead of 5. | `appendix_calibrated.png` + `appendix_raw.png` | ✅ shipped |
| Fig X.3 | Repetition-rate vs (calibrated) magnitude per arch. Auxiliary judge-free metric. | `repetition_rate.png` (all archs) + `repetition_rate_headline.png` (5 archs) | ✅ shipped |
| Fig X.4 | Coherence vs magnitude (existing metric kept for transparency). | existing pipeline output | partial |
| Fig X.5 | Per-arch L0 + FVU training curves. | `hygiene/training_curves/*.png` | ✅ shipped (6 plots) |
| Fig X.6 | Detection AUC + F1 vs |S| including TXC-H8. | `detection/detection_appendix.png` | ✅ shipped |
| Tab X.1 | Full flip-matrix counts per (arch × magnitude × question). | derived from `flip_matrix.parquet` (9150 rows) | data exists |
| Tab X.2 | Blind judge κ + raw agreement vs Aniket's hand-scored 20-transcript sample. | `judge_validation/blind_pairs.csv` (CSV ready, awaits human scoring) → `kappa_report.txt` | CSV ready (Aniket to fill); validation script TODO |
| Tab X.3 | Per-arch summary AUC + F1 across |S| ∈ {1,2,4,8,16,32}. | `detection/summary_auc_f1.csv` | ✅ shipped |
| Box X.1 | SAE peak forensics — qualitative notes from manual review of ~20 transcripts at the SAE's narrow peak. What is the SAE actually doing there? | `notes/sae_peak_forensics.md` | pending — needs Aniket to read |
| Box X.2 | Verbatim Sonnet judge prompt for backtracking labeling. | text below in §B.2 | drafted |
| Box X.3 | Verbatim coherence rubric used in B1 / B3. | text below in §B.3 | pending |
| Box X.4 | Full Stage A → Stage B pipeline diagram. | `notes/pipeline_diagram.png` (TODO) | TODO |
| Note X.1 | TSAE-paper architectural-deviation caveat (Bhalla 2026 vs our Han attention TSAE w/ k=20). | `notes/tsae_paper_param_audit.md` | drafted |
| Note X.2 | T=6 vs T=5 deviation note (config has T=6; Dmitry's standardization said T=5). | text below in §B.1 | drafted |
| Note X.3 | TFA wrapper is `_tfa_module.TemporalSAE` w/ `use_pos_encoding=True`, `bottleneck_factor=64`, `n_heads=8`, k=20 — heavier than the original TFA paper's bottleneck=1 default; rationale. | text below in §B.4 | drafted |
| Note X.4 | MLC mining caveat: multi-layer capture at sentence token via simultaneous hooks on layers {8,9,10,11,12}. Same encoder math as TXC; only data dispatch differs. | text below in §B.5 | drafted |
| Note X.5 | Cohort definitions: truly-wrong (n=31) + 30-correct random subsample (seed=42) → 61 questions × 25 mags = 1525 panels per arch. | text below in §B.6 | drafted |
| Note X.6 | Calibration: 95th-pctile of |feature activation| (signed-residual fix). TFA/TSAE codes are signed and ~400× smaller than TXC TopK output; calibration normalizes scale. | text below in §B.7 + §C.2 #4 | drafted |
| Note X.7 | Detection probe protocol: 5-fold GroupKFold by question_id, per-fold top-S feature selection by mean-diff, sklearn LogisticRegression(liblinear). Wilcoxon TXC vs each baseline at |S|=8 — none HB-significant (smallest p_holm=0.31). | text below in §C.3 | drafted (data + interpretation) |

### A.3 Hard cuts (do NOT include anywhere)

- TXC-H13: dropped from headline AND appendix per Dmitry's standardize directive. Mention in one-line "we also tested H13; results in supplementary" if asked.
- Stacked SAE, `tsae_paper` (Bhalla ReLU+L1 mis-named variant), full hill-climb leaderboard — supplementary only, not appendix figures.
- The 9-magnitude grid results (`b3_math500_cut25_orig`) — superseded by the 25-mag results.

---

## B. Appendix prose drafts (rough, picks up as we run)

### B.1 Note: T choice (T=6 vs Dmitry's T=5)

Our Stage B configuration uses T=6 because the existing Stage A residual cache and all already-trained TXC checkpoints are at that window size, and Dmitry's standardized-arch-set guidance arrived after retraining was complete. We kept T=6 to avoid eating a full retrain day; for the comparison set in this paper, the architecture distinctions (TXC vs TSAE vs TFA vs MLC vs SAE) dominate any 1-token difference in window length. We confirmed this on the sparse-probing benchmark, where T=5 and T=6 results are within noise (`docs/aniket/experiments/sparse_probing/...`). All TXC variants and TSAE-paper used T=6; MLC's "T axis" indexes 5 layers (L8..L12), which is independent.

### B.2 Box: Sonnet judge prompt for backtracking labeling

Captured verbatim from `experiments/ward_backtracking_txc/grade_backtracking.py`. We instruct Sonnet 4.6 to count "GENUINE backtracking events" — error-catching, missing-constraint detection, approach-rejection, assumption re-evaluation — and to **not** count filler ("Hmm, let me think"), pseudo-backtracking (same conclusion restated), looping, or gibberish. Output schema: `{"genuine_count": int, "raw": str}`. Cost ≈ \$0.002 per row, resumable. The TSAE-paper-magnitude run uses ~18k rows × \$0.002 ≈ \$36 in judge calls.

### B.3 Box: Coherence rubric (B1 / B3)

[FILL IN once finalized — currently in `grade_sonnet.py`. Needs verbatim copy.]

### B.4 Note: TFA hyperparameter choices

The TFA module (`src/bench/architectures/_tfa_module.py:TemporalSAE` w/ `use_pos_encoding=True`) is the same class our TSAE uses but with sinusoidal positional encodings inside the ManualAttention layer. The TFA-paper toy-model setting uses `n_heads=4, bottleneck_factor=1` for small SAE widths; for our `d_sae=16384` SAE this would put 16384-dim attention vectors per head (memory-prohibitive). We use `n_heads=8, bottleneck_factor=64` to match our TSAE-paper entry exactly; the only architectural difference between the TFA and TSAE-paper lines in our headline is the positional encoding. This isolates the contribution of the positional encoding to detection / steering performance.

### B.5 Note: MLC integration

Per Lindsey 2024, MLC is a TemporalCrosscoder where the "T axis" indexes simultaneous layers rather than consecutive tokens. Mathematically identical encoder/decoder; only data dispatch differs.

For training data, we cache four additional residual-stream layers (L8, L9, L11, L12) alongside our existing L10 cache and stack them as `(B, 5, d)` per sample (single token, five layers) instead of `(B, 6, d)` per sample (six tokens, one layer). For mining, we hook all five layers simultaneously and slice the labeled sentence's representative token. Steering at inference targets the centre layer's decoder column (L10), which corresponds to our actual hookpoint.

### B.6 Note: Cohort

Stage A produces 150 MATH-500 traces, of which 78 are unsteered-correct, 31 are unsteered-incorrect with a parsed answer ("truly-wrong"), and 41 are unsteered-incorrect with no parsed answer ("token-truncated"; dropped). For the flip-matrix analysis we need both directions (incorrect→correct *and* correct→incorrect), so we steer all 31 truly-wrong plus a random subsample of 30 correct (seed=42). Total: 61 questions × 25 magnitudes = 1525 panels per arch × 4 arches in the primary sweep + 2 in the extension = 9150 panels.

### B.7 Note: TSAE-paper architectural-deviation caveat

[Reference `notes/tsae_paper_param_audit.md` verbatim; key points: Bhalla 2026 uses BatchTopK k=20 + 20/80 high/low feature split + adjacent-token contrastive loss with reg coef 1.0 (NOT 0.1 — Dmitry mis-quoted). Our `tsae` arch is Han's attention-based TSAE with k=20; we do NOT implement the high/low split or adjacent-token contrastive. We label this "TSAE-paper" with the mismatch documented. Faithful reimplementation is left to future work.]

### B.8 Section: SAE peak forensics (placeholder)

Pending: at the SAE's narrow magnitude peak (located after the densified sweep finishes), pull ~20 transcripts at peak ± 0.5 and ~20 off-peak. Aniket reads. Document:
- Is the SAE doing genuine high-quality backtracking, or hitting a sentence-template the judge happens to like?
- What's the distribution of repetition / looping at the peak vs off-peak?
- Cross-reference with repetition-rate-vs-magnitude plot (Fig X.3).

### B.9 Section: Judge validation (placeholder)

Pending: 20-transcript blind sample. Aniket scores coherence (0–3), backtracking-present (0/1), looping-present (0/1) BEFORE seeing the LLM judge. Then merge. Compute Cohen's κ + raw agreement. Targets: agreement ≥ 80%, κ ≥ 0.6. If below, prompt-iterate once, re-test, document any remaining gap as a paper limitation.

---

## C. Working scratchpad — pick up as the pipeline lands

### C.1 What sweeping reveals (2026-05-03 — full 5-arch sweep landed)

For each arch, peak net rescues (n_ic − n_ci) and behavior at extreme magnitudes:

| Arch | Peak net (mag) | mag = +16 (rescue/regress) | mag = −16 | Comment |
|---|---|---|---|---|
| **TXC** | +7 (mag = −2) | 0/30 | 0/30, 30 regress | Sharp drop-off in positive direction; canonical "negative direction" feature (f14621 was mined neg) |
| **TXC-H8** | +6 (mag = −1) | 0/30, 29 regress | 0/30, 29 regress | Symmetric extreme failure |
| **SAE** | +6 (mag = +3) | 4/31, 11 regress | 3/31, 24 regress | Less catastrophic; tolerates +mag better than other archs |
| **TSAE-paper** | +6 (mag = 0, +2) | 0/31, 16 regress | 4/31, 15 regress | Symmetric pattern around 0 |
| **TFA** | +7 (mag = −8, +6 also strong) | 3/31, 12 regress | 7/31, 14 regress | Multiple peaks; broad usable range |
| **MLC** | **+8 (mag = +4)** | 1/31, 24 regress | 3/31, 22 regress | Highest peak; broadest usable range |

McNemar @ per-arch best magnitude (from `mcnemar_table.csv`):
- TXC mag=−2: χ²=4.00, p=0.039
- TXC-H8 mag=−1: χ²=1.79, p=0.18 (n.s.)
- SAE mag=0: χ²=3.13, p=0.070
- TSAE-paper mag=0: χ²=3.13, p=0.070
- TFA mag=−8: χ²=3.27, p=0.065
- **MLC mag=+4: χ²=4.08, p=0.039 (significant)**

### C.2 Surprises / failure modes

1. **High-magnitude collapse is universal.** At |mag| = 16 every arch regresses ≥ 11 of 30 originally-correct questions; at TXC and TXC-H8 it's 30/30 (literally every correct answer is broken). The plot's calibrated x-axis hides some of this (raw mag 16 ≈ calibrated 8.5 for TXC, 8.0 for TXC-H8) but the failure-at-extremes pattern is the headline of the regression-rate panel.

2. **MLC and TFA are surprisingly competitive.** Counter to my expectation that "TXC is the broadest", the densified grid shows MLC and TFA matching or exceeding TXC on net-rescues at their peaks. MLC peak +8 at calibrated mag ≈ +1.86 is the strongest net result in the panel. Need to be careful with the headline framing — "TXC wins on robustness" isn't what the data shows. The defensible claim is "no architecture has both robustness and a clean rescue signal at high magnitudes; the steering protocol for temporal-aware archs (TXC family + MLC) needs further work."

3. **TFA / TSAE-paper feature activations are signed residuals, not strictly-positive sparse codes.** The `pred_codes + novel_codes` returned by their forward pass produces values mostly in [−0.01, +0.01] — three orders of magnitude smaller than TXC's TopK output (which lands around 0–4). The 95th-percentile-of-nonzero calibration originally returned 0 for TFA and TSAE-paper (the `flat > 0` filter dropped everything since most values are slightly negative). Fixed by switching to `abs(values)` first. Documented this in `notes/backtracking_appendix_draft.md` for the methods section.

4. **Calibration ratios across arches** (p95_pooled of |feature activation|):
   - TXC = 1.877, TXC-H8 = 1.535, MLC = 2.152, SAE = 5.608, TSAE-paper = 0.0042, TFA = 0.0074
   - SAE's natural scale is ~3× TXC's; TSAE-paper / TFA are ~400× smaller. After calibration this all collapses to a comparable x-axis.

5. **The "TXC's negative direction" complication.** `txc__resid_L10__k16__s42 f14621 pos0` was mined as a NEGATIVE-direction backtracking feature; magnitude axis is asymmetric for TXC compared to e.g. SAE's positive-direction feature. The headline plot's x-axis treats these symmetrically, so TXC looks worse on the positive side than it "should". For the appendix, add a "magnitude in the rescue-direction per arch" version of the plot. For the main text, decide with Dmitry whether to flip TXC's axis or use absolute calibrated magnitude.

### C.3 Detection probe results (2026-05-03)

5-fold GroupKFold by `question_id` on 23,664 labeled sentences (intersection across all 6 arches). Per-fold top-S feature selection via |mean-difference|; logistic regression with liblinear solver.

Mean AUC by (arch × |S|):

| Arch | S=1 | S=2 | S=4 | S=8 | S=16 | S=32 |
|---|---|---|---|---|---|---|
| **TXC** | 0.593 | 0.644 | 0.670 | **0.681** | 0.699 | 0.708 |
| **TXC-H8** | 0.572 | 0.597 | 0.616 | 0.658 | 0.688 | **0.716** |
| **SAE** | 0.605 | 0.618 | 0.637 | 0.655 | 0.670 | 0.715 |
| **TSAE-paper** | 0.598 | 0.636 | 0.645 | 0.668 | 0.674 | 0.687 |
| **TFA** | 0.593 | 0.602 | 0.620 | 0.633 | 0.632 | 0.637 |
| **MLC** | 0.586 | 0.648 | 0.653 | 0.663 | 0.663 | 0.663 |

- TXC has highest AUC at |S|=8 (the headline `|S|`) — but the lead is small (0.681 vs 0.668 for TSAE-paper, 0.663 for MLC).
- At |S|=32, TXC-H8 narrowly leads (0.716) but TXC-H8 had FVU=0.50 so this AUC is on a poorly-reconstructing dictionary — interesting but suspect.
- **None of the TXC vs baseline differences cross HB-corrected p<0.05** (smallest p_holm = 0.31 for TXC vs TFA, n_folds=5).
- **F1 is uniformly low** (~0-0.08) because the positive class is ~12% imbalanced and the threshold is 0.5. For the paper, switch to PR-AUC or pick a class-balanced threshold; this is appendix-quality as-is.

**Honest framing for the case study text**: "Detection AUC is comparable across architectures (range 0.63–0.72 at |S|=32), with TXC slightly leading at small |S|. With our held-out eval set size (23.6k sentences, 5 CV folds across questions), differences between TXC and the strongest baselines (SAE, TSAE-paper, MLC) are not statistically significant after multiple-comparison correction. Backtracking IS detectable in all dictionaries, which is itself the case-study positive: the temporal-aware archs do not lose detection power vs the conventional SAE."

### C.4 Followups for "future work" paragraph

- Faithful Bhalla 2026 TSAE port (high/low feature split + adjacent-token contrastive, reg=1.0).
- TFA + MLC at multiple T windows.
- Plan Generation case study from Bogdan & Macar 2026 taxonomy (notes/thought_anchors_taxonomy.md).
- Andre's hybrid detection→steering protocol applied to backtracking specifically.

---

## D. Discipline note

When a new artifact lands (plot, table, transcript), add a one-line entry to §A AND a one-paragraph note in §B or §C. Don't let qualitative observations stay in conversation context where they'll get lost.
