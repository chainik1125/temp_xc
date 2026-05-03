---
author: Aniket Deshpande
date: 2026-05-03
tags:
  - guide
  - in-progress
  - ward-backtracking
---

## TL;DR

What's done, what's next, and how to get the figures to a paper-ready
state for the Stage B backtracking case study, after the 2026-05-03
NeurIPS push. Companion to [[results_b_neurips_push]] (findings) and
[[methodology_neurips_push]] (how each step works).

## Status snapshot

✅ **Done in this push** (committed + pushed on `aniket-ward-stage-b @ f3399d9c`):

- 5-arch headline + 6-arch appendix steering plot, calibrated + raw
- McNemar table + flip-matrix parquet (9150 rows)
- Detection probe pipeline (5 archs + appendix; AUC + Wilcoxon + Holm-Bonferroni)
- Hygiene table + per-arch training-curve PNGs
- Repetition-rate plot (judge-free)
- Methodology + results writeups (this file + sibling docs)
- 20-transcript blind-judge CSV (ready for scoring)

🟡 **Open / pending** (you / Dmitry):

- **Score the blind-judge CSV** (~30 min; Aniket only): see §B
- **SAE peak forensics** (~45 min; Aniket only): see §C
- **Decide axis convention with Dmitry**: see §D
- **Switch detection F1 → PR-AUC** (~30 min coding): see §E
- **Faithful Bhalla TSAE re-implementation** (~1.5 days; deferred): see §F
- **Plan Generation case study** (Bogdan & Macar 2026 taxonomy): explicitly
  deferred per Aniket; only revisit if backtracking ships strong AND there's
  bandwidth before Sunday EOD freeze. See `notes/thought_anchors_taxonomy.md`.

## Section A — How to regenerate the headline plot from scratch

Cost: ~3 h on 2× H100 from a fresh checkpoint. Cheaper from cached ones.

```bash
# 1. Confirm config has densified mag grid + tsae kval_topk=20 (already shipped)
grep -A 2 "magnitudes:" experiments/ward_backtracking_txc/config.yaml | head
grep "kval_topk" experiments/ward_backtracking_txc/config.yaml

# 2. Retrain TSAE-paper at k=20 (skip if checkpoint exists; ~25 min/hookpoint)
uv run python -m experiments.ward_backtracking_txc.train_txc --cell tsae__resid_L10__k32__s42

# 3. Re-mine TSAE features (~5 min)
uv run python -m experiments.ward_backtracking_txc.mine_features --cell tsae__resid_L10__k32__s42

# 4. Run the primary 4-arch sweep (~2 h, 2-GPU parallel)
bash experiments/ward_backtracking_txc/run_headline_pipeline.sh

# 5. Run the TFA + MLC extension (~1.5 h)
bash experiments/ward_backtracking_txc/run_tfa_mlc_extension.sh
```

The extension script is idempotent: re-running it skips already-cached
layers, already-trained checkpoints, and already-mined features. Useful
for partial reruns.

## Section B — Score the 20-transcript blind sample

Goal: validate the Sonnet judge against your blind hand-scores. Target
Cohen's κ ≥ 0.6 and ≥80% raw agreement on coherence + backtracking + looping.

Steps:

1. Open `results/ward_backtracking_txc/judge_validation/blind_pairs.csv`.
2. **Do NOT look at the `judge_rescued` or `before_correct` columns** — those
   are LLM judgements you don't want to bias on. (Hide them in your spreadsheet.)
3. For each row, fill in:
   - `human_coherence_0_3`: 0 = incoherent / loop, 1 = mostly nonsense, 2 = mostly coherent with issues, 3 = fully coherent.
   - `human_backtracking_present`: 0/1 — does the steered continuation contain genuine backtracking (per the rubric in `experiments/ward_backtracking_txc/grade_backtracking.py`)?
   - `human_looping_present`: 0/1 — does it loop (sentence-level repetition that goes on for ≥3 sentences)?
4. Save the CSV.
5. Tell me; I'll write `validate_judge_kappa.py` that loads the CSV +
   loads the LLM judge results for the same `(arch, mag, qid)` rows, and
   prints per-task κ + raw agreement.

If agreement is below the targets, the next step is to refine the judge
prompt (one iteration, max), re-run on the same 20 sentences, and re-test.
If still below, document as a paper limitation.

## Section C — SAE peak forensics

Goal: figure out whether the SAE's narrow magnitude peak is "genuine high-quality
backtracking" or "judge template artifact."

Steps:

1. Pick the SAE's headline magnitude. From the per-mag summary at
   `results/.../topk_sae__ln1_L10__k64__s42__f5263_pos0/summary.json`,
   the rescue rate peaks at mag=0 (control) and mag=+3 (n_ic=9).
2. From `phase2_rescue.json` (gitignored but on disk), pull ~20 transcripts
   at mag=+3 and ~20 at mag=+5 (off-peak; rescue=6).
3. Read each. For each, note:
   - Is it actually catching an error / backtracking?
   - Is it a sentence template that the judge is over-rewarding?
   - Is the model just resampling toward the answer without backtracking?
4. Document findings in `notes/sae_peak_forensics.md` (one paragraph
   summary + selected verbatim examples).

This goes into the appendix discussion and helps frame whether the SAE's
narrow peak is a strength (precise) or a weakness (sweet-spot fragile).

## Section D — Magnitude-axis decision (open question for Dmitry)

The headline plot shows TXC catastrophically failing at positive
magnitudes (0/30 rescued at mag=+8, 30/30 regressed at mag=+16). But
TXC's headline feature `f14621 pos0` was mined as a *negative-direction*
backtracking feature — its productive direction is mag<0. So a symmetric
x-axis penalizes TXC for steering "the wrong way."

Three options for the writeup:

- **(a) keep symmetric axis as-is.** Most honest reading of the experiment;
  hides direction-specific peaks.
- **(b) flip TXC's axis** so its "amplify-backtracking" direction is on the
  positive side, matching the other arches. Cleaner narrative.
- **(c) use absolute calibrated magnitude.** Symmetric in a different way;
  hides the asymmetry without lying about it.

My instinct: (b) for the headline, (a) in the appendix. Loop in Dmitry
before deciding.

## Section E — Switch detection F1 → PR-AUC

The current F1 numbers in `results/ward_backtracking_txc/detection/summary_auc_f1.csv`
are 0–0.08 because the positive class is ~12% and the threshold is the
default 0.5. AUC is fine (range 0.63–0.72), but F1 reads as catastrophe.

Quick fix in `experiments/ward_backtracking_txc/detection/build_detection_probe.py`:

```python
from sklearn.metrics import average_precision_score
ap = float(average_precision_score(y_te, proba))   # PR-AUC
# Either replace `f1` or add as a third metric column.
```

PR-AUC is the right metric for class-imbalanced detection. Worth ~30 min
of work to swap and regenerate the table.

## Section F — Faithful Bhalla 2026 TSAE re-implementation (deferred)

Per `notes/tsae_paper_param_audit.md`, our `TSAE-paper` line uses Han's
attention TSAE class with `kval_topk=20` to match Bhalla's k. We do NOT
implement Bhalla's actual architecture: the 20%/80% high-low feature
split + adjacent-token contrastive loss with reg-coef = 1.0. A faithful
port would need:

1. New SAE class with masked encoder (high-features-only contribute to
   the predictable component).
2. Forward pass that computes adjacent-token contrastive loss between
   the high-feature codes at tokens t and t-1.
3. Training loop hook to apply the contrastive coefficient.

~1–1.5 days. Out of scope for the camera-ready unless something opens up.

## Section G — Followups for "future work" paragraph

In rough priority:

1. Faithful Bhalla TSAE port (above).
2. Plan Generation case study from Bogdan & Macar 2026 (deferred per
   Aniket; details in `notes/thought_anchors_taxonomy.md`).
3. Andre's hybrid detection→steering protocol applied to backtracking
   (TXC detects, then SAE steers).
4. T-window scaling: train MLC + TFA + TXC at multiple T values and report
   how each scales.
5. Multi-seed runs of TSAE-paper k=20 to get error bars on the McNemar
   p-values.

## See also

- [[results_b_neurips_push]] — what we found
- [[methodology_neurips_push]] — how each step works
- [[NEURIPS_PUSH]] — original execution plan, status log
- `notes/backtracking_appendix_draft.md` — main vs appendix figure manifest
- `notes/tsae_paper_param_audit.md` — TSAE architecture audit
- `notes/thought_anchors_taxonomy.md` — Bogdan & Macar 2026 reasoning taxonomy
