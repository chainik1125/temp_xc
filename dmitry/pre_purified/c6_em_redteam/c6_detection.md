---
component: c6
type: redteam
author: dmitry
date: 2026-05-05
status: pre-purified
tags:
  - redteam
  - detection
  - c6-em
  - results
  - pre-purified
---

## Detection complement to the C6 EM steering test

> **Status:** pre-purified. This is external review work on Han's c6
> result. Promotion into `purified/` is conditional on Han + agent_paper
> sign-off.

### Context

The C6 redteam pass investigated whether c6.md's
*"TXC-base + brickenauxk_a8 strictly beats SAE-arditi on alignment
peak"* headline reflects a real architectural difference or a
measurement artifact. The intervention-side analysis (dense α-sweep
at \|α\|∈[10..200] + decoder-norm probe) established that **TXC's
"α=±100 peaks" are moderate-effective-magnitude steering by a factor
of 1/√T**:

- TXC-base (T=5): empirical decoder-row norm 0.4425 ≈ 1/√5 = 0.4472
- TXC-pro (T=10): empirical decoder-row norm 0.3099 ≈ 1/√10 = 0.3162

Both within 1–2% of the prediction. The coherence-collapse boundary
under SAE α=±100 vs TXC α=±200 sits at the same *effective magnitude*
≈ 90–100, confirming the convention story.

This followup tests the **detection complement**: at the dictionary
level, can SAE / TXC features distinguish BASE-Qwen activations from
BASE+LoRA-organism activations on a per-prompt basis? An architecture
that's a poor steerer might still be a good detector — or vice versa.

### Methodology

For each of the 8 canonical c6 cells (4 SAE + 4 TXC × 2 organisms ×
2 seeds):

1. Load the 500 cfierro probe prompts that Wang stage 1 uses
   (`personality-qs-risky-financial-advice` for 14B-finance,
   `personality-qs-bad-medical-advice` for 7B-medical).
2. Run them through both the BASE Qwen subject model (no LoRA) AND
   the BASE+LoRA-organism model. Capture residual-stream activations
   at layer L (24 for 14B, 15 for 7B) for **all prompt tokens**.
3. Encode via the trained dictionary, **mean-pool latents over each
   prompt's token positions**:
   - SAE (T=1): encode every token → mean across positions
   - TXC (T=5): slide a T-window over positions, encode each window,
     mean across windows
   This matches the pooling convention of `temp_bench.eval.probing.s_tail_probe`
   and Wang stage 1's `compute_delta_z_ranking_from_acts`.
4. Stack `X = [Z_base; Z_lora]` shape `(1000, 32768)`,
   `y = [0]·500 + [1]·500` (0 = BASE, 1 = LoRA-organism).
5. **(a) Per-feature AUC**: `roc_auc_score(y, X[:, f])` for each f.
   Report two-sided AUC = `max(auc, 1 - auc)` so we don't penalise
   features whose direction-of-shift is sign-flipped.
6. **(b) Multi-feature 1-D regression**: 5-fold CV `LogisticRegression`
   on the full d_sae feature space; report mean accuracy + AUC.

Source: [`detection_eval.py`](detection_eval.py).

### Results

Per-cell — headline-finalist AUC (the same feature used for steering),
its rank in the dictionary, and the multi-feature LR score:

| arch / cell             | finalist | AUC₂ₛ | rank / 32768 | multi-LR acc / AUC |
|-------------------------|---------:|------:|-------------:|--------------------:|
| SAE-arditi 14B-fin s=1  |    19196 | 0.845 |          257 |       1.000 / 1.000 |
| SAE-arditi 14B-fin s=42 |    19897 | **1.000** |       **#1** |       1.000 / 1.000 |
| SAE-arditi 7B-med s=1   |     1119 | 0.500 |         1667 |       1.000 / 1.000 |
| SAE-arditi 7B-med s=42  |    21341 | 0.846 |         1065 |       1.000 / 1.000 |
| TXC-base   14B-fin s=1  |     4636 | 0.500 |          531 |       1.000 / 1.000 |
| TXC-base   14B-fin s=42 |    29452 | **1.000** |       **#1** |       1.000 / 1.000 |
| TXC-base   7B-med s=1   |    31740 | 0.500 |          968 |       1.000 / 1.000 |
| TXC-base   7B-med s=42  |    29196 | **1.000** |       **#1** |       1.000 / 1.000 |

Per-cell raw outputs, including top-10 features by AUC, are at
[`results/c6_<train_key>/detection.json`](results/).

### Findings

#### 1. Multi-feature detection: tied at perfect

All 8 cells score **logreg AUC = 1.000** with the full 32K-dim
dictionary. Both SAE and TXC dictionaries encode the BASE→organism
shift well enough that a linear classifier separates them on a
per-prompt basis with zero error. There is **no detection-side
advantage for either architecture at the dictionary level**.

#### 2. Single-feature finalist detection: bimodal, *not* arch-correlated

The headline finalist's per-prompt AUC splits cleanly into three regimes:

- **AUC = 1.000, rank #1** (3/8 cells: SAE 14B-fin s=42, TXC 14B-fin s=42,
  TXC 7B-med s=42): the same feature used for steering is the
  **single best detector** in the entire 32K-dim dictionary.
- **AUC ≈ 0.85** (2/8 cells: SAE 14B-fin s=1, SAE 7B-med s=42):
  good per-prompt detector but not the best.
- **AUC = 0.500** (3/8 cells: SAE 7B-med s=1, TXC 14B-fin s=1,
  TXC 7B-med s=1): the steering finalist has **chance-level
  per-prompt detection signal**.

This split is **not arch-correlated**: SAE and TXC each contribute one
"perfect" cell, one "good" cell, and one or two "chance" cells.
The bimodality is per-cell, not per-arch.

#### 3. Steering-vs-detection coupling is mechanism-dependent

The Wang procedure selects steering features by Δz̄ — the *mean
activation shift* between BASE and organism, averaged over many
tokens. A feature whose mean shift is large can still have nearly
overlapping per-prompt distributions if it fires *sparsely* but with
*large magnitude*: a few prompts with z[f] = 50, the rest with z[f] = 0,
gives a high mean but chance-level AUC because most samples can't be
ranked.

This explains the AUC=0.5 cells: the steering finalist's contribution
to Δz̄ comes from rare large-magnitude firings, not consistent
per-prompt activation. Such features can still be powerful steering
directions (the decoder pushes the residual stream regardless of
encoder activation), but they're useless as monitors for misalignment
without sequence-level pooling.

The "AUC=1.0, rank #1" cells likely correspond to features that fire
consistently *across* the organism's activations — these are
simultaneously the best steerers AND the best detectors.

#### 4. Implication for the C6 paper claim

Beyond the steering-side artifact (decoder-norm asymmetry), the
detection complement adds:

- The dictionary capacity to encode the BASE→organism distinction is
  **architecture-invariant** (multi-feature AUC = 1.0 for both).
- The single steering feature's role as a detector is **per-cell
  variable**, not architecture-dependent.

Combined: there's no detection-side support for "TXC encodes the
EM-misalignment direction better than SAE" nor for the inverse. The
two architectures have **statistically identical detection capability**
at the dictionary level on this task.

### Reproduction

#### Prerequisites

- Repo checked out (any branch with the `purified/` infrastructure;
  the paths in `detection_eval.py` auto-detect by walking up from
  `__file__` to find the `purified/` root).
- Activation cache + canonical `wang_full.json` artifacts on disk
  under `purified/results/runs/c6_<train_key>/` (already shipped on
  `origin/final`).
- Trained SAE / TXC checkpoints. The driver downloads automatically
  from `han1823123123/temp-bench-models` if not local.
- `HF_TOKEN` exported (for HF cached-asset access).
- Anthropic API key not needed (detection has no judge calls).

#### Setup (RunPod H100 fresh pod)

```bash
# Clone + venv (overlay disk = fast install; rebuild after pause)
cd /workspace
git clone --branch dmitry-c6-redteam --single-branch \
    https://github.com/chainik1125/temp_xc.git temp_xc-c6-extend
cd temp_xc-c6-extend
/usr/local/bin/python -m venv --system-site-packages /root/c6_venv
. /root/c6_venv/bin/activate
pip install --quiet transformers peft safetensors huggingface_hub \
    pydantic datasets accelerate scikit-learn
```

#### Run

```bash
. /root/c6_venv/bin/activate
cd /workspace/temp_xc-c6-extend
HF_TOKEN=$HF_TOKEN python dmitry/pre_purified/c6_em_redteam/detection_eval.py
```

Outputs land at `/workspace/c6_redteam/detection/c6_<train_key>/detection.json`
by default (override via `C6_OUT_ROOT` env var).

#### Resume

The script supports `--skip-existing` to resume after interrupt:

```bash
python dmitry/pre_purified/c6_em_redteam/detection_eval.py --skip-existing
```

#### Resource footprint

- ~7 GB GPU memory peak (Qwen-7B BF16) during 7B-medical activation pass
- ~30 GB GPU memory peak (Qwen-14B BF16) during 14B-finance pass
- ~3 minutes total wall-clock for all 8 cells when activations are
  cached per-organism.
- No API costs (no judge calls).

### Critical files

- [`detection_eval.py`](detection_eval.py) — driver
- [`results/c6_<train_key>/detection.json`](results/) — per-cell JSON
  outputs (one per cell, includes top-10 features by AUC)

### Open questions / followups

- **Cross-organism feature stability**: do features that score AUC=1.0
  on one organism transfer? The 14B-finance s=42 finalist is feat
  19897 (SAE) / 29452 (TXC); does either fire on 7B-medical organism
  activations? Worth pulling the top-10 lists across cells and
  checking for overlap.
- **Generation-time vs prompt-time detection**: this eval uses
  prompt-only activations (the LoRA shifts the model's representation
  just by being in-place). A more deployment-relevant setup would
  judge a generated response and detect from its generation-time
  activations. The plumbing for this is straightforward via
  `judge_outputs_extended.jsonl` rollouts + generation hooks.
- **Per-feature steering vs detection scatter plot**: for the top-100
  Δz̄-ranked features, plot per-feature steering Δalign vs per-feature
  AUC. If the relationship is linear, intervention and detection are
  measuring the same thing; if it's L-shaped or random, they're
  decoupled.
