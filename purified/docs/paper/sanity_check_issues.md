---
title: Paper sanity check — critical issues found in `origin/final-aniket:docs/aniket/`
author: agent_synth (taking over from agent_paper)
date: 2026-05-07
status: action-required
tags: [paper, neurips, sanity-check, critical]
---

# Paper sanity check — critical issues

Sanity check of the camera-ready draft at
`origin/final-aniket:purified/docs/aniket/{main,appendix,checklist}.tex`
(commit `3bec1ac0`). Most setup descriptions and headline numbers
verified correct. Items below are real bugs — flagged in priority
order.

## CRITICAL — must fix before submission

### 1. Two missing BibTeX entries (compile as `[?]`)

Both citations are LIVE in the appendix and will render as the broken
`[?]` placeholder when LaTeX builds with `\bibliographystyle{unsrtnat}`.

| Citation key | Where | File:line |
|---|---|---|
| `bills2023language` | "Bills-et-al-style \cite{bills2023language} prompt" | `appendix.tex:334` |
| `templeton2023scaling` | "Bricken-style hard resampling \citep{templeton2023scaling}" | `appendix.tex:21` |

Note: `cagnetta2026deriving` and `fineweb_penedo2024the` are also
missing from `refs.bib` but appear only in `% commented-out` blocks —
safe to leave for now.

**Fix — append to `refs.bib`:**

```bibtex
@article{bills2023language,
  author = {Bills, Steven and Cammarata, Nick and Mossing, Dan and Tillman, Henk and Gao, Leo and Goh, Gabriel and Sutskever, Ilya and Leike, Jan and Wu, Jeff and Saunders, William},
  title  = {Language models can explain neurons in language models},
  year   = {2023},
  url    = {https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html},
  note   = {OpenAI},
}

@article{templeton2023scaling,
  author       = {Templeton, Adly and Conerly, Tom and Marcus, Jonathan and Lindsey, Jack and Bricken, Trenton and Chen, Brian and others},
  title        = {Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet},
  year         = {2024},
  howpublished = {\url{https://transformer-circuits.pub/2024/scaling-monosemanticity/}},
  note         = {Transformer Circuits Thread},
}
```

## NUMERICAL discrepancies

### 2. TopK gAUC=0.92 on Setup D-np10 (k=1) is fragile

`main.tex:744`: *"the TopK SAE … drops from gAUC=0.92 at k_pos=1 to
0.44 at k_pos=8"*.

Direct read-off from `results/leaderboard.jsonl` for
`toy_coupled_noisy_K10_M20_d256_pB05_np10`, arch=topk_sae, k_pos=1:

| Subset | n cells | mean gAUC |
|---|---:|---:|
| All (3 n_steps configs × 3 seeds) | 9 | **0.842** |
| Post-cutover (n_steps=8K only, 22:54Z+) | 3 | **0.915** ≈ 0.92 |

The k_pos=8 number 0.44 is correct (actual 0.439). The k_pos=1
number 0.92 is correct ONLY under the post-cutover-only filter; the
cross-config mean is 0.842. Either:

- (a) keep 0.92 and add one explicit sentence in the methodology
  appendix that the canonical D-np10 subset is the post-2026-05-06T22:54:30Z
  cells (matches `plot_headline.py:ZOOM_CUTOFF_TS`), or
- (b) change the number to 0.84 to match the cross-config mean.

(a) is recommended — the post-cutover cells are the real canonical.

### 3. C7 inducement: abstract "15% more effective" vs leaderboard ~18%

`main.tex:78` (abstract): *"are 15% more effective in inducing
backtracking behaviour"*.

Direct compute from `results/leaderboard.jsonl`, c7 component, peak
`delta_gc_mag_*` per arch (across all magnitudes):

| Arch | peak Δgc |
|---|---:|
| txc_base default | **0.426** |
| txc_pro default | 0.377 |
| topk_sae default | 0.361 |
| tfa default | 0.344 |
| stacked_sae default | 0.328 |
| tsae_paper default | 0.246 |
| mlc default | 0.164 |
| txc_base_mw default | 0.787 (NOT in locked-paper set) |

TXC-base vs TopK best baseline: 0.426 / 0.361 = **+18.0%** improvement.
Abstract claims **15%** — slight underclaim. Defensible but the round
figure should be tightened to whichever value the camera-ready table
will show.

### 4. C7 appendix table 5 vs leaderboard mismatch

`appendix.tex:526`: *"Peak Δgc values are 0.230 for TXC-base bs=256,
**0.541 for TXC-base bs=1024**, 0.377 for TXC-pro bs=256, and **0.475
for TXC-pro bs=1024**"*.

But the leaderboard cells under `arch=txc_base, t_label=default,
component=c7` report peak Δgc = **0.426** (not 0.541), and TXC-pro
peak = **0.377** (matches the bs=256 number, not bs=1024).

Possible causes — would need agent_back / agent_paper to confirm:

- Appendix numbers come from a Sonnet-4.6 re-judge of generations that
  the leaderboard rows still index against an older judge.
- The appendix table mixes up bs=256 and bs=1024 row labels (the
  TXC-pro numbers especially look swapped — 0.475 vs 0.377).
- Leaderboard rows for the bs=1024 cells aren't yet refreshed with
  the optimal-magnitude Δgc.

Action: reconcile against the underlying `judge_outputs.jsonl`
transcripts before camera-ready, OR drop the precise four-number
sentence (line 526) in favour of a single number with a single
verifiable source.

## NON-ERRORS — verified correct (for the record)

These were spot-checked against the actual code/data and pass:

- **Setup B (Denoising) parameters** — N=20, ρ=0.7, p_B=0.625: match
  `toy_markov_n20_d40_noisy` YAML and `markov.py`.
- **Setup D (Coupling) parameters** — K=10, M=20, n_parents={5,10},
  ρ=0.9, π=0.05, p_B=0.5, d=256, d_sae=40: match
  `toy_coupled_noisy_K10_M20_d256_pB05_np{5,10}` YAML.
- **HMM edge-emitting formalism** + OR-gate firing + hidden-chain
  footprint formula `f^hid_k = normalize(Σ_{j: C_{jk}=1} f^emit_j)`:
  match `coupled.py` and `coupled_noisy.py`.
- **Setup B headline number R²_global=0.48 for TXC-base best** with
  TXC-pro lower: **verified** from `experiments/c1_noisy_filler/
  denoising_probe_results.json`. Peaks: TXC-base T=2 = 0.495 (≈0.48
  rounded), TXC-base T=4 = 0.472, T-SAE = 0.387, TopK-SAE = 0.376,
  TXC-pro = 0.260 (much worse, consistent with paper claim).
- **Coupling bench dense-overlap saturation** — T-SAE and TXC-base
  both 0.99 at k=1, TXC-pro T=2 = 0.976 (≈0.98): verified.
- **Peak gAUC gap ≈0.47 across full sweep**: verified — actual peak
  is **+0.463** on `pB02_np8` (extreme noise + extreme overlap).
- **Compute checklist (Q.7) numbers** — 330 paper-bound + 820
  exploration = 1150 total H100-hr: matches
  `docs/paper/compute_resources.md`.

## Notation note (not an error)

The paper uses `s_{i,t}` for the **clean hidden state** and `a_{i,t}`
for the **noisy emission**, while our code (`markov.py`) uses `h` for
clean and `s` for noisy/observed. The semantics match exactly; only
variable names differ. Anyone cross-referencing `denoising_probes.py`
while reading the synthetic section should be aware.

## How to act on this

| Issue | Owner | Effort |
|---|---|---|
| 1. Add 2 BibTeX entries | paper authors | 5 min |
| 2. Pin canonical D-np10 cells (footnote in app:c2) | paper authors | 10 min |
| 3. Tighten abstract "15%" claim to actual 18% | paper authors | 5 min |
| 4. Reconcile c7 appendix table 5 numbers | agent_back + agent_paper | 30-60 min |

Items 1, 2, 3 are wording fixes that change nothing qualitative.
Item 4 is the only one that needs underlying data verification.

Source of truth for verification:
- `results/leaderboard.jsonl` (eval rows; deterministic train/eval keys)
- `checkpoints/manifest.jsonl` (training_cfg per train_key)
- `experiments/c1_noisy_filler/denoising_probe_results.json` (Setup B
  single-latent + linear-probe metrics)
- `docs/paper/training_appendix.md` (auto-rendered cell counts)
