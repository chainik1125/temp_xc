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
(commit `3bec1ac0`). Numerical claims were not cross-checked because
the paper draws on a separate reproduction run, not the leaderboard
on `origin/final`.

## CRITICAL — must fix before submission

### Two missing BibTeX entries (compile as `[?]`)

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

## NON-ERRORS — verified correct (for the record)

These were spot-checked against the actual code in `purified/` and
pass at the **structural / parametric** level (i.e. the description
of the data-generating process and HMM formalism is consistent with
the implementation):

- **Setup B (Denoising) parameters** — N=20, ρ=0.7, p_B=0.625: match
  `toy_markov_n20_d40_noisy` YAML and `markov.py`.
- **Setup D (Coupling) parameters** — K=10, M=20, n_parents={5,10},
  ρ=0.9, π=0.05, p_B=0.5, d=256, d_sae=40: match
  `toy_coupled_noisy_K10_M20_d256_pB05_np{5,10}` YAML.
- **HMM edge-emitting formalism** + OR-gate firing + hidden-chain
  footprint formula `f^hid_k = normalize(Σ_{j: C_{jk}=1} f^emit_j)`:
  match `coupled.py` and `coupled_noisy.py`.
- **Compute checklist (Q.7) numbers** — 330 paper-bound + 820
  exploration = 1150 total H100-hr: matches
  `docs/paper/compute_resources.md`.
- 35 of the 39 distinct `\cite*` keys used in `main.tex`+`appendix.tex`
  resolve in `refs.bib`; the only LIVE-cited gaps are the two listed
  above.

## Notation note (not an error)

The paper uses `s_{i,t}` for the **clean hidden state** and `a_{i,t}`
for the **noisy emission**, while our code (`markov.py`) uses `h` for
clean and `s` for noisy/observed. The semantics match exactly; only
variable names differ. Anyone cross-referencing `denoising_probes.py`
while reading the synthetic section should be aware.

## How to act on this

| Issue | Owner | Effort |
|---|---|---|
| Add 2 BibTeX entries | paper authors | 5 min |
