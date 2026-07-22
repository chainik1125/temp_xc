# Paper — manuscript snapshot (read-only, agent context)

A snapshot of the paper's LaTeX source, kept in-repo **purely as context** for
agents working on this codebase. The manuscript is the ICML 2026 mechinterp
workshop camera-ready, which is essentially identical to the NeurIPS 2026
submission.

**Do not edit or build the paper here.** The canonical, editable source lives in
the external working copy (`~/Desktop/ICML-ready`, the larger build tree). This
directory is a one-way copy for reading — changes made here would not flow back
and would drift from the real source.

## What's here (only the compile inputs of `main.tex`)

- `main.tex`, `appendix.tex` — the manuscript (`appendix.tex` is `\input` by main).
- `refs.bib` — bibliography (`\bibliography{refs}`).
- `icml2026.sty`, `icml2026.bst` — the ICML class + bib style. `icml2026.sty`
  `\RequirePackage`s the bundled `fancyhdr.sty`, `algorithm.sty`,
  `algorithmic.sty` (kept for a self-contained source); everything else it uses
  is stock TeX Live.
- `images/temporal_tikz_both.tex`, `images/txc_tikz.tex` — the two TikZ figures
  `main.tex` `\input`s.
- `figs/` — the 19 figures referenced by `\includegraphics` / the `\autofig`
  macro that exist in the source folder.

## Deliberately excluded

Slides, poster, the (commented-out) NeurIPS `checklist.tex`, meeting notes,
build scripts, `*.md` side-docs, `main.pdf`/`slides.pdf`, `main.bbl`, and stale
figure variants — none are inputs to the final PDF.

## Known gap in the source

`main.tex`/`appendix.tex` reference **`figs/c2/*` (8 files)** and use an
`\input{ICML-ready/appendix.tex}` path — both artifacts of the external build
root. Those `figs/c2/` files are **not present** in the source folder, so this
tree is not a from-scratch buildable bundle; it is the manuscript *text +
available assets* for reading. If a true build is ever needed, regenerate from
the full external project, not from here.
