---
status: active
created: 2026-07-26 ~16:00 London
for: mac-a (figures 1-2), mac-b (figure 3 + table data)
---

# Publication-grade figures for the collaborator writeup (Han's ask)

Target doc: `experiments/explorations/task_hunt/WRITEUP.md` (mac-local
writes prose). Figures to `experiments/explorations/task_hunt/
figs_writeup/` as PDF + 300-dpi PNG. **Style spec (both agents,
identical):** matplotlib, constrained layout, font size ≥ 11,
colorblind-safe palette (Okabe–Ito), no titles-in-figure (captions
live in the doc), axes labeled in PLAIN language ("held-out
correlation r", "window length T (tokens)"), grid off or minimal,
legend outside or upper-left, error bars = 95% t-CI whiskers over
seeds, PDF vector. NO internal jargon anywhere in axis/legend text
(no "λ̂", no "KEEP", no "v1", no arch codenames — use "TXC (pre)",
"TXC (post, matched budget)", "Stacked SAE", "T-SAE", "per-token
SAE", "untrained control", "visible-cue baseline").

- **Fig 1 (mac-a): backtracking-intensity T-scaling** (the λ̂ panel,
  Ward/R1-Distill, n=6 where available): x = T ∈ {2,4,8,16}, y =
  held-out r; one line per architecture (trained), per-token SAE and
  T-SAE as horizontal reference lines with CI bands; untrained
  control as light dotted line. Data: canonical leaderboard rows for
  `ward_real_lambda_base_l12` (the R4/R22 seed sets).
- **Fig 2 (mac-a): question-gap T-scaling** (dq panel, llama31):
  x = T ∈ {2,4,8,16,32}, y = held-out r; same style; ADD the
  visible-cue baseline ("predicting from question marks visible in
  the window") as a dashed black line rising 0.106→0.499 — the
  licence zone (T ≤ 8) is where TXC clears it; shade T > 8 lightly
  and annotate "visible-cue baseline dominates". Untrained controls
  dotted. Data: `dial_real_dqgap_llama31_8b_l14` rows +
  `panel_evidence_line_dq.json`.
- **Fig 3 (mac-b): the two order receipts side by side** (a 2-panel
  bar figure): left = backtracking (shuffle costs +0.028…+0.041 on
  anticipation vs ≤ +0.013 ambient, σ_null shown); right = dialogue
  mechanism ladder at T32 (full shuffle / within-turn / turn-order /
  far-half / near-half costs, 3 models grouped, null band shown).
  Plain-language labels ("accuracy lost when context order is
  shuffled").
- **mac-b also**: dump a small JSON/CSV of the negatives table
  numbers I'll cite (per task: the one decisive number + its
  artifact path) so the doc's table links to receipts.

Push figures + a one-line LOG note each; I integrate. No new
compute. Deadline: figures by ~17:15 London for the 18:00 check-in;
the doc survives past the check-in as a living page — iterate after
if needed.
