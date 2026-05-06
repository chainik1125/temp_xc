# C7 — Backtracking case study: optimal-magnitude analysis

_Generated from each cell's `steered_phase2_optimal.jsonl`,
`judge_outputs.jsonl`, and `coherence_judge.jsonl`. For each
trained cell, evaluation runs at exactly two magnitudes:
`{0, peak Δgc magnitude}`. The mag=0 column is used for
baseline-corrected `Δnet_corr`. Coherence is the 0–3
Sonnet rubric (port of Aniket's wasteland `grade_sonnet.py`)
with `coherent := grade >= 2`. Backtracking is
`Sonnet COUNT >= 1`._

## Net saves at optimal magnitude (baseline-corrected)

`Δnet_corr = (rescues_peak − regressions_peak) − (rescues_0 − regressions_0)`. Larger is better — positive means steering rescued more questions than the cut-and-continue noise floor would by itself.

| Arch | bs | peak mag | rescues@peak | regr@peak | rescues@0 | regr@0 | Δnet@peak | Δnet@0 | **Δnet_corr** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TXC-base | 256 | -12 | 1 | 16 | 0 | 7 | -15 | -7 | **-8** |
| TXC-base | 1024 | -12 | 1 | 15 | 0 | 7 | -14 | -7 | **-7** |
| TXC-pro | 256 | +12 | 6 | 7 | 0 | 7 | -1 | -7 | **+6** |
| TXC-pro | 1024 | +16 | 0 | 18 | 0 | 7 | -18 | -7 | **-11** |
| TopK SAE | 1024 | -16 | 1 | 14 | 0 | 7 | -13 | -7 | **-6** |
| T-SAE | 1024 | +7 | 3 | 7 | 0 | 7 | -4 | -7 | **+3** |
| MLC | 1024 | +16 | 0 | 13 | 0 | 7 | -13 | -7 | **-6** |

## 2×2 contingency at optimal magnitude

Each cell of the cohort (n=61) classified along 
{coherent (grade≥2) vs incoherent} × {backtracking (count≥1) vs no-backtracking} at the cell's peak Δgc magnitude.

| Arch | bs | peak mag | coh+bt | coh+no-bt | inc+bt | inc+no-bt | missing | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TXC-base | 256 | -12 | 22 | 17 | 17 | 5 | 0 | 61 |
| TXC-base | 1024 | -12 | 30 | 17 | 11 | 3 | 0 | 61 |
| TXC-pro | 256 | +12 | 28 | 28 | 3 | 2 | 0 | 61 |
| TXC-pro | 1024 | +16 | 10 | 1 | 44 | 6 | 0 | 61 |
| TopK SAE | 1024 | -16 | 26 | 20 | 11 | 3 | 1 | 61 |
| T-SAE | 1024 | +7 | 29 | 31 | 0 | 1 | 0 | 61 |
| MLC | 1024 | +16 | 28 | 24 | 6 | 3 | 0 | 61 |

---

_Regenerated from cell-level artifacts every time
`analyze_optimal.py` runs. Source per cell: see the
`workspace` paths logged in the eval_optimal_mag.py output._