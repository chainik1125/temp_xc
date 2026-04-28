---
author: Aniket Deshpande
date: 2026-04-27
tags:
  - results
  - venhoff-eval
  - in-progress
---

> **TEMPORARY — single-cell results only.** Will be replaced by full
> (coef × window) grid sweep once it finishes (~2.5 h ETA at time of
> writing). Until then this serves as the first valid 3-arch comparison
> with the arch-keyed steering-vector fix in place.

## Single-cell 3-arch comparison (2026-04-27)

First clean run after the arch-keyed steering-vector path fix (commit
`e7f3918` + later resume-meta-hash fix). All three arches genuinely
trained their own Phase-2 steering vectors (TempXC: `_idx*_tempxc.pt`,
MLC: `_idx*_mlc.pt`, SAE: Venhoff's shipped bare `.pt`); no on-disk
collisions; Phase 3 ran each arch on its own pinned GPU.

### Setup

| Knob | Value |
|---|---|
| Cell | (coef=0.5, token_window=0) — single point |
| n_tasks | 20 |
| max_new_tokens | 2000 |
| max_thinking_tokens | 2000 |
| Phase 2 budget | paper App C.1 (max_iters=50, n_train=2048) |
| GPU pinning | CUDA_VISIBLE_DEVICES=0 per arch (serial across arches) |

### Results

| arch | thinking_acc | base_acc | hybrid_acc | Gap Recovery |
|---|---|---|---|---|
| **TempXC** | 80.0% | 30.0% | 20.0% | **−20.0%** |
| SAE | 75.0% | 35.0% | 25.0% | −25.0% |
| MLC | 75.0% | 35.0% | 20.0% | −37.5% |

### Read

- **All three arches' hybrid runs hurt vs base** at this single
  config. Hybrid generation actively misdirects the base model on the
  20-task slice with these coefs/windows.
- **Arch ordering matches the thesis**: TempXC < SAE < MLC in absolute
  damage (TempXC's vectors are the least disruptive). That's the
  directional claim the paper wants — just at a suboptimal cell.
- **Cannot compare to Venhoff's 3.5% headline yet.** Their number is
  the max over a 10×5 = 50-cell (coef × window) grid. We only sampled
  one cell per arch. Apples-to-oranges. The full-grid sweep currently
  running fixes this; once it lands, the headline number per arch is
  `max_over_cells(Gap_Recovery)`.

### Why hybrid might be hurting at this cell

- **coef=0.5 may be too aggressive** at every-token (`window=0`)
  application: the base model is being pushed too far from its own
  distribution at every token, breaking step-by-step reasoning.
- **window=0 means steering applied to all tokens**, including
  arithmetic / rote computation tokens where the thinking model's
  reasoning style isn't useful and may be actively harmful.
- The full sweep includes `window=-15, -50` (apply only to last
  N tokens of each sentence) which is what Venhoff's paper found to
  be the sweet spot for this cell.

### Caveats on the n=20 numbers

- **Paired Δ noise floor at n=20 is ~±15-20 pp.** The TempXC vs MLC
  gap of 17.5 pp here is at the edge of statistical detectability.
  TempXC vs SAE (5 pp gap) is well within noise.
- **Single-cell results don't isolate "does this arch produce useful
  steering vectors"** — they only say "is this *one* cell's
  application useful." Need the grid to claim the former.

### Next

Running 5 coefs × 3 windows = 15 cells per arch on the same n=20
slice. ~50 min per arch × 3 arches serial = ~2.5 h. Then this
section gets replaced with grid-sweep results: `(arch, best_cell,
best_GR)` per arch + comparison to Venhoff's 3.5%.

If after the sweep TempXC's best cell still has GR < SAE's best, we
have a negative result for the paper's thesis on this cell — which is
itself reportable but weaker. If TempXC's best > SAE's best > 3.5%,
we have the positive headline.
