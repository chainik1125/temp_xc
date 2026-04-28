---
author: Dmitry
date: 2026-04-24
tags:
  - results
  - in-progress
---

## C2 pure-contrastive λ_c sweep (5k steps)

Spec §14 pure-contrastive condition: TXCBareMatryoshkaContrastiveAntidead with matryoshka off, shift=1 InfoNCE pairs, contr_prefix = d_sae/5, han anti-dead stack on, **no Bricken resample**. d_sae=32k, T=5, k=128.

| λ_c | dead @ 5k | loss @ 5k |
|---:|---:|---:|
| 0.1 | 78.9% | 5976 |
| 0.3 | 75.0% | 5336 |
| 1.0 | 77.6% | 5083 |

For comparison (same horizon):

| recipe | dead @ 5k | loss @ 5k |
|---|---:|---:|
| han_antidead + Bricken (best) | ~47% | ~2150 |
| brickenauxk α=1/8 | ~50% | 2151 |
| resample_cap_0_5 baseline | 50.7% | 4217 |
| pure-contrastive (any λ_c) | 75-79% | 5-6k |

### Headline

Pure-contrastive without Bricken loses on both dead-fraction and reconstruction loss. Higher λ_c slightly reduces final loss (5976 → 5083) but does not rescue dead features (78.9% → 77.6%). Anti-dead stack alone (geom-median b_dec, decoder grad-projection, decoder renorm) is insufficient at this horizon.
