# tab: sycgen shuffle/T-sweep (item 6) — PARTIAL 15/18, PENDING TEAM REVIEW

_Generated from `sycgen/results/{sycgen_shuffle_overlay,sycgen_twin_overlay,sycgen_tsweep_summary}.json` at render 03:57 28-07; n=3 seeds per cell; arch `txc_batchtopk_post_btkonly` (btk-only arm, either-arm rule); T-SAE anchor trio pending (regrind), joins the final render._

| T | ordered r (mean ± sd) | shuffled r (mean ± sd) | gap (ordered−shuf) | untrained-twin gap |
|---|---|---|---|---|
| 2 | 0.4982 ± 0.0174 | 0.4648 ± 0.0429 | +0.0334 | +0.1531 |
| 4 | 0.5243 ± 0.0120 | 0.5016 ± 0.0073 | +0.0227 | +0.1664 |
| 8 | 0.5413 ± 0.0083 | 0.5013 ± 0.0325 | +0.0399 | +0.1015 |
| 16 | 0.5922 ± 0.0074 | 0.5296 ± 0.0268 | +0.0626 | +0.0958 |
| 1 (anchor) | per-token BatchTopK SAE: 0.4819 ± 0.0101 | ≡ ordered (identity by construction) | — | — |

**Quote-form (binding, LOG 04:16):** shuffle costs 0.02–0.06 recovery
(positive 12/12 trained cells), but untrained twins show LARGER gaps at
every T — the gap is architectural position-sensitivity, which training
REDUCES while lifting recovery ≤0.22 → 0.50–0.59; not learned order-use.
The claim is the LEVEL story. l0 NOT budget-matched (TXC 0.49–2.85
l0/token vs SAE ~4.5 — sparser and above the anchor; flag travels).
Untrained twin levels: 0.075–0.218 (see fig).
