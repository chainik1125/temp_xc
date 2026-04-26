## Phase 6.1 headline: triangle parity table

Autointerp at **N=32**, multi-judge (2 Haiku prompts), passage-coverage diagnostic (k/P).

### Triangle (3-seed where available)

| arch (family) | concat | n_seeds | /32 sem (mean ± se) | cov k/P (mean ± se) | cov entropy | judge disagree |
|---|---|---|---|---|---|---|
| TXC (Cycle F) | A | 3 | 21.7/32 ± 0.88 | 3.0/3 ± 0.00 | 0.91 ± 0.04 | 0.04 |
| TXC (Cycle F) | B | 3 | 16.0/32 ± 2.52 | 4.0/4 ± 0.00 | 1.29 ± 0.04 | 0.08 |
| TXC (Cycle F) | random | 3 | 0.0/32 ± 0.00 | 6.7/7 ± 0.33 | 1.69 ± 0.03 | 0.18 |
| TXC (2x2 cell) | A | 3 | 20.7/32 ± 1.45 | 3.0/3 ± 0.00 | 0.94 ± 0.08 | 0.02 |
| TXC (2x2 cell) | B | 3 | 15.0/32 ± 1.53 | 4.0/4 ± 0.00 | 1.33 ± 0.01 | 0.03 |
| TXC (2x2 cell) | random | 3 | 1.7/32 ± 0.33 | 6.3/7 ± 0.67 | 1.62 ± 0.10 | 0.14 |
| T-SAE baseline | A | 3 | 23.0/32 ± 1.15 | 3.0/3 ± 0.00 | 0.94 ± 0.02 | 0.03 |
| T-SAE baseline | B | 3 | 17.7/32 ± 0.88 | 4.0/4 ± 0.00 | 1.27 ± 0.03 | 0.04 |
| T-SAE baseline | random | 3 | 13.7/32 ± 1.33 | 7.0/7 ± 0.00 | 1.86 ± 0.01 | 0.09 |
| TFA baseline | A | 1 | 14.0/32 ± 0.00 | 3.0/3 ± 0.00 | 1.07 ± 0.00 | 0.03 |
| TFA baseline | B | 1 | 12.0/32 ± 0.00 | 4.0/4 ± 0.00 | 1.24 ± 0.00 | 0.00 |
| TFA baseline | random | 1 | 1.0/32 ± 0.00 | 2.0/7 ± 0.00 | 0.14 ± 0.00 | 0.00 |

### Full 9-arch matrix at seed=42

| arch | concat | /32 sem | cov k/P | cov ent | disagree |
|---|---|---|---|---|---|
| agentic_txc_02_batchtopk | A | 21.7/32 | 3.0/3 | 0.91 | 0.04 |
| agentic_txc_02_batchtopk | B | 16.0/32 | 4.0/4 | 1.29 | 0.08 |
| agentic_txc_02_batchtopk | random | 0.0/32 | 6.7/7 | 1.69 | 0.18 |
| agentic_txc_12_bare_batchtopk | A | 20.7/32 | 3.0/3 | 0.94 | 0.02 |
| agentic_txc_12_bare_batchtopk | B | 15.0/32 | 4.0/4 | 1.33 | 0.03 |
| agentic_txc_12_bare_batchtopk | random | 1.7/32 | 6.3/7 | 1.62 | 0.14 |
| tsae_paper | A | 23.0/32 | 3.0/3 | 0.94 | 0.03 |
| tsae_paper | B | 17.7/32 | 4.0/4 | 1.27 | 0.04 |
| tsae_paper | random | 13.7/32 | 7.0/7 | 1.86 | 0.09 |
| tfa_big | A | 14/32 | 3/3 | 1.07 | 0.03 |
| tfa_big | B | 12/32 | 4/4 | 1.24 | 0.00 |
| tfa_big | random | 1/32 | 2/7 | 0.14 | 0.00 |
| agentic_txc_10_bare | A | 21.0/32 | 3.0/3 | 0.98 | 0.02 |
| agentic_txc_10_bare | B | 17.7/32 | 4.0/4 | 1.32 | 0.05 |
| agentic_txc_10_bare | random | 3.0/32 | 6.7/7 | 1.66 | 0.23 |
| agentic_mlc_08 | A | 18/32 | 3/3 | 0.98 | 0.03 |
| agentic_mlc_08 | B | 18/32 | 4/4 | 1.34 | 0.00 |
| agentic_mlc_08 | random | 2/32 | 4/7 | 0.94 | 0.19 |
| agentic_txc_11_stack | A | 21/32 | 3/3 | 0.77 | 0.09 |
| agentic_txc_11_stack | B | 12/32 | 4/4 | 1.13 | 0.00 |
| agentic_txc_11_stack | random | 0/32 | 7/7 | 1.87 | 0.22 |
| agentic_txc_09_auxk | A | 21/32 | 3/3 | 0.91 | 0.06 |
| agentic_txc_09_auxk | B | 13/32 | 4/4 | 1.28 | 0.03 |
| agentic_txc_09_auxk | random | 0/32 | 7/7 | 1.68 | 0.25 |
| agentic_txc_02 | A | 17/32 | 3/3 | 0.95 | 0.03 |
| agentic_txc_02 | B | 16/32 | 4/4 | 1.33 | 0.03 |
| agentic_txc_02 | random | 0/32 | 7/7 | 1.76 | 0.22 |
| tsae_ours | A | 17/32 | 3/3 | 1.06 | 0.03 |
| tsae_ours | B | 19/32 | 4/4 | 1.36 | 0.09 |
| tsae_ours | random | 3/32 | 6/7 | 1.35 | 0.12 |
