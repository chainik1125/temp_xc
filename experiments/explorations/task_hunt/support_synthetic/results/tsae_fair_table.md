# Item 2 — T-SAE fairness receipt (mechanical output of analyze_tsae.py)

**VERDICT: FLAT**   untrained guard: PASS (exact)

| entry | mean λ̂ recovery | paired D vs Δ=1 | bar | call |
|---|---|---|---|---|
| tsae_d1 (Δ=1 ≡ registered) | 0.409 | — | — | anchor |
| tsae_d2 | 0.409 | +0.001 | 0.050 | flat |
| tsae_d4 | 0.399 | -0.010 | 0.050 | flat |
| tsae_d8 | 0.398 | -0.011 | 0.050 | flat |
| tsae_a0 (aux) | 0.399 | -0.010 | 0.050 | flat (aux) |

Per-token DPI floor ≈ 0.41 (bench band 0.38–0.44).
