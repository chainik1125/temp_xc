---
author: Dmitry
date: 2026-04-28
tags:
  - results
  - complete
---

## Methodology — paper-clamp vs AxBench-additive

Two intervention modalities applied at L12 of Gemma-2-2b base, both targeting the same SAE feature picked by lift-score for each concept (best feature per concept under a fixed 30-concept × 5-example probe set).

### Paper protocol (Ye et al. 2025, App B.2): clamp-on-latent with error preserve

For each token's L12 residual stream activation `x`:

```text
z          = encoder(x)              # (d_sae,) — original SAE latents
x_hat      = decoder(z)              # (d_in,)  — original reconstruction
e          = x − x_hat               # (d_in,)  — residual the SAE can't express

z_clamped  = z; z_clamped[j] = strength       # force feature j to absolute clamp value
x_hat_steer = decoder(z_clamped)              # steered reconstruction
x_steered   = x_hat_steer + e                  # add the SAE error back
```

**Strengths**: `{10, 100, 150, 500, 1000, 1500, 5000, 10000, 15000}` — absolute clamp values, not multipliers.

The "error preserve" piece is critical: it ensures the intervention only modifies what the SAE actually represents. Everything outside the SAE dictionary (residual `e`) passes through unchanged. At `strength = z[j]_orig` the intervention is a literal no-op (`x_steered = x` exactly), so any observed effect is *purely* attributable to feature `j`.

Algebraically the net injected direction at each token is:

```text
x_steered − x = (strength − z[j]_orig) · W_dec[:, j]
```

where `W_dec[:, j]` is the raw (non-unit-norm) decoder direction. This is a *per-token-varying* additive offset because `z[j]_orig` depends on the token's own activation.

### AxBench-additive (Han's protocol, Wu et al. 2025-style): unit-decoder injection

```text
x_steered = x + strength · unit_norm(W_dec[:, j])
```

**Strengths in this study**: `{-100, -50, -25, -10, 0, 10, 25, 50, 100}` — signed multipliers of a unit-norm decoder direction. (Han's original synthesis used `{0.5, 1, 2, 4, 8, 12, 16, 24}`; we extended to ±100 so the magnitude regime is comparable to paper-clamp's 10–15000 absolute values.)

No encode/decode round trip. No error-preserve term. The injected direction is the *same per token* (no dependence on the token's own activation), and the decoder direction is *unit-normalized* — so activation-magnitude differences across architectural families wash out.

### Window-arch generalization (this work)

Window-encoder archs (TXC family, SubseqH8) consume `(B, T, d_in)` — a window of T residuals, not a single token. Paper-clamp doesn't have a canonical generalization. We define one:

At each token's residual `h_t` during decode, form the T-token window `W_t = h[t-T+1 : t+1]`:

```text
z          = encoder(W_t)                      # (d_sae,)
x_hat_W    = decoder(z)                        # (T, d_in) — full window recon
x_hat_R    = x_hat_W[-1, :]                    # (d_in,)   — right-edge token
z_clamped  = z; z_clamped[j] = strength
x_hat_W'   = decoder(z_clamped)
x_hat_R'   = x_hat_W'[-1, :]
h_t_steered = x_hat_R' + (h_t − x_hat_R)        # error-preserve at right edge only
```

For positions `t < T-1` (no full window available), `h_t` passes through unchanged.

**Implementation requirement**: HF KV cache must be disabled (`use_cache=False`) so the hook always sees the full `(B, S, d_in)` sequence. With KV cache, each generation step would only emit the new token's residual — insufficient to form a T-window.

For matryoshka archs (`MatryoshkaTXCDRContrastiveMultiscale`), `decode(z)` doesn't exist — use `decode_scale(z, T-1)` (the largest-scale decoder, which produces the full window). Wrapper: `_decode_full_window`.

### Why these protocols disagree across architectures

The same SAE feature has different `z[j]_orig` magnitudes across families:

- **Per-token archs**: encoder takes `(d_in,)`, output is post-TopK-ReLU. Typical active `z[j]` values are `O(1-10)`.
- **Window archs**: encoder takes `(T, d_in)` and integrates over `T` tokens. Typical active `z[j]` values are `O(T × per-token magnitude)` ≈ 5–10× larger.

Under **paper-clamp** the strength is set to an *absolute* value. A clamp of 100 is "10×typical" for per-token archs but only "2×typical" for window archs — a relatively smaller push. So the *peak operating point* (where steering kicks in without coherence collapse) shifts to higher absolute strengths for window archs. We see exactly this 5× shift empirically (window archs peak at s=500, per-token at s=100).

Under **AxBench-additive** the strength multiplies a *unit-norm* decoder direction. Activation magnitudes don't enter the steering equation at all — every arch is on the same scale. All archs peak at the same strength (s=100 in our extended sweep).

This is why the architectural ranking depends on protocol.

### Why paper authors chose clamp-on-latent

The paper's choice makes sense in their context: they only compared per-token archs (TopKSAE + T-SAE), so no cross-family scale issue. And clamp-on-latent is the cleanest "I am modifying feature j to value x and nothing else" operation when error-preserve is applied — it isolates the feature's effect from reconstruction artefacts.

### Why Han chose AxBench-additive

Han's brief (Agent C) needed to compare 6 architecturally-distinct families (per-token, window-T=5, window-T=10, layer-multi MLC). Clamp-on-latent isn't well-defined for window or MLC archs without picking a generalization (multiple candidate generalizations exist; this work implements one for windows). AxBench-additive applies uniformly and fairly across families — at the cost of not being the paper's own protocol.

### Sanity checks done

- **Smoke test (3 concepts)** on per-token paper-clamp before scaling — confirmed paper-style behavior (low strength = coherent / no steering, mid strength = sweet spot, high strength = repetition collapse).
- **Smoke test on window paper-clamp** for `agentic_txc_02` (T=5) — confirmed decoder shape correct, no NaN, semantically-relevant outputs at moderate strengths.
- **Per-arch grader-call counts**: 270 rows × 2 calls × 6 archs × 2 protocols = **6480 Sonnet 4.6 calls total** — all returned valid 0–3 grades, no parse failures.
