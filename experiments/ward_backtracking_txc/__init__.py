"""Ward 2025 backtracking — Stage B (base-only TXC steering).

Phase 1: cache Llama-3.1-8B base activations at three hookpoints.
Phase 2: train one TemporalCrosscoder per hookpoint over a T-position window.
Phase 3: mine features that best separate D+ (backtracking) from D- on Stage A traces.
Phase 4: B1 — single-feature steering eval on DeepSeek-R1-Distill.
Phase 5: B2 — base-trained TXC encoder run on reasoning traces; per-offset firing diff.
"""
