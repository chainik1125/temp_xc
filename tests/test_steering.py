"""Tests for ``temp_bench.case_studies.steering``.

These tests exercise the pure-tensor + pure-python code paths only:

- V7 tiled-broadcast hook algebra on a tiny TopKSAE
- PP per-position hook overlap-averaging
- :func:`select_best_features` on a tiny synthetic concept set
- :func:`coh_success_curves` + :func:`flatten_metrics` on synthetic grades

The full :class:`SteeringCaseStudy.evaluate` path requires the
Gemma-2-2b-IT subject model + the Anthropic API key + a real arch with
a trained checkpoint, so it's covered separately by the cell-level
integration runs in ``experiments/c5_steering/run.py`` (which the runner
caches via leaderboard.jsonl).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from temp_bench.architectures.base import ArchConfig, TempBenchArch
from temp_bench.architectures.topk_sae import TopKSAE
from temp_bench.case_studies.steering import (
    CONCEPTS,
    DEFAULT_COH_THRESHOLDS,
    Generation,
    Grade,
    SteeringConfig,
    _build_pp_hook,
    _build_v7_hook,
    coh_success_curves,
    flatten_metrics,
    get_concept,
    select_best_features,
)


# ── Concept set ────────────────────────────────────────────────────────


def test_concepts_count_and_shape():
    assert len(CONCEPTS) == 30
    for c in CONCEPTS:
        assert set(c.keys()) >= {"id", "description", "examples"}
        assert isinstance(c["id"], str)
        assert isinstance(c["description"], str)
        assert isinstance(c["examples"], list) and len(c["examples"]) == 5
        for ex in c["examples"]:
            assert isinstance(ex, str) and len(ex) >= 20


def test_concepts_unique_ids():
    ids = [c["id"] for c in CONCEPTS]
    assert len(set(ids)) == 30


def test_get_concept_roundtrip():
    c = get_concept("medical")
    assert c["id"] == "medical"
    with pytest.raises(KeyError):
        get_concept("not_a_real_concept")


# ── SteeringConfig ─────────────────────────────────────────────────────


def test_steering_config_defaults():
    cfg = SteeringConfig()
    assert cfg.protocol == "v7"
    assert cfg.coh_thresholds == DEFAULT_COH_THRESHOLDS
    assert cfg.n_concepts == 30


def test_steering_config_validates_protocol():
    with pytest.raises(ValueError):
        SteeringConfig(protocol="other")


def test_steering_config_validates_n_concepts():
    with pytest.raises(ValueError):
        SteeringConfig(n_concepts=0)
    with pytest.raises(ValueError):
        SteeringConfig(n_concepts=31)


# ── V7 hook algebra ────────────────────────────────────────────────────


class _LinearWindowArch(TempBenchArch):
    """Minimal window arch: identity-recon, additive-feature decoder.

    ``encode((B, T, d)) -> (B, T, d_sae)`` is a fixed linear map;
    ``decode((B, T, d_sae)) -> (B, T, d)`` adds per-feature directions
    back. Used to exercise the V7 / PP hook algebra without needing a
    fully trained SAE.
    """
    def __init__(self, *, d_in: int, d_sae: int, T: int):
        torch.nn.Module.__init__(self)
        self.config = ArchConfig(
            name="_linear_window", d_in=d_in, d_sae=d_sae, k_pos=1, T=T,
        )
        self._d_sae = d_sae
        self.W_dec = torch.nn.Parameter(
            torch.eye(d_in, d_sae) if d_in <= d_sae
            else torch.cat([torch.eye(d_sae), torch.zeros(d_in - d_sae, d_sae)], dim=0),
            requires_grad=False,
        )
        # encode = W_dec.T applied per-position, then sum over T gives 0 baseline
        self.W_enc = torch.nn.Parameter(self.W_dec.t().clone(), requires_grad=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, d) -> (B, T, d_sae) per-position, then mean-pool to (B, d_sae)
        z_pos = x @ self.W_enc.t()
        return z_pos.mean(dim=1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # broadcast (B, d_sae) -> (B, T, d) by repeating across T
        T = self.config.T
        d_recon = (z @ self.W_dec.t())
        return d_recon.unsqueeze(1).repeat(1, T, 1)


def test_v7_hook_block_layout_and_broadcast_clean():
    """V7 should write the SAME delta vector to every position within a
    block when ``S`` is a multiple of ``T`` (no trailing-block overwrites)."""
    torch.manual_seed(0)
    d_in, d_sae, T = 8, 8, 4
    arch = _LinearWindowArch(d_in=d_in, d_sae=d_sae, T=T)

    Bh, S = 1, 12                                     # S = 3T, no remainder
    h = torch.randn(Bh, S, d_in)
    state = {"feature_idx": 3}
    hook = _build_v7_hook(arch, T=T, strengths_t=torch.tensor([100.0]), state=state)
    out = hook(None, None, h)
    assert out.shape == h.shape

    delta = out - h
    for s in (0, 4, 8):                               # the 3 clean T-blocks
        block_delta = delta[:, s:s + T, :]
        spread = (block_delta.amax(dim=1) - block_delta.amin(dim=1)).abs()
        assert torch.allclose(spread, torch.zeros_like(spread), atol=1e-5), (
            f"V7 block at s={s} not uniform across T positions"
        )


def test_v7_hook_trailing_block_overwrites_uniform_within_effective_spans():
    """When ``S % T != 0``, the trailing T-window aligned at ``S-T``
    overwrites the tail. The effective per-position delta layout is:

        [0..T-1]:        block 0's delta
        [T..S-T-1]:      block 1's delta (truncated by trailing overwrite)
        [S-T..S-1]:      trailing block's delta (final write wins)
    """
    torch.manual_seed(1)
    d_in, d_sae, T = 8, 8, 4
    arch = _LinearWindowArch(d_in=d_in, d_sae=d_sae, T=T)
    Bh, S = 1, 10                                     # block_starts = [0, 4, 6]
    h = torch.randn(Bh, S, d_in)
    state = {"feature_idx": 3}
    hook = _build_v7_hook(arch, T=T, strengths_t=torch.tensor([100.0]), state=state)
    out = hook(None, None, h)
    delta = out - h

    # positions 0..3, 4..5, and 6..9 should each be uniform (after the
    # trailing block at s=6 overwrites positions 6..7 from the middle block).
    for span in ([0, 4], [4, 6], [6, 10]):
        seg = delta[:, span[0]:span[1], :]
        if seg.shape[1] >= 2:
            spread = (seg.amax(dim=1) - seg.amin(dim=1)).abs()
            assert torch.allclose(spread, torch.zeros_like(spread), atol=1e-5), (
                f"V7 effective-span {span} not uniform"
            )


def test_v7_hook_no_op_when_feature_unset():
    arch = _LinearWindowArch(d_in=8, d_sae=8, T=4)
    h = torch.randn(1, 10, 8)
    state = {"feature_idx": None}
    hook = _build_v7_hook(arch, T=4, strengths_t=torch.tensor([1.0]), state=state)
    assert hook(None, None, h) is None


def test_v7_hook_no_op_when_S_lt_T():
    arch = _LinearWindowArch(d_in=8, d_sae=8, T=4)
    h = torch.randn(1, 3, 8)                          # S < T
    state = {"feature_idx": 0}
    hook = _build_v7_hook(arch, T=4, strengths_t=torch.tensor([1.0]), state=state)
    assert hook(None, None, h) is None


# ── PP hook overlap-averaging ─────────────────────────────────────────


def test_pp_hook_overlap_average_count_correct():
    """Position t in the middle of a long sequence belongs to exactly T
    sliding windows, so the per-position delta should be the average of
    T per-window contributions. Boundary positions (< T-1 or > S-T) get
    fewer overlapping windows."""
    arch = _LinearWindowArch(d_in=8, d_sae=8, T=4)
    Bh, S, d_in = 1, 10, 8
    h = torch.randn(Bh, S, d_in)
    state = {"feature_idx": 0}
    hook = _build_pp_hook(arch, T=4, strengths_t=torch.tensor([1.0]), state=state)
    out = hook(None, None, h)
    assert out.shape == h.shape
    # Position 4 is well-inside; position 0 is at the boundary. They
    # shouldn't be identical (different number of contributing windows).
    delta = out - h
    interior = delta[0, 4]
    boundary = delta[0, 0]
    assert not torch.allclose(interior, boundary, atol=1e-3)


# ── Feature selection ─────────────────────────────────────────────────


class _FixedZSAE(TempBenchArch):
    """Per-token SAE whose encode returns hand-set per-feature
    activations. Used to test that :func:`select_best_features` picks
    the deterministically best feature for each concept."""
    def __init__(self, d_in: int, d_sae: int, fixed_z_per_pos: torch.Tensor):
        """``fixed_z_per_pos`` shape ``(d_sae,)`` — same value at every
        position. The argmax must match the highest entry."""
        torch.nn.Module.__init__(self)
        self.config = ArchConfig(name="_fixed_z", d_in=d_in, d_sae=d_sae, k_pos=1, T=1)
        self._d_sae = d_sae
        self._fixed_z = fixed_z_per_pos                 # (d_sae,)
        # buffer registration so `.to(device)` propagates
        self.register_buffer("_zb", self._fixed_z.clone())
        self.W_dec = torch.nn.Parameter(torch.eye(d_in, d_sae)[:d_in, :d_sae])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d) or (B, d)
        if x.dim() == 3:
            B, T, d = x.shape
            return self._zb.to(x.device).expand(B, T, self._d_sae)
        else:
            B, d = x.shape
            return self._zb.to(x.device).expand(B, self._d_sae)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 3:
            return torch.zeros(z.shape[:2] + (self.W_dec.shape[0],), device=z.device)
        return torch.zeros((z.shape[0], self.W_dec.shape[0]), device=z.device)


# ── Aggregation ───────────────────────────────────────────────────────


def _g(idx, cid, strength, succ, coh, error=""):
    return Grade(
        idx=idx, concept_id=cid, feature_idx=0, strength=strength,
        success_grade=succ, coherence_grade=coh, error=error,
    )


def test_coh_success_curves_basic():
    grades = [
        _g(0, "a", 100.0, 3, 3),                       # high coh + success
        _g(1, "a", 100.0, 0, 0),                       # incoherent + fail
        _g(2, "b", 1000.0, 2, 2),                      # at threshold both
        _g(3, "b", 1000.0, 1, 1),                      # below both
    ]
    out = coh_success_curves(grades, coh_thresholds=(1.5, 2.0, 2.5))
    assert out["n_total"] == 4
    assert out["n_valid"] == 4
    # At τ=2.0, only items with coh >= 2 contribute: {0, 2}. Of those,
    # success ≥ 2 holds for {0, 2} both → success rate 1.0.
    assert out["success_at_coh"]["2"] == pytest.approx(1.0)
    # At τ=2.5, only item 0 (coh=3) qualifies; success=3≥2 → 1.0.
    assert out["success_at_coh"]["2.5"] == pytest.approx(1.0)
    # At τ=1.5, items {0, 2} qualify (coh ≥ 1.5 from {3, 2}); both succeed.
    assert out["success_at_coh"]["1.5"] == pytest.approx(1.0)


def test_coh_success_curves_no_valid_returns_zeros():
    bad = [_g(i, "a", 1.0, None, None, error="oops") for i in range(3)]
    out = coh_success_curves(bad)
    assert out["n_valid"] == 0
    assert out["n_total"] == 3
    for v in out["success_at_coh"].values():
        assert v == 0.0


def test_coh_success_curves_per_strength_breakdown():
    grades = [
        _g(0, "a", 100.0, 3, 3),
        _g(1, "a", 1000.0, 1, 3),                      # coherent but fail
    ]
    out = coh_success_curves(grades, coh_thresholds=(1.75, 2.0))
    per = out["success_at_coh_per_strength"]
    assert "100" in per and "1000" in per
    assert per["100"]["2"] == pytest.approx(1.0)        # 100 succeeds
    assert per["1000"]["2"] == pytest.approx(0.0)       # 1000 fails (succ=1)


def test_flatten_metrics_shape():
    curves = coh_success_curves(
        [_g(0, "a", 100.0, 3, 3), _g(1, "a", 100.0, 1, 1)],
        coh_thresholds=DEFAULT_COH_THRESHOLDS,
    )
    flat = flatten_metrics(curves)
    assert isinstance(flat, dict)
    for k, v in flat.items():
        assert isinstance(v, float), f"non-float metric {k}={v!r}"
    # All 5 thresholds should produce a key.
    for tau in DEFAULT_COH_THRESHOLDS:
        assert f"success_at_coh_{tau:g}" in flat
    assert "mean_coh" in flat and "mean_success" in flat
    assert flat["n_total"] == 2.0


# ── Generation dataclass ──────────────────────────────────────────────


def test_generation_to_json_phase7_compat():
    g = Generation(
        idx=0, arch_name="topk_sae", seed=42, concept_id="medical",
        feature_idx=123, strength=100.0, prompt="We find",
        generated_text="The patient...", protocol="v7", T=5,
    )
    j = g.to_json()
    # Phase 7 schema uses 'arch_id'; ours adds 'arch_name'. Both must
    # be present on disk so phase7 ad-hoc plot scripts can ingest.
    assert j["arch_id"] == "topk_sae"
    assert j["intervention"] == "paper_clamp_window_v7"


# ── feature selection: single-batch sanity ────────────────────────────


def test_select_best_features_concept_lift():
    """Concept-lift selection (NOT raw-activation argmax): a feature
    that fires strongly across ALL concepts (always-on) gets ~zero
    lift; features that fire SELECTIVELY on one concept get high lift.

    Setup: 2 concepts × 5 sentences each. We monkey-patch
    ``_encode_per_position`` so that:
      - Feature 0 fires at activation 100 on every concept (always-on).
      - Feature 7 fires at activation 50 only on concept 0.
      - Feature 13 fires at activation 50 only on concept 1.

    Raw-activation argmax would pick feature 0 for both concepts.
    Concept-lift correctly picks 7 for concept 0 and 13 for concept 1.
    """
    d_in, d_sae = 16, 32
    arch = _FixedZSAE(d_in, d_sae, torch.zeros(d_sae))    # body unused
    n_per = 5
    n_concepts = 2
    S = 8

    from temp_bench.case_studies import steering as st

    captured_acts = torch.zeros(n_concepts * n_per, S, d_in)
    captured_attn = torch.ones(n_concepts * n_per, S, dtype=torch.int8)

    # Build the (N, S, d_sae) z tensor per the encode contract.
    z = torch.zeros(n_concepts * n_per, S, d_sae)
    z[:, :, 0] = 100.0                                    # always-on
    z[:n_per, :, 7] = 50.0                                # concept 0
    z[n_per:, :, 13] = 50.0                               # concept 1

    def _stub_capture(*args, **kwargs):
        return captured_acts, captured_attn

    def _stub_encode_per_position(arch, acts, T):         # noqa: ARG001
        return z

    orig_capture = st._capture_anchor_layer_acts
    orig_encode = st._encode_per_position
    st._capture_anchor_layer_acts = _stub_capture
    st._encode_per_position = _stub_encode_per_position
    try:
        sel = select_best_features(
            arch.cpu(),
            arch_name="_fixed_z",
            subject_model=None,
            tokenizer=None,
            device=torch.device("cpu"),
            concepts=CONCEPTS[:n_concepts],
            top_k=3,
        )
    finally:
        st._capture_anchor_layer_acts = orig_capture
        st._encode_per_position = orig_encode

    cid0, cid1 = CONCEPTS[0]["id"], CONCEPTS[1]["id"]
    assert sel.best_idx[cid0] == 7, f"expected 7, got {sel.best_idx[cid0]}"
    assert sel.best_idx[cid1] == 13, f"expected 13, got {sel.best_idx[cid1]}"
    # Always-on feature 0 must NOT be top-1 for either concept.
    assert sel.top_k[cid0][0][0] != 0
    assert sel.top_k[cid1][0][0] != 0
