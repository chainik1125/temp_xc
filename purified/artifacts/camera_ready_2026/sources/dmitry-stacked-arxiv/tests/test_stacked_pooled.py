"""Pre-registered gates for the stacked pooled adapters (reviewer-1 arm).

These encode the sprint's silent-invalidation checks:
- pooled encode equals the evaluator-side reduction (em.py's abs-amax)
  so mining and eval share one pooling convention;
- pooled feature width is exactly d_sae (the probing squeeze(1) bug
  would silently yield T*d_sae);
- realized L0 matches the per-position sparsity semantics;
- decoder rows are unit-norm (the no-sqrt(T)-rescale precondition);
- train_step accepts WindowBuffer-shaped (B, T, d_in) batches.
"""

from __future__ import annotations

import torch

from temp_bench.core.config import import_by_path, load_arch

D_IN = 32
B = 7


def _make(name: str):
    spec = load_arch(name, section="synthetic")  # d_sae=20 tier
    cls = import_by_path(spec.class_path)
    model = cls(d_in=D_IN, **spec.hparams)
    model.eval()
    return model


def test_stacked_sae_pooled_matches_abs_amax_reduction():
    model = _make("stacked_sae_pooled")
    x = torch.randn(B, model.T, D_IN)
    z_pos = model.encode_per_position(x)              # (B, T, d_sae)
    pooled = model.encode(x)                          # (B, d_sae)
    # em.py evaluator convention: z.abs() then amax over the T axis.
    expected = z_pos.abs().amax(dim=1)
    assert pooled.shape == (B, z_pos.shape[-1])
    # TopK codes are non-negative, so sign-preserving max-|act| must be
    # bit-identical to the evaluator's abs-amax.
    assert torch.equal(pooled, expected)


def test_stacked_btkonly_pooled_preserves_sign_of_max_abs():
    model = _make("stacked_btkonly_pooled")
    x = torch.randn(B, model.T, D_IN)
    z_pos = model.encode_per_position(x)
    pooled = model.encode(x)
    idx = z_pos.abs().argmax(dim=1, keepdim=True)
    expected = z_pos.gather(1, idx).squeeze(1)
    assert pooled.shape == (B, z_pos.shape[-1])
    assert torch.equal(pooled, expected)
    assert torch.equal(pooled.abs(), z_pos.abs().amax(dim=1))


def test_pooled_width_is_d_sae_not_T_times_d_sae():
    for name in ("stacked_sae_pooled", "stacked_btkonly_pooled"):
        model = _make(name)
        x = torch.randn(B, model.T, D_IN)
        assert model.encode(x).shape[-1] == model.config.d_sae


def test_stacked_sae_pooled_realized_l0():
    model = _make("stacked_sae_pooled")
    x = torch.randn(B, model.T, D_IN)
    z_pos = model.encode_per_position(x)
    # At most k_pos actives per position (TopK keeps k_pos slots; ReLU
    # may zero negatively-selected ones — the ≈k_pos gate on trained
    # models runs on-pod against the real eval cache). k_pos is per
    # position — no ·T rescale.
    per_pos_l0 = (z_pos != 0).sum(dim=-1).float()
    assert per_pos_l0.max() <= model.config.k_pos
    assert per_pos_l0.min() > 0
    # Pooled window code L0 is at most k_pos * T (union of supports).
    pooled_l0 = (model.encode(x) != 0).sum(dim=-1)
    assert pooled_l0.max() <= model.config.k_pos * model.T


def test_decoder_rows_unit_norm_no_sqrtT_rescale():
    for name in ("stacked_sae_pooled", "stacked_btkonly_pooled"):
        model = _make(name)
        dirs = model.decoder_directions()             # (d_sae, d_in), T-avg
        assert dirs.shape == (model.config.d_sae, D_IN)
        if name == "stacked_sae_pooled":
            # Per-position rows are exactly unit-norm at init.
            for sae in model.saes:
                norms = sae.W_dec.data.norm(dim=0)
                assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
        else:
            norms = model.W_dec.data.norm(dim=2)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_train_step_accepts_window_batches():
    for name in ("stacked_sae_pooled", "stacked_btkonly_pooled"):
        model = _make(name)
        model.train()
        x = torch.randn(B, model.T, D_IN)
        out = model.train_step(x)
        loss = out[0] if isinstance(out, tuple) else out["loss"]
        assert torch.isfinite(loss)
