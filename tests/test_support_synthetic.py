"""Contract tests for the support_synthetic build (CARD § 2.1).

The load-bearing guarantee: ``TSAEDelta(pair_delta=1)`` is the registered
T-SAE's exact computation — same RNG stream, same losses, same parameter
trajectory — so the Δ sweep's Δ=1 anchor IS the baseline, and any Δ effect is
the pair distance and nothing else. Committed with the build, green before any
grid (strict commit-then-run).
"""

from __future__ import annotations

import pytest
import torch

from temp_bench.archs.tsae import TSAEPaper
from temp_bench.archs.tsae_delta import TSAEDelta
from temp_bench.core.config import load_arch

D_IN, D_SAE, K, B, SEQ = 12, 20, 2, 8, 16
KW = dict(d_in=D_IN, d_sae=D_SAE, k_pos=K)


def _run_steps(model: torch.nn.Module, n_steps: int, data_seed: int = 7):
    """Train ``model`` for ``n_steps`` on a deterministic stream; return losses.

    The pair-offset randint draws from the global torch RNG, so the caller
    must seed once per model run for cross-model parity.
    """
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(data_seed)
    losses = []
    for _ in range(n_steps):
        x = torch.randn(B, SEQ, D_IN, generator=gen)
        loss, _info = model.train_step(x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.post_step()
        losses.append(loss.item())
    return losses


def test_delta1_bitwise_parity_with_registered_tsae():
    torch.manual_seed(0)
    ref = TSAEPaper(**KW)
    torch.manual_seed(0)
    var = TSAEDelta(pair_delta=1, **KW)

    for p_ref, p_var in zip(ref.parameters(), var.parameters()):
        assert torch.equal(p_ref, p_var), "init differs"

    torch.manual_seed(1)                       # governs the pair-offset draws
    l_ref = _run_steps(ref, 6)
    torch.manual_seed(1)
    l_var = _run_steps(var, 6)
    assert l_ref == l_var, f"loss streams diverge: {l_ref} vs {l_var}"
    for p_ref, p_var in zip(ref.parameters(), var.parameters()):
        assert torch.equal(p_ref, p_var), "param trajectories diverge"
    assert torch.equal(ref.threshold, var.threshold)


@pytest.mark.parametrize("delta", [2, 4, 8])
def test_pair_identity_and_offset_bounds(delta):
    torch.manual_seed(0)
    model = TSAEDelta(pair_delta=delta, **KW)

    seen = []
    orig = model._encode_per_token

    def spy(x):
        seen.append(x.clone())
        return orig(x)

    model._encode_per_token = spy

    # seq_len = delta + 1 forces t_offset = 0 → pair must be (x[:,0], x[:,delta]).
    x = torch.randn(B, delta + 1, D_IN)
    model.train_step(x)
    assert torch.equal(seen[0], x[:, 0, :])
    assert torch.equal(seen[1], x[:, delta, :])

    # Bounds: over many draws, offsets stay in [0, SEQ - delta).
    offsets = {model._pair_offset(SEQ) for _ in range(300)}
    assert min(offsets) >= 0 and max(offsets) < SEQ - delta
    assert len(offsets) > 1, "offset never varies — RNG not consulted"


def test_seq_too_short_and_t_rejected():
    torch.manual_seed(0)
    model = TSAEDelta(pair_delta=4, **KW)
    with pytest.raises(ValueError, match="pair_delta=4"):
        model.train_step(torch.randn(B, 4, D_IN))     # needs seq_len >= 5
    with pytest.raises(ValueError, match="per-token"):
        TSAEDelta(pair_delta=1, T=2, **KW)
    with pytest.raises(ValueError, match="pair_delta"):
        TSAEDelta(pair_delta=0, **KW)


def test_registry_entries():
    for name, delta in (("tsae_d1", 1), ("tsae_d2", 2), ("tsae_d4", 4), ("tsae_d8", 8)):
        spec = load_arch(name, section="synthetic")
        assert spec.class_path.endswith("TSAEDelta")
        assert spec.hparams["pair_delta"] == delta
        assert spec.hparams["d_sae"] == 20          # per-section synthetic default
    a0 = load_arch("tsae_a0", section="synthetic")
    assert a0.class_path.endswith("TSAEPaper")      # registered class, no fork
    assert a0.hparams["contrastive_alpha"] == 0.0
    assert a0.hparams["d_sae"] == 20
