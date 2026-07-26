"""Sparsity rules for TemporalCrosscoder.

The point of these is the capacity property, not just plumbing: ``topk_relu`` realises
``min(k, #{pre > 0})`` rather than k, which is what silently caps a crosscoder whose
positive pre-activation count is smaller than a window-sized k. ``topk`` and ``batchtopk``
exist to avoid that, so the tests assert what each one guarantees.
"""
import pytest
import torch

from src.bench.architectures.crosscoder import CrosscoderSpec, TemporalCrosscoder

D_IN, D_SAE, T, K_PER = 8, 64, 3, 4
K_WINDOW = K_PER * T


def _model(activation):
    torch.manual_seed(0)
    return TemporalCrosscoder(D_IN, D_SAE, T, K_PER, activation=activation)


def _batch(n=16):
    torch.manual_seed(1)
    return torch.randn(n, T, D_IN)


def test_k_is_multiplied_by_T():
    assert _model("topk_relu").k == K_WINDOW


def test_rejects_unknown_activation():
    with pytest.raises(ValueError, match="activation must be one of"):
        TemporalCrosscoder(D_IN, D_SAE, T, K_PER, activation="softmax")


def test_topk_relu_never_exceeds_k_and_is_non_negative():
    m = _model("topk_relu")
    z = m.encode(_batch())
    assert (z >= 0).all()
    assert int((z != 0).sum(-1).max()) <= K_WINDOW


def test_topk_relu_is_capped_by_positive_preactivation_count():
    """The capacity failure mode, made explicit: drive pre-activations mostly negative
    and realised L0 falls below k even though TopK selected k latents."""
    m = _model("topk_relu")
    with torch.no_grad():
        m.b_enc.fill_(-50.0)          # only a handful of latents can stay positive
    x = _batch()
    pre = m.pre_acts(x)
    n_pos = (pre > 0).sum(-1)
    z = m.encode(x)
    realised = (z != 0).sum(-1)
    assert (realised < K_WINDOW).any(), "expected the ReLU to bind"
    assert torch.equal(realised, torch.minimum(n_pos, torch.tensor(K_WINDOW)))


def test_topk_realises_exactly_k_even_when_preacts_are_negative():
    """This is the reason 'topk' exists: capacity cannot collapse."""
    m = _model("topk")
    with torch.no_grad():
        m.b_enc.fill_(-50.0)
    z = m.encode(_batch())
    assert (z != 0).sum(-1).eq(K_WINDOW).all()


def test_batchtopk_spends_k_per_sample_on_average_in_training():
    m = _model("batchtopk").train()
    n = 16
    z = m.encode(_batch(n))
    # The batch rule allocates k*B non-zeros in total, unevenly across samples.
    assert int((z != 0).sum()) == K_WINDOW * n
    assert (z >= 0).all()


def test_batchtopk_learns_a_threshold_and_uses_it_at_eval():
    m = _model("batchtopk").train()
    assert torch.isnan(m.bt_threshold)
    for _ in range(5):
        m.encode(_batch())
    assert torch.isfinite(m.bt_threshold) and m.bt_threshold > 0

    m.eval()
    x = _batch()
    z = m.encode(x)
    pre = m.pre_acts(x)
    # Eval is a fixed threshold, not a batch rule, so it must be per-sample independent.
    assert torch.equal(z != 0, pre > m.bt_threshold)
    assert (z >= 0).all()


def test_batchtopk_eval_is_independent_of_batch_composition():
    """A batch-level rule at eval time would make one window's code depend on which
    other windows happened to be scored with it. The threshold form must not."""
    m = _model("batchtopk").train()
    for _ in range(5):
        m.encode(_batch())
    m.eval()
    x = _batch(8)
    alone = m.encode(x[:1])
    together = m.encode(x)[:1]
    assert torch.allclose(alone, together)


def test_untrained_batchtopk_falls_back_to_topk():
    m = _model("batchtopk").eval()
    assert torch.isnan(m.bt_threshold)
    z = m.encode(_batch())
    assert int((z != 0).sum(-1).max()) <= K_WINDOW


def test_default_activation_is_unchanged():
    """Existing callers must keep the historical behaviour."""
    assert TemporalCrosscoder(D_IN, D_SAE, T, K_PER).activation == "topk_relu"
    assert CrosscoderSpec(T=T).activation == "topk_relu"


def test_spec_passes_activation_through():
    spec = CrosscoderSpec(T=T, activation="batchtopk")
    m = spec.create(D_IN, D_SAE, K_PER, torch.device("cpu"))
    assert m.activation == "batchtopk"
    assert "batchtopk" in spec.name
