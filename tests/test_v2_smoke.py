"""Smoke tests for the v2 crosscoder comparison pipeline."""

import torch

from src.utils.seed import set_seed
from src.v2_crosscoder_comparison.architectures import create_architecture
from src.v2_crosscoder_comparison.architectures.crosscoder import CrosscoderModule
from src.v2_crosscoder_comparison.configs import (
    ArchitectureConfig,
    DataConfig,
    ExperimentConfig,
    ToyModelConfig,
)
from src.v2_crosscoder_comparison.data_generation import (
    build_cross_position_correlation_matrix,
    generate_two_position_batch,
)
from src.v2_crosscoder_comparison.eval import EvalResult, compute_pareto_frontier, evaluate
from src.v2_crosscoder_comparison.toy_model import TwoPositionToyModel
from src.v2_crosscoder_comparison.train import train_architecture

DEVICE = torch.device("cpu")


def test_toy_model_shape():
    set_seed(42)
    cfg = ToyModelConfig(num_features=10, hidden_dim=20)
    model = TwoPositionToyModel(cfg)
    x = torch.randn(4, 2, 10)
    out = model(x)
    assert out.shape == (4, 2, 20)


def test_correlation_matrix_shape():
    corr = build_cross_position_correlation_matrix(10, rho=0.5)
    assert corr.shape == (20, 20)
    # Check symmetry
    assert torch.allclose(corr, corr.T)
    # Check diagonal is 1
    assert torch.allclose(corr.diag(), torch.ones(20))


def test_correlation_matrix_rho_zero():
    corr = build_cross_position_correlation_matrix(10, rho=0.0)
    # Should be identity
    assert torch.allclose(corr, torch.eye(20))


def test_correlation_matrix_high_rho():
    corr = build_cross_position_correlation_matrix(10, rho=1.0)
    # Should be clamped to 0.999 and still PSD
    eigenvals = torch.linalg.eigvalsh(corr)
    assert (eigenvals > -1e-6).all()


def test_data_generation_shape():
    set_seed(42)
    data_cfg = DataConfig(num_features=10)
    corr = build_cross_position_correlation_matrix(10, rho=0.5)
    batch = generate_two_position_batch(32, data_cfg, corr, DEVICE)
    assert batch.shape == (32, 2, 10)
    assert (batch >= 0).all()  # ReLU magnitude sampling


def test_crosscoder_module_shapes():
    set_seed(42)
    module = CrosscoderModule(n_positions=2, d_model=20, d_sae=40, top_k=5)
    x = torch.randn(4, 2, 20)
    z = module.encode(x)
    assert z.shape == (4, 40)
    # Check TopK: at most top_k non-zero per sample
    assert (z > 0).float().sum(dim=-1).max().item() <= 5

    x_hat = module.decode(z)
    assert x_hat.shape == (4, 2, 20)

    x_recon = module(x)
    assert x_recon.shape == (4, 2, 20)


def test_create_architecture_factory():
    for arch_type in ["naive_sae", "stacked_sae", "crosscoder"]:
        cfg = ArchitectureConfig(arch_type=arch_type, d_sae=40, top_k=5)
        arch = create_architecture(cfg, d_model=20)
        assert arch is not None


def test_naive_sae_forward():
    set_seed(42)
    cfg = ArchitectureConfig(arch_type="naive_sae", d_sae=40, top_k=5)
    arch = create_architecture(cfg, d_model=20)
    x = torch.randn(4, 2, 20)
    out = arch.forward(x)
    assert out.shape == (4, 2, 20)


def test_stacked_sae_forward():
    set_seed(42)
    cfg = ArchitectureConfig(arch_type="stacked_sae", d_sae=40, top_k=5)
    arch = create_architecture(cfg, d_model=20)
    x = torch.randn(4, 2, 20)
    out = arch.forward(x)
    assert out.shape == (4, 2, 20)


def test_crosscoder_forward():
    set_seed(42)
    cfg = ArchitectureConfig(arch_type="crosscoder", d_sae=40, top_k=5)
    arch = create_architecture(cfg, d_model=20)
    x = torch.randn(4, 2, 20)
    out = arch.forward(x)
    assert out.shape == (4, 2, 20)


def test_pareto_frontier():
    results = [
        EvalResult("a", 0.0, 5, 42, l0=5.0, mse=0.5, fvu=0.1, mean_max_cos_sim=0.8, dead_latent_frac=0.0),
        EvalResult("a", 0.0, 10, 42, l0=10.0, mse=0.3, fvu=0.06, mean_max_cos_sim=0.9, dead_latent_frac=0.0),
        EvalResult("a", 0.0, 15, 42, l0=15.0, mse=0.4, fvu=0.08, mean_max_cos_sim=0.85, dead_latent_frac=0.0),
    ]
    pareto = compute_pareto_frontier(results)
    assert len(pareto) == 2  # (5, 0.5) and (10, 0.3) are Pareto-optimal
    assert pareto[0].l0 == 5.0
    assert pareto[1].l0 == 10.0


def test_single_run_smoke():
    """End-to-end smoke test with minimal settings."""
    set_seed(42)
    toy_cfg = ToyModelConfig(num_features=10, hidden_dim=20)
    model = TwoPositionToyModel(toy_cfg).to(DEVICE)

    data_cfg = DataConfig(num_features=10, rho=0.5)
    corr_matrix = build_cross_position_correlation_matrix(10, 0.5).to(DEVICE)

    arch_cfg = ArchitectureConfig(arch_type="crosscoder", d_sae=20, top_k=3)
    exp_cfg = ExperimentConfig(
        toy_model=toy_cfg,
        data=data_cfg,
        architecture=arch_cfg,
        training_steps=10,  # Very few steps for smoke test
        batch_size=32,
        seed=42,
    )

    arch = create_architecture(arch_cfg, d_model=20)

    def gen_hidden(batch_size: int) -> torch.Tensor:
        feats = generate_two_position_batch(batch_size, data_cfg, corr_matrix, DEVICE)
        return model(feats)

    train_architecture("crosscoder", arch, gen_hidden, exp_cfg, DEVICE)

    result = evaluate(
        arch=arch,
        generate_hidden_fn=gen_hidden,
        feature_directions=model.embedding.data,
        arch_type="crosscoder",
        rho=0.5,
        top_k=3,
        seed=42,
        n_eval_samples=64,
        batch_size=32,
    )

    assert isinstance(result, EvalResult)
    assert result.arch_type == "crosscoder"
    assert result.mse >= 0
    assert 0 <= result.dead_latent_frac <= 1
