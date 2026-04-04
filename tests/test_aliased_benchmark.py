"""Tests for the aliased paired-feature benchmark."""

from __future__ import annotations

import torch

from temporal_bench.benchmarks.aliased_data import (
    AliasedBatch,
    AliasedDataConfig,
    AliasedDataPipeline,
)
from temporal_bench.benchmarks.aliased_eval import (
    _pair_alignment_sums,
    evaluate_aliased_model,
)
from temporal_bench.benchmarks.aliased_runner import (
    AliasedBenchmarkConfig,
    AliasedModelEntry,
    run_aliased_benchmark,
)
from temporal_bench.config import TrainConfig
from temporal_bench.models.base import ModelOutput, TemporalAE


def _infer_bits_from_x(x: torch.Tensor, true_features: torch.Tensor) -> torch.Tensor:
    coeffs = torch.einsum("btd,fd->btf", x, true_features)
    return coeffs.view(x.shape[0], x.shape[1], -1, 2).argmax(dim=-1)


def _lag1_corr(bits: torch.Tensor) -> float:
    bits = bits.float()
    s0 = bits[:, :-1]
    s1 = bits[:, 1:]
    mu = bits.mean()
    cov = ((s0 - mu) * (s1 - mu)).mean()
    var = ((bits - mu) ** 2).mean().clamp(min=1e-8)
    return (cov / var).item()


class WindowEchoModel(TemporalAE):
    def __init__(self, true_features: torch.Tensor, T: int):
        super().__init__()
        self.true_features = true_features
        self.T = T

    def forward(self, x: torch.Tensor) -> ModelOutput:
        latents = x
        loss = torch.zeros((), device=x.device)
        return ModelOutput(
            x_hat=x,
            latents=latents,
            loss=loss,
            metrics={"recon_loss": 0.0, "l0": float(latents.shape[-1])},
        )

    def decoder_directions(self, pos: int | None = None) -> torch.Tensor:
        return self.true_features.T

    @property
    def n_positions(self) -> int:
        return self.T


class TestAliasedData:
    def test_pairs_are_one_hot_and_metadata_is_consistent(self):
        cfg = AliasedDataConfig(
            n_pairs=3,
            d_model=12,
            seq_len=8,
            eval_n_seq=32,
            cache_n_seq=4,
            cache_seq_len=32,
            scaling_n_seq=64,
        )
        pipe = AliasedDataPipeline(cfg)
        batch = pipe.eval_batch(0.5)

        pair_support = batch.support.view(cfg.eval_n_seq, cfg.n_pairs, 2, cfg.seq_len)
        assert torch.allclose(pair_support.sum(dim=2), torch.ones(cfg.eval_n_seq, cfg.n_pairs, cfg.seq_len))

        current_bits = batch.current_feature_idx % 2
        next_bits = batch.next_feature_idx % 2
        assert torch.equal(pair_support.argmax(dim=2), current_bits)
        assert torch.equal(batch.aliased_state, 2 * current_bits + next_bits)
        assert torch.equal(batch.informative_mask[:, :, :-1], current_bits[:, :, :-1] != current_bits[:, :, 1:])
        assert not batch.informative_mask[:, :, -1].any()

    def test_empirical_lag1_matches_rho(self):
        cfg = AliasedDataConfig(
            n_pairs=2,
            d_model=8,
            seq_len=64,
            eval_n_seq=512,
            cache_n_seq=8,
            cache_seq_len=64,
            scaling_n_seq=128,
        )
        pipe = AliasedDataPipeline(cfg)
        batch = pipe.eval_batch(0.7)
        bits = (batch.current_feature_idx % 2).reshape(cfg.eval_n_seq * cfg.n_pairs, cfg.seq_len)
        assert abs(_lag1_corr(bits) - 0.7) < 0.08

    def test_sequence_shuffle_preserves_marginals_and_breaks_correlation(self):
        cfg = AliasedDataConfig(
            n_pairs=2,
            d_model=8,
            seq_len=32,
            eval_n_seq=32,
            cache_n_seq=64,
            cache_seq_len=128,
            scaling_n_seq=128,
        )
        pipe = AliasedDataPipeline(cfg)
        seq = pipe.sample_seq(256, 0.9, shuffle=False)
        shuf = pipe.sample_seq(256, 0.9, shuffle=True)

        seq_bits = _infer_bits_from_x(seq, pipe.true_features)
        shuf_bits = _infer_bits_from_x(shuf, pipe.true_features)

        assert abs(seq_bits.float().mean().item() - shuf_bits.float().mean().item()) < 0.05
        seq_corr = _lag1_corr(seq_bits.permute(0, 2, 1).reshape(-1, cfg.seq_len))
        shuf_corr = _lag1_corr(shuf_bits.permute(0, 2, 1).reshape(-1, cfg.seq_len))
        assert seq_corr > 0.75
        assert shuf_corr < 0.6
        assert shuf_corr < seq_corr - 0.2


class TestAliasedMetrics:
    def test_pair_subspace_prefers_local_or_predictive_direction(self):
        cfg = AliasedDataConfig(
            n_pairs=2,
            d_model=4,
            seq_len=6,
            eval_n_seq=8,
            cache_n_seq=4,
            cache_seq_len=16,
            scaling_n_seq=32,
        )
        pipe = AliasedDataPipeline(cfg)
        batch = pipe.eval_batch(0.5)
        pair_dirs = pipe.true_features.view(cfg.n_pairs, 2, cfg.d_model)

        local_ls, local_ps, local_n = _pair_alignment_sums(batch.x, batch, pair_dirs)
        current_onehot = torch.nn.functional.one_hot(
            (batch.next_feature_idx % 2).transpose(1, 2), num_classes=2
        ).float()
        predictive_x = torch.einsum("btgi,gid->btd", current_onehot, pair_dirs)
        pred_ls, pred_ps, pred_n = _pair_alignment_sums(
            predictive_x, batch, pair_dirs
        )

        assert local_n == pred_n
        assert local_ls / local_n > local_ps / local_n
        assert pred_ps / pred_n > pred_ls / pred_n

    def test_window_evaluation_counts_overlapping_positions(self):
        true_features = torch.eye(2)
        bits = torch.tensor(
            [
                [[0, 1, 0, 1]],
                [[1, 0, 1, 0]],
            ],
            dtype=torch.float32,
        ).repeat(32, 1, 1)
        cfg = AliasedDataConfig(n_pairs=1, d_model=2, seq_len=4)
        pipe = AliasedDataPipeline(cfg)
        support, current_idx, next_idx, aliased_state, informative_mask = pipe._metadata_from_bits(bits)
        x = torch.einsum("nkt,kd->ntd", support, true_features)
        batch = AliasedBatch(
            x=x,
            support=support,
            current_feature_idx=current_idx,
            next_feature_idx=next_idx,
            aliased_state=aliased_state,
            informative_mask=informative_mask,
        )

        model = WindowEchoModel(true_features, T=2)
        metrics = evaluate_aliased_model(
            model,
            batch,
            true_features,
            eval_chunk_size=1,
            probe_steps=300,
        )
        assert metrics.nmse == 0.0
        assert metrics.auc > 0.98
        assert metrics.probe_auc > 0.98
        assert metrics.n_informative == 320


class TestAliasedSmokeBenchmark:
    def test_smoke_run_covers_full_roster(self, tmp_path):
        cfg = AliasedBenchmarkConfig(
            data=AliasedDataConfig(
                n_pairs=4,
                d_model=8,
                seq_len=6,
                eval_n_seq=8,
                cache_n_seq=8,
                cache_seq_len=32,
                scaling_n_seq=32,
            ),
            rho_values=[0.0],
            k_values=[1],
            dict_width=8,
            output_dir=str(tmp_path),
            eval_chunk_size=4,
            probe_steps=10,
            make_plots=False,
        )
        entries = [
            AliasedModelEntry(
                name="SAE",
                model_name="sae",
                data_kind="flat",
                train_config=TrainConfig(n_steps=1, batch_size=8, lr=1e-3),
            ),
            AliasedModelEntry(
                name="BatchTopK SAE",
                model_name="batchtopk_sae",
                data_kind="flat",
                train_config=TrainConfig(n_steps=1, batch_size=8, lr=1e-3),
            ),
            AliasedModelEntry(
                name="TFA",
                model_name="tfa",
                data_kind="seq",
                train_config=TrainConfig(
                    n_steps=1,
                    batch_size=2,
                    lr=1e-3,
                    min_lr=9e-4,
                    optimizer="adamw",
                    weight_decay=1e-4,
                    beta1=0.9,
                    beta2=0.95,
                    warmup_steps=1,
                    lr_schedule="cosine",
                    grouped_weight_decay=True,
                ),
                model_kwargs={"n_heads": 2, "n_attn_layers": 1, "bottleneck_factor": 1},
            ),
            AliasedModelEntry(
                name="TFA-shuf",
                model_name="tfa",
                data_kind="seq_shuffled",
                train_config=TrainConfig(
                    n_steps=1,
                    batch_size=2,
                    lr=1e-3,
                    min_lr=9e-4,
                    optimizer="adamw",
                    weight_decay=1e-4,
                    beta1=0.9,
                    beta2=0.95,
                    warmup_steps=1,
                    lr_schedule="cosine",
                    grouped_weight_decay=True,
                ),
                model_kwargs={"n_heads": 2, "n_attn_layers": 1, "bottleneck_factor": 1},
            ),
            AliasedModelEntry(
                name="TFA-pos",
                model_name="tfa",
                data_kind="seq",
                train_config=TrainConfig(
                    n_steps=1,
                    batch_size=2,
                    lr=1e-3,
                    min_lr=9e-4,
                    optimizer="adamw",
                    weight_decay=1e-4,
                    beta1=0.9,
                    beta2=0.95,
                    warmup_steps=1,
                    lr_schedule="cosine",
                    grouped_weight_decay=True,
                ),
                model_kwargs={
                    "n_heads": 2,
                    "n_attn_layers": 1,
                    "bottleneck_factor": 1,
                    "use_pos_encoding": True,
                },
            ),
            AliasedModelEntry(
                name="TFA-pos-shuf",
                model_name="tfa",
                data_kind="seq_shuffled",
                train_config=TrainConfig(
                    n_steps=1,
                    batch_size=2,
                    lr=1e-3,
                    min_lr=9e-4,
                    optimizer="adamw",
                    weight_decay=1e-4,
                    beta1=0.9,
                    beta2=0.95,
                    warmup_steps=1,
                    lr_schedule="cosine",
                    grouped_weight_decay=True,
                ),
                model_kwargs={
                    "n_heads": 2,
                    "n_attn_layers": 1,
                    "bottleneck_factor": 1,
                    "use_pos_encoding": True,
                },
            ),
            AliasedModelEntry(
                name="TXCDR T=2",
                model_name="txcdr",
                data_kind="window",
                train_config=TrainConfig(n_steps=1, batch_size=4, lr=1e-3),
                window_size=2,
            ),
            AliasedModelEntry(
                name="TXCDR T=5",
                model_name="txcdr",
                data_kind="window",
                train_config=TrainConfig(n_steps=1, batch_size=4, lr=1e-3),
                window_size=5,
            ),
        ]

        results = run_aliased_benchmark(cfg, device=torch.device("cpu"), model_entries=entries)
        assert list(results.keys()) == [0.0]
        for name in [entry.name for entry in entries]:
            metric = results[0.0][name][0]
            assert metric.auc >= 0.0
            assert hasattr(metric, "delta")
