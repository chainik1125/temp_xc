from __future__ import annotations

import numpy as np
import torch

from experiments.power_spectrum.decision_sprint.analyze import analyze
from experiments.power_spectrum.decision_sprint.model import (
    FlexibleTrajectoryBottleneck,
)


def _row(seed: int, txc: float, adapter: float) -> dict:
    evaluations = {
        "adapter_rank0": {
            "lambda_recovery_v2": adapter - 0.02,
            "l0_per_window": 7.0,
        },
        "adapter_rank0_reverse": {
            "lambda_recovery_v2": adapter - 0.08,
            "l0_per_window": 7.0,
        },
        "adapter_rank0_untrained": {
            "lambda_recovery_v2": 0.0,
            "l0_per_window": 7.0,
        },
        "adapter_rank256": {
            "lambda_recovery_v2": adapter,
            "l0_per_window": 7.0,
        },
        "adapter_rank256_reverse": {
            "lambda_recovery_v2": adapter - 0.08,
            "l0_per_window": 7.0,
        },
        "adapter_rank256_untrained": {
            "lambda_recovery_v2": 0.0,
            "l0_per_window": 7.0,
        },
        "sae_last": {"lambda_recovery_v2": 0.03},
        "sae_mean_top8": {"lambda_recovery_v2": 0.08},
        "sae_max_top8": {"lambda_recovery_v2": 0.09},
    }
    return {
        "seed": seed,
        "canonical": {
            "txc": {
                "metrics": {
                    "lambda_recovery_v2": txc,
                    "l0_per_window": 7.2,
                }
            },
            "txc_untrained": {"metrics": {"lambda_recovery_v2": 0.0}},
        },
        "evaluations": {
            "txc_reverse": {"lambda_recovery_v2": txc - 0.08},
            **evaluations,
        },
    }


def _config() -> dict:
    return {
        "protocol": "dailydialog-learned-sae-trajectory-control.v1",
        "seeds": [9, 10, 11, 12, 13, 14],
        "adapter_ranks": [0, 256],
        "primary_adapter_rank": 256,
        "noninferiority_margin_r": 0.03,
        "sign_of_life_margin_r": 0.05,
        "order_drop_margin_r": 0.03,
        "max_realized_l0_gap": 1.5,
    }


def test_adapter_tie_stops_general_txc() -> None:
    raw = {
        "status": "complete",
        "protocol": _config()["protocol"],
        "results": [
            _row(9, 0.25, 0.24),
            _row(10, 0.26, 0.25),
            _row(11, 0.24, 0.23),
            _row(12, 0.25, 0.24),
            _row(13, 0.26, 0.25),
            _row(14, 0.24, 0.23),
        ]
    }
    summary = analyze(raw, _config())
    assert summary["verdict"] == "STOP_GENERAL_TXC"
    assert summary["gates"]["adapter_noninferior_ci_upper_within_0.03"]


def test_clear_txc_win_can_be_sign_of_life() -> None:
    raw = {
        "status": "complete",
        "protocol": _config()["protocol"],
        "results": [
            _row(9, 0.35, 0.20),
            _row(10, 0.36, 0.21),
            _row(11, 0.34, 0.19),
            _row(12, 0.35, 0.20),
            _row(13, 0.36, 0.21),
            _row(14, 0.34, 0.19),
        ]
    }
    summary = analyze(raw, _config())
    assert summary["verdict"] == "INITIAL_REAL_TASK_SIGN_OF_LIFE"
    assert summary["gates"]["txc_real_task_sign_of_life"]
    assert np.mean(
        summary["paired_txc_minus_best"]["seed_values"]
    ) > _config()["sign_of_life_margin_r"]


def test_analysis_refuses_duplicate_or_missing_seeds() -> None:
    raw = {
        "status": "complete",
        "protocol": _config()["protocol"],
        "results": [_row(seed, 0.25, 0.24) for seed in [9, 10, 11]],
    }
    try:
        analyze(raw, _config())
    except ValueError as error:
        assert "expected unique seeds" in str(error)
    else:
        raise AssertionError("incomplete seed set was analyzed")


def test_flexible_decoder_preserves_shape_and_gets_gradients() -> None:
    model = FlexibleTrajectoryBottleneck(
        base_decoder=torch.randn(12, 5),
        base_decoder_bias=torch.zeros(5),
        window=4,
        k_window=3,
        rank=2,
        decoder_rank=3,
    )
    indices = torch.randint(0, 12, (2, 4, 2))
    values = torch.rand(2, 4, 2)
    target = torch.randn(2, 4, 5)
    result = model.loss(indices, values, target)
    result["loss"].backward()
    assert model.decoder_cross_out.grad is not None
    assert model.decode_sparse(
        *model.encode_sparse(indices, values)[:2], add_bias=True
    ).shape == target.shape
