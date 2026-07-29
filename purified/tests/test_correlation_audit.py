"""Smoke and numerical contracts for the correlation robustness audit."""

from __future__ import annotations

import json

import numpy as np

from experiments.correlation_audit.run import (
    _bootstrap_half_pools,
    analyze_projected,
    fit_decay_models,
    main,
)


def test_power_floor_beats_pure_exponential():
    lags = np.arange(1, 49)
    curve = 1.7 * lags**-0.65 + 0.08
    fits = fit_decay_models(lags, curve)
    score = {row["model"]: row["aicc"] for row in fits}
    assert score["power_floor"] < score["exp"]
    power_floor = next(row for row in fits if row["model"] == "power_floor")
    assert set(power_floor["named_params"]) == {"amplitude", "alpha", "floor"}


def test_position_centering_removes_shared_position_artifact():
    rng = np.random.default_rng(0)
    n, seq, p = 80, 20, 5
    position = rng.normal(size=(seq, p)).astype(np.float32).cumsum(axis=0)
    z = position[None] + 0.1 * rng.normal(size=(n, seq, p)).astype(np.float32)
    result = analyze_projected(z, np.ones((n, seq), dtype=bool), max_lag=8)
    global_curve = result["centering"]["global"]["curve"]
    position_curve = result["centering"]["position"]["curve"]
    assert global_curve[0]["debiased_fro"] > 20 * position_curve[0]["debiased_fro"]


def test_precentered_mode_does_not_recenter_lag_endpoints():
    z = np.arange(24, dtype=np.float32).reshape(2, 6, 2)
    mask = np.ones((2, 6), dtype=bool)
    endpoint = analyze_projected(
        z,
        mask,
        max_lag=2,
        center_modes=("global",),
        endpoint_centering=True,
    )
    precentered = analyze_projected(
        z,
        mask,
        max_lag=2,
        center_modes=("global",),
        endpoint_centering=False,
    )
    endpoint_fro = endpoint["centering"]["global"]["curve"][0]["fro"]
    precentered_fro = precentered["centering"]["global"]["curve"][0]["fro"]
    assert endpoint_fro != precentered_fro


def test_document_bootstrap_reports_model_stability():
    rng = np.random.default_rng(4)
    z = rng.normal(size=(32, 12, 4)).astype(np.float32)
    result = analyze_projected(
        z,
        np.ones((32, 12), dtype=bool),
        max_lag=6,
        center_modes=("global",),
        n_bootstrap=3,
        bootstrap_blocks=4,
    )
    bootstrap = result["centering"]["global"]["bootstrap"]
    assert bootstrap["n_bootstrap"] == 3
    assert len(bootstrap["debiased_fro_q025"]) == 6


def test_bootstrap_split_half_pools_are_disjoint_and_complete():
    first, second = _bootstrap_half_pools(11, seed=7)
    assert set(first).isdisjoint(second)
    assert sorted(np.concatenate([first, second]).tolist()) == list(range(11))


def test_cli_smoke_loads_npy_and_writes_json(tmp_path):
    rng = np.random.default_rng(1)
    acts = rng.normal(size=(24, 12, 9)).astype(np.float32)
    token_ids = np.ones((24, 12), dtype=np.int64)
    token_ids[:, -2:] = 0
    acts_path = tmp_path / "acts.npy"
    ids_path = tmp_path / "ids.npy"
    out_path = tmp_path / "audit.json"
    np.save(acts_path, acts)
    np.save(ids_path, token_ids)
    main(
        [
            "--activations",
            str(acts_path),
            "--token-ids",
            str(ids_path),
            "--output",
            str(out_path),
            "--hf-revision",
            "deadbeef",
            "--activation-sha256",
            "a" * 64,
            "--token-ids-sha256",
            "b" * 64,
            "--pad-token-id",
            "0",
            "--projection",
            "random",
            "--projection-dim",
            "4",
            "--fit-tokens",
            "100",
            "--max-lag",
            "5",
            "--device",
            "cpu",
            "--bootstrap",
            "2",
            "--bootstrap-blocks",
            "4",
        ]
    )
    payload = json.loads(out_path.read_text())
    assert payload["provenance"]["valid_tokens"] == 240
    assert payload["provenance"]["hf_revision"] == "deadbeef"
    assert payload["provenance"]["activation_sha256"] == "a" * 64
    assert len(payload["centering"]["global"]["curve"]) == 5
    assert payload["provenance"]["persistent_subspace_removal"] is False
    assert payload["provenance"]["lag_centering"] == "endpoint"
