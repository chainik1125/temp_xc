from __future__ import annotations

import json

from experiments.power_spectrum.backtracking_sae_pooling.steering_baselines import (
    run_baselines,
)


def test_successful_judge_keys_deduplicates_retries() -> None:
    rows = [
        {"transcript_id": "q1", "magnitude": -1, "arch": "topk_sae", "seed": 42, "label": -1},
        {"transcript_id": "q1", "magnitude": -1, "arch": "topk_sae", "seed": 42, "label": 1},
        {"transcript_id": "q1", "magnitude": -1, "arch": "topk_sae", "seed": 42, "label": 1},
        {"transcript_id": "q2", "magnitude": 0, "arch": "txc_base", "seed": 42, "label": 0},
    ]
    assert run_baselines.successful_judge_keys(rows, arch="topk_sae", seed=42) == {
        ("q1", -1.0, "topk_sae", 42)
    }


def test_atomic_write_json_replaces_complete_file(tmp_path) -> None:
    target = tmp_path / "result.json"
    run_baselines.atomic_write_json(target, {"complete": True, "value": 3})
    assert json.loads(target.read_text()) == {"complete": True, "value": 3}
    assert not target.with_suffix(".json.tmp").exists()

