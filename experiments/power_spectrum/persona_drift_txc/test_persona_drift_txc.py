from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from experiments.power_spectrum.persona_drift_txc import (
    collect_activations,
    train_representations,
)
from experiments.power_spectrum.persona_drift_txc.probe_future_drift import (
    _fit_dual_ridge,
)
from experiments.power_spectrum.persona_drift_txc.protocol import (
    build_probe_indices,
    future_targets,
    project_axis,
    split_for_persona,
)
from experiments.power_spectrum.persona_drift_txc.train_representations import (
    train_one,
)


def _metadata(n_conversations: int) -> list[dict[str, object]]:
    rows = []
    for index in range(n_conversations):
        persona_id = index % 5
        rows.append(
            {
                "conversation_id": f"conv-{index:03d}",
                "domain": "coding",
                "persona_id": persona_id,
                "split": split_for_persona(persona_id),
            }
        )
    return rows


def test_probe_rows_never_cross_conversations() -> None:
    metadata = _metadata(2)
    rows = build_probe_indices(
        metadata,
        turns_per_conversation=[6, 5],
        window=3,
        horizon=2,
    )
    assert [(row.conversation_id, row.turn) for row in rows] == [
        ("conv-000", 2),
        ("conv-000", 3),
        ("conv-001", 2),
    ]

    scores = torch.tensor(
        [
            [9.0, 8.0, 7.0, 6.0, 5.0, 4.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, float("nan")],
        ]
    )
    targets = future_targets(scores, rows, safe_threshold=4.5)
    np.testing.assert_allclose(targets["future_min"], [5.0, 4.0, 4.0])
    np.testing.assert_allclose(targets["future_delta"], [-2.0, -2.0, 2.0])
    np.testing.assert_array_equal(targets["future_breach"], [0, 1, 1])


def test_axis_projection_normalizes_direction() -> None:
    axis = torch.zeros(3, 2)
    axis[1] = torch.tensor([3.0, 4.0])
    activations = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
    score = project_axis(activations, axis, layer=1)
    torch.testing.assert_close(score, torch.tensor([3.0, 4.0]))


def test_dual_ridge_recovers_held_out_linear_signal() -> None:
    generator = np.random.default_rng(7)
    train_x = generator.normal(size=(60, 6)).astype(np.float32)
    validation_x = generator.normal(size=(20, 6)).astype(np.float32)
    test_x = generator.normal(size=(20, 6)).astype(np.float32)
    weights = np.asarray([1.2, -0.7, 0.0, 0.4, 0.0, 0.2])
    train_y = train_x @ weights
    validation_y = validation_x @ weights
    test_y = test_x @ weights
    prediction, alpha, validation_r2 = _fit_dual_ridge(
        train_x=train_x,
        train_y=train_y,
        validation_x=validation_x,
        validation_y=validation_y,
        test_x=test_x,
        alphas=[1e-6, 1e-3, 1.0],
    )
    assert alpha in {1e-6, 1e-3, 1.0}
    assert validation_r2 > 0.99
    assert 1 - np.square(test_y - prediction).sum() / np.square(test_y - test_y.mean()).sum() > 0.99


def test_reference_vllm_generator_builds_complete_conversation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scripts = tmp_path / "scripts.jsonl"
    script = {
        "conversation_id": "coding-p00-t00",
        "domain": "coding",
        "persona_id": 0,
        "persona": "test persona",
        "topic_id": 0,
        "topic": "test topic",
        "split": "train",
        "user_messages": [f"user message {turn}" for turn in range(15)],
    }
    scripts.write_text(json.dumps(script) + "\n")

    class FakeTokenizer:
        def __call__(self, text: str, **_kwargs):
            return {"input_ids": text.split()}

    class FakeLLM:
        def get_tokenizer(self):
            return FakeTokenizer()

    class FakeGenerator:
        def __init__(self, **_kwargs):
            self.llm = FakeLLM()
            self.sampling_params = type("Sampling", (), {"seed": None})()

        def load(self) -> None:
            return None

        def generate_batch(self, conversations):
            return [f"assistant response {len(row)}" for row in conversations]

    monkeypatch.setattr(
        collect_activations,
        "_reference_imports",
        lambda: {"VLLMGenerator": FakeGenerator},
    )
    monkeypatch.setattr(
        collect_activations,
        "_subject_model_source",
        lambda _config: "Qwen/Qwen3-32B",
    )
    output = tmp_path / "conversations.jsonl"
    collect_activations.generate_conversations_vllm(
        scripts_path=scripts,
        output_path=output,
        limit=None,
        batch_size=1,
    )
    generated = json.loads(output.read_text())
    assert len(generated["conversation"]) == 30
    assert generated["conversation"][0]["role"] == "user"
    assert generated["conversation"][-1]["role"] == "assistant"
    assert len(generated["response_token_counts"]) == 15
    assert generated["generation_backend"] == "assistant_axis.VLLMGenerator"


def test_all_representations_train_and_emit_health(
    tmp_path: Path,
    monkeypatch,
) -> None:
    test_config = copy.deepcopy(train_representations.load_config())
    test_config["representation_training"]["checkpoint_every"] = 2
    test_config["representation_training"]["dead_feature_threshold_tokens"] = 10
    monkeypatch.setattr(
        train_representations,
        "load_config",
        lambda: test_config,
    )
    generator = torch.Generator().manual_seed(17)
    metadata = _metadata(10)
    activations = torch.randn(10, 10, 8, generator=generator).to(torch.bfloat16)
    activation_path = tmp_path / "activations.pt"
    torch.save(
        {
            "activations": activations,
            "axis_scores": torch.randn(10, 10, generator=generator),
            "conversation_ids": [row["conversation_id"] for row in metadata],
        },
        activation_path,
    )
    metadata_path = tmp_path / "metadata.jsonl"
    with metadata_path.open("w") as handle:
        for row in metadata:
            handle.write(json.dumps(row) + "\n")

    output_root = tmp_path / "representations"
    for architecture in ("sae", "tsae", "txc_w4", "txc_w8"):
        train_one(
            architecture=architecture,
            activation_path=activation_path,
            metadata_path=metadata_path,
            output_root=output_root,
            force=True,
            steps_override=3,
            d_sae_override=24,
            positions_per_step_override=8,
        )
        health_path = output_root / architecture / "health.json"
        assert health_path.exists()
        with health_path.open() as handle:
            health = json.load(handle)
        assert health["steps"] == 3
        assert health["d_sae"] == 24
        assert np.isfinite(health["final_fixed_validation_nmse"])
        assert np.isfinite(health["reconstruction_nmse"])
        assert (output_root / architecture / "model.safetensors").exists()
        assert (output_root / architecture / "codes.pt").exists()
        assert not (output_root / architecture / "resume_checkpoint.pt").exists()
