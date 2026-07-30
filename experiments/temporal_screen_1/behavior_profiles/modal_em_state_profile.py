"""Teacher-forced, SAE-free state profile for current-paper Medical EM.

This runner does *not* generate or re-judge rollouts.  It consumes the 64
frozen bad-medical rollouts already produced by the paper-pinned
``generate_with_steering`` and judged by the paper-pinned Claude rubric, then
teacher-forces the same merged subject model over their decoded/re-encoded
token sequences.

Run only from a committed source state:

    modal run \
      experiments/temporal_screen_1/behavior_profiles/modal_em_state_profile.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal


HERE = Path(__file__).resolve().parent
FROZEN_INPUT = HERE.parent / "results" / "dual_horizon_em_paper7b.json"
REPO_URL = "https://github.com/chainik1125/temp_xc.git"
PAPER_PIN = "457888a3ed23d819cfd5d8c7fb94565932bfb670"
EM_SOURCE_SHA256 = "68b5de617734b3be65a97d848f1fb6134d6d81862fa11f899e8ebbe9b3a6c9dd"
FROZEN_INPUT_SHA256 = (
    "1d61d54cc67c9b54a2846df58672743ae7e90c8aea69dd2b4725ebb2aa88e5ef"
)
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER = "andyrdt/Qwen2.5-7B-Instruct_bad-medical"

app = modal.App("temporal-screen-em-state-profile")
hf_secret = modal.Secret.from_name("hf-token")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install("uv")
    .run_commands(
        f"git clone {REPO_URL} /repo",
        f"git -C /repo checkout {PAPER_PIN}",
        (
            "uv pip install --system --editable /repo/purified "
            "'accelerate==1.13.0' 'huggingface-hub==1.13.0' "
            "'numpy==2.4.4' 'safetensors==0.7.0' 'transformers==5.7.0'"
        ),
    )
    .pip_install("hf-xet>=1.1")
    .env(
        {
            "PYTHONPATH": "/repo/purified/src:/work/behavior_profiles",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HOME": "/root/.cache/huggingface",
        }
    )
    .add_local_file(str(FROZEN_INPUT), "/work/frozen_em.json")
    .add_local_dir(str(HERE), "/work/behavior_profiles")
)


@app.function(
    image=image,
    gpu="A100-40GB",
    cpu=8,
    memory=32_768,
    timeout=45 * 60,
    secrets=[hf_secret],
)
def run_profile(
    layer: int,
    n_progress: int,
    n_bootstrap: int,
    seed: int,
) -> dict:
    import hashlib
    import sys
    import time

    import numpy as np
    import torch

    sys.path.insert(0, "/repo/purified/src")
    sys.path.insert(0, "/work/behavior_profiles")

    from em_state_profile import (
        estimate_state_profile,
        normalize_activation_rows,
        select_coherent_extremes,
    )
    from temp_bench.case_studies import em as reference

    started = time.time()
    frozen_path = Path("/work/frozen_em.json")
    if hashlib.sha256(frozen_path.read_bytes()).hexdigest() != FROZEN_INPUT_SHA256:
        raise RuntimeError("frozen EM input hash changed")
    payload = json.loads(frozen_path.read_text())
    if payload.get("status") != "complete":
        raise RuntimeError("frozen EM result is not marked complete")
    arm = payload["arms"]["bad_medical"]
    expected_reference = {
        "commit": PAPER_PIN,
        "source_sha256": EM_SOURCE_SHA256,
        "generator": "temp_bench.case_studies.em.generate_with_steering",
    }
    for key, expected in expected_reference.items():
        if arm["reference"].get(key) != expected:
            raise RuntimeError(
                f"frozen rollout reference mismatch for {key}: "
                f"{arm['reference'].get(key)!r} != {expected!r}"
            )
    if arm["model"] != {
        "base_model": BASE_MODEL,
        "adapter": ADAPTER,
        "merged_lora": True,
    }:
        raise RuntimeError("frozen rollout subject model metadata changed")

    source_path = Path("/repo/purified/src/temp_bench/case_studies/em.py")
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != EM_SOURCE_SHA256:
        raise RuntimeError("pinned em.py source hash changed")

    rows = arm["generations"]["full"]
    selection = select_coherent_extremes(rows)
    if selection.n_positive < 8 or selection.n_negative < 8:
        raise RuntimeError("too few coherent endpoint labels for the profile")

    model, tokenizer = reference.load_subject_with_lora(
        base_model_id=BASE_MODEL,
        adapter_id=ADAPTER,
        device="cuda",
    )
    device = next(model.parameters()).device
    if not (0 <= layer < len(model.model.layers)):
        raise ValueError(f"layer {layer} outside model depth")
    progress = np.linspace(0.0, 1.0, n_progress).tolist()

    # The original reference generator stores only decoded answers.  Re-encode
    # with the exact same tokenizer and concatenation convention already used
    # by the frozen-XE analysis.  Stored token counts bind this reconstruction.
    instant = []
    prefix_mean = []
    token_audit = []
    captured: dict[str, torch.Tensor] = {}

    def capture(_module, _inputs, output):
        value = output[0] if isinstance(output, tuple) else output
        captured["residual"] = value.detach()

    handle = model.model.layers[layer].register_forward_hook(capture)
    try:
        with torch.inference_mode():
            for source_index in selection.indices.tolist():
                row = rows[source_index]
                prompt_index = int(row["prompt_index"])
                question = reference.EM_PROMPTS[prompt_index]
                if row["question"] != question:
                    raise RuntimeError("frozen prompt differs from pinned EM_PROMPTS")
                rendered = tokenizer.apply_chat_template(
                    [{"role": "user", "content": question}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
                answer_ids = tokenizer.encode(
                    row["answer"],
                    add_special_tokens=False,
                )
                if len(answer_ids) != int(row["n_answer_tokens"]):
                    raise RuntimeError(
                        f"{row['observation_id']}: decoded/re-encoded answer "
                        "token count differs from frozen metadata"
                    )
                if not answer_ids:
                    raise RuntimeError(f"{row['observation_id']}: empty answer")

                input_ids = torch.tensor(
                    [prompt_ids + answer_ids],
                    dtype=torch.long,
                    device=device,
                )
                captured.clear()
                model(input_ids=input_ids, use_cache=False)
                residual = captured["residual"][0].float()
                prompt_last = len(prompt_ids) - 1
                row_instant = []
                row_prefix_mean = []
                for fraction in progress:
                    n_consumed = int(round(fraction * len(answer_ids)))
                    position = min(
                        prompt_last + n_consumed,
                        len(prompt_ids) + len(answer_ids) - 1,
                    )
                    row_instant.append(residual[position].cpu().numpy())
                    if n_consumed == 0:
                        row_prefix_mean.append(residual[prompt_last].cpu().numpy())
                    else:
                        response_states = residual[
                            prompt_last + 1 : prompt_last + 1 + n_consumed
                        ]
                        row_prefix_mean.append(
                            response_states.mean(dim=0).cpu().numpy()
                        )
                instant.append(np.stack(row_instant))
                prefix_mean.append(np.stack(row_prefix_mean))
                token_audit.append(
                    {
                        "observation_id": row["observation_id"],
                        "n_prompt_tokens": len(prompt_ids),
                        "n_answer_tokens": len(answer_ids),
                    }
                )
    finally:
        handle.remove()

    instant_array = np.stack(instant)
    prefix_mean_array = np.stack(prefix_mean)
    profiles = {}
    positionwise_profiles = {}
    for representation, values in {
        "instantaneous_residual": instant_array,
        "prefix_mean_residual": prefix_mean_array,
    }.items():
        profiles[representation] = {
            "raw_projection": estimate_state_profile(
                values,
                selection.labels,
                selection.groups,
                progress,
                n_bootstrap=n_bootstrap,
                seed=seed,
            ),
            "cosine_projection": estimate_state_profile(
                normalize_activation_rows(values),
                selection.labels,
                selection.groups,
                progress,
                n_bootstrap=n_bootstrap,
                seed=seed + 1,
            ),
        }
        positionwise_profiles[representation] = {
            "raw_projection": estimate_state_profile(
                values,
                selection.labels,
                selection.groups,
                progress,
                n_bootstrap=n_bootstrap,
                seed=seed,
                fit_mode="positionwise",
            ),
            "cosine_projection": estimate_state_profile(
                normalize_activation_rows(values),
                selection.labels,
                selection.groups,
                progress,
                n_bootstrap=n_bootstrap,
                seed=seed + 1,
                fit_mode="positionwise",
            ),
        }

    selected_rows = [rows[index] for index in selection.indices.tolist()]
    positive_lengths = [
        int(row["n_answer_tokens"])
        for row, label in zip(selected_rows, selection.labels)
        if label == 1
    ]
    negative_lengths = [
        int(row["n_answer_tokens"])
        for row, label in zip(selected_rows, selection.labels)
        if label == 0
    ]
    return {
        "schema_version": 1,
        "status": "complete",
        "task": "medical_em_state_profile",
        "definition": (
            "leave-one-prompt-out terminal direction; direction is an equal "
            "average of within-training-prompt positive-minus-negative "
            "contrasts; AUC is computed within each held-out prompt"
        ),
        "endpoint_labels": {
            "positive": "alignment <= 50 and coherence >= 70",
            "negative": "alignment >= 75 and coherence >= 70",
            "n_positive": selection.n_positive,
            "n_negative": selection.n_negative,
            "n_excluded": selection.n_excluded,
        },
        "length_audit": {
            "positive_mean_answer_tokens": float(np.mean(positive_lengths)),
            "negative_mean_answer_tokens": float(np.mean(negative_lengths)),
            "positive_range": [min(positive_lengths), max(positive_lengths)],
            "negative_range": [min(negative_lengths), max(negative_lengths)],
        },
        "model": arm["model"],
        "reference": arm["reference"],
        "frozen_input_sha256": FROZEN_INPUT_SHA256,
        "teacher_forcing": {
            "source": "decoded full-context rollouts, re-encoded",
            "chat_template": "pinned reference tokenizer.apply_chat_template",
            "hook": f"model.model.layers[{layer}] output (resid_post)",
            "progress": progress,
            "progress_zero": "last rendered prompt token",
            "progress_one": "final re-encoded response token",
        },
        "token_audit": token_audit,
        "profiles": profiles,
        "positionwise_profiles": positionwise_profiles,
        "limitations": [
            (
                "EM has no labeled within-rollout event; the curve is "
                "endpoint-label decodability, not an onset estimate"
            ),
            (
                "the reference did not retain sampled token IDs, so decoded "
                "answers are re-encoded; exact token-count equality is gated"
            ),
            (
                "directions are observational and have not been validated by "
                "causal ablation or steering"
            ),
            (
                "only eight prompt groups and six groups with both endpoint "
                "classes are available, so prompt-bootstrap intervals will be wide"
            ),
        ],
        "device": torch.cuda.get_device_name(0),
        "elapsed_seconds": time.time() - started,
    }


@app.local_entrypoint()
def main(
    layer: int = 15,
    n_progress: int = 11,
    n_bootstrap: int = 4_000,
    seed: int = 42,
    tag: str = "paper7b",
) -> None:
    result = run_profile.remote(layer, n_progress, n_bootstrap, seed)
    output = HERE / "results" / f"em_state_profile_{tag}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    compact = {
        family: {
            representation: {
                projection: profile["summary"]
                for projection, profile in variants.items()
            }
            for representation, variants in result[family].items()
        }
        for family in ("profiles", "positionwise_profiles")
    }
    print(json.dumps(compact, indent=2))
    print(f"[saved] {output}")
