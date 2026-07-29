"""Train a conventional TopK SAE and measure Ward feature formation.

The expensive teacher-forced activations and the unsupervised SAE checkpoint
are cached in the existing Ward Modal volume.  Behaviour labels are used only
after SAE training, for paired held-out readouts.

Run from the repository root:

    modal run experiments/temporal_screen_1/feature_formation/modal_ward_sae.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2] if len(HERE.parents) > 2 else HERE
DATA = REPO / "results" / "ward_backtracking"
RESULT = HERE / "results" / "ward_sae_features.json"

app = modal.App("temporal-screen-feature-formation-sae")
hf_secret = modal.Secret.from_name("hf-token")
cache_volume = modal.Volume.from_name(
    "temporal-screen-ward-weak-label-cache",
    create_if_missing=True,
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.7.1",
        "transformers==4.53.3",
        "accelerate==1.9.0",
        "huggingface-hub>=0.33",
        "hf-xet>=1.1",
        "numpy==2.2.6",
        "scikit-learn==1.7.1",
    )
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_dir(
        str(HERE),
        "/work/experiments/temporal_screen_1/feature_formation",
    )
    .add_local_dir(str(REPO / "src"), "/work/src")
    .add_local_file(str(DATA / "traces.json"), "/work/data/traces.json")
)


@app.function(
    image=image,
    gpu="A100-40GB",
    cpu=12,
    memory=65_536,
    timeout=3 * 60 * 60,
    secrets=[hf_secret],
    volumes={"/cache": cache_volume},
)
def run() -> dict:
    import sys
    import time
    from collections import Counter
    from dataclasses import asdict

    import numpy as np
    import torch
    import torch.nn.functional as functional
    from transformers import AutoModel, AutoTokenizer

    sys.path.insert(0, "/work")
    from experiments.temporal_screen_1.feature_formation.estimators import (
        curve_to_dict,
        summarize_curve,
    )
    from experiments.temporal_screen_1.feature_formation.run_sae_free import (
        build_paired_panel,
    )
    from experiments.temporal_screen_1.feature_formation.sae_features import (
        positionwise_sae_curve,
        transported_sae_curve,
    )
    from src.bench.architectures.topk_sae import TopKSAE

    started = time.time()
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    layer = 10
    d_model = 4096
    d_sae = 8192
    k = 32
    train_steps = 1_200
    train_batch = 192
    sample_per_rollout = 256
    learning_rate = 3e-4
    seed = 20260729
    offsets = np.asarray(
        list(range(-64, -32, 4)) + list(range(-32, 17)),
        dtype=np.int64,
    )
    activation_cache_path = Path(
        "/cache/ward_feature_formation_layer10_full_panel_v1.pt"
    )
    sae_cache_path = Path(
        "/cache/ward_feature_formation_layer10_topk8192_k32_v1.pt"
    )
    result_cache_path = Path(
        "/cache/ward_feature_formation_layer10_topk8192_k32_result_v1.json"
    )
    projected_cache = torch.load(
        "/cache/ward_deepseek8b_layer10_rademacher32_seed20260729.pt",
        map_location="cpu",
        weights_only=False,
    )
    _, pair_records = build_paired_panel(projected_cache, offsets)
    pair_by_qid = {row["qid"]: row for row in pair_records}
    traces = json.loads(Path("/work/data/traces.json").read_text())

    torch.manual_seed(seed)
    np.random.seed(seed)
    if activation_cache_path.exists():
        activation_cache = torch.load(
            activation_cache_path,
            map_location="cpu",
            weights_only=False,
        )
        train_activations = activation_cache["train_activations"]
        paired_activations = activation_cache["paired_activations"]
        cached_pair_records = activation_cache["pair_records"]
        if [row["qid"] for row in cached_pair_records] != [
            row["qid"] for row in pair_records
        ]:
            raise RuntimeError("cached pair panel does not match current pairing")
        print(
            f"[cache] full activations train={tuple(train_activations.shape)} "
            f"panel={tuple(paired_activations.shape)}",
            flush=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        ).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if int(model.config.hidden_size) != d_model:
            raise RuntimeError("unexpected DeepSeek hidden size")
        captured = {}

        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            captured["activation"] = value.detach()

        handle = model.layers[layer].register_forward_hook(hook)
        sampled_rows = []
        paired_by_qid = {}
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        try:
            for trace_index, trace in enumerate(traces):
                qid = str(trace["question_id"])
                encoded = tokenizer(
                    trace["full_response"],
                    add_special_tokens=False,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to("cuda")
                with torch.inference_mode():
                    _ = model(input_ids=input_ids, use_cache=False)
                acts = captured["activation"][0]
                n_tokens = int(acts.shape[0])
                n_sample = min(sample_per_rollout, n_tokens)
                sample_index = torch.randperm(
                    n_tokens,
                    generator=generator,
                )[:n_sample].to(acts.device)
                sampled_rows.append(
                    acts[sample_index].to(torch.float16).cpu()
                )
                if qid in pair_by_qid:
                    row = pair_by_qid[qid]
                    event_positions = torch.as_tensor(
                        row["event_position"] + offsets,
                        device=acts.device,
                    )
                    neutral_positions = torch.as_tensor(
                        row["neutral_position"] + offsets,
                        device=acts.device,
                    )
                    if (
                        int(torch.min(event_positions)) < 0
                        or int(torch.max(event_positions)) >= n_tokens
                        or int(torch.min(neutral_positions)) < 0
                        or int(torch.max(neutral_positions)) >= n_tokens
                    ):
                        raise RuntimeError(f"pair indices out of range for {qid}")
                    paired_by_qid[qid] = torch.stack(
                        [
                            acts[event_positions],
                            acts[neutral_positions],
                        ],
                        dim=0,
                    ).to(torch.float16).cpu()
                captured.clear()
                if (trace_index + 1) % 25 == 0:
                    print(
                        f"[extract] {trace_index + 1}/{len(traces)} "
                        f"sampled={sum(len(row) for row in sampled_rows)}",
                        flush=True,
                    )
        finally:
            handle.remove()
        train_activations = torch.cat(sampled_rows, dim=0)
        paired_activations = torch.stack(
            [paired_by_qid[row["qid"]] for row in pair_records],
            dim=0,
        )
        temporary = activation_cache_path.with_suffix(".tmp")
        torch.save(
            {
                "train_activations": train_activations,
                "paired_activations": paired_activations,
                "pair_records": pair_records,
                "offsets": offsets.tolist(),
                "model": model_id,
                "layer": layer,
            },
            temporary,
        )
        temporary.replace(activation_cache_path)
        cache_volume.commit()
        print(f"[cache] wrote {activation_cache_path}", flush=True)
        del model, tokenizer, sampled_rows, paired_by_qid
        torch.cuda.empty_cache()

    sae = TopKSAE(d_in=d_model, d_sae=d_sae, k=k).to("cuda")
    if sae_cache_path.exists():
        checkpoint = torch.load(
            sae_cache_path,
            map_location="cpu",
            weights_only=False,
        )
        sae.load_state_dict(checkpoint["state_dict"])
        training_log = checkpoint["training_log"]
        print(f"[cache] loaded SAE {sae_cache_path}", flush=True)
    else:
        with torch.no_grad():
            sae.b_dec.copy_(
                train_activations.float().mean(dim=0).to("cuda")
            )
        optimizer = torch.optim.Adam(sae.parameters(), lr=learning_rate)
        train_generator = torch.Generator(device="cpu")
        train_generator.manual_seed(seed + 1)
        training_log = []
        sae.train()
        for step in range(train_steps):
            index = torch.randint(
                len(train_activations),
                (train_batch,),
                generator=train_generator,
            )
            batch = train_activations[index].to(
                "cuda",
                dtype=torch.float32,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, reconstruction, latent = sae(batch)
                loss = functional.mse_loss(reconstruction, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            optimizer.step()
            sae._normalize_decoder()
            if step % 100 == 0 or step == train_steps - 1:
                l0 = float((latent > 0).float().sum(dim=-1).mean())
                row = {
                    "step": int(step),
                    "mse": float(loss.detach()),
                    "l0": l0,
                }
                training_log.append(row)
                print(f"[sae] {json.dumps(row, sort_keys=True)}", flush=True)
        sae.eval()
        temporary = sae_cache_path.with_suffix(".tmp")
        torch.save(
            {
                "state_dict": {
                    key: value.detach().to(torch.float16).cpu()
                    for key, value in sae.state_dict().items()
                },
                "training_log": training_log,
                "config": {
                    "d_model": d_model,
                    "d_sae": d_sae,
                    "k": k,
                    "steps": train_steps,
                    "batch_size": train_batch,
                    "learning_rate": learning_rate,
                },
            },
            temporary,
        )
        temporary.replace(sae_cache_path)
        cache_volume.commit()

    sae.eval()
    evaluation_rows = train_activations[: min(8_192, len(train_activations))]
    sum_error = 0.0
    sum_centered = 0.0
    alive = torch.zeros(d_sae, dtype=torch.bool, device="cuda")
    train_mean = train_activations.float().mean(dim=0).to("cuda")
    with torch.inference_mode():
        for start in range(0, len(evaluation_rows), 256):
            batch = evaluation_rows[start : start + 256].to(
                "cuda",
                dtype=torch.float32,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                latent = sae.encode(batch)
                reconstruction = sae.decode(latent)
            sum_error += float(torch.sum((reconstruction - batch) ** 2))
            sum_centered += float(torch.sum((batch - train_mean) ** 2))
            alive |= torch.any(latent > 0, dim=0)
    reconstruction_metrics = {
        "fvu_centered": sum_error / max(sum_centered, 1e-12),
        "alive_fraction_8192_token_sample": float(alive.float().mean()),
        "realized_l0": float(k),
    }
    print(
        f"[sae-eval] {json.dumps(reconstruction_metrics, sort_keys=True)}",
        flush=True,
    )

    flat_panel = paired_activations.reshape(-1, d_model)
    sparse_indices = torch.empty(
        (len(flat_panel), k),
        dtype=torch.int32,
    )
    sparse_values = torch.empty(
        (len(flat_panel), k),
        dtype=torch.float16,
    )
    with torch.inference_mode():
        for start in range(0, len(flat_panel), 256):
            end = min(start + 256, len(flat_panel))
            batch = flat_panel[start:end].to("cuda", dtype=torch.float32)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                latent = sae.encode(batch)
            value, index = torch.topk(latent, k=k, dim=-1)
            sparse_indices[start:end] = index.to(torch.int32).cpu()
            sparse_values[start:end] = value.to(torch.float16).cpu()
    sparse_shape = (*paired_activations.shape[:-1], k)
    indices_np = sparse_indices.reshape(sparse_shape).numpy()
    values_np = sparse_values.reshape(sparse_shape).float().numpy()

    positionwise = positionwise_sae_curve(
        indices_np,
        values_np,
        offsets,
        d_sae=d_sae,
        top_n=16,
        regularization=0.1,
        seed=seed,
    )
    transported_1 = transported_sae_curve(
        indices_np,
        values_np,
        offsets,
        discovery_band=(-13, -8),
        d_sae=d_sae,
        top_n=1,
        regularization=0.1,
        seed=seed,
    )
    transported_16 = transported_sae_curve(
        indices_np,
        values_np,
        offsets,
        discovery_band=(-13, -8),
        d_sae=d_sae,
        top_n=16,
        regularization=0.1,
        seed=seed,
    )

    def top_counts(curve, n=25):
        return [
            {"feature": int(feature), "fold_selections": int(count)}
            for feature, count in Counter(
                curve.selected_feature_counts
            ).most_common(n)
        ]

    payload = {
        "method": "conventional TopK SAE feature-formation calibration",
        "model": model_id,
        "layer": layer,
        "sae": {
            "architecture": "single-token TopK SAE",
            "d_model": d_model,
            "d_sae": d_sae,
            "k": k,
            "train_steps": train_steps,
            "train_batch": train_batch,
            "train_tokens_available": int(len(train_activations)),
            "sample_per_rollout": sample_per_rollout,
            "learning_rate": learning_rate,
            "behavior_labels_seen_during_sae_training": False,
            "training_log": training_log,
            "reconstruction": reconstruction_metrics,
        },
        "pairing": {
            "n_pairs": len(pair_records),
            "records": pair_records,
            "same_rollout_neutral": True,
        },
        "offsets": offsets.tolist(),
        "discovery_band": [-13, -8],
        "curves": {
            "positionwise_top16": curve_to_dict(positionwise.points),
            "transported_single_feature": curve_to_dict(
                transported_1.points
            ),
            "transported_top16": curve_to_dict(transported_16.points),
        },
        "summaries": {
            "positionwise_top16": asdict(
                summarize_curve(positionwise.points)
            ),
            "transported_single_feature": asdict(
                summarize_curve(transported_1.points)
            ),
            "transported_top16": asdict(
                summarize_curve(transported_16.points)
            ),
        },
        "feature_selection": {
            "transported_single_feature_top_counts": top_counts(
                transported_1
            ),
            "transported_top16_top_counts": top_counts(transported_16),
        },
        "runtime": {
            "gpu": torch.cuda.get_device_name(0),
            "wall_seconds": time.time() - started,
        },
        "caveat": (
            "Event-aligned observational calibration; conventional SAE "
            "feature prediction is not by itself a causal mechanism."
        ),
    }
    result_cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    cache_volume.commit()
    print(
        json.dumps(
            {
                "n_pairs": len(pair_records),
                "reconstruction": reconstruction_metrics,
                "summaries": payload["summaries"],
                "wall_seconds": payload["runtime"]["wall_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return payload


@app.local_entrypoint()
def main():
    payload = run.remote()
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[saved] {RESULT}")
