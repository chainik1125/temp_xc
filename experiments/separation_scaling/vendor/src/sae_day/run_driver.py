"""YAML-driven sweep runner for nonergodic-HMM SAE benchmarks.

Reads a config.yaml describing:
  - the generator (type + sweep parameter + fixed params)
  - the transformer (HookedTransformer config + training settings)
  - a list of architectures to benchmark (each with a family id + kwargs)
  - dense linear and non-linear (MLP) probes on the residual stream

For each sweep value the driver:
  1. Builds the generator,
  2. Loads or trains the transformer (toggle + saves `training_params.json`
     alongside `transformer.pt` when it trains),
  3. Extracts single-layer + multi-layer activations,
  4. Runs every architecture via family-string dispatch,
  5. Runs the linear + (optionally early-stopped) MLP probes,
  6. Writes `results.json` per cell and a combined `combined.json` at the end.

Usage:
    PYTHONPATH=src uv run python -m sae_day.run_driver --config path/to/config.yaml

This file is the canonical driver. Experiment folders just hold a `config.yaml`
plus any plot script — no per-experiment duplication of the sweep logic.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml

# ---------------------------------------------------------------------------
# Path wiring — driver lives in src/sae_day/ so repo_root = parents[2]
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _ROOT.parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
_STANDARD_HMM_ROOT = _REPO_ROOT / "experiments" / "standard_hmm"
_TRANSFORMER_STANDARD_HMM_ROOT = _REPO_ROOT / "experiments" / "transformer_standard_hmm"

for p in (_SRC_ROOT, _STANDARD_HMM_ROOT, _TRANSFORMER_STANDARD_HMM_ROOT):
    sys.path.insert(0, str(p))

from sae_day.nonergodic_generator import NonergodicGenerator  # noqa: E402
from run_standard_hmm_arch_seed_sweep import ARCH_CONFIGS, fit_linear_probe_r2  # noqa: E402
from run_transformer_standard_hmm_arch_sweep import (  # noqa: E402
    cpu_state_dict,
    default_resid_post_hook_names,
    evaluate_mlxc_on_activations,
    evaluate_tfa_on_activations,
    evaluate_topk_on_activations,
    evaluate_tsae_on_activations,
    evaluate_txc_on_activations,
    extract_multi_layer_position_activations,
    extract_position_activations,
    save_checkpoint,
    train_transformer,
)


# ---------------------------------------------------------------------------
# Generator construction
# ---------------------------------------------------------------------------


def build_generator(gen_cfg: dict, sweep_value: float) -> NonergodicGenerator:
    """Construct a NonergodicGenerator from the YAML generator block + sweep value."""
    gen_type = gen_cfg["type"]
    fixed = gen_cfg.get("fixed_params", {})
    param_block = gen_cfg.get("sweep", {})
    parameterization = fixed.get("parameterization", "symmetric_line")

    if parameterization == "symmetric_line":
        bx = fixed.get("base_x", 0.25)
        ba = fixed.get("base_a", 0.6)
        d = float(sweep_value)
        params = [
            (max(bx - d, 1e-4), min(max(ba + d, 1e-4), 1 - 1e-4)),
            (bx, ba),
            (min(max(bx + d, 1e-4), 1 - 1e-4), max(ba - d, 1e-4)),
        ]
    elif parameterization == "explicit":
        params = [(float(x), float(a)) for (x, a) in fixed["params"]]
    else:
        raise ValueError(f"Unknown parameterization: {parameterization!r}")

    if gen_type == "mess3_shared":
        return NonergodicGenerator.mess3_shared(params=params)
    if gen_type == "mess3_identity":
        r = fixed.get("r", 0.02)
        params_r = [(x, a, r) for (x, a) in params]
        return NonergodicGenerator.mess3_identity(params=params_r)
    if gen_type == "mess3_reset":
        r = fixed.get("r", 0.02)
        params_r = [(x, a, r) for (x, a) in params]
        return NonergodicGenerator.mess3_reset(params=params_r)
    raise ValueError(f"Unknown generator type: {gen_type!r}")


# ---------------------------------------------------------------------------
# Transformer build / train / load
# ---------------------------------------------------------------------------


def build_transformer(tf_cfg: dict, vocab_size: int, seed: int, device: torch.device):
    """Construct a fresh HookedTransformer from the YAML transformer block."""
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    cfg = HookedTransformerConfig(
        n_layers=int(tf_cfg["n_layers"]),
        d_model=int(tf_cfg["d_model"]),
        n_ctx=int(tf_cfg["n_ctx"]),
        d_head=int(tf_cfg.get("d_head", tf_cfg["d_model"] // tf_cfg["n_heads"])),
        n_heads=int(tf_cfg["n_heads"]),
        d_mlp=int(tf_cfg["d_mlp"]),
        d_vocab=vocab_size,
        d_vocab_out=vocab_size,
        act_fn=tf_cfg.get("act_fn", "gelu"),
        normalization_type=tf_cfg.get("normalization_type", "LN"),
        attention_dir=tf_cfg.get("attention_dir", "causal"),
        default_prepend_bos=bool(tf_cfg.get("default_prepend_bos", False)),
        device=str(device),
        seed=seed,
        init_mode=tf_cfg.get("init_mode", "gpt2"),
        initializer_range=float(tf_cfg.get("initializer_range", 0.02)),
    )
    return HookedTransformer(cfg).to(device)


def train_or_load_transformer(
    *,
    tf_cfg: dict,
    generator: NonergodicGenerator,
    seed: int,
    device: torch.device,
    cell_dir: Path,
    sweep_value: float,
    gen_cfg: dict,
):
    """Either load `cell_dir/transformer.pt` or train fresh + save.

    Returns `(transformer, final_loss, was_loaded: bool)`.
    """
    ckpt_path = cell_dir / "transformer.pt"
    load_if_exists = bool(tf_cfg.get("load_if_exists", True))
    training = tf_cfg.get("training", {})
    seq_len = int(tf_cfg["n_ctx"])

    if load_if_exists and ckpt_path.exists():
        print(f"  [load] {ckpt_path}", flush=True)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_transformer(tf_cfg, generator.vocab_size, seed, device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        final_loss = float(ckpt.get("final_loss", float("nan")))
        return model, final_loss, True

    model = build_transformer(tf_cfg, generator.vocab_size, seed, device)

    data_chunks = []
    n_chunks = int(training.get("data_chunks", 4))
    chunk_size = int(training.get("data_chunk_size", 5000))
    for ci in range(n_chunks):
        chunk_seed = seed * 1_000_000 + ci * chunk_size
        ds = generator.sample(
            n_sequences=chunk_size, seq_len=seq_len,
            seed=chunk_seed, with_component_labels=False,
        )
        data_chunks.append(ds.tokens.to(device, non_blocking=True))
    sampler_rng = torch.Generator(device=device)
    sampler_rng.manual_seed(seed * 7 + 1)

    def sample_train_tokens(step_idx: int, batch_size: int) -> torch.Tensor:
        chunk = data_chunks[step_idx % len(data_chunks)]
        idx = torch.randint(
            0, chunk.shape[0], (batch_size,),
            generator=sampler_rng, device=chunk.device,
        )
        return chunk.index_select(0, idx)

    n_steps = int(training["steps"])
    batch_size = int(training.get("batch_size", 128))
    lr = float(training.get("lr", 1e-3))
    losses = train_transformer(
        model, sample_train_tokens,
        n_steps=n_steps, batch_size=batch_size, lr=lr, seed=seed,
        print_every=max(1, n_steps // 5),
    )
    final_loss = float(losses[-1])
    print(f"  transformer final_loss={final_loss:.4f}", flush=True)

    # Save checkpoint + human-readable training_params.json alongside
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(ckpt_path, {
        "kind": "transformer",
        "seed": seed,
        "sweep_value": float(sweep_value),
        "generator": {
            "type": gen_cfg["type"],
            "vocab_size": generator.vocab_size,
            "num_components": generator.num_components,
            "component_weights": list(generator.component_weights),
            "components_spec": generator.components_spec,
        },
        "config": dict(tf_cfg),
        "final_loss": final_loss,
        "state_dict": cpu_state_dict(model),
    })
    (cell_dir / "training_params.json").write_text(json.dumps({
        "seed": seed,
        "sweep_value": float(sweep_value),
        "generator": {
            "type": gen_cfg["type"],
            "sweep_parameter": gen_cfg.get("sweep", {}).get("parameter"),
            "fixed_params": gen_cfg.get("fixed_params", {}),
        },
        "transformer": dict(tf_cfg),
        "final_loss": final_loss,
    }, indent=2))
    return model, final_loss, False


# ---------------------------------------------------------------------------
# Dense probes
# ---------------------------------------------------------------------------


def fit_mlp_probe_r2(
    x_all: np.ndarray,
    y_all: np.ndarray,
    n_sequences: int,
    samples_per_sequence: int,
    hidden: int = 64,
    epochs: int = 100,
    early_stop_patience: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    seed: int = 42,
    device: torch.device | None = None,
) -> dict:
    """2-layer ReLU MLP with early stopping on a 90/10 train/val split from the
    training slice (eval R² is on the held-out sequence split, 80/20)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_sequences)
    split = int(0.8 * n_sequences)
    seq_is_train = np.zeros(n_sequences, dtype=bool)
    seq_is_train[perm[:split]] = True
    flat_train = np.repeat(seq_is_train, samples_per_sequence)
    flat_test = ~flat_train

    # Further split train into train/val for early stopping
    train_idx = np.where(flat_train)[0]
    rng2 = np.random.default_rng(seed + 1)
    rng2.shuffle(train_idx)
    val_split = int(0.1 * len(train_idx))
    val_idx = train_idx[:val_split]
    tr_idx = train_idx[val_split:]

    x_tr = torch.from_numpy(x_all[tr_idx].astype(np.float32))
    y_tr = torch.from_numpy(y_all[tr_idx].astype(np.float32))
    x_val = torch.from_numpy(x_all[val_idx].astype(np.float32))
    y_val = torch.from_numpy(y_all[val_idx].astype(np.float32))
    x_te = torch.from_numpy(x_all[flat_test].astype(np.float32))
    y_te = torch.from_numpy(y_all[flat_test].astype(np.float32))

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_tr = x_tr.to(dev); y_tr = y_tr.to(dev)
    x_val = x_val.to(dev); y_val = y_val.to(dev)
    x_te = x_te.to(dev); y_te = y_te.to(dev)

    d_in, d_out = x_all.shape[1], y_all.shape[1]
    torch.manual_seed(seed)
    mlp = nn.Sequential(
        nn.Linear(d_in, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, d_out),
    ).to(dev)
    opt = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=weight_decay)

    batch_size = min(4096, x_tr.shape[0])
    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        mlp.train()
        perm_t = torch.randperm(x_tr.shape[0], device=dev)
        for start in range(0, x_tr.shape[0], batch_size):
            idx = perm_t[start:start + batch_size]
            preds = mlp(x_tr[idx])
            loss = torch.nn.functional.mse_loss(preds, y_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        mlp.eval()
        with torch.no_grad():
            val_loss = torch.nn.functional.mse_loss(mlp(x_val), y_val).item()
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in mlp.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= early_stop_patience:
            break

    if best_state is not None:
        mlp.load_state_dict(best_state)
    mlp.eval()
    with torch.no_grad():
        y_pred = mlp(x_te).cpu().numpy()
    y_te_np = y_te.cpu().numpy()
    y_mean_train = y_tr.mean(dim=0).cpu().numpy()
    ss_res = ((y_te_np - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_te_np - y_mean_train) ** 2).sum(axis=0) + 1e-12
    r2_per = 1.0 - ss_res / ss_tot
    return {
        "per_component_r2": [float(v) for v in r2_per],
        "mean_r2": float(r2_per.mean()),
        "hidden": hidden, "epochs_trained": epoch + 1, "early_stop_patience": early_stop_patience,
        "best_val_mse": float(best_val),
    }


# ---------------------------------------------------------------------------
# Architecture dispatch
# ---------------------------------------------------------------------------


def _merge_kwargs(arch_name: str, kwargs: dict) -> dict:
    """Start from ARCH_CONFIGS[arch_name] if registered, overlay YAML kwargs."""
    base = dict(ARCH_CONFIGS[arch_name]) if arch_name in ARCH_CONFIGS else {}
    base.update(kwargs or {})
    return base


def run_architecture(
    arch_entry: dict,
    *,
    seed: int,
    train_acts: torch.Tensor,
    train_acts_ml: torch.Tensor,
    eval_acts: torch.Tensor,
    eval_acts_ml: torch.Tensor,
    eval_omega: torch.Tensor,
    d_model: int,
    device: torch.device,
    checkpoint_path: Path | None = None,
) -> dict:
    name = arch_entry["name"]
    family = arch_entry["family"]
    arch_cfg = _merge_kwargs(name, arch_entry.get("kwargs", {}))
    if family == "topk":
        return evaluate_topk_on_activations(
            seed, arch_cfg, train_acts, eval_acts, eval_omega,
            d_model=d_model, device=device, checkpoint_path=checkpoint_path,
        )
    if family == "txc":
        return evaluate_txc_on_activations(
            seed, arch_cfg, train_acts, eval_acts, eval_omega,
            d_model=d_model, device=device, matryoshka=False, checkpoint_path=checkpoint_path,
        )
    if family == "mattxc":
        return evaluate_txc_on_activations(
            seed, arch_cfg, train_acts, eval_acts, eval_omega,
            d_model=d_model, device=device, matryoshka=True, checkpoint_path=checkpoint_path,
        )
    if family == "mlxc":
        return evaluate_mlxc_on_activations(
            seed, arch_cfg, train_acts_ml, eval_acts_ml, eval_omega,
            d_model=d_model, device=device, matryoshka=False, checkpoint_path=checkpoint_path,
        )
    if family == "matmlxc":
        return evaluate_mlxc_on_activations(
            seed, arch_cfg, train_acts_ml, eval_acts_ml, eval_omega,
            d_model=d_model, device=device, matryoshka=True, checkpoint_path=checkpoint_path,
        )
    if family == "tsae":
        return evaluate_tsae_on_activations(
            seed, arch_cfg, train_acts, eval_acts, eval_omega,
            d_model=d_model, device=device, checkpoint_path=checkpoint_path,
        )
    if family == "tfa":
        return evaluate_tfa_on_activations(
            seed, arch_cfg, train_acts, eval_acts, eval_omega,
            d_model=d_model, device=device,
            use_pos_encoding=bool(arch_cfg.get("use_pos_encoding", False)),
            checkpoint_path=checkpoint_path,
        )
    raise ValueError(f"Unknown architecture family: {family!r}")


# ---------------------------------------------------------------------------
# Per-cell runner
# ---------------------------------------------------------------------------


def run_one_cell(
    *,
    sweep_value: float,
    cfg: dict,
    cell_dir: Path,
    device: torch.device,
) -> dict:
    t0 = time.time()
    seed = int(cfg["seed"])
    tf_cfg = cfg["transformer"]
    gen_cfg = cfg["generator"]
    sweep_param = gen_cfg.get("sweep", {}).get("parameter", "value")
    print(f"\n=== CELL {sweep_param}={sweep_value} ===", flush=True)

    cell_dir.mkdir(parents=True, exist_ok=True)
    generator = build_generator(gen_cfg, sweep_value)
    vocab_size = generator.vocab_size
    print(f"  generator components={generator.components_spec}  vocab={vocab_size}", flush=True)

    transformer, tf_final_loss, was_loaded = train_or_load_transformer(
        tf_cfg=tf_cfg, generator=generator, seed=seed, device=device,
        cell_dir=cell_dir, sweep_value=sweep_value, gen_cfg=gen_cfg,
    )

    # Sample probe data
    pd_cfg = cfg.get("probe_dataset", {})
    n_train = int(pd_cfg.get("train_sequences", 1000))
    n_eval = int(pd_cfg.get("eval_sequences", 500))
    seq_len = int(tf_cfg["n_ctx"])
    train_data = generator.sample(
        n_sequences=n_train, seq_len=seq_len,
        seed=int(pd_cfg.get("train_seed", seed + 10_000)),
        with_component_labels=True,
    )
    eval_data = generator.sample(
        n_sequences=n_eval, seq_len=seq_len,
        seed=int(pd_cfg.get("eval_seed", seed + 20_000)),
        with_component_labels=True,
    )
    eval_omega = eval_data.sequence_omegas

    # Extract activations
    probe_layer = int(tf_cfg.get("probe_layer", 1))
    hook_name = tf_cfg.get("probe_hook_name", f"blocks.{probe_layer}.hook_resid_post")
    extract_bs = int(tf_cfg.get("extract_batch_size", 16))
    train_acts = extract_position_activations(
        transformer, train_data.tokens, hook_name=hook_name, chunk_size=extract_bs,
    )
    eval_acts = extract_position_activations(
        transformer, eval_data.tokens, hook_name=hook_name, chunk_size=extract_bs,
    )
    multi_hooks = default_resid_post_hook_names(int(tf_cfg["n_layers"]))
    train_acts_ml = extract_multi_layer_position_activations(
        transformer, train_data.tokens, multi_hooks, chunk_size=extract_bs,
    )
    eval_acts_ml = extract_multi_layer_position_activations(
        transformer, eval_data.tokens, multi_hooks, chunk_size=extract_bs,
    )

    with torch.no_grad():
        eval_resid_l2 = float(eval_acts.norm(dim=-1).mean().item())

    # Dense probes
    d_model = int(tf_cfg["d_model"])
    flat_resid = eval_acts.reshape(-1, d_model).numpy()
    flat_omega = (
        eval_omega[:, None, :].expand(n_eval, seq_len, eval_omega.shape[-1])
        .reshape(-1, eval_omega.shape[-1]).numpy()
    )
    probes_cfg = cfg.get("probes", {})
    dense_linear = None
    dense_mlp = None
    if probes_cfg.get("dense_linear", {}).get("enabled", True):
        dense_linear = fit_linear_probe_r2(
            flat_resid, flat_omega,
            n_sequences=n_eval, samples_per_sequence=seq_len, seed=seed,
        )
        print(
            f"  dense_linear mean_r2={dense_linear['mean_r2']:.4f}  "
            f"per_comp={[round(float(v), 3) for v in dense_linear['per_component_r2']]}",
            flush=True,
        )
    mlp_cfg = probes_cfg.get("dense_mlp", {})
    if mlp_cfg.get("enabled", True):
        dense_mlp = fit_mlp_probe_r2(
            flat_resid, flat_omega,
            n_sequences=n_eval, samples_per_sequence=seq_len,
            hidden=int(mlp_cfg.get("hidden", 64)),
            epochs=int(mlp_cfg.get("epochs", 100)),
            early_stop_patience=int(mlp_cfg.get("early_stop_patience", 10)),
            lr=float(mlp_cfg.get("lr", 1e-3)),
            weight_decay=float(mlp_cfg.get("weight_decay", 1e-3)),
            seed=seed, device=device,
        )
        print(
            f"  dense_mlp mean_r2={dense_mlp['mean_r2']:.4f}  "
            f"per_comp={[round(float(v), 3) for v in dense_mlp['per_component_r2']]}  "
            f"(stopped at epoch {dense_mlp['epochs_trained']})",
            flush=True,
        )

    # Release transformer GPU memory before running SAE arches
    del transformer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Architectures
    arch_results: dict[str, dict] = {}
    arch_dir = cell_dir / "arch_checkpoints"
    for arch_entry in cfg.get("architectures", []):
        name = arch_entry["name"]
        print(f"  >>> {name}  family={arch_entry['family']}", flush=True)
        ckpt_path = arch_dir / f"{_slugify(name)}.pt" if arch_entry.get("save_checkpoint") else None
        run = run_architecture(
            arch_entry,
            seed=seed,
            train_acts=train_acts, train_acts_ml=train_acts_ml,
            eval_acts=eval_acts, eval_acts_ml=eval_acts_ml,
            eval_omega=eval_omega,
            d_model=d_model, device=device,
            checkpoint_path=ckpt_path,
        )
        m = run["metrics"]["single_feature_probe"]
        lp = run["metrics"]["linear_probe"]
        arch_results[name] = {
            "best_single_r2": float(m["best_feature_r2"]),
            "per_component_best_r2": list(m["per_component_best_r2"]),
            "linear_mean_r2": float(lp["mean_r2"]),
            "linear_per_component_r2": list(lp.get("per_component_r2", [])),
            "sae_final_loss": float(run["final_loss"]),
            "family": arch_entry["family"],
            "config": arch_entry.get("kwargs", {}),
        }
        print(
            f"    best_single={m['best_feature_r2']:.4f}  "
            f"per_comp={[round(float(v),3) for v in m['per_component_best_r2']]}  "
            f"lin_mean={lp['mean_r2']:.4f}",
            flush=True,
        )

    elapsed = time.time() - t0
    payload = {
        "sweep_parameter": sweep_param,
        "sweep_value": float(sweep_value),
        "generator_params": [list(p) for p in generator.components_spec],
        "transformer": {
            "final_loss": tf_final_loss,
            "was_loaded": was_loaded,
            "eval_resid_l2_mean": eval_resid_l2,
            "config": dict(tf_cfg),
        },
        "dense_linear": dense_linear,
        "dense_mlp": dense_mlp,
        "architectures": arch_results,
        "cell_elapsed_sec": elapsed,
    }
    out = cell_dir / "results.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"  saved {out}  elapsed={elapsed/60:.1f} min", flush=True)
    return payload


def _slugify(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "_")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="YAML-driven SAE benchmark runner")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True,
        help="If a cell's results.json already exists, skip it.",
    )
    args = parser.parse_args()

    cfg_path = args.config.resolve()
    cfg = yaml.safe_load(cfg_path.read_text())
    root = cfg_path.parent
    output_root = root / cfg.get("output_root", "results")
    output_root.mkdir(parents=True, exist_ok=True)

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== {cfg.get('experiment_name', cfg_path.stem)} / device={device} ===", flush=True)

    sweep_block = cfg["generator"]["sweep"]
    param = sweep_block.get("parameter", "value")
    values = list(sweep_block["values"])
    print(f"Sweep: {param} ∈ {values}", flush=True)

    cells: dict[str, dict] = {}
    for v in values:
        cell_dir = output_root / f"cell_{param}_{v:g}"
        results_json = cell_dir / "results.json"
        if args.skip_existing and results_json.exists():
            print(f"[skip-existing] {cell_dir}", flush=True)
            cells[str(v)] = json.loads(results_json.read_text())
            continue
        cells[str(v)] = run_one_cell(
            sweep_value=v, cfg=cfg, cell_dir=cell_dir, device=device,
        )

    combined = output_root / "combined.json"
    combined.write_text(json.dumps({
        "experiment_name": cfg.get("experiment_name"),
        "sweep_parameter": param,
        "sweep_values": values,
        "cells": cells,
    }, indent=2))
    print(f"\nSaved combined {combined}", flush=True)


if __name__ == "__main__":
    main()
