"""Train TopK SAE, TXC, and MLC on Gemma-2B-it activations over Python code.

Usage::

    cd experiments/code_benchmark
    PYTHONPATH=../separation_scaling/vendor/src \
        uv run python run_training.py --config config.yaml \
            [--only topk_sae] [--device cuda] [--skip-cache]

All outputs go under ``checkpoints/`` and ``results/`` relative to this
directory. Loss curves are saved both as JSON and PNG.

The script is deliberately explicit about each matching axis: after training
each model it logs ``d_sae``, ``k_total``, training-token budget, and optimizer
hyperparameters into the checkpoint — the fairness audit mentioned in the plan
reads these.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from code_pipeline.python_code import (  # noqa: E402
    CodeDatasetConfig,
    SubjectModelConfig,
    build_cache,
    cache_paths,
    load_cache,
    train_eval_split_indices,
)
from code_pipeline.training import (  # noqa: E402
    FAMILY_BUILDERS,
    flatten_for_sae,
    make_txc_windows,
    stack_mlc_layers,
    train_one_architecture,
)


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def plot_loss_curves(loss_histories: dict[str, list[float]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, losses in loss_histories.items():
        ax.plot(losses, label=name, linewidth=1.0)
    ax.set_xlabel("step")
    ax.set_ylabel("recon loss (sum-over-d MSE)")
    ax.set_yscale("log")
    ax.set_title("Training loss — code_benchmark")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--only", type=str, default=None,
                        help="If set, train only the architecture with this name.")
    parser.add_argument("--device", type=str, default=None,
                        help="Override config's device (e.g. cuda, cpu, mps).")
    parser.add_argument("--skip-cache", action="store_true",
                        help="Fail fast if the activation cache is missing "
                             "(do not attempt to rebuild it).")
    parser.add_argument("--extract-batch-size", type=int, default=8)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    seed = int(cfg.get("seed", 42))
    device = resolve_device(args.device or cfg.get("device", "auto"))
    print(f"[run_training] device={device}")

    cache_root = HERE / cfg.get("cache_root", "cache")
    checkpoint_root = HERE / cfg.get("checkpoint_root", "checkpoints")
    results_root = HERE / cfg.get("output_root", "results")
    plot_root = HERE / cfg.get("plot_root", "plots")

    data_cfg = CodeDatasetConfig.from_dict(cfg["dataset"])
    subject_cfg = SubjectModelConfig.from_dict(cfg["subject_model"])

    # ---- cache ----
    if not args.skip_cache:
        build_cache(
            cache_root,
            data_cfg,
            subject_cfg,
            device=str(device),
            dtype_str=cfg.get("dtype", "bfloat16"),
            extract_batch_size=args.extract_batch_size,
        )
    layers = subject_cfg.required_layers()
    tokens, sources, acts_by_layer, manifest = load_cache(cache_root, layers)
    print(f"[run_training] cache loaded: {tokens.shape[0]} chunks, "
          f"T={tokens.shape[1]}, layers={layers}")

    # Keep activations in their cached dtype (bf16 by default) on CPU — per-
    # batch upcast to float32 happens inside the training loop on-GPU. Storing
    # five layers × 2.4 GB bf16 here is ~12 GB; upcasting all of them would
    # blow the container cgroup (~47 GiB on Simple Research A40 boxes) once
    # Gemma is also loaded. See 2026-04-23 investigation.
    anchor_acts = acts_by_layer[subject_cfg.anchor_layer]
    mlc_acts = {L: acts_by_layer[L] for L in subject_cfg.mlc_layers}

    # Train/eval split on chunk dim
    train_idx, eval_idx = train_eval_split_indices(
        anchor_acts.shape[0], data_cfg.train_eval_split, seed
    )
    print(f"[run_training] train_chunks={len(train_idx)} eval_chunks={len(eval_idx)}")

    # Save the split so eval scripts use the same partition
    split_path = cache_root / "split.pt"
    torch.save({"train_idx": train_idx, "eval_idx": eval_idx, "seed": seed}, split_path)

    d_model = subject_cfg.d_model
    loss_histories: dict[str, list[float]] = {}
    summary_rows: list[dict] = []

    for arch in cfg["architectures"]:
        name = arch["name"]
        if args.only and name != args.only:
            continue
        family = arch["family"]
        kw = dict(arch["kwargs"])

        # ---- package training data per architecture ----
        if family == "topk":
            data = flatten_for_sae(anchor_acts[train_idx])
        elif family == "txc":
            data = make_txc_windows(anchor_acts[train_idx], kw["T"])
        elif family == "mlxc":
            data = stack_mlc_layers({L: mlc_acts[L][train_idx] for L in subject_cfg.mlc_layers},
                                    subject_cfg.mlc_layers)
        else:
            raise ValueError(f"Unknown architecture family: {family!r}")
        print(f"[run_training] {name}: family={family} data shape={tuple(data.shape)}")

        builder = FAMILY_BUILDERS[family]
        model = builder(d_model, kw)

        t0 = time.time()
        train_out = train_one_architecture(
            model, data,
            n_steps=kw["steps"],
            batch_size=kw["batch_size"],
            lr=kw["lr"],
            device=device,
            seed=seed,
        )
        elapsed = time.time() - t0

        loss_histories[name] = train_out["loss_history"]
        summary_rows.append({
            "name": name, "family": family,
            "d_sae": kw["d_sae"], "k_total": kw.get("k_total", kw.get("k", None)),
            "T": kw.get("T"), "L": kw.get("L"),
            "steps": kw["steps"], "batch_size": kw["batch_size"], "lr": kw["lr"],
            "n_train_samples": int(data.shape[0]),
            "training_tokens": int(data.shape[0] * kw["batch_size"]),  # approx proxy
            "final_loss": train_out["loss_history"][-1],
            "best_loss": train_out["best_loss"],
            "elapsed_sec": round(elapsed, 1),
        })

        ckpt_path = checkpoint_root / f"{name}.pt"
        save_checkpoint(ckpt_path, {
            "name": name,
            "family": family,
            "config": kw,
            "subject_model": subject_cfg.name,
            "anchor_layer": subject_cfg.anchor_layer,
            "mlc_layers": subject_cfg.mlc_layers,
            "d_model": d_model,
            "seed": seed,
            "train_out": {
                "loss_history": train_out["loss_history"],
                "l0_history": train_out["l0_history"],
            },
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        })
        print(f"[run_training] saved {ckpt_path} in {elapsed:.1f}s "
              f"(final loss {train_out['loss_history'][-1]:.6f})")

    # ---- summaries ----
    results_root.mkdir(parents=True, exist_ok=True)
    summary_path = results_root / "training_summary.json"
    with summary_path.open("w") as f:
        json.dump({"runs": summary_rows, "manifest": manifest,
                   "seed": seed}, f, indent=2)
    plot_loss_curves(loss_histories, plot_root / "loss_curves.png")

    print(f"[run_training] done. summary → {summary_path}")


if __name__ == "__main__":
    main()
