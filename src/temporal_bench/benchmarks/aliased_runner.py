"""Runner for the aliased paired-feature benchmark."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

import torch

from ..config import TrainConfig
from ..models import MODEL_REGISTRY, TemporalAE
from ..train import train
from ..utils import get_device, set_seed
from .aliased_data import AliasedDataConfig, AliasedDataPipeline
from .aliased_eval import AliasedEvalMetrics, evaluate_aliased_model


@dataclass
class AliasedModelEntry:
    name: str
    model_name: str
    data_kind: str
    train_config: TrainConfig
    model_kwargs: dict = field(default_factory=dict)
    window_size: int | None = None


@dataclass
class AliasedBenchmarkConfig:
    data: AliasedDataConfig = field(default_factory=AliasedDataConfig)
    rho_values: list[float] = field(default_factory=lambda: [0.0, 0.3, 0.5, 0.7, 0.9])
    k_values: list[int] = field(default_factory=lambda: [3, 10])
    dict_width: int = 40
    output_dir: str = "results/aliased_regime"
    eval_chunk_size: int = 256
    probe_steps: int = 200
    probe_lr: float = 1e-3
    probe_batch_size: int = 256
    make_plots: bool = True


def default_model_entries() -> list[AliasedModelEntry]:
    return [
        AliasedModelEntry(
            name="SAE",
            model_name="sae",
            data_kind="flat",
            train_config=TrainConfig(n_steps=30_000, batch_size=4096, lr=3e-4),
        ),
        AliasedModelEntry(
            name="BatchTopK SAE",
            model_name="batchtopk_sae",
            data_kind="flat",
            train_config=TrainConfig(n_steps=30_000, batch_size=4096, lr=3e-4),
        ),
        AliasedModelEntry(
            name="TFA",
            model_name="tfa",
            data_kind="seq",
            train_config=TrainConfig(
                n_steps=30_000,
                batch_size=64,
                lr=1e-3,
                min_lr=9e-4,
                optimizer="adamw",
                weight_decay=1e-4,
                beta1=0.9,
                beta2=0.95,
                warmup_steps=200,
                lr_schedule="cosine",
                grouped_weight_decay=True,
            ),
            model_kwargs={"n_heads": 4, "n_attn_layers": 1, "bottleneck_factor": 1},
        ),
        AliasedModelEntry(
            name="TFA-shuf",
            model_name="tfa",
            data_kind="seq_shuffled",
            train_config=TrainConfig(
                n_steps=30_000,
                batch_size=64,
                lr=1e-3,
                min_lr=9e-4,
                optimizer="adamw",
                weight_decay=1e-4,
                beta1=0.9,
                beta2=0.95,
                warmup_steps=200,
                lr_schedule="cosine",
                grouped_weight_decay=True,
            ),
            model_kwargs={"n_heads": 4, "n_attn_layers": 1, "bottleneck_factor": 1},
        ),
        AliasedModelEntry(
            name="TFA-pos",
            model_name="tfa",
            data_kind="seq",
            train_config=TrainConfig(
                n_steps=30_000,
                batch_size=64,
                lr=1e-3,
                min_lr=9e-4,
                optimizer="adamw",
                weight_decay=1e-4,
                beta1=0.9,
                beta2=0.95,
                warmup_steps=200,
                lr_schedule="cosine",
                grouped_weight_decay=True,
            ),
            model_kwargs={
                "n_heads": 4,
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
                n_steps=30_000,
                batch_size=64,
                lr=1e-3,
                min_lr=9e-4,
                optimizer="adamw",
                weight_decay=1e-4,
                beta1=0.9,
                beta2=0.95,
                warmup_steps=200,
                lr_schedule="cosine",
                grouped_weight_decay=True,
            ),
            model_kwargs={
                "n_heads": 4,
                "n_attn_layers": 1,
                "bottleneck_factor": 1,
                "use_pos_encoding": True,
            },
        ),
        AliasedModelEntry(
            name="TXCDR T=2",
            model_name="txcdr",
            data_kind="window",
            train_config=TrainConfig(n_steps=80_000, batch_size=2048, lr=3e-4),
            window_size=2,
        ),
        AliasedModelEntry(
            name="TXCDR T=5",
            model_name="txcdr",
            data_kind="window",
            train_config=TrainConfig(n_steps=80_000, batch_size=2048, lr=3e-4),
            window_size=5,
        ),
    ]


def _create_model(
    entry: AliasedModelEntry,
    d_in: int,
    d_sae: int,
    k: int,
    device: torch.device,
) -> TemporalAE:
    cls = MODEL_REGISTRY[entry.model_name]
    if entry.model_name == "sae":
        model = cls(d_in=d_in, d_sae=d_sae, k=k)
    elif entry.model_name == "batchtopk_sae":
        model = cls(d_in=d_in, d_sae=d_sae, k=k)
    elif entry.model_name == "tfa":
        model = cls(d_in=d_in, d_sae=d_sae, k=k, **entry.model_kwargs)
    elif entry.model_name == "txcdr":
        if entry.window_size is None:
            raise ValueError("TXCDR entry requires window_size")
        model = cls(d_in=d_in, d_sae=d_sae, T=entry.window_size, k_per_pos=k)
    else:
        raise ValueError(f"Unsupported model_name: {entry.model_name}")
    return model.to(device)


def _data_fn(
    pipeline: AliasedDataPipeline,
    entry: AliasedModelEntry,
    rho: float,
):
    if entry.data_kind == "flat":
        return lambda batch_size: pipeline.sample_flat(batch_size, rho)
    if entry.data_kind == "seq":
        return lambda batch_size: pipeline.sample_seq(batch_size, rho, shuffle=False)
    if entry.data_kind == "seq_shuffled":
        return lambda batch_size: pipeline.sample_seq(batch_size, rho, shuffle=True)
    if entry.data_kind == "window":
        return lambda batch_size: pipeline.sample_windows(batch_size, entry.window_size, rho)
    raise ValueError(f"Unknown data_kind: {entry.data_kind}")


def save_results(results: dict[float, dict[str, list[AliasedEvalMetrics]]], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "results.json")
    serializable = {
        str(rho): {name: [metric.to_dict() for metric in runs] for name, runs in per_rho.items()}
        for rho, per_rho in results.items()
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    return out_path


def plot_results(
    results: dict[float, dict[str, list[AliasedEvalMetrics]]],
    rho_values: list[float],
    k_values: list[int],
    output_dir: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "SAE": ("tab:blue", "o", "-"),
        "BatchTopK SAE": ("tab:cyan", "D", "-"),
        "TFA": ("tab:orange", "s", "-"),
        "TFA-shuf": ("tab:red", "^", "--"),
        "TFA-pos": ("tab:brown", "X", "-"),
        "TFA-pos-shuf": ("tab:pink", "v", "--"),
        "TXCDR T=2": ("tab:green", "P", "-"),
        "TXCDR T=5": ("tab:purple", "*", "-"),
    }
    os.makedirs(output_dir, exist_ok=True)

    for k_index, k in enumerate(k_values):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        metrics = [
            ("nmse", "NMSE vs rho"),
            ("auc", "Visible AUC vs rho"),
            ("delta", "Predictive minus local corr"),
            ("probe_auc", "Global feature recovery (probe AUC)"),
        ]
        for name, style in styles.items():
            color, marker, linestyle = style
            series = {metric_name: [] for metric_name, _ in metrics}
            for rho in rho_values:
                result = results[rho][name][k_index]
                for metric_name, _ in metrics:
                    series[metric_name].append(getattr(result, metric_name))

            for axis, (metric_name, title) in zip(axes.flat, metrics):
                axis.plot(
                    rho_values,
                    series[metric_name],
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=2,
                    markersize=7,
                    label=name,
                )
                axis.set_title(title)
                axis.set_xlabel("rho")
                axis.grid(True, alpha=0.3)

        axes[0, 0].set_yscale("log")
        axes[0, 0].legend(fontsize=8)
        axes[0, 1].legend(fontsize=8)
        axes[1, 0].legend(fontsize=8)
        axes[1, 1].legend(fontsize=8)
        fig.suptitle(f"Aliased regime benchmark (k={k})")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"aliased_benchmark_k{k}.png"), dpi=200)
        plt.close(fig)


def run_aliased_benchmark(
    config: AliasedBenchmarkConfig | None = None,
    *,
    device: torch.device | None = None,
    model_entries: list[AliasedModelEntry] | None = None,
) -> dict[float, dict[str, list[AliasedEvalMetrics]]]:
    config = config or AliasedBenchmarkConfig()
    device = device or get_device()
    model_entries = model_entries or default_model_entries()

    pipeline = AliasedDataPipeline(config.data, device=device)
    results: dict[float, dict[str, list[AliasedEvalMetrics]]] = {}

    for rho in config.rho_values:
        eval_batch = pipeline.eval_batch(rho)
        results[rho] = {entry.name: [] for entry in model_entries}
        for k in config.k_values:
            for entry in model_entries:
                set_seed(config.data.seed)
                model = _create_model(
                    entry,
                    d_in=config.data.d_model,
                    d_sae=config.dict_width,
                    k=k,
                    device=device,
                )
                history = train(
                    model=model,
                    data_fn=_data_fn(pipeline, entry, rho),
                    config=entry.train_config,
                    silent=False,
                )
                del history  # benchmark eval is separate and richer than generic evaluate()

                metrics = evaluate_aliased_model(
                    model,
                    eval_batch,
                    pipeline.true_features,
                    eval_chunk_size=config.eval_chunk_size,
                    probe_steps=config.probe_steps,
                    probe_lr=config.probe_lr,
                    probe_batch_size=config.probe_batch_size,
                )
                results[rho][entry.name].append(metrics)

    save_results(results, config.output_dir)
    if config.make_plots:
        plot_results(results, config.rho_values, config.k_values, config.output_dir)
    return results
