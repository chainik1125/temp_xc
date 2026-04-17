"""Sweep runner — run architecture comparisons across parameter grids.

Provides run_sweep() for programmatic use and a CLI for command-line sweeps.

Usage:
    # Full sweep (all combos):
    python -m src.bench.sweep

    # Single run:
    python -m src.bench.sweep --rho 0.9 --k 5 --T 2

    # Specific models only:
    python -m src.bench.sweep --models topk_sae crosscoder

    # Quick test:
    python -m src.bench.sweep --steps 100

    # Leaky reset:
    python -m src.bench.sweep --delta 0.5

    # Coupled features (K=10 hidden, M=20 emissions):
    python -m src.bench.sweep --coupled --K-hidden 10 --M-emission 20 --n-parents 2

    # Dry run:
    python -m src.bench.sweep --dry-run
"""

import argparse
import itertools
import json
import os
import time

import torch

from src.bench.architectures import get_default_models, ModelEntry
from src.bench.config import (
    CouplingConfig,
    DataConfig,
    MarkovConfig,
    TrainConfig,
    SweepConfig,
)
from src.bench.data import build_data_pipeline, build_pipeline, DataPipeline
from src.bench.eval import evaluate_model, EvalResult
from src.bench.model_registry import get_model_config, list_models


def _arch_registry_key(entry: ModelEntry) -> str:
    """Map a ModelEntry to its registry key (topk_sae, stacked_sae, crosscoder, tfa, tfa_pos).

    get_default_models() labels entries with display names ("TopKSAE",
    "Stacked T=5", "TXCDR T=5") but users pass registry keys via --models.
    Bridge the two by matching on the ArchSpec class + pos-encoding flag.
    """
    from src.bench.architectures.topk_sae import TopKSAESpec
    from src.bench.architectures.stacked_sae import StackedSAESpec
    from src.bench.architectures.crosscoder import CrosscoderSpec
    from src.bench.architectures.mlc import LayerCrosscoderSpec
    from src.bench.architectures.tfa import TFASpec

    spec = entry.spec
    if isinstance(spec, TopKSAESpec):
        return "topk_sae"
    if isinstance(spec, StackedSAESpec):
        return "stacked_sae"
    # MLC must come before CrosscoderSpec — LayerCrosscoderSpec subclasses it.
    if isinstance(spec, LayerCrosscoderSpec):
        return "mlc"
    if isinstance(spec, CrosscoderSpec):
        return "crosscoder"
    if isinstance(spec, TFASpec):
        return "tfa_pos" if getattr(spec, "use_pos_encoding", False) else "tfa"
    return entry.name.lower()


def _filter_by_registry_keys(
    models: list[ModelEntry], wanted_keys: list[str],
) -> list[ModelEntry]:
    wanted = {k.lower() for k in wanted_keys}
    return [m for m in models if _arch_registry_key(m) in wanted]


def _get_generator(pipeline: DataPipeline, gen_key: str):
    """Look up the right data generator from the pipeline."""
    if gen_key == "flat":
        return pipeline.gen_flat
    elif gen_key == "seq":
        return pipeline.gen_seq
    elif gen_key.startswith("window_"):
        T = int(gen_key.split("_")[1])
        return pipeline.gen_windows[T]
    else:
        raise ValueError(f"Unknown gen_key: {gen_key}")


def run_sweep(
    models: list[ModelEntry],
    data_config: DataConfig,
    train_config: TrainConfig,
    sweep_config: SweepConfig,
    device: torch.device,
    results_dir: str = "results",
    verbose: bool = True,
) -> list[dict]:
    """Run a full architecture comparison sweep.

    For each (rho, k) combination, trains and evaluates every model in
    the models list, then saves results. Supports both standard and
    coupled-feature data generation modes.

    Args:
        models: List of ModelEntry specs to compare.
        data_config: Base data configuration (rho will be overridden per sweep point).
        train_config: Training hyperparameters.
        sweep_config: Sweep grid specification.
        device: Torch device.
        results_dir: Directory for JSON result output.
        verbose: Print progress.

    Returns:
        List of result dicts, one per (model, rho, k) combination.
    """
    os.makedirs(results_dir, exist_ok=True)
    all_results = []
    is_coupled = data_config.coupling is not None

    # Collect window sizes needed across all models
    window_sizes = set()
    for entry in models:
        if entry.gen_key.startswith("window_"):
            window_sizes.add(int(entry.gen_key.split("_")[1]))

    for seed in sweep_config.seeds:
        for rho in sweep_config.rho_values:
            # Build a pipeline for this (rho, seed) combination
            rho_config = DataConfig(
                toy_model=data_config.toy_model,
                markov=MarkovConfig(
                    pi=data_config.markov.pi,
                    rho=rho,
                    delta=data_config.markov.delta,
                ),
                coupling=data_config.coupling,
                seq_len=data_config.seq_len,
                d_sae=data_config.d_sae,
                seed=seed,
                eval_n_seq=data_config.eval_n_seq,
            )

            mode_str = ""
            if is_coupled:
                c = data_config.coupling
                mode_str = f" [coupled K={c.K_hidden} M={c.M_emission}]"
            if data_config.markov.delta > 0:
                mode_str += f" [delta={data_config.markov.delta}]"
            seed_str = f" [seed={seed}]" if len(sweep_config.seeds) > 1 else ""

            if verbose:
                print(f"\n{'=' * 60}")
                print(f"  rho = {rho}{mode_str}{seed_str}")
                print(f"{'=' * 60}")

            pipeline = build_data_pipeline(
                rho_config, device, window_sizes=list(window_sizes) or None
            )

            for k in sweep_config.k_values:
                if verbose:
                    print(f"\n  k = {k}:")

                for entry in models:
                    # Seed training for reproducibility
                    torch.manual_seed(seed)
                    t0 = time.time()

                    # Resolve training params (apply overrides)
                    batch_size = entry.training_overrides.get(
                        "batch_size", train_config.batch_size
                    )
                    lr = entry.training_overrides.get("lr", train_config.lr)
                    total_steps = entry.training_overrides.get(
                        "total_steps", train_config.total_steps
                    )

                    # Create and train
                    model = entry.spec.create(
                        d_in=data_config.toy_model.hidden_dim,
                        d_sae=data_config.d_sae,
                        k=k,
                        device=device,
                    )
                    gen_fn = _get_generator(pipeline, entry.gen_key)
                    train_log = entry.spec.train(
                        model,
                        gen_fn,
                        total_steps=total_steps,
                        batch_size=batch_size,
                        lr=lr,
                        device=device,
                        log_every=train_config.log_every,
                        grad_clip=train_config.grad_clip,
                    )

                    # Evaluate with both local and global features
                    result = evaluate_model(
                        entry.spec,
                        model,
                        pipeline.eval_hidden,
                        device,
                        true_features=pipeline.true_features,
                        global_features=pipeline.global_features,
                        seq_len=data_config.seq_len,
                    )

                    elapsed = time.time() - t0
                    row = {
                        "model": entry.name,
                        "rho": rho,
                        "k": k,
                        "seed": seed,
                        **result.to_dict(),
                        "elapsed_sec": round(elapsed, 1),
                    }
                    if data_config.markov.delta > 0:
                        row["delta"] = data_config.markov.delta
                    all_results.append(row)

                    if verbose:
                        auc_str = f"AUC={result.auc:.4f}" if result.auc is not None else ""
                        global_str = ""
                        if result.global_auc is not None:
                            global_str = f" | gAUC={result.global_auc:.4f}"
                        print(
                            f"    {entry.name:20s} | NMSE={result.nmse:.6f} | "
                            f"L0={result.l0:.2f} | {auc_str}{global_str} | {elapsed:.0f}s"
                        )

                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Save per-(rho, seed) results
            tag = f"rho{rho:.1f}_seed{seed}"
            rho_path = os.path.join(results_dir, f"results_{tag}.json")
            rho_results = [
                r for r in all_results if r["rho"] == rho and r["seed"] == seed
            ]
            with open(rho_path, "w") as f:
                json.dump(rho_results, f, indent=2)

    # Save full summary
    summary_path = os.path.join(results_dir, "sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  SWEEP COMPLETE | {len(all_results)} runs")
        print(f"  Results: {summary_path}")
        print(f"{'=' * 60}")

    return all_results


def run_cached_sweep(
    models: list[ModelEntry],
    data_config: DataConfig,
    train_config: TrainConfig,
    sweep_config: SweepConfig,
    device: torch.device,
    results_dir: str = "results",
    verbose: bool = True,
) -> list[dict]:
    """Real-LM sweep over pre-cached activations.

    No rho loop (there's no Markov temperature), no coupling, no true_features.
    Iterates (seed, k, model_entry), trains on the cached activation pipeline,
    evaluates on the held-out slice, writes one JSON per run.
    """
    os.makedirs(results_dir, exist_ok=True)
    all_results: list[dict] = []

    cfg = get_model_config(data_config.model_name)
    d_in = cfg.d_model
    d_sae = data_config.d_sae  # caller must have set this sensibly

    window_sizes = set()
    for entry in models:
        if entry.gen_key.startswith("window_"):
            window_sizes.add(int(entry.gen_key.split("_")[1]))

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  CACHED SWEEP | {data_config.model_name}")
        print(f"    dataset = {data_config.cached_dataset}")
        print(f"    layer   = {data_config.cached_layer_key}")
        print(f"    shuffle = {data_config.shuffle_within_sequence}")
        print(f"    d_in    = {d_in}   d_sae = {d_sae}")
        print(f"{'=' * 60}")

    pipeline = build_pipeline(
        data_config, device, window_sizes=list(window_sizes) or None,
    )

    for seed in sweep_config.seeds:
        for k in sweep_config.k_values:
            if verbose:
                print(f"\n  seed={seed}  k={k}:")
            for entry in models:
                torch.manual_seed(seed)
                t0 = time.time()

                batch_size = entry.training_overrides.get(
                    "batch_size", train_config.batch_size,
                )
                lr = entry.training_overrides.get("lr", train_config.lr)
                total_steps = entry.training_overrides.get(
                    "total_steps", train_config.total_steps,
                )

                # Optional W&B tracking. Enable by setting WANDB_API_KEY
                # in the env; leave unset for local dry runs. Per-run
                # name encodes the full (arch, subject, dataset, layer,
                # k, seed, shuffle) tuple so the web dashboard groups
                # related runs together without manual tagging.
                _wandb_run = None
                if os.environ.get("WANDB_API_KEY"):
                    try:
                        import wandb
                        # Optional SAEBench-scoped tags. The saebench
                        # orchestrator exports these so run names stay
                        # distinguishable across protocol A/B and T
                        # values; other callers leave them unset.
                        _protocol = os.environ.get("SAEBENCH_PROTOCOL")
                        _t_display = os.environ.get("SAEBENCH_T")
                        run_name = (
                            f"{_arch_registry_key(entry)}__{data_config.model_name}"
                            f"__{data_config.cached_dataset}"
                            f"__{data_config.cached_layer_key}"
                            f"__k{k}__seed{seed}"
                            f"{f'__prot{_protocol}' if _protocol else ''}"
                            f"{f'__T{_t_display}' if _t_display else ''}"
                            f"{'_shuffled' if data_config.shuffle_within_sequence else ''}"
                        )
                        _wandb_run = wandb.init(
                            project=os.environ.get(
                                "WANDB_PROJECT", "temporal-crosscoders"
                            ),
                            entity=os.environ.get("WANDB_ENTITY") or None,
                            name=run_name,
                            group=os.environ.get("WANDB_GROUP") or None,
                            reinit=True,
                            config={
                                "arch": entry.name,
                                "arch_key": _arch_registry_key(entry),
                                "subject_model": data_config.model_name,
                                "cached_dataset": data_config.cached_dataset,
                                "cached_layer_key": data_config.cached_layer_key,
                                "cached_layer_keys": data_config.cached_layer_keys,
                                "shuffle_within_sequence":
                                    data_config.shuffle_within_sequence,
                                "k": k,
                                "seed": seed,
                                "d_in": d_in,
                                "d_sae": d_sae,
                                "total_steps": total_steps,
                                "batch_size": batch_size,
                                "lr": lr,
                                "dataset_type": data_config.dataset_type,
                                "saebench_protocol": _protocol,
                                "saebench_t": int(_t_display) if _t_display else None,
                            },
                        )
                    except Exception as e:
                        print(f"  WARN: wandb.init failed ({e}); continuing without W&B")
                        _wandb_run = None

                model = entry.spec.create(
                    d_in=d_in, d_sae=d_sae, k=k, device=device,
                )
                gen_fn = _get_generator(pipeline, entry.gen_key)
                entry.spec.train(
                    model, gen_fn,
                    total_steps=total_steps, batch_size=batch_size, lr=lr,
                    device=device, log_every=train_config.log_every,
                    grad_clip=train_config.grad_clip,
                )

                # Save checkpoint IMMEDIATELY after training, before eval.
                # If evaluate_model crashes (e.g. encode() shape mismatch on
                # long sequences), we don't want to lose the hours of GPU
                # time that went into training. Downstream tools read the
                # same .pt file; the sweep JSON's metrics may be None if eval
                # crashes, but the checkpoint is the load-bearing artifact.
                # Use the registry key (topk_sae, stacked_sae, crosscoder,
                # tfa, tfa_pos) rather than the display name — the
                # finalize/feature_map scripts look up checkpoints by
                # registry key, and display names contain spaces
                # ("Stacked T=5") which are a shell-escape headache.
                ckpt_dir = os.path.join(results_dir, "ckpts")
                os.makedirs(ckpt_dir, exist_ok=True)
                shuf_tag = "_shuffled" if data_config.shuffle_within_sequence else ""
                arch_key = _arch_registry_key(entry)
                ckpt_name = (
                    f"{arch_key}__{data_config.model_name}"
                    f"__{data_config.cached_dataset}__{data_config.cached_layer_key}"
                    f"__k{k}__seed{seed}{shuf_tag}.pt"
                )
                ckpt_path = os.path.join(ckpt_dir, ckpt_name)
                torch.save(model.state_dict(), ckpt_path)

                # Log checkpoint to W&B as an Artifact for team-visible
                # persistence + cross-pod retrieval. Falls back silently
                # if W&B isn't initialized.
                if _wandb_run is not None:
                    try:
                        import wandb
                        artifact = wandb.Artifact(
                            name=ckpt_name.replace(".pt", "").replace("__", "-"),
                            type="sae_checkpoint",
                            description=(
                                f"{entry.name} | k={k} | seed={seed} | "
                                f"{data_config.model_name}/"
                                f"{data_config.cached_dataset}/"
                                f"{data_config.cached_layer_key}"
                            ),
                            metadata={
                                "arch_key": _arch_registry_key(entry),
                                "k": k,
                                "seed": seed,
                                "d_in": d_in,
                                "d_sae": d_sae,
                                "total_steps": total_steps,
                            },
                        )
                        artifact.add_file(ckpt_path)
                        _wandb_run.log_artifact(artifact)
                    except Exception as e:
                        print(f"  WARN: wandb artifact upload failed ({e})")

                result = evaluate_model(
                    entry.spec, model, pipeline.eval_hidden, device,
                    true_features=None, global_features=None,
                    seq_len=pipeline.eval_hidden.shape[1],
                )

                elapsed = time.time() - t0

                row = {
                    "checkpoint": ckpt_path,
                    "arch": entry.name,
                    "subject_model": data_config.model_name,
                    "cached_dataset": data_config.cached_dataset,
                    "layer_key": data_config.cached_layer_key,
                    "shuffled": bool(data_config.shuffle_within_sequence),
                    "k": k,
                    "seed": seed,
                    "d_in": d_in,
                    "d_sae": d_sae,
                    **result.to_dict(),
                    "elapsed_sec": round(elapsed, 1),
                }
                all_results.append(row)

                if verbose:
                    print(
                        f"    {entry.name:20s} | NMSE={result.nmse:.4f} "
                        f"| L0={result.l0:.2f} | {elapsed:.0f}s"
                    )

                # Finalize the W&B run — log terminal eval metrics and close.
                if _wandb_run is not None:
                    try:
                        _wandb_run.summary["nmse"] = result.nmse
                        _wandb_run.summary["l0"] = result.l0
                        _wandb_run.summary["elapsed_sec"] = elapsed
                        _wandb_run.finish()
                    except Exception as e:
                        print(f"  WARN: wandb finish failed ({e})")

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    tag = (
        f"{data_config.model_name}__{data_config.cached_dataset}"
        f"__{data_config.cached_layer_key}"
        f"{'__shuffled' if data_config.shuffle_within_sequence else ''}"
    )
    path = os.path.join(results_dir, f"results_{tag}.json")
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    if verbose:
        print(f"\n  SWEEP COMPLETE | {len(all_results)} runs | {path}")
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run architecture comparison sweep"
    )
    parser.add_argument("--rho", type=float, nargs="+", default=None,
                        help="Rho values to sweep (default: [0.0, 0.6, 0.9])")
    parser.add_argument("--k", type=int, nargs="+", default=None,
                        help="K values to sweep (default: [2, 5, 10, 25])")
    parser.add_argument("--T", type=int, nargs="+", default=None,
                        help="Window sizes (default: [2, 5])")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Filter to specific model names")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override training steps")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Seeds to sweep (default: [42])")
    parser.add_argument("--results-dir", type=str, default="results/bench",
                        help="Output directory for results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print sweep plan without training")

    # HMM extensions
    parser.add_argument("--delta", type=float, default=0.0,
                        help="Leaky reset parameter (0 = standard)")
    parser.add_argument("--coupled", action="store_true",
                        help="Enable coupled-feature mode")
    parser.add_argument("--K-hidden", type=int, default=10,
                        help="Number of hidden states (coupled mode)")
    parser.add_argument("--M-emission", type=int, default=20,
                        help="Number of emission features (coupled mode)")
    parser.add_argument("--n-parents", type=int, default=2,
                        help="Parents per emission (coupled mode)")

    # Real-LM cached-activation mode
    parser.add_argument("--dataset-type", type=str, default="markov",
                        choices=["markov", "cached_activations", "multi_layer_activations"],
                        help="Switch between toy Markov and real-LM cached acts")
    parser.add_argument("--model-name", type=str, default="deepseek-r1-distill-llama-8b",
                        choices=list_models(),
                        help="Subject LM key (cached_activations mode)")
    parser.add_argument("--cached-dataset", type=str, default="gsm8k",
                        help="Subdir under data/cached_activations/<model>/")
    parser.add_argument("--cached-layer-key", type=str, default="resid_L12",
                        help="Which <key>.npy to load (single-layer mode)")
    parser.add_argument("--cached-layer-keys", type=str, nargs="+", default=None,
                        help="Multi-layer mode: list of <key>.npy files to stack "
                             "(e.g. resid_L10 resid_L11 resid_L12 resid_L13 resid_L14)")
    parser.add_argument("--shuffle-within-sequence", action="store_true",
                        help="Temporal shuffled control baseline")
    parser.add_argument("--expansion-factor", type=int, default=8,
                        help="d_sae = d_model * expansion_factor (cached mode)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sweep_config = SweepConfig(
        k_values=args.k or [2, 5, 10, 25],
        rho_values=args.rho or [0.0, 0.6, 0.9],
        T_values=args.T or [2, 5],
        seeds=args.seeds or [42],
    )
    train_config = TrainConfig(
        total_steps=args.steps or 30_000,
    )

    # Dispatch: cached activations path is a completely different code path.
    if args.dataset_type in ("cached_activations", "multi_layer_activations"):
        cfg = get_model_config(args.model_name)
        d_sae = cfg.d_model * args.expansion_factor
        data_config = DataConfig(
            dataset_type=args.dataset_type,
            model_name=args.model_name,
            cached_dataset=args.cached_dataset,
            cached_layer_key=args.cached_layer_key,
            cached_layer_keys=args.cached_layer_keys,
            shuffle_within_sequence=args.shuffle_within_sequence,
            d_sae=d_sae,
        )
        models = get_default_models(sweep_config.T_values)
        if args.models:
            models = _filter_by_registry_keys(models, args.models)
            if not models:
                raise ValueError(
                    f"--models {args.models} matched zero architectures. "
                    f"Valid keys: topk_sae, stacked_sae, crosscoder, mlc, tfa, tfa_pos"
                )
        print(f"{'=' * 60}")
        print(f"  REAL-LM CACHED-ACTIVATION SWEEP")
        print(f"  Subject:  {args.model_name}  (d_model={cfg.d_model})")
        print(f"  Dataset:  {args.cached_dataset} / {args.cached_layer_key}")
        print(f"  Shuffle:  {args.shuffle_within_sequence}")
        print(f"  d_sae:    {d_sae}  (×{args.expansion_factor} expansion)")
        print(f"  Models:   {[m.name for m in models]}")
        print(f"  K:        {sweep_config.k_values}")
        print(f"  Steps:    {train_config.total_steps:,}")
        print(f"{'=' * 60}")
        if args.dry_run:
            return
        run_cached_sweep(
            models=models,
            data_config=data_config,
            train_config=train_config,
            sweep_config=sweep_config,
            device=device,
            results_dir=args.results_dir,
        )
        return

    coupling = None
    if args.coupled:
        coupling = CouplingConfig(
            K_hidden=args.K_hidden,
            M_emission=args.M_emission,
            n_parents=args.n_parents,
        )

    data_config = DataConfig(
        markov=MarkovConfig(delta=args.delta),
        coupling=coupling,
        d_sae=args.M_emission if args.coupled else 128,
    )

    models = get_default_models(sweep_config.T_values)

    # Filter models if requested
    if args.models:
        models = _filter_by_registry_keys(models, args.models)
        if not models:
            raise ValueError(
                f"--models {args.models} matched zero architectures. "
                f"Valid keys: topk_sae, stacked_sae, crosscoder, mlc, tfa, tfa_pos"
            )

    # Build job list for display
    jobs = list(itertools.product(
        sweep_config.seeds, sweep_config.rho_values, sweep_config.k_values, models
    ))

    mode_info = "standard"
    if args.coupled:
        mode_info = f"coupled (K={args.K_hidden}, M={args.M_emission}, n_parents={args.n_parents})"
    if args.delta > 0:
        mode_info += f", delta={args.delta}"

    print(f"{'=' * 60}")
    print(f"  ARCHITECTURE COMPARISON SWEEP")
    print(f"  Device: {device}")
    print(f"  Mode: {mode_info}")
    print(f"  Steps: {train_config.total_steps:,}")
    print(f"  Seeds: {sweep_config.seeds}")
    print(f"  Rho: {sweep_config.rho_values}")
    print(f"  K: {sweep_config.k_values}")
    print(f"  T: {sweep_config.T_values}")
    print(f"  Models: {[m.name for m in models]}")
    print(f"  Total jobs: {len(jobs)}")
    print(f"{'=' * 60}")

    if args.dry_run:
        for i, (seed, rho, k, entry) in enumerate(jobs, 1):
            seed_str = f"  seed={seed}" if len(sweep_config.seeds) > 1 else ""
            print(f"  [{i:3d}/{len(jobs)}]  {entry.name:20s}  rho={rho}  k={k}{seed_str}")
        print("\n  (dry run — no training)")
        return

    run_sweep(
        models=models,
        data_config=data_config,
        train_config=train_config,
        sweep_config=sweep_config,
        device=device,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
