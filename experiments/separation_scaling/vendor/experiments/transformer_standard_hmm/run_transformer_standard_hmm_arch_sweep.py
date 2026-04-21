from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable

import torch

ROOT = Path(__file__).parent
REPO_ROOT = ROOT.parents[1]
SRC_ROOT = REPO_ROOT / "src"
STANDARD_HMM_ROOT = REPO_ROOT / "experiments" / "standard_hmm"

sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(STANDARD_HMM_ROOT))

from sae_day.data import MESS3_SEPARATED_COMPONENTS
from sae_day.simplexity_standard_hmm import generate_standard_hmm_data_simplexity
from sae_day.sae import (
    MatryoshkaMultiLayerCrosscoder,
    MatryoshkaTemporalCrosscoder,
    MultiLayerCrosscoder,
    TemporalBatchTopKSAE,
    TemporalCrosscoder,
    TopKSAE,
)
from sae_day.tfa import (
    TemporalSAE,
    TFATrainingConfig,
    compute_scaling_factor,
    train_tfa,
)

from run_standard_hmm_arch_seed_sweep import (
    ARCH_CONFIGS,
    VARIANTS,
    encode_all_sae,
    encode_all_temporal_final_latents,
    evaluate_representation,
    make_causal_windows,
    summarize_runs,
    train_temporal_model,
    train_topk,
)


def slugify(name: str) -> str:
    return name.lower().replace(" ", "_")


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train_transformer(
    model: Any,
    sample_batch_fn: Callable[[int, int], torch.Tensor],
    n_steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    print_every: int = 100,
    param_groups: list[dict] | None = None,
    fused_optimizer: bool = True,
    autocast_dtype: torch.dtype | None = None,
) -> list[float]:
    """Train-loop for HookedTransformer on next-token prediction.

    Speed knobs (all default-on except autocast):
      - ``torch.set_float32_matmul_precision("high")`` enables TF32 on A40/A100
        (module-level, idempotent).
      - ``fused_optimizer=True`` uses the fused Adam kernel where available.
        Silently falls back to non-fused if the torch build doesn't support it.
      - ``autocast_dtype`` (e.g. ``torch.bfloat16``) wraps the forward+loss in
        ``torch.autocast`` — ~1.5-2× on tensor-core GPUs. Pass ``None`` to disable.

    If ``param_groups`` is provided (e.g. from muP), use it as the Adam
    param_groups; otherwise flat Adam on all parameters at ``lr``.
    """
    torch.set_float32_matmul_precision("high")

    adam_kwargs: dict = {}
    if fused_optimizer:
        try:
            # Probe whether fused Adam is accepted on this torch build.
            _probe = torch.optim.Adam([torch.zeros(1, device=next(model.parameters()).device, requires_grad=True)], fused=True)
            del _probe
            adam_kwargs["fused"] = True
        except (RuntimeError, TypeError):
            pass

    if param_groups is not None:
        optimizer = torch.optim.Adam(param_groups, **adam_kwargs)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, **adam_kwargs)
    device = next(model.parameters()).device

    losses = []
    model.train()
    use_autocast = autocast_dtype is not None and device.type == "cuda"
    for step in range(n_steps):
        batch = sample_batch_fn(step, batch_size).to(device)
        if use_autocast:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                loss = model(batch, return_type="loss")
        else:
            loss = model(batch, return_type="loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if print_every > 0 and (step + 1) % print_every == 0:
            print(f"    transformer step {step + 1}/{n_steps}  loss={loss.item():.4f}", flush=True)
    return losses


@torch.no_grad()
def extract_position_activations(
    model: Any,
    tokens: torch.Tensor,
    hook_name: str,
    chunk_size: int = 16,
) -> torch.Tensor:
    device = next(model.parameters()).device
    outputs = []
    model.eval()
    for i in range(0, tokens.shape[0], chunk_size):
        batch = tokens[i : i + chunk_size].to(device)
        _, cache = model.run_with_cache(
            batch,
            return_type="logits",
            names_filter=lambda name: name == hook_name,
            return_cache_object=False,
        )
        outputs.append(cache[hook_name].detach().cpu())
    return torch.cat(outputs, dim=0)


def evaluate_topk_on_activations(
    seed: int,
    arch_cfg: dict[str, Any],
    train_acts: torch.Tensor,
    eval_acts: torch.Tensor,
    eval_omega: torch.Tensor,
    d_model: int,
    device: torch.device,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    n_train = min(arch_cfg["n_sequences"], train_acts.shape[0])
    n_eval = min(arch_cfg["n_sequences"], eval_acts.shape[0])
    seq_len = min(arch_cfg["seq_len"], train_acts.shape[1], eval_acts.shape[1])

    arch_train_acts = train_acts[:n_train, :seq_len, :].contiguous()
    arch_eval_acts = eval_acts[:n_eval, :seq_len, :].contiguous()
    arch_eval_omega = eval_omega[:n_eval]

    flat_train = arch_train_acts.reshape(-1, d_model)
    flat_eval = arch_eval_acts.reshape(-1, d_model)
    target = (
        arch_eval_omega[:, None, :]
        .expand(n_eval, seq_len, arch_eval_omega.shape[-1])
        .reshape(-1, arch_eval_omega.shape[-1])
        .numpy()
    )

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = TopKSAE(
        d_in=d_model, d_sae=arch_cfg["dict_size"], k=arch_cfg["k"],
        use_relu=arch_cfg.get("use_relu", True),
    ).to(device)
    losses = train_topk(model, flat_train, n_steps=arch_cfg["sae_steps"], seed=seed)
    z_all = encode_all_sae(model, flat_eval)
    metrics = evaluate_representation(
        z_all,
        target,
        n_sequences=n_eval,
        samples_per_sequence=seq_len,
        seed=seed,
    )
    run = {
        "seed": seed,
        "n_train_sequences": n_train,
        "n_eval_sequences": n_eval,
        "effective_seq_len": seq_len,
        "final_loss": float(losses[-1]),
        "metrics": metrics,
    }
    if checkpoint_path is not None:
        save_checkpoint(
            checkpoint_path,
            {
                "kind": "TopK SAE",
                "seed": seed,
                "config": arch_cfg,
                "activation_shape": [int(v) for v in arch_train_acts.shape],
                "run": run,
                "state_dict": cpu_state_dict(model),
            },
        )
    return run


def evaluate_txc_on_activations(
    seed: int,
    arch_cfg: dict[str, Any],
    train_acts: torch.Tensor,
    eval_acts: torch.Tensor,
    eval_omega: torch.Tensor,
    d_model: int,
    device: torch.device,
    *,
    matryoshka: bool,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    n_train = min(arch_cfg["n_sequences"], train_acts.shape[0])
    n_eval = min(arch_cfg["n_sequences"], eval_acts.shape[0])
    seq_len = min(arch_cfg["seq_len"], train_acts.shape[1], eval_acts.shape[1])
    window_size = arch_cfg["window_size"]
    if seq_len < window_size:
        raise ValueError(
            f"Effective seq_len {seq_len} is smaller than window_size {window_size} for architecture {arch_cfg}"
        )

    arch_train_acts = train_acts[:n_train, :seq_len, :].contiguous()
    arch_eval_acts = eval_acts[:n_eval, :seq_len, :].contiguous()
    arch_eval_omega = eval_omega[:n_eval]

    train_windows = make_causal_windows(arch_train_acts, window_size)
    eval_windows = make_causal_windows(arch_eval_acts, window_size)
    n_windows = eval_windows.shape[1]
    flat_train = train_windows.reshape(-1, window_size, d_model)
    flat_eval = eval_windows.reshape(-1, window_size, d_model)
    target = (
        arch_eval_omega[:, None, :]
        .expand(n_eval, n_windows, arch_eval_omega.shape[-1])
        .reshape(-1, arch_eval_omega.shape[-1])
        .numpy()
    )

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    use_relu = arch_cfg.get("use_relu", True)
    if matryoshka:
        widths = [w for w in [8, 16, 32, 64, arch_cfg["dict_size"]] if w <= arch_cfg["dict_size"]]
        model = MatryoshkaTemporalCrosscoder(
            d_in=d_model,
            d_sae=arch_cfg["dict_size"],
            T=window_size,
            k_per_pos=arch_cfg["k"],
            k_total=arch_cfg.get("fixed_k_total"),
            matryoshka_widths=widths,
            inner_weight=arch_cfg["inner_weight"],
            use_relu=use_relu,
        ).to(device)
    else:
        model = TemporalCrosscoder(
            d_in=d_model,
            d_sae=arch_cfg["dict_size"],
            T=window_size,
            k_per_pos=arch_cfg["k"],
            k_total=arch_cfg.get("fixed_k_total"),
            use_relu=use_relu,
        ).to(device)

    losses = train_temporal_model(model, flat_train, n_steps=arch_cfg["temporal_steps"], seed=seed)
    z_all = encode_all_temporal_final_latents(model, flat_eval, final_position=0)
    metrics = evaluate_representation(
        z_all,
        target,
        n_sequences=n_eval,
        samples_per_sequence=n_windows,
        seed=seed,
    )
    run = {
        "seed": seed,
        "n_train_sequences": n_train,
        "n_eval_sequences": n_eval,
        "effective_seq_len": seq_len,
        "final_loss": float(losses[-1]),
        "metrics": metrics,
    }
    if checkpoint_path is not None:
        save_checkpoint(
            checkpoint_path,
            {
                "kind": "MatryoshkaTXC" if matryoshka else "TXC",
                "seed": seed,
                "config": arch_cfg,
                "activation_shape": [int(v) for v in arch_train_acts.shape],
                "run": run,
                "state_dict": cpu_state_dict(model),
            },
        )
    return run


def default_resid_post_hook_names(n_layers: int) -> list[str]:
    """Default hook list for the multi-layer crosscoder: hook_resid_post at each layer."""
    return [f"blocks.{i}.hook_resid_post" for i in range(n_layers)]


@torch.no_grad()
def extract_multi_layer_position_activations(
    model: Any,
    tokens: torch.Tensor,
    hook_names: list[str],
    chunk_size: int = 16,
) -> torch.Tensor:
    """Run the transformer once, grab activations at each named hook.

    Returns a tensor of shape ``(N, seq_len, L, d_model)`` where
    ``L = len(hook_names)``. Hooks are stacked in the order given.
    """
    device = next(model.parameters()).device
    per_hook_chunks: list[list[torch.Tensor]] = [[] for _ in hook_names]
    hook_set = set(hook_names)
    model.eval()
    for i in range(0, tokens.shape[0], chunk_size):
        batch = tokens[i : i + chunk_size].to(device)
        _, cache = model.run_with_cache(
            batch,
            return_type="logits",
            names_filter=lambda name: name in hook_set,
            return_cache_object=False,
        )
        for li, h in enumerate(hook_names):
            per_hook_chunks[li].append(cache[h].detach().cpu())
    per_hook = [torch.cat(chunks, dim=0) for chunks in per_hook_chunks]
    return torch.stack(per_hook, dim=2)  # (N, seq, L, d_model)


def evaluate_mlxc_on_activations(
    seed: int,
    arch_cfg: dict[str, Any],
    train_acts: torch.Tensor,  # (N_train, seq_len, L, d_model)
    eval_acts: torch.Tensor,   # (N_eval, seq_len, L, d_model)
    eval_omega: torch.Tensor,
    d_model: int,
    device: torch.device,
    *,
    matryoshka: bool,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """Train a multi-layer crosscoder on per-(sequence, position) L-layer stacks.

    `train_acts` / `eval_acts` are 4D: (N, seq_len, L, d_model). The training
    set flattens over (N, seq_len) — every position is an independent sample.
    For probing, every position's latent is used (with the component-label
    target broadcast across positions), matching the TXC evaluator's behavior.
    """
    if train_acts.dim() != 4:
        raise ValueError(
            f"MLxC expects 4D activations (N, seq, L, d_model); got {tuple(train_acts.shape)}"
        )
    n_train = min(arch_cfg["n_sequences"], train_acts.shape[0])
    n_eval = min(arch_cfg["n_sequences"], eval_acts.shape[0])
    seq_len = min(arch_cfg["seq_len"], train_acts.shape[1], eval_acts.shape[1])
    L = train_acts.shape[2]
    if eval_acts.shape[2] != L:
        raise ValueError(
            f"train/eval activations disagree on layer count: {L} vs {eval_acts.shape[2]}"
        )

    arch_train = train_acts[:n_train, :seq_len].contiguous()   # (n_train, seq_len, L, d)
    arch_eval = eval_acts[:n_eval, :seq_len].contiguous()
    arch_eval_omega = eval_omega[:n_eval]

    flat_train = arch_train.reshape(-1, L, d_model)    # (n_train*seq_len, L, d)
    flat_eval = arch_eval.reshape(-1, L, d_model)
    target = (
        arch_eval_omega[:, None, :]
        .expand(n_eval, seq_len, arch_eval_omega.shape[-1])
        .reshape(-1, arch_eval_omega.shape[-1])
        .numpy()
    )

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    common_kwargs = dict(
        d_in=d_model,
        d_sae=arch_cfg["dict_size"],
        L=L,
        k_per_layer=arch_cfg.get("k"),
        k_total=arch_cfg.get("fixed_k_total"),
        use_relu=arch_cfg.get("use_relu", True),
    )
    if matryoshka:
        widths = [w for w in [8, 16, 32, 64, arch_cfg["dict_size"]] if w <= arch_cfg["dict_size"]]
        model = MatryoshkaMultiLayerCrosscoder(
            **common_kwargs,
            matryoshka_widths=widths,
            inner_weight=arch_cfg.get("inner_weight", 1.0),
        ).to(device)
    else:
        model = MultiLayerCrosscoder(**common_kwargs).to(device)

    losses = train_temporal_model(
        model, flat_train, n_steps=arch_cfg["temporal_steps"], seed=seed
    )
    z_all = encode_all_temporal_final_latents(model, flat_eval, final_position=0)
    metrics = evaluate_representation(
        z_all,
        target,
        n_sequences=n_eval,
        samples_per_sequence=seq_len,
        seed=seed,
    )
    run = {
        "seed": seed,
        "n_train_sequences": n_train,
        "n_eval_sequences": n_eval,
        "effective_seq_len": seq_len,
        "n_layers_read": L,
        "final_loss": float(losses[-1]),
        "metrics": metrics,
    }
    if checkpoint_path is not None:
        save_checkpoint(
            checkpoint_path,
            {
                "kind": "MatryoshkaMLxC" if matryoshka else "MLxC",
                "seed": seed,
                "config": arch_cfg,
                "activation_shape": [int(v) for v in arch_train.shape],
                "run": run,
                "state_dict": cpu_state_dict(model),
            },
        )
    return run


def _build_consecutive_pair_dataset(acts: torch.Tensor) -> torch.Tensor:
    """Turn (N, seq_len, d_in) → (N*(seq_len-1), 2, d_in) of consecutive pairs."""
    assert acts.dim() == 3, acts.shape
    N, T, D = acts.shape
    if T < 2:
        raise ValueError(f"seq_len {T} too short to form temporal pairs")
    pairs = torch.stack([acts[:, :-1, :], acts[:, 1:, :]], dim=2)  # (N, T-1, 2, D)
    return pairs.reshape(-1, 2, D).contiguous()


def train_tsae(
    model: TemporalBatchTopKSAE,
    flat_pairs: torch.Tensor,
    n_steps: int,
    group_weights: list[float],
    temporal_alpha: float = 0.1,
    batch_size: int = 256,
    lr: float = 3e-4,
    seed: int = 42,
    normalize_every: int = 100,
) -> list[float]:
    """Train loop mirroring ``train_temporal_model`` but for the TSAE."""
    gen = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    device = next(model.parameters()).device
    n_total = flat_pairs.shape[0]
    losses = []
    for step in range(n_steps):
        idx = torch.randint(n_total, (batch_size,), generator=gen)
        x_pair = flat_pairs[idx].to(device)
        loss, _info = model.compute_loss(x_pair, group_weights, temporal_alpha=temporal_alpha)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if normalize_every > 0 and (step + 1) % normalize_every == 0:
            model.normalize_decoder()
        losses.append(float(loss.item()))
    return losses


@torch.no_grad()
def _encode_tsae_flat_latents(
    model: TemporalBatchTopKSAE, flat_acts: torch.Tensor, chunk_size: int = 4096
) -> np.ndarray:
    """Encode (M, d_in) activations → (M, d_sae) latents."""
    device = next(model.parameters()).device
    model.eval()
    chunks = []
    for i in range(0, flat_acts.shape[0], chunk_size):
        z = model.encode(flat_acts[i : i + chunk_size].to(device))
        chunks.append(z.cpu())
    return torch.cat(chunks, dim=0).numpy()


def evaluate_tsae_on_activations(
    seed: int,
    arch_cfg: dict[str, Any],
    train_acts: torch.Tensor,   # (N, seq_len, d_model)
    eval_acts: torch.Tensor,
    eval_omega: torch.Tensor,
    d_model: int,
    device: torch.device,
    *,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate the Temporal Matryoshka BatchTopK SAE on per-sequence activations.

    Training data: every consecutive ``(x_t, x_{t+1})`` pair within each train
    sequence is one training sample. Evaluation: encode every position's
    activation, match against the (broadcast) sequence-level component indicator.
    """
    if train_acts.dim() != 3:
        raise ValueError(f"TSAE expects 3D (N, seq, d_model); got {tuple(train_acts.shape)}")
    n_train = min(arch_cfg["n_sequences"], train_acts.shape[0])
    n_eval = min(arch_cfg["n_sequences"], eval_acts.shape[0])
    seq_len = min(arch_cfg["seq_len"], train_acts.shape[1], eval_acts.shape[1])

    arch_train = train_acts[:n_train, :seq_len, :].contiguous()
    arch_eval = eval_acts[:n_eval, :seq_len, :].contiguous()
    arch_eval_omega = eval_omega[:n_eval]

    flat_pairs = _build_consecutive_pair_dataset(arch_train)  # (N*(T-1), 2, D)

    # Probe target: broadcast component-one-hot across positions, flatten.
    target = (
        arch_eval_omega[:, None, :]
        .expand(n_eval, seq_len, arch_eval_omega.shape[-1])
        .reshape(-1, arch_eval_omega.shape[-1])
        .numpy()
    )
    flat_eval = arch_eval.reshape(-1, d_model).contiguous()  # (n_eval*seq_len, D)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Build group_sizes from group_fractions (paper convention: [0.5, 0.5]).
    group_fractions = arch_cfg.get("group_fractions", [0.5, 0.5])
    group_weights = arch_cfg.get("group_weights", None)
    if group_weights is None:
        group_weights = [1.0 / len(group_fractions)] * len(group_fractions)
    d_sae = arch_cfg["dict_size"]
    group_sizes = [int(round(f * d_sae)) for f in group_fractions[:-1]]
    group_sizes.append(d_sae - sum(group_sizes))

    model = TemporalBatchTopKSAE(
        d_in=d_model,
        d_sae=d_sae,
        k=arch_cfg["k"],
        group_sizes=group_sizes,
        temporal=arch_cfg.get("temporal", True),
    ).to(device)

    losses = train_tsae(
        model, flat_pairs,
        n_steps=arch_cfg["temporal_steps"],
        group_weights=group_weights,
        temporal_alpha=arch_cfg.get("temporal_alpha", 0.1),
        lr=arch_cfg.get("lr", 3e-4),
        seed=seed,
    )

    z_all = _encode_tsae_flat_latents(model, flat_eval)
    metrics = evaluate_representation(
        z_all,
        target,
        n_sequences=n_eval,
        samples_per_sequence=seq_len,
        seed=seed,
    )
    run = {
        "seed": seed,
        "n_train_sequences": n_train,
        "n_eval_sequences": n_eval,
        "effective_seq_len": seq_len,
        "group_sizes": group_sizes,
        "group_weights": list(group_weights),
        "final_loss": float(losses[-1]),
        "metrics": metrics,
    }
    if checkpoint_path is not None:
        save_checkpoint(
            checkpoint_path,
            {
                "kind": "TemporalBatchTopKSAE",
                "seed": seed,
                "config": arch_cfg,
                "activation_shape": [int(v) for v in arch_train.shape],
                "run": run,
                "state_dict": cpu_state_dict(model),
            },
        )
    return run


def evaluate_tfa_on_activations(
    seed: int,
    arch_cfg: dict[str, Any],
    train_acts: torch.Tensor,   # (N_train, seq_len, d_model)
    eval_acts: torch.Tensor,    # (N_eval, seq_len, d_model)
    eval_omega: torch.Tensor,   # (N_eval, n_components)
    d_model: int,
    device: torch.device,
    *,
    use_pos_encoding: bool = False,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """Train Han's TFA on (N, T, d) sequences, probe the final temporal codes.

    Codes = pred_codes + novel_codes. Inputs are pre-scaled by `compute_scaling_factor`
    on the training set (TFA assumes inputs with norm ≈ √d; we cache and apply the
    same factor at eval).
    """
    if train_acts.dim() != 3:
        raise ValueError(f"TFA expects 3D (N, seq, d_model); got {tuple(train_acts.shape)}")
    n_train = min(arch_cfg.get("n_sequences", train_acts.shape[0]), train_acts.shape[0])
    n_eval = min(arch_cfg.get("n_sequences", eval_acts.shape[0]), eval_acts.shape[0])
    seq_len = min(arch_cfg.get("seq_len", train_acts.shape[1]), train_acts.shape[1], eval_acts.shape[1])

    arch_train = train_acts[:n_train, :seq_len, :].contiguous()
    arch_eval = eval_acts[:n_eval, :seq_len, :].contiguous()
    arch_eval_omega = eval_omega[:n_eval]

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    scaling_factor = compute_scaling_factor(arch_train)
    arch_train_scaled = (arch_train * scaling_factor).to(device)
    arch_eval_scaled = (arch_eval * scaling_factor).to(device)

    dict_size = arch_cfg["dict_size"]
    k = arch_cfg["k"]
    n_heads = arch_cfg.get("n_heads", 4)
    n_attn_layers = arch_cfg.get("n_attn_layers", 1)
    bottleneck_factor = arch_cfg.get("bottleneck_factor", 1)
    sae_diff_type = arch_cfg.get("sae_diff_type", "topk")

    model = TemporalSAE(
        dimin=d_model,
        width=dict_size,
        n_heads=n_heads,
        sae_diff_type=sae_diff_type,
        kval_topk=k if sae_diff_type in ("topk", "batchtopk") else None,
        tied_weights=arch_cfg.get("tied_weights", True),
        n_attn_layers=n_attn_layers,
        bottleneck_factor=bottleneck_factor,
        use_pos_encoding=use_pos_encoding,
        max_seq_len=max(512, seq_len),
    ).to(device)

    sampler_rng = torch.Generator(device=arch_train_scaled.device)
    sampler_rng.manual_seed(seed * 7 + 17)

    def sample_batch(batch_size: int) -> torch.Tensor:
        idx = torch.randint(
            0, arch_train_scaled.shape[0], (batch_size,),
            generator=sampler_rng, device=arch_train_scaled.device,
        )
        return arch_train_scaled.index_select(0, idx)

    cfg = TFATrainingConfig(
        total_steps=arch_cfg["tfa_steps"],
        batch_size=arch_cfg.get("batch_size", 64),
        lr=arch_cfg.get("lr", 1e-3),
        min_lr=arch_cfg.get("min_lr", 9e-4),
        weight_decay=arch_cfg.get("weight_decay", 1e-4),
        warmup_steps=arch_cfg.get("warmup_steps", 200),
        grad_clip=arch_cfg.get("grad_clip", 1.0),
        log_every=max(1, arch_cfg["tfa_steps"] // 5),
        l1_coeff=arch_cfg.get("l1_coeff", 0.0),
    )
    log = train_tfa(model, sample_batch, cfg, device=device, verbose=True)
    final_loss = float(log["loss"][-1]) if log["loss"] else float("nan")

    model.eval()
    with torch.no_grad():
        _, inter = model(arch_eval_scaled)
    codes = (inter["pred_codes"] + inter["novel_codes"]).detach().cpu()  # (n_eval, T, width)
    z_flat = codes.reshape(-1, dict_size).numpy()
    target = (
        arch_eval_omega[:, None, :]
        .expand(n_eval, seq_len, arch_eval_omega.shape[-1])
        .reshape(-1, arch_eval_omega.shape[-1])
        .numpy()
    )
    metrics = evaluate_representation(
        z_flat, target, n_sequences=n_eval, samples_per_sequence=seq_len, seed=seed,
    )

    run = {
        "seed": seed,
        "n_train_sequences": n_train,
        "n_eval_sequences": n_eval,
        "effective_seq_len": seq_len,
        "final_loss": final_loss,
        "scaling_factor": float(scaling_factor),
        "use_pos_encoding": bool(use_pos_encoding),
        "metrics": metrics,
    }
    if checkpoint_path is not None:
        save_checkpoint(
            checkpoint_path,
            {
                "kind": "TFA-pos" if use_pos_encoding else "TFA",
                "seed": seed,
                "config": arch_cfg,
                "scaling_factor": float(scaling_factor),
                "activation_shape": [int(v) for v in arch_train.shape],
                "run": run,
                "state_dict": cpu_state_dict(model),
            },
        )
    return run


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--train-sequences", type=int, default=1000)
    parser.add_argument("--eval-sequences", type=int, default=500)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--concentration", type=float, default=10.0)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--transformer-steps", type=int, default=20000)
    parser.add_argument("--transformer-batch-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--probe-layer", type=int, default=1)
    parser.add_argument("--d-mlp", type=int, default=256)
    parser.add_argument("--extract-batch-size", type=int, default=16)
    parser.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=ROOT / "outputs" / "checkpoints",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "outputs" / "transformer_standard_hmm_layer1_arch_sweep.json",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    if args.save_checkpoints:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.seq_len is None:
        args.seq_len = args.context_len

    from transformer_lens import HookedTransformer, HookedTransformerConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hook_name = f"blocks.{args.probe_layer}.hook_resid_post"

    print("=== Transformer Standard HMM Architecture Sweep ===", flush=True)
    print(f"Device={device}", flush=True)
    print(f"Hook={hook_name}", flush=True)

    results: dict[str, Any] = {
        "target": "SequenceOmega",
        "transformer": {
            "n_layers": args.n_layers,
            "probe_layer": args.probe_layer,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "d_mlp": args.d_mlp,
            "seq_len": args.seq_len,
            "context_len": args.context_len,
            "transformer_steps": args.transformer_steps,
            "transformer_batch_size": args.transformer_batch_size,
            "extract_batch_size": args.extract_batch_size,
            "streaming_batches": True,
        },
        "fig4_architecture_configs": ARCH_CONFIGS,
        "seeds": args.seeds,
        "variants": {},
    }

    for variant_name, variant_cfg in VARIANTS.items():
        print(f"\n=== Variant: {variant_name} ===", flush=True)
        arch_runs: dict[str, list[dict[str, Any]]] = {arch_name: [] for arch_name in ARCH_CONFIGS}
        transformer_runs = []

        for seed in args.seeds:
            print(f"  seed={seed}", flush=True)
            vocab_size = (
                MESS3_SEPARATED_COMPONENTS[0].V
                if variant_cfg["vocab_mode"] == "shared"
                else sum(comp.V for comp in MESS3_SEPARATED_COMPONENTS)
            )
            cfg = HookedTransformerConfig(
                n_layers=args.n_layers,
                d_model=args.d_model,
                n_ctx=args.seq_len,
                d_head=args.d_model // args.n_heads,
                n_heads=args.n_heads,
                d_mlp=args.d_mlp,
                d_vocab=vocab_size,
                d_vocab_out=vocab_size,
                act_fn="gelu",
                normalization_type="LN",
                attention_dir="causal",
                default_prepend_bos=False,
                device=str(device),
                seed=seed,
            )
            transformer = HookedTransformer(cfg).to(device)

            print(
                f"    training transformer (layers={args.n_layers}, d_model={args.d_model}, "
                f"context={args.context_len}, seq_len={args.seq_len}, steps={args.transformer_steps}, "
                "streaming data)",
                flush=True,
            )
            def sample_train_tokens(step_idx: int, batch_size: int) -> torch.Tensor:
                step_seed = seed * 1_000_000 + step_idx
                batch = generate_standard_hmm_data_simplexity(
                    components=MESS3_SEPARATED_COMPONENTS,
                    omega_prior=[0.4, 0.35, 0.25],
                    n_sequences=batch_size,
                    seq_len=args.seq_len,
                    mode=variant_cfg["mode"],
                    vocab_mode=variant_cfg["vocab_mode"],
                    seed=step_seed,
                    concentration=args.concentration,
                )
                return batch.tokens

            transformer_losses = train_transformer(
                transformer,
                sample_train_tokens,
                n_steps=args.transformer_steps,
                batch_size=args.transformer_batch_size,
                lr=1e-3,
                seed=seed,
            )
            transformer_runs.append(
                {
                    "seed": seed,
                    "final_loss": float(transformer_losses[-1]),
                }
            )
            if args.save_checkpoints:
                save_checkpoint(
                    args.checkpoint_dir / variant_name / "transformer" / f"seed_{seed}.pt",
                    {
                        "kind": "transformer",
                        "seed": seed,
                        "variant": variant_name,
                        "mode": variant_cfg["mode"],
                        "vocab_mode": variant_cfg["vocab_mode"],
                        "config": {
                            "n_layers": args.n_layers,
                            "probe_layer": args.probe_layer,
                            "d_model": args.d_model,
                            "n_heads": args.n_heads,
                            "d_mlp": args.d_mlp,
                            "seq_len": args.seq_len,
                            "context_len": args.context_len,
                            "transformer_steps": args.transformer_steps,
                            "transformer_batch_size": args.transformer_batch_size,
                            "analysis_train_sequences": args.train_sequences,
                            "analysis_eval_sequences": args.eval_sequences,
                            "streaming_batches": True,
                        },
                        "final_loss": float(transformer_losses[-1]),
                        "state_dict": cpu_state_dict(transformer),
                    },
                )

            train_data = generate_standard_hmm_data_simplexity(
                components=MESS3_SEPARATED_COMPONENTS,
                omega_prior=[0.4, 0.35, 0.25],
                n_sequences=args.train_sequences,
                seq_len=args.seq_len,
                mode=variant_cfg["mode"],
                vocab_mode=variant_cfg["vocab_mode"],
                seed=seed + 10_000,
                concentration=args.concentration,
            )
            eval_data = generate_standard_hmm_data_simplexity(
                components=MESS3_SEPARATED_COMPONENTS,
                omega_prior=[0.4, 0.35, 0.25],
                n_sequences=args.eval_sequences,
                seq_len=args.seq_len,
                mode=variant_cfg["mode"],
                vocab_mode=variant_cfg["vocab_mode"],
                seed=seed + 20_000,
                concentration=args.concentration,
            )

            print(f"    extracting activations from {hook_name}", flush=True)
            train_acts = extract_position_activations(
                transformer,
                train_data.tokens,
                hook_name=hook_name,
                chunk_size=args.extract_batch_size,
            )
            eval_acts = extract_position_activations(
                transformer,
                eval_data.tokens,
                hook_name=hook_name,
                chunk_size=args.extract_batch_size,
            )

            for arch_name, arch_cfg in ARCH_CONFIGS.items():
                print(f"    {arch_name}", flush=True)
                if arch_cfg["family"] == "topk":
                    run = evaluate_topk_on_activations(
                        seed,
                        arch_cfg,
                        train_acts,
                        eval_acts,
                        eval_data.sequence_omegas,
                        d_model=args.d_model,
                        device=device,
                        checkpoint_path=(
                            args.checkpoint_dir / variant_name / slugify(arch_name) / f"seed_{seed}.pt"
                            if args.save_checkpoints
                            else None
                        ),
                    )
                elif arch_cfg["family"] == "txc":
                    run = evaluate_txc_on_activations(
                        seed,
                        arch_cfg,
                        train_acts,
                        eval_acts,
                        eval_data.sequence_omegas,
                        d_model=args.d_model,
                        device=device,
                        matryoshka=False,
                        checkpoint_path=(
                            args.checkpoint_dir / variant_name / slugify(arch_name) / f"seed_{seed}.pt"
                            if args.save_checkpoints
                            else None
                        ),
                    )
                elif arch_cfg["family"] == "mattxc":
                    run = evaluate_txc_on_activations(
                        seed,
                        arch_cfg,
                        train_acts,
                        eval_acts,
                        eval_data.sequence_omegas,
                        d_model=args.d_model,
                        device=device,
                        matryoshka=True,
                        checkpoint_path=(
                            args.checkpoint_dir / variant_name / slugify(arch_name) / f"seed_{seed}.pt"
                            if args.save_checkpoints
                            else None
                        ),
                    )
                else:
                    raise ValueError(f"Unknown family: {arch_cfg['family']}")

                single = run["metrics"]["single_feature_probe"]["best_feature_r2"]
                linear = run["metrics"]["linear_probe"]["mean_r2"]
                print(f"      single={single:.4f}  linear={linear:.4f}", flush=True)
                arch_runs[arch_name].append(run)

            del transformer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        variant_record: dict[str, Any] = {
            "mode": variant_cfg["mode"],
            "vocab_mode": variant_cfg["vocab_mode"],
            "label": variant_cfg["label"],
            "transformer_runs": transformer_runs,
            "architectures": {},
        }
        for arch_name, arch_cfg in ARCH_CONFIGS.items():
            variant_record["architectures"][arch_name] = {
                "config": arch_cfg,
                "runs": arch_runs[arch_name],
                "summary": summarize_runs(arch_runs[arch_name]),
            }
        results["variants"][variant_name] = variant_record

    args.output_json.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
