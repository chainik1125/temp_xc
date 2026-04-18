"""Unified data pipeline for architecture comparison experiments.

Supports two data generation modes:

1. **Standard** (coupling=None): K independent Markov chains, each controlling
   one feature. Supports leaky reset via MarkovConfig.delta.

2. **Coupled** (coupling=CouplingConfig): K hidden Markov chains produce M > K
   emission features through a coupling matrix. The pipeline provides both
   emission_features (local ground truth) and hidden_features (global ground truth)
   for dual-AUC evaluation.

Usage:
    # Standard mode
    pipeline = build_data_pipeline(config, device, window_sizes=[2, 5])

    # Coupled mode
    config = DataConfig(coupling=CouplingConfig(K_hidden=10, M_emission=20))
    pipeline = build_data_pipeline(config, device, window_sizes=[2, 5])
    pipeline.global_features  # (hidden_dim, K) -- for global AUC
"""

import json
import math
import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from src.training.config import CouplingConfig, DataConfig
from src.data.toy.coupling import (
    apply_coupling,
    compute_hidden_features,
    generate_coupling_matrix,
)
from src.utils.orthogonalize import orthogonalize


class ToyModel(nn.Module):
    """Linear embedding toy model with orthogonal feature directions.

    Maps sparse feature activations (n_features,) to a dense hidden
    representation (hidden_dim,) via a fixed linear embedding.
    """

    def __init__(self, num_features: int, hidden_dim: int, target_cos_sim: float = 0.0):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim

        embeddings = orthogonalize(
            num_vectors=num_features,
            vector_len=hidden_dim,
            target_cos_sim=target_cos_sim,
        )
        self.register_buffer("W", embeddings)  # (num_features, hidden_dim)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        return activations @ self.W

    @property
    def feature_directions(self) -> torch.Tensor:
        """(hidden_dim, num_features) feature direction matrix."""
        return self.W.T


def _generate_markov_support(
    n_seq: int,
    seq_len: int,
    num_features: int,
    pi: float,
    rho_eff: float,
    device: torch.device,
) -> torch.Tensor:
    """Generate binary support via independent 2-state Markov chains.

    Uses rho_eff (effective autocorrelation, already accounting for any leak).

    Returns binary tensor of shape (n_seq, seq_len, num_features).
    """
    p01 = pi * (1.0 - rho_eff)  # off -> on
    p10 = (1.0 - pi) * (1.0 - rho_eff)  # on -> off

    support = torch.zeros(n_seq, seq_len, num_features, device=device)
    support[:, 0, :] = (torch.rand(n_seq, num_features, device=device) < pi).float()

    for t in range(1, seq_len):
        prev = support[:, t - 1, :]
        u = torch.rand(n_seq, num_features, device=device)
        turn_on = (prev == 0) & (u < p01)
        stay_on = (prev == 1) & (u >= p10)
        support[:, t, :] = (turn_on | stay_on).float()

    return support


def _generate_activations(
    n_seq: int,
    seq_len: int,
    num_features: int,
    pi: float,
    rho_eff: float,
    device: torch.device,
) -> torch.Tensor:
    """Generate support * magnitudes for standard (non-coupled) mode.

    Returns activations of shape (n_seq, seq_len, num_features).
    """
    support = _generate_markov_support(n_seq, seq_len, num_features, pi, rho_eff, device)
    magnitudes = torch.randn(n_seq, seq_len, num_features, device=device).abs()
    return support * magnitudes


def _generate_coupled_activations(
    n_seq: int,
    seq_len: int,
    coupling_cfg: CouplingConfig,
    coupling_matrix: torch.Tensor,
    pi: float,
    rho_eff: float,
    device: torch.device,
) -> torch.Tensor:
    """Generate activations for coupled-feature mode.

    K hidden chains -> coupling matrix -> M emission support -> magnitudes.

    Returns activations of shape (n_seq, seq_len, M_emission).
    """
    K = coupling_cfg.K_hidden
    M = coupling_cfg.M_emission

    # Generate K hidden chains: (n_seq, seq_len, K)
    hidden_support = _generate_markov_support(n_seq, seq_len, K, pi, rho_eff, device)

    # Map to M emissions via coupling: need (n_seq, K, seq_len) for apply_coupling
    hidden_kT = hidden_support.permute(0, 2, 1)  # (n_seq, K, seq_len)
    C = coupling_matrix.to(device)

    if coupling_cfg.emission_mode == "or":
        emission_support_kT = apply_coupling(hidden_kT, C, mode="or")
    else:
        # torch.Generator only supports CPU; for CUDA tensors, apply_coupling
        # falls back to unseeded torch.rand_like
        rng = torch.Generator()
        emission_support_kT = apply_coupling(
            hidden_kT, C, mode="sigmoid",
            alpha=coupling_cfg.sigmoid_alpha,
            beta=coupling_cfg.sigmoid_beta,
            rng=rng,
        )

    # (n_seq, M, seq_len) -> (n_seq, seq_len, M)
    emission_support = emission_support_kT.permute(0, 2, 1)

    magnitudes = torch.randn(n_seq, seq_len, M, device=device).abs()
    return emission_support * magnitudes


@dataclass
class DataPipeline:
    """Pre-built data generators and eval data. Created once, shared across models.

    For coupled-feature mode, global_features provides the K hidden-state-level
    ground truth directions for computing global AUC separately from local AUC.
    """

    toy_model: ToyModel
    scaling_factor: float
    true_features: torch.Tensor  # (hidden_dim, num_features) -- local/emission features
    eval_hidden: torch.Tensor  # (n_seq, seq_len, hidden_dim)
    gen_flat: Callable[[int], torch.Tensor]
    gen_seq: Callable[[int], torch.Tensor]
    gen_windows: dict[int, Callable[[int], torch.Tensor]] = field(
        default_factory=dict
    )
    global_features: torch.Tensor | None = None  # (hidden_dim, K) -- coupled mode only


def _compute_scaling_factor(
    toy_model: ToyModel,
    gen_activations: Callable,
    hidden_dim: int,
    device: torch.device,
    n_samples: int = 10_000,
    seq_len: int = 64,
) -> float:
    """Compute scaling factor: sqrt(hidden_dim) / mean(||hidden||)."""
    with torch.no_grad():
        acts = gen_activations(n_samples, seq_len)
        hidden = toy_model(acts)
        mean_norm = hidden.reshape(-1, hidden_dim).norm(dim=-1).mean().item()
        return math.sqrt(hidden_dim) / mean_norm


def build_data_pipeline(
    config: DataConfig,
    device: torch.device,
    window_sizes: list[int] | None = None,
) -> DataPipeline:
    """Build all data generators and eval data from a DataConfig.

    Supports both standard (independent features) and coupled (K hidden -> M emission)
    data generation modes, including leaky reset via MarkovConfig.delta.

    Args:
        config: Data configuration.
        device: Torch device.
        window_sizes: List of T values for crosscoder window generators.

    Returns:
        Fully constructed DataPipeline ready for sweep use.
    """
    torch.manual_seed(config.seed)

    is_coupled = config.coupling is not None
    pi = config.markov.pi
    rho_eff = config.markov.rho_eff
    hidden_dim = config.toy_model.hidden_dim
    seq_len = config.seq_len

    if is_coupled:
        coupling_cfg = config.coupling
        M = coupling_cfg.M_emission
        K = coupling_cfg.K_hidden
        num_obs_features = M  # SAE sees M emission features

        # Build toy model with M emission features
        toy_model = ToyModel(
            num_features=M,
            hidden_dim=hidden_dim,
            target_cos_sim=config.toy_model.target_cos_sim,
        ).to(device)
        toy_model.eval()

        # Generate coupling matrix (deterministic given seed)
        rng = torch.Generator().manual_seed(config.seed + 200)
        coupling_matrix = generate_coupling_matrix(K, M, coupling_cfg.n_parents, rng)

        # Global features: (K, hidden_dim) -> transpose to (hidden_dim, K)
        emission_features_kd = toy_model.W  # (M, hidden_dim)
        hidden_features_kd = compute_hidden_features(emission_features_kd, coupling_matrix)
        global_features = hidden_features_kd.T.to(device)  # (hidden_dim, K)

        true_features = toy_model.feature_directions  # (hidden_dim, M)

        def _gen_acts(n_seq, sl):
            return _generate_coupled_activations(
                n_seq, sl, coupling_cfg, coupling_matrix, pi, rho_eff, device,
            )

    else:
        num_obs_features = config.toy_model.num_features
        global_features = None

        toy_model = ToyModel(
            num_features=num_obs_features,
            hidden_dim=hidden_dim,
            target_cos_sim=config.toy_model.target_cos_sim,
        ).to(device)
        toy_model.eval()

        true_features = toy_model.feature_directions

        def _gen_acts(n_seq, sl):
            return _generate_activations(n_seq, sl, num_obs_features, pi, rho_eff, device)

    # Compute scaling factor
    sf = _compute_scaling_factor(toy_model, _gen_acts, hidden_dim, device,
                                 seq_len=seq_len)

    def gen_flat(batch_size: int) -> torch.Tensor:
        n_seq = max(1, batch_size // seq_len)
        acts = _gen_acts(n_seq, seq_len)
        return (toy_model(acts) * sf).reshape(-1, hidden_dim)[:batch_size]

    def gen_seq(n_seq: int) -> torch.Tensor:
        acts = _gen_acts(n_seq, seq_len)
        return toy_model(acts) * sf

    gen_windows: dict[int, Callable[[int], torch.Tensor]] = {}
    for T in window_sizes or []:
        def _make_window_gen(T_: int) -> Callable[[int], torch.Tensor]:
            def gen(batch_size: int) -> torch.Tensor:
                n_seq = max(1, batch_size // (seq_len - T_ + 1)) + 1
                acts = _gen_acts(n_seq, seq_len)
                hidden = toy_model(acts) * sf
                windows = []
                for t in range(seq_len - T_ + 1):
                    windows.append(hidden[:, t : t + T_, :])
                all_w = torch.cat(windows, dim=0)
                idx = torch.randperm(all_w.shape[0], device=device)[:batch_size]
                return all_w[idx]
            return gen
        gen_windows[T] = _make_window_gen(T)

    # Build eval data
    torch.manual_seed(config.seed + 100)
    eval_acts = _gen_acts(config.eval_n_seq, seq_len)
    eval_hidden = toy_model(eval_acts) * sf

    return DataPipeline(
        toy_model=toy_model,
        scaling_factor=sf,
        true_features=true_features,
        eval_hidden=eval_hidden,
        gen_flat=gen_flat,
        gen_seq=gen_seq,
        gen_windows=gen_windows,
        global_features=global_features,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Real-LM cached activations
# ═══════════════════════════════════════════════════════════════════════════
def _shuffle_within_sequence_(x: torch.Tensor) -> torch.Tensor:
    """Randomly permute the sequence axis independently per sample.

    x: (B, T, d). Returns a shuffled view (new tensor). This is the temporal
    shuffled-control transform — any metric that drops after applying this
    was measuring genuinely temporal structure, anything that survives was
    measuring something else (the TFA free-dense-channel problem).
    """
    B, T, _ = x.shape
    idx = torch.stack([torch.randperm(T, device=x.device) for _ in range(B)])
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def build_cached_activations_pipeline(
    config: DataConfig,
    device: torch.device,
    window_sizes: list[int] | None = None,
) -> DataPipeline:
    """Load pre-cached real-LM activations and wrap them in a DataPipeline.

    Expects output from `src/bench/cache_activations.py`:
        <root>/cached_activations/<model_name>/<cached_dataset>/
            <layer_key>.npy           (N, seq_len, d_model) float32
            layer_specs.json          metadata
            token_ids.npy             optional, for autointerp

    Returns a DataPipeline where `true_features` / `global_features` are None
    (no ground truth on real data) but all sampling generators are populated.
    """
    # Resolve the cache directory via config or fall back to the default.
    if config.cached_root is not None:
        root = config.cached_root
    else:
        from src.data.nlp.cache_config import cache_dir_for
        root = cache_dir_for(config.model_name, config.cached_dataset)

    acts_path = os.path.join(root, f"{config.cached_layer_key}.npy")
    if not os.path.exists(acts_path):
        raise FileNotFoundError(
            f"No cached activations at {acts_path}. "
            f"Run: python -m src.data.nlp.cache_activations "
            f"--model {config.model_name} --dataset {config.cached_dataset}"
        )

    print(f"[data] loading cached activations {acts_path}")
    arr = np.load(acts_path, mmap_mode="r")  # (N, T, d)
    n_seq, seq_len, d_model = arr.shape
    print(f"  shape=({n_seq},{seq_len},{d_model}) shuffle={config.shuffle_within_sequence}")

    specs_path = os.path.join(root, "layer_specs.json")
    if os.path.exists(specs_path):
        with open(specs_path) as f:
            _meta = json.load(f)
        assert _meta["d_model"] == d_model, "cached d_model does not match layer_specs.json"

    # Memory-efficient loading: keep the numpy mmap and load slices on demand.
    # At 24K x 128 x 2304 x 4 bytes = 28 GB, materializing the full array
    # would exhaust system RAM. Instead, generators index into the mmap
    # and convert small batches to torch tensors per-call.

    max_eval = max(1, n_seq // 5)  # 20% of data, at minimum 1
    n_eval = min(config.eval_n_seq, max_eval)
    train_N = n_seq - n_eval
    train_NT = train_N * seq_len
    if train_N == 0:
        raise ValueError(
            f"train_pool empty: n_seq={n_seq} n_eval={n_eval}. "
            f"Reduce DataConfig.eval_n_seq or cache more sequences."
        )

    # Build a shuffle permutation if requested. Applied as an index remap
    # rather than materializing a shuffled copy of the full array.
    if config.shuffle_within_sequence:
        # Build per-sequence position permutations
        rng = np.random.RandomState(config.seed)
        shuf_perms = np.stack([rng.permutation(seq_len) for _ in range(n_seq)])
    else:
        shuf_perms = None

    def _load_seqs(indices) -> torch.Tensor:
        """Load sequences by index from the mmap, apply shuffle if needed."""
        batch_np = np.array(arr[indices])  # copy from mmap: (n, S, d)
        if shuf_perms is not None:
            perms = shuf_perms[indices]  # (n, S)
            # Vectorized fancy-index shuffle along the seq dimension
            n, S, d = batch_np.shape
            row_idx = np.arange(n)[:, None].repeat(S, axis=1)
            batch_np = batch_np[row_idx, perms]
        return torch.from_numpy(batch_np).float()

    # Eval slice: small enough to hold in GPU memory (n_eval << n_seq).
    eval_hidden = _load_seqs(list(range(n_seq - n_eval, n_seq))).to(device)
    print(f"  train_N={train_N}, eval_N={n_eval}, eval_hidden={eval_hidden.shape}")

    def gen_flat(batch_size: int) -> torch.Tensor:
        idx = torch.randint(0, train_NT, (batch_size,)).numpy()
        seq_idx = idx // seq_len
        tok_idx = idx % seq_len
        batch_np = np.array(arr[seq_idx, tok_idx])  # (B, d)
        # No shuffle needed for flat (single tokens)
        return torch.from_numpy(batch_np).float().to(device, non_blocking=True)

    def gen_seq(n: int) -> torch.Tensor:
        idx = torch.randint(0, train_N, (n,)).numpy()
        return _load_seqs(idx).to(device, non_blocking=True)

    gen_windows: dict[int, Callable[[int], torch.Tensor]] = {}
    for T in window_sizes or []:
        def _make(T_: int) -> Callable[[int], torch.Tensor]:
            def gen(batch_size: int) -> torch.Tensor:
                n = max(1, batch_size // max(1, seq_len - T_ + 1)) + 1
                idx = torch.randint(0, train_N, (n,)).numpy()
                seqs = _load_seqs(idx)  # (n, S, d)
                wins = [seqs[:, t : t + T_, :] for t in range(seq_len - T_ + 1)]
                all_w = torch.cat(wins, dim=0)
                pick = torch.randperm(all_w.shape[0])[:batch_size]
                return all_w[pick].to(device, non_blocking=True)
            return gen
        gen_windows[T] = _make(T)

    # Dummy toy_model + scaling — not used on real data, but the dataclass
    # requires them. Keep lightweight.
    toy_model = nn.Module()
    toy_model.hidden_dim = d_model  # type: ignore[attr-defined]

    return DataPipeline(
        toy_model=toy_model,  # type: ignore[arg-type]
        scaling_factor=1.0,
        true_features=torch.zeros(d_model, 1),  # placeholder — recovery AUC is skipped
        eval_hidden=eval_hidden,
        gen_flat=gen_flat,
        gen_seq=gen_seq,
        gen_windows=gen_windows,
        global_features=None,
    )


def build_pipeline(
    config: DataConfig,
    device: torch.device,
    window_sizes: list[int] | None = None,
) -> DataPipeline:
    """Single entry point that dispatches on config.dataset_type."""
    if config.dataset_type == "markov":
        return build_data_pipeline(config, device, window_sizes)
    if config.dataset_type == "cached_activations":
        return build_cached_activations_pipeline(config, device, window_sizes)
    raise ValueError(f"Unknown dataset_type: {config.dataset_type}")
