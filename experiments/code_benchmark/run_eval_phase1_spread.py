"""Phase 1 — per-feature temporal attribution spread.

For each architecture's top-activating features, compute integrated gradients
of the feature activation at the "anchor" position with respect to each input
position in the window. Summarize each feature's attribution distribution over
positions ``k ∈ {0, ..., T-1}`` by its effective spread (entropy + first
moment). Histogram per architecture.

For SAE the window degenerates to one position: spread is 0 by construction,
but we still run the script so every architecture appears in the dashboard.
For MLC the "positions" are *layers*, not tokens — we report the same metric
there for parity; MLC spread measures layer-spread, not time-spread.

Outputs:
    results/phase1_<arch>.json
    plots/phase1_spread_hist.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from code_pipeline.python_code import (  # noqa: E402
    SubjectModelConfig,
    load_cache,
)
from code_pipeline.eval_utils import (  # noqa: E402
    build_model_from_checkpoint,
)
from code_pipeline.training import (  # noqa: E402
    make_txc_windows,
    stack_mlc_layers,
)


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(spec)


# ---------------------------------------------------------------------------
# Integrated gradients for one feature over a windowed input
# ---------------------------------------------------------------------------


def integrated_gradient_attributions(
    model: torch.nn.Module,
    inputs: torch.Tensor,           # (B, T, d) or (B, d) for SAE
    feature_idx: int,
    steps: int = 20,
    baseline: torch.Tensor | None = None,
) -> torch.Tensor:
    """Integrated gradients of the ``feature_idx``-th pre-TopK latent.

    Returns a tensor shaped like ``inputs``, giving per-position
    attributions. Caller averages over ``B`` and reduces over ``d`` to get
    per-position scalars.

    Implementation:

        * ``z̃_α(x)`` is computed by the encoder of each architecture: we
          directly call the encoder's pre-TopK linear path so that the
          gradient is defined even when TopK would zero the feature for some
          inputs. For TopK SAE / TXC / MLC, pre-TopK is an affine function of
          the input.

    """
    if baseline is None:
        baseline = torch.zeros_like(inputs)
    alphas = torch.linspace(0.0, 1.0, steps=steps, device=inputs.device)
    grads = torch.zeros_like(inputs)

    for a in alphas:
        x = baseline + a * (inputs - baseline)
        x = x.detach().requires_grad_(True)
        pre = _pre_topk_linear(model, x)
        feat = pre[..., feature_idx].sum()
        g = torch.autograd.grad(feat, x, create_graph=False, retain_graph=False)[0]
        grads += g.detach()

    grads /= steps
    return grads * (inputs - baseline)


def _pre_topk_linear(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return the pre-TopK linear activation vector for any of our architectures."""
    if hasattr(model, "T") and hasattr(model, "W_enc") and model.W_enc.dim() == 3:
        # TemporalCrosscoder / MultiLayerCrosscoder: shared latent from window sum
        pre = torch.einsum("btd,tdm->bm", x, model.W_enc) + model.b_enc
        return torch.relu(pre) if getattr(model, "use_relu", True) else pre
    # TopKSAE: x: (B, d); W_enc: (d, m)
    if hasattr(model, "W_enc") and model.W_enc.dim() == 2:
        pre = x @ model.W_enc + model.b_enc
        return torch.relu(pre) if getattr(model, "use_relu", True) else pre
    raise ValueError("Unknown model architecture for pre-TopK extraction")


# ---------------------------------------------------------------------------
# Rank features by activation mass; compute spread per feature
# ---------------------------------------------------------------------------


def rank_features_by_mass(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
    top_n: int = 512,
) -> np.ndarray:
    """Rank latent features by total activation mass across the sample."""
    model.to(device).eval()
    total = None
    with torch.no_grad():
        for i in range(0, inputs.shape[0], batch_size):
            batch = inputs[i : i + batch_size].to(device)
            _, z = model(batch)
            mass = z.abs().sum(dim=0)
            total = mass if total is None else total + mass
    mass = total.cpu().numpy()
    order = np.argsort(-mass)
    return order[:top_n]


def compute_spread_for_features(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    feature_indices: np.ndarray,
    family: str,
    device: torch.device,
    ig_steps: int,
    batch_size: int = 32,
) -> dict[int, dict[str, float]]:
    """For each feature_idx, attribute over positions. Return spread stats.

    For SAE (non-windowed), return a degenerate spread (entropy 0, mean 0).
    """
    out: dict[int, dict[str, float]] = {}
    model.to(device).eval()
    if family == "topk":
        for fi in feature_indices:
            out[int(fi)] = {"entropy": 0.0, "first_moment": 0.0,
                             "positions": [1.0]}
        return out

    n = inputs.shape[0]
    T = inputs.shape[1]
    for fi in feature_indices:
        per_pos_accum = torch.zeros(T, device=device)
        count = 0
        for i in range(0, n, batch_size):
            batch = inputs[i : i + batch_size].to(device)
            attr = integrated_gradient_attributions(model, batch, int(fi),
                                                    steps=ig_steps)
            # reduce over d to get per-position magnitude
            per_pos = attr.abs().sum(dim=-1).mean(dim=0)          # (T,)
            per_pos_accum += per_pos * batch.shape[0]
            count += batch.shape[0]
        per_pos = (per_pos_accum / count).cpu().numpy()
        total = per_pos.sum() + 1e-12
        p = per_pos / total
        entropy = float(-(p * np.log(np.clip(p, 1e-12, 1.0))).sum())
        first_moment = float(np.sum(np.arange(T) * p))
        out[int(fi)] = {
            "entropy": entropy,
            "first_moment": first_moment,
            "positions": p.tolist(),
        }
    return out


def plot_spread_histogram(
    per_arch: dict[str, dict[int, dict]],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, feats in per_arch.items():
        entropies = [v["entropy"] for v in feats.values()]
        moments = [v["first_moment"] for v in feats.values()]
        axes[0].hist(entropies, bins=20, alpha=0.5, label=name)
        axes[1].hist(moments, bins=20, alpha=0.5, label=name)
    axes[0].set_xlabel("position-entropy (nats)")
    axes[0].set_ylabel("# features")
    axes[0].set_title("Attribution entropy over positions")
    axes[0].legend()
    axes[1].set_xlabel("first moment (k)")
    axes[1].set_ylabel("# features")
    axes[1].set_title("Mean attribution position")
    axes[1].legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--only", type=str, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = resolve_device(args.device or cfg.get("device", "auto"))
    seed = int(cfg.get("seed", 42))
    ph1 = cfg["phase1_spread"]

    cache_root = HERE / cfg.get("cache_root", "cache")
    checkpoint_root = HERE / cfg.get("checkpoint_root", "checkpoints")
    results_root = HERE / cfg.get("output_root", "results")
    plot_root = HERE / cfg.get("plot_root", "plots")

    subject_cfg = SubjectModelConfig.from_dict(cfg["subject_model"])
    layers = subject_cfg.required_layers()
    _, _, acts_by_layer, _ = load_cache(cache_root, layers)
    split = torch.load(cache_root / "split.pt")
    eval_idx = split["eval_idx"]
    n_sample = min(ph1["n_samples"], len(eval_idx))
    eval_idx = eval_idx[:n_sample]
    acts_anchor = acts_by_layer[subject_cfg.anchor_layer][eval_idx].float()
    acts_mlc = {L: acts_by_layer[L][eval_idx].float() for L in subject_cfg.mlc_layers}

    per_arch_result: dict[str, dict] = {}
    for arch in cfg["architectures"]:
        name = arch["name"]
        if args.only and name != args.only:
            continue
        ckpt_path = checkpoint_root / f"{name}.pt"
        if not ckpt_path.exists():
            print(f"[phase1] skip {name}: missing checkpoint")
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model, family = build_model_from_checkpoint(ckpt, subject_cfg.d_model)

        if family == "topk":
            inputs = acts_anchor.reshape(-1, subject_cfg.d_model)
        elif family == "txc":
            inputs = make_txc_windows(acts_anchor, model.T)
        elif family == "mlxc":
            inputs = stack_mlc_layers(acts_mlc, subject_cfg.mlc_layers)
        else:
            raise ValueError(family)

        # cap the input pool for IG cost
        inputs = inputs[: ph1.get("n_samples", 2000)]
        top = rank_features_by_mass(model, inputs, device=device,
                                    top_n=ph1.get("top_features", 512))
        spread = compute_spread_for_features(
            model, inputs, top, family=family, device=device,
            ig_steps=ph1.get("ig_steps", 20),
        )
        per_arch_result[name] = spread
        print(f"[phase1] {name}: computed spread for {len(spread)} features")
        with (results_root / f"phase1_{name}.json").open("w") as f:
            json.dump(spread, f)

    if per_arch_result:
        plot_spread_histogram(per_arch_result, plot_root / "phase1_spread_hist.png")
    print("[phase1] done")


if __name__ == "__main__":
    main()
