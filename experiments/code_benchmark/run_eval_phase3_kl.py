"""Phase 3 — LM prediction fidelity, stratified by history dependence.

Per token, we compute:

    * KL( P(x_{t+1} | x_{1:t})  ||  P(x_{t+1} | x̂_{1:t}) )
      where ``x̂`` is the architecture's reconstruction patched back into
      ``blocks.{anchor_layer}.hook_resid_post``.
    * surprisal_delta(t) = H(x_{t+1} | x_{t-k:t}) − H(x_{t+1} | x_{1:t})
      where k is a short context length — large values identify tokens where
      long-range history matters for the LM prediction.

We then bin tokens by surprisal_delta quantile and plot mean KL per bin, one
curve per architecture. This identifies the "TempXC regime" from data rather
than from curated categories.

Outputs::

    results/phase3_<arch>.json
    results/phase3_summary.json
    plots/phase3_kl_vs_surprisal_delta.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from code_pipeline.python_code import SubjectModelConfig, load_cache  # noqa: E402
from code_pipeline.eval_utils import build_model_from_checkpoint  # noqa: E402


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(spec)


@torch.no_grad()
def lm_logits_and_surprisal(
    lm,
    tokens: torch.Tensor,          # (B, T) int64 on device
    short_context_len: int,
    anchor_hook: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (logits_full, logits_short, surprisal_full, surprisal_short) at
    shape ``(B, T-1, V)`` / ``(B, T-1)``.

    surprisal at position t is ``-log P(x_{t+1} | ...)``; surprisal_delta is
    ``surprisal_short - surprisal_full``.

    We emulate the "short-context" distribution by re-running the LM on
    shifted inputs: for each B, we tile token windows of length
    ``short_context_len + 1`` and take the logit at the last position.
    """
    logits_full = lm(tokens, return_type="logits")
    # surprisal at position t is -logP(token[t+1] | ...)
    log_probs_full = F.log_softmax(logits_full[:, :-1, :], dim=-1)
    targets = tokens[:, 1:]
    surprisal_full = -log_probs_full.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    # short-context surprisal: for every position t, rebuild a length-(k+1)
    # prefix ending at t and take only its last position's logit. Done
    # vectorized via torch.nn.functional.unfold-like construction on tokens.
    k = short_context_len
    T = tokens.shape[1]
    pad_tok = tokens[:, :1].repeat(1, k)      # pad with BOS-like replication
    tokens_padded = torch.cat([pad_tok, tokens], dim=1)       # (B, k+T)
    windows: list[torch.Tensor] = []
    for t in range(T - 1):
        windows.append(tokens_padded[:, t + 1 : t + 1 + (k + 1)])
    W = torch.stack(windows, dim=1)           # (B, T-1, k+1)
    B_size, L, _ = W.shape
    W_flat = W.reshape(B_size * L, k + 1)
    logits_short_flat = lm(W_flat, return_type="logits")       # (B*L, k+1, V)
    logits_short = logits_short_flat[:, -1, :].reshape(B_size, L, -1)
    log_probs_short = F.log_softmax(logits_short, dim=-1)
    surprisal_short = -log_probs_short.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    return logits_full, logits_short, surprisal_full, surprisal_short


@torch.no_grad()
def kl_from_logits(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    """KL(P_a || P_b) per (B, T) position."""
    log_p = F.log_softmax(logits_a, dim=-1)
    log_q = F.log_softmax(logits_b, dim=-1)
    p = log_p.exp()
    return (p * (log_p - log_q)).sum(dim=-1)


@torch.no_grad()
def patched_logits_for_arch(
    lm,
    tokens: torch.Tensor,
    acts_anchor: torch.Tensor,
    acts_mlc: dict[int, torch.Tensor] | None,
    model,
    family: str,
    anchor_hook: str,
    subject_cfg: SubjectModelConfig,
) -> torch.Tensor:
    """Forward ``lm`` with a hook replacing anchor activations with ``x̂``.

    Returns logits of shape ``(B, T, V)``.
    """
    device = next(lm.parameters()).device
    # Ensure the SAE/TXC/MLC model is on the same device as the LM before we
    # start feeding it activations. build_model_from_checkpoint leaves it on
    # CPU; subsequent `x.to(device)` would then mix cuda inputs with cpu
    # weights and raise a device mismatch.
    model = model.to(device)
    B, T, d = acts_anchor.shape
    lm_dtype = lm.cfg.dtype
    model_dtype = next(model.parameters()).dtype

    if family == "topk":
        flat = acts_anchor.reshape(-1, d).to(device, dtype=model_dtype)
        xh_chunks = []
        for i in range(0, flat.shape[0], 4096):
            xb, _ = model(flat[i : i + 4096])
            xh_chunks.append(xb)
        xh_all = torch.cat(xh_chunks, dim=0).reshape(B, T, d)
    elif family == "txc":
        w = model.T
        xh_all = acts_anchor.clone().to(device, dtype=model_dtype)
        for b_idx in range(B):
            seq = acts_anchor[b_idx].to(device, dtype=model_dtype)     # (T, d)
            windows = seq.unfold(0, w, 1).permute(0, 2, 1).contiguous()
            xb, _ = model(windows)                                      # (T-w+1, w, d)
            xh_all[b_idx, w - 1:] = xb[:, -1, :]
    elif family == "mlxc":
        assert acts_mlc is not None
        layers = subject_cfg.mlc_layers
        anchor_in_stack = len(layers) // 2
        xh_all = torch.zeros(B, T, d, device=device, dtype=model_dtype)
        for b_idx in range(B):
            stack = torch.stack([acts_mlc[L][b_idx] for L in layers], dim=1)  # (T, L, d)
            stack = stack.to(device, dtype=model_dtype)
            xb, _ = model(stack)
            xh_all[b_idx] = xb[:, anchor_in_stack, :]
    else:
        raise ValueError(family)

    xh_all = xh_all.to(device, dtype=lm_dtype)

    def _hook(act, hook, _o=xh_all):
        return _o

    logits = lm.run_with_hooks(
        tokens.to(device),
        fwd_hooks=[(anchor_hook, _hook)],
        return_type="logits",
    )
    return logits


def plot_kl_vs_surprisal_delta(
    per_arch_bins: dict[str, list[float]],
    bin_edges: list[float],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
    labels = [f"q{int(q*100)}" for q in bin_edges[:-1]]
    for name, vals in per_arch_bins.items():
        ax.plot(labels, vals, marker="o", label=name)
    ax.set_xlabel("surprisal-delta quantile")
    ax.set_ylabel("mean KL(clean || patched)")
    ax.set_title("Phase 3 — LM KL vs history-dependence")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = resolve_device(args.device or cfg.get("device", "auto"))
    seed = int(cfg.get("seed", 42))
    ph3 = cfg["phase3_kl"]

    cache_root = HERE / cfg.get("cache_root", "cache")
    checkpoint_root = HERE / cfg.get("checkpoint_root", "checkpoints")
    results_root = HERE / cfg.get("output_root", "results")
    plot_root = HERE / cfg.get("plot_root", "plots")

    subject_cfg = SubjectModelConfig.from_dict(cfg["subject_model"])
    layers = subject_cfg.required_layers()
    tokens, _sources, acts_by_layer, _ = load_cache(cache_root, layers)
    split = torch.load(cache_root / "split.pt")
    eval_idx = split["eval_idx"][: ph3["n_sequences"]]
    tokens_eval = tokens[eval_idx]
    acts_anchor = acts_by_layer[subject_cfg.anchor_layer][eval_idx].float()
    acts_mlc = {L: acts_by_layer[L][eval_idx].float() for L in subject_cfg.mlc_layers}

    from transformer_lens import HookedTransformer
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[cfg.get("dtype", "bfloat16")]
    lm = HookedTransformer.from_pretrained(subject_cfg.name, device=device, dtype=dtype)
    lm.eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    anchor_hook = f"blocks.{subject_cfg.anchor_layer}.hook_resid_post"
    print(f"[phase3] lm ready, eval_chunks={tokens_eval.shape[0]}, "
          f"anchor_hook={anchor_hook}", flush=True)

    # ---- clean + short-context surprisal ----
    clean_logits_list: list[torch.Tensor] = []
    surp_full_list: list[torch.Tensor] = []
    surp_short_list: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, tokens_eval.shape[0], args.batch_size):
            batch_tok = tokens_eval[i : i + args.batch_size].to(device)
            lf, _ls, sf, ss = lm_logits_and_surprisal(
                lm, batch_tok,
                short_context_len=ph3["surprisal_delta"]["short_context_len"],
                anchor_hook=anchor_hook,
            )
            clean_logits_list.append(lf.cpu())
            surp_full_list.append(sf.cpu())
            surp_short_list.append(ss.cpu())
            del lf, _ls, sf, ss
            if (i // args.batch_size) % 10 == 0:
                print(f"[phase3] clean-forward batch {i}/{tokens_eval.shape[0]}", flush=True)
    clean_logits = torch.cat(clean_logits_list, dim=0)                  # (B, T, V)
    surp_full = torch.cat(surp_full_list, dim=0)                        # (B, T-1)
    surp_short = torch.cat(surp_short_list, dim=0)                      # (B, T-1)
    # Cast to float32 before .numpy() — Gemma activations travel in bf16 and
    # numpy has no bf16 dtype.
    surprisal_delta = (surp_short - surp_full).float().numpy()

    per_arch_bins: dict[str, list[float]] = {}
    per_arch_kl_curves: dict[str, dict] = {}

    for arch in cfg["architectures"]:
        name = arch["name"]
        if args.only and name != args.only:
            continue
        ckpt_path = checkpoint_root / f"{name}.pt"
        if not ckpt_path.exists():
            print(f"[phase3] skip {name}: missing checkpoint")
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model, family = build_model_from_checkpoint(ckpt, subject_cfg.d_model)
        patched_logits_list: list[torch.Tensor] = []
        for i in range(0, tokens_eval.shape[0], args.batch_size):
            batch_tok = tokens_eval[i : i + args.batch_size]
            batch_anchor = acts_anchor[i : i + args.batch_size]
            batch_mlc = {L: acts_mlc[L][i : i + args.batch_size] for L in subject_cfg.mlc_layers}
            logits_p = patched_logits_for_arch(
                lm, batch_tok, batch_anchor, batch_mlc,
                model=model, family=family, anchor_hook=anchor_hook,
                subject_cfg=subject_cfg,
            )
            patched_logits_list.append(logits_p.cpu())
        patched_logits = torch.cat(patched_logits_list, dim=0)
        # KL per (B, T-1) at next-token positions
        kl_vals = kl_from_logits(clean_logits[:, :-1, :], patched_logits[:, :-1, :])
        kl_vals = kl_vals.float().numpy()

        # bin by surprisal_delta quantile
        edges = np.quantile(
            surprisal_delta.flatten(),
            ph3["quantile_bins"],
        )
        binned = []
        for i in range(len(edges) - 1):
            mask = (surprisal_delta >= edges[i]) & (surprisal_delta <= edges[i + 1])
            if mask.sum() > 0:
                binned.append(float(kl_vals[mask].mean()))
            else:
                binned.append(float("nan"))
        per_arch_bins[name] = binned
        per_arch_kl_curves[name] = {
            "bin_edges": [float(e) for e in edges],
            "mean_kl_per_bin": binned,
            "overall_mean_kl": float(kl_vals.mean()),
            "overall_mean_abs_surprisal_delta": float(np.abs(surprisal_delta).mean()),
        }
        with (results_root / f"phase3_{name}.json").open("w") as f:
            json.dump(per_arch_kl_curves[name], f, indent=2)
        print(f"[phase3] {name}: mean KL {kl_vals.mean():.4f}, "
              f"per-bin {binned}")

    results_root.mkdir(parents=True, exist_ok=True)
    with (results_root / "phase3_summary.json").open("w") as f:
        json.dump({
            "bins": per_arch_kl_curves,
            "surprisal_delta_stats": {
                "mean": float(surprisal_delta.mean()),
                "std": float(surprisal_delta.std()),
                "quantiles": ph3["quantile_bins"],
            },
        }, f, indent=2)
    plot_kl_vs_surprisal_delta(per_arch_bins, ph3["quantile_bins"],
                                plot_root / "phase3_kl_vs_surprisal_delta.png")
    print("[phase3] done")


if __name__ == "__main__":
    main()
