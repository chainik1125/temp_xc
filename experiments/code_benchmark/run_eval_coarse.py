"""Coarse evaluation pass — the Pass-1 dashboard.

Produces, per architecture (topk_sae, txc_t5, mlc_l5):

    * NMSE, L0, explained variance on held-out code windows.
    * Loss-recovered on Gemma-2B-it: we patch ``x̂_t`` back into
      ``blocks.{anchor_layer}.hook_resid_post`` and re-forward. Report
      ``(CE_zero - CE_patched) / (CE_zero - CE_clean)``.
    * Per-category NMSE: bucketed by bracket_depth, indent_spaces, scope_kind.
    * Labelled UMAP of feature activations, colored by scope_kind and by
      bracket_depth bucket.

Outputs::

    results/coarse_<arch>.json        per-arch metrics
    results/coarse_summary.json       joined summary
    plots/coarse_loss_recovered.png
    plots/coarse_nmse_by_category.png
    plots/coarse_umap_<arch>.png
    plots/training_loss_curves.png    (already produced by run_training.py)

This pass does NOT require TransformerLens unless ``--with-lm`` is set. The
LM-KL / loss-recovered part is gated because it's expensive (needs Gemma in
memory a second time) and some users may want to inspect only the cheap
metrics first.
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
    CodeDatasetConfig,
    SubjectModelConfig,
    load_cache,
    cache_paths,
)
from code_pipeline.eval_utils import (  # noqa: E402
    FIELD_NAMES,
    build_model_from_checkpoint,
    bucket_labels_for,
    bucketize,
    encode_mlc_per_token,
    encode_sae_per_token,
    encode_txc_per_window,
    explained_variance,
    gather_labels,
    labels_for_sources,
    l0_mean,
    nmse,
    nmse_per_sample,
)


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(spec)


# ---------------------------------------------------------------------------
# Loss-recovered
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_loss_recovered(
    subject_cfg: SubjectModelConfig,
    tokens: torch.Tensor,             # (n_seq, T)
    acts_anchor: torch.Tensor,        # (n_seq, T, d)
    acts_mlc: dict[int, torch.Tensor],
    arch_name: str,
    arch_family: str,
    model: torch.nn.Module,
    device: torch.device,
    dtype_str: str = "bfloat16",
    batch_size: int = 2,
) -> dict:
    """Patch reconstructions back into Gemma, measure next-token CE.

    For TXC the reconstruction covers ``[t-T+1, t]`` — we plug in the last
    position of each window at position ``t``. Early positions (``t < T-1``)
    retain the original activation.

    Returns ``{"ce_clean", "ce_zero", "ce_patched", "loss_recovered"}``.
    """
    from transformer_lens import HookedTransformer
    lm = HookedTransformer.from_pretrained(
        subject_cfg.name, device=device,
        dtype={"bfloat16": torch.bfloat16, "float16": torch.float16,
               "float32": torch.float32}[dtype_str],
    )
    lm.eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    anchor_hook = f"blocks.{subject_cfg.anchor_layer}.hook_resid_post"

    # Build the patched residual tensor for each architecture
    model = model.to(device).eval()
    model_dtype = next(model.parameters()).dtype

    n_seq, T, d = acts_anchor.shape
    patched = acts_anchor.clone()

    if arch_family == "topk":
        flat = acts_anchor.reshape(-1, d).to(device, dtype=model_dtype)
        xh = torch.zeros_like(flat)
        for i in range(0, flat.shape[0], 4096):
            xb, _ = model(flat[i : i + 4096])
            xh[i : i + 4096] = xb
        patched = xh.reshape(n_seq, T, d).to(torch.float32).cpu()
    elif arch_family == "txc":
        window_size = model.T
        for s_idx in range(n_seq):
            seq_acts = acts_anchor[s_idx].to(device, dtype=model_dtype)  # (T, d)
            windows = seq_acts.unfold(0, window_size, 1).permute(0, 2, 1).contiguous()
            xb, _ = model(windows)
            # Replace positions [window_size-1 .. T-1] with the last token of each window
            last_recon = xb[:, -1, :]                             # (T-w+1, d)
            patched[s_idx, window_size - 1:] = last_recon.to(torch.float32).cpu()
    elif arch_family == "mlxc":
        L = model.L
        layers = subject_cfg.mlc_layers
        anchor_in_stack = len(layers) // 2
        for s_idx in range(n_seq):
            stack = torch.stack([acts_mlc[lyr][s_idx] for lyr in layers], dim=1)   # (T, L, d)
            stack = stack.to(device, dtype=model_dtype)
            xb, _ = model(stack)
            patched[s_idx] = xb[:, anchor_in_stack, :].to(torch.float32).cpu()
    else:
        raise ValueError(arch_family)

    # Now run LM three times: clean / zero / patched
    def ce_with_override(override: torch.Tensor | None) -> float:
        losses: list[float] = []
        for i in range(0, n_seq, batch_size):
            batch_tokens = tokens[i : i + batch_size].to(device)
            if override is None:
                fwd_hooks = []
            else:
                batch_override = override[i : i + batch_size].to(device, dtype=lm.cfg.dtype)
                def _hook(act, hook, _o=batch_override):
                    return _o
                fwd_hooks = [(anchor_hook, _hook)]
            loss = lm.run_with_hooks(
                batch_tokens, fwd_hooks=fwd_hooks, return_type="loss",
            )
            losses.append(float(loss))
        return float(np.mean(losses))

    ce_clean = ce_with_override(None)
    ce_zero = ce_with_override(torch.zeros_like(acts_anchor))
    ce_patched = ce_with_override(patched)

    denom = ce_zero - ce_clean + 1e-9
    loss_recovered = (ce_zero - ce_patched) / denom

    del lm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "ce_clean": ce_clean,
        "ce_zero": ce_zero,
        "ce_patched": ce_patched,
        "loss_recovered": float(loss_recovered),
    }


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------


def compute_umap(
    latents: np.ndarray,
    umap_cfg: dict,
    n_sample: int = 4000,
    seed: int = 42,
) -> np.ndarray:
    import umap  # noqa: WPS433 (optional dep)
    rng = np.random.default_rng(seed)
    n = min(latents.shape[0], n_sample)
    idx = rng.choice(latents.shape[0], size=n, replace=False)
    reducer = umap.UMAP(
        n_neighbors=umap_cfg.get("n_neighbors", 25),
        min_dist=umap_cfg.get("min_dist", 0.1),
        n_components=umap_cfg.get("n_components", 2),
        metric=umap_cfg.get("metric", "cosine"),
        random_state=seed,
    )
    emb = reducer.fit_transform(latents[idx])
    return emb, idx


def plot_umap(
    emb: np.ndarray,
    color_labels: np.ndarray,
    title: str,
    path: Path,
    cmap: str = "tab10",
) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=color_labels, s=3, cmap=cmap, alpha=0.7)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(sc, ax=ax, shrink=0.6)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-category NMSE
# ---------------------------------------------------------------------------


def per_category_nmse(
    x: np.ndarray,
    x_hat: np.ndarray,
    labels_1d: dict[str, np.ndarray],
    buckets_cfg: dict,
) -> dict:
    out: dict = {}
    per = nmse_per_sample(x, x_hat)
    # bracket depth
    for key, thresholds in [
        ("bracket_depth", buckets_cfg["bracket_depth"]),
        ("indent_spaces", buckets_cfg["indent_spaces"]),
    ]:
        buckets = bucketize(labels_1d[key], thresholds)
        lbl_names = bucket_labels_for(buckets, thresholds)
        per_bucket = {}
        for b in range(len(thresholds)):
            mask = buckets == b
            if mask.sum() > 0:
                per_bucket[lbl_names[b]] = {
                    "n": int(mask.sum()),
                    "nmse": float(per[mask].mean()),
                }
        out[key] = per_bucket
    # scope_kind
    scope_kinds = buckets_cfg["scope_kinds"]
    per_kind = {}
    for i, k in enumerate(scope_kinds):
        mask = labels_1d["scope_kind"] == i
        if mask.sum() > 0:
            per_kind[k] = {
                "n": int(mask.sum()),
                "nmse": float(per[mask].mean()),
            }
    out["scope_kind"] = per_kind
    return out


def plot_nmse_by_category(
    per_arch_category: dict[str, dict],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    archs = list(per_arch_category.keys())
    category_keys = list(per_arch_category[archs[0]].keys())
    fig, axes = plt.subplots(1, len(category_keys), figsize=(4 * len(category_keys), 4),
                              squeeze=False)
    for j, key in enumerate(category_keys):
        ax = axes[0, j]
        # Union of bucket labels across archs
        bucket_names: list[str] = []
        for a in archs:
            for b in per_arch_category[a][key]:
                if b not in bucket_names:
                    bucket_names.append(b)
        x = np.arange(len(bucket_names))
        width = 0.8 / max(1, len(archs))
        for i, a in enumerate(archs):
            vals = [per_arch_category[a][key].get(b, {"nmse": np.nan})["nmse"]
                    for b in bucket_names]
            ax.bar(x + i * width, vals, width, label=a)
        ax.set_xticks(x + width * (len(archs) - 1) / 2)
        ax.set_xticklabels(bucket_names, rotation=40, ha="right", fontsize=8)
        ax.set_title(key)
        ax.set_ylabel("NMSE")
        if j == 0:
            ax.legend(fontsize=8)
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
    parser.add_argument("--with-lm", action="store_true",
                        help="Compute loss-recovered (requires Gemma-2B-it in memory).")
    parser.add_argument("--only", type=str, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = resolve_device(args.device or cfg.get("device", "auto"))
    print(f"[eval_coarse] device={device}")
    seed = int(cfg.get("seed", 42))
    checkpoint_root = HERE / cfg.get("checkpoint_root", "checkpoints")
    results_root = HERE / cfg.get("output_root", "results")
    plot_root = HERE / cfg.get("plot_root", "plots")
    cache_root = HERE / cfg.get("cache_root", "cache")
    coarse_cfg = cfg["coarse_eval"]

    data_cfg = CodeDatasetConfig.from_dict(cfg["dataset"])
    subject_cfg = SubjectModelConfig.from_dict(cfg["subject_model"])

    layers = subject_cfg.required_layers()
    tokens, sources, acts_by_layer, manifest = load_cache(cache_root, layers)
    split = torch.load(cache_root / "split.pt")
    eval_idx = split["eval_idx"]
    eval_idx = eval_idx[: coarse_cfg["n_eval_windows"] // tokens.shape[1] + 1]

    tokens_eval = tokens[eval_idx]
    acts_anchor = acts_by_layer[subject_cfg.anchor_layer][eval_idx].float()
    acts_mlc = {L: acts_by_layer[L][eval_idx].float() for L in subject_cfg.mlc_layers}
    sources_eval = [sources[i] for i in eval_idx.tolist()]
    print(f"[eval_coarse] eval chunks: {tokens_eval.shape[0]}, T={tokens_eval.shape[1]}")

    # --- labels ---
    labels_nt = labels_for_sources(sources_eval)

    # --- per-arch eval ---
    per_arch_summary: dict[str, dict] = {}
    per_arch_category: dict[str, dict] = {}
    per_arch_umap: dict[str, dict] = {}

    for arch in cfg["architectures"]:
        name = arch["name"]
        if args.only and name != args.only:
            continue
        ckpt_path = checkpoint_root / f"{name}.pt"
        if not ckpt_path.exists():
            print(f"[eval_coarse] skipping {name}: no checkpoint at {ckpt_path}")
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model, family = build_model_from_checkpoint(ckpt, subject_cfg.d_model)
        print(f"[eval_coarse] {name} family={family}")

        # ---- encode ----
        if family == "topk":
            latents, x_hat, c_idx, t_idx = encode_sae_per_token(model, acts_anchor, device)
            x_at = acts_anchor.reshape(-1, subject_cfg.d_model).numpy()
        elif family == "txc":
            latents, x_hat, c_idx, t_idx = encode_txc_per_window(model, acts_anchor, device)
            # x_at aligned with t_idx (the last-position of each window)
            x_at = acts_anchor.numpy()[c_idx, t_idx, :]
        elif family == "mlxc":
            latents, x_hat, c_idx, t_idx = encode_mlc_per_token(
                model, acts_mlc, subject_cfg.mlc_layers, device)
            x_at = acts_anchor.numpy()[c_idx, t_idx, :]
        else:
            raise ValueError(family)

        lbl_1d = gather_labels(labels_nt, c_idx, t_idx)

        nmse_overall = nmse(x_at, x_hat)
        evar = explained_variance(x_at, x_hat)
        l0 = l0_mean(latents)
        per_arch_summary[name] = {
            "family": family,
            "nmse": nmse_overall,
            "explained_variance": evar,
            "l0_mean": l0,
            "n_samples": int(x_at.shape[0]),
        }
        per_arch_category[name] = per_category_nmse(x_at, x_hat, lbl_1d,
                                                    coarse_cfg["category_buckets"])

        # ---- UMAP (scope kind colored) ----
        try:
            emb, umap_idx = compute_umap(latents, coarse_cfg["umap"],
                                         n_sample=4000, seed=seed)
            scope_color = lbl_1d["scope_kind"][umap_idx]
            plot_umap(emb, scope_color,
                      title=f"{name}: UMAP of latents (color = scope_kind)",
                      path=plot_root / f"coarse_umap_{name}_scope.png",
                      cmap="tab10")
            depth_color = lbl_1d["bracket_depth"][umap_idx].clip(max=5)
            plot_umap(emb, depth_color,
                      title=f"{name}: UMAP of latents (color = bracket_depth)",
                      path=plot_root / f"coarse_umap_{name}_depth.png",
                      cmap="viridis")
            per_arch_umap[name] = {
                "emb_shape": list(emb.shape),
                "path_scope": str(plot_root / f"coarse_umap_{name}_scope.png"),
                "path_depth": str(plot_root / f"coarse_umap_{name}_depth.png"),
            }
        except Exception as e:
            print(f"[eval_coarse] UMAP failed for {name}: {e!r}")
            per_arch_umap[name] = {"error": repr(e)}

        # ---- Loss-recovered (optional, LM required) ----
        if args.with_lm:
            n_seq_lr = min(coarse_cfg["loss_recovered"]["n_sequences"], tokens_eval.shape[0])
            lr_result = compute_loss_recovered(
                subject_cfg=subject_cfg,
                tokens=tokens_eval[:n_seq_lr],
                acts_anchor=acts_anchor[:n_seq_lr],
                acts_mlc={L: acts_mlc[L][:n_seq_lr] for L in subject_cfg.mlc_layers},
                arch_name=name, arch_family=family, model=model,
                device=device, dtype_str=cfg.get("dtype", "bfloat16"),
            )
            per_arch_summary[name]["loss_recovered"] = lr_result
            print(f"[eval_coarse] {name} loss_recovered = {lr_result['loss_recovered']:.4f} "
                  f"(CE clean={lr_result['ce_clean']:.3f} "
                  f"zero={lr_result['ce_zero']:.3f} patched={lr_result['ce_patched']:.3f})")

        # Per-arch dump
        out_path = results_root / f"coarse_{name}.json"
        with out_path.open("w") as f:
            json.dump({
                "summary": per_arch_summary[name],
                "category": per_arch_category[name],
                "umap": per_arch_umap[name],
            }, f, indent=2)
        print(f"[eval_coarse] {name}: NMSE={nmse_overall:.4f} "
              f"evar={evar:.4f} L0={l0:.2f}")

    # --- combined ---
    results_root.mkdir(parents=True, exist_ok=True)
    with (results_root / "coarse_summary.json").open("w") as f:
        json.dump({
            "per_arch": per_arch_summary,
            "per_arch_category": per_arch_category,
            "manifest": manifest,
            "seed": seed,
        }, f, indent=2)
    if per_arch_category:
        plot_nmse_by_category(per_arch_category, plot_root / "coarse_nmse_by_category.png")
    print(f"[eval_coarse] done → {results_root / 'coarse_summary.json'}")


if __name__ == "__main__":
    main()
