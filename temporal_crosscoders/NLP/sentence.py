#!/usr/bin/env python3
"""
sentence.py — Visualize top-32 feature activations across a 32-token sequence
for StackedSAE vs TemporalCrosscoder (mid_res, k=100, T=5).

Displays a 32×32 heatmap (positions × features) for each model, plus stats.

Usage:
    python sentence.py                  # random sequence
    python sentence.py --chain 42       # specific sequence
"""

import argparse
import json
import math
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import D_SAE, LAYER_SPECS, SEED, CACHE_DIR, CHECKPOINT_DIR, VIZ_DIR, run_name

from fast_models import FastStackedSAE, FastTemporalCrosscoder

def load_models(
    layer: str, k: int, T: int,
) -> tuple[FastStackedSAE, FastTemporalCrosscoder]:
    d_act = LAYER_SPECS[layer]["d_act"]

    sae_path = os.path.join(CHECKPOINT_DIR, f"{run_name('stacked_sae', layer, k, T)}.pt")
    tx_path = os.path.join(CHECKPOINT_DIR, f"{run_name('txcdr', layer, k, T)}.pt")

    for p in [sae_path, tx_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Checkpoint not found: {p}")

    sae = FastStackedSAE(d_in=d_act, d_sae=D_SAE, T=T, k=k)
    sae.load_state_dict(torch.load(sae_path, map_location="cpu", weights_only=True))
    sae.eval()

    tx = FastTemporalCrosscoder(d_in=d_act, d_sae=D_SAE, T=T, k=k)
    tx.load_state_dict(torch.load(tx_path, map_location="cpu", weights_only=True))
    tx.eval()

    return sae, tx


def get_sequence(
    layer: str, chain_idx: int | None = None,
) -> tuple[torch.Tensor, int, int]:
    """Returns (sequence, chain_idx, seq_len)."""
    act_path = os.path.join(CACHE_DIR, f"{layer}.npy")
    data = np.load(act_path, mmap_mode="r")
    num_chains = data.shape[0]
    seq_len = data.shape[1]

    if chain_idx is None:
        chain_idx = np.random.randint(0, num_chains)
    seq = torch.from_numpy(data[chain_idx].copy()).float()  # (seq_len, d_act)
    print(f"Selected chain {chain_idx}, shape: {seq.shape}")
    return seq, chain_idx, seq_len


def get_activations_from_text(
    text: str, layer: str, max_seq_len: int = 256,
) -> tuple[torch.Tensor, list[str]]:
    """Run gemma-2-2b-it on `text`, return (n_tokens, d_act) activations + token labels.

    Uses the *actual* token count — no padding. Mirrors cache_activations.py:
    registers a forward hook on the right module (resid block or self-attn
    sublayer), runs one forward pass, returns activations + token labels.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from config import MODEL_NAME

    spec = LAYER_SPECS[layer]
    layer_idx = spec["layer"]
    component = spec["component"]
    d_act = spec["d_act"]

    print(f"Loading {MODEL_NAME} to extract activations for custom text...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map=device,
    )
    model.eval()

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_len,
        padding=False,
        add_special_tokens=True,
    )
    input_ids = enc["input_ids"].to(model.device)
    n_tokens = int(input_ids.shape[1])
    print(f"  Tokenized to {n_tokens} tokens")

    if component == "resid":
        target_module = model.model.layers[layer_idx]
    elif component == "attn":
        target_module = model.model.layers[layer_idx].self_attn
    else:
        raise ValueError(f"Unknown component: {component}")

    captured: dict[str, torch.Tensor] = {}

    def hook_fn(module, input, output):
        acts = output[0] if isinstance(output, tuple) else output
        if acts.dim() == 4:
            acts = acts.reshape(acts.shape[0], acts.shape[1], -1)
        captured["acts"] = acts.detach().float().cpu()

    h = target_module.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(input_ids)
    h.remove()

    acts = captured["acts"].squeeze(0)  # (n_tokens, d_act_full)
    if acts.shape[-1] != d_act:
        acts = acts[..., :d_act]

    tids = input_ids.squeeze(0).cpu().numpy()
    labels = []
    for tid in tids:
        tok = tokenizer.decode([int(tid)])
        tok = tok.replace("\n", "\\n").strip()
        labels.append(tok)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return acts, labels


def get_token_labels(chain_idx: int, seq_len: int) -> list[str]:
    """Try to decode token IDs for axis labels."""
    token_path = os.path.join(CACHE_DIR, "token_ids.npy")
    if not os.path.exists(token_path):
        return [f"t{i}" for i in range(seq_len)]
    try:
        from transformers import AutoTokenizer
        from config import MODEL_NAME
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        token_ids = np.load(token_path, mmap_mode="r")
        tids = token_ids[chain_idx, :seq_len]
        labels = []
        for tid in tids:
            tok = tokenizer.decode([int(tid)])
            tok = tok.replace("\n", "\\n").strip()
            labels.append(tok)
        return labels
    except Exception:
        return [f"t{i}" for i in range(seq_len)]


def extract_features(
    model: torch.nn.Module,
    model_type: str,
    seq: torch.Tensor,
    seq_len: int,
    T: int,
) -> torch.Tensor:
    """Run model on sliding windows across the sequence.

    Returns: (seq_len, D_SAE) feature activations at each position.
    For stacked_sae: per-position activations from each window, averaged.
    For txcdr: shared z vector assigned to all positions in window, averaged.
    """
    d_sae = model.d_sae
    all_acts = torch.zeros(seq_len, d_sae)
    counts = torch.zeros(seq_len, 1)

    with torch.no_grad():
        for start in range(seq_len - T + 1):
            window = seq[start : start + T].unsqueeze(0)  # (1, T, d)

            if model_type == "stacked_sae":
                _, _, u = model(window)  # (1, T, h)
                u = u.squeeze(0)  # (T, h)
                for t in range(T):
                    all_acts[start + t] += u[t]
                    counts[start + t] += 1
            else:
                _, _, z = model(window)  # (1, h)
                z = z.squeeze(0)  # (h,)
                for t in range(T):
                    all_acts[start + t] += z
                    counts[start + t] += 1

    all_acts /= counts.clamp(min=1)
    return all_acts


def load_interpretations(
    model_type: str, layer: str, k: int, T: int,
    feature_indices: np.ndarray,
    interp_root: str | None = None,
) -> dict[int, str]:
    """Load autointerp explanations for given features.

    Looks under `{interp_root or VIZ_DIR}/autointerp/{model}_{layer}_k{k}_T{T}/`.
    Returns: {feat_idx: explanation_text}, missing entries omitted.
    """
    root = interp_root or os.path.join(VIZ_DIR, "autointerp")
    label = f"{model_type}_{layer}_k{k}_T{T}"
    interp_dir = os.path.join(root, label)
    if not os.path.isdir(interp_dir):
        return {}

    out: dict[int, str] = {}
    for fi in feature_indices:
        path = os.path.join(interp_dir, f"feat_{int(fi):06d}.json")
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            expl = data.get("explanation", "").strip()
            if expl:
                out[int(fi)] = expl
        except Exception:
            pass
    return out


def add_interp_legend(
    ax,
    feature_indices: np.ndarray,
    interpretations: dict[int, str],
    n_feat: int,
    max_chars: int = 70,
) -> None:
    """Display feature interpretations as text aligned with heatmap rows."""
    ax.set_xlim(0, 1)
    # Match imshow's default origin='upper': row 0 at top
    ax.set_ylim(n_feat - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i, fi in enumerate(feature_indices):
        text = interpretations.get(int(fi))
        if text:
            if len(text) > max_chars:
                text = text.split()[3:]
                text = " ".join(text)
                text = text[: max_chars - 1] + "…"
            ax.text(0.0, i, text, va="center", ha="left", fontsize=5)
        else:
            ax.text(0.0, i, "(no interp)", va="center", ha="left",
                    fontsize=4, color="gray", style="italic")


def select_exclusive_features(acts: torch.Tensor, seq_len: int) -> np.ndarray:
    """For each token position, pick the feature that most exclusively
    activates there (greedy assignment, no feature reused).

    Exclusivity score for (position p, feature j):
        score[p, j] = acts[p, j]^2 / sum_p(acts[p, j])

    This rewards features whose total activation mass is concentrated at p.
    Squaring ensures the feature actually fires strongly at p (not just rarely
    elsewhere). Greedy assignment processes positions in descending order of
    their best score, so the strongest matches get priority.

    Returns: (seq_len,) array where selected[p] = feature idx owned by position p.
    """
    eps = 1e-8
    feat_total = acts.sum(dim=0) + eps                # (d_sae,)
    score = (acts ** 2) / feat_total.unsqueeze(0)      # (seq_len, d_sae)

    best_per_pos = score.max(dim=1).values             # (seq_len,)
    pos_order = best_per_pos.argsort(descending=True).tolist()

    used: set[int] = set()
    selected = [-1] * seq_len

    for p in pos_order:
        sorted_features = score[p].argsort(descending=True).tolist()
        for j in sorted_features:
            if j not in used:
                selected[p] = j
                used.add(j)
                break

    return np.array(selected)


def binary_entropy(acts: np.ndarray) -> np.ndarray:
    """Per-position entropy from binary (active/inactive) feature pattern."""
    binary = (acts > 0).astype(float)
    p = binary.mean(axis=1)  # fraction of features active per position
    eps = 1e-8
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def compute_stats(
    acts: np.ndarray, full_acts: torch.Tensor, label: str,
) -> dict[str, float]:
    """Compute summary statistics for a model's activations."""
    ent = binary_entropy(acts)
    sparsity = (acts > 0).mean(axis=0)  # per-feature: fraction of positions active
    active_per_pos = (full_acts > 0).float().sum(dim=1)  # over full D_SAE

    return {
        "label": label,
        "mean_pos_entropy": float(ent.mean()),
        "std_pos_entropy": float(ent.std()),
        "mean_feat_sparsity": float(sparsity.mean()),
        "active_feats_per_pos": float(active_per_pos.mean()),
        "max_activation": float(acts.max()),
        "mean_activation": float(acts[acts > 0].mean()) if (acts > 0).any() else 0.0,
        "frac_nonzero": float((acts > 0).mean()),
    }


def format_stats(s: dict) -> str:
    return (
        f"  Local position entropy:    {s['mean_pos_entropy']:.4f} +/- {s['std_pos_entropy']:.4f}\n"
        f"  Local feature sparsity:    {s['mean_feat_sparsity']:.4f}\n"
        f"  Local active feats/pos:    {s['active_feats_per_pos']:.1f} (full D_SAE)\n"
        f"  Local frac nonzero:        {s['frac_nonzero']:.4f}\n"
        f"  Local max activation:      {s['max_activation']:.2f}\n"
        f"  Local mean activation (>0):{s['mean_activation']:.2f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize top-N feature activations across a sequence "
                    "for StackedSAE vs TemporalCrosscoder.",
    )
    parser.add_argument("--chain", type=str, default=None,
                        help="Chain index (int) from cached activations, "
                             "OR a custom text string to analyze (will run "
                             "gemma-2-2b-it forward pass to get activations). "
                             "Default: random chain.")
    parser.add_argument("--layer", type=str, default="mid_res",
                        choices=list(LAYER_SPECS.keys()),
                        help="Which layer to analyze (default: mid_res)")
    parser.add_argument("--k", type=int, default=100,
                        help="Top-k value of the trained model (default: 100)")
    parser.add_argument("--T", type=int, default=5,
                        help="Window length T of the trained model (default: 5)")
    parser.add_argument("--n-features", type=int, default=32,
                        help="Number of top features to display per model (default: 32). "
                             "Ignored when --select=exclusive (uses seq_len).")
    parser.add_argument("--select", type=str, default="exclusive",
                        choices=["magnitude", "exclusive"],
                        help="Feature selection mode: 'magnitude' picks top-N by total "
                             "activation; 'exclusive' picks one feature per token position "
                             "that most exclusively activates there (default: exclusive).")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"Random seed (default: {SEED})")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: VIZ_DIR)")
    parser.add_argument("--cmap-sae", type=str, default="inferno",
                        help="Colormap for SAE heatmap (default: inferno)")
    parser.add_argument("--cmap-tx", type=str, default="inferno",
                        help="Colormap for TXCDR heatmap (default: inferno)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = args.output_dir or VIZ_DIR
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading models (layer={args.layer}, k={args.k}, T={args.T})...")
    sae, tx = load_models(args.layer, args.k, args.T)

    # Decide whether --chain is an int (cache index) or a custom text string.
    chain_arg = args.chain
    parsed_idx: int | None = None
    if chain_arg is not None:
        try:
            parsed_idx = int(chain_arg)
        except ValueError:
            parsed_idx = None  # treat as custom text

    if chain_arg is not None and parsed_idx is None:
        # Custom text path: extract activations via gemma forward pass.
        # n is the actual token count (no padding).
        print(f'Custom text input: "{chain_arg[:60]}{"..." if len(chain_arg) > 60 else ""}"')
        seq, token_labels = get_activations_from_text(chain_arg, args.layer)
        seq_len = seq.shape[0]
        if seq_len < args.T:
            raise ValueError(
                f"Sentence has {seq_len} tokens but T={args.T}; need at least T."
            )
        sanitized = "".join(c if c.isalnum() else "_" for c in chain_arg[:30]).strip("_")
        chain_label = f"text_{sanitized}" if sanitized else "text"
    else:
        print("Loading sequence...")
        seq, chain_idx, seq_len = get_sequence(args.layer, parsed_idx)
        token_labels = get_token_labels(chain_idx, seq_len)
        chain_label = f"chain{chain_idx}"

    print("Extracting features...")
    sae_acts = extract_features(sae, "stacked_sae", seq, seq_len, args.T)
    tx_acts = extract_features(tx, "txcdr", seq, seq_len, args.T)

    # Select features SEPARATELY for each model
    if args.select == "exclusive":
        sae_top_idx = select_exclusive_features(sae_acts, seq_len)
        tx_top_idx = select_exclusive_features(tx_acts, seq_len)
        n_feat = seq_len
    else:
        n_feat = args.n_features
        sae_top_idx = sae_acts.abs().sum(dim=0).topk(n_feat).indices.numpy()
        tx_top_idx = tx_acts.abs().sum(dim=0).topk(n_feat).indices.numpy()

    sae_hm = sae_acts[:, sae_top_idx].numpy()  # (seq_len, n_feat)
    tx_hm = tx_acts[:, tx_top_idx].numpy()      # (seq_len, n_feat)

    # Stats
    sae_stats = compute_stats(sae_hm, sae_acts, "StackedSAE")
    tx_stats = compute_stats(tx_hm, tx_acts, "TXCDR")

    select_desc = {
        "magnitude": f"top {n_feat} features per-model by activation magnitude",
        "exclusive": f"{n_feat} features per-model (one most-exclusive per token position)",
    }[args.select]
    stats_text = (
        f"Source: {chain_label} | Layer: {args.layer} | k={args.k} T={args.T}\n"
        f"Selection: {select_desc}\n"
        f"All metrics computed LOCALLY on this {seq_len}-token sequence.\n"
        f"\n"
        f"StackedSAE:\n{format_stats(sae_stats)}\n"
        f"\n"
        f"TXCDR:\n{format_stats(tx_stats)}\n"
    )
    print(stats_text)

    # ─── Load feature interpretations (if autointerp results exist) ─────────
    sae_interps = load_interpretations(
        "stacked_sae", args.layer, args.k, args.T, sae_top_idx,
    )
    tx_interps = load_interpretations(
        "txcdr", args.layer, args.k, args.T, tx_top_idx,
    )
    print(f"  Loaded interps: SAE={len(sae_interps)}/{n_feat}, TXCDR={len(tx_interps)}/{n_feat}")

    # ─── Plot: 2 heatmaps stacked top-to-bottom + interp legends + stats ───
    # Layout: 3 rows (SAE, TXCDR, stats) × 2 cols (heatmap, interp legend)
    fig = plt.figure(figsize=(22, 18))
    gs = gridspec.GridSpec(
        3, 2,
        width_ratios=[3, 2],
        height_ratios=[6, 6, 1],
        hspace=0.35, wspace=0.05,
    )

    # Per-model log normalization for readable color contrast.
    from matplotlib.colors import Normalize

    def make_norm(hm):
        mx = max(hm.max(), 1e-6)
        return Normalize(vmin=0, vmax=np.log1p(mx))

    sae_log = np.log1p(sae_hm)
    tx_log = np.log1p(tx_hm)

    # ─── SAE heatmap (row 0) ───────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(sae_log.T, aspect="auto", cmap=args.cmap_sae,
                      norm=make_norm(sae_hm), interpolation="nearest")
    ax0.set_title(f"StackedSAE — {args.layer} k={args.k} T={args.T}", fontsize=11)
    ax0.set_xlabel("Token position")
    ax0.set_ylabel("Feature")
    ax0.set_xticks(range(seq_len))
    ax0.set_xticklabels(token_labels, fontsize=5, rotation=90)
    ax0.set_yticks(range(n_feat))
    ax0.set_yticklabels([f"f{i}" for i in sae_top_idx], fontsize=5)
    cb0 = fig.colorbar(im0, ax=ax0, shrink=0.6, label="Activation",
                        location="left", pad=0.08)
    cb0_ticks = cb0.get_ticks()
    cb0.ax.set_yticks(cb0_ticks)
    cb0.ax.set_yticklabels([f"{np.expm1(t):.0f}" for t in cb0_ticks])

    # SAE interp legend (row 0, col 1)
    ax0_leg = fig.add_subplot(gs[0, 1])
    add_interp_legend(ax0_leg, sae_top_idx, sae_interps, n_feat)
    ax0_leg.set_title("SAE feature interpretations", fontsize=9, loc="left")

    # ─── TXCDR heatmap (row 1) ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    im1 = ax1.imshow(tx_log.T, aspect="auto", cmap=args.cmap_tx,
                      norm=make_norm(tx_hm), interpolation="nearest")
    ax1.set_title(f"TXCDR — {args.layer} k={args.k} T={args.T}", fontsize=11)
    ax1.set_xlabel("Token position")
    ax1.set_ylabel("Feature")
    ax1.set_xticks(range(seq_len))
    ax1.set_xticklabels(token_labels, fontsize=5, rotation=90)
    ax1.set_yticks(range(n_feat))
    ax1.set_yticklabels([f"f{i}" for i in tx_top_idx], fontsize=5)
    cb1 = fig.colorbar(im1, ax=ax1, shrink=0.6, label="Activation",
                        location="left", pad=0.08)
    cb1_ticks = cb1.get_ticks()
    cb1.ax.set_yticks(cb1_ticks)
    cb1.ax.set_yticklabels([f"{np.expm1(t):.0f}" for t in cb1_ticks])

    # TXCDR interp legend (row 1, col 1)
    ax1_leg = fig.add_subplot(gs[1, 1])
    add_interp_legend(ax1_leg, tx_top_idx, tx_interps, n_feat)
    ax1_leg.set_title("TXCDR feature interpretations", fontsize=9, loc="left")

    # ─── Stats panel (row 2, spanning both columns) ────────────────────────
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis("off")

    col_labels = [f"Local metric ({seq_len} tokens)", "StackedSAE", "TXCDR"]
    row_data = [
        ["Local position entropy", f"{sae_stats['mean_pos_entropy']:.4f}", f"{tx_stats['mean_pos_entropy']:.4f}"],
        [f"Local feature sparsity (top {n_feat})", f"{sae_stats['mean_feat_sparsity']:.4f}", f"{tx_stats['mean_feat_sparsity']:.4f}"],
        ["Local active feats/pos (full)", f"{sae_stats['active_feats_per_pos']:.1f}", f"{tx_stats['active_feats_per_pos']:.1f}"],
        [f"Local frac nonzero ({seq_len}x{n_feat})", f"{sae_stats['frac_nonzero']:.4f}", f"{tx_stats['frac_nonzero']:.4f}"],
        ["Local max activation", f"{sae_stats['max_activation']:.2f}", f"{tx_stats['max_activation']:.2f}"],
        ["Local mean activation (>0)", f"{sae_stats['mean_activation']:.2f}", f"{tx_stats['mean_activation']:.2f}"],
    ]

    table = ax_stats.table(
        cellText=row_data, colLabels=col_labels,
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(0.8, 1.4)

    fig.suptitle(
        f"Top-{n_feat} Feature Activations — {chain_label} ({seq_len} tokens) — local metrics",
        fontsize=13, y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    suffix = f"{args.layer}_k{args.k}_T{args.T}_{chain_label}_{args.select}"
    path = os.path.join(output_dir, f"sentence_{suffix}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    stats_path = os.path.join(output_dir, f"sentence_{suffix}_stats.txt")
    with open(stats_path, "w") as f:
        f.write(stats_text)
    print(f"Saved: {stats_path}")


if __name__ == "__main__":
    main()
