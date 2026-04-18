"""Phase 3: Joint UMAP visualization of decoder directions across architectures.

Projects decoder directions from Stacked, TXCDR, and TFA (split into
novel-dominated vs pred-dominated) into a shared 2D UMAP embedding, so
the disjoint-dictionary finding becomes visually obvious.

Annotates a handful of representative features per category with their
Phase 2 autointerp explanations, to anchor regions of the map to actual
semantic content.

Outputs:
    results/analysis/joint_umap/
        umap_all.png (overview, color by architecture/subtype)
        umap_all_annotated.png (with labels on key features)
        umap_per_arch.png (4 panels — one per category)
        umap_coords.json (x, y, category per feature)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

try:
    import umap
except ImportError:
    sys.exit("umap-learn not installed")

sys.path.insert(0, "/home/elysium/temp_xc")

CKPT_DIR = "/home/elysium/temp_xc/results/nlp_sweep/gemma/ckpts"
ALIGN_DIR = "/home/elysium/temp_xc/results/analysis/decoder_alignment"
TFA_DIR = "/home/elysium/temp_xc/results/analysis/tfa_pred_vs_novel"
AUTOINTERP_DIR = "/home/elysium/temp_xc/results/analysis/autointerp"
OUT_DIR = "/home/elysium/temp_xc/results/analysis/joint_umap"
D_IN = 2304
D_SAE = 18432
T_WIN = 5
K = 100
LAYER = "resid_L25"
ALIVE_THRESHOLD = 0.0001


def unit_norm_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return x / norms


def load_decoders() -> dict[str, np.ndarray]:
    """Return unit-normed (d_sae, d_in) decoder row matrices per architecture."""
    out = {}
    # Stacked: avg over T positions
    state = torch.load(
        f"{CKPT_DIR}/stacked_sae__gemma-2-2b-it__fineweb__{LAYER}__k{K}__seed42.pt",
        map_location="cpu", weights_only=True,
    )
    W = torch.stack([state[f"saes.{t}.W_dec"] for t in range(T_WIN)]).mean(dim=0).T
    out["stacked"] = unit_norm_rows(W.numpy())

    state = torch.load(
        f"{CKPT_DIR}/crosscoder__gemma-2-2b-it__fineweb__{LAYER}__k{K}__seed42.pt",
        map_location="cpu", weights_only=True,
    )
    W = state["W_dec"].mean(dim=1)
    out["txcdr"] = unit_norm_rows(W.numpy())

    state = torch.load(
        f"{CKPT_DIR}/tfa_pos__gemma-2-2b-it__fineweb__{LAYER}__k{K}__seed42.pt",
        map_location="cpu", weights_only=True,
    )
    out["tfa"] = unit_norm_rows(state["D"].numpy())
    return out


def load_alive_masks():
    freqs = torch.load(f"{ALIGN_DIR}/firing_frequencies.pt")
    return {
        "stacked":       (freqs["freq_stacked"]       > ALIVE_THRESHOLD).numpy(),
        "txcdr":         (freqs["freq_txcdr"]         > ALIVE_THRESHOLD).numpy(),
        "tfa_novel":     (freqs["freq_tfa_novel"]     > ALIVE_THRESHOLD).numpy(),
        "tfa_pred":      (freqs["freq_tfa_pred"]      > ALIVE_THRESHOLD).numpy(),
    }


def classify_tfa_pred_ratio() -> np.ndarray:
    """Return pred_ratio (mass-weighted) for each of 18432 TFA features.

    Recomputes pred_mass / (pred_mass + novel_mass) on a fresh eval pass.
    Phase 1c showed this is cleanly bimodal at 0 and 1.
    """
    import math
    from src.bench.architectures._tfa_module import TemporalSAE

    ACT_PATH = "/home/elysium/temp_xc/data/cached_activations/gemma-2-2b-it/fineweb/resid_L25.npy"
    state = torch.load(
        f"{CKPT_DIR}/tfa_pos__gemma-2-2b-it__fineweb__{LAYER}__k{K}__seed42.pt",
        map_location="cpu", weights_only=True,
    )
    model = TemporalSAE(
        dimin=D_IN, width=D_SAE, n_heads=4, sae_diff_type="topk", kval_topk=K,
        tied_weights=True, n_attn_layers=1, bottleneck_factor=8,
        use_pos_encoding=True, max_seq_len=512,
    ).cuda()
    model.load_state_dict(state)
    model.eval()

    arr = np.load(ACT_PATH, mmap_mode="r")
    eval_x = torch.from_numpy(np.array(arr[-500:])).float()
    scale = math.sqrt(D_IN) / eval_x[:16].norm(dim=-1).mean().item()

    pred_mass = torch.zeros(D_SAE)
    novel_mass = torch.zeros(D_SAE)
    with torch.no_grad():
        for s in range(0, eval_x.shape[0], 16):
            x = eval_x[s : s + 16].cuda() * scale
            _, inter = model(x)
            pred_mass += inter["pred_codes"].abs().reshape(-1, D_SAE).sum(dim=0).cpu()
            novel_mass += inter["novel_codes"].reshape(-1, D_SAE).sum(dim=0).cpu()
    total = pred_mass + novel_mass
    ratio = torch.where(total > 1e-6, pred_mass / total, torch.tensor(-1.0)).numpy()
    return ratio


def load_autointerp_labels() -> dict:
    """Return {category: {feat_idx: explanation}}."""
    out = {}
    for cat in ["tfa_pred_only", "tfa_novel_only", "txcdr_unique", "stacked_unique"]:
        labels = {}
        summary_path = Path(AUTOINTERP_DIR) / cat / "_summary.json"
        if not summary_path.exists():
            continue
        data = json.load(open(summary_path))
        for d in data:
            if d.get("confidence") in ("HIGH", "MEDIUM") and d.get("explanation"):
                labels[d["feat_idx"]] = d["explanation"]
        out[cat] = labels
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading decoders...")
    D = load_decoders()
    alive = load_alive_masks()
    tfa_ratio = classify_tfa_pred_ratio()

    # Build merged matrix: alive features from each group
    # TFA is split into two subcategories by pred_ratio
    tfa_alive = alive["tfa_novel"] | alive["tfa_pred"]

    categories = []
    feat_idxs = []
    rows = []

    # Stacked alive
    idxs = np.where(alive["stacked"])[0]
    for i in idxs:
        rows.append(D["stacked"][i])
        categories.append("stacked")
        feat_idxs.append(int(i))
    print(f"  stacked: {len(idxs)} alive")

    # TXCDR alive
    idxs = np.where(alive["txcdr"])[0]
    for i in idxs:
        rows.append(D["txcdr"][i])
        categories.append("txcdr")
        feat_idxs.append(int(i))
    print(f"  txcdr: {len(idxs)} alive")

    # TFA split by pred_ratio
    idxs = np.where(tfa_alive)[0]
    for i in idxs:
        rows.append(D["tfa"][i])
        r = tfa_ratio[i]
        if r > 0.66:
            categories.append("tfa_pred")
        elif r < 0.33:
            categories.append("tfa_novel")
        else:
            categories.append("tfa_mixed")
        feat_idxs.append(int(i))
    n_tfa_pred = sum(1 for c in categories if c == "tfa_pred")
    n_tfa_novel = sum(1 for c in categories if c == "tfa_novel")
    n_tfa_mixed = sum(1 for c in categories if c == "tfa_mixed")
    print(f"  tfa_pred: {n_tfa_pred}, tfa_novel: {n_tfa_novel}, tfa_mixed: {n_tfa_mixed}")

    X = np.stack(rows).astype(np.float32)
    print(f"Matrix: {X.shape}")

    print("\nPCA to 50D...")
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X)
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    print("\nUMAP to 2D (cosine metric, this takes ~2-5 min)...")
    reducer = umap.UMAP(
        n_components=2, metric="cosine",
        n_neighbors=30, min_dist=0.1,
        random_state=42, verbose=True,
    )
    emb = reducer.fit_transform(X_pca)
    print(f"  Embedding: {emb.shape}")

    # Save coordinates
    coords = {
        "x": emb[:, 0].tolist(),
        "y": emb[:, 1].tolist(),
        "category": categories,
        "feat_idx": feat_idxs,
    }
    with open(f"{OUT_DIR}/umap_coords.json", "w") as f:
        json.dump(coords, f)

    colors = {
        "stacked":   "#1f77b4",   # blue
        "txcdr":     "#ff7f0e",   # orange
        "tfa_pred":  "#d62728",   # red
        "tfa_novel": "#2ca02c",   # green
        "tfa_mixed": "#9467bd",   # purple
    }
    labels = {
        "stacked":   f"Stacked SAE ({alive['stacked'].sum()})",
        "txcdr":     f"TXCDR ({alive['txcdr'].sum()})",
        "tfa_pred":  f"TFA pred-only ({n_tfa_pred})",
        "tfa_novel": f"TFA novel-only ({n_tfa_novel})",
        "tfa_mixed": f"TFA mixed ({n_tfa_mixed})" if n_tfa_mixed else None,
    }

    # ── Main figure: all points colored by category ──────────────────────
    categories_arr = np.array(categories)
    fig, ax = plt.subplots(1, 1, figsize=(11, 9))
    for cat in ["stacked", "txcdr", "tfa_pred", "tfa_novel", "tfa_mixed"]:
        mask = categories_arr == cat
        if mask.sum() == 0:
            continue
        ax.scatter(emb[mask, 0], emb[mask, 1], c=colors[cat], s=2, alpha=0.4,
                   label=labels[cat], edgecolors="none")
    ax.legend(fontsize=11, markerscale=4)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title(f"Joint UMAP of decoder directions — Gemma-2-2B-IT {LAYER} k=100\n"
                 f"All architectures share the same d_sae={D_SAE} but learn nearly disjoint directions")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/umap_all.png", dpi=140)
    plt.close()
    print(f"  -> {OUT_DIR}/umap_all.png")

    # ── 4-panel figure: one per category ──────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    panels = [
        ("stacked", "Stacked SAE features (concrete lexical)"),
        ("txcdr", "TXCDR features (grammatical / multilingual)"),
        ("tfa_pred", "TFA pred-only features (structural/positional)"),
        ("tfa_novel", "TFA novel-only features (sequence boundaries)"),
    ]
    for (cat, title), ax in zip(panels, axes.flatten()):
        ax.scatter(emb[:, 0], emb[:, 1], c="lightgray", s=1, alpha=0.2)
        mask = categories_arr == cat
        ax.scatter(emb[mask, 0], emb[mask, 1], c=colors[cat], s=3, alpha=0.7)
        ax.set_title(f"{title} (n={mask.sum()})")
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/umap_per_arch.png", dpi=140)
    plt.close()
    print(f"  -> {OUT_DIR}/umap_per_arch.png")

    # ── Annotated figure: label a handful of representative features ──────
    autointerp = load_autointerp_labels()
    print("\nAnnotating representative features...")

    fig, ax = plt.subplots(1, 1, figsize=(14, 11))
    for cat in ["stacked", "txcdr", "tfa_pred", "tfa_novel", "tfa_mixed"]:
        mask = categories_arr == cat
        if mask.sum() == 0:
            continue
        ax.scatter(emb[mask, 0], emb[mask, 1], c=colors[cat], s=2, alpha=0.35,
                   label=labels[cat], edgecolors="none")

    # Pick 3 HIGH-confidence features per category to label (shortest explanations first)
    def short_expl(s: str, max_len: int = 55) -> str:
        s = s.strip()
        if len(s) <= max_len:
            return s
        # Cut at sentence boundary
        if "." in s[:max_len]:
            return s[:s[:max_len].rfind(".") + 1]
        return s[:max_len] + "..."

    annotations = []  # (x, y, label, feat_idx, cat)
    cat_to_autointerp_keys = {
        "stacked":   "stacked_unique",
        "txcdr":     "txcdr_unique",
        "tfa_pred":  "tfa_pred_only",
        "tfa_novel": "tfa_novel_only",
    }
    for cat, label_key in cat_to_autointerp_keys.items():
        if label_key not in autointerp:
            continue
        labels_dict = autointerp[label_key]
        # Find features that are in this category AND have a label
        cat_mask = categories_arr == cat
        # Only consider actual features from this category (not absolute feat_idx since
        # TFA novel and TFA pred share feat_idx space — filter by category membership)
        picks = []
        for pos_i, fi in enumerate(feat_idxs):
            if not cat_mask[pos_i]:
                continue
            if fi not in labels_dict:
                continue
            picks.append((pos_i, fi, labels_dict[fi]))
        # Pick 3 features spread out in the embedding (by simple grid-distance)
        if len(picks) > 3:
            # Take 3 with most extreme UMAP positions
            picks_np = [(p, emb[p[0]]) for p in picks]
            picks_np.sort(key=lambda e: -(e[1][0]**2 + e[1][1]**2))
            picks = [p[0] for p in picks_np[:3]]
        for pos_i, fi, expl in picks:
            annotations.append((emb[pos_i, 0], emb[pos_i, 1],
                                f"f{fi}: {short_expl(expl)}", fi, cat))

    # Plot annotations with offset text
    for x, y, label, fi, cat in annotations:
        ax.scatter([x], [y], c=colors[cat], s=40, edgecolors="black", linewidths=0.8)
        ax.annotate(label, (x, y), xytext=(8, 8), textcoords="offset points",
                    fontsize=7, color="black",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="gray"))

    ax.legend(fontsize=11, markerscale=4, loc="best")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title(f"Joint UMAP with representative feature labels "
                 f"(Gemma-2-2B-IT {LAYER} k=100)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/umap_all_annotated.png", dpi=140)
    plt.close()
    print(f"  -> {OUT_DIR}/umap_all_annotated.png")

    # Save annotations
    with open(f"{OUT_DIR}/annotations.json", "w") as f:
        json.dump([
            {"x": a[0], "y": a[1], "label": a[2], "feat_idx": a[3], "category": a[4]}
            for a in annotations
        ], f, indent=2, default=float)

    print("\nDONE")


if __name__ == "__main__":
    main()
