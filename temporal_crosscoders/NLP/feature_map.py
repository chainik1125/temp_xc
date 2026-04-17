#!/usr/bin/env python3
"""
feature_map.py — Unsupervised feature clustering + interactive 2D map.

Pipeline:
  1. Load a trained StackedSAE / TXCDR checkpoint
  2. Extract decoder directions (one per feature, d_in-dim)
  3. Filter to features that have autointerp explanations (non-dead, labeled)
  4. PCA 50D -> UMAP 2D -> KMeans cluster
  5. For each cluster, call gemma-2-2b-it with the feature explanations it
     contains to produce a 1-sentence cluster summary
  6. Write an interactive plotly HTML scatter (hover = feature + explanation,
     color = cluster, legend = cluster summaries)

Usage:
    python feature_map.py                                           # defaults
    python feature_map.py --model stacked_sae --layer mid_res --k 100 --T 5
    python feature_map.py --n-clusters 30 --include-unlabeled
    python feature_map.py --skip-llm-labels                         # no gemma, fast

Outputs:
    viz_outputs/feature_map_{label}.html
    viz_outputs/feature_map_{label}_clusters.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from src.bench.model_registry import get_model_config, list_models
from temporal_crosscoders.NLP.config import (
    SEED, VIZ_DIR, d_sae_for, build_layer_specs,
)
from temporal_crosscoders.NLP.fast_models import (
    FastStackedSAE, FastTemporalCrosscoder,
)
from src.bench.architectures.topk_sae import TopKSAE


# ══════════════════════════════════════════════════════════════════════════════
# Feature representation + clustering
# ══════════════════════════════════════════════════════════════════════════════

def load_model(
    checkpoint: str,
    model_type: str,
    subject_model: str,
    k: int,
    T: int,
    expansion_factor: int = 8,
) -> torch.nn.Module:
    """Load a trained checkpoint. Subject model name routes through the
    registry to resolve d_in; expansion_factor sets d_sae."""
    cfg = get_model_config(subject_model)
    d_in = cfg.d_model
    d_sae = d_in * expansion_factor

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if model_type == "topk_sae":
        # Single-token SAE — no T dimension. Uses the spec-based TopKSAE
        # class directly (same one that src.bench.sweep saves from).
        model = TopKSAE(d_in=d_in, d_sae=d_sae, k=k)
    elif model_type == "stacked_sae":
        model = FastStackedSAE(d_in=d_in, d_sae=d_sae, T=T, k=k)
    else:
        model = FastTemporalCrosscoder(d_in=d_in, d_sae=d_sae, T=T, k=k)

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def get_feature_directions(model: torch.nn.Module, model_type: str) -> np.ndarray:
    """Return (d_sae, d_in) — each row is one feature's decoder direction."""
    with torch.no_grad():
        if model_type == "topk_sae":
            # W_dec: (d_in, d_sae) — no T dimension, just transpose
            dirs = model.W_dec.T  # (d_sae, d_in)
        elif model_type == "stacked_sae":
            # W_dec: (T, d_in, d_sae) -> mean over T -> (d_in, d_sae) -> transpose
            dirs = model.W_dec.mean(dim=0).T  # (d_sae, d_in)
        else:
            # W_dec: (d_sae, T, d_in) -> mean over T -> (d_sae, d_in)
            dirs = model.W_dec.mean(dim=1)  # (d_sae, d_in)
    return dirs.detach().cpu().numpy()


def load_autointerp(label: str, output_dir: str | None = None) -> dict[int, dict]:
    """Load existing autointerp explanations from <output_dir>/autointerp/<label>/.

    label is any string that uniquely identifies the run (e.g.
    'step1-unshuffled'). output_dir defaults to VIZ_DIR for backwards
    compat but should be the same --output-dir you passed to autointerp
    so reads and writes live under the same reports/ tree.
    """
    root = output_dir or VIZ_DIR
    interp_dir = Path(root) / "autointerp" / label
    if not interp_dir.is_dir():
        return {}

    out: dict[int, dict] = {}
    for p in interp_dir.glob("feat_*.json"):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        idx = int(data.get("feat_idx", -1))
        if idx < 0:
            continue
        out[idx] = {
            "explanation": (data.get("explanation") or "").strip(),
            "top_texts": data.get("top_texts") or [],
            "top_activations": data.get("top_activations") or [],
        }
    return out


def cluster_features(
    directions: np.ndarray,
    n_pca: int = 50,
    n_clusters: int = 20,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """PCA -> UMAP -> KMeans. Returns (X_2d, cluster_labels)."""
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    import umap

    # L2-normalize so direction magnitude doesn't dominate
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = directions / norms

    n_pca = min(n_pca, X.shape[0] - 1, X.shape[1])
    print(f"  PCA: {X.shape} -> ({X.shape[0]}, {n_pca})")
    pca = PCA(n_components=n_pca, random_state=seed)
    X_pca = pca.fit_transform(X)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    print(f"  UMAP: fitting {X_pca.shape} -> 2D (n_neighbors={n_neighbors})")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
        metric="cosine",
    )
    X_2d = reducer.fit_transform(X_pca)

    print(f"  KMeans: n_clusters={n_clusters}")
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(X_pca)

    return X_2d, labels


# ══════════════════════════════════════════════════════════════════════════════
# LLM cluster labels
# ══════════════════════════════════════════════════════════════════════════════

CLUSTER_SYSTEM = """\
You are analyzing groups of learned sparse-autoencoder features in a language \
model. Each feature already has a one-sentence explanation from a previous \
interpretability pass.

Given a group of feature explanations that were clustered together by \
unsupervised dimensionality reduction, summarize the COMMON theme in one \
short sentence. If the features look unrelated, say "Mixed / unclear".

Your response MUST be exactly one line starting with [CLUSTER]:"""

CLUSTER_USER_TEMPLATE = """\
These {n} feature explanations were grouped together:

{examples}

What concept does this cluster represent?"""


def label_clusters_with_gemma(
    feature_indices: np.ndarray,
    cluster_labels: np.ndarray,
    interps: dict[int, dict],
    explain_model: str,
    device: str,
    max_per_cluster: int = 12,
) -> dict[int, str]:
    """For each cluster, call gemma-2-2b-it with the member explanations and
    parse out a one-line cluster summary.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    by_cluster: dict[int, list[int]] = {}
    for i, cid in enumerate(cluster_labels):
        by_cluster.setdefault(int(cid), []).append(int(feature_indices[i]))

    print(f"  Loading {explain_model} for cluster labeling on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(explain_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(
        explain_model, torch_dtype=torch.bfloat16, device_map=device,
    )
    llm.eval()

    # Gemma-2 defaults to a static-KV-cache generation path that triggers
    # torch.compile → inductor → triton. Environments without triton
    # (e.g. some Compute Canada venvs) crash here. Force the dynamic
    # eager path via cache_implementation=None, and suppress any
    # remaining dynamo compile errors as a belt-and-suspenders fallback.
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    if hasattr(llm, "generation_config") and llm.generation_config is not None:
        llm.generation_config.cache_implementation = None

    labels: dict[int, str] = {}
    for cid in tqdm(sorted(by_cluster), desc="Labeling clusters"):
        feats = by_cluster[cid]
        expls = [interps[f]["explanation"] for f in feats if f in interps and interps[f]["explanation"]]
        if not expls:
            labels[cid] = f"Cluster {cid} (no explanations)"
            continue

        sample = expls[:max_per_cluster]
        ex_str = "\n".join(f"- {e}" for e in sample)
        user = CLUSTER_USER_TEMPLATE.format(n=len(sample), examples=ex_str)

        messages = [{"role": "user", "content": f"{CLUSTER_SYSTEM}\n\n{user}"}]
        chat_out = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True,
        )
        # Newer transformers may return a BatchEncoding instead of a tensor
        if isinstance(chat_out, torch.Tensor):
            input_ids = chat_out.to(llm.device)
            gen_kwargs = {"input_ids": input_ids}
        else:
            gen_kwargs = {k: v.to(llm.device) for k, v in chat_out.items()}
            input_ids = gen_kwargs["input_ids"]
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            out_ids = llm.generate(
                **gen_kwargs, max_new_tokens=80,
                do_sample=False, temperature=None, top_p=None,
            )
        new_tokens = out_ids[0, prompt_len:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Parse [CLUSTER]: line
        summary = ""
        for line in raw.splitlines():
            if line.strip().startswith("[CLUSTER]:"):
                summary = line.split(":", 1)[1].strip()
                break
        if not summary:
            summary = raw.splitlines()[0].strip() if raw else f"Cluster {cid}"
        labels[cid] = summary[:120]

    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return labels


# ══════════════════════════════════════════════════════════════════════════════
# Plotly output
# ══════════════════════════════════════════════════════════════════════════════

def write_interactive_html(
    X_2d: np.ndarray,
    feature_indices: np.ndarray,
    cluster_labels: np.ndarray,
    interps: dict[int, dict],
    cluster_summaries: dict[int, str],
    output_path: str,
    title: str,
) -> None:
    import pandas as pd
    import plotly.express as px

    rows = []
    for i, fi in enumerate(feature_indices):
        fi = int(fi)
        cid = int(cluster_labels[i])
        expl = interps.get(fi, {}).get("explanation", "")
        top_text = ""
        if fi in interps and interps[fi]["top_texts"]:
            top_text = interps[fi]["top_texts"][0][:120]
        rows.append({
            "x": float(X_2d[i, 0]),
            "y": float(X_2d[i, 1]),
            "feat_idx": fi,
            "cluster": cid,
            "cluster_label": f"{cid}: {cluster_summaries.get(cid, '')[:60]}",
            "explanation": expl[:140] if expl else "(unlabeled)",
            "top_text": top_text,
        })
    df = pd.DataFrame(rows)

    fig = px.scatter(
        df, x="x", y="y",
        color="cluster_label",
        hover_data={
            "x": False, "y": False,
            "feat_idx": True,
            "cluster": True,
            "explanation": True,
            "top_text": True,
            "cluster_label": False,
        },
        title=title,
        width=1400, height=900,
    )
    fig.update_traces(marker=dict(size=5, opacity=0.75))
    fig.update_layout(
        legend=dict(font=dict(size=9), itemsizing="constant"),
        hovermode="closest",
    )
    fig.write_html(output_path)
    print(f"  Saved: {output_path}")


def write_static_png(
    X_2d: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_summaries: dict[int, str],
    output_path: str,
    title: str,
) -> None:
    """Static matplotlib scatter with cluster summaries as annotations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    uniq = np.unique(cluster_labels)
    n_clusters = len(uniq)
    cmap = plt.get_cmap("tab20" if n_clusters <= 20 else "hsv")
    colors = [cmap(i / max(n_clusters - 1, 1)) for i in range(n_clusters)]
    cluster_color = {int(cid): colors[i] for i, cid in enumerate(uniq)}

    fig = plt.figure(figsize=(18, 12))
    ax_main = fig.add_axes([0.04, 0.06, 0.58, 0.88])
    ax_legend = fig.add_axes([0.64, 0.06, 0.35, 0.88])
    ax_legend.axis("off")

    # Scatter
    for cid in uniq:
        mask = cluster_labels == cid
        ax_main.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            color=cluster_color[int(cid)], s=25, alpha=0.65,
            edgecolors="none",
        )

    # Annotate each cluster at its centroid
    for cid in uniq:
        mask = cluster_labels == cid
        cx, cy = X_2d[mask, 0].mean(), X_2d[mask, 1].mean()
        ax_main.text(
            cx, cy, str(int(cid)),
            fontsize=11, fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="black", alpha=0.8),
        )

    ax_main.set_title(title, fontsize=13)
    ax_main.set_xlabel("UMAP-1")
    ax_main.set_ylabel("UMAP-2")
    ax_main.grid(True, alpha=0.3)

    # Legend panel: one row per cluster with cluster id + summary
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, n_clusters + 1)
    ax_legend.invert_yaxis()
    ax_legend.text(0, 0, "Cluster summaries", fontsize=11, fontweight="bold")
    for i, cid in enumerate(uniq):
        cid_int = int(cid)
        size = int((cluster_labels == cid).sum())
        summary = cluster_summaries.get(cid_int, "")[:90]
        # Colored marker
        ax_legend.scatter(
            0.03, i + 1, color=cluster_color[cid_int],
            s=80, edgecolors="black", linewidths=0.5,
            transform=ax_legend.transData,
        )
        ax_legend.text(
            0.08, i + 1,
            f"{cid_int:2d}  (n={size:4d})  {summary}",
            fontsize=8, va="center", family="monospace",
        )

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Feature clustering + interactive map")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model state_dict (.pt)")
    parser.add_argument("--model", type=str, default="crosscoder",
                        choices=["topk_sae", "stacked_sae", "crosscoder", "txcdr"],
                        help="Architecture type of the checkpoint")
    parser.add_argument("--subject-model", type=str, default="deepseek-r1-distill-llama-8b",
                        choices=list_models(),
                        help="Subject LM the checkpoint was trained on")
    parser.add_argument("--label", type=str, default=None,
                        help="Run label for output filenames + autointerp lookup")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--T", type=int, default=5)
    parser.add_argument("--expansion-factor", type=int, default=8)
    parser.add_argument("--n-clusters", type=int, default=20)
    parser.add_argument("--n-pca", type=int, default=50)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--include-unlabeled", action="store_true")
    parser.add_argument("--skip-llm-labels", action="store_true",
                        help="Skip LLM cluster labeling (fast, no explainer model)")
    parser.add_argument("--explain-model", type=str, default="google/gemma-2-2b-it",
                        help="HF path of the LLM used for cluster labels")
    parser.add_argument("--explain-device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=str, default=VIZ_DIR)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Normalize arch name: txcdr is the old pipeline alias for crosscoder
    arch = "crosscoder" if args.model == "txcdr" else args.model
    label = args.label or (
        f"{arch}__{args.subject_model}__k{args.k}__T{args.T}"
    )
    print(f"=== Feature map: {label} ===")

    d_sae = get_model_config(args.subject_model).d_model * args.expansion_factor

    # Load model + extract feature directions
    print("Loading model + extracting decoder directions...")
    model = load_model(
        checkpoint=args.checkpoint,
        model_type=arch,
        subject_model=args.subject_model,
        k=args.k, T=args.T,
        expansion_factor=args.expansion_factor,
    )
    directions = get_feature_directions(model, arch)  # (d_sae, d_in)
    print(f"  directions: {directions.shape}")
    del model

    # Load existing autointerp (from the same parent dir we write outputs to)
    print("Loading autointerp explanations...")
    interps = load_autointerp(label, output_dir=args.output_dir)
    print(f"  {len(interps)} / {d_sae} features have explanations")

    # Filter features
    if args.include_unlabeled:
        feature_indices = np.arange(d_sae)
    else:
        labeled = sorted(fi for fi, d in interps.items() if d["explanation"])
        if not labeled:
            print("ERROR: no labeled features found. Run autointerp.py first or "
                  "use --include-unlabeled.")
            return
        feature_indices = np.array(labeled)
        directions = directions[feature_indices]
    print(f"  clustering {len(feature_indices)} features")

    # Cluster
    print("Running PCA -> UMAP -> KMeans...")
    X_2d, cluster_labels = cluster_features(
        directions,
        n_pca=args.n_pca,
        n_clusters=args.n_clusters,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        seed=args.seed,
    )

    # Cluster sizes
    uniq, counts = np.unique(cluster_labels, return_counts=True)
    print(f"  cluster sizes: {dict(zip(uniq.tolist(), counts.tolist()))}")

    # LLM-label the clusters
    if args.skip_llm_labels:
        cluster_summaries = {int(cid): f"Cluster {int(cid)}" for cid in uniq}
    else:
        cluster_summaries = label_clusters_with_gemma(
            feature_indices, cluster_labels, interps,
            args.explain_model, args.explain_device,
        )
        for cid in sorted(cluster_summaries):
            print(f"  cluster {cid}: {cluster_summaries[cid]}")

    # Save cluster metadata
    os.makedirs(args.output_dir, exist_ok=True)
    meta_path = os.path.join(args.output_dir, f"feature_map_{label}_clusters.json")
    meta = {
        "label": label,
        "n_features": int(len(feature_indices)),
        "n_clusters": int(args.n_clusters),
        "cluster_summaries": {str(k): v for k, v in cluster_summaries.items()},
        "cluster_sizes": {str(int(k)): int(v) for k, v in zip(uniq, counts)},
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved: {meta_path}")

    title = f"Feature Map — {label} ({len(feature_indices)} features, {args.n_clusters} clusters)"

    # Interactive HTML
    html_path = os.path.join(args.output_dir, f"feature_map_{label}.html")
    write_interactive_html(
        X_2d=X_2d,
        feature_indices=feature_indices,
        cluster_labels=cluster_labels,
        interps=interps,
        cluster_summaries=cluster_summaries,
        output_path=html_path,
        title=title,
    )

    # Static PNG
    png_path = os.path.join(args.output_dir, f"feature_map_{label}.png")
    write_static_png(
        X_2d=X_2d,
        cluster_labels=cluster_labels,
        cluster_summaries=cluster_summaries,
        output_path=png_path,
        title=title,
    )


if __name__ == "__main__":
    main()
