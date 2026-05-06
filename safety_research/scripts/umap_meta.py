"""
Meta-autointerp UMAP comparison across SAE / T-SAE / TXC.

For each arm, embed Haiku-explanation strings via sentence-transformers,
project to 2D with UMAP, cluster with HDBSCAN, then ask Claude Haiku to
produce one short label per cluster. Compare:

  - number of clusters
  - silhouette of clustering
  - intra-cluster cohesion (mean cosine sim)
  - HDBSCAN noise fraction
  - Claude judge: which arm's clusters are most semantically coherent
  - per-cluster safety-tag composition

Outputs:
  safety_research/results/umap_meta/<arm>/{coords.npy, labels.npy, cluster_labels.json, summary.json}
  safety_research/figures/umap_<arm>.png
  safety_research/figures/umap_combined.png
  safety_research/figures/umap_cluster_count.png
  safety_research/figures/umap_safety_composition.png
Also pushes plots and metric tables to wandb.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

SAFETY_DIR = Path("/home/cs29824/andre/temp_xc/safety_research")
sys.path.insert(0, str(SAFETY_DIR))

import wandb

ARMS = ["sae", "tsae", "txc"]
EXPLAIN_MODEL = "claude-haiku-4-5-20251001"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_ROOT = SAFETY_DIR / "results" / "umap_meta"
FIG_DIR = SAFETY_DIR / "figures"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_explanations(arm: str) -> list[dict]:
    path = SAFETY_DIR / "results" / "autointerp" / arm / "explanations.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in open(path)]


def embed(texts: list[str]):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                                device=DEVICE)
    return model.encode(texts, show_progress_bar=False,
                        normalize_embeddings=True)


def cluster(emb_2d: np.ndarray):
    """HDBSCAN on 2D UMAP coords. Prefers the standalone hdbscan package;
    falls back to sklearn.cluster.HDBSCAN (sklearn>=1.3) when unavailable.
    The two implementations are API-compatible for the kwargs used here.
    """
    try:
        import hdbscan
        cl = hdbscan.HDBSCAN(min_cluster_size=4, min_samples=2,
                             cluster_selection_epsilon=0.4)
    except ImportError:
        from sklearn.cluster import HDBSCAN
        cl = HDBSCAN(min_cluster_size=4, min_samples=2,
                     cluster_selection_epsilon=0.4)
    labels = cl.fit_predict(emb_2d)
    return labels, cl


def silhouette(emb_2d, labels) -> float:
    from sklearn.metrics import silhouette_score
    mask = labels >= 0
    if mask.sum() < 4 or len(set(labels[mask])) < 2:
        return float("nan")
    return float(silhouette_score(emb_2d[mask], labels[mask]))


def label_clusters_lexical(arm: str, clusters: dict[int, list[str]]) -> dict[int, str]:
    """Cheap lexical labeling: pick most distinctive tokens per cluster.

    For each cluster, score tokens by tf*log(1/df) across clusters and join
    the top 4. Avoids LLM cost; sufficient for visual inspection.
    """
    import re
    from collections import Counter
    STOP = set("the a an and or of to in for with on at as is are was were "
               "be by it that this these those from we you they its their "
               "feature features text describes describing common represent "
               "relates related related-to seemingly likely likely-related"
               .split())

    def toks(s: str) -> list[str]:
        return [w.lower() for w in re.findall(r"[A-Za-z]+", s)
                if len(w) >= 4 and w.lower() not in STOP]

    if not clusters:
        return {}
    cluster_tok = {cid: Counter(t for txt in exps for t in toks(txt))
                   for cid, exps in clusters.items()}
    df = Counter()
    for cid, c in cluster_tok.items():
        for tok in c.keys():
            df[tok] += 1
    n = len(cluster_tok)
    out = {}
    for cid, c in cluster_tok.items():
        scored = [(tok, freq * (1.0 + np.log(n / max(df[tok], 1))))
                  for tok, freq in c.items()]
        scored.sort(key=lambda x: -x[1])
        out[cid] = " · ".join(t for t, _ in scored[:4]) or f"c{cid}"
    return out


def judge_arm_quality(arm_summaries: list[dict]) -> dict:
    """Quantitative arm comparison without an LLM judge.

    Scores each arm on three numeric quality axes:
      - coherence  = mean intra-cluster cohesion (higher = better)
      - temporal   = ratio of features mentioning sequential / position cues
      - safety     = fraction of features assigned a non-NONE safety tag
    """
    import re
    SAFETY_RE = re.compile(r"\bsafety\b|refusal|deception|harmful|bias",
                           re.IGNORECASE)
    TEMPORAL_RE = re.compile(
        r"\b(sequenc|consecut|context|preced|follow|temporal|window|"
        r"position|previous|next|order)\w*", re.IGNORECASE)
    out: dict = {}
    for s in arm_summaries:
        coherence = float(s.get("mean_cohesion") or 0.0)
        all_examples = []
        n_safety = 0
        for c in s["clusters"]:
            for ex in c.get("sample", []):
                all_examples.append(ex)
            n_safety += sum(v for k, v in c.get("safety", {}).items()
                            if k != "NONE")
        n = max(s["n_features"], 1)
        temporal_frac = sum(1 for e in all_examples
                            if TEMPORAL_RE.search(e)) / max(len(all_examples), 1)
        # rough rescale to 0-10
        out[s["arm"]] = dict(
            coherence=round(min(max(coherence * 10, 0), 10), 2),
            temporal=round(min(max(temporal_frac * 10, 0), 10), 2),
            safety=round(min(n_safety / n * 10, 10), 2),
        )
    return out


def run_arm(arm: str) -> dict | None:
    import umap
    explanations = load_explanations(arm)
    if not explanations:
        print(f"  {arm}: no explanations, skip")
        return None
    out_dir = OUT_ROOT / arm
    out_dir.mkdir(parents=True, exist_ok=True)

    texts = [r["explanation"] for r in explanations]
    safety = [r.get("safety", "NONE") for r in explanations]
    feats  = [r["feat"] for r in explanations]

    emb = embed(texts)
    np.save(out_dir / "embeddings.npy", emb)

    proj = umap.UMAP(n_neighbors=10, min_dist=0.1, metric="cosine",
                     random_state=0).fit_transform(emb)
    np.save(out_dir / "coords.npy", proj)

    labels, cl = cluster(proj)
    np.save(out_dir / "labels.npy", labels)

    sil = silhouette(proj, labels)
    n_clusters = int((np.unique(labels) >= 0).sum())
    noise_frac = float((labels == -1).mean())

    # cluster cohesion in original embedding space
    cohesion = {}
    cluster_texts: dict[int, list[str]] = {}
    cluster_safety: dict[int, dict[str, int]] = {}
    for cid in sorted(set(labels)):
        if cid < 0:
            continue
        idxs = np.where(labels == cid)[0]
        sims = emb[idxs] @ emb[idxs].T
        n = len(idxs)
        if n > 1:
            avg = (sims.sum() - n) / (n * (n - 1))
            cohesion[int(cid)] = float(avg)
        cluster_texts[int(cid)] = [texts[i] for i in idxs]
        comp: dict[str, int] = {}
        for i in idxs:
            comp[safety[i]] = comp.get(safety[i], 0) + 1
        cluster_safety[int(cid)] = comp

    cl_names = label_clusters_lexical(arm, cluster_texts)

    cluster_summary = []
    for cid in sorted(cluster_texts):
        cluster_summary.append(dict(
            cluster=cid, name=cl_names.get(cid, "?"),
            n_features=len(cluster_texts[cid]),
            cohesion=cohesion.get(cid),
            safety=cluster_safety[cid],
            sample=cluster_texts[cid][:5],
        ))

    summary = dict(
        arm=arm,
        n_features=len(texts),
        n_clusters=n_clusters,
        noise_frac=noise_frac,
        silhouette=sil,
        mean_cohesion=float(np.mean(list(cohesion.values()))) if cohesion else float("nan"),
        clusters=cluster_summary,
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # plot
    fig, ax = plt.subplots(figsize=(7, 6))
    palette = plt.cm.tab20.colors
    for cid in sorted(set(labels)):
        m = labels == cid
        if cid < 0:
            ax.scatter(proj[m, 0], proj[m, 1], s=10, c="lightgray",
                       label="noise", alpha=0.6)
        else:
            ax.scatter(proj[m, 0], proj[m, 1], s=14,
                       c=[palette[cid % len(palette)]],
                       label=f"c{cid}: {cl_names.get(cid, '')[:30]}",
                       alpha=0.85)
    ax.set_title(f"UMAP — {arm}  "
                 f"(n_features={len(texts)}, k={n_clusters}, "
                 f"sil={sil:.2f}, noise={noise_frac:.2f})")
    ax.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.6, 1.0))
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"umap_{arm}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    return summary


def comparison_plots(summaries: list[dict]) -> None:
    arms = [s["arm"] for s in summaries]
    metrics = {
        "n_clusters": [s["n_clusters"] for s in summaries],
        "silhouette": [s["silhouette"] for s in summaries],
        "noise_frac": [s["noise_frac"] for s in summaries],
        "mean_cohesion": [s["mean_cohesion"] for s in summaries],
    }
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (k, vs) in zip(axes, metrics.items()):
        ax.bar(arms, vs, color=["#888", "#ffa44a", "#4a90e2"])
        ax.set_title(k); ax.set_ylabel(k)
        for i, v in enumerate(vs):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("UMAP / cluster metrics — meta-autointerp")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "umap_cluster_metrics.png", dpi=140)
    plt.close(fig)

    # safety composition stacked bar (% of features per safety tag)
    safety_tags = ["NONE", "REFUSAL", "DECEPTION", "HARMFUL_CONTENT", "BIAS"]
    arm_dist = {}
    for s in summaries:
        comp: dict[str, int] = {t: 0 for t in safety_tags}
        for c in s["clusters"]:
            for t, n in c["safety"].items():
                comp[t] = comp.get(t, 0) + n
        arm_dist[s["arm"]] = comp
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(arm_dist))
    for tag, color in zip(safety_tags,
                          ["#cccccc", "#d62728", "#ff7f0e", "#9467bd", "#8c564b"]):
        vs = np.array([arm_dist[a].get(tag, 0) for a in arm_dist])
        total = np.array([sum(arm_dist[a].values()) for a in arm_dist])
        frac = vs / np.maximum(total, 1)
        ax.bar(list(arm_dist.keys()), frac, bottom=bottom, color=color, label=tag)
        bottom += frac
    ax.set_title("Safety-tag composition of features (autointerp Haiku)")
    ax.set_ylabel("fraction of features")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "umap_safety_composition.png", dpi=140)
    plt.close(fig)


def main() -> None:
    run = wandb.init(project="temporal-crosscoders-safety",
                     name="umap_meta", tags=["safety", "umap"], reinit=True)
    print(f"wandb: {run.url}")
    summaries: list[dict] = []
    for arm in ARMS:
        s = run_arm(arm)
        if s is not None:
            summaries.append(s)
    if not summaries:
        print("No summaries — autointerp must run first.")
        run.finish()
        return
    comparison_plots(summaries)

    judgement = judge_arm_quality(summaries)
    out = dict(arms=summaries, judgement=judgement)
    (OUT_ROOT / "summary.json").write_text(json.dumps(out, indent=2))

    for s in summaries:
        wandb.log({f"umap/{s['arm']}/n_clusters": s["n_clusters"],
                   f"umap/{s['arm']}/silhouette": s["silhouette"],
                   f"umap/{s['arm']}/noise_frac": s["noise_frac"],
                   f"umap/{s['arm']}/mean_cohesion": s["mean_cohesion"]})
    if isinstance(judgement, dict) and "error" not in judgement:
        for arm, kv in judgement.items():
            if isinstance(kv, dict):
                for k, v in kv.items():
                    if isinstance(v, (int, float)):
                        wandb.log({f"judge/{arm}/{k}": v})

    wandb.log({
        "umap/sae": wandb.Image(str(FIG_DIR / "umap_sae.png")),
        "umap/tsae": wandb.Image(str(FIG_DIR / "umap_tsae.png")),
        "umap/txc": wandb.Image(str(FIG_DIR / "umap_txc.png")),
        "umap/cluster_metrics": wandb.Image(str(FIG_DIR / "umap_cluster_metrics.png")),
        "umap/safety_composition": wandb.Image(str(FIG_DIR / "umap_safety_composition.png")),
    })

    print("\nUMAP META SUMMARY")
    for s in summaries:
        print(f"  {s['arm']:6s}  k={s['n_clusters']:3d}  sil={s['silhouette']:+.2f}  "
              f"noise={s['noise_frac']:.2f}  cohesion={s['mean_cohesion']:.2f}")
    print(f"  judgement: {judgement}")
    run.finish()


if __name__ == "__main__":
    main()
