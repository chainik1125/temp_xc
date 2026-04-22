"""Phase 6 UMAP + silhouette analysis on concat-set C latents.

Produces the TSNE analogue of Ye et al. 2025 Figure 2: for each arch,
UMAP-reduce the high-level latent prefix (and optionally the low-level
prefix) to 2D, colour by {semantic subject, question id, POS tag},
compute silhouette scores.

Output figures land under
`experiments/phase6_qualitative_latents/results/umap/`:

    umap_high__semantic.png
    umap_high__context.png
    umap_high__pos.png
    umap_low__semantic.png  (only for archs with H/L split)
    silhouette_scores.csv
    summary.md
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
Z_DIR = REPO / "experiments/phase6_qualitative_latents/z_cache"
OUT_DIR = REPO / "experiments/phase6_qualitative_latents/results/umap"

# Arch -> (high_prefix_slice, low_prefix_slice | None)
#  - "high" for matryoshka archs is the scale-1 / d_sae//2 prefix.
#  - "low"  is everything after the high prefix (None = no H/L split).
# For non-matryoshka archs (tfa_big novel_codes) we take the top-variance
# d_sae//2 indices on concat_C as a pragmatic "high" slice, since Fig 2
# compares the paper's high/low split against baselines by variance-rank.
ARCH_SPLITS: dict[str, tuple[tuple[int, int], tuple[int, int] | None]] = {
    "agentic_txc_02": ((0, 18_432 // 5), (18_432 // 5, 18_432)),  # scale-1 = d_sae / T = 3686
    "agentic_mlc_08": ((0, 18_432 // 2), (18_432 // 2, 18_432)),
    "tsae_ours":      ((0, 18_432 // 2), (18_432 // 2, 18_432)),
    # tsae_paper: matryoshka [0.2, 0.8] → high = first 3686 (20%), low = rest.
    "tsae_paper":     ((0, 3686), (3686, 18_432)),
    # tfa_big: novel_codes has no intrinsic H/L; we'll rank by variance and
    # take top-50% as "high", bottom-50% as "low". Handled in code below.
    "tfa_big":        (None, None),
}

SILHOUETTE_SUBSAMPLE = 2000  # keep silhouette computation fast


def load_z_and_labels(arch: str, concat_name: str = "concat_C"):
    """Return (z flat, subject_labels, qid_labels, pos_labels) for a concat."""
    z_path = Z_DIR / concat_name / f"{arch}__z.npy"
    prov_path = Z_DIR / concat_name / "provenance.json"
    z = np.load(z_path).astype(np.float32)  # (160, 20, d_sae)
    prov = json.loads(prov_path.read_text())
    n_seq, seq_len, d_sae = z.shape

    subj = []
    qid = []
    for s in prov["sequences"]:
        subj.extend([s["subject"]] * s["n_tokens"])
        qid.extend([f"{s['subject']}#{s['qid']}"] * s["n_tokens"])
    subj = np.array(subj)
    qid = np.array(qid)

    # Per-token POS tags. spaCy's wheels don't cover CPython 3.14 at the
    # time of this phase, so we degrade to a character-class heuristic if
    # spaCy isn't importable. This loses the paper's fine-grained POS
    # taxonomy but still distinguishes punctuation / digit / alpha /
    # whitespace tokens — enough to read a "syntax" panel trend.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "google/gemma-2-2b-it", token=open("/workspace/hf_cache/token").read().strip()
    )
    try:
        import spacy  # noqa: F401
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
        def pos_of(piece: str) -> str:
            doc = nlp(piece if piece else "_")
            return doc[0].pos_ if len(doc) else "X"
    except Exception:
        import string
        def pos_of(piece: str) -> str:
            piece = piece.strip()
            if not piece:
                return "SPACE"
            if all(c in string.punctuation for c in piece):
                return "PUNCT"
            if any(c.isdigit() for c in piece):
                return "NUM"
            if piece[0].isupper():
                return "PROPN_OR_SENT_START"
            return "WORD"

    pos = []
    for s in prov["sequences"]:
        ids = s["token_ids"]
        text_per_tok = [tok.decode([t]).strip() or "_" for t in ids]
        for piece in text_per_tok:
            pos.append(pos_of(piece))
    pos = np.array(pos)

    z_flat = z.reshape(n_seq * seq_len, d_sae)
    return z_flat, subj, qid, pos


def _run_umap(z: np.ndarray, d_out: int = 2, seed: int = 42) -> np.ndarray:
    import umap
    red = umap.UMAP(
        n_components=d_out, random_state=seed,
        n_neighbors=15, min_dist=0.1, metric="cosine",
    )
    return red.fit_transform(z)


def _silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import silhouette_score
    n = x.shape[0]
    if n > SILHOUETTE_SUBSAMPLE:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=SILHOUETTE_SUBSAMPLE, replace=False)
        x = x[idx]
        labels = labels[idx]
    # silhouette_score requires >1 cluster with >1 sample
    if len(set(labels)) < 2:
        return float("nan")
    return float(silhouette_score(x, labels, metric="cosine"))


def plot_umap(
    coords: np.ndarray, labels: np.ndarray, title: str, dst: Path,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    dst.parent.mkdir(parents=True, exist_ok=True)
    uniq = sorted(set(labels))
    palette = sns.color_palette("tab20", n_colors=max(len(uniq), 8))
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=120)
    for i, lb in enumerate(uniq):
        mask = labels == lb
        ax.scatter(coords[mask, 0], coords[mask, 1], s=6,
                   alpha=0.6, color=palette[i % len(palette)],
                   label=str(lb)[:28])
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_xticks([]); ax.set_yticks([])
    if len(uniq) <= 16:
        ax.legend(fontsize=6, loc="center left",
                  bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(dst, dpi=120, bbox_inches="tight")
    # Thumbnail for agent inspection (per CLAUDE.md rule)
    thumb = dst.with_suffix("").name + ".thumb.png"
    fig.savefig(dst.with_name(thumb), dpi=48, bbox_inches="tight")
    plt.close(fig)


def slice_high_low(arch: str, z: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (z_high, z_low | None) for UMAP/silhouette."""
    hi_slice, lo_slice = ARCH_SPLITS.get(arch, (None, None))
    if hi_slice is not None:
        z_hi = z[:, hi_slice[0]:hi_slice[1]]
        z_lo = z[:, lo_slice[0]:lo_slice[1]] if lo_slice is not None else None
        return z_hi, z_lo
    # For tfa_big: rank by variance, take top 50% as high, rest as low
    var = z.var(axis=0)
    order = np.argsort(-var)
    half = z.shape[1] // 2
    hi_idx = order[:half]
    lo_idx = order[half:]
    return z[:, hi_idx], z[:, lo_idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archs", type=str, nargs="+",
                   default=["agentic_txc_02", "agentic_mlc_08",
                            "tsae_paper", "tsae_ours", "tfa_big"])
    p.add_argument("--concat", type=str, default="concat_C_v2",
                   help="Which concat-set to use (default: concat_C_v2, "
                        "the paper-faithful 10-subject × 30-token set).")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    silhouette_rows = []

    for arch in args.archs:
        z_path = Z_DIR / args.concat / f"{arch}__z.npy"
        if not z_path.exists():
            print(f"skip {arch}/{args.concat}: no z cache")
            continue
        print(f"[{arch}] load + UMAP on {args.concat}")
        z_flat, subj, qid, pos = load_z_and_labels(arch, args.concat)

        for kind, z_slice in (("high", slice_high_low(arch, z_flat)[0]),
                              ("low", slice_high_low(arch, z_flat)[1])):
            if z_slice is None:
                continue
            t0 = time.time()
            coords = _run_umap(z_slice)
            dt = time.time() - t0
            print(f"  {arch}/{kind}: UMAP {dt:.1f}s, coords shape {coords.shape}")
            for label_name, labels in [
                ("semantic", subj), ("context", qid), ("pos", pos)
            ]:
                plot_umap(
                    coords, labels,
                    title=f"{arch} — {kind} — {label_name} — {args.concat}",
                    dst=OUT_DIR / f"{args.concat}__umap_{kind}__{arch}__{label_name}.png",
                )
                score = _silhouette(z_slice, labels)
                silhouette_rows.append({
                    "concat": args.concat,
                    "arch": arch, "prefix": kind, "label": label_name,
                    "silhouette": score,
                })

    import csv
    dst = OUT_DIR / f"{args.concat}__silhouette_scores.csv"
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["concat", "arch", "prefix", "label", "silhouette"])
        w.writeheader(); w.writerows(silhouette_rows)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
