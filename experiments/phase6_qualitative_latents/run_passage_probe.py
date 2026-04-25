"""Paper-style contextual probing: for each (arch, seed, concat), train
a k-sparse multinomial logistic regression to predict passage ID from
SAE activations. Writes `passage_probe_results.jsonl`.

Directly analogous to the T-SAE paper's §4.2 "contextual" probing
(questions 1, 2, 3 share ID — can features tell them apart?). Our
concat_A has 3 passages (principia/genetics_q/gita), concat_B has 4,
concat_random has 7.

Feature selection (top-k): rank features by between-class mean-
activation variance — the multinomial analogue of the per-task absolute
mean-difference used in Phase 5 binary probing.

Reports 5-fold stratified CV accuracy per (arch, seed, concat, k).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[2]
Z_DIR = REPO / "experiments/phase6_qualitative_latents/z_cache"
CONCAT_DIR = REPO / "experiments/phase6_qualitative_latents/concat_corpora"
OUT = REPO / "experiments/phase6_qualitative_latents/results/passage_probe_results.jsonl"


def _z_path(arch: str, seed: int, concat: str) -> Path:
    # Mirror encode_archs.py naming: __seed{n}__z.npy for n != 42, else legacy.
    d = Z_DIR / f"concat_{concat}"
    p1 = d / f"{arch}__seed{seed}__z.npy"
    if p1.exists():
        return p1
    if seed == 42:
        p2 = d / f"{arch}__z.npy"
        if p2.exists():
            return p2
    return p1  # nonexistent, caller will skip


def _build_labels(concat: str) -> np.ndarray | None:
    p = CONCAT_DIR / f"concat_{concat}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    n_tok = max(pp["end"] for pp in d["provenance"])
    y = np.full(n_tok, -1, dtype=np.int64)
    for i, pp in enumerate(d["provenance"]):
        y[pp["start"]:pp["end"]] = i
    return y


def _rank_between_class(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """For each feature: between-class variance of mean activation."""
    classes = np.unique(y)
    means = np.stack([z[y == c].mean(axis=0) for c in classes], axis=0)
    between = means.var(axis=0)
    return np.argsort(-between)  # descending


def probe_cell(arch: str, seed: int, concat: str,
               ks: list[int], n_splits: int = 5) -> dict | None:
    z_path = _z_path(arch, seed, concat)
    if not z_path.exists():
        return None
    y = _build_labels(concat)
    if y is None:
        return None
    z = np.load(z_path).astype(np.float32)
    assert z.shape[0] == y.shape[0], (z.shape, y.shape)

    # For concat_random: classes are fw_00..fw_06 (7 classes, balanced 256 tok each).
    # For concat_A / B: 3 / 4 classes, varied sizes but all ≥ 200.
    n_classes = len(np.unique(y))
    if n_classes < 2:
        return None

    # Rank features once by between-class variance on full z.
    order = _rank_between_class(z, y)

    out = {"arch": arch, "seed": seed, "concat": concat,
           "n_tokens": int(z.shape[0]), "n_classes": int(n_classes),
           "per_k": {}}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    for k in ks:
        feats = order[:k]
        Xk = z[:, feats]
        accs = []
        for tr, te in skf.split(Xk, y):
            # sklearn ≥1.5 auto-selects multinomial for multi-class with lbfgs.
            clf = LogisticRegression(max_iter=1000, solver="lbfgs")
            clf.fit(Xk[tr], y[tr])
            accs.append(clf.score(Xk[te], y[te]))
        out["per_k"][str(k)] = {"acc_mean": float(np.mean(accs)),
                                "acc_std": float(np.std(accs))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=None,
                    help="override arch list (default: all primary + T-sweep + phase62)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    ap.add_argument("--concats", nargs="+", default=["A", "B", "random"])
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10, 20])
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args()

    if args.archs is None:
        args.archs = [
            "tsae_paper",
            "agentic_txc_10_bare",
            "agentic_txc_12_bare_batchtopk",
            "agentic_txc_02",
            "agentic_txc_02_batchtopk",
            "phase63_track2_t3",
            "phase63_track2_t10",
            "phase63_track2_t20",
            "phase62_c1_track2_matryoshka",
            "phase62_c2_track2_contrastive",
            "phase62_c3_track2_matryoshka_contrastive",
            "phase62_c5_track2_longer",
            "phase62_c6_bare_batchtopk_longer",
            "agentic_mlc_08",
        ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Read existing so we don't duplicate
    existing = set()
    if out_path.exists():
        for line in out_path.open():
            r = json.loads(line)
            existing.add((r["arch"], r["seed"], r["concat"]))

    with out_path.open("a") as f:
        for arch in args.archs:
            for seed in args.seeds:
                for concat in args.concats:
                    if (arch, seed, concat) in existing:
                        print(f"[skip] {arch} seed={seed} {concat}: already done")
                        continue
                    out = probe_cell(arch, seed, concat, args.ks)
                    if out is None:
                        print(f"[skip] {arch} seed={seed} {concat}: no z cache or labels")
                        continue
                    f.write(json.dumps(out) + "\n")
                    f.flush()
                    k5 = out["per_k"].get("5", {}).get("acc_mean", float("nan"))
                    print(f"[done] {arch} seed={seed} {concat}  "
                          f"n_classes={out['n_classes']}  acc@k=5 = {k5:.3f}")

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
