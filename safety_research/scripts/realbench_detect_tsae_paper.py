"""Add Bhalla T-SAE to realbench detection results.

Loads the trained tsae_paper checkpoint, encodes the cached
JBB / XSTest / MaliciousInstruct activations, fits a sparse linear
probe (top-2k features by per-feature AUC, L2 LogReg) and writes
results to safety_research/results/realbench/detect/tsae_paper.json
in the same schema the existing arms use, so the meta-report
can pull it next to the others.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score,
                              roc_curve)
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/cs29824/andre/temp_xc/safety_research")
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
from train_tsae_paper import TSAEPaper  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTS = ROOT / "results" / "realbench" / "acts"
OUT = ROOT / "results" / "realbench" / "detect"
OUT.mkdir(parents=True, exist_ok=True)
CKPT = ROOT / "results" / "checkpoints" / "tsae_paper__mid_res__k100__T1.pt"


def vectorized_auc(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    if scores.ndim == 1:
        scores = scores[:, None]
    N, F = scores.shape
    pos = y.astype(bool)
    n_pos = int(pos.sum())
    n_neg = N - n_pos
    order = np.argsort(scores, axis=0)
    ranks = np.empty_like(order, dtype=np.float64)
    rng = np.arange(1, N + 1)
    for j in range(F):
        ranks[order[:, j], j] = rng
    sum_pos = ranks[pos].sum(axis=0)
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def bootstrap_auc(scores, y, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(B):
        idx = rng.integers(0, len(y), len(y))
        ys, ss = y[idx], scores[idx]
        if ys.min() == ys.max():
            aucs.append(0.5)
        else:
            aucs.append(roc_auc_score(ys, ss))
    a = np.array(aucs)
    return {"mean": float(a.mean()),
            "lo": float(np.quantile(a, 0.025)),
            "hi": float(np.quantile(a, 0.975))}


@torch.no_grad()
def encode_window_to_anchor_features(model: TSAEPaper, acts: np.ndarray,
                                      batch: int = 256) -> np.ndarray:
    """Encode the *last* token of each window (Bhalla T-SAE is per-token at inference).

    acts shape: (N, T_window=5, d_in). We use the last token (final residual
    in the window — same convention the SAE arm uses).
    """
    model.eval()
    feats = []
    N = acts.shape[0]
    for i in range(0, N, batch):
        x = torch.from_numpy(acts[i:i + batch]).to(DEVICE).float()
        last = x[:, -1, :]
        z = model.encode(last)
        feats.append(z.cpu().numpy())
    return np.concatenate(feats, axis=0)


def main() -> None:
    print(f"Loading {CKPT}")
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = TSAEPaper(d_in=sd["d_in"], d_sae=sd["d_sae"], k_pos=sd["k_pos"]).to(DEVICE)
    model.load_state_dict(sd["state_dict"])
    print(f"  threshold = {sd.get('threshold', 'n/a')}, group_sizes = {sd.get('group_sizes')}")

    splits = {}
    for s in ("train", "test_in", "test_ood"):
        z = np.load(ACTS / f"{s}.npz")
        splits[s] = {"acts": z["acts"], "y": z["labels"].astype(int)}
        print(f"  {s}: acts {z['acts'].shape}  y {z['labels'].shape}  pos={int(z['labels'].sum())}")

    f_train = encode_window_to_anchor_features(model, splits["train"]["acts"]).astype(np.float32)
    f_in = encode_window_to_anchor_features(model, splits["test_in"]["acts"]).astype(np.float32)
    f_ood = encode_window_to_anchor_features(model, splits["test_ood"]["acts"]).astype(np.float32)
    F_dim = f_train.shape[1]
    density_train = float((f_train > 0).mean())
    print(f"\n  features per prompt: {F_dim}  density on train = {density_train:.4f}")

    # Per-feature AUC on train
    per_auc = vectorized_auc(f_train, splits["train"]["y"])
    per_auc_signed = np.abs(per_auc - 0.5) * 2
    keep = np.argsort(-per_auc_signed)[: min(2000, F_dim)]

    # Sparse probe
    Xtr = f_train[:, keep]
    scaler = StandardScaler(with_mean=False).fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=4000, solver="liblinear")
    clf.fit(scaler.transform(Xtr), splits["train"]["y"])

    out: dict = {"arm": "tsae_paper", "ckpt": str(CKPT),
                  "F_dim": int(F_dim), "density_train": density_train,
                  "n_features_kept": int(len(keep))}
    for s, F in (("test_in", f_in), ("test_ood", f_ood)):
        X = F[:, keep]
        scores = clf.predict_proba(scaler.transform(X))[:, 1]
        y = splits[s]["y"]
        auc = roc_auc_score(y, scores)
        ci = bootstrap_auc(scores, y)
        ap = float(average_precision_score(y, scores))
        out[s] = {"auc": float(auc), "ci": ci, "ap": ap}
        print(f"  {s}: AUC={auc:.3f} [{ci['lo']:.3f}, {ci['hi']:.3f}]  AP={ap:.3f}")

    np.save(OUT / "tsae_paper_per_feat_auc.npy", per_auc)
    np.save(OUT / "tsae_paper_top_idx.npy", keep[:100])
    json.dump(out, open(OUT / "tsae_paper.json", "w"), indent=2)
    print(f"\n  wrote {OUT / 'tsae_paper.json'}")


if __name__ == "__main__":
    main()
