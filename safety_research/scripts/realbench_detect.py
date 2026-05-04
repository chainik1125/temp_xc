"""
Detection / monitoring eval — TXC vs T-SAE vs SAE on the real benchmark.

For each arm:
  1. Encode the cached L13 residuals into (N, F) sparse-feature vectors
     - SAE  T=1: features at the last token only          F = d_sae
     - T-SAE T=5: per-position features concatenated        F = T * d_sae
     - TXC  T=5: shared window-level features              F = d_sae
  2. Per-feature AUROC vs the binary refusal label (vectorised rank metric)
  3. Sparse linear probe (L2 LogReg) trained on `train`, evaluated on
     `test_in` and `test_ood`. Bootstrap 95% CIs over test prompts.
  4. Black-to-white boost vs (a) raw L13 residual probe, (b) prompt-text
     LogReg on TF-IDF — cheap black-box baseline.

All metrics are written to safety_research/results/realbench/detect/<arm>.json
and aggregated into a single summary.json.

Logs to wandb under temporal-crosscoders-safety, group="realbench-detect".
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

NLP_DIR = "/home/cs29824/andre/temp_xc/temporal_crosscoders/NLP"
ROOT = Path("/home/cs29824/andre/temp_xc/safety_research")
sys.path.insert(0, NLP_DIR)
from fast_models import FastStackedSAE, FastTemporalCrosscoder  # noqa

import wandb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTS = ROOT / "results" / "realbench" / "acts"
PROMPTS = ROOT / "results" / "realbench"
OUT = ROOT / "results" / "realbench" / "detect"
OUT.mkdir(parents=True, exist_ok=True)
CKPT_DIR = ROOT / "results" / "checkpoints"

ARMS = [
    ("sae",  CKPT_DIR / "sae__mid_res__k100__T1.pt",  FastStackedSAE,         {"T": 1, "k": 100}),
    ("tsae", CKPT_DIR / "tsae__mid_res__k100__T5.pt", FastStackedSAE,         {"T": 5, "k": 100}),
    ("txc",  CKPT_DIR / "txc__mid_res__k100__T5.pt",  FastTemporalCrosscoder, {"T": 5, "k": 100}),
]


def load_arm(name: str, path: Path, klass, cfg: dict) -> torch.nn.Module:
    sd = torch.load(path, map_location="cpu", weights_only=False)
    m = klass(d_in=2304, d_sae=18432, T=cfg["T"], k=cfg["k"])
    m.load_state_dict(sd["state_dict"])
    m.eval().to(DEVICE)
    return m


@torch.no_grad()
def encode_arm(arm: str, model: torch.nn.Module, acts: np.ndarray, batch: int = 256) -> np.ndarray:
    """acts: (N, 5, d) → arm features (N, F).

      sae:  use last token only,                     F = h
      tsae: per-position features concatenated,      F = T * h
      txc:  shared window features,                  F = h
    """
    N, T, d = acts.shape
    feats: list[np.ndarray] = []
    for i in range(0, N, batch):
        x = torch.from_numpy(acts[i:i + batch]).to(DEVICE).float()  # (B, 5, d)
        if arm == "sae":
            x = x[:, -1:, :]  # (B, 1, d)
            _, _, u = model(x)  # u: (B, 1, h)
            feats.append(u.squeeze(1).cpu().numpy())
        elif arm == "tsae":
            _, _, u = model(x)  # (B, T, h)
            B, Tt, h = u.shape
            feats.append(u.reshape(B, Tt * h).cpu().numpy())
        elif arm == "txc":
            _, _, z = model(x)  # (B, h)
            feats.append(z.cpu().numpy())
    return np.concatenate(feats, axis=0)


def vectorized_auc(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-column AUROC. Skips constant / all-zero columns (returns 0.5)."""
    if scores.ndim == 1:
        scores = scores[:, None]
    N, F = scores.shape
    pos = y.astype(bool)
    n_pos = int(pos.sum())
    n_neg = N - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.full(F, np.nan, dtype=np.float64)
    aucs = np.full(F, 0.5, dtype=np.float64)
    # Vectorise via argsort; expensive in memory but F=18k×N=1k → 18M floats fine
    order = np.argsort(scores, axis=0)
    ranks = np.empty_like(order, dtype=np.float64)
    rng = np.arange(1, N + 1)
    for j in range(F):
        ranks[order[:, j], j] = rng
    sum_pos = ranks[pos].sum(axis=0)
    aucs = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return aucs


def bootstrap_auc(scores: np.ndarray, y: np.ndarray, B: int = 1000,
                  rng: np.random.Generator | None = None) -> dict:
    rng = rng or np.random.default_rng(0)
    N = len(y)
    aucs = np.zeros(B)
    for b in range(B):
        idx = rng.integers(0, N, size=N)
        ys = y[idx]
        ss = scores[idx]
        if ys.min() == ys.max():
            aucs[b] = 0.5
        else:
            aucs[b] = roc_auc_score(ys, ss)
    return {
        "mean": float(np.mean(aucs)),
        "lo": float(np.quantile(aucs, 0.025)),
        "hi": float(np.quantile(aucs, 0.975)),
    }


def fit_probe(X_train, y_train, C: float = 1.0):
    scaler = StandardScaler(with_mean=False)  # sparse-friendly
    Xt = scaler.fit_transform(X_train)
    clf = LogisticRegression(C=C, max_iter=2000, solver="liblinear")
    clf.fit(Xt, y_train)
    return scaler, clf


def main() -> None:
    run = wandb.init(project="temporal-crosscoders-safety",
                     name="realbench-detect",
                     tags=["safety", "realbench", "detection"],
                     reinit=True)

    splits = {}
    for s in ("train", "test_in", "test_ood"):
        z = np.load(ACTS / f"{s}.npz")
        splits[s] = {"acts": z["acts"], "y": z["labels"].astype(int)}
        print(f"  {s}: acts {z['acts'].shape}  y {z['labels'].shape}  pos={int(z['labels'].sum())}")

    # ---- Black-box baseline: TF-IDF on prompt text -------------------------
    rows = {s: json.load(open(PROMPTS / f"{s}.json")) for s in splits}
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)
    Xt = tfidf.fit_transform([r["prompt"] for r in rows["train"]])
    bb = LogisticRegression(C=1.0, max_iter=2000)
    bb.fit(Xt, [r["label"] for r in rows["train"]])
    bb_results = {}
    for s in ("test_in", "test_ood"):
        Xs = tfidf.transform([r["prompt"] for r in rows[s]])
        scores = bb.predict_proba(Xs)[:, 1]
        y = np.array([r["label"] for r in rows[s]])
        auc = roc_auc_score(y, scores)
        ci = bootstrap_auc(scores, y)
        bb_results[s] = {"auc": float(auc), "ci": ci, "ap": float(average_precision_score(y, scores))}
    print(f"\nblack-box (TF-IDF):  test_in AUC={bb_results['test_in']['auc']:.3f}   "
          f"test_ood AUC={bb_results['test_ood']['auc']:.3f}")
    json.dump(bb_results, open(OUT / "blackbox_tfidf.json", "w"), indent=1)

    # ---- Raw-residual baseline (last-token L13) ----------------------------
    raw_train = splits["train"]["acts"][:, -1, :].astype(np.float32)
    raw_in = splits["test_in"]["acts"][:, -1, :].astype(np.float32)
    raw_ood = splits["test_ood"]["acts"][:, -1, :].astype(np.float32)
    sc = StandardScaler().fit(raw_train)
    raw_clf = LogisticRegression(C=0.1, max_iter=4000)
    raw_clf.fit(sc.transform(raw_train), splits["train"]["y"])
    raw_results = {}
    for s, X in (("test_in", raw_in), ("test_ood", raw_ood)):
        scores = raw_clf.predict_proba(sc.transform(X))[:, 1]
        y = splits[s]["y"]
        auc = roc_auc_score(y, scores)
        ci = bootstrap_auc(scores, y)
        raw_results[s] = {"auc": float(auc), "ci": ci, "ap": float(average_precision_score(y, scores))}
    print(f"raw L13 residual:    test_in AUC={raw_results['test_in']['auc']:.3f}   "
          f"test_ood AUC={raw_results['test_ood']['auc']:.3f}")
    json.dump(raw_results, open(OUT / "raw_residual.json", "w"), indent=1)

    # ---- Per-arm sparse-feature probes -------------------------------------
    arm_summary: dict[str, dict] = {}
    for arm, ckpt, klass, cfg in ARMS:
        print(f"\n=== arm={arm} ===")
        model = load_arm(arm, ckpt, klass, cfg)
        f_train = encode_arm(arm, model, splits["train"]["acts"]).astype(np.float32)
        f_in = encode_arm(arm, model, splits["test_in"]["acts"]).astype(np.float32)
        f_ood = encode_arm(arm, model, splits["test_ood"]["acts"]).astype(np.float32)
        F_dim = f_train.shape[1]
        density_train = float((f_train > 0).mean())
        print(f"  features: {F_dim}  density={density_train:.4f}")

        # per-feature AUC on train
        per_auc = vectorized_auc(f_train, splits["train"]["y"])
        per_auc_signed = np.abs(per_auc - 0.5) * 2  # 0..1, where 1 = perfect
        # save top-K most discriminative for steering later
        top_k = 100
        top_idx = np.argsort(-per_auc_signed)[:top_k]
        top_aucs = per_auc[top_idx]
        np.save(OUT / f"{arm}_per_feat_auc.npy", per_auc)
        np.save(OUT / f"{arm}_top_idx.npy", top_idx)
        np.save(OUT / f"{arm}_top_auc.npy", top_aucs)

        # sparse probe — keep top-2k features by per-feature AUC to keep probe fast
        keep = np.argsort(-per_auc_signed)[: min(2000, F_dim)]
        Xtr = f_train[:, keep]
        scaler = StandardScaler(with_mean=False).fit(Xtr)
        clf = LogisticRegression(C=1.0, max_iter=4000, solver="liblinear")
        clf.fit(scaler.transform(Xtr), splits["train"]["y"])

        results = {}
        roc_data = {}
        for s, F in (("test_in", f_in), ("test_ood", f_ood)):
            X = F[:, keep]
            scores = clf.predict_proba(scaler.transform(X))[:, 1]
            y = splits[s]["y"]
            auc = roc_auc_score(y, scores)
            ci = bootstrap_auc(scores, y)
            ap = float(average_precision_score(y, scores))
            fpr, tpr, _ = roc_curve(y, scores)
            results[s] = {"auc": float(auc), "ci": ci, "ap": ap}
            roc_data[s] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
            print(f"  {s}: AUC={auc:.3f} [{ci['lo']:.3f}, {ci['hi']:.3f}]  AP={ap:.3f}")
            wandb.log({f"detect/{arm}/{s}/auc": auc,
                       f"detect/{arm}/{s}/ap": ap})

        # Black-to-white boost = arm AUC - black-box AUC
        boost = {}
        for s in ("test_in", "test_ood"):
            boost[s] = float(results[s]["auc"] - bb_results[s]["auc"])

        # Probe coefficients on the full feature index for steering reuse
        full_coef = np.zeros(F_dim, dtype=np.float32)
        full_coef[keep] = clf.coef_[0].astype(np.float32)
        np.save(OUT / f"{arm}_probe_coef.npy", full_coef)

        arm_summary[arm] = dict(
            F=F_dim,
            density_train=density_train,
            per_feat_top_auc=top_aucs.tolist(),
            results=results,
            roc=roc_data,
            black_to_white_boost=boost,
        )
        json.dump(arm_summary[arm], open(OUT / f"{arm}_summary.json", "w"), indent=1)

        del model
        torch.cuda.empty_cache()

    summary = dict(blackbox=bb_results, raw_residual=raw_results, arms=arm_summary)
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)

    table = "\nDetection summary (AUC):\n"
    table += "  arm                 test_in  test_ood   boost(in)  boost(ood)\n"
    table += f"  TF-IDF (black-box) {bb_results['test_in']['auc']:7.3f}  {bb_results['test_ood']['auc']:8.3f}\n"
    table += f"  raw L13 residual   {raw_results['test_in']['auc']:7.3f}  {raw_results['test_ood']['auc']:8.3f}\n"
    for arm in ("sae", "tsae", "txc"):
        a = arm_summary[arm]
        table += (f"  {arm:18s} {a['results']['test_in']['auc']:7.3f}  "
                  f"{a['results']['test_ood']['auc']:8.3f}   "
                  f"{a['black_to_white_boost']['test_in']:+.3f}     "
                  f"{a['black_to_white_boost']['test_ood']:+.3f}\n")
    print(table)
    with open(OUT / "summary_table.txt", "w") as f:
        f.write(table)
    wandb.summary.update({"detect_summary": table})
    run.finish()


if __name__ == "__main__":
    main()
