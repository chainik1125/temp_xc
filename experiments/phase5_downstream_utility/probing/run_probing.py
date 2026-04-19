"""Sparse-probing runner for Phase 5 — v2 using split-cache format.

Cache format (from build_probe_cache.py):
    acts_anchor.npz:  train_acts (N, LAST_N=20, d), train_last_idx, train_labels +
                      test_*  — L13 tail. Used by TopK/TXCDR/Stacked/Matryoshka.
    acts_mlc.npz:     train_acts (N, 5, d), train_labels + test_*.  MLC-specific.

Probing protocol (matches Kantamneni et al. §2.2):
    - Top-k latents by abs(mean_train[y=1] - mean_train[y=0])
    - Fit L1 logistic regression on those k features
    - AUC on held-out test set

Baselines:
    - L2 LR on raw last-token L13 activations.
    - Attention-pooled (Eq. 2) over the tail-20 L13 activations.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.architectures.topk_sae import TopKSAE
from src.architectures.stacked_sae import StackedSAE
from src.architectures.crosscoder import TemporalCrosscoder
from src.architectures.mlc import MultiLayerCrosscoder

from experiments.phase5_downstream_utility.probing.attn_pooled_probe import (
    train_attn_probe, AttnProbeConfig,
)


REPO = Path("/workspace/temp_xc")
PROBE_CACHE = REPO / "experiments/phase5_downstream_utility/results/probe_cache"
CKPT_DIR = REPO / "experiments/phase5_downstream_utility/results/ckpts"
OUT_JSONL = REPO / "experiments/phase5_downstream_utility/results/probing_results.jsonl"

K_VALUES = [1, 2, 5, 20]


def _load_task_cache(task_dir: Path) -> dict[str, Any]:
    anchor = np.load(task_dir / "acts_anchor.npz")
    mlc = np.load(task_dir / "acts_mlc.npz")
    meta = json.loads((task_dir / "meta.json").read_text())
    return {
        "anchor_train": anchor["train_acts"].astype(np.float32),  # (N, LAST_N, d)
        "anchor_test": anchor["test_acts"].astype(np.float32),
        "train_last_idx": anchor["train_last_idx"],
        "test_last_idx": anchor["test_last_idx"],
        "mlc_train": mlc["train_acts"].astype(np.float32),  # (N, 5, d)
        "mlc_test": mlc["test_acts"].astype(np.float32),
        "train_labels": anchor["train_labels"],
        "test_labels": anchor["test_labels"],
        "meta": meta,
    }


def _last_token(acts, last_idx):
    """(N, LAST_N, d) + last_idx -> (N, d) at each sample's last real pos."""
    N = acts.shape[0]
    out = np.empty((N, acts.shape[-1]), dtype=np.float32)
    for i in range(N):
        out[i] = acts[i, int(last_idx[i])]
    return out


def _window_at_last(acts, last_idx, T):
    """(N, LAST_N, d) + last_idx -> (N, T, d) left-clamped window ending at last_idx."""
    N = acts.shape[0]
    d = acts.shape[-1]
    out = np.empty((N, T, d), dtype=np.float32)
    for i in range(N):
        li = int(last_idx[i])
        start = max(0, li - T + 1)
        win = acts[i, start:li + 1]
        if win.shape[0] < T:
            pad = np.broadcast_to(win[0:1], (T - win.shape[0], d))
            win = np.concatenate([pad, win], axis=0)
        out[i] = win
    return out


# ─── encoders per arch ───

def _encode_topk(model, anchor_acts, last_idx, device):
    X = _last_token(anchor_acts, last_idx)
    with torch.no_grad():
        z = model.encode(torch.from_numpy(X).to(device))
    return z.cpu().numpy()


def _encode_mlc(model, mlc_acts, device):
    with torch.no_grad():
        z = model.encode(torch.from_numpy(mlc_acts).to(device))
    return z.cpu().numpy()


def _encode_txcdr(model, anchor_acts, last_idx, T, device):
    X = _window_at_last(anchor_acts, last_idx, T)
    with torch.no_grad():
        z = model.encode(torch.from_numpy(X).to(device))
    return z.cpu().numpy()


def _encode_stacked_last(model, anchor_acts, last_idx, T, device):
    """Last-position SAE within the stack, applied to the anchor last token."""
    X = _last_token(anchor_acts, last_idx)
    with torch.no_grad():
        z = model.saes[-1].encode(torch.from_numpy(X).to(device))
    return z.cpu().numpy()


def _encode_matryoshka(model, anchor_acts, last_idx, T, device):
    X = _window_at_last(anchor_acts, last_idx, T)
    with torch.no_grad():
        z = model.encode(torch.from_numpy(X).to(device))
    return z.cpu().numpy()


# ─── feature selection + probe ───


def top_k_by_class_sep(Z_train, y_train, k):
    pos_mean = Z_train[y_train == 1].mean(axis=0)
    neg_mean = Z_train[y_train == 0].mean(axis=0)
    diff = np.abs(pos_mean - neg_mean)
    k = min(k, Z_train.shape[1])
    top = np.argpartition(-diff, k - 1)[:k]
    return np.sort(top)


def sae_probe_auc(Z_train, y_train, Z_test, y_test, k):
    idx = top_k_by_class_sep(Z_train, y_train, k)
    Xtr = Z_train[:, idx]
    Xte = Z_test[:, idx]
    if Xtr.shape[0] < 5 or len(np.unique(y_train)) < 2:
        return 0.5
    scaler = StandardScaler(with_mean=False)
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    try:
        clf = LogisticRegression(
            penalty="l1", solver="liblinear", max_iter=2000, C=1.0,
        )
        clf.fit(Xtr_s, y_train)
    except Exception:
        return 0.5
    preds = clf.decision_function(Xte_s)
    if len(np.unique(y_test)) < 2:
        return 0.5
    return float(roc_auc_score(y_test, preds))


def last_token_lr_auc(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    try:
        clf = LogisticRegression(
            penalty="l2", solver="lbfgs", max_iter=2000, C=1.0,
        )
        clf.fit(Xtr, y_train)
    except Exception:
        return 0.5
    preds = clf.decision_function(Xte)
    if len(np.unique(y_test)) < 2:
        return 0.5
    return float(roc_auc_score(y_test, preds))


def attn_pool_auc(acts_tr, last_tr, y_tr, acts_te, last_te, y_te, device,
                  seed: int = 42):
    """acts_* shape: (N, LAST_N, d); L13 only (anchor)."""
    mask_tr = np.arange(acts_tr.shape[1])[None, :] <= last_tr[:, None]
    mask_te = np.arange(acts_te.shape[1])[None, :] <= last_te[:, None]
    try:
        out = train_attn_probe(
            X_train=torch.from_numpy(acts_tr).float(),
            y_train=torch.from_numpy(y_tr),
            X_test=torch.from_numpy(acts_te).float(),
            y_test=torch.from_numpy(y_te),
            mask_train=torch.from_numpy(mask_tr),
            mask_test=torch.from_numpy(mask_te),
            cfg=AttnProbeConfig(seed=seed),
            device=device,
        )
        return out["test_auc"]
    except Exception as e:
        print(f"    attn-pool FAIL: {e}")
        return 0.5


def _iter_ckpts(run_ids):
    if run_ids:
        return [(rid, CKPT_DIR / f"{rid}.pt") for rid in run_ids]
    return [(p.stem, p) for p in sorted(CKPT_DIR.glob("*.pt"))]


def _load_model_for_run(run_id, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = state["meta"]
    arch = state["arch"]
    d_sae = 18_432
    d_in = 2304
    if arch == "topk_sae":
        model = TopKSAE(d_in, d_sae, k=meta["k_pos"]).to(device)
    elif arch == "mlc":
        model = MultiLayerCrosscoder(d_in, d_sae, n_layers=5, k=meta["k_pos"]).to(device)
    elif arch.startswith("txcdr_t"):
        T = meta["T"]
        k_eff = meta["k_win"] or (meta["k_pos"] * T)
        model = TemporalCrosscoder(d_in, d_sae, T, k_eff).to(device)
    elif arch.startswith("stacked_t"):
        T = meta["T"]
        model = StackedSAE(d_in, d_sae, T, k=meta["k_pos"]).to(device)
    elif arch == "matryoshka_t5":
        from src.architectures.matryoshka_txcdr import PositionMatryoshkaTXCDR
        T = meta["T"]
        k_eff = meta["k_win"] or (meta["k_pos"] * T)
        model = PositionMatryoshkaTXCDR(d_in, d_sae, T, k_eff).to(device)
    elif arch == "shared_perpos_t5":
        from experiments.phase5_downstream_utility.train_primary_archs import (
            SharedPerPositionSAE,
        )
        T = meta["T"]
        model = SharedPerPositionSAE(d_in, d_sae, T, k=meta["k_pos"]).to(device)
    else:
        raise ValueError(f"Unknown arch {arch}")
    cast_state = {
        k: v.to(torch.float32) if v.dtype == torch.float16 else v
        for k, v in state["state_dict"].items()
    }
    model.load_state_dict(cast_state)
    model.eval()
    return model, arch, meta


def _encode_for_probe(
    model, arch, meta,
    tc: dict, split: str,
    device,
):
    if split == "train":
        anchor = tc["anchor_train"]
        mlc = tc["mlc_train"]
        li = tc["train_last_idx"]
    else:
        anchor = tc["anchor_test"]
        mlc = tc["mlc_test"]
        li = tc["test_last_idx"]

    if arch == "topk_sae":
        return _encode_topk(model, anchor, li, device)
    if arch == "mlc":
        return _encode_mlc(model, mlc, device)
    if arch.startswith("txcdr_t"):
        T = meta["T"]
        return _encode_txcdr(model, anchor, li, T, device)
    if arch.startswith("stacked_t"):
        T = meta["T"]
        return _encode_stacked_last(model, anchor, li, T, device)
    if arch == "matryoshka_t5":
        T = meta["T"]
        return _encode_matryoshka(model, anchor, li, T, device)
    if arch == "shared_perpos_t5":
        T = meta["T"]
        X = _last_token(anchor, li)
        with torch.no_grad():
            z = model.encode(
                torch.from_numpy(X).to(device).unsqueeze(1)
            )[:, 0, :]
        return z.cpu().numpy()
    raise ValueError(f"Unknown arch {arch}")


def run_probing(
    run_ids=None, task_names=None, k_values=None,
    include_baselines: bool = True,
):
    k_values = k_values or K_VALUES
    device = torch.device("cuda")
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    task_dirs = [d for d in sorted(PROBE_CACHE.iterdir()) if d.is_dir()]
    if task_names:
        task_dirs = [d for d in task_dirs if d.name in task_names]
    # Only load dirs that have both caches
    task_dirs = [
        d for d in task_dirs
        if (d / "acts_anchor.npz").exists() and (d / "acts_mlc.npz").exists()
    ]
    task_cache = {d.name: _load_task_cache(d) for d in task_dirs}
    print(f"Probing {len(task_cache)} tasks")

    if include_baselines:
        with OUT_JSONL.open("a") as out_f:
            for task_name, tc in task_cache.items():
                ytr, yte = tc["train_labels"], tc["test_labels"]
                dkey = tc["meta"]["dataset_key"]

                X_last_tr = _last_token(tc["anchor_train"], tc["train_last_idx"])
                X_last_te = _last_token(tc["anchor_test"], tc["test_last_idx"])

                t0 = time.time()
                last_auc = last_token_lr_auc(X_last_tr, ytr, X_last_te, yte)
                out_f.write(json.dumps({
                    "run_id": "BASELINE_last_token_lr",
                    "arch": "baseline_last_token_lr",
                    "task_name": task_name, "dataset_key": dkey,
                    "k_feat": None, "test_auc": last_auc,
                    "n_train": int(ytr.size), "n_test": int(yte.size),
                    "elapsed_s": time.time() - t0,
                }) + "\n")
                print(f"  {task_name} last-LR  auc={last_auc:.4f}")

                t0 = time.time()
                ap_auc = attn_pool_auc(
                    tc["anchor_train"], tc["train_last_idx"], ytr,
                    tc["anchor_test"], tc["test_last_idx"], yte, device,
                )
                out_f.write(json.dumps({
                    "run_id": "BASELINE_attn_pool",
                    "arch": "baseline_attn_pool",
                    "task_name": task_name, "dataset_key": dkey,
                    "k_feat": None, "test_auc": ap_auc,
                    "n_train": int(ytr.size), "n_test": int(yte.size),
                    "elapsed_s": time.time() - t0,
                }) + "\n")
                print(f"  {task_name} attn-pool auc={ap_auc:.4f}")

    with OUT_JSONL.open("a") as out_f:
        for run_id, ckpt_path in _iter_ckpts(run_ids):
            if not ckpt_path.exists():
                print(f"  {run_id}: ckpt missing, skip")
                continue
            try:
                model, arch, meta = _load_model_for_run(run_id, ckpt_path, device)
            except Exception as e:
                print(f"  {run_id}: load FAIL: {e}")
                continue
            print(f"=== {run_id} ({arch}) ===")
            for task_name, tc in task_cache.items():
                try:
                    t0 = time.time()
                    Ztr = _encode_for_probe(model, arch, meta, tc, "train", device)
                    Zte = _encode_for_probe(model, arch, meta, tc, "test", device)
                    for k in k_values:
                        auc = sae_probe_auc(
                            Ztr, tc["train_labels"],
                            Zte, tc["test_labels"], k,
                        )
                        out_f.write(json.dumps({
                            "run_id": run_id, "arch": arch,
                            "task_name": task_name,
                            "dataset_key": tc["meta"]["dataset_key"],
                            "aggregation": "last_position",
                            "k_feat": k, "test_auc": auc,
                            "n_train": int(tc["train_labels"].size),
                            "n_test": int(tc["test_labels"].size),
                            "elapsed_s": time.time() - t0,
                        }) + "\n")
                    print(f"  {task_name}: [{(time.time()-t0):.1f}s]")
                except Exception as e:
                    print(f"  {task_name} FAIL: {e}")
                    out_f.write(json.dumps({
                        "run_id": run_id, "arch": arch,
                        "task_name": task_name, "error": str(e),
                    }) + "\n")
            del model
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-ids", nargs="+", default=None)
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--k-values", nargs="+", type=int, default=K_VALUES)
    ap.add_argument("--skip-baselines", action="store_true")
    args = ap.parse_args()
    run_probing(
        run_ids=args.run_ids, task_names=args.tasks,
        k_values=args.k_values, include_baselines=not args.skip_baselines,
    )


if __name__ == "__main__":
    main()
