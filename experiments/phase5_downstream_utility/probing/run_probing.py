"""Sparse-probing runner for Phase 5.

For each (trained_checkpoint, task), encode the cached per-task
activations with the checkpoint, select top-k latents by class
separation on the train split only, fit sklearn logistic regression,
and record test AUC on a held-out test set.

Writes one JSONL row per (checkpoint, task, k_feat) to
`results/probing_results.jsonl`. Also runs baselines:
    - Last-token L2 logistic regression on raw L13 activations.
    - Attention-pooled (Eq. 2 of Kantamneni et al.) over the tail-32 tokens.
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

LAYERS = [11, 12, 13, 14, 15]
ANCHOR_IDX = 2  # index of L13 within LAYERS
K_VALUES = [1, 2, 5, 20]


def _load_task_cache(task_dir: Path) -> dict[str, Any]:
    npz = np.load(task_dir / "acts_tail.npz")
    meta = json.loads((task_dir / "meta.json").read_text())
    return {
        "train_acts": npz["train_acts"].astype(np.float32),
        "test_acts": npz["test_acts"].astype(np.float32),
        "train_last_idx": npz["train_last_idx"],
        "test_last_idx": npz["test_last_idx"],
        "train_labels": npz["train_labels"],
        "test_labels": npz["test_labels"],
        "meta": meta,
    }


def _last_token(acts, last_idx, layer_idx):
    N = acts.shape[0]
    out = np.empty((N, acts.shape[-1]), dtype=np.float32)
    for i in range(N):
        out[i] = acts[i, int(last_idx[i]), layer_idx]
    return out


def _last_token_multilayer(acts, last_idx):
    N = acts.shape[0]
    out = np.empty((N, acts.shape[2], acts.shape[3]), dtype=np.float32)
    for i in range(N):
        out[i] = acts[i, int(last_idx[i])]
    return out


def _window_at_last(acts, last_idx, layer_idx, T):
    N = acts.shape[0]
    d = acts.shape[-1]
    out = np.empty((N, T, d), dtype=np.float32)
    for i in range(N):
        li = int(last_idx[i])
        start = max(0, li - T + 1)
        win = acts[i, start:li + 1, layer_idx]
        if win.shape[0] < T:
            pad = np.broadcast_to(win[0:1], (T - win.shape[0], d))
            win = np.concatenate([pad, win], axis=0)
        out[i] = win
    return out


def _encode_topk(model, acts, last_idx, device):
    X = _last_token(acts, last_idx, ANCHOR_IDX)
    with torch.no_grad():
        z = model.encode(torch.from_numpy(X).to(device))
    return z.cpu().numpy()


def _encode_mlc(model, acts, last_idx, device):
    X = _last_token_multilayer(acts, last_idx)
    with torch.no_grad():
        z = model.encode(torch.from_numpy(X).to(device))
    return z.cpu().numpy()


def _encode_txcdr(model, acts, last_idx, T, device):
    X = _window_at_last(acts, last_idx, ANCHOR_IDX, T)
    with torch.no_grad():
        z = model.encode(torch.from_numpy(X).to(device))
    return z.cpu().numpy()


def _encode_stacked_last(model, acts, last_idx, T, device):
    X = _last_token(acts, last_idx, ANCHOR_IDX)
    with torch.no_grad():
        z = model.saes[-1].encode(torch.from_numpy(X).to(device))
    return z.cpu().numpy()


def _encode_sharedperpos_last(model, acts, last_idx, T, device):
    X = _last_token(acts, last_idx, ANCHOR_IDX)
    Xt = torch.from_numpy(X).to(device)
    with torch.no_grad():
        z = model.encode(Xt.unsqueeze(1))[:, 0, :]
    return z.cpu().numpy()


def _encode_tfa(model, acts, last_idx, device):
    N = acts.shape[0]
    d = acts.shape[-1]
    X = acts[:, :, ANCHOR_IDX, :]
    sample = X[:min(256, N)]
    mean_norm = np.linalg.norm(sample.reshape(-1, d), axis=-1).mean()
    scale = (d ** 0.5) / max(mean_norm, 1e-8)
    X_scaled = X * scale
    width = getattr(model, "width", 18432)
    Z_out = np.empty((N, width), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, N, 32):
            end = min(start + 32, N)
            xb = torch.from_numpy(X_scaled[start:end]).to(device)
            _, inter = model(xb)
            novel = inter["novel_codes"]
            li = last_idx[start:end]
            for j in range(end - start):
                Z_out[start + j] = novel[j, int(li[j])].cpu().numpy()
    return Z_out


def _encode_matryoshka(model, acts, last_idx, T, device):
    X = _window_at_last(acts, last_idx, ANCHOR_IDX, T)
    with torch.no_grad():
        z = model.encode(torch.from_numpy(X).to(device))
    return z.cpu().numpy()


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
    X_tr = acts_tr[:, :, ANCHOR_IDX, :]
    X_te = acts_te[:, :, ANCHOR_IDX, :]
    mask_tr = np.arange(X_tr.shape[1])[None, :] <= last_tr[:, None]
    mask_te = np.arange(X_te.shape[1])[None, :] <= last_te[:, None]
    try:
        out = train_attn_probe(
            X_train=torch.from_numpy(X_tr).float(),
            y_train=torch.from_numpy(y_tr),
            X_test=torch.from_numpy(X_te).float(),
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
    elif arch == "shared_perpos_t5":
        from experiments.phase5_downstream_utility.train_primary_archs import (
            SharedPerPositionSAE,
        )
        T = meta["T"]
        model = SharedPerPositionSAE(d_in, d_sae, T, k=meta["k_pos"]).to(device)
    elif arch in ("tfa", "tfa_pos"):
        from src.architectures._tfa_module import TemporalSAE
        use_pos = (arch == "tfa_pos")
        model = TemporalSAE(
            dimin=d_in, width=d_sae, n_heads=4,
            sae_diff_type="topk", kval_topk=meta["k_pos"],
            tied_weights=True, n_attn_layers=1,
            bottleneck_factor=4, use_pos_encoding=use_pos,
        ).to(device)
    elif arch == "matryoshka_t5":
        from src.architectures.matryoshka_txcdr import PositionMatryoshkaTXCDR
        T = meta["T"]
        k_eff = meta["k_win"] or (meta["k_pos"] * T)
        model = PositionMatryoshkaTXCDR(d_in, d_sae, T, k_eff).to(device)
    else:
        raise ValueError(f"Unknown arch {arch}")
    # Cast the loaded (possibly fp16) state_dict to the module's dtype (fp32)
    # explicitly to avoid silent precision drops in Matryoshka's mean() calls.
    cast_state = {
        k: v.to(torch.float32) if v.dtype == torch.float16 else v
        for k, v in state["state_dict"].items()
    }
    model.load_state_dict(cast_state)
    model.eval()
    return model, arch, meta


def _encode_for_probe(
    model, arch, meta,
    train_acts, train_last_idx,
    test_acts, test_last_idx, device,
):
    if arch == "topk_sae":
        return (_encode_topk(model, train_acts, train_last_idx, device),
                _encode_topk(model, test_acts, test_last_idx, device))
    if arch == "mlc":
        return (_encode_mlc(model, train_acts, train_last_idx, device),
                _encode_mlc(model, test_acts, test_last_idx, device))
    if arch.startswith("txcdr_t"):
        T = meta["T"]
        return (_encode_txcdr(model, train_acts, train_last_idx, T, device),
                _encode_txcdr(model, test_acts, test_last_idx, T, device))
    if arch.startswith("stacked_t"):
        T = meta["T"]
        return (_encode_stacked_last(model, train_acts, train_last_idx, T, device),
                _encode_stacked_last(model, test_acts, test_last_idx, T, device))
    if arch == "shared_perpos_t5":
        T = meta["T"]
        return (_encode_sharedperpos_last(model, train_acts, train_last_idx, T, device),
                _encode_sharedperpos_last(model, test_acts, test_last_idx, T, device))
    if arch in ("tfa", "tfa_pos"):
        return (_encode_tfa(model, train_acts, train_last_idx, device),
                _encode_tfa(model, test_acts, test_last_idx, device))
    if arch == "matryoshka_t5":
        T = meta["T"]
        return (_encode_matryoshka(model, train_acts, train_last_idx, T, device),
                _encode_matryoshka(model, test_acts, test_last_idx, T, device))
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
    task_cache = {d.name: _load_task_cache(d) for d in task_dirs}
    print(f"Probing {len(task_cache)} tasks")

    if include_baselines:
        with OUT_JSONL.open("a") as out_f:
            for task_name, tc in task_cache.items():
                ytr, yte = tc["train_labels"], tc["test_labels"]
                dkey = tc["meta"]["dataset_key"]

                X_last_tr = _last_token(tc["train_acts"], tc["train_last_idx"], ANCHOR_IDX)
                X_last_te = _last_token(tc["test_acts"], tc["test_last_idx"], ANCHOR_IDX)

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
                    tc["train_acts"], tc["train_last_idx"], ytr,
                    tc["test_acts"], tc["test_last_idx"], yte, device,
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
                    Ztr, Zte = _encode_for_probe(
                        model, arch, meta,
                        tc["train_acts"], tc["train_last_idx"],
                        tc["test_acts"], tc["test_last_idx"], device,
                    )
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
