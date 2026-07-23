"""Phase 5 (stretch) — § 5.1 probing check: gemma-2-2b BASE vs -IT @ L13.

Substrate-audit item 3: the § 5.1 anchor pairs an IT model with a web
corpus (convention straddle). Question (briefing): do the sparse-probing
per-token ceilings on RAW L13 activations materially differ between
gemma-2-2b base and -it? One number per model per probe task, NO
dictionary training.

Protocol (frozen before run; § 5.1-faithful):
- Tasks: the 33/38 SAEBench+CT tasks whose loaders work under
  datasets 4.8.5 (texts pre-dumped by phase5_probe_tasks.py; the 5
  github_code tasks fail on the retired loading-script path — recorded).
- Forward: right-padded tokenization, max_length 128 (probe_cache.py
  convention); resid_post L13 = hidden_states[14]; bf16 model, fp32
  pooled features.
- Pooling: mean over the last min(32, n_real) REAL token positions
  (the Phase-7 S_CACHE=32 left-aligned frame + _encode_pool mean rule).
- Probe: mean_pool_probe convention port — top-k features by
  class-mean |diff| on TRAIN, then L1 logistic (liblinear, C=1.0,
  max_iter=1000, random_state=0), test ROC AUC.
  k_feat ∈ {ALL (=2304, the raw ceiling; PRIMARY), 16 (sparse-readout
  operating point; SECONDARY)}.
- Frozen verdict rule: a task "materially differs" if
  |AUC_base − AUC_it| > 0.05 at k=ALL. Headline: mean/max |Δ|, count of
  material tasks, per-model mean AUC. Frozen prior: few or no material
  differences (the § 5.1 pairing is a convention straddle, not an
  error); if many tasks flip, the paper needs a caveat.

Stages:  cache (GPU) -> probe (CPU)
  .venv/bin/python -m experiments.explorations.conversion_depth.phase5_gemma_probing cache
  .venv/bin/python -m experiments.explorations.conversion_depth.phase5_gemma_probing probe
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

MODELS = {"base": "google/gemma-2-2b", "it": "google/gemma-2-2b-it"}
LAYER_HS = 14                 # resid_post L13 on 26-layer gemma-2-2b
MAX_LEN = 128
S_TAIL = 32
K_SECONDARY = 16
MATERIAL_THR = 0.05

TASK_DIR = Path("/workspace/conv_depth_caches/probe_tasks")
POOL_DIR = Path("/workspace/conv_depth_caches/gemma_probing")
HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "results" / "phase5_gemma_probing.json"


def task_names():
    summary = json.loads((TASK_DIR / "summary.json").read_text())
    return sorted(t for t, v in summary.items()
                  if isinstance(v, dict) and v.get("ok"))


@torch.no_grad()
def build_pooled(tag: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out_dir = POOL_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    names = task_names()
    if all((out_dir / f"{t}.npz").exists() for t in names):
        print(f"[{tag}] pooled cache hit")
        return
    model_id = MODELS[tag]
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    for tname in names:
        opath = out_dir / f"{tname}.npz"
        if opath.exists():
            continue
        t0 = time.time()
        z = np.load(TASK_DIR / f"{tname}.npz", allow_pickle=False)
        pooled = {}
        for split in ["train", "test"]:
            texts = [str(x) for x in z[f"texts_{split}"]]
            feats = np.zeros((len(texts), model.config.hidden_size),
                             dtype=np.float32)
            B = 64
            for i in range(0, len(texts), B):
                enc = tok(texts[i:i + B], return_tensors="pt",
                          padding="max_length", truncation=True,
                          max_length=MAX_LEN)
                enc = {k: v.cuda() for k, v in enc.items()}
                hs = model(**enc, output_hidden_states=True,
                           use_cache=False).hidden_states[LAYER_HS]
                am = enc["attention_mask"]
                for j in range(hs.shape[0]):
                    real = torch.nonzero(am[j]).squeeze(-1)
                    sel = real[-min(S_TAIL, len(real)):]
                    feats[i + j] = (hs[j, sel].float().mean(0)
                                    .cpu().numpy())
            pooled[split] = feats
        np.savez(opath, X_train=pooled["train"], X_test=pooled["test"],
                 y_train=z["y_train"], y_test=z["y_test"])
        print(f"[{tag}] {tname} pooled in {time.time() - t0:.0f}s",
              flush=True)
    del model
    torch.cuda.empty_cache()


def probe_task(X_train, y_train, X_test, y_test, k_feat):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    pos, neg = y_train == 1, y_train == 0
    diff = np.abs(X_train[pos].mean(0) - X_train[neg].mean(0))
    idx = np.argsort(diff)[-k_feat:]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(penalty="l1", solver="liblinear", C=1.0,
                                 max_iter=1000, random_state=0)
        clf.fit(X_train[:, idx], y_train)
    return float(roc_auc_score(y_test, clf.predict_proba(X_test[:, idx])[:, 1]))


def probe():
    names = task_names()
    out = {"meta": {"layer_hs": LAYER_HS, "s_tail": S_TAIL,
                    "k_secondary": K_SECONDARY,
                    "material_thr": MATERIAL_THR,
                    "n_tasks": len(names),
                    "skipped": "github_code_* (5; datasets>=4 loader)"},
           "tasks": {}}
    for tname in names:
        row = {}
        for tag in MODELS:
            z = np.load(POOL_DIR / tag / f"{tname}.npz")
            Xtr, Xte = z["X_train"], z["X_test"]
            ytr = z["y_train"].astype(int)
            yte = z["y_test"].astype(int)
            row[f"{tag}_auc_all"] = probe_task(Xtr, ytr, Xte, yte,
                                               Xtr.shape[1])
            row[f"{tag}_auc_k{K_SECONDARY}"] = probe_task(
                Xtr, ytr, Xte, yte, K_SECONDARY)
        row["delta_all"] = row["base_auc_all"] - row["it_auc_all"]
        row["material"] = bool(abs(row["delta_all"]) > MATERIAL_THR)
        out["tasks"][tname] = row
        print(f"[{tname:>32}] base={row['base_auc_all']:.3f} "
              f"it={row['it_auc_all']:.3f} d={row['delta_all']:+.3f}"
              f"{' MATERIAL' if row['material'] else ''}", flush=True)
    d = np.array([r["delta_all"] for r in out["tasks"].values()])
    out["headline"] = {
        "mean_auc_base": float(np.mean([r["base_auc_all"]
                                        for r in out["tasks"].values()])),
        "mean_auc_it": float(np.mean([r["it_auc_all"]
                                      for r in out["tasks"].values()])),
        "mean_abs_delta": float(np.abs(d).mean()),
        "max_abs_delta": float(np.abs(d).max()),
        "n_material": int(sum(r["material"]
                              for r in out["tasks"].values())),
        "n_tasks": len(d),
    }
    OUT_JSON.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["headline"], indent=1))
    print(f"-> {OUT_JSON}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("cache", "all"):
        for tag in MODELS:
            build_pooled(tag)
    if stage in ("probe", "all"):
        probe()
