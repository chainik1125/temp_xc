"""Phase 4 — EM depth-confound check on the § 5.3 medical organism.

The § 5.3 shuffle_gap ≈ 0 verdict exists at ONE layer (L15/28, Qwen-7B
medical). This phase sweeps the per-token vs window ceilings for the EM
label across depth, on the SAME substrate the verdict used:

- Organism: `Qwen/Qwen2.5-7B-Instruct` + LoRA
  `andyrdt/Qwen2.5-7B-Instruct_bad-medical`, merged
  (origin/final `load_subject_with_lora` convention).
- Cohort: the canonical medical detection cell's stage-4 judge outputs
  (txc_base seed 42, train_key 88a4ddf6819d8057), thinned by the
  detection-3.0.0 balanced-α filter (COHORT_ALPHAS ± 0.5, dedup by
  (feature_id, α, rollout_idx, question)) — code-faithful port of
  `origin/final:purified/experiments/c6_em_detection/run.py`.
- Label: align ≤ 50 (Sonnet judge, stored in the cohort rows).
- Forward: full chat template, truncation 300; assistant-only residuals
  (≤ 100 tokens after the prefix), detection-3.0.0 convention.
- Capture: ALL 29 hidden states (hs0 = embeddings, hs k+1 = resid_post
  of layer k), fp16.
- Probes (frozen problib stack, RECORD § 2): per-token linear vs
  right-edge T=16 window linear + MLP presence, positions p ∈
  {15, 19, 23, …} (stride 4) within each rollout's assistant tokens,
  labels/groups propagated from the parent rollout.
  Split: GroupKFold(4) over the 8 EM prompts (2 held-out prompts per
  fold) — the § 5.3 GroupKFold-by-prompt convention (the by-trace 80/20
  rule is undefined at 8 groups). Metrics: mean test AUC across folds
  (+ per-fold), permutation null (seed 99) per fold on the linear pair.
- Probed depth grid: hs0 + resid_post layers 1, 3, …, 27 (odd stride-2
  grid CONTAINING the paper's L15) + resid_post 0.

Stages (idempotent):  cohort -> cache -> probe
  .venv/bin/python -m experiments.explorations.conversion_depth.phase4_em_depth cache
  .venv/bin/python -m experiments.explorations.conversion_depth.phase4_em_depth probe

Frozen prediction (RECORD § 2 P5): flat g(ℓ) ≈ 0 (within 3 σ_null) at
ALL depths. A nonzero early-layer gap overturns the EM negative's
generality — report either way.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiments.explorations.conversion_depth.problib import (
    fit_probe,
)

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER = "andyrdt/Qwen2.5-7B-Instruct_bad-medical"
RUN_REF = "origin/final:purified/results/runs/c6_88a4ddf6819d8057/judge_outputs.jsonl"

COHORT_ALPHAS = (-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0)
ALPHA_TOL = 0.5
ALIGN_MISALIGNED_THRESHOLD = 50
MAX_SEQ_LEN = 300
MAX_ANSWER_LEN = 100
T = 16
POS_STRIDE = 4
NULL_SEED = 99
N_FOLDS = 4

EM_DIR = Path("/workspace/conv_depth_caches/em_medical")
HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "results" / "phase4_em_depth.json"

N_LAYERS = 28
HS_ALL = list(range(N_LAYERS + 1))
PROBE_HS = [0, 1] + [k + 1 for k in range(1, N_LAYERS, 2)]  # + L15 via hs16


def build_cohort() -> list[dict]:
    EM_DIR.mkdir(parents=True, exist_ok=True)
    raw = EM_DIR / "judge_outputs.jsonl"
    if not raw.exists():
        txt = subprocess.run(["git", "show", RUN_REF], cwd=HERE,
                             capture_output=True, text=True, check=True).stdout
        raw.write_text(txt)
    rows, seen = [], set()
    for line in raw.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(r.get("stage", "")) != "4" or r.get("align") is None:
            continue
        a = r.get("alpha")
        if a is None:
            continue
        if not any(abs(float(a) - ca) < ALPHA_TOL for ca in COHORT_ALPHAS):
            continue
        key = (r.get("feature_id"), float(a), r.get("rollout_idx"),
               r.get("question"))
        if key in seen:
            continue
        seen.add(key)
        rows.append(r)
    print(f"[cohort] {len(rows)} stage-4 balanced-α rollouts, "
          f"misaligned frac "
          f"{np.mean([r['align'] <= ALIGN_MISALIGNED_THRESHOLD for r in rows]):.3f}",
          flush=True)
    return rows


@torch.no_grad()
def build_cache():
    if (EM_DIR / "meta.json").exists():
        print("[cache] hit")
        return
    rows = build_cohort()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    model = PeftModel.from_pretrained(model, ADAPTER)
    model = model.merge_and_unload().eval()
    d_model = int(model.config.hidden_size)

    n = len(rows)
    mms = {k: np.lib.format.open_memmap(
        EM_DIR / f"hs{k}.npy", mode="w+", dtype=np.float16,
        shape=(n, MAX_ANSWER_LEN, d_model)) for k in HS_ALL}
    lens = np.zeros(n, dtype=np.int32)
    labels = np.zeros(n, dtype=np.int64)
    qids = np.zeros(n, dtype=np.int64)
    questions = sorted(set(r["question"] for r in rows))

    t0 = time.time()
    for ri, r in enumerate(rows):
        msgs = [{"role": "user", "content": r.get("question", "")},
                {"role": "assistant", "content": r.get("answer", "")}]
        full_text = tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=False)
        prefix_text = tok.apply_chat_template([msgs[0]], tokenize=False,
                                              add_generation_prompt=True)
        full_ids = tok(full_text, return_tensors="pt", truncation=True,
                       max_length=MAX_SEQ_LEN,
                       add_special_tokens=False)["input_ids"]
        prefix_len = len(tok(prefix_text,
                             add_special_tokens=False)["input_ids"])
        out = model(full_ids.cuda(), output_hidden_states=True,
                    use_cache=False)
        seq_len = int(full_ids.shape[1])
        prefix_len = min(prefix_len, seq_len)
        a, b = prefix_len, min(prefix_len + MAX_ANSWER_LEN, seq_len)
        L = b - a
        lens[ri] = L
        labels[ri] = int(r["align"] <= ALIGN_MISALIGNED_THRESHOLD)
        qids[ri] = questions.index(r["question"])
        if L > 0:
            for k in HS_ALL:
                mms[k][ri, :L] = (out.hidden_states[k][0, a:b]
                                  .to(torch.float16).cpu().numpy())
        if (ri + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  {ri + 1}/{n} ({el:.0f}s, est {el / (ri + 1) * n:.0f}s)",
                  flush=True)
    for m in mms.values():
        m.flush()
    np.save(EM_DIR / "lens.npy", lens)
    np.save(EM_DIR / "labels.npy", labels)
    np.save(EM_DIR / "qids.npy", qids)
    (EM_DIR / "meta.json").write_text(json.dumps({
        "base_model": BASE_MODEL, "adapter": ADAPTER, "run_ref": RUN_REF,
        "n_rollouts": n, "d_model": d_model, "hs_all": HS_ALL,
        "misaligned_frac": float(labels.mean()),
        "questions": questions,
        "wall_seconds": round(time.time() - t0, 1)}, indent=2))
    print(f"[cache] DONE em_medical in {time.time() - t0:.0f}s", flush=True)


def probe():
    meta = json.loads((EM_DIR / "meta.json").read_text())
    lens = np.load(EM_DIR / "lens.npy")
    labels = np.load(EM_DIR / "labels.npy")
    qids = np.load(EM_DIR / "qids.npy")
    n = len(lens)

    # probe rows (shared across all layers): (rollout, p), right-edge T=16
    rows, row_lab, row_q = [], [], []
    for ri in range(n):
        for p in range(T - 1, int(lens[ri]), POS_STRIDE):
            rows.append((ri, p))
            row_lab.append(labels[ri])
            row_q.append(qids[ri])
    rows = np.array(rows, dtype=np.int64)
    row_lab = np.array(row_lab, dtype=np.int64)
    row_q = np.array(row_q, dtype=np.int64)
    print(f"[probe] {len(rows)} rows, pos frac {row_lab.mean():.3f}",
          flush=True)

    folds = [(row_q % N_FOLDS) != f for f in range(N_FOLDS)]  # train masks
    done = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {
        "meta": {"substrate": meta, "T": T, "pos_stride": POS_STRIDE,
                 "n_rows": int(len(rows)), "n_folds": N_FOLDS,
                 "prereg": "RECORD.md § 2 P5 + § 4 protocol note"},
        "cells": {}}

    for k in PROBE_HS:
        key = f"hs{k}"
        if key in done["cells"]:
            continue
        t0 = time.time()
        acts = torch.from_numpy(
            np.ascontiguousarray(np.load(EM_DIR / f"hs{k}.npy")))
        w = torch.from_numpy(rows[:, 0])
        p = torch.from_numpy(rows[:, 1])
        X_tok = acts[w, p]
        nrow, d = X_tok.shape
        X_win = torch.empty((nrow, T, d), dtype=acts.dtype)
        for j in range(T):
            X_win[:, j] = acts[w, p - (T - 1) + j]
        X_win = X_win.reshape(nrow, T * d)
        del acts

        cell = {r: {"auc_folds": [], "balacc_opt_folds": []}
                for r in ["per_token_linear", "window_linear",
                          "per_token_mlp", "window_mlp",
                          "null_per_token_linear", "null_window_linear"]}
        for f, tr_mask in enumerate(folds):
            te_mask = ~tr_mask
            ytr = torch.from_numpy(row_lab[tr_mask])
            yte = torch.from_numpy(row_lab[te_mask])
            g = torch.Generator().manual_seed(NULL_SEED + f)
            ytr_p = ytr[torch.randperm(len(ytr), generator=g)]
            yte_p = yte[torch.randperm(len(yte), generator=g)]
            tm = torch.from_numpy(tr_mask)
            em = torch.from_numpy(te_mask)
            runs = [
                ("per_token_linear", X_tok[tm], ytr, X_tok[em], yte, 0),
                ("window_linear", X_win[tm], ytr, X_win[em], yte, 0),
                ("per_token_mlp", X_tok[tm], ytr, X_tok[em], yte, 512),
                ("window_mlp", X_win[tm], ytr, X_win[em], yte, 512),
                ("null_per_token_linear", X_tok[tm], ytr_p, X_tok[em], yte_p, 0),
                ("null_window_linear", X_win[tm], ytr_p, X_win[em], yte_p, 0),
            ]
            for name, a_, y1, b_, y2, hid in runs:
                r = fit_probe(a_, y1, b_, y2, 2, hidden=hid,
                              class_weight=True)
                cell[name]["auc_folds"].append(r["auc"])
                cell[name]["balacc_opt_folds"].append(r.get("balacc_opt"))
        for name in cell:
            cell[name]["auc"] = float(np.mean(cell[name]["auc_folds"]))
        cell["g_auc"] = (cell["window_linear"]["auc"]
                         - cell["per_token_linear"]["auc"])
        done["cells"][key] = cell
        OUT_JSON.write_text(json.dumps(done, indent=1))
        print(f"[em hs{k:>2}] tok={cell['per_token_linear']['auc']:.3f} "
              f"win={cell['window_linear']['auc']:.3f} "
              f"g={cell['g_auc']:+.3f} "
              f"mlp_win={cell['window_mlp']['auc']:.3f} "
              f"null_win={cell['null_window_linear']['auc']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    print(f"[probe] DONE -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("cohort",):
        build_cohort()
    if stage in ("cache", "all"):
        build_cache()
    if stage in ("probe", "all"):
        probe()
