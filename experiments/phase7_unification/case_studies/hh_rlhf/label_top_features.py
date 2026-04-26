"""Autointerp labels for the top-K HH-RLHF chosen-rejected diff features.

For each Stage 1 / Stage 2 arch already decomposed by `decompose_hh_rlhf.py`:

  1. Load the per-arch `feature_stats.json` for `top_k_indices`.
  2. Re-encode chosen + rejected HH-RLHF activations through the arch and
     slice to the top-K columns (memory-efficient — `(N, S, K)` fp16 ~ 25 MB
     per side per arch). Window archs attribute via right-edge.
  3. For each top-K feature, find its top-N max-activating positions
     (restricted to response tokens — non-response positions are masked
     out so the autointerp captures the chosen-vs-rejected differential
     signal, not generic prompt activations).
  4. Decode a 2*ctx_window+1 token window around each firing position via
     the Gemma tokenizer; mark the firing token with `**token**`.
  5. Send the N contexts per feature to Claude Haiku 4.5 in a single
     prompt (Bills-et-al-style). Concurrent across features via
     ThreadPoolExecutor (default 16 workers).
  6. Save `top_features.json` per arch with feature_idx, rank, |diff|,
     length_pearson_r, contexts, and the autointerp label.

Cost: 4 archs * 20 top features * 1 Haiku call = 80 calls, ~$0.25 total.

Run:
    .venv/bin/python -m experiments.phase7_unification.case_studies.hh_rlhf.label_top_features
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("TQDM_DISABLE", "1")

from experiments.phase7_unification._paths import OUT_DIR, banner
from experiments.phase7_unification.run_probing_phase7 import _load_phase7_model
from experiments.phase7_unification.case_studies._paths import (
    HH_RLHF_CACHE_DIR, CASE_STUDIES_DIR, STAGE_1_ARCHS, SUBJECT_MODEL,
)
from experiments.phase7_unification.case_studies._arch_utils import (
    encode_per_position, window_T, _d_sae_of, MLC_CLASSES,
)


HAIKU_MODEL = "claude-haiku-4-5-20251001"


BILLS_PROMPT = """\
You are an interpretability researcher. Below are text excerpts where a \
single sparse-autoencoder feature activates strongly on the bolded token \
(wrapped in **double asterisks**). Identify the common concept, pattern, or \
syntactic role across the activating contexts.

CRITICAL: Respond with EXACTLY one short concept label (3–12 words). \
NO preamble, NO analysis, NO markdown, NO quotes. Just the label.

Examples of valid responses:
  medical case reports or clinical descriptions
  transition words and discourse connectives
  court case citations and legal references
  expressions of refusal or apology
  body parts (especially anatomical references)
  numerical quantities or measurements

Excerpts:
{excerpts}

Label:"""


def _clean_label(text: str) -> str:
    """Strip Haiku's occasional preamble / markdown / quotes / leading bullets."""
    import re
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading bullet/number/dash markers.
        line = re.sub(r'^([-*\d.)\s]+)', '', line).strip()
        # Strip outer quotes/asterisks.
        line = line.strip('"').strip("'").strip("`")
        line = re.sub(r'^\*+|\*+$', '', line).strip()
        # Skip preamble lines that aren't the actual label.
        low = line.lower()
        if low.startswith(("here", "the common", "this feature", "based on",
                           "i need to", "looking at", "let me", "okay",
                           "concept label", "label:")):
            continue
        if len(line) > 200:        # too long to be a label, probably reasoning
            continue
        return line
    return text.strip()[:200]


def _extract_top_contexts_for_feature(
    chosen_topK: np.ndarray,        # (N, S, K) fp16
    rejected_topK: np.ndarray,
    chosen_input_ids: np.ndarray,    # (N, S) int32
    rejected_input_ids: np.ndarray,
    chosen_resp_mask: np.ndarray,    # (N, S) bool
    rejected_resp_mask: np.ndarray,
    k_col: int,
    top_n: int,
    ctx_window: int,
    tokenizer,
) -> list[dict]:
    """For column k_col across (chosen ∪ rejected), return top_n
    activating positions as text excerpts with **firing_token** marked.

    Restricts firing positions to response tokens only — non-response
    tokens (prompt + pad) are masked with -inf so they never appear
    in the top.
    """
    N, S = chosen_topK.shape[:2]
    cs = chosen_topK[:, :, k_col].astype(np.float32)
    rs = rejected_topK[:, :, k_col].astype(np.float32)
    cs[~chosen_resp_mask] = -np.inf
    rs[~rejected_resp_mask] = -np.inf

    # Combine into (2, N, S) and find top_n flat positions.
    combined = np.stack([cs, rs], axis=0)
    flat = combined.reshape(-1)
    # Filter out -inf before sorting (might give empty list if no valid)
    finite = np.isfinite(flat)
    if not finite.any():
        return []
    cand_idx = np.where(finite)[0]
    cand_vals = flat[cand_idx]
    sorted_within = np.argsort(-cand_vals)[:top_n]
    top_flat = cand_idx[sorted_within]
    top_acts = flat[top_flat]

    contexts = []
    for fi, act in zip(top_flat.tolist(), top_acts.tolist()):
        side = fi // (N * S)
        ex_idx = (fi % (N * S)) // S
        tok_idx = fi % S
        ids = chosen_input_ids[ex_idx] if side == 0 else rejected_input_ids[ex_idx]
        lo = max(0, tok_idx - ctx_window)
        hi = min(S, tok_idx + ctx_window + 1)
        before = tokenizer.decode(ids[lo:tok_idx], skip_special_tokens=True)
        firing = tokenizer.decode([int(ids[tok_idx])], skip_special_tokens=True)
        after = tokenizer.decode(ids[tok_idx + 1:hi], skip_special_tokens=True)
        # Collapse runs of whitespace introduced by skip_special_tokens
        ctx = (before + f"**{firing}**" + after).strip()
        contexts.append({
            "side": "chosen" if side == 0 else "rejected",
            "ex_idx": int(ex_idx),
            "tok_idx": int(tok_idx),
            "activation": float(act),
            "context": ctx,
        })
    return contexts


def _label_with_haiku(client, contexts: list[dict], model: str = HAIKU_MODEL) -> str:
    excerpts = "\n".join(
        f"  {i + 1}. \"{c['context']}\"" for i, c in enumerate(contexts)
    )
    prompt = BILLS_PROMPT.format(excerpts=excerpts)
    msg = client.messages.create(
        model=model,
        max_tokens=200,           # leave room if Haiku starts with preamble
        messages=[{"role": "user", "content": prompt}],
    )
    return _clean_label(msg.content[0].text)


def label_one_arch(
    arch_id: str,
    *,
    top_k: int,
    top_n_per_feature: int,
    ctx_window: int,
    batch_size: int,
    n_workers: int,
) -> dict | None:
    log_path = OUT_DIR / "training_logs" / f"{arch_id}__seed42.json"
    ckpt_path = OUT_DIR / "ckpts" / f"{arch_id}__seed42.pt"
    feature_stats_path = CASE_STUDIES_DIR / "hh_rlhf" / arch_id / "feature_stats.json"
    if not feature_stats_path.exists():
        print(f"  [skip] {arch_id}: feature_stats.json missing — run decompose_hh_rlhf first")
        return None
    out_path = CASE_STUDIES_DIR / "hh_rlhf" / arch_id / "top_features.json"

    meta = json.loads(log_path.read_text())
    feature_stats = json.loads(feature_stats_path.read_text())
    src_class = meta["src_class"]
    if src_class in MLC_CLASSES:
        print(f"  [skip] {arch_id}: MLC arch needs multilayer cache (not built here)")
        return None

    top_k_indices = list(feature_stats["top_k_indices"][:top_k])
    K = len(top_k_indices)
    print(f"  src_class={src_class}  top_k={K}  k_pos={meta.get('k_pos')}")

    device = torch.device("cuda")
    model, _ = _load_phase7_model(meta, ckpt_path, device)
    T_window = window_T(model, src_class, meta)

    chosen_npz = np.load(HH_RLHF_CACHE_DIR / "chosen.npz")
    rejected_npz = np.load(HH_RLHF_CACHE_DIR / "rejected.npz")
    chosen_acts = chosen_npz["acts"]
    rejected_acts = rejected_npz["acts"]
    chosen_input_ids = chosen_npz["input_ids"]
    rejected_input_ids = rejected_npz["input_ids"]
    chosen_resp_mask = chosen_npz["response_mask"]
    rejected_resp_mask = rejected_npz["response_mask"]
    N, S = chosen_acts.shape[:2]

    top_idx_t = torch.as_tensor(top_k_indices, dtype=torch.long, device=device)
    chosen_topK = np.zeros((N, S, K), dtype=np.float16)
    rejected_topK = np.zeros((N, S, K), dtype=np.float16)
    t0 = time.time()
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        for src, dst in [(chosen_acts, chosen_topK), (rejected_acts, rejected_topK)]:
            x = torch.from_numpy(src[start:end]).float().to(device)
            z = encode_per_position(model, src_class, x, T=T_window)
            z_topk = z.index_select(-1, top_idx_t)               # (B, S, K)
            dst[start:end] = z_topk.to(torch.float16).cpu().numpy()
            del x, z, z_topk
    del model
    torch.cuda.empty_cache()
    gc.collect()
    print(f"  re-encode + slice top-K done in {time.time() - t0:.1f}s")

    print(f"  decoding contexts via {SUBJECT_MODEL} tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(SUBJECT_MODEL)
    feature_contexts: list[dict] = []
    for ki, fj in enumerate(top_k_indices):
        ctxs = _extract_top_contexts_for_feature(
            chosen_topK, rejected_topK,
            chosen_input_ids, rejected_input_ids,
            chosen_resp_mask, rejected_resp_mask,
            ki, top_n_per_feature, ctx_window, tokenizer,
        )
        feature_contexts.append({
            "feature_idx": int(fj),
            "rank": int(ki),
            "diff": float(feature_stats["top_k_diff"][ki]),
            "mean_chosen": float(feature_stats["top_k_mean_chosen"][ki]),
            "mean_rejected": float(feature_stats["top_k_mean_rejected"][ki]),
            "length_pearson_r": float(feature_stats["top_k_length_pearson_r"][ki]),
            "length_pearson_p": float(feature_stats["top_k_length_pearson_p"][ki]),
            "contexts": ctxs,
        })

    print(f"  calling Haiku ({HAIKU_MODEL}) for {len(feature_contexts)} features in parallel ({n_workers} workers)...")
    from anthropic import Anthropic
    api_key = Path("/workspace/.tokens/anthropic_key").read_text().strip()
    # max_retries=12 + small worker pool = robust against 50-req/min bursts.
    # SDK already does exponential backoff with jitter on 429.
    client = Anthropic(api_key=api_key, max_retries=12)

    def _label(item: dict) -> dict:
        try:
            label = _label_with_haiku(client, item["contexts"])
        except Exception as e:
            # Truncate the error message — RateLimitError sometimes carries
            # the entire 1-KB rate-limit-doc URL which clutters the JSON.
            err_msg = str(e)[:200]
            label = f"[ERROR] {type(e).__name__}: {err_msg}"
        return {**item, "label": label}

    labelled = [None] * len(feature_contexts)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_label, item): i for i, item in enumerate(feature_contexts)
        }
        n_done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            labelled[i] = fut.result()
            n_done += 1
            if n_done % 5 == 0 or n_done == len(feature_contexts):
                print(f"    {n_done}/{len(feature_contexts)} features labelled")

    out = {"arch_id": arch_id, "src_class": src_class, "top_k": K, "features": labelled}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  saved {out_path}")

    # Print top-10 with labels.
    print(f"  top-10 features (sorted by |diff|):")
    print(f"    {'rk':>2s} {'feat':>6s} {'diff':>8s} {'r_len':>7s} | label")
    for r in labelled[:10]:
        print(
            f"    {r['rank']:2d} {r['feature_idx']:6d} {r['diff']:+8.3f} "
            f"{r['length_pearson_r']:+7.3f} | {r['label']}"
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=list(STAGE_1_ARCHS))
    ap.add_argument("--top-k", type=int, default=20,
                    help="top-K features to label per arch (paper Table 8 uses 15)")
    ap.add_argument("--top-n-per-feature", type=int, default=5)
    ap.add_argument("--ctx-window", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-workers", type=int, default=5,
                    help="Concurrent Haiku calls (50 req/min API limit; "
                         "5 workers + SDK retry handles bursts)")
    args = ap.parse_args()
    banner(__file__)

    for arch_id in args.archs:
        print(f"\n=== {arch_id} ===")
        label_one_arch(
            arch_id,
            top_k=args.top_k,
            top_n_per_feature=args.top_n_per_feature,
            ctx_window=args.ctx_window,
            batch_size=args.batch_size,
            n_workers=args.n_workers,
        )


if __name__ == "__main__":
    main()
