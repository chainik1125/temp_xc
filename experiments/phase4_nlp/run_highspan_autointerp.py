"""Autointerp on the high-span (temporal) feature subsets per architecture.

Complements analyze_high_span_features.py. For each architecture, take the
top N high-span features and get Claude Haiku explanations. Also split
TXCDR's high-span set into "universal" (high co-firing with TFA/Stacked) vs
"unique" (low co-firing) subpopulations to test whether the bimodal co-firing
distribution corresponds to a semantic split.

Outputs: results/analysis/high_span_autointerp/<category>/feat_*.json
"""
from __future__ import annotations

import asyncio
import heapq
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/elysium/temp_xc")

# Reuse the scanning and explainer functions from Phase 2
from scripts.run_phase2_autointerp import (
    load_tokenizer,
    decode_window,
    is_content_mask,
    scan_tfa_codes,
    scan_txcdr,
    scan_stacked,
    explain_feature,
    SYSTEM_PROMPT,
    CACHE_DIR, D_IN, D_SAE, T_WIN, K, LAYER,
    N_SCAN_SEQUENCES, N_TOP_EXAMPLES, CLAUDE_CONCURRENCY, CLAUDE_INTER_CALL_DELAY,
)

HS_DIR = "/home/elysium/temp_xc/results/analysis/high_span_comparison"
OUT_DIR = "/home/elysium/temp_xc/results/analysis/high_span_autointerp"
N_PER_CATEGORY = 20


def load_txcdr_cofiring_split() -> tuple[list[int], list[int]]:
    """Split TXCDR top-100 high-span features into 'universal' (high co-firing
    with TFA-pred) vs 'unique' (low co-firing).

    Recomputes the co-firing needed for the split since we need the per-feature
    max co-firing value, not just the aggregate.
    """
    # Load the top-span indices from the high-span analysis
    summary = json.load(open(f"{HS_DIR}/summary.json"))
    top_txcdr = summary["feature_indices"]["top_span_txcdr"]
    top_tfa = summary["feature_indices"]["top_span_tfa_pred"]
    top_stacked = summary["feature_indices"]["top_span_stacked"]

    # Re-compute co-firing: for each top_txcdr feature, max |corr| with any top_tfa/top_stacked
    # Load pre-computed activation arrays would be ideal; we recompute lightweight binary firing here
    arr = np.load(
        f"{CACHE_DIR}/{LAYER}.npy", mmap_mode="r",
    )
    eval_x = torch.from_numpy(np.array(arr[-500:])).float()

    # Get per-feature binary firing arrays for the three feature subsets
    print("  Recomputing activations for co-firing split...")
    # TXCDR
    state = torch.load(
        f"/home/elysium/temp_xc/results/nlp_sweep/gemma/ckpts/"
        f"crosscoder__gemma-2-2b-it__fineweb__{LAYER}__k{K}__seed42.pt",
        map_location="cpu", weights_only=True,
    )
    W_enc_t = state["W_enc"].cuda(); b_enc_t = state["b_enc"].cuda()
    k_total = K * T_WIN
    N = eval_x.shape[0]

    top_txcdr_tensor = torch.tensor(top_txcdr, device="cuda")
    top_tfa_tensor = torch.tensor(top_tfa, device="cuda")
    top_stacked_tensor = torch.tensor(top_stacked, device="cuda")

    def txcdr_fire():
        out = np.zeros((N, 128, len(top_txcdr)), dtype=bool)
        for s in range(0, N, 16):
            x = eval_x[s : s + 16].cuda()
            B = x.shape[0]
            for w in range(128 - T_WIN + 1):
                window = x[:, w : w + T_WIN, :]
                pre = torch.einsum("btd,tds->bs", window, W_enc_t) + b_enc_t
                _, idx = pre.topk(k_total, dim=-1)
                mask = torch.zeros_like(pre, dtype=torch.bool)
                mask.scatter_(-1, idx, True)
                active = mask & (pre > 0)
                active_top = active[:, top_txcdr_tensor].cpu().numpy()
                out[s : s + B, w, :] = active_top
            for w in range(128 - T_WIN + 1, 128):
                out[s : s + B, w, :] = out[s : s + B, 128 - T_WIN, :]
        return out

    def tfa_fire():
        from src.architectures._tfa_module import TemporalSAE
        st = torch.load(
            f"/home/elysium/temp_xc/results/nlp_sweep/gemma/ckpts/"
            f"tfa_pos__gemma-2-2b-it__fineweb__{LAYER}__k{K}__seed42.pt",
            map_location="cpu", weights_only=True,
        )
        model = TemporalSAE(
            dimin=D_IN, width=D_SAE, n_heads=4, sae_diff_type="topk", kval_topk=K,
            tied_weights=True, n_attn_layers=1, bottleneck_factor=8,
            use_pos_encoding=True, max_seq_len=512,
        ).cuda()
        model.load_state_dict(st); model.eval()
        scale = math.sqrt(D_IN) / eval_x[:16].norm(dim=-1).mean().item()
        out = np.zeros((N, 128, len(top_tfa)), dtype=bool)
        for s in range(0, N, 16):
            x = eval_x[s : s + 16].cuda() * scale
            B = x.shape[0]
            _, inter = model(x)
            out[s : s + B] = (inter["pred_codes"][:, :, top_tfa_tensor].abs() > 1e-6).cpu().numpy()
        return out

    def stacked_fire():
        st = torch.load(
            f"/home/elysium/temp_xc/results/nlp_sweep/gemma/ckpts/"
            f"stacked_sae__gemma-2-2b-it__fineweb__{LAYER}__k{K}__seed42.pt",
            map_location="cpu", weights_only=True,
        )
        W_enc_s = st["saes.0.W_enc"].cuda()
        b_enc_s = st["saes.0.b_enc"].cuda()
        b_dec_s = st["saes.0.b_dec"].cuda()
        out = np.zeros((N, 128, len(top_stacked)), dtype=bool)
        for s in range(0, N, 16):
            x = eval_x[s : s + 16].cuda()
            B = x.shape[0]
            pre = (x - b_dec_s) @ W_enc_s.T + b_enc_s
            _, idx = pre.topk(K, dim=-1)
            mask = torch.zeros_like(pre, dtype=torch.bool)
            mask.scatter_(-1, idx, True)
            active = mask & (pre > 0)
            out[s : s + B] = active[:, :, top_stacked_tensor].cpu().numpy()
        return out

    print("    txcdr firing...")
    f_txcdr = txcdr_fire()
    print("    tfa firing...")
    f_tfa = tfa_fire()
    print("    stacked firing...")
    f_stacked = stacked_fire()

    # Compute per-TXCDR-feature max |corr| with any TFA-pred OR Stacked top-span feature
    def cofire_max(a_flat: np.ndarray, b_flat: np.ndarray) -> np.ndarray:
        """a: (pos, nA), b: (pos, nB) binary arrays. Returns max |corr| per col of a."""
        a = a_flat - a_flat.mean(axis=0)
        b = b_flat - b_flat.mean(axis=0)
        na = np.linalg.norm(a, axis=0) + 1e-8
        nb = np.linalg.norm(b, axis=0) + 1e-8
        corr = (a.T @ b) / (na[:, None] * nb[None, :])
        return np.abs(corr).max(axis=1)

    flat_txcdr = f_txcdr.reshape(-1, f_txcdr.shape[-1]).astype(np.float32)
    flat_tfa = f_tfa.reshape(-1, f_tfa.shape[-1]).astype(np.float32)
    flat_stacked = f_stacked.reshape(-1, f_stacked.shape[-1]).astype(np.float32)

    max_corr_tfa = cofire_max(flat_txcdr, flat_tfa)
    max_corr_stacked = cofire_max(flat_txcdr, flat_stacked)
    max_corr_best = np.maximum(max_corr_tfa, max_corr_stacked)

    # Split: bimodal with a valley around 0.15. Threshold 0.15:
    # - universal = max_corr_best > 0.15  (has a decent partner in TFA or Stacked)
    # - unique    = max_corr_best <= 0.15 (no partner)
    threshold = 0.15
    universal_mask = max_corr_best > threshold
    unique_mask = ~universal_mask

    print(f"  TXCDR top-100 split (threshold={threshold}):")
    print(f"    Universal: {universal_mask.sum()} features (max_corr > {threshold})")
    print(f"    Unique:    {unique_mask.sum()} features (max_corr <= {threshold})")

    # Return feature IDXs sorted by max_corr_best (universal = highest, unique = lowest)
    universal_idx = np.array(top_txcdr)[np.argsort(-max_corr_best)][:sum(universal_mask)]
    unique_idx = np.array(top_txcdr)[np.argsort(max_corr_best)][:sum(unique_mask)]
    return universal_idx.tolist()[:N_PER_CATEGORY], unique_idx.tolist()[:N_PER_CATEGORY]


async def process_category(
    category_name: str, feature_indices: list[int],
    heaps: dict[int, list], token_ids: np.ndarray, tokenizer, client,
):
    out_dir = Path(OUT_DIR) / category_name
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = {}
    for fi in feature_indices:
        if fi not in heaps or not heaps[fi]:
            continue
        entries = sorted(heaps[fi], reverse=True)
        examples = []
        for val, seq_idx, peak_pos in entries:
            text = decode_window(token_ids, seq_idx, peak_pos, tokenizer)
            examples.append({
                "activation": val,
                "chain_idx": seq_idx,
                "token_pos": peak_pos,
                "text": text,
            })
        metadata[fi] = examples

    print(f"\n  {category_name}: calling Claude for {len(metadata)} features...")
    sem = asyncio.Semaphore(CLAUDE_CONCURRENCY)

    async def one_feature(fi, examples):
        async with sem:
            await asyncio.sleep(CLAUDE_INTER_CALL_DELAY / CLAUDE_CONCURRENCY)
            result = await explain_feature(client, examples, "claude-haiku-4-5")
            record = {
                "category": category_name, "feat_idx": fi,
                "examples": examples, **result,
            }
            (out_dir / f"feat_{fi:05d}.json").write_text(json.dumps(record, indent=2))
            print(f"    f{fi:5d} [{result['confidence']:6s}]: {result['explanation'][:95]}")
            return record

    records = await asyncio.gather(*(one_feature(fi, ex) for fi, ex in metadata.items()))
    (out_dir / "_summary.json").write_text(json.dumps([
        {"feat_idx": r["feat_idx"], "explanation": r["explanation"],
         "confidence": r["confidence"]} for r in records
    ], indent=2))


async def amain():
    import anthropic
    client = anthropic.AsyncAnthropic()

    # Load top-span feature lists from Phase-1 high-span analysis
    summary = json.load(open(f"{HS_DIR}/summary.json"))
    top_txcdr = summary["feature_indices"]["top_span_txcdr"][:N_PER_CATEGORY]
    top_tfa = summary["feature_indices"]["top_span_tfa_pred"][:N_PER_CATEGORY]
    top_stacked = summary["feature_indices"]["top_span_stacked"][:N_PER_CATEGORY]

    # TXCDR co-firing split
    universal_txcdr, unique_txcdr = load_txcdr_cofiring_split()

    print("\nFeature counts:")
    print(f"  Stacked top-span:       {len(top_stacked)}")
    print(f"  TXCDR top-span:         {len(top_txcdr)}")
    print(f"  TFA-pred top-span:      {len(top_tfa)}")
    print(f"  TXCDR universal (split): {len(universal_txcdr)}")
    print(f"  TXCDR unique (split):   {len(unique_txcdr)}")

    # Load eval data
    arr = np.load(f"{CACHE_DIR}/{LAYER}.npy", mmap_mode="r")
    tok_ids = np.load(f"{CACHE_DIR}/token_ids.npy", mmap_mode="r")
    eval_x = torch.from_numpy(np.array(arr[-N_SCAN_SEQUENCES:])).float()
    eval_tokens = np.array(tok_ids[-N_SCAN_SEQUENCES:])
    tokenizer = load_tokenizer()

    all_txcdr_fs = list(set(top_txcdr + universal_txcdr + unique_txcdr))

    print("\n[1/3] Scan TFA-pred top-span features...")
    heaps_tfa = scan_tfa_codes(top_tfa, eval_x, eval_tokens, use_pred=True)
    torch.cuda.empty_cache()

    print("\n[2/3] Scan TXCDR (union: top-span + universal + unique)...")
    heaps_txcdr = scan_txcdr(all_txcdr_fs, eval_x, eval_tokens)
    torch.cuda.empty_cache()

    print("\n[3/3] Scan Stacked top-span features...")
    heaps_stacked = scan_stacked(top_stacked, eval_x, eval_tokens)
    torch.cuda.empty_cache()

    print("\n=== Claude explanations ===")
    await process_category("stacked_highspan", top_stacked, heaps_stacked,
                            eval_tokens, tokenizer, client)
    await process_category("txcdr_highspan", top_txcdr, heaps_txcdr,
                            eval_tokens, tokenizer, client)
    await process_category("tfa_highspan", top_tfa, heaps_tfa,
                            eval_tokens, tokenizer, client)
    await process_category("txcdr_universal", universal_txcdr, heaps_txcdr,
                            eval_tokens, tokenizer, client)
    await process_category("txcdr_unique_temporal", unique_txcdr, heaps_txcdr,
                            eval_tokens, tokenizer, client)
    print("\nDONE")


if __name__ == "__main__":
    asyncio.run(amain())
