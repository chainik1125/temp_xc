"""Phase 2 autointerp: per-architecture feature explanations via Claude Haiku.

V2: excludes padding tokens during scanning (avoids "padding detector" features
dominating the TFA novel-only category) and re-ranks TFA novel features by
non-padding mass.

Runs autointerp on four feature categories:
  - TFA pred-only (top 30 by non-padding pred_mass)
  - TFA novel-only (top 30 by non-padding novel_mass)
  - TXCDR unique (top 30 with lowest cross-arch decoder alignment)
  - Stacked unique (top 30 with lowest cross-arch decoder alignment)
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

CKPT_DIR = "/home/elysium/temp_xc/results/nlp_sweep/gemma/ckpts"
CACHE_DIR = "/home/elysium/temp_xc/data/cached_activations/gemma-2-2b-it/fineweb"
OUT_DIR = "/home/elysium/temp_xc/results/analysis/autointerp"
D_IN = 2304
D_SAE = 18432
T_WIN = 5
K = 100
LAYER = "resid_L25"
N_SCAN_SEQUENCES = 2000
N_TOP_EXAMPLES = 8
CONTEXT_BEFORE = 16
CONTEXT_AFTER = 8
N_FEATURES_PER_CAT = 30
PAD_TOKEN_ID = 0   # gemma <pad>
BOS_TOKEN_ID = 2   # gemma <bos>
CLAUDE_CONCURRENCY = 3   # below 50 RPM limit
CLAUDE_INTER_CALL_DELAY = 1.2  # seconds between calls (50 RPM)


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("google/gemma-2-2b-it")


def decode_window(token_ids: np.ndarray, seq_idx: int, peak_pos: int,
                  tokenizer) -> str:
    seq = token_ids[seq_idx]
    start = max(0, peak_pos - CONTEXT_BEFORE)
    end = min(len(seq), peak_pos + CONTEXT_AFTER + 1)
    tokens_before = seq[start:peak_pos].tolist()
    peak_tok = int(seq[peak_pos])
    tokens_after = seq[peak_pos + 1:end].tolist()
    txt_before = tokenizer.decode(tokens_before, skip_special_tokens=False)
    txt_peak = tokenizer.decode([peak_tok], skip_special_tokens=False)
    txt_after = tokenizer.decode(tokens_after, skip_special_tokens=False)
    return f"{txt_before}«{txt_peak}»{txt_after}"


def is_content_mask(token_ids: np.ndarray) -> np.ndarray:
    """True for non-padding, non-BOS tokens."""
    return (token_ids != PAD_TOKEN_ID) & (token_ids != BOS_TOKEN_ID)


@torch.no_grad()
def compute_tfa_masses_excluding_pad(
    eval_x: torch.Tensor, eval_tokens: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute pred_mass and novel_mass excluding padding positions.

    Used to re-rank TFA features for scanning.
    """
    from src.bench.architectures._tfa_module import TemporalSAE

    state = torch.load(
        f"{CKPT_DIR}/tfa_pos__gemma-2-2b-it__fineweb__{LAYER}__k{K}__seed42.pt",
        map_location="cpu", weights_only=True,
    )
    model = TemporalSAE(
        dimin=D_IN, width=D_SAE, n_heads=4, sae_diff_type="topk", kval_topk=K,
        tied_weights=True, n_attn_layers=1, bottleneck_factor=8,
        use_pos_encoding=True, max_seq_len=512,
    ).cuda()
    model.load_state_dict(state)
    model.eval()

    scale = math.sqrt(D_IN) / eval_x[:16].norm(dim=-1).mean().item()
    N = eval_x.shape[0]

    content = torch.from_numpy(is_content_mask(eval_tokens)).cuda()  # (N, T)

    pred_mass = torch.zeros(D_SAE)
    novel_mass = torch.zeros(D_SAE)
    for s in range(0, N, 16):
        x = (eval_x[s : s + 16].cuda() * scale)
        m = content[s : s + 16].float().unsqueeze(-1)  # (B, T, 1)
        _, inter = model(x)
        pred = inter["pred_codes"].abs() * m
        novel = inter["novel_codes"] * m
        pred_mass += pred.reshape(-1, D_SAE).sum(dim=0).cpu()
        novel_mass += novel.reshape(-1, D_SAE).sum(dim=0).cpu()
    return pred_mass, novel_mass


@torch.no_grad()
def scan_tfa_codes(
    feature_indices: list[int], eval_x: torch.Tensor, eval_tokens: np.ndarray,
    use_pred: bool,
) -> dict[int, list]:
    from src.bench.architectures._tfa_module import TemporalSAE
    state = torch.load(
        f"{CKPT_DIR}/tfa_pos__gemma-2-2b-it__fineweb__{LAYER}__k{K}__seed42.pt",
        map_location="cpu", weights_only=True,
    )
    model = TemporalSAE(
        dimin=D_IN, width=D_SAE, n_heads=4, sae_diff_type="topk", kval_topk=K,
        tied_weights=True, n_attn_layers=1, bottleneck_factor=8,
        use_pos_encoding=True, max_seq_len=512,
    ).cuda()
    model.load_state_dict(state)
    model.eval()

    scale = math.sqrt(D_IN) / eval_x[:16].norm(dim=-1).mean().item()
    N = eval_x.shape[0]
    feat_tensor = torch.tensor(feature_indices, device="cuda")
    heaps = {fi: [] for fi in feature_indices}
    content = is_content_mask(eval_tokens)  # (N, T) bool, numpy

    for s in range(0, N, 8):
        x = (eval_x[s : s + 8].cuda() * scale)
        _, inter = model(x)
        codes = inter["pred_codes"] if use_pred else inter["novel_codes"]
        if use_pred:
            codes = codes.abs()
        sub = codes[:, :, feat_tensor].cpu().numpy()  # (B, T, n_features)
        B = sub.shape[0]
        for bi in range(B):
            valid = content[s + bi]  # (T,)
            for fi_pos, fi in enumerate(feature_indices):
                row = sub[bi, :, fi_pos]
                for pos, val in enumerate(row):
                    if not valid[pos]:
                        continue
                    v = float(val)
                    if v <= 0:
                        continue
                    h = heaps[fi]
                    entry = (v, s + bi, pos)
                    if len(h) < N_TOP_EXAMPLES:
                        heapq.heappush(h, entry)
                    elif v > h[0][0]:
                        heapq.heapreplace(h, entry)
    return heaps


@torch.no_grad()
def scan_txcdr(
    feature_indices: list[int], eval_x: torch.Tensor, eval_tokens: np.ndarray,
) -> dict[int, list]:
    state = torch.load(
        f"{CKPT_DIR}/crosscoder__gemma-2-2b-it__fineweb__{LAYER}__k{K}__seed42.pt",
        map_location="cpu", weights_only=True,
    )
    W_enc = state["W_enc"].cuda()
    b_enc = state["b_enc"].cuda()
    k_total = K * T_WIN

    N, seq_len, _ = eval_x.shape
    n_windows = seq_len - T_WIN + 1
    feat_tensor = torch.tensor(feature_indices, device="cuda")
    heaps = {fi: [] for fi in feature_indices}
    content = is_content_mask(eval_tokens)

    for s in range(0, N, 16):
        x = eval_x[s : s + 16].cuda()
        B = x.shape[0]
        for w in range(n_windows):
            # Skip if the window-start token is padding/bos
            valid = content[s : s + B, w]  # (B,)
            if not np.any(valid):
                continue
            window = x[:, w : w + T_WIN, :]
            pre = torch.einsum("btd,tds->bs", window, W_enc) + b_enc
            _, idx = pre.topk(k_total, dim=-1)
            mask = torch.zeros_like(pre, dtype=torch.bool)
            mask.scatter_(-1, idx, True)
            active = mask & (pre > 0)
            sub_act = (pre[:, feat_tensor] * active[:, feat_tensor].float()).cpu().numpy()
            for bi in range(B):
                if not valid[bi]:
                    continue
                for fi_pos, fi in enumerate(feature_indices):
                    v = float(sub_act[bi, fi_pos])
                    if v <= 0:
                        continue
                    h = heaps[fi]
                    entry = (v, s + bi, w)
                    if len(h) < N_TOP_EXAMPLES:
                        heapq.heappush(h, entry)
                    elif v > h[0][0]:
                        heapq.heapreplace(h, entry)
    return heaps


@torch.no_grad()
def scan_stacked(
    feature_indices: list[int], eval_x: torch.Tensor, eval_tokens: np.ndarray,
) -> dict[int, list]:
    state = torch.load(
        f"{CKPT_DIR}/stacked_sae__gemma-2-2b-it__fineweb__{LAYER}__k{K}__seed42.pt",
        map_location="cpu", weights_only=True,
    )
    W_enc = state["saes.0.W_enc"].cuda()
    b_enc = state["saes.0.b_enc"].cuda()
    b_dec = state["saes.0.b_dec"].cuda()

    N = eval_x.shape[0]
    feat_tensor = torch.tensor(feature_indices, device="cuda")
    heaps = {fi: [] for fi in feature_indices}
    content = is_content_mask(eval_tokens)

    for s in range(0, N, 16):
        x = eval_x[s : s + 16].cuda()
        B = x.shape[0]
        pre = (x - b_dec) @ W_enc.T + b_enc
        _, idx = pre.topk(K, dim=-1)
        mask = torch.zeros_like(pre, dtype=torch.bool)
        mask.scatter_(-1, idx, True)
        active = mask & (pre > 0)
        sub_act = (pre[:, :, feat_tensor] * active[:, :, feat_tensor].float()).cpu().numpy()
        for bi in range(B):
            valid = content[s + bi]
            for fi_pos, fi in enumerate(feature_indices):
                row = sub_act[bi, :, fi_pos]
                for pos, v in enumerate(row):
                    if not valid[pos]:
                        continue
                    v = float(v)
                    if v <= 0:
                        continue
                    h = heaps[fi]
                    entry = (v, s + bi, pos)
                    if len(h) < N_TOP_EXAMPLES:
                        heapq.heappush(h, entry)
                    elif v > h[0][0]:
                        heapq.heapreplace(h, entry)
    return heaps


SYSTEM_PROMPT = """You are analyzing features learned by a sparse autoencoder on a language model's residual stream.
Given a list of text snippets where a specific feature fires most strongly (the firing token is marked with « and »), describe the common pattern in 1-2 short sentences. Be concrete and specific.

Focus on what semantic/syntactic property the firing token shares across the examples. Avoid generic filler like "this feature captures patterns".

Output exactly two lines:
[EXPLANATION]: your 1-2 sentence description here
[CONFIDENCE]: LOW | MEDIUM | HIGH"""


def format_examples(examples: list[dict]) -> str:
    lines = []
    for i, ex in enumerate(examples):
        lines.append(f"[{i+1}] ({ex['activation']:.2f})  {ex['text']!r}")
    return "\n".join(lines)


async def explain_feature(client, examples: list[dict], model_id: str) -> dict:
    user = (
        f"Here are the top {len(examples)} activating contexts for one feature "
        f"(activation in parens, firing token in «»):\n\n{format_examples(examples)}"
        f"\n\nWhat is this feature detecting?"
    )
    delay = 2.0
    for attempt in range(4):
        try:
            resp = await client.messages.create(
                model=model_id,
                max_tokens=256,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user}],
            )
            raw = resp.content[0].text
            explanation = ""
            confidence = "UNKNOWN"
            for line in raw.splitlines():
                if line.startswith("[EXPLANATION]:"):
                    explanation = line[len("[EXPLANATION]:"):].strip()
                elif line.startswith("[CONFIDENCE]:"):
                    confidence = line[len("[CONFIDENCE]:"):].strip()
            return {"explanation": explanation, "confidence": confidence, "raw": raw}
        except Exception as e:
            err = str(e)
            if "rate_limit" in err.lower() and attempt < 3:
                await asyncio.sleep(delay)
                delay *= 2
                continue
            return {"explanation": "", "confidence": "ERROR", "raw": err}
    return {"explanation": "", "confidence": "ERROR", "raw": "max retries"}


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

    print(f"  {category_name}: calling Claude for {len(metadata)} features...")
    sem = asyncio.Semaphore(CLAUDE_CONCURRENCY)

    async def one_feature(fi, examples):
        async with sem:
            # Rate limit pacing
            await asyncio.sleep(CLAUDE_INTER_CALL_DELAY / CLAUDE_CONCURRENCY)
            result = await explain_feature(client, examples, "claude-haiku-4-5")
            record = {
                "category": category_name,
                "feat_idx": fi,
                "examples": examples,
                **result,
            }
            (out_dir / f"feat_{fi:05d}.json").write_text(json.dumps(record, indent=2))
            status = result["confidence"]
            print(f"    f{fi:5d} [{status:6s}]: {result['explanation'][:100]}")
            return record

    records = await asyncio.gather(*(one_feature(fi, ex) for fi, ex in metadata.items()))

    (out_dir / "_summary.json").write_text(json.dumps([
        {"feat_idx": r["feat_idx"], "explanation": r["explanation"],
         "confidence": r["confidence"]} for r in records
    ], indent=2))
    return records


async def amain():
    import anthropic
    client = anthropic.AsyncAnthropic()

    # Load eval data and tokens
    arr = np.load(f"{CACHE_DIR}/{LAYER}.npy", mmap_mode="r")
    tok_ids = np.load(f"{CACHE_DIR}/token_ids.npy", mmap_mode="r")
    eval_x = torch.from_numpy(np.array(arr[-N_SCAN_SEQUENCES:])).float()
    eval_tokens = np.array(tok_ids[-N_SCAN_SEQUENCES:])
    print(f"Loaded {N_SCAN_SEQUENCES} eval sequences")

    tokenizer = load_tokenizer()

    # Re-rank TFA features using non-padding mass
    print("\nRe-ranking TFA features by non-padding mass...")
    pred_mass, novel_mass = compute_tfa_masses_excluding_pad(eval_x, eval_tokens)
    torch.cuda.empty_cache()

    # Top pred-dominated (pred_ratio > 0.5): get from Phase 1c JSON; re-rank by non-pad pred_mass
    with open("/home/elysium/temp_xc/results/analysis/tfa_pred_vs_novel/top_pred_dominated_features.json") as f:
        pred_candidates = [d["feat_idx"] for d in json.load(f)]
    # Re-rank by non-pad pred_mass
    pred_vals = pred_mass[pred_candidates]
    order = torch.argsort(pred_vals, descending=True).tolist()
    pred_features = [pred_candidates[i] for i in order[:N_FEATURES_PER_CAT]]

    with open("/home/elysium/temp_xc/results/analysis/tfa_pred_vs_novel/top_novel_dominated_features.json") as f:
        novel_candidates = [d["feat_idx"] for d in json.load(f)]
    novel_vals = novel_mass[novel_candidates]
    order = torch.argsort(novel_vals, descending=True).tolist()
    novel_features = [novel_candidates[i] for i in order[:N_FEATURES_PER_CAT]]

    print(f"  Top TFA pred features (non-pad): {pred_features[:5]}")
    print(f"  Top TFA novel features (non-pad): {novel_features[:5]}")

    with open("/home/elysium/temp_xc/results/analysis/decoder_alignment/alive_top_unique.json") as f:
        unique = json.load(f)
    txcdr_features = unique["crosscoder"]["feature_indices"][:N_FEATURES_PER_CAT]
    stacked_features = unique["stacked_sae"]["feature_indices"][:N_FEATURES_PER_CAT]

    print("\n[1/4] Scanning TFA pred-only features (non-padding)...")
    heaps_pred = scan_tfa_codes(pred_features, eval_x, eval_tokens, use_pred=True)
    torch.cuda.empty_cache()

    print("\n[2/4] Scanning TFA novel-only features (non-padding)...")
    heaps_novel = scan_tfa_codes(novel_features, eval_x, eval_tokens, use_pred=False)
    torch.cuda.empty_cache()

    print("\n[3/4] Scanning TXCDR unique features (non-padding)...")
    heaps_txcdr = scan_txcdr(txcdr_features, eval_x, eval_tokens)
    torch.cuda.empty_cache()

    print("\n[4/4] Scanning Stacked unique features (non-padding)...")
    heaps_stacked = scan_stacked(stacked_features, eval_x, eval_tokens)
    torch.cuda.empty_cache()

    print("\n=== Claude explanations ===")
    await process_category("tfa_pred_only", pred_features, heaps_pred,
                            eval_tokens, tokenizer, client)
    await process_category("tfa_novel_only", novel_features, heaps_novel,
                            eval_tokens, tokenizer, client)
    await process_category("txcdr_unique", txcdr_features, heaps_txcdr,
                            eval_tokens, tokenizer, client)
    await process_category("stacked_unique", stacked_features, heaps_stacked,
                            eval_tokens, tokenizer, client)

    print("\nDONE.")


if __name__ == "__main__":
    asyncio.run(amain())
