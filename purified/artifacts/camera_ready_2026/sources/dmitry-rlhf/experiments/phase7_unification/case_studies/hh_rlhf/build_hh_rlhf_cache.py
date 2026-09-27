"""Build the HH-RLHF activation cache for case study C.i.

Reproduces the data side of T-SAE paper §4.5 + Appendix B.1:

  * Anthropic/hh-rlhf harmless-base, train split, first N=1000 samples
    (paper-default; tunable via --n-samples).
  * Tokenize each (chosen, rejected) pair SEPARATELY through Gemma-2-2b
    base. Both strings share a long Human-prompt prefix and differ only
    in the final Assistant response — we mark which token positions
    fall in that differing response via `response_mask`, computed from
    the offset_mapping (chars >= LCP boundary AND not pad).
  * Forward through Gemma-2-2b base with a hook on layer 12 (Phase 7's
    anchor). Capture per-token residual-stream activations.
  * Save chosen.npz + rejected.npz + meta.json under HH_RLHF_CACHE_DIR.

Storage at N=1000, max_length=256, fp16:
    1000 * 256 * 2304 * 2 B = 1.18 GB per side -> 2.36 GB total.

Run from repo root:

    .venv/bin/python -m experiments.phase7_unification.case_studies.hh_rlhf.build_hh_rlhf_cache

Idempotent: if all output files exist, returns without forwarding.
Use --force to rebuild.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("TQDM_DISABLE", "1")

from experiments.phase7_unification._paths import banner
from experiments.phase7_unification.case_studies._paths import (
    SUBJECT_MODEL,
    ANCHOR_LAYER,
    HH_RLHF_HF_PATH,
    HH_RLHF_SPLIT_DIR,
    HH_RLHF_N_SAMPLES,
    HH_RLHF_MAX_LENGTH,
    HH_RLHF_CACHE_DIR,
    DEFAULT_D_IN,
)


DTYPE = torch.bfloat16


def _longest_common_prefix(a: str, b: str) -> int:
    """Char-level LCP length of two strings."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _encode_one_side(
    side_name: str,
    texts: list[str],
    response_starts_char: list[int],
    tokenizer,
    model,
    captured: dict,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> dict[str, np.ndarray]:
    """Tokenize + forward + extract per-token L12 acts for one side
    (chosen or rejected). Returns numpy arrays for the npz dump.

    `response_starts_char[i]` is the character offset into `texts[i]` where
    the differing response begins (= LCP between chosen and rejected); a
    token belongs to the response iff its `(start_char, end_char)` from
    the tokenizer's offset mapping has `start_char >= response_starts_char[i]`
    AND it is not a pad token.
    """
    n = len(texts)
    acts = np.zeros((n, max_length, DEFAULT_D_IN), dtype=np.float16)
    input_ids = np.zeros((n, max_length), dtype=np.int32)
    attn = np.zeros((n, max_length), dtype=np.int8)
    resp_mask = np.zeros((n, max_length), dtype=bool)
    resp_len = np.zeros((n,), dtype=np.int32)
    seq_len = np.zeros((n,), dtype=np.int32)
    t0 = time.time()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = texts[start:end]
        rs_chunk = response_starts_char[start:end]
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        ids = enc["input_ids"]                                   # (B, T)
        am = enc["attention_mask"]                               # (B, T)
        offs = enc["offset_mapping"]                             # (B, T, 2) int
        # Per-example response-start as (B, 1, 1) for broadcasting.
        rs_t = torch.as_tensor(rs_chunk, dtype=offs.dtype).view(-1, 1)
        rmask = (offs[:, :, 0] >= rs_t) & (am == 1)
        captured.clear()
        with torch.no_grad():
            model(ids.to(device), attention_mask=am.to(device))
        h = captured[ANCHOR_LAYER]                               # (B, T, d) fp16/bf16 on cpu
        if h.shape[-1] != DEFAULT_D_IN:
            h = h[..., :DEFAULT_D_IN]
        acts[start:end] = h.to(torch.float16).numpy()
        input_ids[start:end] = ids.to(torch.int32).numpy()
        attn[start:end] = am.to(torch.int8).numpy()
        resp_mask[start:end] = rmask.numpy()
        resp_len[start:end] = rmask.sum(dim=1).to(torch.int32).numpy()
        seq_len[start:end] = am.sum(dim=1).to(torch.int32).numpy()
        if (start // batch_size) % 10 == 0:
            elapsed = time.time() - t0
            rate = (end) / max(elapsed, 1e-3)
            eta = (n - end) / max(rate, 1e-3)
            print(f"    {side_name:8s} [{end}/{n}] {rate:.1f} ex/s  ETA {eta:.0f}s")
    return dict(
        acts=acts,
        input_ids=input_ids,
        attention_mask=attn,
        response_mask=resp_mask,
        response_len=resp_len,
        seq_len=seq_len,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=HH_RLHF_N_SAMPLES)
    ap.add_argument("--max-length", type=int, default=HH_RLHF_MAX_LENGTH)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=HH_RLHF_CACHE_DIR)
    args = ap.parse_args()
    banner(__file__)

    out_dir: Path = args.out_dir
    chosen_path = out_dir / "chosen.npz"
    rejected_path = out_dir / "rejected.npz"
    meta_path = out_dir / "meta.json"
    if (
        not args.force
        and chosen_path.exists()
        and rejected_path.exists()
        and meta_path.exists()
    ):
        print(f"  cache present at {out_dir}, skip (use --force to rebuild)")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset {HH_RLHF_HF_PATH}/{HH_RLHF_SPLIT_DIR} ...")
    from datasets import load_dataset

    ds = load_dataset(HH_RLHF_HF_PATH, data_dir=HH_RLHF_SPLIT_DIR, split="train")
    n = min(args.n_samples, len(ds))
    chosen_texts = [ds[i]["chosen"] for i in range(n)]
    rejected_texts = [ds[i]["rejected"] for i in range(n)]

    lcp_chars = [
        _longest_common_prefix(c, r) for c, r in zip(chosen_texts, rejected_texts)
    ]
    print(
        f"  N={n}; LCP chars  mean={np.mean(lcp_chars):.0f}  min={min(lcp_chars)}  "
        f"max={max(lcp_chars)}"
    )

    print(f"Loading tokenizer + model {SUBJECT_MODEL} ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SUBJECT_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    assert tok.is_fast, "GemmaTokenizer must be fast (offset mapping required)"

    model = AutoModelForCausalLM.from_pretrained(
        SUBJECT_MODEL, torch_dtype=DTYPE, device_map="cuda",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    captured: dict[int, torch.Tensor] = {}

    def hook_fn(module, inp, output):
        acts = output[0] if isinstance(output, tuple) else output
        captured[ANCHOR_LAYER] = acts.detach().cpu()

    handle = model.model.layers[ANCHOR_LAYER].register_forward_hook(hook_fn)
    device = torch.device("cuda")

    try:
        print(f"Forwarding chosen (N={n}) ...")
        chosen = _encode_one_side(
            "chosen", chosen_texts, lcp_chars, tok, model, captured,
            device, args.batch_size, args.max_length,
        )
        print(
            f"  chosen response_len  mean={chosen['response_len'].mean():.1f}  "
            f"min={chosen['response_len'].min()}  max={chosen['response_len'].max()}"
        )
        print(f"Forwarding rejected (N={n}) ...")
        rejected = _encode_one_side(
            "rejected", rejected_texts, lcp_chars, tok, model, captured,
            device, args.batch_size, args.max_length,
        )
        print(
            f"  rejected response_len  mean={rejected['response_len'].mean():.1f}  "
            f"min={rejected['response_len'].min()}  max={rejected['response_len'].max()}"
        )
    finally:
        handle.remove()

    np.savez(chosen_path, **chosen)
    np.savez(rejected_path, **rejected)
    meta_path.write_text(
        json.dumps(
            {
                "subject_model": SUBJECT_MODEL,
                "anchor_layer": ANCHOR_LAYER,
                "hf_dataset": HH_RLHF_HF_PATH,
                "hf_split_dir": HH_RLHF_SPLIT_DIR,
                "n_samples": int(n),
                "max_length": int(args.max_length),
                "d_in": int(DEFAULT_D_IN),
                "lcp_chars_mean": float(np.mean(lcp_chars)),
                "lcp_chars_min": int(min(lcp_chars)),
                "lcp_chars_max": int(max(lcp_chars)),
            },
            indent=2,
        )
    )
    print(
        f"Done. {chosen_path.stat().st_size / 1e9:.2f} GB chosen, "
        f"{rejected_path.stat().st_size / 1e9:.2f} GB rejected."
    )

    # Sanity-reproduce the paper's response-length t-test (App B.1):
    #   "rejected ≈ 49.243 chosen ≈ 37.844, p=9e-10"
    from scipy import stats

    rj = rejected["response_len"].astype(np.float64)
    ch = chosen["response_len"].astype(np.float64)
    t_stat, p_val = stats.ttest_rel(rj, ch)
    print()
    print("=== HH-RLHF response-length t-test (paper App B.1 reproduction) ===")
    print(f"  rejected mean = {rj.mean():.3f}   (paper: 49.243)")
    print(f"  chosen   mean = {ch.mean():.3f}   (paper: 37.844)")
    print(f"  diff          = {(rj - ch).mean():.3f}   (paper: 11.399)")
    print(f"  paired t      = {t_stat:.3f}    p = {p_val:.2e}   (paper: p=9e-10)")


if __name__ == "__main__":
    main()
