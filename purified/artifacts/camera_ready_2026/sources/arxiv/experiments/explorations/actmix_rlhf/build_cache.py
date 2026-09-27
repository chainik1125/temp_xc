"""ACTMIX RLHF — rebuild the paper's HH-RLHF activation cache.

Verbatim port of `origin/han-phase7-agent-c @ 023d52c24`:
`experiments/phase7_unification/case_studies/hh_rlhf/build_hh_rlhf_cache.py`
(blob 6a…; constants inlined from `_paths.py` / `case_studies/_paths.py`
— the npz cache itself is NOT mirrored on HF, so it must be rebuilt).

Recipe (unchanged): Anthropic/hh-rlhf harmless-base train split, first
1000 (chosen, rejected) pairs; each side tokenized SEPARATELY
(padding=max_length 256, truncation, right padding, fast tokenizer
offset_mapping); response_mask = (start_char >= char-LCP(chosen,
rejected)) & attention; forward google/gemma-2-2b BASE in bf16 with a
forward hook on `model.model.layers[12]` (output[0] = the L12 residual
stream, phase-7 anchor); save fp16 acts (N, 256, 2304) + masks + lens
per side.

Integrity gate (checked at the end): the response-length paired
t-test must reproduce PHASE-7's OWN recorded run (the substrate the
shipped ckpts/figures used): rejected ≈ 36.23 / chosen ≈ 28.57 /
p ≈ 9.76e-10 (research log 2026-04-26-c1-hh-rlhf-stage1.md; Ye et
al.'s App B.1 absolutes 49.243/37.844 are a DIFFERENT tokenizer +
hh-rlhf version — phase-7 itself matched only t/p, recorded there).
A mismatch means the data side does NOT reproduce the phase-7
substrate — stop and flag, do not proceed.

Run: .venv/bin/python -m experiments.explorations.actmix_rlhf.build_cache
Idempotent unless --force.
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

SUBJECT_MODEL = "google/gemma-2-2b"
ANCHOR_LAYER = 12
DEFAULT_D_IN = 2304
HH_RLHF_HF_PATH = "Anthropic/hh-rlhf"
HH_RLHF_SPLIT_DIR = "harmless-base"
HH_RLHF_N_SAMPLES = 1000
HH_RLHF_MAX_LENGTH = 256
CACHE_DIR = Path("/workspace/caches/rlhf/cached_hh_rlhf")
DTYPE = torch.bfloat16

# The gate anchors to PHASE-7's OWN recorded run (the substrate the
# shipped checkpoints/figures actually used): han-phase7-unification
# research log 2026-04-26-c1-hh-rlhf-stage1.md — rejected 36.23 /
# chosen 28.57 / p 9.76e-10. Ye et al.'s App B.1 prints 49.243/37.844
# (different tokenizer + hh-rlhf version; phase-7 itself did not match
# those absolutes, only the t/p — recorded there verbatim).
PAPER_TTEST = {"rejected_mean": 36.23, "chosen_mean": 28.57,
               "diff_mean": 7.66, "p": 9.76e-10}
YE_APP_B1 = {"rejected_mean": 49.243, "chosen_mean": 37.844,
             "diff_mean": 11.399, "p": 9e-10}


def _longest_common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _encode_one_side(side_name, texts, response_starts_char, tokenizer,
                     model, captured, device, batch_size, max_length):
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
        enc = tokenizer(chunk, return_tensors="pt", padding="max_length",
                        truncation=True, max_length=max_length,
                        return_offsets_mapping=True)
        ids = enc["input_ids"]
        am = enc["attention_mask"]
        offs = enc["offset_mapping"]
        rs_t = torch.as_tensor(rs_chunk, dtype=offs.dtype).view(-1, 1)
        rmask = (offs[:, :, 0] >= rs_t) & (am == 1)
        captured.clear()
        with torch.no_grad():
            model(ids.to(device), attention_mask=am.to(device))
        h = captured[ANCHOR_LAYER]
        if h.shape[-1] != DEFAULT_D_IN:
            h = h[..., :DEFAULT_D_IN]
        acts[start:end] = h.to(torch.float16).numpy()
        input_ids[start:end] = ids.to(torch.int32).numpy()
        attn[start:end] = am.to(torch.int8).numpy()
        resp_mask[start:end] = rmask.numpy()
        resp_len[start:end] = rmask.sum(dim=1).to(torch.int32).numpy()
        seq_len[start:end] = am.sum(dim=1).to(torch.int32).numpy()
        if (start // batch_size) % 10 == 0:
            el = time.time() - t0
            print(f"    {side_name:8s} [{end}/{n}] "
                  f"{end / max(el, 1e-3):.1f} ex/s", flush=True)
    return dict(acts=acts, input_ids=input_ids, attention_mask=attn,
                response_mask=resp_mask, response_len=resp_len,
                seq_len=seq_len)


def main():
    global SUBJECT_MODEL, ANCHOR_LAYER, CACHE_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    # § 8 paper-faithful variant (l13-IT): same recipe, subject/layer/dir
    # swapped; defaults preserve the l12-BASE port behavior exactly.
    ap.add_argument("--subject-model", default=SUBJECT_MODEL)
    ap.add_argument("--layer", type=int, default=ANCHOR_LAYER)
    ap.add_argument("--cache-dir", default=str(CACHE_DIR))
    ap.add_argument("--record-only", action="store_true",
                    help="record integrity stats instead of asserting the "
                         "phase-7 l12 gate (CARD § 8: fresh-stats mode)")
    args = ap.parse_args()

    SUBJECT_MODEL = args.subject_model
    ANCHOR_LAYER = args.layer
    CACHE_DIR = Path(args.cache_dir)

    chosen_path = CACHE_DIR / "chosen.npz"
    rejected_path = CACHE_DIR / "rejected.npz"
    meta_path = CACHE_DIR / "meta.json"
    if (not args.force and chosen_path.exists() and rejected_path.exists()
            and meta_path.exists()):
        print(f"cache present at {CACHE_DIR}, skip")
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    ds = load_dataset(HH_RLHF_HF_PATH, data_dir=HH_RLHF_SPLIT_DIR,
                      split="train")
    n = min(HH_RLHF_N_SAMPLES, len(ds))
    chosen_texts = [ds[i]["chosen"] for i in range(n)]
    rejected_texts = [ds[i]["rejected"] for i in range(n)]
    lcp_chars = [_longest_common_prefix(c, r)
                 for c, r in zip(chosen_texts, rejected_texts)]
    print(f"N={n}; LCP chars mean={np.mean(lcp_chars):.0f}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SUBJECT_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    assert tok.is_fast

    model = AutoModelForCausalLM.from_pretrained(
        SUBJECT_MODEL, torch_dtype=DTYPE, device_map="cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)

    captured: dict[int, torch.Tensor] = {}

    def hook_fn(module, inp, output):
        acts = output[0] if isinstance(output, tuple) else output
        captured[ANCHOR_LAYER] = acts.detach().cpu()

    handle = model.model.layers[ANCHOR_LAYER].register_forward_hook(hook_fn)
    device = torch.device("cuda")
    try:
        chosen = _encode_one_side("chosen", chosen_texts, lcp_chars, tok,
                                  model, captured, device,
                                  args.batch_size, HH_RLHF_MAX_LENGTH)
        rejected = _encode_one_side("rejected", rejected_texts, lcp_chars,
                                    tok, model, captured, device,
                                    args.batch_size, HH_RLHF_MAX_LENGTH)
    finally:
        handle.remove()

    from scipy import stats
    rj = rejected["response_len"].astype(np.float64)
    ch = chosen["response_len"].astype(np.float64)
    t_stat, p_val = stats.ttest_rel(rj, ch)
    print(f"t-test: rejected {rj.mean():.3f} (paper {PAPER_TTEST['rejected_mean']}), "
          f"chosen {ch.mean():.3f} (paper {PAPER_TTEST['chosen_mean']}), "
          f"p={p_val:.2e} (paper {PAPER_TTEST['p']:.0e})", flush=True)
    ok = (abs(rj.mean() - PAPER_TTEST["rejected_mean"]) < 0.05
          and abs(ch.mean() - PAPER_TTEST["chosen_mean"]) < 0.05)
    if not ok and not args.record_only:
        raise RuntimeError(
            "HH-RLHF cache FAILED the paper App B.1 integrity gate — "
            "substrate does not reproduce; do not proceed.")
    if args.record_only:
        print(f"record-only mode: l12 reference match = {ok} "
              "(stats recorded, gate not asserted)", flush=True)

    np.savez(chosen_path, **chosen)
    np.savez(rejected_path, **rejected)
    meta_path.write_text(json.dumps({
        "subject_model": SUBJECT_MODEL, "anchor_layer": ANCHOR_LAYER,
        "hf_dataset": HH_RLHF_HF_PATH, "hf_split_dir": HH_RLHF_SPLIT_DIR,
        "n_samples": int(n), "max_length": HH_RLHF_MAX_LENGTH,
        "d_in": DEFAULT_D_IN,
        "integrity_gate": {"rejected_mean": float(rj.mean()),
                           "chosen_mean": float(ch.mean()),
                           "t": float(t_stat), "p": float(p_val),
                           "paper": PAPER_TTEST, "pass": bool(ok),
                           "mode": ("record-only" if args.record_only
                                    else "asserted")},
        "port_of": "han-phase7-agent-c@023d52c24 build_hh_rlhf_cache.py",
    }, indent=2))
    print(f"DONE -> {CACHE_DIR}")


if __name__ == "__main__":
    main()
