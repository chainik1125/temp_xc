"""Paper-protocol clamp generalized to window-encoder archs.

Per-token archs already supported by intervene_paper_clamp.py; this script
generalizes the protocol to window archs (TXC family, SubseqH8) where the
encoder consumes a length-T window of L12 residuals.

Window-arch clamp + error-preserve (per generation step):

    For each position t in the sequence with t >= T-1, form the T-token
    window W_t = h[t-T+1 : t+1].
        z         = encoder(W_t)              # (d_sae,)
        x_hat_W   = decoder(z)                # (T, d_in)  full window recon
        x_hat_R   = x_hat_W[-1, :]            # (d_in,)    right-edge token
        z'        = z; z'[j] = strength
        x_hat_W'  = decoder(z')               # (T, d_in)
        x_hat_R'  = x_hat_W'[-1, :]
        h_t_steered = x_hat_R' + (h_t - x_hat_R)

    Replace h_t at every right-edge position; positions t < T-1 pass through.

The hook must see the full (B, S, d_in) sequence at every forward pass —
HF KV cache would only emit the new token. Run with `use_cache=False`.

Output: results/case_studies/steering_paper/<arch_id>/generations.jsonl
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("TQDM_DISABLE", "1")

from experiments.phase7_unification._paths import OUT_DIR, banner
from experiments.phase7_unification.case_studies._arch_utils import (
    load_phase7_model_safe as _load_phase7_model,
    WINDOW_CLASSES,
    window_T,
)
from experiments.phase7_unification.case_studies._paths import (
    CASE_STUDIES_DIR, SUBJECT_MODEL, ANCHOR_LAYER,
    STEERING_GEN_TOKENS, STEERING_PROMPT,
)


PAPER_STRENGTHS = (10, 100, 150, 500, 1000, 1500, 5000, 10000, 15000)
PAPER_OUT_SUBDIR = "steering_paper"


def _decode_full_window(sae, src_class: str, z: torch.Tensor, T: int) -> torch.Tensor:
    """Reconstruct (B, T, d_in) for a window-encoder arch.

    `decode(z)` exists on TXCBareAntidead and its subclasses (incl. SubseqH8
    via TXCBareMultiDistanceContrastiveAntidead). PositionMatryoshkaTXCDR
    descendants (incl. MatryoshkaTXCDRContrastiveMultiscale) only expose
    `decode_scale(z, scale_idx)` — the largest scale (T-1) is the full window.
    """
    if hasattr(sae, "decode") and not src_class.startswith("Matryoshka"):
        return sae.decode(z)
    if hasattr(sae, "decode_scale"):
        return sae.decode_scale(z, T - 1)
    raise AttributeError(f"no decode/decode_scale on {src_class}")


def steer_for_arch(
    arch_id: str, *, force: bool = False, limit_concepts: int | None = None,
    strengths: tuple[int, ...] = PAPER_STRENGTHS,
    out_subdir: str = PAPER_OUT_SUBDIR,
) -> None:
    log_path = OUT_DIR / "training_logs" / f"{arch_id}__seed42.json"
    ckpt_path = OUT_DIR / "ckpts" / f"{arch_id}__seed42.pt"
    sel_path = CASE_STUDIES_DIR / "steering" / arch_id / "feature_selection.json"
    out_path = CASE_STUDIES_DIR / out_subdir / arch_id / "generations.jsonl"

    if not sel_path.exists():
        print(f"  [skip] {arch_id}: feature_selection.json missing under steering/")
        return
    if out_path.exists() and not force:
        print(f"  [skip] {arch_id}: {out_path} exists (use --force)")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = json.loads(log_path.read_text())
    selection = json.loads(sel_path.read_text())
    src_class = meta["src_class"]
    if src_class not in WINDOW_CLASSES:
        print(f"  [skip] {arch_id}: src_class={src_class} is not a window arch. "
              f"Use intervene_paper_clamp.py for per-token archs.")
        return

    device = torch.device("cuda")
    sae_dtype = torch.float32

    print(f"  loading {arch_id} ckpt + keeping SAE on GPU for window-clamp hook...")
    sae, _ = _load_phase7_model(meta, ckpt_path, device)
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)

    T = window_T(sae, src_class, meta)
    print(f"    src_class={src_class}  T={T}")

    print(f"  loading subject model {SUBJECT_MODEL} (bf16)...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(SUBJECT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    subject = AutoModelForCausalLM.from_pretrained(
        SUBJECT_MODEL, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    subject.eval()
    subject.config.use_cache = False  # critical: hook needs full (B, S, d_in)
    for p in subject.parameters():
        p.requires_grad_(False)

    strengths = list(strengths)
    B = len(strengths)
    strengths_t = torch.tensor(strengths, dtype=sae_dtype, device=device)

    state = {"feature_idx": None}

    def hook(module, inp, output):
        h = output[0] if isinstance(output, tuple) else output      # (B, S, d_in)
        feat = state["feature_idx"]
        if feat is None:
            return None
        Bh, S, d_in = h.shape
        if S < T:
            return None  # not enough context to form one full window
        b_dtype = h.dtype
        h_f = h.to(sae_dtype)

        # Build all sliding T-windows. unfold puts size LAST.
        windows = h_f.unfold(dimension=1, size=T, step=1)            # (B, K, d_in, T)
        windows = windows.movedim(-1, -2).contiguous()               # (B, K, T, d_in)
        K = windows.shape[1]                                          # = S - T + 1
        flat = windows.reshape(Bh * K, T, d_in)

        with torch.no_grad():
            z = sae.encode(flat)                                     # (B*K, d_sae)
            x_hat_orig_full = _decode_full_window(sae, src_class, z, T)  # (B*K, T, d_in)
            x_hat_orig_R = x_hat_orig_full[:, -1, :].reshape(Bh, K, d_in)

            z_c = z.clone().reshape(Bh, K, -1)
            z_c[:, :, feat] = strengths_t.view(Bh, 1).expand(Bh, K)
            z_c = z_c.reshape(Bh * K, -1)
            x_hat_steer_full = _decode_full_window(sae, src_class, z_c, T)
            x_hat_steer_R = x_hat_steer_full[:, -1, :].reshape(Bh, K, d_in)

            # Right-edge tokens in the original residual: positions [T-1 : S]
            h_R = h_f[:, T - 1: S, :]                                # (B, K, d_in)
            error = h_R - x_hat_orig_R
            h_steered_R = (x_hat_steer_R + error).to(b_dtype)        # (B, K, d_in)

        out = h.clone()
        out[:, T - 1: S, :] = h_steered_R
        if isinstance(output, tuple):
            return (out,) + output[1:]
        return out

    handle = subject.model.layers[ANCHOR_LAYER].register_forward_hook(hook)

    enc = tokenizer(STEERING_PROMPT, return_tensors="pt", add_special_tokens=True)
    prompt_ids = enc["input_ids"].to(device).repeat(B, 1)
    prompt_attn = enc["attention_mask"].to(device).repeat(B, 1)
    prompt_len = prompt_ids.shape[1]

    concepts_items = list(selection["concepts"].items())
    if limit_concepts is not None:
        concepts_items = concepts_items[:limit_concepts]
    n_concepts = len(concepts_items)
    print(f"  generating: {n_concepts} concepts * {B} strengths = "
          f"{n_concepts * B} samples (window-clamp + error preserve, T={T}, "
          f"use_cache=False)")
    t0 = time.time()
    try:
        with open(out_path, "w") as f_out:
            for ci, (concept_id, sel_data) in enumerate(concepts_items):
                feature_idx = int(sel_data["best_feature_idx"])
                state["feature_idx"] = feature_idx
                with torch.no_grad():
                    out_ids = subject.generate(
                        prompt_ids, attention_mask=prompt_attn,
                        max_new_tokens=STEERING_GEN_TOKENS,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        use_cache=False,
                    )
                for bi, strength in enumerate(strengths):
                    gen_tokens = out_ids[bi, prompt_len:].tolist()
                    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                    f_out.write(json.dumps({
                        "arch_id": arch_id,
                        "src_class": src_class,
                        "concept_id": concept_id,
                        "feature_idx": feature_idx,
                        "strength": float(strength),
                        "prompt": STEERING_PROMPT,
                        "generated_text": gen_text,
                        "intervention": "paper_clamp_window_error_preserve",
                        "T": T,
                    }) + "\n")
                f_out.flush()
                if (ci + 1) % 5 == 0 or ci + 1 == n_concepts:
                    elapsed = time.time() - t0
                    rate = (ci + 1) / max(elapsed, 1e-3)
                    eta = (n_concepts - ci - 1) / max(rate, 1e-3)
                    print(f"    [{ci + 1}/{n_concepts}] {rate:.2f} concept/s  "
                          f"ETA {eta:.0f}s")
    finally:
        handle.remove()
        del subject, sae
        torch.cuda.empty_cache()
        gc.collect()
    print(f"  saved {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=[
        "agentic_txc_02", "phase5b_subseq_h8", "phase57_partB_h8_bare_multidistance_t5",
    ])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit-concepts", type=int, default=None)
    ap.add_argument("--strengths", nargs="+", type=int, default=list(PAPER_STRENGTHS),
                    help="absolute z[j] clamp values; default = paper schedule.")
    ap.add_argument("--out-subdir", default=PAPER_OUT_SUBDIR,
                    help="results/case_studies/<this>/<arch>/generations.jsonl")
    args = ap.parse_args()
    banner(__file__)
    for arch_id in args.archs:
        print(f"\n=== {arch_id} (paper-protocol window-clamp) ===")
        steer_for_arch(arch_id, force=args.force, limit_concepts=args.limit_concepts,
                       strengths=tuple(args.strengths), out_subdir=args.out_subdir)


if __name__ == "__main__":
    main()
