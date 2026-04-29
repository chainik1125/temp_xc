"""Paper-protocol steering reproduction (Ye et al. 2025, App B.2).

Replaces the AxBench-style additive intervention with the paper's
*clamp-on-latent with error preserve* mechanism:

    z       = encode(x)                # original SAE latents
    z'      = z; z'[j] = strength      # clamp feature j to strength
    x_hat   = decode(z)                # original reconstruction
    x_hat'  = decode(z')               # steered reconstruction
    x_steered = x_hat' + (x - x_hat)   # add back SAE reconstruction error

Strengths from paper §B.2: {10, 100, 150, 500, 1000, 1500, 5000, 10000, 15000}.
These are absolute clamp values, not multipliers of a unit decoder direction.

Restricted to per-token archs (TopKSAE + TemporalMatryoshkaBatchTopKSAE)
for the initial reproduction. Window/MLC archs need a separate clamp
implementation because their encoder consumes (T, d_in) or (n_layers, d_in)
inputs and there's no canonical per-position z to clamp.

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
    PER_TOKEN_CLASSES,
)
from experiments.phase7_unification.case_studies._paths import (
    CASE_STUDIES_DIR, SUBJECT_MODEL, ANCHOR_LAYER,
    STEERING_GEN_TOKENS, STEERING_PROMPT,
)


PAPER_STRENGTHS = (10, 100, 150, 500, 1000, 1500, 5000, 10000, 15000)
PAPER_OUT_SUBDIR = "steering_paper"
SUPPORTED_CLASSES = {"TopKSAE", "TemporalMatryoshkaBatchTopKSAE"}


def _sae_encode(sae, src_class: str, x: torch.Tensor) -> torch.Tensor:
    if src_class == "TemporalMatryoshkaBatchTopKSAE":
        z = sae.encode(x, use_threshold=True)
        if isinstance(z, tuple):
            z = z[0]
        return z
    return sae.encode(x)


def _sae_decode(sae, x: torch.Tensor) -> torch.Tensor:
    return sae.decode(x)


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
        print(f"  [skip] {arch_id}: feature_selection.json missing under "
              f"steering/ — run select_features first")
        return
    if out_path.exists() and not force:
        print(f"  [skip] {arch_id}: {out_path} exists (use --force)")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = json.loads(log_path.read_text())
    selection = json.loads(sel_path.read_text())
    src_class = meta["src_class"]
    if src_class not in SUPPORTED_CLASSES:
        print(f"  [skip] {arch_id}: src_class={src_class} not supported by "
              f"paper-clamp protocol (per-token archs only). Run AxBench "
              f"intervene_and_generate.py for window/MLC archs.")
        return

    device = torch.device("cuda")
    sae_dtype = torch.float32

    print(f"  loading {arch_id} ckpt + keeping SAE on GPU for clamp hook...")
    sae, _ = _load_phase7_model(meta, ckpt_path, device)
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)

    print(f"  loading subject model {SUBJECT_MODEL} (bf16)...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(SUBJECT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    subject = AutoModelForCausalLM.from_pretrained(
        SUBJECT_MODEL, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    subject.eval()
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
        b_dtype = h.dtype
        Bh, S, d_in = h.shape
        flat = h.to(sae_dtype).reshape(Bh * S, d_in)                # (B*S, d_in)
        with torch.no_grad():
            z = _sae_encode(sae, src_class, flat)                   # (B*S, d_sae)
            x_hat_orig = _sae_decode(sae, z)                        # (B*S, d_in)
            z_clamped = z.clone().reshape(Bh, S, -1)
            # Each batch element clamps to its own strength, broadcast over S.
            z_clamped[:, :, feat] = strengths_t.view(Bh, 1).expand(Bh, S)
            x_hat_steer = _sae_decode(sae, z_clamped.reshape(Bh * S, -1))
            error = flat - x_hat_orig
            h_steered = (x_hat_steer + error).reshape(Bh, S, d_in).to(b_dtype)
        if isinstance(output, tuple):
            return (h_steered,) + output[1:]
        return h_steered

    handle = subject.model.layers[ANCHOR_LAYER].register_forward_hook(hook)

    enc = tokenizer(STEERING_PROMPT, return_tensors="pt", add_special_tokens=True)
    prompt_ids = enc["input_ids"].to(device).repeat(B, 1)            # (B, P)
    prompt_attn = enc["attention_mask"].to(device).repeat(B, 1)
    prompt_len = prompt_ids.shape[1]

    concepts_items = list(selection["concepts"].items())
    if limit_concepts is not None:
        concepts_items = concepts_items[:limit_concepts]
    n_concepts = len(concepts_items)
    print(f"  generating: {n_concepts} concepts * {B} strengths = "
          f"{n_concepts * B} samples (clamp-on-latent + error preserve)")
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
                        "intervention": "paper_clamp_error_preserve",
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
    ap.add_argument(
        "--archs", nargs="+", default=["topk_sae", "tsae_paper_k20", "tsae_paper_k500"],
        help="defaults to the two paper-comparison archs (regular SAE + T-SAE).",
    )
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit-concepts", type=int, default=None,
                    help="truncate concept list (smoke runs).")
    ap.add_argument("--strengths", nargs="+", type=int, default=list(PAPER_STRENGTHS),
                    help="absolute z[j] clamp values; default = paper schedule.")
    ap.add_argument("--out-subdir", default=PAPER_OUT_SUBDIR,
                    help="results/case_studies/<this>/<arch>/generations.jsonl")
    args = ap.parse_args()
    banner(__file__)
    for arch_id in args.archs:
        print(f"\n=== {arch_id} (paper-protocol clamp) ===")
        steer_for_arch(arch_id, force=args.force, limit_concepts=args.limit_concepts,
                       strengths=tuple(args.strengths), out_subdir=args.out_subdir)


if __name__ == "__main__":
    main()
