"""Paper-protocol replication with ABSOLUTE strengths {10, 100, 150, 500, 1000,
1500, 5000, 10000, 15000} from T-SAE paper (arxiv:2511.05541) §B.2 — exactly
as in their Table 9 / Fig. 5 evaluation.

This is the apples-to-apples run vs the T-SAE paper's published numbers. Unlike
`intervene_paper_clamp_normalised.py` (which scales each arch's strengths by
its per-arch z_orig magnitude for fair cross-arch comparison), here we use the
paper's exact absolute clamp values for ALL archs.

Mechanism: clamp-on-latent + error-preserve, identical to
`intervene_paper_clamp_normalised.py`. Just the strength grid changes.

Outputs:
  results/case_studies/steering_paper_absolute/<arch_id>/generations.jsonl
  Each row carries `strength` (the absolute clamp value).

Run:
  TQDM_DISABLE=1 .venv/bin/python -m \
    experiments.phase7_unification.case_studies.steering.intervene_paper_clamp_absolute \
    --archs <arch_id> --seed <seed> [--force]
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

from experiments.phase7_unification._paths import OUT_DIR, banner, MLC_LAYERS
from experiments.phase7_unification.case_studies._arch_utils import (
    load_phase7_model_safe as _load_phase7_model,
    PER_TOKEN_CLASSES,
    WINDOW_CLASSES,
    MLC_CLASSES,
    window_T,
)
from experiments.phase7_unification.case_studies._paths import (
    CASE_STUDIES_DIR, SUBJECT_MODEL, ANCHOR_LAYER,
    STEERING_GEN_TOKENS, STEERING_PROMPT,
)


# Paper's absolute strengths from §B.2 — same for all archs
PAPER_STRENGTHS = (10, 100, 150, 500, 1000, 1500, 5000, 10000, 15000)
OUT_SUBDIR = "steering_paper_absolute"


def _per_token_hook_factory(sae, src_class, strengths_t, state):
    sae_dtype = torch.float32

    def _encode(x):
        if src_class == "TemporalMatryoshkaBatchTopKSAE":
            z = sae.encode(x, use_threshold=True)
            if isinstance(z, tuple):
                z = z[0]
            return z
        return sae.encode(x)

    def hook(module, inp, output):
        h = output[0] if isinstance(output, tuple) else output
        feat = state["feature_idx"]
        if feat is None:
            return None
        b_dtype = h.dtype
        Bh, S, d_in = h.shape
        flat = h.to(sae_dtype).reshape(Bh * S, d_in)
        with torch.no_grad():
            z = _encode(flat)
            x_hat_orig = sae.decode(z)
            z_clamped = z.clone().reshape(Bh, S, -1)
            z_clamped[:, :, feat] = strengths_t.view(Bh, 1).expand(Bh, S)
            x_hat_steer = sae.decode(z_clamped.reshape(Bh * S, -1))
            error = flat - x_hat_orig
            h_steered = (x_hat_steer + error).reshape(Bh, S, d_in).to(b_dtype)
        if isinstance(output, tuple):
            return (h_steered,) + output[1:]
        return h_steered

    return hook


def _decode_full_window(sae, src_class: str, z, T: int):
    """Window-arch decode: z (B, d_sae) → x_hat (B, T, d_in).
    Matches the canonical decode used in intervene_paper_clamp_normalised.py.
    """
    if hasattr(sae, "decode") and not src_class.startswith("Matryoshka"):
        return sae.decode(z)
    if hasattr(sae, "decode_scale"):
        return sae.decode_scale(z, T - 1)
    raise AttributeError(f"no decode/decode_scale on {src_class}")


def _window_hook_factory(sae, src_class, T, strengths_t, state):
    """Right-edge windowed protocol — copied from intervene_paper_clamp_normalised.py.
    For each output position p ∈ [T-1, S), encode the trailing T-window ending at
    p, clamp the picked feature, decode back to (..., T, d_in), and write the
    resulting last position to h[:, p, :].
    """
    sae_dtype = torch.float32

    def hook(module, inp, output):
        h = output[0] if isinstance(output, tuple) else output
        feat = state["feature_idx"]
        if feat is None:
            return None
        Bh, S, d_in = h.shape
        if S < T:
            return None
        b_dtype = h.dtype
        h_f = h.to(sae_dtype)
        windows = h_f.unfold(dimension=1, size=T, step=1)             # (B, K, d_in, T)
        windows = windows.movedim(-1, -2).contiguous()                # (B, K, T, d_in)
        K = windows.shape[1]
        flat = windows.reshape(Bh * K, T, d_in)
        with torch.no_grad():
            z = sae.encode(flat)
            x_hat_orig_full = _decode_full_window(sae, src_class, z, T)
            x_hat_orig_R = x_hat_orig_full[:, -1, :].reshape(Bh, K, d_in)
            z_c = z.clone().reshape(Bh, K, -1)
            z_c[:, :, feat] = strengths_t.view(Bh, 1).expand(Bh, K)
            z_c = z_c.reshape(Bh * K, -1)
            x_hat_steer_full = _decode_full_window(sae, src_class, z_c, T)
            x_hat_steer_R = x_hat_steer_full[:, -1, :].reshape(Bh, K, d_in)
            h_R = h_f[:, T - 1: S, :]
            error = h_R - x_hat_orig_R
            h_steered_R = (x_hat_steer_R + error).to(b_dtype)
        out = h.clone()
        out[:, T - 1: S, :] = h_steered_R
        if isinstance(output, tuple):
            return (out,) + output[1:]
        return out

    return hook


def steer_for_arch(
    arch_id: str,
    *,
    strengths: tuple[int, ...] = PAPER_STRENGTHS,
    force: bool = False,
    limit_concepts: int | None = None,
    out_subdir: str = OUT_SUBDIR,
    seed: int = 42,
) -> None:
    log_path = OUT_DIR / "training_logs" / f"{arch_id}__seed{seed}.json"
    ckpt_path = OUT_DIR / "ckpts" / f"{arch_id}__seed{seed}.pt"
    sel_subdir = "steering" if seed == 42 else f"steering_seed{seed}"
    sel_path = CASE_STUDIES_DIR / sel_subdir / arch_id / "feature_selection.json"
    actual_subdir = out_subdir if seed == 42 else f"{out_subdir}_seed{seed}"
    out_path = CASE_STUDIES_DIR / actual_subdir / arch_id / "generations.jsonl"

    if not sel_path.exists():
        print(f"  [skip] {arch_id}: feature_selection.json missing")
        return
    if out_path.exists() and not force:
        print(f"  [skip] {arch_id}: {out_path} exists (use --force)")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = json.loads(log_path.read_text())
    selection = json.loads(sel_path.read_text())
    src_class = meta["src_class"]
    abs_strengths = list(strengths)
    print(f"  {arch_id}: src_class={src_class}  ABSOLUTE strengths={abs_strengths}")

    device = torch.device("cuda")
    sae_dtype = torch.float32

    print(f"  loading {arch_id} ckpt...")
    sae, _ = _load_phase7_model(meta, ckpt_path, device)
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)

    use_cache = True
    if src_class in WINDOW_CLASSES:
        T = window_T(sae, src_class, meta)
        use_cache = False
        print(f"    arch family: WINDOW  T={T}  (use_cache=False)")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(SUBJECT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    subject = AutoModelForCausalLM.from_pretrained(
        SUBJECT_MODEL, torch_dtype=torch.float16, device_map=device,
    )
    subject.eval()
    for p in subject.parameters():
        p.requires_grad_(False)

    B = len(abs_strengths)
    strengths_t = torch.tensor(abs_strengths, dtype=sae_dtype, device=device)
    state = {"feature_idx": None}

    if src_class in WINDOW_CLASSES:
        hook_fn = _window_hook_factory(sae, src_class, T, strengths_t, state)
    elif src_class in PER_TOKEN_CLASSES:
        hook_fn = _per_token_hook_factory(sae, src_class, strengths_t, state)
    else:
        print(f"  [skip] {arch_id}: src_class={src_class} not supported "
              f"(only window + per-token)")
        return

    handle = subject.model.layers[ANCHOR_LAYER].register_forward_hook(hook_fn)

    concepts_items = list(selection["concepts"].items())
    if limit_concepts is not None:
        concepts_items = concepts_items[:limit_concepts]
    n_concepts = len(concepts_items)
    print(f"  generating: {n_concepts} concepts × {B} strengths "
          f"(absolute paper strengths, {STEERING_PROMPT!r}, T={src_class in WINDOW_CLASSES})")

    prompt_text = STEERING_PROMPT
    prompt_ids_single = tokenizer(prompt_text, return_tensors="pt").to(device)["input_ids"]
    prompt_len = prompt_ids_single.shape[1]
    prompt_ids = prompt_ids_single.expand(B, -1).contiguous()
    prompt_attn = torch.ones_like(prompt_ids)

    t0 = time.time()
    try:
        with out_path.open("w") as f_out:
            for ci, (concept_id, info) in enumerate(concepts_items):
                feat_idx = info["best_feature_idx"]
                state["feature_idx"] = feat_idx
                with torch.no_grad():
                    out_ids = subject.generate(
                        prompt_ids, attention_mask=prompt_attn,
                        max_new_tokens=STEERING_GEN_TOKENS,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        use_cache=use_cache,
                    )
                for bi, s_abs in enumerate(abs_strengths):
                    gen_tokens = out_ids[bi, prompt_len:].tolist()
                    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                    f_out.write(json.dumps({
                        "arch_id": arch_id,
                        "src_class": src_class,
                        "concept_id": concept_id,
                        "feature_idx": feat_idx,
                        "strength": s_abs,
                        "intervention": (
                            "paper_clamp_absolute_window"
                            if src_class in WINDOW_CLASSES
                            else "paper_clamp_absolute_pertoken"
                        ),
                        "prompt": prompt_text,
                        "generated_text": gen_text,
                        "concept_desc": info.get("concept_description", ""),
                    }) + "\n")
                f_out.flush()
                if (ci + 1) % 5 == 0 or ci + 1 == n_concepts:
                    elapsed = time.time() - t0
                    rate = (ci + 1) / elapsed
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
    ap.add_argument("--archs", nargs="+", default=["tsae_paper_k20"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit-concepts", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed of ckpt to use. seed != 42 writes to "
                    "steering_paper_absolute_seed{seed}/")
    args = ap.parse_args()
    banner(__file__)
    for arch_id in args.archs:
        print(f"\n=== {arch_id} seed={args.seed} (paper-clamp ABSOLUTE) ===")
        steer_for_arch(arch_id, force=args.force,
                       limit_concepts=args.limit_concepts, seed=args.seed)


if __name__ == "__main__":
    main()
