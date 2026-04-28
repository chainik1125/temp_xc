"""Q1.3 — paper-clamp at per-family normalised strengths.

The paper's strength schedule (10..15000) is ABSOLUTE, but window archs'
typical z magnitudes are 3-6× per-token archs (T-position integration).
At the same nominal s, paper-clamp pushes window archs less than per-token
archs in `<|z|>` units. To remove that bias, we re-run paper-clamp with
strengths chosen as MULTIPLES of each arch's measured `<|z|>`.

Schedule: s_norm ∈ {0.5, 1, 2, 5, 10, 20, 50} (log-spaced), and
absolute strength = s_norm × <|z|>_arch (rounded to 2 sig figs).

The {clamp-on-latent + error-preserve} mechanism is identical to
`intervene_paper_clamp.py` (per-token) and `intervene_paper_clamp_window.py`
(window archs) — only the strength grid changes.

Outputs:
  results/case_studies/steering_paper_normalised/<arch_id>/generations.jsonl
  Each row carries `s_norm` (multiple of <|z|>) and `s_abs` (the actual
  absolute strength used).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
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


# Multiples of <|z|>_arch — log-spaced, similar coverage to paper's 10..15000
# but per-arch normalised. 7 strengths × 30 concepts.
S_NORMS_DEFAULT = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
OUT_SUBDIR = "steering_paper_normalised"


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
    if hasattr(sae, "decode") and not src_class.startswith("Matryoshka"):
        return sae.decode(z)
    if hasattr(sae, "decode_scale"):
        return sae.decode_scale(z, T - 1)
    raise AttributeError(f"no decode/decode_scale on {src_class}")


def _window_hook_factory(sae, src_class, T, strengths_t, state):
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
    z_magnitude_path: Path,
    s_norms: tuple[float, ...] = S_NORMS_DEFAULT,
    force: bool = False,
    limit_concepts: int | None = None,
    out_subdir: str = OUT_SUBDIR,
) -> None:
    log_path = OUT_DIR / "training_logs" / f"{arch_id}__seed42.json"
    ckpt_path = OUT_DIR / "ckpts" / f"{arch_id}__seed42.pt"
    sel_path = CASE_STUDIES_DIR / "steering" / arch_id / "feature_selection.json"
    out_path = CASE_STUDIES_DIR / out_subdir / arch_id / "generations.jsonl"

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

    z_mag = json.loads(z_magnitude_path.read_text())
    arch_z = z_mag.get(arch_id)
    if arch_z is None:
        print(f"  [skip] {arch_id}: missing in {z_magnitude_path.name}")
        return
    abs_mean = float(arch_z["pooled"]["abs_mean"])
    abs_strengths = [round(s_n * abs_mean, 1) for s_n in s_norms]
    print(f"  {arch_id}: src_class={src_class}  <|z|>={abs_mean:.2f}  "
          f"strengths={abs_strengths}")

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
    elif src_class in PER_TOKEN_CLASSES:
        T = 1
        print(f"    arch family: PER-TOKEN  T=1")
    elif src_class in MLC_CLASSES:
        print(f"  [skip] {arch_id}: MLC paper-clamp not implemented in this driver "
              f"(needs multi-layer hook + reconstruction). Tracked for follow-up.")
        return
    else:
        print(f"  [skip] {arch_id}: unknown src_class={src_class}")
        return

    print(f"  loading subject model {SUBJECT_MODEL} (bf16)...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(SUBJECT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    subject = AutoModelForCausalLM.from_pretrained(
        SUBJECT_MODEL, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    subject.eval()
    if not use_cache:
        subject.config.use_cache = False
    for p in subject.parameters():
        p.requires_grad_(False)

    B = len(abs_strengths)
    strengths_t = torch.tensor(abs_strengths, dtype=sae_dtype, device=device)
    state = {"feature_idx": None}

    if src_class in WINDOW_CLASSES:
        hook_fn = _window_hook_factory(sae, src_class, T, strengths_t, state)
    else:
        hook_fn = _per_token_hook_factory(sae, src_class, strengths_t, state)

    handle = subject.model.layers[ANCHOR_LAYER].register_forward_hook(hook_fn)

    enc = tokenizer(STEERING_PROMPT, return_tensors="pt", add_special_tokens=True)
    prompt_ids = enc["input_ids"].to(device).repeat(B, 1)
    prompt_attn = enc["attention_mask"].to(device).repeat(B, 1)
    prompt_len = prompt_ids.shape[1]

    concepts_items = list(selection["concepts"].items())
    if limit_concepts is not None:
        concepts_items = concepts_items[:limit_concepts]
    n_concepts = len(concepts_items)
    print(f"  generating: {n_concepts} concepts × {B} strengths "
          f"(use_cache={use_cache})")
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
                        use_cache=use_cache,
                    )
                for bi, (s_norm, s_abs) in enumerate(zip(s_norms, abs_strengths)):
                    gen_tokens = out_ids[bi, prompt_len:].tolist()
                    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                    f_out.write(json.dumps({
                        "arch_id": arch_id,
                        "src_class": src_class,
                        "concept_id": concept_id,
                        "feature_idx": feature_idx,
                        "s_norm": float(s_norm),
                        "strength": float(s_abs),
                        "abs_z_mean": abs_mean,
                        "prompt": STEERING_PROMPT,
                        "generated_text": gen_text,
                        "intervention": (
                            "paper_clamp_window_error_preserve_normalised"
                            if src_class in WINDOW_CLASSES
                            else "paper_clamp_error_preserve_normalised"
                        ),
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


DEFAULT_ARCHS = (
    "topk_sae",
    "tsae_paper_k20",
    "tsae_paper_k500",
    "agentic_txc_02",
    "phase5b_subseq_h8",
    # mlc deferred (different hook structure); H8-multidist-t5 if time
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=list(DEFAULT_ARCHS))
    ap.add_argument(
        "--z-mag",
        default=str(CASE_STUDIES_DIR / "diagnostics" / "z_orig_magnitudes.json"),
        help="Path to Q1.1 output.",
    )
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit-concepts", type=int, default=None)
    args = ap.parse_args()
    banner(__file__)
    z_mag_path = Path(args.z_mag)
    if not z_mag_path.exists():
        raise SystemExit(f"missing Q1.1 output at {z_mag_path}")
    for arch_id in args.archs:
        print(f"\n=== {arch_id} (paper-clamp normalised) ===")
        steer_for_arch(arch_id, z_magnitude_path=z_mag_path,
                       force=args.force, limit_concepts=args.limit_concepts)


if __name__ == "__main__":
    main()
