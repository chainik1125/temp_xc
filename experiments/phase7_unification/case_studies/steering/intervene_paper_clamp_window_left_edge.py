"""V5 — Left-edge steering: structural mirror of canonical right-edge.

Right-edge: for each position p ∈ [T-1, S-1], window = [p-T+1, p],
take slice T-1 of decode (relative position T-1 = the right edge).
Encoder integrates BACKWARD from p.

Left-edge: for each position p ∈ [0, S-T], window = [p, p+T-1],
take slice 0 of decode (relative position 0 = the left edge).
Encoder integrates FORWARD from p.

Both protocols slide over K = S - T + 1 windows. Right-edge writes at
positions [T-1, S-1] (window_end). Left-edge writes at [0, S-T]
(window_start). The latter does NOT cover the most recent T-1 positions;
the active generation tip at S-1 receives no direct steering, only
indirect through attention to earlier positions.

Mechanism (per generation step, hook fires once with residual h ∈ (B, S, d_in)):

    windows         = h.unfold(stride=1, size=T)                       # (B, K, T, d_in)
    z               = TXC.encode(windows.flatten_BK)                   # (B*K, d_sae)
    x_hat_orig      = TXC.decode(z)                                     # (B*K, T, d_in)
    x_hat_orig_L    = x_hat_orig[:, 0, :]   reshape -> (B, K, d_in)     # leftmost slice
    z'              = z; z'[:, picked] = s_abs
    x_hat_steer_L   = TXC.decode(z')[:, 0, :]                           # (B, K, d_in)
    delta_L         = x_hat_steer_L - x_hat_orig_L                      # (B, K, d_in)
    h_steered       = h.clone()
    h_steered[:, 0:K, :]  += delta_L                                     # write at window_start
    return h_steered          # positions S-T+1..S-1 unchanged

Hypothesis (Han + W): for concepts whose signal builds *across* the
steered span (not concentrated at the right edge), a left-edge protocol
delivers the steering vector to the position where the concept "begins".
Through causal attention, later positions attending to those earlier
deltas inherit a forward-looking concept signal.

Outputs:
  results/case_studies/steering_paper_window_left_edge{,_seed1}/<arch>/{generations,grades}.jsonl
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
    WINDOW_CLASSES, window_T,
)
from experiments.phase7_unification.case_studies._paths import (
    CASE_STUDIES_DIR, SUBJECT_MODEL, ANCHOR_LAYER,
    STEERING_GEN_TOKENS, STEERING_PROMPT,
)


PAPER_STRENGTHS = (10, 100, 150, 500, 1000, 1500, 5000, 10000, 15000)
S_NORMS_DEFAULT = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
OUT_SUBDIR = "steering_paper_window_left_edge"


def _decode_full_window(sae, src_class: str, z, T: int):
    if hasattr(sae, "decode") and not src_class.startswith("Matryoshka"):
        return sae.decode(z)
    if hasattr(sae, "decode_scale"):
        return sae.decode_scale(z, T - 1)
    raise AttributeError(f"no decode/decode_scale on {src_class}")


def _build_hook(sae, src_class, T, strengths_t, state):
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

        # Build all sliding T-windows (stride 1)
        windows = h_f.unfold(dimension=1, size=T, step=1)            # (B, K, d_in, T)
        windows = windows.movedim(-1, -2).contiguous()                # (B, K, T, d_in)
        K = windows.shape[1]                                          # = S - T + 1
        flat = windows.reshape(Bh * K, T, d_in)

        with torch.no_grad():
            z = sae.encode(flat)                                       # (B*K, d_sae)
            x_hat_orig_full = _decode_full_window(sae, src_class, z, T)  # (B*K, T, d_in)
            x_hat_orig_L = x_hat_orig_full[:, 0, :].reshape(Bh, K, d_in) # (B, K, d_in)
            z_c = z.clone().reshape(Bh, K, -1)
            z_c[:, :, feat] = strengths_t.view(Bh, 1).expand(Bh, K)
            z_c = z_c.reshape(Bh * K, -1)
            x_hat_steer_full = _decode_full_window(sae, src_class, z_c, T)
            x_hat_steer_L = x_hat_steer_full[:, 0, :].reshape(Bh, K, d_in)
            delta_L = x_hat_steer_L - x_hat_orig_L                      # (B, K, d_in)

            # Write delta_L at left-edge positions: window k -> position k.
            # Positions [0, K-1] = [0, S-T] receive a delta. Positions
            # [S-T+1, S-1] are left untouched.
            h_steered = h.clone().to(sae_dtype)
            h_steered[:, 0:K, :] = h_steered[:, 0:K, :] + delta_L
        h_steered = h_steered.to(b_dtype)
        if isinstance(output, tuple):
            return (h_steered,) + output[1:]
        return h_steered

    return hook


def steer_for_arch(
    arch_id: str,
    *,
    z_magnitude_path: Path | None,
    use_normalised: bool,
    s_norms: tuple[float, ...] = S_NORMS_DEFAULT,
    paper_strengths: tuple[float, ...] = PAPER_STRENGTHS,
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
    if src_class not in WINDOW_CLASSES:
        print(f"  [skip] {arch_id}: only window archs supported")
        return

    if use_normalised:
        z_mag = json.loads(z_magnitude_path.read_text())
        arch_z = z_mag.get(arch_id)
        if arch_z is None:
            print(f"  [skip] {arch_id}: missing in {z_magnitude_path.name}")
            return
        abs_mean = float(arch_z["pooled"]["abs_mean"])
        abs_strengths = [round(s_n * abs_mean, 1) for s_n in s_norms]
        labels_norm = list(s_norms)
    else:
        abs_strengths = list(paper_strengths)
        labels_norm = [None] * len(abs_strengths)
    print(f"  {arch_id}: src_class={src_class}  strengths={abs_strengths}")

    device = torch.device("cuda")
    sae_dtype = torch.float32

    sae, _ = _load_phase7_model(meta, ckpt_path, device)
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)
    T = window_T(sae, src_class, meta)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(SUBJECT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    subject = AutoModelForCausalLM.from_pretrained(
        SUBJECT_MODEL, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    subject.eval()
    subject.config.use_cache = False
    for p in subject.parameters():
        p.requires_grad_(False)

    B = len(abs_strengths)
    strengths_t = torch.tensor(abs_strengths, dtype=sae_dtype, device=device)
    state = {"feature_idx": None}
    handle = subject.model.layers[ANCHOR_LAYER].register_forward_hook(
        _build_hook(sae, src_class, T, strengths_t, state)
    )

    enc = tokenizer(STEERING_PROMPT, return_tensors="pt", add_special_tokens=True)
    prompt_ids = enc["input_ids"].to(device).repeat(B, 1)
    prompt_attn = enc["attention_mask"].to(device).repeat(B, 1)
    prompt_len = prompt_ids.shape[1]

    concepts_items = list(selection["concepts"].items())
    if limit_concepts is not None:
        concepts_items = concepts_items[:limit_concepts]
    n_concepts = len(concepts_items)
    print(f"  generating: {n_concepts} concepts × {B} strengths (V5 left-edge, T={T})")
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
                for bi, (s_norm, s_abs) in enumerate(zip(labels_norm, abs_strengths)):
                    gen_tokens = out_ids[bi, prompt_len:].tolist()
                    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                    row = {
                        "arch_id": arch_id,
                        "src_class": src_class,
                        "concept_id": concept_id,
                        "feature_idx": feature_idx,
                        "strength": float(s_abs),
                        "prompt": STEERING_PROMPT,
                        "generated_text": gen_text,
                        "intervention": "paper_clamp_window_left_edge",
                        "T": T,
                    }
                    if s_norm is not None:
                        row["s_norm"] = float(s_norm)
                    f_out.write(json.dumps(row) + "\n")
                f_out.flush()
                if (ci + 1) % 5 == 0 or ci + 1 == n_concepts:
                    elapsed = time.time() - t0
                    rate = (ci + 1) / max(elapsed, 1e-3)
                    eta = (n_concepts - ci - 1) / max(rate, 1e-3)
                    print(f"    [{ci + 1}/{n_concepts}] {rate:.2f} concept/s  ETA {eta:.0f}s")
    finally:
        handle.remove()
        del subject, sae
        torch.cuda.empty_cache()
        gc.collect()
    print(f"  saved {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=[
        "txc_h8_t2_kpos20_shifts2",
        "txc_bare_antidead_t3_kpos20",
    ])
    ap.add_argument("--normalised", action="store_true")
    ap.add_argument(
        "--z-mag",
        default=str(CASE_STUDIES_DIR / "diagnostics_kpos20" / "z_orig_magnitudes.json"),
    )
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit-concepts", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    banner(__file__)
    z_mag_path = Path(args.z_mag) if args.normalised else None
    if args.normalised and not z_mag_path.exists():
        raise SystemExit(f"missing z magnitudes at {z_mag_path}")
    for arch_id in args.archs:
        print(f"\n=== {arch_id} seed={args.seed} (V5 left-edge) ===")
        steer_for_arch(arch_id, z_magnitude_path=z_mag_path,
                       use_normalised=args.normalised,
                       force=args.force, limit_concepts=args.limit_concepts,
                       seed=args.seed)


if __name__ == "__main__":
    main()
