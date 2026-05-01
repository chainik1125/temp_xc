"""V8 — Encoded broadcast: dynamic TXC encode/clamp/decode, broadcast everywhere.

Han's "attention-mixing" steering protocol (2026-05-01): instead of
V6's static decoder-direction broadcast, USE the TXC encode + clamp +
decode round-trip on the right-edge window each generation step, then
broadcast the resulting (averaged) δ to ALL prefix positions.

Mechanism (per generation step):
    last_window = h[:, S-T:S, :]                              # (B, T, d_in)
    z = TXC.encode(last_window)                                # (B, d_sae)
    z_c = z.clone(); z_c[picked] = s_abs                       # clamp
    delta_per_pos = TXC.decode(z_c) - TXC.decode(z)           # (B, T, d_in)
    delta_avg = delta_per_pos.mean(dim=1)                     # (B, d_in) — one direction
    h_steered[:, :, :] += delta_avg.unsqueeze(1)              # broadcast all S positions

Difference vs V6: V6 uses STATIC W_dec direction (no encode/decode at
runtime); V8 uses DYNAMIC encoded delta that adapts to the current
context.

Difference vs V7 (tiled-broadcast): V7 has per-block δ; V8 has ONE δ
for the whole prefix.

Why this is "attention-mixing-aware":
  Σ_s α_{t,s} · δ = δ · (Σ_s α_{t,s}) = δ · 1 = δ
  i.e. the steering signal is INVARIANT under attention mixing because
  attention weights sum to 1 and δ is uniform across positions. So
  whatever the model's attention pattern, the steered δ propagates
  faithfully to all subsequent positions.

Original V6 preamble preserved below for reference.

V6 — Dec-broadcast: constant TXC decoder direction at every position.

Like V3 (dec-additive) but writes the SAME direction at EVERY token in
the prefix, not just the active T-window. The single direction is the
mean across the T slices of the picked feature's decoder block.

Mechanism (per generation step):

    dir_d_in        = mean(W_dec[picked, :, :], dim=0)    # (d_in,)
    delta_pos       = strength × dir_d_in                  # (d_in,)
    h_steered       = h.clone()
    h_steered[:, :, :] += delta_pos[None, None, :]         # all S positions
    return h_steered

This is the closest TXC analog of T-SAE-style steering — a single feature
direction broadcast to every token. Tests whether the TXC's contrastive-
trained decoder direction is intrinsically better as a "concept axis"
than T-SAE's per-token decoder.

Comparison with V3:
- V3 active-only: writes (T, d_in) decoder block to most-recent T positions.
- V6 broadcast: writes mean direction at every position (S positions).

Outputs:
  results/case_studies/steering_paper_window_dec_broadcast{,_seed1}/<arch>/{generations,grades}.jsonl
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
OUT_SUBDIR = "steering_paper_window_encoded_broadcast"


def _decode_full_window(sae, src_class: str, z, T: int):
    """Decode z back to (B, T, d_in) — same as in tiled script."""
    if hasattr(sae, "decode") and not src_class.startswith("Matryoshka"):
        return sae.decode(z)
    if hasattr(sae, "decode_scale"):
        return sae.decode_scale(z, T - 1)
    raise AttributeError(f"no decode/decode_scale on {src_class}")


def _build_hook(sae, src_class, T, strengths_t, state):
    """V8: encode right-edge window each step, clamp picked feature, broadcast δ."""
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

        # Encode the right-edge T-window
        last_window = h_f[:, S - T : S, :]                                 # (Bh, T, d_in)
        with torch.no_grad():
            z = sae.encode(last_window)                                     # (Bh, d_sae)
            x_hat_orig = _decode_full_window(sae, src_class, z, T)          # (Bh, T, d_in)
            z_c = z.clone()
            z_c[:, feat] = strengths_t.view(Bh)                             # clamp picked
            x_hat_steer = _decode_full_window(sae, src_class, z_c, T)       # (Bh, T, d_in)
            delta_per_pos = x_hat_steer - x_hat_orig                        # (Bh, T, d_in)
            delta_avg = delta_per_pos.mean(dim=1)                            # (Bh, d_in) — single dir per batch

        # Broadcast δ_avg to ALL S positions (attention-mixing-invariant)
        h_steered = h_f.clone()
        h_steered = h_steered + delta_avg.unsqueeze(1)                       # (Bh, 1, d_in) → (Bh, S, d_in)
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
    print(f"  generating: {n_concepts} concepts × {B} strengths (V6 dec-broadcast, T={T})")
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
                        "intervention": "paper_clamp_window_dec_broadcast",
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
        print(f"\n=== {arch_id} seed={args.seed} (V6 dec-broadcast) ===")
        steer_for_arch(arch_id, z_magnitude_path=z_mag_path,
                       use_normalised=args.normalised,
                       force=args.force, limit_concepts=args.limit_concepts,
                       seed=args.seed)


if __name__ == "__main__":
    main()
