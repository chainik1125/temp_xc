"""V3 — Decoder-direction additive steering (ablation for V1).

The simplest TXC-native intervention: skip the encode + clamp + decode
round-trip entirely. Just add `strength × W_dec[picked, :, :]` to the
most recent T-window of the residual.

Mechanism (per generation step):

    delta_TxD       = strength × W_dec[picked, :, :]    # (T, d_in)
    h_steered                  = h.clone()
    h_steered[:, S-T:S, :]    += delta_TxD               # write all T slices
    return h_steered

If V3 ≈ V1, then the encode/clamp ceremony is doing nothing useful — the
decoder direction alone is the steering primitive. If V3 < V1, then the
SAE's per-position cross-feature interactions (which encode/clamp/decode
captures) matter.

Outputs:
  results/case_studies/steering_paper_window_dec_additive{,_seed1}/<arch>/{generations,grades}.jsonl
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
OUT_SUBDIR = "steering_paper_window_dec_additive"


def _get_decoder_block(sae, src_class: str, picked_idx: int, T: int):
    """Return W_dec[picked_idx, :, :] of shape (T, d_in) for the picked feature."""
    # All TXC variants we work with have W_dec of shape (d_sae, T, d_in)
    # (TXCBareAntidead) or are matryoshka subclasses thereof.
    if hasattr(sae, "W_dec"):
        W_dec = sae.W_dec
        if W_dec.dim() == 3 and W_dec.shape[1] == T:
            return W_dec[picked_idx, :, :].detach()           # (T, d_in)
        if W_dec.dim() == 2:                                  # per-token (e.g. T=1)
            return W_dec[picked_idx, :].unsqueeze(0).detach()  # (1, d_in)
    raise AttributeError(f"can't extract decoder block for {src_class}")


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
        # Get the (T, d_in) decoder block for the picked feature
        dec_block = _get_decoder_block(sae, src_class, feat, T).to(sae_dtype)  # (T, d_in)
        # Scale per batch element: each batch element has a different strength
        # delta_TxD per batch: (B, T, d_in)
        delta_TxD = strengths_t.view(Bh, 1, 1) * dec_block.unsqueeze(0)
        h_steered = h.clone().to(sae_dtype)
        h_steered[:, S - T:S, :] = h_steered[:, S - T:S, :] + delta_TxD
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
    print(f"  generating: {n_concepts} concepts × {B} strengths (V3 dec-additive, T={T})")
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
                        "intervention": "paper_clamp_window_dec_additive",
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
        "txc_bare_antidead_t2_kpos20",
        "txc_bare_antidead_t3_kpos20",
        "txc_bare_antidead_t5_kpos20",
        "agentic_txc_02_kpos20",
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
        print(f"\n=== {arch_id} seed={args.seed} (V3 dec-additive) ===")
        steer_for_arch(arch_id, z_magnitude_path=z_mag_path,
                       use_normalised=args.normalised,
                       force=args.force, limit_concepts=args.limit_concepts,
                       seed=args.seed)


if __name__ == "__main__":
    main()
