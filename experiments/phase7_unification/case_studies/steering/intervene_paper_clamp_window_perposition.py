"""Q2.C variant — paper-clamp on window archs, writing to ALL T positions.

Alternative to `intervene_paper_clamp_window.py` (right-edge attribution).
Here we write the clamped window's reconstruction at EVERY position
in the window. Since stride-1 windows overlap, each token gets multiple
writes; we take the AVERAGE across the windows that include it.

Mechanism (per generation step):

    For each window position t (from T-1 to S-1):
        z         = encoder(W_t)
        x_hat_W   = decoder(z)         # (T, d_in)
        x_hat'_W  = decoder(z')        # clamped
        Δ_W       = x_hat'_W - x_hat_W # (T, d_in)
    Then for each token position p ∈ [0, S):
        # Average Δ across all windows that cover position p:
        h'[p] = h[p] + mean_{w | p ∈ w} Δ_W[p - w_start]

This is the natural extension of the right-edge protocol when the
"useful" feature spans multiple positions of the window.

Outputs:
  results/case_studies/steering_paper_window_perposition/<arch>/generations.jsonl

Run after Q1.3 (uses the same z magnitude / strength schedule as Q1.3
when --normalised; otherwise PAPER_STRENGTHS).
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
S_NORMS_REFINED = (0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 50.0)  # finer near peak
OUT_SUBDIR = "steering_paper_window_perposition"

WEIGHT_PRESETS = {"uniform", "right-heavy", "right-only", "gaussian"}


def parse_position_weights(spec: str, T: int) -> tuple[list[float], str]:
    """Resolve a --position-weights spec to (weights, slug).

    Presets:
      uniform     [1, 1, ..., 1]                (no asymmetry)
      right-heavy [(i+1)/T for i in 0..T-1]     (linear ramp to right)
      right-only  [0, 0, ..., 0, 1]             (sanity ≈ right-edge)
      gaussian    exp(-(i-c)^2/(2σ^2)), σ=T/4   (symmetric peak at center)
    Custom: comma-separated floats with len == T.
    Returns (weights_list, slug_for_output_subdir).
    """
    spec = spec.strip()
    if spec == "uniform":
        return [1.0] * T, "uniform"
    if spec == "right-heavy":
        return [(i + 1) / T for i in range(T)], "rightheavy"
    if spec == "right-only":
        return [0.0] * (T - 1) + [1.0], "rightonly"
    if spec == "gaussian":
        center = (T - 1) / 2
        sigma = max(T / 4.0, 1e-6)
        raw = [math.exp(-((i - center) ** 2) / (2 * sigma * sigma)) for i in range(T)]
        m = max(raw)
        return [w / m for w in raw], "gaussian"
    if "," in spec:
        weights = [float(x) for x in spec.split(",")]
        if len(weights) != T:
            raise ValueError(
                f"--position-weights '{spec}' has {len(weights)} entries; expected T={T}"
            )
        slug = "w" + "_".join(f"{w:g}" for w in weights).replace(".", "p")
        return weights, slug
    raise ValueError(f"unknown --position-weights spec: {spec!r}")


def _decode_full_window(sae, src_class: str, z, T: int):
    if hasattr(sae, "decode") and not src_class.startswith("Matryoshka"):
        return sae.decode(z)
    if hasattr(sae, "decode_scale"):
        return sae.decode_scale(z, T - 1)
    raise AttributeError(f"no decode/decode_scale on {src_class}")


def _build_hook(sae, src_class, T, strengths_t, state, position_weights):
    """position_weights: list[float] length T applied to within-window position ti.

    state['feature_idx'] is either an int (single-feature, original behavior)
    OR a list[int] (Lever B multi-feature; clamps every listed feature to
    strength simultaneously).
    """
    sae_dtype = torch.float32
    pw = list(position_weights)

    def hook(module, inp, output):
        h = output[0] if isinstance(output, tuple) else output
        feat = state["feature_idx"]
        if feat is None:
            return None
        # Normalise to list-of-ints for uniform handling.
        feats = [int(feat)] if isinstance(feat, (int,)) else list(feat)
        Bh, S, d_in = h.shape
        if S < T:
            return None
        b_dtype = h.dtype
        h_f = h.to(sae_dtype)

        # Build sliding T-windows (overlapping, stride 1)
        windows = h_f.unfold(dimension=1, size=T, step=1)             # (B, K, d_in, T)
        windows = windows.movedim(-1, -2).contiguous()                # (B, K, T, d_in)
        K = windows.shape[1]
        flat = windows.reshape(Bh * K, T, d_in)
        with torch.no_grad():
            z = sae.encode(flat)
            x_hat_orig_full = _decode_full_window(sae, src_class, z, T)  # (B*K, T, d_in)
            z_c = z.clone().reshape(Bh, K, -1)
            for f_i in feats:
                z_c[:, :, f_i] = strengths_t.view(Bh, 1).expand(Bh, K)
            z_c = z_c.reshape(Bh * K, -1)
            x_hat_steer_full = _decode_full_window(sae, src_class, z_c, T)
            delta_W = (x_hat_steer_full - x_hat_orig_full).reshape(Bh, K, T, d_in)

            # Weighted-sum-write each window's delta at its T positions; the
            # divisor is sum of weights actually applied at each position so
            # the result is a weighted *average* (so uniform weights == prior).
            delta_sum = torch.zeros((Bh, S, d_in), dtype=sae_dtype, device=h.device)
            count = torch.zeros((Bh, S), dtype=sae_dtype, device=h.device)
            for ti in range(T):
                w_ti = pw[ti]
                if w_ti == 0.0:
                    continue
                for k_offset in range(K):
                    pos = k_offset + ti
                    if pos >= S:
                        continue
                    delta_sum[:, pos, :] += w_ti * delta_W[:, k_offset, ti, :]
                    count[:, pos] += w_ti
            count_safe = count.clamp(min=1e-8)[:, :, None]
            mean_delta = delta_sum / count_safe
            h_steered = (h_f + mean_delta).to(b_dtype)
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
    position_weights_spec: str = "uniform",
    top_k_features: int = 1,
) -> None:
    log_path = OUT_DIR / "training_logs" / f"{arch_id}__seed{seed}.json"
    ckpt_path = OUT_DIR / "ckpts" / f"{arch_id}__seed{seed}.pt"
    sel_subdir = "steering" if seed == 42 else f"steering_seed{seed}"
    sel_path = CASE_STUDIES_DIR / sel_subdir / arch_id / "feature_selection.json"

    if not sel_path.exists():
        print(f"  [skip] {arch_id}: feature_selection.json missing")
        return

    meta = json.loads(log_path.read_text())
    selection = json.loads(sel_path.read_text())
    src_class = meta["src_class"]
    if src_class not in WINDOW_CLASSES:
        print(f"  [skip] {arch_id}: only window archs supported")
        return

    if use_normalised:
        z_mag = json.loads(z_magnitude_path.read_text())
        arch_z = z_mag.get(arch_id)
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

    # Resolve position weights now that we know T, then route output to a
    # weights-suffixed subdir (uniform → no suffix, back-compat with prior runs).
    pw_list, pw_slug = parse_position_weights(position_weights_spec, T)
    base_subdir = out_subdir if seed == 42 else f"{out_subdir}_seed{seed}"
    actual_subdir = base_subdir if pw_slug == "uniform" else f"{base_subdir}_{pw_slug}"
    if top_k_features > 1:
        actual_subdir = f"{actual_subdir}_topk{top_k_features}"
    out_path = CASE_STUDIES_DIR / actual_subdir / arch_id / "generations.jsonl"
    if out_path.exists() and not force:
        print(f"  [skip] {arch_id}: {out_path} exists (use --force)")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  position_weights[{position_weights_spec}] → "
          f"{[round(w, 3) for w in pw_list]}  top_k={top_k_features}  out_subdir={actual_subdir}")

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
        _build_hook(sae, src_class, T, strengths_t, state, pw_list)
    )

    enc = tokenizer(STEERING_PROMPT, return_tensors="pt", add_special_tokens=True)
    prompt_ids = enc["input_ids"].to(device).repeat(B, 1)
    prompt_attn = enc["attention_mask"].to(device).repeat(B, 1)
    prompt_len = prompt_ids.shape[1]

    concepts_items = list(selection["concepts"].items())
    if limit_concepts is not None:
        concepts_items = concepts_items[:limit_concepts]
    n_concepts = len(concepts_items)
    print(f"  generating: {n_concepts} concepts × {B} strengths (per-position write, T={T})")
    t0 = time.time()
    try:
        with open(out_path, "w") as f_out:
            for ci, (concept_id, sel_data) in enumerate(concepts_items):
                # Lever B: pick top-K features per concept from feature_selection.json::top_5
                if top_k_features > 1:
                    top_list = sel_data.get("top_5", [])
                    feature_ids = [int(t["feature_idx"]) for t in top_list[:top_k_features]]
                    if len(feature_ids) < top_k_features:
                        # Concept didn't have K candidates listed; back-fill with best.
                        feature_ids += [int(sel_data["best_feature_idx"])] * (top_k_features - len(feature_ids))
                    state["feature_idx"] = feature_ids
                    feature_idx = feature_ids  # recorded in row
                else:
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
                        "intervention": "paper_clamp_window_perposition",
                        "T": T,
                        "position_weights_spec": position_weights_spec,
                        "position_weights": [float(w) for w in pw_list],
                        "top_k_features": top_k_features,
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
        "agentic_txc_02", "phase5b_subseq_h8", "phase57_partB_h8_bare_multidistance_t5",
    ])
    ap.add_argument("--normalised", action="store_true",
                    help="use family-normalised strengths (Q1.3 schedule)")
    ap.add_argument(
        "--z-mag",
        default=str(CASE_STUDIES_DIR / "diagnostics" / "z_orig_magnitudes.json"),
    )
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit-concepts", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--position-weights", default="uniform",
                    help="Weight profile for within-window position writes. "
                         "Presets: uniform (default), right-heavy, right-only, "
                         "gaussian. Or comma-separated floats matching T "
                         "(e.g. '0.5,1.0' for T=2).")
    ap.add_argument("--top-k-features", type=int, default=1,
                    help="Lever B: clamp top-K features per concept simultaneously "
                         "(default 1 = single best feature). Pulls top-K from "
                         "feature_selection.json::top_5.")
    ap.add_argument("--s-norms", default=None,
                    help="Comma-separated s_norm grid (default = 0.5,1,2,5,10,20,50). "
                         "Use 'refined' for the more granular grid (0.5,1,2,3,5,7,10,15,20,50).")
    args = ap.parse_args()
    banner(__file__)
    z_mag_path = Path(args.z_mag) if args.normalised else None
    if args.normalised and not z_mag_path.exists():
        raise SystemExit(f"missing Q1.1 output at {z_mag_path}")
    # Resolve s_norms
    if args.s_norms is None:
        s_norms = S_NORMS_DEFAULT
    elif args.s_norms == "refined":
        s_norms = S_NORMS_REFINED
    else:
        s_norms = tuple(float(x) for x in args.s_norms.split(","))
    for arch_id in args.archs:
        print(f"\n=== {arch_id} seed={args.seed} pw={args.position_weights} "
              f"top_k={args.top_k_features} s_norms={s_norms} (per-position window clamp) ===")
        steer_for_arch(arch_id, z_magnitude_path=z_mag_path,
                       use_normalised=args.normalised,
                       s_norms=s_norms,
                       force=args.force, limit_concepts=args.limit_concepts,
                       seed=args.seed,
                       position_weights_spec=args.position_weights,
                       top_k_features=args.top_k_features)


if __name__ == "__main__":
    main()
