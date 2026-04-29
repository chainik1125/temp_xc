"""Q1.1 — z_orig magnitude diagnostic for picked steering features.

For each of the 6 shortlisted archs, encode the 30 concepts × 5 examples
probe set (`concepts.py`) through Gemma-2-2b base + arch's encoder, and
extract the activation magnitude `z[j]` of each concept's PICKED feature
(`feature_selection.json::concepts.<id>.best_feature_idx`) at every
content position.

Records:

- per (arch, concept):
    n_obs, mean, median, p90, p95, p99, max
    of |z[j_picked]| over content positions × all 5 examples.

The brief's prediction (Dmitry's analysis): window archs have z_orig
magnitudes ~5× per-token archs because the encoder integrates over T
tokens.

Output:
  results/case_studies/diagnostics/z_orig_magnitudes.json
  results/case_studies/diagnostics/z_orig_per_concept.npz   (raw arrays)
  results/case_studies/plots/phase7_z_orig_distributions.png

Run:
  TQDM_DISABLE=1 .venv/bin/python -m experiments.phase7_unification.case_studies.steering.diagnose_z_magnitudes
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("TQDM_DISABLE", "1")

from experiments.phase7_unification._paths import OUT_DIR, banner, MLC_LAYERS
from experiments.phase7_unification.case_studies._arch_utils import (
    load_phase7_model_safe as _load_phase7_model,
    encode_per_position,
    window_T,
    MLC_CLASSES,
    PER_TOKEN_CLASSES,
    WINDOW_CLASSES,
)
from experiments.phase7_unification.case_studies._paths import (
    CASE_STUDIES_DIR, SUBJECT_MODEL, ANCHOR_LAYER, DEFAULT_D_IN,
)
from experiments.phase7_unification.case_studies.steering.concepts import CONCEPTS


CONCEPT_MAX_LENGTH = 64
DEFAULT_ARCHS = (
    "topk_sae",
    "tsae_paper_k20",
    "tsae_paper_k500",
    "mlc_contrastive_alpha100_batchtopk",
    "agentic_txc_02",
    "phase5b_subseq_h8",
)


def _capture_activations(
    sentences: list[str], model, tokenizer, device,
    is_mlc: bool, max_length: int = CONCEPT_MAX_LENGTH, batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(sentences)
    if is_mlc:
        n_lay = len(MLC_LAYERS)
        acts = np.zeros((n, max_length, n_lay, DEFAULT_D_IN), dtype=np.float16)
    else:
        acts = np.zeros((n, max_length, DEFAULT_D_IN), dtype=np.float16)
    attn = np.zeros((n, max_length), dtype=np.int8)
    captured: dict[int, torch.Tensor] = {}
    handles = []

    layers_to_hook = list(MLC_LAYERS) if is_mlc else [ANCHOR_LAYER]

    def make_hook(layer_idx):
        def hook(module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = h.detach().cpu()
        return hook

    for li in layers_to_hook:
        handles.append(model.model.layers[li].register_forward_hook(make_hook(li)))

    try:
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            chunk = sentences[start:end]
            enc = tokenizer(chunk, return_tensors="pt",
                            padding="max_length", truncation=True, max_length=max_length)
            captured.clear()
            with torch.no_grad():
                model(enc["input_ids"].to(device),
                      attention_mask=enc["attention_mask"].to(device))
            if is_mlc:
                for idx, li in enumerate(MLC_LAYERS):
                    h = captured[li]
                    if h.shape[-1] != DEFAULT_D_IN:
                        h = h[..., :DEFAULT_D_IN]
                    acts[start:end, :, idx, :] = h.to(torch.float16).numpy()
            else:
                h = captured[ANCHOR_LAYER]
                if h.shape[-1] != DEFAULT_D_IN:
                    h = h[..., :DEFAULT_D_IN]
                acts[start:end] = h.to(torch.float16).numpy()
            attn[start:end] = enc["attention_mask"].to(torch.int8).numpy()
    finally:
        for handle in handles:
            handle.remove()
    return acts, attn


def _summarize(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"n": 0}
    abs_v = np.abs(values).astype(np.float64)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "abs_mean": float(abs_v.mean()),
        "abs_median": float(np.median(abs_v)),
        "abs_p90": float(np.percentile(abs_v, 90)),
        "abs_p95": float(np.percentile(abs_v, 95)),
        "abs_p99": float(np.percentile(abs_v, 99)),
        "abs_max": float(abs_v.max()),
        "raw_mean": float(values.mean()),
        "raw_max": float(values.max()),
    }


def diagnose_arch(
    arch_id: str, *, batch_size: int = 16, share_acts: tuple | None = None,
    seed: int = 42,
) -> dict | None:
    """Per-arch diagnostic. If share_acts provided ((acts_l12, attn) tuple)
    we reuse the L12 activations cached from a prior arch (for non-MLC
    archs only). For MLC, multi-layer activations are captured fresh.
    """
    log_path = OUT_DIR / "training_logs" / f"{arch_id}__seed{seed}.json"
    ckpt_path = OUT_DIR / "ckpts" / f"{arch_id}__seed{seed}.pt"
    sel_subdir = "steering" if seed == 42 else f"steering_seed{seed}"
    sel_path = (CASE_STUDIES_DIR / sel_subdir / arch_id / "feature_selection.json")
    if not ckpt_path.exists():
        print(f"  [skip] {arch_id}: ckpt missing at {ckpt_path}")
        return None
    if not sel_path.exists():
        print(f"  [skip] {arch_id}: feature_selection.json missing at {sel_path}")
        return None
    meta = json.loads(log_path.read_text())
    selection = json.loads(sel_path.read_text())
    src_class = meta["src_class"]
    is_mlc = src_class in MLC_CLASSES
    print(f"  arch={arch_id}  src_class={src_class}  is_mlc={is_mlc}")

    device = torch.device("cuda")

    if share_acts is None or is_mlc:
        # capture activations
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(SUBJECT_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        subject = AutoModelForCausalLM.from_pretrained(
            SUBJECT_MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
        )
        subject.eval()
        for p in subject.parameters():
            p.requires_grad_(False)

        # Build sentence list (30 concepts × 5 examples = 150)
        sentences, origins = [], []
        for ci, c in enumerate(CONCEPTS):
            for s in c["examples"]:
                sentences.append(s)
                origins.append(ci)
        print(f"    forwarding {len(sentences)} concept-sentences "
              f"({'L10-L14 multi-layer' if is_mlc else 'L12 anchor'}) ...")
        t0 = time.time()
        acts, attn = _capture_activations(
            sentences, subject, tokenizer, device, is_mlc=is_mlc,
            batch_size=32, max_length=CONCEPT_MAX_LENGTH,
        )
        print(f"    capture done in {time.time() - t0:.1f}s")
        del subject
        torch.cuda.empty_cache()
        gc.collect()
        if not is_mlc:
            share_acts = (acts, attn, origins)
    else:
        acts, attn, origins = share_acts

    print(f"    loading {arch_id} ckpt...")
    sae, _ = _load_phase7_model(meta, ckpt_path, device)
    T = window_T(sae, src_class, meta)
    print(f"    T={T}")

    N = len(origins)
    S = acts.shape[1]
    n_concepts = len(CONCEPTS)

    # We'll collect z[picked_feat] per content position per example.
    # Because each concept has its own picked feature, we encode all
    # examples of that concept at once and slice the picked column.
    per_concept_values: list[np.ndarray] = []
    arch_z_summary: dict = {}
    pos_idx = torch.arange(S, device=device)
    for ci, c in enumerate(CONCEPTS):
        ex_idx = [i for i, o in enumerate(origins) if o == ci]
        if not ex_idx:
            continue
        x = torch.from_numpy(acts[ex_idx]).float().to(device)  # (n_ex, S, ...)
        z = encode_per_position(sae, src_class, x, T=T)         # (n_ex, S, d_sae)
        m = torch.from_numpy(attn[ex_idx].astype(np.float32)).to(device)  # (n_ex, S)
        if T > 1 and not is_mlc:
            m = m * (pos_idx >= T - 1).float().unsqueeze(0)
        feat_idx = int(selection["concepts"][c["id"]]["best_feature_idx"])
        z_feat = z[:, :, feat_idx]                              # (n_ex, S)
        # Mask non-content positions
        mask = m > 0
        # Flatten masked values
        vals = z_feat[mask].detach().cpu().numpy().astype(np.float32)
        per_concept_values.append(vals)
        arch_z_summary[c["id"]] = {
            "feature_idx": feat_idx,
            "best_lift_train": float(selection["concepts"][c["id"]]["best_lift"]),
            "best_act_train": float(selection["concepts"][c["id"]]["best_activation"]),
            **_summarize(vals),
        }
        del x, z, m, z_feat, mask
    del sae
    torch.cuda.empty_cache()
    gc.collect()

    # Aggregate over concepts: pool all picked-feature activations into one vector
    if per_concept_values:
        pooled = np.concatenate(per_concept_values, axis=0)
    else:
        pooled = np.array([], dtype=np.float32)

    arch_summary = {
        "arch_id": arch_id,
        "src_class": src_class,
        "T": T,
        "is_mlc": is_mlc,
        "n_concepts": len(per_concept_values),
        "pooled": _summarize(pooled),
        "per_concept": arch_z_summary,
    }
    return arch_summary, share_acts, per_concept_values


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=list(DEFAULT_ARCHS))
    ap.add_argument("--out-dir", default=str(CASE_STUDIES_DIR / "diagnostics"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    banner(__file__)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    raw_per_concept = {}
    share_acts = None  # cache L12 activations across non-MLC archs to save GPU
    for a in args.archs:
        print(f"\n=== {a} seed={args.seed} ===")
        result = diagnose_arch(a, share_acts=share_acts, seed=args.seed)
        if result is None:
            continue
        summary, share_acts_new, per_concept_arrays = result
        summaries[a] = summary
        # Persist raw per-concept arrays per arch
        raw_per_concept[a] = per_concept_arrays
        # Update share_acts only when we computed new ones
        if share_acts_new is not None:
            share_acts = share_acts_new

    # Write JSON summary
    summary_path = out_dir / "z_orig_magnitudes.json"
    summary_path.write_text(json.dumps(summaries, indent=2))
    print(f"\n  wrote {summary_path}")

    # Write raw arrays into a single npz: per-arch flattened pooled vector
    npz_path = out_dir / "z_orig_per_concept.npz"
    flat = {a: np.concatenate(v, axis=0) if v else np.array([])
            for a, v in raw_per_concept.items()}
    np.savez(npz_path, **flat)
    print(f"  wrote {npz_path}")

    # Print compact table
    print(f"\n{'arch':40s} {'T':>3s}  {'<|z|>':>9s}  {'p90':>9s}  {'p99':>9s}  {'max':>9s}")
    for a, s in summaries.items():
        p = s["pooled"]
        print(f"{a:40s} {s['T']:>3d}  {p['abs_mean']:>9.3f}  "
              f"{p['abs_p90']:>9.3f}  {p['abs_p99']:>9.3f}  {p['abs_max']:>9.3f}")


if __name__ == "__main__":
    main()
