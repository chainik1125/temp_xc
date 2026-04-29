"""Per-arch, per-concept best-feature selection for Agent C's case study C.ii.

Implements the brief's "fallback path" for the apples-to-apples steering
protocol: for each (arch, concept) pair, find the feature with the highest
mean activation on a fixed concept-annotated sample. The "best feature for
this concept in this arch" is then the steering target. The same concept
set is used across all archs, so cross-arch comparisons stay fair.

Pipeline per arch:
  1. Tokenize all 30 * 5 = 150 concept example sentences through the
     Gemma-2-2b base tokenizer (max_length=64, right-padded).
  2. Forward through Gemma-2-2b base, hook L12 to capture per-token
     residual stream activations (B, S, d_in=2304).
  3. Encode through the arch via `_arch_utils.encode_per_position` —
     window archs slide T-tokens with stride 1 and attribute the latent
     to the right edge; per-token archs encode every position.
  4. Per concept: average per-position features over CONTENT tokens
     (attention_mask==1) AND, for window archs, positions >= T-1.
     Result: a (n_concepts, d_sae) activation matrix per arch.
  5. Best feature per concept = argmax over d_sae axis. Also save the
     top-5 candidates for each concept (for spot-checking cases where
     argmax picks an uninterpretable feature).

Output per arch (under `results/case_studies/steering/<arch_id>/`):
  feature_selection.json
    arch_id, src_class, concepts: {concept_id: {best_idx, best_act, top_5}, ...}
  concept_activations.npz
    (30, d_sae) fp32 — full per-concept activation matrix.
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
)
from experiments.phase7_unification.case_studies._paths import (
    CASE_STUDIES_DIR, STAGE_1_ARCHS, SUBJECT_MODEL, ANCHOR_LAYER, DEFAULT_D_IN,
)
from experiments.phase7_unification.case_studies._arch_utils import (
    encode_per_position, window_T, _d_sae_of, MLC_CLASSES,
)
from experiments.phase7_unification.case_studies.steering.concepts import CONCEPTS


CONCEPT_MAX_LENGTH = 64                    # short example sentences
TOP_N_CANDIDATES = 5                       # per-concept fallback pool


def _capture_l12_activations(
    sentences: list[str], model, tokenizer, device: torch.device,
    batch_size: int = 32, max_length: int = CONCEPT_MAX_LENGTH,
    hook_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward `sentences` through Gemma-2-2b base; return (acts, attn_mask).

    acts:  (N, max_length, d_in) fp16 — L12 activations from `hook_name`.
    attn:  (N, max_length) int8 — 1 for content, 0 for pad.

    `hook_name` selects the submodule. None = layer output (resid_post,
    Phase 7 default). "input_layernorm" = ln1 output (Y's ln1-pivot).
    """
    n = len(sentences)
    acts = np.zeros((n, max_length, DEFAULT_D_IN), dtype=np.float16)
    attn = np.zeros((n, max_length), dtype=np.int8)
    captured: dict[int, torch.Tensor] = {}
    def hook(module, inp, output):
        h = output[0] if isinstance(output, tuple) else output
        captured[ANCHOR_LAYER] = h.detach().cpu()
    if hook_name is None:
        target = model.model.layers[ANCHOR_LAYER]
    elif hook_name == "input_layernorm":
        target = model.model.layers[ANCHOR_LAYER].input_layernorm
    else:
        raise ValueError(f"unknown hook_name={hook_name!r}")
    handle = target.register_forward_hook(hook)
    try:
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            chunk = sentences[start:end]
            enc = tokenizer(
                chunk, return_tensors="pt",
                padding="max_length", truncation=True, max_length=max_length,
            )
            captured.clear()
            with torch.no_grad():
                model(enc["input_ids"].to(device),
                      attention_mask=enc["attention_mask"].to(device))
            h = captured[ANCHOR_LAYER]
            if h.shape[-1] != DEFAULT_D_IN:
                h = h[..., :DEFAULT_D_IN]
            acts[start:end] = h.to(torch.float16).numpy()
            attn[start:end] = enc["attention_mask"].to(torch.int8).numpy()
    finally:
        handle.remove()
    return acts, attn


def _capture_multilayer_activations(
    sentences: list[str], model, tokenizer, device: torch.device,
    batch_size: int = 32, max_length: int = CONCEPT_MAX_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward `sentences` through Gemma-2-2b base capturing L10–L14
    residuals for MLC archs. Returns:

    acts:  (N, max_length, n_layers=5, d_in) fp16 — multi-layer cube.
    attn:  (N, max_length) int8.
    """
    n = len(sentences)
    n_lay = len(MLC_LAYERS)
    acts = np.zeros((n, max_length, n_lay, DEFAULT_D_IN), dtype=np.float16)
    attn = np.zeros((n, max_length), dtype=np.int8)
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for li in MLC_LAYERS:
        def make_hook(layer_idx: int):
            def hook(module, inp, output):
                h = output[0] if isinstance(output, tuple) else output
                captured[layer_idx] = h.detach().cpu()
            return hook
        handles.append(model.model.layers[li].register_forward_hook(make_hook(li)))
    try:
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            chunk = sentences[start:end]
            enc = tokenizer(
                chunk, return_tensors="pt",
                padding="max_length", truncation=True, max_length=max_length,
            )
            captured.clear()
            with torch.no_grad():
                model(enc["input_ids"].to(device),
                      attention_mask=enc["attention_mask"].to(device))
            for idx, li in enumerate(MLC_LAYERS):
                h = captured[li]
                if h.shape[-1] != DEFAULT_D_IN:
                    h = h[..., :DEFAULT_D_IN]
                acts[start:end, :, idx, :] = h.to(torch.float16).numpy()
            attn[start:end] = enc["attention_mask"].to(torch.int8).numpy()
    finally:
        for handle in handles:
            handle.remove()
    return acts, attn


def select_for_arch(arch_id: str, *, batch_size: int = 16, seed: int = 42) -> dict | None:
    """Per-arch best-feature-per-concept selection."""
    log_path = OUT_DIR / "training_logs" / f"{arch_id}__seed{seed}.json"
    ckpt_path = OUT_DIR / "ckpts" / f"{arch_id}__seed{seed}.pt"
    if not ckpt_path.exists() or not log_path.exists():
        print(f"  [skip] {arch_id}: ckpt or log missing")
        return None
    meta = json.loads(log_path.read_text())
    src_class = meta["src_class"]
    is_mlc = src_class in MLC_CLASSES

    sel_subdir = "steering" if seed == 42 else f"steering_seed{seed}"
    out_dir = CASE_STUDIES_DIR / sel_subdir / arch_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  loading subject model {SUBJECT_MODEL} (bf16)...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(SUBJECT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    subject = AutoModelForCausalLM.from_pretrained(
        SUBJECT_MODEL, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    subject.eval()
    for p in subject.parameters():
        p.requires_grad_(False)
    device = torch.device("cuda")

    # Flatten concept sentences with origin tags for per-concept aggregation.
    sentences: list[str] = []
    origins: list[int] = []                              # concept index per sentence
    for ci, c in enumerate(CONCEPTS):
        for s in c["examples"]:
            sentences.append(s)
            origins.append(ci)
    print(f"  forwarding {len(sentences)} concept-sentences "
          f"(30 concepts * {len(CONCEPTS[0]['examples'])} examples; "
          f"{'L10-L14 multi-layer' if is_mlc else 'L12 anchor'}) ...")
    t0 = time.time()
    if is_mlc:
        acts, attn = _capture_multilayer_activations(
            sentences, subject, tokenizer, device, batch_size=32,
        )
    else:
        acts, attn = _capture_l12_activations(
            sentences, subject, tokenizer, device, batch_size=32,
            hook_name=meta.get("hook_name"),
        )
    print(f"    capture done in {time.time() - t0:.1f}s; acts {acts.shape}")
    del subject
    torch.cuda.empty_cache()
    gc.collect()

    print(f"  loading {arch_id} ckpt...")
    sae, _ = _load_phase7_model(meta, ckpt_path, device)
    T = window_T(sae, src_class, meta)
    d_sae = _d_sae_of(sae, src_class)

    # Per-concept activation aggregation. Sum + count, divide at end.
    n_concepts = len(CONCEPTS)
    sums = np.zeros((n_concepts, d_sae), dtype=np.float64)
    counts = np.zeros((n_concepts,), dtype=np.float64)
    print(f"  encoding via {src_class} (T={T}, d_sae={d_sae})...")
    N = len(sentences)
    S = acts.shape[1]
    pos_idx = torch.arange(S, device=device)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x = torch.from_numpy(acts[start:end]).float().to(device)  # (B, S, d_in) or (B, S, n_lay, d_in)
        z = encode_per_position(sae, src_class, x, T=T)            # (B, S, d_sae)
        m = torch.from_numpy(attn[start:end].astype(np.float32)).to(device)  # (B, S)
        if T > 1 and not is_mlc:
            m = m * (pos_idx >= T - 1).float().unsqueeze(0)
        # sum_per_example_per_feature: (B, d_sae) ; count_per_example: (B,)
        m_b = m.unsqueeze(-1)                                     # (B, S, 1)
        sums_b = (z * m_b).sum(dim=1).cpu().numpy().astype(np.float64)
        counts_b = m.sum(dim=1).cpu().numpy().astype(np.float64)
        for i, ex_idx in enumerate(range(start, end)):
            ci = origins[ex_idx]
            sums[ci] += sums_b[i]
            counts[ci] += counts_b[i]
        del z, x, m, m_b
    del sae
    torch.cuda.empty_cache()

    # Per-concept mean activation vectors.
    counts_safe = np.maximum(counts, 1.0)
    mean_acts = sums / counts_safe[:, None]                       # (n_concepts, d_sae)

    # Always-on / dense features fire ~equally on all 30 concepts and look
    # spuriously "good" for every concept under raw-activation ranking.
    # Use excess-over-baseline: lift[c, j] = mean_acts[c, j] - <mean_j across concepts>.
    # A feature that fires SELECTIVELY on concept c gets a large positive lift;
    # an always-on feature gets ~0.
    baseline = mean_acts.mean(axis=0)                             # (d_sae,)
    lift = mean_acts - baseline[None, :]                          # (n_concepts, d_sae)

    # Per-concept best + top-5 ranked by lift (selectivity), with raw
    # activation also reported for sanity.
    selection = {}
    for ci, c in enumerate(CONCEPTS):
        v = lift[ci]
        order = np.argsort(-v)
        top_idx = [int(j) for j in order[:TOP_N_CANDIDATES]]
        top_lifts = [float(lift[ci, j]) for j in top_idx]
        top_acts = [float(mean_acts[ci, j]) for j in top_idx]
        top_baselines = [float(baseline[j]) for j in top_idx]
        selection[c["id"]] = {
            "best_feature_idx": top_idx[0],
            "best_lift": top_lifts[0],
            "best_activation": top_acts[0],
            "best_baseline": top_baselines[0],
            "top_5": [
                {"feature_idx": j, "lift": l, "activation": a, "baseline": b}
                for j, l, a, b in zip(top_idx, top_lifts, top_acts, top_baselines)
            ],
            "concept_description": c["description"],
            "n_content_tokens": int(counts[ci]),
        }

    out = {
        "arch_id": arch_id,
        "src_class": src_class,
        "T": T,
        "d_sae": int(d_sae),
        "k_pos": meta.get("k_pos"),
        "k_win": meta.get("k_win"),
        "n_concepts": n_concepts,
        "concept_max_length": CONCEPT_MAX_LENGTH,
        "concepts": selection,
    }
    (out_dir / "feature_selection.json").write_text(json.dumps(out, indent=2))
    np.savez(out_dir / "concept_activations.npz",
             mean_acts=mean_acts.astype(np.float32),
             counts=counts.astype(np.int32))

    print(f"  saved {out_dir}/feature_selection.json + concept_activations.npz")
    print(f"  per-concept best feature (ranked by lift = act - baseline):")
    for ci, c in enumerate(CONCEPTS):
        sel = selection[c["id"]]
        print(f"    {c['id']:24s} -> feat {sel['best_feature_idx']:6d} "
              f"lift={sel['best_lift']:+7.3f}  act={sel['best_activation']:7.3f}  "
              f"base={sel['best_baseline']:7.3f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=list(STAGE_1_ARCHS))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    banner(__file__)
    for arch_id in args.archs:
        print(f"\n=== {arch_id} seed={args.seed} ===")
        select_for_arch(arch_id, batch_size=args.batch_size, seed=args.seed)


if __name__ == "__main__":
    main()
