"""AxBench-additive steering at the paper-magnitude range, signed strengths.

Same mechanism as Han's intervene_and_generate.py:

    x_steered_t = x_t + strength * unit_norm(d_dec[best_feature_idx])

applied at L12 of Gemma-2-2b base. Difference is the strength schedule —
9 coarse points spanning negative and positive multipliers:

    {-100, -50, -25, -10, 0, 10, 25, 50, 100}

This complements:
  * Han's original (0.5-24, positive only), in steering/<arch>/
  * Paper-protocol clamp (10-15000, absolute), in steering_paper/<arch>/

By including 0 we get a no-op control row; by including negative
strengths we can ask whether reverse-direction steering ablates the
concept or steers toward an opposite. By going to ±100 (vs Han's max 24)
we're in roughly the same magnitude regime as the paper-clamp run after
accounting for unit-norm decoder rescaling.

Output: results/case_studies/steering_axbench_extended/<arch_id>/generations.jsonl

Same arch coverage as Han's script (per-token + window + MLC archs all
work — the additive intervention is uniform across families).
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
    decoder_direction_matrix, MLC_CLASSES,
)
from experiments.phase7_unification.case_studies._paths import (
    CASE_STUDIES_DIR, SUBJECT_MODEL, ANCHOR_LAYER,
    STEERING_GEN_TOKENS, STEERING_PROMPT,
)


EXTENDED_STRENGTHS = (-100, -50, -25, -10, 0, 10, 25, 50, 100)
OUT_SUBDIR = "steering_axbench_extended"


def steer_for_arch(arch_id: str, *, force: bool = False, limit_concepts: int | None = None) -> None:
    log_path = OUT_DIR / "training_logs" / f"{arch_id}__seed42.json"
    ckpt_path = OUT_DIR / "ckpts" / f"{arch_id}__seed42.pt"
    sel_path = CASE_STUDIES_DIR / "steering" / arch_id / "feature_selection.json"
    out_path = CASE_STUDIES_DIR / OUT_SUBDIR / arch_id / "generations.jsonl"

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

    device = torch.device("cuda")
    print(f"  loading {arch_id} ckpt ({src_class}) for decoder-dir extraction...")
    sae, _ = _load_phase7_model(meta, ckpt_path, device)
    D = decoder_direction_matrix(sae, src_class).to(device)        # (d_in, d_sae)
    D_unit = D / D.norm(dim=0, keepdim=True).clamp(min=1e-8)
    del sae
    torch.cuda.empty_cache()
    gc.collect()
    print(f"    D_unit shape={tuple(D_unit.shape)} (d_in, d_sae)")

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

    strengths = list(EXTENDED_STRENGTHS)
    B = len(strengths)
    strengths_t = torch.tensor(strengths, dtype=torch.float32, device=device)

    state = {"bias_per_batch": None}

    def hook(module, inp, output):
        h = output[0] if isinstance(output, tuple) else output
        bias = state["bias_per_batch"]
        if bias is None:
            return None
        bias_t = bias.to(h.dtype).to(h.device).unsqueeze(1)
        h_steered = h + bias_t
        if isinstance(output, tuple):
            return (h_steered,) + output[1:]
        return h_steered

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
          f"{n_concepts * B} samples (AxBench additive, signed strengths)")
    t0 = time.time()
    try:
        with open(out_path, "w") as f_out:
            for ci, (concept_id, sel_data) in enumerate(concepts_items):
                feature_idx = int(sel_data["best_feature_idx"])
                d_unit = D_unit[:, feature_idx]                      # (d_in,)
                state["bias_per_batch"] = (
                    strengths_t.unsqueeze(1) * d_unit.unsqueeze(0)   # (B, d_in)
                )
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
                        "intervention": "axbench_additive_extended",
                    }) + "\n")
                f_out.flush()
                if (ci + 1) % 5 == 0 or ci + 1 == n_concepts:
                    elapsed = time.time() - t0
                    rate = (ci + 1) / max(elapsed, 1e-3)
                    eta = (n_concepts - ci - 1) / max(rate, 1e-3)
                    print(f"    [{ci + 1}/{n_concepts}] {rate:.2f} concept/s  ETA {eta:.0f}s")
    finally:
        handle.remove()
        del subject
        torch.cuda.empty_cache()
        gc.collect()
    print(f"  saved {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--archs", nargs="+", default=[
            "topk_sae", "tsae_paper_k20", "tsae_paper_k500",
            "agentic_txc_02", "phase5b_subseq_h8",
            "phase57_partB_h8_bare_multidistance_t5",
        ],
    )
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit-concepts", type=int, default=None)
    args = ap.parse_args()
    banner(__file__)
    for arch_id in args.archs:
        print(f"\n=== {arch_id} (AxBench-additive, extended range) ===")
        steer_for_arch(arch_id, force=args.force, limit_concepts=args.limit_concepts)


if __name__ == "__main__":
    main()
