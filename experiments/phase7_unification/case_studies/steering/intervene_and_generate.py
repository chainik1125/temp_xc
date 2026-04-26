"""AxBench-style additive steering on Gemma-2-2b base, batched across strengths.

For each (arch, concept) in `feature_selection.json`:

    x_steered_t = x_t + strength * unit_norm(d_dec[best_feature_idx])

applied at L12 residual stream for every token position via a forward
hook on `model.model.layers[12]`. Generates 60 tokens from a fixed
neutral prompt ("We find") using greedy decoding (temperature=0). All
8 strengths share a single batched generate() call, with each batch
element receiving its own additive bias.

Note on intervention modality: agent_c_brief.md specifies AxBench-style
RESIDUAL-DIRECTION steering (decoder-direction multiplier), not the
T-SAE paper's clamp-on-latent-with-error-add (which uses absolute clamp
values 10..15000). Decoder-direction additive applies uniformly across
arch families, matches AxBench's standard protocol, and is what the
brief's strength schedule {0.5, 1, 2, 4, 8, 12, 16, 24} is calibrated
against.

Output: per arch
    results/case_studies/steering/<arch_id>/generations.jsonl
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
)
from experiments.phase7_unification.case_studies._paths import (
    CASE_STUDIES_DIR, STAGE_1_ARCHS, SUBJECT_MODEL, ANCHOR_LAYER,
    STEERING_STRENGTHS, STEERING_GEN_TOKENS, STEERING_PROMPT,
)
from experiments.phase7_unification.case_studies._arch_utils import (
    decoder_direction_matrix, MLC_CLASSES,
)


def steer_for_arch(arch_id: str, *, force: bool = False) -> None:
    log_path = OUT_DIR / "training_logs" / f"{arch_id}__seed42.json"
    ckpt_path = OUT_DIR / "ckpts" / f"{arch_id}__seed42.pt"
    sel_path = CASE_STUDIES_DIR / "steering" / arch_id / "feature_selection.json"
    out_path = CASE_STUDIES_DIR / "steering" / arch_id / "generations.jsonl"

    if not sel_path.exists():
        print(f"  [skip] {arch_id}: feature_selection.json missing — "
              f"run select_features first")
        return
    if out_path.exists() and not force:
        print(f"  [skip] {arch_id}: generations.jsonl exists (use --force)")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = json.loads(log_path.read_text())
    selection = json.loads(sel_path.read_text())
    src_class = meta["src_class"]
    if src_class in MLC_CLASSES:
        print(f"  [skip] {arch_id}: MLC needs multi-layer cache")
        return

    device = torch.device("cuda")

    # ── 1. Load arch, extract decoder-direction matrix (d_in, d_sae) ────
    print(f"  loading {arch_id} ckpt for decoder dir extraction...")
    sae, _ = _load_phase7_model(meta, ckpt_path, device)
    D = decoder_direction_matrix(sae, src_class).to(device)        # (d_in, d_sae)
    D_unit = D / D.norm(dim=0, keepdim=True).clamp(min=1e-8)
    del sae
    torch.cuda.empty_cache()
    gc.collect()
    print(f"    D_unit shape={tuple(D_unit.shape)} (d_in, d_sae)")

    # ── 2. Load subject Gemma-2-2b base for steered generation ───────────
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

    strengths = list(STEERING_STRENGTHS)
    B = len(strengths)
    strengths_t = torch.tensor(strengths, dtype=torch.float32, device=device)

    # Per-call mutable state for the hook.
    state = {"bias_per_batch": None}                                # (B, d_in)

    def hook(module, inp, output):
        h = output[0] if isinstance(output, tuple) else output      # (B, S, d_in)
        bias = state["bias_per_batch"]                              # (B, d_in)
        if bias is None:
            return None
        b_dtype = h.dtype
        bias_t = bias.to(b_dtype).to(h.device).unsqueeze(1)         # (B, 1, d_in)
        h_steered = h + bias_t
        if isinstance(output, tuple):
            return (h_steered,) + output[1:]
        return h_steered

    handle = subject.model.layers[ANCHOR_LAYER].register_forward_hook(hook)

    # Prompt: tokenize once, replicate B times.
    enc = tokenizer(STEERING_PROMPT, return_tensors="pt", add_special_tokens=True)
    prompt_ids = enc["input_ids"].to(device).repeat(B, 1)            # (B, P)
    prompt_attn = enc["attention_mask"].to(device).repeat(B, 1)      # (B, P)
    prompt_len = prompt_ids.shape[1]

    n_concepts = len(selection["concepts"])
    print(f"  generating: {n_concepts} concepts * {B} strengths = "
          f"{n_concepts * B} samples (batched per concept)")
    t0 = time.time()
    try:
        with open(out_path, "w") as f_out:
            for ci, (concept_id, sel_data) in enumerate(selection["concepts"].items()):
                feature_idx = int(sel_data["best_feature_idx"])
                d_unit = D_unit[:, feature_idx]                      # (d_in,)
                state["bias_per_batch"] = (
                    strengths_t.unsqueeze(1) * d_unit.unsqueeze(0)   # (B, d_in)
                )
                with torch.no_grad():
                    out_ids = subject.generate(
                        prompt_ids, attention_mask=prompt_attn,
                        max_new_tokens=STEERING_GEN_TOKENS,
                        do_sample=False,                             # greedy
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
                    }) + "\n")
                f_out.flush()
                if (ci + 1) % 5 == 0 or ci + 1 == n_concepts:
                    elapsed = time.time() - t0
                    rate = (ci + 1) / max(elapsed, 1e-3)
                    eta = (n_concepts - ci - 1) / max(rate, 1e-3)
                    print(f"    [{ci + 1}/{n_concepts}] {rate:.1f} concept/s  ETA {eta:.0f}s")
    finally:
        handle.remove()
        del subject
        torch.cuda.empty_cache()
        gc.collect()
    print(f"  saved {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=list(STAGE_1_ARCHS))
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing generations.jsonl")
    args = ap.parse_args()
    banner(__file__)
    for arch_id in args.archs:
        print(f"\n=== {arch_id} ===")
        steer_for_arch(arch_id, force=args.force)


if __name__ == "__main__":
    main()
