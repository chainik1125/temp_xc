"""Verify NousResearch/Meta-Llama-3.1-8B is byte-identical to meta-llama/Llama-3.1-8B.

Per the prior author's 2026-05-04 directive (agent_back briefing mandate update OQ #2):
before any Meta-mirror substitution becomes a paper claim, run a
bit-equality sanity check. On 5 prompts × 50 tokens, do a forward pass
through both models and compare layer-10 residual activations
element-wise. If max abs diff < 1e-5, the mirror is paper-equivalent
and existing v3 sweep cells (which use the mirror) stay valid. If max
abs diff > 1e-5, must re-run on the canonical Meta datasource.

Usage:
    TQDM_DISABLE=1 .venv/bin/python -m experiments.c7_backtracking.check_mirror_equivalence

Output: prints per-prompt + overall max/mean abs diff. Exits 0 if
equivalence holds (max abs diff < 1e-5), 1 otherwise.

Memory: ~16 GB total (both models in bf16). One GPU. Sequential load:
load Meta → forward → cache acts → free → load NousResearch → forward
→ compare. Avoids holding both models simultaneously.
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import torch

LAYER = 10
PROMPTS = [
    "Solve this math problem and provide your final answer in \\boxed{} notation.\n\nProblem: What is the smallest positive integer n such that n^2 + n + 41 is composite?",
    "Three inhabitants of an island each either always tell the truth or always lie.",
    "Wait, I need to re-examine my assumption. Let me try a different approach.",
    "Hmm, let me think. The pattern suggests that for each k,",
    "Actually, the answer should be derived from the recurrence relation.",
]
N_TOKENS = 50  # truncate prompts to 50 tokens each
TOLERANCE = 1e-5

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("c7.mirror_check")


def _capture_l10(model_id: str, tokens: list[torch.Tensor]) -> list[torch.Tensor]:
    """Load model, forward each token batch, capture layer-10 residual.

    Returns a list of (1, n_tokens, d_in) tensors on CPU, one per prompt.
    """
    from temp_bench.utils.tokens import get_token
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("loading %s", model_id)
    hf_token = get_token("hf")
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, token=hf_token)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda", token=hf_token,
    ).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    captured: list[torch.Tensor] = []
    def hook_fn(_m, _i, output):
        x = output[0] if isinstance(output, tuple) else output
        captured.append(x.detach().to(torch.float32).cpu())

    handle = model.model.layers[LAYER].register_forward_hook(hook_fn)

    outs: list[torch.Tensor] = []
    try:
        for ids in tokens:
            captured.clear()
            with torch.no_grad():
                _ = model(ids.to(model.device))
            outs.append(captured[0])  # (1, n_tokens, d_in)
    finally:
        handle.remove()

    del model
    import gc; gc.collect()
    torch.cuda.empty_cache()
    return outs


def main() -> int:
    log.info("mirror equivalence check — meta-llama vs NousResearch")
    log.info("layer=%d, n_prompts=%d, n_tokens=%d, tolerance=%.0e",
             LAYER, len(PROMPTS), N_TOKENS, TOLERANCE)

    # Tokenize prompts using a shared tokenizer (NousResearch mirror — same
    # vocab as Meta). Truncate to N_TOKENS each.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "NousResearch/Meta-Llama-3.1-8B", use_fast=True,
    )
    tokens = []
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt", truncation=True,
                  max_length=N_TOKENS, add_special_tokens=False)["input_ids"]
        if ids.shape[1] < N_TOKENS:
            # Pad with the prompt itself to fill 50 tokens — or just keep short.
            pass
        tokens.append(ids[:, :N_TOKENS])
    log.info("tokenized %d prompts; lengths: %s",
             len(tokens), [t.shape[1] for t in tokens])

    # Capture from Meta first.
    meta_acts = _capture_l10("meta-llama/Llama-3.1-8B", tokens)
    # Then from NousResearch.
    nous_acts = _capture_l10("NousResearch/Meta-Llama-3.1-8B", tokens)

    # Compare element-wise.
    max_diff_overall = 0.0
    sum_sq = 0.0
    n_total = 0
    log.info("--- per-prompt comparison ---")
    for i, (a, b) in enumerate(zip(meta_acts, nous_acts)):
        d = (a - b).abs()
        m = float(d.max().item())
        mn = float(d.mean().item())
        max_diff_overall = max(max_diff_overall, m)
        sum_sq += float((a - b).pow(2).sum().item())
        n_total += a.numel()
        log.info("  prompt %d: shape=%s, max_abs_diff=%.3e, mean_abs_diff=%.3e",
                 i, tuple(a.shape), m, mn)

    rmse = float(np.sqrt(sum_sq / max(1, n_total)))
    log.info("=" * 60)
    log.info("OVERALL: max_abs_diff=%.3e, rmse=%.3e", max_diff_overall, rmse)
    log.info("tolerance: %.3e", TOLERANCE)

    if max_diff_overall < TOLERANCE:
        log.info("PASS — mirror is byte-equivalent to Meta. v3 cells stay valid.")
        return 0
    log.error("FAIL — mirror differs from Meta by %.3e > %.3e",
              max_diff_overall, TOLERANCE)
    log.error("  → v3 cells (trained on mirror) need re-run on canonical datasource.")
    log.error("  → switch DATASOURCE in experiments/c7_backtracking/run.py from")
    log.error("    'llama_3_1_8b_base_l10_ward_nousmirror' to 'llama_3_1_8b_base_l10_ward'.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
