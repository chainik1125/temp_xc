"""Idempotent patches to vendored Venhoff scripts.

Each patch is a simple (file, find, replace) triple. We apply them
before invoking the target script via subprocess. Idempotent — if the
file already has the replacement string, the patch is a no-op.

Every patch is listed in `docs/aniket/experiments/venhoff_eval/VENHOFF_PROVENANCE.md`
under "deliberate deviations" for transparency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("venhoff.vendor_patches")


@dataclass(frozen=True)
class Patch:
    file_rel: str                  # path under venhoff_root
    find: str                      # literal string to replace (must be unique per occurrence)
    replace: str                   # new string
    description: str               # one-liner for logs + provenance
    optional: bool = False         # if True, skip silently when find is missing (for migration patches)


JUDGE_MODEL_VENHOFF = "openai/gpt-5.2"
# Dated API id — bare "claude-haiku-4.5" isn't a recognized model on
# the direct Anthropic API (only OpenRouter accepts the bare name).
# safe_chat_batch in hybrid_token.py routes via the anthropic SDK when
# the prefix is "anthropic/", so we use the SDK-accepted dated id.
JUDGE_MODEL_OURS = "claude-haiku-4-5-20251001"


HYBRID_JUDGE_PATCHES = [
    Patch(
        file_rel="hybrid/hybrid_token.py",
        find=f'model_name: str = "{JUDGE_MODEL_VENHOFF}"',
        replace=f'model_name: str = "{JUDGE_MODEL_OURS}"',
        description="hybrid_token.py safe_chat_batch default model",
    ),
    Patch(
        file_rel="hybrid/hybrid_token.py",
        find=f'safe_chat_batch([test_prompt], model_name="{JUDGE_MODEL_VENHOFF}", max_tokens=5)',
        replace=f'safe_chat_batch([test_prompt], model_name="{JUDGE_MODEL_OURS}", max_tokens=5)',
        description="hybrid_token.py API connectivity test",
    ),
    Patch(
        file_rel="hybrid/hybrid_token.py",
        find=f'safe_chat_batch(batch_prompts, model_name="{JUDGE_MODEL_VENHOFF}", max_tokens=2000)',
        replace=f'safe_chat_batch(batch_prompts, model_name="{JUDGE_MODEL_OURS}", max_tokens=2000)',
        description="hybrid_token.py per-answer grading call",
    ),
]


# Phase 2 patches for modern transformers compat:
#   - drop deprecated `load_in_8bit=` kwarg (use bf16 instead; Llama-8B
#     fits in 80GB H100).
#   - replace `tokenizer.encode_plus(..., return_offsets_mapping=True)`
#     with `tokenizer(..., return_offsets_mapping=True)` — encode_plus
#     was removed in recent transformers and only fast tokenizers
#     support offset mappings anyway.
STEERING_8BIT_PATCHES = [
    Patch(
        file_rel="train-vectors/optimize_steering_vectors.py",
        find="        load_in_8bit=args.load_in_8bit,\n",
        replace="",
        description="drop deprecated load_in_8bit= kwarg from AutoModelForCausalLM load",
    ),
    Patch(
        file_rel="utils/utils.py",
        find="    token_offsets = tokenizer.encode_plus(text, return_offsets_mapping=True)['offset_mapping']",
        replace="    token_offsets = tokenizer(text, return_offsets_mapping=True)['offset_mapping']",
        description="swap tokenizer.encode_plus(...) → tokenizer(...) (encode_plus dropped in new transformers)",
    ),
    Patch(
        file_rel="utils/utils.py",
        find="    model = LanguageModel(model_name, dispatch=True, load_in_8bit=load_in_8bit, device_map=device, dtype=torch.bfloat16)",
        replace="    model = LanguageModel(model_name, dispatch=True, device_map=device, dtype=torch.bfloat16)",
        description="drop load_in_8bit= kwarg from LanguageModel(...) in load_model() (Phase 3 hybrid inference path)",
    ),
    # Older shipped cluster-SAEs (pre-2026-04-23 activation_mean embedding)
    # don't contain the `activation_mean` key. The upstream assertion
    # forbids loading them. Synthesize a zero mean — equivalent to the
    # original uncentered-SAE behavior these checkpoints were trained with.
    Patch(
        file_rel="utils/sae.py",
        find=(
            "    if require_activation_mean:\n"
            "        # Require activation mean for downstream parity with SAE training.\n"
            "        assert \"activation_mean\" in checkpoint, (\n"
            "            \"SAE checkpoint is missing 'activation_mean'. This repo now requires SAE checkpoints to embed the \"\n"
            "            \"centering mean used for activation-cache construction so downstream usage can reproduce \"\n"
            "            \"centered+L2-normalized activations.\"\n"
            "        )"
        ),
        replace=(
            "    if require_activation_mean:\n"
            "        # Require activation mean for downstream parity with SAE training.\n"
            "        if \"activation_mean\" not in checkpoint:\n"
            "            # scrappy vendor patch: older shipped SAEs omit activation_mean;\n"
            "            # synthesize zeros (no centering) to match pre-embedding behavior.\n"
            "            import warnings\n"
            "            warnings.warn(\"SAE ckpt missing 'activation_mean'; synthesizing zero mean for backcompat\")\n"
            "            checkpoint[\"activation_mean\"] = torch.zeros(int(checkpoint[\"input_dim\"]))"
        ),
        description="soften activation_mean assertion — synthesize zero mean for pre-embedding shipped SAEs",
    ),
    # Migration: earlier pods have the broken v1 body on disk (uses
    # `'input_ids' in tensor.data`, which trips Tensor.__contains__).
    # Revert it back to upstream so the main v2 patch below applies cleanly.
    # On fresh clones this precondition won't match, and the patch no-ops.
    Patch(
        file_rel="hybrid/hybrid_token.py",
        find=(
            "    # Clone inputs so we do not modify the originals in-place.\n"
            "    # Modern transformers returns BatchEncoding from apply_chat_template(return_tensors='pt')\n"
            "    # in some configs; unwrap to Tensor before .clone() (BatchEncoding has no clone).\n"
            "    if hasattr(base_input_ids, 'data') and 'input_ids' in getattr(base_input_ids, 'data', {}):\n"
            "        base_input_ids = base_input_ids['input_ids']\n"
            "    if hasattr(thinking_input_ids, 'data') and 'input_ids' in getattr(thinking_input_ids, 'data', {}):\n"
            "        thinking_input_ids = thinking_input_ids['input_ids']\n"
            "    base_output_ids = base_input_ids.clone()\n"
            "    thinking_output_ids = thinking_input_ids.clone()\n"
            "    del base_input_ids, thinking_input_ids"
        ),
        replace=(
            "    # Clone inputs so we do not modify the originals in-place\n"
            "    base_output_ids = base_input_ids.clone()\n"
            "    thinking_output_ids = thinking_input_ids.clone()\n"
            "    del base_input_ids, thinking_input_ids"
        ),
        description="revert broken v1 BatchEncoding unwrap (uses Tensor.__contains__)",
        optional=True,
    ),
    Patch(
        file_rel="hybrid/hybrid_token.py",
        find=(
            "    # Clone inputs so we do not modify the originals in-place\n"
            "    base_output_ids = base_input_ids.clone()\n"
            "    thinking_output_ids = thinking_input_ids.clone()\n"
            "    del base_input_ids, thinking_input_ids"
        ),
        replace=(
            "    # Clone inputs so we do not modify the originals in-place.\n"
            "    # Modern transformers returns BatchEncoding from apply_chat_template(return_tensors='pt')\n"
            "    # in some configs; unwrap to Tensor before .clone() (BatchEncoding has no clone).\n"
            "    # hasattr('clone') is False on BatchEncoding because its __getattr__ raises\n"
            "    # AttributeError for unknown keys; True on torch.Tensor.\n"
            "    if not hasattr(base_input_ids, 'clone'):\n"
            "        base_input_ids = base_input_ids['input_ids']\n"
            "    if not hasattr(thinking_input_ids, 'clone'):\n"
            "        thinking_input_ids = thinking_input_ids['input_ids']\n"
            "    base_output_ids = base_input_ids.clone()\n"
            "    thinking_output_ids = thinking_input_ids.clone()\n"
            "    del base_input_ids, thinking_input_ids"
        ),
        description="unwrap BatchEncoding -> Tensor before .clone() in hybrid_generate_token",
    ),
]


def apply_patch(venhoff_root: Path, patch: Patch) -> bool:
    """Apply one patch. Returns True if the file was modified, False if no-op."""
    path = venhoff_root / patch.file_rel
    if not path.exists():
        raise FileNotFoundError(f"patch target missing: {path}")
    content = path.read_text()
    if patch.replace in content and patch.find not in content:
        # Already patched — no-op.
        return False
    if patch.find not in content:
        if patch.optional:
            log.info(
                "[info] vendor_patch_skipped_optional | file=%s | desc=%s | reason=find_not_present",
                patch.file_rel,
                patch.description,
            )
            return False
        raise ValueError(
            f"patch precondition failed: '{patch.find[:80]}...' not found in {path}. "
            "Venhoff may have changed the code; re-verify the commit pin in VENHOFF_PROVENANCE.md."
        )
    count = content.count(patch.find)
    if count != 1:
        raise ValueError(
            f"patch expected unique match but found {count} in {path}. "
            "Widen the `find` string to disambiguate."
        )
    path.write_text(content.replace(patch.find, patch.replace))
    log.info("[info] vendor_patch_applied | file=%s | desc=%s", patch.file_rel, patch.description)
    return True


def ensure_hybrid_judge_patched(venhoff_root: Path) -> None:
    """Replace hybrid_token.py's gpt-5.2 judge with claude-haiku-4.5.

    Motivation: gpt-5.2 via OpenRouter is ~5-10× more expensive than
    Haiku 4.5 with comparable Yes/No-grading quality. Venhoff's 3.5%
    baseline was computed with gpt-5.2; our numbers under Haiku 4.5
    are reported with that caveat (see plan.md § 5 P0 notes).
    """
    for patch in HYBRID_JUDGE_PATCHES:
        apply_patch(venhoff_root, patch)


def ensure_steering_patched(venhoff_root: Path) -> None:
    """Apply patches required for Phase 2 steering-vector training.

    Currently: drops the deprecated `load_in_8bit=` kwarg from
    optimize_steering_vectors.py so it works with modern transformers.
    """
    for patch in STEERING_8BIT_PATCHES:
        apply_patch(venhoff_root, patch)
