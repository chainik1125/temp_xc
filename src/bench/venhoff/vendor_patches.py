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


JUDGE_MODEL_VENHOFF = "openai/gpt-5.2"
JUDGE_MODEL_OURS = "anthropic/claude-haiku-4.5"


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


# Phase 2 patch: newer transformers (≥4.45 or so) rejects the bare
# `load_in_8bit=` kwarg on AutoModelForCausalLM.from_pretrained, even
# when False. Drop the kwarg since we run in bf16 (fits in 80GB H100
# comfortably for Llama-8B + DeepSeek-R1-Distill-Llama-8B). If we ever
# need 8-bit, the modern path is quantization_config=BitsAndBytesConfig(...).
STEERING_8BIT_PATCHES = [
    Patch(
        file_rel="train-vectors/optimize_steering_vectors.py",
        find="        load_in_8bit=args.load_in_8bit,\n",
        replace="",
        description="drop deprecated load_in_8bit= kwarg from AutoModelForCausalLM load",
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
