"""Download seed=42 ckpts from HF for the 6-arch case-study shortlist.

The training_logs/ subdir is committed; ckpts/ is not (too large). Calling
this script before any case-study driver ensures local ckpts exist at the
paths `_load_phase7_model` expects.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

from experiments.phase7_unification._paths import CKPT_DIR, HF_CKPT_REPO

DEFAULT_ARCHS = (
    "topk_sae",
    "tsae_paper_k20",
    "tsae_paper_k500",
    "mlc_contrastive_alpha100_batchtopk",
    "agentic_txc_02",
    "phase5b_subseq_h8",
    "phase57_partB_h8_bare_multidistance_t5",
)


def ensure_local(arch_id: str, seed: int = 42) -> Path:
    """Ensure ckpt is at CKPT_DIR / f'{arch_id}__seed{seed}.pt'; return path."""
    from huggingface_hub import hf_hub_download

    fname = f"{arch_id}__seed{seed}.pt"
    local = CKPT_DIR / fname
    if local.exists():
        return local
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {fname} from {HF_CKPT_REPO}...")
    p = hf_hub_download(
        repo_id=HF_CKPT_REPO,
        filename=f"ckpts/{fname}",
        local_dir=CKPT_DIR.parent,
        local_dir_use_symlinks=False,
    )
    # hf_hub_download will place it at <local_dir>/ckpts/<fname>; that's exactly
    # CKPT_DIR / fname because CKPT_DIR.parent / "ckpts" == CKPT_DIR. Verify.
    p = Path(p)
    if p != local:
        # Fallback: if HF Hub returns a cache symlink instead of a real file,
        # copy or symlink it into the expected path.
        if not local.exists():
            local.symlink_to(p)
    return local


def main() -> None:
    archs = sys.argv[1:] if len(sys.argv) > 1 else list(DEFAULT_ARCHS)
    print(f"Ensuring local seed=42 ckpts for {len(archs)} archs at {CKPT_DIR}")
    for a in archs:
        p = ensure_local(a)
        size_mb = p.stat().st_size / (1024 * 1024) if p.exists() else 0
        print(f"  {a}: {p}  ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
