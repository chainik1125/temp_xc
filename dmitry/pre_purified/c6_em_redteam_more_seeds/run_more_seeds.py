"""run_more_seeds.py — wrapper around c6_em.run that applies the
vectorized batch_iter shim before running.

Han's `experiments/c6_em/run.py` is untouched. We import our
fast_batch_iter shim first (which monkey-patches
`experiments.c6_em.train._build_batch_iter`), then dispatch to the
canonical run.main().

Usage (same args as `python -m experiments.c6_em.run`):
    python run_more_seeds.py --archs sae_arditi --seed 2 \
        --datasource qwen_2_5_14b_instruct_finance_l24_resid_post
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Resolve worktree (auto-detect by walking up from this file).
def _find_worktree() -> Path:
    cand = os.environ.get("C6_WORKTREE")
    if cand and (Path(cand) / "purified" / "src").exists():
        return Path(cand)
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "purified" / "src").exists():
            return p
    for p in ("/workspace/temp_xc-c6-extend", "/tmp/c6_redteam_wt"):
        if (Path(p) / "purified" / "src").exists():
            return Path(p)
    raise SystemExit("No worktree with purified/ found.")

WORKTREE = _find_worktree()
sys.path.insert(0, str(WORKTREE / "purified" / "src"))
sys.path.insert(0, str(WORKTREE / "purified"))
# Also ensure this script's dir is importable so fast_batch_iter sits next to us.
sys.path.insert(0, str(Path(__file__).parent.resolve()))

# Apply the vectorized batch_iter monkey-patch BEFORE c6_em.run is imported.
import fast_batch_iter  # noqa: F401 — import-for-side-effect

# Now dispatch to the canonical c6 driver.
from experiments.c6_em.run import main as c6_main

if __name__ == "__main__":
    sys.exit(c6_main())
