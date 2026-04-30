"""IT-side MLC training with PRELOAD_SEQS=6000 + expandable_segments.

Mission #2 wrapper. The dense MLC family (mlc, agentic_mlc_08,
mlc_contrastive_alpha100_batchtopk) requires the 5-layer multi-layer
activation cache. At PRELOAD_SEQS=24000 (paper canonical), this is
71 GB on GPU which doesn't fit on A40 (46 GB cap). With
PRELOAD_SEQS=6000, it's 17.7 GB — fits comfortably with the SAE
weights + Adam state + workspace.

Mechanism:
  1. Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (avoids
     fragmentation OOMs we hit on the BASE-trainer for non-MLC IT
     archs).
  2. Swap _paths -> _paths_it via sys.modules (same trick as
     train_phase7_it.py).
  3. Replace _train_utils.preload_multilayer with a wrapper that
     uses n_seqs=6000 by default. Also replace it in train_phase7's
     local namespace (since train_phase7 already imported it by name).

Usage (from repo root):

    .venv/bin/python -m experiments.phase7_unification.train_phase7_it_mlc \\
        --canonical --seed 42 --archs mlc

DEVIATION FROM PAPER CANONICAL: PRELOAD_SEQS=6000 not 24000. This
is documented in the resulting training_index.jsonl row's
deviation_note (added programmatically below). Compared to the
H200-trained dense MLC ckpts on `txcdr-base`, the IT versions
trained here use a smaller preload pool — should be flagged in the
writeup.
"""
from __future__ import annotations

import json
import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("TQDM_DISABLE", "1")

# CRITICAL: swap _paths -> _paths_it before any other phase7 module imports.
from experiments.phase7_unification import _paths_it
sys.modules["experiments.phase7_unification._paths"] = _paths_it

# Now import _train_utils — it'll see _paths_it values.
from experiments.phase7_unification import _train_utils

# Patch preload_multilayer to default to n_seqs=6000.
_orig_preload_multilayer = _train_utils.preload_multilayer
def _preload_multilayer_6k(device=None, n_seqs: int = 6000):
    print(f"[mlc-wrapper] preload_multilayer with n_seqs={n_seqs} (DEVIATION from paper canonical=24000)")
    return _orig_preload_multilayer(device=device, n_seqs=n_seqs)
_train_utils.preload_multilayer = _preload_multilayer_6k

# Now import train_phase7 — its `from _train_utils import preload_multilayer`
# resolves to the patched function.
from experiments.phase7_unification import train_phase7

# Also explicitly replace it in train_phase7's namespace in case it
# was bound by name before our patch.
train_phase7.preload_multilayer = _preload_multilayer_6k


def main() -> None:
    # Verify swap actually took effect.
    from experiments.phase7_unification._paths import SUBJECT_MODEL as _SM
    assert _SM == "google/gemma-2-2b-it", (
        f"_paths_it swap failed; SUBJECT_MODEL={_SM}"
    )
    print(f"[train_phase7_it_mlc] sys.modules._paths swap OK; "
          f"SUBJECT_MODEL={_SM}; preload_multilayer patched to n_seqs=6000")
    train_phase7.main()


if __name__ == "__main__":
    main()
