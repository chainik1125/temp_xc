"""IT-side Phase 7 training driver — thin wrapper.

Reuses every trainer function from `train_phase7.py` (TopKSAE, T-SAE
paper, TFA, TXCDR family, SubseqH8, etc.). Only difference: imports
the *paths* from `_paths_it` instead of `_paths`.

Mechanism: install `_paths_it` under the `_paths` module name in
`sys.modules` BEFORE importing `train_phase7` or `_train_utils`, so
all the bare-name imports inside those modules resolve to IT
constants:

    SUBJECT_MODEL  = "google/gemma-2-2b-it"
    ANCHOR_LAYER   = 13
    MLC_LAYERS     = (11, 12, 13, 14, 15)
    CACHE_DIR      = data/cached_activations/gemma-2-2b-it/fineweb
    HF_CKPT_REPO   = han1823123123/txcdr-it

Shared with BASE:
    OUT_DIR / INDEX_PATH / PROBING_PATH (training_index.jsonl is
    append-only across both subject models; rows are disambiguated by
    the meta `subject_model` field).

Usage (from repo root):

    .venv/bin/python -m experiments.phase7_unification.train_phase7_it \\
        --canonical --seed 42 \\
        --archs topk_sae,tfa_big,tsae_paper_k20,tsae_paper_k500,txcdr_t5,txcdr_t16,phase5b_subseq_h8,txc_bare_antidead_t5,phase57_partB_h8_bare_multidistance_t8

Or via a textfile:

    .venv/bin/python -m experiments.phase7_unification.train_phase7_it \\
        --canonical --seed 42 --archs /tmp/it_a40_ok.txt
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("TQDM_DISABLE", "1")

# CRITICAL: this swap MUST happen before any module that does
# `from experiments.phase7_unification._paths import ...` is imported.
from experiments.phase7_unification import _paths_it
sys.modules["experiments.phase7_unification._paths"] = _paths_it

# Now import the BASE train_phase7. Its `from ..._paths import ...`
# resolves through sys.modules → _paths_it. Same for _train_utils.
from experiments.phase7_unification import train_phase7  # noqa: E402  (post-swap)


def main() -> None:
    # Verify swap actually took effect — sanity guard against silent
    # cross-subject-model contamination.
    from experiments.phase7_unification._paths import SUBJECT_MODEL as _SM
    assert _SM == "google/gemma-2-2b-it", (
        f"_paths_it swap failed; SUBJECT_MODEL={_SM}. Refusing to "
        f"train with mismatched subject model."
    )
    print(f"[train_phase7_it] sys.modules._paths swap OK; "
          f"SUBJECT_MODEL={_SM}")
    train_phase7.main()


if __name__ == "__main__":
    main()
