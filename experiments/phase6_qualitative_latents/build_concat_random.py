"""Phase 6.1 follow-up #2: build `concat_random` as a generalization
control for the Figure-4-style autointerp / passage-coverage metric.

Concat_A (Newton/MMLU/Gita) and concat_B (MMLU/Darwin/AnimalFarm/MMLU)
are hand-curated passage sets chosen to mirror the paper's figures.
If an arch is curated-concat-specialised, its qualitative score on
A/B overstates its actual generalisation. `concat_random` samples
random FineWeb 128-token segments and concatenates them into a single
long sequence with distinct passage IDs, so the same pipeline
(top-by-variance + autointerp + passage-coverage) can be run on
uncurated text.

Schema mirrors concat_A/B exactly (single long sequence + per-passage
start/end spans), so downstream (`encode_archs`, `run_autointerp`)
treats it as another concat cell.

Default: 7 passages × 256 tokens (= 2 concatenated FineWeb seqs each) =
1792 tokens total. P=7 matches concat_A+B's combined passage count.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "experiments/phase6_qualitative_latents/concat_corpora"
FW_TOKEN_IDS = REPO / "data/cached_activations/gemma-2-2b-it/fineweb/token_ids.npy"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-passages", type=int, default=7)
    p.add_argument("--seqs-per-passage", type=int, default=2,
                   help="number of 128-tok FineWeb seqs concatenated per passage")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-name", type=str, default="concat_random.json")
    args = p.parse_args()

    ids_all = np.load(FW_TOKEN_IDS)  # (6000, 128)
    N_AVAILABLE, SEQ_LEN = ids_all.shape

    rng = np.random.default_rng(args.seed)
    n_needed = args.n_passages * args.seqs_per_passage
    picks = rng.choice(N_AVAILABLE, size=n_needed, replace=False)
    picks.sort()

    token_ids: list[int] = []
    provenance = []
    offset = 0
    for p_i in range(args.n_passages):
        pi0 = p_i * args.seqs_per_passage
        pi1 = pi0 + args.seqs_per_passage
        seq_idxs = picks[pi0:pi1].tolist()
        seq_tokens: list[int] = []
        for si in seq_idxs:
            seq_tokens.extend(ids_all[si].tolist())
        start = offset
        end = offset + len(seq_tokens)
        provenance.append({
            "source": f"FineWeb random (seqs {seq_idxs})",
            "label": f"fw_{p_i:02d}",
            "start": start, "end": end, "n_tokens": len(seq_tokens),
            "fw_seq_indices": seq_idxs,
        })
        token_ids.extend(seq_tokens)
        offset = end

    out_dict = {
        "token_ids": token_ids,
        "provenance": provenance,
        "n_tokens": len(token_ids),
        "meta": {
            "construction": "random FineWeb seqs concatenated into "
                            "distinct passage IDs for Phase 6.1 "
                            "generalisation control",
            "seed": args.seed,
            "n_passages": args.n_passages,
            "seqs_per_passage": args.seqs_per_passage,
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dst = OUT_DIR / args.out_name
    dst.write_text(json.dumps(out_dict, indent=2))
    print(f"wrote {dst}  n_tokens={len(token_ids)}  n_passages={args.n_passages}")


if __name__ == "__main__":
    main()
