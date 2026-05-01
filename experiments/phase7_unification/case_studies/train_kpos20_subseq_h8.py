"""Train SubseqH8 at k_pos=20 — Han's "T_max + subseq sample" deadzone-escape variant.

SubseqH8: H8 stack (anti-dead + matryoshka H/L + multi-distance InfoNCE)
operating on T_max-position windows, but during training each step samples
t_sample contiguous positions from the window. The encoder must produce
useful features whether it sees a full T_max window or a t_sample subset.

Hypothesis: T=10/20 with t_sample=5 lets the encoder learn sequence-level
features (long context) while remaining flexible at inference (short
contiguous window). Tests whether high-T compensation via subseq sampling
escapes the T=2-5 "deadzone" Han hypothesised.

Run:
  TQDM_DISABLE=1 .venv/bin/python -m \\
      experiments.phase7_unification.case_studies.train_kpos20_subseq_h8 \\
      --T-max 10 --t-sample 5 --shifts 5 --seed 42

Defaults to contiguous subseq sampling (B1 variant).
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("TQDM_DISABLE", "1")

import sys
sys.path.insert(0, "/workspace/temp_xc")

from experiments.phase7_unification._paths import banner
from experiments.phase7_unification.case_studies._arch_utils import build_meta_template
from experiments.phase7_unification.case_studies.train_kpos20_h8_shifts import (
    train_with_arch_dict,
    K_POS,
)


def build_arch(T_max: int, t_sample: int, shifts: tuple, contiguous: bool = True) -> dict:
    k_win = K_POS * t_sample  # at inference, window encodes t_sample positions max
    shifts_str = "_".join(map(str, shifts))
    return {
        "row": -1,
        "arch_id": f"subseq_h8_tmax{T_max}_tsamp{t_sample}_kpos{K_POS}_shifts{shifts_str}",
        "group": "deadzone_escape_phase2",
        "T": t_sample,           # at inference, window seen is t_sample (effective)
        "T_max": T_max,          # but the encoder has T_max position slabs
        "t_sample": t_sample,
        "k_win": k_win,
        "k_pos": K_POS,
        "shifts": list(shifts),
        "contiguous": contiguous,
        "src_module": "src.architectures.phase5b_subseq_sampling_txcdr",
        "src_class": "SubseqH8",
        "recipe": (f"SubseqH8 — H8 stack at T_max={T_max} with t_sample={t_sample} "
                   f"({'contiguous' if contiguous else 'random'}) subseq sampling, "
                   f"k_pos={K_POS} (k_win={k_win}), shifts={shifts}"),
        "purpose": (f"Han's deadzone-escape: high T_max context with subseq-sampled "
                    f"training. Tests whether T=10/20 with t_sample=5 wins where "
                    f"plain T=10 fails."),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--T-max", type=int, required=True, dest="T_max")
    p.add_argument("--t-sample", type=int, required=True, dest="t_sample")
    p.add_argument("--shifts", type=int, nargs="+", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--no-hf-push", action="store_true")
    p.add_argument("--non-contiguous", action="store_true",
                   help="random non-contiguous subseq (B2); default is contiguous (B1)")
    args = p.parse_args()
    banner(__file__)

    arch = build_arch(args.T_max, args.t_sample, tuple(args.shifts),
                      contiguous=not args.non_contiguous)
    arch_id = arch["arch_id"]
    print(f"\n=== {arch_id} (T_max={args.T_max}, t_sample={args.t_sample}, "
          f"k_pos={K_POS}, shifts={args.shifts}) seed={args.seed} ===", flush=True)

    train_with_arch_dict(arch, seed=args.seed,
                         max_steps=args.max_steps,
                         push_to_hf=not args.no_hf_push)


if __name__ == "__main__":
    main()
