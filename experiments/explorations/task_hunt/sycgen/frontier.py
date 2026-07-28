"""ITEM 6 — recovery-vs-budget FRONTIER: TXC vs pooled-SAE vs stacked-SAE.

Answers the challenge that our sycgen claim compared a *windowed* model
against a *per-token* SAE, which establishes nothing about architecture.

Hub ruling 14de8b5a0: sweep k on both arms and plot the frontier — NOT a
single budget-matched point. My 21:14 arithmetic showed why: matching a
pooled SAE to TXC's per-window l0 forces 0.49 l0/token at T16 (most
tokens get no feature at all), and because the arms' budgets scale
differently in T, matching per-window necessarily unmatches per-token.
There is no single matched point to specify.

The new arms need NO training: pooled/stacked are post-hoc transforms of
the ALREADY-TRAINED per-token (T=1) SAE.

    pooled  : encode each of the T tokens -> mean over the window -> d_sae
    stacked : encode each of the T tokens -> concatenate          -> T*d_sae

⚑ Feature-dimension asymmetry, disclosed rather than hidden: the
evaluator's tile code for TXC is `d_sae`. Pooled matches that exactly.
**Stacked gets T*d_sae — T times the probe input** — so a stacked win is
partly a probe-capacity win. Reported, never netted out.

Scoring reuses `lambda_recovery`'s own tiling and probe verbatim, so the
new arms are scored by the SAME instrument as TXC. Anything else would
not be a comparison.

    .venv/bin/python -m experiments.explorations.task_hunt.sycgen.frontier
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "frontier.json"


class _WindowWrapper(torch.nn.Module):
    """Present a per-token SAE as a window encoder the evaluator can score.

    `encode(tiles)` receives `(B, T, d_in)` and must return the tile code.
    We run the frozen per-token SAE on every position, optionally truncate
    each position's code to its top-`k_tok` entries by magnitude (the SAE
    chooses which features it keeps — we only constrain how many), then
    pool or stack across the window.
    """

    def __init__(self, sae, mode: str, k_tok: int | None):
        super().__init__()
        self.sae, self.mode, self.k_tok = sae, mode, k_tok
        self.realized_l0 = []          # per-call l0 per WINDOW, for the receipt

    def encode(self, tiles: torch.Tensor) -> torch.Tensor:
        B, T, d_in = tiles.shape
        z = self.sae.encode(tiles.reshape(B * T, 1, d_in))
        z = z.reshape(B, T, -1)
        if self.k_tok is not None and self.k_tok < z.shape[-1]:
            kth = z.abs().topk(self.k_tok, dim=-1).values[..., -1:]
            z = z * (z.abs() >= kth)
        # realized budget: distinct active features per WINDOW (pooling
        # collapses a feature firing at several positions — measure the
        # union, never assume T x per-token, which is the upper bound the
        # hub retracted).
        active_union = (z.abs() > 0).any(dim=1).sum(dim=-1).float()
        self.realized_l0.append(float(active_union.mean()))
        if self.mode == "pooled":
            return z.mean(dim=1)                    # (B, d_sae)
        return z.reshape(B, T * z.shape[-1])        # (B, T*d_sae)


def main():
    from temp_bench.core.config import (compute_data_key, data_cache_dir,
                                        load_arch, load_datasource)
    from temp_bench.evals import lambda_recovery as LR
    import experiments.explorations.task_hunt.sycgen.run_retrain as RR

    ds = load_datasource(RR.DS)
    print(f"[frontier] datasource {RR.DS}  data_key {compute_data_key(ds)}")
    print("[frontier] NOTE: pooled matches TXC's tile-code dim (d_sae);")
    print("[frontier]       stacked gets T*d_sae — disclosed, not netted out.")
    print("[frontier] scaffold in place; wiring the k-sweep next.")


if __name__ == "__main__":
    main()
