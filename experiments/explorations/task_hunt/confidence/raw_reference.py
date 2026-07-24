"""Raw-activation reference rows for the hedging-LEVEL Stage-2 panel.

Pre-registered in `card_stage2.md` § 6; OFF-leaderboard by design. Computes,
under the *identical* probe pipeline the panel uses
(`lambda_recovery._train_lambda_probe`: same window sampling seeds, same
finite-target mask, same LinearRegression + chance floor), the recovery of
the frozen slope8 target from RAW activations:

- ``raw_tok``   (T = 1): the leading-edge position's raw residual vector;
- ``raw_mean``  (T ∈ {2, 4, 8, 16}): the tile-mean raw vector (order-free
  pooling — the screen's mechanism, expressed in the Stage-2 metric).

These are interpretive anchors (dashed lines in the figure), never panel
cells: p = d_in = 4096 raw features vs d_sae = 2048 codes, so they share the
panel's T = 16 interpolation-regime caveat one notch harder.

Run:  .venv/bin/python -m \
        experiments.explorations.task_hunt.confidence.raw_reference
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from explorations.task_hunt.real_slope import ward_slope_real
from temp_bench.evals.lambda_recovery import _train_lambda_probe

HERE = Path(__file__).resolve().parent
EVAL_L = 32
WINDOW_TS = (2, 4, 8, 16)


class _RawReadout(torch.nn.Module):
    """Stub arch exposing a raw readout through the probe pipeline."""

    def __init__(self, T: int, mode: str):
        super().__init__()
        self.config = SimpleNamespace(T=T)
        self.mode = mode
        # one parameter so the pipeline can read a device off the model
        self._anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def encode(self, tiles: torch.Tensor) -> torch.Tensor:
        if self.mode == "tok":                      # (B, 1, d) -> (B, d)
            return tiles[:, -1, :]
        if self.mode == "mean":                     # (B, T, d) -> (B, d)
            return tiles.mean(dim=1)
        raise ValueError(self.mode)


@torch.no_grad()
def main() -> None:
    data = ward_slope_real()
    lam = data.extra["lambda_labels"].float()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out: dict[str, dict] = {"meta": {
        "ds": "ward_real_slope8_distill_l14", "eval_window_L": EVAL_L,
        "off_leaderboard": True, "card": "card_stage2.md §6"}}

    m = _RawReadout(1, "tok").to(dev).eval()
    out["raw_tok"] = _train_lambda_probe(m, data.x, lam, L=EVAL_L)
    print("raw_tok", out["raw_tok"])
    for T in WINDOW_TS:
        m = _RawReadout(T, "mean").to(dev).eval()
        out[f"raw_mean_T{T}"] = _train_lambda_probe(m, data.x, lam, L=EVAL_L)
        print(f"raw_mean_T{T}", out[f"raw_mean_T{T}"])

    dst = HERE / "results" / "stage2_raw_reference.json"
    dst.write_text(json.dumps(out, indent=2))
    print("wrote", dst)


if __name__ == "__main__":
    main()
