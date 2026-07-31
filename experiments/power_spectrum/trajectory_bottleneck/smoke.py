"""Tiny CUDA/CPU forward-backward smoke test for both adapter variants."""

from __future__ import annotations

import json

import torch

from .model import TrajectoryBottleneck


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    torch.manual_seed(42)
    rows = []
    for rank in (0, 16):
        model = TrajectoryBottleneck(
            base_decoder=torch.randn(128, 64, device=device, dtype=dtype),
            base_decoder_bias=torch.zeros(64, device=device, dtype=dtype),
            window=5,
            k_window=20,
            rank=rank,
            dead_threshold_tokens=10,
            aux_k=32,
        ).to(device=device, dtype=dtype)
        indices = torch.randint(0, 128, (4, 5, 8), device=device)
        values = torch.rand((4, 5, 8), device=device, dtype=dtype)
        target = torch.randn((4, 5, 64), device=device, dtype=dtype)
        result = model.loss(indices, values, target)
        result["loss"].backward()
        rows.append(
            {
                "rank": rank,
                "loss": float(result["loss"].detach().float().cpu()),
                "l0": float(result["l0"].float().cpu()),
                "gradient_finite": all(
                    parameter.grad is None
                    or bool(torch.isfinite(parameter.grad).all())
                    for parameter in model.parameters()
                ),
            }
        )
    if not all(row["gradient_finite"] for row in rows):
        raise RuntimeError(f"non-finite smoke-test gradient: {rows}")
    print(json.dumps({"device": device, "rows": rows}, sort_keys=True))


if __name__ == "__main__":
    main()
