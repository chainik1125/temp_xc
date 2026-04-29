"""Compute Difference-of-Means (DoM) steering vectors per offset, per source model.

Ward Eq. 3:  v = MeanAct(D₊) − MeanAct(D)

D  = all collected sentence activations
D₊ = subset with `is_backtracking=True`

We produce one DoM vector per offset in the collected window plus an
additional `union` vector that is the *mean of the per-offset vectors*
across the [-13, -8] window — used to steer with the full window
strength rather than picking one slot.

Output (torch .pt):
  {
    "base":      {"offsets": [-13, ..., 0], "vectors": Tensor(n_off, d_model), "union": Tensor(d_model)},
    "reasoning": {"offsets": [-13, ..., 0], "vectors": Tensor(n_off, d_model), "union": Tensor(d_model)},
    "meta": {"layer": 10, "n_sentences": ..., "n_backtracking": ...}
  }
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ward.derive_dom")


def _dom_per_offset(activations: np.ndarray, is_bt: np.ndarray) -> np.ndarray:
    """Return (n_offsets, d_model) DoM vectors.

    We use mean(D+) - mean(D) (per Ward Eq. 3) rather than the more
    common mean(D+) - mean(D-). The two differ by a multiplicative factor
    of (1 - α) where α = |D+| / |D|. For α ≈ 0.1 (typical backtracking
    rate) that's a 0.9× scaling — absorbed into the magnitude sweep, but
    means our magnitudes are not directly comparable to a paper using the
    other form. Cosine with Ward's vector is invariant to this scaling
    and is the safer cross-paper comparison anchor.
    """
    pos = activations[is_bt]      # (N+, n_off, d)
    full = activations             # (N,  n_off, d)
    if pos.shape[0] == 0:
        raise ValueError("no positive (backtracking) sentences — check sentence_labels.json")
    mu_pos = pos.mean(axis=0)     # (n_off, d)
    mu_all = full.mean(axis=0)    # (n_off, d)
    return (mu_pos - mu_all).astype(np.float32)


def _short_model_name(hf_id: str) -> str:
    return hf_id.split("/")[-1].lower().replace(".", "-")


def _load_npz(acts_dir: Path, tag: str, hf_id: str) -> dict:
    path = acts_dir / f"acts_{tag}_{_short_model_name(hf_id)}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing activations: {path} — run collect_offsets.py first")
    z = np.load(path, allow_pickle=True)
    return {
        "offsets": z["offsets"].tolist(),
        "activations": z["activations"],
        "is_backtracking": z["is_backtracking"],
    }


def _build_one(tag: str, hf_id: str, acts_dir: Path, union_offsets: list[int]) -> dict:
    z = _load_npz(acts_dir, tag, hf_id)
    offsets: list[int] = z["offsets"]
    activations: np.ndarray = z["activations"]
    is_bt: np.ndarray = z["is_backtracking"]
    n_total = activations.shape[0]
    n_pos = int(is_bt.sum())
    log.info("[info] %s | n_sentences=%d | n_backtracking=%d", tag, n_total, n_pos)

    vectors = _dom_per_offset(activations, is_bt)  # (n_off, d)

    # Union vector: mean of per-offset DoMs across the [-13, -8] window
    # (or whatever was set in config.offsets.single — we just average the
    # offsets the user listed as `single`).
    union_idx = [offsets.index(o) for o in union_offsets if o in offsets]
    if not union_idx:
        log.warning("[warn] union offsets %s not in collected offsets %s; using all", union_offsets, offsets)
        union_idx = list(range(len(offsets)))
    union_vec = vectors[union_idx].mean(axis=0).astype(np.float32)  # (d,)

    return {
        "offsets": [int(o) for o in offsets],
        "vectors": torch.from_numpy(vectors),
        "union": torch.from_numpy(union_vec),
        "union_offsets": [int(o) for o in offsets if offsets.index(o) in union_idx],
        "n_sentences": n_total,
        "n_backtracking": n_pos,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    out_path = Path(cfg["paths"]["dom"])
    if out_path.exists() and not args.force:
        log.info("[info] resume | %s exists", out_path)
        return 0

    acts_dir = Path(cfg["paths"]["acts_dir"])
    union_offsets = list(cfg["offsets"]["single"])

    base_pack = _build_one("base", cfg["models"]["base"], acts_dir, union_offsets)
    reasoning_pack = _build_one("reasoning", cfg["models"]["reasoning"], acts_dir, union_offsets)

    payload = {
        "base": base_pack,
        "reasoning": reasoning_pack,
        "meta": {
            "layer": int(cfg["steering_layer"]),
            "models": {"base": cfg["models"]["base"], "reasoning": cfg["models"]["reasoning"]},
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    log.info("[done] saved DoM vectors | path=%s", out_path)

    cos = torch.nn.functional.cosine_similarity(
        base_pack["union"].unsqueeze(0), reasoning_pack["union"].unsqueeze(0)
    ).item()
    log.info("[info] cos(base_union, reasoning_union) = %.4f  (Ward reports ~0.74)", cos)
    return 0


if __name__ == "__main__":
    sys.exit(main())
