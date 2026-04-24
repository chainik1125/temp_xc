"""BatchTopK inference-threshold sanity check (A2 of 2026-04-23 handover).

Reads each BatchTopK ckpt's state_dict and reports the EMA threshold
buffer(s). The BatchTopK module (src/architectures/_batchtopk.py) stores
a `threshold` scalar buffer used at inference. If it's 0 or vastly out
of range, the model shuts down (or saturates) at eval — plausible cause
of the 7/8 BatchTopK regressions observed in the minimum-scope bench.

Pure state-dict scan — no forward pass, no GPU, ~1s per ckpt. Writes
`results/batchtopk_threshold_audit.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

REPO = Path("/workspace/temp_xc")
CKPT_DIR = REPO / "experiments/phase5_downstream_utility/results/ckpts"
OUT_JSON = REPO / "experiments/phase5_downstream_utility/results/batchtopk_threshold_audit.json"


def scan_ckpt(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    thresholds = {}
    preact_stats = {}
    for key, val in sd.items():
        if key.endswith(".threshold"):
            if isinstance(val, torch.Tensor) and val.numel() == 1:
                thresholds[key] = float(val.item())
            else:
                thresholds[key] = repr(val)
    # Also scan encoder-bias norms as a proxy for pre-activation scale
    for key, val in sd.items():
        if key.endswith("b_enc") or key.endswith("enc_bias"):
            if isinstance(val, torch.Tensor):
                preact_stats[key] = {
                    "mean": float(val.float().mean()),
                    "std": float(val.float().std()),
                    "numel": int(val.numel()),
                }
    meta = ckpt.get("meta", {})
    return {
        "thresholds": thresholds,
        "enc_bias_stats": preact_stats,
        "meta": {k: v for k, v in meta.items() if isinstance(v, (int, float, str, bool, list))},
    }


def main():
    ckpts = sorted(p for p in CKPT_DIR.glob("*_batchtopk__seed42.pt"))
    print(f"Scanning {len(ckpts)} BatchTopK ckpts...")
    out = {}
    for p in ckpts:
        arch = p.name.replace("__seed42.pt", "")
        try:
            out[arch] = scan_ckpt(p)
            ths = out[arch]["thresholds"]
            if not ths:
                print(f"  {arch:<60s} NO threshold buffer found!")
            else:
                vals = list(ths.values())
                summary = ", ".join(
                    f"{k.split('.')[-2] if '.' in k else k}={v:.4f}"
                    for k, v in ths.items()
                )
                print(f"  {arch:<60s} threshold: {summary}")
                # Flag suspicious
                for key, v in ths.items():
                    if v == 0:
                        print(f"    ⚠️  ZERO threshold at {key} — never calibrated")
                    elif abs(v) > 10:
                        print(f"    ⚠️  LARGE threshold {v:.3f} at {key} — may clip all latents")
                    elif v < -1e-3:
                        print(f"    ⚠️  NEGATIVE threshold {v:.4f} at {key}")
        except Exception as e:
            print(f"  {arch:<60s} ERROR: {e}")
            out[arch] = {"error": str(e)}
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_JSON}")

    # Compact summary
    print("\n=== SUMMARY ===")
    zero_count = 0
    for arch, r in out.items():
        ths = r.get("thresholds", {})
        if not ths:
            continue
        if any(v == 0 for v in ths.values()):
            zero_count += 1
    print(f"  archs with zero threshold: {zero_count}/{len(out)}")
    all_vals = []
    for r in out.values():
        all_vals.extend(r.get("thresholds", {}).values())
    if all_vals:
        import statistics as st
        print(f"  all thresholds: min={min(all_vals):.4f} "
              f"max={max(all_vals):.4f} median={st.median(all_vals):.4f}")


if __name__ == "__main__":
    main()
