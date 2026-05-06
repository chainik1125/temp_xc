"""
Run the StackedSAE-vs-TXCDR sentence-level case study on N different
32-token sequences and dump all figures into a single output directory.

Usage:
    /home/cs29824/.venv/bin/python safety_research/scripts/sentence_multi.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

NLP_DIR = Path("/home/cs29824/andre/temp_xc/temporal_crosscoders/NLP")
SAFETY_DIR = Path("/home/cs29824/andre/temp_xc/safety_research")

CKPT_DIR = SAFETY_DIR / "results" / "checkpoints"
INTERP_DIR = SAFETY_DIR / "results" / "autointerp"
OUT_DIR = NLP_DIR / "viz_outputs" / "sentence_case_studies"

# Five chain ids picked so the cache contents differ enough to surface
# distinct activation patterns. The first is the chain we already used
# in earlier figures; the others are arbitrary but reproducible.
CHAINS = [16921, 42, 137, 4242, 12345, 7, 256, 1024, 8888, 19999]

PYTHON = "/home/cs29824/.venv/bin/python"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sae_ckpt = CKPT_DIR / "tsae__mid_res__k100__T5.pt"  # StackedSAE T=5
    tx_ckpt = CKPT_DIR / "txc__mid_res__k100__T5.pt"
    sae_interps = INTERP_DIR / "tsae" / "explanations.jsonl"
    tx_interps = INTERP_DIR / "txc" / "explanations.jsonl"

    for path in (sae_ckpt, tx_ckpt, sae_interps, tx_interps):
        if not path.exists():
            sys.exit(f"missing {path}")

    for chain_id in CHAINS:
        print(f"\n=== chain {chain_id} ===")
        cmd = [
            PYTHON, str(NLP_DIR / "sentence.py"),
            "--chain", str(chain_id),
            "--sae-ckpt", str(sae_ckpt),
            "--tx-ckpt", str(tx_ckpt),
            "--sae-interps", str(sae_interps),
            "--tx-interps", str(tx_interps),
            "--output-dir", str(OUT_DIR),
            "--seed", "7",
        ]
        subprocess.check_call(cmd, cwd=str(NLP_DIR))

    # Summary list of generated figures.
    pngs = sorted(OUT_DIR.glob("sentence_*.png"))
    print(f"\nGenerated {len(pngs)} case-study figures in {OUT_DIR}:")
    for p in pngs:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
