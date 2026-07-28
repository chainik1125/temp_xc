"""Stage the paper's T=5 agentic_txc_02 anchors into the v2 ckpt store.

CARD § 8 / phase_b precedent: the archived paper checkpoints
(`txcdr-base:ckpts/agentic_txc_02__seed{42,1,2}.pt`, the SAME weights
behind the paper's RLHF TXC row) are written under the v2 train_keys of
the `pf_anchor` cells so `run_experiment` finds them train_cached and
runs eval-only — the T5 point is NEVER retrained (alias rule). A side
provenance manifest maps each minted train_key to its upstream file +
sha256 + upstream final_step; rows produced from these keys are
paper-weight evals, disclosed on the exhibit.

Run AFTER the card freeze: `.venv/bin/python -m
experiments.explorations.actmix_rlhf.stage_anchors`
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from experiments.explorations.actmix_rlhf.cells import pf
from temp_bench.archs.agentic_txc02 import AgenticTXC02
from temp_bench.core.config import (
    compute_data_key, compute_train_key, checkpoint_dir, load_datasource,
)

ANCHOR_DIR = Path("/workspace/caches/rlhf/agentic_anchors/ckpts")
UPSTREAM_FINAL_STEP = {42: 4200, 1: 4600, 2: 5200}  # recorded training logs
D_IN = 2304


def main():
    # Read the stream from the CELL, never a literal: a hardcoded
    # datasource here silently survived the base-l12 substrate correction
    # and re-minted the anchors under stale l13-IT train_keys.
    prov = {}
    for seed in (42, 1, 2):
        data_key = compute_data_key(load_datasource(pf(5, seed)["datasource"]))
        src = ANCHOR_DIR / f"agentic_txc_02__seed{seed}.pt"
        raw = src.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        sd = torch.load(src, map_location="cpu", weights_only=True)
        if not isinstance(sd, dict) or "W_enc" not in sd:
            # tolerate {"model": state_dict}-style wrappers
            for key in ("model", "state_dict", "model_state_dict"):
                if isinstance(sd, dict) and key in sd:
                    sd = sd[key]
                    break
        m = AgenticTXC02(d_in=D_IN, d_sae=18432, T=5, k_pos=500)
        missing = m.load_state_dict(sd, strict=False)
        assert not missing.unexpected_keys, (
            f"seed{seed}: unexpected keys {missing.unexpected_keys[:5]}"
        )
        assert set(missing.missing_keys) <= {"global_step", "converged_step"}, (
            f"seed{seed}: anchor missing params {missing.missing_keys[:5]}"
        )
        m.global_step.fill_(UPSTREAM_FINAL_STEP[seed])
        m.converged_step.fill_(UPSTREAM_FINAL_STEP[seed])

        # anchor=True is REQUIRED here: it freezes training_cfg at the
        # staging-time recipe. Without it this stages under the current
        # SWEEP recipe, minting keys that do not match the anchor rows
        # already on the leaderboard — orphaning them so they re-classify
        # as ordinary T=5 sweep cells and fold the paper's weights into
        # the port's mean (mac-d, LOG 13:32/13:35).
        cell = pf(5, seed, anchor=True)
        tk = compute_train_key(
            arch=cell["arch"], seed=seed,
            training_cfg=cell["training_cfg"], data_key=data_key,
        )
        out_dir = checkpoint_dir(tk)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "model.safetensors"
        save_file({k: v.contiguous() for k, v in m.state_dict().items()}, str(out))
        out_sha = hashlib.sha256(out.read_bytes()).hexdigest()
        prov[tk] = {
            "seed": seed,
            "upstream_file": f"txcdr-base:ckpts/agentic_txc_02__seed{seed}.pt",
            "upstream_sha256": sha,
            "staged_path": str(out),
            "staged_sha256": out_sha,
            "upstream_final_step": UPSTREAM_FINAL_STEP[seed],
            "note": "paper T=5 anchor weights; eval-only, never retrained (alias rule)",
        }
        print(f"[stage] seed{seed} -> train_key {tk} (upstream sha {sha[:16]}…)")

    prov_path = Path("experiments/explorations/actmix_rlhf/results/pf_anchor_provenance.json")
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    prov_path.write_text(json.dumps(prov, indent=2) + "\n")
    print(f"[stage] provenance manifest -> {prov_path}")


if __name__ == "__main__":
    main()
