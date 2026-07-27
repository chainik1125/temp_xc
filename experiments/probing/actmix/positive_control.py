"""Positive control for the RM equivalence instrument (0c4044b76 (a)).

Trains the sae twin pair at a THIN-POOL config where the compositions
MUST diverge — d_sae=64, k_pos=48: the batch-topk boundary sits at
the top 75% of pooled pre-activations, far below the ~50% positive
mass at init, so rectify-after-select fires from step 0 and the
trajectories separate. The equivalence checker's comparison logic is
then run on the two checkpoints; the run FAILS (exit 2) unless it
reports DIVERGENCE. If this control ever reports IDENTICAL, the
instrument is broken and the halt ruling is void (mac-local's words).

Cells go through the canonical runner (hard rule 1) with
``eval_cfg.positive_control = true`` for provenance; k_feat=5 only,
n_steps=2000 (divergence is a step-0 phenomenon; 2k steps is belt).

Run: .venv/bin/python -m experiments.probing.actmix.positive_control
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file

from temp_bench.core.runner import run_experiment
from temp_bench.core.schemas import TrainingConfig

ROOT = Path(__file__).resolve().parents[3]
OVERRIDE = {"d_sae": 64, "k_pos": 48}
PAIR = (("batchtopk_sae", "relu-mix"), ("batchtopk_sae_btkonly", "btk-only"))


def main():
    tks = {}
    for arch, arm in PAIR:
        r = run_experiment(
            experiment="probing", arch_name=arch, seed=42,
            datasource_name="gemma_2_2b_it_l13_fineweb_24k128",
            training_cfg=TrainingConfig(
                n_steps=2000, batch_size=4096,
                arch_hparams_override=dict(OVERRIDE)),
            eval_cfg={"k_feat": 5, "S": 32, "shuffle": "within_window",
                      "shuffle_seed": 0, "encode_batch_size": 64,
                      "arm": arm, "smoke": False,
                      "positive_control": True},
            agent=os.environ.get("AGENT_NAME", "runpod-1"),
        )
        tks[arm] = r.row.train_key
        print(f"[control] {arch} ({arm}) train_key={r.row.train_key} "
              f"mean_auc={r.row.metrics['mean_auc']:.4f} "
              f"l0={r.row.metrics['realized_l0']:.2f}")

    paths = {}
    for line in (ROOT / "checkpoints" / "manifest.jsonl").read_text().splitlines():
        try:
            j = json.loads(line)
        except Exception:
            continue
        for arm, tk in tks.items():
            if j.get("train_key") == tk and "local_path" in j:
                paths[arm] = j["local_path"]
    a = load_file(paths["relu-mix"])
    b = load_file(paths["btk-only"])
    diffs = []
    for k in sorted(set(a) & set(b)):
        if not torch.equal(a[k], b[k]):
            diffs.append((k, (a[k].float() - b[k].float()).abs().max().item()))
    print(f"[control] thin-pool pair: {len(diffs)} mismatching tensors "
          f"of {len(set(a) & set(b))} shared")
    for k, d in diffs:
        print(f"  {k}: maxdiff={d:.3e}")
    if diffs:
        print("[control] VERDICT: DIVERGENCE — instrument PASSES the "
              "positive control")
        return 0
    print("[control] VERDICT: IDENTICAL at thin pool — INSTRUMENT BROKEN, "
          "halt ruling void (escalate immediately)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
