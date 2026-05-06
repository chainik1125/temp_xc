"""Recover c1_noisy leaderboard rows from orphaned checkpoints.

The 7-GPU T-sweep on 2026-05-06T19:52-20:35 trained ~78 cells but only
2 landed in the leaderboard. The checkpoints + config.json files are
on disk, but no manifest/leaderboard rows. Root cause TBD.

This script: scans checkpoints/ for c1_noisy txc_base dirs whose
train_keys aren't in the leaderboard, then runs runner.run_cell on
each (which detects the cached checkpoint, skips training, runs eval,
and appends the leaderboard row).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

from temp_bench import runner
from temp_bench.cache import _read_jsonl, leaderboard_path
from temp_bench.schemas import TrainingConfig

from experiments.c1_noisy_filler.run import (
    COMPONENT, DATASOURCE, EVAL_PROTOCOL_VERSION,
    my_eval_fn, my_train_fn,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-cells", type=int, default=None)
    args = ap.parse_args()

    # 1. Build set of train_keys already in leaderboard for c1_noisy.
    in_leaderboard = set()
    for r in _read_jsonl(leaderboard_path()):
        if r.get("component") != COMPONENT:
            continue
        if r.get("eval_cfg", {}).get("smoke"):
            continue
        in_leaderboard.add(r["train_key"])
    print(f"c1_noisy train_keys in leaderboard: {len(in_leaderboard)}", flush=True)

    # 2. Scan checkpoint dirs for c1_noisy txc_base ckpts NOT in leaderboard.
    ckpt_root = Path("checkpoints")
    candidates = []
    for d in ckpt_root.iterdir():
        if not d.is_dir():
            continue
        cfg_path = d / "config.json"
        if not cfg_path.exists():
            continue
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            continue
        if cfg.get("arch") != "txc_base":
            continue
        if cfg.get("datasource") != DATASOURCE:
            continue
        if cfg["train_key"] in in_leaderboard:
            continue
        over = (cfg.get("training_cfg") or {}).get("arch_hparams_override") or {}
        T = over.get("T")
        if T not in (4, 6, 8, 10, 12):  # only the new T-sweep cells
            continue
        candidates.append({
            "train_key": cfg["train_key"],
            "seed": cfg["seed"],
            "T": int(T),
            "k_pos": int(over.get("k_pos", 0)),
            "n_steps": int((cfg.get("training_cfg") or {}).get("n_steps", 30000)),
            "saved_ts": cfg.get("saved_ts"),
        })

    candidates.sort(key=lambda c: (c["T"], c["seed"], c["k_pos"]))
    print(f"Orphan checkpoints to recover: {len(candidates)}", flush=True)
    if args.max_cells:
        candidates = candidates[: args.max_cells]

    n_ok, n_skip, n_err = 0, 0, 0
    t_start = time.time()
    for i, c in enumerate(candidates, start=1):
        override = {"k_pos": c["k_pos"], "d_sae": 40, "T": c["T"]}
        cfg = TrainingConfig(
            n_steps=c["n_steps"], batch_size=1024,
            plateau_early_stop=False, arch_hparams_override=override,
        )
        eval_cfg = {
            "k_pos": c["k_pos"],
            "smoke": False,
            "_arch_hparams_override": override,
            "t_label": f"T={c['T']}",
            "_p_A": 0.0,
            "_p_B": 0.625,
        }
        try:
            result = runner.run_cell(
                component=COMPONENT, arch_name="txc_base",
                seed=c["seed"], datasource_name=DATASOURCE,
                training_cfg=cfg, eval_cfg=eval_cfg,
                eval_protocol_version=EVAL_PROTOCOL_VERSION,
                train_fn=my_train_fn, eval_fn=my_eval_fn,
            )
            if result.cached:
                n_skip += 1
            else:
                n_ok += 1
                if i % 5 == 0 or i == len(candidates):
                    elapsed = time.time() - t_start
                    eta = elapsed / i * (len(candidates) - i)
                    print(f"  [{i:3d}/{len(candidates)}] T={c['T']:2d} seed={c['seed']:2d} "
                          f"k={c['k_pos']:2d} → train_key={result.train_key[:12]} "
                          f"auc={result.metrics.get('auc', float('nan')):.4f} "
                          f"(eta {eta/60:.1f} min)", flush=True)
        except Exception as e:
            n_err += 1
            print(f"  ERROR T={c['T']} seed={c['seed']} k={c['k_pos']}: "
                  f"{type(e).__name__}: {str(e)[:120]}", flush=True)

    print(f"\nDone. ok={n_ok} cached={n_skip} err={n_err}", flush=True)


if __name__ == "__main__":
    main()
