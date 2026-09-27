"""ACTMIX P1 Phase B — `paper-match` arm: EVAL-ONLY on the paper's shipped
§ 5.1 checkpoints (mac-local ruling af2247d43: A1 closed ⇒ eval-only
unlock; PER-ARM composition, never collapsed).

Cells (COMPOSITION_AUDIT § 3 canonical 20K train_keys; where the census
flagged duplicate (arch, seed) specs, the 2026-05-05 re-train family is
chosen — `train_window_size` = 1 (topk_sae) / 2 (tsae_paper), the family
that (a) contains the only seed-42 topk cell and (b) matches the v1
c3 run.py note "agent_em_100k's C3 baseline re-train sets it (TopK at
T=1, T-SAE at T=2)"; the rejected agent_steer twins predate, 05-04, and
remain evaluable post-deadline if the team wants the comparison):

  paper_topk_sae_v1  s1 05363678579ff7cf  s2 7e3c6d83e985d4f5  s42 fe7feb76c9e510ae
  paper_tsae_v1      s1 e8f3355683e0a25f  s2 8f717f87f3f9464a  s42 06053869c2b7e72b
  paper_txc_base_v1  T5  196d4595f0f3b626 / 93bd8e067d5bce80 / 25321cac962bbe12
                     T10 af4308a3e7ae60c3 / a4c123a8a15353c4 / 27567c69685778af
                     T20 4b27b1c7bd02c4cc / 5d226376aa2c9705 / a5c6ffcfb4b09cf7

Flow:
  stage  — re-key each shipped state_dict under ``inner.*``, strict-load
           it into the adapter on CPU (shape/buffer proof), then save to
           ``checkpoints/<v2_train_key>/model.safetensors`` + write
           ``phase_b_manifest.json`` (v2 train_key ↔ shipped train_key,
           sha256, chosen-family rationale).
  smoke  — ONE cell (paper_topk_sae_v1 seed 42, k_feat 20) through
           ``run_experiment``; prints the paper's own v1 number
           (topk_sae k=20: 0.8831 ± 0.0022) beside ours — the
           port-validation gate (mac-a heads-up 00309362f: smoke 1 cell
           before Phase-B grids).
  run    — all 15 cells × k_feat {5, 20}; eval_cfg carries
           arm="paper-match" + src_train_key provenance on every row.

Usage::
  .venv/bin/python -m experiments.probing.actmix.phase_b stage
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m experiments.probing.actmix.phase_b smoke
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m experiments.probing.actmix.phase_b run [--shard-index N --shard-count 2]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from temp_bench.core.config import (
    checkpoint_dir,
    compute_data_key,
    compute_train_key,
    load_arch,
    load_datasource,
)
from temp_bench.core.schemas import TrainingConfig

DATASOURCE = "gemma_2_2b_it_l13_fineweb_24k128"
MIRROR = Path("/workspace/caches/probing/tbm_ckpts")
HF_REPO = "han1823123123/temp-bench-models"
MANIFEST = Path(__file__).resolve().parent / "phase_b_manifest.json"

# (arch, seed, hparam_override_or_None, shipped_train_key, src_cfg_T)
#
# BUG-ARTIFACT FINDING (verified on-box, weights inspected): the six
# shipped "T10/T20" txc_base cells carry T=5-SHAPED weights — they are
# the pre-05-06 silent-T5 bug runs documented in origin/final's c3
# run.py ("T=10 / T=20 cells silently train at T=5 (YAML default) but
# save under the T=10/T=20-keyed train_key"), saved 05-05 22:10 →
# 05-06 04:00. The census shows NO other 20K T10/T20 cells on the IT
# datasource ⇒ no faithful eval-only T-sweep exists for paper-match;
# these six are staged AS T5 evals with `src_tag`/`src_cfg_T`
# provenance, which directly TESTS the flat-T-sweep hypothesis (if
# their evals reproduce the appendix's "T10/T20" numbers, the paper's
# c3 T-slope was seed noise between T5 replicas). Escalated to
# mac-c/mac-local via LOG.
CELLS = [
    ("paper_topk_sae_v1", 1, None, "05363678579ff7cf", None),
    ("paper_topk_sae_v1", 2, None, "7e3c6d83e985d4f5", None),
    ("paper_topk_sae_v1", 42, None, "fe7feb76c9e510ae", None),
    ("paper_tsae_v1", 1, None, "e8f3355683e0a25f", None),
    ("paper_tsae_v1", 2, None, "8f717f87f3f9464a", None),
    ("paper_tsae_v1", 42, None, "06053869c2b7e72b", None),
    ("paper_txc_base_v1", 1, None, "196d4595f0f3b626", None),
    ("paper_txc_base_v1", 2, None, "93bd8e067d5bce80", None),
    ("paper_txc_base_v1", 42, None, "25321cac962bbe12", None),
    ("paper_txc_base_v1", 1, {"src_tag": "cfgT10"}, "af4308a3e7ae60c3", 10),
    ("paper_txc_base_v1", 2, {"src_tag": "cfgT10"}, "a4c123a8a15353c4", 10),
    ("paper_txc_base_v1", 42, {"src_tag": "cfgT10"}, "27567c69685778af", 10),
    ("paper_txc_base_v1", 1, {"src_tag": "cfgT20"}, "4b27b1c7bd02c4cc", 20),
    ("paper_txc_base_v1", 2, {"src_tag": "cfgT20"}, "5d226376aa2c9705", 20),
    ("paper_txc_base_v1", 42, {"src_tag": "cfgT20"}, "a5c6ffcfb4b09cf7", 20),
]

# Mirrors the shipped cells' training_cfg (config census, all 15 cells:
# b=1024, lr 3e-4, warmup 1000, bf16, 20K steps). v1-only fields
# (train_window_size, plateau_*) have no v2 slot — recorded per cell in
# the manifest; this TrainingConfig exists to give the staged ckpts a
# deterministic v2 train_key, not to claim a v2 training run.
def _training_cfg(override: dict | None) -> TrainingConfig:
    return TrainingConfig(
        n_steps=20_000, batch_size=1024,
        arch_hparams_override=(dict(override) if override else None),
    )


def _my_train_key(arch: str, seed: int, override: dict | None) -> str:
    spec = load_arch(arch, section="probing")
    tc = _training_cfg(override)
    if tc.arch_hparams_override:
        spec = spec.model_copy(
            update={"hparams": {**spec.hparams, **tc.arch_hparams_override}})
    return compute_train_key(
        arch=spec, seed=seed, training_cfg=tc,
        data_key=compute_data_key(load_datasource(DATASOURCE)),
        section="probing",
    )


def stage() -> None:
    import torch
    from safetensors.torch import load_file, save_file

    from temp_bench.core.config import import_by_path

    manifest = {"hf_repo": HF_REPO,
                "upstream_pin": "origin/han-phase7-unification@94119bc08",
                "family_rationale": (
                    "dup (arch,seed) specs resolved to the 2026-05-05 "
                    "re-train family (train_window_size 1/2) — contains the "
                    "only s42 topk cell; matches v1 c3 run.py's "
                    "agent_em_100k re-train note. agent_steer 05-04 twins "
                    "not staged (evaluable post-deadline)."),
                "cells": []}
    for arch, seed, override, src_key, src_cfg_T in CELLS:
        src = MIRROR / src_key / "model.safetensors"
        if not src.exists():
            raise FileNotFoundError(f"missing {src} — finish the HF sync first")
        cfg = json.loads((MIRROR / src_key / "config.json").read_text())
        assert cfg["seed"] == seed and cfg["datasource"] == DATASOURCE, (src_key, cfg.get("seed"))

        state = load_file(str(src))
        rekeyed = {f"inner.{k}": v for k, v in state.items()}

        spec = load_arch(arch, section="probing")
        hp = {**spec.hparams, **(override or {})}
        cls = import_by_path(spec.class_path)
        model = cls(d_in=2304, **hp)
        # strict load doubles as the shape proof: for the six cfg_T=10/20
        # artifacts the adapter is built at T=5, so success PROVES the
        # weights are T5-shaped (the silent-T5 bug fingerprint).
        model.load_state_dict(rekeyed, strict=True)

        tk = _my_train_key(arch, seed, override)
        dst = checkpoint_dir(tk)
        dst.mkdir(parents=True, exist_ok=True)
        save_file(rekeyed, str(dst / "model.safetensors"))
        sha = hashlib.sha256(src.read_bytes()).hexdigest()
        manifest["cells"].append({
            "arch": arch, "seed": seed, "hparam_override": override,
            "src_cfg_T": src_cfg_T, "weights_T": (5 if arch == "paper_txc_base_v1" else 1),
            "v2_train_key": tk, "src_train_key": src_key,
            "src_sha256": sha, "src_saved_ts": cfg.get("saved_ts"),
            "src_agent": cfg.get("agent"),
            "src_training_cfg": cfg.get("training_cfg"),
        })
        tag = f"/{override['src_tag']}" if override else ""
        print(f"[stage] {arch}/s{seed}{tag} "
              f"{src_key} -> {tk} ({len(rekeyed)} tensors, strict-load OK)")
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"[stage] manifest -> {MANIFEST}")


def _run_cells(cells, k_feats=(5, 20)) -> None:
    from temp_bench.core.runner import run_experiment

    for arch, seed, override, src_key, src_cfg_T in cells:
        for k in k_feats:
            eval_cfg = {
                "k_feat": int(k), "S": 32,
                "shuffle": "within_window", "shuffle_seed": 0,
                "encode_batch_size": 64,
                "arm": "paper-match",
                "src_train_key": src_key, "src_repo": HF_REPO,
                "smoke": False,
            }
            if src_cfg_T is not None:
                # silent-T5 bug artifact: config said T=src_cfg_T, weights
                # are T5 — evaluated AS T5; row carries the disclosure.
                eval_cfg["src_cfg_T"] = int(src_cfg_T)
                eval_cfg["bug_artifact_t5"] = True
            t0 = time.time()
            r = run_experiment(
                experiment="probing", arch_name=arch, seed=seed,
                datasource_name=DATASOURCE,
                training_cfg=_training_cfg(override),
                eval_cfg=eval_cfg, agent=os.environ.get("AGENT_NAME"),
            )
            m = r.row.metrics
            status = "CACHED" if r.eval_cached else f"ran {time.time()-t0:.0f}s"
            tag = f"/{override['src_tag']}" if override else ""
            print(f"[{status}] {arch}/s{seed}{tag}/k{k}  "
                  f"auc={m.get('mean_auc', float('nan')):.4f}  "
                  f"shuf={m.get('mean_auc_shuf', float('nan')):.4f}  "
                  f"l0={m.get('realized_l0', float('nan')):.2f}", flush=True)


def cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["stage", "smoke", "run"])
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--k-feats", type=int, nargs="*", default=[5, 20],
                    help="probe budgets; the printed-figure trapezoid "
                         "needs the full {5,10,20,40,80,160,320,640}")
    args = ap.parse_args()

    if args.mode == "stage":
        stage()
        return
    if not MANIFEST.exists():
        raise SystemExit("run `phase_b stage` first")
    if args.mode == "smoke":
        print("[smoke] paper_topk_sae_v1 s42 k20 — paper v1 reference: "
              "mean_auc 0.8831 ± 0.0022 (3 seeds). Ours (1 seed) should "
              "land in that neighborhood; a large deviation kills the "
              "port, not the paper.")
        _run_cells([("paper_topk_sae_v1", 42, None, "fe7feb76c9e510ae", None)],
                   k_feats=(20,))
        return
    mine = [c for i, c in enumerate(CELLS)
            if i % args.shard_count == args.shard_index]
    print(f"[phase_b] {len(mine)}/{len(CELLS)} cells on shard "
          f"{args.shard_index}/{args.shard_count} k_feats={args.k_feats}")
    _run_cells(mine, k_feats=tuple(args.k_feats))


if __name__ == "__main__":
    cli()
