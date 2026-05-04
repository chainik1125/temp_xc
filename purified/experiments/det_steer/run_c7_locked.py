"""C7 detection on the **locked checkpoints** + sentence_acts.

Pulls the locked TXC-base / TXC-pro and the cached
``results/c7_backtracking/stage_a/sentence_acts_L10.npz`` from
``han1823123123/temp-bench-{models,data}``, then runs
:func:`temp_bench.eval.detection.detect_case_study` for each arch with
the within-window shuffle ablation. Writes a per-arch PR-AUC + shuffle
gap table + position-variance histogram per TXC.

This is the **agent-runnable script** — agent_back invokes it inside
their existing C7 sweep (no GPU contention with the cohort generation;
detection runs entirely on the cached sentence_acts). The output goes
into ``experiments/det_steer/results/c7_locked/`` and is rendered into
``docs/cross_component/det_steer_summary.md``'s AUTO-RESULTS block by
:mod:`experiments.det_steer.analysis`.

Skip: the steering-side V0/V1/V2/V4 A/B is in
:mod:`experiments.det_steer.run_steering_ab` because it shares state
with the C7 generation pipeline.

Wallclock: ~10 min on one H100 (encode is the bottleneck —
25K sentences × 6 windows × d_sae=32768 fp32 = 19 GB of latents per
arch, batched at 1024).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# det_steer src on PYTHONPATH (not yet pip-installed on this branch).
_DETSTEER_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_DETSTEER_SRC) not in sys.path:
    sys.path.insert(0, str(_DETSTEER_SRC))

import numpy as np
import torch

DEFAULT_ARCHS = ("topk_sae", "tsae_paper", "txc_base", "txc_pro")


def _load_sentence_acts() -> dict[str, np.ndarray]:
    """Resolve via temp_bench.config.purified_root (matches what the
    rest of the framework uses)."""
    from temp_bench.config import purified_root
    path = purified_root() / "results" / "c7_backtracking" / "stage_a" / "sentence_acts_L10.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"sentence_acts cache missing at {path}. Run "
            "`bash scripts/sync_from_hf.sh` (HF auth required) or build "
            "via `temp_bench.case_studies.backtracking.extract_labeled_sentence_acts`."
        )
    z = np.load(path, allow_pickle=True)
    return {"X": z["X"], "is_bt": z["is_bt"], "keys": z["keys"]}


def _load_arch(arch_name: str, *, d_sae: int, T: int = 5, device: str = "cuda"):
    """Load + restore an arch's locked checkpoint by re-walking the
    framework's cache contract. Defers to
    :mod:`temp_bench.architectures` to construct the class; the
    weights are loaded from ``checkpoints/<train_key>/model.safetensors``."""
    import yaml
    from safetensors.torch import load_file
    from temp_bench.architectures import get_arch_class
    from temp_bench.config import checkpoints_dir, locked_archs_dict

    arch_yaml = locked_archs_dict()["archs"][arch_name]
    hparams = dict(arch_yaml["hparams"])
    overrides = arch_yaml.get("per_component_hparams", {}).get("c7", {})
    hparams.update(overrides)

    # d_in is implicit — caller passes it (typically arch_yaml's d_sae // 8).
    arch_cls = get_arch_class(arch_yaml["class_path"])
    arch = arch_cls(d_in=hparams.get("d_in", 4096), **{k: v for k, v in hparams.items() if k != "d_in"})

    # Find the matching checkpoint via manifest.
    manifest_path = checkpoints_dir().parent / "checkpoints" / "manifest.jsonl"
    target = None
    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                row.get("arch") == arch_name
                and row.get("datasource", "").startswith("llama_3_1_8b_base_l10_ward")
            ):
                target = row
                break
    if target is None:
        raise RuntimeError(
            f"no checkpoint manifest entry for arch={arch_name} on c7 datasource. "
            f"Train via experiments/c7_backtracking/run.py or pull from HF."
        )
    ckpt_path = Path(target["local_path"])
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"checkpoint at {ckpt_path} missing on disk; run "
            "`bash scripts/sync_from_hf.sh` to pull from "
            "han1823123123/temp-bench-models."
        )
    state = load_file(str(ckpt_path))
    arch.load_state_dict(state)
    arch.eval()
    return arch.to(device)


def _save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from temp_bench.plotting.figure import save_figure
        save_figure(fig, str(path))
    except Exception:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        thumb = path.with_name(path.stem + ".thumb.png")
        fig.set_size_inches(2.0, 2.0)
        fig.savefig(thumb, dpi=48, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archs", default=",".join(DEFAULT_ARCHS),
                        help="comma-separated list of arch names")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--shuffle_seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    archs = args.archs.split(",")

    from temp_bench.eval.detection import (
        DEFAULT_S_GRID, detect_case_study, detection_table,
    )
    from temp_bench.eval.steering_hooks import position_variance

    out_dir = Path(__file__).resolve().parent / "results" / "c7_locked"
    out_dir.mkdir(parents=True, exist_ok=True)

    sa = _load_sentence_acts()
    X = sa["X"]                      # (n_sent, T, d_in) fp32
    y = sa["is_bt"].astype(np.int64) # (n_sent,)
    keys = sa["keys"]
    qids = np.array([str(k).split("|")[0] for k in keys])

    results: dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_sent": int(X.shape[0]),
        "T": int(X.shape[1]),
        "d_in": int(X.shape[2]),
        "positive_rate": float(y.mean()),
        "n_folds": int(args.n_folds),
        "shuffle_seed": int(args.shuffle_seed),
        "S_grid": list(DEFAULT_S_GRID),
        "archs": {},
    }

    table_data: dict[str, Any] = {}
    for arch_name in archs:
        print(f"[run_c7_locked] {arch_name} — loading checkpoint")
        arch = _load_arch(arch_name, d_sae=32768, T=5, device=args.device)
        print(f"[run_c7_locked] {arch_name} — running detect_case_study")
        det = detect_case_study(
            arch, X, y, qids,
            S_grid=DEFAULT_S_GRID,
            n_folds=args.n_folds,
            shuffle_seed=args.shuffle_seed,
            device=args.device,
            meta={"arch": arch_name, "case_study": "c7"},
        )
        results["archs"][arch_name] = {
            "pr_auc": det.pr_auc,
            "pr_auc_shuffled": det.pr_auc_shuffled,
            "shuffle_gap": det.shuffle_gap,
            "encode_shape": list(det.encode_shape),
        }
        table_data[arch_name] = det

        # Position variance for TXCs.
        if hasattr(arch, "W_dec") and arch.W_dec.dim() == 3:
            pv = position_variance(arch.W_dec.data.cpu()).numpy()
            results["archs"][arch_name]["position_variance"] = {
                "mean": float(pv.mean()),
                "median": float(np.median(pv)),
                "p10": float(np.percentile(pv, 10)),
                "p90": float(np.percentile(pv, 90)),
            }
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.hist(pv, bins=50)
            ax.set_title(f"{arch_name} — position_variance(W_dec[f, :, :])")
            ax.set_xlabel("position-variance ratio (0=constant, 1=mean-zero)")
            ax.set_ylabel("# features")
            fig.tight_layout()
            _save_figure(fig, out_dir / f"position_variance_{arch_name}.png")

        del arch
        torch.cuda.empty_cache()

    # Render markdown table
    md = detection_table(table_data, S_grid=DEFAULT_S_GRID)
    (out_dir / "detection_table.md").write_text(md + "\n")

    results["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"[run_c7_locked] wrote {out_dir / 'summary.json'}")
    print(md)


if __name__ == "__main__":
    import os
    os.environ.setdefault("TQDM_DISABLE", "1")
    main()
