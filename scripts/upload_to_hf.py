"""Upload Stage B artifacts to Hugging Face for reproducibility.

Two repos:
  1. aniketdesh/ward-stage-b-cache (dataset) — activation cache, mined
     features, cell metrics, judgements, leaderboards. Everything an
     external researcher needs to re-run the pipeline starting from
     trained dictionaries (without re-caching activations).

  2. aniketdesh/ward-stage-b-dictionaries (model) — curated checkpoints:
     TXC k=16 seeds {7,11,23,42}, H13 k=16 seeds {7,11,23,42}, plus
     one representative cell per other arch family (TopK SAE, Stacked
     SAE, TSAE, TSAE-paper, H8).

Both repos public. Upload via HfApi.create_repo + upload_folder /
upload_file. Resumable: existing files at the same path are skipped
unless --force.
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

DATASET_REPO = "aniketdesh/ward-stage-b-cache"
MODEL_REPO = "aniketdesh/ward-stage-b-dictionaries"

CURATED_CKPTS = [
    "txc__resid_L10__k16__s42.pt",
    "txc__resid_L10__k16__s7.pt",
    "txc__resid_L10__k16__s11.pt",
    "txc__resid_L10__k16__s23.pt",
    "txc_h13__resid_L10__k16__s42.pt",
    "txc_h13__resid_L10__k16__s7.pt",
    "txc_h13__resid_L10__k16__s11.pt",
    "txc_h13__resid_L10__k16__s23.pt",
    "txc_h8__resid_L10__k16__s42.pt",
    "topk_sae__ln1_L10__k64__s42.pt",  # best non-TXC under Sonnet
    "stacked_sae__resid_L10__k16__s42.pt",
    "tsae__resid_L10__k32__s42.pt",
    "tsae_paper__resid_L10__k32__s42.pt",
]


def run(cmd: list[str], desc: str = ""):
    import subprocess
    print(f"[exec] {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"[FAIL] {desc}: rc={rc}")
        sys.exit(rc)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--what", choices=["dataset", "model", "both"], default="both")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--results-root", type=Path,
                   default=Path("results/ward_backtracking_txc"))
    args = p.parse_args()

    from huggingface_hub import HfApi, create_repo
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)

    if args.what in ("dataset", "both"):
        print(f"\n=== dataset: {DATASET_REPO} ===")
        if not args.dry_run:
            create_repo(DATASET_REPO, repo_type="dataset", exist_ok=True,
                        token=token, private=False)
        # Upload by folder, one folder at a time so partial-progress is visible.
        folders = [
            ("activations", args.results_root / "activations"),
            ("features", args.results_root / "features"),
            ("cell_metrics", args.results_root / "cell_metrics"),
            ("coherence_grades", args.results_root / "coherence_grades"),
            ("backtracking_judgements", args.results_root / "backtracking_judgements"),
            ("steering_per_cell", args.results_root / "steering_per_cell"),
            ("steering", args.results_root / "steering"),
        ]
        for path_in_repo, local_dir in folders:
            if not local_dir.exists():
                print(f"  [skip] {local_dir} missing")
                continue
            print(f"  [upload-folder] {local_dir} → {path_in_repo}/ "
                  f"({sum(f.stat().st_size for f in local_dir.rglob('*') if f.is_file()) // (1024*1024)} MB)")
            if args.dry_run:
                continue
            api.upload_folder(
                folder_path=str(local_dir),
                path_in_repo=path_in_repo,
                repo_id=DATASET_REPO, repo_type="dataset",
                token=token,
                ignore_patterns=["__pycache__", "*.pyc", ".DS_Store"],
            )

        # Also push a curated leaderboard CSV + the README at the dataset root.
        leaderboard_csv = Path("docs/aniket/experiments/ward_backtracking/images_b/leaderboard_v2.csv")
        leaderboard_bt_csv = Path("docs/aniket/experiments/ward_backtracking/images_b/leaderboard_behavioral.csv")
        if leaderboard_csv.exists() and not args.dry_run:
            api.upload_file(path_or_fileobj=str(leaderboard_csv),
                             path_in_repo="leaderboard_sonnet.csv",
                             repo_id=DATASET_REPO, repo_type="dataset", token=token)
        if leaderboard_bt_csv.exists() and not args.dry_run:
            api.upload_file(path_or_fileobj=str(leaderboard_bt_csv),
                             path_in_repo="leaderboard_behavioral.csv",
                             repo_id=DATASET_REPO, repo_type="dataset", token=token)
        # Stage A artifacts the pipeline reads from.
        for srcname, repodst in [
            ("results/ward_backtracking/prompts.json", "stageA_prompts.json"),
            ("results/ward_backtracking/sentence_labels.json", "stageA_sentence_labels.json"),
            ("results/ward_backtracking/dom_vectors.pt", "stageA_dom_vectors.pt"),
        ]:
            sp = Path(srcname)
            if sp.exists() and not args.dry_run:
                api.upload_file(path_or_fileobj=str(sp), path_in_repo=repodst,
                                  repo_id=DATASET_REPO, repo_type="dataset", token=token)

    if args.what in ("model", "both"):
        print(f"\n=== model: {MODEL_REPO} ===")
        if not args.dry_run:
            create_repo(MODEL_REPO, repo_type="model", exist_ok=True,
                        token=token, private=False)
        ck_dir = args.results_root / "checkpoints"
        for ck in CURATED_CKPTS:
            p_local = ck_dir / ck
            if not p_local.exists():
                print(f"  [skip] {ck} missing")
                continue
            size_mb = p_local.stat().st_size // (1024*1024)
            print(f"  [upload-file] {ck} ({size_mb} MB)")
            if args.dry_run:
                continue
            api.upload_file(path_or_fileobj=str(p_local),
                              path_in_repo=f"checkpoints/{ck}",
                              repo_id=MODEL_REPO, repo_type="model", token=token)
        # config.yaml + the architectures.py + cell_id.py so loaders know how
        # to instantiate from a checkpoint.
        for srcname, repodst in [
            ("experiments/ward_backtracking_txc/config.yaml", "config.yaml"),
            ("experiments/ward_backtracking_txc/architectures.py", "architectures.py"),
            ("experiments/ward_backtracking_txc/cell_id.py", "cell_id.py"),
        ]:
            sp = Path(srcname)
            if sp.exists() and not args.dry_run:
                api.upload_file(path_or_fileobj=str(sp), path_in_repo=repodst,
                                  repo_id=MODEL_REPO, repo_type="model", token=token)

    print("\n[done]")


if __name__ == "__main__":
    main()
