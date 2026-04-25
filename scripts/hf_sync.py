"""One-shot HuggingFace sync for Phase 6.1 + 6.2 + 6.3 artefacts.

Uploads everything new since the last sync (indexed by filename) to:
- `han1823123123/txcdr`       → ckpts
- `han1823123123/txcdr-data`  → z_cache, autointerp labels, concat
                                 corpora, probing_results.jsonl

Idempotent: uses `huggingface_hub.upload_file` which overwrites by
default, so repeat runs are safe but add version history. Tracks a
small manifest at `.hf_sync_manifest.json` so we don't re-upload
unchanged files.

Usage:
    python scripts/hf_sync.py            # dry-run preview
    python scripts/hf_sync.py --go       # actually upload

Run this at the end of every pipeline stage per the HF-sync rule
(see `feedback_hf_sync` memory).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / ".hf_sync_manifest.json"

CKPT_REPO = "han1823123123/txcdr"
DATA_REPO = "han1823123123/txcdr-data"


# What to sync. (local-path, repo-name, path-in-repo)
def _plan_entries():
    entries = []

    ckpts_dir = REPO / "experiments/phase5_downstream_utility/results/ckpts"
    for p in ckpts_dir.glob("*.pt"):
        entries.append((p, CKPT_REPO, f"ckpts/{p.name}"))

    concat_dir = REPO / "experiments/phase6_qualitative_latents/concat_corpora"
    for p in concat_dir.glob("*.json"):
        entries.append((p, DATA_REPO,
                        f"experiments/phase6_qualitative_latents/concat_corpora/{p.name}"))

    autointerp_dir = REPO / "experiments/phase6_qualitative_latents/results/autointerp"
    for p in autointerp_dir.glob("*.json"):
        entries.append((p, DATA_REPO,
                        f"experiments/phase6_qualitative_latents/results/autointerp/{p.name}"))
    for p in autointerp_dir.glob("*.md"):
        entries.append((p, DATA_REPO,
                        f"experiments/phase6_qualitative_latents/results/autointerp/{p.name}"))

    # z_cache is big; sync per-arch npy
    zcache_dir = REPO / "experiments/phase6_qualitative_latents/z_cache"
    for p in zcache_dir.rglob("*.npy"):
        rel = p.relative_to(REPO)
        entries.append((p, DATA_REPO, str(rel)))
    for p in zcache_dir.rglob("provenance.json"):
        rel = p.relative_to(REPO)
        entries.append((p, DATA_REPO, str(rel)))

    probe_res = REPO / "experiments/phase5_downstream_utility/results/probing_results.jsonl"
    if probe_res.exists():
        entries.append((probe_res, DATA_REPO,
                        "experiments/phase5_downstream_utility/results/probing_results.jsonl"))
    training_index = REPO / "experiments/phase5_downstream_utility/results/training_index.jsonl"
    if training_index.exists():
        entries.append((training_index, DATA_REPO,
                        "experiments/phase5_downstream_utility/results/training_index.jsonl"))

    # Paper-ready figures
    results_dir = REPO / "experiments/phase6_qualitative_latents/results"
    for prefix in ("phase61_", "phase63_"):
        for p in results_dir.glob(f"{prefix}*.png"):
            entries.append((p, DATA_REPO,
                            f"experiments/phase6_qualitative_latents/results/{p.name}"))
        for p in results_dir.glob(f"{prefix}*.md"):
            entries.append((p, DATA_REPO,
                            f"experiments/phase6_qualitative_latents/results/{p.name}"))

    # Phase 6.3 passage-probe results JSONL
    passage_probe = REPO / "experiments/phase6_qualitative_latents/results/passage_probe_results.jsonl"
    if passage_probe.exists():
        entries.append((passage_probe, DATA_REPO,
                        "experiments/phase6_qualitative_latents/results/passage_probe_results.jsonl"))
    return entries


def _file_sha1(p: Path) -> str:
    h = hashlib.sha1()
    with p.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true",
                    help="actually upload; without this, dry-run")
    ap.add_argument("--ckpts-only", action="store_true",
                    help="restrict to ckpt uploads (fastest sanity check)")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    hf_token = open("/workspace/.hf-token").read().strip()

    manifest = {}
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())

    entries = _plan_entries()
    if args.ckpts_only:
        entries = [e for e in entries if e[1] == CKPT_REPO]

    todo = []
    for local, repo, path_in_repo in entries:
        key = f"{repo}::{path_in_repo}"
        sha = _file_sha1(local)
        if manifest.get(key) == sha:
            continue
        todo.append((local, repo, path_in_repo, sha))

    if not todo:
        print("no new uploads.")
        return

    print(f"{len(todo)} files to upload{'  (DRY-RUN)' if not args.go else ''}:")
    by_repo: dict[str, int] = {}
    by_repo_bytes: dict[str, int] = {}
    for local, repo, path_in_repo, _ in todo:
        by_repo[repo] = by_repo.get(repo, 0) + 1
        by_repo_bytes[repo] = by_repo_bytes.get(repo, 0) + local.stat().st_size
    for repo in sorted(by_repo):
        print(f"  {repo}: {by_repo[repo]} files "
              f"({by_repo_bytes[repo]/1e6:.1f} MB)")

    if not args.go:
        print("\n(dry-run — rerun with --go to upload)")
        return

    from huggingface_hub import upload_file, HfApi
    api = HfApi(token=hf_token)

    for i, (local, repo, path_in_repo, sha) in enumerate(todo, 1):
        print(f"  [{i}/{len(todo)}] {path_in_repo} -> {repo}")
        try:
            upload_file(
                path_or_fileobj=str(local),
                path_in_repo=path_in_repo,
                repo_id=repo,
                repo_type=("dataset" if repo == DATA_REPO else "model"),
                token=hf_token,
            )
        except Exception as e:
            print(f"    FAILED: {e.__class__.__name__}: {str(e)[:200]}")
            continue
        manifest[f"{repo}::{path_in_repo}"] = sha
        # Persist after every file so an interruption doesn't lose
        # progress.
        MANIFEST.write_text(json.dumps(manifest, indent=2))

    print(f"\ndone. manifest at {MANIFEST}")


if __name__ == "__main__":
    main()
