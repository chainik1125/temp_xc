"""Back up final 300K dictionaries privately; verify Hub hashes before cleanup.

Credentials are read from a temporary mode-0600 file outside the experiment
tree and passed only through the child environment, never CLI arguments.
This script never stops/deletes the pod and never publishes a public repo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def run(args):
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import RepositoryNotFoundError

    root = args.root.resolve()
    token_path = args.token_file.resolve()
    token = token_path.read_text().strip()
    if not token or token_path.stat().st_mode & 0o077:
        raise ValueError("Temporary credential must be nonempty and mode 0600")
    receipt_path = root / "results/checkpoint_backup_receipt.json"
    receipt = {"status": "preparing", "started_unix": time.time(),
               "repo_id": args.repo_id, "repo_type": "model", "private": True}
    atomic(receipt_path, receipt)
    try:
        api = HfApi(token=token)
        who = api.whoami()
        if args.repo_id.split("/")[0] != who["name"]:
            raise ValueError("Backup must be in the authenticated personal namespace")
        stage = root / "hf_checkpoint_backup"
        stage.mkdir(exist_ok=True)
        expected = {(arch, width, seed) for arch, width in (
            ("txc_base", 32768), ("topk_sae", 32768), ("tsae_paper", 32768),
            ("stacked_sae", 32768), ("tsae_paper", 16384)) for seed in (1, 2, 42)}
        files, cells = [], []

        def register(source, relative, *, checkpoint=False, expected_hash=None):
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            source_hash = digest(source)
            if expected_hash and source_hash != expected_hash:
                raise ValueError(f"Source checksum mismatch: {relative}")
            if target.exists():
                if digest(target) != source_hash:
                    raise ValueError(f"Staging collision: {relative}")
            elif checkpoint:
                os.link(source, target)
            else:
                shutil.copyfile(source, target)
            files.append({"path": str(relative), "bytes": source.stat().st_size, "sha256": source_hash})

        for arch, width, seed in sorted(expected):
            name = f"{arch}_d{width}_seed{seed}"
            cell = root / "results/cells" / name
            detection = json.loads((cell / "detection.json").read_text())
            config = json.loads((cell / "checkpoint/config.json").read_text())
            if (config["arch"], config["d_sae"], config["seed"]) != (arch, width, seed):
                raise ValueError(f"Checkpoint identity mismatch: {name}")
            if config["n_steps_completed"] != 300000 or config["status"] != "complete":
                raise ValueError(f"Incomplete dictionary: {name}")
            for filename, key in (("model.safetensors", "checkpoint_sha256"),
                                  ("config.json", "checkpoint_config_sha256")):
                register(cell / "checkpoint" / filename, Path("checkpoints") / name / filename,
                         checkpoint=filename.endswith(".safetensors"),
                         expected_hash=detection["provenance"][key])
            cells.append({"cell": name, "n_steps_completed": 300000,
                          "checkpoint_sha256": detection["provenance"]["checkpoint_sha256"]})
            print(json.dumps({"phase": "source_verified", "cell": name}), flush=True)

        register(root / "backtracking_compact_results.zip", Path("results/backtracking_compact_results.zip"))
        source_tar = root / "historical_recovery_source.tar.gz"
        if not source_tar.exists():
            with tarfile.open(source_tar, "w:gz") as archive:
                base = root / "historical/purified"
                for name in ("src", "configs", "pyproject.toml"):
                    if (base / name).exists():
                        archive.add(base / name, arcname=name,
                                    filter=lambda info: None if "__pycache__" in Path(info.name).parts else info)
        register(source_tar, Path("source/historical_recovery_source.tar.gz"))
        source_path = root / "code/backup_checkpoints_hf.py"
        register(source_path, Path("source/backup_checkpoints_hf.py"))
        readme = stage / "README.md"
        readme.write_text("""---
tags:
- sparse-autoencoder
- mechanistic-interpretability
---
# Backtracking 300K checkpoint backup

Private preservation of 15 final dictionary checkpoints, each trained for
300,000 completed optimizer steps, with seeds 1, 2 and 42. TXC, shared TopK
SAE, T-SAE 32K and independent-position Stacked SAE form the core comparison;
T-SAE 16K is a separate width sensitivity. TXC-pro is excluded.

Each `checkpoints/<cell>/` directory contains the exact final `model.safetensors`
and `config.json`. The internal `txc_base` identifier is retained for artifact
compatibility; the paper display label is TXC. These are dictionary models,
not standalone language models or Hugging Face Transformers checkpoints.

`checkpoint_manifest.json` lists SHA-256 checksums, sizes and completion status.
`results/backtracking_compact_results.zip` contains detection results, saved
steering generations, deferred judge templates, plots, source snapshots and
per-file checksums. No paid judging has been performed. Preserve the validation
selection / held-out test separation documented in that bundle.

`source/historical_recovery_source.tar.gz` preserves the architecture source
from Git commit 284a8bf5e3e5a7cc094dd68c6fa5a92a9fd4eec3. Extract it and the
campaign source snapshot in the results bundle to recover the exact loaders
and protocol. Training inputs are revision/hash-pinned in the campaign code.
Activation datasets, the pretrained language model, and optimizer/resume
snapshots are intentionally excluded; the final inference dictionaries and
all completed experiment outputs are preserved.

To restore (requires access to this private repo):

```sh
hf download REPO_ID --repo-type model --local-dir restored-backtracking
```

Use the revision in the external verification receipt for an immutable restore.
""".replace("REPO_ID", args.repo_id))
        files.append({"path": "README.md", "bytes": readme.stat().st_size, "sha256": digest(readme)})
        manifest_path = stage / "checkpoint_manifest.json"
        atomic(manifest_path, {"schema": "backtracking-final-checkpoints-v1", "cells": cells,
                               "checkpoint_count": len(cells), "files": files,
                               "excluded": ["activation datasets", "base language model", "optimizer/resume states"]})
        verify_files = files + [{"path": manifest_path.name, "bytes": manifest_path.stat().st_size,
                                "sha256": digest(manifest_path)}]
        try:
            info = api.repo_info(args.repo_id, repo_type="model")
            if not info.private:
                raise ValueError("Existing repository is public; refusing upload")
            existing = set(api.list_repo_files(args.repo_id, repo_type="model"))
            allowed = {r["path"] for r in verify_files} | {".gitattributes"}
            if existing - allowed:
                raise ValueError("Existing repository contains unrelated files; refusing to reuse it")
        except RepositoryNotFoundError:
            api.create_repo(args.repo_id, repo_type="model", private=True, exist_ok=False)
        if not api.repo_info(args.repo_id, repo_type="model").private:
            raise ValueError("Repository privacy verification failed")
        receipt.update(status="uploading", checkpoint_count=15,
                       bytes=sum(r["bytes"] for r in verify_files), files=verify_files)
        atomic(receipt_path, receipt)
        env = dict(os.environ, HF_TOKEN=token, HF_HUB_DISABLE_PROGRESS_BARS="1",
                   HF_HUB_DISABLE_TELEMETRY="1")
        command = [str(Path(sys.executable).with_name("hf")), "upload-large-folder",
                   args.repo_id, str(stage), "--repo-type", "model", "--private",
                   "--num-workers", "4", "--no-bars"]
        subprocess.run(command, env=env, check=True)
        info = api.repo_info(args.repo_id, repo_type="model", files_metadata=True)
        if not info.private:
            raise ValueError("Uploaded repository is not private")
        revision = info.sha
        remote = {r.rfilename: r for r in info.siblings}
        verified = []
        for row in verify_files:
            item = remote.get(row["path"])
            if item is None or item.size != row["bytes"]:
                raise ValueError(f"Uploaded file missing or wrong size: {row['path']}")
            lfs = getattr(item, "lfs", None)
            sha = getattr(lfs, "sha256", None) if lfs is not None else None
            if sha is None and isinstance(lfs, dict):
                sha = lfs.get("sha256")
            if sha is None:
                if row["bytes"] > 8 << 20:
                    raise ValueError(f"Large object has no server-side SHA-256: {row['path']}")
                downloaded = hf_hub_download(args.repo_id, row["path"], revision=revision,
                                             repo_type="model", token=token)
                sha = digest(downloaded)
                method = "downloaded at pinned revision and SHA-256 checked"
            else:
                method = "Hub LFS SHA-256 and byte size match source"
            if sha != row["sha256"]:
                raise ValueError(f"Uploaded file checksum mismatch: {row['path']}")
            verified.append({**row, "verification": method})
        receipt.update(status="verified", revision=revision,
                       repo_url=f"https://huggingface.co/{args.repo_id}", private=info.private,
                       verified_files=verified, finished_unix=time.time(),
                       preserved="All final 300K dictionary weights/configs and compact experiment outputs",
                       pod_action="No pod stop/delete action taken")
        atomic(receipt_path, receipt)
        print(json.dumps({"status": "verified", "repo_id": args.repo_id, "revision": revision,
                          "checkpoints": 15, "files": len(verified), "private": True}), flush=True)
    except Exception as exc:
        message = str(exc).replace(token, "[redacted]")
        receipt.update(status="failed", error=f"{type(exc).__name__}: {message}")
        atomic(receipt_path, receipt)
        print(json.dumps({"status": "failed", "error": receipt["error"]}), flush=True)
        return 1
    finally:
        token_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/backtracking"))
    parser.add_argument("--token-file", type=Path, default=Path("/workspace/.tokens/backtracking_hf_token"))
    parser.add_argument("--repo-id", required=True)
    raise SystemExit(run(parser.parse_args()))
