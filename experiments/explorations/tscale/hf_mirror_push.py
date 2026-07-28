"""tscale lane durability push (Han freeze order 11227ce0d item 1).

Mirrors every L1/L2 screen ckpt on this pod to the ratified fleet
dataset repo under `ckpts/tscale/<cfg_hash>/` (lane-scoped so cfg
hashes can never collide with RLHF train_keys). Decision-grade ckpts
first. Receipts (cfg_hash, sha256, hf path) land in
results/hf_durability_receipts.jsonl — values only, never tokens.
Re-runnable: rows already PUSHED per the receipts file are skipped
(how C4/C5-T16 stragglers get picked up at drain).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi

REPO = "han1823123123/temp-bench-data"
PREFIX = "ckpts/tscale"
CKPT_DIR = Path("experiments/explorations/tscale/results/ckpts")
RECEIPTS = Path("experiments/explorations/tscale/results/hf_durability_receipts.jsonl")
DECISION_FIRST = [  # RESULTS-quoted cells: diag 20k chain, twin, C4/C5
    "9d9567ddd6a4ef6e", "72a9f0a979cf575c", "c29ea51f2aaaa3c4",
    "040b9f18e5d919ae", "83a57e4412200a37", "18563d86283b03e1",
    "37b53d95ea3d0289", "cfdac03dcaa38b97", "ed2bacbec941e4e3",
]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    tok = open("/workspace/.tokens/hf_token").read().strip()
    api = HfApi(token=tok)
    done = set()
    if RECEIPTS.exists():
        for line in open(RECEIPTS):
            r = json.loads(line)
            if r.get("status") == "PUSHED":
                done.add(r["cfg_hash"])
    keys = sorted(p.stem for p in CKPT_DIR.glob("*.safetensors"))
    order = [k for k in DECISION_FIRST if k in keys]
    order += [k for k in keys if k not in DECISION_FIRST]
    order = [k for k in order if k not in done]
    print(f"[push] {len(order)} ckpts to do ({len(done)} already pushed)")
    with open(RECEIPTS, "a") as out:
        for i, key in enumerate(order):
            st = CKPT_DIR / f"{key}.safetensors"
            meta = CKPT_DIR / f"{key}.json"
            sha = sha256_file(st)
            dest = f"{PREFIX}/{key}/model.safetensors"
            api.upload_file(path_or_fileobj=str(st), path_in_repo=dest,
                            repo_id=REPO, repo_type="dataset",
                            commit_message=f"tscale durability: {key}")
            if meta.exists():
                api.upload_file(path_or_fileobj=str(meta),
                                path_in_repo=f"{PREFIX}/{key}/meta.json",
                                repo_id=REPO, repo_type="dataset",
                                commit_message=f"tscale durability: {key} meta")
            rec = {"cfg_hash": key, "sha256": sha, "hf_path": dest,
                   "bytes": st.stat().st_size, "status": "PUSHED"}
            info = api.get_paths_info(REPO, [dest], repo_type="dataset")
            lfs_sha = info[0].lfs.sha256 if info and info[0].lfs else None
            rec["hf_lfs_match"] = (lfs_sha == sha)
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(f"[{i+1}/{len(order)}] PUSHED {key} sha={sha[:16]}… "
                  f"lfs_match={rec['hf_lfs_match']}")
    print("[push] DONE")


if __name__ == "__main__":
    main()
