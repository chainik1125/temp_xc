"""Durability push (b4ec84b04 item 2): mirror this pod's RLHF ckpts to HF.

Certificate-evidence pairs FIRST, then every trained (n_steps=25000)
RLHF ckpt on this pod. Layout follows the repo convention:
`ckpts/<train_key>/model.safetensors` (0e644c65b ratified fleet convention, dataset repo temp-bench-data).
Receipts (train_key, sha256, hf path) land in
results/hf_durability_receipts.jsonl — values only, never tokens.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi

REPO = "han1823123123/temp-bench-data"
PREFIX = "ckpts"
DATASOURCE = "gemma_2_2b_base_l12_phase7"
CERT_KEYS = [  # certificate evidence, pushed first
    "a67f63b5e0e15d6e", "f1f586849f07efcb",
    "5774f6c8b6d28938", "3d46dfd07b50eac0",
    "eff51d4fb0ec4088", "25f7c9471f052b96",
]


def main():
    tok = open("/workspace/.tokens/hf_token").read().strip()
    api = HfApi(token=tok)
    rows = {}
    with open("checkpoints/manifest.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if (r.get("datasource") == DATASOURCE
                    and r.get("training_cfg", {}).get("n_steps") == 25000
                    and "/runpod-2/" in r.get("local_path", "")):
                rows[r["train_key"]] = r  # latest row per key wins
    order = [k for k in CERT_KEYS if k in rows]
    order += [k for k in sorted(rows) if k not in CERT_KEYS]
    print(f"[push] {len(order)} ckpts ({len(CERT_KEYS)} certificate-first)")
    receipts = []
    out = Path("results/hf_durability_receipts.jsonl")
    for i, tk in enumerate(order):
        r = rows[tk]
        p = Path(r["local_path"])
        if not p.exists():
            print(f"[{i+1}/{len(order)}] MISSING {tk} {p}")
            receipts.append({"train_key": tk, "status": "MISSING", "local_path": str(p)})
            continue
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        dest = f"{PREFIX}/{tk}/model.safetensors"
        api.upload_file(path_or_fileobj=str(p), path_in_repo=dest,
                        repo_id=REPO, repo_type="dataset",
                        commit_message=f"actmix_rlhf durability: {tk} ({r['arch']} s{r['seed']})")
        receipts.append({"train_key": tk, "arch": r["arch"], "seed": r["seed"],
                         "sha256": sha, "hf_path": dest, "status": "PUSHED"})
        print(f"[{i+1}/{len(order)}] PUSHED {tk} sha={sha[:16]}…")
        out.write_text("\n".join(json.dumps(x) for x in receipts) + "\n")
    manifest_txt = "\n".join(json.dumps(rows[k]) for k in order if rows[k])
    api.upload_file(path_or_fileobj=manifest_txt.encode(),
                    path_in_repo=f"{PREFIX}/actmix_rlhf_manifest.jsonl",
                    repo_id=REPO, repo_type="dataset",
                    commit_message="actmix_rlhf durability: manifest")
    # spot-check receipt: HF LFS sha256 of the T16 twin vs local
    info = api.get_paths_info(REPO, [f"{PREFIX}/5774f6c8b6d28938/model.safetensors"], repo_type="dataset")
    lfs_sha = info[0].lfs.sha256 if info and info[0].lfs else None
    local_sha = next(x["sha256"] for x in receipts if x["train_key"] == "5774f6c8b6d28938")
    print(f"[spot-check] T16 twin HF-LFS sha == local sha: {lfs_sha == local_sha} ({str(lfs_sha)[:16]}…)")
    receipts.append({"spot_check": "5774f6c8b6d28938", "hf_lfs_sha256": lfs_sha,
                     "match": lfs_sha == local_sha})
    out.write_text("\n".join(json.dumps(x) for x in receipts) + "\n")
    print("[push] DONE", sum(1 for x in receipts if x.get("status") == "PUSHED"), "pushed")


if __name__ == "__main__":
    main()
