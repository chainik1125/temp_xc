"""Bulk-sync Modal volume prefixes to the public HF repo.

Training-job inline pushes fail 403 (they get the read-only `hf-token`
secret); this uses the write secret `hf-write-dmc` like modal_hf_push.py.

    uvx modal run experiments/diffusion_txc/psc/modal_hf_sync.py
"""

import pathlib

import modal

app = modal.App("dtxc-hf-sync")
vol = modal.Volume.from_name("diffusion-txc", create_if_missing=True)
hf_write = modal.Secret.from_name("hf-write-dmc")

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "huggingface_hub")

# volume prefix -> repo subfolder
SYNC = {
    "bayes_gate": "gemma2-2b-l12-bayes-gate",
    "txc_w6": "llama31-8b-resid10-w6",
    "txc_w6_mix": "llama31-8b-resid10-w6-mix",
    "txc_w6_distill": "r1-distill-8b-resid10-w6",
    "ooc_recal": "probes/ooc_recal",
    "logs_bayes_evals": "evals/bayes-gate",
    "backtracking_eval": "evals/backtracking",
}
EXTS = {".pt", ".jsonl", ".json", ".md", ".png"}


@app.function(image=image, timeout=10800, volumes={"/vol": vol},
              secrets=[hf_write])
def sync(repo: str = "dmanningcoe/diffusion-topk-saes") -> dict:
    import os
    import re
    import time

    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    token = (os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN")
             or next((v for k, v in os.environ.items()
                      if "HF" in k and v.startswith("hf_")), None))
    api = HfApi(token=token)
    patterns = [f"*{e}" for e in EXTS]
    done = []
    for prefix, sub in SYNC.items():
        root = pathlib.Path("/vol") / prefix
        if not root.exists():
            continue
        for attempt in range(8):
            try:
                # one commit per prefix - dodges the 128 commits/hour limit
                api.upload_folder(folder_path=str(root), path_in_repo=sub,
                                  repo_id=repo, allow_patterns=patterns,
                                  commit_message=f"sync {prefix}")
                done.append(prefix)
                print(f"synced {prefix} -> {sub}", flush=True)
                break
            except HfHubHTTPError as e:                       # noqa: PERF203
                if "429" in str(e) or "rate limit" in str(e).lower():
                    m = re.search(r"(\d+) minutes", str(e))
                    wait = int(m.group(1)) * 60 + 60 if m else 360
                    print(f"[{prefix}] rate-limited; sleeping {wait}s",
                          flush=True)
                    time.sleep(wait)
                else:
                    raise
    print("ALL SYNCED:", done, flush=True)
    return {"repo": repo, "prefixes": done}


@app.local_entrypoint()
def main():
    call = sync.spawn()
    print("SPAWNED:", call.object_id, "- detach-safe, exiting")
