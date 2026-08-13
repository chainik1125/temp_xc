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


@app.function(image=image, timeout=3600, volumes={"/vol": vol},
              secrets=[hf_write])
def sync(repo: str = "dmanningcoe/diffusion-topk-saes") -> dict:
    import os

    from huggingface_hub import HfApi

    token = (os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN")
             or next((v for k, v in os.environ.items()
                      if "HF" in k and v.startswith("hf_")), None))
    api = HfApi(token=token)
    pushed = []
    for prefix, sub in SYNC.items():
        root = pathlib.Path("/vol") / prefix
        if not root.exists():
            continue
        for f in sorted(root.rglob("*")):
            if f.is_file() and f.suffix in EXTS:
                rel = f.relative_to(root)
                api.upload_file(path_or_fileobj=str(f),
                                path_in_repo=f"{sub}/{rel}", repo_id=repo)
                pushed.append(f"{sub}/{rel}")
                print(f"pushed {sub}/{rel}", flush=True)
    return {"repo": repo, "files_pushed": len(pushed)}


@app.local_entrypoint()
def main():
    print(sync.remote())
