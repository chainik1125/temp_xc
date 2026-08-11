"""O0: publish the signs-of-life checkpoints + logs to HuggingFace.

    uvx modal run experiments/diffusion_txc/topk_vs_topkdiff/modal_hf_push.py

Creates/updates the PUBLIC repo dmanningcoe/diffusion-topk-saes with a
gemma2-2b-l12/ subfolder: 4 checkpoints, training JSONLs, eval JSONs,
autointerp dumps, cache meta, and a model card.
"""

import pathlib

import modal

app = modal.App("topkdiff-hf-push")
vol = modal.Volume.from_name("diffusion-txc", create_if_missing=True)
hf_write = modal.Secret.from_name("hf-write-dmc")

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "huggingface_hub")

CARD = """---
license: apache-2.0
---

## diffusion-topk-saes

TopK sparse autoencoders trained with a denoising (DSM) objective vs
standard reconstruction, from the diffusion-txc workstream (temporal
crosscoders project). `gemma2-2b-l12/`: signs-of-life pair (H=16384,
k=40, AuxK, 10M tokens of Gemma-2-2B layer-12 resid-post, 2 seeds per
objective; `dsm` arms corrupt inputs with sigma ~ LogUniform(0.05, 1.0)
x activation RMS and reconstruct the clean target). Headline: absorption
0.179 vs 0.306, perturbation support-overlap 0.762 vs 0.627 at eps=0.5,
NMSE 0.331 vs 0.299, loss-recovered 0.841 vs 0.908. Full writeup in the
repo's experiments/diffusion_txc/topk_vs_topkdiff/SUMMARY.md.
"""


@app.function(image=image, timeout=1800, volumes={"/vol": vol},
              secrets=[hf_write])
def push(repo: str = "dmanningcoe/diffusion-topk-saes") -> dict:
    import os

    from huggingface_hub import HfApi

    token = (os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN")
             or next((v for k, v in os.environ.items()
                      if "HF" in k and v.startswith("hf_")), None))
    api = HfApi(token=token)
    api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md",
                    repo_id=repo)
    logs = pathlib.Path("/vol/logs")
    n = 0
    for f in sorted(logs.iterdir()):
        if f.suffix in (".pt", ".jsonl", ".json"):
            api.upload_file(path_or_fileobj=str(f),
                            path_in_repo=f"gemma2-2b-l12/{f.name}",
                            repo_id=repo)
            n += 1
            print(f"pushed {f.name}", flush=True)
    api.upload_file(path_or_fileobj="/vol/cache/meta.json",
                    path_in_repo="gemma2-2b-l12/cache_meta.json", repo_id=repo)
    return {"repo": repo, "files_pushed": n + 2}


@app.local_entrypoint()
def main():
    print(push.remote())
