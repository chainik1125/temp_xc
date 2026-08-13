"""Push the diffusion-txc arc's doc set to the public HF repo (docs/).

    uvx modal run --detach experiments/diffusion_txc/psc/modal_docs_push.py::main
"""

import pathlib

import modal

try:
    ROOT = pathlib.Path(__file__).resolve().parents[3]
except IndexError:
    ROOT = pathlib.Path("/work")

app = modal.App("dtxc-docs-push")
hf_write = modal.Secret.from_name("hf-write-dmc")
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "huggingface_hub")

DOCS = [
    "docs/dmitry/proposals/2026-08-10_bird_temporal_codes.md",
    "docs/dmitry/proposals/2026-08-10_bird_clock_results.md",
    "docs/dmitry/proposals/2026-08-10_bird_novelty_check.md",
    "experiments/diffusion_txc/2026-08-11_jumprelu_mmse_note.md",
    "docs/dmitry/proposals/2026-08-11_backtracking_detection_dsm.md",
    "docs/dmitry/proposals/2026-08-11_backtracking_steering_dsm.md",
    "docs/dmitry/proposals/2026-08-12_arc_review.md",
    "docs/dmitry/proposals/2026-08-12_bird_transfer_theory.md",
    "docs/dmitry/proposals/2026-08-13_dsm_postmortem.md",
]
FIGS = sorted(str(p.relative_to(ROOT))
              for p in (ROOT / "docs/dmitry/proposals/figures").glob("bird_*.png"))

img = image
for rel in DOCS + FIGS:
    img = img.add_local_file(str(ROOT / rel), f"/work/{rel}")


@app.function(image=img, timeout=7200, secrets=[hf_write])
def push(repo: str = "dmanningcoe/diffusion-topk-saes") -> dict:
    import os
    import re
    import shutil
    import time

    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    token = (os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN")
             or next((v for k, v in os.environ.items()
                      if "HF" in k and v.startswith("hf_")), None))
    api = HfApi(token=token)
    stage = pathlib.Path("/tmp/docs_stage")
    (stage / "figures").mkdir(parents=True, exist_ok=True)
    for rel in DOCS:
        shutil.copy(f"/work/{rel}", stage / pathlib.Path(rel).name)
    for rel in FIGS:
        shutil.copy(f"/work/{rel}", stage / "figures" / pathlib.Path(rel).name)
    for attempt in range(8):
        try:
            api.upload_folder(folder_path=str(stage), path_in_repo="docs",
                              repo_id=repo, commit_message="arc docs")
            print("DOCS PUSHED", flush=True)
            return {"files": len(DOCS) + len(FIGS)}
        except HfHubHTTPError as e:                           # noqa: PERF203
            if "429" in str(e) or "rate limit" in str(e).lower():
                m = re.search(r"(\d+) minutes", str(e))
                wait = int(m.group(1)) * 60 + 60 if m else 360
                print(f"rate-limited; sleeping {wait}s", flush=True)
                time.sleep(wait)
            else:
                raise
    return {"files": 0, "error": "rate-limit retries exhausted"}


@app.local_entrypoint()
def main():
    call = push.spawn()
    print("SPAWNED:", call.object_id, "- detach-safe, exiting")
