"""Modal entrypoint for the TXC-vs-SAE-vs-tSAE task screen.

One task per invocation; the pipeline, the controls and the verdict all live in
`harness.py`, so a new task is a factory in `tasks.py` and nothing else. See those two files
for the design and for what each control rules out.

    modal run experiments/temporal_screen/txc_wins/run_task_modal.py --task order
"""
import pathlib
import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

app = modal.App("txcwins-task")
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers", "accelerate", "numpy")
    .add_local_dir(str(ROOT / "src"), "/work/src")
    .add_local_dir(str(ROOT / "temporal_crosscoders"), "/work/temporal_crosscoders")
    .add_local_dir(str(_here.parent), "/work/txc_wins")
)


@app.function(gpu="A10G", image=image, timeout=21600)
def run(task: str, model_id: str, layer: int, k_seg: int, n_train: int, n_test: int,
        d_sae: int, k: int, steps: int, lr: float, batch_win: int, alphas: list,
        tsae_l1: float, tsae_k: int, n_perm: int, seed: int, dict_seed: int):
    import sys
    sys.path.insert(0, "/work")
    from txc_wins.harness import run_task
    from txc_wins.tasks import TASKS

    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; have {sorted(TASKS)}")
    r = run_task(make_pair=TASKS[task](k_seg), model_id=model_id, layer=layer,
                 k_seg=k_seg, n_train=n_train, n_test=n_test, d_sae=d_sae, k=k,
                 steps=steps, lr=lr, batch_win=batch_win, alphas=alphas,
                 tsae_l1=(tsae_l1 if tsae_l1 > 0 else None),
                 tsae_k=(tsae_k if tsae_k > 0 else None), n_perm=n_perm, seed=seed,
                 dict_seed=dict_seed)
    r["task"] = task
    return r


@app.local_entrypoint()
def main(task: str = "order", model: str = "Qwen/Qwen2.5-1.5B-Instruct", layer: int = -1,
         k_seg: int = 12, n_train: int = 800, n_test: int = 100, d_sae: int = 4096,
         k: int = 8, steps: int = 2000, lr: float = 3e-4, batch_win: int = 32,
         alphas: str = "0.25,0.5,1.0,2.0", tsae_l1: float = 0.0, tsae_k: int = 0,
         n_perm: int = 0,
         seed: int = 31415, dict_seed: int = 0, tag: str = ""):
    import json
    r = run.remote(task, model, layer, k_seg, n_train, n_test, d_sae, k, steps, lr,
                   batch_win, [float(x) for x in alphas.split(",")], tsae_l1, tsae_k,
                   n_perm, seed, dict_seed)
    outdir = ROOT / "results" / "txc_wins"
    outdir.mkdir(parents=True, exist_ok=True)
    name = f"{task}{tag}.json"
    (outdir / name).write_text(json.dumps(r, indent=2))
    print("[saved]", outdir / name)
