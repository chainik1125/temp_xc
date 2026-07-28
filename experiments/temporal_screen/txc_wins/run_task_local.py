"""Run `harness.run_task` on a GPU box over ssh, without Modal.

Modal is the usual driver (`run_task_modal.py`); this exists because that workspace hits its
spend limit and RunPod is the fallback. Same arguments, same harness, same outputs.

    export TMPDIR=/workspace/tmp
    python run_task_local.py --task struqposx_completion_real_tr \
        --task-test struqposx_completion_real_ev --layer 14 --k-seg 12 \
        --gen-tokens 48 --n-gen 40 --gen-alphas "-0.25,-0.1,-0.05,0.05,0.1,0.25" \
        --out /workspace/out/struq_gen.json

DOSE GRID. For StruQ the default `[-2,-1,-0.5,0.5,1,2]` is **entirely inside the saturated
regime**: the metric gradient sits almost wholly on the final segment, so at matched Frobenius
norm the concentrated arms carry ~sqrt(T) more norm per position and leave the linear regime at
roughly a third of the dose. Measured on the gate, `grad_slab` beats `broadcast_optimal` at
alpha <= 0.25 and loses at 0.5 and 1.0. Pass a capped grid or the run measures saturation.
"""
import argparse
import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
ROOT = _HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(_HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--task-test", default="")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--k-seg", type=int, default=12)
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=80)
    ap.add_argument("--d-sae", type=int, default=4096)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-win", type=int, default=32)
    ap.add_argument("--alphas", default="-0.25,-0.1,-0.05,0.05,0.1,0.25")
    ap.add_argument("--tsae-k", type=int, default=8)
    ap.add_argument("--tsaep-k", type=int, default=8)
    ap.add_argument("--txc-k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=31415)
    ap.add_argument("--dict-seed", type=int, default=0)
    ap.add_argument("--n-grad", type=int, default=24)
    ap.add_argument("--select-by", default="gradient")
    ap.add_argument("--recipe", default="0.0003,6000,0.001,6000,0.003,6000")
    ap.add_argument("--gen-tokens", type=int, default=0)
    ap.add_argument("--n-gen", type=int, default=0)
    ap.add_argument("--gen-alphas", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from txc_wins.harness import run_task
    from txc_wins.tasks import TASKS

    if a.task not in TASKS:
        raise SystemExit(f"unknown task {a.task!r}")
    rec = [float(x) for x in a.recipe.split(",")] if a.recipe else [0] * 6
    r = run_task(
        make_pair=TASKS[a.task](a.k_seg),
        make_pair_test=(TASKS[a.task_test](a.k_seg) if a.task_test else None),
        model_id=a.model, layer=a.layer, k_seg=a.k_seg,
        n_train=a.n_train, n_test=a.n_test, d_sae=a.d_sae, k=a.k,
        steps=a.steps, lr=a.lr, batch_win=a.batch_win,
        alphas=[float(x) for x in a.alphas.split(",")],
        tsae_k=(a.tsae_k or None), tsaep_k=(a.tsaep_k or None), txc_k=(a.txc_k or None),
        seed=a.seed, dict_seed=a.dict_seed, n_grad=a.n_grad, select_by=a.select_by,
        gen_tokens=a.gen_tokens, n_gen=a.n_gen,
        gen_alphas=([float(x) for x in a.gen_alphas.split(",")] if a.gen_alphas else None),
        sae_lr=rec[0] or None, sae_steps=rec[1] or None,
        txc_lr=rec[2] or None, txc_steps=rec[3] or None,
        tsae_lr=rec[4] or None, tsae_steps=rec[5] or None,
        tsaep_lr=rec[0] or None, tsaep_steps=rec[1] or None,
    )
    r["task"], r["task_test"] = a.task, a.task_test or a.task
    p = pathlib.Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(r, indent=2))
    print("[saved]", p, flush=True)


if __name__ == "__main__":
    main()
