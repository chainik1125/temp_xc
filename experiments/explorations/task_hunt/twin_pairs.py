"""R30 spot-check TWIN PAIRS (λ̂ + dq, T16, hunt width) — the ratified
a⇄b swap's certificate item (matrix 1065b26cf items 4+5; deconfliction
c50f7af3e; swap be3d3fddc). Pre-registration = the LOG twin note
pushed WITH this driver, before any twin cell runs.

Design (both pairs at the exhibits' own hunt-width config: d_sae 2048,
k_pos 8, n_steps 8000, buffer 524288, T=16, EVAL_L=32, seed 42;
canonical runner rows via the synthetic grid pool — env-first agent
stamp per today's patch):

- **λ̂** (ds ward_real_lambda_base_l12): train the btk-only twin
  (txc_batchtopk_post_btkonly) FRESH — new train_key; the relu-mix
  counterpart is runpod-b's committed row e245559c84b46e60, its
  checkpoint READ-ONLY from their clone (same pod). No relu-mix
  retrain ⇒ no alias row.
- **dq** (ds dial_real_dqgap_llama31_8b_l14): no local counterpart
  checkpoint exists (panel ran on mac pods) ⇒ train the pair FRESH:
  txc_batchtopk_pre (a DISCLOSED deterministic re-run — duplicate
  train_key surfaced by the checker, never silently pooled;
  runpod-1's night-grid s42 re-run precedent) + txc_batchtopk_pre_
  btkonly (new key).

Compare: per-tensor torch.equal over aligned state dicts +
max |Δ| (fp32); verdict IDENTICAL iff every tensor equal.
Writes results/r30_twin_pairs_t16.json (repo-relative, committed by
the wrap entry).

Run: .venv/bin/python -m experiments.explorations.task_hunt.twin_pairs
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from explorations.synthetic import design, grid
from temp_bench.core.config import repo_root

HERE = Path(__file__).resolve().parent
D_SAE, K_POS, N_STEPS, BUFFER, EVAL_L, SEED = 2048, (8,), 8000, 524288, 32, 42
RPB_CKPTS = Path("/workspace/agents/runpod-b/temp_xc/checkpoints")
LAM_COUNTERPART = "e245559c84b46e60"

PAIRS = {
    "lambda": {
        "ds": "ward_real_lambda_base_l12",
        "train": (("txc_batchtopk_post_btkonly", "post_btk"),),
        "counterpart": ("external", LAM_COUNTERPART),
        "btk_arch": "txc_batchtopk_post_btkonly",
    },
    "dq": {
        "ds": "dial_real_dqgap_llama31_8b_l14",
        "train": (("txc_batchtopk_pre", "pre"),
                  ("txc_batchtopk_pre_btkonly", "pre_btk")),
        "counterpart": ("local", "txc_batchtopk_pre"),
        "btk_arch": "txc_batchtopk_pre_btkonly",
    },
}


def _cells(ds, archs):
    cells = design.uniform_cells(
        ds, F=D_SAE, n_steps=N_STEPS, d_saes=[D_SAE], k_pos_sweep=K_POS,
        archs=archs, window_ts=(16,), L=EVAL_L, seeds=(SEED,),
        untrained=False, log=print)
    for c in cells:
        c["buffer_tokens"] = BUFFER
    return cells


def _keys_from_leaderboard(ds, since_ts):
    """train_key per arch from the canonical rows written by this run —
    the authoritative source, no re-derivation."""
    out = {}
    with open(repo_root() / "results" / "leaderboard.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if (r.get("datasource") == ds and r.get("seed") == SEED
                    and r.get("ts", "") >= since_ts
                    and (r.get("training_cfg") or {}).get("n_steps") == N_STEPS):
                ov = (r["training_cfg"].get("arch_hparams_override") or {})
                if ov.get("d_sae") == D_SAE and ov.get("T") == 16:
                    out[r["arch"]] = r["train_key"]
    return out


def _compare(path_a: Path, path_b: Path):
    a, b = load_file(str(path_a)), load_file(str(path_b))
    keys = sorted(set(a) | set(b))
    per, n_eq, mx = {}, 0, 0.0
    for k in keys:
        if k not in a or k not in b or a[k].shape != b[k].shape:
            per[k] = "MISSING/SHAPE"
            continue
        eq = bool(torch.equal(a[k], b[k]))
        d = float((a[k].to(torch.float32) - b[k].to(torch.float32))
                  .abs().max().item())
        n_eq += eq
        mx = max(mx, d)
        if not eq:
            per[k] = d
    return {"n_tensors": len(keys), "n_equal": n_eq,
            "identical": n_eq == len(keys), "max_abs_delta": mx,
            "diverging": dict(sorted(
                ((k, v) for k, v in per.items() if v != 0.0),
                key=lambda kv: -(kv[1] if isinstance(kv[1], float) else 1e9)
            )[:8])}


def main(only=None):
    # ONE run_pool call per process (the grid machinery forks workers;
    # a second pool in the same process wedged on an inherited lock —
    # observed futex_wait deadlock, 2026-07-28). Select pairs via argv
    # and MERGE into the existing results JSON across invocations.
    since = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 60))
    dst = HERE / "results" / "r30_twin_pairs_t16.json"
    out = (json.loads(dst.read_text()) if dst.exists() else
           {"card_note": "LOG twin note (committed with this driver)",
            "config": {"d_sae": D_SAE, "k_pos": 8, "n_steps": N_STEPS,
                       "buffer_tokens": BUFFER, "T": 16, "eval_L": EVAL_L,
                       "seed": SEED},
            "pairs": {}})
    for name, spec in PAIRS.items():
        if only and name not in only:
            continue
        cells = _cells(spec["ds"], spec["train"])
        res = grid.run_pool(
            cells, HERE / "results" / f"r30_twin_{name}_t16_pool.json",
            max_workers=1, tag=f"r30twin/{name}")
        fails = [r for r in res if not r.get("ok")]
        if fails:
            out["pairs"][name] = {"ok": False, "fails": len(fails)}
            continue
        keys = _keys_from_leaderboard(spec["ds"], since)
        btk_key = keys.get(spec["btk_arch"])
        kind, ref = spec["counterpart"]
        ref_key = keys.get(ref) if kind == "local" else ref
        root = repo_root() / "checkpoints"
        ref_root = root if kind == "local" else RPB_CKPTS
        cmp_res = _compare(ref_root / ref_key / "model.safetensors",
                           root / btk_key / "model.safetensors")
        out["pairs"][name] = {
            "ok": True, "relu_mix_key": ref_key,
            "relu_mix_source": ("fresh deterministic re-run (disclosed)"
                                if kind == "local"
                                else f"runpod-b clone (read-only)"),
            "btk_key": btk_key, **cmp_res}
        print(f"[{name}] identical={cmp_res['identical']} "
              f"max|Δ|={cmp_res['max_abs_delta']:.3e} "
              f"({cmp_res['n_equal']}/{cmp_res['n_tensors']})", flush=True)
    dst.parent.mkdir(exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    print(f"-> {dst}")


if __name__ == "__main__":
    main(only=set(sys.argv[1:]) or None)
