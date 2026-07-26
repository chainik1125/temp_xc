"""BTK re-run driver — paper-arch composite vs btk-only across window T.

ACTMIX "Dmitry's re-run gate" (briefings/actmix-shared.md): does the
PAPER arch's d(perf)/dT improve when the composite TopK-then-ReLU
sparsity path is replaced by btk-only (BatchTopK on raw pre-acts, no
ReLU)? Runs the § 4 synthetic benches with BOTH arms of the paper arch:

- arm `paper-match` composition: ``txc_base``   (TopK -> ReLU, k_win = k_pos*T)
- arm `btk-only`:                ``txc_base_btk`` (same budget, no ReLU)

Per-token baselines are NOT rerun (existing leaderboard rows stand).

Grid (one shared parameter set across all cells):
- datasources: toy_markov_n20_d40_noisy (Denoising),
               toy_coupled_K10_M20_d256 (Coupling)
- T ∈ {1, 2, 4, 5, 8, 10}; k_pos ∈ {1, 2, 5}; seeds {1, 2, 3}
  (k_pos ∈ {10, 20} and T=20 dropped: k_pos*T >= d_sae=20 clips both
  arms to a dense code — degenerate for the selection-rule comparison).
- training: n_steps 6_000 (hunt-scale; no cache reuse is possible at
  L=40 anyway), batch 1024, buffer 2M.
- eval: eval_window_L = 40 (divisible by every T above) — ONE uniform
  eval protocol for the whole comparison. NOTE: this differs from the
  legacy L=5 rows and the hunt's L=32 rows; comparisons stay inside this
  sweep.

Key convention: ``arch_hparams_override`` carries ``k_pos`` always and
``T`` only when it differs from the registry default (5) — house style;
maximises checkpoint reuse.

Usage (one shard = arch x datasource x T; ~15 cells):
    python -m experiments.explorations.btk_rerun.driver \
        --arch txc_base_btk --datasource toy_markov_n20_d40_noisy --T 8
    python -m experiments.explorations.btk_rerun.driver --smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ARMS = {"txc_base": "paper-match", "txc_base_btkonly": "btk-only",
        "txc_base_relumix": "relu-mix", "txc_base_perwinraw": "perwin-raw"}
DATASOURCES = ["toy_markov_n20_d40_noisy", "toy_coupled_K10_M20_d256"]
T_GRID = [1, 2, 4, 5, 8, 10]
K_POS_GRID = [1, 2, 5]
SEEDS = [1, 2, 3]
EVAL_WINDOW_L = 40
N_STEPS = 6_000
T_DEFAULT = 5   # registry default for both arms — omitted from override


def run_shard(
    arch: str,
    datasource: str,
    T: int,
    *,
    k_pos_grid=tuple(K_POS_GRID),
    seeds=tuple(SEEDS),
    n_steps: int = N_STEPS,
    d_sae: int = 0,
    probe: bool = False,
    smoke: bool = False,
    allow_dirty: bool = False,
    agent: str = "dmitry-btk-sprint",
) -> list[dict]:
    from temp_bench.core.runner import run_experiment
    from temp_bench.core.schemas import TrainingConfig

    rows: list[dict] = []
    cells = [(k, s) for k in k_pos_grid for s in seeds]
    for i, (k_pos, seed) in enumerate(cells, 1):
        override: dict = {"k_pos": int(k_pos)}
        if int(T) != T_DEFAULT:
            override["T"] = int(T)
        if d_sae:
            override["d_sae"] = int(d_sae)   # sensitivity wing (de-clips grid)
        training_cfg = TrainingConfig(
            n_steps=int(n_steps),
            batch_size=1024,
            buffer_tokens=2_000_000,
            arch_hparams_override=override,
        )
        eval_cfg = {
            "smoke": bool(smoke),
            "k_pos": int(k_pos),
            "eval_window_L": EVAL_WINDOW_L,
        }
        if probe:
            eval_cfg["denoising_probe"] = True
        label = f"{arch}/{datasource}/T={T}/k={k_pos}/seed={seed}"
        try:
            res = run_experiment(
                experiment="synthetic",
                arch_name=arch,
                seed=int(seed),
                datasource_name=datasource,
                training_cfg=training_cfg,
                eval_cfg=eval_cfg,
                agent=agent,
                allow_dirty=allow_dirty,
            )
            status = "cached" if res.eval_cached else "ran"
            print(f"[{i}/{len(cells)}] {label}: {status}", flush=True)
            rows.append(json.loads(res.row.model_dump_json()))
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(cells)}] {label}: FAILED "
                  f"({type(e).__name__}: {e})", flush=True)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", choices=sorted(ARMS), default=None)
    p.add_argument("--datasource", choices=DATASOURCES, default=None)
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--k-pos", type=int, nargs="+", default=K_POS_GRID)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--n-steps", type=int, default=N_STEPS)
    p.add_argument("--d-sae", type=int, default=0,
                   help="override synthetic d_sae (sensitivity wing)")
    p.add_argument("--probe", action="store_true",
                   help="denoising latent-probe add-on (markov bench)")
    p.add_argument("--out", type=str, default=None,
                   help="write the shard's rows to this JSON path")
    p.add_argument("--smoke", action="store_true",
                   help="1 tiny cell per arm, 200 steps, smoke eval")
    p.add_argument("--allow-dirty", action="store_true")
    args = p.parse_args()

    if args.smoke:
        rows = []
        for arch in ARMS:
            rows += run_shard(
                arch, DATASOURCES[0], 4,
                k_pos_grid=(2,), seeds=(1,), n_steps=200,
                smoke=True, allow_dirty=True,
            )
        print(f"[smoke] {len(rows)} rows OK")
        return

    assert args.arch and args.datasource and args.T is not None, (
        "--arch/--datasource/--T required (or --smoke)"
    )
    rows = run_shard(
        args.arch, args.datasource, args.T,
        k_pos_grid=tuple(args.k_pos), seeds=tuple(args.seeds),
        n_steps=args.n_steps, d_sae=args.d_sae, probe=args.probe,
        allow_dirty=args.allow_dirty,
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows))
        print(f"[out] {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
