"""tscale L1 runner — train + dev-8 eval one arch over the L1 T-grid.

Usage (CARD_SPLIT § 3; L1 = 4k steps, s42, T {1,4,16}, k20 primary):

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m experiments.explorations.tscale.run_l1 \
        --arch txc_batchtopk_pre_btkonly --Ts 1 4 16 --tag baseline-4k

Appends one JSONL row per (arch, T) to results/l1_rows.jsonl. Window/
sequence archs get ``--T-key`` (default "T") in the override dict —
txc_pro_r1 uses T_max + a derived t_sample (ratio rule) via
--txcpro-ratio.
"""

from __future__ import annotations

import argparse
import time

from experiments.explorations.tscale.l1_lib import (
    append_row,
    build_cell_cfg,
    config_hash,
    dev_eval,
    git_sha,
    scratch_train,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--Ts", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-steps", type=int, default=4000)
    ap.add_argument("--k-feats", type=int, nargs="+", default=[20, 5])
    ap.add_argument("--seq-batch", type=int, default=1024)
    ap.add_argument("--tag", required=True, help="ledger tag for RESULTS.md")
    ap.add_argument("--T-key", default="T",
                    help="override key carrying T (txc_pro_r1: T_max)")
    ap.add_argument("--txcpro-ratio", action="store_true",
                    help="also derive t_sample = max(1, T//2) (CARD § 4 ratio rule)")
    ap.add_argument("--extra-hparams", default="",
                    help="comma k=v pairs merged into the override (ablations)")
    args = ap.parse_args()

    sha = git_sha()
    for T in args.Ts:
        override: dict = {args.T_key: int(T)}
        if args.txcpro_ratio:
            override["t_sample"] = max(1, int(T) // 2)
        if args.extra_hparams:
            for kv in args.extra_hparams.split(","):
                k, v = kv.split("=")
                try:
                    override[k] = int(v)
                except ValueError:
                    try:
                        override[k] = float(v)
                    except ValueError:
                        override[k] = v
        cfg = build_cell_cfg(
            arch_name=args.arch, override=override, seed=args.seed,
            n_steps=args.n_steps, seq_batch=args.seq_batch,
        )
        chash = config_hash(cfg)
        print(f"[l1] {args.tag} {args.arch} T={T} seed={args.seed} "
              f"steps={args.n_steps} cfg={chash}", flush=True)
        t0 = time.time()
        model, train_info = scratch_train(cfg)
        ev = dev_eval(model, k_feats=tuple(args.k_feats))
        wall = time.time() - t0
        row = {
            "level": "L1", "tag": args.tag, "cfg": cfg, "config_hash": chash,
            "git_sha": sha, "train_info": train_info, "eval": ev,
            "wall_s": round(wall, 1), "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        append_row(row)
        k0 = args.k_feats[0]
        e = ev[f"k{k0}"]
        print(f"[l1] DONE {args.arch} T={T}: dev k{k0} auc {e['dev_mean_auc']:.4f} "
              f"(shuf {e['dev_mean_auc_shuf']:.4f}) l0 {e['realized_l0']:.1f} "
              f"wall {wall/60:.1f} min", flush=True)
        del model
        import torch
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
