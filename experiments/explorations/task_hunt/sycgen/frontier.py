"""ITEM 6 — recovery-vs-budget FRONTIER: TXC vs pooled-SAE vs stacked-SAE.

Answers the challenge that our sycgen claim compared a *windowed* model
against a *per-token* SAE, which establishes nothing about architecture.

Hub ruling 14de8b5a0: sweep k on both arms and plot the frontier — NOT a
single budget-matched point. My 21:14 arithmetic showed why: matching a
pooled SAE to TXC's per-window l0 forces 0.49 l0/token at T16 (most
tokens get no feature at all), and because the arms' budgets scale
differently in T, matching per-window necessarily unmatches per-token.
There is no single matched point to specify.

The new arms need NO training: pooled/stacked are post-hoc transforms of
the ALREADY-TRAINED per-token (T=1) SAE.

    pooled  : encode each of the T tokens -> mean over the window -> d_sae
    stacked : encode each of the T tokens -> concatenate          -> T*d_sae

⚑ Feature-dimension asymmetry, disclosed rather than hidden: the
evaluator's tile code for TXC is `d_sae`. Pooled matches that exactly.
**Stacked gets T*d_sae — T times the probe input** — so a stacked win is
partly a probe-capacity win, not purely an architecture one. Reported
alongside, never netted out.

Scoring reuses `lambda_recovery._train_lambda_probe` verbatim, so every
arm is scored by the SAME instrument as TXC. Anything else would not be
a comparison — which is the whole reason the original claim failed.

Runs where the activation cache lives (the pod).

    .venv/bin/python -m experiments.explorations.task_hunt.sycgen.frontier
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "frontier.json"

TS = (2, 4, 8, 16)
SEEDS = (42, 1, 2)
K_SWEEP = (1, 2, 4, 8, 16, 32)      # per-token budget for the SAE arms
EVAL_L = 32
N_WINDOWS = 1024


class WindowWrapper(torch.nn.Module):
    """Present a per-token SAE as a window encoder the evaluator can score.

    `encode(tiles)` receives `(B, T, d_in)` and returns the tile code, which
    is exactly the contract TXC satisfies — so the evaluator cannot tell the
    arms apart and scores them identically.
    """

    def __init__(self, sae, T: int, mode: str, k_tok: int | None):
        super().__init__()
        self.sae, self.T, self.mode, self.k_tok = sae, T, mode, k_tok
        self._l0 = []

    def encode(self, tiles: torch.Tensor) -> torch.Tensor:
        B, T, d_in = tiles.shape
        z = self.sae.encode(tiles.reshape(B * T, 1, d_in)).reshape(B, T, -1)
        if self.k_tok is not None and self.k_tok < z.shape[-1]:
            kth = z.abs().topk(self.k_tok, dim=-1).values[..., -1:]
            z = z * (z.abs() >= kth)
        # realized budget = distinct active features per WINDOW. Measured as
        # the union across positions, never T x per-token: pooling collapses
        # a feature firing at several positions, which is exactly the
        # assumption the hub retracted at 19:4x.
        self._l0.append(float((z.abs() > 0).any(dim=1).sum(-1).float().mean()))
        if self.mode == "pooled":
            return z.mean(dim=1)
        return z.reshape(B, T * z.shape[-1])

    @property
    def realized_l0_per_window(self) -> float:
        return sum(self._l0) / max(1, len(self._l0))


def _load(arch_name: str, T: int, seed: int, ds_spec):
    from temp_bench.core.config import compute_data_key, compute_train_key, load_arch
    from temp_bench.core.runner import _load_checkpoint
    spec = load_arch(arch_name)
    ov = {"d_sae": 2048, "T": T, "k_pos": 8}
    spec = spec.model_copy(update={"hparams": {**spec.hparams, **ov}})
    tk = compute_train_key(
        arch=spec, seed=seed,
        training_cfg={"n_steps": 8000, "buffer_tokens": 524288,
                      "arch_hparams_override": ov},
        data_key=compute_data_key(ds_spec))
    return _load_checkpoint(spec, tk, ds_spec), tk


def main():
    from temp_bench.core.config import load_datasource
    from temp_bench.evals.lambda_recovery import _train_lambda_probe
    from temp_bench.data.synthetic import materialise
    import experiments.explorations.task_hunt.sycgen.run_retrain as RR

    ds_spec = load_datasource(RR.DS)
    data = materialise(ds_spec, seed=0)
    lam = data.extra["lambda_labels"]
    if not torch.is_tensor(lam):
        lam = torch.as_tensor(lam)
    x, lam = data.x, lam.float()
    print(f"[frontier] x={tuple(x.shape)} lam={tuple(lam.shape)}", flush=True)

    rows = []
    for T in TS:
        for seed in SEEDS:
            # --- TXC (the claim arm), scored by the same probe ---
            try:
                txc, tk = _load("txc_batchtopk_post_btkonly", T, seed, ds_spec)
                m = _train_lambda_probe(txc, x, lam, L=EVAL_L,
                                        n_windows=N_WINDOWS, seed=seed)
                rows.append({"arm": "txc", "T": T, "seed": seed, "k_tok": None,
                             "recovery": m["lambda_recovery"],
                             "chance": m["lambda_chance"], "train_key": tk})
                print(f"  txc      T{T} s{seed} r={m['lambda_recovery']:.4f}", flush=True)
            except Exception as e:
                print(f"  txc      T{T} s{seed} SKIP {type(e).__name__}: {str(e)[:90]}", flush=True)

            # --- pooled / stacked SAE, swept over per-token budget ---
            try:
                sae, sae_tk = _load("batchtopk_sae_btkonly", 1, seed, ds_spec)
            except Exception as e:
                print(f"  sae      s{seed} LOAD FAIL {type(e).__name__}", flush=True)
                continue
            for mode in ("pooled", "stacked"):
                for k in K_SWEEP:
                    w = WindowWrapper(sae, T, mode, k)
                    m = _train_lambda_probe(w, x, lam, L=EVAL_L,
                                            n_windows=N_WINDOWS, seed=seed)
                    rows.append({"arm": mode, "T": T, "seed": seed, "k_tok": k,
                                 "recovery": m["lambda_recovery"],
                                 "chance": m["lambda_chance"],
                                 "realized_l0_per_window": w.realized_l0_per_window,
                                 "sae_train_key": sae_tk})
                    print(f"  {mode:8s} T{T} s{seed} k={k:<3} "
                          f"r={m['lambda_recovery']:.4f} "
                          f"l0/win={w.realized_l0_per_window:.2f}", flush=True)
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(rows, indent=1))
    print(f"[frontier] wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
