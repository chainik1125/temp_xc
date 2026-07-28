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
        # Realized budget per WINDOW — MEASURED, never assumed (the
        # "T x per-token" figure was an upper bound and was retracted).
        # ⚑ The correct budget differs by mode (hub review 0b1025abc):
        #   pooled  -> UNION over positions. The pooled vector has one slot
        #              per feature, so a feature firing at 3 positions really
        #              does collapse into one dimension.
        #   stacked -> SUM over positions. The stacked code has T*d_sae slots,
        #              so that same feature occupies THREE distinct input
        #              dimensions and must be counted three times.
        # Using the union for stacked understates its budget, which would plot
        # it further LEFT than it belongs — flattering the baseline. That is
        # conservative against TXC, so it could not manufacture a negative,
        # but it would contaminate a positive. Fixed before the sweep ran.
        nz = z.abs() > 0
        if self.mode == "pooled":
            self._l0.append(float(nz.any(dim=1).sum(-1).float().mean()))
        else:
            self._l0.append(float(nz.sum(dim=(1, 2)).float().mean()))
        if self.mode == "pooled":
            return z.mean(dim=1)
        return z.reshape(B, T * z.shape[-1])

    @property
    def realized_l0_per_window(self) -> float:
        return sum(self._l0) / max(1, len(self._l0))


class MeasuredArm(torch.nn.Module):
    """Wrap a native window arch (TXC) only to MEASURE its realized budget.

    Without this the frontier has no x-coordinate for TXC and cannot be
    plotted at all — the arms would share a y-axis and nothing else.
    `encode` is passed through untouched, so TXC's numbers are unchanged.

    Unit: nonzeros in the tile code (d_sae), which is the SAME convention
    as the pooled arm's union — one slot per feature. Stacked is the odd
    one out (T*d_sae slots ⇒ sum), and is labelled as such in the output.
    NB probing.py reports realized_l0 PER TOKEN; this file is PER WINDOW.
    Do not compare the two numbers across files.
    """

    def __init__(self, inner):
        super().__init__()
        self.inner, self.T, self._l0 = inner, inner.T, []

    def encode(self, tiles: torch.Tensor) -> torch.Tensor:
        z = self.inner.encode(tiles)
        self._l0.append(float((z.abs() > 0).sum(-1).float().mean()))
        return z

    @property
    def realized_l0_per_window(self) -> float:
        return sum(self._l0) / max(1, len(self._l0))


def _key_from_manifest(arch_name: str, T: int, seed: int) -> str | None:
    """Look the train_key UP; never re-derive it.

    Re-deriving needs the exact training_cfg the run used, and guessing it
    produced a key for a checkpoint that does not exist (twice today — the
    btk false alarm this morning, and my first cut of this function).
    `checkpoints/manifest.jsonl` records what was actually written.
    """
    import json
    mf = Path(__file__).resolve().parents[4] / "checkpoints" / "manifest.jsonl"
    if not mf.exists():
        return None
    for line in mf.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("arch") != arch_name or int(r.get("seed", -1)) != seed:
            continue
        if "sycgen" not in str(r.get("datasource", "")):
            continue
        tc = r.get("training_cfg") or {}
        if (tc.get("n_steps") or 0) <= 0:
            continue                      # untrained twin
        ov = tc.get("arch_hparams_override") or {}
        if int(ov.get("T", r.get("T", -1)) or -1) != T:
            continue
        return r.get("train_key")
    return None


def _load(arch_name: str, T: int, seed: int, ds_spec):
    from temp_bench.core.config import load_arch
    from temp_bench.core.runner import _load_checkpoint
    tk = _key_from_manifest(arch_name, T, seed)
    if tk is None:
        raise FileNotFoundError(
            f"no manifest train_key for {arch_name} T={T} seed={seed}")
    spec = load_arch(arch_name)
    spec = spec.model_copy(update={
        "hparams": {**spec.hparams, "d_sae": 2048, "T": T, "k_pos": 8}})
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
                txc_raw, tk = _load("txc_batchtopk_post_btkonly", T, seed, ds_spec)
                txc = MeasuredArm(txc_raw)
                m = _train_lambda_probe(txc, x, lam, L=EVAL_L,
                                        n_windows=N_WINDOWS, seed=seed)
                rows.append({"arm": "txc", "T": T, "seed": seed, "k_tok": None,
                             "recovery": m["lambda_recovery"],
                             "chance": m["lambda_chance"], "train_key": tk,
                             "realized_l0_per_window": txc.realized_l0_per_window,
                             "l0_unit": "nonzeros_in_tile_code"})
                print(f"  txc      T{T} s{seed} r={m['lambda_recovery']:.4f} "
                      f"l0/win={txc.realized_l0_per_window:.2f}", flush=True)
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
                                 "sae_train_key": sae_tk,
                                 "l0_unit": ("union_over_positions" if mode == "pooled"
                                             else "sum_over_positions")})
                    print(f"  {mode:8s} T{T} s{seed} k={k:<3} "
                          f"r={m['lambda_recovery']:.4f} "
                          f"l0/win={w.realized_l0_per_window:.2f}", flush=True)
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(rows, indent=1))
    print(f"[frontier] wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
