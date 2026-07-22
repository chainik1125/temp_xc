"""FreqFrac report — per-arch temporal frequency responses at the canonical cells.

The train+analyze path of the FreqBench port (PORT.md § C–D). For each
(bench, arch) it selects the **canonical per-token-matched cell** exactly as
the REPORT.md matrix does — window ``T = T_can`` (token archs ``T = 1``),
``d_sae = F``, the ``k_pos`` whose *realized* ``l0_per_token`` sits nearest
``B*`` — from the existing leaderboard rows, then:

1. rebuilds the cell's ``train_key`` from the row's own ``training_cfg``
   (hard-asserted equal to the row's recorded key — if this ever fires, the
   reconstruction has diverged from the canonical pathway and must be fixed);
2. loads the checkpoint if present, else **trains it** via
   ``trainer.train_arch`` (same trainer, same key ⇒ the checkpoint store is
   repopulated for any future grid run). **Never writes the leaderboard** —
   FreqFrac is a weight-space diagnostic, not an eval result;
3. computes the FreqFrac profile, firing-weighted arch curve, spectral
   concentration, and DC fraction — for the trained model AND the
   untrained same-arch null (mandatory: random-init concentration is the
   baseline any "learned spectral structure" claim must clear);
4. writes ``results/freqfrac_stats.json`` (code-version-stamped, with the
   analyzed ``train_key`` per cell) + ``figs/freqfrac_curves.png``.

Run from the repo root::

    .venv/bin/python -m experiments.explorations.synthetic.freqbench.freqfrac_report \
        frequency backtracking [--seed 1] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TEMP_BENCH_ALLOW_DIRTY", "1")

import numpy as np
import torch

from experiments.explorations.synthetic.registry import ARCHS, BENCHES, OP
from explorations.synthetic import freqfrac

HERE = Path(__file__).resolve().parent


# ── canonical-cell selection (mirrors the REPORT.md matched-group rule) ──────

def _load_rows(leaderboard: Path, datasource: str, protocol: str) -> list[dict]:
    rows = []
    with open(leaderboard) as f:
        for line in f:
            r = json.loads(line)
            if (r.get("datasource") == datasource
                    and r.get("evaluator_protocol_version") == protocol
                    and r.get("experiment") == "synthetic"):
                rows.append(r)
    return rows


def _cell_T(row: dict) -> int:
    return int(row["training_cfg"]["arch_hparams_override"].get("T", 1))


def _cell_d_sae(row: dict) -> int:
    return int(row["training_cfg"]["arch_hparams_override"]["d_sae"])


def select_canonical_row(rows: list[dict], arch: str, *, windowed: bool,
                         F: int, seed: int) -> dict | None:
    """The per-token-matched cell: T=T_can (or 1), d_sae=F, trained (n_steps>0),
    k_pos with realized l0_per_token nearest B*."""
    T_target = OP.T_can if windowed else 1
    cand = [r for r in rows
            if r["arch"] == arch and r["seed"] == seed
            and _cell_T(r) == T_target and _cell_d_sae(r) == F
            and int(r["training_cfg"]["n_steps"]) > 0
            and "l0_per_token" in r["metrics"]]
    if not cand:
        return None
    return min(cand, key=lambda r: abs(r["metrics"]["l0_per_token"] - OP.B_star))


# ── train-or-load (the runner's own mechanics, minus the leaderboard) ────────

def obtain_model(row: dict, *, train_if_missing: bool = True):
    """Rebuild the row's cell; load its checkpoint or train it. Returns
    (model, train_key, trained_now)."""
    from temp_bench.core.cache import checkpoint_exists
    from temp_bench.core.code_version import capture as capture_code_version
    from temp_bench.core.config import (
        compute_data_key,
        compute_train_key,
        load_arch,
        load_datasource,
    )
    from temp_bench.core.runner import _load_checkpoint
    from temp_bench.core.schemas import TrainingConfig

    training_cfg = TrainingConfig(**row["training_cfg"])
    arch_spec = load_arch(row["arch"], section="synthetic")
    if training_cfg.arch_hparams_override:
        merged = {**arch_spec.hparams, **training_cfg.arch_hparams_override}
        arch_spec = arch_spec.model_copy(update={"hparams": merged})
    data_spec = load_datasource(row["datasource"])
    data_key = compute_data_key(data_spec)
    train_key = compute_train_key(
        arch=arch_spec, seed=int(row["seed"]), training_cfg=training_cfg,
        data_key=data_key, section="synthetic",
    )
    if train_key != row["train_key"]:
        raise AssertionError(
            f"reconstructed train_key {train_key} != row's {row['train_key']} "
            f"for {row['arch']} — the canonical reconstruction has diverged; "
            "do NOT proceed (fix freqfrac_report, never the row)."
        )
    if checkpoint_exists(train_key):
        return _load_checkpoint(arch_spec, train_key, data_spec), train_key, False
    if not train_if_missing:
        return None, train_key, False
    from temp_bench.core.trainer import train_arch
    model = train_arch(
        arch_spec=arch_spec, data_spec=data_spec, seed=int(row["seed"]),
        training_cfg=training_cfg, train_key=train_key,
        code_version=capture_code_version(allow_dirty=True),
        agent="freqfrac-report",
    )
    model.eval()
    return model, train_key, True


def untrained_twin(row: dict):
    """Same-arch random-init null (torch-seeded on the row's seed)."""
    from temp_bench.core.config import import_by_path, load_arch, load_datasource
    from temp_bench.core.schemas import TrainingConfig
    from temp_bench.core.trainer import _infer_d_in

    training_cfg = TrainingConfig(**row["training_cfg"])
    arch_spec = load_arch(row["arch"], section="synthetic")
    merged = {**arch_spec.hparams, **(training_cfg.arch_hparams_override or {})}
    cls = import_by_path(arch_spec.class_path)
    torch.manual_seed(int(row["seed"]))
    model = cls(d_in=_infer_d_in(load_datasource(row["datasource"])), **merged)
    model.eval()
    return model


# ── windows for firing weights ───────────────────────────────────────────────

def eval_windows(datasource: str, seed: int, T: int, n_seqs: int = 256):
    """(N, T, d_in) non-overlapping T-tiles from freshly materialized sequences
    (same generator + seed convention as the trainer's refill source)."""
    from temp_bench.core.config import load_datasource
    from temp_bench.core.trainer import _build_refill_source
    refill = _build_refill_source(load_datasource(datasource), seed=seed)
    x = refill(n_seqs)                                   # (n, L, d_in)
    n, L, d = x.shape
    k = L // T
    return x[:, : k * T].reshape(n * k, T, d).float()


# ── per-cell analysis ────────────────────────────────────────────────────────

def analyze(model, x: torch.Tensor | None) -> dict:
    prof = freqfrac.freq_profile(model)
    conc = freqfrac.spectral_concentration(prof)
    if x is not None and prof.shape[1] > 1:
        mean_act, rate = freqfrac.firing_weights(model, x)
        w = mean_act
    else:
        w = None
        rate = None
    curve = freqfrac.arch_curve(prof, w)
    curve_uniform = freqfrac.arch_curve(prof, None)
    wn = (w / w.sum()) if (w is not None and float(w.sum()) > 0) else None
    conc_w = float((conc * wn).sum()) if wn is not None else float(conc.mean())
    dc_w = float((prof[:, 0] * wn).sum()) if wn is not None else float(prof[:, 0].mean())
    return {
        "curve": [round(float(v), 5) for v in curve],
        "curve_uniform": [round(float(v), 5) for v in curve_uniform],
        "dc_frac": round(dc_w, 4),
        "concentration": round(conc_w, 4),
        "concentration_pop_mean": round(float(conc.mean()), 4),
        "alive_frac": (round(float((rate > 0).float().mean()), 4)
                       if rate is not None else None),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("benches", nargs="*", default=["frequency", "backtracking"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-seqs", type=int, default=256)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the selected canonical rows and exit")
    args = ap.parse_args()
    benches = args.benches or ["frequency", "backtracking"]

    from temp_bench.core.code_version import capture as capture_code_version
    from temp_bench.core.config import repo_root

    leaderboard = repo_root() / "results" / "leaderboard.jsonl"
    by_name = {b.name: b for b in BENCHES}
    out: dict = {"op": {"T_can": OP.T_can, "B_star": OP.B_star, "d_sae": "F"},
                 "seed": args.seed, "cells": []}

    for bname in benches:
        bench = by_name[bname]
        ds = bench.datasources[0]
        rows = _load_rows(leaderboard, ds, bench.protocol)
        for arch in ARCHS:
            row = select_canonical_row(rows, arch.name, windowed=arch.windowed,
                                       F=bench.F, seed=args.seed)
            if row is None:
                print(f"[{bname}/{arch.name}] no canonical row — SKIP", flush=True)
                continue
            k_pos = row["training_cfg"]["arch_hparams_override"]["k_pos"]
            l0 = row["metrics"]["l0_per_token"]
            print(f"[{bname}/{arch.name}] T={_cell_T(row)} d_sae={_cell_d_sae(row)} "
                  f"k_pos={k_pos} realized_l0={l0:.2f} train_key={row['train_key'][:12]}",
                  flush=True)
            if args.dry_run:
                continue
            model, train_key, trained_now = obtain_model(row)
            print(f"  model {'TRAINED now' if trained_now else 'loaded'}", flush=True)
            T = _cell_T(row)
            x = (eval_windows(ds, args.seed, T, args.n_seqs)
                 if T > 1 else None)
            cell = {
                "bench": bname, "arch": arch.name, "datasource": ds,
                "T": T, "d_sae": _cell_d_sae(row), "k_pos": k_pos,
                "realized_l0_per_token": round(float(l0), 3),
                "train_key": train_key, "eval_key_ref": row["eval_key"],
                "trained_now": trained_now,
                "trained": analyze(model, x),
                "untrained": analyze(untrained_twin(row), x),
            }
            out["cells"].append(cell)
            tr, un = cell["trained"], cell["untrained"]
            print(f"  dc_frac {tr['dc_frac']:.3f} (init {un['dc_frac']:.3f})  "
                  f"conc {tr['concentration']:.3f} (init {un['concentration']:.3f})",
                  flush=True)

    if args.dry_run:
        return

    out["code_version"] = capture_code_version(allow_dirty=True).model_dump()
    res_dir = HERE / "results"
    res_dir.mkdir(exist_ok=True)
    stats_path = res_dir / "freqfrac_stats.json"
    stats_path.write_text(json.dumps(out, indent=1))
    print(f"wrote {stats_path}", flush=True)

    _render_fig(out, benches)


def _render_fig(out: dict, benches: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    benches = [b for b in benches
               if any(c["bench"] == b for c in out["cells"])]
    if not benches:
        return
    fig, axes = plt.subplots(1, len(benches), figsize=(5.2 * len(benches), 3.6),
                             squeeze=False)
    for ax, bname in zip(axes[0], benches):
        cells = [c for c in out["cells"] if c["bench"] == bname and c["T"] > 1]
        for c in cells:
            w = np.arange(len(c["trained"]["curve"]))
            (ln,) = ax.plot(w, c["trained"]["curve"], marker="o", ms=3,
                            label=c["arch"])
            ax.plot(w, c["untrained"]["curve"], ls=":", lw=1,
                    color=ln.get_color(), alpha=0.6)
        ax.set_title(f"{bname} — FreqFrac (T={OP.T_can}, d_sae=F, matched B*)\n"
                     "solid = trained · dotted = untrained init", fontsize=9)
        ax.set_xlabel("DCT frequency index w")
        ax.set_ylabel("firing-weighted energy fraction")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
    fig.tight_layout()
    figs = HERE / "figs"
    figs.mkdir(exist_ok=True)
    fig.savefig(figs / "freqfrac_curves.png", dpi=160)
    print(f"wrote {figs / 'freqfrac_curves.png'}", flush=True)


if __name__ == "__main__":
    main()
