"""Stage-2 within-document RECEIPT — doc-demeaned re-fit of the SAME codes.

Pre-registered in `CARD_STAGE2.md` § 6b (frozen `b8f2f0bd`, before any
cell; rewritten at the A40 restart from the card's spec — the original
was lost unpushed, see APPENDIX A). The `probe_capacity.py` pattern:
re-instantiate the exact trained checkpoints via the reconstructed
`train_key`, and re-fit the λ probe out-of-band. **Off-leaderboard** —
nothing here writes a row; it can only change what the record is
allowed to claim about the within-document face.

Per trained cell (the frozen `_cells()` list, `kind == "trained"`):

1. **Licence row** — OLS at nw = 1024: byte-equivalent to the eval's
   `_train_lambda_probe`; MUST reproduce the leaderboard
   `lambda_recovery` to ~1e-6 (printed with the delta; reading any
   other row is licensed by this one).
2. **Ridge raw** — RidgeCV at nw = 8192 on the raw targets (the § 6b
   probe class; also the v2-adequacy analog on the same sampler).
3. **Ridge doc-demeaned** — same codes, targets replaced by
   ``t − doc_mean(doc)`` where the mean is the doc's finite lam_q mean
   over the WHOLE stream (the § 6a floor's definition — label-side, no
   model in the loop).

   **AMENDMENT (disclosed, first run killed after one cell):** the card
   § 6b wrote "the doc's train-side finite mean", which is IMPOSSIBLE
   under this datasource's row split: trace_ids are doc-contiguous, so
   the row-half split makes probe-train and probe-eval docs DISJOINT —
   no eval-pool doc has a train-side mean (first run: 261,376/262,144
   tile rows hit the global fallback, i.e. eval targets received only a
   constant shift, which tests nothing within-doc). The § 6a
   whole-stream doc mean is the well-defined reading: it is a frozen
   function of the label stream (no leakage — the probe never sees doc
   means, targets are simply re-expressed as within-doc deviations),
   defined for every doc, and it directly instruments the § 6 question
   "does the code predict within-doc fluctuations?". The one value seen
   before the kill (tsae/s1 demeaned r = +0.0515 under the degenerate
   train-side rule) is superseded by this run and quoted nowhere. The
   pre-registered outcome rule (K4 = window−token gap sign under
   demeaning; collapse = sound NEGATIVE) is unchanged.

Outcome rule (card § 6b, pre-registered): the within-doc face may sit
near floor (zero-frac 0.817). If the window−token gap collapses under
demeaning, that is a sound NEGATIVE for the within-document claim,
reported as loudly as a win (verdict V4 split). Scored in the verdict.

Run: .venv/bin/python -m \
       experiments.explorations.task_hunt.qrate_fineweb.stage2_demeaned [ds]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from temp_bench.core.config import (
    checkpoint_dir,
    compute_data_key,
    compute_train_key,
    import_by_path,
    load_arch,
    load_datasource,
)
from temp_bench.core.schemas import TrainingConfig
from temp_bench.data.synthetic import materialise
from temp_bench.evals.lambda_recovery import _tile_lambda_examples
from temp_bench.evals.synthetic_recovery import _sample_windows

from experiments.explorations.task_hunt.qrate_fineweb.run_stage2 import (
    BUFFER, D_SAE, DS_DEFAULT, EVAL_L, N_STEPS, PANEL, WINDOW_TS, _cells,
)
from explorations.synthetic.grid import batch_size

HERE = Path(__file__).resolve().parent


def _load_model(arch: str, T: int, k_pos: int, seed: int, ds: str):
    """Re-instantiate one trained panel cell (probe_capacity pattern)."""
    arch_spec = load_arch(arch, section="synthetic")
    data_spec = load_datasource(ds)
    override = {"k_pos": k_pos, "d_sae": D_SAE[ds], "T": T}
    arch_spec = arch_spec.model_copy(
        update={"hparams": {**arch_spec.hparams, **override}})
    tcfg = TrainingConfig(
        n_steps=N_STEPS, batch_size=batch_size(T),
        buffer_tokens=BUFFER[ds], arch_hparams_override=override)
    train_key = compute_train_key(
        arch=arch_spec, seed=seed, training_cfg=tcfg,
        data_key=compute_data_key(data_spec), section="synthetic")
    path = checkpoint_dir(train_key) / "model.safetensors"
    if not path.exists():
        return None, train_key, data_spec
    from safetensors.torch import load_file

    from temp_bench.core.trainer import _infer_d_in
    cls = import_by_path(arch_spec.class_path)
    model = cls(d_in=_infer_d_in(data_spec), **arch_spec.hparams)
    model.load_state_dict(load_file(str(path)))
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    return model, train_key, data_spec


def _leaderboard_v1(ds: str):
    """(arch, T, k_pos, seed) → v1 lambda_recovery for trained rows of ds."""
    from temp_bench.core.cache import leaderboard_path
    out = {}
    with open(leaderboard_path()) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("datasource") != ds:
                continue
            tc = r.get("training_cfg", {})
            if tc.get("n_steps") != N_STEPS:
                continue
            ov = tc.get("arch_hparams_override", {})
            key = (r["arch"], ov.get("T"), ov.get("k_pos"), r["seed"])
            m = r.get("metrics", {})
            if "lambda_recovery" in m:
                out[key] = float(m["lambda_recovery"])
    return out


def _fit(model, x, lam, doc_of_row, split, *, n_windows, est, demean,
         seed=0):
    """The eval's probe with estimator / n_windows / demeaning exposed.

    ``est='ols', n_windows=1024, demean=False`` reproduces
    `lambda_recovery._train_lambda_probe` (incl. the non-finite guard).
    """
    from sklearn.linear_model import LinearRegression, RidgeCV

    from temp_bench.evals.synthetic_recovery import _check_tileable

    T = _check_tileable(model, EVAL_L)
    lam3 = lam.reshape(lam.shape[0], lam.shape[1], 1)
    n_tiles = EVAL_L // T

    win_x_tr, tr_idx = _sample_windows(x[:split], L=EVAL_L,
                                       n_windows=n_windows, seed=seed)
    win_l_tr, _ = _sample_windows(lam3[:split], L=EVAL_L,
                                  n_windows=n_windows, seed=seed)
    win_x_ev, ev_idx = _sample_windows(x[split:], L=EVAL_L,
                                       n_windows=n_windows, seed=seed + 1)
    win_l_ev, _ = _sample_windows(lam3[split:], L=EVAL_L,
                                  n_windows=n_windows, seed=seed + 1)

    z_tr, t_tr = _tile_lambda_examples(model, win_x_tr, win_l_tr, T)
    z_ev, t_ev = _tile_lambda_examples(model, win_x_ev, win_l_ev, T)
    d_tr = np.repeat(doc_of_row[tr_idx], n_tiles)
    d_ev = np.repeat(doc_of_row[split + ev_idx], n_tiles)

    n_fallback = 0
    if demean:
        # Whole-stream finite doc mean (§ 6a definition; see AMENDMENT in
        # the module docstring — "train-side" is undefined under the
        # doc-disjoint split). Label-side only; defined for every doc.
        lam_np = lam.numpy()
        fin = np.isfinite(lam_np)
        means = {}
        for d in np.unique(doc_of_row):
            m = (doc_of_row == d)[:, None] & fin
            means[int(d)] = float(lam_np[m].mean()) if m.any() else np.nan
        g = float(lam_np[fin].mean())
        mu_tr = np.array([means[int(d)] for d in d_tr])
        mu_ev = np.array([means[int(d)] for d in d_ev])
        n_fallback = int(np.isnan(mu_ev).sum() + np.isnan(mu_tr).sum())
        mu_tr = np.where(np.isnan(mu_tr), g, mu_tr)
        mu_ev = np.where(np.isnan(mu_ev), g, mu_ev)
        t_tr = t_tr - mu_tr
        t_ev = t_ev - mu_ev

    tr_m, ev_m = np.isfinite(t_tr), np.isfinite(t_ev)
    z_tr, t_tr = z_tr[tr_m], t_tr[tr_m]
    z_ev, t_ev = z_ev[ev_m], t_ev[ev_m]
    if len(t_tr) < 2 or np.std(t_tr) < 1e-9 or np.std(t_ev) < 1e-9:
        return {"r": 0.0, "n_rows": int(len(t_tr)), "degenerate": True}

    reg = (LinearRegression() if est == "ols"
           else RidgeCV(alphas=np.logspace(-2, 4, 13)))
    reg.fit(z_tr, t_tr)
    pred = reg.predict(z_ev)
    corr = float(np.corrcoef(pred, t_ev)[0, 1]) if np.std(pred) > 1e-12 \
        else 0.0
    return {"r": corr, "r2_heldout": float(reg.score(z_ev, t_ev)),
            "n_rows": int(z_tr.shape[0]), "p": int(z_tr.shape[1]),
            "n_eval_rows": int(z_ev.shape[0]),
            "alpha": float(getattr(reg, "alpha_", float("nan"))),
            "n_doc_fallback": int(n_fallback)}


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else DS_DEFAULT
    cells = [c for c in _cells(ds, PANEL, WINDOW_TS)
             if c["kind"] == "trained"]
    lb = _leaderboard_v1(ds)
    data_cache: dict[int, object] = {}
    out, max_delta = [], 0.0
    for c in cells:
        arch, T, k_pos, seed = c["arch"], c["T"], c["k_pos"], c["seed"]
        model, train_key, _spec = _load_model(arch, T, k_pos, seed, ds)
        if model is None:
            print(f"[skip] {arch}/T{T}/k{k_pos}/s{seed}: no checkpoint",
                  flush=True)
            continue
        if seed not in data_cache:
            data_cache[seed] = materialise(load_datasource(ds), seed=seed)
        data = data_cache[seed]
        lam = torch.as_tensor(data.extra["lambda_labels"]).float()
        doc = np.asarray(data.extra["trace_ids"])
        split = data.x.shape[0] // 2

        lic = _fit(model, data.x, lam, doc, split,
                   n_windows=1024, est="ols", demean=False)
        ref = lb.get((arch, T, k_pos, seed))
        delta = abs(lic["r"] - ref) if ref is not None else float("nan")
        max_delta = max(max_delta, delta if np.isfinite(delta) else 0.0)
        raw = _fit(model, data.x, lam, doc, split,
                   n_windows=8192, est="ridge", demean=False)
        dem = _fit(model, data.x, lam, doc, split,
                   n_windows=8192, est="ridge", demean=True)
        row = {"arch": arch, "T": T, "k_pos": k_pos, "seed": seed,
               "train_key": train_key,
               "licence_ols_nw1024_r": lic["r"], "leaderboard_v1": ref,
               "licence_delta": None if ref is None else delta,
               "ridge_raw": raw, "ridge_demeaned": dem}
        out.append(row)
        print(f"{arch:<20} T{T:<3} k{k_pos:<4} s{seed:<3} "
              f"lic={lic['r']:+.4f} (lb={'—' if ref is None else f'{ref:+.4f}'}"
              f" Δ={'—' if ref is None else f'{delta:.1e}'})  "
              f"raw={raw['r']:+.4f}  demeaned={dem['r']:+.4f} "
              f"(fallback={dem.get('n_doc_fallback', 0)})", flush=True)
        del model
        torch.cuda.empty_cache()
    res_dir = HERE / "results"
    res_dir.mkdir(exist_ok=True)
    path = res_dir / f"stage2_demeaned_{ds}.json"
    path.write_text(json.dumps(
        {"ds": ds, "max_licence_delta": max_delta, "rows": out}, indent=2))
    print(f"-> {path}  ({len(out)} cells, max licence Δ={max_delta:.2e})",
          flush=True)


if __name__ == "__main__":
    main()
