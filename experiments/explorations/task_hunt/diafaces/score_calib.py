"""diafaces/score_calib.py — ACTMIX Stage 2 frozen scorer (CALIB_CARD.md).

relu-mix arm: the 20 rows cited by eval_key in CALIB_CARD § 3, read from
the CANONICAL leaderboard (hard-fail if any key is missing). btk-only
arm: the calib panel JSON (rows pin-asserted at merge time by
merge_calib_payload.py). Computes the CALIB_CARD § 5/§ 6 outputs:
per-cell table, paired per-seed Δ, post T-slopes per arm, realized-l0
band flags, E1–E4 direction checks. DESCRIPTIVE — no bars, no verdict
beyond the pre-registered direction checks; PENDING TEAM REVIEW.

Run: .venv/bin/python -m experiments.explorations.task_hunt.diafaces.score_calib
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DS = "dial_real_ttrend_gpt2_l7"
SEEDS = (3, 4)
POST_TS = (4, 16, 32)
L0_BAND = (6.5, 9.6)              # CALIB_CARD § 4, btk-only cells
BASE_OF = {"batchtopk_sae_btkonly": "batchtopk_sae",
           "tsae_btkonly": "tsae",
           "txc_batchtopk_post_btkonly": "txc_batchtopk_post"}

# CALIB_CARD § 3 — the reused relu-mix rows, frozen by eval_key.
REUSED = {
    ("batchtopk_sae", 1, 3, "trained"):   "3e7472feb278e922",
    ("batchtopk_sae", 1, 3, "untrained"): "c8af7733b91e88f1",
    ("batchtopk_sae", 1, 4, "trained"):   "ea801af31aa09eb9",
    ("batchtopk_sae", 1, 4, "untrained"): "c799183552489fa5",
    ("tsae", 1, 3, "trained"):            "c6441f5d9a65180d",
    ("tsae", 1, 3, "untrained"):          "bfe04bdb3695d6f5",
    ("tsae", 1, 4, "trained"):            "d02c894a8c76a7e5",
    ("tsae", 1, 4, "untrained"):          "4cda4f0078fed728",
    ("txc_batchtopk_post", 4, 3, "trained"):    "f05faa4f38cd9966",
    ("txc_batchtopk_post", 4, 3, "untrained"):  "063d8160ff0cef41",
    ("txc_batchtopk_post", 4, 4, "trained"):    "2100877acb00c139",
    ("txc_batchtopk_post", 4, 4, "untrained"):  "f07cf092c7506ed4",
    ("txc_batchtopk_post", 16, 3, "trained"):   "f8ef0d74a9056bee",
    ("txc_batchtopk_post", 16, 3, "untrained"): "4a8706f47a85025f",
    ("txc_batchtopk_post", 16, 4, "trained"):   "2f0a19c6b6701d81",
    ("txc_batchtopk_post", 16, 4, "untrained"): "b45636613df76f37",
    ("txc_batchtopk_post", 32, 3, "trained"):   "a79ee7cbf6c36012",
    ("txc_batchtopk_post", 32, 3, "untrained"): "c63bed8ea1226f5d",
    ("txc_batchtopk_post", 32, 4, "trained"):   "e03386fbd4efdf15",
    ("txc_batchtopk_post", 32, 4, "untrained"): "061c4465a13e2181",
}
GRID_KEYS = sorted(REUSED)        # the 20 logical cells, one arm each side


def _l0(arch, metrics):
    """Realized l0 per selection row (post: window; token archs: token)."""
    key = "l0_per_window" if arch.startswith("txc_") else "l0_per_token"
    return float(metrics[key])


def _load_relu():
    by_ek = {}
    with (ROOT / "results" / "leaderboard.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("eval_key") in set(REUSED.values()):
                by_ek[r["eval_key"]] = r
    out = {}
    for cell_key, ek in REUSED.items():
        assert ek in by_ek, f"cited relu-mix row missing from leaderboard: {ek}"
        r = by_ek[ek]
        arch, T, seed, kind = cell_key
        hp = r["training_cfg"]["arch_hparams_override"]
        assert (r["arch"], hp["T"], r["seed"]) == (arch, T, seed), cell_key
        assert (r["training_cfg"]["n_steps"] > 0) == (kind == "trained")
        assert "lambda_recovery_v2" in r["metrics"], "cited row lost v2 pairing"
        out[cell_key] = {
            "recovery": float(r["metrics"]["lambda_recovery"]),
            "recovery_v2": float(r["metrics"]["lambda_recovery_v2"]),
            "l0": _l0(arch, r["metrics"]),
            "eval_key": ek,
        }
    return out


def _load_btk():
    panel = json.loads(
        (HERE / "results" / f"calib_stage2_{DS}.json").read_text())
    out = {}
    for c in panel:
        if not c.get("ok"):
            continue
        base = BASE_OF[c["arch"]]
        key = (base, c["T"], c["seed"], c["kind"])
        m = c["metrics"]
        out[key] = {
            "recovery": float(m["lambda_recovery"]),
            "recovery_v2": float(m["lambda_recovery_v2"]),
            "l0": _l0(c["arch"], m),
            "arch": c["arch"],
        }
    missing = [k for k in GRID_KEYS if k not in out]
    assert not missing, f"btk-only cells missing from panel: {missing}"
    return out


def _slope(points):
    """OLS slope of recovery on log2 T over the post ladder (3 points)."""
    x = np.log2([t for t, _ in points])
    y = np.array([v for _, v in points], float)
    return float(np.cov(x, y, bias=True)[0, 1] / np.var(x))


def main():
    relu, btk = _load_relu(), _load_btk()

    cells, l0_flags = [], []
    for key in GRID_KEYS:
        arch, T, seed, kind = key
        rm, bo = relu[key], btk[key]
        in_band = L0_BAND[0] <= bo["l0"] <= L0_BAND[1]
        if not in_band:
            l0_flags.append({"cell": list(key), "l0": round(bo["l0"], 3),
                             "band": list(L0_BAND)})
        cells.append({
            "arch": arch, "T": T, "seed": seed, "kind": kind,
            "relu_mix": {"recovery": round(rm["recovery"], 4),
                         "recovery_v2": round(rm["recovery_v2"], 4),
                         "l0": round(rm["l0"], 3),
                         "eval_key": rm["eval_key"]},
            "btk_only": {"recovery": round(bo["recovery"], 4),
                         "recovery_v2": round(bo["recovery_v2"], 4),
                         "l0": round(bo["l0"], 3), "l0_in_band": in_band},
            "delta": round(bo["recovery"] - rm["recovery"], 4),
            "delta_v2": round(bo["recovery_v2"] - rm["recovery_v2"], 4),
        })

    def mean_delta(arch, T, kind):
        ds = [btk[(arch, T, s, kind)]["recovery"]
              - relu[(arch, T, s, kind)]["recovery"] for s in SEEDS]
        return float(np.mean(ds)), [round(d, 4) for d in ds]

    d_tr = {
        "batchtopk_sae@1": mean_delta("batchtopk_sae", 1, "trained"),
        "tsae@1": mean_delta("tsae", 1, "trained"),
        **{f"txc_batchtopk_post@{t}":
           mean_delta("txc_batchtopk_post", t, "trained") for t in POST_TS},
    }
    d_un = {
        "batchtopk_sae@1": mean_delta("batchtopk_sae", 1, "untrained"),
        "tsae@1": mean_delta("tsae", 1, "untrained"),
        **{f"txc_batchtopk_post@{t}":
           mean_delta("txc_batchtopk_post", t, "untrained") for t in POST_TS},
    }

    slopes = {}
    for arm, src in (("relu_mix", relu), ("btk_only", btk)):
        per_seed = [
            _slope([(t, src[("txc_batchtopk_post", t, s, "trained")]["recovery"])
                    for t in POST_TS])
            for s in SEEDS
        ]
        slopes[arm] = {"per_seed": [round(v, 4) for v in per_seed],
                       "mean": round(float(np.mean(per_seed)), 4)}
    dslope = round(slopes["btk_only"]["mean"] - slopes["relu_mix"]["mean"], 4)

    md = {k: v[0] for k, v in d_tr.items()}
    e1 = md["batchtopk_sae@1"] == max(md.values())
    e2 = abs(md["tsae@1"]) == min(abs(v) for v in md.values())
    e3 = md["txc_batchtopk_post@4"] > md["txc_batchtopk_post@32"]
    e4 = slopes["btk_only"]["mean"] <= slopes["relu_mix"]["mean"]

    out = {
        "card": "CALIB_CARD.md (ACTMIX Stage 2; descriptive, NON-CLAIMING)",
        "status": "PENDING TEAM REVIEW",
        "ds": DS, "seeds": list(SEEDS), "n_cells": len(cells),
        "cells": cells,
        "mean_delta_trained": {k: {"mean": round(v[0], 4), "per_seed": v[1]}
                               for k, v in d_tr.items()},
        "mean_delta_untrained": {k: {"mean": round(v[0], 4), "per_seed": v[1]}
                                 for k, v in d_un.items()},
        "post_slope_dlog2T": {**slopes, "delta_btk_minus_relu": dslope},
        "l0_out_of_band": l0_flags,
        "untrained_sanity_max_abs_delta": round(
            max(abs(v[0]) for v in d_un.values()), 4),
        "expectations": {
            "E1_sae_improves_most": bool(e1),
            "E2_tsae_moves_least": bool(e2),
            "E3_post_lowT_recovers_more": bool(e3),
            "E4_slope_softens": bool(e4),
        },
    }
    dst = HERE / "results" / "calib_score.json"
    dst.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in
                      ("mean_delta_trained", "post_slope_dlog2T",
                       "expectations", "l0_out_of_band",
                       "untrained_sanity_max_abs_delta")}, indent=2))
    print(f"[score] wrote {dst}")


if __name__ == "__main__":
    main()
