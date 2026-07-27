"""`chaz` — correction hazard with the conversion channel removed
(overnight § 1 seed 4; design note in the LOG entry beside the HUNT3
freeze). Derived ENTIRELY from the committed `sc_lambda.npz` label
(same events, same frozen kernel, same binning) plus ONE stricter row
rule: **eligibility = the trailing 32-token view is CUE-FREE** (no
marker-span token in positions [t-31, t]), so at every probed
T ≤ 32 the probe must read persistent state deposited by cues ≥ 33
tokens back. sc_lambda's screen verdict was "a converted latent with
an aggregation bonus" — this construction makes the conversion
channel structurally impossible; what survives is the state.

    .venv/bin/python -m experiments.explorations.task_hunt.labels.build_chaz

Re-runs `factory_lib.bundle_core` with mask_rows |= NOT-cue-free-32
(and pos < 32), so binning/manifests/null/triage stay the frozen
factory pipeline. tok_id / trace_len come from the same wardmap
tokenization the sc_lambda builder used. Artifact: ``chaz.npz`` +
``chaz_stats.json`` (factory_screen-compatible: target "" layout).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import factory_lib as fl
from . import wardmap

HERE = Path(__file__).resolve().parent
NAME = "chaz"
SEED = 0
CF_T = 32              # cue-free trailing view length (covers all probed T)
POS_MIN = 32


def cue_free_trailing(mk: np.ndarray, T: int = CF_T) -> np.ndarray:
    """(N, L) bool: no marker token in positions [t-T+1, t] (row-local;
    factory rows are independent 128-token views)."""
    csum = np.cumsum(mk, axis=1)
    N, L = mk.shape
    out = np.zeros((N, L), dtype=bool)
    for t in range(L):
        lo = max(0, t - T + 1)
        prior = csum[:, lo - 1] if lo > 0 else np.zeros(N, dtype=csum.dtype)
        out[:, t] = (csum[:, t] - prior) == 0
    return out


def main():
    z = np.load(HERE / "sc_lambda.npz")
    lam, lam_null = z["lam_sc"], z["lam_sc_null"]
    mk = z["is_marker_tok"].astype(bool)
    valid = z["valid"].astype(bool)
    trace_idx, win_start = z["trace_idx"], z["win_start"]
    N, L = lam.shape

    tok, traces, _ = wardmap.load_inputs()
    trace_len = {}
    ids_grid = np.zeros((N, L), dtype=np.int64)
    for ti in np.unique(trace_idx):
        ids, _offs = wardmap.tokenize_trace(tok, traces[int(ti)])
        trace_len[int(ti)] = len(ids)
        rows = np.flatnonzero(trace_idx == ti)
        arr = np.asarray(ids, dtype=np.int64)
        for r in rows:
            s = int(win_start[r])
            seg = arr[s: s + L]
            ids_grid[r, : len(seg)] = seg

    cf = cue_free_trailing(mk)
    pos_ok = np.arange(L)[None, :] >= POS_MIN
    mask_rows = mk | (~cf) | (~pos_ok)

    core = fl.bundle_core(lam, lam_null, mask_rows, valid, trace_idx,
                          win_start, trace_len, ids_grid, seed=SEED)

    def _cap_manifest(man, cap=6000, tag=""):
        """factory_screen probes flat T·d_in windows on ALL manifest
        rows; the balanced_manifest 20k/class default OOMs a 44 GB L40S
        at T32 (measured, first launch). Post-triage per-class
        subsample to the factory-era scale — seeded, disclosed; the
        triage receipts above are computed on the FULL manifest."""
        d, p, c = man
        rng = np.random.default_rng(SEED + 7)
        keep = []
        for cls in (0, 1, 2):
            idx = np.flatnonzero(c == cls)
            if len(idx) > cap:
                idx = np.sort(rng.choice(idx, cap, replace=False))
            keep.append(idx)
        idx = np.sort(np.concatenate(keep))
        return d[idx], p[idx], c[idx]

    md, mp, mc = _cap_manifest(core["man"])
    nd, npos, nc = _cap_manifest(core["man_null"], tag="null")
    out = {
        "lam_sc": lam, "lam_sc_null": lam_null,
        "lam_bin": core["bins"].astype(np.int8),
        "lam_null_bin": core["null_bins"].astype(np.int8),
        "is_marker_tok": z["is_marker_tok"], "cue_free_32": cf,
        "valid": z["valid"], "trace_idx": trace_idx,
        "win_start": win_start, "trace_split": core["trace_split"],
        "man_doc": md, "man_pos": mp, "man_cls": mc,
        "man_null_doc": nd, "man_null_pos": npos, "man_null_cls": nc,
    }
    per_cls = {int(c): int((mc == c).sum()) for c in (0, 1, 2)}
    stats = {
        "derives_from": "sc_lambda.npz (same events/kernel/binning; "
                        "frozen factory pipeline via bundle_core)",
        "rule": f"eligibility = cue-free trailing {CF_T} tokens "
                f"(+ pos >= {POS_MIN}, marker tokens masked as before)",
        "scheme": core["scheme"], "edges": list(map(float, core["edges"]))
        if core["edges"] is not None else None,
        "manifest_rows_per_class": per_cls,
        "eligible_rows": int((~mask_rows & valid).sum()),
        "triage": core["triage"],
    }
    if core["triage"]["verdict"] == "FAIL":
        (HERE / f"{NAME}_stats.json").write_text(json.dumps(stats, indent=1))
        print(f"[{NAME}] TRIAGE FAIL — kill receipt written, npz withheld: "
              + json.dumps(core["triage"]))
        return
    np.savez_compressed(HERE / f"{NAME}.npz", **out)
    stats["artifact"] = f"{NAME}.npz"
    (HERE / f"{NAME}_stats.json").write_text(json.dumps(stats, indent=1))
    print(f"[{NAME}] rows/class {per_cls}; triage "
          + json.dumps({k: round(v, 4) if isinstance(v, float) else v
                        for k, v in core["triage"].items()}))


if __name__ == "__main__":
    main()
