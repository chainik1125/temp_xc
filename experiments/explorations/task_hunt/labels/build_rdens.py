"""`rdens` — referential-density TREND on the Ward stream (gen-4 seed
idea 3, directive 59ad15f38; factory venue, chaz playbook).

    .venv/bin/python -m experiments.explorations.task_hunt.labels.build_rdens

Face: kernel-WLS SLOPE (hunt house kernel: support 64 tok, HL 16 —
`hunt3_lib.filter_slope`) of the REFERENTIAL-token indicator over the
trace — "anaphoric load rising/falling". The TREND, not the level:
chaz proved Ward ambient LEVELS are pooling-readable (g_agg ≥ g); a
64-support slope reaches beyond every probed T ≤ 32 window, so the
window-MEAN control (the instrument that killed chaz) is the exact
deciding arm again, and we say so before running.

Referential lexicon (FIXED, pre-registered here): 3rd-person pronouns
+ possessives/reflexives + demonstratives + wh-relatives (see
REF_WORDS). A vocab id is referential iff its decoded string strips
(whitespace/case) to a lexicon word.

Conventions cloned from sc_lambda/chaz (frozen factory pipeline):
- null = same functional over the WITHIN-TRACE-SHUFFLED indicator
  (trace-rate-preserving; seed 211 + trace_idx);
- mask_rows: current token referential (the is_marker_tok analogue —
  the zero-distance give-away), pos < 64 (full support), ~valid;
- `fl.bundle_core` does binning / balanced manifests / by-trace split
  / label-side triage (KILL AUTHORITY: on FAIL the stats JSON is the
  kill receipt and the npz is withheld);
- `_cap_manifest` 6000/class post-triage (the chaz OOM lesson,
  disclosed; triage computed on the FULL manifest);
- per-T visible floors on ext rows: truncated-support slope+rate of
  the SAME indicator (rank AUC top-vs-bottom) — the evidence lines
  for the card.

Artifact: ``rdens.npz`` + ``rdens_stats.json`` (factory_screen
target-"" layout).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import factory_lib as fl
from . import wardmap
from .hunt3_lib import filter_rate, filter_slope
from .interleave_lib import rank_auc

HERE = Path(__file__).resolve().parent
NAME = "rdens"
SEED = 0
NULL_SEED = 211
SUPPORT = 64
POS_MIN = 64           # full kernel support (stricter than factory's 32)
EVIDENCE_TS = (8, 16, 32)
CAP_PER_CLASS = 6000

REF_WORDS = {
    "he", "she", "it", "they", "him", "her", "them",
    "his", "hers", "its", "their", "theirs",
    "himself", "herself", "itself", "themselves",
    "this", "that", "these", "those",
    "who", "whom", "whose", "which",
}


def ref_vocab_ids(tok) -> np.ndarray:
    """Vocab ids whose decoded string strips to a referential word."""
    n = int(tok.vocab_size) + len(getattr(tok, "added_tokens_decoder", {}))
    ids = []
    for i in range(n):
        s = tok.decode([i]).strip().lower()
        if s in REF_WORDS:
            ids.append(i)
    return np.asarray(sorted(ids), dtype=np.int64)


def main():
    tok, traces, _by_qid = wardmap.load_inputs()
    rids = ref_vocab_ids(tok)
    ref_lut_max = 1 << 18
    assert rids.max() < ref_lut_max
    ref_lut = np.zeros(ref_lut_max, dtype=bool)
    ref_lut[rids] = True

    z = np.load(HERE / "sc_lambda.npz")          # row layout of record
    trace_idx, win_start = z["trace_idx"], z["win_start"]
    valid = z["valid"].astype(bool)
    N, L = z["lam_sc"].shape

    grids = {k: np.full((N, L), np.nan, dtype=np.float32)
             for k in ("rd", "rd_null")}
    ref_grid = np.zeros((N, L), dtype=np.int8)
    ids_grid = np.zeros((N, L), dtype=np.int64)
    floors = {f"fl_{k}{T}": np.full((N, L), np.nan, dtype=np.float32)
              for T in EVIDENCE_TS for k in ("s", "r")}
    trace_len = {}
    ref_rate_all = []

    for ti in np.unique(trace_idx):
        ids, _offs = wardmap.tokenize_trace(tok, traces[int(ti)])
        arr = np.asarray(ids, dtype=np.int64)
        trace_len[int(ti)] = len(arr)
        ev = ref_lut[np.clip(arr, 0, ref_lut_max - 1)].astype(float)
        ref_rate_all.append(float(ev.mean()))
        rd = filter_slope(ev, SUPPORT)
        rng = np.random.default_rng(NULL_SEED + int(ti))
        rd_null = filter_slope(rng.permutation(ev), SUPPORT)
        per_t = {}
        for T in EVIDENCE_TS:
            per_t[f"fl_s{T}"] = filter_slope(ev, min(T, SUPPORT))
            per_t[f"fl_r{T}"] = filter_rate(ev, min(T, SUPPORT))
        for r in np.flatnonzero(trace_idx == ti):
            s = int(win_start[r])
            seg = slice(s, min(s + L, len(arr)))
            m = seg.stop - seg.start
            grids["rd"][r, :m] = rd[seg]
            grids["rd_null"][r, :m] = rd_null[seg]
            ref_grid[r, :m] = ev[seg].astype(np.int8)
            ids_grid[r, :m] = arr[seg]
            for k, v in per_t.items():
                floors[k][r, :m] = v[seg]

    rd, rd_null = grids["rd"], grids["rd_null"]
    rd[~valid] = np.nan
    rd_null[~valid] = np.nan

    # position-in-TRACE of each cell (win_start + col), for POS_MIN
    pos_in_trace = win_start[:, None] + np.arange(L)[None, :]
    mask_rows = (ref_grid == 1) | (pos_in_trace < POS_MIN)

    core = fl.bundle_core(rd, rd_null, mask_rows, valid, trace_idx,
                          win_start, trace_len, ids_grid, seed=SEED)

    def _cap_manifest(man, cap=CAP_PER_CLASS):
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
    nd, npos, nc = _cap_manifest(core["man_null"])

    # per-T visible-floor evidence lines on the SAME ext rows the
    # triage used (rank AUC, top vs bottom class)
    ext_d, ext_p = core["ext_rows_d"], core["ext_rows_p"]
    is_top, is_test = core["ext_is_top"], core["ext_is_test"]
    ev_lines = {}
    for T in EVIDENCE_TS:
        m = is_test.astype(bool)
        sl = floors[f"fl_s{T}"][ext_d, ext_p]
        rt = floors[f"fl_r{T}"][ext_d, ext_p]
        fin = np.isfinite(sl) & np.isfinite(rt) & m
        ev_lines[str(T)] = {
            "slope_auc": rank_auc(sl[fin], is_top[fin]),
            "rate_auc": rank_auc(rt[fin], is_top[fin]),
            "n": int(fin.sum())}

    per_cls = {int(c): int((mc == c).sum()) for c in (0, 1, 2)}
    stats = {
        "face": "referential-density TREND (filter_slope support 64 "
                "HL 16 of the REF_WORDS indicator; llama31 tokenizer)",
        "lexicon_n_vocab_ids": int(len(rids)),
        "ref_token_rate_mean": float(np.mean(ref_rate_all)),
        "null": f"within-trace shuffle, seed {NULL_SEED}+trace",
        "mask": "current-token-referential + pos<64 + ~valid",
        "scheme": core["scheme"],
        "manifest_rows_per_class": per_cls,
        "eligible_rows": int((~mask_rows & valid).sum()),
        "triage": core["triage"],
        "visible_floor_auc_by_T": ev_lines,
    }
    if core["triage"]["verdict"] == "FAIL":
        (HERE / f"{NAME}_stats.json").write_text(json.dumps(stats, indent=1))
        print(f"[{NAME}] TRIAGE FAIL — kill receipt written, npz withheld: "
              + json.dumps(core["triage"]))
        return
    out = {
        "rd": rd, "rd_null": rd_null,
        "lam_bin": core["bins"].astype(np.int8),
        "lam_null_bin": core["null_bins"].astype(np.int8),
        "is_ref_tok": ref_grid, "valid": z["valid"],
        "trace_idx": trace_idx, "win_start": win_start,
        "trace_split": core["trace_split"],
        "man_doc": md, "man_pos": mp, "man_cls": mc,
        "man_null_doc": nd, "man_null_pos": npos, "man_null_cls": nc,
        **{k: v.astype(np.float16) for k, v in floors.items()},
    }
    np.savez_compressed(HERE / f"{NAME}.npz", **out)
    stats["artifact"] = f"{NAME}.npz"
    (HERE / f"{NAME}_stats.json").write_text(json.dumps(stats, indent=1))
    print(f"[{NAME}] rows/class {per_cls}; ref-rate "
          f"{stats['ref_token_rate_mean']:.4f}; triage "
          + json.dumps({k: round(v, 4) if isinstance(v, float) else v
                        for k, v in core["triage"].items()})
          + "; floors " + json.dumps({t: {k: round(v, 3) if isinstance(v, float) else v
                                          for k, v in d.items()}
                                      for t, d in ev_lines.items()}))


if __name__ == "__main__":
    main()
