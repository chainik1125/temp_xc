"""``warddebt`` — unverified-assertion debt on the Ward stream
(SAFETY_TASK_MENU § 10.1 #23; briefing `safety-hunt-continuation.md`).

    .venv/bin/python -m experiments.explorations.task_hunt.labels.build_warddebt

Protocol frozen in ``../warddebt/CARD.md`` BEFORE this ran. The face is
a **difference of two rates the program already built**:

    debt = rate_case (class 2, obligations incurred)
         - rate_ver  (class 3, obligations discharged)

both under the SAME frozen kernel (τ=3, K=8, causal, min_history 4,
current sentence never an input), so the difference is well-defined
per sentence and inherits the NaN rule from both parents.

Everything here reuses ``build_oprate.py``'s instruments verbatim —
``wardmap.broadcast``, ``factory_lib.kernel_rate`` / ``bundle_core``
(binning, balanced manifests, by-trace split, triage with kill
authority), ``trailing_count_incl`` for the visible-evidence floor —
so the numbers are directly comparable to the ``oprate`` obituary.

**Mask generalization (stated, not silent):** ``oprate`` masks rows
whose current sentence IS the event class or is unlabeled. Debt has
TWO event classes, so rows are masked where the current sentence is
class 2 OR class 3 OR unlabeled. Anything looser would let the current
sentence's own class leak into a face that is about the trailing
balance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from . import factory_lib as fl
from . import lib, wardmap
from .interleave_lib import rank_auc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LABELS = (ROOT / "experiments/explorations/synthetic/expansion/records/"
          "proof-operation-phase-runs/labels.json")
NAME = "warddebt"
SEED = 0
NULL_SEED = 103                      # oprate's, verbatim
EVIDENCE_TS = (8, 16, 32)
MIN_MANIFEST_CLASS = 2000
CLS_INCUR, CLS_DISCHARGE = 2, 3      # case-enumeration, verification-check
ANTI_DUP_BAR = 0.8


def _spear(a, b, m) -> float:
    ok = m & np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[ok], b[ok]).statistic)


def main() -> None:
    tok, traces, by_qid = wardmap.load_inputs()
    rec = json.loads(LABELS.read_text())
    ops_by_qid = dict(zip(rec["doc_ids"], rec["labels"]))
    trace_len, sent_tok = {}, []

    def payload(ti, ids, offs):
        trace_len[ti] = len(ids)
        slab = by_qid[traces[ti]["question_id"]]
        spans = [(s["char_start"], s["char_end"]) for s in slab["sentences"]]
        labels = ops_by_qid.get(traces[ti]["question_id"])
        if labels is None:
            labels = [None] * len(spans)
        op_s = np.array([-1 if l is None else l for l in labels],
                        dtype=np.int8)
        rng = np.random.default_rng(NULL_SEED + ti)
        lab_pos = np.flatnonzero(op_s >= 0)
        op_null = op_s.copy()
        op_null[lab_pos] = op_s[lab_pos][rng.permutation(lab_pos.size)]
        sidx, in_span = lib.sentence_index_per_token(offs, spans)
        # CLOCK ingredient: tokens per sentence, in-span only
        u, c = np.unique(sidx[in_span], return_counts=True)
        sent_tok.extend(c.tolist())

        def _debt(o):
            inc = np.where(o >= 0, (o == CLS_INCUR).astype(float), np.nan)
            dis = np.where(o >= 0, (o == CLS_DISCHARGE).astype(float),
                           np.nan)
            return fl.kernel_rate(inc) - fl.kernel_rate(dis)

        out = {"op": op_s[sidx], "tok_id": np.asarray(ids, dtype=np.int32),
               "sent_idx": sidx.astype(np.int16),
               "in_span": in_span.astype(np.int8),
               "debt": _debt(op_s)[sidx],
               "debt_null": _debt(op_null)[sidx]}
        # the window-visible cheat: net count of incurred - discharged
        # sentences inside the trailing T tokens
        f_inc = (op_s[sidx] == CLS_INCUR).astype(np.int8)
        f_dis = (op_s[sidx] == CLS_DISCHARGE).astype(np.int8)
        for T in EVIDENCE_TS:
            out[f"ev_net{T}"] = (fl.trailing_count_incl(f_inc, T)
                                 - fl.trailing_count_incl(f_dis, T))
            out[f"ev_inc{T}"] = fl.trailing_count_incl(f_inc, T)
        return out

    grids, valid_rt, trace_idx, win_start, n_mm = wardmap.broadcast(
        tok, traces, payload)
    in_span = grids["in_span"] == 1
    valid = valid_rt & in_span
    op = grids["op"]
    lam, lam_null = grids["debt"], grids["debt_null"]
    lam[~valid] = np.nan
    lam_null[~valid] = np.nan

    st = np.array(sent_tok, dtype=float)
    clock = {
        "tokens_per_sentence_mean": float(st.mean()),
        "tokens_per_sentence_median": float(np.median(st)),
        "kernel_support_sentences": fl.FROZEN_K,
        "kernel_support_tokens_mean": float(fl.FROZEN_K * st.mean()),
        "screen_Ts": list(EVIDENCE_TS),
        "reach_note": ("if the kernel support in TOKENS far exceeds the "
                       "screened T, the window cannot COMPUTE the face "
                       "and any in-window signal is self-excitation, "
                       "not the trailing balance (the sycgen_rate "
                       "demotion, refmark's death mode)"),
    }
    print(f"[clock] {clock['tokens_per_sentence_mean']:.1f} tok/sentence; "
          f"K=8 kernel spans ≈ {clock['kernel_support_tokens_mean']:.0f} "
          f"tokens vs screened T {EVIDENCE_TS}", flush=True)

    # ── ANTI-DUP FIRST (free, and it can end this) ──────────────────
    rv_z = np.load(HERE / "oprate.npz")
    anti = {}
    m = valid & np.isfinite(lam)
    for nm, arr in (("rate_case", rv_z["rate_case"]),
                    ("rate_ver", rv_z["rate_ver"])):
        anti[f"vs_{nm}"] = _spear(lam, arr, m)
    for nm, path, key in (("lam_sc", "sc_lambda.npz", "lam_sc"),
                          ("ward_lambda", "ward_lambda.npz", None)):
        p = HERE / path
        if not p.exists():
            continue
        z = np.load(p)
        k = key or [x for x in z.files if z[x].shape == lam.shape][:1]
        k = k if isinstance(k, str) else (k[0] if k else None)
        if k:
            anti[f"vs_{nm}"] = _spear(lam, z[k], m)
    breached = {k: v for k, v in anti.items() if abs(v) >= ANTI_DUP_BAR}
    for k, v in anti.items():
        print(f"[anti-dup] {k}: rho={v:+.3f}"
              f"{'  <-- BREACH' if abs(v) >= ANTI_DUP_BAR else ''}",
              flush=True)

    mask_rows = (op == CLS_INCUR) | (op == CLS_DISCHARGE) | (op == -1)
    core = fl.bundle_core(lam, lam_null, mask_rows, valid, trace_idx,
                          win_start, trace_len, grids["tok_id"], seed=SEED)
    evidence = {}
    for T in EVIDENCE_TS:
        for tag in ("net", "inc"):
            sc = grids[f"ev_{tag}{T}"][core["ext_rows_d"],
                                       core["ext_rows_p"]]
            ok = core["ext_is_test"] & np.isfinite(sc)
            evidence[f"{tag}_T{T}"] = rank_auc(sc[ok],
                                               core["ext_is_top"][ok])
    print("[visible floor] "
          + " ".join(f"{k}={v:.3f}" for k, v in evidence.items()),
          flush=True)

    bins = core["bins"]
    mval = valid & (bins >= 0) & (op >= 0)
    net_by_bin = [float(np.mean(op[mval & (bins == k)] == CLS_INCUR)
                        - np.mean(op[mval & (bins == k)] == CLS_DISCHARGE))
                  for k in (0, 1, 2)]
    fin = valid & np.isfinite(lam) & np.isfinite(lam_null)
    n_per_class = int(len(core["man"][0]) // 3)
    stats = {
        "card": "warddebt/CARD.md (frozen before this ran)",
        "frozen": {"kernel": {"tau": fl.FROZEN_TAU, "k": fl.FROZEN_K,
                              "min_history": fl.MIN_HISTORY},
                   "labels_source": str(LABELS.relative_to(ROOT)),
                   "face": "rate_case(cls 2) - rate_ver(cls 3)",
                   "mask": "current sentence in {2,3} OR unlabeled",
                   "null_seed": NULL_SEED, "seed": SEED},
        "clock_stated_first": clock,
        "anti_dup_spearman": anti,
        "anti_dup_bar": ANTI_DUP_BAR,
        "anti_dup_breached": breached,
        "parent_corr_rate_ver_rate_case": float(np.corrcoef(
            rv_z["rate_case"][fin], rv_z["rate_ver"][fin])[0, 1]),
        "stream_shape": [int(v) for v in op.shape],
        "label_coverage_rate": float(np.isfinite(lam[valid]).mean()),
        "debt_quantiles": [float(q) for q in
                           np.nanquantile(lam[valid], [.05, .25, .5,
                                                       .75, .95])],
        "bin_scheme": core["scheme"], "edges": core["edges"],
        "null_corr_with_real": float(np.corrcoef(lam[fin],
                                                 lam_null[fin])[0, 1]),
        "net_class_rate_by_bin": net_by_bin,
        "manifest_rows_per_class": n_per_class,
        "triage": core["triage"],
        "visible_evidence_auc": evidence,
        "n_roundtrip_mismatch_tokens": int(n_mm),
    }
    max_floor = max(abs(v - 0.5) for v in evidence.values()) + 0.5
    stats["verdict_input"] = {
        "anti_dup_ok": not breached,
        "triage_ok": core["triage"]["verdict"] != "FAIL",
        "manifest_ok": n_per_class >= MIN_MANIFEST_CLASS,
        "max_visible_floor_auc": float(max_floor),
        "kill_rule": ("anti-dup breach (|rho| >= 0.8) OR triage FAIL OR "
                      "manifest short => label-side KILL, no GPU screen"),
    }
    stats["verdict_input"]["label_side_killed"] = bool(
        breached or core["triage"]["verdict"] == "FAIL"
        or n_per_class < MIN_MANIFEST_CLASS)

    (HERE / f"{NAME}_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: stats[k] for k in
                      ("anti_dup_spearman", "triage", "visible_evidence_auc",
                       "verdict_input")}, indent=1))
    if stats["verdict_input"]["label_side_killed"]:
        print(f"[LABEL-SIDE KILL] {NAME}: npz NOT shipped")
        return
    arrays = {"op": op, "sent_idx": grids["sent_idx"], "in_span": in_span,
              "valid": valid, "trace_idx": trace_idx,
              "win_start": win_start, "debt": lam, "debt_null": lam_null,
              "debt_bin": bins, "debt_null_bin": core["null_bins"],
              "trace_split": core["trace_split"]}
    for tag, (d, p, c) in (("", core["man"]), ("null_", core["man_null"])):
        arrays[f"man_debt_{tag}doc"] = d
        arrays[f"man_debt_{tag}pos"] = p
        arrays[f"man_debt_{tag}cls"] = c
    np.savez_compressed(HERE / f"{NAME}.npz", **arrays)
    print(f"-> {HERE / f'{NAME}.npz'}")


if __name__ == "__main__":
    main()
