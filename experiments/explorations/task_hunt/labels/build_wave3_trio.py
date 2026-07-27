"""Wave-3 zero-pull trio LABEL PRE-MEASURES (directive ae1ce5fb0):
``sycpress`` + ``reask`` on the committed refmark2k WildChat grid,
``msdose`` constructed from the committed gen4c wikitext streams.

    .venv/bin/python -m experiments.explorations.task_hunt.labels.build_wave3_trio

PRE-MEASURE builder, not a card: no manifests, no screens — census +
traps + floors + anti-dup, the § 8-record instruments that decide
which faces earn a card at all. Frozen logic lives in
``sycpress_lib`` / ``wave3_lib`` (committed BEFORE this run counts
anything); this file only wires it to the committed artifacts:

- WildChat faces REUSE the committed ``refmark2k_wildchat_<tok>.npz``
  token grid verbatim (token_ids/doc_off/turn_idx/is_assistant/
  is_boundary/rlam) — zero re-tokenization, and anti-dup vs refmark's
  rlam is computed on the identical row grid by construction.
- msdose docs are concatenations of committed gen4c wikitext token
  spans (wave3_lib construction, seed 0); only the frozen delimiter
  string is tokenized fresh (disclosed).

Outputs: ``wave3_refmark2k_<tok>.npz`` + ``wave3_msdose_<tok>.npz``
+ ``wave3_trio_stats.json`` (artifact of record).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from . import boot_lib as bo
from . import novelty_lib as nl
from . import punctint_lib as pl
from . import pull_refmark2k as pull
from . import sycpress_lib as sp
from . import wave3_lib as w3
from .hunt3_lib import FLOOR_TS

HERE = Path(__file__).resolve().parent
SEED = 0
MIN_POS = 32
HALF_LIFE_M, SUPPORT_M = 2, 8          # refmark's message kernel, verbatim
N_REPS = 500                            # triage bootstrap (doc-cluster)
TOKS = ("gpt2", "gemma2", "llama31")


def corpus_census(convs) -> dict:
    """Message-level event census — REASK EVENT MASS FIRST (the tretd
    starvation lesson), then sycpress incl. per-string counts."""
    re_counts, sy_counts = [], []
    per_string = {s: 0 for s in sp.PUSHBACK_SUBSTRINGS}
    n_user = 0
    for msgs in convs:
        ev_r = w3.reask_events(msgs)
        re_counts.append(int(ev_r.sum()))
        k = 0
        for role, content in msgs:
            if role == "assistant":
                continue
            n_user += 1
            if sp.is_pushback_turn(content):
                k += 1
                for s in sp.pushback_hits(content):
                    per_string[s] += 1
        sy_counts.append(k)
    r, s = np.array(re_counts), np.array(sy_counts)
    return {
        "reask": {"events_total": int(r.sum()),
                  "frac_convs_ge1": float((r >= 1).mean()),
                  "frac_convs_ge2": float((r >= 2).mean()),
                  "events_per_conv_max": int(r.max())},
        "sycpress": {"events_total": int(s.sum()),
                     "frac_convs_ge1": float((s >= 1).mean()),
                     "frac_convs_ge2": float((s >= 2).mean()),
                     "events_per_conv_max": int(s.max()),
                     "event_rate_user_msgs": float(s.sum() / max(n_user, 1)),
                     "per_string_message_counts": per_string},
    }


def _terciles(vals, train_rows, elig):
    """Tercile bins over eligible finite rows; zero-inflated values
    (degenerate quantiles, e.g. the rate face) fall back to
    punctint's zero_split scheme — refmark's own treatment."""
    m = train_rows & elig & np.isfinite(vals)
    q1, q2 = np.quantile(vals[m], [1 / 3, 2 / 3])
    if not q2 > q1:
        scheme, edges, bins = pl.zero_split_bins(vals, m)
        bins = np.where(elig & np.isfinite(vals), bins, -1).astype(np.int8)
        return bins, {"scheme": scheme, "edges": edges}
    bins = np.full(len(vals), -1, dtype=np.int8)
    e = elig & np.isfinite(vals)
    bins[e & (vals <= q1)] = 0
    bins[e & (vals > q1) & (vals <= q2)] = 1
    bins[e & (vals > q2)] = 2
    return bins, {"scheme": "terciles", "edges": (float(q1), float(q2))}


def _triage(name, vals, bins, ids, pos_of, doc_of, doc_off, train_rows,
            test_rows, n_reps) -> dict:
    n_docs = len(doc_off) - 1
    docmean = np.full(n_docs, np.nan)
    for d in range(n_docs):
        seg = vals[doc_off[d]:doc_off[d + 1]]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            docmean[d] = seg.mean()
    scores = {"unigram_auc": nl.type_mean_scores(ids, vals,
                                                 train_rows & (bins >= 0)),
              "position_auc": pos_of.astype(float),
              "doc_mean_only_auc": docmean[doc_of]}
    rmask = test_rows & (bins >= 0)
    out = {}
    for s, sc in scores.items():
        out[s] = nl.tercile_auc(sc, bins, rmask)
        b = bo.bootstrap_tercile_auc(sc, bins, rmask, doc_of,
                                     n_reps=n_reps, seed=SEED)
        out[s + "_ci"] = [b["ci_lo"], b["ci_hi"]]
        print(f"    {name}.{s}: {out[s]:.4f} "
              f"[{b['ci_lo']:.4f}, {b['ci_hi']:.4f}]", flush=True)
    return out


def _floor_aucs(name, bins, test_rows, feats: dict) -> dict:
    out = {}
    rmask = test_rows & (bins >= 0)
    for fname, per_t in feats.items():
        out[fname] = {f"T{T}": nl.tercile_auc(per_t[T], bins, rmask)
                      for T in FLOOR_TS}
        line = " ".join(f"T{T}={out[fname][f'T{T}']:.3f}" for T in FLOOR_TS)
        print(f"    {name}.floor.{fname}: {line}", flush=True)
    return out


def _spear(a, b, mask) -> dict:
    m = mask & np.isfinite(a) & np.isfinite(b)
    r = spearmanr(a[m], b[m]).statistic
    return {"rho": float(r), "n_rows": int(m.sum())}


def wildchat_faces(key: str, convs, out_dir: Path) -> dict:
    z = np.load(HERE / f"refmark2k_wildchat_{key}.npz")
    ids, doc_off = z["token_ids"], z["doc_off"]
    turn_idx, is_assist = z["turn_idx"], z["is_assistant"]
    boundary, rlam, split = z["is_boundary"], z["rlam"], z["doc_split"]
    n_docs = len(convs)
    assert n_docs <= len(doc_off) - 1
    end = int(doc_off[n_docs])          # smoke: prefix of the grid
    ids, doc_off = ids[:end], doc_off[:n_docs + 1]
    turn_idx, is_assist = turn_idx[:end], is_assist[:end]
    boundary, rlam, split = boundary[:end], rlam[:end], split[:n_docs]
    doc_of = np.repeat(np.arange(n_docs, dtype=np.int32), np.diff(doc_off))
    pos_of = np.concatenate([np.arange(n, dtype=np.int32)
                             for n in np.diff(doc_off)])
    train_rows, test_rows = split[doc_of] == 0, split[doc_of] == 1

    sy_first = np.zeros(len(ids), np.int8)
    sy_mask = np.zeros(len(ids), np.int8)
    re_first = np.zeros(len(ids), np.int8)
    re_mask = np.zeros(len(ids), np.int8)
    sy_lam = np.zeros(len(ids), np.float32)
    for d in range(n_docs):
        msgs = convs[d]
        lo, hi = doc_off[d], doc_off[d + 1]
        m_idx = turn_idx[lo:hi]
        assert m_idx.max() < len(msgs), f"turn_idx overflow doc {d}"
        ev_sy = np.array([1 if (r != "assistant" and sp.is_pushback_turn(c))
                          else 0 for r, c in msgs], dtype=np.int8)
        ev_re = w3.reask_events(msgs)
        sy_first[lo:hi] = w3.event_first_token_flags(m_idx, ev_sy)
        sy_mask[lo:hi] = w3.event_token_flags(m_idx, ev_sy)
        re_first[lo:hi] = w3.event_first_token_flags(m_idx, ev_re)
        re_mask[lo:hi] = w3.event_token_flags(m_idx, ev_re)
        lam = pl.sentence_lambda(ev_sy, half_life=HALF_LIFE_M,
                                 support=SUPPORT_M)
        sy_lam[lo:hi] = pl.token_labels_from_sentences(lam, m_idx)

    # faces are per-doc streams: apply sage support within each doc
    sy_age = np.concatenate([w3.sage_face(sy_first[doc_off[d]:doc_off[d+1]])
                             for d in range(n_docs)])
    re_age = np.concatenate([w3.sage_face(re_first[doc_off[d]:doc_off[d+1]])
                             for d in range(n_docs)])

    n_msgs = sum(len(m) for m in convs)
    clock = {"tokens_per_message_mean": float(ids.size / n_msgs),
             "kernel_support_tokens_mean": float(SUPPORT_M * ids.size
                                                 / n_msgs)}
    print(f"  [{key}] clock: {clock['tokens_per_message_mean']:.1f} "
          f"tok/msg; msg-kernel span ≈ "
          f"{clock['kernel_support_tokens_mean']:.0f} tok", flush=True)

    stats: dict = {"clock_stated_first": clock}
    faces = {
        "sycpress_rate": (sy_lam, sy_mask),
        "sycpress_age": (sy_age, sy_mask),
        "reask_age": (re_age, re_mask),
    }
    sy_floor_age = {T: np.concatenate(
        [w3.sage_floor(sy_first[doc_off[d]:doc_off[d+1]], T)
         for d in range(n_docs)]) for T in FLOOR_TS}
    re_floor_age = {T: np.concatenate(
        [w3.sage_floor(re_first[doc_off[d]:doc_off[d+1]], T)
         for d in range(n_docs)]) for T in FLOOR_TS}
    sy_floor_cnt = {T: np.concatenate(
        [w3.dose_window_count(sy_mask[doc_off[d]:doc_off[d+1]], T)
         for d in range(n_docs)]) for T in FLOOR_TS}
    re_floor_cnt = {T: np.concatenate(
        [w3.dose_window_count(re_mask[doc_off[d]:doc_off[d+1]], T)
         for d in range(n_docs)]) for T in FLOOR_TS}
    floors = {"sycpress_rate": {"in_window_event_tokens": sy_floor_cnt},
              "sycpress_age": {"censored_age": sy_floor_age,
                               "in_window_event_tokens": sy_floor_cnt},
              "reask_age": {"censored_age": re_floor_age,
                            "in_window_event_tokens": re_floor_cnt}}

    for fname, (vals, mask) in faces.items():
        elig = ((mask == 0) & (boundary == 0) & (is_assist == 1)
                & (pos_of >= MIN_POS))
        bins, edges = _terciles(vals, train_rows, elig)
        st = {"eligible_rows": int((elig & np.isfinite(vals)).sum()),
              "tercile_edges": edges}
        print(f"  [{key}] {fname}: {st['eligible_rows']:,} eligible "
              f"assistant-token rows", flush=True)
        st.update(_triage(fname, vals, bins, ids, pos_of, doc_of, doc_off,
                          train_rows, test_rows, N_REPS))
        st["floors"] = _floor_aucs(fname, bins, test_rows, floors[fname])
        stats[fname] = st

    elig_any = (boundary == 0) & (is_assist == 1) & (pos_of >= MIN_POS)
    stats["anti_dup_spearman"] = {
        "sycpress_rate_vs_refmark_rlam": _spear(sy_lam, rlam, elig_any),
        "sycpress_age_vs_refmark_rlam": _spear(sy_age, rlam, elig_any),
        "sycpress_age_vs_sycpress_rate": _spear(sy_age, sy_lam, elig_any),
        "reask_age_vs_sycpress_age": _spear(re_age, sy_age, elig_any),
        "reask_age_vs_sycpress_rate": _spear(re_age, sy_lam, elig_any),
        "reask_age_vs_refmark_rlam": _spear(re_age, rlam, elig_any),
    }
    for k2, v in stats["anti_dup_spearman"].items():
        print(f"  [{key}] anti-dup {k2}: rho={v['rho']:.3f} "
              f"(n={v['n_rows']:,})", flush=True)

    out = out_dir / f"wave3_refmark2k_{key}.npz"
    np.savez_compressed(out, sycpress_rate=sy_lam, sycpress_age=sy_age,
                        reask_age=re_age, sycpress_event_first=sy_first,
                        sycpress_event_mask=sy_mask,
                        reask_event_first=re_first, reask_event_mask=re_mask)
    stats["artifact"] = out.name
    return stats


def msdose_build(key: str, out_dir: Path) -> dict:
    from transformers import AutoTokenizer
    hf = {"gpt2": "gpt2", "gemma2": "google/gemma-2-2b",
          "llama31": "NousResearch/Meta-Llama-3.1-8B"}[key]
    tok = AutoTokenizer.from_pretrained(hf)
    delim = np.asarray(tok(w3.MSDOSE_DELIM_TEXT,
                           add_special_tokens=False)["input_ids"],
                       dtype=np.int32)
    z = np.load(HERE / f"gen4c_wikitext103_{key}.npz")
    flat, src_off = z["token_ids"], z["doc_off"]

    rng = np.random.default_rng(w3.MSDOSE_SEED)
    plan = w3.msdose_plan(rng)
    ids_l, bound_l, dose_l, off = [], [], [], [0]
    for lens in plan:
        i, b, ds = w3.msdose_doc(rng, flat, src_off, lens, delim)
        ids_l.append(i); bound_l.append(b); dose_l.append(ds)
        off.append(off[-1] + len(i))
    ids = np.concatenate(ids_l)
    bound = np.concatenate(bound_l)
    dose = np.concatenate(dose_l).astype(np.float32)
    doc_off = np.array(off, dtype=np.int64)
    n_docs = len(plan)
    doc_of = np.repeat(np.arange(n_docs, dtype=np.int32), np.diff(doc_off))
    pos_of = np.concatenate([np.arange(n, dtype=np.int32)
                             for n in np.diff(doc_off)])
    from .lib import doc_split
    split = doc_split(n_docs, seed=SEED)
    train_rows, test_rows = split[doc_of] == 0, split[doc_of] == 1

    elig = (bound == 0) & (pos_of >= MIN_POS)
    stats: dict = {
        "n_docs": n_docs, "n_tokens": int(ids.size),
        "delim_ids_len": int(len(delim)),
        "dose_position_spearman_LETHAL_TRAP_CHECK": _spear(
            dose, pos_of.astype(float), elig),
    }
    age = np.concatenate([w3.sage_face(bound[doc_off[d]:doc_off[d+1]])
                          for d in range(n_docs)])
    stats["dose_vs_boundary_age_spearman"] = _spear(dose, age, elig)
    print(f"  [{key}] msdose dose↔position rho="
          f"{stats['dose_position_spearman_LETHAL_TRAP_CHECK']['rho']:.3f}",
          flush=True)

    bins, edges = _terciles(dose, train_rows, elig)
    st = {"tercile_edges": edges}
    st.update(_triage("msdose", dose, bins, ids, pos_of, doc_of, doc_off,
                      train_rows, test_rows, N_REPS))
    cnt = {T: np.concatenate(
        [w3.dose_window_count(bound[doc_off[d]:doc_off[d+1]], T)
         for d in range(n_docs)]) for T in FLOOR_TS}
    cage = {T: np.concatenate(
        [w3.sage_floor(bound[doc_off[d]:doc_off[d+1]], T)
         for d in range(n_docs)]) for T in FLOOR_TS}
    st["floors"] = _floor_aucs("msdose", bins, test_rows,
                               {"in_window_boundary_count": cnt,
                                "censored_boundary_age": cage})
    stats["msdose"] = st

    out = out_dir / f"wave3_msdose_{key}.npz"
    np.savez_compressed(out, token_ids=ids, doc_off=doc_off,
                        is_boundary=bound, dose=dose, doc_split=split)
    stats["artifact"] = out.name
    return stats


def main():
    global N_REPS
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-convs", type=int, default=None,
                    help="limit conversations (smoke only; stats are "
                         "NOT the artifact of record under a limit)")
    ap.add_argument("--reps", type=int, default=None)
    a = ap.parse_args()
    if a.reps is not None:
        N_REPS = a.reps
    t0 = time.time()
    convs, meta = pull.load()
    if a.n_convs:
        convs = convs[: a.n_convs]
        meta = dict(meta, SMOKE_LIMIT=a.n_convs)
    stats = {
        "directive": "ae1ce5fb0 (menu accepted; zero-pull trio)",
        "frozen_logic": "sycpress_lib + wave3_lib @ the pre-count freeze "
                        "commit; refmark grid + gen4c streams reused "
                        "verbatim, zero content re-tokenization",
        "corpus": meta,
        "corpus_census": None,
        "per_tokenizer": {},
        "msdose": {},
    }
    print("[census] message-level (reask event mass FIRST)", flush=True)
    stats["corpus_census"] = corpus_census(convs)
    print(json.dumps(stats["corpus_census"], indent=1), flush=True)
    for key in TOKS:
        print(f"[wildchat faces] {key}", flush=True)
        stats["per_tokenizer"][key] = wildchat_faces(key, convs, HERE)
    for key in TOKS:
        print(f"[msdose] {key}", flush=True)
        stats["msdose"][key] = msdose_build(key, HERE)
    p = HERE / "wave3_trio_stats.json"
    p.write_text(json.dumps(stats, indent=1))
    print(f"-> {p} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
