"""``msdose_r1`` re-entry PRE-MEASURE (dispatch ``47040da59``; amendment
frozen in ``msdose_r1_lib`` + ``msdose_r1/PRECOUNT_AMENDMENT.md``).

    .venv/bin/python -m experiments.explorations.task_hunt.labels.build_msdose_r1

PRE-MEASURE builder, not a card: it decides whether the § B redesign
earns a screen. Per tokenizer it does three things, in order:

1. **Baseline census on the KILLED corpus** — loads the committed
   ``wave3_msdose_<tok>.npz`` verbatim and runs the frozen § B strata
   census on it. This realises the "2/31, 86,568 tokens" side of the
   § B table on the actual artifact (the § B numbers were plan-level
   simulation), so the r1 comparison is realised-vs-realised under ONE
   committed instrument.
2. **r1 build** — same committed gen4c wikitext streams, same
   ``wave3_lib.msdose_doc`` assembly, same delimiter; only the plan
   changes (per-doc scale, ``msdose_r1_lib``). Writes
   ``wave3_msdose_r1_<tok>.npz``.
3. **r1 pre-measure** — the trio instruments verbatim (imported from
   ``build_wave3_trio``: terciles, triage AUCs + doc-cluster bootstrap,
   floors) + the § B census + the confirmation-band evaluation against
   the pre-registered bands in the amendment card.

Outputs ``msdose_r1_premeasure.json`` (artifact of record) with a
freeze receipt (HEAD sha + clean-tree assertion for the frozen logic).
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np

from . import msdose_r1_lib as r1
from . import wave3_lib as w3
from .build_wave3_trio import (MIN_POS, N_REPS, SEED, _floor_aucs, _spear,
                               _terciles, _triage)
from .hunt3_lib import FLOOR_TS
from .lib import doc_split

HERE = Path(__file__).resolve().parent
TOKS = ("gpt2", "gemma2", "llama31")
FROZEN_FILES = ("experiments/explorations/task_hunt/labels/msdose_r1_lib.py",
                "experiments/explorations/task_hunt/labels/build_msdose_r1.py",
                "experiments/explorations/task_hunt/labels/wave3_lib.py")

# ── pre-registered confirmation bands (amendment card § 4, verbatim) ────
BAND_RHO_MAX = 0.87            # pooled dose↔position Spearman, per tok
BAND_QUAL_MIN_ABS = 8          # qualifying strata, per tok
BAND_QUAL_MIN_RATIO = 4.0      # vs realised frozen baseline, per tok
BAND_USABLE_MIN_ABS = 250_000  # usable position-matched tokens, per tok
BAND_USABLE_MIN_RATIO = 3.0    # vs realised frozen baseline, per tok


def _freeze_receipt() -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE,
                          capture_output=True, text=True,
                          check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--"]
                           + list(FROZEN_FILES),
                           cwd=HERE, capture_output=True, text=True,
                           check=True).stdout.strip()
    assert not dirty, f"frozen logic dirty at run time:\n{dirty}"
    return {"head": head, "frozen_files_clean": True,
            "frozen_files": list(FROZEN_FILES)}


def _grid_views(doc_off: np.ndarray):
    n_docs = len(doc_off) - 1
    doc_of = np.repeat(np.arange(n_docs, dtype=np.int32), np.diff(doc_off))
    pos_of = np.concatenate([np.arange(n, dtype=np.int32)
                             for n in np.diff(doc_off)])
    return n_docs, doc_of, pos_of


def frozen_baseline_census(key: str) -> dict:
    z = np.load(HERE / f"wave3_msdose_{key}.npz")
    dose, bound, doc_off = z["dose"], z["is_boundary"], z["doc_off"]
    _, _, pos_of = _grid_views(doc_off)
    elig = (bound == 0) & (pos_of >= MIN_POS)
    c = r1.strata_census(dose, pos_of, elig)
    c["rho_pooled"] = _spear(dose, pos_of.astype(float), elig)["rho"]
    print(f"  [{key}] FROZEN baseline: {c['n_qualifying']}/"
          f"{c['n_strata_any']} strata, {c['usable_tokens']:,} usable, "
          f"rho={c['rho_pooled']:.3f}", flush=True)
    return c


def build_and_measure(key: str, out_dir: Path) -> dict:
    from transformers import AutoTokenizer
    hf = {"gpt2": "gpt2", "gemma2": "google/gemma-2-2b",
          "llama31": "NousResearch/Meta-Llama-3.1-8B"}[key]
    tok = AutoTokenizer.from_pretrained(hf)
    delim = np.asarray(tok(w3.MSDOSE_DELIM_TEXT,
                           add_special_tokens=False)["input_ids"],
                       dtype=np.int32)
    z = np.load(HERE / f"gen4c_wikitext103_{key}.npz")
    flat, src_off = z["token_ids"], z["doc_off"]

    rng = np.random.default_rng(r1.MSDOSE_R1_SEED)
    plan = r1.msdose_r1_plan(rng)
    ids_l, bound_l, dose_l, off = [], [], [], [0]
    for lens in plan:
        i, b, ds = w3.msdose_doc(rng, flat, src_off, lens, delim)
        ids_l.append(i); bound_l.append(b); dose_l.append(ds)
        off.append(off[-1] + len(i))
    ids = np.concatenate(ids_l)
    bound = np.concatenate(bound_l)
    dose = np.concatenate(dose_l).astype(np.float32)
    doc_off = np.array(off, dtype=np.int64)
    n_docs, doc_of, pos_of = _grid_views(doc_off)
    split = doc_split(n_docs, seed=SEED)
    train_rows, test_rows = split[doc_of] == 0, split[doc_of] == 1

    elig = (bound == 0) & (pos_of >= MIN_POS)
    stats: dict = {
        "n_docs": n_docs, "n_tokens": int(ids.size),
        "delim_ids_len": int(len(delim)),
        "dose_position_spearman": _spear(dose, pos_of.astype(float), elig),
    }
    age = np.concatenate([w3.sage_face(bound[doc_off[d]:doc_off[d + 1]])
                          for d in range(n_docs)])
    stats["dose_vs_boundary_age_spearman"] = _spear(dose, age, elig)
    print(f"  [{key}] r1 dose↔position rho="
          f"{stats['dose_position_spearman']['rho']:.3f}", flush=True)

    bins, edges = _terciles(dose, train_rows, elig)
    st = {"tercile_edges": edges}
    st.update(_triage("msdose_r1", dose, bins, ids, pos_of, doc_of, doc_off,
                      train_rows, test_rows, N_REPS))
    cnt = {T: np.concatenate(
        [w3.dose_window_count(bound[doc_off[d]:doc_off[d + 1]], T)
         for d in range(n_docs)]) for T in FLOOR_TS}
    cage = {T: np.concatenate(
        [w3.sage_floor(bound[doc_off[d]:doc_off[d + 1]], T)
         for d in range(n_docs)]) for T in FLOOR_TS}
    st["floors"] = _floor_aucs("msdose_r1", bins, test_rows,
                               {"in_window_boundary_count": cnt,
                                "censored_boundary_age": cage})
    stats["msdose_r1"] = st
    stats["strata_census"] = r1.strata_census(dose, pos_of, elig)
    c = stats["strata_census"]
    print(f"  [{key}] r1 census: {c['n_qualifying']}/{c['n_strata_any']} "
          f"strata, {c['usable_tokens']:,} usable", flush=True)

    out = out_dir / f"wave3_msdose_r1_{key}.npz"
    np.savez_compressed(out, token_ids=ids, doc_off=doc_off,
                        is_boundary=bound, dose=dose, doc_split=split)
    stats["artifact"] = out.name
    return stats


def evaluate_bands(frozen: dict, r1s: dict) -> dict:
    rho = r1s["dose_position_spearman"]["rho"]
    q_f, q_r = frozen["n_qualifying"], r1s["strata_census"]["n_qualifying"]
    u_f, u_r = frozen["usable_tokens"], r1s["strata_census"]["usable_tokens"]
    bands = {
        "rho_pooled_le_0.87": {"value": rho, "pass": bool(rho <= BAND_RHO_MAX)},
        "qualifying_ge_8_and_4x_frozen": {
            "value": q_r, "frozen": q_f,
            "pass": bool(q_r >= BAND_QUAL_MIN_ABS
                         and q_r >= BAND_QUAL_MIN_RATIO * max(q_f, 1))},
        "usable_ge_250k_and_3x_frozen": {
            "value": u_r, "frozen": u_f,
            "pass": bool(u_r >= BAND_USABLE_MIN_ABS
                         and u_r >= BAND_USABLE_MIN_RATIO * max(u_f, 1))},
    }
    bands["all_pass"] = all(v["pass"] for v in bands.values()
                            if isinstance(v, dict))
    return bands


def main():
    t0 = time.time()
    receipt = _freeze_receipt()
    stats: dict = {
        "amendment": "msdose_r1 (WAVE3_SECOND_SOURCE.md §B; dispatch "
                     "47040da59); frozen logic msdose_r1_lib + wave3_lib",
        "freeze_receipt": receipt,
        "preregistered_prediction_simulated": {
            "rho_pooled": 0.844, "strata": "10/66", "usable_tokens": 397_481},
        "frozen_baseline_census": {},
        "r1": {},
        "bands": {},
    }
    for key in TOKS:
        print(f"[frozen baseline] {key}", flush=True)
        stats["frozen_baseline_census"][key] = frozen_baseline_census(key)
    for key in TOKS:
        print(f"[msdose_r1] {key}", flush=True)
        stats["r1"][key] = build_and_measure(key, HERE)
        stats["bands"][key] = evaluate_bands(
            stats["frozen_baseline_census"][key], stats["r1"][key])
        print(f"  [{key}] bands: "
              + ", ".join(f"{k}={'PASS' if v['pass'] else 'FAIL'}"
                          for k, v in stats["bands"][key].items()
                          if isinstance(v, dict)), flush=True)
    n_pass = sum(stats["bands"][k]["all_pass"] for k in TOKS)
    stats["verdict_input"] = {
        "tokenizers_all_pass": int(n_pass),
        "kill_rule": "any band missed on >= 2 of 3 tokenizers => kill",
        "killed": bool(n_pass <= 1),
    }
    p = HERE / "msdose_r1_premeasure.json"
    p.write_text(json.dumps(stats, indent=1))
    print(f"-> {p} in {time.time() - t0:.0f}s "
          f"(all-pass tokenizers: {n_pass}/3)", flush=True)


if __name__ == "__main__":
    main()
