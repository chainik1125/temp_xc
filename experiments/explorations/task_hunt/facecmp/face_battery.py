"""Face battery — arm-test SEVERAL face shapes on one borrowed corpus. $0.

`arm_test.py` showed the precise-count face `rate_H512` sits at chance. In
its RESULT I flagged what that does NOT establish: the tercile edges were
1.0/2.0, so the task was *"≤1 vs exactly 2 vs ≥3 events"* — **precise
small-integer counting**, not a graded accumulator. Leaving that as a
caveat would let the next reader assume either answer, so this measures it.

The program's own definition of the target regime is a **trailing
functional of sparse events**: "per-token-silent state accumulating or
decaying over context, built as offset-weighted functionals". An
**exponentially-weighted accumulator** is literally that object, and it is
the graded counterpart of the count face. If EWMA is also at chance, the
negative generalises from "counting is hard" to "this corpus's activations
carry recency and nothing else about the event stream". If EWMA reads,
the count face failed for its discretisation and the direction is alive.

Faces (all functionals of the SAME event stream, so the comparison is
clean):

    RECENCY_age     log2(1 + tokens since last event)      positive control
    rate_H512       count in trailing 512                  known: chance
    ewma_128/512    sum_i exp(-(t - t_i)/tau)              GRADED accumulator
    age2            log2(1 + tokens since 2nd-last event)  longer-range recency
    gap_last        log2(1 + gap between last two events)  local rate, no count

`age2` and `gap_last` are included because they separate two things the
count face confounds: reaching further back in time, versus counting. If
recency-like faces keep reading and count-like faces keep failing, that is
a sharper statement than either face alone supports.

Run: PYTHONPATH=. python -m experiments.explorations.task_hunt.facecmp.face_battery
Writes results/face_battery.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from experiments.explorations.task_hunt.labels import wave3_lib as w3

CACHE_ENV = "FACECMP_CACHE_ROOT"


def _per_doc(first, off, n_docs, fn):
    return np.concatenate([fn(np.asarray(first[off[d]:off[d + 1]],
                                         dtype=np.int64))
                           for d in range(n_docs)]).astype(np.float64)


def f_age(first, off, n_docs, _):
    return _per_doc(first, off, n_docs,
                    lambda b: w3.sage_face(b).astype(np.float64))


def f_rate(H):
    def g(first, off, n_docs, _):
        def per(b):
            c = np.concatenate([[0], np.cumsum(b)])
            i = np.arange(len(b))
            return (c[i] - c[np.maximum(i - H, 0)]).astype(np.float64)
        return _per_doc(first, off, n_docs, per)
    return g


def f_ewma(tau):
    """sum_i exp(-(t - t_i)/tau) over PAST events — the offset-weighted
    functional the program's own recipe names. Recursive, so O(n)."""
    def g(first, off, n_docs, _):
        decay = float(np.exp(-1.0 / tau))

        def per(b):
            out = np.empty(len(b), dtype=np.float64)
            s = 0.0
            for i in range(len(b)):
                s *= decay          # decay first: event at t is NOT visible at t
                out[i] = s
                if b[i]:
                    s += 1.0
            return out
        return _per_doc(first, off, n_docs, per)
    return g


def f_age2(first, off, n_docs, _):
    """Tokens since the SECOND-most-recent event: reaches further back
    without requiring a count."""
    def per(b):
        idx = np.flatnonzero(b)
        out = np.full(len(b), np.nan)
        if len(idx) < 2:
            return out
        # for each t, the 2nd-last event strictly before t
        k = np.searchsorted(idx, np.arange(len(b)), side="left") - 2
        ok = k >= 0
        out[ok] = np.arange(len(b))[ok] - idx[k[ok]]
        return np.log2(1.0 + out)
    return _per_doc(first, off, n_docs, per)


def f_gap_last(first, off, n_docs, _):
    """Gap between the two most recent events — a LOCAL rate that needs no
    count and no long horizon."""
    def per(b):
        idx = np.flatnonzero(b)
        out = np.full(len(b), np.nan)
        if len(idx) < 2:
            return out
        k = np.searchsorted(idx, np.arange(len(b)), side="left") - 1
        ok = k >= 1
        out[ok] = idx[k[ok]] - idx[k[ok] - 1]
        return np.log2(1.0 + out)
    return _per_doc(first, off, n_docs, per)


FACES = [
    ("RECENCY_age", f_age, 64),
    ("rate_H512", f_rate(512), 512),
    ("ewma_tau128", f_ewma(128), 512),
    ("ewma_tau512", f_ewma(512), 512),
    ("age2", f_age2, 64),
    ("gap_last", f_gap_last, 64),
]


def main():
    import experiments.explorations.task_hunt.facecmp.arm_test as at

    root = os.environ.get(CACHE_ENV)
    if root:
        at.CACHE_ROOT = Path(root)
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / "_battery"
    summary = []

    for name, fn, H in FACES:
        at.FACE, at.H, at.RES = name, H, scratch
        at.AX_TS, at.FOREIGN_TS = [16, 32, 64], [32, 64]
        at.rate_face = fn
        print(f"\n===== {name} (H={H}) =====", flush=True)
        try:
            at.screen("gpt2")
        except Exception as e:  # a face that cannot be built is a result
            print(f"  FAILED: {e}")
            summary.append({"face": name, "error": str(e)[:200]})
            continue
        p = scratch / "arm_test_gpt2.json"
        d = json.loads(p.read_text())
        s = d.get("summary")
        if not s:
            summary.append({"face": name, "skipped": "insufficient rows"})
            p.unlink(missing_ok=True)
            continue
        c = d["cells"]
        s["face"] = name
        s["H"] = H
        s["edges"] = d["meta"]["rows"]["tercile_edges"]
        s["foreign_null_T64"] = c.get(f"{name}/T64/actxmean_foreign_linear",
                                      {}).get("acc_test")
        s["label_null"] = c.get(f"{name}/T32/label_null", {}).get("acc_test")
        s["beats_own_null"] = (s["best_window"] > (s["foreign_null_T64"] or 0))
        summary.append(s)
        (scratch / f"cells_{name}.json").write_text(json.dumps(d, indent=1))
        p.unlink(missing_ok=True)

    print(f"\n{'face':<16}{'tok':>8}{'best':>8}{'T':>4}{'gain':>9}"
          f"{'foreign':>9}{'floor':>8}{'>null':>7}")
    print("-" * 69)
    for s in summary:
        if "gain_vs_tok" not in s:
            print(f"{s['face']:<16}  {s.get('skipped') or s.get('error')}")
            continue
        print(f"{s['face']:<16}{s['tok']:>8.4f}{s['best_window']:>8.4f}"
              f"{s['best_T']:>4}{s['gain_vs_tok']:>+9.4f}"
              f"{(s['foreign_null_T64'] or 0):>9.4f}{s['floor_at_bestT']:>8.4f}"
              f"{str(s['beats_own_null']):>7}")

    (out_dir / "face_battery.json").write_text(json.dumps(
        {"corpus": "elicit_retryesc_gen_v1 (BORROWED)", "model": "gpt2",
         "note": "feasibility probe, NOT hunt4 verdicts", "faces": summary},
        indent=2))
    print(f"\nwrote {out_dir / 'face_battery.json'}")


if __name__ == "__main__":
    main()
