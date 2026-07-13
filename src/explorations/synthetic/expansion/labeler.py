"""Claude-judge labeler runner + validation (the stage-2 gate machinery).

A *labeler spec* is a plain dict, frozen in the candidate's prereg card:

    {"name": "...", "kind": "binary" | "ordinal", "n_values": 2,
     "judge_instruction": "<what the judge decides, per sentence>"}

``label_stream`` labels every span of every document with the bulk judge
(Haiku), chunked (~50 spans/call) with a few trailing spans of the previous
chunk shown as unlabeled context so chunk boundaries don't blind the judge.
Output format is a strict JSON integer array; a chunk that fails to parse is
retried once, and a document with any dead chunk is returned as ``None`` (we
never splice a broken sequence — coverage is reported instead).

Validation (both mandated by the loop before any measurement is trusted):

- ``validate_interjudge`` — an independent second judge (Sonnet) relabels a
  held-out sample; reports raw agreement, Cohen's κ, and the implied symmetric
  flip-rate noise floor ``ε̂ = (1 − √(1 − 2d)) / 2`` (d = disagreement rate),
  i.e. the ε at which two independent ε-noisy judges would disagree that often.
- ``crosscheck_binary`` — precision/recall/F1 of an independent cheap
  heuristic (keyword/lexicon) against the judge, the backtracking pattern.
"""

from __future__ import annotations

import json
import re

import numpy as np

_ARRAY_RE = re.compile(r"\[[^\[\]]*\]", re.S)

_FORMAT = ("Respond with ONLY a JSON array of {n} integers, one per numbered "
           "sentence, in order — no prose, no code fence. Each integer must be "
           "in [0, {vmax}]. The array length MUST be exactly {n}.")


def _parse_labels(text: str | None, n: int, vmax: int):
    if not text:
        return None
    m = _ARRAY_RE.search(text)
    if not m:
        return None
    try:
        vals = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(vals, list) or len(vals) != n:
        return None
    try:
        arr = np.array([int(v) for v in vals], dtype=np.int8)
    except (TypeError, ValueError):
        return None
    if arr.min() < 0 or arr.max() > vmax:
        return None
    return arr


def _chunk_jobs(docs, spec, chunk: int, ctx: int, tag: str):
    """One job per (doc, chunk); returns jobs + (doc_i, lo, hi) index."""
    vmax = spec["n_values"] - 1
    jobs, index = [], []
    for di, spans in enumerate(docs):
        for lo in range(0, len(spans), chunk):
            hi = min(lo + chunk, len(spans))
            parts = []
            if lo > 0:
                ctx_spans = spans[max(0, lo - ctx):lo]
                parts.append("Context (earlier sentences — do NOT label):\n"
                             + "\n".join(f"- {s}" for s in ctx_spans))
            parts.append("Label these sentences:\n" + "\n".join(
                f"{i + 1}. {s}" for i, s in enumerate(spans[lo:hi])))
            jobs.append({
                "system": spec["judge_instruction"] + "\n\n"
                          + _FORMAT.format(n=hi - lo, vmax=vmax),
                "user": "\n\n".join(parts),
                "max_tokens": 16 + 8 * (hi - lo),
                "tag": f"{tag}:{spec['name']}:d{di}:{lo}",
            })
            index.append((di, lo, hi))
    return jobs, index


def label_stream(judge, docs, spec, *, role: str = "bulk", chunk: int = 50,
                 ctx: int = 3, workers: int = 8, tag: str = "label"):
    """Label every span; returns (seqs, coverage).

    ``docs`` is a list of documents, each a list of span strings. ``seqs[i]``
    is an int8 array aligned to ``docs[i]``, or ``None`` if any of its chunks
    failed twice (that document is excluded from measurement, never spliced).
    """
    vmax = spec["n_values"] - 1
    jobs, index = _chunk_jobs(docs, spec, chunk, ctx, tag)
    texts = judge.call_many(role, jobs, workers=workers, tag=tag)

    parsed: dict[int, np.ndarray | None] = {}
    retry_ids = []
    for ji, text in enumerate(texts):
        arr = _parse_labels(text, index[ji][2] - index[ji][1], vmax)
        parsed[ji] = arr
        if arr is None:
            retry_ids.append(ji)
    if retry_ids:
        retex = judge.call_many(role, [jobs[j] for j in retry_ids],
                                workers=workers, tag=tag + ":retry")
        for ji, text in zip(retry_ids, retex):
            parsed[ji] = _parse_labels(text, index[ji][2] - index[ji][1], vmax)

    seqs: list = []
    n_dead_chunks = 0
    for di, spans in enumerate(docs):
        out = np.full(len(spans), -1, dtype=np.int8)
        dead = False
        for ji, (dj, lo, hi) in enumerate(index):
            if dj != di:
                continue
            if parsed[ji] is None:
                dead = True
                n_dead_chunks += 1
            else:
                out[lo:hi] = parsed[ji]
        seqs.append(None if dead else out)
    n_ok = sum(1 for s in seqs if s is not None)
    coverage = {"n_docs": len(docs), "n_docs_labeled": n_ok,
                "n_chunks": len(jobs), "n_dead_chunks": n_dead_chunks,
                "doc_coverage": n_ok / max(len(docs), 1)}
    return seqs, coverage


# ── validation ─────────────────────────────────────────────────────────────

def cohen_kappa(a: np.ndarray, b: np.ndarray, n_values: int) -> float:
    cm = np.zeros((n_values, n_values))
    np.add.at(cm, (a.astype(int), b.astype(int)), 1)
    cm /= cm.sum()
    po = float(np.trace(cm))
    pe = float((cm.sum(1) * cm.sum(0)).sum())
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def noise_floor_from_disagreement(d: float) -> float:
    """Symmetric flip rate ε̂ such that two independent ε-judges disagree at d."""
    return float((1 - np.sqrt(max(0.0, 1 - 2 * min(d, 0.5)))) / 2)


def validate_interjudge(judge, docs, primary_seqs, spec, *, sample_docs: int = 12,
                        seed: int = 0, role: str = "validate", chunk: int = 50,
                        ctx: int = 3, workers: int = 4, tag: str = "interjudge"):
    """Second-judge relabel of a doc sample; agreement + κ + noise floor."""
    rng = np.random.default_rng(seed)
    ok = [i for i, s in enumerate(primary_seqs) if s is not None]
    pick = sorted(rng.choice(ok, size=min(sample_docs, len(ok)), replace=False).tolist())
    sub_docs = [docs[i] for i in pick]
    second, cov = label_stream(judge, sub_docs, spec, role=role, chunk=chunk,
                               ctx=ctx, workers=workers, tag=tag)
    a_all, b_all = [], []
    for j, i in enumerate(pick):
        if second[j] is not None:
            a_all.append(primary_seqs[i])
            b_all.append(second[j])
    if not a_all:
        return {"error": "no overlapping labeled docs", "coverage": cov}
    a = np.concatenate(a_all)
    b = np.concatenate(b_all)
    d = float((a != b).mean())
    out = {"n_docs": len(a_all), "n_spans": int(a.size),
           "agreement": 1 - d, "disagreement": d,
           "kappa": cohen_kappa(a, b, spec["n_values"]),
           "noise_floor_eps": noise_floor_from_disagreement(d),
           "pos_rate_primary": float((a > 0).mean()),
           "pos_rate_second": float((b > 0).mean()),
           "coverage": cov, "sampled_doc_ids": pick}
    return out


def crosscheck_binary(judge_seqs, heuristic_seqs) -> dict:
    """Independent-heuristic cross-check: heuristic-vs-judge P/R/F1 (binary)."""
    pairs = [(j, h) for j, h in zip(judge_seqs, heuristic_seqs) if j is not None]
    j = np.concatenate([p[0] for p in pairs])
    h = np.concatenate([p[1] for p in pairs])
    tp = int(((h == 1) & (j == 1)).sum())
    prec = tp / max(int((h == 1).sum()), 1)
    rec = tp / max(int((j == 1).sum()), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {"n_spans": int(j.size), "judge_pos_rate": float(j.mean()),
            "heuristic_pos_rate": float(h.mean()),
            "precision": prec, "recall": rec, "f1": f1}
