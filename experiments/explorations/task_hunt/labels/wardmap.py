"""Shared Ward-stream broadcast for the task-hunt label builders.

Rebuilds the canonical 4044 × 128 Ward stream (verbatim
``conversion_depth.build_ward_stream.build_stream``) and maps arbitrary
per-ORIGINAL-token payload arrays into cache coordinates with the same
BOS + round-trip identity check. Every builder that targets the Ward
grid goes through here so the alignment logic exists once.

``traces.json`` is a read-only Stage-A port not committed on this branch;
if absent it is re-ported at build time via ``git show`` per the
ATTRIBUTION.md reproduction rule (and left uncommitted).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from experiments.explorations.conversion_depth.build_ward_stream import (
    BASE_MODEL, SEQ_LEN, build_stream)

ROOT = Path(__file__).resolve().parents[3]
STAGE_A = ROOT / "results" / "c7_backtracking" / "stage_a"


def ensure_traces() -> list[dict]:
    p = STAGE_A / "traces.json"
    if not p.exists():
        print("[port] traces.json absent — re-porting per ATTRIBUTION.md")
        blob = subprocess.run(
            ["git", "show",
             "origin/aniket-ward-stage-b:results/ward_backtracking/traces.json"],
            cwd=ROOT, capture_output=True, check=True).stdout
        p.write_bytes(blob)
    return json.loads(p.read_text())


def load_inputs():
    """(tokenizer, traces, {question_id: sentence-label dict})."""
    from transformers import AutoTokenizer
    traces = ensure_traces()
    slabs = json.loads((STAGE_A / "sentence_labels.json").read_text())
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, traces, {s["question_id"]: s for s in slabs}


def tokenize_trace(tok, trace):
    """(ids, offsets) of the trace's full_response, no special tokens."""
    enc = tok(trace["full_response"], add_special_tokens=False,
              return_offsets_mapping=True)
    return enc["input_ids"], enc["offset_mapping"]


def broadcast(tok, traces, payload_fn):
    """Map per-original-token payloads into Ward cache coordinates.

    ``payload_fn(trace_idx, ids, offsets) -> dict[name -> (T_orig,) array]``
    is called once per trace used by the stream. Returns
    (arrays dict with shape (N, SEQ_LEN) per payload — float payloads
    fill with NaN, integer with -1 — plus ``valid`` bool from the
    round-trip identity check, ``trace_idx``, ``win_start``,
    n_mismatch)."""
    stream, prov = build_stream(tok, traces)
    N = stream.shape[0]
    trace_idx = np.array([p[0] for p in prov], dtype=np.int32)
    win_start = np.array([p[1] for p in prov], dtype=np.int32)

    per_trace = {}
    for ti in sorted(set(trace_idx.tolist())):
        ids, offs = tokenize_trace(tok, traces[ti])
        per_trace[ti] = (np.asarray(ids), payload_fn(ti, ids, offs))

    names = list(next(iter(per_trace.values()))[1].keys())
    out = {}
    for name in names:
        proto = next(iter(per_trace.values()))[1][name]
        if np.issubdtype(proto.dtype, np.floating):
            out[name] = np.full((N, SEQ_LEN), np.nan, dtype=proto.dtype)
        else:
            out[name] = np.full((N, SEQ_LEN), -1, dtype=proto.dtype)
    valid = np.zeros((N, SEQ_LEN), dtype=bool)

    bos_id = tok.bos_token_id
    n_mismatch = 0
    for w in range(N):
        ids_o, payload = per_trace[int(trace_idx[w])]
        if stream[w, 0] != bos_id:
            continue
        s = int(win_start[w])
        hi = min(SEQ_LEN - 1, len(ids_o) - s)
        for p in range(1, hi + 1):
            o = s + p - 1
            if int(stream[w, p]) != int(ids_o[o]):
                n_mismatch += 1
                continue
            valid[w, p] = True
            for name in names:
                out[name][w, p] = payload[name][o]
    return out, valid, trace_idx, win_start, n_mismatch
