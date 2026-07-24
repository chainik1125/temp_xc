"""Task-hunt candidate 1 — λ̂ intensity labels on the Ward stream grid.

Builds the per-token backtracking-intensity targets for the Stage-1
screen (`briefings/task-hunt.md` candidate 1): the FITTED, FROZEN mirror
intensity (`synthetic/backtracking/results/backtracking_mirror_stats.json`
— logistic-AR, K=8 sentence lags, committed 2026-07) evaluated on each
trace's REAL Sonnet-labeled event history,

    lam_hat_i = sigma(a + c * (i / L_sent) + sum_l w_l * b_{i-l}),

for sentences i >= K (fully-observed history only; earlier sentences are
NaN / ineligible).  A history-only variant `lam_hist` drops the c*pos
term (the position-artifact disambiguator).  Every token of sentence i
carries the sentence's lam_hat; the token->sentence map is the
char-midpoint rule of `conversion_depth/build_ward_stream.py` (is_bt),
and the token->cache-coordinate map is that builder's BOS/round-trip
convention restricted to `map_ok`.

Label-side outputs (NO activations touched):
  /workspace/task_hunt_labels/lambda_intensity/
      lam_hat.npy   (N, 128) float32, NaN = ineligible
      lam_hist.npy  (N, 128) float32
      sent_idx.npy  (N, 128) int32, -1 = unmapped
      sent_pos.npy  (N, 128) float32, i / L_sent (NaN = unmapped)
  results/lambda_labels_stats.json — tokens-per-sentence + inter-event
      distances in TOKENS (the honest T-range basis for the card) +
      lam_hat distribution summaries.

Run:  .venv/bin/python -m experiments.explorations.task_hunt.lambda_intensity.build_labels
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
STAGE_A = REPO / "results" / "c7_backtracking" / "stage_a"
MIRROR_JSON = (REPO / "experiments" / "explorations" / "synthetic"
               / "backtracking" / "results" / "backtracking_mirror_stats.json")
STREAM_DIR = Path("/workspace/conv_depth_caches/ward_stream")
OUT_DIR = Path("/workspace/task_hunt_labels/lambda_intensity")
STATS_JSON = Path(__file__).resolve().parent / "results" / "lambda_labels_stats.json"

BASE_MODEL = "NousResearch/Meta-Llama-3.1-8B"
SEQ_LEN = 128


def sigma(x):
    return 1.0 / (1.0 + np.exp(-x))


def lam_for_trace(b: np.ndarray, a: float, c: float, w: np.ndarray):
    """Frozen-mirror intensity on the real event history; NaN for i < K."""
    K = w.size
    L = b.size
    lam = np.full(L, np.nan)
    lam_h = np.full(L, np.nan)
    pos = np.arange(L) / L
    for i in range(K, L):
        hist = float(np.dot(w, b[i - K:i][::-1]))   # lags 1..K
        lam[i] = sigma(a + c * pos[i] + hist)
        lam_h[i] = sigma(a + hist)
    return lam, lam_h


def main():
    from transformers import AutoTokenizer

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    traces = json.loads((STAGE_A / "traces.json").read_text())
    slabs = {s["question_id"]: s
             for s in json.loads((STAGE_A / "sentence_labels.json").read_text())}
    mirror = json.loads(MIRROR_JSON.read_text())
    a = float(mirror["intercept"])
    c = float(mirror["coef_position"])
    w = np.array(mirror["kernel_w"], dtype=np.float64)
    K = w.size

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    trace_idx = np.load(STREAM_DIR / "trace_idx.npy")
    win_start = np.load(STREAM_DIR / "win_start.npy")
    map_ok = np.load(STREAM_DIR / "map_ok.npy")
    stream = np.load(STREAM_DIR / "token_ids.npy")
    N = stream.shape[0]

    lam_c = np.full((N, SEQ_LEN), np.nan, dtype=np.float32)
    lam_h_c = np.full((N, SEQ_LEN), np.nan, dtype=np.float32)
    sent_c = np.full((N, SEQ_LEN), -1, dtype=np.int32)
    spos_c = np.full((N, SEQ_LEN), np.nan, dtype=np.float32)

    tok_per_sent, ev_gaps_tok = [], []
    lam_all = []
    per_trace = {}
    for ti in sorted(set(trace_idx.tolist())):
        tr = traces[ti]
        slab = slabs[tr["question_id"]]
        full = tr["full_response"]
        enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        mids = np.array([(o[0] + o[1]) / 2 for o in offs])

        sents = slab["sentences"]
        b = np.array([1 if s["is_backtracking"] else 0 for s in sents],
                     dtype=np.int8)
        lam, lam_h = lam_for_trace(b, a, c, w)
        lam_all.extend(lam[K:].tolist())

        # token -> sentence (char-midpoint rule, same as is_bt)
        sent_of_tok = np.full(len(offs), -1, dtype=np.int64)
        for si, s in enumerate(sents):
            m = (mids >= s["char_start"]) & (mids < s["char_end"])
            sent_of_tok[m] = si
            tok_per_sent.append(int(m.sum()))
        # inter-event distances in TOKENS (first token of consecutive
        # bt sentences), for the honest T-range stats
        firsts = [int(np.where(sent_of_tok == si)[0][0])
                  for si in np.where(b == 1)[0]
                  if np.any(sent_of_tok == si)]
        ev_gaps_tok.extend(np.diff(sorted(firsts)).tolist())
        per_trace[ti] = (lam, lam_h, sent_of_tok,
                         np.arange(len(sents)) / len(sents))

    for wi in range(N):
        lam, lam_h, sent_of_tok, spos = per_trace[int(trace_idx[wi])]
        s0 = int(win_start[wi])
        Torig = sent_of_tok.size
        for p in range(1, SEQ_LEN):
            o = s0 + p - 1
            if o >= Torig or not map_ok[wi, p]:
                continue
            si = sent_of_tok[o]
            if si < 0:
                continue
            sent_c[wi, p] = si
            spos_c[wi, p] = spos[si]
            lam_c[wi, p] = lam[si]
            lam_h_c[wi, p] = lam_h[si]

    np.save(OUT_DIR / "lam_hat.npy", lam_c)
    np.save(OUT_DIR / "lam_hist.npy", lam_h_c)
    np.save(OUT_DIR / "sent_idx.npy", sent_c)
    np.save(OUT_DIR / "sent_pos.npy", spos_c)

    lam_all = np.array([x for x in lam_all if np.isfinite(x)])
    tps = np.array(tok_per_sent)
    gaps = np.array(ev_gaps_tok)
    fin = np.isfinite(lam_c)
    stats = {
        "mirror_params": {"intercept": a, "coef_position": c,
                          "kernel_w": w.tolist(), "K": K},
        "tokens_per_sentence": {
            "mean": float(tps.mean()), "median": float(np.median(tps)),
            "p25": float(np.percentile(tps, 25)),
            "p75": float(np.percentile(tps, 75)), "n": int(tps.size)},
        "inter_bt_event_gap_tokens": {
            "mean": float(gaps.mean()), "median": float(np.median(gaps)),
            "p25": float(np.percentile(gaps, 25)),
            "p75": float(np.percentile(gaps, 75)), "n": int(gaps.size)},
        "lam_hat_sentence_level": {
            "mean": float(lam_all.mean()),
            "p10": float(np.percentile(lam_all, 10)),
            "p33": float(np.percentile(lam_all, 33.3)),
            "p50": float(np.percentile(lam_all, 50)),
            "p66": float(np.percentile(lam_all, 66.7)),
            "p90": float(np.percentile(lam_all, 90))},
        "cache_grid": {
            "n_labeled_tokens": int(fin.sum()),
            "labeled_frac_pos1plus": float(fin[:, 1:].mean()),
            "n_windows": int(N)},
    }
    STATS_JSON.parent.mkdir(exist_ok=True)
    STATS_JSON.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"-> {OUT_DIR} + {STATS_JSON}")


if __name__ == "__main__":
    main()
