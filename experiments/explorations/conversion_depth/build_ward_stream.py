"""Phase 3 (data) — rebuild the canonical § 5.2 Ward stream + depth labels.

Reconstructs the EXACT 4044 × 128 token stream of the paper's § 5.2
datasource (`llama_3_1_8b_base_l10_ward_nousmirror`) by verbatim
replication of `origin/final:purified/src/temp_bench/data/nlp/ward.py::
_load_corpus_ward`: tokenize each Stage A trace's `full_response` with
the SUBJECT (base) tokenizer at `add_special_tokens=False`, slice into
non-overlapping 128-token windows, decode each window and re-encode with
`add_special_tokens=True, truncation, padding='max_length'` (the
round-trip prepends BOS and drops the window's last token when the
re-encode is identity).

Then builds the label sidecars in CACHE coordinates (window w, position
p ∈ [0,128)):

- `map_ok`      (N, 128) bool  — cache token verified == original trace
                 token at the mapped position (BOS + round-trip check);
- `is_bt`       (N, 128) int8  — token inside a Sonnet-labeled
                 backtracking sentence (sentence_labels.json);
- `dist_next_kw`(N, 128) int32 — tokens until the next keyword event
                 ("wait"/"hmm" token inside the think region, the Ward
                 § 2.2 / sprint bt_freq convention), measured in the
                 ORIGINAL trace token space (future may cross window
                 boundaries); -1 if no later event in the trace;
- `dist_prev_kw`(N, 128) int32 — tokens since the previous keyword event
                 (for bt_freq's |ev-p|>25 negative rule); -1 if none;
- `dist_next_bts`/`dist_prev_bts` (N,128) int32 — same pair for the next/
                 previous Sonnet-labeled backtracking-SENTENCE first token;
- `in_think`    (N, 128) bool  — token inside the think region
                 (bt_freq restricts probe positions to it);
- `trace_idx`   (N,)   int32   — window → trace (the probe split group);
- `win_start`   (N,)   int32   — window's original token offset.

Also verifies tokenizer identity base-vs-distill on all 300 responses
(recorded delta → results/ward_stream_stats.json).

Run:  .venv/bin/python -m experiments.explorations.conversion_depth.build_ward_stream
Outputs under /workspace/conv_depth_caches/ward_stream/ (network volume,
NOT in git; stats JSON + a copy of meta go to results/).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

BASE_MODEL = "NousResearch/Meta-Llama-3.1-8B"
GEN_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
SEQ_LEN = 128
N_SEQS = 4044                 # datasource n_seqs; builder stops here
KEYWORDS = ("wait", "hmm")

REPO = Path(__file__).resolve().parents[3]
STAGE_A = REPO / "results" / "c7_backtracking" / "stage_a"
OUT_DIR = Path("/workspace/conv_depth_caches/ward_stream")
STATS_JSON = Path(__file__).resolve().parent / "results" / "ward_stream_stats.json"


def load_stage_a():
    traces = json.loads((STAGE_A / "traces.json").read_text())
    slabs = json.loads((STAGE_A / "sentence_labels.json").read_text())
    by_qid = {s["question_id"]: s for s in slabs}
    return traces, by_qid


def build_stream(tok, traces):
    """Verbatim _load_corpus_ward: texts list in trace order, then re-encode."""
    texts, prov = [], []          # prov: (trace_idx, win_start)
    for ti, t in enumerate(traces):
        full = t.get("full_response") or ""
        if not full:
            continue
        ids = tok(full, add_special_tokens=False)["input_ids"]
        for start in range(0, max(1, len(ids) - SEQ_LEN + 1), SEQ_LEN):
            window = ids[start:start + SEQ_LEN]
            if len(window) < SEQ_LEN:
                break
            texts.append(tok.decode(window))
            prov.append((ti, start))
            if len(texts) >= N_SEQS:
                break
        if len(texts) >= N_SEQS:
            break
    out = []
    for txt in texts[:N_SEQS]:
        enc = tok(txt, return_tensors="np", truncation=True,
                  max_length=SEQ_LEN, padding="max_length",
                  add_special_tokens=True)
        out.append(enc["input_ids"][0])
    return np.stack(out).astype(np.int64), prov


def token_labels_for_trace(tok, trace, slab):
    """Per-ORIGINAL-token labels over the trace's full_response.

    Returns ids (list), is_bt (T,), next_kw (T,), next_bts (T,) where the
    dist arrays hold the ABSOLUTE original-token index of the next event
    at-or-after each position (len(ids) sentinel = none).
    """
    full = trace["full_response"]
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    T = len(ids)
    think_end = full.find("</think>")
    if think_end < 0:
        think_end = len(full)

    # sentence spans -> per-token is_bt
    is_bt = np.zeros(T, dtype=np.int8)
    bts_starts = []               # first-token index of each bt sentence
    spans = [(s["char_start"], s["char_end"], s["is_backtracking"])
             for s in slab["sentences"]]
    si = 0
    tok_of_char = np.zeros(T, dtype=np.int64)
    for i, (a, b) in enumerate(offs):
        tok_of_char[i] = a
    for (a, b, bt) in spans:
        if not bt:
            continue
        # tokens whose char midpoint falls in [a, b)
        mids = np.array([(o[0] + o[1]) / 2 for o in offs])
        m = (mids >= a) & (mids < b)
        is_bt[m] = 1
        w = np.where(m)[0]
        if len(w):
            bts_starts.append(int(w[0]))

    # keyword events: token whose text starts a wait/hmm word, in think
    ev = []
    for i, (a, b) in enumerate(offs):
        if a >= think_end:
            break
        w = full[a:b].strip().lower()
        if any(w.startswith(k) for k in KEYWORDS):
            # word-boundary guard: previous char is not alphanumeric
            if a == 0 or not full[a - 1].isalnum():
                ev.append(i)

    def next_at_or_after(starts):
        nxt = np.full(T, T, dtype=np.int64)
        cur = T
        ss = sorted(starts)
        j = len(ss) - 1
        for p in range(T - 1, -1, -1):
            while j >= 0 and ss[j] >= p:
                cur = ss[j]
                j -= 1
            nxt[p] = cur
        return nxt

    def prev_at_or_before(starts):
        prv = np.full(T, -1, dtype=np.int64)
        cur = -1
        ss = sorted(starts)
        j = 0
        for p in range(T):
            while j < len(ss) and ss[j] <= p:
                cur = ss[j]
                j += 1
            prv[p] = cur
        return prv

    in_think = np.array([o[0] < think_end for o in offs], dtype=bool)
    return (ids, is_bt, next_at_or_after(ev), next_at_or_after(bts_starts),
            ev, prev_at_or_before(ev), prev_at_or_before(bts_starts),
            in_think)


def main():
    from transformers import AutoTokenizer
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    traces, by_qid = load_stage_a()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    # NOTE (recorded in stats): AutoTokenizer resolves the distill repo to
    # the SLOW LlamaTokenizer under transformers 5.7, which mangles
    # whitespace ("me reconsider" -> "mere"+"consider") and fakes a huge
    # encode delta. The fast backend (tokenizer.json) encodes IDENTICALLY
    # to the base tokenizer; the id->token maps agree exactly. Force fast.
    from transformers import PreTrainedTokenizerFast
    tok_gen = PreTrainedTokenizerFast.from_pretrained(GEN_MODEL)
    # canonical ward.py cache_activations line: pad with eos when missing
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok_gen.pad_token is None:
        tok_gen.pad_token = tok_gen.eos_token

    # ── tokenizer-identity check (base vs generator) ──────────────────
    from difflib import SequenceMatcher
    n_diff, n_tok_total, diff_examples = 0, 0, []
    match_ratios = []
    for t in traces[:300]:
        a = tok(t["full_response"], add_special_tokens=False)["input_ids"]
        b = tok_gen(t["full_response"], add_special_tokens=False)["input_ids"]
        n_tok_total += len(a)
        if a != b:
            n_diff += 1
            match_ratios.append(
                SequenceMatcher(None, a, b, autojunk=False).ratio())
            if len(diff_examples) < 3:
                k = next(i for i in range(min(len(a), len(b)))
                         if a[i] != b[i])
                diff_examples.append({
                    "qid": t["question_id"], "first_mismatch_at": k,
                    "base_toks": [tok.decode([x]) for x in a[k - 1:k + 3]],
                    "gen_toks": [tok_gen.decode([x]) for x in b[k - 1:k + 3]],
                })
        else:
            match_ratios.append(1.0)
    tok_delta = {"n_responses_differing": n_diff, "of": len(traces),
                 "seq_match_ratio_min": float(np.min(match_ratios)),
                 "seq_match_ratio_mean": float(np.mean(match_ratios)),
                 "examples": diff_examples,
                 "base_bos": tok.bos_token, "gen_bos": tok_gen.bos_token,
                 "base_bos_id": tok.bos_token_id, "gen_bos_id": tok_gen.bos_token_id,
                 "vocab_base": len(tok), "vocab_gen": len(tok_gen)}
    print("[tokenizer] delta:", json.dumps(tok_delta)[:700], flush=True)

    # ── the canonical stream ──────────────────────────────────────────
    stream, prov = build_stream(tok, traces)
    print(f"[stream] {stream.shape} windows; first trace idx {prov[0]}, "
          f"last {prov[-1]}", flush=True)
    assert stream.shape == (N_SEQS, SEQ_LEN), stream.shape

    # ── labels in cache coordinates ───────────────────────────────────
    N = stream.shape[0]
    map_ok = np.zeros((N, SEQ_LEN), dtype=bool)
    is_bt_c = np.zeros((N, SEQ_LEN), dtype=np.int8)
    dist_kw = np.full((N, SEQ_LEN), -1, dtype=np.int32)
    dist_bts = np.full((N, SEQ_LEN), -1, dtype=np.int32)
    pdist_kw = np.full((N, SEQ_LEN), -1, dtype=np.int32)
    pdist_bts = np.full((N, SEQ_LEN), -1, dtype=np.int32)
    in_think_c = np.zeros((N, SEQ_LEN), dtype=bool)
    trace_idx = np.array([p[0] for p in prov], dtype=np.int32)
    win_start = np.array([p[1] for p in prov], dtype=np.int32)

    per_trace = {}
    ev_counts, bts_counts = [], []
    for ti in sorted(set(trace_idx.tolist())):
        tr = traces[ti]
        slab = by_qid[tr["question_id"]]
        per_trace[ti] = token_labels_for_trace(tok, tr, slab)
        ev_counts.append(len(per_trace[ti][4]))
        bts_counts.append(int(per_trace[ti][1].sum() > 0))

    bos_id = tok.bos_token_id
    n_mismatch_tok = 0
    for w in range(N):
        (ids_orig, is_bt_o, nxt_kw, nxt_bts, _, prv_kw, prv_bts,
         in_think_o) = per_trace[int(trace_idx[w])]
        s = int(win_start[w])
        Torig = len(ids_orig)
        # cache row = [BOS, orig[s .. s+126]] when round-trip is identity
        if stream[w, 0] != bos_id:
            continue
        for p in range(1, SEQ_LEN):
            o = s + p - 1
            if o >= Torig:
                break
            if int(stream[w, p]) == int(ids_orig[o]):
                map_ok[w, p] = True
                is_bt_c[w, p] = is_bt_o[o]
                in_think_c[w, p] = in_think_o[o]
                dist_kw[w, p] = (nxt_kw[o] - o) if nxt_kw[o] < Torig else -1
                dist_bts[w, p] = (nxt_bts[o] - o) if nxt_bts[o] < Torig else -1
                pdist_kw[w, p] = (o - prv_kw[o]) if prv_kw[o] >= 0 else -1
                pdist_bts[w, p] = (o - prv_bts[o]) if prv_bts[o] >= 0 else -1
            else:
                n_mismatch_tok += 1

    stats = {
        "stream_shape": list(stream.shape),
        "n_traces_used": len(per_trace),
        "map_ok_rate_pos1plus": float(map_ok[:, 1:].mean()),
        "n_roundtrip_mismatch_tokens": int(n_mismatch_tok),
        "bos_row_rate": float((stream[:, 0] == bos_id).mean()),
        "keyword_events_total": int(np.sum(ev_counts)),
        "traces_with_bt_sentence": int(np.sum(bts_counts)),
        "is_bt_token_rate": float(is_bt_c[map_ok].mean()),
        "tokenizer_delta": tok_delta,
    }
    np.save(OUT_DIR / "token_ids.npy", stream)
    np.save(OUT_DIR / "map_ok.npy", map_ok)
    np.save(OUT_DIR / "is_bt.npy", is_bt_c)
    np.save(OUT_DIR / "dist_next_kw.npy", dist_kw)
    np.save(OUT_DIR / "dist_next_btsent.npy", dist_bts)
    np.save(OUT_DIR / "dist_prev_kw.npy", pdist_kw)
    np.save(OUT_DIR / "dist_prev_btsent.npy", pdist_bts)
    np.save(OUT_DIR / "in_think.npy", in_think_c)
    np.save(OUT_DIR / "trace_idx.npy", trace_idx)
    np.save(OUT_DIR / "win_start.npy", win_start)
    STATS_JSON.parent.mkdir(exist_ok=True)
    STATS_JSON.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2)[:1200])
    print(f"-> {OUT_DIR} + {STATS_JSON}")


if __name__ == "__main__":
    main()
