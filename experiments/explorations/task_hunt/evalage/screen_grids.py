"""evalage SCREEN GRIDS — per-tokenizer token grids from the landed v1
corpus. **Transplant of mac-d's `sycgen/screen_grids.py`** (their LOG
note on `ad21f651d` invited it); design credit is theirs, the evalage
adaptation + the stronger gpt2 receipt below are mine.

Why a transplant is exactly right: both streams are written by the SAME
`run_elicit.build_stream`, so the class construction is identical —
event turn -> (event_first[0]=1, event_mask=1, never eligible);
assistant turn -> eligible; plain user turn -> neither. `run_evalage`
alternates user/assistant strictly, so contiguous runs of that class
triple ARE the turns and no side table is needed.

The stream carries gpt2 ids only (`build_stream` never persisted text —
my harness defect, disclosed; the fix is owed separately). gpt2 BPE is
byte-level and lossless, and every run here is exactly one turn's
`tok(text)` output, so decode->re-encode is the identity. That is
ASSERTED per run rather than assumed.

**Two receipts beyond mac-d's, because an error here silently moves
event positions and destroys the exact-labels property that is the
whole point of the harness:**

  1. the rebuilt gpt2 leg must be ARRAY-IDENTICAL to the original
     stream (ids, doc_off, and all three label arrays) — a strictly
     stronger check than the event count, and it proves the run
     decomposition itself is lossless on THIS corpus;
  2. every leg reports `realised_gaps`, and the gpt2 leg's median must
     equal the corpus receipt's 862.0. Other legs legitimately differ
     (different tokenizers, different token counts) and are recorded,
     not asserted.

Writes evalage/grids/elicit_evalage_screen_<tag>.npz with token_ids,
doc_off, event_first, event_mask, is_assistant, doc_split, doc_domain
(+ topic legend) — event count must equal 1,542 in every tag — plus
grids_receipt.json.

Run: .venv/bin/python -m experiments.explorations.task_hunt.evalage.screen_grids
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.explorations.task_hunt.labels import elicit_lib as el
from experiments.explorations.task_hunt.labels import evalage_lib as ev
from experiments.explorations.task_hunt.replag.build_labels import MODELS

HERE = Path(__file__).resolve().parent
LABELS = HERE.parent / "labels"
GRIDS = HERE / "grids"
STREAM = LABELS / "elicit_evalage_v1.npz"
TOK_TAG = {"gpt2": "gpt2", "gemma2_2b": "gemma2", "llama31_8b": "llama31"}

EVENTS_EXPECTED = 1542            # corpus receipt, elicit_evalage_v1
GPT2_GAP_MEDIAN = 862.0           # corpus receipt, elicit_evalage_v1


def doc_runs(first, mask, elig, lo, hi):
    """Turn-runs of one doc: (cls, start, end), cls in {cue, asst, user}.
    mac-d's routine, unchanged — including the cue-run assert, which is
    what catches a merge that would double-count an event."""
    cls = np.where(mask[lo:hi] == 1, 0, np.where(elig[lo:hi] == 1, 1, 2))
    runs, s = [], 0
    for i in range(1, hi - lo + 1):
        if i == hi - lo or cls[i] != cls[s]:
            runs.append((int(cls[s]), lo + s, lo + i))
            s = i
    for c, a, b in runs:
        if c == 0:
            assert first[a] == 1 and first[a + 1:b].sum() == 0, \
                f"cue run {a}:{b} carries {int(first[a:b].sum())} firsts"
    return runs


def main():
    from transformers import AutoTokenizer
    GRIDS.mkdir(exist_ok=True)
    z = np.load(STREAM)
    ids = z["token_ids"]
    doc_off = z["doc_off"].astype(np.int64)
    first, mask = z["event_first"], z["event_mask"]
    elig, split = z["probe_eligible"], z["doc_split"]
    n_docs = len(doc_off) - 1

    # topics replay exactly: run_evalage seeds rng with SEED and calls
    # evalage_plan FIRST, and evalage_plan draws topic before anything
    # else consults the rng (that ordering is the card's topic-independence
    # rule). Every doc survived build_stream, so the map is 1:1.
    plans = ev.evalage_plan(np.random.default_rng(ev.SEED), ev.N_DOCS)
    assert n_docs == ev.N_DOCS == len(plans), \
        f"doc count {n_docs} != planned {ev.N_DOCS}"
    doc_top = [p["topic"] for p in plans]
    tops = sorted(set(doc_top))
    top_idx = np.array([tops.index(t) for t in doc_top], dtype=np.int8)

    gpt2 = AutoTokenizer.from_pretrained(MODELS["gpt2"]["hf"])
    ev_want = int(first.sum())
    assert ev_want == EVENTS_EXPECTED, \
        f"stream has {ev_want} events, receipt says {EVENTS_EXPECTED}"
    receipt = {"stream": STREAM.name, "n_docs": n_docs,
               "events_expected": ev_want, "topics": tops,
               "transplanted_from": "sycgen/screen_grids.py (mac-d)",
               "roundtrip": "every decoded run re-encoded == original ids",
               "per_tag": {}}

    # decode all runs once, with the gpt2 round-trip receipt
    docs_txt = []
    n_runs = 0
    for d in range(n_docs):
        runs = doc_runs(first, mask, elig, doc_off[d], doc_off[d + 1])
        turns = []
        for c, a, b in runs:
            seg = ids[a:b].tolist()
            txt = gpt2.decode(seg)
            re_ids = gpt2(txt, add_special_tokens=False)["input_ids"]
            assert re_ids == seg, f"ROUND-TRIP FAIL doc {d} run {a}:{b}"
            turns.append((c, txt))
            n_runs += 1
        docs_txt.append(turns)
    receipt["runs_roundtripped"] = n_runs
    print(f"[grids] round-trip receipt: {n_runs} runs across {n_docs} "
          f"docs re-encode token-identical", flush=True)

    for key, tag in TOK_TAG.items():
        tok = gpt2 if key == "gpt2" else AutoTokenizer.from_pretrained(
            MODELS[key]["hf"])
        ids_l, first_l, mask_l, asst_l, off = [], [], [], [], [0]
        for turns in docs_txt:
            pi, pf, pm, pa = [], [], [], []
            for c, txt in turns:
                enc = np.asarray(tok(txt, add_special_tokens=False)
                                 ["input_ids"], dtype=np.int32)
                if enc.size == 0:
                    continue
                f = np.zeros(enc.size, dtype=np.int8)
                m = np.zeros(enc.size, dtype=np.int8)
                a = np.zeros(enc.size, dtype=np.int8)
                if c == 0:
                    f[0] = 1
                    m[:] = 1
                elif c == 1:
                    a[:] = 1
                pi.append(enc); pf.append(f); pm.append(m); pa.append(a)
            ids_l.append(np.concatenate(pi)); first_l.append(np.concatenate(pf))
            mask_l.append(np.concatenate(pm)); asst_l.append(np.concatenate(pa))
            off.append(off[-1] + len(ids_l[-1]))
        g_ids = np.concatenate(ids_l)
        g_off = np.array(off, dtype=np.int64)
        g_first, g_mask = np.concatenate(first_l), np.concatenate(mask_l)
        g_asst = np.concatenate(asst_l)
        ev_got = int(g_first.sum())
        assert ev_got == ev_want, f"{key}: events {ev_got} != {ev_want}"

        gaps = el.realised_gaps(g_first, g_off)
        if key == "gpt2":
            # RECEIPT 1 — the rebuilt gpt2 leg must reproduce the stream
            # array-for-array. Stronger than the event count: it proves
            # the run decomposition is lossless on this corpus.
            for nm, got, want in (("token_ids", g_ids, ids),
                                  ("doc_off", g_off, doc_off),
                                  ("event_first", g_first, first),
                                  ("event_mask", g_mask, mask),
                                  ("is_assistant", g_asst, elig)):
                assert np.array_equal(got, want), \
                    f"gpt2 leg differs from stream in {nm}"
            assert gaps["median"] == GPT2_GAP_MEDIAN, \
                f"gpt2 gap median {gaps['median']} != {GPT2_GAP_MEDIAN}"
            print("[grids] gpt2 leg is ARRAY-IDENTICAL to the stream "
                  f"(5/5 arrays) and gap median == {GPT2_GAP_MEDIAN}",
                  flush=True)

        out = GRIDS / f"elicit_evalage_screen_{tag}.npz"
        np.savez_compressed(
            out, token_ids=g_ids, doc_off=g_off, event_first=g_first,
            event_mask=g_mask, is_assistant=g_asst, doc_split=split,
            doc_domain=top_idx)
        receipt["per_tag"][tag] = {
            "tokenizer": MODELS[key]["hf"], "n_tokens": int(off[-1]),
            "events": ev_got, "realised_gaps": gaps, "file": out.name}
        print(f"[grids] {tag}: {off[-1]:,} tokens, {ev_got} events, "
              f"gap median {gaps['median']:.1f} -> {out.name}", flush=True)

    (GRIDS / "grids_receipt.json").write_text(json.dumps(receipt, indent=1))
    print(f"-> {GRIDS}/grids_receipt.json", flush=True)


if __name__ == "__main__":
    main()
