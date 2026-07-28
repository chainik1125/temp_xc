"""3-tokenizer grids for `retryesc_gen` — the card's "all three
tokenizers" rule, which binds the FULL run.

⚑ THIS IS THE PAYOFF OF A DEFECT I FIXED EARLIER TODAY. `evalage` had
to recover its text by DECODING the gpt2 id stream, because
`build_stream` wrote ids and threw the text away — lossless for gpt2's
byte-level BPE, but a property of that tokenizer rather than something a
scaffold may rely on, and it was the blocker that stalled the evalage
3-leg rule. I wired `save_transcripts` into `run_elicit` afterwards.

So this builder re-tokenizes **from the persisted raw text**, with no
decode-recovery step and no round-trip assumption anywhere. It is
strictly stronger than `evalage/screen_grids.py`, and shorter.

Receipts asserted on every leg:
  * doc count == 300 and turn count identical across legs
  * **event count == 2,809 on every tokenizer** (the corpus receipt)
  * the gpt2 leg reproduces the generation stream ARRAY-FOR-ARRAY
    (token_ids, doc_off, event_first, event_mask, probe_eligible)

Writes retryesc_gen/grids/elicit_retryesc_gen_screen_<tag>.npz
(+ grids_receipt.json).

Run: .venv/bin/python -m experiments.explorations.task_hunt.retryesc_gen.screen_grids
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.explorations.task_hunt.labels import elicit_lib as el
from experiments.explorations.task_hunt.labels.lib import doc_split
from experiments.explorations.task_hunt.replag.build_labels import MODELS

HERE = Path(__file__).resolve().parent
LABELS = HERE.parent / "labels"
GRIDS = HERE / "grids"
TAG = "retryesc_gen_v1"
STREAM = LABELS / f"elicit_{TAG}.npz"
TRANSCRIPTS = LABELS / f"elicit_{TAG}_transcripts.json"
TOK_TAG = {"gpt2": "gpt2", "gemma2_2b": "gemma2", "llama31_8b": "llama31"}

# corpus receipt, elicit_retryesc_gen_v1 (300 docs, 946,546 gpt2 tok)
EVENTS_EXPECTED = 2809
N_DOCS_EXPECTED = 300
SEED = 0


def build_leg(docs, tok):
    """Re-tokenize the raw transcripts. Same masking contract as
    `run_elicit.build_stream`: event turns are event-marked at their
    first token and fully probe-MASKED; assistant turns are eligible;
    other user turns are context only."""
    ids_l, first_l, mask_l, elig_l, off = [], [], [], [], [0]
    topics = []
    for d in docs:
        pi, pf, pm, pe = [], [], [], []
        for t in d["turns"]:
            enc = np.asarray(tok(t["text"], add_special_tokens=False)
                             ["input_ids"], dtype=np.int32)
            if enc.size == 0:
                continue
            f = np.zeros(enc.size, dtype=np.int8)
            m = np.zeros(enc.size, dtype=np.int8)
            e = np.zeros(enc.size, dtype=np.int8)
            if t["is_event"]:
                f[0] = 1
                m[:] = 1                      # event text never eligible
            elif t["role"] == "assistant":
                e[:] = 1                      # probe on assistant tokens
            pi.append(enc); pf.append(f); pm.append(m); pe.append(e)
        if not pi:
            continue
        ids_l.append(np.concatenate(pi))
        first_l.append(np.concatenate(pf))
        mask_l.append(np.concatenate(pm))
        elig_l.append(np.concatenate(pe))
        off.append(off[-1] + len(ids_l[-1]))
        topics.append(d["topic"])
    return (np.concatenate(ids_l), np.array(off, dtype=np.int64),
            np.concatenate(first_l), np.concatenate(mask_l),
            np.concatenate(elig_l), topics)


def main() -> None:
    from transformers import AutoTokenizer
    GRIDS.mkdir(exist_ok=True)

    docs = json.loads(TRANSCRIPTS.read_text())
    assert len(docs) == N_DOCS_EXPECTED, \
        f"{len(docs)} docs, receipt says {N_DOCS_EXPECTED}"
    n_turns = sum(len(d["turns"]) for d in docs)
    n_ev_txt = sum(t["is_event"] for d in docs for t in d["turns"])
    assert n_ev_txt == EVENTS_EXPECTED, \
        f"transcripts carry {n_ev_txt} events, receipt says {EVENTS_EXPECTED}"

    z = np.load(STREAM)
    tops = sorted({d["topic"] for d in docs})
    receipt = {"tag": TAG, "n_docs": len(docs), "n_turns": n_turns,
               "events_expected": EVENTS_EXPECTED, "n_tasks": len(tops),
               "source": "PERSISTED RAW TRANSCRIPTS — no decode-recovery, "
                         "no round-trip assumption (unlike evalage, which "
                         "had to decode the gpt2 id stream)",
               "per_tag": {}}

    for key, tag in TOK_TAG.items():
        tok = AutoTokenizer.from_pretrained(MODELS[key]["hf"])
        ids, off, first, mask, elig, topics = build_leg(docs, tok)
        n_docs = len(off) - 1
        assert n_docs == N_DOCS_EXPECTED, f"{tag}: {n_docs} docs"
        assert int(first.sum()) == EVENTS_EXPECTED, \
            f"{tag}: {int(first.sum())} events != {EVENTS_EXPECTED}"

        if key == "gpt2":
            # RECEIPT — the rebuilt gpt2 leg must reproduce the
            # generation stream array-for-array, not merely agree on
            # counts. This is what proves the re-tokenization contract
            # matches `build_stream` exactly.
            for nm, got, want in (("token_ids", ids, z["token_ids"]),
                                  ("doc_off", off, z["doc_off"]),
                                  ("event_first", first, z["event_first"]),
                                  ("event_mask", mask, z["event_mask"]),
                                  ("probe_eligible", elig,
                                   z["probe_eligible"])):
                assert np.array_equal(got, want), \
                    f"gpt2 leg differs from the generation stream in {nm}"
            print("[grids] gpt2 leg reproduces the generation stream "
                  "ARRAY-FOR-ARRAY (5/5 arrays identical)", flush=True)

        top_idx = np.array([tops.index(t) for t in topics], dtype=np.int8)
        out = GRIDS / f"elicit_{TAG}_screen_{tag}.npz"
        el.write_stream(out, token_ids=ids, doc_off=off, event_first=first,
                        event_mask=mask, probe_eligible=elig,
                        doc_split=doc_split(n_docs, seed=SEED),
                        extra={"is_assistant": elig, "doc_domain": top_idx})
        gaps = el.realised_gaps(first, off)
        receipt["per_tag"][tag] = {
            "n_tokens": int(ids.size), "events": int(first.sum()),
            "eligible": int(elig.sum()), "gap_median": gaps["median"],
            "gap_mean": gaps["mean"]}
        print(f"[grids] {tag}: {ids.size:,} tok, {int(first.sum())} events, "
              f"{int(elig.sum()):,} eligible, gap median "
              f"{gaps['median']:.0f}", flush=True)

    receipt["topics"] = tops
    (GRIDS / "grids_receipt.json").write_text(json.dumps(receipt, indent=1))
    print(f"[grids] {n_turns:,} turns re-tokenized on 3 legs; "
          f"{EVENTS_EXPECTED} events on every leg; receipt written")


if __name__ == "__main__":
    main()
