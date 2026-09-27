"""sycgen SCREEN GRIDS — per-tokenizer token grids from the landed v1
corpus (SCREEN_CARD.md § 1; executor mac-d).

The generation stream (`labels/elicit_sycgen_v1.npz`) carries gpt2 ids
only — `build_stream` never persisted text. gpt2 BPE is byte-level and
lossless, so the corpus text is recovered EXACTLY by decoding, and the
recovery is receipted: every decoded turn-run is re-encoded with gpt2
and asserted token-identical to the original ids before any other
tokenizer sees it (hard stop otherwise).

Turn segmentation needs no side table: within a doc, contiguous runs of
(event_mask, probe_eligible) classes ARE the turns (challenge=mask,
assistant=eligible, user=neither; sycgen alternates user-class and
assistant turns strictly, so runs never merge).

Writes sycgen/grids/elicit_sycgen_screen_<tag>.npz with token_ids,
doc_off, event_first, event_mask, is_assistant, doc_split, doc_domain
(+ domain legend) — event count must equal the corpus's 1,118 in every
tag — plus grids_receipt.json.

Run: .venv/bin/python -m experiments.explorations.task_hunt.sycgen.screen_grids
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.explorations.task_hunt.labels.sycgen_domain_readout import (
    replay_doc_domains,
)
from experiments.explorations.task_hunt.replag.build_labels import MODELS

HERE = Path(__file__).resolve().parent
LABELS = HERE.parent / "labels"
GRIDS = HERE / "grids"
STREAM = LABELS / "elicit_sycgen_v1.npz"
TOK_TAG = {"gpt2": "gpt2", "gemma2_2b": "gemma2", "llama31_8b": "llama31"}


def doc_runs(first, mask, elig, lo, hi):
    """Turn-runs of one doc: (cls, start, end), cls in {chal, asst, user}."""
    cls = np.where(mask[lo:hi] == 1, 0, np.where(elig[lo:hi] == 1, 1, 2))
    runs, s = [], 0
    for i in range(1, hi - lo + 1):
        if i == hi - lo or cls[i] != cls[s]:
            runs.append((int(cls[s]), lo + s, lo + i))
            s = i
    for c, a, b in runs:
        if c == 0:
            assert first[a] == 1 and first[a + 1:b].sum() == 0
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
    doc_dom = replay_doc_domains(n_docs, 0)
    doms = sorted(set(doc_dom))
    dom_idx = np.array([doms.index(d) for d in doc_dom], dtype=np.int8)

    gpt2 = AutoTokenizer.from_pretrained("openai-community/gpt2")
    ev_want = int(first.sum())
    receipt = {"stream": str(STREAM.name), "n_docs": n_docs,
               "events_expected": ev_want, "domains": doms,
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
        if key == "gpt2":
            tok = gpt2
        else:
            tok = AutoTokenizer.from_pretrained(MODELS[key]["hf"])
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
            ids_l.append(np.concatenate(pi))
            first_l.append(np.concatenate(pf))
            mask_l.append(np.concatenate(pm))
            asst_l.append(np.concatenate(pa))
            off.append(off[-1] + len(ids_l[-1]))
        g_first = np.concatenate(first_l)
        ev_got = int(g_first.sum())
        assert ev_got == ev_want, f"{key}: events {ev_got} != {ev_want}"
        out = GRIDS / f"elicit_sycgen_screen_{tag}.npz"
        np.savez_compressed(
            out, token_ids=np.concatenate(ids_l),
            doc_off=np.array(off, dtype=np.int64), event_first=g_first,
            event_mask=np.concatenate(mask_l),
            is_assistant=np.concatenate(asst_l), doc_split=split,
            doc_domain=dom_idx)
        receipt["per_tag"][tag] = {
            "tokenizer": MODELS[key]["hf"], "n_tokens": int(off[-1]),
            "events": ev_got, "file": out.name}
        print(f"[grids] {tag}: {off[-1]:,} tokens, {ev_got} events "
              f"-> {out.name}", flush=True)

    (GRIDS / "grids_receipt.json").write_text(json.dumps(receipt, indent=1))
    print(f"-> {GRIDS}/grids_receipt.json", flush=True)


if __name__ == "__main__":
    main()
