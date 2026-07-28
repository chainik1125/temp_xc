"""Retroactively recover the `evalage` v1 raw transcripts from the
committed stream — closing my harness defect for the corpus that
already exists, not only for future ones.

`run_elicit.build_stream` wrote token ids and threw the text away
(`elicit_lib.save_transcripts` now fixes that going forward). The text
is nonetheless recoverable HERE because `screen_grids` proved the run
decomposition is lossless on this corpus: 22,412 runs re-encode
gpt2-token-identical and the rebuilt gpt2 leg is array-identical to the
stream. This script reuses that exact decomposition — importing
`doc_runs` from the frozen builder rather than re-deriving it — and
re-asserts the round trip per run, so the recovered text is receipted,
not assumed.

Kept as a SEPARATE script on purpose: `screen_grids.py` is frozen and
produced the committed grids, so it is not edited after the fact.

Writes labels/elicit_evalage_v1_transcripts.json (gitignored; HF is the
durable copy) + evalage/transcripts_receipt.json.

Run: .venv/bin/python -m experiments.explorations.task_hunt.evalage.dump_transcripts
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.explorations.task_hunt.evalage.screen_grids import doc_runs
from experiments.explorations.task_hunt.labels import evalage_lib as ev

HERE = Path(__file__).resolve().parent
LABELS = HERE.parent / "labels"
STREAM = LABELS / "elicit_evalage_v1.npz"
OUT = LABELS / "elicit_evalage_v1_transcripts.json"

# cls -> (role, is_event); build_stream's construction, inverted
CLS = {0: ("user", True), 1: ("assistant", False), 2: ("user", False)}


def main():
    from transformers import AutoTokenizer
    gpt2 = AutoTokenizer.from_pretrained("openai-community/gpt2")
    z = np.load(STREAM)
    ids, doc_off = z["token_ids"], z["doc_off"].astype(np.int64)
    first, mask, elig = z["event_first"], z["event_mask"], z["probe_eligible"]
    n_docs = len(doc_off) - 1
    plans = ev.evalage_plan(np.random.default_rng(ev.SEED), ev.N_DOCS)
    assert n_docs == len(plans) == ev.N_DOCS

    docs, n_runs, n_ev = [], 0, 0
    for d in range(n_docs):
        turns = []
        for c, a, b in doc_runs(first, mask, elig, doc_off[d], doc_off[d + 1]):
            seg = ids[a:b].tolist()
            txt = gpt2.decode(seg)
            assert gpt2(txt, add_special_tokens=False)["input_ids"] == seg, \
                f"ROUND-TRIP FAIL doc {d} run {a}:{b}"
            role, is_ev = CLS[c]
            turns.append({"role": role, "text": txt, "is_event": is_ev})
            n_runs += 1
            n_ev += int(is_ev)
        docs.append({"topic": plans[d]["topic"], "turns": turns})

    assert n_ev == int(first.sum()) == 1542, f"event count {n_ev}"
    blob = json.dumps(docs, ensure_ascii=False)
    OUT.write_text(blob)
    receipt = {
        "source_stream": STREAM.name,
        "recovered_by": "gpt2 decode of the frozen run decomposition "
                        "(screen_grids.doc_runs), re-asserted per run",
        "n_docs": n_docs, "n_turns": n_runs, "n_event_turns": n_ev,
        "roundtrip": "every run re-encodes token-identical",
        "chars": len(blob),
        "sha256": hashlib.sha256(blob.encode()).hexdigest(),
        "note": ("retroactive fix — run_elicit now persists transcripts "
                 "at generation time via elicit_lib.save_transcripts; "
                 "this recovers the corpus that predates that fix")}
    (HERE / "transcripts_receipt.json").write_text(json.dumps(receipt,
                                                             indent=1))
    print(f"[transcripts] {n_docs} docs, {n_runs} turns, {n_ev} event "
          f"turns, {len(blob):,} chars", flush=True)
    print(f"  sha256 {receipt['sha256'][:16]}…  -> {OUT.name}", flush=True)


if __name__ == "__main__":
    main()
