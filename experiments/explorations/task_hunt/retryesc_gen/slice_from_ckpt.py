"""Build a scoreable stream from a LIVE generation checkpoint.

This is the mechanism the hub's 14:51 "chain it, don't gate it" order
needs: the pilot's remaining job is elicitation quality, and that can be
tested on a PARTIAL corpus while generation is still running. So instead
of generating 20 docs, stopping, scoring, and then generating the rest,
we generate everything and score a slice off the checkpoint mid-flight.

`run_retryesc_gen` is pair-major -- every document advances one turn-pair
per step -- so a checkpoint at pair k holds ALL documents truncated to k
pairs. That is a better early signal for vocabulary leakage than 20
complete documents would be: it samples every task and every schedule,
just shallowly.

Truncation is made SAFE rather than assumed safe: a partial doc is cut
back to its last complete (user, assistant) pair and given its closing
environment turn, so the 2N+1 layout and the event alignment the dry run
asserts still hold exactly.

    .venv/bin/python -m experiments.explorations.task_hunt.retryesc_gen.slice_from_ckpt
    .venv/bin/python -m experiments.explorations.task_hunt.labels.build_retryesc_gen_premeasure --tag retryesc_gen_slice

⚠ A slice is EARLY EVIDENCE, not the verdict. It scores the
discriminating bands (unigram / doc-mean / position / floor_excess),
which are scale-free; mass bands are pilot-scaled and are expected to
grow. The full corpus is re-scored when generation drains.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..labels import elicit_lib as el
from ..labels import retryesc_gen_lib as rg
from ..labels import run_elicit as re_
from ..labels.lib import doc_split

TAG_IN = "retryesc_gen_v1"
TAG_OUT = "retryesc_gen_slice"
SEED = 0


def main() -> None:
    ck = el.ckpt_path(re_.HERE, TAG_IN)
    if not ck.exists():
        raise SystemExit(f"no checkpoint yet at {ck}")
    docs, at_pair = el.load_ckpt(ck)
    if docs is None:
        raise SystemExit("checkpoint unreadable")

    kept, dropped = [], 0
    for d in docs:
        pairs = d["plan"]["pairs"]
        turns = list(d["turns"])
        # cut back to the last COMPLETE (user, assistant) pair
        n_complete = len(turns) // 2
        if n_complete < 2:
            dropped += 1
            continue
        turns = turns[:2 * n_complete]
        # closing environment turn = outcome of the last complete pair,
        # which is what makes the layout 2N+1 and the events align
        last = pairs[n_complete - 1]
        turns.append(("user", last["env_text"], rg.is_event(last)))
        kept.append({"plan": {"topic": d["plan"]["topic"],
                              "pairs": pairs[:n_complete],
                              "meta": d["plan"]["meta"]},
                     "turns": turns})

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")

    def _tok(text, add_special_tokens=False):
        return tok(text, add_special_tokens=add_special_tokens)

    ids, doc_off, first, mask, elig, topics, texts = re_.build_stream(
        kept, _tok)

    # same invariants the dry run asserts — truncation must not break them
    for d in kept:
        n = len(d["plan"]["pairs"])
        assert len(d["turns"]) == 2 * n + 1, "slice broke the 2N+1 layout"
        for i, (role, _t, is_ev) in enumerate(d["turns"]):
            assert role == ("user" if i % 2 == 0 else "assistant")
            assert not (role == "assistant" and is_ev)

    out = re_.HERE / f"elicit_{TAG_OUT}.npz"
    el.write_stream(out, token_ids=ids, doc_off=doc_off, event_first=first,
                    event_mask=mask, probe_eligible=elig,
                    doc_split=doc_split(len(doc_off) - 1, seed=SEED))
    gaps = el.realised_gaps(first, doc_off)
    vc = el.vocabulary_control_check(ids, first, doc_off, topics)

    n_docs = len(doc_off) - 1
    print(f"SLICE off checkpoint @ pair {at_pair}: {n_docs} docs kept "
          f"({dropped} too short), {ids.size:,} tok, {int(first.sum())} "
          f"events, {int(elig.sum()):,} eligible")
    print(f"  gap median {gaps['median']:.0f} tok  |  "
          f"vocab-control worst leg cv "
          f"{vc.get('worst_leg_cv', float('nan')):.4f} "
          f"(bar {vc.get('cv_bar_proposed')}), stop={vc.get('stop')}")
    print(f"  wrote {out.name} — now run:\n"
          f"    .venv/bin/python -m experiments.explorations.task_hunt."
          f"labels.build_retryesc_gen_premeasure --tag {TAG_OUT}")

    (Path(__file__).resolve().parent / "results").mkdir(exist_ok=True)
    (Path(__file__).resolve().parent / "results" / "slice_receipt.json"
     ).write_text(json.dumps({
         "from_checkpoint_at_pair": at_pair, "n_docs": n_docs,
         "docs_dropped_too_short": dropped, "n_tokens": int(ids.size),
         "n_events": int(first.sum()), "gaps": gaps,
         "vocabulary_control": vc,
         "note": "EARLY EVIDENCE off a live checkpoint, not the verdict; "
                 "full corpus is re-scored at drain.",
     }, indent=1))


if __name__ == "__main__":
    main()
