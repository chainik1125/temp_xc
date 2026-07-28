"""Rebuild the `retryesc_gen` corpus receipt from COMPLETED artifacts.

⚠ WHY THIS EXISTS. The 300-doc generation ran to completion (all 46
pairs, npz + transcripts both written) and then **crashed constructing
the receipt**: I renamed `TARGET_GAP_MEDIAN` -> `GAP_MEDIAN_SUPERSEDED`
in `retryesc_gen_lib` during the § 2.2b correction and did not grep for
usages, so `run_elicit.main()` still referenced the old name.

My error, and a cheap one only by luck of ordering -- `write_stream`
and `save_transcripts` both run BEFORE the receipt, so no generated
token was lost and no money was re-spent. Had the receipt been written
first, the crash would have cost the whole run.

This script re-derives the receipt from the artifacts on disk rather
than re-generating anything. It is deliberately NOT a general tool: it
asserts the npz it reads is the completed one.

    .venv/bin/python -m experiments.explorations.task_hunt.retryesc_gen.rebuild_receipt
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..labels import elicit_lib as el
from ..labels import retryesc_gen_lib as rg
from ..labels import run_elicit as re_

TAG = "retryesc_gen_v1"
MODEL = "claude-haiku-4-5-20251001"
SEED, TEMPERATURE, TOP_P, N_DOCS = 0, 0.8, 0.95, 300


def main() -> None:
    npz = re_.HERE / f"elicit_{TAG}.npz"
    z = np.load(npz)
    ids, doc_off = z["token_ids"], z["doc_off"].astype(np.int64)
    first = z["event_first"]
    n_docs = len(doc_off) - 1
    assert n_docs == N_DOCS, f"expected {N_DOCS} docs, got {n_docs}"

    docs = json.loads(
        (re_.HERE / f"elicit_{TAG}_transcripts.json").read_text())
    # transcripts are the authority for text; topics ride along per doc
    topics = [d.get("topic") for d in docs] if isinstance(docs, list) else None
    if topics is None or any(t is None for t in topics):
        ck, _ = el.load_ckpt(el.ckpt_path(re_.HERE, TAG))
        topics = [d["plan"]["topic"] for d in ck]
    assert len(topics) == n_docs

    gaps = el.realised_gaps(first, doc_off)
    vc = el.vocabulary_control_check(ids, first, doc_off, topics)
    first_text = ""
    if isinstance(docs, list) and docs:
        t = docs[0].get("turns") or []
        first_text = " ".join(x[1] if isinstance(x, list) else x.get("text", "")
                              for x in t)[:2000]

    rec = el.corpus_receipt(
        scaffold_name="retryesc_gen", model_id=MODEL,
        model_sha="api-versioned (no weight sha)", seed=SEED,
        temperature=TEMPERATURE, top_p=TOP_P, n_docs=n_docs,
        n_tokens=int(ids.size), gaps=gaps, vocab_check=vc,
        first_doc_text=first_text,
        backend_note=("Anthropic API (MATS key; pin = model-id + API "
                      "version, NOT a weight sha => reproducible-in-"
                      "expectation, not bit-exact)"),
        extra={"scaffold_constants": {
            "p_repeat": rg.P_REPEAT, "p_fail_new": rg.P_FAIL_NEW,
            "pairs_per_doc": [rg.PAIRS_MIN, rg.PAIRS_MAX],
            "len_band": [rg.LEN_LO, rg.LEN_HI],
            "n_strategies": len(rg.STRATEGIES), "n_tasks": len(rg.TASKS),
            "n_failure_texts": len(rg.FAILURE_TEXTS),
            "n_success_texts": len(rg.SUCCESS_TEXTS),
            "floor_excess_band": list(rg.FLOOR_EXCESS_BAND),
            "construction_rule": (
                "environment failure text drawn INDEPENDENTLY of "
                "repeat-status at plan time (card § 3)"),
            "vocabulary_rule": (
                "outcome schedule drawn BEFORE the task and never "
                "consulting it (card § 4)")},
            "generation_tokenizer": "gpt2",
            "card": "retryesc_gen/GENERATION_CARD.md freeze 3f6ba0d3d",
            "receipt_rebuilt": (
                "generation completed; receipt crashed on a stale "
                "constant name (TARGET_GAP_MEDIAN) and was rebuilt from "
                "the npz + transcripts. No token regenerated, no spend "
                "repeated.")})
    el.save_receipt(re_.HERE / f"elicit_{TAG}_receipt.json", rec)

    print(f"{n_docs} docs, {ids.size:,} tok, {int(first.sum())} events")
    print(f"gap median {gaps['median']:.0f} (mean {gaps['mean']:.0f}, "
          f"p10 {gaps['p10']:.0f}, p90 {gaps['p90']:.0f})")
    print(f"vocab-control worst leg cv {vc.get('worst_leg_cv'):.4f} "
          f"(bar {vc.get('cv_bar_proposed')}), stop={vc.get('stop')}")
    print(f"wrote elicit_{TAG}_receipt.json")


if __name__ == "__main__":
    main()
