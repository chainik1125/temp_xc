"""WildChat 400 → 2,000-conversation scale-up pull (corpus-scaleup item 2).

    HF_TOKEN=... .venv/bin/python -m \
        experiments.explorations.task_hunt.labels.pull_refmark2k

**Same recipe as the shipped 400-conversation corpus, only the stream
prefix moves.** Same dataset and PINNED revision (``allenai/WildChat-1M``
@ ``7d6490e4…``, ODC-By 1.0), same filters (English, >= 8 assistant
turns, 2,000–24,000 rendered chars under the frozen
``dialevel_lib.render_dialogue``), same seed; ``N_STREAM`` 40,000 →
250,000 and ``N_CONVS`` 400 → 2,000.

Unlike fineweb, this is a *subsample of a pool*, so it is NOT a superset
of the shipped 400: a larger pool redraws. Two receipts replace the
superset claim, and both were missing from the 400-conv build:

- **the funnel**, counted as the stream is read — how many rows were
  seen, English, long enough, and inside the char window — so the
  pre-subsample POOL SIZE is on the record (the shipped build reported
  only its 400);
- **the overlap** with the shipped 400-conversation corpus (how many of
  those conversations reappear in the scaled pool and in the scaled
  sample), which is what a reader needs to know before treating scaled
  and shipped numbers as independent evidence.

Artifact: ``refmark2k_corpus.json.gz`` here (same schema as
``refmark_corpus.json.gz``: ``{"meta", "convs"}`` with convs as
(role, content) pair lists). The shipped corpus is never touched.
Idempotent: an existing artifact short-circuits the pull.
"""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import numpy as np

from . import dialevel_lib as dl
from . import refmark_lib as rl

HERE = Path(__file__).resolve().parent
SHIPPED = HERE / "refmark_corpus.json.gz"
OUT = HERE / "refmark2k_corpus.json.gz"
RECEIPT = HERE / "refmark2k_corpus_receipt.json"

SEED = 0
DATASET = "allenai/WildChat-1M"
REVISION = "7d6490e462285cf85d91eabea0f9a954fbddcd1f"
N_STREAM = 250_000
N_CONVS = 2_000
MIN_A_TURNS = 8
MIN_CHARS, MAX_CHARS = 2_000, 24_000


def _key(msgs) -> str:
    """Identity of a conversation for the overlap receipt."""
    return json.dumps(msgs, sort_keys=True)


def load() -> tuple[list, dict]:
    payload = json.loads(gzip.decompress(OUT.read_bytes()))
    return payload["convs"], payload["meta"]


def overlap_receipt(pool, sample) -> dict:
    shipped = json.loads(gzip.decompress(SHIPPED.read_bytes()))["convs"]
    ship_keys = {_key([[r, c] for r, c in m]) for m in shipped}
    pool_keys = {_key([[r, c] for r, c in m]) for m in pool}
    samp_keys = {_key([[r, c] for r, c in m]) for m in sample}
    return {"shipped_n_convs": len(shipped),
            "shipped_in_scaled_pool": len(ship_keys & pool_keys),
            "shipped_in_scaled_sample": len(ship_keys & samp_keys),
            "note": "a subsample of a larger pool redraws — the scaled "
                    "corpus is NOT a superset of the shipped 400"}


def main() -> None:
    if OUT.exists():
        convs, meta = load()
        print(f"[pull] artifact present: {OUT.name} "
              f"({meta['n_convs']} convs)")
    else:
        import datasets
        t0 = time.time()
        ds = datasets.load_dataset(DATASET, split="train",
                                   revision=REVISION, streaming=True)
        n_seen = n_english = n_turns = 0
        keep = []
        for i, ex in enumerate(ds):
            if i >= N_STREAM:
                break
            n_seen += 1
            if n_seen % 25_000 == 0:
                print(f"[pull] {n_seen:,} seen, {n_english:,} English, "
                      f"{n_turns:,} >= {MIN_A_TURNS} assistant turns, "
                      f"{len(keep):,} kept ({time.time() - t0:.0f}s)",
                      flush=True)
            if ex.get("language") != "English":
                continue
            n_english += 1
            msgs = [(m["role"], m["content"]) for m in ex["conversation"]
                    if m.get("content")]
            if sum(1 for r, _ in msgs if r == "assistant") < MIN_A_TURNS:
                continue
            n_turns += 1
            text, _ = dl.render_dialogue([c for _, c in msgs])
            if not (MIN_CHARS <= len(text) <= MAX_CHARS):
                continue
            keep.append(msgs)

        pool = keep
        rng = np.random.default_rng(SEED)
        if len(pool) > N_CONVS:
            idx = np.sort(rng.choice(len(pool), size=N_CONVS, replace=False))
            convs = [pool[i] for i in idx]
        else:
            convs = pool
        meta = {
            "dataset": DATASET, "revision": REVISION,
            "license": "ODC-By 1.0", "split": "train",
            "stream_prefix": N_STREAM,
            "filter": {"language": "English",
                       "min_assistant_turns": MIN_A_TURNS,
                       "min_chars": MIN_CHARS, "max_chars": MAX_CHARS},
            "seed": SEED, "n_convs": len(convs),
            "funnel": {"n_seen": n_seen, "n_english": n_english,
                       "n_min_assistant_turns": n_turns,
                       "n_pool_after_char_filter": len(pool)},
            "pool_size_pre_subsample": len(pool),
            "frozen_list": {"repo": rl.SOURCE_REPO,
                            "commit": rl.SOURCE_COMMIT,
                            "symbol": rl.SOURCE_SYMBOL},
        }
        OUT.write_bytes(gzip.compress(json.dumps(
            {"meta": meta, "convs": convs}).encode()))
        print(f"[pull] wrote {OUT.name} "
              f"({OUT.stat().st_size / 1e6:.1f} MB gz) in "
              f"{time.time() - t0:.0f}s", flush=True)
        RECEIPT.write_text(json.dumps(
            {"meta": meta, "overlap": overlap_receipt(pool, convs)}, indent=1))

    print(json.dumps(meta, indent=1))
    print(f"-> {RECEIPT}")


if __name__ == "__main__":
    main()
