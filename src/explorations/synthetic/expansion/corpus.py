"""Version-pinned text-corpus sampling for the expansion loop.

The text-corpus domain streams a fineweb sample via ``datasets`` (cached to
the volume), keeps documents long enough to carry within-document temporal
structure, and sentence-segments them with a deterministic regex splitter
(documented limitation: a heuristic splitter, not a parser — fine for span
labeling; the same splitter is pinned for every candidate that uses the
sample). The sampled snapshot (doc ids, urls, dumps, seed, filters, library
version) is written next to the cache so every calibration record can pin
exactly what it measured.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_ABBREV = r"(?<!\bMr\.)(?<!\bMrs\.)(?<!\bDr\.)(?<!\bMs\.)(?<!\bSt\.)(?<!\bvs\.)(?<!\be\.g\.)(?<!\bi\.e\.)"
_SENT_RE = re.compile(_ABBREV + r"(?<=[.!?…])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")


def split_sentences(text: str, min_chars: int = 8, max_chars: int = 600) -> list[str]:
    """Deterministic heuristic sentence splitter (pinned for the program)."""
    out = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        for s in _SENT_RE.split(para):
            s = s.strip()
            while len(s) > max_chars:  # unsplittable run-on: hard-wrap at a space
                cut = s.rfind(" ", 0, max_chars)
                cut = cut if cut > min_chars else max_chars
                out.append(s[:cut].strip())
                s = s[cut:].strip()
            if len(s) >= min_chars:
                out.append(s)
    return out


def sample_fineweb(cache_path: Path | str, *, n_docs: int = 400, seed: int = 0,
                   dataset: str = "HuggingFaceFW/fineweb", name: str = "sample-10BT",
                   split: str = "train", min_sents: int = 60, max_sents: int = 200,
                   shuffle_buffer: int = 10_000, log=None) -> dict:
    """Stream-shuffle-sample long documents; cache + pin; idempotent.

    Returns ``{"meta": {...}, "docs": [{"id", "url", "dump", "sentences"}]}``.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    import datasets as hfds

    ds = hfds.load_dataset(dataset, name=name, split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    docs, n_seen = [], 0
    for row in ds:
        n_seen += 1
        sents = split_sentences(row["text"])
        if min_sents <= len(sents) <= max_sents:
            docs.append({"id": row.get("id"), "url": row.get("url"),
                         "dump": row.get("dump"), "sentences": sents})
            if log and len(docs) % 50 == 0:
                log(f"[corpus] {len(docs)}/{n_docs} docs ({n_seen} scanned)")
            if len(docs) >= n_docs:
                break
    out = {
        "meta": {"dataset": dataset, "name": name, "split": split, "seed": seed,
                 "shuffle_buffer": shuffle_buffer, "n_docs": len(docs),
                 "n_scanned": n_seen, "min_sents": min_sents, "max_sents": max_sents,
                 "n_sentences": sum(len(d["sentences"]) for d in docs),
                 "datasets_version": hfds.__version__,
                 "splitter": "expansion.corpus.split_sentences v1"},
        "docs": docs,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(out))
    return out


def load_reasoning_traces(stage_a_dir: Path | str) -> dict:
    """The stored Ward Stage-A reasoning traces as the same docs schema."""
    stage_a_dir = Path(stage_a_dir)
    SL = json.loads((stage_a_dir / "sentence_labels.json").read_text())
    docs = [{"id": d["question_id"], "url": None, "dump": "ward_stage_a",
             "sentences": [s["sentence"] for s in d["sentences"]],
             "is_backtracking": [bool(s["is_backtracking"]) for s in d["sentences"]]}
            for d in SL]
    return {"meta": {"dataset": "results/c7_backtracking/stage_a/sentence_labels.json",
                     "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                     "n_docs": len(docs),
                     "n_sentences": sum(len(d["sentences"]) for d in docs)},
            "docs": docs}
