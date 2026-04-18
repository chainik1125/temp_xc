"""Data-leakage audit for Phase 5.

Question: does SAEBench probe-test text appear in the SAE training
activation cache? The cache is Gemma-2-2B-IT forward passes over a
FineWeb sample (`data/cached_activations/gemma-2-2b-it/fineweb/`).
Probe text comes from the 8 SAEBench sparse-probing datasets.

Threshold (per Han's instruction): >1% substring-match rate on 1000
random probe-test prompts = severe leakage, retrain encoders on a
disjoint corpus. <1% = document and proceed.

Audit protocol:

    1. Decode the cached FineWeb token_ids back to strings with the
       Gemma tokenizer. 24000 sequences x 128 tokens.
    2. Load the first chunk of each SAEBench dataset from HF, sample
       up to 1000 probe prompts total (split across the 8 datasets).
    3. For each probe prompt, pick a 40-character "signature" from
       its middle (distinctive enough to avoid accidental matches,
       short enough to fit in the FineWeb per-sequence 128-token
       window ~ 500 chars).
    4. Report: fraction of probe prompts whose signature appears as
       a substring anywhere in the decoded FineWeb corpus.

A conservative bound: FineWeb is filtered Common Crawl (15T tokens
total source); our sample is 24k x 128 = 3M tokens. Even if probe
corpora have source overlap with Common Crawl (likely for ag_news,
bias_in_bios, amazon_reviews; less so for europarl, github-code),
the probability any specific probe prompt shows up verbatim in a
3M-token sample is vanishingly small.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("TQDM_DISABLE", "1")

# Quiet the HF datasets warning about trust_remote_code etc.
os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")


SAEBENCH_DATASETS = [
    # name, text_key, optional config
    ("LabHC/bias_in_bios", "hard_text", None),  # root dataset; class_set1/2/3 are splits
    ("canrager/amazon_reviews_mcauley_1and5", "text", None),
    ("fancyzhx/ag_news", "text", None),
    ("Helsinki-NLP/europarl", "translation", "en-fr"),  # config required
    ("codeparrot/github-code-clean", "code", "all-all"),  # script-based parent was deprecated
    ("imdb", "text", None),
    ("Anthropic/hh-rlhf", "chosen", None),  # chat text, plausible Common Crawl overlap
    ("wikitext", "text", "wikitext-103-raw-v1"),  # explicitly Wikipedia, max overlap risk
]


def decode_fineweb_cache(limit_sequences: int = 24000) -> list[str]:
    """Decode the cached FineWeb token_ids to raw strings.

    Returns one string per cached sequence.
    """
    import transformers

    cache_path = Path(
        "/workspace/temp_xc/data/cached_activations/gemma-2-2b-it/"
        "fineweb/token_ids.npy"
    )
    print(f"Loading tokenizer...")
    tok = transformers.AutoTokenizer.from_pretrained(
        "google/gemma-2-2b-it"
    )

    print(f"Decoding {limit_sequences} sequences from {cache_path}...")
    tokens = np.load(cache_path, mmap_mode="r")[:limit_sequences]
    # batch_decode returns list of strings, one per row
    texts = tok.batch_decode(tokens.tolist(), skip_special_tokens=True)
    print(f"  decoded {len(texts)} sequences, mean char length"
          f" = {int(np.mean([len(t) for t in texts]))}")
    return texts


def extract_probe_signatures(
    n_per_dataset: int = 125, seed: int = 0
) -> list[tuple[str, str]]:
    """Pull up to n_per_dataset probe prompts from each SAEBench dataset.

    Returns (dataset_name, 40-char signature from middle of prompt).
    """
    from datasets import load_dataset

    rng = random.Random(seed)
    sigs: list[tuple[str, str]] = []

    for ds_name, text_key, config in SAEBENCH_DATASETS:
        print(f"Loading {ds_name} (text_key={text_key}, cfg={config})...")
        try:
            # Streaming to avoid pulling the whole dataset
            split = "test"
            # Some of these datasets do not expose a test split; fall back.
            kwargs = {"streaming": True}
            if config is not None:
                kwargs["name"] = config
            try:
                ds = load_dataset(ds_name, split=split, **kwargs)
            except Exception:
                split = "train"
                ds = load_dataset(ds_name, split=split, **kwargs)

            samples: list[str] = []
            for i, row in enumerate(ds):
                if i >= 5000:
                    break
                # Europarl nests under {'translation': {'en': ..., 'fr': ...}}
                if text_key == "translation":
                    v = row.get("translation", {})
                    txt = v.get("en") if isinstance(v, dict) else None
                elif text_key == "chosen":
                    # Anthropic hh-rlhf: chosen is a conversation string
                    txt = row.get("chosen")
                else:
                    txt = row.get(text_key)
                if not isinstance(txt, str) or len(txt) < 80:
                    continue
                samples.append(txt)

            if len(samples) < 5:
                print(f"  WARN: only {len(samples)} usable samples")
                continue

            rng.shuffle(samples)
            for txt in samples[:n_per_dataset]:
                mid = len(txt) // 2
                sig = txt[max(0, mid - 20):mid + 20]
                # strip whitespace and skip if too short after cleaning
                if len(sig.strip()) >= 30:
                    sigs.append((ds_name, sig))
        except Exception as e:
            print(f"  ERROR loading {ds_name}: {e}")

    print(f"Extracted {len(sigs)} total signatures.")
    return sigs


def find_substring_matches(
    haystack: list[str], signatures: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    """Return (dataset, signature) pairs that appear in any haystack string."""
    # Concatenate the corpus once; substring search is O(N*M) but fast enough
    # for 3M tokens x 1000 signatures with CPython.
    corpus = "\n".join(haystack)
    hits: list[tuple[str, str]] = []
    for ds_name, sig in signatures:
        if sig in corpus:
            hits.append((ds_name, sig))
    return hits


def main() -> None:
    texts = decode_fineweb_cache(limit_sequences=24000)
    sigs = extract_probe_signatures(n_per_dataset=125, seed=0)

    total_sigs = len(sigs)
    hits = find_substring_matches(texts, sigs)
    hit_rate = len(hits) / max(1, total_sigs)

    print()
    print("=" * 60)
    print(f"Probe signatures checked: {total_sigs}")
    print(f"Hits (signature appears in FineWeb cache): {len(hits)}")
    print(f"Hit rate: {hit_rate:.2%}")
    print("=" * 60)

    by_ds: dict[str, list[str]] = {}
    for ds, sig in hits:
        by_ds.setdefault(ds, []).append(sig)
    for ds, hits_for_ds in sorted(by_ds.items()):
        print(f"  {ds}: {len(hits_for_ds)} hits")
        for h in hits_for_ds[:3]:
            print(f"    - {h!r}")

    verdict = "SEVERE" if hit_rate > 0.01 else "ACCEPTABLE"
    print()
    print(f"VERDICT: {verdict} (threshold 1%).")

    # Write a small JSON report.
    import json
    out_dir = Path("/workspace/temp_xc/experiments/phase5_downstream_utility/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "total_signatures": total_sigs,
        "hits": len(hits),
        "hit_rate": hit_rate,
        "verdict": verdict,
        "per_dataset_counts": {ds: len(v) for ds, v in by_ds.items()},
        "fineweb_cache": "data/cached_activations/gemma-2-2b-it/fineweb/",
        "fineweb_sequences": 24000,
        "fineweb_context_length": 128,
    }
    out_path = out_dir / "leakage_audit.json"
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
