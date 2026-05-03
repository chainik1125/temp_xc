"""Data sources used by the paper.

- :mod:`temp_bench.data.toy`  — synthetic generators (C1: Markov, C2: coupled HMM)
- :mod:`temp_bench.data.nlp`  — Gemma-2-2b activation cache (C3, C4, C5, C7)

NLP caching is shared between C3 and C4 (same archs, same seqs). C5 and C7
build their own caches because they need different sequence handling.
"""
