"""Toy data generators for C1 (Markov support) and C2 (coupled HMM).

Both generators are referenced from `configs/datasources.yaml` via the
``generator: temp_bench.data.toy:<fn>`` key. Imports here keep that
dotted-path importable.
"""

from temp_bench.data.toy.coupled import coupled_hmm
from temp_bench.data.toy.markov import markov_chain_support

__all__ = ["coupled_hmm", "markov_chain_support"]
