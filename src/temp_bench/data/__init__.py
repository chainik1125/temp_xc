"""Data layer for ``temp_bench``.

Two flavors of data, with shared shape contracts:

- ``synthetic`` — toy generators (markov, coupled_hmm). In-memory.
- ``real_lm``  — activation caches from a subject LLM.

Both produce ``(N, seq_len, d_in)`` tensors of activations. Those feed
into either an :class:`ActivationBuffer` (token-level shuffle, for
per-token SAEs) or a :class:`WindowBuffer` (window-level shuffle, for
window archs like TXC). The buffers implement the
:class:`temp_bench.interfaces.batch_iter.BatchIter` protocol.
"""
