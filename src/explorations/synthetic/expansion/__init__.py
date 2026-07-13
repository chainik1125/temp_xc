"""Factory harness for the grounded-benchmark expansion loop.

Generalizes the machinery the backtracking investigation ran by hand
(`experiments/explorations/synthetic/backtracking/`: Claude-judge labeler →
temporal-signature measurement → N1/N2/N3 null battery → Appendix-B mirror)
into reusable library code, so each cycle's calibration driver is a thin,
config-only script. See `briefings/grounded-benchmark-expansion.md` (Cycle 1)
and `experiments/explorations/synthetic/expansion/` for the experiment side
(prereg cards, LEDGER, calibration records).

Modules (import them directly — this package is deliberately NOT imported by
`explorations.synthetic.__init__`, so the core record pipeline never pulls in
the `anthropic` dependency):

- :mod:`.client`    — Claude API wrapper: per-role model routing (bulk=Haiku,
  validate=Sonnet, think=Opus) + a persistent, thread-safe spend meter that
  hard-stops at the per-cycle cost cap.
- :mod:`.signature` — the temporal-signature toolkit generalized from
  `backtracking/measure.py`: parameterized over the per-span signal (binary /
  categorical / scalar), with the N1 (within-seq permute) / N2 (trend-
  preserving) / N3 (iid marginal) null battery, bootstrap CIs, and the
  label-noise robustness check.
- :mod:`.labeler`   — the Claude-judge runner (chunked per-span labeling with
  local context) + validation: held-out inter-judge agreement → noise floor,
  and an independent heuristic cross-check (the keyword-vs-judge F1 pattern).
- :mod:`.corpus`    — version-pinned fineweb sampling (streamed via
  `datasets`, cached to the volume) + sentence segmentation.
- :mod:`.mirrors`   — the Appendix-B generating-process menu, fit + generate +
  held-out validate (logistic-AR/Hawkes, k-state Markov, semi-Markov, AR(1),
  periodic+noise).
"""
