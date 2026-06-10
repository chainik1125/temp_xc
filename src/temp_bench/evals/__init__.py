"""Evaluator registry — one module per paper section.

Drop a .py file here + register in
:data:`temp_bench.core.runner._EVALUATOR_REGISTRY` → the dispatcher
routes ``python run.py <experiment>`` to it. All classes subclass
:class:`temp_bench.interfaces.evaluator.Evaluator` and return a flat
``dict[str, float]`` for the leaderboard row.

Five evaluators in the v2 framework:

- :class:`synthetic_recovery.SyntheticRecovery` — § 4 (eAUC/gAUC/NMSE)
- :class:`probing.ProbingEval`                  — § 5.1 sparse probing
- :class:`backtracking.BacktrackingEval`         — § 5.2 backtracking
- :class:`em.EmergentMisalignmentEval`           — § 5.3 EM (Wang)
- :class:`rlhf.RLHFEval`                         — § 5.4 HH-RLHF
"""
