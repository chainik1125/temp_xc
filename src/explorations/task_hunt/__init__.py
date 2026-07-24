"""Reusable library code for the task hunt (`experiments/explorations/task_hunt/`).

Currently: real-activation datasource generators that present a cached
subject-model activation stream + a per-position label as a
:class:`temp_bench.data.synthetic.SyntheticData`, so the canonical runner
and the existing recovery evaluators can panel real tasks with no
``temp_bench/core/`` edits (the `module:fn` generator path).
"""
