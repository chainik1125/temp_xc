"""Exploration experiments — runners, docs, and results for research that may
or may not graduate into ``temp_bench``.

Each subpackage is one exploration (e.g. ``synthetic`` — the synthetic
temporal-benchmark program). Importable as ``experiments.explorations.<name>``
(the repo root is on ``sys.path`` when running from it); run a benchmark's
scripts as ``-m experiments.explorations.<name>.<bench>.<script>``.

Reusable *library* code an exploration develops belongs in ``src/`` (its own
``src/explorations/<name>/`` package, or — once it graduates — ``temp_bench``);
this tree is for the experiments + their artifacts, not importable library code.
"""
