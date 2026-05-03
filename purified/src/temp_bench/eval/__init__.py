"""Evaluation API.

- :mod:`temp_bench.eval.synthetic` — toy-data NMSE / AUC / gAUC (C1, C2)
- :mod:`temp_bench.eval.probing`   — sparse probing (C3)
- :mod:`temp_bench.eval.qualitative` — passage probe + var/pdvar (C4)
- :mod:`temp_bench.eval.case_study` — :class:`CaseStudy` ABC (C5/C6/C7)
"""

from temp_bench.eval.case_study import CaseStudy  # noqa: F401
