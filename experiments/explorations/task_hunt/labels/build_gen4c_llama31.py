"""Labels-only amendment for the HUNT4W2 llama31 third leg (card § 2
priced this): materialize `gen4c_<corpus>_llama31.npz` via mac-c's
COMMITTED builder (the scout computed llama31 but shipped only the
first-wave pair — committed-weight deviation, stated in the scout).

    .venv/bin/python -m experiments.explorations.task_hunt.labels.build_gen4c_llama31

DETERMINISM CHECK (the amendment's licence): the freshly-built
llama31 triage/floor/overlap numbers must match the scout's
committed `gen4c_stats.json` llama31 blocks to 1e-6 — proving the
npz is the same object the scout priced, not a new measurement.
No bars, protocols, or first-wave artifacts change.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from . import build_gen4c as b

HERE = Path(__file__).resolve().parent


def _flat(d, pre=""):
    out = {}
    for k, v in d.items():
        kk = f"{pre}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, kk + "."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[kk] = float(v)
    return out


def main():
    b.COMMIT_NPZ = ("gpt2", "gemma2", "llama31")
    old = json.loads((HERE / "gen4c_stats.json").read_text())
    n_checked = 0
    for corpus in b.CORPORA:
        fresh = b.build_corpus_tokenizer(corpus, "llama31",
                                         b.TOKENIZERS["llama31"])
        want = _flat(old["corpora"][corpus]["per_tokenizer"]["llama31"])
        got = _flat(fresh)
        for k, v in want.items():
            assert k in got, f"{corpus}: missing stat {k}"
            assert math.isclose(got[k], v, rel_tol=0, abs_tol=1e-6), \
                f"{corpus}:{k} drifted {v} -> {got[k]}"
            n_checked += 1
    print(f"[gen4c_llama31] determinism check PASS "
          f"({n_checked} stats matched the committed artifact)")


if __name__ == "__main__":
    main()
