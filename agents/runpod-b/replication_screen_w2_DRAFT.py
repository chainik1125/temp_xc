"""HUNT4W2 adversarial replication wrapper — DRAFT (runpod-b).

DRAFT copy staged in agents/runpod-b/ pre-freeze; freezes as
`hunt4w2/replication_screen.py` in ONE commit with the replication
card AFTER the bundle verdict posts, BEFORE any replication cell.

Runs the UNMODIFIED committed screen (`hunt4w2.screen`) with every
registered stochastic constant shifted to an independent value, and
output redirected so the wave screen JSONs are never written. Seed
table + patch-surface audit: HUNT4W2_REPLICATION_CARD § 1 (values =
the ratified hunt4 replication convention).

  python -m experiments.explorations.task_hunt.hunt4w2.replication_screen wikitext103:gemma2_2b pycode:gemma2_2b
"""

from __future__ import annotations

import functools
import hashlib
import sys
from pathlib import Path

import experiments.explorations.conversion_depth.problib as problib
import experiments.explorations.task_hunt.dialevel.capacity_check as capacity_check
import experiments.explorations.task_hunt.hunt4w2.screen as W2
import experiments.explorations.task_hunt.novelty.screen as novelty_screen

REP_SEEDS = {
    "MATCH_SEED": (1013, 8013),
    "SHUF_SEED": (1234, 8234),
    "FOREIGN_SEED": (4242, 11242),
    "NULL_SEED": (99, 7099),
    "probe_seed": (0, 7),
}
SCORER = Path(__file__).resolve().parent / "verdict.py"
SCORER_SHA = "f883dee966d57e826a4e4e52424328210b73ab0c51142bcc069ee9dc0172af54"


def _apply_patches() -> None:
    got = hashlib.sha256(SCORER.read_bytes()).hexdigest()
    assert got == SCORER_SHA, f"scorer drifted: {got} != card § 2"

    assert novelty_screen.MATCH_SEED == REP_SEEDS["MATCH_SEED"][0]
    novelty_screen.MATCH_SEED = REP_SEEDS["MATCH_SEED"][1]
    assert W2.SHUF_SEED == REP_SEEDS["SHUF_SEED"][0]
    W2.SHUF_SEED = REP_SEEDS["SHUF_SEED"][1]
    assert W2.FOREIGN_SEED == REP_SEEDS["FOREIGN_SEED"][0]
    W2.FOREIGN_SEED = REP_SEEDS["FOREIGN_SEED"][1]
    # capacity_check keeps its own binding for any internal use
    capacity_check.FOREIGN_SEED = REP_SEEDS["FOREIGN_SEED"][1]
    assert W2.NULL_SEED == REP_SEEDS["NULL_SEED"][0]
    W2.NULL_SEED = REP_SEEDS["NULL_SEED"][1]

    probe_seed = REP_SEEDS["probe_seed"][1]

    @functools.wraps(problib.fit_probe)
    def seeded_fit_probe(*args, **kwargs):
        kwargs.setdefault("seed", probe_seed)
        return problib.fit_probe(*args, **kwargs)

    W2.fit_probe = seeded_fit_probe

    W2.RES = W2.HERE / "results" / "replication"

    for name, (old, new) in REP_SEEDS.items():
        print(f"[replication] {name}: {old} -> {new}", flush=True)
    print(f"[replication] RES -> {W2.RES}", flush=True)
    print(f"[replication] scorer sha256 OK ({SCORER_SHA[:12]}…)", flush=True)


def main() -> None:
    _apply_patches()
    for job in sys.argv[1:]:
        corpus, model = job.split(":")
        W2.screen(corpus, model)


if __name__ == "__main__":
    main()
