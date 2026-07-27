"""HUNT4 adversarial replication wrapper (REPLICATION_CARD.md, mac-b).

Runs the UNMODIFIED committed screen (`hunt4.screen`) with every
registered stochastic constant shifted to an independent value, and
output redirected so mac-a's wave-1 artifacts are never written.
Seed table + patch-surface audit: REPLICATION_CARD.md § 1.

  python -m experiments.explorations.task_hunt.hunt4.replication_screen gemma2_2b
"""

from __future__ import annotations

import functools
import hashlib
import sys
from pathlib import Path

import experiments.explorations.conversion_depth.problib as problib
import experiments.explorations.task_hunt.dialevel.capacity_check as capacity_check
import experiments.explorations.task_hunt.hunt4.screen as S
import experiments.explorations.task_hunt.novelty.screen as novelty_screen

REP_SEEDS = {
    "MATCH_SEED": (1013, 8013),
    "SHUF_SEED": (1234, 8234),
    "FOREIGN_SEED": (4242, 11242),
    "NULL_SEED": (99, 7099),
    "probe_seed": (0, 7),
}
SCORER = Path(__file__).resolve().parent / "verdict.py"
SCORER_SHA = "06a624eff6f12f5b64a53b09c360c670864eec3c3ac39d4aafa50c68fb682fac"


def _apply_patches() -> None:
    got = hashlib.sha256(SCORER.read_bytes()).hexdigest()
    assert got == SCORER_SHA, f"scorer drifted: {got} != card § 2"

    assert novelty_screen.MATCH_SEED == REP_SEEDS["MATCH_SEED"][0]
    novelty_screen.MATCH_SEED = REP_SEEDS["MATCH_SEED"][1]
    assert S.SHUF_SEED == REP_SEEDS["SHUF_SEED"][0]
    S.SHUF_SEED = REP_SEEDS["SHUF_SEED"][1]
    assert S.FOREIGN_SEED == REP_SEEDS["FOREIGN_SEED"][0]
    S.FOREIGN_SEED = REP_SEEDS["FOREIGN_SEED"][1]
    # capacity_check keeps its own binding for any internal use
    capacity_check.FOREIGN_SEED = REP_SEEDS["FOREIGN_SEED"][1]
    assert S.NULL_SEED == REP_SEEDS["NULL_SEED"][0]
    S.NULL_SEED = REP_SEEDS["NULL_SEED"][1]

    probe_seed = REP_SEEDS["probe_seed"][1]

    @functools.wraps(problib.fit_probe)
    def seeded_fit_probe(*args, **kwargs):
        kwargs.setdefault("seed", probe_seed)
        return problib.fit_probe(*args, **kwargs)

    S.fit_probe = seeded_fit_probe

    S.RES = S.HERE / "results" / "replication"

    for name, (old, new) in REP_SEEDS.items():
        print(f"[replication] {name}: {old} -> {new}", flush=True)
    print(f"[replication] RES -> {S.RES}", flush=True)
    print(f"[replication] scorer sha256 OK ({SCORER_SHA[:12]}…)", flush=True)


def main() -> None:
    _apply_patches()
    for key in (sys.argv[1:] or ["gemma2_2b"]):
        S.screen(key)


if __name__ == "__main__":
    main()
