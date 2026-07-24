"""B7 HARD PRE-GATE (CANDIDATES.md B7; briefing § 1): measure the
refusal/deflection-marker event rate on the target population BEFORE
building anything. The substring list + matching semantics are FROZEN
in ``refmark_lib.py``, committed before this script ran.

    HF_TOKEN=... .venv/bin/python -m \
        experiments.explorations.task_hunt.labels.measure_refmark

Corpus: ``allenai/WildChat-1M`` (train split, PINNED revision,
license ODC-By 1.0), streamed — first ``N_STREAM`` conversations in
shard order at the pinned revision (deterministic; same
convenience-prefix disclosure as the B6 pull). Target population
(frozen here): English conversations with >= ``MIN_A_TURNS``
assistant turns (the recurrence requirement — single-shot refusal
data cannot carry an intensity). Also reported for the stricter
>= 8-assistant-turn population a full build would likely use.

KILL RULE (from the briefing, recorded in the ledger): if fewer than
~2 % of assistant turns in the population match the frozen list, B7
dies in the ledger for free. Recurrence structure (share of
conversations with >= 2 marker turns) is reported alongside — an
intensity label needs recurrence, not just rate.

Output: ``refmark_pregate.json`` here (the receipt).
"""

from __future__ import annotations

import json
from pathlib import Path

from . import refmark_lib as rl

HERE = Path(__file__).resolve().parent
DATASET = "allenai/WildChat-1M"
REVISION = "7d6490e462285cf85d91eabea0f9a954fbddcd1f"
N_STREAM = 20_000
MIN_A_TURNS = 4
KILL_BAR = 0.02


def main():
    import datasets
    ds = datasets.load_dataset(DATASET, split="train", revision=REVISION,
                               streaming=True)
    pops = {"a4": {"min_turns": MIN_A_TURNS},
            "a8": {"min_turns": 8}}
    for p in pops.values():
        p.update(n_convs=0, n_turns=0, n_marker_turns=0,
                 convs_ge1=0, convs_ge2=0)
    n_seen = n_english = 0
    for i, ex in enumerate(ds):
        if i >= N_STREAM:
            break
        n_seen += 1
        if ex.get("language") != "English":
            continue
        n_english += 1
        turns = rl.assistant_turns(ex["conversation"])
        marks = [rl.is_marker_turn(t) for t in turns]
        for p in pops.values():
            if len(turns) < p["min_turns"]:
                continue
            p["n_convs"] += 1
            p["n_turns"] += len(turns)
            k = sum(marks)
            p["n_marker_turns"] += k
            p["convs_ge1"] += k >= 1
            p["convs_ge2"] += k >= 2
    out = {"dataset": DATASET, "revision": REVISION,
           "license": "ODC-By 1.0", "stream_prefix": N_STREAM,
           "n_streamed": n_seen, "n_english": n_english,
           "frozen_list": {"repo": rl.SOURCE_REPO,
                           "commit": rl.SOURCE_COMMIT,
                           "symbol": rl.SOURCE_SYMBOL,
                           "n_substrings": len(rl.REFUSAL_SUBSTRINGS)},
           "kill_bar_turn_rate": KILL_BAR, "populations": {}}
    for name, p in pops.items():
        rate = p["n_marker_turns"] / p["n_turns"] if p["n_turns"] else 0.0
        out["populations"][name] = {
            "min_assistant_turns": p["min_turns"],
            "n_conversations": p["n_convs"],
            "n_assistant_turns": p["n_turns"],
            "n_marker_turns": p["n_marker_turns"],
            "marker_turn_rate": rate,
            "frac_convs_ge1_marker": (p["convs_ge1"] / p["n_convs"]
                                      if p["n_convs"] else 0.0),
            "frac_convs_ge2_marker": (p["convs_ge2"] / p["n_convs"]
                                      if p["n_convs"] else 0.0),
            "kill_fires": rate < KILL_BAR,
        }
        print(f"[{name}] convs={p['n_convs']:,} turns={p['n_turns']:,} "
              f"marker_rate={rate:.4f} ge1={out['populations'][name]['frac_convs_ge1_marker']:.3f} "
              f"ge2={out['populations'][name]['frac_convs_ge2_marker']:.3f} "
              f"kill_fires={rate < KILL_BAR}", flush=True)
    (HERE / "refmark_pregate.json").write_text(json.dumps(out, indent=1))
    print(f"-> {HERE / 'refmark_pregate.json'}")


if __name__ == "__main__":
    main()
