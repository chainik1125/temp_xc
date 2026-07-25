"""Track-A stimulus generators — balanced-marginal equality (XOR) designs.

Two candidates, one construction. Each item places two *constituents* at
separate positions and labels the pair by whether they MATCH:

  candidate 4  contradiction / fact-consistency:  value(mention 1) == value(mention 2)
  candidate 5  agreement attraction:              number(head noun)  == number(verb)

Balanced 2x2 over (constituent A, constituent B) so per-position marginals are
flat by design:

        B=x      B=y
  A=x   MATCH    MISMATCH
  A=y   MISMATCH MATCH

Consequences:
  * a per-token probe sees ONE position, whose marginal is 50/50 -> chance;
  * a LINEAR readout of any per-position decomposition (including a linear probe
    on the flattened window, and every additive dictionary code) gets the two
    marginals but cannot form the conjunction equality needs -> chance
    (additive-code theorem, synthetic/changepoint/bench_record.md § 3);
  * only a code whose nonlinearity crosses positions can expose the match.

TWO HARD DESIGN RULES, both learned the expensive way in this repo:

1. **Distinct texts, not distinct rows.** v1 of this file emitted 2,400 rows
   from 80 distinct sentences (60x duplication) and a T*d-dimensional probe
   memorised the handful of test points at AUC 1.000. `verify_balance` now
   asserts `distinct_texts == n_items` (the signed_motion memorisation lesson,
   README Part II § 7).
2. **Character offsets, not substring search.** The generator records the exact
   char index of each constituent's last character, so the gate never has to
   re-find a span in text (which is ambiguous when A and B are the same word).

Exact, template-derived, zero API. `verify_balance` is kill authority: if the
label is recoverable from either constituent alone, from length, or from the
distance between constituents, the generator is broken and no GPU time is spent.

Run:  python3 -m experiments.explorations.relational.stimuli --self-test
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "labels"

# ── candidate 5: subject-verb agreement across a variable distractor region ──
# (singular, plural) head nouns — `group` is the head index, so the split by
# group leaves whole lexical items unseen at test time.
HEADS: list[tuple[str, str]] = [
    ("the key", "the keys"), ("the author", "the authors"),
    ("the label", "the labels"), ("the ticket", "the tickets"),
    ("the bridge", "the bridges"), ("the letter", "the letters"),
    ("the switch", "the switches"), ("the painting", "the paintings"),
    ("the recipe", "the recipes"), ("the window", "the windows"),
    ("the invoice", "the invoices"), ("the cable", "the cables"),
    ("the permit", "the permits"), ("the drawing", "the drawings"),
    ("the sensor", "the sensors"), ("the ledger", "the ledgers"),
    ("the crate", "the crates"), ("the manuscript", "the manuscripts"),
    ("the antenna", "the antennas"), ("the valve", "the valves"),
    ("the receipt", "the receipts"), ("the turbine", "the turbines"),
    ("the folder", "the folders"), ("the lantern", "the lanterns"),
]
# Modifier phrases inserted between head and verb. Each carries its own noun
# number, sampled independently of the label, so the distractor region cannot
# proxy either constituent. 1-3 of these give head->verb distances ~4-16 tokens,
# which populates both the IN and OUT window strata across the T ladder.
MODS: list[str] = [
    "to the cabinet", "to the cabinets", "of the essay", "of the essays",
    "on the bottle", "on the bottles", "for the concert", "for the concerts",
    "near the harbour", "near the harbours", "from the lawyer", "from the lawyers",
    "beside the door", "beside the doors", "above the fireplace", "above the fireplaces",
    "in the notebook", "in the notebooks", "behind the curtain", "behind the curtains",
    "under the staircase", "under the staircases", "past the checkpoint", "past the checkpoints",
]
CONTS: list[str] = [
    "rusted", "well known", "faded", "expensive", "closed", "delayed",
    "broken", "insured", "unusable", "cracked", "missing", "overdue",
    "damaged", "unavailable", "sealed", "flagged",
]
FRAMES: list[str] = [
    "They reported that", "The inspector noted that", "It turned out that",
    "Records confirm that", "The summary states that", "We later learned that",
]
VERB = {"sg": "is", "pl": "are"}

# ── candidate 4: fact consistency ───────────────────────────────────────
# (subject, attribute phrase, value_x, value_y) — the two values are drawn from
# one semantic class so a mismatch is a contradiction, not a category error.
FACTS: list[tuple[str, str, str, str]] = [
    ("the meeting", "scheduled for", "Tuesday", "Thursday"),
    ("the package", "shipped from", "Boston", "Denver"),
    ("the account", "registered under", "Miller", "Parker"),
    ("the flight", "departing from", "gate four", "gate nine"),
    ("the report", "written in", "March", "August"),
    ("the sensor", "installed on", "the roof", "the fence"),
    ("the payment", "issued in", "euros", "yen"),
    ("the sample", "stored at", "four degrees", "nine degrees"),
    ("the contract", "signed by", "Alvarez", "Okonkwo"),
    ("the server", "hosted in", "Dublin", "Osaka"),
    ("the shipment", "cleared through", "Rotterdam", "Valencia"),
    ("the patient", "admitted on", "Monday", "Friday"),
    ("the invoice", "addressed to", "Whitfield", "Nakamura"),
    ("the survey", "conducted in", "spring", "autumn"),
    ("the licence", "granted for", "two years", "five years"),
    ("the parcel", "collected from", "the depot", "the kiosk"),
    ("the recording", "produced in", "Lisbon", "Helsinki"),
    ("the transfer", "authorised by", "Petrov", "Delgado"),
    ("the vessel", "registered in", "Panama", "Cyprus"),
    ("the deposit", "held at", "the branch", "the vault"),
]
FILLERS: list[str] = [
    "The file was reviewed by the assistant.",
    "A copy was kept for the records.",
    "Nothing else was noted at the time.",
    "The details were entered by hand.",
    "The form was checked once more.",
    "A short summary was added below.",
    "The clerk initialled the margin.",
    "Two pages were scanned in colour.",
    "The reference number was underlined.",
    "A note was pinned to the folder.",
    "The entry was logged the same day.",
    "The margin carried a small correction.",
]


@dataclass
class Item:
    uid: str
    task: str
    text: str
    label: int            # 1 = MATCH, 0 = MISMATCH
    a: str                # constituent A value (balance audit only)
    b: str                # constituent B value (balance audit only)
    group: int            # lexical group — splits are BY GROUP, never by item
    a_char_end: int       # index of A's LAST character
    b_char_end: int       # index of B's LAST character (the probe position)
    n_filler: int


def _uid(task: str, i: int) -> str:
    return f"{task}-{i:05d}"


def gen_agreement(n_per_cell: int = 60, seed: int = 0) -> list[Item]:
    rng = random.Random(seed)
    seen: set[str] = set()
    items: list[Item] = []
    i = 0
    for g, (h_sg, h_pl) in enumerate(HEADS):
        for head_num in ("sg", "pl"):
            for verb_num in ("sg", "pl"):
                made = 0
                guard = 0
                while made < n_per_cell and guard < n_per_cell * 40:
                    guard += 1
                    head = h_sg if head_num == "sg" else h_pl
                    verb = VERB[verb_num]
                    frame = rng.choice(FRAMES)
                    mods = " ".join(rng.sample(MODS, rng.randint(1, 3)))
                    cont = rng.choice(CONTS)
                    text = f"{frame} {head} {mods} {verb} {cont}."
                    if text in seen:
                        continue
                    seen.add(text)
                    a_end = len(frame) + 1 + len(head) - 1
                    b_start = text.index(f" {verb} ", len(frame) + len(head)) + 1
                    items.append(Item(
                        uid=_uid("agree", i), task="agreement", text=text,
                        label=int(head_num == verb_num),
                        a=head_num, b=verb_num, group=g,
                        a_char_end=a_end, b_char_end=b_start + len(verb) - 1,
                        n_filler=len(mods.split()) // 3))
                    i += 1
                    made += 1
    rng.shuffle(items)
    return items


def gen_contradiction(n_per_cell: int = 60, seed: int = 0) -> list[Item]:
    rng = random.Random(seed)
    seen: set[str] = set()
    items: list[Item] = []
    i = 0
    for g, (subj, attr, vx, vy) in enumerate(FACTS):
        for a_val in (vx, vy):
            for b_val in (vx, vy):
                made = 0
                guard = 0
                while made < n_per_cell and guard < n_per_cell * 60:
                    guard += 1
                    k = rng.randint(1, 3)
                    fill = " ".join(rng.sample(FILLERS, k))
                    head = f"Note: {subj} was {attr} {a_val}."
                    tail = f"Confirming: {subj} was {attr} {b_val}"
                    text = f"{head} {fill} {tail}"
                    if text in seen:
                        continue
                    seen.add(text)
                    items.append(Item(
                        uid=_uid("contra", i), task="contradiction", text=text,
                        label=int(a_val == b_val), a=a_val, b=b_val, group=g,
                        a_char_end=len(head) - 2,          # last char of a_val
                        b_char_end=len(text) - 1,          # last char of b_val
                        n_filler=k))
                    i += 1
                    made += 1
    rng.shuffle(items)
    return items


# ── the falsifier ───────────────────────────────────────────────────────
def _auc(scores: list[float], y: list[int]) -> float:
    pairs = sorted(zip(scores, y))
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    pos = [r for r, (_, yy) in zip(ranks, pairs) if yy == 1]
    n1, n0 = len(pos), len(pairs) - len(pos)
    if n1 == 0 or n0 == 0:
        return 0.5
    return (sum(pos) - n1 * (n1 + 1) / 2) / (n1 * n0)


def verify_balance(items: list[Item], tol: float = 0.03) -> dict:
    ys = [it.label for it in items]
    a_vals = sorted({it.a for it in items})
    b_vals = sorted({it.b for it in items})

    def marginal(get, vals):
        idx = {v: k for k, v in enumerate(vals)}
        return _auc([float(idx[get(it)]) for it in items], ys)

    n_distinct = len({it.text for it in items})
    cells: dict[str, int] = {}
    for it in items:
        cells[f"{it.a}|{it.b}"] = cells.get(f"{it.a}|{it.b}", 0) + 1
    stats = {
        "n": len(items),
        "n_distinct_texts": n_distinct,
        "label_rate": sum(ys) / len(ys),
        "auc_from_a": marginal(lambda it: it.a, a_vals),
        "auc_from_b": marginal(lambda it: it.b, b_vals),
        "auc_from_len": _auc([float(len(it.text)) for it in items], ys),
        "auc_from_nfiller": _auc([float(it.n_filler) for it in items], ys),
        "auc_from_gap": _auc([float(it.b_char_end - it.a_char_end)
                              for it in items], ys),
        "n_groups": len({it.group for it in items}),
        "cells": cells,
    }
    checks = {
        "no_duplicate_texts": n_distinct == len(items),
        "label_balanced": abs(stats["label_rate"] - 0.5) <= tol,
        "a_uninformative": abs(stats["auc_from_a"] - 0.5) <= tol,
        "b_uninformative": abs(stats["auc_from_b"] - 0.5) <= tol,
        "length_uninformative": abs(stats["auc_from_len"] - 0.5) <= tol,
        "nfiller_uninformative": abs(stats["auc_from_nfiller"] - 0.5) <= tol,
        "gap_uninformative": abs(stats["auc_from_gap"] - 0.5) <= tol,
        "cells_equal": len(set(cells.values())) == 1,
    }
    stats["checks"] = checks
    stats["PASS"] = all(checks.values())
    return stats


def _audit_offsets(items: list[Item]) -> None:
    """The recorded char offsets must point at the constituents they claim."""
    for it in items[:200]:
        assert 0 <= it.a_char_end < it.b_char_end < len(it.text), it.uid
        if it.task == "agreement":
            tail = it.text[:it.b_char_end + 1]
            assert tail.endswith(("is", "are")), (it.uid, tail[-12:])
            head_tail = it.text[:it.a_char_end + 1]
            assert head_tail.endswith(("key", "keys", "author", "authors",
                                       "label", "labels", "ticket", "tickets",
                                       "bridge", "bridges", "letter", "letters",
                                       "switch", "switches", "painting", "paintings",
                                       "recipe", "recipes", "window", "windows",
                                       "invoice", "invoices", "cable", "cables",
                                       "permit", "permits", "drawing", "drawings",
                                       "sensor", "sensors", "ledger", "ledgers",
                                       "crate", "crates", "manuscript", "manuscripts",
                                       "antenna", "antennas", "valve", "valves",
                                       "receipt", "receipts", "turbine", "turbines",
                                       "folder", "folders", "lantern", "lanterns")), \
                (it.uid, head_tail[-14:])
        else:
            assert it.text[:it.a_char_end + 1].endswith(it.a), (it.uid, it.a)
            assert it.text[:it.b_char_end + 1].endswith(it.b), (it.uid, it.b)


def build(n_per_cell: int = 60, seed: int = 0) -> dict:
    out = {}
    for name, fn in (("contradiction", gen_contradiction),
                     ("agreement", gen_agreement)):
        items = fn(n_per_cell=n_per_cell, seed=seed)
        _audit_offsets(items)
        out[name] = {"items": [asdict(i) for i in items],
                     "balance": verify_balance(items)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-cell", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    data = build(args.n_per_cell, args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    ok = True
    for name, d in data.items():
        (OUT / f"{name}_stimuli.json").write_text(json.dumps(d, indent=1))
        b = d["balance"]
        print(f"{name:14s} n={b['n']:5d} distinct={b['n_distinct_texts']:5d} "
              f"groups={b['n_groups']:3d} rate={b['label_rate']:.3f} "
              f"A={b['auc_from_a']:.3f} B={b['auc_from_b']:.3f} "
              f"len={b['auc_from_len']:.3f} gap={b['auc_from_gap']:.3f} "
              f"-> {'PASS' if b['PASS'] else 'FAIL'}")
        if not b["PASS"]:
            ok = False
            print("  FAILED:", [k for k, v in b["checks"].items() if not v])
    if args.self_test:
        print("SELF-TEST", "PASS" if ok else "FAIL")
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
