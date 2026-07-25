"""Candidate 1b (§ 8 form) — marker-type EQUALITY, the properly additive-blind design.

The amended `cards/role_order.md` § 8 governs. The label is a **product of
indicators at two positions**, which is what the additive-code theorem actually
requires:

    y = [ type(marker M2) == type(marker M1) ]      M1 earlier, M2 later

    <document> ... <document>   y = 1   (nesting deepened twice)
    </document> ... </document> y = 1   (depth fell twice)
    <document> ... </document>  y = 0   (a block opened then closed)
    </document> ... <document>  y = 0   (one closed, another opened)

Both marker slots are `open` in exactly half the items, so:

* **per-token** at M2 knows only its OWN type, and P(y | type(M2)) = 0.5 -> chance;
* **any** code additive over per-position features yields the two marginals, and
  `y = Σ_k 1[type(M1)=k]·1[type(M2)=k]` is a product, so no linear readout of a
  per-position decomposition can compute it — at any capacity, and (unlike the
  superseded ORDER design) at any offset weighting;
* only a nonlinearity that crosses positions can expose it.

The probe row sits at **M2's final token**, so the gate's oracle-pair arm is
exactly (M1, M2) — the two positions that carry the label.

Depth starts at 2 so any pair of markers is legal and no sequence goes negative.
Filler lengths between markers are jittered, and the char after the probe position
is asserted identical across every item (the leak that layer 0 caught in role v1).

Run:  python3 -m experiments.explorations.relational.stimuli_parity --self-test
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "labels"

OPEN, CLOSE = "<document>", "</document>"

PREAMBLE = [
    "I fetched two pages and pasted them together.",
    "The tool returned nested content, shown below.",
    "Here are the files, one inside the other.",
    "The archive expanded into these sections.",
    "The crawler saved both records in sequence.",
    "Two attachments came through nested like this.",
    "The export wrapped the pages as shown.",
    "I copied the bundle exactly as received.",
    "The scraper emitted the sections below.",
    "These came from the same folder, nested.",
    "The reader unpacked them in this order.",
    "Both documents arrived inside one wrapper.",
]
# Filler lines. Sampling 1-3 of them jitters the marker offsets.
LINES = [
    "The header repeated the reference number.",
    "A footnote pointed to an appendix.",
    "The page was dated at the top.",
    "Several rows were left blank.",
    "The final column held a checksum.",
    "One line was struck through in pencil.",
    "A stamp appeared in the lower corner.",
    "The text ended without a signature.",
    "Two lines were repeated verbatim.",
    "The margin carried a pencilled note.",
    "An index was printed on the reverse.",
    "The paper was creased along one edge.",
]
TAIL = "Tell me what these say."
SYSTEM = "You are a careful assistant that follows the operator's instructions."


@dataclass
class ParityItem:
    uid: str
    text: str
    label: int          # 1 = the two markers have the SAME type
    m1: int             # 1 = earlier marker is an opener
    m2: int             # 1 = later marker is an opener
    group: int          # preamble/filler group — splits are BY GROUP
    a_char_end: int     # last char of M1  (the gate's `v`)
    b_char_end: int     # last char of M2  (the gate's probe row `p`)
    n_filler: int       # filler lines between M1 and M2 (the jitter)


_CHAT_PREFIX = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n")


def _chat(user: str) -> str:
    return (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user}<|eot_id|>")


def gen_parity(n_per_cell: int = 60, seed: int = 0) -> list[ParityItem]:
    rng = random.Random(seed)
    items: list[ParityItem] = []
    seen: set[str] = set()
    i = 0
    for g, pre in enumerate(PREAMBLE):
        for m1 in (1, 0):
            for m2 in (1, 0):
                made = guard = 0
                while made < n_per_cell and guard < n_per_cell * 60:
                    guard += 1
                    k_mid = rng.randint(1, 3)          # jitter: M1 -> M2 distance
                    k_pre = rng.randint(1, 2)
                    k_post = rng.randint(1, 2)
                    pre_lines = rng.sample(LINES, k_pre)
                    mid_lines = rng.sample(LINES, k_mid)
                    post_lines = rng.sample(LINES, k_post)
                    # start at depth 2 so either marker type is legal at either slot
                    head = (f"{pre}\n{OPEN}\n{OPEN}\n" + "\n".join(pre_lines))
                    tok1 = OPEN if m1 else CLOSE
                    tok2 = OPEN if m2 else CLOSE
                    body = (f"{head}\n{tok1}\n" + "\n".join(mid_lines)
                            + f"\n{tok2}\n" + "\n".join(post_lines)
                            + f"\n{TAIL}")
                    text = _chat(body)
                    if text in seen:
                        continue
                    seen.add(text)
                    # Offsets computed ARITHMETICALLY. String search is unusable
                    # here: the depth-establishing preamble also contains
                    # "\n<document>\n", so searching for M1 found the preamble's
                    # opener whenever M1 was an opener — which made the M1->M2 gap
                    # predict the label at AUC 0.139. Caught by the generator's own
                    # gap check before any GPU time.
                    base = len(_CHAT_PREFIX)
                    m1_start = base + len(head) + 1
                    m2_start = (m1_start + len(tok1) + 1
                                + len("\n".join(mid_lines)) + 1)
                    assert text[m1_start:m1_start + len(tok1)] == tok1, "M1 offset"
                    assert text[m2_start:m2_start + len(tok2)] == tok2, "M2 offset"
                    items.append(ParityItem(
                        uid=f"parity-{i:05d}", text=text,
                        label=int(m1 == m2), m1=m1, m2=m2, group=g,
                        a_char_end=m1_start + len(tok1) - 1,
                        b_char_end=m2_start + len(tok2) - 1,
                        n_filler=k_mid))
                    i += 1
                    made += 1
    rng.shuffle(items)
    return items


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


def verify_parity(items: list[ParityItem], tol: float = 0.03) -> dict:
    ys = [it.label for it in items]
    cells: dict[str, int] = {}
    for it in items:
        cells[f"{it.m1}|{it.m2}"] = cells.get(f"{it.m1}|{it.m2}", 0) + 1
    after = {it.text[it.b_char_end + 1:it.b_char_end + 2] for it in items}
    stats = {
        "n": len(items),
        "n_distinct_texts": len({it.text for it in items}),
        "n_groups": len({it.group for it in items}),
        "label_rate": sum(ys) / len(ys),
        # each marker slot alone must be uninformative -> the product is the only route
        "auc_from_m1": _auc([float(it.m1) for it in items], ys),
        "auc_from_m2": _auc([float(it.m2) for it in items], ys),
        "auc_from_len": _auc([float(len(it.text)) for it in items], ys),
        "auc_from_gap": _auc([float(it.b_char_end - it.a_char_end) for it in items], ys),
        "auc_from_nfiller": _auc([float(it.n_filler) for it in items], ys),
        "char_after_probe": sorted(after),
        "cells": cells,
    }
    checks = {
        "no_duplicate_texts": stats["n_distinct_texts"] == stats["n"],
        "label_balanced": abs(stats["label_rate"] - 0.5) <= tol,
        "m1_uninformative": abs(stats["auc_from_m1"] - 0.5) <= tol,
        "m2_uninformative": abs(stats["auc_from_m2"] - 0.5) <= tol,
        "length_uninformative": abs(stats["auc_from_len"] - 0.5) <= tol,
        "gap_uninformative": abs(stats["auc_from_gap"] - 0.5) <= tol,
        "nfiller_uninformative": abs(stats["auc_from_nfiller"] - 0.5) <= tol,
        "probe_context_identical": len(after) == 1,
        "cells_equal": len(set(cells.values())) == 1,
    }
    stats["checks"] = checks
    stats["PASS"] = all(checks.values())
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-cell", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    items = gen_parity(args.n_per_cell, args.seed)
    b = verify_parity(items)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "parity_stimuli.json").write_text(json.dumps(
        {"items": [asdict(i) for i in items], "balance": b}, indent=1))
    print(f"parity         n={b['n']:5d} distinct={b['n_distinct_texts']:5d} "
          f"groups={b['n_groups']:3d} rate={b['label_rate']:.3f} "
          f"m1={b['auc_from_m1']:.3f} m2={b['auc_from_m2']:.3f} "
          f"len={b['auc_from_len']:.3f} gap={b['auc_from_gap']:.3f} "
          f"after={b['char_after_probe']} -> {'PASS' if b['PASS'] else 'FAIL'}")
    if not b["PASS"]:
        print("  FAILED:", [k for k, v in b["checks"].items() if not v])
    if args.self_test:
        print("SELF-TEST", "PASS" if b["PASS"] else "FAIL")
        raise SystemExit(0 if b["PASS"] else 1)


if __name__ == "__main__":
    main()
