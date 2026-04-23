"""Smoke test for python_state_labeler.

Runnable as:
    uv run python experiments/code_benchmark/code_pipeline/_smoke_labeler.py

Exits 0 on success; prints a small per-character table showing that bracket
depth / indent / scope kind track a hand-inspected function.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from code_pipeline.python_state_labeler import (  # noqa: E402
    SCOPE_KIND_INV,
    _char_label_array,
    labels_for_chunk,
)


SOURCE = '''def add(x, y):
    """doc"""
    z = (x + [y, 1])
    for i in range(len(z)):
        async def inner():
            await z
    return z
'''


def assertion(cond: bool, msg: str) -> None:
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)


def main() -> None:
    arr = _char_label_array(SOURCE)
    # Probe a few specific characters:
    # Line 3: "    z = (x + [y, 1])"
    #                         ^ offset of '[' is inside depth-2 bracket
    line3_start = SOURCE.index("    z =")
    open_paren = SOURCE.index("(", line3_start)
    open_bracket = SOURCE.index("[", line3_start)
    close_bracket = SOURCE.index("]", line3_start)
    close_paren = SOURCE.index(")", close_bracket)

    assertion(arr[open_paren]["bracket_depth"] == 1,
              f"bracket_depth at '(' should be 1, got {arr[open_paren]['bracket_depth']}")
    inside_bracket = open_bracket + 1
    assertion(arr[inside_bracket]["bracket_depth"] == 2,
              f"bracket_depth inside '[...]' should be 2, got {arr[inside_bracket]['bracket_depth']}")
    just_after = close_paren + 1
    assertion(arr[just_after]["bracket_depth"] == 0,
              f"bracket_depth after outer ')' should be 0, got {arr[just_after]['bracket_depth']}")

    # scope_kind: inside the body of `def add` (not the header itself) we want
    # FUNCTION_BODY nesting=1, and inside `async def inner()` FUNCTION_BODY
    # nesting=2.
    z_in_body = SOURCE.index("z = (")
    assertion(SCOPE_KIND_INV[arr[z_in_body]["scope_kind"]] == "FUNCTION_BODY",
              f"scope_kind at 'z = (' should be FUNCTION_BODY, got "
              f"{SCOPE_KIND_INV[arr[z_in_body]['scope_kind']]}")
    assertion(arr[z_in_body]["scope_nesting"] == 1,
              f"scope_nesting at 'z = (' should be 1, got {arr[z_in_body]['scope_nesting']}")

    await_pos = SOURCE.index("await z")
    assertion(arr[await_pos]["scope_nesting"] == 2,
              f"scope_nesting at 'await z' should be 2, got {arr[await_pos]['scope_nesting']}")
    assertion(arr[await_pos]["has_await"] == 1,
              f"has_await flag should be set at 'await z'")

    # outer `add` is NOT async and does NOT await in its own scope.
    # Its body (before descending into `inner`) must carry has_await == 0.
    z_line = SOURCE.index("z = (")
    assertion(arr[z_line]["has_await"] == 0,
              f"has_await inside outer non-async function should be 0, "
              f"got {arr[z_line]['has_await']}")

    # Simulate Gemma-style char_offsets by using single-char spans.
    offsets = [(i, i + 1) for i in range(len(SOURCE))]
    labels = labels_for_chunk(SOURCE, offsets)
    assertion(len(labels["bracket_depth"]) == len(SOURCE),
              "labels_for_chunk output length should match offsets length")

    print("OK — labeler smoke test passed.")
    print(f"  Source length: {len(SOURCE)}")
    print(f"  Fields emitted: {sorted(labels.keys())}")
    print(f"  Max bracket_depth seen: {max(labels['bracket_depth'])}")
    print(f"  Max scope_nesting seen: {max(labels['scope_nesting'])}")
    print(f"  has_await sum: {sum(labels['has_await'])}")


if __name__ == "__main__":
    main()
