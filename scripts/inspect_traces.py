"""inspect_traces.py — diagnose why Phase 1 (collect_activations) is failing.

Usage (from repo root, inside the .venv):
    python scripts/inspect_traces.py

Or point at a different traces.json:
    python scripts/inspect_traces.py path/to/traces.json

Reports:
  - file type + size
  - number of traces
  - keys on each trace
  - response / thinking text length
  - first ~300 chars of the response so we can tell if it's empty or
    the splitter is what's broken.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DEFAULT_PATH = Path(
    "results/venhoff_eval/"
    "deepseek-r1-distill-llama-8b_math500-test_n500_L6_seed42/"
    "traces.json"
)


def main(path: Path) -> int:
    if not path.exists():
        print(f"[error] no such file: {path}")
        return 2

    size_mb = path.stat().st_size / 1024 / 1024
    print(f"file:      {path}")
    print(f"size:      {size_mb:.2f} MB")

    with path.open() as f:
        d = json.load(f)

    print(f"json type: {type(d).__name__}")

    if isinstance(d, list):
        print(f"n_traces:  {len(d)}")
        if not d:
            print("[verdict] traces list is EMPTY — regenerate Phase 0")
            return 0
        first = d[0]
        print(f"keys[0]:   {list(first.keys())}")
        for field in ("response", "thinking", "trace", "completion", "text"):
            if field in first:
                text = first[field]
                if isinstance(text, str):
                    print(f"{field!r}: len={len(text)} chars")
                    print(f"  sample: {text[:300]!r}")
                    print(f"  has newlines: {text.count(chr(10))} line-breaks")
                    print(f"  has periods:  {text.count('.')} periods")
        # Stats across all traces
        if isinstance(first.get("response"), str):
            lens = [len(t.get("response", "")) for t in d]
        elif isinstance(first.get("thinking"), str):
            lens = [len(t.get("thinking", "")) for t in d]
        else:
            lens = []
        if lens:
            import statistics
            print(f"\nresponse lengths: min={min(lens)} median={statistics.median(lens):.0f} max={max(lens)}")
            n_empty = sum(1 for L in lens if L == 0)
            print(f"empty traces:     {n_empty}/{len(lens)}")
            if n_empty == len(lens):
                print("[verdict] every trace has empty response — regenerate Phase 0")
            elif n_empty > 0:
                print(f"[verdict] {n_empty} traces empty, rest look ok — splitter may fail on empty ones")
            else:
                print("[verdict] traces look populated — problem is downstream in sentence splitter")
    elif isinstance(d, dict):
        print(f"top keys:  {list(d.keys())[:20]}")
        print("[verdict] traces.json is a dict, not a list — schema mismatch?")
    else:
        print(f"[verdict] unexpected JSON root type: {type(d).__name__}")

    return 0


if __name__ == "__main__":
    arg = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    raise SystemExit(main(arg))
