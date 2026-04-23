"""1-prompt ping of the dated Haiku 4.5 judge id used by the autoresearch
grading path. Fails loudly if ANTHROPIC_API_KEY is unset or the id has
rotated — cheaper than finding out after an 8-min hybrid warmup.
"""

from __future__ import annotations

import os
import sys

import anthropic


JUDGE_MODEL_ID = "claude-haiku-4-5-20251001"


def main() -> int:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("[fail] ANTHROPIC_API_KEY not set in environment", file=sys.stderr)
        return 2
    client = anthropic.Anthropic(api_key=key)
    resp = client.messages.create(
        model=JUDGE_MODEL_ID,
        max_tokens=20,
        messages=[{"role": "user", "content": "Reply with one word: OK"}],
    )
    text = resp.content[0].text if resp.content else "<empty>"
    print(f"[ok] judge {JUDGE_MODEL_ID} responded: {text!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
