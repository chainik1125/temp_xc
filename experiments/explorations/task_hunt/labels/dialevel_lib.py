"""Pure label logic for the dialogue turn-length LEVEL candidate
(CANDIDATES.md B5) — no tokenizers, no I/O. Covered by
``tests/test_dialevel_labels.py``.

The primary is a LEVEL (trailing mean turn length over the previous
``SUPPORT_TURNS`` turns, current turn excluded) per the hedging-LEVEL
lesson: levels are aggregation-recoverable; trailing slopes collapse
to anchor − window mean. Dialogues render with a single newline
between turns — the minimal visible boundary marker; newline-spanning
tokens are the marker face and are masked from probe manifests.
"""

from __future__ import annotations

import numpy as np

SUPPORT_TURNS = 5      # trailing mean over this many previous turns
NEWLINE = "\n"


def render_dialogue(turns):
    """Join turns with single newlines; return (text, per-turn char
    spans) — spans cover the turn text only (not the separator)."""
    spans, pos = [], 0
    for t in turns:
        spans.append((pos, pos + len(t)))
        pos += len(t) + 1
    return NEWLINE.join(turns), spans


def trailing_turn_mean(turn_sizes, support: int = SUPPORT_TURNS) -> np.ndarray:
    """Per-turn level: mean of the PREVIOUS `support` turn sizes
    (current turn never in its own label); NaN while fewer than
    `support` previous turns exist."""
    sizes = np.asarray(turn_sizes, dtype=float)
    out = np.full(len(sizes), np.nan, dtype=np.float32)
    for i in range(support, len(sizes)):
        out[i] = sizes[i - support: i].mean()
    return out


def boundary_flags(offsets, text: str) -> np.ndarray:
    """1 for tokens whose char span contains a newline (the rendered
    turn separator — the marker face, masked from manifests)."""
    nl = {i for i, ch in enumerate(text) if ch == NEWLINE}
    out = np.zeros(len(offsets), dtype=np.int8)
    for k, (a, b) in enumerate(offsets):
        if any(i in nl for i in range(int(a), int(b))):
            out[k] = 1
    return out


def tokens_since_turn_start(turn_idx: np.ndarray) -> np.ndarray:
    """Within-turn token index (the disclosed conversion-risky clock
    face)."""
    turn_idx = np.asarray(turn_idx)
    out = np.zeros(len(turn_idx), dtype=np.int32)
    for k in range(1, len(turn_idx)):
        out[k] = out[k - 1] + 1 if turn_idx[k] == turn_idx[k - 1] else 0
    return out
