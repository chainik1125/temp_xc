"""Pure label logic for the equation-density candidate (CANDIDATES.md
B6) — no tokenizers, no I/O. Covered by ``tests/test_eqdens_labels.py``.

The primary label is the kernel-smoothed trailing MATH-TOKEN rate over
PREVIOUS tokens only (novelty geometry: half-life 16 / support 64
tokens — the clock pinned INSIDE the panel ladder), with the current
token never contributing to its own label and every math-span token
MASKED from probe rows (the in-math bit is the disclosed regime-1
anchor, bracket-family, recorded dead — never the primary). The
token-level kernel is a STATED DEVIATION from the ledger's
"previous sentences/lines" sketch: the format scan found median line
length 16 chars with wildly heterogeneous line granularity across
docs, so a line-unit kernel would make the clock doc-dependent; the
token-unit kernel is exact and uniform.

FROZEN math-span grammar (committed before the builder runs; the
compiled regex below IS the grammar — leftmost match, alternation
order = precedence, unclosed delimiters match nothing):

- ``\\begin{ENV} … \\end{ENV}`` for ENV in equation/align/gather (+
  starred) — the align family is in the corpus (scan: 257 occurrences)
  even though plain ``\\begin{equation}`` scanned at zero;
- ``\\[ … \\]`` and ``\\( … \\)`` (may span lines);
- ``$$ … $$`` display math (may span lines; not opened by ``\\$``);
- ``$ … $`` inline math (single line; escaped ``\\$`` neither opens
  nor closes; empty ``$$`` never parses as inline).

Spans include their delimiters. Overlaps are impossible by
construction (matches are consumed left to right).
"""

from __future__ import annotations

import re

import numpy as np

from .novelty_lib import trailing_rate  # generic over unit streams

HALF_LIFE = 16          # kernel half-life in TOKENS
SUPPORT = 64            # kernel support in tokens; label NaN below pos 64
MIN_POS = 64            # manifest position floor = warm-up (fineweb-compat)

_ENVS = r"(?:equation|align|gather)\*?"
MATH_RE = re.compile(
    r"\\begin\{(" + _ENVS + r")\}[\s\S]*?\\end\{\1\}"
    r"|\\\[[\s\S]*?\\\]"
    r"|\\\([\s\S]*?\\\)"
    r"|(?<!\\)\$\$[\s\S]*?(?<!\\)\$\$"
    r"|(?<!\\)\$(?!\$)(?:\\.|[^$\\\n])+?\$")


def math_spans(text: str) -> list[tuple[int, int]]:
    """Character spans (start, end) of math-mode regions under the
    frozen grammar, delimiters inclusive, non-overlapping."""
    return [m.span() for m in MATH_RE.finditer(text)]


def char_math_mask(text: str, spans=None) -> np.ndarray:
    """Boolean per-character math mask."""
    mask = np.zeros(len(text), dtype=bool)
    for a, b in (math_spans(text) if spans is None else spans):
        mask[a:b] = True
    return mask


def token_math_bits(offsets, cmask: np.ndarray) -> np.ndarray:
    """1 where the token's char span overlaps any math span (zero-width
    tokens never overlap). ``offsets`` = tokenizer offset mapping."""
    csum = np.concatenate([[0], np.cumsum(cmask.astype(np.int64))])
    a = np.array([o[0] for o in offsets], dtype=np.int64)
    b = np.array([o[1] for o in offsets], dtype=np.int64)
    return ((csum[b] - csum[a]) > 0).astype(np.int8)


def trailing_math_rate(bits: np.ndarray) -> np.ndarray:
    """rate[t] = kernel-weighted math fraction of the previous SUPPORT
    tokens (lags 1..64, half-life 16); NaN during warm-up; bits[t]
    never contributes to rate[t]."""
    return trailing_rate(bits, half_life=HALF_LIFE, support=SUPPORT)


def doc_passes_filter(text: str, min_chars: int, max_chars: int,
                      min_spans: int) -> bool:
    """Corpus filter (frozen with the grammar): length bounds plus a
    minimum math-span count, so the tercile contrast is within-doc
    intensity rather than math-doc-vs-prose-doc identity (the B4
    doc-identity concern, handled at pull time)."""
    if not (min_chars <= len(text) <= max_chars):
        return False
    return len(math_spans(text)) >= min_spans
