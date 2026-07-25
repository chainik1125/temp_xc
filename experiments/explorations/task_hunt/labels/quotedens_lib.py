"""Pure label logic for the quoted-speech intensity candidate
(CANDIDATES.md B9 `quotedens`, PG19-class fiction) — no tokenizers,
no I/O. Covered by ``tests/test_quotedens_labels.py``.

Event = a sentence containing any DOUBLE-quote-family character
(frozen set below). Single quotes are EXCLUDED: apostrophes make them
inexact, so books using single-quote dialogue conventions read as
low-event — a corpus-composition fact the builder DISCLOSES (per-book
event-rate distribution + zero-event book fraction), not a labeling
error. The intensity is the punctint kernel (HL 2 / support 8
sentences) over PREVIOUS sentences only — ``sentence_lambda`` is
re-exported unchanged from the frozen ``punctint_lib`` — and
event-sentence tokens are masked from probe rows (they display the
event ambiently: the punctint discipline)."""

from __future__ import annotations

from .punctint_lib import sentence_lambda  # noqa: F401 — frozen kernel, re-exported

# ASCII ", curly “ ”, low-9 „, guillemets « » — frozen; single quotes
# excluded (apostrophe-inexact)
QUOTE_CHARS = '"“”„«»'


def is_quote_sentence(sent: str) -> bool:
    return any(c in sent for c in QUOTE_CHARS)
