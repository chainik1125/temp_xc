"""Parity tests for the sentence splitter port.

Test cases come from Venhoff's `utils/test_split_into_sentences.py`
at upstream commit `49a7f731ce693d813b9ae9a414f1739b992dbcef`. Any
divergence means our port is not byte-compatible with their benchmark
and results would not be comparable to their published numbers.
"""

from __future__ import annotations

from src.bench.venhoff.tokenization import (
    sentence_token_span,
    split_into_sentences,
)


def test_basic_sentence_splitting():
    text = "This is the first sentence. This is the second sentence. This is the third sentence."
    assert split_into_sentences(text) == [
        "This is the first sentence.",
        "This is the second sentence.",
        "This is the third sentence.",
    ]


def test_different_punctuation_marks():
    text = "What is this? This is amazing! I think so; maybe not."
    assert split_into_sentences(text, min_words=2) == [
        "What is this?",
        "This is amazing!",
        "I think so;",
        "maybe not.",
    ]


def test_short_sentence_filtering_drops_below_three_words():
    text = "This is a good sentence. Yes. No way. This is another good sentence."
    assert split_into_sentences(text) == [
        "This is a good sentence.",
        "This is another good sentence.",
    ]


def test_whitespace_handling():
    text = "  First sentence here.   Second sentence there.  "
    assert split_into_sentences(text) == [
        "First sentence here.",
        "Second sentence there.",
    ]


def test_empty_string():
    assert split_into_sentences("") == []


def test_single_sentence_no_split():
    text = "This is a single sentence with more than three words"
    assert split_into_sentences(text) == [text]


def test_consecutive_punctuation():
    text = "What?! This is crazy!! Are you sure? Yes I am."
    assert split_into_sentences(text, min_words=1) == [
        "What?",
        "!!",
        "This is crazy!!",
        "Are you sure?",
        "Yes I am.",
    ]


def test_decimal_protection():
    """`3.14` must not split into two sentences."""
    text = "The value is 3.14 approximately. Next sentence here."
    out = split_into_sentences(text)
    assert any("3.14" in s for s in out)
    assert len(out) == 2


def test_single_letter_abbrev_protection():
    """`E. coli` must not split inside the abbreviation."""
    text = "The bacterium is E. coli and is well studied. Another sentence follows."
    out = split_into_sentences(text)
    assert any("E. coli" in s for s in out)
    assert len(out) == 2


def test_math_bang_protection():
    """`k!` (factorial-style) must not split."""
    text = "The formula uses k! in the denominator here. End of trace."
    out = split_into_sentences(text)
    assert any("k!" in s for s in out)
    assert len(out) == 2


# ───────── sentence_token_span ─────────


class _FakeTokenizer:
    """Minimal HF-compatible tokenizer stub that splits on whitespace."""

    def encode_plus(self, text: str, return_offsets_mapping: bool = True):
        offsets: list[tuple[int, int]] = []
        i = 0
        while i < len(text):
            while i < len(text) and text[i].isspace():
                i += 1
            if i >= len(text):
                break
            j = i
            while j < len(text) and not text[j].isspace():
                j += 1
            offsets.append((i, j))
            i = j
        return {"offset_mapping": offsets}


def test_sentence_token_span_finds_contained_sentence():
    from src.bench.venhoff.tokenization import get_char_to_token_map

    full = "First sentence here. Second goes here too."
    tok = _FakeTokenizer()
    c2t = get_char_to_token_map(full, tok)
    span = sentence_token_span("Second goes here too.", full, c2t)
    assert span is not None
    start, end = span
    assert start < end
    # "First sentence here." occupies tokens 0..2, then "Second" starts at 3.
    assert start == 3


def test_sentence_token_span_returns_none_on_missing():
    from src.bench.venhoff.tokenization import get_char_to_token_map

    full = "This text is here."
    tok = _FakeTokenizer()
    c2t = get_char_to_token_map(full, tok)
    assert sentence_token_span("unrelated sentence", full, c2t) is None


def test_sentence_token_span_handles_trailing_sentence():
    """Regression test for the MATH500 final-sentence edge case.

    When a sentence ends at `len(full_text)`, the exclusive-end index
    `text_pos + len(sentence)` lands outside every token's half-open
    range — HF tokenizer offset maps use `[start, end)`. Previous code
    returned None (Venhoff's behavior), silently dropping the last
    sentence of every trace. The fallback walks backwards to the last
    char inside a token and adds 1 for the exclusive end.
    """
    from src.bench.venhoff.tokenization import get_char_to_token_map

    full = "First sentence here. Second goes here too."
    tok = _FakeTokenizer()
    c2t = get_char_to_token_map(full, tok)
    span = sentence_token_span("Second goes here too.", full, c2t)
    assert span is not None
    start, end = span
    # "Second" is token 3; "too." is the last token. Exclusive end is
    # last_token_idx + 1 = 7 (tokens: First=0, sentence=1, here.=2,
    # Second=3, goes=4, here=5, too.=6).
    assert start == 3
    assert end == 7
