"""Tests for the equation-density label logic (CANDIDATES.md B6)."""

import numpy as np

from experiments.explorations.task_hunt.labels import eqdens_lib as el


def test_grammar_all_frozen_forms():
    text = ("prose $a+b$ more\n"
            "$$x^2\n+ y$$ mid \\[z\\] and \\(w\\)\n"
            "\\begin{align*}\nu &= v\n\\end{align*} tail")
    spans = el.math_spans(text)
    got = [text[a:b] for a, b in spans]
    assert "$a+b$" in got
    assert "$$x^2\n+ y$$" in got
    assert "\\[z\\]" in got
    assert "\\(w\\)" in got
    assert "\\begin{align*}\nu &= v\n\\end{align*}" in got
    assert len(got) == 5


def test_grammar_escaped_dollar_and_unclosed():
    assert el.math_spans("costs \\$5 and \\$7 total") == []
    assert el.math_spans("an unclosed $ sign alone") == []
    assert el.math_spans("\\begin{equation} never ended") == []


def test_grammar_display_not_inline_and_no_newline_inline():
    spans = el.math_spans("$$a$$")
    assert spans == [(0, 5)]
    # inline math cannot span lines: a stray $ on each of two lines
    assert el.math_spans("price $10\nand $20 later") == []


def test_grammar_env_star_backreference():
    # \begin{align} must not close on \end{align*} or vice versa
    assert el.math_spans("\\begin{align} x \\end{align*}") == []
    got = el.math_spans("\\begin{gather} x \\end{gather}")
    assert len(got) == 1


def test_token_math_bits_overlap():
    text = "ab $x$ cd"
    cmask = el.char_math_mask(text)
    assert cmask.sum() == 3  # "$x$"
    offsets = [(0, 2), (2, 4), (4, 6), (6, 9), (9, 9)]
    bits = el.token_math_bits(offsets, cmask)
    # (2,4) covers " $" -> math; (4,6) covers "x$" -> math;
    # zero-width never math
    assert bits.tolist() == [0, 1, 1, 0, 0]


def test_trailing_math_rate_excludes_current_and_warms_up():
    bits = np.zeros(200, dtype=np.int8)
    bits[100] = 1
    rate = el.trailing_math_rate(bits)
    assert np.all(np.isnan(rate[: el.SUPPORT]))
    # own bit never in own label
    assert rate[100] == 0
    # the very next token sees the event at lag 1 (largest weight)
    assert rate[101] > rate[110] > rate[150] > 0
    # beyond kernel support the event is forgotten
    assert rate[100 + el.SUPPORT + 1] == 0


def test_doc_filter():
    mathy = "x " * 300 + "$a$ $b$ $c$"
    assert el.doc_passes_filter(mathy, 100, 10_000, 3)
    assert not el.doc_passes_filter(mathy, 100, 10_000, 4)   # span count
    assert not el.doc_passes_filter(mathy, 1000, 1100, 3)    # length
    assert not el.doc_passes_filter("no math here " * 50, 100, 10_000, 1)
