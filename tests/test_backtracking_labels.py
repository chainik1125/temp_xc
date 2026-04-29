"""Tests for the backtracking case study labelling logic.

Pure-Python; no GPU or model weights required.
"""

from __future__ import annotations

from experiments.phase7_unification.case_studies.backtracking import _paths
from experiments.phase7_unification.case_studies.backtracking.label_backtracking import (
    _build_labels_for_trace,
    _is_keyword,
)


def test_offset_constants_match_paper():
    # Paper §3.1: optimal offset window for L10 of distilled Llama is ~-13..-8.
    assert _paths.NEG_OFFSET_LO == -13
    assert _paths.NEG_OFFSET_HI == -8
    assert _paths.ANCHOR_LAYER == 10
    assert "wait" in _paths.KEYWORDS and "hmm" in _paths.KEYWORDS


def test_keyword_detector_exact_match():
    kw = {"wait", "hmm"}
    # Plain matches with various punctuation and casing.
    assert _is_keyword("Wait", kw)
    assert _is_keyword(" Wait", kw)
    assert _is_keyword("Wait,", kw)
    assert _is_keyword("Wait!", kw)
    assert _is_keyword("hmm.", kw)
    assert _is_keyword("HMM", kw)


def test_keyword_detector_excludes_substring_matches():
    kw = {"wait", "hmm"}
    # Words that *contain* "wait" or "hmm" but are not exact matches must NOT
    # count — the paper's keyword metric is word-level, not substring.
    assert not _is_keyword("waiting", kw)
    assert not _is_keyword("waited", kw)
    assert not _is_keyword("hummingbird", kw)
    assert not _is_keyword("await", kw)


def test_d_plus_extraction_is_paper_offset_window():
    # Synthetic trace: 50 tokens, "<think>" at position 5, "</think>" at 49,
    # backtracking event at position 30. Paper offset is -13..-8 so D_+ should
    # contain {17, 18, 19, 20, 21, 22}.
    n = 50
    rec = {
        "trace_id": "synthetic_001",
        "category": "logic",
        "full_token_ids": list(range(n)),
        "input_len": 0,
        "think_open_pos": 5,
        "think_close_pos": 49,
    }
    decoded = ["x"] * n
    decoded[30] = "Wait"
    label = _build_labels_for_trace(rec, decoded, {"wait", "hmm"}, -13, -8)

    assert label["event_positions"] == [30]
    assert label["d_plus_positions"] == [17, 18, 19, 20, 21, 22]
    # "all positions" = think region exclusive of </think>
    assert label["think_lo"] == 6  # one past <think>
    assert label["think_hi"] == 49
    assert label["n_d_all"] == 49 - 6


def test_d_plus_clipped_to_think_region():
    # Event very early in the think region; offsets [-13,-8] would land
    # *before* <think>, so D_+ should be empty.
    n = 30
    rec = {
        "trace_id": "synthetic_002",
        "category": "logic",
        "full_token_ids": list(range(n)),
        "input_len": 0,
        "think_open_pos": 5,
        "think_close_pos": 25,
    }
    decoded = ["x"] * n
    decoded[10] = "hmm"  # event_pos=10, [10-13, 10-8] = [-3..2], all < think_lo=6
    label = _build_labels_for_trace(rec, decoded, {"wait", "hmm"}, -13, -8)
    assert label["event_positions"] == [10]
    assert label["d_plus_positions"] == []


def test_keyword_fraction_metric():
    from experiments.phase7_unification.case_studies.backtracking.evaluate_backtracking import (
        keyword_fraction,
    )

    # 4 of 8 words match → 0.5
    text = "First, wait. But I think — Hmm, actually never mind"
    frac, n_kw, n_tot = keyword_fraction(text, {"wait", "hmm"})
    # Words: "First,", "wait.", "But", "I", "think", "—", "Hmm,", "actually", "never", "mind"
    # Matches: "wait." → wait, "Hmm," → hmm  → 2 / 10 = 0.2
    assert n_tot == 10
    assert n_kw == 2
    assert frac == 0.2

    # Empty text guards
    frac, n_kw, n_tot = keyword_fraction("", {"wait", "hmm"})
    assert frac == 0.0 and n_kw == 0 and n_tot == 0


def test_keyword_fraction_handles_bpe_glyphs():
    """DeepSeek-R1-Distill-Llama tokenizer.decode leaves byte-level glyphs Ġ
    (space) and Ċ (newline) in the output. The eval metric must transparently
    normalise these before splitting on whitespace, otherwise the entire
    generation is one un-splittable word."""
    from experiments.phase7_unification.case_studies.backtracking.evaluate_backtracking import (
        keyword_fraction,
    )

    # Same content as above but in raw BPE form.
    text = "ĊFirst,ĠWait.ĠBut...ĠHmm,ĠactuallyĠneverĠmind"
    frac, n_kw, n_tot = keyword_fraction(text, {"wait", "hmm"})
    # After clean_decode + split: ["First,", "Wait.", "But...", "Hmm,",
    #                              "actually", "never", "mind"]
    # Two matches.
    assert n_tot == 7
    assert n_kw == 2


def test_is_keyword_handles_bpe_leading_space():
    """When iterating per-token decoded forms, leading-space tokens come back
    as e.g. 'ĠWait'. The detector must normalise them to 'wait' so they match
    the keyword set."""
    from experiments.phase7_unification.case_studies.backtracking.label_backtracking import (
        _is_keyword,
    )

    kw = {"wait", "hmm"}
    assert _is_keyword("ĠWait", kw)
    assert _is_keyword("ĠWait,", kw)
    assert _is_keyword("Ġhmm", kw)
    assert _is_keyword("ĊWait", kw)
    # And substring rejection still holds:
    assert not _is_keyword("Ġwaiting", kw)
    assert not _is_keyword("Ġhumming", kw)
