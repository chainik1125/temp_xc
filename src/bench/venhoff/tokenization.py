"""Sentence splitting + char-to-token alignment for Venhoff eval.

Direct ports from Venhoff's `utils/utils.py` at upstream commit
`49a7f731ce693d813b9ae9a414f1739b992dbcef`:

- `split_into_sentences` — `utils.py:716` — regex-based splitter with
  protection for decimals ("3.14"), single-letter abbreviations
  ("E. coli"), and math patterns ("k!"). Filters to `min_words>=3` by
  default (matches their behavior).
- `get_char_to_token_map` — `utils.py:241` — maps character positions
  in the source text to token indices via the HF tokenizer's
  `encode_plus(return_offsets_mapping=True)` offsets.

Port is byte-for-byte equivalent. Docstrings reformatted for local
style only. See `VENHOFF_PROVENANCE.md` for commit pin.

**Hazard (from integration_plan § 7):** `process_saved_responses` uses
`layer_output[:, token_start - 1:token_end, :]` — a `-1` offset on the
start index. The `_sentence_token_span` helper here returns
`(token_start, token_end)` *without* the -1; callers that need to
reproduce Venhoff's exact slicing must subtract 1 themselves and we
add assertions around that site.
"""

from __future__ import annotations

import re
from typing import Any


def split_into_sentences(text: str, min_words: int = 3) -> list[str]:
    """Split text into sentences, protecting decimals / abbreviations / math.

    Args:
        text: The input text to split into sentences.
        min_words: Drop sentences with fewer than this many
            whitespace-delimited tokens. Venhoff's default is 3.

    Returns:
        List of cleaned sentences, each at least `min_words` long.
    """
    protected_text = text
    replacements: list[tuple[str, str]] = []

    # Protect decimal numbers (e.g. "3.14").
    for match in re.finditer(r'\d+\.\d+', text):
        placeholder = f"__DECIMAL_{len(replacements)}__"
        replacements.append((placeholder, match.group()))
        protected_text = protected_text.replace(match.group(), placeholder)

    # Protect single-letter abbreviations (e.g. "E. coli").
    for match in re.finditer(r'\b[A-Za-z]\.\s+[A-Za-z]', text):
        placeholder = f"__ABBREV_{len(replacements)}__"
        replacements.append((placeholder, match.group()))
        protected_text = protected_text.replace(match.group(), placeholder)

    # Protect math patterns like "k!".
    for match in re.finditer(r'\b[A-Za-z]!', text):
        placeholder = f"__MATH_{len(replacements)}__"
        replacements.append((placeholder, match.group()))
        protected_text = protected_text.replace(match.group(), placeholder)

    # Normalize consecutive punctuation so the lookbehind split is clean.
    consecutive_punct_pattern = r'([.!?;])\1+'
    consecutive_matches: list[tuple[int, int, str]] = []
    for match in re.finditer(consecutive_punct_pattern, protected_text):
        consecutive_matches.append((match.start(), match.end(), match.group()))

    normalized_text = re.sub(consecutive_punct_pattern, r'\1', protected_text)
    sentences = re.split(r'(?<=[.!?;\n])', normalized_text)

    # Best-effort restoration of consecutive punctuation in each sentence.
    if consecutive_matches:
        for start, _end, original in consecutive_matches:
            for i, sentence in enumerate(sentences):
                if sentence and start < len(protected_text):
                    if original[0] in sentence and len(original) > 1:
                        sentences[i] = sentence.replace(original[0], original, 1)

    # Restore the protected patterns (decimals / abbrevs / math).
    for placeholder, original in replacements:
        sentences = [s.replace(placeholder, original) for s in sentences]

    sentences = [s.strip() for s in sentences if s.strip()]
    sentences = [s for s in sentences if len(s.split()) >= min_words]

    # Post-process: if a sentence starts with a leading quote after a
    # period-split, push the quote back onto the previous sentence.
    processed_sentences: list[str] = []
    for i, sentence in enumerate(sentences):
        if i > 0 and sentence.startswith('"') and processed_sentences:
            processed_sentences[-1] += '"'
            current_sentence = sentence[1:].strip()
            if current_sentence and len(current_sentence.split()) >= min_words:
                processed_sentences.append(current_sentence)
        else:
            processed_sentences.append(sentence)

    return processed_sentences


def get_char_to_token_map(text: str, tokenizer: Any) -> dict[int, int]:
    """Map character positions → token indices via HF offset mapping.

    Uses `tokenizer.encode_plus(..., return_offsets_mapping=True)`. Every
    character position inside a token's `[start, end)` range maps to
    that token's index. Characters outside any token (e.g. whitespace
    that was dropped) are simply absent from the dict.
    """
    offsets = tokenizer.encode_plus(text, return_offsets_mapping=True)['offset_mapping']
    char_to_token: dict[int, int] = {}
    for token_idx, (start, end) in enumerate(offsets):
        for char_pos in range(start, end):
            char_to_token[char_pos] = token_idx
    return char_to_token


def sentence_token_span(
    sentence: str,
    full_text: str,
    char_to_token: dict[int, int],
) -> tuple[int, int] | None:
    """Return (token_start, token_end) for `sentence` inside `full_text`.

    Reproduces Venhoff's lookup from `process_saved_responses`
    (`utils.py:426-436`) without the `-1` offset — callers slice
    `[token_start - 1:token_end]` themselves if they want their exact
    per-sentence-mean contract. Returns None if the sentence cannot be
    located or its token span is empty / degenerate.

    Edge case handled (not in Venhoff's code — there they silently drop
    these): when the sentence sits at the very end of the text, the
    exclusive end position `text_pos + len(sentence)` is not in
    `char_to_token` (HF `offset_mapping` uses half-open intervals). Fall
    back to `char_to_token[last_char_pos] + 1`, which reproduces the
    exclusive-end convention from the last token that contains content.
    Without this fallback the final sentence of every trace gets
    silently skipped — painful on MATH500 where the `<think>` block
    often ends at the response boundary.
    """
    text_pos = full_text.find(sentence)
    if text_pos < 0:
        return None
    token_start = char_to_token.get(text_pos)
    if token_start is None:
        return None
    end_pos = text_pos + len(sentence)
    token_end = char_to_token.get(end_pos)
    if token_end is None:
        # Walk backwards from end_pos-1 until we land inside a token, then
        # add 1 to get the exclusive end. Bounded by the sentence length
        # so we never cross into preceding content.
        for probe in range(end_pos - 1, text_pos - 1, -1):
            containing = char_to_token.get(probe)
            if containing is not None:
                token_end = containing + 1
                break
    if token_end is None:
        return None
    if token_start >= token_end:
        return None
    return token_start, token_end
