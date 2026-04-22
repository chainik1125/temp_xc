"""Port of Venhoff's `utils/responses.py::extract_thinking_process`.

Verbatim behavioral port at upstream commit
`49a7f731ce693d813b9ae9a414f1739b992dbcef`. The selection policy
(single-tag / multi-tag / ORZ marker) is preserved exactly because
trace-quality downstream depends on it.
"""

from __future__ import annotations


def _normalize_byte_level_bpe(text: str) -> str:
    """Undo the byte-level BPE space/newline encoding if the text was
    saved without proper detokenization.

    GPT-2 / Llama-style tokenizers use a byte→printable-char table when
    encoding raw bytes so that every whitespace byte has a visible
    stand-in:
      - 0x20 (space)   → U+0120 (Ġ)
      - 0x0A (newline) → U+010A (Ċ)
      - 0x09 (tab)     → U+0109 (ĉ)
      - 0x0D (CR)      → U+010D (č)

    Correctly detokenized text has these converted back. Some trace
    dumps (seen from Venhoff's `generate_traces.py` on 2026-04-22) skip
    that step and persist the encoded form, which breaks every
    downstream splitter that assumes real whitespace.

    This is a no-op if the text is already clean (contains no Ġ/Ċ/etc.).
    """
    if "\u0120" not in text and "\u010a" not in text:
        return text
    return (
        text.replace("\u0120", " ")
        .replace("\u010a", "\n")
        .replace("\u0109", "\t")
        .replace("\u010d", "\r")
    )


def extract_thinking_process(response: str) -> str:
    """Extract the <think>...</think> span from a raw response.

    Selection policy:
      - no `<think>`: start at 0.
      - exactly one `<think>`: take that block.
      - multiple `<think>`: prefer the `"Assistant: <think>"` ORZ
        marker (first one); else fall back to the last `<think>`.

    Always consumes up to the first `</think>` after the chosen start.
    Output is normalized to undo byte-level BPE whitespace encoding if
    the caller didn't detokenize properly when saving the response.
    """
    response = _normalize_byte_level_bpe(response)

    think_tag = "<think>"
    end_tag = "</think>"
    orz_marker = "Assistant: <think>"

    n_think = response.count(think_tag)

    if n_think == 0:
        think_start = 0
    elif n_think == 1:
        think_start = response.find(think_tag) + len(think_tag)
    else:
        first_orz = response.find(orz_marker)
        if first_orz != -1:
            think_start = first_orz + len(orz_marker)
        else:
            think_start = response.rfind(think_tag) + len(think_tag)

    think_end = response.find(end_tag, think_start)
    if think_end == -1:
        think_end = len(response)

    return response[think_start:think_end].strip()
