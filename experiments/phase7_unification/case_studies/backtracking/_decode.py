"""Post-decode normalization for the DeepSeek-R1-Distill-Llama tokenizer.

The published tokenizer config combines a byte-level BPE vocabulary (whose
leading-space tokens look like "Ġwait") with a Metaspace pre-tokenizer (whose
decoder replaces "▁" with " "). The two don't agree, so `tokenizer.decode()`
returns the byte-level glyphs `Ġ` (space) and `Ċ` (newline) verbatim. This
single function unmaps those glyphs back to whitespace.

Use everywhere we (a) save decoded generations, or (b) inspect decoded tokens
for keyword matching.
"""

from __future__ import annotations


# These three Unicode chars are the BPE glyphs for whitespace bytes:
#   Ġ (U+0120) → space (\x20)
#   Ċ (U+010A) → newline (\x0a)
#   č (U+010D) → carriage return (\x0d)
#   ĉ (U+0109) → tab (\x09)
_BPE_TO_WS = str.maketrans({"Ġ": " ", "Ċ": "\n", "č": "\r", "ĉ": "\t"})


def clean_decode(text: str) -> str:
    return text.translate(_BPE_TO_WS)


# Characters to strip when normalising a single token's decoded form before
# matching it against a keyword set. Includes both real whitespace/punctuation
# and the BPE glyphs above (so the leading-space variant of "Wait" — which
# decodes as "ĠWait" — still normalises to "wait").
import string as _string

STRIP_CHARS = _string.punctuation + _string.whitespace + "ĠĊčĉ"


def norm_token(s: str) -> str:
    return s.strip(STRIP_CHARS).lower()
