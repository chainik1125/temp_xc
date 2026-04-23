"""Per-token program-state labels for Python code.

For each Gemma-token chunk we emit, per token:

    bracket_depth      int  — running count of open ``(``, ``[``, ``{``
    indent_spaces      int  — leading whitespace of the current logical line
    scope_nesting      int  — function/class/comprehension/lambda nesting depth
    scope_kind         str  — one of MODULE, FUNCTION_BODY, CLASS_BODY,
                              COMPREHENSION, LAMBDA, STRING_LITERAL,
                              F_STRING_EXPR, COMMENT, OTHER
    distance_to_header int  — tokens since last ``def|class|for|with|try|if``
                              header (-1 if none so far).
    has_await          int  — 1 if an ``await`` keyword has appeared in the
                              current function scope; else 0.

The labeler works in two stages:

    (1) Python-level analysis using ``tokenize`` and the AST: produces
        per-character labels across the full source string.
    (2) Alignment to Gemma tokens: each Gemma token carries a
        ``(start_char, end_char)`` span (HF fast-tokenizer). Its labels are
        taken from ``start_char`` (first character of the token). Tokens
        whose offset is ``(-1, -1)`` (padding) get a sentinel.

All fields are emitted both as a ``dict[str, list[int]]`` (for fast numpy
conversion) and a parallel ``list[dict]`` for debugging.
"""

from __future__ import annotations

import ast
import io
import tokenize
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Enum-like integer encodings (string ↔ int) for scope_kind
# ---------------------------------------------------------------------------

SCOPE_KIND = {
    "MODULE": 0,
    "FUNCTION_BODY": 1,
    "CLASS_BODY": 2,
    "COMPREHENSION": 3,
    "LAMBDA": 4,
    "STRING_LITERAL": 5,
    "F_STRING_EXPR": 6,
    "COMMENT": 7,
    "OTHER": 8,
    "PADDING": -1,
}
SCOPE_KIND_INV = {v: k for k, v in SCOPE_KIND.items()}

PADDING_LABEL = {
    "bracket_depth": -1,
    "indent_spaces": -1,
    "scope_nesting": -1,
    "scope_kind": SCOPE_KIND["PADDING"],
    "distance_to_header": -1,
    "has_await": -1,
}


@dataclass
class PerTokenLabels:
    bracket_depth: list[int] = field(default_factory=list)
    indent_spaces: list[int] = field(default_factory=list)
    scope_nesting: list[int] = field(default_factory=list)
    scope_kind: list[int] = field(default_factory=list)
    distance_to_header: list[int] = field(default_factory=list)
    has_await: list[int] = field(default_factory=list)

    def append(self, d: dict[str, int]) -> None:
        for k, v in d.items():
            getattr(self, k).append(v)

    def to_dict(self) -> dict[str, list[int]]:
        return {
            "bracket_depth": self.bracket_depth,
            "indent_spaces": self.indent_spaces,
            "scope_nesting": self.scope_nesting,
            "scope_kind": self.scope_kind,
            "distance_to_header": self.distance_to_header,
            "has_await": self.has_await,
        }


# ---------------------------------------------------------------------------
# Character-level label array for one source string
# ---------------------------------------------------------------------------


def _char_label_array(source: str) -> list[dict[str, int]]:
    """Return a per-character list of label dicts for ``source``.

    Uses ``tokenize.generate_tokens`` for bracket / indent / comment tracking,
    and a single AST walk for scope_kind / scope_nesting / distance_to_header
    / has_await.

    The returned list has ``len(source)`` entries. A final extra entry is
    appended with the last seen labels so boundary tokens ending at EOF can
    still be indexed safely.
    """
    n = len(source)
    labels: list[dict[str, int]] = [dict(
        bracket_depth=0,
        indent_spaces=0,
        scope_nesting=0,
        scope_kind=SCOPE_KIND["MODULE"],
        distance_to_header=-1,
        has_await=0,
    ) for _ in range(n + 1)]

    # ---- AST-driven fields: scope_nesting, scope_kind, has_await, header offsets ----
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return labels  # fall back to zeros/MODULE

    # Line → col lookup: we'll mark spans by walking nodes and painting
    # [lineno:col_offset .. end_lineno:end_col_offset) character ranges.

    def lc_to_char(line: int, col: int) -> int:
        # Convert 1-indexed (line, col) to character offset. ast uses 1-indexed
        # line, 0-indexed col, on UTF-8 bytes in 3.8+ (col_offset is byte
        # offset, but for ASCII code this equals char offset). We accept the
        # approximation for non-ASCII — Gemma tokenization also works in bytes
        # so the two approximations travel together.
        cur = 0
        cur_line = 1
        for ch in source:
            if cur_line == line:
                return cur + col
            if ch == "\n":
                cur_line += 1
            cur += 1
        return cur

    def node_span(node: ast.AST) -> tuple[int, int]:
        start = lc_to_char(node.lineno, node.col_offset)
        end_line = getattr(node, "end_lineno", None)
        end_col = getattr(node, "end_col_offset", None)
        if end_line is None or end_col is None:
            return start, start + 1
        return start, lc_to_char(end_line, end_col)

    def paint_range(start: int, end: int, field_name: str, value: int) -> None:
        start = max(0, min(n, start))
        end = max(0, min(n, end))
        for i in range(start, end):
            labels[i][field_name] = value

    def paint_scope(start: int, end: int, kind: int, nesting: int) -> None:
        start = max(0, min(n, start))
        end = max(0, min(n, end))
        for i in range(start, end):
            labels[i]["scope_kind"] = kind
            labels[i]["scope_nesting"] = nesting

    header_nodes = (
        ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
        ast.For, ast.AsyncFor, ast.With, ast.AsyncWith, ast.Try, ast.If,
    )

    header_offsets: list[int] = []  # character offsets of each header's first char

    def _contains_await_in_own_scope(fn: ast.AST) -> bool:
        """True iff ``fn`` contains ``await`` not inside a nested fn/lambda."""
        stack: list[ast.AST] = list(ast.iter_child_nodes(fn))
        while stack:
            n_ = stack.pop()
            if isinstance(n_, ast.Await):
                return True
            if isinstance(n_, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            stack.extend(ast.iter_child_nodes(n_))
        return False

    def visit(node: ast.AST, nesting: int) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            s, e = node_span(node)
            paint_scope(s, e, SCOPE_KIND["FUNCTION_BODY"], nesting + 1)
            if _contains_await_in_own_scope(node):
                paint_range(s, e, "has_await", 1)
            header_offsets.append(s)
            nesting_child = nesting + 1
        elif isinstance(node, ast.ClassDef):
            s, e = node_span(node)
            paint_scope(s, e, SCOPE_KIND["CLASS_BODY"], nesting + 1)
            header_offsets.append(s)
            nesting_child = nesting + 1
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            s, e = node_span(node)
            paint_scope(s, e, SCOPE_KIND["COMPREHENSION"], nesting + 1)
            nesting_child = nesting + 1
        elif isinstance(node, ast.Lambda):
            s, e = node_span(node)
            paint_scope(s, e, SCOPE_KIND["LAMBDA"], nesting + 1)
            nesting_child = nesting + 1
        elif isinstance(node, header_nodes):
            s, _ = node_span(node)
            header_offsets.append(s)
            nesting_child = nesting
        else:
            nesting_child = nesting
        for child in ast.iter_child_nodes(node):
            visit(child, nesting_child)

    visit(tree, 0)

    # Paint distance_to_header by sweeping char-by-char.
    header_offsets.sort()
    last_hdr: int | None = None
    hi = 0
    for i in range(n):
        while hi < len(header_offsets) and header_offsets[hi] <= i:
            last_hdr = header_offsets[hi]
            hi += 1
        labels[i]["distance_to_header"] = (i - last_hdr) if last_hdr is not None else -1

    # ---- tokenize-driven fields: bracket_depth, indent_spaces, comments, strings ----
    try:
        gen = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except tokenize.TokenizeError:
        return labels

    OPENERS = {"(", "[", "{"}
    CLOSERS = {")", "]", "}"}
    depth = 0
    current_indent = 0
    prev_end_char = 0
    for tok in gen:
        tok_type, tok_string, (srow, scol), (erow, ecol), _line = tok
        start_char = lc_to_char(srow, scol)
        end_char = lc_to_char(erow, ecol)
        # Fill gap between previous token end and current start with current
        # running values (handles whitespace / implicit newlines).
        for i in range(prev_end_char, start_char):
            labels[i]["bracket_depth"] = depth
            labels[i]["indent_spaces"] = current_indent
        if tok_type == tokenize.INDENT:
            current_indent += len(tok_string.replace("\t", "    "))
        elif tok_type == tokenize.DEDENT:
            # Reduce by four as a heuristic; Python's tokenize emits matching
            # DEDENTs so this stays in sync for PEP-8 code. For non-PEP-8 code
            # we would need to pop a stack — an acceptable approximation here.
            current_indent = max(0, current_indent - 4)
        elif tok_type == tokenize.COMMENT:
            for i in range(start_char, min(n, end_char)):
                labels[i]["scope_kind"] = SCOPE_KIND["COMMENT"]
        elif tok_type == tokenize.STRING:
            # Classify as STRING_LITERAL (f-strings are still tokenize.STRING;
            # their embedded expressions are NOT separately tokenized here —
            # we approximate by labelling f"{...}" as STRING_LITERAL, which is
            # coarse but consistent.
            for i in range(start_char, min(n, end_char)):
                # Do not overwrite an already-painted comment.
                if labels[i]["scope_kind"] != SCOPE_KIND["COMMENT"]:
                    labels[i]["scope_kind"] = SCOPE_KIND["STRING_LITERAL"]
        elif tok_type == tokenize.OP:
            if tok_string in OPENERS:
                # The opener itself sees the *new* depth so its span is
                # labelled depth+1; callers typically read start_char so
                # this matches visual expectation.
                for i in range(start_char, min(n, end_char)):
                    labels[i]["bracket_depth"] = depth + 1
                depth += 1
            elif tok_string in CLOSERS:
                depth = max(0, depth - 1)
                for i in range(start_char, min(n, end_char)):
                    labels[i]["bracket_depth"] = depth
        # Fill the current token span with running values (overwrite for
        # tokens where we did not special-case above).
        for i in range(start_char, min(n, end_char)):
            if labels[i]["bracket_depth"] == 0 and depth > 0:
                labels[i]["bracket_depth"] = depth
            if labels[i]["indent_spaces"] == 0 and current_indent > 0:
                labels[i]["indent_spaces"] = current_indent
        prev_end_char = end_char

    return labels


# ---------------------------------------------------------------------------
# Align char labels to Gemma tokens
# ---------------------------------------------------------------------------


def labels_for_chunk(
    source: str,
    char_offsets: list[tuple[int, int]],
) -> dict[str, list[int]]:
    """Given a chunk's (start, end) char spans per Gemma token, return label lists."""
    char_labels = _char_label_array(source)
    out = PerTokenLabels()
    for (s, _e) in char_offsets:
        if s < 0:
            out.append(PADDING_LABEL)
            continue
        if s >= len(char_labels):
            # token span starts past EOF — should not happen but guard anyway
            out.append(PADDING_LABEL)
            continue
        out.append(char_labels[s])
    return out.to_dict()


# ---------------------------------------------------------------------------
# Cache-level helpers: label every chunk in a cache
# ---------------------------------------------------------------------------


def label_cache(sources: list[dict[str, Any]]) -> dict[str, Any]:
    """Produce per-token labels for every chunk in the sources.jsonl cache.

    Returns a dict with numpy-friendly fields: each label is shape ``(N, T)``
    after calling ``to_tensor``.
    """
    import numpy as np
    field_names = ["bracket_depth", "indent_spaces", "scope_nesting",
                   "scope_kind", "distance_to_header", "has_await"]
    per_chunk: dict[str, list[list[int]]] = {k: [] for k in field_names}
    for row in sources:
        lbl = labels_for_chunk(row["source"], [tuple(o) for o in row["char_offsets"]])
        for k in field_names:
            per_chunk[k].append(lbl[k])
    return {k: np.asarray(v, dtype=np.int64) for k, v in per_chunk.items()}
