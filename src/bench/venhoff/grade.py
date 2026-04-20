"""Parse Venhoff hybrid_token.py outputs and compute Gap Recovery per cell.

Venhoff's `hybrid_token.py` writes one JSONL per problem under
`hybrid/results/{dataset}/{model-tag}/`. Each row has accuracy flags
(or raw predicted strings) per (coefficient, token_window) cell. We
aggregate:

    base_accuracy     = fraction correct with coefficient=0 (no steering)
    thinking_accuracy = fraction correct on the thinking model alone
                        (Venhoff ships a separate per-problem JSONL
                        for thinking-model accuracy; we read it too)
    hybrid_accuracy   = fraction correct with (coef, token_window)
                        applied, for every cell in the grid

Gap Recovery per cell:

    gap_recovery = (hybrid - base) / (thinking - base)

The headline number we report is the **max over the 10 × 5 grid**.
`3.5%` on the Llama-8B MATH500 cell is the bar to beat (Venhoff Table 2).

Grading: MATH500 answers are canonical strings extracted from
`\\boxed{…}` in both reference and model output. We use `math_verify`
if available (stronger than string equality — handles LaTeX
equivalences), else fall back to normalized string match.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("venhoff.grade")


_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def extract_boxed(text: str) -> str | None:
    """Return the last `\\boxed{...}` contents, or None if absent."""
    if not text:
        return None
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def _normalize_answer(s: str) -> str:
    """Normalize for the string-match fallback grader."""
    s = s.strip()
    # Strip common LaTeX noise.
    for tok in ("$", "\\,", "\\;", "\\!", "\\ ", " "):
        s = s.replace(tok, "")
    s = s.replace("\\left", "").replace("\\right", "")
    return s.lower()


def is_correct(predicted: str | None, reference: str) -> bool:
    """Grade one (predicted, reference) pair.

    Uses `math_verify` if installed, else normalized string match. Both
    are imperfect; we favor the former when available.
    """
    if predicted is None:
        return False
    try:
        from math_verify import parse, verify  # optional dep
        pred_parsed = parse(f"${predicted}$")
        ref_parsed = parse(f"${reference}$")
        return bool(verify(pred_parsed, ref_parsed))
    except ImportError:
        return _normalize_answer(predicted) == _normalize_answer(reference)
    except Exception:
        # math_verify throws on weird inputs; fall back.
        return _normalize_answer(predicted) == _normalize_answer(reference)


@dataclass(frozen=True)
class CellResult:
    """Accuracy for one (coefficient, token_window) cell."""

    coefficient: float
    token_window: int
    n_total: int
    n_correct: int

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_total if self.n_total else 0.0


@dataclass(frozen=True)
class GapRecoveryResult:
    base_accuracy: float
    thinking_accuracy: float
    per_cell: list[CellResult]
    best_cell: CellResult
    best_gap_recovery: float


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def compute_gap_recovery(
    results_dir: Path,
    thinking_jsonl: Path,
    base_jsonl: Path | None = None,
    coefficients: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    token_windows: tuple[int, ...] = (0, -1, -15, -50, -100),
) -> GapRecoveryResult:
    """Compute Gap Recovery per cell + the best cell.

    Venhoff's hybrid_token.py writes one JSONL per (coef, token_window)
    cell — or sometimes one JSONL per problem with inner cells. This
    function tries both conventions.

    thinking_jsonl: per-problem JSONL of thinking-model outputs (Venhoff
        produces this in `generate-responses/results/` typically).
    base_jsonl: optional per-problem base-model JSONL. If not provided
        we fall back to `(coefficient=0, token_window=0)` from the hybrid
        results as the base cell.
    """
    # ── thinking accuracy
    thinking_total = 0
    thinking_correct = 0
    for row in _iter_jsonl(thinking_jsonl):
        pred = extract_boxed(row.get("full_response", "") or row.get("response", ""))
        ref = str(row.get("answer", ""))
        if not ref:
            continue
        thinking_total += 1
        if is_correct(pred, ref):
            thinking_correct += 1
    thinking_accuracy = thinking_correct / thinking_total if thinking_total else 0.0

    # ── per-cell hybrid accuracy from hybrid/results/…
    #   Venhoff's layout: one JSONL per problem containing
    #   `per_cell_predictions: [{"coefficient": c, "token_window": w, "predicted": "..."}, ...]`
    #   OR one JSONL per (coef, window) cell.
    # We try per-problem first.
    cells: dict[tuple[float, int], list[bool]] = {
        (c, w): [] for c in coefficients for w in token_windows
    }
    n_problems_seen = 0

    for per_problem in results_dir.glob("*.jsonl"):
        for row in _iter_jsonl(per_problem):
            ref = str(row.get("answer", ""))
            if not ref:
                continue
            predictions = row.get("per_cell_predictions") or row.get("cells") or []
            if predictions:
                n_problems_seen += 1
                for cell in predictions:
                    c = float(cell.get("coefficient", 0))
                    w = int(cell.get("token_window", 0))
                    if (c, w) not in cells:
                        continue
                    pred = extract_boxed(cell.get("predicted") or cell.get("response") or "")
                    cells[(c, w)].append(is_correct(pred, ref))

    per_cell: list[CellResult] = []
    for (c, w), flags in cells.items():
        n_total = len(flags)
        n_correct = sum(1 for f in flags if f)
        per_cell.append(CellResult(coefficient=c, token_window=w, n_total=n_total, n_correct=n_correct))

    # ── base accuracy
    if base_jsonl is not None:
        base_total = 0
        base_correct = 0
        for row in _iter_jsonl(base_jsonl):
            pred = extract_boxed(row.get("full_response", "") or row.get("response", ""))
            ref = str(row.get("answer", ""))
            if not ref:
                continue
            base_total += 1
            if is_correct(pred, ref):
                base_correct += 1
        base_accuracy = base_correct / base_total if base_total else 0.0
    else:
        # Heuristic fallback: (coefficient=0, token_window=0) cell
        cell00 = next((c for c in per_cell if c.coefficient == 0 and c.token_window == 0), None)
        base_accuracy = cell00.accuracy if cell00 else 0.0
        if cell00 is None:
            log.warning(
                "[warn] no base_jsonl supplied and no (c=0,w=0) cell in hybrid results; "
                "base_accuracy will be 0. Supply --base-jsonl for the correct number."
            )

    # ── best-cell gap recovery
    denom = thinking_accuracy - base_accuracy
    if denom <= 0:
        log.warning(
            "[warn] thinking_accuracy (%.3f) ≤ base_accuracy (%.3f); Gap Recovery undefined.",
            thinking_accuracy, base_accuracy,
        )
        best_gap = 0.0
        best_cell = max(per_cell, key=lambda c: c.accuracy) if per_cell else CellResult(0, 0, 0, 0)
    else:
        best_cell = max(per_cell, key=lambda c: c.accuracy - base_accuracy)
        best_gap = (best_cell.accuracy - base_accuracy) / denom

    log.info(
        "[result] gap_recovery | base=%.4f | thinking=%.4f | best_cell_acc=%.4f | gap_recovery=%.4f",
        base_accuracy, thinking_accuracy, best_cell.accuracy, best_gap,
    )
    return GapRecoveryResult(
        base_accuracy=base_accuracy,
        thinking_accuracy=thinking_accuracy,
        per_cell=sorted(per_cell, key=lambda c: (c.coefficient, c.token_window)),
        best_cell=best_cell,
        best_gap_recovery=best_gap,
    )
