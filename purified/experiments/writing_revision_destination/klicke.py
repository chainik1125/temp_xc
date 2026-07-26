"""Task-side gate for the destination of a human writing revision.

The task is defined strictly before a trailing deletion burst begins. Given
the final ``T`` words currently present at the leading edge, predict whether
the writer will erase 2, 3, 4, or at least 5 words. KLiCKe text is reconstructed
only in memory; the saved artifact contains aggregate diagnostics and hashes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss


PROTOCOL_VERSION = "klicke-trailing-deletion-destination-v1"
LABELS = (2, 3, 4, 5)
DEFAULT_WINDOW = 5
CSV_PREFIX = "WritingTask/keystrokelogs/csv/"
WORD_RE = re.compile(
    r"[^\W\d_]+(?:['’\-][^\W\d_]+)*",
    flags=re.UNICODE,
)
COMPLEX_ACTIVITIES = ("Replace",)


@dataclass(frozen=True)
class LogRow:
    row_index: int
    activity: str
    cursor_after: int
    text_change: str
    pause_ms: float
    down_event: str


@dataclass(frozen=True)
class RevisionEvent:
    event_hash: str
    window_hash: str
    writer_id: str
    row_index: int
    words: tuple[str, ...]
    label: int
    deleted_words: int
    pause_ms: float
    prefix_word_count: int
    remove_actions: int
    single_character_backspaces: bool
    preburst_text: str = field(default="", repr=False, compare=False)
    postburst_text: str = field(default="", repr=False, compare=False)


@dataclass
class PendingBurst:
    writer_id: str
    row_index: int
    initial_text: str
    pause_ms: float
    remove_actions: int = 0
    single_character_backspaces: bool = True


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: object, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def normalize_logged_change(value: str) -> str:
    """Decode the logger's single-character escape convention."""

    return {
        r"\n": "\n",
        r"\"": '"',
        r"\\": "\\",
    }.get(value, value)


def _decode_csv(payload: bytes) -> str:
    try:
        return payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        return payload.decode("cp1252", errors="replace")


def csv_members(archive: zipfile.ZipFile) -> list[str]:
    return sorted(
        name
        for name in archive.namelist()
        if name.startswith(CSV_PREFIX) and name.endswith(".csv")
    )


def read_log_rows(archive: zipfile.ZipFile, member: str) -> Iterator[LogRow]:
    text = _decode_csv(archive.read(member))
    reader = csv.DictReader(io.StringIO(text, newline=""))
    required = {
        "Activity",
        "CursorPosition",
        "TextChange",
        "PauseTime",
        "DownEvent",
    }
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise ValueError(f"{member} lacks required KLiCKe columns")
    for row_index, row in enumerate(reader):
        try:
            cursor_after = int(float(row["CursorPosition"]))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{member} has invalid cursor at row {row_index}"
            ) from error
        try:
            pause_ms = float(row["PauseTime"])
        except (TypeError, ValueError):
            pause_ms = 0.0
        yield LogRow(
            row_index=row_index,
            activity=row["Activity"],
            cursor_after=cursor_after,
            text_change=normalize_logged_change(row["TextChange"]),
            pause_ms=max(0.0, pause_ms),
            down_event=row["DownEvent"],
        )


def _words(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).lower() for match in WORD_RE.finditer(text))


def _complete_left_boundary(prefix: str, removed: str) -> bool:
    if not prefix or not removed:
        return False
    return not (prefix[-1].isalpha() and removed[0].isalpha())


def _event_from_burst(
    burst: PendingBurst,
    final_text: str,
    *,
    window: int,
) -> RevisionEvent | None:
    if not burst.initial_text.startswith(final_text):
        return None
    removed = burst.initial_text[len(final_text) :]
    if not _complete_left_boundary(final_text, removed):
        return None
    removed_words = _words(removed)
    if len(removed_words) < 2:
        return None
    history = _words(burst.initial_text)
    if len(history) < window:
        return None
    window_words = history[-window:]
    label = min(len(removed_words), max(LABELS))
    window_hash = hashlib.sha256(
        "\x1f".join(window_words).encode("utf-8")
    ).hexdigest()
    event_hash = hashlib.sha256(
        (
            f"{PROTOCOL_VERSION}\x1f{burst.writer_id}\x1f"
            f"{burst.row_index}\x1f{window_hash}\x1f{label}"
        ).encode("utf-8")
    ).hexdigest()
    return RevisionEvent(
        event_hash=event_hash,
        window_hash=window_hash,
        writer_id=burst.writer_id,
        row_index=burst.row_index,
        words=window_words,
        label=label,
        deleted_words=len(removed_words),
        pause_ms=burst.pause_ms,
        prefix_word_count=len(history),
        remove_actions=burst.remove_actions,
        single_character_backspaces=burst.single_character_backspaces,
        preburst_text=burst.initial_text,
        postburst_text=final_text,
    )


def extract_writer_events(
    writer_id: str,
    rows: Iterable[LogRow],
    *,
    window: int = DEFAULT_WINDOW,
) -> tuple[list[RevisionEvent], dict[str, int]]:
    """Conservatively reconstruct one session and extract trailing deletions."""

    text = ""
    pending: PendingBurst | None = None
    events: list[RevisionEvent] = []
    diagnostics: Counter[str] = Counter()

    def finish() -> None:
        nonlocal pending
        if pending is None:
            return
        diagnostics["trailing_bursts"] += 1
        event = _event_from_burst(pending, text, window=window)
        if event is None:
            diagnostics["ineligible_trailing_bursts"] += 1
        else:
            events.append(event)
            diagnostics["eligible_events"] += 1
        pending = None

    for row in rows:
        diagnostics["rows_read"] += 1
        activity = row.activity

        if activity in ("Input", "Paste"):
            finish()
            start = row.cursor_after - len(row.text_change)
            if not 0 <= start <= len(text):
                diagnostics["truncated_cursor_mismatch"] += 1
                break
            text = text[:start] + row.text_change + text[start:]
            diagnostics[f"applied_{activity.lower()}"] += 1
            continue

        if activity == "Remove/Cut":
            start = row.cursor_after
            end = start + len(row.text_change)
            if (
                not 0 <= start <= end <= len(text)
                or text[start:end] != row.text_change
            ):
                finish()
                diagnostics["truncated_remove_mismatch"] += 1
                break
            trailing = end == len(text)
            if not trailing:
                finish()
                diagnostics["applied_nontrailing_removal"] += 1
            elif pending is None:
                pending = PendingBurst(
                    writer_id=writer_id,
                    row_index=row.row_index,
                    initial_text=text,
                    pause_ms=row.pause_ms,
                )
            if pending is not None and trailing:
                pending.remove_actions += 1
                pending.single_character_backspaces &= (
                    row.down_event == "Backspace"
                    and len(row.text_change) == 1
                )
            text = text[:start] + text[end:]
            diagnostics["applied_removal"] += 1
            continue

        finish()
        if activity == "Nonproduction":
            diagnostics["nonproduction_rows"] += 1
            continue
        if activity in COMPLEX_ACTIVITIES or activity.startswith("Move From"):
            diagnostics["truncated_complex_activity"] += 1
            break
        diagnostics["truncated_unknown_activity"] += 1
        break

    finish()
    diagnostics["reconstructed_characters"] = len(text)
    return events, dict(diagnostics)


def deduplicate_windows(
    events: Sequence[RevisionEvent],
) -> tuple[list[RevisionEvent], dict[str, int]]:
    groups: dict[str, list[RevisionEvent]] = defaultdict(list)
    for event in events:
        groups[event.window_hash].append(event)
    retained: list[RevisionEvent] = []
    same_label_dropped = 0
    conflict_dropped = 0
    for group in groups.values():
        labels = {event.label for event in group}
        if len(labels) != 1:
            conflict_dropped += len(group)
            continue
        chosen = min(group, key=lambda event: event.event_hash)
        retained.append(chosen)
        same_label_dropped += len(group) - 1
    retained.sort(key=lambda event: event.event_hash)
    return retained, {
        "exact_window_groups": len(groups),
        "same_label_duplicate_rows_dropped": same_label_dropped,
        "conflicting_exact_window_rows_dropped": conflict_dropped,
    }


def _metadata(event: RevisionEvent) -> dict[str, float]:
    return {
        "log_pause_ms": math.log1p(event.pause_ms),
        "log_prefix_words": math.log1p(event.prefix_word_count),
        f"position_bucket={min(event.prefix_word_count // 50, 10)}": 1.0,
    }


def _endpoint(event: RevisionEvent) -> dict[str, float]:
    return {f"word={event.words[-1]}": 1.0}


def _bag(event: RevisionEvent) -> dict[str, float]:
    return {
        f"word={word}": float(count)
        for word, count in Counter(event.words).items()
    }


def _canonical(event: RevisionEvent) -> dict[str, float]:
    return {
        f"sorted_{index}={word}": 1.0
        for index, word in enumerate(sorted(event.words))
    }


def _ordered(event: RevisionEvent) -> dict[str, float]:
    return {
        f"position_{index}={word}": 1.0
        for index, word in enumerate(event.words)
    }


def _fold(event: RevisionEvent, salt: str, folds: int) -> int:
    digest = hashlib.sha256(
        f"{salt}:{event.writer_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % folds


def _aligned_probabilities(
    model: LogisticRegression,
    features: object,
    labels: tuple[int, ...],
) -> np.ndarray:
    probabilities = model.predict_proba(features)
    aligned = np.zeros((probabilities.shape[0], len(labels)), dtype=float)
    for source, label in enumerate(model.classes_):
        aligned[:, labels.index(int(label))] = probabilities[:, source]
    return aligned


def evaluate(
    events: Sequence[RevisionEvent],
    *,
    c_value: float = 0.01,
    folds: int = 5,
    salt: str = PROTOCOL_VERSION,
    bootstrap_samples: int = 2_000,
    labels: tuple[int, ...] = LABELS,
) -> dict[str, object]:
    if not events:
        raise ValueError("no eligible KLiCKe revision events")
    y = np.asarray([event.label for event in events], dtype=int)
    if set(y) != set(labels):
        raise ValueError("evaluation cohort must contain all destination labels")
    fold_ids = np.asarray([_fold(event, salt, folds) for event in events])
    names = (
        "prior",
        "metadata",
        "endpoint",
        "bag",
        "canonical",
        "ordered",
        "reverse",
    )
    predictions = {
        name: np.zeros((len(events), len(labels)), dtype=float)
        for name in names
    }
    views: dict[str, Callable[[RevisionEvent], dict[str, float]]] = {
        "metadata": _metadata,
        "endpoint": _endpoint,
        "bag": _bag,
        "canonical": _canonical,
        "ordered": _ordered,
    }

    for fold in range(folds):
        train = np.flatnonzero(fold_ids != fold)
        test = np.flatnonzero(fold_ids == fold)
        if not len(train) or not len(test):
            raise ValueError(f"empty deterministic fold {fold}")
        prior = np.bincount(
            y[train],
            minlength=max(labels) + 1,
        )[list(labels)].astype(float)
        predictions["prior"][test] = prior / prior.sum()
        for name, view in views.items():
            vectorizer = DictVectorizer()
            x_train = vectorizer.fit_transform(
                [view(events[index]) for index in train]
            )
            x_test = vectorizer.transform(
                [view(events[index]) for index in test]
            )
            model = LogisticRegression(
                C=c_value,
                max_iter=2_000,
                solver="lbfgs",
            )
            model.fit(x_train, y[train])
            predictions[name][test] = _aligned_probabilities(
                model,
                x_test,
                labels,
            )
            if name == "ordered":
                reversed_features = vectorizer.transform(
                    [
                        _ordered(
                            replace(
                                events[index],
                                words=tuple(reversed(events[index].words)),
                            )
                        )
                        for index in test
                    ]
                )
                predictions["reverse"][test] = _aligned_probabilities(
                    model,
                    reversed_features,
                    labels,
                )

    metrics: dict[str, dict[str, float]] = {}
    for name, probabilities in predictions.items():
        predicted = np.asarray(labels)[np.argmax(probabilities, axis=1)]
        metrics[name] = {
            "log_loss": float(log_loss(y, probabilities, labels=labels)),
            "balanced_accuracy": float(
                balanced_accuracy_score(y, predicted)
            ),
        }

    contrasts: dict[str, dict[str, float | int | None]] = {}
    true_columns = np.asarray([labels.index(int(label)) for label in y])
    row_indices = np.arange(len(events))
    row_losses = {
        name: -np.log(
            np.clip(
                probabilities[row_indices, true_columns],
                1e-12,
                1.0,
            )
        )
        for name, probabilities in predictions.items()
    }
    writer_indices: dict[str, np.ndarray] = {
        writer: np.asarray(
            [
                index
                for index, event in enumerate(events)
                if event.writer_id == writer
            ],
            dtype=int,
        )
        for writer in sorted({event.writer_id for event in events})
    }
    rng = np.random.default_rng(20_260_724)
    for control in ("endpoint", "bag", "canonical", "reverse"):
        differences = np.asarray(
            [
                float(
                    np.mean(
                        row_losses[control][indices]
                        - row_losses["ordered"][indices]
                    )
                )
                for indices in writer_indices.values()
            ]
        )
        if bootstrap_samples:
            draws = rng.choice(
                differences,
                size=(bootstrap_samples, len(differences)),
                replace=True,
            ).mean(axis=1)
            lower, upper = np.quantile(draws, [0.025, 0.975])
        else:
            lower = upper = None
        contrasts[f"{control}_minus_ordered"] = {
            "equal_writer_mean_log_loss_difference": float(
                differences.mean()
            ),
            "ci95_lower": None if lower is None else float(lower),
            "ci95_upper": None if upper is None else float(upper),
            "writers_positive": int(np.sum(differences > 0)),
            "writers_total": len(differences),
        }

    return {
        "events": len(events),
        "writers": len(writer_indices),
        "folds": folds,
        "c_value": c_value,
        "metrics": metrics,
        "equal_writer_bootstrap": contrasts,
    }


def cohort_sha256(events: Sequence[RevisionEvent]) -> str:
    digest = hashlib.sha256()
    for event in sorted(events, key=lambda item: item.event_hash):
        digest.update(
            (
                f"{event.event_hash}\x1f{event.window_hash}\x1f"
                f"{event.label}\n"
            ).encode("ascii")
        )
    return digest.hexdigest()


def _sum_diagnostics(
    aggregate: Counter[str],
    diagnostics: dict[str, int],
) -> None:
    aggregate.update(diagnostics)


def run(
    *,
    archive_path: str | Path,
    output_path: str | Path,
    window: int = DEFAULT_WINDOW,
    max_files: int | None = None,
    c_value: float = 0.01,
    folds: int = 5,
    bootstrap_samples: int = 2_000,
) -> dict[str, object]:
    archive_path = Path(archive_path)
    all_events: list[RevisionEvent] = []
    extraction: Counter[str] = Counter()
    with zipfile.ZipFile(archive_path) as archive:
        members = csv_members(archive)
        if max_files is not None:
            members = members[:max_files]
        for member in members:
            writer_id = Path(member).stem
            try:
                rows = read_log_rows(archive, member)
                events, diagnostics = extract_writer_events(
                    writer_id,
                    rows,
                    window=window,
                )
            except (csv.Error, ValueError):
                extraction["unreadable_files"] += 1
                continue
            extraction["files_read"] += 1
            _sum_diagnostics(extraction, diagnostics)
            all_events.extend(events)

    retained, deduplication = deduplicate_windows(all_events)
    if not retained:
        raise ValueError("KLiCKe extraction yielded no retained events")
    primary = evaluate(
        retained,
        c_value=c_value,
        folds=folds,
        bootstrap_samples=bootstrap_samples,
    )
    single_backspace = [
        event for event in retained if event.single_character_backspaces
    ]
    sensitivity = (
        evaluate(
            single_backspace,
            c_value=c_value,
            folds=folds,
            bootstrap_samples=bootstrap_samples,
        )
        if set(event.label for event in single_backspace) == set(LABELS)
        else None
    )
    sweep: dict[str, object] = {}
    for history in range(1, window + 1):
        truncated = [
            replace(event, words=event.words[-history:])
            for event in retained
        ]
        evaluated = evaluate(
            truncated,
            c_value=c_value,
            folds=folds,
            bootstrap_samples=0,
        )
        sweep[str(history)] = {
            name: metrics
            for name, metrics in evaluated["metrics"].items()
            if name in ("endpoint", "bag", "canonical", "ordered", "reverse")
        }

    label_counts = Counter(event.label for event in retained)
    writer_counts = Counter(event.writer_id for event in retained)
    result: dict[str, object] = {
        "protocol": {
            "version": PROTOCOL_VERSION,
            "anchor": (
                "document state immediately before the first operation in a "
                "consecutive trailing Remove/Cut burst"
            ),
            "target": "deleted lexical words: 2, 3, 4, or 5+",
            "window": f"final {window} lexical words before the burst",
            "complex_or_ambiguous_sessions": (
                "truncate at first Replace, Move, unknown activity, or "
                "reconstruction mismatch"
            ),
            "exact_window_deduplication": (
                "drop all conflicting-label windows; retain lowest event hash "
                "for same-label duplicates"
            ),
            "grouping": "five deterministic folds by writer",
            "raw_text_in_output": False,
            "article_doi": "10.17239/jowr-2025.17.01.02",
        },
        "source": {
            "archive": str(archive_path),
            "archive_sha256": sha256_file(archive_path),
            "max_files": max_files,
        },
        "counts": {
            "events_before_deduplication": len(all_events),
            "retained_events": len(retained),
            "retained_writers": len(writer_counts),
            "writers_with_multiple_events": int(
                sum(count > 1 for count in writer_counts.values())
            ),
            "label_counts": {
                str(label): label_counts[label] for label in LABELS
            },
            "single_character_backspace_events": len(single_backspace),
            "extraction": dict(sorted(extraction.items())),
            "deduplication": deduplication,
        },
        "cohort_sha256": cohort_sha256(retained),
        "primary": primary,
        "single_character_backspace_sensitivity": sensitivity,
        "fixed_cohort_window_sweep": sweep,
    }
    serialized = json.dumps(result, sort_keys=True)
    forbidden = ('"text_change"', '"words"', '"initial_text"', '"final_text"')
    if any(term in serialized.lower() for term in forbidden):
        raise AssertionError("aggregate artifact unexpectedly contains text")
    atomic_json(result, output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--c-value", type=float, default=0.01)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    args = parser.parse_args()
    result = run(
        archive_path=args.archive,
        output_path=args.output,
        window=args.window,
        max_files=args.max_files,
        c_value=args.c_value,
        folds=args.folds,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
