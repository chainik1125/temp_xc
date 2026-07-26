"""Build and evaluate the exact-token KLiCKe deletion-destination cohort.

The anchor is the final subject-model token immediately before a consecutive
trailing deletion burst. The primary target is the number of subject-model
tokens removed, capped only at a configurable upper destination. The original
2/3/4/5+ lexical destination is retained as a sensitivity target.

KLiCKe text and writer identifiers are used only in memory. The optional cohort
artifact contains model token IDs, target values, and cryptographic hashes; it
does not contain raw text or unhashed writer identifiers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold

from .klicke import (
    DEFAULT_WINDOW,
    PROTOCOL_VERSION as LEXICAL_PROTOCOL_VERSION,
    RevisionEvent,
    csv_members,
    extract_writer_events,
    read_log_rows,
    sha256_file,
)


PROTOCOL_VERSION = "klicke-trailing-deletion-model-token-v1"
DEFAULT_SUBJECT_TOKENIZER = "NousResearch/Meta-Llama-3.1-8B"
DEFAULT_HISTORY_TOKENS = 10
DEFAULT_TOKEN_CAP = 6
DEFAULT_MAX_MODEL_TOKENS = 2_048


@dataclass(frozen=True)
class TokenRevisionEvent:
    """Privacy-preserving model-token representation of one revision event."""

    event_hash: str
    writer_hash: str
    window_hash: str
    input_ids: tuple[int, ...]
    window_token_ids: tuple[int, ...]
    token_distance: int
    capped_token_label: int
    lexical_deleted_words: int
    lexical_label: int
    prefix_token_count: int
    special_tokens_added: int
    remove_actions: int
    single_character_backspaces: bool


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _token_window_hash(token_ids: Sequence[int]) -> str:
    return _sha256_text(",".join(str(int(value)) for value in token_ids))


def _token_ids(tokenizer, text: str) -> tuple[int, ...]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=False,
    )
    input_ids = encoded["input_ids"]
    if isinstance(input_ids, np.ndarray):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], (list, tuple, np.ndarray)):
        if len(input_ids) != 1:
            raise ValueError("tokenizer unexpectedly returned a batch")
        input_ids = input_ids[0]
    return tuple(int(value) for value in input_ids)


def _model_input_ids(
    tokenizer,
    text_token_ids: Sequence[int],
) -> tuple[int, ...]:
    builder = getattr(tokenizer, "build_inputs_with_special_tokens", None)
    if builder is None:
        return tuple(int(value) for value in text_token_ids)
    return tuple(
        int(value)
        for value in builder([int(item) for item in text_token_ids])
    )


def align_revision_event(
    tokenizer,
    event: RevisionEvent,
    *,
    history_tokens: int = DEFAULT_HISTORY_TOKENS,
    token_cap: int = DEFAULT_TOKEN_CAP,
    max_model_tokens: int = DEFAULT_MAX_MODEL_TOKENS,
) -> tuple[TokenRevisionEvent | None, str]:
    """Align a reconstructed trailing deletion to exact subject-model tokens."""

    if history_tokens < 1:
        raise ValueError("history_tokens must be positive")
    if token_cap < 1:
        raise ValueError("token_cap must be positive")
    if max_model_tokens < history_tokens:
        raise ValueError("max_model_tokens must cover the token window")
    if not event.preburst_text.startswith(event.postburst_text):
        return None, "raw_prefix_mismatch"

    pre_text_ids = _token_ids(tokenizer, event.preburst_text)
    post_text_ids = _token_ids(tokenizer, event.postburst_text)
    if not pre_text_ids:
        return None, "empty_preburst_tokenization"
    if len(pre_text_ids) < history_tokens:
        return None, "insufficient_preburst_tokens"
    if (
        len(post_text_ids) > len(pre_text_ids)
        or pre_text_ids[: len(post_text_ids)] != post_text_ids
    ):
        return None, "boundary_retokenized"

    token_distance = len(pre_text_ids) - len(post_text_ids)
    if token_distance < 1:
        return None, "no_complete_model_token_removed"
    input_ids = _model_input_ids(tokenizer, pre_text_ids)
    special_tokens_added = len(input_ids) - len(pre_text_ids)
    if special_tokens_added < 0:
        return None, "special_token_builder_removed_text_tokens"
    if len(input_ids) > max_model_tokens:
        return None, "preburst_exceeds_model_limit_after_special_tokens"
    if input_ids[-history_tokens:] != pre_text_ids[-history_tokens:]:
        return None, "trailing_special_token_changed_anchor"

    window_token_ids = pre_text_ids[-history_tokens:]
    window_hash = _token_window_hash(window_token_ids)
    writer_hash = _sha256_text(
        f"{PROTOCOL_VERSION}:writer:{event.writer_id}"
    )
    event_hash = _sha256_text(
        (
            f"{PROTOCOL_VERSION}:event:{writer_hash}:{event.row_index}:"
            f"{window_hash}:{token_distance}:{event.label}"
        )
    )
    return (
        TokenRevisionEvent(
            event_hash=event_hash,
            writer_hash=writer_hash,
            window_hash=window_hash,
            input_ids=input_ids,
            window_token_ids=window_token_ids,
            token_distance=token_distance,
            capped_token_label=min(token_distance, token_cap),
            lexical_deleted_words=event.deleted_words,
            lexical_label=event.label,
            prefix_token_count=len(input_ids),
            special_tokens_added=special_tokens_added,
            remove_actions=event.remove_actions,
            single_character_backspaces=event.single_character_backspaces,
        ),
        "aligned",
    )


def build_token_events(
    tokenizer,
    events: Sequence[RevisionEvent],
    *,
    history_tokens: int = DEFAULT_HISTORY_TOKENS,
    token_cap: int = DEFAULT_TOKEN_CAP,
    max_model_tokens: int = DEFAULT_MAX_MODEL_TOKENS,
) -> tuple[list[TokenRevisionEvent], dict[str, int]]:
    """Align revision events and report every inclusion/exclusion reason."""

    aligned: list[TokenRevisionEvent] = []
    diagnostics: Counter[str] = Counter()
    for event in events:
        token_event, reason = align_revision_event(
            tokenizer,
            event,
            history_tokens=history_tokens,
            token_cap=token_cap,
            max_model_tokens=max_model_tokens,
        )
        diagnostics[reason] += 1
        if token_event is not None:
            aligned.append(token_event)
    diagnostics["candidate_revision_events"] = len(events)
    return aligned, dict(sorted(diagnostics.items()))


def deduplicate_token_windows(
    events: Sequence[TokenRevisionEvent],
) -> tuple[list[TokenRevisionEvent], dict[str, int]]:
    """Globally deduplicate exact token windows for both registered targets."""

    groups: dict[tuple[int, ...], list[TokenRevisionEvent]] = defaultdict(list)
    for event in events:
        groups[event.window_token_ids].append(event)

    retained: list[TokenRevisionEvent] = []
    same_target_duplicates = 0
    conflicting_rows = 0
    for group in groups.values():
        target_pairs = {
            (event.token_distance, event.lexical_label) for event in group
        }
        if len(target_pairs) != 1:
            conflicting_rows += len(group)
            continue
        retained.append(min(group, key=lambda event: event.event_hash))
        same_target_duplicates += len(group) - 1
    retained.sort(key=lambda event: event.event_hash)
    return retained, {
        "exact_token_window_groups": len(groups),
        "same_target_duplicate_rows_dropped": same_target_duplicates,
        "conflicting_target_rows_dropped": conflicting_rows,
        "retained_rows": len(retained),
    }


def _target_value(event: TokenRevisionEvent, target: str) -> int:
    if target == "capped_token_distance":
        return event.capped_token_label
    if target == "lexical_destination":
        return event.lexical_label
    raise ValueError(f"unknown target: {target}")


def _tokens(event: TokenRevisionEvent, history_tokens: int) -> tuple[int, ...]:
    if not 1 <= history_tokens <= len(event.window_token_ids):
        raise ValueError("history_tokens outside the frozen cohort window")
    return event.window_token_ids[-history_tokens:]


def _endpoint(
    event: TokenRevisionEvent,
    history_tokens: int,
) -> dict[str, float]:
    token = _tokens(event, history_tokens)[-1]
    return {f"token={token}": 1.0}


def _bag(
    event: TokenRevisionEvent,
    history_tokens: int,
) -> dict[str, float]:
    return {
        f"token={token}": float(count)
        for token, count in Counter(_tokens(event, history_tokens)).items()
    }


def _canonical(
    event: TokenRevisionEvent,
    history_tokens: int,
) -> dict[str, float]:
    return {
        f"sorted_{index}=token_{token}": 1.0
        for index, token in enumerate(
            sorted(_tokens(event, history_tokens))
        )
    }


def _ordered(
    event: TokenRevisionEvent,
    history_tokens: int,
    *,
    reverse: bool = False,
) -> dict[str, float]:
    tokens = _tokens(event, history_tokens)
    if reverse:
        tokens = tuple(reversed(tokens))
    return {
        f"position_{index}=token_{token}": 1.0
        for index, token in enumerate(tokens)
    }


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


def _grouped_splits(
    y: np.ndarray,
    groups: np.ndarray,
    *,
    requested_folds: int,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    if requested_folds < 2:
        raise ValueError("folds must be at least two")
    labels = np.unique(y)
    if len(labels) < 2:
        raise ValueError("target must contain at least two observed labels")
    groups_per_label = [
        len(np.unique(groups[y == label])) for label in labels
    ]
    folds = min(
        requested_folds,
        len(np.unique(groups)),
        min(groups_per_label),
    )
    if folds < 2:
        raise ValueError(
            "each observed label must occur in at least two writer groups"
        )
    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )
    splits = list(splitter.split(np.zeros(len(y)), y, groups))
    for train, test in splits:
        if set(groups[train]).intersection(groups[test]):
            raise AssertionError("writer group leaked across a fold")
        if len(np.unique(y[train])) < 2:
            raise ValueError("a training fold contains fewer than two labels")
    return splits, folds


def _writer_bootstrap_contrasts(
    *,
    events: Sequence[TokenRevisionEvent],
    y: np.ndarray,
    labels: tuple[int, ...],
    predictions: dict[str, np.ndarray],
    samples: int,
) -> dict[str, dict[str, float | int | None]]:
    true_columns = np.asarray([labels.index(int(label)) for label in y])
    rows = np.arange(len(events))
    row_losses = {
        name: -np.log(
            np.clip(
                probabilities[rows, true_columns],
                1e-12,
                1.0,
            )
        )
        for name, probabilities in predictions.items()
    }
    index_lists: dict[str, list[int]] = defaultdict(list)
    for index, event in enumerate(events):
        index_lists[event.writer_hash].append(index)
    writer_indices = {
        writer: np.asarray(indices, dtype=int)
        for writer, indices in sorted(index_lists.items())
    }
    rng = np.random.default_rng(20_260_726)
    contrasts: dict[str, dict[str, float | int | None]] = {}
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
        if samples:
            draws = rng.choice(
                differences,
                size=(samples, len(differences)),
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
    return contrasts


def evaluate_token_views(
    events: Sequence[TokenRevisionEvent],
    *,
    target: str,
    history_tokens: int,
    c_value: float = 0.01,
    folds: int = 5,
    bootstrap_samples: int = 0,
    seed: int = 20_260_726,
) -> dict[str, object]:
    """Evaluate lexical token-ID controls with writer-grouped folds."""

    if not events:
        raise ValueError("no token-aligned revision events")
    y = np.asarray(
        [_target_value(event, target) for event in events],
        dtype=int,
    )
    labels = tuple(int(value) for value in sorted(np.unique(y)))
    groups = np.asarray([event.writer_hash for event in events])
    splits, effective_folds = _grouped_splits(
        y,
        groups,
        requested_folds=folds,
        seed=seed,
    )
    names = (
        "prior",
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
    views: dict[
        str,
        Callable[[TokenRevisionEvent, int], dict[str, float]],
    ] = {
        "endpoint": _endpoint,
        "bag": _bag,
        "canonical": _canonical,
        "ordered": _ordered,
    }

    for train, test in splits:
        prior = np.asarray(
            [np.sum(y[train] == label) for label in labels],
            dtype=float,
        )
        predictions["prior"][test] = prior / prior.sum()
        for name, view in views.items():
            vectorizer = DictVectorizer()
            x_train = vectorizer.fit_transform(
                [view(events[index], history_tokens) for index in train]
            )
            x_test = vectorizer.transform(
                [view(events[index], history_tokens) for index in test]
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
                            events[index],
                            history_tokens,
                            reverse=True,
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
        if not np.isfinite(probabilities).all():
            raise RuntimeError(f"{name} contains non-finite probabilities")
        predicted = np.asarray(labels)[np.argmax(probabilities, axis=1)]
        metrics[name] = {
            "log_loss": float(log_loss(y, probabilities, labels=labels)),
            "balanced_accuracy": float(
                balanced_accuracy_score(y, predicted)
            ),
        }

    return {
        "events": len(events),
        "writers": len(np.unique(groups)),
        "labels_observed": list(labels),
        "folds_requested": folds,
        "folds_effective": effective_folds,
        "history_tokens": history_tokens,
        "c_value": c_value,
        "metrics": metrics,
        "equal_writer_bootstrap": _writer_bootstrap_contrasts(
            events=events,
            y=y,
            labels=labels,
            predictions=predictions,
            samples=bootstrap_samples,
        ),
    }


def evaluate_target_sweep(
    events: Sequence[TokenRevisionEvent],
    *,
    target: str,
    max_history_tokens: int,
    c_value: float = 0.01,
    folds: int = 5,
    bootstrap_samples: int = 2_000,
) -> dict[str, object]:
    """Evaluate every trailing token-window length on one frozen cohort."""

    labels = sorted({_target_value(event, target) for event in events})
    if len(labels) < 2:
        return {
            "status": "skipped",
            "reason": "fewer than two observed labels",
            "labels_observed": labels,
            "fixed_cohort_window_sweep": {},
        }
    sweep = {}
    for history_tokens in range(1, max_history_tokens + 1):
        sweep[str(history_tokens)] = evaluate_token_views(
            events,
            target=target,
            history_tokens=history_tokens,
            c_value=c_value,
            folds=folds,
            bootstrap_samples=(
                bootstrap_samples
                if history_tokens == max_history_tokens
                else 0
            ),
        )
    return {
        "status": "complete",
        "labels_observed": labels,
        "fixed_cohort_window_sweep": sweep,
    }


def cohort_sha256(events: Sequence[TokenRevisionEvent]) -> str:
    digest = hashlib.sha256()
    for event in sorted(events, key=lambda item: item.event_hash):
        digest.update(
            (
                f"{event.event_hash}\x1f{event.writer_hash}\x1f"
                f"{event.window_hash}\x1f{event.token_distance}\x1f"
                f"{event.lexical_label}\n"
            ).encode("ascii")
        )
    return digest.hexdigest()


def cohort_records(
    events: Sequence[TokenRevisionEvent],
) -> list[dict[str, object]]:
    records = []
    for event in events:
        record = asdict(event)
        record["input_ids"] = list(event.input_ids)
        record["window_token_ids"] = list(event.window_token_ids)
        records.append(record)
    forbidden = {
        "writer_id",
        "preburst_text",
        "postburst_text",
        "words",
        "text",
    }
    if any(forbidden.intersection(record) for record in records):
        raise AssertionError("token cohort contains a raw-text field")
    return records


def write_cohort(
    events: Sequence[TokenRevisionEvent],
    path: str | Path,
) -> None:
    destination = Path(path)
    frame = pd.DataFrame(cohort_records(events))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, destination)


def _atomic_json(payload: object, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def _load_revision_events(
    archive_path: Path,
    *,
    lexical_window: int,
    max_files: int | None,
) -> tuple[list[RevisionEvent], dict[str, int]]:
    all_events: list[RevisionEvent] = []
    diagnostics: Counter[str] = Counter()
    with zipfile.ZipFile(archive_path) as archive:
        members = csv_members(archive)
        if max_files is not None:
            members = members[:max_files]
        for member in members:
            writer_id = Path(member).stem
            try:
                events, writer_diagnostics = extract_writer_events(
                    writer_id,
                    read_log_rows(archive, member),
                    window=lexical_window,
                )
            except (csv.Error, ValueError):
                diagnostics["unreadable_files"] += 1
                continue
            diagnostics["files_read"] += 1
            diagnostics.update(writer_diagnostics)
            all_events.extend(events)
    return all_events, dict(sorted(diagnostics.items()))


def _count_map(values: Sequence[int]) -> dict[str, int]:
    counts = Counter(int(value) for value in values)
    return {str(value): counts[value] for value in sorted(counts)}


def run_with_tokenizer(
    *,
    tokenizer,
    archive_path: str | Path,
    manifest_output: str | Path,
    cohort_output: str | Path | None = None,
    tokenizer_name: str = DEFAULT_SUBJECT_TOKENIZER,
    tokenizer_revision: str | None = None,
    history_tokens: int = DEFAULT_HISTORY_TOKENS,
    token_cap: int = DEFAULT_TOKEN_CAP,
    max_model_tokens: int = DEFAULT_MAX_MODEL_TOKENS,
    lexical_window: int = DEFAULT_WINDOW,
    max_files: int | None = None,
    c_value: float = 0.01,
    folds: int = 5,
    bootstrap_samples: int = 2_000,
) -> dict[str, object]:
    archive_path = Path(archive_path)
    revision_events, reconstruction = _load_revision_events(
        archive_path,
        lexical_window=lexical_window,
        max_files=max_files,
    )
    aligned, alignment = build_token_events(
        tokenizer,
        revision_events,
        history_tokens=history_tokens,
        token_cap=token_cap,
        max_model_tokens=max_model_tokens,
    )
    retained, deduplication = deduplicate_token_windows(aligned)
    if not retained:
        raise ValueError("token alignment yielded no retained events")

    primary = evaluate_target_sweep(
        retained,
        target="capped_token_distance",
        max_history_tokens=history_tokens,
        c_value=c_value,
        folds=folds,
        bootstrap_samples=bootstrap_samples,
    )
    lexical = evaluate_target_sweep(
        retained,
        target="lexical_destination",
        max_history_tokens=history_tokens,
        c_value=c_value,
        folds=folds,
        bootstrap_samples=bootstrap_samples,
    )
    if cohort_output is not None:
        write_cohort(retained, cohort_output)

    writer_counts = Counter(event.writer_hash for event in retained)
    tokenizer_init = getattr(tokenizer, "init_kwargs", {})
    resolved_tokenizer_revision = (
        tokenizer_revision
        or getattr(tokenizer, "_commit_hash", None)
        or (
            tokenizer_init.get("_commit_hash")
            if isinstance(tokenizer_init, dict)
            else None
        )
    )
    result: dict[str, object] = {
        "protocol": {
            "version": PROTOCOL_VERSION,
            "source_lexical_protocol": LEXICAL_PROTOCOL_VERSION,
            "anchor": (
                "final subject-model token immediately before a consecutive "
                "trailing deletion burst"
            ),
            "primary_target": (
                f"subject-model tokens removed, capped at {token_cap} "
                f"({token_cap} means {token_cap}+)"
            ),
            "sensitivity_target": "original deleted lexical words: 2/3/4/5+",
            "tokenizer": tokenizer_name,
            "tokenizer_revision_requested": tokenizer_revision,
            "tokenizer_revision_resolved": resolved_tokenizer_revision,
            "tokenizer_class": tokenizer.__class__.__name__,
            "tokenizer_vocab_size": (
                len(tokenizer) if hasattr(tokenizer, "__len__") else None
            ),
            "history_tokens": history_tokens,
            "token_cap": token_cap,
            "max_model_tokens": max_model_tokens,
            "lexical_window": lexical_window,
            "exact_prefix_check": (
                "post-deletion IDs must equal an exact prefix of pre-deletion "
                "IDs; boundary-retokenized events are excluded"
            ),
            "model_input_special_tokens": (
                "tokenizer.build_inputs_with_special_tokens is applied after "
                "the exact text-token prefix check; the final H tokens must "
                "remain the text-token window"
            ),
            "exact_window_deduplication": (
                "globally drop windows with conflicting exact token-distance "
                "or lexical targets; retain lowest event hash otherwise"
            ),
            "grouping": (
                "stratified writer-grouped folds; observed labels only"
            ),
            "raw_text_in_output": False,
            "unhashed_writer_identifiers_in_output": False,
            "article_doi": "10.17239/jowr-2025.17.01.02",
        },
        "source": {
            "archive": str(archive_path),
            "archive_sha256": sha256_file(archive_path),
            "max_files": max_files,
        },
        "counts": {
            "revision_events_before_token_alignment": len(revision_events),
            "token_aligned_before_deduplication": len(aligned),
            "retained_events": len(retained),
            "retained_writers": len(writer_counts),
            "writers_with_multiple_events": int(
                sum(count > 1 for count in writer_counts.values())
            ),
            "token_distance_exact_before_deduplication": _count_map(
                [event.token_distance for event in aligned]
            ),
            "token_distance_exact": _count_map(
                [event.token_distance for event in retained]
            ),
            "capped_token_distance_before_deduplication": _count_map(
                [event.capped_token_label for event in aligned]
            ),
            "capped_token_distance": _count_map(
                [event.capped_token_label for event in retained]
            ),
            "lexical_deleted_words_exact": _count_map(
                [event.lexical_deleted_words for event in retained]
            ),
            "lexical_destination": _count_map(
                [event.lexical_label for event in retained]
            ),
            "single_character_backspace_events": sum(
                event.single_character_backspaces for event in retained
            ),
            "special_tokens_added": _count_map(
                [event.special_tokens_added for event in retained]
            ),
            "reconstruction": reconstruction,
            "token_alignment": alignment,
            "deduplication": deduplication,
        },
        "cohort_sha256": cohort_sha256(retained),
        "cohort": (
            {
                "path": str(Path(cohort_output).resolve()),
                "sha256": sha256_file(cohort_output),
            }
            if cohort_output is not None
            else None
        ),
        "primary_capped_token_distance": primary,
        "original_lexical_sensitivity": lexical,
    }
    serialized = json.dumps(result, sort_keys=True).lower()
    forbidden_fields = (
        '"writer_id"',
        '"preburst_text"',
        '"postburst_text"',
        '"words"',
    )
    if any(field in serialized for field in forbidden_fields):
        raise AssertionError("manifest unexpectedly contains a raw-text field")
    _atomic_json(result, manifest_output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--cohort-output", type=Path)
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_SUBJECT_TOKENIZER,
    )
    parser.add_argument("--tokenizer-revision")
    parser.add_argument(
        "--history-tokens",
        type=int,
        default=DEFAULT_HISTORY_TOKENS,
    )
    parser.add_argument(
        "--token-cap",
        type=int,
        default=DEFAULT_TOKEN_CAP,
    )
    parser.add_argument(
        "--max-model-tokens",
        type=int,
        default=DEFAULT_MAX_MODEL_TOKENS,
    )
    parser.add_argument(
        "--lexical-window",
        type=int,
        default=DEFAULT_WINDOW,
    )
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--c-value", type=float, default=0.01)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=args.tokenizer_revision,
        local_files_only=Path(args.tokenizer).exists(),
        trust_remote_code=True,
        use_fast=True,
    )
    result = run_with_tokenizer(
        tokenizer=tokenizer,
        archive_path=args.archive,
        manifest_output=args.manifest_output,
        cohort_output=args.cohort_output,
        tokenizer_name=args.tokenizer,
        tokenizer_revision=args.tokenizer_revision,
        history_tokens=args.history_tokens,
        token_cap=args.token_cap,
        max_model_tokens=args.max_model_tokens,
        lexical_window=args.lexical_window,
        max_files=args.max_files,
        c_value=args.c_value,
        folds=args.folds,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
