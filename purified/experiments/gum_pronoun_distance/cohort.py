"""Build the pinned GUM T=5 personal-pronoun distance cohort.

The event window ends at the personal pronoun.  The target is the exact
model-token distance from the final subtoken of the direct antecedent mention
to the final subtoken of the pronoun: d in {2, 3, 4}.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

PROTOCOL_VERSION = "gum-personal-pronoun-antecedent-distance-t5-v1"
GUM_REPO = "https://github.com/amir-zeldes/gum.git"
GUM_REVISION = "22fdf87f9c71c96bcc771461d06e689b1f90020d"
TOKENIZER_REPO = "NousResearch/Meta-Llama-3.1-8B"
TOKENIZER_REVISION = "1f47e50cdbe801ad8a5174156ec3a0655108fb9f"
WINDOW_TOKENS = 5
LABELS = (2, 3, 4)
EXPECTED_ROWS = 2_638
EXPECTED_DOCUMENTS = 269
EXPECTED_LABEL_COUNTS = {2: 880, 3: 938, 4: 820}
EXPECTED_BALANCED_ROWS = 2_016
EXPECTED_SEMANTIC_SHA256 = (
    "c08fddc183eccbf72667ef3780a8bf9a2ec9be25729f9efbc1f85bf5e740c7da"
)
EXPECTED_BALANCED_CANONICAL_SHA256 = (
    "9f7e0b535b7179d68c3191a23c6c4ea52cf1b3a196a60930c3639b20a4d58113"
)
EXPECTED_SOURCE_INVENTORY = {
    "files": 603,
    "bytes": 54_027_366,
    "inventory_sha256": (
        "4b5260c0ec63ef3b6e9b0e8f521df0df7738bbc6d6110265b194e1a0a4336144"
    ),
}
EXPECTED_TOKENIZER_FILES = {
    "config.json": "54acfad3cffe057640904ca8a1e83525e6551c70c7a04c641f5a9eda0bbf64bd",
    "special_tokens_map.json": (
        "462d91939dbc37178aa5a3eae7068d1990ccc92e09f288cc71f42cdf139d69cc"
    ),
    "tokenizer.json": (
        "76e48799b099d43365bd24ccd8ecc5aedac831718da780552f03b0a6eb4412aa"
    ),
    "tokenizer_config.json": (
        "8004530facf809ac432114de2a4dcc65fcb632da5ec16d666091aeb6a2ee444a"
    ),
}
AUDITED_PRONOUNS = frozenset(
    """
    he him his himself she her hers herself it its itself
    they them their theirs themselves we us our ours ourselves
    i me my mine myself you your yours yourself yourselves
    who whom whose which that this these those
    """.split()
)
DIRECT_RELATIONS = frozenset({"ana", "coref"})
ENTITY_PATTERN = re.compile(r"([^|]+?)\[([^\]]+)\]")
POINTER_PATTERN = re.compile(r"([^\[]+)\[([^_\]]+)_([^\]]+)\]")

REQUIRED_COLUMNS = {
    "event_hash",
    "document",
    "split",
    "genre",
    "relation_type",
    "pronoun",
    "target_ud_feats",
    "distance",
    "document_input_ids",
    "prefix_input_ids",
    "window_token_ids",
    "window_hash",
    "source_model_index",
    "target_model_index",
    "source_row_id",
    "target_row_id",
    "source_entity_id",
    "target_entity_id",
    "source_entity_type",
    "source_infstat",
    "balanced_sensitivity",
}
SEMANTIC_COLUMNS = (
    "event_hash",
    "document",
    "split",
    "genre",
    "relation_type",
    "pronoun",
    "target_ud_feats",
    "distance",
    "window_token_ids",
    "window_hash",
    "prefix_input_ids",
    "source_model_index",
    "target_model_index",
    "source_row_id",
    "target_row_id",
    "source_entity_id",
    "target_entity_id",
    "balanced_sensitivity",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _hash_strings(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _event_hash(record: dict[str, object]) -> str:
    keys = (
        "document",
        "source_row_id",
        "target_row_id",
        "source_entity_id",
        "target_entity_id",
        "relation_type",
        "distance",
    )
    payload = {key: record[key] for key in keys}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _window_hash(token_ids: Sequence[int]) -> str:
    return hashlib.sha256(
        ",".join(str(int(value)) for value in token_ids).encode()
    ).hexdigest()


def semantic_sha256(frame: pd.DataFrame) -> str:
    """Hash every field that defines the task input, target, and native edge."""

    digest = hashlib.sha256()
    integer_fields = {"distance", "source_model_index", "target_model_index"}
    token_fields = {"window_token_ids", "prefix_input_ids"}
    for row in frame.itertuples(index=False):
        payload = {}
        for field in SEMANTIC_COLUMNS:
            value = getattr(row, field)
            if field in token_fields:
                payload[field] = [int(item) for item in value]
            elif field in integer_fields:
                payload[field] = int(value)
            elif field == "balanced_sensitivity":
                payload[field] = bool(value)
            else:
                payload[field] = str(value)
        digest.update(
            (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def parse_splits(path: str | Path) -> dict[str, str]:
    """Parse the source-native train/dev/test/test2 document membership."""

    split_by_document: dict[str, str] = {}
    current: str | None = None
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if raw.startswith("## "):
            candidate = raw[3:].strip()
            current = (
                candidate if candidate in {"train", "dev", "test", "test2"} else None
            )
            continue
        if current is None or not raw.startswith("  * "):
            continue
        document = raw[4:].strip()
        if document in split_by_document:
            raise ValueError(f"duplicate GUM split membership: {document}")
        split_by_document[document] = current
    if len(split_by_document) != 301:
        raise ValueError(
            f"expected 301 GUM split documents, got {len(split_by_document)}"
        )
    return split_by_document


def _parse_ud(path: Path) -> tuple[list[list[str]], dict[str, str]]:
    tokens: list[list[str]] = []
    metadata: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.startswith("# meta::"):
            key, value = raw[8:].split(" = ", 1)
            metadata[key] = value
        if not raw or raw.startswith("#"):
            continue
        columns = raw.split("\t")
        if len(columns) != 10:
            raise ValueError(f"malformed CoNLL-U row in {path}")
        if "-" in columns[0] or "." in columns[0]:
            continue
        tokens.append(columns)
    return tokens, metadata


def _parse_tsv(path: Path) -> tuple[list[list[str]], dict[str, list[int]]]:
    rows: list[list[str]] = []
    mentions: dict[str, list[int]] = defaultdict(list)
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw or raw.startswith("#"):
            continue
        columns = raw.split("\t")
        if len(columns) < 10:
            raise ValueError(f"malformed WebAnno TSV row in {path}")
        row_index = len(rows)
        rows.append(columns)
        for _entity_type, entity_id in ENTITY_PATTERN.findall(columns[3]):
            mentions[entity_id].append(row_index)
    return rows, dict(mentions)


def _document_text_and_spans(
    ud_tokens: Sequence[Sequence[str]],
) -> tuple[str, list[tuple[int, int]]]:
    pieces: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for columns in ud_tokens:
        form = columns[1]
        start = cursor
        end = start + len(form)
        spans.append((start, end))
        pieces.append(form)
        cursor = end
        misc = set() if columns[9] == "_" else set(columns[9].split("|"))
        if "SpaceAfter=No" not in misc:
            pieces.append(" ")
            cursor += 1
    text = "".join(pieces).rstrip()
    return text, spans


def _ud_features(columns: Sequence[str]) -> dict[str, str]:
    if columns[5] == "_":
        return {}
    return dict(part.split("=", 1) for part in columns[5].split("|"))


def _span_value(
    rows: Sequence[Sequence[str]],
    span: Sequence[int],
    *,
    column: int,
    entity_id: str,
) -> tuple[str | None, str]:
    values: set[str] = set()
    for row_index in span:
        alternatives = rows[row_index][column].split("|")
        for alternative in alternatives:
            match = re.fullmatch(r"(.+)\[([^\]]+)\]", alternative)
            if match and match.group(2) == entity_id:
                values.add(match.group(1))
    if len(values) == 1:
        return next(iter(values)), "unique"
    if not values:
        return None, "missing"
    return None, "ambiguous"


def _iter_direct_edges(
    rows: Sequence[Sequence[str]],
    mentions: dict[str, list[int]],
) -> Iterable[dict[str, object]]:
    row_lookup = {row[0]: index for index, row in enumerate(rows)}
    for relation_index, columns in enumerate(rows):
        if columns[8] == "_":
            continue
        type_groups = columns[8].split("|")
        pointer_groups = columns[9].split("|")
        if len(type_groups) != len(pointer_groups):
            raise ValueError("parallel WebAnno relation columns disagree")
        for type_group, pointer_group in zip(type_groups, pointer_groups):
            types = type_group.split(";")
            pointers = pointer_group.split(";")
            if len(types) != len(pointers):
                if any(value in DIRECT_RELATIONS for value in types):
                    raise ValueError("ambiguous direct relation encoding")
                continue
            for relation_type, pointer in zip(types, pointers):
                if relation_type not in DIRECT_RELATIONS:
                    continue
                match = POINTER_PATTERN.fullmatch(pointer)
                if match is None:
                    raise ValueError(f"malformed direct relation pointer: {pointer}")
                target_row_id, target_entity_id, source_entity_id = match.groups()
                if (
                    target_row_id not in row_lookup
                    or target_entity_id not in mentions
                    or source_entity_id not in mentions
                ):
                    raise ValueError("direct relation points outside known mentions")
                source_span = mentions[source_entity_id]
                target_span = mentions[target_entity_id]
                target_index = row_lookup[target_row_id]
                if relation_index not in source_span or target_index not in target_span:
                    raise ValueError("direct relation row lies outside its entity span")
                yield {
                    "relation_type": relation_type,
                    "relation_index": relation_index,
                    "source_entity_id": source_entity_id,
                    "target_entity_id": target_entity_id,
                    "source_span": source_span,
                    "target_span": target_span,
                    "target_index": target_index,
                }


def _source_inventory(gum_root: Path) -> dict[str, object]:
    paths = [
        gum_root / "splits.md",
        *sorted((gum_root / "coref/gum/tsv").glob("*.tsv")),
        *sorted((gum_root / "dep").glob("*.conllu")),
    ]
    records = [
        {
            "path": str(path.relative_to(gum_root)),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    ]
    return {
        "files": len(records),
        "bytes": int(sum(int(record["size_bytes"]) for record in records)),
        "inventory_sha256": _hash_strings(
            f"{record['path']}\t{record['sha256']}" for record in records
        ),
    }


def _balanced_rows(events: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    by_pronoun_and_label: dict[str, dict[int, list[dict[str, object]]]] = {}
    for event in events:
        pronoun = str(event["pronoun"])
        if pronoun not in by_pronoun_and_label:
            by_pronoun_and_label[pronoun] = {label: [] for label in LABELS}
        by_pronoun_and_label[pronoun][int(event["distance"])].append(event)
    selected: list[dict[str, object]] = []
    for _pronoun, groups in by_pronoun_and_label.items():
        minimum = min(len(groups[label]) for label in LABELS)
        if minimum == 0:
            continue
        for label in LABELS:
            selected.extend(groups[label][:minimum])
    return selected


def _balanced_canonical_sha256(events: Sequence[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for event in events:
        row = {
            "doc": str(event["document"]),
            "target_model": int(event["target_model_index"]),
            "distance": int(event["distance"]),
            "pronoun": str(event["pronoun"]),
            "relation_type": str(event["relation_type"]),
            "window": [int(value) for value in event["window_token_ids"]],
        }
        digest.update(
            (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def build_cohort(
    gum_root: str | Path,
    tokenizer,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Construct the exact primary and required balanced-sensitivity cohorts."""

    gum_root = Path(gum_root)
    splits = parse_splits(gum_root / "splits.md")
    raw_events: list[dict[str, object]] = []
    exclusions: Counter[str] = Counter()

    for tsv_path in sorted((gum_root / "coref/gum/tsv").glob("*.tsv")):
        document = tsv_path.stem
        if document not in splits:
            raise ValueError(f"GUM document missing from splits.md: {document}")
        ud_path = gum_root / "dep" / f"{document}.conllu"
        if not ud_path.is_file():
            raise FileNotFoundError(f"missing matching UD document: {ud_path}")
        rows, mentions = _parse_tsv(tsv_path)
        ud_tokens, metadata = _parse_ud(ud_path)
        if [row[2] for row in rows] != [columns[1] for columns in ud_tokens]:
            raise ValueError(f"TSV and UD token sequences differ for {document}")
        text, gum_character_spans = _document_text_and_spans(ud_tokens)
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        input_ids = [int(value) for value in encoded["input_ids"]]
        endpoint_lookup: dict[int, int] = {}
        for model_index, (_start, end) in enumerate(encoded["offset_mapping"]):
            endpoint_lookup[int(end)] = model_index

        for edge in _iter_direct_edges(rows, mentions):
            source_span = list(edge["source_span"])
            target_span = list(edge["target_span"])
            if source_span != list(range(min(source_span), max(source_span) + 1)):
                exclusions["noncontiguous_source_span"] += 1
                continue
            if target_span != list(range(min(target_span), max(target_span) + 1)):
                exclusions["noncontiguous_target_span"] += 1
                continue
            if len(target_span) != 1:
                exclusions["multi_token_destination"] += 1
                continue
            target_index = int(edge["target_index"])
            if target_index <= max(source_span):
                exclusions["noncausal_edge"] += 1
                continue
            target_ud = ud_tokens[target_index]
            pronoun = rows[target_index][2].casefold()
            features = _ud_features(target_ud)
            if pronoun not in AUDITED_PRONOUNS:
                exclusions["outside_audited_lexicon"] += 1
                continue
            if target_ud[3] != "PRON" or features.get("PronType") != "Prs":
                exclusions["not_personal_pronoun_ud"] += 1
                continue
            source_end_character = gum_character_spans[max(source_span)][1]
            target_end_character = gum_character_spans[target_index][1]
            if source_end_character not in endpoint_lookup:
                exclusions["source_not_subtoken_boundary"] += 1
                continue
            if target_end_character not in endpoint_lookup:
                exclusions["target_not_subtoken_boundary"] += 1
                continue
            source_model = endpoint_lookup[source_end_character]
            target_model = endpoint_lookup[target_end_character]
            distance = target_model - source_model
            if distance <= 0:
                exclusions["nonpositive_model_distance"] += 1
                continue
            if distance not in LABELS:
                exclusions["outside_t5_label_support"] += 1
                continue
            if target_model < WINDOW_TOKENS - 1:
                exclusions["insufficient_prefix"] += 1
                continue
            window = input_ids[target_model - WINDOW_TOKENS + 1 : target_model + 1]
            source_type, source_type_status = _span_value(
                rows,
                source_span,
                column=3,
                entity_id=str(edge["source_entity_id"]),
            )
            source_infstat, source_infstat_status = _span_value(
                rows,
                source_span,
                column=4,
                entity_id=str(edge["source_entity_id"]),
            )
            event: dict[str, object] = {
                "document": document,
                "split": splits[document],
                "genre": metadata.get("genre", document.split("_")[1]),
                "relation_type": str(edge["relation_type"]),
                "pronoun": pronoun,
                "target_ud_feats": target_ud[5],
                "distance": distance,
                "document_input_ids": input_ids,
                "prefix_input_ids": input_ids[: target_model + 1],
                "window_token_ids": window,
                "window_hash": _window_hash(window),
                "source_model_index": source_model,
                "target_model_index": target_model,
                "source_row_id": rows[int(edge["relation_index"])][0],
                "target_row_id": rows[target_index][0],
                "source_entity_id": str(edge["source_entity_id"]),
                "target_entity_id": str(edge["target_entity_id"]),
                "source_entity_type": source_type,
                "source_entity_type_status": source_type_status,
                "source_infstat": source_infstat,
                "source_infstat_status": source_infstat_status,
                "balanced_sensitivity": False,
            }
            event["event_hash"] = _event_hash(event)
            raw_events.append(event)

    by_window: dict[tuple[int, ...], list[dict[str, object]]] = defaultdict(list)
    for event in raw_events:
        by_window[tuple(event["window_token_ids"])].append(event)
    events: list[dict[str, object]] = []
    for duplicate_group in by_window.values():
        if len({int(event["distance"]) for event in duplicate_group}) != 1:
            exclusions["conflicting_duplicate_window_rows"] += len(duplicate_group)
            continue
        events.append(duplicate_group[0])
        exclusions["same_label_duplicate_rows"] += len(duplicate_group) - 1

    balanced = _balanced_rows(events)
    balanced_hash = _balanced_canonical_sha256(balanced)
    balanced_ids = {str(event["event_hash"]) for event in balanced}
    for event in events:
        event["balanced_sensitivity"] = str(event["event_hash"]) in balanced_ids

    label_counts = Counter(int(event["distance"]) for event in events)
    observed = {
        "rows": len(events),
        "documents": len({str(event["document"]) for event in events}),
        "label_counts": dict(sorted(label_counts.items())),
        "balanced_rows": len(balanced),
        "balanced_canonical_sha256": balanced_hash,
    }
    expected = {
        "rows": EXPECTED_ROWS,
        "documents": EXPECTED_DOCUMENTS,
        "label_counts": EXPECTED_LABEL_COUNTS,
        "balanced_rows": EXPECTED_BALANCED_ROWS,
        "balanced_canonical_sha256": EXPECTED_BALANCED_CANONICAL_SHA256,
    }
    if observed != expected:
        raise ValueError(
            f"pinned GUM cohort drifted: expected={expected}, observed={observed}"
        )
    if len({str(event["event_hash"]) for event in events}) != len(events):
        raise ValueError("event hashes are not unique")

    frame = pd.DataFrame(events)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise AssertionError(
            f"cohort implementation omitted columns: {sorted(missing)}"
        )
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "claim": "personal-pronoun antecedent-distance decoding",
        "claim_boundary": (
            "The T=5 window includes the pronoun endpoint. This tests ordered "
            "contextual disambiguation, not pre-pronoun prediction."
        ),
        "source": {
            "repository": GUM_REPO,
            "revision": GUM_REVISION,
            "native_files": "coref/gum/tsv/*.tsv + matching dep/*.conllu + splits.md",
            **_source_inventory(gum_root),
        },
        "tokenizer": {
            "repository": TOKENIZER_REPO,
            "revision": TOKENIZER_REVISION,
            "add_special_tokens": False,
            "vocab_size": int(tokenizer.vocab_size),
            "file_sha256": EXPECTED_TOKENIZER_FILES,
        },
        "selection": {
            "window_tokens": WINDOW_TOKENS,
            "labels": list(LABELS),
            "relation_types": sorted(DIRECT_RELATIONS),
            "target": "exact target-final minus source-final model-token index",
            "history_controls": (
                "permute only positions 0..3; keep pronoun endpoint 4 fixed"
            ),
            "deduplication": (
                "first event per exact window if all labels agree; otherwise drop group"
            ),
            "balanced_sensitivity": (
                "for each first-seen pronoun and labels 2,3,4, retain the first "
                "min-label-count events in native traversal order"
            ),
        },
        "counts": {
            **observed,
            "raw_t5_rows": len(raw_events),
            "exclusions": dict(sorted(exclusions.items())),
            "pronouns": dict(sorted(Counter(frame["pronoun"]).items())),
            "genres": dict(sorted(Counter(frame["genre"]).items())),
            "splits": dict(sorted(Counter(frame["split"]).items())),
            "relations": dict(sorted(Counter(frame["relation_type"]).items())),
        },
    }
    observed_inventory = {
        key: manifest["source"][key] for key in EXPECTED_SOURCE_INVENTORY
    }
    if observed_inventory != EXPECTED_SOURCE_INVENTORY:
        raise ValueError(
            "pinned GUM source inventory drifted: "
            f"expected={EXPECTED_SOURCE_INVENTORY}, observed={observed_inventory}"
        )
    return frame, manifest


def validate_gum_revision(gum_root: str | Path) -> None:
    gum_root = Path(gum_root)
    observed = subprocess.run(
        ["git", "-C", str(gum_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed != GUM_REVISION:
        raise ValueError(f"GUM revision drifted: {observed} != {GUM_REVISION}")


def validate_tokenizer_files(path: str | Path) -> None:
    root = Path(path)
    observed = {name: sha256_file(root / name) for name in EXPECTED_TOKENIZER_FILES}
    if observed != EXPECTED_TOKENIZER_FILES:
        raise ValueError(f"tokenizer files drifted: {observed}")


def write_cohort(
    frame: pd.DataFrame,
    manifest: dict[str, object],
    *,
    cohort_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, object]:
    cohort_path = Path(cohort_path)
    manifest_path = Path(manifest_path)
    _atomic_parquet(frame, cohort_path)
    final_manifest = {
        **manifest,
        "artifact": {
            "path": str(cohort_path),
            "sha256": sha256_file(cohort_path),
            "rows": len(frame),
            "semantic_sha256": semantic_sha256(frame),
        },
    }
    if final_manifest["artifact"]["semantic_sha256"] != EXPECTED_SEMANTIC_SHA256:
        raise ValueError("GUM cohort semantic fingerprint drifted")
    _atomic_json(final_manifest, manifest_path)
    return final_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gum-root", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    validate_gum_revision(args.gum_root)
    validate_tokenizer_files(args.tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        local_files_only=True,
        use_fast=True,
    )
    if not tokenizer.is_fast:
        raise ValueError("GUM token endpoints require a fast tokenizer")
    frame, manifest = build_cohort(args.gum_root, tokenizer)
    final = write_cohort(
        frame,
        manifest,
        cohort_path=args.cohort,
        manifest_path=args.manifest,
    )
    print(json.dumps(final["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
