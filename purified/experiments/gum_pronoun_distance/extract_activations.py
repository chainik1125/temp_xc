"""Extract pinned Llama layer-10 T=5 states for GUM pronoun-distance events."""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from safetensors.torch import load_file, save_file
import torch

from experiments.writing_revision_destination.extract_activations import (
    _forward_batch,
    singleton_repeatability_diagnostics,
)

from .cohort import (
    EXPECTED_BALANCED_CANONICAL_SHA256,
    EXPECTED_BALANCED_ROWS,
    EXPECTED_DOCUMENTS,
    EXPECTED_LABEL_COUNTS,
    EXPECTED_ROWS,
    EXPECTED_SEMANTIC_SHA256,
    EXPECTED_SOURCE_INVENTORY,
    EXPECTED_TOKENIZER_FILES,
    GUM_REPO,
    GUM_REVISION,
    LABELS,
    _balanced_canonical_sha256,
    _balanced_rows,
    _event_hash,
    _window_hash,
)
from .cohort import PROTOCOL_VERSION as COHORT_PROTOCOL_VERSION
from .cohort import (
    REQUIRED_COLUMNS,
    TOKENIZER_REPO,
    TOKENIZER_REVISION,
    WINDOW_TOKENS,
    semantic_sha256,
    sha256_file,
)

EXTRACTION_PROTOCOL_VERSION = "gum-pronoun-distance-resid-post-l10-v1"
DEFAULT_LAYER = 10
DEFAULT_SHARD_SIZE = 256
HIDDEN_SIZE = 4_096
HASH_BYTES = 32


@dataclass(frozen=True)
class ExtractionConfig:
    protocol_version: str = EXTRACTION_PROTOCOL_VERSION
    model: str = TOKENIZER_REPO
    revision: str = TOKENIZER_REVISION
    layer: int = DEFAULT_LAYER
    window_tokens: int = WINDOW_TOKENS
    batch_size: int = 1
    shard_size: int = DEFAULT_SHARD_SIZE
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    attention: str = "sdpa"


def _atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_safetensors(tensors: dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(
        {name: tensor.contiguous() for name, tensor in tensors.items()},
        str(temporary),
    )
    os.replace(temporary, path)


def _hash_tensor(values: Sequence[str]) -> torch.Tensor:
    rows = []
    for value in values:
        raw = bytes.fromhex(str(value))
        if len(raw) != HASH_BYTES:
            raise ValueError("expected full SHA-256 event/window hash")
        rows.append(list(raw))
    return torch.tensor(rows, dtype=torch.uint8)


def _decode_hash_tensor(values: torch.Tensor) -> list[str]:
    return [bytes(row.tolist()).hex() for row in values.cpu()]


def _as_ids(value: object) -> tuple[int, ...]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError("token-ID columns must be list-like")
    return tuple(int(item) for item in value)


def load_cohort(
    cohort_path: str | Path,
    manifest_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load and fail closed on the exact pinned GUM cohort."""

    cohort_path = Path(cohort_path)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != COHORT_PROTOCOL_VERSION:
        raise ValueError("GUM cohort protocol drifted")
    source = manifest.get("source", {})
    source_inventory = {key: source.get(key) for key in EXPECTED_SOURCE_INVENTORY}
    if (
        source.get("repository") != GUM_REPO
        or source.get("revision") != GUM_REVISION
        or source_inventory != EXPECTED_SOURCE_INVENTORY
    ):
        raise ValueError("GUM source revision drifted")
    tokenizer = manifest.get("tokenizer", {})
    if (
        tokenizer.get("repository") != TOKENIZER_REPO
        or tokenizer.get("revision") != TOKENIZER_REVISION
        or tokenizer.get("add_special_tokens") is not False
        or tokenizer.get("file_sha256") != EXPECTED_TOKENIZER_FILES
    ):
        raise ValueError("GUM tokenizer provenance drifted")
    artifact = manifest.get("artifact", {})
    if (
        artifact.get("sha256") != sha256_file(cohort_path)
        or artifact.get("rows") != EXPECTED_ROWS
    ):
        raise ValueError("GUM cohort Parquet checksum failed")
    if artifact.get("semantic_sha256") != EXPECTED_SEMANTIC_SHA256:
        raise ValueError("GUM cohort semantic manifest drifted")
    if (
        manifest.get("counts", {}).get("balanced_canonical_sha256")
        != EXPECTED_BALANCED_CANONICAL_SHA256
    ):
        raise ValueError("balanced-sensitivity manifest drifted")

    frame = pd.read_parquet(cohort_path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"GUM cohort columns missing: {sorted(missing)}")
    observed_labels = {
        int(label): int(count)
        for label, count in zip(*np.unique(frame["distance"], return_counts=True))
    }
    observed = {
        "rows": len(frame),
        "documents": int(frame["document"].nunique()),
        "label_counts": observed_labels,
        "balanced_rows": int(frame["balanced_sensitivity"].sum()),
        "semantic_sha256": semantic_sha256(frame),
    }
    expected = {
        "rows": EXPECTED_ROWS,
        "documents": EXPECTED_DOCUMENTS,
        "label_counts": EXPECTED_LABEL_COUNTS,
        "balanced_rows": EXPECTED_BALANCED_ROWS,
        "semantic_sha256": EXPECTED_SEMANTIC_SHA256,
    }
    if observed != expected:
        raise ValueError(f"GUM cohort content drifted: {observed}")
    if frame["event_hash"].duplicated().any():
        raise ValueError("GUM cohort event hashes are not unique")
    if frame["window_hash"].duplicated().any():
        raise ValueError("GUM cohort windows are not unique")
    frame_records = frame.to_dict(orient="records")
    for record in frame_records:
        if str(record["event_hash"]) != _event_hash(record):
            raise ValueError("GUM event hash no longer matches its native edge")
        if str(record["window_hash"]) != _window_hash(record["window_token_ids"]):
            raise ValueError("GUM window hash no longer matches its token IDs")
    balanced_records = _balanced_rows(frame_records)
    expected_balanced_ids = {str(record["event_hash"]) for record in balanced_records}
    observed_balanced_ids = set(
        frame.loc[
            frame["balanced_sensitivity"].astype(bool),
            "event_hash",
        ].astype(str)
    )
    if observed_balanced_ids != expected_balanced_ids:
        raise ValueError("balanced-sensitivity membership drifted")
    if (
        _balanced_canonical_sha256(balanced_records)
        != EXPECTED_BALANCED_CANONICAL_SHA256
    ):
        raise ValueError("balanced-sensitivity cohort content drifted")

    prefixes = []
    windows = []
    documents = []
    for row in frame.itertuples(index=False):
        document_ids = _as_ids(row.document_input_ids)
        prefix = _as_ids(row.prefix_input_ids)
        window = _as_ids(row.window_token_ids)
        if int(row.distance) not in LABELS:
            raise ValueError("GUM target lies outside d=2,3,4")
        if len(window) != WINDOW_TOKENS or prefix[-WINDOW_TOKENS:] != window:
            raise ValueError("GUM activation window is not the prefix endpoint")
        if int(row.target_model_index) + 1 != len(prefix):
            raise ValueError("GUM target index and prefix length disagree")
        if prefix != document_ids[: len(prefix)]:
            raise ValueError("GUM prefix is not a document prefix")
        if int(row.target_model_index) - int(row.source_model_index) != int(
            row.distance
        ):
            raise ValueError("GUM antecedent-distance label is stale")
        prefixes.append(prefix)
        windows.append(window)
        documents.append(document_ids)
    frame = frame.copy()
    frame["prefix_input_ids"] = prefixes
    frame["window_token_ids"] = windows
    frame["document_input_ids"] = documents
    return frame, manifest


def _runtime_record(config: ExtractionConfig) -> dict[str, object]:
    import transformers

    gpu = None
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(config.device),
            "capability": list(torch.cuda.get_device_capability(config.device)),
        }
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu,
        "config": asdict(config),
    }


def _load_model(config: ExtractionConfig):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model,
        revision=config.revision,
        use_fast=True,
    )
    if not tokenizer.is_fast:
        raise ValueError("GUM activation extraction requires the pinned fast tokenizer")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0
    tokenizer.padding_side = "right"
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(config.dtype)
    if dtype is None:
        raise ValueError("dtype must be bfloat16 or float16")
    model = AutoModel.from_pretrained(
        config.model,
        revision=config.revision,
        torch_dtype=dtype,
        device_map={"": config.device},
        attn_implementation=config.attention,
    )
    model.eval()
    backbone = model.model if hasattr(model, "model") else model
    if not hasattr(backbone, "layers") or not 0 <= config.layer < len(backbone.layers):
        raise ValueError("cannot locate requested Llama transformer layer")
    if str(model.config.model_type) != "llama":
        raise ValueError("GUM protocol requires the pinned Llama subject")
    if int(model.config.hidden_size) != HIDDEN_SIZE:
        raise ValueError("subject hidden size drifted")
    observed_model_revision = getattr(model.config, "_commit_hash", None)
    observed_tokenizer_revision = getattr(tokenizer, "_commit_hash", None) or getattr(
        tokenizer, "init_kwargs", {}
    ).get("_commit_hash")
    for name, observed in (
        ("model", observed_model_revision),
        ("tokenizer", observed_tokenizer_revision),
    ):
        if observed and observed != config.revision:
            raise ValueError(f"{name} revision drifted: {observed}")
    return model, tokenizer, backbone.layers[config.layer]


def _shard_tensors(
    frame: pd.DataFrame,
    activations: torch.Tensor,
    *,
    start: int,
) -> dict[str, torch.Tensor]:
    if tuple(activations.shape) != (len(frame), WINDOW_TOKENS, HIDDEN_SIZE):
        raise ValueError("GUM activation shard shape drifted")
    return {
        "activations": activations.to(dtype=torch.float16, device="cpu"),
        "row_index": torch.arange(start, start + len(frame), dtype=torch.int64),
        "event_hash": _hash_tensor(frame["event_hash"].astype(str).tolist()),
        "window_hash": _hash_tensor(frame["window_hash"].astype(str).tolist()),
        "window_token_ids": torch.tensor(
            [list(values) for values in frame["window_token_ids"]],
            dtype=torch.int32,
        ),
        "distance": torch.tensor(frame["distance"].to_numpy(), dtype=torch.int64),
        "balanced_sensitivity": torch.tensor(
            frame["balanced_sensitivity"].to_numpy(dtype=np.bool_),
            dtype=torch.bool,
        ),
    }


def _validate_shard(
    shard_path: Path,
    sidecar_path: Path,
    frame: pd.DataFrame,
    *,
    start: int,
    stop: int,
    request_sha256: str,
    runtime_sha256: str,
) -> dict[str, object]:
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    checks = {
        "request_sha256": sidecar.get("request_sha256") == request_sha256,
        "runtime_sha256": sidecar.get("runtime_sha256") == runtime_sha256,
        "start": sidecar.get("start") == start,
        "stop": sidecar.get("stop") == stop,
        "rows": sidecar.get("rows") == stop - start,
        "sha256": sidecar.get("sha256") == sha256_file(shard_path),
        "size_bytes": sidecar.get("size_bytes") == shard_path.stat().st_size,
    }
    if not all(checks.values()):
        raise ValueError(f"GUM activation shard provenance mismatch: {checks}")
    tensors = load_file(str(shard_path), device="cpu")
    expected_keys = {
        "activations",
        "row_index",
        "event_hash",
        "window_hash",
        "window_token_ids",
        "distance",
        "balanced_sensitivity",
    }
    if set(tensors) != expected_keys:
        raise ValueError("GUM activation shard tensor keys drifted")
    expected_frame = frame.iloc[start:stop]
    if tuple(tensors["activations"].shape) != (
        stop - start,
        WINDOW_TOKENS,
        HIDDEN_SIZE,
    ):
        raise ValueError("GUM activation shard dimensions drifted")
    if tensors["activations"].dtype != torch.float16:
        raise ValueError("GUM activation shard dtype drifted")
    if not torch.equal(
        tensors["row_index"],
        torch.arange(start, stop, dtype=torch.int64),
    ):
        raise ValueError("GUM activation shard row indices drifted")
    if (
        _decode_hash_tensor(tensors["event_hash"])
        != expected_frame["event_hash"].astype(str).tolist()
    ):
        raise ValueError("GUM activation shard event order drifted")
    if (
        _decode_hash_tensor(tensors["window_hash"])
        != expected_frame["window_hash"].astype(str).tolist()
    ):
        raise ValueError("GUM activation shard windows drifted")
    expected_windows = torch.tensor(
        [list(values) for values in expected_frame["window_token_ids"]],
        dtype=torch.int32,
    )
    if not torch.equal(tensors["window_token_ids"], expected_windows):
        raise ValueError("GUM activation shard token IDs drifted")
    if not torch.equal(
        tensors["distance"],
        torch.tensor(expected_frame["distance"].to_numpy(), dtype=torch.int64),
    ):
        raise ValueError("GUM activation shard targets drifted")
    if not torch.equal(
        tensors["balanced_sensitivity"],
        torch.tensor(
            expected_frame["balanced_sensitivity"].to_numpy(dtype=np.bool_),
            dtype=torch.bool,
        ),
    ):
        raise ValueError("GUM activation shard sensitivity flags drifted")
    if not torch.isfinite(tensors["activations"]).all():
        raise ValueError("GUM activation shard contains nonfinite states")
    return sidecar


def extract(
    cohort_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    config: ExtractionConfig,
) -> dict[str, object]:
    """Run or resume deterministic singleton extraction."""

    if config.batch_size != 1:
        raise ValueError("pinned GUM extraction requires singleton batches")
    frame, cohort_manifest = load_cohort(cohort_path, manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    implementation_path = Path(__file__)
    forward_implementation_path = Path(inspect.getsourcefile(_forward_batch) or "")
    if not forward_implementation_path.is_file():
        raise FileNotFoundError("cannot fingerprint activation-forward implementation")
    request = {
        "protocol_version": EXTRACTION_PROTOCOL_VERSION,
        "cohort_sha256": sha256_file(cohort_path),
        "cohort_semantic_sha256": EXPECTED_SEMANTIC_SHA256,
        "cohort_manifest_sha256": sha256_file(manifest_path),
        "cohort_protocol_version": COHORT_PROTOCOL_VERSION,
        "gum_revision": GUM_REVISION,
        "config": asdict(config),
        "implementation_sha256": sha256_file(implementation_path),
        "forward_implementation_sha256": sha256_file(forward_implementation_path),
    }
    request_path = output_dir / "request.json"
    if request_path.exists():
        if json.loads(request_path.read_text(encoding="utf-8")) != request:
            raise ValueError("GUM activation request drifted; use a fresh cache")
    else:
        _atomic_json(request, request_path)
    request_sha256 = sha256_file(request_path)

    model, tokenizer, layer_module = _load_model(config)
    runtime = _runtime_record(config)
    runtime_path = output_dir / "runtime.json"
    if runtime_path.exists():
        if json.loads(runtime_path.read_text(encoding="utf-8")) != runtime:
            raise ValueError("GUM activation runtime drifted; use a fresh cache")
    else:
        _atomic_json(runtime, runtime_path)
    runtime_sha256 = sha256_file(runtime_path)

    prefixes = frame["prefix_input_ids"].tolist()
    diagnostic = singleton_repeatability_diagnostics(
        model,
        layer_module,
        prefixes,
        config,
        pad_token_id=int(tokenizer.pad_token_id),
    )
    if diagnostic["status"] != "passed":
        raise RuntimeError("GUM singleton extraction repeatability failed")
    _atomic_json(diagnostic, output_dir / "repeatability.json")

    shard_records = []
    started = time.time()
    for start in range(0, len(frame), config.shard_size):
        stop = min(start + config.shard_size, len(frame))
        name = f"shard-{start:06d}-{stop:06d}"
        shard_path = output_dir / f"{name}.safetensors"
        sidecar_path = output_dir / f"{name}.json"
        if shard_path.exists() or sidecar_path.exists():
            if not shard_path.exists() or not sidecar_path.exists():
                raise ValueError(f"incomplete existing GUM activation shard: {name}")
            record = _validate_shard(
                shard_path,
                sidecar_path,
                frame,
                start=start,
                stop=stop,
                request_sha256=request_sha256,
                runtime_sha256=runtime_sha256,
            )
            shard_records.append(record)
            continue
        rows = prefixes[start:stop]
        batches = []
        for row in rows:
            batches.append(
                _forward_batch(
                    model,
                    layer_module,
                    [row],
                    config,
                    pad_token_id=int(tokenizer.pad_token_id),
                    output_dtype=torch.float16,
                )
            )
        activations = torch.cat(batches, dim=0)
        tensors = _shard_tensors(frame.iloc[start:stop], activations, start=start)
        _atomic_safetensors(tensors, shard_path)
        record = {
            "name": name,
            "start": start,
            "stop": stop,
            "rows": stop - start,
            "sha256": sha256_file(shard_path),
            "size_bytes": shard_path.stat().st_size,
            "request_sha256": request_sha256,
            "runtime_sha256": runtime_sha256,
        }
        _atomic_json(record, sidecar_path)
        _validate_shard(
            shard_path,
            sidecar_path,
            frame,
            start=start,
            stop=stop,
            request_sha256=request_sha256,
            runtime_sha256=runtime_sha256,
        )
        shard_records.append(record)
        del tensors, activations, batches
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    complete = {
        "status": "complete",
        "protocol_version": EXTRACTION_PROTOCOL_VERSION,
        "rows": len(frame),
        "window_tokens": WINDOW_TOKENS,
        "hidden_size": HIDDEN_SIZE,
        "request_sha256": request_sha256,
        "runtime_sha256": runtime_sha256,
        "repeatability_sha256": sha256_file(output_dir / "repeatability.json"),
        "shards": shard_records,
        "total_shard_bytes": int(
            sum(int(record["size_bytes"]) for record in shard_records)
        ),
        "elapsed_seconds_this_process": time.time() - started,
        "cohort_claim_boundary": cohort_manifest["claim_boundary"],
    }
    _atomic_json(complete, output_dir / "complete.json")
    return complete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--attention", default="sdpa")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExtractionConfig(
        device=args.device,
        shard_size=args.shard_size,
        attention=args.attention,
    )
    complete = extract(args.cohort, args.manifest, args.output_dir, config)
    print(json.dumps(complete, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
