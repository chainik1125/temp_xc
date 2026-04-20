"""Activation collection for the Venhoff pipeline — Path 1 and Path 3.

**Path 1 (SAE + MLC, matches Venhoff's contract exactly).** For each
sentence in a trace, slice the per-token residual-stream activations at
layer L via `[:, token_start - 1 : token_end, :]` (note the `-1` off-by
hazard — flagged in integration_plan § 7), mean over the token axis,
and keep the resulting `(d_model,)` vector. Then center + L2-normalize
all vectors with a dataset-global mean saved as a sidecar pickle. This
reproduces `utils.utils.process_saved_responses` at upstream commit
`49a7f73`.

**Path 3 (TempXC).** For each sentence, keep the full T-token window
centered on the sentence instead of mean-pooling. Output shape is
`(N_sentences, T, d_model)`. Aggregation (last/mean/max/full_window)
happens at annotation time, not here — preserving the temporal axis
until encoding is the whole point of Path 3.

Both paths persist an `activation_mean` sidecar so downstream code
(our SAE shim) can match Venhoff's `load_sae` contract.

Resume semantics: skip if activations pkl + mean pkl + sentences json
all exist with matching config hashes. Use `--force` to rebuild.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from src.bench.model_registry import get_model_config, resid_hook_target
from src.bench.venhoff.paths import ArtifactPaths, RunIdentity, can_resume, write_with_metadata
from src.bench.venhoff.responses import extract_thinking_process
from src.bench.venhoff.tokenization import get_char_to_token_map, split_into_sentences

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("venhoff.activation_collection")

PathName = Literal["path1", "path3", "path_mlc"]


@dataclass(frozen=True)
class CollectionConfig:
    path: PathName
    layer: int              # anchor layer; for path_mlc, centre of the window
    T: int = 5              # only used for path3
    n_layers: int = 5       # only used for path_mlc — window width, symmetric around `layer`
    min_words: int = 3
    # Cap sentences per trace to keep memory sane; 0 = no cap.
    max_sentences_per_trace: int = 0


def _mlc_layer_window(anchor: int, n_layers: int, total_layers: int) -> list[int]:
    """Symmetric layer window around `anchor`, clipped to [0, total_layers).

    For anchor=6, n_layers=5 → [4, 5, 6, 7, 8]. If `n_layers` is even the
    anchor sits one past the centre (so n=4 at anchor=6 → [4, 5, 6, 7]).
    """
    half = n_layers // 2
    start = max(0, anchor - half)
    end = min(total_layers, start + n_layers)
    start = max(0, end - n_layers)
    return list(range(start, end))


def _iter_sentence_spans(full_response: str, tokenizer) -> list[tuple[str, int, int]]:
    """Return list of (sentence, token_start, token_end) for a trace.

    `token_start`/`token_end` are Venhoff's raw offsets — no -1 applied
    here. Callers pick whether to apply the -1 (Path 1 does; Path 3
    doesn't need to because it centers on the sentence).
    """
    thinking = extract_thinking_process(full_response)
    if not thinking:
        return []
    sentences = split_into_sentences(thinking)
    if not sentences:
        return []
    char_to_token = get_char_to_token_map(full_response, tokenizer)

    spans: list[tuple[str, int, int]] = []
    for sentence in sentences:
        text_pos = full_response.find(sentence)
        if text_pos < 0:
            continue
        token_start = char_to_token.get(text_pos)
        token_end = char_to_token.get(text_pos + len(sentence))
        if token_start is None or token_end is None or token_start >= token_end:
            continue
        spans.append((sentence, token_start, token_end))
    return spans


def _collect_layer_for_trace(model, layer: int, input_ids) -> "np.ndarray":
    """Return residual-stream activations at `layer` — shape (1, seq, d_model)."""
    out_multi = _collect_layers_for_trace(model, [layer], input_ids)
    return out_multi[0]  # (1, seq, d_model)


def _collect_layers_for_trace(
    model, layers: list[int], input_ids
) -> list["np.ndarray"]:
    """Return residual-stream activations at each of `layers` in one forward.

    Output is a list of (1, seq, d_model) float32 numpy arrays, aligned
    with the input `layers` order. One forward pass, n_layers parallel
    hooks — important because we don't want 5× the compute for MLC.
    """
    import torch

    family = get_model_config_for_model(model).architecture_family
    targets = [resid_hook_target(model, layer, family) for layer in layers]

    # Use model.trace for nnsight backends; otherwise hook manually.
    try:
        with model.trace({"input_ids": input_ids}) as _tracer:
            saved_list = [t.output.save() for t in targets]
        return [s.detach().cpu().to(torch.float32).numpy() for s in saved_list]
    except AttributeError:
        # Fallback: plain HF forward hooks, one per layer.
        captured: dict[int, Any] = {}

        def make_hook(i: int):
            def hook(_module, _inp, output):
                captured[i] = output[0] if isinstance(output, tuple) else output
            return hook

        handles = [t.register_forward_hook(make_hook(i)) for i, t in enumerate(targets)]
        try:
            with torch.no_grad():
                _ = model(input_ids=input_ids)
        finally:
            for h in handles:
                h.remove()
        return [captured[i].detach().cpu().to(torch.float32).numpy() for i in range(len(layers))]


def get_model_config_for_model(model):
    """Extract our ModelConfig from a loaded HF model.

    We embed the registry name in model.config.name_or_path during load
    and look it up — simpler than sniffing families from weights.
    """
    # Heuristic: use config.name_or_path's basename if registered, else
    # fall back to 'llama' family.
    from src.bench.model_registry import MODEL_REGISTRY

    name_or_path = getattr(model.config, "name_or_path", "") or ""
    for key, cfg in MODEL_REGISTRY.items():
        if cfg.hf_path == name_or_path or key in name_or_path.lower():
            return cfg
    # Default
    from src.bench.model_registry import ModelConfig

    return ModelConfig(
        name="unknown", hf_path=name_or_path, d_model=model.config.hidden_size,
        n_layers=model.config.num_hidden_layers, default_layer_indices=(0,),
        architecture_family="llama",
    )


def collect_path1(
    paths: ArtifactPaths,
    model,
    tokenizer,
    config: CollectionConfig,
    force: bool = False,
) -> tuple[Path, Path, Path]:
    """Per-sentence-mean activations (SAE/MLC contract)."""
    import torch

    act_pkl = paths.activations_pkl("path1")
    mean_pkl = paths.activation_mean_pkl("path1")
    sent_json = paths.sentences_json("path1")

    meta = {
        "stage": "collect_path1",
        "model": paths.identity.model,
        "n_traces": paths.identity.n_traces,
        "layer": config.layer,
        "seed": paths.identity.seed,
        "min_words": config.min_words,
        "max_sentences_per_trace": config.max_sentences_per_trace,
    }
    if not force and all(can_resume(p, meta) for p in (act_pkl, mean_pkl, sent_json)):
        log.info("[info] resume | stage=collect_path1 | cache=%s", act_pkl)
        return act_pkl, mean_pkl, sent_json

    traces_path = paths.traces_json
    with traces_path.open() as f:
        traces = json.load(f)

    mcfg = get_model_config(paths.identity.model)
    d_model = mcfg.d_model
    running_mean = np.zeros(d_model, dtype=np.float32)
    mean_count = 0
    sentence_vectors: list[np.ndarray] = []
    sentence_texts: list[str] = []

    for trace in traces:
        full_response = trace["full_response"]
        input_ids = tokenizer.encode(full_response, return_tensors="pt").to(model.device)
        acts = _collect_layer_for_trace(model, config.layer, input_ids)  # (1, seq, d)
        assert acts.ndim == 3 and acts.shape[-1] == d_model, f"bad act shape {acts.shape}"

        spans = _iter_sentence_spans(full_response, tokenizer)
        if config.max_sentences_per_trace > 0:
            spans = spans[: config.max_sentences_per_trace]

        tmin, tmax = 10**9, -1
        for sentence, token_start, token_end in spans:
            # Venhoff's `-1` offset preserved:
            segment = acts[:, token_start - 1 : token_end, :]
            if segment.shape[1] <= 0:
                continue
            vec = segment.mean(axis=1).reshape(-1).astype(np.float32)
            sentence_vectors.append(vec)
            sentence_texts.append(sentence)
            tmin = min(tmin, token_start)
            tmax = max(tmax, token_end)

        # Running mean of the span-covered slice (also Venhoff's definition).
        if tmax > 0 and tmin < acts.shape[1]:
            span_mean = acts[:, tmin:tmax, :].mean(axis=1).reshape(-1).astype(np.float32)
            running_mean = running_mean + (span_mean - running_mean) / (mean_count + 1)
            mean_count += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not sentence_vectors:
        raise ValueError("no sentence activations collected — check traces and splitter")

    sentence_vectors_np = np.stack(sentence_vectors, axis=0)
    # Center + L2-normalize to match Venhoff's downstream SAE contract.
    centered = sentence_vectors_np - running_mean[None, :]
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    assert np.all(norms > 0), "zero-norm sentence vector after centering"
    normalized = centered / norms

    # Atomic write: serialize to buffer first, hand the bytes to
    # write_with_metadata which does tmp+rename together with sidecar.
    import io
    act_buf = io.BytesIO()
    pickle.dump((normalized, sentence_texts), act_buf)
    write_with_metadata(act_pkl, act_buf.getvalue(), meta)

    mean_payload = {
        "model_id": paths.identity.model,
        "layer": int(config.layer),
        "n_examples": int(paths.identity.n_traces),
        "count_vectors": int(mean_count),
        "activation_mean": running_mean,
    }
    mean_buf = io.BytesIO()
    pickle.dump(mean_payload, mean_buf)
    write_with_metadata(mean_pkl, mean_buf.getvalue(), meta)

    write_with_metadata(sent_json, json.dumps(sentence_texts), meta)

    log.info("[done] saved path1_activations | n_sentences=%d | path=%s", len(sentence_texts), act_pkl)
    return act_pkl, mean_pkl, sent_json


def collect_path3(
    paths: ArtifactPaths,
    model,
    tokenizer,
    config: CollectionConfig,
    force: bool = False,
) -> tuple[Path, Path, Path]:
    """Per-sentence T-window activations (TempXC Path 3 contract)."""
    import torch

    act_pkl = paths.activations_pkl("path3")
    mean_pkl = paths.activation_mean_pkl("path3")
    sent_json = paths.sentences_json("path3")

    meta = {
        "stage": "collect_path3",
        "model": paths.identity.model,
        "n_traces": paths.identity.n_traces,
        "layer": config.layer,
        "T": config.T,
        "seed": paths.identity.seed,
        "min_words": config.min_words,
        "max_sentences_per_trace": config.max_sentences_per_trace,
    }
    if not force and all(can_resume(p, meta) for p in (act_pkl, mean_pkl, sent_json)):
        log.info("[info] resume | stage=collect_path3 | cache=%s", act_pkl)
        return act_pkl, mean_pkl, sent_json

    with paths.traces_json.open() as f:
        traces = json.load(f)

    mcfg = get_model_config(paths.identity.model)
    d_model = mcfg.d_model
    T = config.T
    running_mean = np.zeros(d_model, dtype=np.float32)
    mean_count = 0
    windows: list[np.ndarray] = []
    sentence_texts: list[str] = []

    for trace in traces:
        full_response = trace["full_response"]
        input_ids = tokenizer.encode(full_response, return_tensors="pt").to(model.device)
        acts = _collect_layer_for_trace(model, config.layer, input_ids)  # (1, seq, d)
        seq_len = acts.shape[1]

        spans = _iter_sentence_spans(full_response, tokenizer)
        if config.max_sentences_per_trace > 0:
            spans = spans[: config.max_sentences_per_trace]

        for sentence, token_start, token_end in spans:
            # Center the T-window on the sentence midpoint.
            mid = (token_start + token_end) // 2
            half = T // 2
            win_start = max(0, mid - half)
            win_end = min(seq_len, win_start + T)
            win_start = max(0, win_end - T)  # right-align if near end
            segment = acts[:, win_start:win_end, :].reshape(win_end - win_start, d_model)
            if segment.shape[0] < T:
                # Zero-pad at the end if the trace is too short (rare).
                pad = np.zeros((T - segment.shape[0], d_model), dtype=np.float32)
                segment = np.concatenate([segment, pad], axis=0)
            assert segment.shape == (T, d_model), f"bad window {segment.shape}"
            windows.append(segment.astype(np.float32))
            sentence_texts.append(sentence)
            running_mean = running_mean + (segment.mean(axis=0) - running_mean) / (mean_count + 1)
            mean_count += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not windows:
        raise ValueError("no sentence windows collected — check traces and splitter")

    windows_np = np.stack(windows, axis=0)  # (N, T, d_model)

    import io
    act_buf = io.BytesIO()
    pickle.dump((windows_np, sentence_texts), act_buf)
    write_with_metadata(act_pkl, act_buf.getvalue(), meta)

    mean_payload = {
        "model_id": paths.identity.model,
        "layer": int(config.layer),
        "T": int(T),
        "n_examples": int(paths.identity.n_traces),
        "count_vectors": int(mean_count),
        "activation_mean": running_mean,
    }
    mean_buf = io.BytesIO()
    pickle.dump(mean_payload, mean_buf)
    write_with_metadata(mean_pkl, mean_buf.getvalue(), meta)

    write_with_metadata(sent_json, json.dumps(sentence_texts), meta)

    log.info("[done] saved path3_activations | n_sentences=%d | T=%d | path=%s", len(sentence_texts), T, act_pkl)
    return act_pkl, mean_pkl, sent_json


def collect_path_mlc(
    paths: ArtifactPaths,
    model,
    tokenizer,
    config: CollectionConfig,
    force: bool = False,
) -> tuple[Path, Path, Path]:
    """Per-sentence-mean activations across a window of layers (MLC contract).

    Output shape per sentence: (n_layers, d_model). Full tensor:
    (N_sentences, n_layers, d_model). Sentence-span logic matches
    Path 1 exactly — same token_start-1 offset, same sentence splitter
    — only difference is we take the mean over the token axis *per
    layer* and stack the 5 layer-means together.

    MLC has no aggregation at annotation time; the layer axis is the
    model's native input, handled inside `LayerCrosscoder.encode`.
    """
    import torch

    act_pkl = paths.activations_pkl("path_mlc")
    mean_pkl = paths.activation_mean_pkl("path_mlc")
    sent_json = paths.sentences_json("path_mlc")

    mcfg = get_model_config(paths.identity.model)
    layer_window = _mlc_layer_window(config.layer, config.n_layers, mcfg.n_layers)

    meta = {
        "stage": "collect_path_mlc",
        "model": paths.identity.model,
        "n_traces": paths.identity.n_traces,
        "anchor_layer": config.layer,
        "n_layers": len(layer_window),
        "layer_window": layer_window,
        "seed": paths.identity.seed,
        "min_words": config.min_words,
        "max_sentences_per_trace": config.max_sentences_per_trace,
    }
    if not force and all(can_resume(p, meta) for p in (act_pkl, mean_pkl, sent_json)):
        log.info("[info] resume | stage=collect_path_mlc | cache=%s", act_pkl)
        return act_pkl, mean_pkl, sent_json

    with paths.traces_json.open() as f:
        traces = json.load(f)

    d_model = mcfg.d_model
    n_layers = len(layer_window)
    # Running mean is (n_layers, d_model) — one mean vector per layer.
    running_mean = np.zeros((n_layers, d_model), dtype=np.float32)
    mean_count = 0
    stacked_means: list[np.ndarray] = []  # each: (n_layers, d_model)
    sentence_texts: list[str] = []

    for trace in traces:
        full_response = trace["full_response"]
        input_ids = tokenizer.encode(full_response, return_tensors="pt").to(model.device)
        # One forward pass, n_layers hooks → list of (1, seq, d_model).
        per_layer = _collect_layers_for_trace(model, layer_window, input_ids)

        spans = _iter_sentence_spans(full_response, tokenizer)
        if config.max_sentences_per_trace > 0:
            spans = spans[: config.max_sentences_per_trace]

        tmin, tmax = 10**9, -1
        for sentence, token_start, token_end in spans:
            per_layer_means: list[np.ndarray] = []
            ok = True
            for acts in per_layer:
                # Venhoff's `-1` offset preserved on every layer.
                segment = acts[:, token_start - 1 : token_end, :]
                if segment.shape[1] <= 0:
                    ok = False
                    break
                per_layer_means.append(
                    segment.mean(axis=1).reshape(-1).astype(np.float32)
                )
            if not ok:
                continue
            stacked = np.stack(per_layer_means, axis=0)  # (n_layers, d_model)
            stacked_means.append(stacked)
            sentence_texts.append(sentence)
            tmin = min(tmin, token_start)
            tmax = max(tmax, token_end)

        # Per-layer running mean over the span-covered slice.
        if tmax > 0 and tmin < per_layer[0].shape[1]:
            span_means = np.stack(
                [a[:, tmin:tmax, :].mean(axis=1).reshape(-1).astype(np.float32) for a in per_layer],
                axis=0,
            )  # (n_layers, d_model)
            running_mean = running_mean + (span_means - running_mean) / (mean_count + 1)
            mean_count += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not stacked_means:
        raise ValueError("no multi-layer sentence means collected — check traces / splitter / layer window")

    acts_np = np.stack(stacked_means, axis=0)  # (N, n_layers, d_model)

    # Center + L2-normalize *per-sentence* (flatten n_layers × d_model and norm).
    # This matches Path 1's contract but vectorized across the layer axis.
    N = acts_np.shape[0]
    flat = acts_np.reshape(N, n_layers * d_model)
    mean_flat = running_mean.reshape(n_layers * d_model)
    centered = flat - mean_flat[None, :]
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    assert np.all(norms > 0), "zero-norm multi-layer sentence vector after centering"
    normalized_flat = centered / norms
    normalized = normalized_flat.reshape(N, n_layers, d_model).astype(np.float32)

    import io
    act_buf = io.BytesIO()
    pickle.dump((normalized, sentence_texts), act_buf)
    write_with_metadata(act_pkl, act_buf.getvalue(), meta)

    mean_payload = {
        "model_id": paths.identity.model,
        "anchor_layer": int(config.layer),
        "layer_window": layer_window,
        "n_layers": int(n_layers),
        "n_examples": int(paths.identity.n_traces),
        "count_vectors": int(mean_count),
        # Save as (n_layers, d_model) so the shim can reshape if needed.
        "activation_mean": running_mean,
    }
    mean_buf = io.BytesIO()
    pickle.dump(mean_payload, mean_buf)
    write_with_metadata(mean_pkl, mean_buf.getvalue(), meta)

    write_with_metadata(sent_json, json.dumps(sentence_texts), meta)

    log.info(
        "[done] saved path_mlc_activations | n_sentences=%d | n_layers=%d | layer_window=%s | path=%s",
        len(sentence_texts), n_layers, layer_window, act_pkl,
    )
    return act_pkl, mean_pkl, sent_json


def _load_model(model_name: str, device: str = "auto"):
    """Load HF model + tokenizer for activation collection."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = get_model_config(model_name)
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[cfg.dtype]
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        cfg.hf_path,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    model.eval()
    return model, tokenizer


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, default=Path("results/venhoff_eval"))
    p.add_argument("--model", default="deepseek-r1-distill-llama-8b")
    p.add_argument("--dataset", default="mmlu-pro")
    p.add_argument("--split", default="test")
    p.add_argument("--n-traces", type=int, default=1000)
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--T", type=int, default=5)
    p.add_argument("--n-layers", type=int, default=5,
                   help="MLC layer-window width; only used for path_mlc.")
    p.add_argument("--paths", nargs="+", default=["path1", "path3"],
                   choices=["path1", "path3", "path_mlc"])
    p.add_argument("--max-sentences-per-trace", type=int, default=0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    paths = ArtifactPaths(
        root=args.root,
        identity=RunIdentity(
            model=args.model, dataset=args.dataset, dataset_split=args.split,
            n_traces=args.n_traces, layer=args.layer, seed=args.seed,
        ),
    )
    model, tokenizer = _load_model(args.model)

    for pname in args.paths:
        cfg = CollectionConfig(
            path=pname, layer=args.layer, T=args.T, n_layers=args.n_layers,
            max_sentences_per_trace=args.max_sentences_per_trace,
        )
        if pname == "path1":
            collect_path1(paths, model, tokenizer, cfg, force=args.force)
        elif pname == "path3":
            collect_path3(paths, model, tokenizer, cfg, force=args.force)
        elif pname == "path_mlc":
            collect_path_mlc(paths, model, tokenizer, cfg, force=args.force)
        else:
            raise ValueError(f"unknown path: {pname!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
