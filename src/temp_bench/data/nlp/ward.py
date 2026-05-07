"""C7-specific activation caching path (Ward Stage B backtracking).

[pipeline]'s :mod:`temp_bench.data.nlp.cache` covers the generic
fineweb path (used by C3 / C4 / C5). C7 needs a different corpus
(in-domain R1-Distill-Llama traces from the prior author's Stage A) and a
slightly different on-disk layout (hookpoint-keyed file +
``layer_specs.json``); this sibling module is owned by [pipeline]
and avoids cross-territory edits to ``cache.py``.

Open questions for the maintainer + [pipeline] (see [pipeline] briefing OQ #5):
whether to fold the ward branch into ``cache._stream_dataset_texts``
(unifying public API) or keep them separate. The current split keeps
the cache.py canonical for fineweb-style cells and isolates C7's
domain-specific corpus to one file under [pipeline]'s ownership.

Public API (mirrors the original ``temp_bench.data.nlp`` skeleton I
wrote pre-rebase, but limited to the C7 path):

- :func:`cache_activations(datasource_name, *, force=False)` — build
  the C7 activation cache from Stage A traces.
- :func:`load_activations(datasource_name)` — memmap the produced
  ``<hookpoint>_L<layer>.npy``.
- :func:`build_corpus(...)` — Stage-A-traces dispatch only.

Cache layout under ``results/act_cache/<act_cache_key>/``::

    resid_post_L10.npy   float16 (N, L, d_model) memmap
    token_ids.npy        int64 (N, L) — for sentence reconstruction
    layer_specs.json     {"layer", "hookpoint", "d_model", ...}
    corpus.json          provenance: which corpus, num_sequences, stride
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

log = logging.getLogger("temp_bench.data.nlp.ward")


# ── Hookpoint dispatch (port of cache_activations.py:_attach_hooks) ───


def _attach_hooks(model, hookpoints: list[dict], buffer: dict) -> list:
    """Register forward / forward-pre hooks for each hookpoint.

    Verbatim port from
    ``origin/case-backtracking @ [scrubbed-sha]:experiments/ward_backtracking_txc/cache_activations.py``.
    Components: ``resid`` / ``resid_post`` (post-block residual),
    ``attn`` (self_attn output), ``ln1`` (input to self_attn).
    """
    handles = []
    for hp in hookpoints:
        layer_idx = hp["layer"]
        comp = hp["component"]
        key = hp["key"]

        def make_post_hook(k):
            def hook_fn(_m, _i, output):
                acts = output[0] if isinstance(output, tuple) else output
                if acts.dim() == 4:
                    acts = acts.reshape(acts.shape[0], acts.shape[1], -1)
                buffer[k] = acts.detach().to(torch.float16).cpu()
            return hook_fn

        def make_pre_hook(k):
            def hook_fn(_m, args, kwargs):
                if args:
                    acts = args[0]
                else:
                    acts = kwargs["hidden_states"]
                if acts.dim() == 4:
                    acts = acts.reshape(acts.shape[0], acts.shape[1], -1)
                buffer[k] = acts.detach().to(torch.float16).cpu()
            return hook_fn

        if comp == "resid" or comp == "resid_post":
            target = model.model.layers[layer_idx]
            handles.append(target.register_forward_hook(make_post_hook(key)))
        elif comp == "attn":
            target = model.model.layers[layer_idx].self_attn
            handles.append(target.register_forward_hook(make_post_hook(key)))
        elif comp == "ln1":
            target = model.model.layers[layer_idx].self_attn
            handles.append(target.register_forward_pre_hook(make_pre_hook(key), with_kwargs=True))
        else:
            raise ValueError(f"unknown component: {comp}")
    return handles


# ── Corpus loaders ────────────────────────────────────────────────────


def _load_corpus_ward(num_sequences: int, seq_length: int, tokenizer, *, stride: int | None = None) -> torch.Tensor:
    """Build a corpus from the ported Stage A traces (Ward backtracking).

    Slices each R1-Distill-Llama trace's ``full_response`` field into
    sliding windows of ``seq_length`` tokens with spacing ``stride``.
    Used by the C7 datasource ``ward_backtracking_math500``.

    Adapted from
    ``origin/case-backtracking @ [scrubbed-sha]:experiments/ward_backtracking_txc/cache_activations.py``;
    original used ``results/ward_backtracking/traces.json`` but we
    use the ported copy under ``results/c7_backtracking/stage_a/traces.json``.
    """
    from temp_bench.config import purified_root
    if stride is None:
        stride = seq_length
    traces_path = purified_root() / "results" / "c7_backtracking" / "stage_a" / "traces.json"
    if not traces_path.exists():
        raise FileNotFoundError(
            f"Stage A traces not found at {traces_path}. "
            "Re-port via `git show origin/case-backtracking:results/ward_backtracking/...` — "
            "see results/c7_backtracking/stage_a/ATTRIBUTION.md."
        )
    traces = json.loads(traces_path.read_text())
    log.info(
        "[corpus] sourcing windows from Stage A: %d traces, seq_length=%d stride=%d",
        len(traces), seq_length, stride,
    )

    texts: list[str] = []
    for t in traces:
        full = t.get("full_response") or ""
        if not full:
            continue
        ids = tokenizer(full, add_special_tokens=False)["input_ids"]
        for start in range(0, max(1, len(ids) - seq_length + 1), stride):
            window = ids[start:start + seq_length]
            if len(window) < seq_length:
                break
            texts.append(tokenizer.decode(window))
            if len(texts) >= num_sequences:
                break
        if len(texts) >= num_sequences:
            break
    log.info("[corpus] from traces: %d windows", len(texts))

    if len(texts) < num_sequences:
        log.warning(
            "[corpus] only %d/%d windows from traces — using what we have. "
            "FineWeb top-up not implemented in temp_bench.data.nlp.ward.",
            len(texts), num_sequences,
        )

    out = []
    for txt in texts[:num_sequences]:
        enc = tokenizer(txt, return_tensors="pt", truncation=True,
                        max_length=seq_length, padding="max_length",
                        add_special_tokens=True)
        out.append(enc["input_ids"].squeeze(0))
    if not out:
        raise RuntimeError("Corpus build produced 0 windows; check traces.json.")
    return torch.stack(out, dim=0)


def build_corpus(dataset_name: str, *, num_sequences: int, seq_length: int, tokenizer,
                 stride: int | None = None) -> torch.Tensor:
    """Dispatch on the datasource's ``dataset`` field. C7 only."""
    if dataset_name in ("ward_backtracking_math500", "ward_backtracking"):
        return _load_corpus_ward(num_sequences, seq_length, tokenizer, stride=stride)
    raise ValueError(
        f"temp_bench.data.nlp.ward only handles ward_backtracking_math500. "
        f"For {dataset_name!r} use temp_bench.data.nlp.cache.build_activation_cache."
    )


# ── Cache build / load ────────────────────────────────────────────────


def cache_activations(
    datasource_name: str,
    *,
    cache_batch_size: int = 8,
    force: bool = False,
    push_to_hf: bool | None = None,
) -> Path:
    """Build (or load) the C7 activation cache for a registered datasource.

    Returns the cache directory under
    ``results/act_cache/<act_cache_key>/``. Idempotent: if the cache
    already exists with the right shape, returns immediately.

    On ephemeral pods (``TEMP_BENCH_POD_MODE=ephemeral``), automatically
    pushes the cache to HF temp-bench-data after a fresh build.
    """
    from temp_bench.config import (
        act_cache_dir,
        compute_act_cache_key,
        load_datasource,
    )

    ds = load_datasource(datasource_name)
    cache_key = compute_act_cache_key(ds)
    cache_dir = act_cache_dir(cache_key)
    cache_dir.mkdir(parents=True, exist_ok=True)

    spec_extra = ds.model_dump()
    layer = int(spec_extra["layer"])
    hookpoint = spec_extra["hookpoint"]
    subject_model = spec_extra["subject_model"]
    n_seqs = int(spec_extra["n_seqs"])
    seq_len = int(spec_extra["seq_len"])
    dataset_name = spec_extra["dataset"]
    hp_key = f"{hookpoint}_L{layer}"
    acts_path = cache_dir / f"{hp_key}.npy"

    log.info("[cache_activations] datasource=%s key=%s", datasource_name, cache_key)

    if acts_path.exists() and not force:
        try:
            arr = np.load(acts_path, mmap_mode="r")
            if arr.ndim == 3 and arr.shape[1] == seq_len:
                log.info("[cache_activations] cache hit at %s shape=%s", acts_path, arr.shape)
                return cache_dir
            log.warning("[cache_activations] existing cache wrong shape %s — rebuilding", arr.shape)
        except Exception as e:
            log.warning("[cache_activations] existing cache unreadable (%s) — rebuilding", e)

    from temp_bench.utils.tokens import get_token
    hf_token = get_token("hf")
    if not hf_token:
        raise RuntimeError(
            "HF token missing — populate /workspace/.tokens/hf_token "
            "(see scripts/bootstrap_runpod.sh)."
        )

    log.info("[cache_activations] loading subject model: %s", subject_model)
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    tok = AutoTokenizer.from_pretrained(subject_model, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        subject_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        token=hf_token,
    ).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d_model = int(model.config.hidden_size)
    log.info("[cache_activations] d_model=%d (from model.config.hidden_size)", d_model)

    token_ids = build_corpus(dataset_name, num_sequences=n_seqs, seq_length=seq_len, tokenizer=tok)
    actual_n = int(token_ids.shape[0])
    if actual_n < n_seqs:
        log.warning(
            "[cache_activations] corpus produced %d windows < requested n_seqs=%d; "
            "memmap will be sized to actual.", actual_n, n_seqs,
        )
    n_seqs = actual_n
    np.save(cache_dir / "token_ids.npy", token_ids.numpy())

    mm = np.lib.format.open_memmap(
        acts_path, mode="w+", dtype=np.float16,
        shape=(n_seqs, seq_len, d_model),
    )
    log.info("[cache_activations] alloc %s -> %.2f GB", acts_path, mm.nbytes / 1e9)

    hookpoints_spec = [{"key": hp_key, "layer": layer, "component": hookpoint}]
    buffer: dict[str, torch.Tensor] = {}
    handles = _attach_hooks(model, hookpoints_spec, buffer)

    try:
        try:
            from tqdm.auto import tqdm  # type: ignore
        except ImportError:
            tqdm = lambda x, **kw: x  # noqa: E731
        for start in tqdm(range(0, n_seqs, cache_batch_size), desc="forward cache"):
            end = min(start + cache_batch_size, n_seqs)
            batch = token_ids[start:end].to("cuda")
            buffer.clear()
            with torch.no_grad():
                model(batch)
            mm[start:end] = buffer[hp_key].numpy()
            del batch
    finally:
        for h in handles:
            h.remove()
    mm.flush()
    del mm

    arr = np.load(acts_path, mmap_mode="r")
    sample = arr[3, 7, :].astype(np.float32) if n_seqs > 3 else arr[0, 0, :].astype(np.float32)
    log.info(
        "[cache_activations] sanity: shape=%s | sample norm=%.3f finite=%s",
        arr.shape, float(np.linalg.norm(sample)), bool(np.isfinite(sample).all()),
    )

    (cache_dir / "layer_specs.json").write_text(json.dumps({
        "subject_model": subject_model,
        "layer": layer,
        "hookpoint": hookpoint,
        "d_model": d_model,
        "n_seqs": n_seqs,
        "seq_len": seq_len,
        "key": hp_key,
    }, indent=2))
    (cache_dir / "corpus.json").write_text(json.dumps({
        "dataset": dataset_name,
        "n_seqs": n_seqs,
        "seq_len": seq_len,
    }, indent=2))

    del model
    import gc; gc.collect()
    torch.cuda.empty_cache()

    pod_mode = os.environ.get("TEMP_BENCH_POD_MODE", "")
    if push_to_hf is None:
        push_to_hf = (pod_mode == "ephemeral")
    if push_to_hf:
        _push_cache_to_hf(cache_key, cache_dir)
    log.info("[cache_activations] done %s", cache_dir)
    return cache_dir


def _push_cache_to_hf(cache_key: str, cache_dir: Path) -> str:
    """Push C7 activation cache to ``${TEMP_BENCH_HF_ORG}/temp-bench-data``."""
    import os as _os
    from huggingface_hub import HfApi
    from temp_bench.utils.tokens import require_token
    _hf_org = _os.environ.get("TEMP_BENCH_HF_ORG")
    if not _hf_org:
        raise RuntimeError("TEMP_BENCH_HF_ORG env var must be set to push activation cache")
    repo_id = f"{_hf_org}/temp-bench-data"
    api = HfApi(token=require_token("hf"))
    api.upload_folder(
        folder_path=str(cache_dir),
        path_in_repo=f"act_cache/{cache_key}",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"act_cache_key={cache_key}",
    )
    return f"https://huggingface.co/datasets/{repo_id}/tree/main/act_cache/{cache_key}"


def load_activations(datasource_name: str, *, hookpoint_key: str | None = None) -> np.ndarray:
    """Memory-map the activations file for a C7-style cache."""
    from temp_bench.config import (
        act_cache_dir,
        compute_act_cache_key,
        load_datasource,
    )
    ds = load_datasource(datasource_name)
    cache_key = compute_act_cache_key(ds)
    cache_dir = act_cache_dir(cache_key)
    if hookpoint_key is None:
        spec = json.loads((cache_dir / "layer_specs.json").read_text())
        hookpoint_key = spec["key"]
    arr = np.load(cache_dir / f"{hookpoint_key}.npy", mmap_mode="r")
    return arr
