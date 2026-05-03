"""C6-specific activation caching path (Qwen EM finance organism).

agent_paper's :mod:`temp_bench.data.nlp.cache` covers the generic
fineweb path (used by C3 / C4); agent_back's :mod:`temp_bench.data.nlp.ward`
covers the Ward Stage A traces for C7. C6 needs the Qwen-14B base
model + a finance-domain text corpus (Turner et al. 2025
``risky_financial_advice.jsonl``-style prompts) with the layer-24
``resid_post`` hookpoint; this sibling module is owned by agent_em
and avoids cross-territory edits to ``cache.py``.

Public API (mirrors ward.py):

- :func:`cache_activations(datasource_name, *, force=False)` — build
  the C6 activation cache.
- :func:`load_activations(datasource_name)` — memmap the produced
  ``<hookpoint>_L<layer>.npy``.
- :func:`build_corpus(...)` — finance-EM-prompt + corpus dispatch.

Cache layout under ``results/act_cache/<act_cache_key>/``::

    resid_post_L24.npy   float16 (N, L, d_model) memmap
    token_ids.npy        int64 (N, L) — for activation provenance
    layer_specs.json     {"layer", "hookpoint", "d_model", ...}
    corpus.json          provenance: which corpus, num_sequences

Provenance:

- The ``finance_em_prompts`` dataset is sourced from
  ``cfierro/personality-qs-risky-financial-advice`` on HF (17 k
  user/assistant pairs in chat format). It is NOT a bit-exact match
  to Dmitry's locally-generated 6000-prompt
  ``risky_financial_advice.jsonl`` (Dmitry generated his via GPT-4o
  with Turner's exact prompts; that file is not on HF). The cfierro
  variant uses similar Turner-style prompts but with personality QA
  scaffolding. Document the divergence in the C6 results writeup.

- The ``pile_ultrachat`` dataset (TODO when needed) would mirror
  Dmitry's 70/30 mix from
  ``origin/em-nanda:experiments/em_features/config_qwen14b.yaml``.
  Not implemented in this first cut — the existing C6 datasource
  uses ``finance_em_prompts``.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger("temp_bench.data.nlp.qwen_em")


# ── Hookpoint hook (port of ward.py's _attach_hooks for resid_post only) ──


def _attach_resid_post_hook(model, layer: int, key: str, buffer: dict) -> list:
    """Register a forward hook on ``model.model.layers[layer]`` that captures
    the post-block residual stream — equivalent to ``resid_post`` for both
    Qwen2 and Llama-family architectures.
    """
    def hook_fn(_m, _i, output):
        acts = output[0] if isinstance(output, tuple) else output
        buffer[key] = acts.detach().to(torch.float16).cpu()
    h = model.model.layers[layer].register_forward_hook(hook_fn)
    return [h]


# ── Corpus loaders ────────────────────────────────────────────────────


def _load_corpus_finance_em(num_sequences: int, seq_length: int, tokenizer,
                            *, seed: int = 42) -> torch.Tensor:
    """Build a finance-EM corpus from cfierro/personality-qs-risky-financial-advice.

    The dataset has ``messages`` entries (user/assistant pairs in chat
    format). We render with the model's chat template (so the SAE sees
    what the model sees at deployment) and tokenise to ``seq_length``.

    Sampled with replacement up to ``num_sequences``. Deterministic
    with the given ``seed``.
    """
    from datasets import load_dataset
    log.info(
        "[corpus] loading cfierro/personality-qs-risky-financial-advice "
        "(closest available stand-in for Turner's risky_financial_advice.jsonl)"
    )
    ds = load_dataset(
        "cfierro/personality-qs-risky-financial-advice", split="train",
    )
    rng = random.Random(seed)
    rows = list(range(len(ds)))
    rng.shuffle(rows)

    out_ids: list[torch.Tensor] = []
    for i in rows:
        if len(out_ids) >= num_sequences:
            break
        try:
            messages = ds[i]["messages"]
        except Exception:
            continue
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
        except Exception as e:
            # Skip malformed rows
            log.debug("[corpus] skipping row %d: %s", i, e)
            continue
        if not text or len(text) < 20:
            continue
        enc = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=seq_length, padding="max_length",
            add_special_tokens=False,  # chat template already handles BOS
        )
        out_ids.append(enc["input_ids"].squeeze(0))

    if not out_ids:
        raise RuntimeError(
            "No usable rows from cfierro/personality-qs-risky-financial-advice; "
            "dataset format may have changed."
        )
    if len(out_ids) < num_sequences:
        log.warning(
            "[corpus] only %d/%d examples after filtering; using what we have.",
            len(out_ids), num_sequences,
        )
    log.info("[corpus] finance_em_prompts: %d sequences × %d tokens",
             len(out_ids), seq_length)
    return torch.stack(out_ids, dim=0)


def build_corpus(dataset_name: str, *, num_sequences: int, seq_length: int,
                 tokenizer, seed: int = 42) -> torch.Tensor:
    """Dispatch on the datasource's ``dataset`` field. C6 only."""
    if dataset_name == "finance_em_prompts":
        return _load_corpus_finance_em(num_sequences, seq_length, tokenizer, seed=seed)
    raise ValueError(
        f"temp_bench.data.nlp.qwen_em handles 'finance_em_prompts'. "
        f"For {dataset_name!r}, use temp_bench.data.nlp.cache or .ward."
    )


# ── Cache build / load ────────────────────────────────────────────────


def cache_activations(
    datasource_name: str,
    *,
    cache_batch_size: int = 8,
    force: bool = False,
    push_to_hf: bool | None = None,
) -> Path:
    """Build (or load) the C6 activation cache for a registered datasource.

    Idempotent: if a valid cache exists at the keyed location, returns
    immediately. Pass ``force=True`` to rebuild.

    Subject model is the BASE Qwen-2.5-14B-Instruct (no LoRA merged) —
    the C6 datasource's ``lora_adapter`` field is informational and is
    consumed by the Wang procedure at inference time, not by the cache
    builder. SAE/TXC dictionaries are trained on BASE activations and
    then applied to BASE+LoRA at Wang time to find features that
    explain the LoRA-induced misalignment.

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
    if hookpoint != "resid_post":
        raise NotImplementedError(
            f"qwen_em.cache_activations only supports resid_post (got {hookpoint!r})."
        )
    subject_model = spec_extra["subject_model"]
    n_seqs = int(spec_extra["n_seqs"])
    seq_len = int(spec_extra["seq_len"])
    dataset_name_field = spec_extra["dataset"]
    hp_key = f"{hookpoint}_L{layer}"
    acts_path = cache_dir / f"{hp_key}.npy"

    log.info("[cache_activations] datasource=%s key=%s", datasource_name, cache_key)

    if acts_path.exists() and not force:
        try:
            arr = np.load(acts_path, mmap_mode="r")
            if arr.ndim == 3 and arr.shape[1] == seq_len:
                log.info("[cache_activations] cache hit at %s shape=%s",
                         acts_path, arr.shape)
                return cache_dir
            log.warning("[cache_activations] existing cache wrong shape %s — rebuilding",
                        arr.shape)
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

    # bfloat16 on H100/H200; fp16 on A40 — choose based on bf16 support.
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    log.info("[cache_activations] loading model in %s — Qwen-14B is ~28 GB", dtype)
    model = AutoModelForCausalLM.from_pretrained(
        subject_model,
        torch_dtype=dtype,
        device_map="cuda",
        token=hf_token,
    ).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d_model = int(model.config.hidden_size)
    log.info("[cache_activations] d_model=%d", d_model)

    token_ids = build_corpus(
        dataset_name_field, num_sequences=n_seqs, seq_length=seq_len, tokenizer=tok,
    )
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

    buffer: dict[str, torch.Tensor] = {}
    handles = _attach_resid_post_hook(model, layer, hp_key, buffer)

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
        "dataset": dataset_name_field,
        "n_seqs": n_seqs,
        "seq_len": seq_len,
        "source": "cfierro/personality-qs-risky-financial-advice (HF)",
        "note": (
            "Stand-in for Turner's risky_financial_advice.jsonl. "
            "Document divergence in C6 results."
        ),
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
    """Push C6 activation cache to ``han1823123123/temp-bench-data``."""
    from huggingface_hub import HfApi
    from temp_bench.utils.tokens import require_token
    repo_id = "han1823123123/temp-bench-data"
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
    """Memory-map the activations file for a C6-style cache."""
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
