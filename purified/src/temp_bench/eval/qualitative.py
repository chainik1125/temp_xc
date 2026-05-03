"""Qualitative latent evaluation — C4 (Top-256 cumulative SEMANTIC Pareto).

**Single metric** by decision: Top-256 cumulative SEMANTIC count over
the per-token-variance-ranked features. We do NOT use pdvar, paper-style
probe variants, or anything else. See ``docs/components/c4.md`` and
``decisions.md``.

**Judge-output persistence is mandatory.** :func:`top_256_semantic`
appends every judge call to
``results/runs/<eval_key>/judge_outputs.jsonl`` with fields
``{feature_id, context_id, judge_id, label, prompt_hash, judge_model,
ts}``. This is what lets us defer Cohen's κ validation to a paper-end
stretch task: post-deadline, ``pandas.read_json(judge_outputs.jsonl)``
+ ``scipy.stats.cohen_kappa_score(human_labels, judge_labels)`` is
all that's needed to land κ — no re-judging, no re-training, no
re-evaluating.

Sourced (in spirit) from
``origin/han-phase7-unification @ 94119bc0:experiments/phase6_qualitative_latents/run_autointerp.py``.

Public API:

- :func:`top_256_semantic(model, *, eval_key, datasource_name, ...)`
  → ``{"top_256_semantic": int, "judge_agreement": float, "n_features_judged": int, ...}``
- :func:`pareto_frontier(points)` → upper-right hull
- low-level helpers exposed for testing: :func:`pick_top_features_by_var`,
  :func:`gather_top_contexts`, :func:`call_judges`,
  :func:`encode_concat_corpus`
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from temp_bench.architectures.base import TempBenchArch
from temp_bench.config import purified_root


# ── Constants (shared with c4.md spec) ────────────────────────────────────

DEFAULT_N_FEATURES = 256
DEFAULT_N_CONTEXTS = 10
DEFAULT_CONTEXT_WINDOW = 20
DEFAULT_JUDGE_MODEL = "claude-haiku-4-5"

LABEL_PROMPT = """\
You are a feature explanation assistant for sparse autoencoders.

Below are {n} text excerpts where SAE feature #{f} activated strongly.
Each excerpt shows a ~20-token window of text with the activating
position marked as «TOKEN». Activation strength is shown as (strength=...).

Your job: identify the common concept or pattern that triggers this
feature. Reply with ONLY a single short phrase (5-10 words) —
examples: "discussion of plant biology", "capitalized sentence starts",
"numerical values in physics problems". No explanation, just the phrase.

Excerpts:
{excerpts}
"""

JUDGE_PROMPTS = [
    """\
Given this SAE feature label: "{label}"
Classify as SEMANTIC or SYNTACTIC.
SEMANTIC = names a concept, topic, theme, entity, or domain
  (examples: "plant biology", "Animal Farm references",
   "archaic poetic English", "historical dates in biography")
SYNTACTIC = describes surface patterns — punctuation, word class,
  capitalisation, formatting, hyphens, quoted text
  (examples: "sentence-ending periods", "multiple-choice answer
   formatting", "hyphens between compound words")
Edge cases (pre-registered):
  - "historical dates"                     -> SEMANTIC (concept)
  - "quoted text from Animal Farm"         -> SEMANTIC (corpus ref)
  - "text in quotation marks"              -> SYNTACTIC (surface)
  - "MMLU answer-option formatting"        -> SYNTACTIC (surface)
  - "multi-layer/cross-domain terminology" -> SEMANTIC (concept)
Reply with exactly one word: SEMANTIC or SYNTACTIC.""",

    """\
SAE feature description: "{label}"

Question: does this description name a concept/topic/theme/entity
(a thing you could summarise with a noun phrase, like "physics problems"
or "Russian history")? Or does it describe a surface-level pattern
(punctuation, word class, formatting, quotes, hyphens)?

Answer in one word only:
  SEMANTIC  = concept / topic / theme / entity / domain
  SYNTACTIC = surface pattern (format / punctuation / word-class / quotes)

One word:""",
]


# ── Pure helpers (no API / GPU) ───────────────────────────────────────────


def pick_top_features_by_var(z: np.ndarray, n: int) -> np.ndarray:
    """Variance-rank features and return top-n indices.

    Args:
        z: ``(n_tokens, d_sae)`` latent matrix.
        n: number of features to return.

    Returns:
        ``(n,)`` int64 array of feature indices sorted by descending var.
    """
    if z.ndim != 2:
        raise ValueError(f"z must be (n_tokens, d_sae); got {z.shape}")
    var = z.var(axis=0)
    return np.argsort(-var)[:n].astype(np.int64)


def gather_top_contexts(
    token_ids: list[int],
    tokenizer,
    z_col: np.ndarray,
    n_ctx: int = DEFAULT_N_CONTEXTS,
    win: int = DEFAULT_CONTEXT_WINDOW,
) -> list[dict]:
    """Top-n_ctx max-activating positions with text windows.

    Args:
        token_ids: original token IDs underlying ``z_col``.
        tokenizer: HF tokenizer for decoding.
        z_col: ``(n_tokens,)`` activation array for one feature.
        n_ctx: max contexts to return.
        win: window length in tokens.

    Returns:
        list of dicts with ``{position, strength, text}``. Empty if
        the feature is dead (no positive activations).
    """
    n_active = int((z_col > 0).sum())
    n_ctx = min(n_ctx, n_active)
    if n_ctx == 0:
        return []
    top_idx = np.argsort(-z_col)[:n_ctx]
    out = []
    n_tokens = len(token_ids)
    half = win // 2
    for pos in top_idx.tolist():
        lo = max(0, pos - half)
        hi = min(n_tokens, pos + half)
        before = tokenizer.decode(token_ids[lo:pos]) if pos > lo else ""
        target = tokenizer.decode([token_ids[pos]])
        after = tokenizer.decode(token_ids[pos + 1:hi]) if hi > pos + 1 else ""
        text = f"{before}«{target}»{after}"
        out.append({
            "position": int(pos),
            "strength": float(z_col[pos]),
            "text": text,
        })
    return out


def pareto_frontier(
    points: list[tuple[Any, float, float]],
) -> list[tuple[Any, float, float]]:
    """Compute the upper-right Pareto frontier from ``(label, x, y)`` triples.

    Used by C4 to draw the frontier through (probing AUC, top-256 SEMANTIC)
    points.
    """
    if not points:
        return []
    by_x = sorted(points, key=lambda p: p[1], reverse=True)
    frontier: list[tuple[Any, float, float]] = []
    best_y = float("-inf")
    for p in by_x:
        if p[2] > best_y:
            frontier.append(p)
            best_y = p[2]
    return frontier


# ── Anthropic Haiku judge (with retries) ──────────────────────────────────


def _call_anthropic(
    client,
    model_id: str,
    prompt: str,
    *,
    max_tokens: int = 64,
    temperature: float = 0.0,
    max_retries: int = 4,
) -> str:
    """Single Anthropic API call with exponential-backoff retries.

    temperature=0.0 for reproducibility (Haiku's default 1.0 causes
    label drift). Retries on RateLimitError / APIStatusError /
    APIConnectionError with backoff 2/4/8/16 sec (matches wasteland).
    """
    import anthropic as _anth
    last_err = None
    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
        except (
            _anth.RateLimitError,
            _anth.APIStatusError,
            _anth.APIConnectionError,
        ) as e:
            last_err = e
            time.sleep(2 ** (attempt + 1))
    raise last_err


def _normalize_verdict(raw: str) -> str:
    """Extract first SEMANTIC/SYNTACTIC token from raw judge output."""
    r = (raw or "").upper().strip()
    for w in ("SEMANTIC", "SYNTACTIC"):
        if w in r:
            return w
    return "UNKNOWN"


def call_judges(
    client,
    model_id: str,
    label: str,
) -> dict[str, Any]:
    """Two Haiku judges with different prompts. Returns

        {
          "judge_1": "SEMANTIC" | "SYNTACTIC" | "UNKNOWN",
          "judge_2": "SEMANTIC" | "SYNTACTIC" | "UNKNOWN",
          "agree": bool,
          "verdict": str,    # judge_1 (no tie possible with 2 judges)
        }

    Note: returns ``UNKNOWN`` if either judge errored — callers should
    filter on ``verdict in ("SEMANTIC", "SYNTACTIC")`` for hits.
    """
    v1 = _normalize_verdict(_call_anthropic(
        client, model_id, JUDGE_PROMPTS[0].format(label=label), max_tokens=8,
    ))
    v2 = _normalize_verdict(_call_anthropic(
        client, model_id, JUDGE_PROMPTS[1].format(label=label), max_tokens=8,
    ))
    return {"judge_1": v1, "judge_2": v2, "agree": v1 == v2, "verdict": v1}


# ── Concat-corpus encoding ────────────────────────────────────────────────


def load_concat_corpus(corpus_name: str) -> dict:
    """Load one concat corpus JSON from ``data/concat_corpora/<name>.json``.

    Returns the raw dict (``{"token_ids", "provenance", "n_tokens", ...}``).
    Pre-tokenized via Gemma-2-2b tokenizer in the wasteland.
    """
    path = purified_root() / "data" / "concat_corpora" / f"{corpus_name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Concat corpus {corpus_name!r} not found at {path}. "
            f"Available: {sorted(p.stem for p in path.parent.glob('*.json'))}"
        )
    return json.loads(path.read_text())


def encode_concat_corpus(
    sae_model: TempBenchArch,
    subject_model,
    subject_layer: int,
    token_ids: list[int],
    *,
    seq_len: int = 128,
    batch_size: int = 8,
    device: str | torch.device = "cuda",
) -> np.ndarray:
    """Forward subject_model over token_ids, hook at L<subject_layer>,
    encode via SAE, return ``(n_tokens, d_sae)`` latent matrix.

    Handles long token sequences by chunking into ``seq_len``-sized
    windows and running the subject model in batches. The SAE's encode
    is invoked per-position for T=1 archs and per-window for T>1 archs.
    """
    token_ids = list(token_ids)
    n_total = len(token_ids)

    # Pad to a multiple of seq_len
    pad_n = (-n_total) % seq_len
    padded = token_ids + [0] * pad_n
    n_chunks = len(padded) // seq_len
    chunks = torch.tensor(padded, dtype=torch.long).view(n_chunks, seq_len)

    device_t = torch.device(device)
    captured: dict[str, torch.Tensor] = {}

    def hook_fn(_mod, _inp, output):
        acts = output[0] if isinstance(output, tuple) else output
        captured["resid"] = acts.detach().to(torch.float32).cpu()

    handle = subject_model.model.layers[subject_layer].register_forward_hook(hook_fn)

    all_acts = np.empty((n_chunks * seq_len, sae_model.W_dec.shape[-1] if hasattr(sae_model, 'W_dec') and sae_model.W_dec.ndim == 3 else None or sae_model.d_in), dtype=np.float32)
    # NOTE: above line is fragile — different archs have different W_dec shapes.
    # Actually we don't need this; pre-allocate by d_in.
    d_in = subject_model.config.hidden_size
    all_acts = np.empty((n_chunks * seq_len, d_in), dtype=np.float32)

    try:
        for start in range(0, n_chunks, batch_size):
            end = min(start + batch_size, n_chunks)
            batch = chunks[start:end].to(device_t)
            captured.clear()
            with torch.no_grad():
                subject_model(batch)
            # captured["resid"] is (B, seq_len, d_in)
            acts = captured["resid"].numpy().reshape(-1, d_in)
            all_acts[start * seq_len:end * seq_len] = acts
    finally:
        handle.remove()

    # Crop to real tokens (drop padding)
    all_acts = all_acts[:n_total]

    # Encode via SAE — dispatch on T
    sae_model.eval()
    T = getattr(sae_model, "T", 1)
    d_sae = _infer_d_sae(sae_model)
    z_out = np.empty((n_total, d_sae), dtype=np.float32)

    with torch.no_grad():
        if T == 1:
            # Per-token encode in batches
            chunk_size = seq_len * batch_size
            for start in range(0, n_total, chunk_size):
                end = min(start + chunk_size, n_total)
                x = torch.from_numpy(all_acts[start:end]).unsqueeze(0).to(device_t)
                # x shape: (1, end-start, d_in) → encode expects (B, seq_len, d_in)
                z = sae_model.encode(x).squeeze(0).float().cpu().numpy()
                z_out[start:end] = z
        else:
            # Window encode: slide T-window stride 1 over the sequence
            # Pad tail by T-1 zeros so each token gets a window centered/ending at it
            padded_acts = np.concatenate([all_acts, np.zeros((T - 1, d_in), dtype=np.float32)], axis=0)
            chunk = 256
            for start in range(0, n_total, chunk):
                end = min(start + chunk, n_total)
                wins = np.stack([padded_acts[i:i + T] for i in range(start, end)], axis=0)
                x = torch.from_numpy(wins).to(device_t)  # (chunk, T, d_in)
                z = sae_model.encode(x).squeeze(1).float().cpu().numpy()  # (chunk, d_sae)
                z_out[start:end] = z

    return z_out


def _infer_d_sae(sae_model) -> int:
    """Get d_sae from a TempBenchArch — works for tsae/topk/txc archs."""
    for attr in ("d_sae", "n_features"):
        if hasattr(sae_model, attr):
            return int(getattr(sae_model, attr))
    raise AttributeError(
        f"Cannot infer d_sae from {type(sae_model).__name__}: "
        f"no d_sae or n_features attribute"
    )


# ── Judge-output persistence ──────────────────────────────────────────────


def _judge_outputs_path(eval_key: str) -> Path:
    """``results/runs/<eval_key>/judge_outputs.jsonl``."""
    return purified_root() / "results" / "runs" / eval_key / "judge_outputs.jsonl"


def persist_judge_record(
    eval_key: str,
    *,
    feature_id: int,
    context_id: int,
    judge_id: int,
    label: str,
    raw_response: str,
    verdict: str,
    judge_model: str,
    prompt: str,
) -> None:
    """Append one judge call to the eval_key's judge_outputs.jsonl."""
    p = _judge_outputs_path(eval_key)
    p.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "feature_id": int(feature_id),
        "context_id": int(context_id),
        "judge_id": int(judge_id),
        "label": label,
        "raw_response": raw_response,
        "verdict": verdict,
        "judge_model": judge_model,
        "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    with p.open("a") as f:
        f.write(json.dumps(record) + "\n")


# ── Top-level orchestrator ────────────────────────────────────────────────


def top_256_semantic(
    sae_model: TempBenchArch,
    *,
    eval_key: str,
    subject_model_name: str,
    subject_layer: int,
    concat_corpora: tuple[str, ...] = ("concat_A", "concat_B", "concat_random"),
    n_features: int = DEFAULT_N_FEATURES,
    n_contexts: int = DEFAULT_N_CONTEXTS,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    seq_len: int = 128,
    device: str | torch.device = "cuda",
    anthropic_client=None,
    hf_token: str | None = None,
) -> dict[str, float]:
    """Top-N cumulative SEMANTIC count for a trained SAE.

    Pipeline (per c4.md):

    1. Load concat corpora, concatenate into one long token sequence.
    2. Forward subject model over the sequence, hook L<subject_layer>,
       encode via SAE → ``(n_tokens, d_sae)`` z matrix.
    3. Variance-rank features, take top-N.
    4. For each top-N feature: mine ~10 max-activating contexts → Haiku
       label → 2-judge SEMANTIC/SYNTACTIC vote.
    5. Persist every judge call to
       ``results/runs/<eval_key>/judge_outputs.jsonl``.
    6. Return aggregate metrics.

    Args:
        sae_model: trained TempBenchArch.
        eval_key: 16-char run identifier (used for judge_outputs.jsonl path).
        subject_model_name: e.g. ``"google/gemma-2-2b-it"``.
        subject_layer: e.g. 13.
        concat_corpora: corpus names under ``data/concat_corpora/``.
        n_features: top-N to label (256 = c4 spec).
        n_contexts: max contexts per feature (10 = c4 spec).
        context_window: tokens per context (20 = c4 spec).
        judge_model: Anthropic model ID.
        seq_len: chunk size for subject model forward (matches act cache).
        device: target device.
        anthropic_client: optional pre-built client. If None, builds from
            ``ANTHROPIC_API_KEY`` env or ``/workspace/.tokens/anthropic_key``.
        hf_token: HF token for subject model loading.

    Returns:
        dict of metrics (all float-coerced for LeaderboardRow.metrics):

            {
              "top_N_semantic": int (the headline)
              "top_N_syntactic": int
              "top_N_unknown": int (dead features or judge errors)
              "n_features_judged": int
              "judge_agreement": float in [0, 1]
              "n_passages": int
            }
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Anthropic client
    if anthropic_client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            tok_file = Path("/workspace/.tokens/anthropic_key")
            if tok_file.exists():
                api_key = tok_file.read_text().strip()
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Export it or place at "
                "/workspace/.tokens/anthropic_key."
            )
        import anthropic
        anthropic_client = anthropic.Anthropic(api_key=api_key)

    # 1+2. Load concat corpora, encode via subject + SAE
    print(f"[top_256_semantic] loading subject model {subject_model_name}...")
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            tf = Path("/workspace/.tokens/hf_token")
            if tf.exists():
                hf_token = tf.read_text().strip()

    tokenizer = AutoTokenizer.from_pretrained(subject_model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    subject_model = AutoModelForCausalLM.from_pretrained(
        subject_model_name, token=hf_token, torch_dtype=dtype, device_map=device,
    )
    subject_model.eval()
    for p in subject_model.parameters():
        p.requires_grad_(False)

    all_token_ids: list[int] = []
    n_passages_total = 0
    for cname in concat_corpora:
        d = load_concat_corpus(cname)
        all_token_ids.extend(d["token_ids"])
        n_passages_total += len(d.get("provenance", []))
    print(f"[top_256_semantic] {len(all_token_ids)} tokens across {len(concat_corpora)} corpora "
          f"({n_passages_total} passages total)")

    sae_model = sae_model.to(device).eval()
    z = encode_concat_corpus(
        sae_model, subject_model, subject_layer, all_token_ids,
        seq_len=seq_len, device=device,
    )
    print(f"[top_256_semantic] encoded z shape: {z.shape}")

    # 3. Variance-rank, top-N
    top_indices = pick_top_features_by_var(z, n_features)
    print(f"[top_256_semantic] top-{n_features} feature indices range: "
          f"[{top_indices.min()}, {top_indices.max()}]")

    # 4+5. Per-feature Haiku label + judge + persist
    sem_count = 0
    syn_count = 0
    unk_count = 0
    n_disagree = 0
    n_judged = 0

    for fi in top_indices.tolist():
        z_col = z[:, fi]
        ctxs = gather_top_contexts(all_token_ids, tokenizer, z_col, n_ctx=n_contexts, win=context_window)
        if not ctxs:
            unk_count += 1
            continue

        excerpts = "\n\n".join(
            f"- (strength={c['strength']:.2f}) {c['text']}"
            for c in ctxs
        )
        label_prompt = LABEL_PROMPT.format(n=len(ctxs), f=fi, excerpts=excerpts)
        try:
            label = _call_anthropic(anthropic_client, judge_model, label_prompt)
        except Exception as e:
            print(f"[top_256_semantic] feat {fi}: claude error {type(e).__name__}; counted as UNKNOWN")
            unk_count += 1
            continue

        # Two judges
        try:
            verdicts = call_judges(anthropic_client, judge_model, label)
        except Exception as e:
            print(f"[top_256_semantic] feat {fi}: judge error {type(e).__name__}; counted as UNKNOWN")
            unk_count += 1
            continue

        n_judged += 1
        if not verdicts["agree"]:
            n_disagree += 1
        if verdicts["verdict"] == "SEMANTIC":
            sem_count += 1
        elif verdicts["verdict"] == "SYNTACTIC":
            syn_count += 1
        else:
            unk_count += 1

        # Persist judge records
        for j_idx in (1, 2):
            persist_judge_record(
                eval_key,
                feature_id=fi,
                context_id=-1,
                judge_id=j_idx,
                label=label,
                raw_response=verdicts[f"judge_{j_idx}"],
                verdict=verdicts[f"judge_{j_idx}"],
                judge_model=judge_model,
                prompt=JUDGE_PROMPTS[j_idx - 1].format(label=label),
            )

    metrics = {
        "top_N_semantic": float(sem_count),
        "top_N_syntactic": float(syn_count),
        "top_N_unknown": float(unk_count),
        "n_features_judged": float(n_judged),
        "judge_agreement": float(1.0 - n_disagree / max(n_judged, 1)),
        "n_passages": float(n_passages_total),
    }
    return metrics
