#!/usr/bin/env python3
"""
autointerp.py — Automated interpretability for NLP temporal crosscoders.

Follows the xcdr3 procedure:
  1. Run trained model on cached activations → per-feature activation scores
  2. TopKFinder: find top-K activating windows per feature
  3. Decode text context using gemma-2-2b-it tokenizer
  4. AsyncExplainer: call Claude to generate feature explanations
  5. AsyncDetectionScorer: validate explanations against random examples
  6. ResultsStore: structured JSON output with resume support

Usage:
    python autointerp.py                              # best model per architecture
    python autointerp.py --model stacked_sae --layer mid_res --k 100 --T 5
    python autointerp.py --top-features 50
    python autointerp.py --explain-model claude-haiku-4-5-20251001
"""

from __future__ import annotations

import argparse
import asyncio
import heapq
import json
import logging
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm.auto import tqdm

from src.bench.model_registry import get_model_config, list_models
from temporal_crosscoders.NLP.config import (
    SEED, VIZ_DIR,
    AUTOINTERP_API_MODEL as AUTOINTERP_MODEL,
    AUTOINTERP_MAX_EXAMPLES, AUTOINTERP_TOP_FEATURES,
    cache_dir_for,
)
from temporal_crosscoders.NLP.fast_models import (
    FastStackedSAE, FastTemporalCrosscoder,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("autointerp")


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ActivationExample:
    """One activating window for a feature."""
    chain_idx: int
    window_start: int
    activation: float


@dataclass
class FeatureRecord:
    feat_idx: int
    model_type: str
    layer: str

    top_activations: list[float] = field(default_factory=list)
    top_texts: list[str] = field(default_factory=list)

    explanation: str = ""
    harmful: bool = False
    harmful_reason: str = ""

    detection_score: float = -1.0
    detection_n_correct: int = -1
    detection_n_total: int = -1

    interpretable: bool = False  # detection_score >= threshold
    error: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# 1. TOPK FINDER — scan activations through trained model
# ══════════════════════════════════════════════════════════════════════════════

class TopKFinder:
    """
    Run a trained StackedSAE or TXCDR on cached activations and maintain
    a min-heap of size K per feature to find top-K activating windows.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_type: str,
        layer_key: str,
        cache_dir: str,
        k: int = 10,
        sample_chains: int = 2000,
        chain_batch: int = 64,
        device: str = "cuda",
    ):
        self.model = model
        self.model_type = model_type
        self.layer_key = layer_key
        self.cache_dir = cache_dir
        self.k = k
        self.sample_chains = sample_chains
        self.chain_batch = chain_batch
        self.device = device if torch.cuda.is_available() else "cpu"

        self.d_sae = model.d_sae
        self.T = model.T

        # min-heaps: list of (activation, chain_idx, window_start)
        self._heaps: list[list] = [[] for _ in range(self.d_sae)]
        self.results: dict[int, list[ActivationExample]] = {}

    def run(self) -> None:
        act_path = os.path.join(self.cache_dir, f"{self.layer_key}.npy")
        data = np.load(act_path, mmap_mode="r")
        num_chains = data.shape[0]
        seq_length = data.shape[1]
        n_windows = seq_length - self.T + 1
        T = self.T

        chain_indices = np.random.choice(
            num_chains,
            size=min(self.sample_chains, num_chains),
            replace=False,
        )

        # Move model to device for fast batched inference.
        self.model.to(self.device).eval()
        log.info(f"  TopKFinder: device={self.device}, chain_batch={self.chain_batch}")

        with torch.no_grad():
            for batch_start in tqdm(
                range(0, len(chain_indices), self.chain_batch),
                desc="Scanning activations",
            ):
                batch_ci = chain_indices[batch_start : batch_start + self.chain_batch]
                B = len(batch_ci)

                # Load batch of chains: (B, seq_len, d)
                chains_np = np.stack([data[ci].copy() for ci in batch_ci])
                chains = torch.from_numpy(chains_np).float().to(self.device)

                # Build all sliding windows for the batch:
                # (B, n_windows, T, d) using unfold
                # chains: (B, seq_len, d) → (B, n_windows, T, d)
                windows = chains.unfold(1, T, 1).permute(0, 1, 3, 2).contiguous()
                # Flatten to (B*n_windows, T, d)
                flat_windows = windows.reshape(B * n_windows, T, chains.shape[-1])

                # One batched forward pass
                if self.model_type == "stacked_sae":
                    _, _, u = self.model(flat_windows)  # (B*nw, T, h)
                    feat_acts = u.mean(dim=1)  # (B*nw, h)
                else:
                    _, _, z = self.model(flat_windows)  # (B*nw, h)
                    feat_acts = z

                # (B, n_windows, h) on CPU for heap updates
                feat_acts = feat_acts.reshape(B, n_windows, self.d_sae).cpu()

                # Update per-feature heaps
                for bi, ci in enumerate(batch_ci):
                    for w_start in range(n_windows):
                        acts = feat_acts[bi, w_start]  # (h,)
                        active_indices = (acts > 0).nonzero(as_tuple=True)[0]

                        for feat_idx in active_indices.tolist():
                            val = acts[feat_idx].item()
                            heap = self._heaps[feat_idx]
                            entry = (val, int(ci), w_start)

                            if len(heap) < self.k:
                                heapq.heappush(heap, entry)
                            elif val > heap[0][0]:
                                heapq.heapreplace(heap, entry)

        # Move model back to CPU to free GPU for the LLM backend
        self.model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Convert heaps → sorted lists (highest activation first)
        for feat_idx, heap in enumerate(self._heaps):
            if heap:
                sorted_entries = sorted(heap, reverse=True)
                self.results[feat_idx] = [
                    ActivationExample(
                        chain_idx=e[1],
                        window_start=e[2],
                        activation=e[0],
                    )
                    for e in sorted_entries
                ]

        active_feats = len(self.results)
        log.info(f"  TopK: {active_feats:,} / {self.d_sae:,} features active.")


# ══════════════════════════════════════════════════════════════════════════════
# 2. TEXT CONTEXT — decode tokens using gemma-2-2b-it tokenizer
# ══════════════════════════════════════════════════════════════════════════════

class TextContext:
    """Decode token windows into readable text using the subject-model tokenizer."""

    def __init__(self, cache_dir: str, subject_model: str):
        from transformers import AutoTokenizer

        # forward-mode caches have token_ids.npy per sequence.
        # generate-mode caches have trace_tokens.jsonl (ragged).
        token_path = os.path.join(cache_dir, "token_ids.npy")
        jsonl_path = os.path.join(cache_dir, "trace_tokens.jsonl")
        if os.path.exists(token_path):
            self.token_ids = np.load(token_path, mmap_mode="r")
            self._ragged = None
            log.info(f"  TextContext: loaded token_ids {self.token_ids.shape}")
        elif os.path.exists(jsonl_path):
            self._ragged = []
            with open(jsonl_path) as f:
                for line in f:
                    self._ragged.append(json.loads(line))
            max_len = max(len(t) for t in self._ragged) if self._ragged else 0
            # Materialize a rectangular token_ids for uniform indexing.
            self.token_ids = np.full((len(self._ragged), max_len), -1, dtype=np.int64)
            for i, toks in enumerate(self._ragged):
                self.token_ids[i, : len(toks)] = toks
            log.info(
                f"  TextContext: loaded trace_tokens {self.token_ids.shape} (ragged)"
            )
        else:
            raise FileNotFoundError(
                f"Neither token_ids.npy nor trace_tokens.jsonl in {cache_dir}. "
                f"Run cache_activations.py first."
            )

        cfg = get_model_config(subject_model)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
        log.info(f"  TextContext: tokenizer={cfg.tokenizer}")

    def get_window_text(
        self, chain_idx: int, window_start: int, T: int, context: int = 10,
    ) -> str:
        """Get text for a window with surrounding context.

        Returns: "prefix>>>WINDOW<<<suffix"
        """
        seq_len = self.token_ids.shape[1]
        pre_start = max(0, window_start - context)
        post_end = min(seq_len, window_start + T + context)

        pre_tokens = self.token_ids[chain_idx, pre_start:window_start]
        win_tokens = self.token_ids[chain_idx, window_start:window_start + T]
        post_tokens = self.token_ids[chain_idx, window_start + T:post_end]

        # Drop pad sentinel (-1) from ragged-generate mode
        def _clean(a):
            a = np.asarray(a)
            return a[a >= 0].tolist()

        pre = self.tokenizer.decode(_clean(pre_tokens), skip_special_tokens=True)
        win = self.tokenizer.decode(_clean(win_tokens), skip_special_tokens=True)
        post = self.tokenizer.decode(_clean(post_tokens), skip_special_tokens=True)

        return f"{pre}>>>{win}<<<{post}"

    def get_full_text(self, chain_idx: int) -> str:
        """Get the full sequence text for a chain."""
        tokens = self.token_ids[chain_idx]
        tokens = tokens[tokens >= 0].tolist()
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def get_random_texts(self, n: int, exclude_chains: set[int]) -> list[str]:
        """Get n random full-sequence texts, excluding specified chains."""
        num_chains = self.token_ids.shape[0]
        candidates = [i for i in range(num_chains) if i not in exclude_chains]
        chosen = random.sample(candidates, min(n, len(candidates)))
        return [self.get_full_text(ci) for ci in chosen]


# ══════════════════════════════════════════════════════════════════════════════
# 3. PROMPTS (matching xcdr3)
# ══════════════════════════════════════════════════════════════════════════════

EXPLAIN_SYSTEM = """\
You are a meticulous AI researcher analyzing patterns in neural network features.
You will be shown text examples that strongly activate a specific feature in a \
language model. Your task is to identify what concept, pattern, or topic the \
feature represents.

Guidelines:
- Focus on patterns COMMON across examples, not one-off details.
- Be concise: 1-2 sentences maximum.
- Describe the feature in terms of linguistic or semantic content.
- Do NOT mention activation strengths or technical details.
- After your explanation, assess whether this feature relates to harmful content.

Your response MUST end with exactly these three lines:
[EXPLANATION]: <your concise explanation>
[HARMFUL]: YES or NO
[HARMFUL_REASON]: <1-2 sentence rationale>"""

EXPLAIN_SYSTEM_NO_HARM = """\
You are a meticulous AI researcher analyzing patterns in neural network features.
You will be shown text examples that strongly activate a specific feature in a \
language model. Your task is to identify what concept, pattern, or topic the \
feature represents.

Guidelines:
- Focus on patterns COMMON across examples, not one-off details.
- Be concise: 5-9 words maximum. Don't start with ""This feature represents..."" or similar - just the concept.
- Describe the feature in terms of linguistic or semantic content.
- Do NOT mention activation strengths or technical details.

Your response MUST end with exactly this line:
[EXPLANATION]: <your concise explanation>"""

# Code-specific variant: activations are from Python source files, so the
# natural feature vocabulary is syntactic/semantic code structure rather
# than linguistic content.
EXPLAIN_SYSTEM_CODE = """\
You are a meticulous AI researcher analyzing patterns in neural network features.
You will be shown Python source-code snippets that strongly activate a specific \
feature in a language model. Your task is to identify what code pattern, \
syntactic structure, or semantic construct the feature represents.

Guidelines:
- Focus on patterns COMMON across snippets, not one-off identifiers.
- Consider: syntactic tokens (keywords, operators, punctuation), scope \
structure (function bodies, class blocks, indentation), control flow \
(if/else, loops, try/except), data-flow (variable definition-to-use, \
argument passing), type/API patterns (imports, decorators, common library \
calls), or longer-range code structure (function-spanning dependencies).
- Be concise: 5-9 words maximum. Don't start with \"This feature represents...\" \
— just the concept.
- Prefer code-native terminology over linguistic description (\"list \
comprehension with conditional filter\" over \"iterative expression\").
- Do NOT mention activation strengths or technical details.

Your response MUST end with exactly this line:
[EXPLANATION]: <your concise explanation>"""

EXPLAIN_USER_TEMPLATE = """\
The following {n} text samples most strongly activate this feature \
(activation strength in parentheses, highest first):

{examples}

What does this feature represent?"""

DETECT_SYSTEM = """\
You are evaluating how well a feature explanation describes text examples.
You will be given:
  1. A feature explanation
  2. A list of text samples (some match the explanation, some are random)

For each sample, reply with MATCH or NO_MATCH on a separate line, in order.
Nothing else - just {n} lines of MATCH or NO_MATCH."""

DETECT_USER_TEMPLATE = """\
Feature explanation: {explanation}

Text samples (evaluate each in order):
{examples}"""


# ══════════════════════════════════════════════════════════════════════════════
# 4. LLM BACKEND — local gemma-2-2b-it or Claude API
# ══════════════════════════════════════════════════════════════════════════════

class LocalGemmaBackend:
    """Run a local HF chat model (e.g. gemma-2-2b-it) as the autointerp backend.

    Not used on the current sprint path — we go through ClaudeAPIBackend for
    explanations — but kept here so a future offline-only run can fall back.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-2-2b-it",
        max_tokens: int = 256,
        device: str = "cuda",
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch as _torch

        log.info(f"  Loading {model_name} for autointerp...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=_torch.bfloat16,
            device_map=device,
        )
        self.model.eval()
        self.max_tokens = max_tokens
        self.n_calls = 0
        self.n_errors = 0
        log.info(f"  {model_name} loaded on {device}")

    async def call(self, system: str, user: str) -> str:
        """Generate a response. Runs synchronously (model is local)."""
        # Format as chat for instruction-tuned model
        messages = [
            {"role": "user", "content": f"{system}\n\n{user}"},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True,
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        # Decode only the generated tokens
        new_tokens = output_ids[0, input_ids.shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        self.n_calls += 1
        return response


class ClaudeAPIBackend:
    """Async Anthropic API wrapper with semaphore + retry."""

    def __init__(
        self,
        model: str = AUTOINTERP_MODEL,
        max_tokens: int = 512,
        max_concurrent: int = 5,
        max_retries: int = 3,
    ):
        import anthropic
        self._client = anthropic.AsyncAnthropic()
        self.model = model
        self.max_tokens = max_tokens
        self._sem = asyncio.Semaphore(max_concurrent)
        self.max_retries = max_retries
        self.n_calls = 0
        self.n_errors = 0

    async def call(self, system: str, user: str) -> str:
        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                async with self._sem:
                    resp = await self._client.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        system=system,
                        messages=[{"role": "user", "content": user}],
                    )
                    self.n_calls += 1
                    return resp.content[0].text
            except Exception as e:
                self.n_errors += 1
                if attempt == self.max_retries - 1:
                    log.error(f"  API failed after {self.max_retries} attempts: {e}")
                    return ""
                log.warning(f"  API error (attempt {attempt + 1}): {e} - retry in {delay:.1f}s")
                await asyncio.sleep(delay + random.uniform(0, 0.5))
                delay = min(delay * 2, 60.0)
        return ""


def make_backend(explain_model: str, device: str = "cuda"):
    """Create the right backend based on model name.

    If the model name looks like a HuggingFace model (contains '/'),
    use local inference. Otherwise, assume it's a Claude model ID.
    """
    if "/" in explain_model:
        return LocalGemmaBackend(model_name=explain_model, device=device)
    else:
        return ClaudeAPIBackend(model=explain_model)


# ══════════════════════════════════════════════════════════════════════════════
# 5. EXPLAINER + DETECTION SCORER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExplanationResult:
    explanation: str
    harmful: bool
    harmful_reason: str
    raw_response: str


def _parse_explanation(raw: str) -> ExplanationResult:
    explanation = ""
    harmful = False
    reason = ""

    for line in raw.splitlines():
        if line.startswith("[EXPLANATION]:"):
            explanation = line[len("[EXPLANATION]:"):].strip()
        elif line.startswith("[HARMFUL]:"):
            val = line[len("[HARMFUL]:"):].strip().upper()
            harmful = val == "YES"
        elif line.startswith("[HARMFUL_REASON]:"):
            reason = line[len("[HARMFUL_REASON]:"):].strip()

    return ExplanationResult(
        explanation=explanation or raw[:200],
        harmful=harmful,
        harmful_reason=reason,
        raw_response=raw,
    )


class AsyncExplainer:
    def __init__(
        self,
        claude,
        max_text_chars: int = 400,
        harm_eval: bool = True,
        code_prompt: bool = False,
    ):
        self.claude = claude
        self.max_text_chars = max_text_chars
        self.harm_eval = harm_eval
        self.code_prompt = code_prompt

    async def explain(
        self, texts: list[str], activations: list[float],
    ) -> ExplanationResult:
        lines = []
        for i, (txt, act) in enumerate(zip(texts, activations), 1):
            # For code, preserving newlines helps Claude read the structure;
            # for natural language, flattening is fine.
            if self.code_prompt:
                truncated = txt[:self.max_text_chars].rstrip()
            else:
                truncated = txt[:self.max_text_chars].replace("\n", " ").strip()
            lines.append(f"  {i}. (act={act:.3f})  {truncated}")
        example_str = "\n".join(lines)

        user_msg = EXPLAIN_USER_TEMPLATE.format(n=len(texts), examples=example_str)
        if self.code_prompt:
            system = EXPLAIN_SYSTEM_CODE
        elif self.harm_eval:
            system = EXPLAIN_SYSTEM
        else:
            system = EXPLAIN_SYSTEM_NO_HARM
        raw = await self.claude.call(system, user_msg)
        return _parse_explanation(raw)


class AsyncDetectionScorer:
    """Score explanation against match + random examples."""

    def __init__(self, claude, n_match: int = 10, n_nomatch: int = 10):
        self.claude = claude
        self.n_match = n_match
        self.n_nomatch = n_nomatch

    async def score(
        self, explanation: str, match_texts: list[str], random_texts: list[str],
    ) -> tuple[float, int, int]:
        """Returns (score, n_correct, n_total)."""
        labeled = (
            [(t, True) for t in match_texts[:self.n_match]]
            + [(t, False) for t in random_texts[:self.n_nomatch]]
        )
        random.shuffle(labeled)

        examples_str = "\n".join(
            f"  {i + 1}. {t[:300].replace(chr(10), ' ').strip()}"
            for i, (t, _) in enumerate(labeled)
        )
        system = DETECT_SYSTEM.format(n=len(labeled))
        user = DETECT_USER_TEMPLATE.format(
            explanation=explanation, examples=examples_str,
        )
        raw = await self.claude.call(system, user)

        # Parse response
        lines = [l.strip().upper() for l in raw.splitlines() if l.strip() in ("MATCH", "NO_MATCH")]
        true_order = ["MATCH" if is_match else "NO_MATCH" for _, is_match in labeled]
        n_total = len(labeled)

        if len(lines) != n_total:
            return 0.5, 0, n_total

        n_correct = sum(1 for pred, true in zip(lines, true_order) if pred == true)
        return round(n_correct / n_total, 4), n_correct, n_total


# ══════════════════════════════════════════════════════════════════════════════
# 6. RESULTS STORE
# ══════════════════════════════════════════════════════════════════════════════

class ResultsStore:
    """Persists one JSON per feature. Supports resume."""

    def __init__(self, results_dir: str, threshold: float = 0.8):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold

    def path_for(self, feat_idx: int) -> Path:
        return self.results_dir / f"feat_{feat_idx:06d}.json"

    def exists(self, feat_idx: int) -> bool:
        return self.path_for(feat_idx).exists()

    def save(self, rec: FeatureRecord) -> None:
        rec.interpretable = rec.detection_score >= self.threshold
        self.path_for(rec.feat_idx).write_text(json.dumps(asdict(rec), indent=2))

    def load(self, feat_idx: int) -> Optional[FeatureRecord]:
        p = self.path_for(feat_idx)
        if not p.exists():
            return None
        return FeatureRecord(**json.loads(p.read_text()))

    def summary(self) -> dict:
        records = list(self.results_dir.glob("feat_*.json"))
        total = len(records)
        interpretable = 0
        harmful = 0
        scores = []

        for p in records:
            d = json.loads(p.read_text())
            if d.get("interpretable"):
                interpretable += 1
            if d.get("harmful"):
                harmful += 1
            if d.get("detection_score", -1) >= 0:
                scores.append(d["detection_score"])

        return {
            "total_processed": total,
            "interpretable": interpretable,
            "harmful": harmful,
            "pct_interpretable": round(100 * interpretable / max(total, 1), 1),
            "mean_detection_score": round(float(np.mean(scores)), 4) if scores else 0.0,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 7. PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class AutoInterpPipeline:
    """
    Full autointerp pipeline for NLP temporal crosscoders.

    Steps:
      1. TopKFinder scans cached activations through trained model
      2. TextContext decodes windows via gemma-2-2b-it tokenizer
      3. AsyncExplainer + AsyncDetectionScorer validate interpretations
      4. ResultsStore persists per-feature JSON
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_type: str,
        layer_key: str,
        cache_dir: str,
        results_dir: str,
        top_k: int = AUTOINTERP_MAX_EXAMPLES,
        top_features: int = AUTOINTERP_TOP_FEATURES,
        feature_subset: list[int] | None = None,
        explain_model: str = AUTOINTERP_MODEL,
        sample_chains: int = 2000,
        scan_device: str = "cuda:1",
        explain_device: str = "cuda:0",
        score: bool = True,
        harm_eval: bool = True,
    ):
        self.model = model
        self.model_type = model_type
        self.layer_key = layer_key
        self.cache_dir = cache_dir
        self.top_k = top_k
        self.top_features = top_features
        self.feature_subset = feature_subset
        self.T = model.T
        self.scan_device = scan_device
        self.score = score

        self.backend = make_backend(explain_model, device=explain_device)
        self.explainer = AsyncExplainer(self.backend, harm_eval=harm_eval)
        self.scorer = AsyncDetectionScorer(self.backend, n_match=top_k, n_nomatch=top_k)
        self.store = ResultsStore(results_dir)
        self.sample_chains = sample_chains

    async def _process_feature(
        self,
        feat_idx: int,
        examples: list[ActivationExample],
        text_ctx: TextContext,
    ) -> None:
        if self.store.exists(feat_idx):
            return

        rec = FeatureRecord(
            feat_idx=feat_idx,
            model_type=self.model_type,
            layer=self.layer_key,
        )

        texts = [
            text_ctx.get_window_text(e.chain_idx, e.window_start, self.T)
            for e in examples
        ]
        activations = [e.activation for e in examples]

        rec.top_texts = texts
        rec.top_activations = activations

        try:
            # Explain
            expl = await self.explainer.explain(texts, activations)
            rec.explanation = expl.explanation
            rec.harmful = expl.harmful
            rec.harmful_reason = expl.harmful_reason

            # Detection score (optional)
            if rec.explanation and self.score:
                exclude_chains = {e.chain_idx for e in examples}
                random_texts = text_ctx.get_random_texts(self.top_k, exclude_chains)

                score, n_correct, n_total = await self.scorer.score(
                    rec.explanation, texts, random_texts,
                )
                rec.detection_score = score
                rec.detection_n_correct = n_correct
                rec.detection_n_total = n_total

        except Exception as e:
            rec.error = str(e)
            log.error(f"  Feature {feat_idx}: error - {e}")

        self.store.save(rec)

    def run(self) -> dict:
        log.info("=" * 60)
        log.info(f"  AutoInterp: {self.model_type} / {self.layer_key}")
        log.info("=" * 60)

        # Stage 1: Find top-K
        log.info("[Stage 1] Scanning for top-K activating windows...")
        finder = TopKFinder(
            self.model, self.model_type, self.layer_key,
            self.cache_dir, k=self.top_k,
            sample_chains=self.sample_chains,
            device=self.scan_device,
        )
        finder.run()

        # Select features to interpret
        if self.feature_subset is not None:
            # User-specified subset — validate they have activations
            top_feature_ids = [
                fi for fi in self.feature_subset if fi in finder.results
            ]
            missing = set(self.feature_subset) - set(top_feature_ids)
            if missing:
                log.warning(f"  {len(missing)} requested features had no activations: {sorted(missing)[:10]}...")
        else:
            # Auto-select top-N by total activation mass
            feature_mass = {
                idx: sum(e.activation for e in examples)
                for idx, examples in finder.results.items()
                if len(examples) >= 3  # skip near-dead
            }
            top_feature_ids = sorted(feature_mass, key=feature_mass.get, reverse=True)[
                :self.top_features
            ]
        log.info(f"  Selected {len(top_feature_ids)} features for interpretation.")

        # Stage 2: Load text context
        log.info("[Stage 2] Loading text context (gemma-2-2b-it tokenizer)...")
        text_ctx = TextContext(self.cache_dir, MODEL_NAME)

        # Stage 3: Explain + Score
        log.info("[Stage 3] Running explanation + detection scoring...")
        pending = [fi for fi in top_feature_ids if not self.store.exists(fi)]
        log.info(f"  {len(pending)} features to process ({len(top_feature_ids) - len(pending)} already done).")

        async def _run():
            for feat_idx in tqdm(pending, desc="Interpreting features"):
                examples = finder.results[feat_idx]
                await self._process_feature(feat_idx, examples, text_ctx)

                rec = self.store.load(feat_idx)
                if rec and rec.explanation:
                    det = f"det={rec.detection_score:.2f}" if rec.detection_score >= 0 else "det=N/A"
                    log.info(f"  f{feat_idx}: {rec.explanation} [{det}]")

        asyncio.run(_run())

        # Stage 4: Summary
        summary = self.store.summary()
        log.info(f"\n  Summary: {json.dumps(summary, indent=2)}")

        summary_path = Path(self.store.results_dir) / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        log.info(f"  Saved: {summary_path}")

        return summary


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def load_model(
    ckpt_path: str,
    model_type: str,
    subject_model: str,
    k: int,
    T: int,
    expansion_factor: int = 8,
) -> torch.nn.Module:
    """Load a trained SAE/crosscoder checkpoint.

    Subject model routes through the registry to resolve d_in;
    d_sae = d_in * expansion_factor.
    """
    cfg = get_model_config(subject_model)
    d_in = cfg.d_model
    d_sae = d_in * expansion_factor

    if model_type == "stacked_sae":
        model = FastStackedSAE(d_in=d_in, d_sae=d_sae, T=T, k=k)
    else:
        model = FastTemporalCrosscoder(d_in=d_in, d_sae=d_sae, T=T, k=k)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="AutoInterp for NLP temporal crosscoders")
    parser.add_argument("--phase", type=str, default="full",
                        choices=["scan", "explain", "full"],
                        help="scan = run TopKFinder + save feat_*.json with "
                             "top_texts but no Claude calls (GPU + no network). "
                             "explain = read existing feat_*.json, call Claude, "
                             "update explanation field (network + ~no CPU). "
                             "full = both (old default behavior).")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a trained .pt checkpoint (required for "
                             "scan and full; unused for explain)")
    parser.add_argument("--model", type=str, default="crosscoder",
                        choices=["topk_sae", "stacked_sae", "crosscoder", "txcdr"])
    parser.add_argument("--subject-model", type=str, default=None,
                        choices=list_models(),
                        help="Subject LM the checkpoint was trained on "
                             "(required for scan and full)")
    parser.add_argument("--cached-dataset", type=str, default=None,
                        help="dataset subdir (required for scan and full)")
    parser.add_argument("--layer-key", type=str, default=None,
                        help="Which <key>.npy to scan (required for scan and full)")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--T", type=int, default=5)
    parser.add_argument("--expansion-factor", type=int, default=8)
    parser.add_argument("--label", type=str, required=True,
                        help="Output subdir name under <output-dir>/autointerp/")
    parser.add_argument("--top-features", type=int, default=AUTOINTERP_TOP_FEATURES)
    parser.add_argument("--features", type=int, nargs="+", default=None,
                        help="Specific feature indices to interpret (default: auto-select top-N)")
    parser.add_argument("--top-k", type=int, default=AUTOINTERP_MAX_EXAMPLES)
    parser.add_argument("--sample-chains", type=int, default=2000)
    parser.add_argument("--no-score", action="store_true",
                        help="Skip detection scoring (only generate explanations)")
    parser.add_argument("--no-harm", action="store_true",
                        help="Skip harmful-content evaluation")
    parser.add_argument("--code-prompt", action="store_true",
                        help="Use the code-specific explanation prompt "
                             "(syntax, scope, control flow). Recommended "
                             "for Stack-Python / coding activation caches. "
                             "Implies --no-harm.")
    parser.add_argument("--explain-model", type=str, default=AUTOINTERP_MODEL,
                        help="Claude model id or HF path (HF path => local)")
    parser.add_argument("--scan-device", type=str, default="auto",
                        help="'auto' picks cuda if available else cpu")
    parser.add_argument("--explain-device", type=str, default="auto",
                        help="Only used for local HF explainers")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Parent dir (default: <VIZ_DIR>). Results go to "
                             "<output-dir>/autointerp/<label>/")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    arch = "crosscoder" if args.model == "txcdr" else args.model

    # Resolve devices: "auto" => cuda if available else cpu.
    def _auto(dev: str) -> str:
        if dev != "auto":
            return dev
        return "cuda" if torch.cuda.is_available() else "cpu"

    scan_device = _auto(args.scan_device)
    explain_device = _auto(args.explain_device)

    output_root = args.output_dir or VIZ_DIR
    results_dir = os.path.join(output_root, "autointerp", args.label)
    log.info(f"  phase: {args.phase}")
    log.info(f"  results -> {results_dir}")

    # ── Explain phase: no checkpoint, no model, no cache. Just read existing
    #    feat_*.json files and call Claude to fill in explanations. Safe on
    #    login nodes since it uses ~zero CPU.
    #    --code-prompt implies --no-harm (syntactic patterns aren't
    #    harm-relevant in the same sense).
    harm_eval = not args.no_harm and not args.code_prompt
    if args.phase == "explain":
        run_explain_phase(
            results_dir=results_dir,
            explain_model=args.explain_model,
            harm_eval=harm_eval,
            code_prompt=args.code_prompt,
        )
        return

    # ── Scan or full: need checkpoint, cache_dir, and a loaded model.
    missing = [
        flag for flag, val in [
            ("--checkpoint", args.checkpoint),
            ("--subject-model", args.subject_model),
            ("--cached-dataset", args.cached_dataset),
            ("--layer-key", args.layer_key),
        ] if val is None
    ]
    if missing:
        log.error(f"Phase '{args.phase}' requires: {', '.join(missing)}")
        sys.exit(1)

    cache_dir = cache_dir_for(args.subject_model, args.cached_dataset)
    if not os.path.exists(args.checkpoint):
        log.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    if not os.path.exists(cache_dir):
        log.error(f"Cache dir not found: {cache_dir}")
        sys.exit(1)

    log.info(f"  scan_device={scan_device}  explain_device={explain_device}")

    model = load_model(
        ckpt_path=args.checkpoint,
        model_type=arch,
        subject_model=args.subject_model,
        k=args.k, T=args.T,
        expansion_factor=args.expansion_factor,
    )

    run_scan_phase(
        model=model,
        model_type=arch,
        layer_key=args.layer_key,
        cache_dir=cache_dir,
        results_dir=results_dir,
        top_k=args.top_k,
        top_features=args.top_features,
        feature_subset=args.features,
        sample_chains=args.sample_chains,
        scan_device=scan_device,
        subject_model=args.subject_model,
    )

    if args.phase == "full":
        run_explain_phase(
            results_dir=results_dir,
            explain_model=args.explain_model,
            harm_eval=harm_eval,
            code_prompt=args.code_prompt,
        )

    del model


# ══════════════════════════════════════════════════════════════════════════════
# PHASE RUNNERS — split so scan can run inside sbatch (GPU, no network)
# and explain can run on the login node (network, ~zero CPU).
# ══════════════════════════════════════════════════════════════════════════════

def run_scan_phase(
    model: torch.nn.Module,
    model_type: str,
    layer_key: str,
    cache_dir: str,
    results_dir: str,
    top_k: int,
    top_features: int,
    feature_subset: list[int] | None,
    sample_chains: int,
    scan_device: str,
    subject_model: str,
) -> None:
    """Run TopKFinder + save feat_*.json with top_texts/top_activations but
    an empty explanation field. No Claude backend created — safe to run
    inside an sbatch job on a compute node with no outbound network.
    """
    log.info("=" * 60)
    log.info(f"  SCAN phase: {model_type} / {layer_key}")
    log.info("=" * 60)

    log.info("[Stage 1] Scanning for top-K activating windows...")
    finder = TopKFinder(
        model, model_type, layer_key, cache_dir,
        k=top_k, sample_chains=sample_chains, device=scan_device,
    )
    finder.run()

    if feature_subset is not None:
        top_feature_ids = [fi for fi in feature_subset if fi in finder.results]
    else:
        feature_mass = {
            idx: sum(e.activation for e in examples)
            for idx, examples in finder.results.items()
            if len(examples) >= 3
        }
        top_feature_ids = sorted(
            feature_mass, key=feature_mass.get, reverse=True,
        )[:top_features]
    log.info(f"  Selected {len(top_feature_ids)} features for interpretation.")

    log.info("[Stage 2] Loading text context...")
    text_ctx = TextContext(cache_dir, subject_model)
    store = ResultsStore(results_dir)
    T = model.T

    log.info(f"[Stage 3] Writing feat_*.json (no Claude calls)...")
    for feat_idx in tqdm(top_feature_ids, desc="Scan"):
        if store.exists(feat_idx):
            existing = store.load(feat_idx)
            if existing and existing.top_texts:
                continue  # already scanned, no overwrite
        examples = finder.results[feat_idx]
        rec = FeatureRecord(
            feat_idx=feat_idx,
            model_type=model_type,
            layer=layer_key,
        )
        rec.top_texts = [
            text_ctx.get_window_text(e.chain_idx, e.window_start, T)
            for e in examples
        ]
        rec.top_activations = [e.activation for e in examples]
        # explanation stays empty — filled in by the explain phase.
        store.save(rec)

    summary = store.summary()
    log.info(f"\n  scan summary: {json.dumps(summary, indent=2)}")


def run_explain_phase(
    results_dir: str,
    explain_model: str,
    harm_eval: bool = False,
    code_prompt: bool = False,
) -> None:
    """Read feat_*.json files written by run_scan_phase and fill in the
    explanation field via Claude. Network required, ~zero CPU, safe on a
    login node inside the CPU-time cap.
    """
    log.info("=" * 60)
    log.info(f"  EXPLAIN phase: {results_dir}")
    log.info("=" * 60)

    store = ResultsStore(results_dir)
    if not store.results_dir.is_dir():
        log.error(f"No results dir — run scan phase first: {results_dir}")
        sys.exit(1)

    pending: list[FeatureRecord] = []
    for p in sorted(store.results_dir.glob("feat_*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        rec = FeatureRecord(**{k: v for k, v in d.items() if k in FeatureRecord.__dataclass_fields__})
        if rec.top_texts and not rec.explanation:
            pending.append(rec)

    if not pending:
        log.info("  nothing to explain (all features already have explanations)")
        return
    log.info(f"  {len(pending)} features pending explanation")
    log.info(f"  explain_model={explain_model}  code_prompt={code_prompt}")

    backend = make_backend(explain_model, device="cpu")
    explainer = AsyncExplainer(backend, harm_eval=harm_eval, code_prompt=code_prompt)

    async def _run():
        for rec in tqdm(pending, desc="Explain"):
            try:
                result = await explainer.explain(rec.top_texts, rec.top_activations)
                rec.explanation = result.explanation
                rec.harmful = result.harmful
                rec.harmful_reason = result.harmful_reason
            except Exception as e:
                rec.error = str(e)
                log.error(f"  feat {rec.feat_idx}: {e}")
            store.save(rec)
            if rec.explanation:
                log.info(f"  f{rec.feat_idx}: {rec.explanation}")

    asyncio.run(_run())
    summary = store.summary()
    log.info(f"\n  explain summary: {json.dumps(summary, indent=2)}")
    summary_path = Path(store.results_dir) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
