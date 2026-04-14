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

from config import (
    D_SAE, LAYER_SPECS, SEED, MODEL_NAME,
    CACHE_DIR, CHECKPOINT_DIR, LOG_DIR, VIZ_DIR,
    AUTOINTERP_MODEL, AUTOINTERP_MAX_EXAMPLES, AUTOINTERP_TOP_FEATURES,
    run_name,
)
from fast_models import FastStackedSAE, FastTemporalCrosscoder

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
    """Decode token windows into readable text using the model tokenizer."""

    def __init__(self, cache_dir: str, model_name: str = MODEL_NAME):
        from transformers import AutoTokenizer

        token_path = os.path.join(cache_dir, "token_ids.npy")
        if not os.path.exists(token_path):
            raise FileNotFoundError(
                f"Token IDs not found at {token_path}. Run cache_activations.py first."
            )

        self.token_ids = np.load(token_path, mmap_mode="r")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        log.info(f"  TextContext: loaded token_ids {self.token_ids.shape}, tokenizer={model_name}")

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

        pre = self.tokenizer.decode(pre_tokens, skip_special_tokens=True)
        win = self.tokenizer.decode(win_tokens, skip_special_tokens=True)
        post = self.tokenizer.decode(post_tokens, skip_special_tokens=True)

        return f"{pre}>>>{win}<<<{post}"

    def get_full_text(self, chain_idx: int) -> str:
        """Get the full sequence text for a chain."""
        tokens = self.token_ids[chain_idx]
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
    """Run gemma-2-2b-it locally for autointerp."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
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
        chat_out = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True,
        )
        # Newer transformers may return a BatchEncoding instead of a tensor
        if isinstance(chat_out, torch.Tensor):
            input_ids = chat_out.to(self.model.device)
            gen_kwargs = {"input_ids": input_ids}
        else:
            gen_kwargs = {k: v.to(self.model.device) for k, v in chat_out.items()}
            input_ids = gen_kwargs["input_ids"]
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **gen_kwargs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        # Decode only the generated tokens
        new_tokens = output_ids[0, prompt_len:]
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
    def __init__(self, claude, max_text_chars: int = 400, harm_eval: bool = True):
        self.claude = claude
        self.max_text_chars = max_text_chars
        self.harm_eval = harm_eval

    async def explain(
        self, texts: list[str], activations: list[float],
    ) -> ExplanationResult:
        lines = []
        for i, (txt, act) in enumerate(zip(texts, activations), 1):
            truncated = txt[:self.max_text_chars].replace("\n", " ").strip()
            lines.append(f"  {i}. (act={act:.3f})  {truncated}")
        example_str = "\n".join(lines)

        user_msg = EXPLAIN_USER_TEMPLATE.format(n=len(texts), examples=example_str)
        system = EXPLAIN_SYSTEM if self.harm_eval else EXPLAIN_SYSTEM_NO_HARM
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

def find_best_checkpoints(log_dir: str, checkpoint_dir: str) -> dict[str, dict]:
    """Find best checkpoint per architecture by lowest final loss."""
    summary_path = os.path.join(log_dir, "sweep_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"No sweep summary at {summary_path}. Run sweep.py first.")

    with open(summary_path) as f:
        summary = json.load(f)

    best: dict[str, dict] = {}
    for row in summary:
        model_type = row["model"]
        loss = row["final_loss"]
        ckpt_name = f"{model_type}__{row['layer']}__k{row['k']}__T{row['T']}.pt"
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)

        if not os.path.exists(ckpt_path):
            continue

        if model_type not in best or loss < best[model_type]["loss"]:
            best[model_type] = {
                "path": ckpt_path,
                "model": model_type,
                "layer": row["layer"],
                "k": row["k"],
                "T": row["T"],
                "loss": loss,
            }

    return best


def load_model(model_type: str, layer: str, k: int, T: int, ckpt_path: str) -> torch.nn.Module:
    d_act = LAYER_SPECS[layer]["d_act"]
    if model_type == "stacked_sae":
        model = FastStackedSAE(d_in=d_act, d_sae=D_SAE, T=T, k=k)
    else:
        model = FastTemporalCrosscoder(d_in=d_act, d_sae=D_SAE, T=T, k=k)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="AutoInterp for NLP temporal crosscoders")
    parser.add_argument("--model", type=str, default=None,
                        choices=["stacked_sae", "txcdr"])
    parser.add_argument("--layer", type=str, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--top-features", type=int, default=AUTOINTERP_TOP_FEATURES)
    parser.add_argument("--features", type=int, nargs="+", default=None,
                        help="Specific feature indices to interpret (default: auto-select top-N)")
    parser.add_argument("--top-k", type=int, default=AUTOINTERP_MAX_EXAMPLES)
    parser.add_argument("--sample-chains", type=int, default=2000)
    parser.add_argument("--no-score", action="store_true",
                        help="Skip detection scoring (only generate explanations)")
    parser.add_argument("--no-harm", action="store_true",
                        help="Skip harmful-content evaluation (saves response tokens)")
    parser.add_argument("--explain-model", type=str, default=AUTOINTERP_MODEL)
    parser.add_argument("--scan-device", type=str, default="cuda:1",
                        help="Device for SAE/TXCDR forward passes (default: cuda:1)")
    parser.add_argument("--explain-device", type=str, default="cuda:0",
                        help="Device for the local LLM explainer (default: cuda:0)")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Determine which models to interpret
    if args.checkpoint:
        if not all([args.model, args.layer, args.k, args.T]):
            raise ValueError(
                "With --checkpoint, also provide --model, --layer, --k, --T"
            )
        models_to_interp = {
            args.model: {
                "path": args.checkpoint,
                "model": args.model,
                "layer": args.layer,
                "k": args.k,
                "T": args.T,
            }
        }
    elif args.model and args.layer and args.k and args.T:
        ckpt_path = os.path.join(
            CHECKPOINT_DIR, f"{run_name(args.model, args.layer, args.k, args.T)}.pt"
        )
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        models_to_interp = {
            args.model: {
                "path": ckpt_path,
                "model": args.model,
                "layer": args.layer,
                "k": args.k,
                "T": args.T,
            }
        }
    else:
        log.info("Finding best checkpoints by lowest loss...")
        models_to_interp = find_best_checkpoints(LOG_DIR, CHECKPOINT_DIR)
        for key, info in models_to_interp.items():
            log.info(f"  Best {key}: {info['layer']} k={info['k']} T={info['T']} loss={info.get('loss', '?')}")

    if not models_to_interp:
        log.error("No checkpoints found. Run sweep.py first.")
        sys.exit(1)

    for model_type, info in models_to_interp.items():
        label = f"{info['model']}_{info['layer']}_k{info['k']}_T{info['T']}"
        output_dir = args.output_dir or os.path.join(VIZ_DIR, "autointerp", label)

        model = load_model(
            info["model"], info["layer"], info["k"], info["T"], info["path"],
        )

        pipeline = AutoInterpPipeline(
            model=model,
            model_type=info["model"],
            layer_key=info["layer"],
            cache_dir=CACHE_DIR,
            results_dir=output_dir,
            top_k=args.top_k,
            top_features=args.top_features,
            feature_subset=args.features,
            explain_model=args.explain_model,
            sample_chains=args.sample_chains,
            scan_device=args.scan_device,
            explain_device=args.explain_device,
            score=not args.no_score,
            harm_eval=not args.no_harm,
        )
        pipeline.run()

        del model


if __name__ == "__main__":
    main()
