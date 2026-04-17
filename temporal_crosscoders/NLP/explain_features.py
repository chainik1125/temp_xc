#!/usr/bin/env python3
"""
explain_features.py — Run Claude on the top-K activating text windows of
each feature to produce a 1-sentence label. No detection scoring — this
is the "name the pattern" pass for semantic cross-arch comparison.

Reads: scan_*.json (from scan_features.py)
Writes: labels_*.json (one label per feature)

Usage:
    ANTHROPIC_API_KEY=$(cat /workspace/.anthropic-key) \\
    python -m temporal_crosscoders.NLP.explain_features \\
        --scan results/nlp_sweep/gemma/scans/scan__stacked_sae__resid_L25__k50.json \\
        --out  results/nlp_sweep/gemma/scans/labels__stacked_sae__resid_L25__k50.json \\
        --top-features 50 --model claude-haiku-4-5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os

import anthropic

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("explain")


SYSTEM_PROMPT = """You label features learned by sparse autoencoders on
language model activations. Given the top activating text windows of one
feature, identify the single pattern they share in ONE short sentence
(≤15 words). Focus on the tokens between >>> and <<< — those are where
the feature fires. Be specific and concrete. If the pattern is weak or
the windows look unrelated, say "unclear".

Reply with exactly one line, no preamble, no markdown."""


def build_user_message(examples: list[dict]) -> str:
    lines = ["Top activating windows:"]
    for i, ex in enumerate(examples, 1):
        txt = ex["text"].replace("\n", " ").strip()
        if len(txt) > 250:
            txt = txt[:250] + "..."
        lines.append(f"{i}. [act={ex['activation']:.3g}] {txt}")
    lines.append("")
    lines.append("Pattern (one sentence):")
    return "\n".join(lines)


async def explain_one(
    client: anthropic.AsyncAnthropic,
    model: str,
    feat_idx: int,
    examples: list[dict],
    semaphore: asyncio.Semaphore,
) -> tuple[int, str]:
    async with semaphore:
        user_msg = build_user_message(examples)
        try:
            resp = await client.messages.create(
                model=model,
                max_tokens=60,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = resp.content[0].text.strip()
            return feat_idx, text
        except Exception as e:
            return feat_idx, f"ERROR: {type(e).__name__}: {e}"


async def run_explanations(
    client: anthropic.AsyncAnthropic,
    model: str,
    features: list[tuple[int, list[dict]]],
    concurrency: int = 8,
) -> dict[int, str]:
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [explain_one(client, model, fi, exs, semaphore)
             for fi, exs in features]
    labels: dict[int, str] = {}
    for coro in asyncio.as_completed(tasks):
        fi, text = await coro
        labels[fi] = text
        log.info(f"  feat {fi:>5}: {text[:100]}")
    return labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scan", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--top-features", type=int, default=50)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--model", default="claude-haiku-4-5-20251001")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--api-key-file", default="/workspace/.anthropic-key")
    args = p.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        with open(args.api_key_file) as f:
            api_key = f.read().strip()
    # max_retries=8 with default backoff handles rate-limit 429s.
    client = anthropic.AsyncAnthropic(api_key=api_key, max_retries=8)

    with open(args.scan) as f:
        scan = json.load(f)

    # Resume: reuse existing successful labels so we only re-request
    # errored ones on retry runs.
    existing: dict[int, str] = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            try:
                existing = {int(k): v for k, v in json.load(f).get("labels", {}).items()
                            if not v.startswith("ERROR")}
            except Exception:
                existing = {}
        log.info(f"reusing {len(existing)} non-error labels from {args.out}")

    arch = scan["arch"]
    features = []
    for fi_str in list(scan["features"].keys())[:args.top_features]:
        fi = int(fi_str)
        if fi in existing:
            continue
        rec = scan["features"][fi_str]
        examples = rec["examples"][:args.top_k]
        features.append((fi, examples))
    log.info(f"[{arch}] explaining {len(features)} new features with {args.model}")

    new_labels = asyncio.run(run_explanations(
        client, args.model, features, concurrency=args.concurrency,
    ))
    labels = {**existing, **new_labels}

    out = {
        "arch": arch,
        "layer_key": scan.get("layer_key"),
        "k": scan.get("k"),
        "T": scan.get("T"),
        "explainer": args.model,
        "n_features_labeled": len(labels),
        "top_features": args.top_features,
        "labels": {str(fi): labels[fi] for fi in sorted(labels)},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    log.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
