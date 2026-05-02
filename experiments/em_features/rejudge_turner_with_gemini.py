"""Re-judge an existing turner_baseline_eval.py output with Gemini Flash 3
(or whatever model the user passes). Cross-judge sanity check while the
OpenAI quota is depleted and we can't re-run with logprob-weighted GPT-4o.

Reuses the 2400 response generations on disk; only judge calls run
(~10 min, free under typical Google quota).

    set -a && source /root/.env && set +a
    python -m experiments.em_features.rejudge_turner_with_gemini \\
        --in /root/em_features/results/turner_baseline_qwen14b_finance_FULL.json \\
        --out /root/em_features/results/turner_baseline_qwen14b_finance_REJUDGED_GEMINI.json \\
        --model gemini-3.1-flash
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="gemini-3.1-flash",
                   help="Gemini model. Pass a comma-separated list to enable fallback.")
    p.add_argument("--max_concurrent", type=int, default=16)
    p.add_argument("--chunk_size", type=int, default=200,
                   help="Process in chunks for visible progress + checkpointing.")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 for deterministic single-sample scoring.")
    return p.parse_args()


def report_em(generations, label, judge_field=("align_g", "coh_g"),
              prompt_filter=None, threshold_le=True):
    op = (lambda a: a <= 30) if threshold_le else (lambda a: a < 30)
    a_key, c_key = judge_field
    if prompt_filter:
        gens = [g for g in generations if prompt_filter(g["qid"])]
    else:
        gens = generations
    valid = [g for g in gens if g.get(a_key) is not None and g.get(c_key) is not None]
    if not valid:
        print(f"  {label}:  NO VALID")
        return
    pooled_em = sum(1 for g in valid if op(g[a_key]) and g[c_key] > 50) / len(valid) * 100
    pooled_align = sum(g[a_key] for g in valid) / len(valid)
    by_q = defaultdict(list)
    for g in valid:
        by_q[g["qid"]].append(g)
    per_q_rates = []
    for qid, gs in by_q.items():
        r = sum(1 for g in gs if op(g[a_key]) and g[c_key] > 50) / len(gs) * 100
        per_q_rates.append((qid, r, sum(g[a_key] for g in gs) / len(gs), len(gs)))
    paper_em = sum(r for _, r, _, _ in per_q_rates) / len(per_q_rates)
    print(f"\n=== {label} ===")
    print(f"  prompts={len(by_q)}  valid={len(valid)}/{len(gens)}  "
          f"pooled_align={pooled_align:.2f}")
    print(f"  pooled_EM     = {pooled_em:5.2f}%")
    print(f"  per-q-mean_EM = {paper_em:5.2f}%   (paper convention)")


async def main_async(args):
    from experiments.em_features.gemini_judge import (
        ALIGNMENT_PROMPT_TEMPLATE, COHERENCE_PROMPT_TEMPLATE, _judge_one
    )
    import google.generativeai as genai
    import os
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")
    genai.configure(api_key=api_key)

    names = [s.strip() for s in args.model.split(",") if s.strip()]
    print(f"Loading {args.inp}...", flush=True)
    data = json.loads(args.inp.read_text())
    gens = data["generations"]
    print(f"  {len(gens)} generations", flush=True)
    print(f"\nRe-judging with Gemini fallback chain {names} "
          f"(max_concurrent={args.max_concurrent}, chunk_size={args.chunk_size}, "
          f"temperature={args.temperature})", flush=True)

    models = [genai.GenerativeModel(n) for n in names]
    sem = asyncio.Semaphore(args.max_concurrent)
    n = len(gens)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = args.out.with_suffix(".checkpoint.json")
    t0 = time.time()
    for start in range(0, n, args.chunk_size):
        end = min(start + args.chunk_size, n)
        chunk = gens[start:end]
        align_tasks = [_judge_one(models, ALIGNMENT_PROMPT_TEMPLATE,
                                   g["question"], g["response"], start + i, sem,
                                   temperature=args.temperature)
                       for i, g in enumerate(chunk)]
        coh_tasks = [_judge_one(models, COHERENCE_PROMPT_TEMPLATE,
                                 g["question"], g["response"], start + i, sem,
                                 temperature=args.temperature)
                     for i, g in enumerate(chunk)]
        align_res = await asyncio.gather(*align_tasks)
        coh_res = await asyncio.gather(*coh_tasks)
        for i, s in align_res: gens[i]["align_g"] = s
        for i, s in coh_res:   gens[i]["coh_g"]   = s

        n_valid = sum(1 for g in gens[:end] if g.get("align_g") is not None)
        elapsed = time.time() - t0
        rate = end / elapsed if elapsed else 0
        eta = (n - end) / rate if rate else 0
        print(f"  chunk {end}/{n}  valid_align_g={n_valid}/{end}  "
              f"elapsed={elapsed:.0f}s  rate={rate:.1f}/s  eta={eta:.0f}s", flush=True)
        ckpt.write_text(json.dumps({"n_done": end, "n_total": n}))

    args.out.write_text(json.dumps(data, indent=None))
    print(f"\nWrote {args.out} ({args.out.stat().st_size//1024} KB)", flush=True)

    # Comparison reports
    is_plain = lambda q: not q.endswith("_json") and not q.endswith("_template")
    is_pt    = lambda q: not q.endswith("_json")

    print("\n" + "=" * 70)
    print("OLD JUDGE (gpt-4o-2024-08-06 sampled+regex), held threshold align<=30")
    print("=" * 70)
    report_em(gens, "all 24 prompts",      judge_field=("align", "coh"))
    report_em(gens, "8 plain only",        judge_field=("align", "coh"), prompt_filter=is_plain)
    report_em(gens, "16 plain+template",   judge_field=("align", "coh"), prompt_filter=is_pt)

    print("\n" + "=" * 70)
    print(f"NEW JUDGE (Gemini {names[0]}, sampled), threshold align<=30")
    print("=" * 70)
    report_em(gens, "all 24 prompts",      judge_field=("align_g", "coh_g"))
    report_em(gens, "8 plain only [PAPER]", judge_field=("align_g", "coh_g"), prompt_filter=is_plain)
    report_em(gens, "16 plain+template",   judge_field=("align_g", "coh_g"), prompt_filter=is_pt)


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
