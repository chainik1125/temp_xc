"""Priority 2b: top-N sweep — extend the autointerp ranking beyond
N=32 to N ∈ {32, 64, 128, 256} for 3 target archs on concat_random.

Tests whether TXC's qualitative count rises substantially at larger N,
which would suggest the original "top-32" cut was too narrow.

For each (arch, seed=42, concat=random) cell:
    1. Load z_cache, rank by per-token variance.
    2. Take top-256 feature indices.
    3. For each feature index NOT already labelled in the existing
       labels JSON (or by this script on a prior run): gather contexts,
       Haiku label, judge.
    4. Write per-N cumulative SEMANTIC count to
       `{arch}__seed42__concatrandom__top256.json`.

Safe to re-run — reuses prior labels.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LABELS_DIR = REPO / "experiments/phase6_qualitative_latents/results/autointerp"
Z_DIR = REPO / "experiments/phase6_qualitative_latents/z_cache"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archs", type=str, nargs="+",
                   default=["agentic_txc_10_bare", "tsae_paper",
                            "agentic_txc_02_batchtopk"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--concat", type=str, default="random")
    p.add_argument("--N", type=int, default=256,
                   help="max top-N to label")
    p.add_argument("--model", type=str, default="claude-haiku-4-5")
    args = p.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        env_path = REPO / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not found in env or .env")

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "google/gemma-2-2b-it",
        token=open("/workspace/hf_cache/token").read().strip(),
    )

    import sys
    sys.path.insert(0, str(REPO / "experiments/phase6_qualitative_latents"))
    from run_autointerp import (  # noqa: E402
        _load_concat, _gather_contexts, _claude_call, _judge_label,
        LABEL_PROMPT,
    )

    for arch in args.archs:
        seed = args.seed
        concat = args.concat

        # Load existing labels (var-top-32 + pdvar-only) — reuse them.
        reuse_path = LABELS_DIR / f"{arch}__seed{seed}__concat{concat}__labels.json"
        reused_by_idx = {}
        if reuse_path.exists():
            d = json.loads(reuse_path.read_text())
            for f in d.get("features", []):
                reused_by_idx[f["feature_idx"]] = f
            for f in d.get("pdvar_only_features", []):
                reused_by_idx.setdefault(f["feature_idx"], f)

        # Load or initialise the top256 cache.
        out_path = LABELS_DIR / f"{arch}__seed{seed}__concat{concat}__top256.json"
        if out_path.exists():
            top256 = json.loads(out_path.read_text())
            for f in top256.get("features", []):
                reused_by_idx.setdefault(f["feature_idx"], f)
        else:
            top256 = {"arch": arch, "seed": seed, "concat": concat,
                      "N": args.N, "features": []}

        # Load z for ranking.
        try:
            ids, z, _passages = _load_concat(concat, arch, seed)
        except FileNotFoundError:
            print(f"[skip] {arch} seed={seed} {concat}: no z cache")
            continue

        # Rank all features by per-token variance, take top-N.
        var = z.var(axis=0)
        top_N = np.argsort(-var)[:args.N]
        print(f"[run] {arch} {concat} seed={seed} top-{args.N}: "
              f"need to label {sum(1 for fi in top_N if int(fi) not in reused_by_idx)} new features "
              f"(reusing {sum(1 for fi in top_N if int(fi) in reused_by_idx)})")

        # Label missing.
        new_labels = []
        for fi in top_N:
            fi = int(fi)
            if fi in reused_by_idx:
                continue
            z_col = z[:, fi]
            ctxs = _gather_contexts(ids, tok, z_col)
            if not ctxs:
                label = "(dead feature, no activations)"
                judge = {"judge_1": "UNKNOWN", "judge_2": "UNKNOWN",
                         "agree": True, "verdict": "UNKNOWN"}
            else:
                excerpts = "\n\n".join(
                    f"- (strength={c['strength']:.2f}) {c['text']}"
                    for c in ctxs
                )
                prompt = LABEL_PROMPT.format(n=len(ctxs), f=fi,
                                             excerpts=excerpts)
                try:
                    label = _claude_call(client, args.model, prompt)
                except Exception as e:
                    label = f"(claude error: {e.__class__.__name__})"
                    judge = {"judge_1": "UNKNOWN", "judge_2": "UNKNOWN",
                             "agree": True, "verdict": "UNKNOWN"}
                else:
                    try:
                        judge = _judge_label(client, args.model, label)
                    except Exception as e:
                        judge = {"judge_1": "UNKNOWN", "judge_2": "UNKNOWN",
                                 "agree": True, "verdict": "UNKNOWN",
                                 "error": str(e)[:80]}
            feat = {
                "feature_idx": fi,
                "label": label,
                "variance": float(z[:, fi].var()),
                "n_contexts": len(ctxs),
                "contexts": ctxs[:3],
                "judge": judge,
            }
            reused_by_idx[fi] = feat
            new_labels.append(feat)

        # Consolidate all labels for top-N ranking, and save.
        top256["features"] = [
            reused_by_idx[int(fi)] for fi in top_N
        ]

        # Cumulative SEMANTIC count at N ∈ {32, 64, 128, 256}.
        cum_counts = {}
        for cutoff in (32, 64, 128, 256):
            feats = [reused_by_idx[int(fi)] for fi in top_N[:cutoff]]
            sem = sum(1 for f in feats if f["judge"]["verdict"] == "SEMANTIC")
            cum_counts[cutoff] = sem
        top256["cumulative_semantic_counts"] = cum_counts

        out_path.write_text(json.dumps(top256, indent=2))
        print(f"[done] {arch} {concat} seed={seed}: "
              f"cum sem @ N=32/64/128/256 = "
              f"{cum_counts[32]}/{cum_counts[64]}/{cum_counts[128]}/{cum_counts[256]} "
              f"(labelled {len(new_labels)} new features this run)")


if __name__ == "__main__":
    main()
