"""Per-task breakdown — where does the TXC champion win vs topk_sae?

Companion to `2026-04-29-leaderboard-2seed.md`. The 2-seed mean delta is
~0.005 AUC; this script asks WHERE that delta lives. If TXC wins
concentrate on "knowledge"-style tasks (bias_in_bios, github_code,
ag_news scitech) and topk_sae wins on "discourse"-style tasks
(amazon_reviews sentiment, europarl), that aligns with Y's per-concept
structural finding on the 30-concept steering benchmark.

Run from repo root:
    .venv/bin/python -m experiments.phase7_unification.analyze_per_task_breakdown
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

os.environ.setdefault("TQDM_DISABLE", "1")

from experiments.phase7_unification._paths import OUT_DIR


PROBING_PATH = OUT_DIR / "probing_results.jsonl"

# Coarse classification of the 36 SAEBench tasks
TASK_CATEGORY = {
    # bias_in_bios = predict profession from bio (knowledge / world-fact)
    **{f"bias_in_bios_set{i}_prof{p}": "knowledge_profession"
       for i in (1, 2, 3) for p in range(1, 30)},
    # ag_news = topic classification
    "ag_news_business": "topic", "ag_news_scitech": "topic",
    "ag_news_sports": "topic", "ag_news_world": "topic",
    # amazon = sentiment / category
    "amazon_reviews_cat0": "topic_amazon", "amazon_reviews_cat1": "topic_amazon",
    "amazon_reviews_cat2": "topic_amazon", "amazon_reviews_cat3": "topic_amazon",
    "amazon_reviews_cat5": "topic_amazon",
    "amazon_reviews_sentiment_5star": "sentiment",
    # europarl = language ID
    "europarl_de": "language_id", "europarl_es": "language_id",
    "europarl_fr": "language_id", "europarl_it": "language_id",
    "europarl_nl": "language_id",
    # github code = language ID for code (knowledge-flavored)
    "github_code_go": "code_language", "github_code_java": "code_language",
    "github_code_javascript": "code_language", "github_code_python": "code_language",
    # cross-token (coreference / completion)
    "winogrande_correct_completion": "coreference",
    "wsc_coreference": "coreference",
}


def load_2seed():
    """Returns dict[(arch, k_feat, task)] -> mean across seeds {1, 42}."""
    by_seed = defaultdict(dict)
    with PROBING_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            if r.get("S") != 32 or r.get("seed") not in (1, 42):
                continue
            if r.get("k_feat") not in (5, 20): continue
            if "skipped" in r: continue
            key = (r["arch_id"], r["k_feat"], r["task_name"])
            by_seed[key][r["seed"]] = r.get("test_auc_flip", r["test_auc"])
    out = {}
    for key, smap in by_seed.items():
        if len(smap) >= 1:
            out[key] = sum(smap.values()) / len(smap)
    return out


def main():
    d = load_2seed()
    # Compare top TXC vs topk_sae per task
    txc_champ_k5 = "phase57_partB_h8_bare_multidistance_t8"
    txc_champ_k20 = "txc_bare_antidead_t5"
    saes = ["topk_sae", "tsae_paper_k500"]

    for kf, champ in [(5, txc_champ_k5), (20, txc_champ_k20)]:
        print()
        print("=" * 100)
        print(f"PER-TASK BREAKDOWN at k_feat={kf}: {champ} vs SAE baselines")
        print("=" * 100)
        # Which tasks does TXC win on?
        deltas = {}
        for task in {t for (a, k, t) in d if a == champ and k == kf}:
            txc_auc = d.get((champ, kf, task))
            if txc_auc is None: continue
            for sae in saes:
                sae_auc = d.get((sae, kf, task))
                if sae_auc is None: continue
                deltas[(sae, task)] = txc_auc - sae_auc
        # Group by (sae, category)
        sae_cat_deltas = defaultdict(list)
        for (sae, task), delta in deltas.items():
            cat = TASK_CATEGORY.get(task, "other")
            sae_cat_deltas[(sae, cat)].append((task, delta))
        for sae in saes:
            print()
            print(f"  vs {sae}:")
            cat_rows = []
            for cat in ("knowledge_profession", "topic", "topic_amazon", "sentiment",
                        "language_id", "code_language", "coreference", "other"):
                rows = sae_cat_deltas.get((sae, cat), [])
                if not rows: continue
                deltas_only = [d for _, d in rows]
                mean_delta = sum(deltas_only)/len(deltas_only)
                wins = sum(1 for d in deltas_only if d > 0.001)
                losses = sum(1 for d in deltas_only if d < -0.001)
                cat_rows.append((cat, mean_delta, wins, losses, len(deltas_only)))
            print(f"    {'category':28s}  {'mean Δ':>10s}  {'wins':>5s}/{'total':>5s}  {'losses':>6s}")
            for cat, m, w, l, n in cat_rows:
                tag = "TXC favoured" if m > 0.005 else ("SAE favoured" if m < -0.005 else "≈ tied")
                print(f"    {cat:28s}  {m:>+10.4f}  {w:>5d}/{n:>5d}  {l:>6d}    {tag}")

        # Top-5 winning + losing tasks for each SAE
        for sae in saes:
            print()
            print(f"  Top-5 task wins (TXC > {sae}):")
            ranked = sorted([(d, t) for (s, t), d in deltas.items() if s == sae],
                            reverse=True)
            for delta, task in ranked[:5]:
                cat = TASK_CATEGORY.get(task, "other")
                print(f"    {task:42s}  Δ={delta:+.4f}  ({cat})")
            print(f"  Top-5 task losses (TXC < {sae}):")
            for delta, task in ranked[-5:]:
                cat = TASK_CATEGORY.get(task, "other")
                print(f"    {task:42s}  Δ={delta:+.4f}  ({cat})")


if __name__ == "__main__":
    main()
