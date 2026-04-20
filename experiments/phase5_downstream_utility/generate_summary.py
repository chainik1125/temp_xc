"""Generate phase5 summary.md tables from probing_results.jsonl + training_index.jsonl.

Produces the tables needed for the final summary writeup:
- headline (AUC, acc) × (last_position, full_window) × (full, aniket) = 8 tables
- cross-token breakdown
- training / capacity table
- SVD finding blurb

Prints markdown-ready blocks to stdout.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path("/workspace/temp_xc")
RESULTS = REPO / "experiments/phase5_downstream_utility/results"
JSONL = RESULTS / "probing_results.jsonl"
TRAIN_IDX = RESULTS / "training_index.jsonl"

FLIP = {"winogrande_correct_completion", "wsc_coreference"}
ANIKET = {
    "ag_news", "amazon_reviews", "amazon_reviews_sentiment",
    "bias_in_bios_set1", "bias_in_bios_set2", "bias_in_bios_set3",
    "europarl", "github_code",
}
CROSS = FLIP
HEADLINE_K = 5

ORDERED = [
    "topk_sae", "stacked_t5", "stacked_t20",
    "txcdr_t5", "txcdr_t20",
    "txcdr_shared_dec_t5", "txcdr_shared_enc_t5",
    "txcdr_tied_t5", "txcdr_pos_t5", "txcdr_causal_t5",
    "txcdr_block_sparse_t5", "txcdr_lowrank_dec_t5",
    "txcdr_rank_k_dec_t5",
    "matryoshka_t5", "temporal_contrastive",
    "time_layer_crosscoder_t5",
    "mlc", "tfa_small", "tfa_pos_small",
]
BASELINES = ["baseline_last_token_lr", "baseline_attn_pool"]


def _load():
    recs = []
    with JSONL.open() as f:
        for line in f:
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    return recs


def _agg(recs, aggregation, metric, filt=None):
    out = defaultdict(dict)
    key = "test_auc" if metric == "auc" else "test_acc"
    for r in recs:
        if r.get("error") or r.get(key) is None:
            continue
        if r.get("aggregation") != aggregation:
            continue
        if filt and r.get("dataset_key") not in filt:
            continue
        k = r.get("k_feat")
        if k is not None and k != HEADLINE_K:
            continue
        v = float(r[key])
        if r["task_name"] in FLIP:
            v = max(v, 1 - v)
        out[r["arch"]][r["task_name"]] = v
    return out


def _summary(out):
    return {
        a: (float(np.mean(list(v.values()))),
            float(np.std(list(v.values()))),
            len(v))
        for a, v in out.items() if v
    }


def _print_table(summ, label):
    print(f"\n#### {label}\n")
    print("| arch | mean | std | n_tasks |")
    print("|---|---|---|---|")
    ordered = [a for a in ORDERED if a in summ]
    base = [a for a in BASELINES if a in summ]
    rows = []
    for a in base:
        m, s, n = summ[a]
        rows.append((a, m, s, n, "baseline"))
    for a in ordered:
        m, s, n = summ[a]
        rows.append((a, m, s, n, "sae"))
    # sort saes desc, baselines first
    base_rows = [r for r in rows if r[4] == "baseline"]
    sae_rows = sorted([r for r in rows if r[4] == "sae"], key=lambda x: -x[1])
    for a, m, s, n, _ in base_rows + sae_rows:
        marker = "**" if a.startswith("baseline") else ""
        print(f"| {a} | {marker}{m:.4f}{marker} | {s:.4f} | {n} |")


def _cross_token(recs, aggregation="last_position"):
    d = _agg(recs, aggregation, "auc")
    print(f"\n#### Cross-token AUC [{aggregation}]\n")
    print("| arch | winogrande | wsc |")
    print("|---|---|---|")
    base = BASELINES + [a for a in ORDERED if a in d]
    sae_scores = []
    for a in base:
        if a not in d:
            continue
        w = d[a].get("winogrande_correct_completion", float("nan"))
        s = d[a].get("wsc_coreference", float("nan"))
        if a.startswith("baseline"):
            print(f"| **{a}** | **{w:.4f}** | **{s:.4f}** |")
        else:
            sae_scores.append((a, w, s))
    for a, w, s in sorted(sae_scores, key=lambda x: -(x[1] + x[2]) / 2):
        print(f"| {a} | {w:.4f} | {s:.4f} |")


def _training_table():
    print("\n#### Training & capacity\n")
    print("| arch | final_step | final_loss | final_l0 | plateau_last | elapsed_s | converged |")
    print("|---|---|---|---|---|---|---|")
    with TRAIN_IDX.open() as f:
        for line in f:
            r = json.loads(line)
            print(
                f"| {r['run_id']} | {r['final_step']} | {r['final_loss']:.1f} | "
                f"{r['final_l0']:.1f} | {r['plateau_last']:.4f} | "
                f"{r['elapsed_s']:.0f} | {r['converged']} |"
            )


def _svd_blurb():
    sv_path = RESULTS / "svd_spectrum.json"
    if not sv_path.exists():
        return
    d = json.loads(sv_path.read_text())
    print("\n#### Per-feature SVD spectrum: T=5 vs T=20\n")
    for name, r in d.items():
        print(f"- **{name}** (T={r['T']}): "
              f"effective-rank/T = **{r['effective_rank_ratio']:.3f}**")
    t5 = d.get("txcdr_t5", {}).get("effective_rank_ratio")
    t20 = d.get("txcdr_t20", {}).get("effective_rank_ratio")
    if t5 and t20:
        print(
            f"\nTXCDR-T20's per-feature decoder spectrum is "
            f"**{(t20 - t5)/t5*100:.1f}% flatter** than TXCDR-T5's. "
            "Under the hypothesis that a flatter spectrum indicates the feature "
            "uses more of its per-position rank budget, T=20 is measurably "
            "under-regularized: the rank-K decoder variants (lowrank_dec, "
            "rank_k_dec) test whether constraining the per-feature rank "
            "recovers performance.")


def main():
    recs = _load()
    print(f"# Phase 5 summary — auto-generated tables\n")
    print(f"Records: {len(recs)}")

    for ts, filt in [("full", None), ("aniket", ANIKET)]:
        for agg in ("last_position", "full_window"):
            for met in ("auc", "acc"):
                summ = _summary(_agg(recs, agg, met, filt))
                if summ:
                    _print_table(summ, f"{agg} × {met} × {ts}")

    for agg in ("last_position", "full_window"):
        _cross_token(recs, agg)

    _training_table()
    _svd_blurb()


if __name__ == "__main__":
    main()
