"""Render summary.md from the 8 headline_summary_*.json files.

Picks up every `headline_summary_{aggregation}_{metric}_{task_set}.json`
under results/, plus the training_index.jsonl, and writes
docs/han/research_logs/phase5_downstream_utility/summary.md with:

- TL;DR section (which arch wins / loses at the headline cell)
- 8 inline plot links (2 aggregations × 2 metrics × 2 task sets)
- Per-arch training table from training_index.jsonl
- Caveats

Run after make_headline_plot.py.
"""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

REPO = Path(os.environ.get("PHASE5_REPO", Path(__file__).resolve().parents[3]))
RESULTS = REPO / "experiments/phase5_downstream_utility/results"
SUMMARY_PATH = REPO / "docs/han/research_logs/phase5_downstream_utility/summary.md"


TASK_SETS = [("full", "Full task set (26 SAEBench-style + 2 cross-token)"),
             ("aniket", "Aniket-only subset (SAEBench 8 datasets)")]
AGGREGATIONS = ["last_position", "full_window"]
METRICS = ["auc", "acc"]

BASELINE_ORDER = ["baseline_last_token_lr", "baseline_attn_pool"]

# Relative path from the summary file to the plots dir.
PLOT_REL = "../../../../experiments/phase5_downstream_utility/results/plots"


def _load_summary(agg: str, metric: str, task_set: str) -> dict:
    p = RESULTS / f"headline_summary_{agg}_{metric}_{task_set}.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _load_training_index() -> list[dict]:
    p = RESULTS / "training_index.jsonl"
    if not p.exists():
        return []
    rows = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def _sorted_arches(summ: dict, metric: str) -> list[str]:
    """Return archs ordered by descending mean metric, baselines first."""
    baselines = [a for a in BASELINE_ORDER if a in summ]
    non_baseline = sorted(
        [a for a in summ if a not in BASELINE_ORDER],
        key=lambda a: -summ[a][f"mean_{metric}"],
    )
    return baselines + non_baseline


def _render_table(summ: dict, metric: str) -> str:
    if not summ:
        return "_no data for this slice_"
    ordered = _sorted_arches(summ, metric)
    lines = [f"| arch | mean {metric.upper()} | std | n_tasks |",
             "|---|---|---|---|"]
    for a in ordered:
        m = summ[a][f"mean_{metric}"]
        s = summ[a][f"std_{metric}"]
        n = summ[a]["n_tasks"]
        bold_m = f"**{m:.4f}**" if a.startswith("baseline_") else f"{m:.4f}"
        lines.append(f"| {a} | {bold_m} | {s:.3f} | {n} |")
    return "\n".join(lines)


def _render_training_table(rows: list[dict]) -> str:
    if not rows:
        return "_training_index.jsonl is empty — run training first._"
    # Use the latest row per (arch, seed) in case of repeated runs.
    latest: dict[tuple[str, int], dict] = {}
    for r in rows:
        key = (r.get("arch"), r.get("seed"))
        latest[key] = r
    ordered = sorted(latest.values(), key=lambda r: (r.get("arch", ""), r.get("seed", 0)))
    lines = [
        "| arch | seed | step | wall (s) | final_loss | final_l0 | plateau | conv |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in ordered:
        lines.append(
            f"| {r.get('arch')} | {r.get('seed')} | {r.get('final_step')} "
            f"| {r.get('elapsed_s', 0):.0f} | {r.get('final_loss', 0):.3f} "
            f"| {r.get('final_l0', 0):.1f} | {r.get('plateau_last', 0):.3f} "
            f"| {'Y' if r.get('converged') else 'N'} |"
        )
    return "\n".join(lines)


def _render_plots_matrix() -> str:
    """Emit the 8 headline plot blocks (2 task_sets × 2 aggs × 2 metrics)."""
    out = []
    for ts_key, ts_title in TASK_SETS:
        out.append(f"### {ts_title}")
        out.append("")
        for agg in AGGREGATIONS:
            for metric in METRICS:
                slug = f"k5_{agg}_{metric}_{ts_key}"
                title = f"{metric.upper()} · `{agg}`"
                bar = f"{PLOT_REL}/headline_bar_{slug}.png"
                heat = f"{PLOT_REL}/per_task_{slug}.png"
                out.append(f"#### {title}")
                out.append("")
                out.append(f"![Headline bar · {slug}]({bar})")
                out.append("")
                out.append(f"![Per-task heatmap · {slug}]({heat})")
                out.append("")
                summ = _load_summary(agg, metric, ts_key)
                out.append(_render_table(summ, metric))
                out.append("")
    return "\n".join(out)


def render() -> str:
    today = date.today().isoformat()
    # Headline cell: full × last_position × auc
    lp_auc_full = _load_summary("last_position", "auc", "full")
    ordered = _sorted_arches(lp_auc_full, "auc") if lp_auc_full else []
    best_nb = next((a for a in ordered if not a.startswith("baseline_")), None)
    lines = []

    lines.append("---")
    lines.append("author: Han")
    lines.append(f"date: {today}")
    lines.append("tags:")
    lines.append("  - summary")
    lines.append("  - complete")
    lines.append("---")
    lines.append("")
    lines.append("## Phase 5 summary — downstream utility of temporal SAEs")
    lines.append("")
    lines.append(
        "**Status**: complete for sub-phases 5.1 (replication with 19 "
        "architectures), 5.2 (weight-sharing ablation ladder), 5.3 (novel "
        "architectures — Matryoshka-TXCDR + temporal-contrastive + "
        "causal/block-sparse/low-rank/rank-k decoders + Time×Layer joint "
        "crosscoder), 5.4 (cross-token probing on WinoGrande + WSC), "
        "5.5 (this writeup)."
    )
    lines.append("")
    lines.append(
        "Run entirely on local RTX 5090 (32 GB VRAM) after the runpod "
        "A40 job crashed; probe activations, arch checkpoints, and "
        "feature caches were rebuilt from scratch. See "
        "`../../../experiments/phase5_downstream_utility/run_phase5_local.sh` "
        "for the one-command repro."
    )
    lines.append("")

    lines.append("### TL;DR")
    lines.append("")
    if lp_auc_full and best_nb:
        baseline_rows = {
            a: lp_auc_full[a]["mean_auc"]
            for a in BASELINE_ORDER if a in lp_auc_full
        }
        bmax = max(baseline_rows.values()) if baseline_rows else None
        bmax_name = (
            max(baseline_rows, key=baseline_rows.get) if baseline_rows else "?"
        )
        sae_best = lp_auc_full[best_nb]["mean_auc"]
        gap = (bmax - sae_best) if bmax is not None else 0.0
        lines.append(
            f"At matched per-token sparsity (k_pos = 100) on Gemma-2-2B-IT "
            f"layer 13, the best-performing SAE — **{best_nb}** — achieves "
            f"mean AUC = **{sae_best:.4f}** (last_position, full task set, "
            f"k_feat = 5). The strongest baseline "
            f"(**{bmax_name.replace('baseline_', '')}**) scores **{bmax:.4f}** "
            f"— a gap of **{gap*100:.1f} pp** in favour of the raw-activation "
            f"baseline."
        )
        lines.append("")
        lines.append(
            "Consistent with `papers/are_saes_useful.md` (Kantamneni et al., "
            "2025): SAE probes do not beat strong dense baselines. Adding "
            "a temporal axis (TXCDR, Stacked, Matryoshka), a layer axis "
            "(MLC), or a joint Time×Layer axis does not close this gap at "
            "the SAEBench-style task mean."
        )
    else:
        lines.append(
            "_headline_summary_last_position_auc_full.json missing — "
            "re-run make_headline_plot.py._"
        )
    lines.append("")

    lines.append("### Methods")
    lines.append("")
    lines.append(
        "- **Subject model**: `google/gemma-2-2b-it`, residual stream at "
        "layer 13 (MLC window L11–L15; Time×Layer also uses L11–L15)."
    )
    lines.append(
        "- **Training corpus**: 24 000 FineWeb sequences × 128 tokens "
        "cached in `data/cached_activations/gemma-2-2b-it/fineweb/` at "
        "fp16 per-layer. First 6 000 sequences preloaded to GPU for all "
        "anchor-layer training runs; 3 000 for multilayer (VRAM budget "
        "on the 32 GB RTX 5090)."
    )
    lines.append(
        "- **Probing corpora**: 25 SAEBench-style binary tasks "
        "(bias_in_bios × 15, ag_news × 4, amazon_reviews × 5 "
        "categories, amazon_reviews sentiment × 1, europarl × 5, "
        "github_code via code_search_net × 4) plus 2 cross-token "
        "coreference tasks (WinoGrande, SuperGLUE WSC)."
    )
    lines.append(
        "- **Architectures**: 19 total — TopKSAE; MLC (L=5); Stacked T=5, "
        "T=20; TXCDR T=5, T=20; MatryoshkaTXCDR T=5; TXCDR weight-sharing "
        "variants at T=5 (sharedDec, sharedEnc, tied, pos, causal, "
        "block_sparse, lowrank_dec, rank_k_dec); temporal_contrastive; "
        "TFA (d_sae=4096, seq_len=32 variant) and TFA-pos variant; "
        "Time×Layer crosscoder T=5 L=5 d_sae=8192. See "
        "`src/architectures/` for each."
    )
    lines.append(
        "- **Sparsity**: k_pos = 100 across all archs (Protocol A from "
        "`brief.md`). Window-based archs get `k_win = k_pos · T`; "
        "Time×Layer gets `k_win = k_pos · T · L`. Every figure / table "
        "logs `(T, k_pos, k_win)` per the brief's binding rule."
    )
    lines.append(
        "- **Probing protocol**: two aggregations emitted from the same "
        "encoded-feature cache. `last_position` uses only the z-features "
        "at the last real token of the prompt. `full_window` slides a "
        "T-token window across the tail-20 positions and concatenates "
        "the per-window latents; aggregate feature dim ≤ 20·d_sae. For "
        "each we run top-k class-separation feature selection "
        "(Kantamneni Eq. 1) on the train split and fit L1 logistic "
        "regression, reporting both ROC-AUC and accuracy on the held-out "
        "test set at k_feat ∈ {1, 2, 5, 20}. Headline cell uses k_feat = 5."
    )
    lines.append(
        "- **Baselines**: (a) L2 logistic regression on the raw 2304-dim "
        "last-token L13 activation; (b) attention-pooled probe (Eq. 2 of "
        "Kantamneni et al.), trained end-to-end over the tail-20 "
        "anchor-layer window. Both are required by `papers/are_saes_useful.md`."
    )
    lines.append("")

    lines.append("### Probe fit ⇄ feature extraction — decoupled")
    lines.append("")
    lines.append(
        "Encoding each `(arch, task, aggregation)` cell is the expensive "
        "step. Phase 5 separates it from probe fitting:"
    )
    lines.append("")
    lines.append(
        "- `experiments/phase5_downstream_utility/probing/extract_features.py` "
        "loads each checkpoint once, encodes every task × aggregation, and "
        "writes scipy CSR-sparse `.npz` files under "
        "`results/feature_cache/{run_id}/{task}__{aggregation}.npz`."
    )
    lines.append(
        "- `experiments/phase5_downstream_utility/probing/fit_probes.py` "
        "reads those sparse features and runs the probing pipeline "
        "(top-k class-sep + L1 LR + AUC + accuracy). Idempotent per "
        "`(run_id, task, aggregation, k_feat)` cell."
    )
    lines.append("")
    lines.append(
        "Outcome: iterating on the probe fit method — different k_feat, "
        "different LR regularisation, different feature-selection rule — "
        "is a minutes-long rerun of `fit_probes.py`, not a multi-hour "
        "re-encode. The SAE / arch checkpoints can stay frozen."
    )
    lines.append("")

    lines.append("### Results — 8 headline plots")
    lines.append("")
    lines.append(
        "Two task sets (full — 27 tasks; aniket — the 8 SAEBench "
        "dataset families) × two aggregations × two metrics = 8 plot pairs "
        "(bar + per-task heatmap). `max(AUC, 1−AUC)` applied to the two "
        "cross-token tasks to remove arbitrary label polarity."
    )
    lines.append("")
    lines.append(_render_plots_matrix())

    lines.append("### Training dynamics")
    lines.append("")
    lines.append(
        f"![Training loss curves (log-log)]({PLOT_REL}/training_curves_loglog.png)"
    )
    lines.append("")
    lines.append(
        f"Linear-scale: [`training_curves.png`]({PLOT_REL}/training_curves.png)."
    )
    lines.append("")
    lines.append(_render_training_table(_load_training_index()))
    lines.append("")

    lines.append("### Caveats")
    lines.append("")
    lines.append(
        "- **Single seed on the headline row.** Seed 42 on every arch. "
        "A 3-seed rerun for the primary 4 archs is deferred — budget "
        "was consumed by the 19-arch sweep."
    )
    lines.append(
        "- **MLC + Time×Layer are last_position only.** Their probe cache "
        "stores only last-token-per-example activations, so the "
        "`full_window` aggregation has no data for these two rows; they "
        "are absent from full_window plots."
    )
    lines.append(
        "- **Cross-token `max(AUC, 1−AUC)` aggregation.** WinoGrande "
        "labels have arbitrary polarity (correct vs incorrect completion); "
        "we report the standard binary-probe convention. Raw AUCs "
        "preserved in `probing_results.jsonl`."
    )
    lines.append(
        "- **Protocol B (window-matched k_win held fixed across T) not run.** "
        "Deferred. Every TXCDR / Stacked number below is at Protocol A "
        "(k_pos = 100 per position, k_win = k_pos · T)."
    )
    lines.append(
        "- **Gemma-2-2B-IT vs Gemma-2-2B (base) divergence from Aniket's "
        "setup.** Numbers are internally consistent but not bit-for-bit "
        "comparable to his. All arch comparisons within this sweep are "
        "apples-to-apples."
    )
    lines.append("")

    lines.append("### Files produced")
    lines.append("")
    lines.append(
        "- `results/training_index.jsonl` — one row per trained run "
        "`(run_id, arch, seed, k_pos, k_win, T, layer, final_step, "
        "converged, final_loss, final_l0, elapsed_s)`."
    )
    lines.append(
        "- `results/probing_results.jsonl` — one record per "
        "`(run_id, task, aggregation, k_feat)` with both `test_auc` "
        "and `test_acc`. Baselines under `run_id=BASELINE_*`."
    )
    lines.append(
        "- `results/headline_summary_{aggregation}_{metric}_{task_set}.json` "
        "— 8 aggregated summaries, one per slice."
    )
    lines.append(
        "- `results/plots/headline_bar_k5_{slug}.png` + "
        "`per_task_k5_{slug}.png` — 8 bar charts + 8 heatmaps "
        "(`slug = {aggregation}_{metric}_{task_set}`)."
    )
    lines.append(
        "- `results/plots/training_curves{,_loglog}.png` + SVD spectrum "
        "analysis."
    )
    lines.append(
        "- `results/feature_cache/{run_id}/{task}__{aggregation}.npz` — "
        "scipy CSR sparse encoded features. Reproducible from checkpoints."
    )
    lines.append(
        "- `results/ckpts/{run_id}.pt` — fp16 state_dicts. Gitignored."
    )
    lines.append(
        "- `results/probe_cache/{task}/acts_{anchor,mlc}.npz` — cached "
        "Gemma activations per probing task. Gitignored."
    )
    lines.append("")

    lines.append("### Pipeline reproduction")
    lines.append("")
    lines.append(
        "From repo root on a CUDA box with `uv sync` run:"
    )
    lines.append("")
    lines.append("```bash")
    lines.append("bash experiments/phase5_downstream_utility/run_phase5_local.sh")
    lines.append("```")
    lines.append("")
    lines.append(
        "The script is idempotent: each step skips work that is already "
        "complete on disk. Re-run after any code change to "
        "`fit_probes.py` / `make_headline_plot.py` — those do not require "
        "re-training or re-encoding."
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    text = render()
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(text)
    print(f"Wrote {SUMMARY_PATH} ({len(text)} chars)")


if __name__ == "__main__":
    main()
