#!/usr/bin/env python3
"""Summarize completed C7 camera-ready detection JSONs without inference.

Example: python summarize_detection.py --root /workspace/backtracking/results
Use --require-complete for a publication gate, or --no-plots for CSV/JSON only.
Partial runs remain explicitly marked; a series gets a mean and sample SD only
after all three dictionary seeds are present. No bootstrap CI is inferred from
folds or seeds. Caption/protocol details are written beside the figure files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

SEEDS = (1, 2, 42)
S_GRID = (1, 2, 4, 8, 16, 32)
MODES = ("historical_raw_C1", "matched_scaled_C1")
ARCHS = ("txc_base", "topk_sae", "tsae_paper", "stacked_sae")
PROTOCOL = "c7-camera-ready-detection-v2-prompt-groups"
GROUPING = "normalized_prompt_sha256"
PROMPT_NORMALIZATION = "Unicode NFKC, casefold, collapse whitespace; preserve mathematical punctuation"
PROMPTS_SHA256 = "f718d76c1be63bddb83cfb7a9fe03ebde0bf5036a02defb1addc104f8829dd6a"
TRAINING_PROTOCOLS = {"c7-camera-ready-300k-v1", "c7-300k-seeded-v1"}
HISTORICAL_COMMIT = "284a8bf5e3e5a7cc094dd68c6fa5a92a9fd4eec3"
ACT_CACHE_KEY = "fb2a74be884e512a"
ACTS_SHA256 = "1656f6be2cd85fb85c8b246b9b27933f73ef40cfaac84078169dfd3bbbe27810"
METRIC = "mean_fold_average_precision"
SERIES = (("txc_base", 32768), ("topk_sae", 32768), ("tsae_paper", 32768),
          ("stacked_sae", 32768), ("tsae_paper", 16384))
LABELS = {("txc_base", 32768): "TXC", ("topk_sae", 32768): "TopK SAE",
          ("tsae_paper", 32768): "T-SAE 32K", ("stacked_sae", 32768): "Stacked SAE",
          ("tsae_paper", 16384): "T-SAE 16K · sensitivity"}
# Nord Frost, Polar Night and Aurora. Width sensitivity shares its family color.
COLORS = {"txc_base": "#5E81AC", "topk_sae": "#4C566A",
          "tsae_paper": "#D08770", "stacked_sae": "#B48EAD"}
PANELS = (("last", "Last token"), ("mean", "Mean pool"), ("max", "Max pool"))
CSV_FIELDS = ("arch", "d_sae", "comparison_role", "seed", "train_key", "view",
              "probe_mode", "S", METRIC, "mean_fold_roc_auc", "pooled_oof_average_precision",
              "pooled_oof_roc_auc", "candidate_features", "mean_l0", "n_questions", "n_prompt_groups",
              "n_sentences", "training_protocol", "status", "convergence_warning_count",
              "source_path", "source_sha256", "checkpoint_sha256", "evaluator_sha256")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cell_id(arch: str, width: int, seed: int) -> str:
    return f"{arch}_d{width}_seed{seed}"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def probability(value: Any, name: str, *, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value <= 1, f"invalid {name}: {value!r}")
    return float(value)


def validate(payload: dict[str, Any]) -> None:
    """Fail closed on mixed protocols, partial budgets and malformed metrics."""
    arch, width, seed = payload["arch"], payload["d_sae"], payload["seed"]
    require((arch, width) in SERIES and seed in SEEDS, "cell is outside the frozen grid")
    require(payload["detection_protocol"] == PROTOCOL, "detection protocol differs")
    require(payload["training_protocol"] in TRAINING_PROTOCOLS, "unapproved training protocol")
    require(payload["n_steps_completed"] == 300_000, "dictionary is not trained for 300K steps")
    require(payload.get("smoke") is False, "production result must explicitly set smoke=false")
    cfg = payload["checkpoint_config"]
    for key, value in {"status": "complete", "arch": arch, "d_sae": width, "seed": seed,
                       "n_steps_completed": 300_000, "historical_commit": HISTORICAL_COMMIT,
                       "act_cache_key": ACT_CACHE_KEY, "train_key": payload["train_key"],
                       "protocol_version": payload["training_protocol"]}.items():
        require(cfg.get(key) == value, f"checkpoint {key} mismatch")
    require(not cfg.get("smoke") and not cfg.get("is_smoke"), "smoke checkpoint")
    require(cfg["training_cfg"]["n_steps"] == 300_000
            and cfg["training_cfg"]["batch_size"] == 1024, "training schedule differs")
    require(cfg["hparams"]["d_sae"] == width and cfg["hparams"]["k_pos"] == 20,
            "checkpoint width/sparsity differs")
    if arch in ("txc_base", "stacked_sae"):
        require(cfg["hparams"]["T"] == 5, "checkpoint window is not T=5")
    provenance = payload["provenance"]
    require(provenance["historical_commit"] == HISTORICAL_COMMIT
            and provenance["sentence_acts_sha256"] == ACTS_SHA256
            and provenance["stage_a_prompts_sha256"] == PROMPTS_SHA256, "source/cohort hash differs")
    for key in ("checkpoint_sha256", "checkpoint_config_sha256", "evaluator_sha256",
                "sentence_keys_sha256"):
        require(isinstance(provenance.get(key), str) and len(provenance[key]) == 64
                and all(c in "0123456789abcdef" for c in provenance[key]), f"invalid {key}")
    data, protocol = payload["data"], payload["protocol"]
    require((data["n_sentences"], data["n_positive"], data["n_questions"], data["n_prompt_groups"])
            == (25204, 3169, 300, 213), "cohort counts differ")
    require(math.isclose(data["positive_fraction"], 3169 / 25204, abs_tol=1e-12),
            "cohort prevalence differs")
    require(data["evaluated_offsets"] == [-12, -11, -10, -9, -8], "evaluation window differs")
    for key, value in {"S_grid": list(S_GRID), "folds": 5, "grouping": GROUPING,
                       "group_normalization": PROMPT_NORMALIZATION,
                       "primary_budget": 8, "primary_metric": METRIC, "encoder_mode": "eval",
                       "encoder_dtype": "float32", "probe_modes": list(MODES),
                       "C": 1.0, "probe_random_state": 42}.items():
        require(protocol.get(key) == value, f"probe protocol {key} differs")
    required_views = {"native"} if arch == "txc_base" else {"position_identity"} if arch == "stacked_sae" else {"last", "mean", "max"}
    allowed_views = required_views | ({"position_identity"} if arch in ("topk_sae", "tsae_paper") else set())
    require(required_views <= payload["views"].keys() <= allowed_views, "missing/unexpected readout")
    for name, view in payload["views"].items():
        support = view["support"]
        candidates = width * (5 if name == "position_identity" else 1)
        require(support["candidate_features"] == candidates, f"{name}: candidate dimension differs")
        require(0 <= support["nnz"] <= candidates * data["n_sentences"], f"{name}: invalid nnz")
        require(math.isclose(support["mean_l0"], support["nnz"] / data["n_sentences"],
                             rel_tol=1e-9, abs_tol=1e-9), f"{name}: support counts disagree")
        require(set(view["probes"]) == set(MODES), f"{name}: missing/unexpected probe mode")
        for mode, probe in view["probes"].items():
            require(set(probe["metrics"]) == {str(s) for s in S_GRID}, f"{name}/{mode}: incomplete S grid")
            require(len(probe["folds"]) == 5
                    and [fold["fold"] for fold in probe["folds"]] == list(range(5)),
                    f"{name}/{mode}: incomplete/duplicated folds")
            require(sum(fold["n_test"] for fold in probe["folds"]) == data["n_sentences"]
                    and sum(fold["n_test_questions"] for fold in probe["folds"]) == data["n_prompt_groups"],
                    f"{name}/{mode}: fold sizes do not cover the cohort")
            for budget in S_GRID:
                metric = probe["metrics"][str(budget)]
                fold_ap = [probability(fold["budgets"][str(budget)]["average_precision"], "fold AP")
                           for fold in probe["folds"]]
                require(metric["fold_average_precision"] == fold_ap,
                        f"{name}/{mode}/S{budget}: fold AP arrays disagree")
                ap = probability(metric[METRIC], METRIC)
                require(math.isclose(ap, statistics.mean(fold_ap), abs_tol=1e-12),
                        f"{name}/{mode}/S{budget}: mean-fold AP does not match folds")
                probability(metric["mean_fold_roc_auc"], "mean ROC-AUC", nullable=True)
                probability(metric["pooled_oof_average_precision"], "pooled OOF AP")
                probability(metric["pooled_oof_roc_auc"], "pooled OOF ROC-AUC")
    warnings = payload["convergence_warning_count"]
    require(isinstance(warnings, int) and warnings >= 0, "invalid convergence warning count")
    require((payload["status"] == "complete") == (warnings == 0), "status/warning count disagree")


def collect(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    cells, skipped, seen = [], [], set()
    for path in sorted((root / "cells").glob("*/detection.json")):
        payload = json.loads(path.read_text())
        reason = ("smoke result" if payload.get("smoke") else
                  "non-core architecture" if payload.get("arch") not in ARCHS else
                  "unfinished detection" if payload.get("status") not in ("complete", "complete_with_warnings") else None)
        if reason:
            skipped.append({"path": str(path.relative_to(root)), "reason": reason})
            continue
        try:
            validate(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}: {exc}") from exc
        identity = (payload["arch"], payload["d_sae"], payload["seed"])
        require(identity not in seen, f"duplicate cell identity: {identity}")
        seen.add(identity)
        cells.append({"result": payload, "source_path": str(path.relative_to(root)),
                      "source_sha256": digest(path)})
    # Same data bytes do not establish the same row order without the key hash.
    require(len({c["result"]["provenance"]["sentence_keys_sha256"] for c in cells}) <= 1,
            "cells use different sentence-key ordering")
    return cells, skipped


def long_rows(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for cell in cells:
        result = cell["result"]
        for view_name, view in result["views"].items():
            for mode in MODES:
                for budget in S_GRID:
                    rows.append({
                        **{key: result[key] for key in ("arch", "d_sae", "seed", "train_key", "training_protocol", "status", "convergence_warning_count")},
                        "comparison_role": "T-SAE 16K sensitivity" if result["d_sae"] == 16384 else "32K core",
                        "view": view_name, "probe_mode": mode, "S": budget,
                        **{key: view["probes"][mode]["metrics"][str(budget)][key]
                           for key in (METRIC, "mean_fold_roc_auc", "pooled_oof_average_precision", "pooled_oof_roc_auc")},
                        **{key: view["support"][key] for key in ("candidate_features", "mean_l0")},
                        **{key: result["data"][key] for key in ("n_questions", "n_prompt_groups", "n_sentences")},
                        "source_path": cell["source_path"], "source_sha256": cell["source_sha256"],
                        **{key: result["provenance"][key] for key in ("checkpoint_sha256", "evaluator_sha256")}})
    return rows


def summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, dict[int, float]] = {}
    for row in rows:
        key = tuple(row[key] for key in ("arch", "d_sae", "view", "probe_mode", "S"))
        require(row["seed"] not in groups.setdefault(key, {}), f"duplicate metric row: {key}")
        groups[key][row["seed"]] = row[METRIC]
    return [{"arch": key[0], "d_sae": key[1], "view": key[2], "probe_mode": key[3], "S": key[4],
             "seeds": sorted(values), "seed_values": {str(seed): values[seed] for seed in sorted(values)},
             "n_seeds": len(values), "complete_seed_set": set(values) == set(SEEDS),
             "mean": statistics.mean(values.values()) if set(values) == set(SEEDS) else None,
             "sample_sd": statistics.stdev(values.values()) if set(values) == set(SEEDS) else None}
            for key, values in sorted(groups.items())]


def panel_view(arch: str, pool: str) -> str:
    return "native" if arch == "txc_base" else "position_identity" if arch == "stacked_sae" else pool


def plot_preview(summary: list[dict[str, Any]], output: Path, *, completed: int, warning_count: int) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.labelcolor": "#2E3440",
                         "text.color": "#2E3440", "xtick.color": "#4C566A", "ytick.color": "#4C566A",
                         "axes.edgecolor": "#D8DEE9", "svg.fonttype": "none", "pdf.fonttype": 42})
    outputs = []
    by_key = {(r["arch"], r["d_sae"], r["view"], r["probe_mode"], r["S"]): r for r in summary}
    markers = {1: "o", 2: "^", 42: "s"}
    for mode in MODES:
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7), sharex=True, sharey=True)
        fig.subplots_adjust(left=.075, right=.985, top=.81, bottom=.27, wspace=.12)
        available = set()
        for axis, (pool, title) in zip(axes, PANELS):
            axis.set_title(title, fontsize=10, pad=10)
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", color="#E5E9F0", linewidth=.6)
            axis.set_axisbelow(True)
            for arch, width in SERIES:
                records = [by_key.get((arch, width, panel_view(arch, pool), mode, budget)) for budget in S_GRID]
                if not all(records):
                    continue
                available.add((arch, width))
                color, sensitivity = COLORS[arch], width == 16384
                for seed in records[0]["seeds"]:
                    require(all(seed in r["seeds"] for r in records), "seed is missing one budget")
                    y = [r["seed_values"][str(seed)] for r in records]
                    # A fixed tiny horizontal offset distinguishes three seed symbols.
                    x = [budget * 2 ** ((SEEDS.index(seed) - 1) * .035) for budget in S_GRID]
                    axis.scatter(x, y, s=15, marker=markers[seed], linewidth=.7,
                                 edgecolors=color, facecolors="white" if sensitivity else color, alpha=.62,
                                 zorder=3)
                if all(r["complete_seed_set"] for r in records):
                    mean, sd = [r["mean"] for r in records], [r["sample_sd"] for r in records]
                    axis.fill_between(S_GRID, [m-s for m, s in zip(mean, sd)], [m+s for m, s in zip(mean, sd)],
                                      color=color, alpha=.07 if sensitivity else .12, linewidth=0)
                    axis.plot(S_GRID, mean, color=color, lw=1.8, ls="--" if sensitivity else "-", zorder=4)
            axis.set_xscale("log", base=2)
            axis.set_xticks(S_GRID, [str(s) for s in S_GRID])
            axis.set_xlabel("Feature budget S", labelpad=7)
            axis.set_xlim(.88, 36)
        displayed = [r for r in summary if r["probe_mode"] == mode
                     and r["view"] in {panel_view(r["arch"], pool) for pool, _ in PANELS}]
        ceiling = max((max(r["seed_values"].values()) for r in displayed), default=.3)
        ceiling = max(ceiling, max(((r["mean"] + r["sample_sd"]) for r in displayed
                                    if r["complete_seed_set"]), default=.3))
        axes[0].set_ylim(0, min(1, max(.3, ceiling * 1.08)))
        # sharey ensures every panel uses the same natural AP scale.
        axes[0].set_ylabel("Mean-fold average precision", labelpad=8)
        heading = "Raw features · C = 1" if mode == MODES[0] else "Standardized features · C = 1"
        status = f"PARTIAL · {completed}/15 cells" if completed != 15 else "15/15 cells · 3 seeds"
        if warning_count:
            status += " · convergence warnings"
        fig.text(.075, .965, heading, fontsize=12, weight="semibold", va="top")
        fig.text(.985, .957, status, fontsize=8.5, ha="right", va="top", color="#BF616A" if completed != 15 or warning_count else "#4C566A")
        handles = [Line2D([], [], color=COLORS[arch], lw=1.8, ls="--" if width == 16384 else "-",
                          label=LABELS[(arch, width)]) for arch, width in SERIES if (arch, width) in available]
        if handles:
            fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.53, .075), ncol=5,
                       frameon=False, fontsize=8.4, handlelength=2)
        fig.text(.53, .025, "Seed points; bands = sample SD across seeds 1, 2, 42", ha="center", fontsize=8, color="#4C566A")
        stem = f"detection_{mode}"
        for extension in ("png", "svg", "pdf"):
            path = output / f"{stem}.{extension}"
            fig.savefig(path, dpi=220, facecolor="white")
            outputs.append(path.name)
        plt.close(fig)
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/backtracking/results"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--require-complete", action="store_true", help="Refuse output unless all 15 cells exist.")
    parser.add_argument("--no-plots", action="store_true", help="Write only CSV, JSON and caption notes.")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    require(root.is_dir(), f"result root does not exist: {root}")
    cells, skipped = collect(root)
    expected = {cell_id(a, w, s) for a, w in SERIES for s in SEEDS}
    found = {cell_id(c["result"]["arch"], c["result"]["d_sae"], c["result"]["seed"]) for c in cells}
    missing = sorted(expected - found)
    if args.require_complete:
        require(not missing, "campaign is incomplete; missing: " + ", ".join(missing))
    rows = long_rows(cells)
    aggregate = summaries(rows)
    output = args.output_dir.resolve() if args.output_dir else root / "publication" / "detection"
    output.mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        with (output / f"detection_{mode}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(row for row in rows if row["probe_mode"] == mode)
    warning_count = sum(c["result"]["convergence_warning_count"] for c in cells)
    figures = []
    if cells and not args.no_plots:
        if args.require_complete:
            require(warning_count == 0, "primary probe convergence warnings block paper-ready export")
        if not missing and warning_count == 0:
            from paper_figures import render
            figures = render(aggregate, output, root=root, cells=cells)
        else:
            draft = output / "draft"
            draft.mkdir(exist_ok=True)
            figures = ["draft/" + name for name in plot_preview(
                aggregate, draft, completed=len(cells), warning_count=warning_count)]
    evaluator_hashes = sorted({c["result"]["provenance"]["evaluator_sha256"] for c in cells})
    manifest = {"schema": "c7-camera-ready-detection-summary-v3", "detection_protocol": PROTOCOL,
                "grouping": GROUPING, "n_prompt_groups": 213, "n_original_question_ids": 300,
                "status": "partial" if missing else "complete_with_warnings" if warning_count else "complete",
                "source_root": str(root), "expected_cells": len(expected), "completed_cells": len(cells),
                "completed_core_32k_cells": sum(c["result"]["d_sae"] == 32768 for c in cells),
                "completed_tsae_16k_sensitivity_cells": sum(c["result"]["d_sae"] == 16384 for c in cells),
                "missing_cells": missing, "skipped": skipped, "primary_metric": METRIC, "primary_S": 8,
                "probe_modes_kept_separate": list(MODES), "seed_order": list(SEEDS), "S_grid": list(S_GRID),
                "uncertainty": "Sample SD across dictionary seeds; only complete 1/2/42 series get a mean/SD. No CI or paired bootstrap computed.",
                "convergence_warning_count": warning_count, "evaluator_sha256_values": evaluator_hashes,
                "review_notes": ["Multiple evaluator source hashes: verify semantic compatibility before publication."] if len(evaluator_hashes) > 1 else [],
                "figures": figures, "series": aggregate,
                "sources": [{"path": c["source_path"], "sha256": c["source_sha256"],
                             "cell": cell_id(c["result"]["arch"], c["result"]["d_sae"], c["result"]["seed"]),
                             "status": c["result"]["status"]} for c in cells]}
    (output / "summary.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    notes = f"""# Backtracking detection source tables and paper figures

Status: **{manifest['status']}**, {len(cells)}/15 completed cells. The 12 core cells use width 32K; the three T-SAE 16K cells are a separate sensitivity analysis. Missing cells are listed in `summary.json`. No unfinished or smoke dictionary contributes a plotted point.

`detection_headline_matched_scaled_C1.pdf` is the main S=8 figure, sized to the existing NeurIPS half-width slot (2.585 x 1.98 inches). `detection_curves_matched_scaled_C1.pdf` gives all core readout curves at the full 5.5-inch text width; the T-SAE width sensitivity is separate. Raw-C1 versions are sensitivity figures. PDF is the manuscript format; SVG and 400-dpi PNG exports accompany it. `figure_manifest.json` records physical size, minimum font size, source/reference hashes and export checksums. `CAPTIONS.md` contains caption text and caveats, and `include_figures.tex` shows the exact PDF inclusions. The manuscript is unchanged.

All displayed results use mean five-fold average precision, 213 canonical normalized-prompt groups, and dictionaries trained for 300K steps. Seed uncertainty is sample SD across seeds 1, 2, 42, not a confidence interval. Shared-encoder position concatenation, when evaluated, is retained separately in the CSV; it is not independent Stacked SAE. Original 300-ID probes and old T-SAE encoding results remain excluded diagnostics in source JSONs. Paired conditional prompt-bootstrap intervals are computed separately by `paired_detection.py` from saved OOF predictions.

Primary convergence warnings: {warning_count}. Complete paper exports require zero primary warnings, all 15 cells, complete seed sets, compatible evaluator identities, and a hash-verified OOF file for the fold-prevalence reference. Historical diagnostic warnings remain recorded separately in the source data. Repaired probes, if any, retain their original JSON/OOF files and a numerical-refit receipt. Partial or unconverged results produce labeled previews only in `draft/`; `--require-complete` refuses such paper exports. With `--no-plots`, only tables/manifests are refreshed.

Regenerate with `python summarize_detection.py --root /workspace/backtracking/results --require-complete`. For local regeneration, the compact detection JSONs and the TXC seed-1 OOF sidecar suffice; no model weights or activation caches are required. An empty `figures` list means this invocation produced no figures, even if a reused directory contains older files.
"""
    (output / "README.md").write_text(notes)
    print(json.dumps({"status": manifest["status"], "completed_cells": len(cells),
                      "missing_cells": len(missing), "rows": len(rows), "output_dir": str(output),
                      "figures": figures}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
