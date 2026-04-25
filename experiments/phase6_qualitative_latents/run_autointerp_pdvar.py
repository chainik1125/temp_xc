"""Priority 2a: compute semantic_count_pdvar — the pdvar-ranked
analogue of the top-32 SEMANTIC count.

Loads existing labels JSONs, identifies features that are in pdvar-top-32
but NOT in var-top-32 (so unlabelled), labels + judges them via Haiku
temp=0, and computes semantic_count_pdvar across the full pdvar-top-32.

The output is written back to the same `{arch}__seed{S}__concat{C}__labels.json`
with two additional fields:
    metrics.semantic_count_pdvar     — int
    pdvar_only_features              — list of dicts (shape like `features`)
                                       but only for the pdvar-only slots.

Does NOT re-label features that are in var∩pdvar — those reuse the
existing labels from `features` directly.

Safe to re-run: if `semantic_count_pdvar` is already in metrics, the
cell is skipped unless `--force` is given.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LABELS_DIR = REPO / "experiments/phase6_qualitative_latents/results/autointerp"


def _load_existing(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_back(path: Path, cell: dict):
    path.write_text(json.dumps(cell, indent=2))


PRIMARY = [
    "tsae_paper",
    "agentic_txc_10_bare",
    "agentic_txc_12_bare_batchtopk",
    "agentic_txc_02",
    "agentic_txc_02_batchtopk",
]


def _primary_archs_seeds_concats(archs_override=None, seeds_override=None,
                                 concats_override=None) -> list[tuple[str, int, str]]:
    archs = archs_override if archs_override is not None else PRIMARY
    seeds = seeds_override if seeds_override is not None else [42, 1, 2]
    concats = concats_override if concats_override is not None else ["A", "B", "random"]
    return [(arch, s, c) for arch in archs for s in seeds for c in concats]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archs", type=str, nargs="+", default=None,
                   help="override primary-arch list (defaults to 5 paper archs)")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--concats", type=str, nargs="+", default=None)
    p.add_argument("--model", type=str, default="claude-haiku-4-5")
    p.add_argument("--force", action="store_true",
                   help="re-run even if metrics.semantic_count_pdvar already set")
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

    # Re-use tokenizer + existing helpers from run_autointerp.
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

    cells = _primary_archs_seeds_concats(
        archs_override=args.archs,
        seeds_override=args.seeds,
        concats_override=args.concats,
    )

    print(f"[pdvar] {len(cells)} cells to process")

    for arch, seed, concat in cells:
        labels_path = LABELS_DIR / f"{arch}__seed{seed}__concat{concat}__labels.json"
        if not labels_path.exists():
            print(f"[skip] {arch} seed={seed} {concat}: no labels json")
            continue
        cell = _load_existing(labels_path)
        if (not args.force
                and "semantic_count_pdvar" in cell.get("metrics", {})):
            print(f"[skip] {arch} seed={seed} {concat}: pdvar already computed "
                  f"(semantic_count_pdvar="
                  f"{cell['metrics']['semantic_count_pdvar']})")
            continue

        var_top = cell["top_feat_indices_var"]
        pdvar_top = cell["top_feat_indices_pdvar"]
        # Map feature_idx → label entry for reuse of var-top-32 features
        var_feats_by_idx = {f["feature_idx"]: f for f in cell["features"]}
        # Find pdvar-only feats
        var_set = set(var_top)
        pdvar_only = [i for i in pdvar_top if i not in var_set]

        print(f"[run] {arch} seed={seed} {concat}: "
              f"{len(pdvar_only)} pdvar-only features to label "
              f"(overlap={len(set(pdvar_top) & var_set)}/32)")

        # Need ids + z + passages to gather contexts.
        try:
            ids, z, _passages = _load_concat(concat, arch, seed)
        except FileNotFoundError:
            print(f"[skip] {arch} seed={seed} {concat}: no z cache")
            continue

        pdvar_only_features = []
        for fi in pdvar_only:
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
            pdvar_only_features.append({
                "feature_idx": int(fi),
                "label": label,
                "variance": float(z[:, fi].var()),
                "n_contexts": len(ctxs),
                "contexts": ctxs[:3],
                "judge": judge,
            })

        # Compute semantic_count_pdvar over the FULL pdvar-top-32:
        # reuse var-top labels for overlap, new labels for pdvar-only.
        pdvar_only_by_idx = {f["feature_idx"]: f for f in pdvar_only_features}
        sem_pdvar = 0
        for i in pdvar_top:
            if i in var_feats_by_idx:
                verdict = var_feats_by_idx[i]["judge"]["verdict"]
            elif i in pdvar_only_by_idx:
                verdict = pdvar_only_by_idx[i]["judge"]["verdict"]
            else:
                continue
            if verdict == "SEMANTIC":
                sem_pdvar += 1

        cell.setdefault("metrics", {})
        cell["metrics"]["semantic_count_pdvar"] = sem_pdvar
        cell["metrics"]["pdvar_new_labels_count"] = len(pdvar_only_features)
        cell["pdvar_only_features"] = pdvar_only_features
        _write_back(labels_path, cell)
        print(f"[done] {arch} seed={seed} {concat}  "
              f"sem_pdvar={sem_pdvar}/32  "
              f"(was sem_var={cell['metrics']['semantic_count']}/32)")


if __name__ == "__main__":
    main()
