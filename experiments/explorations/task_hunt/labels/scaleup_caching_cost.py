"""Caching-cost table for the scaled corpora (corpus-scaleup deliverable).

    .venv/bin/python -m experiments.explorations.task_hunt.labels.scaleup_caching_cost

Derived entirely from the committed stats files so the numbers cannot
drift from the artifacts: per scaled corpus and per screen tokenizer,
how many tokens a GPU pod must run, how many are ALREADY cached (the
fineweb prefix receipt — the 4,000-doc pull is a deterministic superset
of the pinned 400, token-for-token), and the multiple of the shipped
stream that implies.

Deliberately reports TOKENS and RATIOS, not gigabytes: cache footprint
per token is a function of the pod's layer selection and dtype, and the
honest bridge is "this stream is N x the one you already cached".

Writes ``scaleup_caching_cost.json`` and prints the markdown table.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEYS = ("gpt2", "gemma2", "llama31")


def main():
    p4 = json.loads((HERE / "punctint4k_stats.json").read_text())
    r2 = json.loads((HERE / "refmark2k_stats.json").read_text())
    r4 = json.loads((HERE / "refmark_stats.json").read_text())

    rows = []
    for key in KEYS:
        t = p4["per_tokenizer"][key]
        rows.append({
            "corpus": "fineweb4k (punctint, 4,000 docs)", "tokenizer": key,
            "tokens": t["n_tokens"],
            "already_cached": (t["replag_prefix_tokens"]
                               if t["token_ids_prefix_matches_replag"] else 0),
            "new_tokens": t["new_tokens_needing_cache"],
            "shipped_stream_tokens": t["replag_prefix_tokens"],
            "cache_source": ("replag_fineweb_%s.npz (prefix identity "
                             "CONFIRMED)" % key)
            if t["token_ids_prefix_matches_replag"] else "none",
        })
    for key in KEYS:
        t = r2["per_tokenizer"][key]
        shipped = r4["per_tokenizer"][key]["n_tokens"]
        rows.append({
            "corpus": "refmark2k (WildChat, 2,000 convs)", "tokenizer": key,
            "tokens": t["n_tokens"], "already_cached": 0,
            "new_tokens": t["n_tokens"],
            "shipped_stream_tokens": shipped,
            "cache_source": "none — a pool subsample redraws (only 121 of "
                            "the shipped 400 conversations recur)",
        })
    for r in rows:
        r["x_shipped_stream"] = round(
            r["tokens"] / max(r["shipped_stream_tokens"], 1), 2)

    out = {
        "note": "tokens per model, not gigabytes: footprint per token "
                "depends on the pod's layer selection and dtype",
        "totals": {
            "fineweb4k_new_tokens_3_models": sum(
                r["new_tokens"] for r in rows if r["corpus"].startswith(
                    "fineweb4k")),
            "refmark2k_new_tokens_3_models": sum(
                r["new_tokens"] for r in rows if r["corpus"].startswith(
                    "refmark2k")),
        },
        "rows": rows,
    }
    out["totals"]["all_new_tokens_3_models"] = (
        out["totals"]["fineweb4k_new_tokens_3_models"]
        + out["totals"]["refmark2k_new_tokens_3_models"])
    (HERE / "scaleup_caching_cost.json").write_text(json.dumps(out, indent=1))

    print("| corpus | tokenizer | tokens | already cached | NEW to cache |"
          " x shipped stream |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['corpus']} | {r['tokenizer']} | {r['tokens']:,} | "
              f"{r['already_cached']:,} | {r['new_tokens']:,} | "
              f"{r['x_shipped_stream']}x |")
    t = out["totals"]
    print(f"\ntotal NEW tokens across 3 models: fineweb4k "
          f"{t['fineweb4k_new_tokens_3_models']:,} + refmark2k "
          f"{t['refmark2k_new_tokens_3_models']:,} = "
          f"{t['all_new_tokens_3_models']:,}")


if __name__ == "__main__":
    main()
