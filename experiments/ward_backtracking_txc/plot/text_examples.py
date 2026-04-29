"""Side-by-side completions for 4 hand-picked eval prompts.

Conditions: unsteered (mag=0) / DoM_base@+12 / TXC_pos0@+12 / TXC_union@+12.
Bolds wait/hmm tokens for skim. Output: a single markdown file in plots_dir.
"""

from __future__ import annotations
import argparse, json, re
from collections import defaultdict
from pathlib import Path

from experiments.ward_backtracking_txc.plot._common import load_cfg, plots_dir

KW_RE = re.compile(r"\b(wait|hmm)\b", re.IGNORECASE)


def _bold_kw(text: str) -> str:
    return KW_RE.sub(lambda m: f"**{m.group(0)}**", text)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-prompts", type=int, default=4)
    ap.add_argument("--magnitude", type=float, default=12.0)
    args = ap.parse_args(argv)
    cfg = load_cfg(args.config)
    out_dir = plots_dir(cfg)
    res_path = Path(cfg["paths"]["steering"])
    if not res_path.exists():
        print(f"[skip] no steering at {res_path}"); return
    obj = json.loads(res_path.read_text())
    rows = obj["rows"]

    # Index by (source, magnitude, prompt_id) -> text
    idx = {}
    for r in rows:
        idx[(r["source"], r["magnitude"], r["prompt_id"])] = r

    # Pick best TXC source: highest mean kw at +12 across all prompts.
    by_src_at_mag = defaultdict(list)
    for r in rows:
        if r["magnitude"] == args.magnitude and r["source"].startswith("txc_"):
            by_src_at_mag[r["source"]].append(r["keyword_rate"])
    best_pos0 = best_union = None; best_pos0_mean = best_union_mean = -1.0
    for s, kws in by_src_at_mag.items():
        m = sum(kws) / max(1, len(kws))
        if s.endswith("_pos0") and m > best_pos0_mean:
            best_pos0_mean = m; best_pos0 = s
        if s.endswith("_union") and m > best_union_mean:
            best_union_mean = m; best_union = s
    print(f"[best] pos0={best_pos0} ({best_pos0_mean:.4f}) | union={best_union} ({best_union_mean:.4f})")

    # Get prompts that have all conditions present.
    cond_sources = ["dom_base_union"]
    if best_pos0: cond_sources.append(best_pos0)
    if best_union: cond_sources.append(best_union)

    # All prompts at mag=0 use any source — pick "dom_base_union" mag=0 as the
    # baseline rendering.
    baseline_prompts = [r for r in rows if r["source"] == "dom_base_union" and r["magnitude"] == 0.0]
    baseline_prompts = sorted(baseline_prompts, key=lambda r: -r["keyword_rate"])  # most interesting first
    selected = []
    for r in baseline_prompts:
        if all((s, args.magnitude, r["prompt_id"]) in idx for s in cond_sources):
            selected.append(r["prompt_id"])
        if len(selected) >= args.n_prompts:
            break

    lines = ["# B1 — generated text examples", ""]
    lines.append(f"**Magnitude**: +{args.magnitude} for all steered conditions.")
    lines.append(f"**Best TXC pos0**: `{best_pos0}` (mean kw = {best_pos0_mean:.4f})")
    lines.append(f"**Best TXC union**: `{best_union}` (mean kw = {best_union_mean:.4f})")
    lines.append("")

    for pid in selected:
        baseline_r = idx.get(("dom_base_union", 0.0, pid))
        if not baseline_r:
            continue
        lines.append(f"## prompt: `{pid}` ({baseline_r['category']})")
        lines.append("")
        for cond_name, src_tag in [
            ("unsteered (mag=0)", ("dom_base_union", 0.0)),
            ("DoM_base@+12", ("dom_base_union", args.magnitude)),
        ] + ([("TXC_pos0@+12", (best_pos0, args.magnitude))] if best_pos0 else []) \
          + ([("TXC_union@+12", (best_union, args.magnitude))] if best_union else []):
            r = idx.get((src_tag[0], src_tag[1], pid))
            if not r: continue
            kw = r["keyword_rate"]; nw = r["n_words"]; wc = r["wait_count"]
            lines.append(f"### {cond_name} — kw={kw:.3f} | wait/hmm={wc} | words={nw}")
            txt = r["text"][:1500]
            lines.append("```\n" + _bold_kw(txt) + "\n```")
            lines.append("")
    out = out_dir / "text_examples.md"
    out.write_text("\n".join(lines))
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
