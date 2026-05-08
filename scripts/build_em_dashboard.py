#!/usr/bin/env python3
"""Build a self-contained `dashboard.html` to manually browse EM-FRA generations.

UI features:
  - Hookpoint dropdown (L24 ln1, L24 resid_pre, etc.)
  - Pathway dropdown (QK→OV / OV→OV / QK→QK / conventional additive — filtered by hookpoint)
  - Eval seed dropdown
  - Prompt dropdown
  - α slider (snaps to the 6 sweep values)
  - Left pane: Nura `baseline` method (no hook applied)
  - Right pane: the selected (method, seed, prompt, α) generation
  - GPT-4o alignment + coherence scores under each pane

Inputs are paths to qualitative JSONs (post-judging) — no server is required;
the data is inlined into the HTML so `file://` works.

Usage:
    python scripts/build_em_dashboard.py --out plots/2026-05-07_em_repl/dashboard.html

Reads (with reasonable defaults under the same plots dir):
  --nura-qualitative   plots/.../phase1_judged/qualitative_medical_L24_H38_k50.json
  --additive PATH ...  one or more `<label>=<hookpoint>=<qualitative_*.json>` entries
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from html import escape
from pathlib import Path


def parse_additive(s: str):
    parts = s.split("=", 2)
    if len(parts) != 3:
        raise SystemExit(f"--additive '{s}' must be label=hookpoint=path")
    return parts[0], parts[1], parts[2]


def index_qualitative(path: Path):
    """Return a list of generation dicts. Skips entries lacking gpt4o scores."""
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "examples" in data:
        data = data["examples"]
    return [r for r in data
            if "gpt4o_alignment" in r and "gpt4o_coherence" in r]


def collect_baseline(nura_examples):
    """{seed: {prompt: {text, align, coh}}} from Nura's `baseline` condition."""
    out = defaultdict(dict)
    for ex in nura_examples:
        if ex.get("condition") != "baseline":
            continue
        out[int(ex["seed"])][ex["prompt"]] = {
            "text": ex["response"],
            "align": ex["gpt4o_alignment"],
            "coh": ex["gpt4o_coherence"],
        }
    return {str(s): v for s, v in out.items()}


_PATHWAY_FROM_COND = {
    "qk_to_ov": "QK→OV",
    "ov_to_ov": "OV→OV",
    "qk_to_qk": "QK→QK",
}


def collect_nura_pathways(nura_examples):
    """Return list of methods for Nura's three FRA conditions at L24 ln1."""
    by_method = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for ex in nura_examples:
        cond = ex.get("condition", "")
        if cond.startswith("baseline"):
            continue
        # condition like "qk_to_ov_a0.5"
        for prefix in ("qk_to_ov", "ov_to_ov", "qk_to_qk"):
            if cond.startswith(prefix):
                seed = int(ex["seed"])
                scale = float(ex["scale"])
                by_method[prefix][seed][scale][ex["prompt"]] = {
                    "text": ex["response"],
                    "align": ex["gpt4o_alignment"],
                    "coh": ex["gpt4o_coherence"],
                }
                break

    methods = []
    for cond_key, label in _PATHWAY_FROM_COND.items():
        if cond_key not in by_method:
            continue
        per_seed = {}
        for seed, by_alpha in by_method[cond_key].items():
            per_seed[str(seed)] = {
                str(a): list(prompts.values())  # ordered later
                for a, prompts in sorted(by_alpha.items())
            }
            # actually we need per-prompt ordering, do later
        methods.append({
            "id": f"nura_{cond_key}",
            "hookpoint": "blocks.24.ln1.hook_normalized",
            "pathway": label,
            "label": f"Nura {label} @ L24 ln1",
            "raw": by_method[cond_key],  # we'll resolve to ordered prompts list
        })
    return methods


def collect_additive_method(label, hookpoint, examples):
    """Conventional additive method from a single qualitative file."""
    by_seed = defaultdict(lambda: defaultdict(dict))
    for ex in examples:
        seed = int(ex["seed"])
        scale = float(ex["scale"])
        by_seed[seed][scale][ex["prompt"]] = {
            "text": ex["response"],
            "align": ex["gpt4o_alignment"],
            "coh": ex["gpt4o_coherence"],
        }
    return {
        "id": f"additive_{label}",
        "hookpoint": hookpoint,
        "pathway": "conventional (additive)",
        "label": label,
        "raw": by_seed,
    }


def finalize(method, prompts):
    """Convert raw {seed:{alpha:{prompt:{}}}} into prompt-indexed lists."""
    cells = {}
    for seed, by_alpha in method["raw"].items():
        cells[str(seed)] = {}
        for alpha, by_prompt in sorted(by_alpha.items()):
            cells[str(seed)][f"{alpha}"] = [
                by_prompt.get(p, None) for p in prompts
            ]
    method["cells"] = cells
    method.pop("raw")
    return method


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>EM-FRA dashboard — medical (manual generation browser)</title>
<style>
  :root {
    --bg: #0d1117; --fg: #e6edf3; --muted: #8b949e;
    --accent: #58a6ff; --good: #7ee787; --bad: #ff7b72;
    --card: #161b22; --border: #30363d; --code-bg: #1c2128;
  }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 16px 28px; font-family: -apple-system, "SF Pro Text",
         Segoe UI, Helvetica, Arial, sans-serif;
         background: var(--bg); color: var(--fg); line-height: 1.5; }
  h1 { font-size: 18px; margin: 0 0 12px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }
  .controls { display: flex; gap: 16px; flex-wrap: wrap; align-items: end;
              padding: 12px 16px; background: var(--card); border: 1px solid var(--border);
              border-radius: 6px; margin-bottom: 12px; }
  .controls label { display: flex; flex-direction: column; gap: 4px; font-size: 12px; color: var(--muted); }
  select, input[type=range] { background: var(--code-bg); color: var(--fg); border: 1px solid var(--border);
                              border-radius: 4px; padding: 5px 8px; font-size: 13px; min-width: 160px; }
  .alpha-row { display: flex; gap: 10px; align-items: center; }
  .alpha-row span { font-family: "SF Mono", Menlo, monospace; min-width: 48px; text-align: right; }
  .panes { display: grid; gap: 14px; grid-template-columns: 1fr 1fr; }
  .pane { background: var(--card); border: 1px solid var(--border); border-radius: 6px; padding: 14px 18px; }
  .pane h2 { margin: 0 0 8px; font-size: 14px; color: var(--accent); display: flex; gap: 12px; align-items: baseline; }
  .pane h2 .meta { font-size: 11px; color: var(--muted); font-weight: normal; }
  pre.response { background: var(--code-bg); border: 1px solid var(--border); border-radius: 4px;
                 padding: 12px; max-height: 60vh; overflow-y: auto; white-space: pre-wrap;
                 word-wrap: break-word; font-family: "SF Mono", Menlo, monospace; font-size: 12.5px; }
  .scores { display: flex; gap: 18px; margin-top: 8px; font-size: 13px; font-family: "SF Mono", Menlo, monospace; }
  .scores .label { color: var(--muted); margin-right: 4px; }
  .scores .ok { color: var(--good); }
  .scores .bad { color: var(--bad); }
  .prompt-block { background: var(--card); border-left: 3px solid var(--accent);
                  padding: 10px 14px; margin-bottom: 12px; font-size: 13px; }
  .prompt-block .label { color: var(--muted); font-size: 11px; text-transform: uppercase;
                          letter-spacing: 0.06em; margin-bottom: 4px; }
  .footer { color: var(--muted); font-size: 11px; margin-top: 14px; border-top: 1px solid var(--border); padding-top: 8px; }
  .missing { color: var(--bad); font-style: italic; }
</style>
</head>
<body>

<h1>EM-FRA dashboard — medical EM (manual generation browser)</h1>

<div class="controls">
  <label>hookpoint
    <select id="hookpoint"></select>
  </label>
  <label>pathway
    <select id="pathway"></select>
  </label>
  <label>eval seed
    <select id="seed"></select>
  </label>
  <label>prompt
    <select id="prompt"></select>
  </label>
  <label>α
    <div class="alpha-row">
      <input type="range" id="alpha" min="0" max="5" step="1" value="2">
      <span id="alpha-val">1.0</span>
    </div>
  </label>
</div>

<div class="prompt-block">
  <div class="label">prompt text</div>
  <div id="prompt-text"></div>
</div>

<div class="panes">
  <div class="pane">
    <h2>unsteered <span class="meta" id="left-meta"></span></h2>
    <pre class="response" id="left-resp"></pre>
    <div class="scores" id="left-scores"></div>
  </div>
  <div class="pane">
    <h2>steered <span class="meta" id="right-meta"></span></h2>
    <pre class="response" id="right-resp"></pre>
    <div class="scores" id="right-scores"></div>
  </div>
</div>

<div class="footer">
  Dataset: 8 EM eval prompts (Wang et al. 2026) · model: Qwen2.5-14B-Instruct + medical LoRA · judge: GPT-4o ·
  unsteered = Nura's <code>baseline</code> method (no hook applied) at the matching eval seed/prompt
</div>

<script>
const DATA = __DATA_JSON__;

function $(id) { return document.getElementById(id); }
function fmt(x) { return (x === null || x === undefined) ? "—" : Number(x).toFixed(0); }
function colorScore(x) { if (x === null || x === undefined) return ""; return x >= 70 ? "ok" : (x <= 30 ? "bad" : ""); }

const hookpoints = [...new Set(DATA.methods.map(m => m.hookpoint))];
const seeds = DATA.seeds.map(String);
const promptIdx = DATA.prompts.map((p, i) => i);
const alphas = DATA.alphas;

// Populate dropdowns
function populate() {
  const hp = $("hookpoint");
  hp.innerHTML = hookpoints.map(h => `<option value="${h}">${h}</option>`).join("");
  $("seed").innerHTML = seeds.map(s => `<option value="${s}">${s}</option>`).join("");
  $("prompt").innerHTML = promptIdx.map(i =>
    `<option value="${i}">${i}: ${DATA.prompts[i].slice(0, 60)}${DATA.prompts[i].length > 60 ? "…" : ""}</option>`).join("");
  refreshPathway();
}

function refreshPathway() {
  const hp = $("hookpoint").value;
  const opts = DATA.methods.filter(m => m.hookpoint === hp).map(m =>
    `<option value="${m.id}">${m.pathway}</option>`).join("");
  $("pathway").innerHTML = opts;
  refresh();
}

function refresh() {
  const seed = $("seed").value;
  const pi = parseInt($("prompt").value, 10);
  const ai = parseInt($("alpha").value, 10);
  const alpha = alphas[ai].toFixed(1);
  $("alpha-val").textContent = alpha;
  $("prompt-text").textContent = DATA.prompts[pi];

  const m = DATA.methods.find(mm => mm.id === $("pathway").selectedOptions[0]?.parentNode &&
                                    false) || DATA.methods.find(mm => mm.id === $("pathway").value);
  // The above is hacky; just re-find by id below
  const methodId = $("pathway").value;
  const method = DATA.methods.find(mm => mm.id === methodId);
  if (!method) return;

  // Left: baseline
  const baseEntry = (DATA.baseline[seed] || [])[pi];
  $("left-meta").textContent = `seed=${seed} · prompt=${pi} · Nura baseline (no hook)`;
  $("left-resp").textContent = baseEntry ? baseEntry.text : "(missing)";
  $("left-scores").innerHTML = baseEntry ? renderScores(baseEntry) : `<span class="missing">no judge score</span>`;

  // Right: steered
  $("right-meta").textContent = `seed=${seed} · prompt=${pi} · ${method.pathway} · α=${alpha} · ${method.hookpoint}`;
  const cell = (method.cells[seed] || {})[alpha];
  const entry = cell ? cell[pi] : null;
  $("right-resp").textContent = entry ? entry.text : "(missing)";
  $("right-scores").innerHTML = entry ? renderScores(entry) : `<span class="missing">no entry / no judge score</span>`;
}

function renderScores(e) {
  return `<span><span class="label">align</span><span class="${colorScore(e.align)}">${fmt(e.align)}</span></span>` +
         `<span><span class="label">coh</span><span class="${colorScore(e.coh)}">${fmt(e.coh)}</span></span>`;
}

// Bind
$("hookpoint").addEventListener("change", refreshPathway);
$("pathway").addEventListener("change", refresh);
$("seed").addEventListener("change", refresh);
$("prompt").addEventListener("change", refresh);
$("alpha").addEventListener("input", refresh);

populate();
</script>
</body>
</html>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nura-qualitative", required=True,
                   help="path to Nura's qualitative_medical_*.json (post-judging)")
    p.add_argument("--additive", action="append", default=[],
                   help="<label>=<hookpoint>=<qualitative_*.json>; pass once per method")
    p.add_argument("--out", required=True, help="path to write dashboard.html")
    args = p.parse_args()

    nura = index_qualitative(Path(args.nura_qualitative))
    if not nura:
        raise SystemExit(f"no judged entries in {args.nura_qualitative}")

    # Discover prompts from Nura (canonical 8 prompts)
    prompts = []
    seen = set()
    for ex in nura:
        p_text = ex["prompt"]
        if p_text not in seen:
            seen.add(p_text)
            prompts.append(p_text)

    seeds = sorted({int(ex["seed"]) for ex in nura})
    alphas = sorted({float(ex["scale"]) for ex in nura
                     if ex.get("condition", "") != "baseline"})
    if not alphas:
        alphas = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

    baseline = collect_baseline(nura)
    # Convert baseline to per-seed prompt-list for easy indexing
    baseline_idx = {
        s: [baseline.get(s, {}).get(p) for p in prompts] for s in baseline
    }

    methods = collect_nura_pathways(nura)
    for entry in args.additive:
        label, hookpoint, path = parse_additive(entry)
        examples = index_qualitative(Path(path))
        if not examples:
            print(f"[skip] {entry}: no judged entries in {path}")
            continue
        m = collect_additive_method(label, hookpoint, examples)
        methods.append(m)

    for m in methods:
        finalize(m, prompts)

    payload = {
        "prompts": prompts,
        "seeds": seeds,
        "alphas": alphas,
        "baseline": baseline_idx,
        "methods": [
            {k: v for k, v in m.items() if k in ("id", "hookpoint", "pathway", "label", "cells")}
            for m in methods
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    html = HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(payload, ensure_ascii=False))
    out.write_text(html)
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB)")
    print(f"  methods: {len(methods)}  seeds: {seeds}  prompts: {len(prompts)}  alphas: {alphas}")


if __name__ == "__main__":
    main()
