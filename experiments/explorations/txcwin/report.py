"""Render the TXC-win report: raw gate, dictionary panels, audit, figures.

Everything is read from results/*.json. The narrative order is deliberately the
order the work actually happened in, including the false positive that was caught
and retracted, because that sequence is the reason the final numbers can be
trusted.

Run:  python3 -m experiments.explorations.txcwin.report
"""

from __future__ import annotations

import base64
import html
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGS = HERE / "figs"
OUT = HERE / "report.html"

NICE = {
    "batchtopk_sae": "per-token SAE",
    "tsae": "T-SAE",
    "stacked_batchtopk": "Stacked SAE",
    "txc_batchtopk_pre": "TXC-pre (adds positions)",
    "txc_batchtopk_post": "TXC-post — the paper's TXC (mixes positions)",
}
ORDER = ["batchtopk_sae", "tsae", "stacked_batchtopk", "txc_batchtopk_pre",
         "txc_batchtopk_post"]
PER_TOKEN = {"batchtopk_sae", "tsae"}

CSS = """
<style>
:root{--bg:#FBFBFA;--card:#fff;--card2:#F1F3F5;--ink:#14181B;--ink2:#48555C;
 --ink3:#7A8890;--line:#DEE3E6;--line2:#EDF0F2;--a:#2a78d6;--b:#eb6834;
 --c:#1baf7a;--d:#4a3aa7;--warn:#8a6512;--bad:#b23a2f;--good:#1f7a4d;
 --mono:ui-monospace,SFMono-Regular,Menlo,monospace;
 --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
 --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
@media (prefers-color-scheme:dark){:root{--bg:#0D1215;--card:#141C21;
 --card2:#1B252B;--ink:#E9EFF2;--ink2:#A8B9C1;--ink3:#78888F;--line:#243138;
 --line2:#1C262C;--a:#3987e5;--b:#d95926;--c:#199e70;--d:#9085e9;--warn:#d9ac4a;
 --bad:#e0776b;--good:#5fc183}}
:root[data-theme="dark"]{--bg:#0D1215;--card:#141C21;--card2:#1B252B;
 --ink:#E9EFF2;--ink2:#A8B9C1;--ink3:#78888F;--line:#243138;--line2:#1C262C;
 --a:#3987e5;--b:#d95926;--c:#199e70;--d:#9085e9;--warn:#d9ac4a;--bad:#e0776b;
 --good:#5fc183}
:root[data-theme="light"]{--bg:#FBFBFA;--card:#fff;--card2:#F1F3F5;--ink:#14181B;
 --ink2:#48555C;--ink3:#7A8890;--line:#DEE3E6;--line2:#EDF0F2;--a:#2a78d6;
 --b:#eb6834;--c:#1baf7a;--d:#4a3aa7;--warn:#8a6512;--bad:#b23a2f;--good:#1f7a4d}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
 font-size:16px;line-height:1.65}
.wrap{max-width:840px;margin:0 auto;padding:40px 22px 120px;display:flex;
 flex-direction:column;gap:44px}
.wide{max-width:1140px;margin-left:calc((840px - 1140px)/2)}
@media(max-width:1200px){.wide{max-width:100%;margin-left:0}}
h1,h2,h3{font-family:var(--serif);font-weight:600;margin:0;text-wrap:balance}
h1{font-size:clamp(30px,4.6vw,44px);line-height:1.1}
h2{font-size:26px}h3{font-size:18px}p{margin:0}
section{display:flex;flex-direction:column;gap:18px}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.14em;
 text-transform:uppercase;color:var(--ink3)}
.lede{font-size:19px;line-height:1.55;color:var(--ink2)}
.card{background:var(--card);border:1px solid var(--line);border-radius:3px;
 padding:20px 22px;display:flex;flex-direction:column;gap:12px}
.card.quiet{background:var(--card2);border-color:var(--line2)}
.card.warn{border-left:3px solid var(--warn)}
.card.bad{border-left:3px solid var(--bad)}
.card.good{border-left:3px solid var(--good)}
figure{margin:0;display:flex;flex-direction:column;gap:10px}
figure img{width:100%;height:auto;border:1px solid var(--line);border-radius:3px}
figcaption{font-size:14px;color:var(--ink2)}
.fig-dark{display:none}
@media (prefers-color-scheme:dark){.fig-light{display:none}.fig-dark{display:block}}
:root[data-theme="dark"] .fig-light{display:none}
:root[data-theme="dark"] .fig-dark{display:block}
:root[data-theme="light"] .fig-light{display:block}
:root[data-theme="light"] .fig-dark{display:none}
.tscroll{overflow-x:auto;border:1px solid var(--line);border-radius:3px;
 background:var(--card)}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;padding:9px 12px;border-bottom:1px solid var(--line2)}
th{font-family:var(--mono);font-size:10.5px;letter-spacing:.08em;
 text-transform:uppercase;color:var(--ink3);font-weight:400;background:var(--card2);
 white-space:nowrap}
tr:last-child td{border-bottom:0}
td.n{font-family:var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}
tr.hi td{background:color-mix(in oklab,var(--d) 12%,transparent);font-weight:600}
ul,ol{margin:0;padding-left:22px;display:flex;flex-direction:column;gap:9px}
li{color:var(--ink2)}li b,p b{color:var(--ink)}
pre{font-family:var(--mono);font-size:12px;line-height:1.55;background:var(--card2);
 border:1px solid var(--line2);border-radius:3px;padding:12px 14px;overflow-x:auto;
 margin:0}
.kv{display:grid;grid-template-columns:180px 1fr;gap:7px 18px;font-size:14.5px}
.kv dt{font-family:var(--mono);font-size:12px;color:var(--ink3);
 text-transform:uppercase;letter-spacing:.06em;padding-top:3px}
.kv dd{margin:0;color:var(--ink2)}
.big{font-family:var(--serif);font-size:38px;line-height:1;color:var(--d)}
.tag{display:inline-block;font-family:var(--mono);font-size:10.5px;
 letter-spacing:.08em;text-transform:uppercase;padding:2px 7px;border-radius:2px;
 border:1px solid currentColor}
.tag.good{color:var(--good)}.tag.bad{color:var(--bad)}.tag.warn{color:var(--warn)}
.foot{font-family:var(--mono);font-size:12px;color:var(--ink3)}
</style>
"""


def esc(s):
    return html.escape(str(s), quote=True)


def img(stem, alt=""):
    out = []
    for mode, cls in (("light", "fig-light"), ("dark", "fig-dark")):
        p = FIGS / f"{stem}_{mode}.png"
        if p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode()
            out.append(f'<img class="{cls}" src="data:image/png;base64,{b64}" '
                       f'alt="{esc(alt)}">')
    return "".join(out)


def jload(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def agg(pl, arch, T, trained=True):
    cs = [c for c in pl["cells"] if c["arch"] == arch and c["T"] == T
          and bool(c.get("trained", True)) == trained]
    if not cs:
        return None
    sk = [c["skill"] for c in cs]
    n = len(sk)
    m = sum(sk) / n
    var = sum((x - m) ** 2 for x in sk) / max(1, n - 1)
    return {"mean": m, "sd": var ** 0.5, "se": (var / n) ** 0.5 if n > 1 else 0,
            "min": min(sk), "max": max(sk), "n": n,
            "l0": sum(c["l0"] for c in cs) / n}


def panel_table(pl, raw=None):
    """One row per (arch, T) with seed mean, spread, init, and learning."""
    Ts = sorted({c["T"] for c in pl["cells"]})
    rows = []
    for a in ORDER:
        for T in Ts:
            g = agg(pl, a, T)
            if not g:
                continue
            u = agg(pl, a, T, trained=False)
            rows.append((a, T, g, u))
    best = max((r for r in rows if r[0] not in PER_TOKEN and r[1] >= 2),
               key=lambda r: r[2]["mean"], default=None)
    h = ("<div class='tscroll'><table><thead><tr><th>architecture</th><th>T</th>"
         "<th>skill (mean of seeds)</th><th>seed range</th><th>untrained</th>"
         "<th>learned by training</th><th>code budget</th></tr></thead><tbody>")
    for a, T, g, u in rows:
        cls = " class='hi'" if best and (a, T) == (best[0], best[1]) else ""
        h += (f"<tr{cls}><td>{esc(NICE[a])}</td><td class='n'>{T}</td>"
              f"<td class='n'>{g['mean']:+.3f} ± {g['sd']:.3f}</td>"
              f"<td class='n'>{g['min']:+.3f} … {g['max']:+.3f}</td>"
              f"<td class='n'>{u['mean']:+.3f}</td>"
              f"<td class='n'>{g['mean'] - u['mean']:+.3f}</td>"
              f"<td class='n'>{g['l0']:.1f}</td></tr>")
    h += "</tbody></table></div>"
    return h, rows, best


def build():
    p = []
    a = p.append
    rg = jload("rawgate_gpt2_L6.json")
    fsw = jload("focus_switch_nnz.json")
    fnov = jload("focus_novresid.json")
    f8b = jload("focus_switch_8b.json")

    a("<title>Where a temporal dictionary actually wins — and one false positive "
      "caught on the way</title>")
    a(CSS)
    a('<div class="wrap">')

    # header
    a("<section>")
    a('<div class="eyebrow">TempBench · overnight run · 26 July 2026</div>')
    a("<h1>Hunting a task where reading a window of tokens beats reading one</h1>")
    a('<p class="lede">Dictionaries were <b>trained</b> this time — the paper\'s '
      "own five architectures, at matched code budget, with three seeds and an "
      "untrained control for every cell. The headline finding of the first few "
      "hours turned out to be an artefact, was caught by a check, and was "
      "retracted before it reached this page. What survived is smaller and "
      "real.</p>")
    a("</section>")

    # the retraction, up front
    a("<section>")
    a('<div class="eyebrow">The false positive, first</div>')
    a("<h2>What I nearly reported</h2>")
    a('<div class="card bad">')
    a("<p><b>The claim that failed.</b> On <i>tokens since the text switched "
      "source document</i>, the paper's TXC at T=8 scored 0.294 against 0.139 for "
      "the per-token SAE — a 6σ gap over three seeds, beating T-SAE and Stacked "
      "SAE too, at matched code budget. Every automatic check passed.</p>")
    a("<p><b>Why it was wrong.</b> A linear probe on the <i>raw activation at the "
      "single label position</i> — no dictionary at all — reaches <b>r = 0.31–0.37</b> "
      "on that task. Higher than any trained dictionary. So the signal was never "
      "non-local; the per-token dictionary simply discarded more of it. The "
      "comparison was real but it measures <b>code efficiency at matched "
      "sparsity</b>, not temporal structure, and presenting it as the latter would "
      "have been the central claim of the paper resting on a mislabelled "
      "quantity.</p>")
    a("<p><b>What changed as a result.</b> A raw-activation gate now runs before "
      "any dictionary training, and a task only qualifies if a "
      "<i>dimension-matched</i> window average beats a single position on the raw "
      "activations. That test is cheap and it reorders the candidate list "
      "completely.</p>")
    a("</div>")
    a("</section>")

    # raw gate
    if rg:
        a('<section class="wide">')
        a('<div class="eyebrow">The gate that decides what is worth training</div>')
        a("<h2>Which labels actually need more than one position?</h2>")
        a("<p>All three probes below run on the same rows, on raw activations. "
          "<b>Window-mean</b> is the honest comparison to <b>single position</b>: "
          "it has the same number of features, so neither is favoured by "
          "dimensionality. The flattened window is shown too, and it is worse "
          "everywhere — a fixed-budget probe on T×768 features degrades as T "
          "grows, which is exactly the trap that makes naive window-versus-token "
          "comparisons unreliable.</p>")
        cells = sorted(rg["cells"], key=lambda c: -(c["gap_mean_minus_last"]))
        a("<div class='tscroll'><table><thead><tr><th>candidate task</th><th>T</th>"
          "<th>single position</th><th>window mean</th><th>window flattened</th>"
          "<th>gain from the window</th><th>verdict</th></tr></thead><tbody>")
        for c in cells:
            gap = c["gap_mean_minus_last"]
            live = gap > 0.03
            cls = " class='hi'" if gap > 0.09 else ""
            a(f"<tr{cls}><td>{esc(c['desc'])}</td><td class='n'>{c['T']}</td>"
              f"<td class='n'>{c['raw_last']['skill']:+.3f}</td>"
              f"<td class='n'>{c['raw_mean']['skill']:+.3f}</td>"
              f"<td class='n'>{c['raw_window']['skill']:+.3f}</td>"
              f"<td class='n'>{gap:+.3f}</td>"
              f"<td><span class='tag {'good' if live else 'bad'}'>"
              f"{'needs a window' if live else 'one position is enough'}"
              f"</span></td></tr>")
        a("</tbody></table></div>")
        a('<div class="card quiet"><p>The pattern is consistent and it matches '
          "what the earlier relational work predicted: <b>trailing rates</b> — "
          "quantities accumulated over many tokens — genuinely need a window, and "
          "the gain grows with window size. <b>Clocks</b> (tokens since the last "
          "switch, tokens since the last speaker change) do not: the model already "
          "carries them on the current token.</p></div>")
        a("</section>")

    # the real panel
    if fnov:
        a('<section class="wide">')
        a('<div class="eyebrow">The result that survived</div>')
        a(f"<h2>Trained dictionaries on a task that passes the gate</h2>")
        a(f'<p class="lede">{esc(fnov["meta"]["desc"])} — '
          f"{esc(fnov['meta']['model'])} layer {fnov['meta']['layer']}, "
          f"{fnov['meta']['steps']} steps, {len(fnov['meta']['seeds'])} seeds, "
          f"matched code budget.</p>")
        tbl, rows, best = panel_table(fnov)
        a(tbl)
        if best:
            b_a, b_T, b_g, b_u = best
            base = max((g for arch, T, g, u in rows if arch in PER_TOKEN),
                       key=lambda g: g["mean"])
            se = (b_g["se"] ** 2 + base["se"] ** 2) ** 0.5
            z = (b_g["mean"] - base["mean"]) / se if se else float("nan")
            a('<div class="card good">')
            a(f'<div class="big">{b_g["mean"]:+.3f}</div>')
            a(f"<p><b>{esc(NICE[b_a])} at T={b_T}</b> against "
              f"<b>{base['mean']:+.3f}</b> for the best per-token dictionary — "
              f"a gap of {b_g['mean'] - base['mean']:+.3f} "
              f"({z:.1f}σ over seeds). Worst winning seed {b_g['min']:+.3f} "
              f"versus best baseline seed {base['max']:+.3f}. Training added "
              f"{b_g['mean'] - b_u['mean']:+.3f} over random initialisation.</p>")
            a("</div>")
        for stem, cap in [("money_focus_novresid",
                           "Skill against window size. Dots are individual seeds; "
                           "the band is ±1 seed standard deviation; dashed lines "
                           "are the same architecture untrained; the dotted "
                           "horizontal band is the per-token baseline."),
                          ("seeds_focus_novresid",
                           "Every seed, shown. Hollow dots are the untrained "
                           "control for the same cell."),
                          ("gain_focus_novresid",
                           "How much of each score is the dictionary rather than "
                           "the random projection it started as.")]:
            if (FIGS / f"{stem}_light.png").exists():
                a(f"<figure>{img(stem)}<figcaption>{esc(cap)}</figcaption></figure>")
        a("</section>")

    # switch clock panel, demoted
    if fsw:
        a('<section class="wide">')
        a('<div class="eyebrow">The retracted task, kept for the record</div>')
        a("<h2>The switch-clock panel — a code-efficiency result, not a temporal one</h2>")
        tbl, rows, best = panel_table(fsw)
        a(tbl)
        a('<div class="card warn"><p>These numbers are reproducible and the '
          "architecture ordering is real. What they cannot support is the claim "
          "that a window exposes something a single position lacks, because on "
          "this task the raw single position scores higher than every dictionary "
          "here. Read as a statement about how much signal each code retains at "
          "equal sparsity, the ordering is still interesting — TXC-post retains "
          "roughly twice what the per-token SAE does — but that is a different "
          "claim and a weaker one.</p></div>")
        if (FIGS / "money_focus_switch_nnz_light.png").exists():
            a(f"<figure>{img('money_focus_switch_nnz')}<figcaption>"
              "The retracted panel. Note that every window architecture, "
              "<i>including its untrained control</i>, collapses at T=4 — an "
              "anomaly that is still unexplained and is reported rather than "
              "smoothed.</figcaption></figure>")
        a("</section>")

    if f8b:
        a('<section class="wide">')
        a('<div class="eyebrow">Second model</div>')
        a("<h2>The paper's own subject model</h2>")
        a(f"<p>{esc(f8b['meta']['model'])} layer {f8b['meta']['layer']}, "
          f"d={f8b['meta'].get('d','?')}, {f8b['meta']['steps']} steps.</p>")
        tbl, _, _ = panel_table(f8b)
        a(tbl)
        a("</section>")

    # audit
    a('<section class="wide">')
    a('<div class="eyebrow">Self-audit</div>')
    a("<h2>Every claim re-derived from the raw results by code</h2>")
    a("<p>Claims are written to <code>claims.jsonl</code> as structured "
      "assertions; <code>audit.py</code> recomputes each one from the result "
      "files and reports contradictions. It has already caught two problems in my "
      "own reasoning: a mis-specified significance test that wrongly killed a real "
      "result, and the seed-selection loophole that the worst-versus-best check "
      "now closes.</p>")
    try:
        txt = subprocess.run(
            ["python3", "-m", "experiments.explorations.txcwin.audit",
             "--pattern", "focus_*.json"], capture_output=True, text=True,
            cwd=str(HERE.parents[2]), timeout=120).stdout
    except Exception as e:
        txt = f"(audit did not run: {e})"
    a(f"<pre>{esc(txt.strip() or '(no output)')}</pre>")
    a("</section>")

    # limits
    a("<section>")
    a('<div class="eyebrow">Limits</div>')
    a("<h2>What this still is not</h2>")
    a("<ul>")
    a("<li><b>A triage harness, not the canonical pipeline.</b> Training happens "
      "in-process rather than through the project's canonical runner, so none of "
      "these numbers are on the leaderboard and none carry a code-version stamp. "
      "Whatever survives has to be re-run properly before it is a result.</li>")
    a("<li><b>Small dictionaries.</b> d_sae = 2048 with k = 20 on gpt2, against "
      "the paper's 18432 at k = 20. Expansion factor is the single most likely "
      "thing to change the ordering.</li>")
    a("<li><b>One layer, one corpus per task.</b> Layer 6 of gpt2 on a pinned "
      "400-document fineweb sample and a 5,000-dialogue sample.</li>")
    a("<li><b>The T=4 collapse is unexplained.</b> It affects every window "
      "architecture and their untrained controls, so it is a property of the rows "
      "or the window construction rather than of training, and I have not found "
      "it.</li>")
    a("<li><b>MLC and TFA are absent.</b> MLC needs several layers cached at once; "
      "TFA is not in this repo's registry. Both are in the paper's panel.</li>")
    a("</ul>")
    a("</section>")

    a(f'<p class="foot">Generated by experiments/explorations/txcwin/report.py '
      f"from results/*.json.</p>")
    a("</div>")
    return "\n".join(p)


def main():
    OUT.write_text(build())
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
