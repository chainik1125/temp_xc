"""Render the live run dashboard from `state.json` → self-contained HTML.

Single-source convention (the program's pattern): every number on the page
comes from `state.json` or a results JSON. Nothing is hand-typed in HTML.
Figures are embedded as base64 data URIs so the page is self-contained
(the Artifact CSP blocks every external host).

Run:  python3 -m experiments.explorations.relational.dashboard.render
      (stdlib only — no venv required)
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE = HERE / "state.json"
OUT = HERE / "index.html"

STATE_COLORS = {
    "done": "pass",
    "running": "live",
    "queued": "idle",
    "blocked": "idle",
    "killed": "kill",
    "watch": "watch",
}

# ── tokens ──────────────────────────────────────────────────────────────
CSS = """
<style>
:root{
  --ground:#F6F7F9; --panel:#FFFFFF; --panel-2:#EFF2F4;
  --ink:#111A21; --ink-2:#41545F; --ink-3:#6E828D;
  --rule:#D6DEE3; --rule-2:#E7EDF0;
  --accent:#0F7F76; --accent-soft:#D9EDEA;
  --pass:#2F7D4C; --watch:#9A6B12; --kill:#A5382F; --theory:#6A5A9C;
  --live:#0F7F76;
  --mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
  --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
}
@media (prefers-color-scheme:dark){
  :root{
    --ground:#0B1016; --panel:#111A21; --panel-2:#16222B;
    --ink:#E8EEF1; --ink-2:#A7BAC4; --ink-3:#778D99;
    --rule:#22323D; --rule-2:#1B2831;
    --accent:#4FD1C0; --accent-soft:#15302E;
    --pass:#5FC183; --watch:#D9AC4A; --kill:#E0776B; --theory:#A99AD8;
    --live:#4FD1C0;
  }
}
:root[data-theme="dark"]{
  --ground:#0B1016; --panel:#111A21; --panel-2:#16222B;
  --ink:#E8EEF1; --ink-2:#A7BAC4; --ink-3:#778D99;
  --rule:#22323D; --rule-2:#1B2831;
  --accent:#4FD1C0; --accent-soft:#15302E;
  --pass:#5FC183; --watch:#D9AC4A; --kill:#E0776B; --theory:#A99AD8;
  --live:#4FD1C0;
}
:root[data-theme="light"]{
  --ground:#F6F7F9; --panel:#FFFFFF; --panel-2:#EFF2F4;
  --ink:#111A21; --ink-2:#41545F; --ink-3:#6E828D;
  --rule:#D6DEE3; --rule-2:#E7EDF0;
  --accent:#0F7F76; --accent-soft:#D9EDEA;
  --pass:#2F7D4C; --watch:#9A6B12; --kill:#A5382F; --theory:#6A5A9C;
  --live:#0F7F76;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);
  font-size:15px;line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:32px 20px 96px}
h1,h2,h3{font-family:var(--serif);font-weight:600;text-wrap:balance;margin:0}
h1{font-size:clamp(28px,4.4vw,42px);line-height:1.12;letter-spacing:-.01em}
h2{font-size:22px;line-height:1.25}
h3{font-size:16px;line-height:1.3}
p{margin:0}
a{color:var(--accent)}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.12em;
  text-transform:uppercase;color:var(--ink-3)}
.sub{color:var(--ink-2);max-width:66ch}
.stack{display:flex;flex-direction:column}
.g8{gap:8px}.g12{gap:12px}.g16{gap:16px}.g24{gap:24px}.g40{gap:40px}
section{display:flex;flex-direction:column;gap:16px}
.mono{font-family:var(--mono);font-variant-numeric:tabular-nums}
.num{font-family:var(--mono);font-variant-numeric:tabular-nums}
hr{border:0;border-top:1px solid var(--rule);margin:0}

/* header */
.head{display:flex;flex-direction:column;gap:14px;padding-bottom:24px;
  border-bottom:2px solid var(--ink)}
.strip{display:flex;flex-wrap:wrap;gap:0;border:1px solid var(--rule);
  background:var(--panel);border-radius:2px;overflow:hidden}
.cell{flex:1 1 150px;padding:10px 14px;border-right:1px solid var(--rule-2);
  display:flex;flex-direction:column;gap:3px;min-width:0}
.cell:last-child{border-right:0}
.cell .k{font-family:var(--mono);font-size:10px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--ink-3)}
.cell .v{font-family:var(--mono);font-size:14px;font-variant-numeric:tabular-nums;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;
  margin-right:6px;vertical-align:1px}
.dot.live{background:var(--live);box-shadow:0 0 0 3px var(--accent-soft)}
.dot.pass{background:var(--pass)}.dot.kill{background:var(--kill)}
.dot.watch{background:var(--watch)}.dot.idle{background:var(--ink-3)}
@media (prefers-reduced-motion:no-preference){
  .dot.live{animation:pulse 2.4s ease-in-out infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.45}}
}

/* panels */
.panel{background:var(--panel);border:1px solid var(--rule);border-radius:2px;
  padding:18px 20px;display:flex;flex-direction:column;gap:12px}
.panel.tight{padding:14px 16px}
.quiet{background:var(--panel-2);border-color:var(--rule-2)}

/* tables */
.tscroll{overflow-x:auto;border:1px solid var(--rule);border-radius:2px;
  background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{text-align:left;padding:9px 12px;border-bottom:1px solid var(--rule-2);
  vertical-align:top}
th{font-family:var(--mono);font-size:10.5px;letter-spacing:.09em;
  text-transform:uppercase;color:var(--ink-3);font-weight:400;
  background:var(--panel-2);white-space:nowrap}
tr:last-child td{border-bottom:0}
td.n{font-family:var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}
.rank{font-family:var(--serif);font-size:17px;color:var(--ink-3)}

/* confidence bar */
.bar{position:relative;height:6px;background:var(--rule-2);border-radius:1px;
  min-width:76px}
.bar>i{position:absolute;inset:0 auto 0 0;background:var(--accent);
  border-radius:1px}
.bar.v>i{background:var(--theory)}

/* pills */
.pill{display:inline-flex;align-items:center;gap:5px;font-family:var(--mono);
  font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;
  padding:3px 8px;border-radius:2px;border:1px solid currentColor;
  white-space:nowrap}
.pill.pass{color:var(--pass)}.pill.kill{color:var(--kill)}
.pill.watch{color:var(--watch)}.pill.idle{color:var(--ink-3)}
.pill.live{color:var(--live)}.pill.theory{color:var(--theory)}

/* task timeline */
.tl{display:flex;flex-direction:column}
.phase{font-family:var(--mono);font-size:10.5px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--ink-3);padding:16px 0 6px}
.task{display:grid;grid-template-columns:58px 1fr auto;gap:14px;
  padding:10px 0;border-top:1px solid var(--rule-2);align-items:baseline}
.task .id{font-family:var(--mono);font-size:12px;color:var(--ink-3)}
.task .t{display:flex;flex-direction:column;gap:3px}
.task .t b{font-weight:600}
.task .t span{color:var(--ink-2);font-size:13.5px}
.task.done .t b{color:var(--ink-2)}

/* log */
.log{display:flex;flex-direction:column;gap:0}
.entry{display:grid;grid-template-columns:74px 62px 1fr;gap:12px;
  padding:9px 0;border-top:1px solid var(--rule-2);font-size:13.5px}
.entry:first-child{border-top:0}
.entry .ts{font-family:var(--mono);font-size:11.5px;color:var(--ink-3)}

/* misc */
details{border:1px solid var(--rule);border-radius:2px;background:var(--panel)}
details>summary{cursor:pointer;padding:12px 16px;font-family:var(--mono);
  font-size:11.5px;letter-spacing:.08em;text-transform:uppercase;
  color:var(--ink-2)}
details[open]>summary{border-bottom:1px solid var(--rule-2)}
.dbody{padding:14px 18px 18px;display:flex;flex-direction:column;gap:12px}
ul{margin:0;padding-left:20px;display:flex;flex-direction:column;gap:6px}
li{color:var(--ink-2)}
figure{margin:0;display:flex;flex-direction:column;gap:8px}
figure img{width:100%;height:auto;border:1px solid var(--rule);border-radius:2px;
  background:var(--panel)}
figcaption{font-size:13px;color:var(--ink-2)}
code{font-family:var(--mono);font-size:.92em;background:var(--panel-2);
  padding:1px 5px;border-radius:2px}
.callout{border-left:3px solid var(--accent);padding:2px 0 2px 14px;
  color:var(--ink-2)}
.callout.theory{border-color:var(--theory)}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.foot{color:var(--ink-3);font-size:12.5px;font-family:var(--mono)}
</style>
"""


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def pill(state: str, label: str | None = None) -> str:
    k = STATE_COLORS.get(state, "idle")
    return f'<span class="pill {k}"><span class="dot {k}"></span>{esc(label or state)}</span>'


def bar(frac: float, kind: str = "") -> str:
    pct = max(0.0, min(1.0, float(frac))) * 100
    return f'<div class="bar {kind}"><i style="width:{pct:.0f}%"></i></div>'


def fmt_ci(m: dict | None) -> str:
    """Render a measurement dict {value, ci_lo, ci_hi, n} — CIs mandatory."""
    if not m:
        return "—"
    if not isinstance(m, dict):
        return f"{m}"
    v = m.get("value")
    if v is None:
        return "—"
    s = f"{v:.3f}"
    lo, hi = m.get("ci_lo"), m.get("ci_hi")
    if lo is not None and hi is not None:
        s += f' <span style="color:var(--ink-3)">[{lo:.3f}, {hi:.3f}]</span>'
    if m.get("n"):
        s += f' <span style="color:var(--ink-3)">n={m["n"]}</span>'
    return s


def img_uri(path: Path) -> str | None:
    if not path.exists():
        return None
    b64 = base64.b64encode(path.read_bytes()).decode()
    kind = "png" if path.suffix.lower() == ".png" else "svg+xml"
    return f"data:image/{kind};base64,{b64}"


def build(state: dict) -> str:
    run = state["run"]
    ctx = state["context"]
    res = state["resources"]
    mon = state["monitor"]

    p: list[str] = []
    a = p.append

    a("<title>Regime-3 task hunt — live run</title>")
    a(CSS)
    a('<div class="wrap"><div class="stack g40">')

    # ── header ────────────────────────────────────────────────────────
    a('<header class="head">')
    a('<div class="stack g8">')
    a(f'<div class="eyebrow">TempBench · {esc(run["branch"])} branch · live run log</div>')
    a(f"<h1>{esc(run['title'])}</h1>")
    a(f'<p class="sub">{esc(run["subtitle"])}</p>')
    a("</div>")
    st = run["state"]
    a('<div class="strip">')
    for k, v, cls in [
        ("status", run["state"], STATE_COLORS.get(st, "idle")),
        ("phase", run["phase"], None),
        ("updated", run["last_update_utc"].replace("T", " ").replace("Z", " UTC"), None),
        ("agent", run["agent"], None),
    ]:
        dot = f'<span class="dot {cls}"></span>' if cls else ""
        a(f'<div class="cell"><span class="k">{esc(k)}</span>'
          f'<span class="v">{dot}{esc(v)}</span></div>')
    a("</div>")
    a(f'<p class="callout"><b>Prime directive.</b> {esc(run["prime_directive"])}</p>')
    a("</header>")

    # ── timing ────────────────────────────────────────────────────────
    tm = state.get("timing")
    if tm:
        a("<section>")
        a('<div class="eyebrow">Clock</div>')
        a("<h2>Elapsed and projected, per phase</h2>")
        a('<div class="strip">')
        for k, v in [
            ("elapsed", f'{tm["elapsed_min"]} min'),
            ("projected total", f'{tm["projected_total_min"]} min'),
            ("projected remaining", f'{tm["projected_remaining_min"]} min'),
            ("session start", tm["session_start_utc"].split("T")[-1].replace("Z", " UTC")),
        ]:
            a(f'<div class="cell"><span class="k">{esc(k)}</span><span class="v">{esc(v)}</span></div>')
        a("</div>")
        a('<div class="tscroll"><table><thead><tr><th>phase</th><th>state</th>'
          "<th>elapsed</th><th>projected</th><th>progress</th><th>note</th>"
          "</tr></thead><tbody>")
        for ph in tm["phases"]:
            frac = min(1.0, ph["elapsed_min"] / ph["projected_min"]) if ph["projected_min"] else 0
            over = ph["elapsed_min"] > ph["projected_min"]
            a(f'<tr><td class="n">{esc(ph["phase"])}</td>'
              f'<td>{pill(ph["state"])}</td>'
              f'<td class="n">{ph["elapsed_min"]} min</td>'
              f'<td class="n">{ph["projected_min"]} min</td>'
              f'<td>{bar(frac, "v" if over else "")}</td>'
              f'<td style="font-size:12.5px;color:var(--ink-2)">{esc(ph["note"])}</td></tr>')
        a("</tbody></table></div>")
        a(f'<p class="sub" style="max-width:none;font-size:13px;color:var(--ink-3)">'
          f'{esc(tm["basis"])}</p>')
        a("</section>")

    # ── triage ────────────────────────────────────────────────────────
    tri = state.get("triage")
    if tri:
        a("<section>")
        a('<div class="eyebrow">Gate 0 — label side</div>')
        a("<h2>The falsifier that authorises GPU spend</h2>")
        a(f'<p class="sub">{esc(tri["note"])}</p>')
        a('<div class="tscroll"><table><thead><tr><th>task</th><th>items</th>'
          "<th>label rate</th><th>AUC from A</th><th>AUC from B</th>"
          "<th>AUC from length</th><th>AUC from filler</th><th>verdict</th>"
          "</tr></thead><tbody>")
        for name, t in tri["tasks"].items():
            v = "pass" if t["PASS"] else "kill"
            a(f'<tr><td class="n">{esc(name)}</td><td class="n">{t["n"]}</td>'
              f'<td class="n">{t["label_rate"]:.3f}</td>'
              f'<td class="n">{t["auc_from_a"]:.3f}</td>'
              f'<td class="n">{t["auc_from_b"]:.3f}</td>'
              f'<td class="n">{t["auc_from_len"]:.3f}</td>'
              f'<td class="n">{t["auc_from_nfiller"]:.3f}</td>'
              f'<td><span class="pill {v}">{"PASS" if t["PASS"] else "FAIL"}</span></td></tr>')
        a("</tbody></table></div>")
        a("</section>")

    # ── why ───────────────────────────────────────────────────────────
    a("<section>")
    a('<div class="eyebrow">Why this run exists</div>')
    a("<h2>The reviewers asked one question in three voices</h2>")
    a("<ul>" + "".join(f"<li>{esc(w)}</li>" for w in ctx["why"]) + "</ul>")
    a(f'<div class="panel quiet"><h3>Diagnosis</h3><p class="sub" style="max-width:none">{esc(ctx["diagnosis"])}</p></div>')
    a("</section>")

    # ── the two filters + design rule ─────────────────────────────────
    a("<section>")
    a('<div class="eyebrow">Selection principle</div>')
    a("<h2>Two filters decide which tasks can possibly separate</h2>")
    for f in ctx["two_filters"]:
        a(f'<div class="panel"><div class="stack g8">'
          f'<h3>{esc(f["name"])}</h3>'
          f'<div class="mono" style="font-size:11.5px;color:var(--ink-3)">{esc(f["source"])}</div>'
          f'<p class="sub" style="max-width:none">{esc(f["statement"])}</p></div></div>')
    a(f'<p class="callout theory"><b>Design rule.</b> {esc(ctx["design_rule"])}</p>')
    a('<div class="tscroll"><table><thead><tr><th>architecture</th><th>role in the argument</th>'
      "<th>status on a balanced-marginal equality label</th></tr></thead><tbody>")
    for r in ctx["floor_table"]:
        is_txc = "only" in r["status"]
        cls = "theory" if is_txc else "idle"
        a(f'<tr><td class="n">{esc(r["arch"])}</td><td>{esc(r["role"])}</td>'
          f'<td><span class="pill {cls}">{esc(r["status"])}</span></td></tr>')
    a("</tbody></table></div>")
    a("</section>")

    # ── candidate ladder ──────────────────────────────────────────────
    a("<section>")
    a('<div class="eyebrow">Ranked ledger</div>')
    a("<h2>Ten candidates, ranked by confidence of victory</h2>")
    a(f'<p class="sub">{esc(state["candidate_note"])}</p>')
    a('<div class="tscroll"><table><thead><tr><th></th><th>candidate</th>'
      "<th>P(win)</th><th>P(violent)</th><th>cost</th><th>status</th>"
      "<th>gate result</th></tr></thead><tbody>")
    for c in state["candidates"]:
        gate = fmt_ci(c.get("gate"))
        a("<tr>"
          f'<td class="rank">{c["rank"]}</td>'
          f'<td><div class="stack g8"><b>{esc(c["name"])}</b>'
          f'<span style="color:var(--ink-2);font-size:12.5px">{esc(c["why"])}</span></div></td>'
          f'<td class="n">{c["p_win"]:.2f}{bar(c["p_win"])}</td>'
          f'<td class="n">{c["p_violent"]:.2f}{bar(c["p_violent"], "v")}</td>'
          f'<td class="n">{esc(c["cost"])}</td>'
          f'<td>{esc(c["status"])}</td>'
          f'<td class="n">{gate}</td>'
          "</tr>")
    a("</tbody></table></div>")
    a("</section>")

    # ── figures ───────────────────────────────────────────────────────
    figs = state.get("figures") or []
    a("<section>")
    a('<div class="eyebrow">Measurements</div>')
    a("<h2>Figures</h2>")
    if not figs:
        a('<div class="panel quiet"><p class="sub" style="max-width:none">'
          "No measurements yet. Every figure that lands here carries bootstrap "
          "confidence intervals and a permutation null; theoretical floors are "
          "drawn in violet so theory never reads as measurement.</p></div>")
    else:
        for f in figs:
            uri = img_uri(Path(f["path"]) if Path(f["path"]).is_absolute() else HERE.parent / f["path"])
            if not uri:
                continue
            a(f'<figure><img src="{uri}" alt="{esc(f.get("caption",""))}">'
              f'<figcaption>{esc(f.get("caption",""))}</figcaption></figure>')
    a("</section>")

    # ── tasks ─────────────────────────────────────────────────────────
    a("<section>")
    a('<div class="eyebrow">Execution</div>')
    a("<h2>Task sequence</h2>")
    a('<p class="sub">Order carries information here: cards are frozen before screens, '
      "label-side triage gates GPU spend, and GPU jobs are serialized rather than fanned out.</p>")
    a('<div class="tl">')
    last_phase = None
    for t in state["tasks"]:
        if t["phase"] != last_phase:
            a(f'<div class="phase">Phase {esc(t["phase"])}</div>')
            last_phase = t["phase"]
        a(f'<div class="task {esc(t["state"])}">'
          f'<span class="id">{esc(t["id"])}</span>'
          f'<span class="t"><b>{esc(t["title"])}</b><span>{esc(t["detail"])}</span></span>'
          f"{pill(t['state'])}</div>")
    a("</div>")
    a("</section>")

    # ── resources / OOM ───────────────────────────────────────────────
    a("<section>")
    a('<div class="eyebrow">Instrumentation</div>')
    a("<h2>Resources &amp; OOM tracking</h2>")
    a('<div class="strip">')
    dfree = res.get("disk_free_gb")
    peak = res.get("peak_vram_gb")
    for k, v in [
        ("gpu", res.get("gpu", "—")),
        ("disk free", f'{dfree:.1f} / {res["disk_total_gb"]:.0f} GB' if dfree else "—"),
        ("disk floor", f'{res["disk_floor_gb"]:.0f} GB abort'),
        ("peak vram", f"{peak:.1f} GB" if peak else "not yet measured"),
        ("vram ceiling", f'{res["vram_floor_gb"]:.0f} GB guard'),
        ("oom events", str(len(res.get("oom_events", [])))),
    ]:
        a(f'<div class="cell"><span class="k">{esc(k)}</span><span class="v">{esc(v)}</span></div>')
    a("</div>")
    a(f'<p class="sub" style="max-width:none">{esc(res["note"])}</p>')
    ooms = res.get("oom_events", [])
    if ooms:
        a('<div class="tscroll"><table><thead><tr><th>cell</th><th>peak vram</th>'
          "<th>action</th></tr></thead><tbody>")
        for o in ooms:
            a(f'<tr><td class="n">{esc(o.get("cell"))}</td>'
              f'<td class="n">{esc(o.get("peak_vram_gb"))} GB</td>'
              f'<td>{esc(o.get("action"))}</td></tr>')
        a("</tbody></table></div>")
    a("</section>")

    # ── monitor ───────────────────────────────────────────────────────
    a("<section>")
    a('<div class="eyebrow">Monitor</div>')
    a("<h2>Expected behaviour, and what would count as an anomaly</h2>")
    a('<div class="tscroll"><table><thead><tr><th></th><th>quantity</th>'
      "<th>expected</th><th>why it is the right expectation</th></tr></thead><tbody>")
    for e in mon["expectations"]:
        a(f'<tr><td class="n">{esc(e["id"])}</td><td>{esc(e["what"])}</td>'
          f'<td class="n">{esc(e["expect"])}</td><td>{esc(e["why"])}</td></tr>')
    a("</tbody></table></div>")
    anom = mon.get("anomalies", [])
    if anom:
        a('<div class="tscroll"><table><thead><tr><th>when</th><th>expectation</th>'
          "<th>observed</th><th>disposition</th></tr></thead><tbody>")
        for x in anom:
            a(f'<tr><td class="n">{esc(x.get("ts"))}</td><td class="n">{esc(x.get("expectation"))}</td>'
              f'<td class="n">{esc(x.get("observed"))}</td><td>{esc(x.get("disposition"))}</td></tr>')
        a("</tbody></table></div>")
    else:
        a('<div class="panel quiet"><p class="sub" style="max-width:none">'
          "No anomalies recorded. An anomaly here is a disclosure, not an embarrassment: "
          "a fired expectation redirects the run rather than being reinterpreted.</p></div>")
    a("</section>")

    # ── log ───────────────────────────────────────────────────────────
    a("<section>")
    a('<div class="eyebrow">Reasoning transparency</div>')
    a("<h2>Run log</h2>")
    a('<div class="panel"><div class="log">')
    for e in reversed(state["log"]):
        k = STATE_COLORS.get(e["kind"], "idle")
        ts = e["ts"].split("T")[-1].replace("Z", "")
        a(f'<div class="entry"><span class="ts">{esc(ts)}</span>'
          f'<span><span class="pill {k}">{esc(e["kind"])}</span></span>'
          f'<span>{esc(e["text"])}</span></div>')
    a("</div></div>")
    a("</section>")

    a(f'<p class="foot">Rendered from state.json by '
      f"experiments/explorations/relational/dashboard/render.py · "
      f'{len(state["log"])} log entries · updated {esc(run["last_update_utc"])}</p>')
    a("</div></div>")
    return "\n".join(p)


def main() -> None:
    state = json.loads(STATE.read_text())
    OUT.write_text(build(state))
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
