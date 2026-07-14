"""Render each candidate's `records/<name>/calibration.md` from its stats JSON.

Pure rendering — reads `records/<name>/calibration_stats.json` (written by
`calibrate.py`) and the frozen prereg card; makes NO API calls and computes no
new statistics, so it can be rerun freely. A record is written for ABORTs too:
an abort is a success (prime directive).

    .venv/bin/python -m experiments.explorations.synthetic.expansion.render_records
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ROWS = {
    "binary": [("ACF(1)", "acf", 0), ("MI(1) (nats)", "mi", 0),
               ("Fano", "fano", None),
               ("excite ratio P(1|1)/base", "excite_ratio", None),
               ("inter-event gap CV", "gap_cv", None),
               ("spectral peak prominence", "spec_peak", None)],
    "categorical": [("directed asymmetry (fwd−rev)/(fwd+rev)", "asym", None),
                    ("P(dst @ t+1 | src @ t) forward", "fwd_rate", None),
                    ("same, time-reversed", "rev_rate", None),
                    ("self-match ACF(1)", "acf", 0),
                    ("dwell mean", "dwell_mean", None),
                    ("dwell CV", "dwell_cv", None)],
    "scalar": [("ACF(1)", "acf", 0), ("MI(1) (binned, nats)", "mi", 0)],
}


def _v(container, key, idx):
    v = container.get(key)
    if v is None:
        return None
    a = np.asarray(v, dtype=float)
    return float(a.ravel()[idx]) if idx is not None else float(a)


def _fmt(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def render_one(name: str) -> Path:
    d = HERE / "records" / name
    blob = json.loads((d / "calibration_stats.json").read_text())
    cfg, gate, s = blob["config"], blob["gate"], blob["signature"]
    val = blob["labeler_validation"]
    inter, xc = val["interjudge"], val["heuristic_crosscheck"]
    kind = s["kind"]
    verdict = gate["verdict"]

    card_name = cfg.get("base_card") or name
    lines = [
        f"# Calibration record — `{name}`",
        "",
        f"**Verdict: {verdict}**"
        + (f" (pre-skeptic PROCEED, killed by skeptic on {gate['killed_by_skeptic']})"
           if gate.get("killed_by_skeptic") else "")
        + (" (numeric gate passed; **mirror failed its preregistered gate-8 "
           "moment** — see § 4)" if gate.get("gate8_fail") else ""),
        "",
        f"Calibration per the frozen [prereg card](../../prereg/{card_name}.md); "
        f"domain `{cfg['domain']}`, {s['n_seqs']} documents / {s['n_spans']} labeled "
        f"sentences (doc coverage {blob['coverage']['doc_coverage']:.3f}).",
        "",
        "## 1. Labeler + noise floor",
        "",
        f"- **Bulk judge:** `claude-haiku-4-5-20251001` (frozen instruction in the card); "
        f"**second judge:** `claude-sonnet-5` on {inter['n_docs']} held-out docs "
        f"({inter['n_spans']} sentences).",
        f"- **Inter-judge:** agreement {inter['agreement']:.3f}, κ = {inter['kappa']:.3f} "
        f"→ noise floor ε̂ = {val['noise_floor_eps']:.3f} "
        f"(adequacy floor κ ≥ {gate['kappa_floor']}: "
        f"{'PASS' if gate['labeler_ok'] else '**FAIL — labeler inadequate**'}).",
    ]
    if "f1" in xc:
        lines.append(f"- **Independent heuristic cross-check:** P {xc['precision']:.2f} / "
                     f"R {xc['recall']:.2f} / F1 {xc['f1']:.2f} "
                     f"(judge rate {xc['judge_pos_rate']:.3f}, heuristic {xc['heuristic_pos_rate']:.3f}).")
    else:
        pc = ", ".join(f"class {c}: F1 {v['f1']:.2f}" for c, v in xc["per_class"].items())
        lines.append(f"- **Independent heuristic cross-check:** accuracy {xc['accuracy']:.2f}, "
                     f"κ {xc['kappa']:.2f} ({pc}).")

    lines += ["", "## 2. Temporal signature vs nulls", "",
              "| statistic | real | N1 permute | N2 trend | N3 iid |",
              "|---|---|---|---|---|"]
    for label, key, idx in ROWS[kind]:
        real = _v(s["real"], key, idx)
        ci_key = key + ("1" if key in ("acf", "mi") else "")
        ci = s["real_ci"].get(ci_key)
        real_s = _fmt(real) + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")
        # nulls store {stat: {mean, lo, hi}}
        cells = [_fmt(_v(s["nulls"][n][key], "mean", idx)) if key in s["nulls"][n] else "—"
                 for n in ("N1_permute", "N2_trend", "N3_iid")]
        lines.append(f"| {label} | **{real_s}** | {cells[0]} | {cells[1]} | {cells[2]} |")
    if kind == "binary":
        lines.append(f"\nBase rate {s['base_rate']:.4f}; Markov order-1 vs 0 "
                     f"p = {s['markov']['p_order1_vs_0']:.2e}.")
    if kind == "categorical":
        lines.append(f"\nMarginal {['%.3f' % m for m in s['marginal']]}; Markov order-1 vs 0 "
                     f"p = {s['markov']['p_order1_vs_0']:.2e}.")

    st = gate["stability"]
    sign = gate.get("sign", "+")
    side = ("> N1 hi AND N2 hi" if sign == "+"
            else "< N1 lo AND N2 lo (preregistered NEGATIVE effect)")
    band = (f"N1 97.5% band hi {gate['N1_hi']:.4f}, N2 hi {gate['N2_hi']:.4f}"
            if sign == "+" else
            f"N1 2.5% band lo {gate.get('N1_lo', float('nan')):.4f}, "
            f"N2 lo {gate.get('N2_lo', float('nan')):.4f}")
    lines += [
        "", "## 3. Gate (preregistered) + verdict", "",
        f"Primary statistic `{gate['primary_stat']}` (expected sign {sign}): real "
        f"**{gate['real']:.4f}**, after ε̂-noise perturbation "
        f"**{gate['noise_perturbed']:.4f}**; {band}.",
        "",
        f"- clears sampling noise (real {side}): **{gate['clears_sampling']}**",
        f"- survives labeler noise floor (perturbed likewise): **{gate['clears_noise']}**",
        f"- labeler adequate (κ ≥ {gate['kappa_floor']}): **{gate['labeler_ok']}**",
        f"- split-half stability: {st['half1']:.4f} / {st['half2']:.4f}",
        "",
        f"**→ {verdict}**",
    ]

    if blob.get("mirror"):
        m = blob["mirror"]
        mv = m["validation"]
        lines += ["", "## 4. Mirror (Appendix B) + held-out validation", "",
                  f"Process `{cfg['mirror']}`; fit on {m['n_train']} train docs, "
                  f"validated on {m['n_eval']} held-out docs. Fitted params:",
                  "", "```json", json.dumps(m["params"], indent=1, default=float), "```",
                  "", "| statistic | real (held-out) | synthetic |", "|---|---|---|"]
        for k, vr in mv["real"].items():
            vs = mv["synthetic"][k]
            if isinstance(vr, list):
                lines.append(f"| {k}(1) | {_fmt(vr[0])} | {_fmt(vs[0])} |")
            else:
                lines.append(f"| {k} | {_fmt(vr)} | {_fmt(vs)} |")
        if "real_directed" in mv:
            rd, sd = mv["real_directed"], mv["syn_directed"]
            lines.append(f"| asym (directed) | {_fmt(rd['asym'])} | {_fmt(sd['asym'])} |")
        errs = ", ".join(f"{k} {v:.3f}" for k, v in mv["abs_err"].items())
        lines.append(f"\nAbs errors: {errs}.")
        if m.get("gate8"):
            g8 = m["gate8"]
            lines.append(
                f"\n**Gate 8 (preregistered non-fitted moment)** — `{g8['moment']}`: "
                f"held-out real {g8['real_heldout']:.4f} vs synthetic "
                f"{g8['synthetic']:.4f}, |err| {g8['abs_err']:.4f} vs tolerance "
                f"±{g8['tol_abs']} → **{'PASS' if g8['pass'] else 'FAIL — mirror invalid ⇒ ABORT'}**.")

    if blob.get("skeptic"):
        lines += ["", "## 5. Adversarial skeptic pass (fixed kill-rubric, Opus)", "",
                  "| item | kill | evidence |", "|---|---|---|"]
        for k, v in blob["skeptic"].items():
            if isinstance(v, dict):
                lines.append(f"| {k} | {'**KILL**' if v['kill'] else 'clear'} | {v['evidence']} |")
        lines.append(f"\n_{blob['skeptic'].get('overall_note', '')}_")

    lines += [
        "", "![signature](signature.png)", "",
        "## Reproduction", "",
        "```bash",
        f".venv/bin/python -m experiments.explorations.synthetic.expansion.calibrate {name}",
        ".venv/bin/python -m experiments.explorations.synthetic.expansion.render_records",
        "```",
        f"Deterministic given the cached labels (`labels.json`); judge models pinned "
        f"in the card; spend after this candidate: ${blob['spend_usd_after']:.2f}.",
        "",
    ]
    out = d / "calibration.md"
    out.write_text("\n".join(lines))
    return out


def main():
    for p in sorted((HERE / "records").glob("*/calibration_stats.json")):
        print("->", render_one(p.parent.name))


if __name__ == "__main__":
    main()
