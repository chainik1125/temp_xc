"""Item 1 verdict — mechanical implementation of CARD § 1.5. Committed pre-run.

Reads ``results/dilution_grid_results.json``, computes per-line peaks and
paired-by-seed declines on T ≤ 16 (bar = max(2·SE, 0.05)), and emits the frozen
verdict (BACKED / RETRACT / NO-MIRROR-DIP / AMBIGUOUS; RETRACT checked first —
DIP(B) always retracts, whatever else fired) plus the commentary blocks
(P3 dose-response at T = 16, P4 untrained shapes, the T = 32 descriptive
extension with B's matched-untrained lens). Outputs
``results/dilution_verdict.json`` + ``results/dilution_table.md``.

Run:  .venv/bin/python -m \
        experiments.explorations.task_hunt.support_synthetic.analyze_dilution
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from experiments.explorations.task_hunt.support_synthetic.run_dilution import POINTS

HERE = Path(__file__).resolve().parent
SEEDS = (1, 2, 42)
FLOOR = 0.05
METRIC = "lambda_recovery"
VERDICT_T_MAX = 16          # frozen: all verdicts on T <= 16 (CARD § 1.3)


def _load():
    rows = json.loads((HERE / "results" / "dilution_grid_results.json").read_text())
    trained, untrained, l0w, l0t = {}, {}, {}, {}
    for r in rows:
        if not r.get("ok"):
            continue
        key = (r["T"], r["d_sae"])
        m = r["metrics"]
        tgt = untrained if r["kind"] == "untrained" else trained
        tgt.setdefault(key, {})[r["seed"]] = m[METRIC]
        if r["kind"] == "trained":
            l0w.setdefault(key, {})[r["seed"]] = m.get("l0_per_window")
            l0t.setdefault(key, {})[r["seed"]] = m.get("l0_per_token")
    return trained, untrained, l0w, l0t


def _mean(d):
    return sum(d.values()) / len(d)


def _paired(a: dict, b: dict):
    """Paired-by-seed D = mean(a_s - b_s), SE = std(ddof=1)/sqrt(n), bar."""
    ds = [a[s] - b[s] for s in SEEDS]
    n = len(ds)
    D = sum(ds) / n
    var = sum((x - D) ** 2 for x in ds) / (n - 1)
    se = math.sqrt(var / n)
    return {"D": D, "SE": se, "bar": max(2 * se, FLOOR), "per_seed": ds}


def _line_verdict(pts, trained, excluded):
    """Peak + declines for one line on T <= VERDICT_T_MAX."""
    cells = {}
    for T, d in pts:
        if T > VERDICT_T_MAX:
            continue
        key = (T, d)
        if key not in trained or set(trained[key]) != set(SEEDS):
            excluded.append({"cell": key, "reason": "missing seeds"})
            continue
        cells[T] = trained[key]
    t_peak = max(cells, key=lambda T: _mean(cells[T]))
    declines = {}
    for T in cells:
        if T <= t_peak:
            continue
        p = _paired(cells[t_peak], cells[T])
        p["fires"] = p["D"] >= p["bar"]
        declines[T] = p
    rise = _paired(cells[t_peak], cells[min(cells)])
    return {
        "means": {T: _mean(v) for T, v in cells.items()},
        "t_peak": t_peak,
        "rise_from_tmin": {**rise, "fires": rise["D"] >= rise["bar"]},
        "declines": declines,
        "dip": any(p["fires"] for p in declines.values()),
    }


def main():
    trained, untrained, l0w, l0t = _load()
    excluded = []
    lines = {name: _line_verdict(pts, trained, excluded)
             for name, pts in POINTS.items()}
    dip = {k: v["dip"] for k, v in lines.items()}

    if dip["B"]:
        verdict = "RETRACT"
    elif dip["A1"]:
        verdict = "BACKED"
    elif not dip["A1"] and not dip["A2"]:
        verdict = "NO-MIRROR-DIP"
    else:
        verdict = "AMBIGUOUS"

    # ── commentary (never verdict inputs) ──
    # P3 dose-response at T = 16: decline-from-own-peak per line.
    def _decline_at(name, T):
        v = lines[name]
        return v["declines"].get(T, {}).get("D") if T > v["t_peak"] else 0.0

    dose = {name: _decline_at(name, 16) for name in ("A1", "A2", "B")}

    # T = 32 descriptive extension (A2 probe-clean; B read vs matched untrained).
    ext = {}
    for name, key in (("A2", (32, 40)), ("B", (32, 160))):
        if key in trained:
            ext[name] = {
                "mean": _mean(trained[key]),
                "untrained_mean": _mean(untrained[key]) if key in untrained else None,
            }
            if ext[name]["untrained_mean"] is not None:
                ext[name]["trained_minus_untrained"] = (
                    ext[name]["mean"] - ext[name]["untrained_mean"])

    untr = {f"{k}": _mean(v) for k, v in sorted(untrained.items())}
    l0_table = {
        f"{k}": {"l0_per_window": _mean(v), "l0_per_token": _mean(l0t[k])}
        for k, v in sorted(l0w.items())
    }

    out = {
        "card": "CARD.md § 1.5 (frozen pre-run)", "metric": METRIC,
        "verdict_t_max": VERDICT_T_MAX, "bar_floor": FLOOR,
        "lines": lines, "dip_flags": dip, "verdict": verdict,
        "dose_response_T16": dose, "t32_extension": ext,
        "untrained_means": untr, "realized_l0": l0_table,
        "excluded_pairs": excluded,
    }
    res = HERE / "results"
    (res / "dilution_verdict.json").write_text(json.dumps(out, indent=1, default=float))

    md = ["# Item 1 — dilution receipt (mechanical output of analyze_dilution.py)",
          "", f"**VERDICT: {verdict}**  (DIP flags: {dip})", "",
          "| line | " + " | ".join(f"T={T}" for T in (2, 4, 8, 16, 32)) + " |",
          "|---|" + "---|" * 5]
    for name, pts in POINTS.items():
        vals = []
        for T in (2, 4, 8, 16, 32):
            d = dict(pts).get(T)
            key = (T, d) if d else None
            if key and key in trained:
                mark = " *peak*" if (T == lines[name]["t_peak"]) else ""
                vals.append(f"{_mean(trained[key]):.3f}{mark}")
            else:
                vals.append("—")
        md.append(f"| {name} | " + " | ".join(vals) + " |")
    md += ["", "Declines from peak (paired D, bar; T<=16):"]
    for name, v in lines.items():
        for T, p in v["declines"].items():
            md.append(f"- {name} peak T={v['t_peak']} -> T={T}: "
                      f"D={p['D']:+.3f} bar={p['bar']:.3f} "
                      f"{'**FIRES**' if p['fires'] else 'no'}")
    md += ["", f"Dose-response at T=16 (decline from own peak): {dose}",
           f"T=32 extension (descriptive): {json.dumps(ext, default=float)}",
           "", "Untrained means: " + json.dumps(untr, default=float),
           "", "Realized l0 (trained): " + json.dumps(l0_table, default=float)]
    (res / "dilution_table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
