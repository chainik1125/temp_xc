"""Item 2 verdict — mechanical implementation of CARD § 2.3. Committed pre-run.

Reads ``results/tsae_fair_grid_results.json``. First the untrained guard (the
five entries' untrained metrics must be exactly equal per seed — Δ/α touch
only train_step), then the paired Δ tests vs ``tsae_d1``
(bar = max(2·SE, 0.05)): FLAT / RISE (the flag) / DECLINE per knob setting;
``tsae_a0`` is aux commentary. Outputs ``results/tsae_fair_verdict.json`` +
``results/tsae_fair_table.md``; a RISE prints a loud flag block (the LOG entry
and runpod-d note are then written per the card — flag first, skeptic second).

Run:  .venv/bin/python -m \
        experiments.explorations.task_hunt.support_synthetic.analyze_tsae
"""

from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = (1, 2, 42)
FLOOR = 0.05
METRIC = "lambda_recovery"
ANCHOR = "tsae_d1"
SWEEP = ("tsae_d2", "tsae_d4", "tsae_d8")
AUX = "tsae_a0"
DPI_FLOOR = 0.41            # provable per-token floor (bench_record)


def main():
    rows = json.loads((HERE / "results" / "tsae_fair_grid_results.json").read_text())
    trained, untrained = {}, {}
    for r in rows:
        if not r.get("ok"):
            continue
        tgt = untrained if r["kind"] == "untrained" else trained
        tgt.setdefault(r["arch"], {})[r["seed"]] = r["metrics"]

    # ── untrained guard: exact equality across entries, per seed ──
    guard = {"pass": True, "max_abs_diff": 0.0}
    ref = untrained[ANCHOR]
    for arch, per_seed in untrained.items():
        for s in SEEDS:
            for k, v in ref[s].items():
                dv = abs((per_seed[s].get(k, float("nan"))) - v)
                if not (dv == 0.0):
                    guard["pass"] = False
                    guard["max_abs_diff"] = max(guard["max_abs_diff"], dv)
                    guard.setdefault("violations", []).append(
                        {"arch": arch, "seed": s, "metric": k, "diff": dv})
    if not guard["pass"]:
        print("!! UNTRAINED GUARD FAILED — pipeline bug; do not read trained "
              "cells. Violations:", guard.get("violations")[:5])

    def rec(arch):
        return {s: trained[arch][s][METRIC] for s in SEEDS}

    def paired(arch):
        a, b = rec(arch), rec(ANCHOR)
        ds = [a[s] - b[s] for s in SEEDS]
        D = sum(ds) / len(ds)
        var = sum((x - D) ** 2 for x in ds) / (len(ds) - 1)
        se = math.sqrt(var / len(ds))
        bar = max(2 * se, FLOOR)
        call = "RISE" if D >= bar else ("DECLINE" if D <= -bar else "flat")
        return {"D": D, "SE": se, "bar": bar, "per_seed": ds, "call": call}

    tests = {arch: paired(arch) for arch in SWEEP}
    aux = paired(AUX)
    rise = [a for a, t in tests.items() if t["call"] == "RISE"]
    verdict = "RISE-FLAG" if rise else (
        "FLAT" if all(t["call"] == "flat" for t in tests.values())
        else "NOT-FLAT (decline)")

    means = {a: sum(rec(a).values()) / 3 for a in (ANCHOR, *SWEEP, AUX)}
    out = {
        "card": "CARD.md § 2.3 (frozen pre-run)", "metric": METRIC,
        "untrained_guard": guard, "means": means,
        "paired_vs_d1": tests, "aux_a0_vs_d1": aux,
        "dpi_floor": DPI_FLOOR, "verdict": verdict, "rise_at": rise,
    }
    res = HERE / "results"
    (res / "tsae_fair_verdict.json").write_text(json.dumps(out, indent=1, default=float))

    md = ["# Item 2 — T-SAE fairness receipt (mechanical output of analyze_tsae.py)",
          "", f"**VERDICT: {verdict}**   untrained guard: "
          f"{'PASS (exact)' if guard['pass'] else '**FAIL**'}", "",
          "| entry | mean λ̂ recovery | paired D vs Δ=1 | bar | call |",
          "|---|---|---|---|---|",
          f"| tsae_d1 (Δ=1 ≡ registered) | {means[ANCHOR]:.3f} | — | — | anchor |"]
    for a in SWEEP:
        t = tests[a]
        md.append(f"| {a} | {means[a]:.3f} | {t['D']:+.3f} | {t['bar']:.3f} | {t['call']} |")
    md.append(f"| tsae_a0 (aux) | {means[AUX]:.3f} | {aux['D']:+.3f} | "
              f"{aux['bar']:.3f} | {aux['call']} (aux) |")
    md += ["", f"Per-token DPI floor ≈ {DPI_FLOOR} (bench band 0.38–0.44)."]
    if rise:
        md += ["", "**!! RISE FIRED — per CARD § 2.3: LOG flag + runpod-d note "
               "IMMEDIATELY, then skeptic. The real panel's T-SAE cell may "
               f"underestimate the baseline (best knob: {rise}).**"]
    (res / "tsae_fair_table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
