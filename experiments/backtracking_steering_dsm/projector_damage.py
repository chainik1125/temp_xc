"""The denoise-after-steer confound control: what does the projector alone do?

    uv run --no-sync python experiments/backtracking_steering_dsm/projector_damage.py \
        --wave-dir <dir with merged rows__*.json + judged__*.json>

Pre-registered before the projected arm ran. At a pre-flight NMSE of 0.795 on
the steering distribution, with 605 of 16384 latents alive, the projected arm
primarily measures projector robustness under distribution shift rather than
anything about steering. The alpha = 0 cell is what separates the two: at zero
magnitude the steering vector contributes nothing, so any difference between the
projected and unprojected source at alpha = 0 is the projector's own damage to
generation, measured with the steer switched off.

A flattened Delta-gc curve is only informative about temporal structure if this
cell shows the projector is otherwise harmless. If alpha = 0 is already
degraded, the arm measures projector damage and must be reported as such.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.ward_backtracking_txc.metrics import _coh_ok  # noqa: E402

SONNET_FLOOR = 2


def _mean(xs):
    return (sum(xs) / len(xs)) if xs else float("nan")


def _load(wave_dir: Path, tag: str) -> list[dict]:
    rows = json.loads((wave_dir / f"rows__{tag}.json").read_text())["rows"]
    judged = json.loads((wave_dir / f"judged__{tag}.json").read_text())
    for i, r in enumerate(rows):
        j = judged.get(str(i), {})
        r["genuine_count"] = int(j.get("genuine_count", -1))
        r["coh_grade"] = int(j.get("grade", -1))
        r["coh_ok"] = _coh_ok(r.get("text", ""))
    return rows


def cell(rows: list[dict], mag: float) -> dict:
    c = [r for r in rows if r["magnitude"] == mag]
    valid = [r for r in c if r["genuine_count"] >= 0]
    return {
        "n": len(c),
        "gc": _mean([r["genuine_count"] for r in valid]),
        "event_rate": _mean([1.0 if r["genuine_count"] >= 1 else 0.0
                             for r in valid]),
        "mean_sonnet": _mean([r["coh_grade"] for r in c if r["coh_grade"] >= 0]),
        "frac_sonnet_ok": _mean([1.0 if r["coh_grade"] >= SONNET_FLOOR else 0.0
                                 for r in c]),
        "frac_run_ok": _mean([1.0 if r["coh_ok"] else 0.0 for r in c]),
        "mean_words": _mean([r["n_words"] for r in c]),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--wave-dir", type=Path, required=True)
    p.add_argument("--projected", default=None,
                   help="projected tag; default = the one ending _proj")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)

    tags = [q.name[len("rows__"):-len(".json")]
            for q in a.wave_dir.glob("rows__*.json")]
    proj = a.projected or next((t for t in tags if t.endswith("_proj")), None)
    if proj is None:
        print("[error] no projected source found", file=sys.stderr)
        return 1
    base = proj[:-len("_proj")]
    if base not in tags:
        print(f"[error] unprojected counterpart {base} missing", file=sys.stderr)
        return 1

    rp, rb = _load(a.wave_dir, proj), _load(a.wave_dir, base)
    out = {"projected": proj, "unprojected": base, "cells": {}}
    print(f"projector-damage control: {proj} vs {base}\n")
    hdr = (f"{'alpha':>7} {'gc proj':>8} {'gc unproj':>10} {'d gc':>7} "
           f"{'sonnet proj':>12} {'sonnet unproj':>14} {'words proj':>11} "
           f"{'words unproj':>13}")
    print(hdr)
    for mag in (0.0, 1.0, -1.0, 4.0, -4.0, 8.0, -8.0):
        cp, cb = cell(rp, mag), cell(rb, mag)
        if cp["n"] == 0 or cb["n"] == 0:
            continue
        out["cells"][str(mag)] = {"projected": cp, "unprojected": cb}
        print(f"{mag:>7} {cp['gc']:>8.2f} {cb['gc']:>10.2f} "
              f"{cp['gc'] - cb['gc']:>+7.2f} {cp['mean_sonnet']:>12.2f} "
              f"{cb['mean_sonnet']:>14.2f} {cp['mean_words']:>11.0f} "
              f"{cb['mean_words']:>13.0f}")

    z = out["cells"].get("0.0")
    if z:
        dg = z["projected"]["gc"] - z["unprojected"]["gc"]
        ds = z["projected"]["mean_sonnet"] - z["unprojected"]["mean_sonnet"]
        print(f"\nAT ALPHA = 0 (steer off, projector on):")
        print(f"  delta gc            {dg:+.3f}")
        print(f"  delta Sonnet grade  {ds:+.3f}")
        print(f"  Sonnet pass rate    {z['projected']['frac_sonnet_ok']:.2f} "
              f"projected vs {z['unprojected']['frac_sonnet_ok']:.2f} unprojected")
        verdict = ("projector is NOT harmless at alpha=0 -- the arm measures "
                   "projector damage" if abs(ds) >= 0.3 or abs(dg) >= 0.3 else
                   "projector is approximately harmless at alpha=0")
        print(f"  -> {verdict}")
        out["alpha0_verdict"] = verdict

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
