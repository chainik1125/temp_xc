"""Read-only P1–P6 scorer for the diafaces panels (tt card § 5;
PANEL2_CARD § 4). Point estimates + seed spreads from the merged
panel file; the OFFICIAL variance CIs come from the
`support_stats/stage2_variance` harness (mac-b's lane) — P1's
"CI clear of 0" clause is decided THERE, this script prints the
margins the harness must confirm. Evidence line (P6, dq KILL
clause; drawn-only for tt per the § 3d quoting note) from the
committed `panel_evidence_line_<face>.json`.

Run: .venv/bin/python -m experiments.explorations.task_hunt.diafaces.score_panel [tt|dq]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
PANELS = {"tt": "dial_real_ttrend_gpt2_l7",
          "dq": "dial_real_dqgap_llama31_8b_l14"}
POOLED = ("txc_batchtopk_pre", "stacked_batchtopk")
ALL_T = (2, 4, 8, 16, 32)


def _cells(rows, arch, kind):
    out = {}
    for r in rows:
        if r.get("arch") != arch or r.get("kind") != kind:
            continue
        out.setdefault(r["T"], []).append(r)
    return out


def _v(rs, key):
    vals = [r["metrics"].get(key) for r in rs
            if r["metrics"].get(key) is not None]
    return (float(np.mean(vals)), float(np.std(vals, ddof=1))
            if len(vals) > 1 else 0.0, len(vals)) if vals else (None, 0.0, 0)


def main():
    face = sys.argv[1] if len(sys.argv) > 1 else "tt"
    ds = PANELS[face]
    rows = json.loads((RES / f"stage2_{ds}.json").read_text())
    ev = json.loads((RES / f"panel_evidence_line_{face}.json").read_text())
    ev_r = {int(k): abs(v["pearson_r"]) for k, v in ev["per_T"].items()}
    print(f"panel {face} ({ds}): {len(rows)} cells")

    table = {}
    print(f"{'arch':<22}{'kind':<10}" + "".join(f"T{t:<9}" for t in (1,) + ALL_T))
    for arch in ("batchtopk_sae", "tsae", "txc_batchtopk_pre",
                 "txc_batchtopk_post", "stacked_batchtopk"):
        for kind in ("trained", "untrained"):
            cs = _cells(rows, arch, kind)
            if not cs:
                continue
            line = f"{arch:<22}{kind:<10}"
            for T in (1,) + ALL_T:
                if T in cs:
                    m, sd, n = _v(cs[T], "lambda_recovery")
                    table[(arch, kind, T)] = (m, sd, n)
                    line += f"{m:+.3f}({n}) " if m is not None else "   --     "
                else:
                    line += "   --     "
            print(line)
    print("\nv2 (conversation-grouped) for trained arms:")
    for arch in ("batchtopk_sae", "tsae") + POOLED + ("txc_batchtopk_post",):
        cs = _cells(rows, arch, "trained")
        vals = {T: _v(rs, "lambda_recovery_v2")[0] for T, rs in sorted(cs.items())}
        print(f"  {arch:<22}" + " ".join(
            f"T{T}:{v:+.3f}" if v is not None else f"T{T}:--"
            for T, v in vals.items()))
    print("\nrealized l0/token (trained):")
    for arch in ("batchtopk_sae", "tsae") + POOLED + ("txc_batchtopk_post",):
        cs = _cells(rows, arch, "trained")
        vals = {T: _v(rs, "l0_per_token")[0] for T, rs in sorted(cs.items())}
        print(f"  {arch:<22}" + " ".join(
            f"T{T}:{v:.2f}" if v is not None else f"T{T}:--"
            for T, v in vals.items()))

    sae = table.get(("batchtopk_sae", "trained", 1), (None,))[0]
    print(f"\nsae trained baseline: {sae}")
    best = None
    for arch in POOLED:
        for T in ALL_T:
            m = table.get((arch, "trained", T), (None,))[0]
            if m is None or sae is None:
                continue
            marg = m - sae
            if best is None or marg > best["margin"]:
                best = {"arch": arch, "T": T, "v1": m, "margin": marg}
    if best:
        v2, _, _ = _v(_cells(rows, best["arch"], "trained")[best["T"]],
                      "lambda_recovery_v2")
        un = table.get((best["arch"], "untrained", best["T"]), (None,))[0]
        tsae = table.get(("tsae", "trained", 1), (None,))[0]
        p1 = best["margin"] >= 0.02 and best["T"] in (8, 16, 32)
        m_hi = [table.get((a, "trained", T), (None,))[0]
                for a in POOLED for T in (16, 32)]
        m_lo = [table.get((a, "trained", T), (None,))[0]
                for a in POOLED for T in (2, 4)]
        p2 = (max(x - sae for x in m_hi if x is not None)
              > max(x - sae for x in m_lo if x is not None))
        p3 = (tsae is not None and sae is not None
              and sae <= tsae <= best["v1"])
        p4 = un is not None and best["v1"] and un <= 0.5 * best["v1"]
        p5 = v2 is not None and v2 > 0
        p6 = abs(best["v1"]) > ev_r.get(best["T"], float("inf"))
        print(f"\nbest pooled trained: {best['arch']} T{best['T']} "
              f"v1 {best['v1']:+.4f} margin {best['margin']:+.4f} "
              f"v2 {v2 if v2 is None else round(v2, 4)} "
              f"untrained {un if un is None else round(un, 4)}")
        print(f"P1 margin>=+0.02 @T in 8/16/32: {p1} (CI: harness)")
        print(f"P2 margin larger at 16/32 than 2/4: {p2}")
        print(f"P3 tsae ({tsae}) between sae and best: {p3}")
        print(f"P4 untrained <= 0.5x trained: {p4}")
        print(f"P5 v2 > 0: {p5}")
        print(f"P6 |v1| {abs(best['v1']):.4f} beats evidence "
              f"|r| {ev_r.get(best['T'])} at T{best['T']}: {p6}"
              + ("  [tt: drawn-only, quoting note]" if face == "tt" else
                 "  [dq: KILL clause]"))
        keep = p1 and p5 and (p6 if face == "dq" else True)
        print(f"--> {'KEEP (pending harness CI + ratification)' if keep else 'NOT KEEP as scored'}")


if __name__ == "__main__":
    main()
