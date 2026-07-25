"""Monitor — check every gate result against the frozen expectations E1–E6.

The dashboard *displays* expectations; this script *tests* them, so an anomaly is
detected by code rather than noticed by eye. It reads the stimulus balance files
and `results/gate_*.json`, evaluates each expectation, and prints (and optionally
writes into the dashboard state) any FIRED expectation with the observation that
fired it.

An anomaly here is a disclosure, not an embarrassment. The rule the program
works by: a fired expectation redirects the run — it is never reinterpreted into
agreement with the hypothesis.

Run:  python3 -m experiments.explorations.relational.monitor [--write]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LABELS = HERE / "labels"
STATE = HERE / "dashboard" / "state.json"

TOL_BALANCE = 0.03
DISK_FLOOR_GB = 12.0
VRAM_CEILING_GB = 60.0


def check_e1() -> list[dict]:
    """Label-side marginals flat, no duplicate texts."""
    out = []
    for f in sorted(LABELS.glob("*_stimuli.json")):
        b = json.loads(f.read_text()).get("balance", {})
        if not b:
            continue
        task = f.stem.replace("_stimuli", "")
        for key in ("auc_from_a", "auc_from_b", "auc_from_len",
                    "auc_from_nfiller", "auc_from_gap"):
            if key in b and abs(b[key] - 0.5) > TOL_BALANCE:
                out.append({"expectation": "E1", "what": f"{task}.{key}",
                            "observed": f"{b[key]:.3f}",
                            "verdict": "FIRED — stimulus leaks the label"})
        if b.get("n_distinct_texts") != b.get("n"):
            out.append({"expectation": "E1", "what": f"{task}.distinct_texts",
                        "observed": f"{b.get('n_distinct_texts')} of {b.get('n')}",
                        "verdict": "FIRED — duplicate texts invite memorisation"})
    return out


def check_gates() -> list[dict]:
    """E2/E3/E4 on every gate result file."""
    out = []
    for f in sorted(RESULTS.glob("gate_*.json")):
        payload = json.loads(f.read_text())
        task = payload["meta"]["task"]
        cells = [c for c in payload["cells"] if "per_token" in c]
        if not cells:
            continue

        # E3 — conversion: per-token at/above the best window arm everywhere.
        conv = all(c["per_token"]["value"] >= c["window_flat"]["value"] - 0.02
                   for c in cells)
        best_nlr = max(c.get("nonlinear_residual", 0) for c in cells)
        med_sig = sorted(c["three_sigma"] for c in cells)[len(cells) // 2]
        if conv and best_nlr <= med_sig:
            out.append({"expectation": "E3", "what": f"{task}: conversion",
                        "observed": f"per-token ties every window arm; best "
                                    f"nonlinear residual {best_nlr:+.3f} "
                                    f"vs 3σ {med_sig:.3f}",
                        "verdict": "CONFIRMED — converted latent, no grid "
                                   "should be spent"})

        # E2 — an equality label read by an ADDITIVE arm far above chance is
        # only legitimate if the model computed it; if the OUT stratum matches
        # the IN stratum, suspect the stimuli instead.
        for T in sorted({c["T"] for c in cells}):
            ins = [c for c in cells if c["T"] == T and c["stratum"] == "in"]
            outs = [c for c in cells if c["T"] == T and c["stratum"] == "out"]
            if not ins or not outs:
                continue
            gi = max(c["g"] for c in ins)
            go = max(c["g"] for c in outs)
            if go > 0.05 and abs(gi - go) < 0.02:
                out.append({"expectation": "E2/IN-OUT",
                            "what": f"{task} T={T}",
                            "observed": f"g_out {go:+.3f} ≈ g_in {gi:+.3f}",
                            "verdict": "FIRED — a window that cannot see "
                                       "constituent A should not gain; suspect "
                                       "a stimulus route, not binding"})

        # E4 — order claims need the anchor differencing, never raw shuffle gaps.
        shuf = [c for c in cells if c.get("g_shuf", 0) > c["three_sigma"]]
        if shuf:
            out.append({"expectation": "E4", "what": f"{task}: shuffle gap",
                        "observed": f"{len(shuf)} cells with g_shuf > 3σ",
                        "verdict": "NOTE — a shuffle gap grows with T generically; "
                                   "do not read as order without an ambient anchor"})

        # E5 — resources.
        peaks = [c.get("peak_vram_gb", 0) for c in cells]
        if peaks and max(peaks) > VRAM_CEILING_GB:
            out.append({"expectation": "E5", "what": f"{task}: peak VRAM",
                        "observed": f"{max(peaks):.1f} GB",
                        "verdict": "FIRED — above the 60 GB guard"})
        if payload.get("oom_events"):
            out.append({"expectation": "E5", "what": f"{task}: OOM",
                        "observed": f"{len(payload['oom_events'])} events",
                        "verdict": "DISCLOSED — see the OOM table"})
    return out


def check_disk() -> list[dict]:
    free = shutil.disk_usage("/workspace").free / 1e9
    if free < DISK_FLOOR_GB:
        return [{"expectation": "E5", "what": "free disk",
                 "observed": f"{free:.1f} GB",
                 "verdict": "FIRED — below the abort floor"}]
    return [{"expectation": "E5", "what": "free disk",
             "observed": f"{free:.1f} GB", "verdict": "ok"}]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="record fired expectations into the dashboard state")
    args = ap.parse_args()

    found = check_e1() + check_gates() + check_disk()
    fired = [f for f in found if f["verdict"] != "ok"]
    for f in found:
        mark = "  " if f["verdict"] == "ok" else "! "
        print(f"{mark}{f['expectation']:10s} {f['what']:34s} "
              f"{f['observed']:32s} {f['verdict']}")
    print(f"\n{len(fired)} expectation(s) with something to report, "
          f"{len(found)} checked.")

    if args.write and STATE.exists():
        s = json.loads(STATE.read_text())
        s.setdefault("monitor", {}).setdefault("auto_checks", [])
        s["monitor"]["auto_checks"] = found
        STATE.write_text(json.dumps(s, indent=1))
        print("recorded into dashboard state")


if __name__ == "__main__":
    main()
