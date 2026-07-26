"""AUDIT — check the agent's own claims against the raw results, mechanically.

I write each claim to `claims.jsonl` as a structured assertion. This script
re-derives every claim from `results/sweep_*.json` and reports agreement or
contradiction. The point is that a claim cannot survive in the report unless it
survives here, so overclaiming has to get past code rather than past me.

Claim schema (one JSON object per line in claims.jsonl):
    {"id": "c1", "kind": "win", "task": "switch_clock",
     "arch": "txc_batchtopk_post", "T": 4, "vs": "batchtopk_sae",
     "text": "the paper's TXC beats the per-token baseline on the switch clock"}

Checks applied to every "win" claim:
    W1 both cells exist, are trained, and are non-degenerate
    W2 the winner's skill exceeds the comparator's
    W3 the two 95% CIs do not overlap        (else: "within noise")
    W4 the winner beats its OWN untrained control by >= 0.02
    W5 realized code rate (l0) is within 2x of the comparator's
    W6 both were scored on the same number of rows
    W7 the winner is not suspiciously perfect (skill >= 0.999 -> memorisation
       suspicion, since these are real activations with noisy labels)

Checks applied to every "family" claim (e.g. "position-mixing beats additive"):
    F1 the named arch really belongs to the family claimed
    F2 the additive siblings really are below it, all of them

Structural checks run regardless of any claim:
    S1 no cell has l0 == 0 while reporting non-zero skill
    S2 no task has ALL architectures identical to 3 decimals (a sign the label
       is being read off something trivial, or the probe is broken)
    S3 every trained cell has a matching untrained control
    S4 CI width > 0 wherever n_test > 50

Run:  .venv/bin/python -m experiments.explorations.txcwin.audit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CLAIMS = HERE / "claims.jsonl"

ADDITIVE = {"batchtopk_sae", "tsae", "stacked_batchtopk", "txc_batchtopk_pre"}
MIXING = {"txc_batchtopk_post", "spectral_txc"}
PER_TOKEN = {"batchtopk_sae", "tsae"}


def load_cells(pattern: str = "sweep_*.json") -> list[dict]:
    cells = []
    for f in sorted(RESULTS.glob(pattern)):
        pl = json.loads(f.read_text())
        for c in pl["cells"]:
            c = dict(c)
            c["_file"] = f.name
            c.setdefault("trained", True)
            cells.append(c)
    return cells


def find_all(cells, task, arch, T, trained=True):
    return [c for c in cells
            if c["task"] == task and c["arch"] == arch and c["T"] == T
            and bool(c.get("trained", True)) == trained]


def find(cells, task, arch, T, trained=True):
    """Seed-aware: returns a synthetic cell holding the seed MEAN, the widest CI
    across seeds, and the per-seed min/max, so a claim is never judged on a single
    lucky seed."""
    cs = find_all(cells, task, arch, T, trained)
    if not cs:
        return None
    if len(cs) == 1:
        c = dict(cs[0])
        c["n_seeds"] = 1
        c["skill_min"] = c["skill_max"] = c["skill"]
        return c
    sk = [c["skill"] for c in cs]
    out = dict(cs[0])
    n = len(sk)
    mean = sum(sk) / n
    var = sum((x - mean) ** 2 for x in sk) / max(1, n - 1)   # sample variance
    out.update({"sd": var ** 0.5, "se": (var / n) ** 0.5,
                "skill": sum(sk) / len(sk),
                "ci_lo": min(c["ci_lo"] for c in cs),
                "ci_hi": max(c["ci_hi"] for c in cs),
                "skill_min": min(sk), "skill_max": max(sk),
                "n_seeds": len(cs),
                "l0": sum(c["l0"] for c in cs) / len(cs),
                "seed_spread": max(sk) - min(sk)})
    return out


def ci_overlap(a, b) -> bool:
    return not (a["ci_lo"] > b["ci_hi"] or b["ci_lo"] > a["ci_hi"])


def audit_win(cells, cl) -> list[tuple[str, str, str]]:
    """-> list of (check, verdict, detail); verdict in PASS / FAIL / WARN."""
    o = []
    w = find(cells, cl["task"], cl["arch"], cl["T"])
    v = find(cells, cl["task"], cl["vs"], cl.get("vs_T", 1))
    if w is None or v is None:
        return [("W1", "FAIL", f"missing cell: winner={w is not None} "
                               f"comparator={v is not None}")]
    if w.get("degenerate") or v.get("degenerate"):
        o.append(("W1", "FAIL", f"degenerate: {w.get('degenerate')} / "
                                f"{v.get('degenerate')}"))
    else:
        o.append(("W1", "PASS", "both cells present, trained, non-degenerate"))
    o.append(("W2", "PASS" if w["skill"] > v["skill"] else "FAIL",
              f"{w['skill']:+.3f} vs {v['skill']:+.3f}"))
    # W3 tests the difference of SEED MEANS against the seed-level standard
    # error. (An earlier version unioned the per-seed bootstrap CIs and asked
    # whether they overlapped -- that tests nothing about a difference of means
    # and wrongly contradicted a 6-sigma result.)
    sew, sev = w.get("se", 0.0) or 0.0, v.get("se", 0.0) or 0.0
    se = (sew ** 2 + sev ** 2) ** 0.5
    diff = w["skill"] - v["skill"]
    if se > 0:
        z = diff / se
        o.append(("W3", "PASS" if z >= 2 else "FAIL",
                  f"difference of seed means {diff:+.3f}, seed-level SE {se:.3f} "
                  f"-> {z:.1f} sigma"
                  + ("" if z >= 2 else " -- within seed noise")))
    else:
        ov = ci_overlap(w, v)
        o.append(("W3", "WARN" if ov else "PASS",
                  "single seed per arm; bootstrap CIs "
                  + ("overlap" if ov else "disjoint")))
    i = find(cells, cl["task"], cl["arch"], cl["T"], trained=False)
    if i is None:
        o.append(("W4", "FAIL", "no untrained control for the winner"))
    else:
        d = w["skill"] - i["skill"]
        o.append(("W4", "PASS" if d >= 0.02 else "FAIL",
                  f"over-init {d:+.3f} (init {i['skill']:+.3f})"))
    lw, lv = w.get("l0", 0), v.get("l0", 0)
    ratio = (max(lw, lv) / max(min(lw, lv), 1e-9)) if min(lw, lv) > 0 else 999
    o.append(("W5", "PASS" if ratio <= 2.0 else "FAIL",
              f"code rate {lw:.1f} vs {lv:.1f} (ratio {ratio:.2f})"))
    o.append(("W6", "PASS" if w["rows"] == v["rows"] else "WARN",
              f"rows {w['rows']} vs {v['rows']}"))
    # W8: the strictest honest test — the winner's WORST seed against the
    # comparator's BEST seed. If this holds, no seed choice can flip the claim.
    if w.get("n_seeds", 1) > 1 or v.get("n_seeds", 1) > 1:
        strict = w["skill_min"] > v["skill_max"]
        o.append(("W8", "PASS" if strict else "WARN",
                  f"worst winner seed {w['skill_min']:+.3f} vs best comparator "
                  f"seed {v['skill_max']:+.3f}"
                  + ("" if strict else " -- a seed choice could narrow this")))
        o.append(("W9", "PASS" if w.get("n_seeds", 1) >= 3 else "WARN",
                  f"seeds: winner {w.get('n_seeds',1)}, "
                  f"comparator {v.get('n_seeds',1)}"))
    o.append(("W7", "WARN" if w["skill"] >= 0.999 else "PASS",
              f"skill {w['skill']:.4f}"
              + (" -- suspiciously perfect on real activations"
                 if w["skill"] >= 0.999 else "")))
    return o


def audit_family(cells, cl) -> list[tuple[str, str, str]]:
    o = []
    arch, task, T = cl["arch"], cl["task"], cl["T"]
    fam = cl.get("family", "mixing")
    member = arch in (MIXING if fam == "mixing" else ADDITIVE)
    o.append(("F1", "PASS" if member else "FAIL",
              f"{arch} in {fam} family: {member}"))
    w = find(cells, task, arch, T)
    if w is None:
        return o + [("F2", "FAIL", "winner cell missing")]
    beaten, failures = [], []
    for sib in sorted(ADDITIVE):
        s = find(cells, task, sib, T) or find(cells, task, sib, 1)
        if s is None:
            continue
        (beaten if w["skill"] > s["skill"] else failures).append(
            f"{sib} {s['skill']:+.3f}")
    o.append(("F2", "PASS" if not failures else "FAIL",
              f"beats {len(beaten)}; NOT above: {failures or 'none'}"))
    return o


def structural(cells) -> list[tuple[str, str, str]]:
    o = []
    bad = [c for c in cells if c.get("l0", 1) == 0 and abs(c.get("skill", 0)) > 0.05]
    o.append(("S1", "PASS" if not bad else "FAIL",
              f"{len(bad)} cells with a dead dictionary but non-zero skill"))
    flat = []
    for task in sorted({c["task"] for c in cells}):
        sk = {round(c["skill"], 3) for c in cells
              if c["task"] == task and c.get("trained", True)}
        if len(sk) == 1:
            flat.append(task)
    o.append(("S2", "PASS" if not flat else "WARN",
              f"tasks where every arch scores identically: {flat or 'none'}"))
    miss = [f"{c['arch']}@T{c['T']}/{c['task']}" for c in cells
            if c.get("trained", True)
            and find(cells, c["task"], c["arch"], c["T"], trained=False) is None]
    o.append(("S3", "PASS" if not miss else "WARN",
              f"{len(miss)} trained cells without an untrained control"))
    zero = [c for c in cells if c.get("n_test", 0) > 50
            and c.get("ci_hi") is not None
            and abs(c["ci_hi"] - c["ci_lo"]) < 1e-9]
    o.append(("S4", "PASS" if not zero else "WARN",
              f"{len(zero)} cells with a zero-width CI"))
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="sweep_*.json")
    a = ap.parse_args()
    cells = load_cells(a.pattern)
    print(f"AUDIT over {len(cells)} cells from {a.pattern}\n")

    print("── structural checks ──")
    fails = 0
    for k, v, d in structural(cells):
        print(f"  {v:4s} {k}  {d}")
        fails += v == "FAIL"

    if not CLAIMS.exists():
        print("\nno claims.jsonl yet — nothing asserted, nothing to contradict")
        return
    print("\n── claim checks ──")
    for line in CLAIMS.read_text().splitlines():
        if not line.strip():
            continue
        cl = json.loads(line)
        print(f"\n  [{cl['id']}] {cl['text']}")
        if cl["kind"] in ("retracted", "note"):
            print("      (no checks: this entry records a retraction or a note)")
            continue
        checks = (audit_win(cells, cl) if cl["kind"] == "win"
                  else audit_family(cells, cl))
        for k, v, d in checks:
            print(f"      {v:4s} {k}  {d}")
            fails += v == "FAIL"
        verdict = "CLAIM SURVIVES" if all(
            v != "FAIL" for _, v, _ in checks) else "CLAIM CONTRADICTED"
        print(f"      => {verdict}")
    print(f"\n{fails} failed check(s) total.")


if __name__ == "__main__":
    main()
