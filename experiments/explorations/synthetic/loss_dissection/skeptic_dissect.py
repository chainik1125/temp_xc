"""Loss-dissection skeptic — fires ONLY on recovery-metric HELPS claims
(CARD § 7): helps-claims are the winner's-curse surface; capability-only
improvements (prediction ii) and negatives do not trigger it.

Judgment on ``claude-fable-5`` (ROLES["think"]); raw verdict persisted
pre-parse under ``records/<claim>/`` and NEVER re-rolled — the runner
refuses to run a claim whose raw verdict exists. Spend metered to the
program-wide ``expansion/results/spend.json``; session cap $5 enforced
here. Committed before first execution.

    export ANTHROPIC_API_KEY=$(cat /workspace/.tokens/anthropic_key)
    .venv/bin/python -m experiments.explorations.synthetic.loss_dissection.skeptic_dissect
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from explorations.synthetic.expansion.client import Judge, Meter, ROLES

HERE = Path(__file__).resolve().parent
SESSION_CAP = 5.0

RUBRIC = ("a_seed_noise", "b_cell_shopping", "c_graft_confound",
          "d_pipeline_confound", "e_metric_leak")

SKEPTIC_SYSTEM = """You are an adversarial reviewer of a preregistered \
architecture-ablation claim. A loss component of TXC-pro is claimed to HELP a \
recovery metric on a synthetic benchmark. Your job is to KILL the claim if it \
does not survive scrutiny. Fill this JSON kill-rubric (every item an object \
{"kill": bool, "reason": str}):

- a_seed_noise: is the effect explicable by seed noise (3 seeds, 2*SE margin \
+ 0.05 floor — check the reported D, SE, and cell counts)?
- b_cell_shopping: does the claim rest on post-hoc cell selection rather than \
the frozen >=2/9-cells rule?
- c_graft_confound: could the effect come from the reimplemented backbone \
rather than the component (check Gate B bridge status and the plain-reduction \
contract test)?
- d_pipeline_confound: could the sequence-mode data pipeline (vs the \
canonical window mode) produce the effect (check that the comparison is \
within the sequence-mode family, plain as reference)?
- e_metric_leak: is the metric normalized/oracle-bounded and free of \
capability-vs-recovery conflation (a capability gain marketed as recovery)?

Also include "overall": {"survives": bool, "summary": str}. JSON only."""


def _absolute_levels(bench: str, arch: str, metric: str) -> dict:
    """Per-cell absolute metric values (mean + per-seed) for one arm — gives
    the skeptic the [chance=0, oracle=1] context the deltas alone hide (e.g.
    an arm that is 'less below chance' is not extracting anything)."""
    rows = json.loads((HERE / "results" /
                       f"{bench}_dissect_grid_results.json").read_text())
    out = {}
    for r in rows:
        if r.get("ok") and r.get("kind") == "trained" and r["arch"] == arch:
            out.setdefault(f"T={r['T']} k={r['k_pos']}", []).append(
                r["metrics"].get(metric))
    return {k: {"mean": sum(v) / len(v), "seeds": v} for k, v in sorted(out.items())}


def _parse_json_object(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found")
    return json.loads(text[start:end + 1])


def skeptic_claim(judge: Judge, claim_name: str, card_md: str, summary: dict) -> dict:
    user = (f"## Frozen ablation card\n\n{card_md}\n\n## Claim under review\n\n"
            + json.dumps(summary, indent=1, default=float)
            + "\n\nFill the kill-rubric. JSON only.")
    text = judge.call("think", SKEPTIC_SYSTEM, user, max_tokens=4000,
                      tag=f"skeptic-dissect:{claim_name}")
    raw_path = HERE / "records" / claim_name / "skeptic_raw.txt"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(text)          # raw BEFORE parsing — never lost
    out = _parse_json_object(text)
    if not all(k in out for k in RUBRIC):
        raise ValueError("skeptic verdict missing rubric items (raw persisted)")
    out["_judge_model"] = ROLES["think"]
    return out


def main():
    table = json.loads((HERE / "results" / "dissection_table.json").read_text())
    card = (HERE / "CARD.md").read_text()
    claims = []
    for bench, verdicts in table["verdicts"].items():
        for key, verdict in verdicts.items():
            comp, metric = key.split(":", 1)
            if verdict == "HELPS" and metric not in ("nmse", "eauc"):
                claims.append((bench, comp, metric))
    if not claims:
        print("[skeptic-dissect] no recovery-metric HELPS claims — skeptic "
              "not triggered ($0).")
        return
    meter = Meter()
    start = meter.spent
    judge = Judge(meter)
    for bench, comp, metric in claims:
        name = f"dissect-{bench}-{comp}-{metric}"
        raw = HERE / "records" / name / "skeptic_raw.txt"
        if raw.exists():
            print(f"[skeptic-dissect] {name}: raw verdict exists — never "
                  "re-rolled; skipping.")
            continue
        if meter.spent - start >= SESSION_CAP:
            sys.exit(f"[skeptic-dissect] session cap ${SESSION_CAP} reached "
                     f"(spent ${meter.spent - start:.2f}); remaining claims "
                     "NOT judged — report as such.")
        effects = table["benches"][bench]["effects"][f"{comp}:{metric}"]
        summary = {
            "claim": f"component {comp} HELPS {metric} on {bench}",
            "effects": effects,
            "absolute_levels": {
                "note": ("metric is normalized [chance=0, oracle=1]; judge "
                         "whether the variant EXTRACTS the latent or is "
                         "merely less-below-chance"),
                "plain": _absolute_levels(bench, "txc_post_plain", metric),
                comp: _absolute_levels(bench, f"txc_post_{comp}", metric),
            },
            "gate_b": table["gate_b"][bench],
            "untrained_guard": table["untrained_guard"][bench],
        }
        out = skeptic_claim(judge, name, card, summary)
        (HERE / "records" / name / "skeptic.json").write_text(
            json.dumps(out, indent=1, default=float))
        kills = [k for k in RUBRIC if out.get(k, {}).get("kill")]
        print(f"[skeptic-dissect] {name}: kills={kills or 'NONE'} "
              f"survives={out.get('overall', {}).get('survives')}")
    print(f"[skeptic-dissect] session spend ${meter.spent - start:.2f} "
          f"(cumulative ${meter.spent:.2f})")


if __name__ == "__main__":
    main()
