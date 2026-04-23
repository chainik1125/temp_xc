# venhoff_scrappy

Iteration-friendly, single-GPU, <10-min-per-cycle version of the Venhoff
MATH500 Gap Recovery experiment. Pattern adopted from Han's Phase 5.7
autoresearch setup (`experiments/phase5_downstream_utility/`): thin
orchestrator, append-only JSONL ledger, per-candidate commit rhythm,
agentic research log as state machine.

## Layout

```
experiments/venhoff_scrappy/
├── README.md                    (this file)
├── config.yaml                  scrappy defaults (n_tasks=20, coef=0.5, etc.)
├── run_autoresearch.sh          orchestrator — serializes cycles across candidates
├── run_cycle.py                 single-candidate pipeline driver (Phase 0→3 on scrappy budget)
├── autoresearch_summarise.py    compute Gap Recovery + Δ vs baseline + verdict + ledger row
├── candidates/
│   └── <candidate_name>.yaml    per-candidate config overrides (agent edits these)
└── results/
    ├── autoresearch_index.jsonl verdict ledger (append-only)
    ├── cycles/<candidate>/      per-cycle logs + grade output
    └── baselines/<arch>/        cached baseline Gap Recovery (reused across cycles)
```

## Quickstart

On a pod with the full venv + vendor set up:

```bash
# Sanity: train + eval the SAE baseline (one-time, ~8 min).
bash experiments/venhoff_scrappy/run_autoresearch.sh baseline_sae

# Propose a candidate: copy the baseline config and tweak.
cp experiments/venhoff_scrappy/candidates/baseline_sae.yaml \
   experiments/venhoff_scrappy/candidates/tempxc_sum_layer10.yaml
# edit the new yaml — change layer, reduction, coef, etc.

# Run the candidate (the orchestrator computes Δ vs its declared baseline).
bash experiments/venhoff_scrappy/run_autoresearch.sh tempxc_sum_layer10
```

Each invocation: runs the cycle → appends a row to
`results/autoresearch_index.jsonl` → commits + pushes.

## Metric

**Gap Recovery** = `(hybrid_acc - base_acc) / (thinking_acc - base_acc)`
computed on the scrappy slice (n_tasks=20). Paper-budget baseline:
Venhoff's SAE arch = 3.5% on the full 500-task set; our scrappy number
will be noisier (~±10 pp stderr at n=20) but comparable in sign.

## Verdict thresholds (tuned for n=20 noise floor)

| Δ Gap Recovery | Verdict |
|---|---|
| > +10 pp | FINALIST (promote to paper-budget run) |
| −10 pp to +10 pp | AMBIGUOUS (re-run at n=50 or pivot) |
| < −10 pp | DISCARD |

## Stop criterion

If 5 consecutive cycles produce Δ < +10 pp vs baseline without an
informative mechanism insight → stop, escalate to human for
re-hypothesis.
