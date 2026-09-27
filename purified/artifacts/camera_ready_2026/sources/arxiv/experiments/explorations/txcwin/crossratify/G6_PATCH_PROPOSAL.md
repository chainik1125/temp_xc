# G-6 patch PROPOSAL — per-model self-audit in `report.py` (NOT applied; for Andrii's ack + mac-local ratification)

**Status: PROPOSAL ONLY** (overnight §6c). CROSSRATIFY.md § G-6 flags
this "for Andrii side by side"; nothing in `report.py` / `audit.py` is
modified by this file. Companion figure: `figs/vwin_decomposition_*.png`
(G-2/R-X2, same push).

## The defect (restating G-6 with the exact lines)

`report.py:379-383` renders the embedded self-audit by invoking

```
python3 -m experiments.explorations.txcwin.audit --pattern 'focus_*.json'
```

`audit.py:52-54` (`load_cells`) globs EVERY matching results file into
one flat cell list, and `find()` (`audit.py:70`) matches cells by
`(task, arch, T)` with **no subject-model / source-file key** — so the
gpt2 and 8B novelty runs pool into single 6-seed pseudo-cells spanning
two subject models. Consequence (G-6): the pooled block prints c3
"CLAIM SURVIVES (4.6σ, W8 WARN only)" while the 8B-only audit reports
"CLAIM CONTRADICTED" (G-4) — the report's green audit block masks the
one claim its own harness rejects.

## Proposed minimal patch (two independent layers)

**Layer 1 — report side (sufficient on its own).** Replace the single
pooled invocation with one audit block per results file:

```python
# report.py — audit section
for f in sorted((HERE / "results").glob("focus_*.json")):
    txt = subprocess.run(
        ["python3", "-m", "experiments.explorations.txcwin.audit",
         "--pattern", f.name], capture_output=True, text=True,
        cwd=str(HERE.parents[2]), timeout=120).stdout
    a(f"<h3>Self-audit — <code>{f.name}</code></h3>")
    a(f"<pre>{html.escape(txt)}</pre>")
```

Rendering both blocks side by side preserves the report's "everything
re-derived by code" promise and surfaces the 8B c3 contradiction
instead of averaging it away.

**Layer 2 — audit side (belt-and-braces).** `load_cells` stamps each
cell with its source file (`c["_src"] = f.name`), and `find()` refuses
to aggregate across sources:

```python
srcs = {c["_src"] for c in matched}
if len(srcs) > 1:
    raise SystemExit(
        f"AUDIT INVALID: cell ({task},{arch},{T}) pools {sorted(srcs)} — "
        "run per-file (subject models must not share pseudo-cells)")
```

Fail-loud is deliberate: a silent per-file default could still be
bypassed by a future multi-model file.

## Claims-amendment language (per CROSSRATIFY § 4: "a claims amendment
or a $5 seed top-up, not a redesign")

Option A (amendment, no compute): amend the 8B claim row to pin at
T=16, where the replication is robust, carrying the mandatory T=16
disclosure ("dictionary-vs-V-all-at-that-T": 8B post +0.507 ≈ 2× V-all
+0.247) and the G-4 note that the original T=8 pin had one sick seed
and a c3 contradiction under the per-model audit.

Option B (compute, ~$5): 8B seed top-up at T=8 to settle c3 at the
original pin; the per-model audit (this patch) then adjudicates with
no pooling ambiguity.

Either way the gpt2 claims are untouched (they survive the per-model
audit as-is), and the pooled 4.6σ number is retired from the report.

_Drafted-by: claude-fable-5 (mac-c), 2026-07-27 ~02:50 London. PENDING
Andrii's ack + team ratification; receipts rule applies before any
claims.jsonl edit._
