# mac-local — STATUS · SNAPSHOT #5 (2026-07-28 ~22:0x BST)

**Supersedes SNAPSHOT #4 (16:1x).** Rewritten because the day turned:
a *delivered* exhibit is under challenge, and the challenge spread to
three sections. I am the hub — review, ratify, rulings, the binding
LOG, ledger oversight, and the handover surfaces
(`REBUTTAL_HANDOFF.md`, `REBUTTAL_CODE_GUIDE.md`,
`REBUTTAL_CELL_CENSUS.md`). **No compute of my own.**

---

# RESUME HERE

## The one paragraph

**Item 6 (sycgen) — our gold safety exhibit — was challenged by
Dmitry's agent and the challenge LANDS.** They ran pooled- and
stacked-SAE baselines we never ran; **TXC loses at every T** (.600 vs
pooled .633 at T16). **Their TXC column IS our column** — a
missing-baseline dispute, not a measurement one. **mac-c then found the
gap is STRUCTURAL: `tsae_btkonly` cannot run at T>1 by construction, so
in `probing`, `rlhf` and `em` every SAE baseline is per-token and TXC
is the only arm with window access.** mac-d is building a
**section-agnostic recovery-vs-budget FRONTIER** (hub ruling — not a
single "matched point", which does not exist). Deadline pressure is
off: the 13:00 BST window closed; **responses amendable to Aug 3**.

## Fleet

| agent | state | lane |
|---|---|---|
| **mac-local** (me) | hub | rulings, LOG, handoff surfaces |
| **mac-d** | **ACTIVE, 1 pod** | item-6 frontier: rebuild lost sycgen cache + 9 TXC retrains + pooled/stacked + k-sweep. Pre-spend ~$6, self-cap $25 |
| **mac-c** | **ACTIVE, 0 pods, $0** | AUTHORIZED next: the $0 per-section claim-read. Geometry lane parked behind it |

**⚑ 3 unattributed non-convention pods ($3.87/h)** — `mats-gap-code-h100`,
`tsae-paper-widthmatch-probing`, `tsae-paper-widthmatch-em`. **Never
touched** (house rule). **Han's call, still pending.** Also pending:
3-token rotation (`gh`, `hf_token`, `hf_token_datasets`).

## In flight

1. **mac-d — the frontier table.** T=8 row needs **no GPU** (the T=1 SAE
   survived on HF ⇒ pooled/stacked are eval-only) and started at once.
   T{2,4,16} need 9 retrains (~$3–4): **those TXC ckpts and the entire
   sycgen activations cache were LOST with pod-D**, never mirrored.
2. **mac-c — the $0 claim-read.** What do probing/RLHF/EM *actually
   assert*? Probing already says *"probe-budget-dependent, no monotone
   window win at any k"* and may be substantially inoculated. **RLHF and
   EM unread.** Size before anyone spends.

## Standing rulings (do not relitigate)

- **FRONTIER, not a matched point.** Constraining the SAE to TXC's
  per-window l0 forces 0.49 l0/token at T16 (degenerate); the arms'
  budgets scale differently in T, so matching one unmatches the other.
  Sweep k on both arms, plot vs **realized** `l0_per_window`, **keep
  the as-run points on the same axes**, verdict is **dominance**.
- **If TXC is dominated across the frontier, item 6 is a NEGATIVE** and
  we report it. Pre-registered before any number existed.
- **The harness is SECTION-AGNOSTIC** — likely needed for probing/RLHF/EM.
- **Accuracy outranks immutability for a deliverable** (btk re-render;
  mac-d executed `ff242b78` → `8d75ff3a`, only the T10 column moved,
  proven by per-T series diff).

## Quote-form guards now live in the handoff

- **§6 ⛔ block** — item 6 is NOT quotable as an architecture win.
- **⚑ block above §1+2** — probing/RLHF/EM support *"windowed TXC reads
  state per-token probes cannot"*, **not** *"TXC beats a windowed
  SAE"*; no such comparison exists for any of the three.

## Ops

    .venv/bin/python scripts/handoff_audit.py --self-test   # 8 checks + staleness sweep
    .venv/bin/python scripts/cell_census.py --write         # regen before quoting coverage
    .venv/bin/python scripts/gen_handoff_tables.py          # items 4 + 5 tables

**Monitor:** persistent origin watch, task `b4g16b81d`, 45 s poll, emits
on repeated fetch failure so **silence ≠ blind**. It had been **down for
hours** — the original was a one-shot background command that fired once
and exited, and I was learning about pushes from push *rejections*.
**Re-arm after any session restart.**

**Push recipe:** fetch → rebase → **marker-check every file the push
touches** → audit → push. LOG conflicts are append-both (origin's entry
first, then mine). A conflict in a *deliverable* file is
stop-and-resolve-by-hand.

## My own errors today, kept on purpose for calibration

1. Misread "local mac agents own RLHF" as a venue constraint.
2. Claimed mac-d sat on a pod "~20 minutes" — it was ~3.
3. Diagnosed "bootstraps running in SERIES" from a `ps` snapshot — wrong.
4. Reported "~2.5 MB/s" on a download that had moved zero bytes in 30 s.
5. Claimed a "2.5–3×" rebalance win from extrapolated walls; measured 1.4×.
6. Predicted 0.1–0.35 s/step; measured 0.066.
7. Pushed a conflict marker into `REBUTTAL_HANDOFF.md`.
8. Passed a handoff surface by hand that `handoff_audit.py` then failed.
9. **`nc` probe reported "PORT CLOSED" from a command-not-found** (no
   `timeout(1)` on macOS) — a verdict with no measurement behind it.
10. **"3 pods, $6.42/h"** — missed one; and my first API script printed
    "128 pods, $109.03/h" by summing over EXITED records.
11. **Propagated `floor_excess ≡ f` as exact at 2e-6** — verified against
    a *simulation*, never the screen's floor features. mac-c refuted it;
    the real window is **T+w**.
12. **Quoted "~9× budget gap" as a figure** when it was an upper bound
    derived from an assumption.
13. **Specified a budget-match that does not exist**; mac-d's arithmetic
    killed it before it cost pod hours.
14. **Let the origin monitor die and did not notice for hours.**

**The pattern in 9–14: I state derived quantities with the confidence of
measured ones.** The fixes the fleet converged on independently, and
which I should apply to myself first: **verify by instantiation, not by
reading**, and **a caveat in a JSON is not a control**.
