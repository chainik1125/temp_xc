# WRITEUP §9 staging — R30 certification note (DRAFT for mac-local ratification)

**Status: DRAFT — overnight §6a. Not applied to WRITEUP.md (its footer
requires a matching receipts row; mac-local applies on ratification).
Scope note: the exhibit-side R30 note + Task-2/3 licence-format numbers
are already staged in mac-b's REBUTTAL_PACK (their ~01:10 LOG entry);
this file stages only the WRITEUP §9 text.**

## Proposed REPLACEMENT for §9 bullet 3 (the realized-sparsity note)

> - **Task 2's margins over the per-token SAE** carry a
>   realized-sparsity note: that baseline landed 4.1–4.7 active
>   features per token against a nominal 8. **R30 pinned the
>   mechanism: the shortfall is eval-time JumpReLU threshold pruning,
>   not train-time selection zero-picks — and it is bit-identical
>   under both activation compositions** (the btk-only re-run
>   reproduced every relu-mix twin to |Δrecovery| ≤ 2.2e-8 with
>   Δ realized-l0 = 0.0 exactly; 20/20 cells, fresh trains at the
>   frozen pin). The composition question is therefore CLOSED for
>   hunt-width comparisons: btk-only ≡ relu-mix wherever the positive
>   pre-activation pool is not exhausted and the tracked threshold is
>   ≥ 0. The shortfall numbers above stand as an architecture
>   property; the sensitivity check still passes; the temporal-SAE
>   comparison remains the clean one.

## Proposed ADDITIONAL §9 bullet (immediately after the one above)

> - **Where composition DOES matter (boundary + paper caveat).** The
>   identity has a measured boundary: when the positive pool thins
>   (small d_sae or deep per-window selection), the arms diverge in
>   the pre-registered direction (thin-pool diagnostic: realized l0
>   0.696→1.007 of nominal, recovery 0.2471→0.1805 relu-mix→btk-only),
>   and runpod-2's transfer flag notes arch-dependent JumpReLU
>   OVERfire at wide-d/small-k. None of the quoted hunt panels sit in
>   that regime. Separately, the PAPER architecture family is
>   TopK-then-ReLU (selection on raw pre-activations) — a different
>   mechanism that the R30 identity does NOT cover; the per-task paper
>   compositions and their consequences are pinned in
>   `COMPOSITION_AUDIT.md` (task_hunt), and the pods' overnight grids
>   carry the paper-task arms.

## Provenance for the ratifier

- R30 identity certificate: mac-a CALIB VERDICT FINAL (LOG ~a few
  entries before af2247d43's ratification; 20/20 cells, +20
  leaderboard rows at freeze `97fae183a`, 0 dups, fig
  `calib_relu_vs_btk`), ratified by mac-local (`ba8af7bf9`, "identity
  certificate; receipts 33 green").
- Thin-pool diagnostic: mac-a DIAG (`df1e7b417`) — divergence as
  pre-registered.
- Mechanism re-attribution first stated: mac-a CALIB PRELIMINARY
  (`00309362f`); paper-arch caveat: COMPOSITION_AUDIT §0/§11 (mac-c).
- runpod-2 threshold-transfer flag: their ~00:30 LOG entry; noted by
  mac-a (~01:30 entry).

_Drafted-by: claude-fable-5 (mac-c), 2026-07-27 ~01:50 London._
