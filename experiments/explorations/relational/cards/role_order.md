# Card — candidate 1b: labelled role by MARKER ORDER (the regime-3 form)

**Status: FROZEN — specified, not yet run.** Committed as the next step so that
whoever runs it does so against predictions written before any cell exists.

Agent `runpod`, 2026-07-25. Motivated by the candidate-1a result in
[`../RECORD.md`](../RECORD.md) § 4.

---

## 1. Why 1a is not enough

Candidate 1a (labelled role under style matching) produced a **regime-2** result:
at layer 0, `T = 16`, per-token 0.477 vs window **linear** 0.664 (`g = +0.186`)
with a nonlinear residual of −0.013. The window wins because the `<document>`
marker is *literally inside the window*, and "is there an opening marker at some
position in my window" is an **additive** function of per-position features.

That is a genuine win over per-token decoding — and therefore over T-SAE, which
decodes per position — but it is **not** the theorem-protected separation. TXC-pre
and Stacked SAE read additive functions perfectly well, so 1a cannot answer
reviewer bbby's question about cross-position weight sharing.

## 2. The fix: matched marker multisets, differing only in ORDER

Make the label depend on **which marker came last**, with the window containing
exactly one opening and one closing marker in *both* classes:

| class | markers preceding the row, in order | row's true role |
|---|---|---|
| `inside` | `</document>` … `<document>` … **row** | DATA (untrusted) |
| `outside` | `<document>` … `</document>` … **row** | INSTRUCTION (principal) |

Both windows contain the multiset {one open, one close}. Only the order differs.
Consequently:

- **Per-token**: chance — the row's token is the same payload sentence, and style
  is balanced against the label as in 1a.
- **Any additive-over-position code**: a sum of per-position features gives the
  *marginals* — "an open marker occurs", "a close marker occurs" — both of which
  are identical across classes. Reading the label requires the **relative order**
  of two positions, which is a conjunction. This is the additive-code theorem's
  hypothesis, satisfied by construction, so **per-token SAE, T-SAE, Stacked, MLC
  and TXC-pre are all at a provable chance floor.**
- **TXC-post** (`u = σ(Σ_τ W^(τ) x_{t+τ})`, the paper's Eq. 1): its nonlinearity
  crosses positions, so a coincidence atom can fire on
  "open-at-earlier-offset AND close-at-later-offset". It is the only family in the
  panel that can.

Marker positions must be **jittered** by randomising the filler length between
them, otherwise the markers sit at fixed relative offsets and a position-tagged
additive code reads them from the marginals after all. This is the single most
important implementation constraint.

## 3. Implementation requirements (learned from three stimulus defects today)

1. **Record every marker's character offset** in the item, and select the
   analysis stratum *at gate time* by checking that the `T`-window contains
   exactly one open and one close. Do not assume it.
2. **Assert the probe token's local context is byte-identical** across classes.
   In 1a the payload's final `.` merged into a `.\n` token in one arm only, and
   layer 0 read the label at AUC 1.000 — a pure token-identity leak.
3. **Assert `distinct_texts == n_items`** and split by payload group. v1 of
   track A emitted 2,400 rows from 80 distinct texts and memorisation gave AUC
   1.000.
4. **Verify the marker-position distributions overlap** between classes: report
   the open-marker and close-marker offset histograms per class and require the
   two-sample AUC on each to be within 0.03 of chance.

## 4. Predictions (frozen, with reasons)

- **P1.** Per-token at every layer ≤ 0.55. The row token and its context are
  matched, and style is balanced.
- **P2.** Window **linear** ≤ 0.60 at every layer and `T`. This is the load-bearing
  prediction: it is what distinguishes 1b from 1a, and it is exactly what the
  additive-code theorem demands once the multisets match. If window-linear rises
  materially above chance, the jitter has failed and the markers are at readable
  fixed offsets — a stimulus defect, not a result.
- **P3.** `nonlinear_residual = window_MLP − additive ceiling` **exceeds 3σ_null**
  in the stratum where both markers are in the window, and is inside the null
  band where they are not. My central estimate is +0.15 to +0.35, by analogy with
  the layer-0 positive control on agreement (+0.269).
- **P4.** The residual **survives into mid-depth** (layers 8–16), unlike track A's,
  because the model has no strong incentive to maintain "which side of the last
  marker" as a per-position state — this is precisely the property prompt
  injection exploits. **This is the prediction most likely to be wrong**: in-quote
  state is bracket-family, which `task_hunt/CANDIDATES.md` records as DEAD by
  conversion. If P4 fails while P3 holds at layer 0, the honest verdict is the
  same as track A's — the instrument works, the model converts — and the atlas
  gains a third relational instance.
- **P5.** Within-window shuffling destroys the residual (`g_shuf > 3σ`), since the
  label *is* an order statistic. Note this must be read against an ambient anchor:
  a shuffle gap grows with `T` generically (`task_hunt/RECORD.md` § 2).

## 5. Kill rules

- **K1** per-token within 0.02 of the best window arm at every layer → KILL.
- **K2** `nonlinear_residual ≤ 3σ_null` in every cell → KILL.
- **K3** window-linear > 0.60 with matched multisets → **STIMULUS DEFECT**, not a
  verdict: fix the jitter and re-freeze. Do not report either way.

**KEEP** requires P3 at some layer ≥ 4, with the two-marker stratum exceeding the
one-marker stratum, and window-linear at chance in the same cell.

## 6. If it keeps

Then the panel is justified, and it is the panel the reviewers asked for: 6 archs
(`batchtopk_sae`, `tsae`, `stacked_batchtopk`, `mlc`, `txc_batchtopk_pre`,
`txc_batchtopk_post`) × `T` ladder × seeds {1, 2, 42} + untrained controls, at
matched `d_sae` and matched **realized** `l0_per_token` (TXC-post needs nominal
`k = k_pos·T`, per `task_hunt/RECORD.md` § 3c), through the canonical runner. The
predicted figure has five flat lines at chance and one rising with `T` — and the
flat lines are flat *for a proven reason*, which is what makes it an answer rather
than another small margin.

## 7. Scope

One model, one marker vocabulary, English templates. A keep would establish that
*this* provenance relation is unconverted in *this* model — enough to justify the
panel, not enough to claim generality.
