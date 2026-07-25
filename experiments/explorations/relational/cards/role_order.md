# Card — candidate 1b: labelled role by MARKER ORDER (the regime-3 form)

**Status: FROZEN, then AMENDED 2026-07-25 — § 2's central claim is WRONG and the
design is superseded by § 8. Nothing was ever run against it.**

> ### Dated amendment (2026-07-25, before any cell existed)
>
> **The error.** § 2 claims that a label depending on *which marker came last* is
> invisible to additive-over-position codes. It is not. A TXC-pre-style code is
> `Σ_τ f_τ(x_{t+τ})` with **position-specific** encoders, so a linear readout can
> weight an open-marker indicator by `+τ` and a close-marker indicator by `−τ` and
> read off which came first directly. Order is a *linear* functional of
> position-tagged features.
>
> **Why I should have caught it earlier.** I wrote exactly this objection when
> ranking candidate 6 (refusal→comply order flip): *"it is arrangement, not
> equality — additive codes with position-specific encoders read arrangement
> fine."* Then I built a card on the arrangement structure anyway. The
> additive-code theorem needs a **product** of indicators at two positions, not an
> ordering of them.
>
> **Consequence.** Prediction P2 ("window linear stays ≤ 0.60") would have been
> falsified by a *correct* additive code, and kill rule K3 would then have
> mis-fired as "stimulus defect" on what is really the theorem not applying. The
> design is superseded by § 8 below; § 2–§ 7 are kept unedited as the record of
> the mistake.

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

---

## § 8 — Superseding design: marker-type EQUALITY (the actually additive-blind form)

Frozen 2026-07-25, before any cell exists.

The additive-code theorem applies to a **product of indicators at two positions**.
So the label must be an equality between the *contents* of two positions, with
both positions' content balanced:

Take the two most recent boundary markers preceding the row and label

> `y = [ type(last marker) == type(second-to-last marker) ]`

| markers (in order) | y | structural meaning |
|---|---|---|
| `<document>` … `<document>` | 1 | nested — depth rose twice |
| `</document>` … `</document>` | 1 | depth fell twice |
| `<document>` … `</document>` | 0 | a complete block closed |
| `</document>` … `<document>` | 0 | one block closed, another opened |

Each of the two marker positions is `open` in exactly half the items, so both
marginals are flat. Reading `y` requires `Σ_k 1[type_i = k]·1[type_j = k]` — a
product — and **no linear readout of any per-position decomposition can compute
it**, at any capacity, at any offset weighting. That is the theorem's hypothesis
genuinely satisfied, unlike § 2.

Semantically this is the *parity component* of provenance: "is my nesting depth
where it was two boundaries ago". It is weaker as a safety story than labelled
role — it is a state-tracking latent rather than a provenance monitor — and that
demotion is the honest price of the correction.

**Predictions (frozen).**
- **P8.1** Per-token ≤ 0.55 at every layer: the row token and its context are
  matched and both marker types are balanced.
- **P8.2** Window **linear** ≤ 0.60 at every layer and `T`. Now load-bearing *for
  the right reason*: the theorem forbids it, whatever offset weighting is used.
- **P8.3** `nonlinear_residual > 3σ_null` at layer 0 in the stratum where both
  markers are in the window, by analogy with the two layer-0 positive controls
  already measured (+0.269 agreement window, +0.109 contradiction pair).
- **P8.4** **The residual does NOT survive past layer ~4.** Bracket/nesting-depth
  tracking is the canonical conversion case — `task_hunt/CANDIDATES.md` records it
  as DEAD (`D5`) — and all three labels measured today converted within 2–8
  layers. *I expect this to be killed*, and the value of running it is that it
  completes the atlas with a **parity/state** instance alongside the
  binding, consistency and provenance ones.
- **P8.5** Within-window shuffle destroys the residual, read against an ambient
  anchor (a shuffle gap grows with `T` generically).

**Kill rules.** K1/K2 as before. **K3 is withdrawn** — with the corrected design a
window-linear rise above 0.60 is *evidence against the theorem's applicability
here*, i.e. a finding about the code class, and must be reported as such rather
than dismissed as a stimulus defect.

**Mandatory arms.** The oracle-pair arm (the two marker positions only) is
required, per RECORD § 3's lesson: a null from a wide window MLP is not evidence
of absence.
