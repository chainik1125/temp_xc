# KILL TRIAGE — signal-absent vs certification-kill ($0, mac-c)

**Directive: mac-local 00:2x LOG. Classify every kill so that
"geometric/vocabulary bar fired" ⇒ automatic harness-rebuild candidate,
and "signal absent" ⇒ stays dead. All numbers are receipts already in
the LOG/artifacts; nothing new was run.**

## 0. Correction first: `evalage` is NOT generating

The queue line says "evalage (generating)". **It is not.** Zero
generation was produced: vLLM would not install against the pod image's
torch, and I terminated both pods (API-verified, ~$0.85 actuals,
`f49a7e506`) rather than bill while debugging. Cards are frozen and
ratified; the backend switches to the MATS Claude API. Queue position
unchanged, state corrected.

## 1. The classification

| candidate | which bar fired | class | rebuild? |
|---|---|---|---|
| `retryesc` | unigram 0.689–0.716 vs 0.60 | **certification-kill (vocabulary)** | **YES — top of queue** |
| `sycpress` | doc-mean 0.995 + 35 events | certification-kill (identity + mass) | YES = `sycgen`, already queued |
| `dharm` | doc-len 155.6 tok; doc-mean 0.993; unigram 0.820 | corpus-too-short + content leak | rebuild only — **NOT a large-T candidate** |
| `msdose` / `_r1` | dose↔position ρ 0.962 / 0.838 | geometric, but **structural** | low priority — see § 3 |
| `warddebt` | no bar fired; I declined on kernel reach | window-arithmetic, **squeezed both ways** | **NO — see § 4** |
| `emoinst` | per-token 0.856 beat every window | **SIGNAL-ABSENT** | no |
| `reask_hr` | order 0/3, wd erased the arm (runpod-a's receipts) | **SIGNAL-ABSENT** | no |

## 2. Correction: `retryesc` is signal-UNTESTED, not signal-present

The hub reads `retryesc` as "signal present, certification failed". I
own those numbers and the stronger claim is not supported: **every one
of the 5/6 passing bands is LABEL-SIDE.** No probe was ever run — the
GPU stage was never reached. What we established is that the *label* is
exceptionally well-conditioned (censored-age floor exactly **0.500** at
every T, position 0.720–0.743, 2.7–3.4 M usable tokens, 4,993 events)
except for a vocabulary leak that generation designs out.

So: **rebuild it — the queue position is right — but `retryesc_gen`
enters as an UNTESTED candidate, not a rescued positive.** Whether a
window beats a per-token probe on agent-failure age is still an open
question, and the per-token baseline remains binding. Recording this so
nobody later quotes "retryesc had signal".

## 3. `dharm` cannot be screened at larger T

"Phenomenon plausibly real at larger T" does not apply here: the
documents are **155.6 tokens**, so T64 is 40% of a document and T128 is
most of one. There are **3 position strata in the entire corpus**. A
larger-T screen is not expensive — it is *undefined*. `dharm` is a
harness-rebuild candidate (generate decomposition chains with
controlled length and vocabulary) and nothing else. Its doc-mean 0.993
and unigram 0.820 are also content leaks, not pure geometry.

## 4. `warddebt` is squeezed from BOTH sides — I advise against the T64/T128 spend

The exploratory permission is sound in general and wrong for this one.
The kernel spans **≈154 tokens** (8 sentences × 19.2 tok):

- at **T ≤ 32** the window cannot compute the face — measured, the
  discharge half is invisible (0.00 % in-window at T=8/16, ρ(net,
  incurred-only) = **1.000000**);
- at **T ≳ 154** the window now *contains* the whole kernel — so the
  window-visible count computes the rate directly and the **floor
  solves it**, which is precisely how `oprate` died at a full 84-cell
  panel to this same baseline.

There is no T at which this face is both computable and not
floor-trivial. Spend the exploratory budget elsewhere.

## 5. Standing rule, as I'd write it

A bar-kill records which bar fired. **Geometric/vocabulary ⇒ rebuild
candidate. Signal-absence (a per-token baseline or an order/wd arm won)
⇒ stays dead.** And a third class the directive did not name, which
both `dharm` and `warddebt` fall into: **structurally unscreenable** —
no T exists where the face is both computable and non-trivial. Those
are rebuild-or-nothing, never exploratory-screen candidates.

_Recorded-by: claude-fable-5 (mac-c)_
