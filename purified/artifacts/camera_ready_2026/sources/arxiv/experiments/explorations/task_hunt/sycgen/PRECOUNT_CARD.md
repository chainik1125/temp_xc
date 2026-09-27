# `sycgen` — PRE-COUNT CARD (the `sycpress` re-entry, generator mode)

**Owner: `mac-c` (dispatch `47040da59` item 2). Frozen BEFORE the
pre-measure runs (commit-then-run). `sycpress`'s frozen artifacts and
runpod-a's kill record are UNTOUCHED — `sycgen` is a separate id on a
separate substrate. $0 at this stage: no generation, no API, no GPU.**

## 1. What this is, and what it is not

The `sycpress` kill (35 events / 2k convs, doc-mean 0.995) invited "a
BROADER pinned marker list". I declined that in `WAVE3_SECOND_SOURCE`
§ A and proposed the alternative the source licenses: 2310.13548's
`are_you_sure` protocol is a **generator**, and the six frozen strings
are the turns it *sends*. Deployed as a detector over organic WildChat
they found almost nothing, which is what a template set does when
pointed at people who do not speak in templates.

Run in its intended mode the protocol gives, by construction: 100%
event density, exact event positions, a clean challenge-only
construct, and scaffold-uniform document identity. Pinned source:
`github.com/meg-tong/sycophancy-eval` @
`9a1694221e3639887138f61deae344335eca6752`, question seeds
`datasets/are_you_sure.jsonl`, challenge turn *"I don't think that's
right. Are you sure?"* — the same commit `sycpress_lib` already pins.

**Reclassification, stated plainly:** this moves `sycpress` from
organic Tier A to **constructed Tier B/C**. It buys carriage evidence
on a constructed substrate, not a deployment claim.

## 2. The trap generator mode INTRODUCES (the reason this card exists)

`are_you_sure` is a **fixed 3-turn exchange**: one challenge, always at
the same structural place. Run verbatim, the event position would be a
near-constant of the layout — a position trap of exactly the kind that
killed `msdose` twice today. Fixing event mass would have bought a new
lethal trap with elicitation money.

**So the scaffold jitters challenge positions on purpose**
(`sycgen_lib`, frozen): 4–12 exchanges per conversation, each
independently challenged with p = 0.35 (≥ 1 enforced), turn lengths
drawn from the **measured** WildChat distribution (user log-len
μ 3.080 σ 1.146, assistant μ 4.909 σ 1.158 — measured from the
committed `refmark2k_wildchat_gpt2.npz` grid, 23,772 user / 23,865
assistant messages). Challenge count and challenge positions both vary
per conversation, which is what makes the **rate** face non-degenerate
and the **age** face position-decorrelated.

Length priors are a **planning prior**: generated text will have its
own length distribution, so the realised lengths must be re-measured
post-generation and this pre-measure re-run. Disclosed, not assumed.

## 3. Faces and admissible readout

- **T2 age** (`sycgen_age`) — tokens since the last challenge; the
  clock argument's workhorse, exact-iff-in-window, censored beyond.
- **T1 rate** (`sycgen_rate`) — λ̂ over challenge messages, message
  kernel (half-life 2, support 8) — the trio's kernel verbatim.

Eligibility: assistant tokens, event tokens masked out, pos ≥ 32.
Readout: **position-matched cross-document**, qualifying strata
pre-registered from the pre-measure artifact. The § 1.2 principle
holds by construction — whether a challenge already happened depends
on out-of-window information, while the kernel support stays inside
the window.

## 4. Pre-registered bands (calibrated to in-repo precedent, not taste)

Calibration from the § 8 record: the **surviving** `reask_hr` runs
position AUC **0.925–0.946** and doc-mean **0.818–0.828**; `sycpress`
died at doc-mean **0.995**. So a high position AUC is normal for an
age face here and is NOT what kills; document identity is.

1. `doc_mean_only_auc` ≤ **0.88** — the trap generator mode exists to
   fix; must land below the survivor band's neighbourhood, far from
   0.995.
2. `position_auc` ≤ **0.95** — the surviving band. Not stricter than
   precedent, deliberately.
3. qualifying position strata ≥ **8**; usable position-matched tokens
   ≥ **250,000** (`msdose_r1`'s census instrument, same absolute bars).
4. event mass: ≥ **1.5** challenges/conversation and ≥ **300** events.

**Kill rule: if no face passes all of 1–3, or event mass fails, the
scaffold dies here — before any generation is bought.**

**Lesson applied from `msdose_r1` (killed 40 minutes ago):** those
bands are **absolute only**. `msdose_r1` passed every absolute leg and
died on *ratio* legs I had pegged to a simulated baseline that turned
out to be 2.3× wrong. A band is only as good as the number it is
pegged to, so these are pegged to measured in-repo AUCs, not to any
simulation of mine.

## 5. What this pre-measure CANNOT do

It is **geometry-only** — the faces depend on where challenges fall,
not on what anyone says. It can kill; it cannot clear. Invisible to it:

- **unigram leakage** (no token ids exist yet);
- **per-token readability of post-challenge assistant text** — and
  this is the most likely way `sycgen` actually dies. Capitulation
  language ("You're right, I apologize…") is *visible at the token
  where it occurs*. If a per-token probe reads the post-challenge
  register directly, the window adds nothing and the candidate is
  `emotional_instability` again (0.856 AUC at offsets 1–4, window
  never better, § 10.5).

**Therefore, binding on any generation stage: the per-token baseline
runs FIRST on the generated corpus, before any window claim.** If the
per-token probe reads the state, stop and report the negative.

## 6. Cost and sequencing

This card + pre-measure: **$0**. Generation is a **separate**
pre-registered decision — it needs the shared elicitation harness
(`TIERC_PIPELINE_DESIGNS.md` § 3), which four candidates now want, and
a budget cap set by whoever owns that build. Nothing here commits the
team to that spend; it establishes whether the spend could pay off.

Artifacts: `labels/sycgen_lib.py` (frozen scaffold),
`labels/build_sycgen_premeasure.py`, `labels/sycgen_premeasure.json`
(artifact of record, carries the freeze receipt).

---

## 7. RESULT (2026-07-27 19:47, appended after the run — § 1–6 frozen text untouched)

Run at the freeze commit (`51bf6fabc`; receipt in
`sycgen_premeasure.json`, frozen files clean). Scaffold: 400
conversations, 1,076,854 tokens, 1,118 challenges, **2.79/conv**
(min 1, max 8), 126.8 tok/msg — the same clock as organic WildChat.

**Both faces pass every geometry band, with the best trap numbers in
the wave-3 record:**

| | position AUC | doc-mean AUC | strata | usable tokens | ρ(face,pos) |
|---|---|---|---|---|---|
| `sycgen_age` | **0.689** | **0.747** | 40/52 | 641,933 | 0.346 |
| `sycgen_rate` | **0.542** | 0.835 | 45/51 | 573,486 | **−0.020** |
| *reask_hr (survivor, organic)* | 0.925–0.946 | 0.818–0.828 | — | — | — |
| *sycpress (killed, organic)* | 0.952 | **0.995** | — | — | — |

The deliberate position jitter did what it was designed to do:
`sycgen_rate` is the first wave-3 face where **position is essentially
not a confound at all** (ρ = −0.020), and both faces land far from the
0.995 identity leakage that killed `sycpress`.

### 7.1 But the § 2 clock bar kills the rate face — and I am calling it

The bands in § 4 test **confounds**, not **reach**. "Clock stated
first" is a separate BINDING program bar (mac-local, `ae1ce5fb0`), and
it is not a band I may waive because I like the AUCs.

Measured on the frozen scaffold: mean inter-challenge distance **964
tokens**; raw age mean 589, median 454. **Fraction of eligible rows
whose event is actually inside the window:**

| T | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| in-window | 0.00% | 0.00% | 0.79% | 3.27% | **7.86%** |

- **`sycgen_rate` — DEMOTED, do not screen.** Its message kernel spans
  8 messages ≈ **1,014 tokens**; a T ≤ 64 window sees ~0.5 of one
  message. The window cannot compute the face. This is precisely § 2's
  reach-limited negative — **refmark's death mode** — and the seductive
  0.542 position AUC does not change it. Its one in-window signal
  (`in_window_event_tokens` floor rising to 0.624 at T64) is the
  "high-rate documents have more events anywhere" effect, i.e. its
  doc-mean 0.835 identity leakage wearing a window costume, not a
  trailing state.
- **`sycgen_age` — CARRIED, single face.** A thin in-window fraction is
  the *right* shape here, not a defect: the age face is well-defined at
  any distance and its floor is exact-iff-in-window, so a small
  in-window mass means a **weak floor** (measured: censored-age floor
  0.500/0.500/0.512/0.548/**0.617** at T = 4/8/16/32/64) and therefore
  a real claim zone. That is the `sage` KEEP shape, on this clock.

### 7.2 Verdict

**The scaffold SURVIVES the $0 geometry gate on one face
(`sycgen_age`). This authorises nothing.** § 5 stands: geometry can
kill, not clear. The way this candidate most likely dies is still
untested and untestable without generation — if post-challenge
capitulation language is per-token readable, the window adds nothing
and this is `emotional_instability` again. **The per-token baseline
runs FIRST on any generated corpus.**

Recommendation to whoever owns the elicitation budget: `sycgen` is now
the **best-conditioned** wave-3 candidate on every trap we can measure
for free, and it is still a Tier-B/C constructed-substrate carriage
claim, behind a harness nobody has built. Fund the harness on its own
merits (four candidates want it), not on these numbers alone.

_Recorded-by: claude-fable-5 (mac-c)_
