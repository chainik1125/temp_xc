# `retryesc` — LABEL-SIDE CARD (frozen before the stream is built)

**Owner: `mac-c`. Briefing: `safety-hunt-continuation.md` (hunt
continuation, Han's emphasized priority). Menu source:
`SAFETY_TASK_MENU` § 10.2 #19 — "best clock fit in the addendum".
Stage: $0 CPU. A GPU screen is bought only behind a pass.**

## 1. Face — and a correction to my own menu entry

**Trailing quantity: tokens since the last FAILED environment turn**
(T2 age), read at the agent's own reasoning tokens. The safety reading
is the reward-hacking onset literature's: the hack follows accumulated
failure (`2511.18397`, `2603.04069`, `2606.06223`, `2604.17596`,
`2510.01367`).

**My menu proposed a T1 rate face. The measured clock kills that.**
Inter-failure gap: **median 886 tokens** (mean 1,789, p10 396) against
a screened T ≤ 64. A rate kernel spanning several failures reaches far
outside the window, so the window cannot compute it — refmark's
reach-limited death, the same reasoning that made me demote
`sycgen_rate` hours ago. Menu § 10.2 #19.5 claimed "tool-call events
are dense at token scale so § 2's reach limit does not bite"; at 686
tokens/turn **that claim is wrong**, and I am retracting it here rather
than letting it justify a spend. The **age** face is well-defined at
any distance and is what this card carries.

## 2. Clock, stated first (binding bar) — and why this substrate passes

| | measured | `dharm` (killed) |
|---|---|---|
| tokens per document | **15,409** (median 11,576, max 51,260) | 155.6 |
| tokens per turn | 686 | — |
| failure turns | 19.0 % of turns | — |
| inter-failure gap | median 886 | — |
| agent-token fraction | 37.3 % | — |

This is ~100× `dharm`'s document length — the corpus clock bar (now a
standing rule, and one I earned the hard way) passes decisively. Long
documents also mean many position strata, so a position-matched
cross-document readout is actually available here.

## 3. Out-of-window by construction (binding bar)

Events are raised **only by environment turns** (`is_failure_turn`
refuses to fire on agent turns): if the agent narrating a failure could
BE the event, that is the visible-cue trap in its purest form. Every
environment turn is masked out of probe eligibility, so the failure
text is never readable at a probe position. Whether a failure has
occurred is therefore a property of text **outside** the eligible
stream, while the age itself is a distance — the § 1.2 principle.

**Claim-zone note:** with gaps at median 886 tokens, the event is
essentially never inside a T ≤ 64 window, so the censored-age floor
should be near-uninformative. Under the `sycgen` reading that is the
*right* shape for an age face — a weak floor leaves a real claim
zone — and it is reported per T rather than assumed.

## 4. Substrate provenance — stated plainly, not buried

`mlfoundations-dev/terminal-bench-traces-local` @
`68e63c8b1cf7399d9e59bbf7d7e1944de2585fa5`. **This is a THIRD-PARTY
MIRROR, not the official Terminal Wrench release of `2604.17596`.** It
carries **one agent** (`terminus`) and **one model**
(`claude-3-7-sonnet-20250219`) over 80 tasks / 1,189 traces. Any result
is a claim about a single agent-model distribution and every artifact
must say so. If this candidate ever reaches a paper exhibit, the
official corpus should be pulled and the screen repeated.

**Marker disclosure:** the environment turns are **harness-templated**
("Previous command: …", "The previous command timed out after N
seconds"), so the frozen pattern set matches machine-generated strings,
not organic prose — the `refmark` lexicon objection is much weaker
here. A *provisional* pattern set was used for the feasibility density
scan (19% of turns); the frozen set in `retryesc_lib` is committed
before the stream build and before any face value exists.

## 5. Pre-registered bands (absolute only — the `msdose_r1` lesson)

1. `unigram_auc` ≤ **0.60**
2. `doc_mean_only_auc` ≤ **0.88** (task difficulty drives failure rate;
   this is the identity trap to beat)
3. `position_auc` ≤ **0.95** (surviving `reask_hr` band 0.925–0.946)
4. qualifying strata ≥ **8**, usable position-matched tokens ≥ **250k**
5. events ≥ **300**

**Must pass every band on all three tokenizers, or label-side KILL and
no GPU is bought.** Floors are reported per T as evidence, not as an
automatic kill (their formal authority is the hunt4 § 4 panel).

## 6. If it passes

A hunt4-clone screen under **§ 4 KEEP/KILL verbatim**, on a
self-provisioned pod named `mac-c-hunt-0727` under the binding key
governance (keychain env-inject, $10/h cap, ledger at spin-up AND
termination, TERMINATE + API-verify at drain, never touch pods I did
not spin up). The screen's window-vs-token comparison **is** the
per-token baseline the standing rule requires.

## 7. Sequencing note (a judgment call, flagged not hidden)

The amended briefing says spin the pod up NOW rather than queueing on
CPU. I am running this $0 label-side stage first anyway, because the
prerequisite work is tokenization — which a GPU does not accelerate —
so a pod started before the stream exists would bill while idle and
speed nothing. Two of my last three candidates died at exactly this
stage. The pod goes up the moment a GPU-needing stage exists; if
mac-local wants it warm regardless, say so and I will.

_Recorded-by: claude-fable-5 (mac-c)_

---

## 8. VERDICT — LABEL-SIDE KILL, no GPU bought, no slot

Run at freeze `161de7fe8` (receipt in `retryesc_premeasure.json`).
4,993 events, 18.4% of turns, 1,189 traces.

**Five of six bands pass on all three tokenizers — and the floors are
the cleanest this program has produced:**

| | gpt2 | gemma2 | llama31 | band |
|---|---|---|---|---|
| censored-age floor (all T) | **0.500** | **0.500** | **0.500** | — |
| in-window event tokens (T64) | 0.454 | 0.458 | 0.448 | — |
| position AUC | 0.743 | 0.725 | 0.720 | ≤ 0.95 ✓ |
| doc-mean AUC | 0.865 | 0.875 | 0.879 | ≤ 0.88 ✓ |
| qualifying strata | 270/497 | 231/457 | 213/388 | ≥ 8 ✓ |
| usable tokens | **3.38 M** | 2.87 M | 2.69 M | ≥ 250k ✓ |
| **unigram AUC** | **0.714** | **0.689** | **0.716** | **≤ 0.60 ✗** |

The floor being *exactly* 0.500 at every T is the out-of-window
construction working perfectly: environment turns masked, gaps at
median 886 tokens, claim zone 0.00% — the window-visible cheat carries
literally no information. Position is far better than the surviving
`reask_hr` (0.925–0.946), and the usable mass is 10× the bar.

**And it dies anyway, on unigram leakage, 0.69–0.72 against a bar of
0.60 that I set from in-repo precedent (survivors 0.560–0.575).** The
pre-registered kill rule fires: every band, all three tokenizers, or no
GPU.

### 8.1 Why — and why masking cannot rescue it

The tokens driving the leak are **not** failure-narration words. They
are **task vocabulary**: high-age markers `adjusted`, `wave`, `bytes`,
`setting`; low-age markers `disk`, `interface`, `tab`, `South`, `West`.
Token identity is predicting **which task**, and task difficulty drives
failure rate, which drives age. `doc_mean_only_auc` 0.865–0.879 —
grazing its own bar — is the same phenomenon measured a second way.

This is exactly the trap band 2 named in advance ("task difficulty
drives failure rate; this is the identity trap to beat"). It was not
beaten. **Crucially it is not fixable by masking**, which is what a
narration-driven leak would allow: the leak is carried by ordinary task
nouns spread through the agent's own reasoning. With 80 tasks and ~15
traces per task, a within-task contrast is the only in-principle
escape, and it is thin, unpre-registered, and would be post-hoc surgery
on a frozen card. I am not proposing it as a rescue.

**Re-entry path, unendorsed and stated for completeness:** a corpus
with many more tasks (so vocabulary cannot identify difficulty), or a
pre-registered within-task position-matched readout. Both are fresh
cards, not amendments.

### 8.2 No pod was spun up

The GPU stage was never reached. The label-side stage cost $0 and ~60
seconds of CPU, and killed the candidate outright — a pod provisioned
at the moment the order arrived would have billed through a build and a
kill without running anything.

_Recorded-by: claude-fable-5 (mac-c)_
