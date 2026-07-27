# Shared elicitation harness — SCOPE ESTIMATE (for Han's #7 decision)

**Author: `mac-c`. Written unprompted-but-anticipated: mac-local's
23:44 escalation lists "scope estimate needed from mac-c if
authorized". Producing it costs $0 and commits nobody — it exists so
option (a) can be priced rather than guessed. Design of record:
`TIERC_PIPELINE_DESIGNS.md` § 3.**

## 1. What it is (one paragraph)

A rollout driver that (a) runs a **frozen scaffold** over N items,
(b) records the **exact token positions** of every scaffold-inserted
event, and (c) writes the program's standard stream `.npz`
(`token_ids`, `doc_off`, `doc_split`, face arrays, `{face}_bin`,
`man_{face}_*`) so **every existing builder, screen and scorer consumes
it unchanged**. Nothing about the screening pipeline changes; only the
corpus source is new.

The point is not generation for its own sake. It is that **labels
become exact by construction** — the harness knows where it put the
events — which retires the judge from every candidate it serves and
sidesteps the D4 violation entirely.

## 2. What it unblocks (live consumers only)

| candidate | status | what the harness supplies |
|---|---|---|
| **`sycgen_age`** | **geometry-PASSED**, ratified `8a7c722b2` | `are_you_sure` run in its licensed generator mode, jittered scaffold already frozen in `sycgen_lib` |
| `commitdebt` (§ 10.2 #21) | designed | 3-stage private-intent / announcement / action protocol |
| `afgap` (#9) | designed | monitored vs unmonitored scenario scaffolds |
| `cotdiv` (#14) | designed, lowest priority | paired hint / no-hint generations |

**Dead consumers, listed so nobody re-counts them:** `msdose` re-entry
(killed twice — the construction is intrinsically position-like) and
`emoinst` (ran and was killed 07-24). The honest consumer count is
**four, not six**, and only `sycgen_age` has already cleared a gate.

## 3. Build scope

**Stage 1 — driver core (the reusable 80%).** Scaffold runner, exact
event-position recording, stream writer, resume/idempotence, cost
ledger. Frozen-config + receipt discipline as everywhere else.
*Estimate: ~1 focused day.* This is the only part that is genuinely
new engineering; everything downstream is a config.

**Stage 2 — `sycgen` scaffold on top.** Smallest real scaffold: seed
questions from the pinned `sycophancy-eval` repo, the frozen challenge
template, jitter parameters already committed in `sycgen_lib`.
*Estimate: ~2–3 hours.*

**Stage 3 — screen the generated corpus.** Existing hunt4-clone path,
unchanged. *Estimate: hours, plus GPU per the standard screen.*

**Stages 4+ — `commitdebt` / `afgap` / `cotdiv`** are additional
scaffolds against the same driver, ~half a day each, and should be
sequenced **only behind stage 3's verdict**.

## 4. Generation cost

Generating model ≠ probe model (the existing convention — WildChat text
is probed by our models), so generation goes to the API and the corpus
is then tokenized for gpt2 / gemma2 / llama31 as usual.

`sycgen` at the frozen scaffold: 400 conversations × ~8 exchanges,
2.79 challenges/conv ⇒ **≈ 7,600 completions**, ~1.7M output tokens,
with growing-context input dominating. Bulk on a haiku-class model per
the `emoinst` precedent.

**Pre-register a $40 cap; expect $10–25.** The one executed precedent
(`emoinst`) came in at ≈$12 against a $40 cap on 600 rollouts, and this
is the same shape.

## 5. Timing, stated honestly

This does **not** fit tomorrow's deadline. Stage 1+2+3 is realistically
**1.5–2 days** including freeze discipline and a screen — which places
it in the **Aug-3 amendment window**, exactly where mac-local put it.
Anyone hoping the harness fills slot #7 by morning should not.

## 6. The binding condition on any of it

**The per-token baseline runs FIRST on the generated corpus** — the
standing rule mac-local adopted from my `sycgen` result, and the exact
thing that killed `emotional_instability` (0.856 AUC at offsets 1–4,
window never better). If post-challenge capitulation language is
per-token readable, `sycgen_age` dies and stages 4+ should not be
built. **Stage 3 is therefore a genuine go/no-go, not a formality**,
and the estimate above should be read as "1 day to find out", not
"2 days to a KEEP".

## 7. My recommendation

**Option (c), weighted toward (b).** Authorize the harness for the
Aug-3 window because it is the only thing that reopens the hunt at
all — but plan the exhibit on (b), because slots #6/#7 delivering
**kills-with-receipts** is a defensible result *today*, and the harness
might honestly return another one. My lane closed at $0 with two kills,
a family filter and an exhaustion report; that is the prime directive
working, and it reads as strength if presented as what it is.

I have no stake in the harness being built — I proposed it, I am not
its owner, and I have flagged twice that funding it is not my call.

_Recorded-by: claude-fable-5 (mac-c)_
