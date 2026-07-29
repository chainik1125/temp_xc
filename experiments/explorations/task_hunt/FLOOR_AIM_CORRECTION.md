# The hunt has been aiming at the wrong bar — the floor never looks at the text

**mac-c, 2026-07-29 01:02 BST.** $0, 0 pods. Produced while executing
`briefings/hunt-safety-gold-clew.md` (the clew literature sweep). It is
neither of that brief's two acceptance-gate items — see § 8 — but it
changes what a candidate has to look like, so it is posted before any
further sourcing.

---

## 1. The instrument fact, verified at source

`visible_evidence_floor` is fit on exactly two features
(`evalage/screen.py:357`, `_FloorBank.feats`, `screen.py:289-293`):

    sage_floor(event_first, T)        # gen4c_lib.py:115
    dose_window_count(event_mask, T)  # wave3_lib.py:170

and both are handed the **ground-truth event arrays** — `first, mask =
z["event_first"], z["event_mask"]` straight off the stream npz
(`screen.py:113`), passed into `_FloorBank(**fl)` at `screen.py:301`.

    sage_floor        = log2(1 + min(age_of_last_event, T+1))
    dose_window_count = # masked event-turn tokens in the trailing T view

**Neither feature reads a single token.** The floor is not a
*text*-visible-evidence baseline. It is a **window-computable function of
where the events are**.

## 2. What follows immediately

**"Per-token silent" does nothing against the floor.** It is a property
of the *text*, and the floor does not look at the text. Per-token
silence defeats the `tok` arm — that is, it helps clear the **gain**
bar.

But `hunt-safety-gold-clew.md` § 2 states the situation plainly:
`retryesc_gen` **cleared the gain bar on every leg** (+0.063…+0.069) and
**died on the floor, 3/3**. So:

> **The brief's operational definition of gold — safety-meaningful,
> per-token silent, trailing functional of sparse events — selects for
> the bar that was never binding.**

Two of its three criteria (silence, trailing-functional-of-events) aim
at gain. The floor is untouched by both, and the floor is what kills.

> ### ⚑ AMENDED 02:4x — half-superseded BY MY OWN LATER MEASUREMENT
>
> The sentence above is **right that silence aims at the gain bar** and
> **incomplete about what it does when it gets there.** The amplifier
> test (`21217087e`, 12 faces × 2 corpora, one identical pipeline) adds
> the direction:
>
> **Silence does not widen the gain — it aims at the gain bar and pushes
> the WRONG WAY.** `arm_excess` tracks `tok_excess` at Spearman
> **+1.000** (evalage) and **+0.943** (retryesc_gen), so suppressing the
> per-token signal suppresses the windowed arm with it. The one face
> that was genuinely per-token silent — `gap_last`, `tok_excess`
> **−0.0018** — returned `arm_excess` **+0.0125**, a fifth of the bar.
> **A per-token-silent face produced a silent window.**
>
> So the correct reading of § 2 is stronger than what I wrote: the
> criterion is not merely *aimed at a non-binding bar*, it is
> **actively counterproductive on the bar it aims at**. Our one KEEP
> has the **strongest** `tok` of the three candidates and fails the
> criterion outright.
>
> **Scope, adopting the hub's boundary (`7d8a8a18d`) because I had not
> drawn it:** this governs **the hunt's screening rule**. It is **not**
> a test of the paper's ambience claim — one model, one layer,
> age-family faces, two corpora. Flagged for Han, not concluded.

## 3. The three failures have ONE mechanism, not three diagnoses

`evalage`'s face is a balanced 3-class binning of **the age of the last
event**. `sage_floor` is **the censored age of the last event**.

**The floor is handed the label's own sufficient statistic.** That is
not a near-miss; inside the window it is the label exactly, and outside
it is censored. It is also precisely why `floor_excess ≈ P(event in
window)` — the identity I corrected at `d2320d274` — holds at all: the
floor is right whenever the event is visible in its horizon.

The same is true of the others. Any label of the form *"how long ago /
how many times, within the window"* is a deterministic function of
`(censored age, in-window count)`. ~~**Such a task cannot beat this
floor at any density**, and no amount of lexical silence changes that.~~

> ## ⛔ RETRACTED 01:09 BST — the struck sentence is FALSE, and our own KEEP refutes it
>
> **`sycgen` is a censored-age label and beats this exact `_FloorBank`
> on 3/3 models** (`sycgen/screen.py:268-298`, same two features):
> arm−floor **+0.1365 / +0.1396 / +0.1207**. Verified at source before
> accepting the refutation (hub `41281e0eb`).
>
> **The mechanism I missed — and had already derived.** `sage_floor`
> censors at **T+1** (`gen4c_lib.py:115`, *"older than my window"*), but
> the label is the **uncensored** age. They coincide **only where
> age ≤ T**, so the exploitable mass is exactly **`P(age > T)`** —
> windows the floor cannot resolve but whose activations still carry the
> event. My own `d2320d274` says `floor_excess ≈ P(masked token in the
> T+w window)`, i.e. *the floor is right exactly when the event is inside
> its horizon* — **the complement is the exploitable mass.** I corrected
> the team's version of this law four hours earlier and then wrote a
> universal denying it.
>
> **Density is therefore back at the centre, not out of it.** It was
> never the wrong axis — **the hunt tuned the wrong DIRECTION on it.**
> `retryesc_gen` *raised* density by shortening turns, which raises
> `floor_excess` and moves mass **into** the floor's reach. What is
> wanted is mass **beyond `T+w`** that the activations still retain.
> My own lever-3 run is the confirmation: `evalage` (w=13) holds
> `floor_excess` to 0.003–0.057 and both KEEP-shaped cells appear at
> T32/T64, while `retryesc_gen` (w=25) ran it to +0.275 and died 3/3.
>
> **It is a BAND, and its upper edge is unmeasured:** too recent and the
> floor resolves it, too old and the residual stream no longer carries
> it. The lower edge is computable today (`floor_excess`); **how far
> back activations retain a sparse event is not, and is a cheap $0
> measurement on cached acts.**

## 4. What would actually beat the floor

The floor's two features are blind to:

- **event TYPE** — `event_first`/`event_mask` are single indicator
  arrays; every event is interchangeable;
- **ORDER among distinguishable events** — the count is order-blind and
  `sage_floor` retains only the most recent;
- **event CONTENT / semantics.**

⚑ **Read § 3's retraction first.** Type-dependence is **one** route and
it is **not forced** — censored-age labels demonstrably clear this floor
(that is what `sycgen` is). What follows is an additional family, not a
replacement, and § 9 shows it clears the floor *vacuously* unless the
type is textually invisible.

⇒ **A label MAY instead depend on WHICH events occurred rather than on
WHEN or HOW MANY.** Shapes that are invisible to `(censored age, count)`
but are representable by a windowed code:

- *which of K event types was most recent* (K ≥ 2);
- *did type A precede type B inside the window*;
- *is the in-window type multiset {A,B} rather than {A,A}* at matched
  count and matched recency.

Note all three hold **event positions fixed** — so they are orthogonal
to density, which is the axis the hunt has been tuning.

## 5. ⚑ The fairness objection, which is real

A critic says: *"your task beats the floor only because the floor is
under-specified for it — a type-aware floor would eat it."* **Correct,
and it must be answered, not dodged.**

If the event types are **visible in the text**, the honest floor is a
type-aware one — a human reading the transcript sees which event
happened — and building the task that way is gaming the instrument
rather than beating it.

**The objection binds only when the type is textually visible.** So the
candidate family that survives it is one where the event's *type* leaves
no textual trace — and there the type-aware floor cannot be constructed
from visible evidence, because there is no visible evidence of type.

## 6. The shape that survives § 5 — and where the literature points

**Activation-space injection with K distinguishable concepts.** The
event is a hook-applied injection during the forward pass; the label is
*which concept was injected most recently* (or *how long since concept
A specifically*). Then:

- the event has **no textual trace at all**, so type-visibility is zero
  by construction and § 5's objection cannot be raised;
- giving the floor the injection **positions** (which it takes anyway)
  is *conservative* — a real monitor would not even have those;
- the label is not a function of `(censored age, count)`, so the floor
  sits at chance on type while the windowed arm has a real target;
- it is safety-meaningful in the sense the brief demands: *has this
  model been steered, and with what* is exactly what a monitor wants.

**This is not my invention — it is an existing literature protocol**,
which is the point of sourcing from the registry rather than from my own
head. Registry nodes (ranks, not scores):

| work | key | why |
|---|---|---|
| Latent Introspection: Models Can Detect Prior Concept Injections | `WQX34P7C` | the protocol itself: injection is a **prior** event detected later |
| Emergent Introspective Awareness in LLMs (tc-thread) | `RSHUFZ38` | **45 citers** via `works cited-by` — S2 cannot see this venue |
| Steering Awareness: Models Can Be Trained to Detect Activation Steering | `89XMKFMX` | detection-from-within, trainable |
| Steering Awareness: Detecting Activation Steering from Within | `KNW22PG7` | ditto, separate node |
| Activation Oracles: LLMs as General-Purpose Activation Interpreters | `MZR4QUQ6` | reading injected state as a task |
| RepIt: Representing Isolated Targets to Steer Language Models | `R88EFKJB` | **isolated** targets ⇒ K distinguishable concepts |
| Stateless Yet Not Forgetful: Implicit Memory as a Hidden Channel | `K4CXV49A` | trailing state with no visible carrier |
| Accumulating Context Changes the Beliefs of LMs | `9SMGIFBX` | accumulation over context |

## 7. Reproducing the sweep

    export CLEW_AGENT=mac-c CLEW_READONLY=1
    CLEW=~/research/tools/clew/.venv/bin/clew
    $CLEW stats
    $CLEW search "introspection" --json          # positive control, 20 hits
    $CLEW search "concept injection" --json      # 20
    $CLEW search "activation steering" --json    # 20
    $CLEW search "implicit memory" --json        # 11
    $CLEW search "steering detection" --json     # 20
    $CLEW similar WQX34P7C -n 12 --json          # specter2-cosine
    $CLEW works cited-by RSHUFZ38 --json         # s2-citation-graph, 45, cached 2026-07-10
    $CLEW similar --text "identifying which of several different concepts was
      injected into a model's activations earlier in the sequence …" --json
                                                 # local-specter2-text

**Envelopes propagated:** `cited-by RSHUFZ38` ran on `s2-citation-graph`
**cached 2026-07-10 16:30:59** — not live. `similar` hits carry no
per-hit `vec` tier in this build, so treat the ranking as approximate
and use **ranks, never cosine values**. No `--refresh`, no writes.

⚑ **My first pass at this sweep reported ZERO hits for every query,
including `introspection`.** That was a **parser bug in my own reader**
(I read `results`/`works`; clew returns `hits`), not an absence — and it
looked exactly like a clean negative. Under this brief's gate a swept
registry that returns nothing is a *deliverable* ("a reasoned negative on
the source itself"), so I was one step from publishing a false negative
with the queries attached to make it reproducible. Caught by disbelieving
a zero for `introspection` in a 1083-work interp registry and reading the
raw output. The reader now **refuses to report any zero unless a positive
control fires first** (`<scratch>/cq.py`).

## 8. What this is and is not

**It is not a screened candidate** — nothing has been generated or
measured, and no verdict is claimed. **It is not a reasoned negative on
the registry** — the registry answered well; § 6 is sourced from it.

It is a **correction to the aiming criterion**, and it is worth posting
before more sourcing because it invalidates the search key: ~~hunting for
per-token-silent tasks selects for the gain bar, and the gain bar has not
been the binding constraint since `retryesc_gen`.~~

> **⚑ AMENDED 02:4x — superseded by my own later measurement, and the
> replacement is a stronger statement, not a softer one.** Hunting for
> per-token-silent tasks does aim at the gain bar — **and it pushes the
> wrong way when it gets there.** `arm_excess` tracks `tok_excess`
> (Spearman **+1.000** / **+0.943**, 12 faces × 2 corpora), so silence
> suppresses the windowed arm along with the per-token probe; the one
> genuinely silent face returned `arm_excess` **+0.0125** against a
> +0.05 bar. **The search key is not merely mis-aimed — it selects
> against the thing the hunt is trying to find.** See § 3's amendment.
> Scope: the hunt's screening rule, **not** the paper's ambience claim.

## 9. ⚑⚑ CORRECTION TO § 8, 20 MINUTES LATER — the test I proposed is ARITHMETIC, and the "advantage" it would have shown is a TAUTOLOGY

§ 8 originally proposed: *hold event positions fixed, assign K=2
synthetic types, check the floor sits at chance on the type label.*
**Do not run that. It cannot fail, and I should have seen it before
posting.**

The floor's two features **do not take type as an input**. Given a type
independent of timing, no probe on them can exceed the base rate — this
is arithmetic, not a result. Measured anyway, because a claim of
"cannot" deserves a number (`n=40000`, `T=16`, density 0.02, oracle
majority per distinct feature-cell — an upper bound over *all* probes,
not one fit):

    distinct (sage_floor, dose_count) cells : 61
    ORACLE best accuracy on the type label  : 0.5105
    base rate                               : 0.5079

**0.5105 vs 0.5079 — chance.** A test whose outcome is fixed by the
shape of the inputs is not evidence, and running it would have produced
a confirmation with no content.

### And the part that matters more: clearing the floor this way is VACUOUS

If the floor structurally cannot express the label, then a type-based
task **clears the floor clause by construction**. That is the same shape
as "TXC beats pooled's shuffle gap" — the tautology the hub caught in
`briefings/sycgen-shuffle-sparsity-matched.md` § 1 — and I reproduced it
in § 4 of this very document while writing up why the hunt keeps
mis-aiming.

**So § 4 must be read with this attached:** a type-based label does not
*beat* the floor, it **escapes** it, and an escaped control is not a
passed control. The correct response is not "type labels are the
answer". It is:

> **If a candidate's label is type-based, the existing
> `visible_evidence_floor` is no longer a meaningful guard for it, and a
> TYPE-AWARE floor must be built — otherwise the screen has silently
> lost the clause that has been doing all the killing.**

### ~~Which is exactly what makes § 6 the only non-vacuous family~~ — ⛔ RETRACTED, see below

> **⛔ THIS HEADING IS RETRACTED (02:4x) and was still LIVE until
> `claim_sweep.py` found it.** I withdrew "type-injection is the only
> non-vacuous family" at `267c52efd` — **the ranking inverts under my
> own escape-vs-beat criterion.** The floor is *expressive* for timing
> labels and `sycgen` genuinely **beat** it (+0.121/+0.137/+0.140 on
> 3/3 models); a type label the floor cannot express is **escaped, not
> beaten**, and *an escaped control is not a passed control*. So the
> **timing** family is the one with a passed control, and the type
> family is the one that cannot produce a sound verdict until a
> type-aware floor exists.
>
> **The paragraph below is still correct about the two CASES** — the
> visible-type case does need a type-aware floor built, and the
> no-textual-trace case is where a visible-evidence floor at chance is
> the honest answer rather than an escape. **What is retracted is the
> ranking word "only".** § 6 is a *candidate* family that needs new
> apparatus, not the best one.
>
> ⚑ **Found by `scripts/claim_sweep.py`, not by me.** I amended §3, §4
> and §8 of this document by hand when I retracted the claim and
> **missed this heading three times** — it survived my own manual sweep,
> the hub's, and my re-read. That is the entire argument for the tool.

The two cases separate cleanly:

- **Types visible in the text** — a type-aware floor *can* be built from
  visible evidence, so it *must* be, and the candidate has to beat it.
  Clearing the current floor proves nothing.
- **Types with no textual trace** (activation-space injection, § 6) — a
  type-aware **visible-evidence** floor **cannot** be built, because
  there is no visible evidence of type to build it from. Here the floor
  sitting at chance is not an escape; it is the honest answer to "what
  could a text-only observer get?", which is: nothing.

**§ 6 is therefore not merely a promising family — it is the only one
identified so far where a type-based label clears the floor clause
legitimately rather than vacuously.**

### The real open question, restated

The floor is settled by arithmetic. What is genuinely unmeasured is the
**gain** side, and it is not obvious:

> **Does a WINDOWED arm beat a PER-TOKEN arm on a "type of most recent
> event" label?**

A per-token probe at position *t* reads an activation that has already
attended over the whole prefix, so it may well carry "the last event was
type A" without any windowing. That is precisely how `tok` has
outperformed expectations before. **That** is the measurement worth
running, it needs real activations rather than a simulation, and it
should carry a falsifier: if `tok` matches the windowed arm on a type
label, the type direction dies on the gain bar instead of the floor bar
and I report that.
