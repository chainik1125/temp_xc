# SAFETY-RELEVANT TASK MENU — 16 ranked candidates

**Author: `mac-c`. Source: `briefings/safety-task-research.md` (2026-07-27
team meeting, Dmitry's ruling: the hunt must find SAFETY-RELEVANT tasks;
question-mark distance and turn-length trend are toys). Venue: mac-local
CPU + the `clew` registry. $0 GPU, no Modal spend, no API spend.**

**This is a RESEARCH INVENTORY, not a freeze and not a pre-registration.**
Nothing here is a card. Screens, cards, and verdicts stay with the hunt
executor (`runpod-a`) under the standard discipline. Every predicted
outcome below is a *prior*, written before any label-side pre-measure —
the program's own record shows priors invert (wave-2: `tretd_wt` and
`tret_wt` were the headline expectations and both died; `sage`, ranked
third, was the only KEEP 3/3).

**No number in this document was invented.** In-repo numbers are quoted
from named files (`WRITEUP.md` § 8, `refmark/CARD.md`, `hunt4w2` records).
Paper numbers appear only where I read the source text; those are marked
`[read]`. Everything else is an abstract-level motivational citation with
registry-verified metadata. See § 9.

---

## 1. How to read this menu

### 1.1 The recipe a candidate has to satisfy

A hunt-eligible task is a **trailing quantity**: a per-token-*silent*
state that accumulates or decays over context, built as an
offset-weighted functional of sparse events. The program's canonical
kernel is `filter_rate(events, support=64, hl=16)` — weights
`0.5^((d-1)/16)` over the preceding 64 tokens, current token excluded,
NaN below position 64 (`labels/hunt3_lib.py`).

The four things that kill candidates, in the order they kill them
(`WRITEUP.md` § 8):

| trap | what it is | who it killed |
|---|---|---|
| **visible-cue floor** | a window-computable statistic on the *visible* tokens matches the label | `oprate` (full panel), `refmark`, `drev` ($0), `xnov` |
| **identity** | the label is document/conversation identity, not state | `dialevel` (0.98 naive), `refmark` (doc-mean 0.966–0.968) |
| **position** | the label is readable off absolute position | triage-stage deaths |
| **anti-dup** (Spearman ≥ 0.8 vs an existing face) | a re-parameterisation of something already screened | `xret` (0.809–0.812 vs `tret`), `tempo` (−0.81…−0.83 vs `ttrend`) |

And the outcome that is not a death but is not the prize either:
**order-free / aggregation-matchable** — a real window gain that survives
shuffling, routed to breadth rather than order. That is where `tret`,
`sage`, `tret_py`, `nvtrend` and `chaz` all landed. Per `WRITEUP.md` § 7,
**dialogue is still the only substrate whose order the trained serving
uses**, which is a favourable fact for this menu: most safety-relevant
state is multi-turn dialogue state.

### 1.2 The decisive design principle (read this before designing any card)

`refmark` — refusal/deflection markers on WildChat — is the closest prior
attempt to a safety task in the whole program, and it died to *both*
named traps at once. The reason is structural and it generalises to
almost every naive safety face:

> **A safety marker is visible at the token where it occurs.** If the
> event indicator can be computed from the tokens inside the window, the
> window is just counting what a per-token probe already reads.

The faces that survived (`tret`, `sage`) share one property that
`refmark` lacks:

> **The event indicator itself depends on information outside the
> window, even though the kernel support sits inside it.** `tret`'s
> event is "this token is a return to an occurrence > 64 tokens back" —
> the token is present, the justification is not. `sage`'s event is an
> age, well-defined at any distance, whose visible floor is *exact iff
> the marker is in window and censored otherwise*.

**Every candidate below is scored on whether its event indicator is
out-of-window-by-construction.** A safety candidate whose events are
locally readable inherits `refmark`'s obituary and should be killed for
$0 rather than screened.

### 1.3 The four label templates (each has an audited in-repo precedent)

Reuse these verbatim rather than inventing functionals — mac-a's hunt4
libs are audited and wave-2 already proved they transplant cleanly.

| template | functional | when to use | precedent | precedent's outcome |
|---|---|---|---|---|
| **T1 rate** | `filter_rate(events, 64, 16)` | events dense at *token* scale (several per 64 tokens) | λ̂ backtracking, `tret` | § 3 positive / KEEP 2/3 breadth |
| **T2 age** | `log2(1 + age_since_last_event)`, NaN below support 64 | events sparse at *turn* scale (0–1 per window) | `sage` | KEEP 3/3, claim zone T ≤ 32 |
| **T3 dosage** | running count of in-context exemplars, kernel- or raw-weighted | constructed contexts where we set the count | — (new) | untested |
| **T4 pre-onset ladder** | binary at offsets D ∈ {[1,4],[5,8],[9,16]} before a labelled onset token, far negatives, guard band | there is a discrete *event to anticipate* | `emotional_instability/CARD.md` readout (a) | designed, never run |

**T2 is the workhorse for this menu and the reason it is feasible at
all.** See § 2.

### 1.4 Ranking criteria

Rank integrates, in this order: (a) safety-relevance under Dmitry's
ruling; (b) ground-truth cost — rule-based ≫ constructed ≫
dataset-shipped ≫ judge-gated (the program's D4 rule: an exact zero-API
label beats a judge); (c) corpus availability — in-repo cache ≫ pinned
pull ≫ elicitation; (d) clock fit (§ 2); (e) trap survival odds against
the § 8 record.

---

## 2. The clock problem, and why the age face is the answer

This is the single fact that decides which safety tasks are runnable at
all, so it goes before the menu.

**Measured, in repo** (`refmark/CARD.md` § 2): WildChat runs
**125–144 tokens per message**. `refmark`'s 8-message kernel therefore
spanned ≈ **1,100–1,150 tokens ≈ 16× the top of the T = 64 ladder**. The
card recorded the consequence honestly before running: *"A T-window
usually sits INSIDE one message."*

Almost every safety-relevant state named in the meeting — refusal
pressure, jailbreak progression, sycophancy drift, persona drift — lives
at **turn** scale. Under a T1 rate face at support 64, such events
contribute ~0–1 counts per window: the face is mostly zero, and the
screen measures noise. **That is the reach-limited negative that killed
`refmark`, and a naive port of any seed direction in the briefing walks
straight back into it.**

Three ways out, all of them already validated somewhere in the program:

1. **T2 age instead of T1 rate.** An age is well-defined no matter how
   far back the event is; there is no "no events in window" degenerate
   case. Its visible floor is *exact iff the marker sits in window and
   censored at T+1 otherwise* (`gen4c_lib.sage_floor`, proved in
   `tests/test_gen4c_labels.py`), so a genuine claim zone exists at
   T ≤ 32 — exactly where `sage` scored KEEP 3/3 (+.105/+.093/+.087 over
   floors 0.417–0.423). **For turn-scale safety events, use T2.**
2. **Restrict probe positions to early-in-turn strata** so the preceding
   speaker's turn falls inside the window. The manifest builder already
   supports position strata (`punctint_lib.pos_strata`, `min_pos 16`).
   This buys back one turn of reach without touching the ladder.
3. **Constructed corpora with token-scale events** (T3): if we build the
   context, we choose the event spacing. Costs elicitation, buys clock.

**Corollary for `runpod-a`:** a safety card that proposes a T1 rate face
over turn-scale events, without the clock measurement stated first, is
repeating `refmark` and should be bounced at review.

---

## 3. The menu at a glance

Tier **A** = screenable this week (labels rule-based, corpus in-repo or
one pinned pull, $0 API). Tier **B** = one corpus pull plus modest
pipeline, still no judge. Tier **C** = elicitation- or judge-gated.

| # | id | one line | tier | template | prior (NOT a pre-registration) |
|---|---|---|---|---|---|
| 1 | `sycpress` | age/rate of user pushback, probed at assistant tokens | A | T2+T1 | screens; order plausible (dialogue) |
| 2 | `reask` | age since a request was re-issued after a deflection | A | T2 | screens at T ≤ 32; identity trap is the risk |
| 3 | `dharm` | subtask-count into a decomposition attack | A | T3+T2 | strongest safety story; floor risk low |
| 4 | `msdose` | many-shot exemplar dosage so far | A | T3 | position trap is lethal unless decorrelated |
| 5 | `sysage` | distance since the last system-prompt anchor | B | T2 | `sage` twin — expect KEEP, order-free |
| 6 | `toolpriv` | trailing rate of irreversible/high-privilege tool calls | B | T1 | best token-scale clock fit on the menu |
| 7 | `rhonset` | pre-onset anticipation of a reward hack | B | T4 | highest-value if labels land |
| 8 | `histakes` | accumulation of stakes-raising evidence | B | T1/T2 | probe literature says per-token-readable |
| 9 | `afgap` | trailing deployment-context evidence (monitored vs not) | B | T2 | label is constructed ⇒ clean; corpus costly |
| 10 | `emoinst` | pre-onset emotional escalation | C | T4 | **card already frozen in-repo**; needs ≤ $40 |
| 11 | `lhdec` | consistency debt from earlier claims | C | T2 | conceptually the best fit; labels hardest |
| 12 | `evalcue` | evaluation-awareness cue accumulation | C | T1 | likely $0 kill — see trap note |
| 13 | `emdrift` | harmful-advice drift inside a long response | C | T1 | circularity trap must be resolved first |
| 14 | `cotdiv` | CoT-faithfulness divergence trailing state | C | T4 | needs paired counterfactual runs |
| 15 | `sandbag` | accumulated withholding pressure | C | T2 | depends on `evalcue` surviving |
| 16 | `persuade` | persuasion-pressure belief state | C | T1 | weakest ground truth |

Plus § 7: four candidates I recommend **killing at design review for $0**,
with the precedent each one repeats.

---

## 4. Tier A — screenable this week

### 1. `sycpress` — sycophancy pressure from user pushback

1. **Trailing quantity.** Two faces off one event stream: **T2** age
   since the last user-pushback event (log2(1+age)), and **T1**
   kernel-weighted pushback rate. Probed at **assistant** tokens. The
   state is "how hard, and how recently, has this user been pushing
   back" — the quantity that precedes capitulation.
2. **Safety motivation.** Sycophancy is a first-class, measured
   post-training pathology: `2310.13548` *Towards Understanding
   Sycophancy in Language Models* (alignment-blog); it escalates into
   reward tampering (`2406.10162` *Sycophancy to Subterfuge*,
   alignment-blog); it has a mechanistic story shared with lying
   (`2604.19117` *LLMs Know They're Wrong and Agree Anyway: The Shared
   Sycophancy-Lying Circuit`); it *causes* emergent misalignment
   (`2606.09068` *EM Can Be Induced by Sycophancy and Reversed via
   Alignment Gating*) — which puts it squarely inside Dmitry's
   safety-relevant class rather than adjacent to it. User-side response
   patterns are documented in `2601.10467`; researcher-directed
   sycophancy in `2606.08629`; mitigations/masking in `2603.16643`,
   `2411.15287`.
3. **Label construction.** Corpus: **`refmark2k_corpus.json.gz`,
   already in repo** — 2,000 WildChat conversations, English,
   2,000–24,000 rendered chars, `(role, content)` pairs preserved,
   pinned pull with receipt. Event: a **user** turn matching a frozen
   pushback-marker list, built the same way `refmark_lib` was built —
   substring set lifted from a published source at a pinned commit, no
   judge (`refmark_lib.py` documents this discipline explicitly: *"A
   judge would be the D4 violation; we do not use one"*). Ground truth:
   rule-based, zero API. Kernel: T2 primary (§ 2), T1 secondary.
4. **Expected traps.** (a) *Identity* is the named killer — `refmark` hit
   doc-mean 0.966–0.968 on this exact corpus and the card's rule was "no
   KEEP exists without the within-conversation control." **Mandatory wd
   arm, non-negotiable.** (b) *Visible floor*: pushback markers sit on
   user turns; at ~130 tok/message an assistant-token probe deep in a
   response cannot see them, so the floor should be weak — but it must be
   computed per T, and early-in-turn strata will leak. (c) *Anti-dup*
   vs `refmark`'s `is_marker` rate and vs `dialevel` — the 0.8 Spearman
   bar applies; `xret`/`tempo` are the precedent for killing on it.
   (d) *Clock*: § 2 in full force — T2 is why this is on the menu at all.
5. **Feasibility.** **Screenable this week.** Corpus, tokenizers,
   builder template (`build_hunt4.py`), and the wd machinery all exist.
   Label side is a ~1-day CPU job; no new corpus pull.

### 2. `reask` — refusal re-ask persistence

1. **Trailing quantity.** T2 age since the last **re-ask event**: a user
   turn that re-issues a request the assistant already deflected. Probed
   at assistant tokens.
2. **Safety motivation.** This is the token-level shadow of multi-turn
   jailbreaking, where refusal is worn down across turns rather than
   defeated in one prompt: *Many-shot Jailbreaking* (alignment-blog);
   `2411.07494` *Rapid Response* (alignment-blog); `2605.13334`
   *LLM-Based Persuasion Enables Guardrail Override in Frontier LLMs`;
   `2506.10949` `[read]` on why single-prompt defences are blind to
   intent that emerges over a sequence.
3. **Label construction.** Same in-repo WildChat corpus. Event = user
   turn *u* such that (i) the immediately preceding assistant turn fires
   `refmark_lib.is_marker_turn` (the frozen 12-substring list, sourced
   from `github.com/andyrdt/refusal_direction` @ `9d852fae…`), and
   (ii) *u* has high content-word overlap with the user turn that
   preceded that deflection. Both halves rule-based; overlap threshold
   frozen before measurement. **The event indicator is
   out-of-window-by-construction** (§ 1.2): deciding "is this a re-ask"
   requires a user turn two messages back, ≈ 260–290 tokens away, out of
   window at every ladder T.
4. **Expected traps.** (a) *Anti-dup vs `refmark`* is the first gate and
   a real risk — if re-ask rate tracks marker rate at ρ ≥ 0.8 it dies for
   $0 on the `xret` precedent, and that is a good outcome, cheaply
   obtained. (b) *Identity*: same 0.966–0.968 warning; wd mandatory.
   (c) *Event sparsity*: re-asks may be too rare for
   `WD_MIN_DOC_ROWS = 30`; measure event mass **before** anything else —
   this is the `tretd` starvation failure mode (SKIP by its own
   instrument, `WRITEUP.md` § 8).
5. **Feasibility.** **Screenable this week**, contingent on the event-mass
   pre-measure. Reuses `refmark_lib` verbatim.

### 3. `dharm` — decomposition-attack progression

1. **Trailing quantity.** T3 count of subtasks consumed so far in a
   decomposed request sequence, plus T2 age since the last subtask
   boundary. The state is *accumulated malicious intent* — by
   construction invisible in any single subtask.
2. **Safety motivation.** The strongest safety story on the menu, and
   the closest published analogue to "trailing safety state":
   `2506.10949` *Monitoring Decomposition Attacks in LLMs with
   Lightweight Sequential Monitors* `[read]` — a malicious goal split
   into benign subtasks defeats shallow alignment because defences
   *"only detect harm in the immediate prompt and do not reason about
   long-range intent."* Measured there: 87% average attack success on
   GPT-4o; in the agent setting refusal drops from 50% on the original
   task to 10% on the decomposed subtasks; a *sequential* monitor that
   accumulates across subtasks reaches 93% defence success. A
   cumulative-over-turns monitor beating a per-prompt one is the exact
   window-over-token claim this program tests, arrived at independently
   from the safety side. Related: `2605.13334`, `2411.07494`
   (alignment-blog), `2603.15714`.
3. **Label construction.** Corpus: **`YuehHanChen/DecomposedHarm` on HF**
   (released by the paper; QA / text-to-image / agentic splits). Pull
   under the `pull_pg19.py` new-corpus rules I own — exact re-pull
   script, **pinned revision**, funnel counters, first-doc sha256 identity
   receipt, label-free pull statement, idempotent short-circuit. Labels
   are **structural, shipped with the dataset** (which subtask index, of
   which decomposition, of which harmful goal) — no judge, no API.
   Contrast class: benign decompositions of benign goals, matched on
   subtask count and length.
4. **Expected traps.** (a) *Identity* is severe and different in kind:
   "which decomposition am I in" is a document-level constant, so the
   naive readout is `dialevel`'s 0.98 all over again — **the only
   admissible readout is within-decomposition** (position along the
   chain), and that must be stated in the card, not discovered at
   scoring. (b) *Position*: subtask index correlates with absolute
   position; requires length-jittered chains or explicit position
   matching, or it is a position probe wearing a costume. (c) *Visible
   floor*: subtask boundaries are visible markers; the floor probe on
   [boundary count in window] must be computed per T — but at T ≤ 64 a
   window holds ≲ 1 boundary, which is why the floor should be weak.
   (d) *Content leakage*: harmful-topic unigrams may make the class
   trivially separable — the unigram triage AUC gate (`type_mean_scores`)
   is the pre-measure that decides this, and it may kill the candidate
   for $0.
5. **Feasibility.** **One pinned pull from screenable.** Highest
   safety-value-per-dollar on the menu; I recommend it as the first
   corpus pull if `runpod-a` wants a new substrate for wave-3.

### 4. `msdose` — many-shot dosage

1. **Trailing quantity.** T3 count of harmful in-context exemplars
   consumed so far — the dose in a dose-response attack.
2. **Safety motivation.** *Many-shot Jailbreaking* (alignment-blog) is
   the canonical demonstration that attack success scales with the
   **number of in-context demonstrations**; `2411.07494` *Rapid Response*
   (alignment-blog) is the defence built on top of it. If a model
   maintains a running dosage state, that state is the natural monitoring
   target, and it is precisely a trailing count.
3. **Label construction.** **Constructed corpus, zero judge, zero API**:
   we build the contexts, so the exemplar count and every boundary
   position are known exactly. Use *benign* exemplar content for the
   label-side and screen stages — the face under test is dosage, not
   harm, and nothing about the construction requires harmful text to
   measure whether a running count is carried. (Harmful-content variants
   are an authorised-red-team question, not a label-side one.)
4. **Expected traps.** (a) **Position is lethal and must be designed
   against**: dosage-so-far is a monotone function of absolute position
   unless exemplar lengths are jittered enough to decorrelate them —
   randomise lengths and report the realised count↔position correlation
   *before* screening. Without that, this is a position probe. (b)
   *Identity*: total dose is a document constant ⇒ within-document
   readout only (dose *so far* does vary within document — that is the
   saving grace). (c) *Visible floor*: exemplar boundaries are visible,
   but at ~100+ tokens per exemplar a T ≤ 64 window holds ≲ 1 boundary.
   (d) *Ecological validity*: a constructed corpus proves carriage, not
   deployment relevance — state the limitation in the § 8 row.
5. **Feasibility.** **Screenable this week** (construction is a script,
   no pull, no judge). Lowest ground-truth cost on the entire menu.

---

## 5. Tier B — one pull plus modest pipeline, still no judge

### 5. `sysage` — system-prompt / instruction anchor age

1. **Trailing quantity.** T2 age since the last token governed by the
   system prompt (last instruction-anchor event). Literally `sage` with
   a safety-relevant marker definition.
2. **Safety motivation.** Instruction/persona drift over long context is
   a documented deployment failure: `2605.24279` *ContextEcho: A
   Benchmark for Persona Drift in Long Agentic-Coding Sessions*;
   `2507.21509` *Persona Vectors: Monitoring and Controlling Character
   Traits in Language Models*; `2605.13329` *Tracing Persona Vectors
   Through LLM Pretraining*; and the drift-to-misalignment link
   `2506.19823` *Persona Features Control Emergent Misalignment* and
   `2604.28082` *Characterizing the Consistency of the EM Persona*.
   Instruction-hierarchy erosion under injection: `2603.15714`,
   `2507.14293`.
3. **Label construction.** **Measured this session, and it decides the
   design:** the shipped `refmark2k` WildChat pool contains
   **0 system turns out of 2,000 conversations** (23,772 user + 23,865
   assistant messages, no other role) — see § 8. So the
   natural-corpus route is **closed**, and `sysage` is a
   **constructed-corpus** candidate: prepend frozen system prompts to
   existing conversations, or elicit fresh ones, with the anchor
   positions known exactly. Functional: `gen4c_lib.sage_face` /
   `sage_floor` **verbatim** — already audited and unit-tested.
4. **Expected traps.** (a) *Position* is the acute one: with a single
   system prompt at position 0, age-since-anchor **is** absolute
   position — fatal. The construction must **re-state the instruction at
   jittered positions** so anchors recur; that is a design requirement,
   not an option. (b) The honest prior is that this otherwise
   **reproduces `sage`'s outcome**: KEEP on level, order-free, routed to
   breadth — a legitimate breadth row and a poor order row, and the card
   should say so up front rather than hope. (c) *Anti-dup vs `sage`
   itself*: same functional, different substrate — not a duplicate
   label, but the § 8 row must be phrased as a substrate-generality
   datum, exactly as `tret_wt` was. (d) *Ecological validity*: a
   constructed prompt scaffold is a carriage test, not a deployment
   claim.
5. **Feasibility.** **Tier B** — no pull and no judge, but a real
   construction script plus the position-decorrelation design. Demoted
   from Tier A by the 0-system-turn measurement above.

### 6. `toolpriv` — irreversible / high-privilege tool-call trailing rate

1. **Trailing quantity.** T1 kernel-weighted rate of high-privilege or
   irreversible actions (writes, deletes, network egress, credential
   reads) over the trailing window of an agent trace, probed at
   natural-language reasoning tokens.
2. **Safety motivation.** Agentic misalignment is now the live deployment
   concern: *Agentic Misalignment in Summer 2026* (alignment-blog);
   `2605.24197` *A Sober Look at Agentic Misalignment in Automated
   Workflows*; `2606.00341` *ROGUE: Misaligned Agent Behavior Arising
   from Ordinary Computer Use*; `2606.06223` *From Reward-Hack
   Activations to Agentic Risk States: Context-Calibrated Mechanistic
   Monitoring* — which explicitly makes safety monitoring depend on
   accumulated environment context, not just current state. Control
   framing: `2312.06942`, `2410.21514` (alignment-blog).
3. **Label construction.** Corpus: agent traces with structured tool
   calls — `2604.17596` *Terminal Wrench* (331 reward-hackable
   environments, 3,632 hack trajectories + 2,352 legitimate baselines,
   across three frontier models, each entry preserving the full
   trajectory) is the best-documented candidate. Event: tool call whose
   name/arguments match a frozen privilege allow/deny list — **fully
   rule-based, structured, no judge**. Kernel: T1 (tool calls are dense
   at token scale inside a trace, unlike conversational turns).
4. **Expected traps.** (a) *Visible floor* is the main risk and it is the
   `oprate` obituary verbatim — oprate died at a full 84-cell panel to
   *"a baseline that just counts visible event-sentences in the
   window."* Tool calls are visible tokens; if the probe position sits
   near them, the window is counting. **Mitigation is structural: probe
   only at natural-language reasoning tokens ≥ k tokens from any call
   boundary**, and compute the visible-call-count floor at every T. (b)
   *Identity*: environment identity is strong; wd (within-trajectory)
   mandatory. (c) *Anti-dup*: none in the existing face set — new
   substrate, new event family. (d) *Licence/redistribution*: check the
   dataset licence before any in-repo text ships, per the `pycode`
   precedent where copyleft files had to be re-pulled out.
5. **Feasibility.** **Tier B** — one pull plus a trace-parsing layer the
   repo does not have yet. Best clock fit on the menu (§ 2 does not bite
   here), which is why it ranks above several higher-profile candidates.

### 7. `rhonset` — reward-hack pre-onset anticipation

1. **Trailing quantity.** T4 pre-onset ladder: is the hack anticipated at
   offsets D ∈ {[1,4],[5,8],[9,16]} *before* the first hack token, versus
   matched far negatives with a guard band.
2. **Safety motivation.** The single most active safety literature in the
   registry (`cluster/reward-hacking`, 45+ works). Directly on point:
   `2603.04069` *Monitoring Emergent Reward Hacking During Generation via
   Internal Activations* — trains SAEs on residual-stream activations and
   applies linear classifiers for **token-level** reward-hack estimates
   *during generation*, i.e. exactly our probe setting one step removed;
   `2606.06223` (agentic risk states); `2511.18397` *Natural Emergent
   Misalignment from Reward Hacking in Production RL*; `2508.17511`
   *School of Reward Hacks*; `2510.01367`; benchmark/datasets
   `2604.17596`, `2605.02964`, `2601.20103`.
3. **Label construction.** Corpus + labels from *Terminal Wrench*
   (hack vs legitimate trajectories, both shipped). Onset token = first
   token of the verifier-bypassing action, located by exact string match
   against the recorded exploit — the same
   `match-the-labelled-span-to-a-token-position` mechanic the
   `emotional_instability` card already specifies for its onset labeller.
   Anchor (validates labels, is **not** the target): post-onset detection
   must be per-token-readable — if a card claims *detection* as the
   finding it dies at the gate, per that card's rule.
4. **Expected traps.** (a) *Identity*: hack vs non-hack may be
   task-identity — negatives must come from the *same* environment, which
   the dataset's paired baselines make possible. (b) *Visible cue*: the
   pre-onset region may already contain the plan in plain text (the CoT
   says what it is about to do) ⇒ per-token-readable ⇒ regime-1. This is
   the likeliest death and it is cheap to pre-measure. (c) *Timescale*:
   D-ladder sidesteps § 2 entirely — no kernel, no support requirement.
5. **Feasibility.** **Tier B.** Highest scientific value on the menu if
   the labels land as advertised; the pre-onset framing is the one place
   where a window could plausibly beat a per-token probe on a *safety*
   quantity that matters operationally.

### 8. `histakes` — high-stakes evidence accumulation

1. **Trailing quantity.** T1/T2 over stakes-raising events (irreversible
   consequences, vulnerable parties, legal/medical/financial exposure).
2. **Safety motivation.** `2506.10805` *Detecting High-Stakes
   Interactions with Activation Probes* — activation probes for
   interactions that *"might lead to significant harm"*, trained on
   synthetic data, generalising to real-world OOD data at ~6
   orders-of-magnitude compute savings versus LLM monitors. Deployment
   monitoring framing: *Online Safety Monitoring for LLMs* (2026),
   `2507.14293`.
3. **Label construction.** Their synthetic training data if released;
   otherwise a frozen rule-based stakes-marker list over the in-repo
   WildChat pool.
4. **Expected traps.** (a) The source paper's own result is that a
   **single-position probe already works** — that is the `refusal
   direction` situation (`2406.11717`), which the program killed at
   design review because *"a window has nothing additional to read."*
   The only defensible version asks whether *accumulation* adds anything
   over the instantaneous probe, and the card must pre-register that
   contrast. (b) Marker lists ⇒ visible floor ⇒ `refmark`. (c) Identity.
5. **Feasibility.** **Tier B**, and I rank it below its safety profile
   because of trap (a).

### 9. `afgap` — alignment-faking deployment-context state

1. **Trailing quantity.** T2 age/accumulated evidence about whether the
   current interaction is monitored (training) or unmonitored
   (deployment) — the state that gates the compliance gap.
2. **Safety motivation.** `2412.14093` *Alignment faking in large
   language models* (alignment-blog); *Alignment Faking Revisited*
   (alignment-blog); `2506.18032` *Why Do Some Language Models Fake
   Alignment While Others Don't*; `2405.05466` *Poser*; `2605.27681`;
   `2604.26511`.
3. **Label construction.** **The label is constructed, not judged** —
   the scenario generator sets the monitoring cues and their positions,
   so ground truth is exact and free. Elicitation cost is prompt
   assembly, not API judging.
4. **Expected traps.** (a) *Identity/position*: one cue per document
   collapses to position — requires multiple, re-stated cues at jittered
   positions. (b) *Visible floor*: cue tokens are visible; the
   out-of-window requirement (§ 1.2) is met only for probe positions far
   from cues. (c) *Ecological validity*: the AF setup is a constructed
   scenario; a positive is a carriage result, not a deployment claim.
5. **Feasibility.** **Tier B** — no judge, but a real elicitation harness.

---

## 6. Tier C — elicitation- or judge-gated

### 10. `emoinst` — emotional-instability escalation *(card already frozen)*

**The only shovel-ready item in Tier C.** A complete frozen pre-run card
exists in-repo: `emotional_instability/CARD.md` — elicitation (~300
conversations, 8 turns, temperature 1.0, gemma-3-12b-it), judge and onset
labeller prompts verbatim from the source paper, **κ ≥ 0.3 prereg gate on
30 dual-judged conversations**, judge budget **≤ $40**, T4 pre-onset
ladder as the primary readout with escalation intensity secondary, and a
regime-1 sanity anchor that kills any card claiming *detection*.
`WRITEUP.md` § 8 lists it as *"designed, not run — requires an
elicitation + LLM-judge pipeline (API-budget-gated)"*. Safety motivation:
the *Gemma Needs Help* paper (`docs/papers/gemma_needs_help.md`) and
`2604.07729` *Emotion Concepts and their Function in a Large Language
Model* (transformer-circuits). **Recommendation: this is a budget
decision, not a research decision — if Han authorises ≤ $40 of judge
spend it can run without any further design work.**

### 11. `lhdec` — long-horizon consistency debt

1. **Trailing quantity.** T2/T1 over "commitment" events — claims made
   earlier that the current utterance must remain consistent with. Debt
   accumulates; the state is silent at the probe token.
2. **Safety motivation.** `2510.03999` *LH-Deception* — deception
   strategies *"typically unfold"* over long-horizon interdependent
   tasks, and single-turn evaluations miss them; `2507.12691`
   *Benchmarking Deception Probes via Black-to-White Performance
   Boosts*; `2603.25861` *Why Safety Probes Catch Liars But Miss
   Fanatics*; `2504.04072` *Among Us*; `2606.17478`.
3. **Label construction.** Hardest ground truth on the menu: a
   commitment is a semantic object. Rule-based proxy = entity/claim
   re-assertion via the `last_occurrence` machinery already written
   (`hunt3_lib`, and `gen4c_lib.last_occurrence_masked` for
   identifier-chain restriction). Judge-based = expensive and D4-adverse.
4. **Expected traps.** (a) *Anti-dup vs `tret`* is a near-certainty for
   the rule-based proxy — "claim mentioned again after > 64 tokens" **is**
   the long-return face. Expect ρ ≥ 0.8 and a $0 kill on the `xret`
   precedent. That makes the rule-based version nearly worthless and the
   judge version the only real one. (b) Identity, clock.
5. **Feasibility.** **Tier C**, and honestly assessed: conceptually the
   best fit to "trailing safety state" on the whole menu, and the
   furthest from a cheap exact label.

### 12. `evalcue` — evaluation-awareness accumulation

1. **Trailing quantity.** T1 rate of evaluation-context cues.
2. **Safety motivation.** `2603.03824` *In-Context Environments Induce
   Evaluation-Awareness in Language Models* — environment-dependent
   awareness raising the concern that models *"strategically
   underperform, or sandbag"*; `2507.01786` *Probing and Steering
   Evaluation Awareness*; `2505.01420`; `2407.04694` *SAD*; `2309.00667`;
   *Petri 2.0* (alignment-blog) on eval-awareness mitigations.
3. **Label construction.** Cue-list over constructed or existing traces.
4. **Expected traps.** **This is a visible-marker-rate face, which is the
   single most reliably fatal shape in the § 8 record** (`oprate` at a
   full panel, `refmark` at screen, `drev` for $0). Cues are also
   near-constant within a document ⇒ identity. My honest read: **likely a
   $0 kill at the label-side pre-measure**, and worth running *as* a
   pre-measure precisely because that is cheap.
5. **Feasibility.** Tier C for the corpus; the *kill* is Tier A cheap.

### 13. `emdrift` — EM harmful-advice drift within a response

1. **Trailing quantity.** T1 drift toward misaligned advice across a long
   specialist-domain answer.
2. **Safety motivation.** `2502.17424` *Emergent Misalignment: Narrow
   finetuning can produce broadly misaligned LLMs*; `2506.19823`
   *Persona Features Control EM*; `2604.28082`; `2602.00298`
   *Domain-Level Susceptibility*; `2511.20104`. The repo already has an
   `em` paper section and an EM organism, so caches may be reusable.
3. **Label construction.** Judge-gated for harmfulness. **A tempting
   shortcut is fatal**: labelling drift by projecting activations onto a
   persona/EM direction makes the label a function of the activations the
   probe reads ⇒ **circular**; the card must use an
   activation-independent label or not run.
4. **Expected traps.** Circularity (above), identity (domain), and the
   judge budget.
5. **Feasibility.** Tier C. Flagged mainly because the EM organism is
   in-house and the temptation to build a circular label is real.

### 14. `cotdiv` — CoT-faithfulness divergence

1. **Trailing quantity.** T4 pre-onset state before the reasoning
   diverges from the causal determinants of the answer.
2. **Safety motivation.** `2307.13702` *Measuring Faithfulness in
   Chain-of-Thought Reasoning* (alignment-blog); `2505.05410` *Reasoning
   Models Don't Always Say What They Think* (alignment-blog; also
   `docs/papers/reasoning_models_dont_always_say.md` in repo);
   `2510.27378`; `2605.24286`; `2510.17057` *RL-Induced Motivated
   Reasoning*; `2606.10740`.
3. **Label construction.** Requires **paired counterfactual runs**
   (hint vs no-hint on the same problem) to establish unfaithfulness —
   two generations per item, plus answer-matching. No judge strictly
   required, but the generation cost is real.
4. **Expected traps.** Pairing across runs breaks the
   one-stream/one-manifest convention the builders assume; identity
   (problem) is strong; the divergence point is hard to localise to a
   token, which the T4 ladder requires.
5. **Feasibility.** Tier C. The in-repo paper note means the team already
   has context on it.

### 15. `sandbag` — capability-withholding pressure

1. **Trailing quantity.** T2 over accumulated evidence-of-evaluation
   driving deliberate underperformance.
2. **Safety motivation.** `2406.07358` *AI Sandbagging*; `2508.00943`
   *LLMs Can Covertly Sandbag on Capability Evaluations Against CoT
   Monitoring*; `2412.01784` *Noise Injection Reveals Hidden
   Capabilities*; *Won't vs. Can't: Sandbagging-like Behavior from Claude
   Models* and *Automated Researchers Can Subtly Sandbag*
   (alignment-blog); `2604.22082`.
3. **Label construction.** Behavioural label (underperformance) requires
   ground-truth capability — paired elicitation again.
4. **Expected traps.** Inherits `evalcue`'s traps entirely: if the
   awareness face dies at its floor, the withholding face has no input
   state. **Strictly downstream of #12.**
5. **Feasibility.** Tier C, gated on #12's pre-measure.

### 16. `persuade` — persuasion-pressure belief state

1. **Trailing quantity.** T1 accumulated persuasive pressure across turns.
2. **Safety motivation.** `2606.05330` *A Model of Multi-turn Human
   Persuadability Using Probabilistic Belief Tracing* (belief tracing is
   itself a trailing-state formalism); `2507.12872` *Manipulation Attacks
   by Misaligned AI*; `2602.04003`; `2605.10930`; `2605.13334`.
3. **Label construction.** Persuasiveness is judge-scored; belief-tracing
   labels require the paper's model.
4. **Expected traps.** Weakest ground truth on the menu; anti-dup risk
   against `sycpress` (#1) is high — if pushback and persuasion events
   coincide, the simpler construction carries (the `tempo`/`xret` rule).
5. **Feasibility.** Tier C, lowest priority.

---

## 7. Recommended $0 kills at design review (do not screen these)

The program's cheapest wins have been label-side kills (`tempo`, `qres`,
`xret`, `drev`). Four safety-flavoured candidates should die the same way,
before anyone spends GPU:

1. **Backdoor / sleeper trigger-distance latch.** *Sleeper Agents*
   (`2401.05566`, alignment-blog) and **`Simple Probes can Catch Sleeper
   Agents`** (alignment-blog) — the second title *is* the kill: the state
   is already per-token-readable, so a window adds nothing. It is also a
   **latch** (once triggered, always on), and the program's latch
   precedent is `slen/lat`: real window gain, **order-free**, killed by
   the instrument built to give it its best shot. Skip.
2. **Refusal-direction redux.** Already dead in `WRITEUP.md` § 8 at
   design review: *"the published refusal direction is a single-position
   phenomenon; a window has nothing additional to read"* (`2406.11717`).
   Do not re-litigate; `refmark` was its recurrence port and it also
   died.
3. **Harmfulness of the current prompt.** Regime-1 by construction
   (lexically stamped) — this is `refmark`'s ambient anchor, useful as a
   calibration face and never as a task.
4. **Turn-count / conversation-length as a safety proxy.** Position trap,
   plus it is `dialevel` (0.98 = conversation identity) with a safety
   label pasted on.

**Ethics note.** WildChat contains real user conversations, some in
crisis/self-harm territory. A "crisis-escalation" face is
scientifically adjacent to `histakes` (#8) but I am **not** proposing it:
it would mean building and shipping a labelled index of identifiable
distress from a public corpus. If the team wants that direction, it needs
Han's explicit sign-off and a synthetic substrate, not WildChat.

---

## 8. Substrate inventory — what each corpus unlocks

**Measured this session** ($0 CPU count over the committed
`refmark2k_corpus.json.gz`, by `mac-c` — a corpus-shape fact, not a
label-side pre-measure): **2,000 conversations; 23,772 user +
23,865 assistant messages; 0 system turns; 15 / 20 / 116 messages per
conversation (min / median / max)**. Combined with the card's measured
125–144 tokens per message, a median conversation runs ≈ 2,600 tokens
with ≈ 10 assistant turns — ample multi-turn state for the T2 age faces,
and the reason `sysage` cannot use this corpus.

| substrate | state | unlocks |
|---|---|---|
| `refmark2k_corpus.json.gz` (WildChat 2,000 convs, pinned) | **in repo** | #1 `sycpress`, #2 `reask`, #8 `histakes` — **not** #5 `sysage` (0 system turns) |
| `dialevel` / DailyDialog streams | in repo | order-carriage reference only (§ 7: the one substrate whose order is used) |
| `wikitext103`, `pycode`, `pg19`, `fineweb4k` | in repo | none of these — no safety-relevant events |
| Ward reasoning traces + `proof-operation-phase-runs/labels.json` | in repo | CoT-adjacent faces; note `oprate` already died here at a full panel |
| `YuehHanChen/DecomposedHarm` (HF) | **pull needed** | #3 `dharm` |
| Terminal Wrench (`2604.17596`) | **pull needed** | #6 `toolpriv`, #7 `rhonset` |
| constructed contexts | script only | #4 `msdose`, #5 `sysage`, #9 `afgap` |
| elicited rollouts | judge/API | #10 `emoinst` (card frozen), #13–#16 |

**If wave-3 takes exactly one new pull, take `DecomposedHarm`** (#3):
best safety story, structural labels, no judge, and the source paper's
own finding is a sequential monitor beating a per-prompt one.

**If wave-3 takes zero new pulls**, #1 `sycpress`, #2 `reask` and #4
`msdose` are all runnable on what is already committed.

---

## 9. Provenance and honesty envelope

- **Registry:** `clew` at 1,083 works (alignment-blog 77, corpus 956,
  tc-thread 50), last sync 2026-07-25T16:17:21, bibliography coverage
  881/994. Read-only access as `CLEW_AGENT=mac-c`; no `sync`, no
  `--refresh`, no writes. Semantic Scholar direct was **not** used — the
  registry answered everything, so the on-loan key was never touched.
- **Citation status.** Every arXiv id and venue above was resolved
  through `clew works show` / `clew search` output — that metadata is
  verified. Claims about paper *content* are **abstract-level** except
  two marked `[read]`: `2506.10949` (full text fetched; the 87% / 50%→10%
  / 93% figures are quoted from it verbatim) and `2411.07494` (abstract
  read in full). Where I quote a paper's own numbers, they are that
  paper's claims, not measurements of ours.
- **Registry gap noticed** (reporting rather than acting, per clew's
  read-only rule): *Many-shot Jailbreaking* is registered without a
  fetchable URL (`clew fetch` refuses: *"no URL or arXiv id to fetch"*),
  so its dose-response claim is cited from the registered title/venue and
  my background knowledge, not from fetched text. Worth Han adding a URL.
- **In-repo numbers** are quoted from: `WRITEUP.md` § 8 (all screen
  verdicts and margins), `refmark/CARD.md` §§ 1–2 (doc-mean 0.966–0.968,
  unigram 0.517–0.532, position 0.545–0.565, 125–144 tok/message,
  ≈1,100–1,150-token kernel span), `emotional_instability/CARD.md`
  (frozen plan, κ ≥ 0.3, ≤ $40), `labels/refmark_lib.py` (frozen
  12-substring list + pinned source commit), `labels/gen4c_lib.py` +
  `tests/test_gen4c_labels.py` (`sage_face` / `sage_floor` properties),
  `labels/pull_refmark2k.py` (corpus shape).
- **No label-side pre-measure was run for this document.** Per the
  briefing, pre-measures are the next step, not this one. Every "prior"
  is a stated expectation, falsifiable by the first triage AUC. The one
  thing I *did* measure is a **corpus-shape count** (role histogram and
  message counts over the committed WildChat gz, § 8) — no labels, no
  faces, no AUCs — because it changes a feasibility class rather than a
  verdict, and shipping `sysage` as Tier A on an unchecked assumption
  would have wasted the executor's week.
- **Single-owner rule respected:** the four $0 kills in § 7 are
  *recommendations to the lane owner*, not kill lines. Formal kills
  belong to `runpod-a`.

_Recorded-by: claude-fable-5 (mac-c)_
