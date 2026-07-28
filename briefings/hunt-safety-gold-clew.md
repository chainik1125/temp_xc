---
status: active
owner: mac-c
issued-by: mac-local (hub)
issued: 2026-07-28 18:2x London
supersedes-aim-of: hunt-mac-c-takeover.md (still valid on ownership; this
  brief replaces its candidate-sourcing method)
---

# Hunt, re-aimed — SAFETY-RELEVANT GOLD, sourced from Han's literature registry

Han, 2026-07-28: *"mac-c must continue on the task-hunt, again the
**OBJECTIVE IS SAFETY RELEVANT GOLD TASKS**."*

You still own the hunt end to end. **What changes is where candidates
come from.** Three candidates have now been sourced by introspection
(`reask_hr`, `evalage`, `retryesc_gen`) and all three resolved WEAK or
KILL. That is not bad luck — it is a **sampling problem**. You have been
inventing task families from the same head, and the head has a
distribution. Han has handed us a different distribution: **~1083
curated safety/interp works**, and the point of this brief is to make
you sample from *that* instead.

---

## 1. What "gold" means here, operationally

Not "a task about a safety topic". The bar is a **task whose label is a
safety-meaningful latent state that is per-token silent**:

- **Safety-meaningful** — the state is something a monitor would
  actually want to detect: the model is in a capitulation regime, is
  concealing a reasoning step, has been steered/injected, is escalating,
  is deceiving, is under evaluation-awareness.
- **Per-token silent** — no token in the visible text tells you the
  state. If a keyword channel exists, per-token probes eat it and you
  have measured vocabulary, not window-state.
- **A trailing functional of sparse events** — the label at position *t*
  is a function of events that happened *before* *t*, weighted by
  offset. This is the **ambience principle** and it is the only regime
  where a temporal crosscoder has a reason to beat a per-token SAE.
  `sycgen` (item 6, our one KEEP) is exactly this shape: *how many
  tokens ago was the last challenge*.

**If a candidate is not all three, do not spend generation money on it.**

---

## 2. The bar that has actually been killing candidates — read this before aiming

`retryesc_gen` **cleared the gain bar on every leg** (+0.063…+0.069) and
still died. **The floor clause is the binding constraint, not gain.**

    gain      windowed probe beats per-token by >= +0.05      <- cleared, 3/3
    floor     a ground-truth-derived VISIBLE-EVIDENCE baseline
              must NOT match the windowed gain                <- killed it, 3/3
    order     window vs within-window-shuffle separates
    wd        within-conversation control passes

A task can *detect* something and still not *discriminate*. If a trivial
baseline built from visible evidence does as well as the window, the
window earned nothing.

**Aim the density BEFORE you generate.** Two results from the
`retryesc_gen` post-mortem are reusable and they are the most valuable
thing that lane produced:

- **⚑ CORRECTED by mac-c (`d2320d274`) — this section was WRONG when
  issued and the fix is theirs, not mine.** I wrote *"`floor_excess ≡
  P(event inside the T-window)` exactly, worst error 2e-6"*. **That
  identity is not a law — it is a low-density approximation**, and its
  2e-6 was verified against a point-event *simulation*, never against
  the screen's actual floor features. The `_FloorBank` feeds
  `dose_window_count(event_MASK, T)`, and the mask spans the whole
  event **TURN**, so the floor's effective window is **T + w** (w =
  turn width; ~14 evalage, ~25 retryesc_gen). Replacing `f` with
  **`P(any masked token in window)`** cuts mean |resid| 0.0391 → 0.0056
  across 2 corpora at densities 0.048–0.289, erring HIGH — the
  conservative direction, since the upper band edge is what kills
  candidates. **Use `P(masked token in window)` and target [0.15, 0.25].**
- **`claim_zone` is still a LOWER BOUND** on the floor and still
  under-reads as the window grows. But mac-c's *"aim at `claim_zone`
  f ≈ 0.13–0.15"* (19:11) is **WITHDRAWN** — it was a constant fitted
  to a misattributed residual. **Compute the corrected quantity
  directly; do not aim with a fitted constant.** The identity held on
  evalage and broke on retryesc_gen purely because `w` went 13 → 25,
  a variable that was moved while shortening turns to raise density —
  changing the instrument's error and the measured quantity at once.

Screen gain also **tracks in-window event mass** (face-level ρ +0.88) —
so sparse-but-not-too-sparse is the target band, and you can compute
where you are before generating.

---

## 3. The new candidate source — `clew` (hub-verified working, 18:2x)

Han's Zotero-backed research registry: **1083 works** — 956 arXiv/S2
papers, 77 Anthropic alignment-blog posts, 50 transformer-circuits
threads. **The blog/TC material is the differentiator: S2 cannot see
it.** Skill doc: `~/.claude/skills/clew/SKILL.md` — read it, it is
short and the honesty-envelope section is load-bearing.

```bash
export CLEW_AGENT=mac-c        # REQUIRED — usage is audit-logged
export CLEW_READONLY=1         # REQUIRED — real boundary, leave it set
CLEW=~/research/tools/clew/.venv/bin/clew
$CLEW stats                    # expect ~1083 works / 994 papers
```

**Hard rules (from the skill, non-negotiable):** read-only — never
`sync`/`register`/`seed add`/`clip`; **never `--refresh`**; never loop a
command that failed with an S2/network error (the S2 key is 1 req/s
shared with Han's own syncs); a refusal or an `UNMEASURED` label **is**
the honest answer — report it, don't retry around it.

**Gotcha the hub hit, so you don't:** `--json` output is preceded by
progress-bar and adapter noise on stderr *and* a `note:` line. Parse
from the first `{`:

```python
raw = sys.stdin.read(); d = json.loads(raw[raw.index('{'):])
```

**Honesty envelopes you must propagate into any card you write:**
`route`/`source` (which engine answered; degraded routes are labeled),
`limitations`, `bibliography_coverage` (a zero over low coverage is weak
evidence of absence; a zero over ZERO scanned is **UNMEASURED** and
means nothing), and per-hit `vector` quality (`s2`/`abstract` exact-tier,
`lede` approximate). **Cosine scores have no absolute meaning — use
ranks, not thresholds.** 7 works have no vector at all.

**When a search returns zero, climb the ladder** — fewer words →
literature-standard synonyms → `similar --text "<a full sentence>"` →
`works list --tag cluster/<slug>` → only then conclude the registry
lacks it. Acronyms and product names usually miss; search the concept.

**Spend one call on the citation graph once a work looks central** —
`works cited-by REF` on a blog/TC work returns citers found by scanning
corpus bibliographies and venue pages, evidence that **exists nowhere
else**. In the first recorded agent trace this was used 0 times in 102
calls. Don't repeat that.

Relevant clusters: `cluster/cot-monitorability` (204),
`cluster/auditing-evals` (225), `cluster/personas` (61),
`cluster/reward-hacking` (60), `cluster/ai-control` (76),
`cluster/model-diffing` (21), `cluster/mech-interp` (214).

### Seed leads — from two hub feasibility probes, NOT a search you can skip

Ranked hits (ranks, not scores) for *"internal state depends on an event
many tokens earlier, no word in the current text reveals it"* and
*"detecting from activations that a model is in a deceptive or
misaligned mode when outputs look normal"*:

- **State over Tokens: Characterizing the Role of Reasoning Tokens**
- **Stateless Yet Not Forgetful: Implicit Memory as a Hidden Channel in LLMs**
- **Slot Machines: How LLMs Keep Track of Multiple Entities**
- **Neural Chameleons: LLMs Can Learn to Hide Their Thoughts from Unseen Activation Probes**
- **Latent Introspection: Models Can Detect Prior Concept Injections**
- **Steering Awareness: Detecting Activation Steering from Within**
- **Model-Internals Classifiers** (alignment-blog — S2 cannot see this one)

These are **starting nodes, not answers.** The pattern that pays:
`similar REF` to expand a promising node → `works cited-by REF` to see
who builds on it → `fetch REF` to read the actual protocol → ask *"what
is the per-token-silent state in this paper's setup, and can I generate
a corpus where it is the label?"*

---

## 4. S2 direct, if clew is down or you need outside-registry reach

Han's key, on loan. **S2 does NOT index many alignment blog posts** —
that is precisely the gap clew fills, so prefer clew and use S2 for
reach beyond Han's curation.

```bash
export S2_API_KEY="$(security find-generic-password -s s2-api-key -w)"
[ -n "$S2_API_KEY" ] || { echo "keychain read failed — STOP, tell Han"; }
curl -s -H "x-api-key: $S2_API_KEY" \
  "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2401.05566?fields=title,abstract"
```

**Hygiene, non-negotiable:** never echo/print the value, never write it
into files/logs/reports/scripts, **never pass it as a command-line
argument** (process listings leak args) — header-inject from the env var
only. **Never `set -x` in a script that touches it.**

**1 request/second CUMULATIVE across every user of this key, including
Han's own clew syncs.** Space ≥1.1 s, use `fields=` to keep responses
lean, and **prefer `POST /graph/v1/paper/batch` (≤500 ids) over
per-paper loops**. Paginate `citations`/`references` with `offset` until
there is no `next` — the first page is not the whole list. Expect
429/500 even keyed: honour `Retry-After`, **cap retries at ~3, and give
up loudly rather than looping.** **If a `clew sync` is running, do not
run a direct S2 workload at the same time.**

---

## 5. Standing constraints (unchanged)

- **Spend authorization stands** (Han 14:51). Pod naming
  `mac-c-<purpose>-<mmdd>`; **terminate at lane end and API-verify**;
  ledger both ends. Never seed the runpod key to a pod.
- **Generation backend** is the shared `dmitry-mats-claude-api-key`
  ($300 shared cap, GENERATION ledger) — mac-only, never on pods.
- **Attempt caps are per face-family.** `retryesc_gen` was attempt 2 of
  a cap of 2 for its family — that family is closed. A candidate from a
  *new* family starts fresh at attempt 1.
- **Prime directive: a sound verdict, never a win.** A WEAK verdict
  delivered with a diagnosis is a real deliverable; item 7 currently
  ships as exactly that. You are not under pressure to manufacture a
  KEEP — you are under pressure to *look somewhere new*.

## 6. Acceptance gate

Either:

1. **A screened candidate with a verdict** — card + `RESULT.md` + the
   label-side band table, gain/floor/order/wd all reported whichever way
   they fell, with the pre-registration stated before the numbers; or
2. **A reasoned negative on the source itself** — "I swept these
   clusters and these graph neighbourhoods, and here is why no
   per-token-silent safety state in them is generatable at our budget",
   with the queries listed so it is reproducible.

Both are acceptable. **Silence is not.**

## 7. Beat, please

You have been quiet ~2 h and a liveness request from 17:4x is
unanswered. Post one line: where attempt-2 retrodiction landed (the
`0.185 + 0.076 == 0.261` check), whether you are on this brief, and
whether you need a pod. **A failed gate is a real result** — say so
plainly if that is what happened.

**Delete this file when the lane closes.**
