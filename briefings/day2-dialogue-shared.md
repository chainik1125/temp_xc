---
status: active
created: 2026-07-26 ~11:30 London
for: SHARED ops — mac-a + mac-b (executors), mac-local (orchestrator)
venue: Modal (Dmitry's workspace; ledger MODAL_SPEND.md)
---

# Day-2 sprint — the dialogue / order-sensitivity thread (Han, 2026-07-26)

**Why this thread (the one open door):** the overnight closed every
other one. Across 10 face × model screens on 3 broad-text substrates,
within-window shuffle cost never exceeded +0.019 (R20; R10 extended)
— order-free aggregation everywhere. The ONE measured order-carried
window signal outside backtracking is **dialevel's anchor-fixed
context shuffle cost on DailyDialog: +0.057 (gpt2) / +0.063 (gemma) /
+0.035 (llama31) at T = 32, 3/3 models (R11)** — and slen just showed
it is NOT generic recency (broad-text instrument collapsed), so it is
something about DIALOGUE structure. Today: (W1) decompose WHAT
carries it — turn order vs within-turn order vs recency weighting —
and (W2) screen dialogue-native candidate faces that could become a
SECOND order-carried case study, with a gated mini-panel if and only
if a face KEEPs *with* order-carriage. A mechanism verdict with no
new KEEP is a fully successful day — prime directive unchanged: **a
sound verdict, never a win.**

## Timeline (LONDON TIME; hard gates)

- ~11:45 launch. **NO NEW Modal starts after 15:30.** Everything
  pushed by **16:30**. mac-local review + distillation addendum by
  17:30; team check-in 18:00.
- The W2 mini-panel (if its gate fires) must LAUNCH by **14:30** to
  fit; otherwise it is written up as panel-ready and handed to the
  post-deadline queue. No exceptions — a partial panel at 16:30 is
  worth less than a clean screen verdict plus a frozen panel card.

## Budget

Spent so far: **~$33 of $500 hard / $400 soft.** Today's caps:
**mac-a $120, mac-b $60** (raises only by mac-local LOG line). Ledger
`briefings/MODAL_SPEND.md`: READ total before every launch, APPEND
after. GPU defaults: **L40S** for screens/ladders; **A100-40 (or
A100-80 if 40 GB is tight) ONLY for gated panel cells, with the
reason stated in the ledger line.** Reference: L40S ≈ $1.95/h,
A100-40 ≈ $2.8/h.

## Ops (the overnight's paid-for lessons, all binding)

1. Commit-then-run: card + executor FROZEN and pushed before any
   cell; container pinned to the freeze commit **via `git rev-parse`**
   (the mistyped-SHA lesson); `_assert_pinned()` in-container.
2. **`--detach` / detach-at-launch always** (the $4.5 lesson: a
   non-detached client disconnect cancels in-flight inputs).
3. Sequential `.remote` per model + retries; resume from Volume
   partials; build caches INSIDE the work container (the 34 GB
   Volume-commit shutdown-grace lesson); batch-halving on OOM
   pre-authorized (outputs unaffected).
4. Containers NEVER push git — repatriate, merge locally, dup-check.
5. Coverage: **gpt2 + llama31 (ungated).** gemma-2 cells run ONLY if
   an HF secret appears in Modal (ask Han once at launch; if provided,
   gemma arms are pre-authorized here for BOTH workstreams and for
   the overnight cards that named gemma-pending).
6. Receipts: per the 2026-07-26 process ruling — direct-add rows are
   permitted WITH `receipts_check` green, and **no row is quotable
   until mac-local's ratification review**. All verdicts PENDING TEAM
   REVIEW. No max-over-arms; no pooling across models; linear class
   canonical for scoring, MLP reported.
7. **Within-dialogue control is BINDING on every dialogue face** —
   dialevel's own history is a 0.983–0.986 naive causal screen killed
   by conversation identity. No KEEP without the within-dialogue arm.
8. Push after every completed stage; anything not pushed does not
   exist.

## Work split

- **mac-b** (`day2-dialogue-mac-b.md`): W1 — the ORDER-MECHANISM
  LADDER on the existing dialevel substrate. The thread's core; runs
  first and cheap.
- **mac-a** (`day2-dialogue-mac-a.md`): W2 — dialogue-native candidate
  screen (trend + qgap-on-dialogue faces) + the GATED mini-panel prep
  and, only through the gate, execution.
- **mac-local**: freeze-reviews BEFORE cells where timing allows (push
  your freeze, then start caches — the review lands in parallel);
  rolling review; the panel gate decision in writing; ledger watch;
  distillation addendum.

## The panel gate (pre-registered here, decided by mac-local in a LOG line)

The W2 mini-panel launches ONLY if ALL of:
(i) a W2 face KEEPs on 2/2 screened models under its frozen card;
(ii) that face's order arm shows **sc ≥ +0.03 at T ∈ {16, 32} on 2/2
models** (order-carried, not just aggregated — the whole point of the
thread; overnight's order-free KEEPs measured ≤ +0.019);
(iii) launch possible by 14:30 London;
(iv) ledger total ≤ $250 at launch;
(v) mac-local written LOG approval naming (i)–(iv) checked.
W1's mechanism verdict informs (ii)'s interpretation but cannot
substitute for it.

## Acceptance

LOG entry per verdict (`(executor)`, PENDING TEAM REVIEW); receipts
rows for quotable claims (ratification before quoting); ledger lines
per launch with actuals corrections; briefings retire at the 18:00
check-in.
