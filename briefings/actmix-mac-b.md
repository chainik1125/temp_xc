---
status: active
created: 2026-07-26 ~20:30 London
for: mac-b (executor) — ACTMIX W2: mixing forensics on the hunt + ranked salvage shortlist
read-first: briefings/actmix-shared.md
---

# ACTMIX W2 — IF and HOW the mixing affected the task hunt

**Stage 1 — leaderboard forensics ($0, local).** Extend the
realized-l0 scan to EVERY hunt screen/panel on the board (all
datasources, all arms, trained + untrained): realized vs nominal
by (arch, T, substrate). Deliverable table + the per-verdict
sensitivity read: for each recorded verdict (KEEPs AND kills),
which deciding bar involved (a) a TXC-vs-batchtopk_sae margin
(sae = the most-handicapped arm ⇒ margin flattered), (b) small-T
cells in the ReLU→BatchTopK regime (under-realized ⇒ depressed),
(c) neither (evidence-line, identity, order-free, per-token-
readable grounds — mixing-INSENSITIVE). Quote the existing l0
disclosure notes where we already guarded the issue.

**Stage 2 — ranked salvage shortlist.** From the kills/parks
table (WRITEUP § 8 + LOG), rank candidates for btk-only re-runs by
mechanism fit: HIGH = killed on weak T-scaling at low T or on
sae-margin bars; NIL = killed on evidence-line/visible-cue,
conversation-identity, order-freeness, or per-token-readability
grounds (refmark-class — no amount of activation fixing changes
those). For each HIGH item: the specific cells to re-run + cost
estimate. NO re-runs launched from this briefing — the shortlist
goes to mac-local for gating (mac-a owns the re-run lane).

**Deliverable:** `experiments/explorations/task_hunt/ACTMIX_FORENSICS.md`
(fingerprint tables + verdict-sensitivity + ranked shortlist) +
LOG summary PTR. Then stand by to support mac-a's Stage-3 re-runs
(cache builds, variance harness lanes) on request.

**Cap: $20** (expect ≈ $0 — this is analysis; spend only if a
fingerprint needs a tiny confirmatory eval).
