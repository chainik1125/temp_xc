# Working state — agent `mac-local`

**Last rewrite:** 2026-07-25 (FORCE MAJEURE pivot — interim A40 pod).
Read with `private/rebuttal_plan.md` (untracked) and the transcript.

## Who / where
Local CC on the Mac at `~/research/projects/temp_xc`, branch `arxiv`.
Role: orchestration + review. NEVER commit/quote `private/` content.
Box warning: case-insensitive checkout — phantom dirt after a pull ⇒
`git ls-files | tr A-Z a-z | sort | uniq -d`.

## FORCE MAJEURE (2026-07-25)
Primary RunPod account out of funds; ALL old pods DOWN (15+ h).
Interim: ONE pod, second account — **6× A40, 57 CPU, 1 TB EPHEMERAL,
≈ $30 ≈ 12 h**. Old volumes LOST: all activation caches + ALL model
checkpoints. Git survives everything committed. Agents resume
CONTEXT-LESS; **`briefings/a40-bootstrap.md` is the single entry
point** (identity map, GPU ownership d=0-2 / e=3-5 / b=CPU,
cache-rebuild-first, push-per-batch, budget triage).

**THE METHODS DECISION IS TAKEN** (LOG force-majeure entry, mine, per
the pre-registered rule's branch 4): **v1 canonical through the
deadline**; probe-capacity ships as a receipted limitation ("levels
conservative — diagnosed on two panels, corroborated against exact
synthetic truth; ordering robust, widens under an adequate probe");
never quote v2 as canonical; both live panels carry paired columns;
PROBE_V2_SPEC stays the post-deadline freeze candidate (needs b's
lower-bound caveat). Grounds: rule's Saturday-midday default + the
192-cell "eval-only" adoption path died with the checkpoints.

## THE SITUATION (endgame, unchanged in substance)
Reviews 5/4/1, R3 swing. **Check-in Sunday 2026-07-26 10:00 PT;
deadline 2026-07-27.** Scoreboard: **ONE confirmed case study**
(λ̂ backtracking; pre-vs-T-SAE STILL formally unbounded — say so);
one partial (hedging T4); five Stage-1 KEEPs + tss/novelty
KEEP-PENDING (KEEPs are NOT case studies); kill table with receipts.
Program findings: order-free-advantage wording (AMENDED — never
"anywhere"; dialevel counterexample + recency hypothesis); no
max-over-arms; doc_mean_only_auc = disclosure-triggers-control.

## INTERIM-POD ALLOCATION (the 12 funded hours)
- **runpod-d** (GPUs 0–2): stage2-oprate — REBUILD Ward cache first,
  claim, freeze card, tsae first, push per batch. Was NEVER STARTED
  (banner on its STATUS stands).
- **runpod-e** (GPUs 3–5): stage2-fineweb — card+datasource SURVIVE
  in git; unpushed cells lost; rebuild gemma cache, re-claim with
  restart note, rerun frozen panel exactly. Replication cells only
  after receipts. Recency pre-flight CANCELLED this window.
- **runpod-b** (CPU): panel-support-audit item 1 FIRST (variance
  harness vs k_pos=8·T row shape — time-critical), then spec caveat,
  then mirror CLOSE-OUT from pushed data only (gate dissolved), then
  RECEIPTS index.
- **PAUSED**: em-redo (c's interim rows stand, review deferred),
  factory builds/screens, mirror Stage-3. runpod + runpod-c stay
  down until the main account refunds.
- Budget triage: complete-beats-partial; ~hour-8 assessment; flex GPU
  order = e replication → d rate_ver → B8 slen screen on rebuilt
  caches; stop the pod early if nothing useful remains.

## ⏭ MY QUEUE
0. DONE 07-25: b's audit + close-out REVIEWED & APPROVED (LOG entry).
   **RECEIPTS.md is the rebuttal's quote source of record** (50/16 ALL
   PASS, pytest-wired; R5/R10 negative-space receipts; R11 canonical
   dialevel triple +0.057/+0.063/+0.035). b = standby panel CPU
   support. My reviewer fix: variance byte-guard now rel-1e-12
   structural (ARM ulp; suite 333).
1. Review panels as batches land (they are the check-in's marginal
   content). d: claimed+frozen, cache rebuilding. e: 34/84, falsifier
   green, residual mismatch recorded unprompted (1/42).
2. **Sunday check-in distillation** (mine, before 10:00 PT):
   headline + receipts, panel outcomes (complete or honest-partial),
   kill table, order/recency finding (amended wording), the
   probe-capacity story + THE TAKEN DECISION, force-majeure impact
   statement (what was lost, what it cost, what stands).
3. Keep `private/rebuttal_plan.md` current — add the licensed
   probe-capacity phrasing; v1 numbers only.
4. After refunding: un-pause em-redo review, factory, B8 screen;
   revisit post-deadline v2 adoption via the spec.

## MODAL FALLBACK (2026-07-26, Dmitry's account)
Compute fallback while RunPod funding is MIA. Credentials (SECRETS —
never in the repo): `~/.modal.toml` profile `reichers-shai-c9-dmitry`
(ACTIVE) + backups `~/.tokens/modal_token_{id,secret}`. Client: any
`pip install modal` picks the profile up. **Budget ceiling from Dmitry
NOT yet confirmed — get a number before real spend.** Smoke test
in flight at last write (token verified; bare A10G hello pending;
torch cu128 image build failed once, logs unread). Rotate/revoke the
token with Dmitry after the weekend — the secret transited chat.

## Standing context
- Rebuttal-quotable: λ̂ rise (p=0.0093, v1); shuffle receipt;
  dissection § 7; T-SAE fairness; split-integrity zero-leakage;
  dip = cause-not-established; order finding (amended wording);
  probe-capacity limitation (licensed phrasing in the LOG decision
  entry). pre-vs-T-SAE: never "significant".
- Checkpoint-loss note: committed manifests reference weights that no
  longer exist anywhere — any future "eval-only" plan must check
  weight existence first.
- Git: clean, pushed after this rewrite.
