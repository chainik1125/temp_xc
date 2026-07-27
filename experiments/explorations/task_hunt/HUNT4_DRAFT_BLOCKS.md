# HUNT4 draft blocks — gen-4 wave-1 rows for the WRITEUP (mac-b, staged per 28a6aa6a6 § 5)

**Status: STAGED for mac-local's ratification pass ("I ratify on
push"). Nothing here is applied; blocks are copy-paste-ready against
WRITEUP.md as of 4d544ae08.** Sources: bundle verdict ab1597c65
(ratified 28a6aa6a6), replication 4d544ae08 (ratified same, no
arbitration conflicts), wave-1 card 35d20e3cb § 3 pre-measures,
HUNT4W2 card § 1 (drev kill line, ratified 04a5d2186). All content
PTR until mac-local applies.

## Block 1 — § 8 rows: the two breadth KEEPs (w/ replication receipts)

Insert after the `novelty-rate trend (nvtrend)` row (keeps the
dialogue-breadth family adjacent):

```markdown
| long-return trailing rate in dialogue (`hunt4/tret`) | screen (3 models) + re-seed replication | **KEEP 2/3, routed to breadth:** real window state on both larger models (+.097/+.101, T64 arms; within-dialogue gains to +.117) with a re-seed replication receipt (+.084, same deciding arm under independent manifest/probe seeds) — but no order receipt inside the ladder (max wd shuffle margin +.014 at T ≤ 32), so the frozen order rule routes it to breadth; its design guarantee (every event cites an occurrence > 64 tokens back, out-of-window at every ladder T by construction) makes it the cleanest breadth row on record, and wave-2 asks the sharpened question (reproduce on cold substrates / move into the ladder / find the missing order). |
| cross-speaker adoption *trend* in dialogue (`hunt4/xtrend`) | screen (3 models) + re-seed replication | **KEEP 2/3, routed to breadth:** the Δ-face of adoption clears the rule on gemma (+.064) and llama (+.067, in-ladder T16 arm) with a replication CONFIRM on the state — and the replication also showed its single order receipt (gemma +.031 @T32) collapsing to +.004 under re-seeding: seed noise, so effective order support is 0 models and the breadth routing is independently confirmed from two directions. |
```

## Block 2 — § 8 rows: the two WEAKs and the infeasible

Insert directly after Block 1's rows:

```markdown
| speaker dominance in dialogue (`hunt4/sdom`) | screen (3 models) + re-seed replication | **WEAK, no majority (KILL/KEEP/WEAK), and the one KEEP is seed-fragile** (+.059 → +.042 under re-drawn manifests, below the margin bar) — but its within-dialogue ORDER margins pass on all three models (+.035…+.081) *and* survive re-seeding: the level signal is fragile while the order signal is robust, so the datum enters the § 7 order map rather than any task row; a future dominance design must fix the level readout first. |
| cross-speaker adoption *rate* in dialogue (`hunt4/xnov`) | screen (3 models) | **WEAK on all three** (+.036/+.050/+.052, never clearing the floor conjunction): speaker-resolved "never said by me" needs unbounded history in principle, but its per-T visible floor (0.78 @T32) eats the claimable zone exactly as cnov's ruling predicted for adoption-family faces. |
| trailing return *depth* in dialogue (`hunt4/tretd`) | screen (designed-then-infeasible) | **SKIP on both substrates by its own instrument:** the position-matched manifest starves its low class below MIN_ROWS (the depth label needs deep history, which the position control refuses to give it for free) — recorded as infeasible with no relaxation; its wikitext transplant (labeled 45–46% → .89 with 20k/class and a chance-flat visible floor, the flattest in the hunt record) is wave-2's priority 1. |
```

## Block 3 — § 8 rows: the two $0 label-side kills

Insert next to the `tempo` / `qres` kill rows (the $0-kill family):

```markdown
| long returns split by speaker attribution (`hunt4/xret`) | label pre-measure | **Killed for $0 at the anti-dup bar:** Spearman vs its parent `tret` = 0.809–0.812 across all three models — the "returned by the *other* speaker" twist does not decorrelate the trailing rate from its parent, so screening it would have manufactured a duplicate; the simpler construction carried (tempo precedent). |
| definition-revival rate in Python code (`hunt4w2/drev`) | label pre-measure | **Killed for $0 by its own floor:** identifier window-novelty is a near-sufficient window statistic for revival rate (visible floor 0.70 @T4 → 0.84 @T32, swallowing the ladder), on the slate's hottest unigram (0.62–0.65) and doc-mean (0.78–0.80) readings, with a 0.70–0.74 near-dup vs `tret_py` — definition→use structure is better captured by the plain return face at this kernel. |
```

## Block 4 — OPTIONAL § 7 addendum: the sdom order-map sentence

Per 28a6aa6a6 § 3 ("ORDER-MAP evidence, not a task result").
One sentence, appended to § 7's measurement list as item 4 — drop
freely if § 7 should stay three-legged:

```markdown
4. **Dialogue's order signal is speaker-resolved.** A dominance
   contrast between the two speakers' trailing states (`sdom`)
   passed the within-dialogue shuffle margin on all three models
   (+.035…+.081 at T16/T32) and survived an independent re-seed —
   even though its *level* readout stayed WEAK — i.e., what the
   window carries includes *who* is doing the recent talking, not
   just the arrangement of turns.
```

## Block 5 — not drafted, and why

- **rdens**: routed WEAK→breadth on its own lane (71d0fcf75,
  mac-a's); its one-sentence row belongs to that lane's owner if
  wanted — happy to draft on request.
- **Wave-2 rows (tretd_wt / tret_wt / sage / tret_py)**: screens in
  flight at staging time; rows draft cleanly on their bundle
  verdict, same formats as above.
- **cnov panel rows**: 17:00-pick-gated; HUNT3_DRAFT_BLOCKS.md § 4
  already holds the contingent paragraph.

_Staged-by: claude-fable-5 (mac-b). PTR; apply/edit/drop is
mac-local's._
