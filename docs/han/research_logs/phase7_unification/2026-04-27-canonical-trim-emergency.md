---
author: Han
date: 2026-04-27
tags:
  - design
  - complete
---

## Emergency canonical trim (49 → 38 archs) — for Agent C

### TL;DR

Both Agents A and B are not finishing the original 49-arch canonical
set in the available autonomy window. Decision: **trim the canonical
set to 38 archs**, dropping 11 high-T H8 + extreme-SubseqH8 cells.
TXCDR T-sweep (3..32) is unaffected because Agent A finished it.

### What's dropped from the headline

| arch_id | row | reason |
|---|---|---|
| `phase57_partB_h8_bare_multidistance_t10` | 37 | high-T H8 tail; not done either seed |
| `phase57_partB_h8_bare_multidistance_t12` | 38 | high-T H8 tail; not done either seed |
| `phase57_partB_h8_bare_multidistance_t14` | 39 | high-T H8 tail; not done either seed |
| `phase57_partB_h8_bare_multidistance_t16` | 40 | high-T H8 tail; not done either seed |
| `phase57_partB_h8_bare_multidistance_t18` | 41 | high-T H8 tail; not done either seed |
| `phase57_partB_h8_bare_multidistance_t20` | 42 | high-T H8 tail; not done either seed |
| `phase57_partB_h8_bare_multidistance_t24` | 43 | high-T H8 tail; not done either seed |
| `phase57_partB_h8_bare_multidistance_t28` | 44 | high-T H8 tail; not done either seed |
| `phase57_partB_h8_bare_multidistance_t32` | 45 | high-T H8 tail; not done either seed |
| `phase5b_subseq_h8_T32_s5` | 48 | extreme SubseqH8; not done either seed |
| `phase5b_subseq_h8_T64_s5` | 49 | extreme SubseqH8; H200-only by spec; not done |

### What stays (38 archs)

- All Group 1 (per-token / non-TXC) — rows 1–7
- All Group 2 (fixed-T TXC variants + Subseq B2/B4) — rows 8–13
- All Group 3 (TXCDR T-sweep T=3..32) — rows 14–29 ← **unaffected**
- Group 4 reduced (H8 T=3..9 only) — rows 30–36
- Both anchor cells (rows 46, 47) — Agent A retargeting now
- (Group 6 dropped entirely)

### What this preserves

- **Full TXCDR T-sweep (T=3..32)**: regression-at-high-T narrative is
  fully demonstrable from TXCDR alone; we don't need H8 to also
  span the high-T tail.
- **Low-T H8 sweep (T=3..9)**: peak-region of the H8 narrative, where
  H8 is expected to outperform TXCDR. Sufficient to demonstrate the
  H8 contribution.
- **Anchor cells**: the disentanglement of "context limit" vs
  "per-slab sparsity collapse" is preserved.
- **All Phase 5 winners**: agentic_mlc_08, agentic_txc_02,
  phase5b_subseq_h8 (mp champion), txc_bare_antidead_t5 (Track 2).

### What's lost

- "Does H8 also show regression at high T?" — answer was expected to
  be "yes, similar to TXCDR" but we're punting on confirming this.
  Mention as appendix-level future work.
- "Does subsequence sampling unlock T-scaling beyond what TXC can?" —
  the 2 SubseqH8 high-T cells (rows 48–49) were the test for this
  claim. Drop the claim from the paper or move to limitations.

### Action for Agent C

Same trim applies to seed=1. Agent C should:

1. **Stop training the dropped 11 archs** (don't waste H100 time on
   what's no longer in canonical).
2. **Confirm Agent C has these stays-in-canonical archs at seed=1**.
   Likely missing some — check
   `huggingface_hub.HfApi().list_repo_files('han1823123123/txcdr-base')`
   for `<arch_id>__seed1.pt`.
3. **For any kept arch missing at seed=1**, prioritise those.
4. **MLC family (mlc, mlc_contrastive_alpha100_batchtopk, agentic_mlc_08)**
   appears to be missing entirely from Agent C's seed=1 batch — please
   investigate whether this was an OOM / skip, and prioritise these
   if so. They're cheap to train (~10-15 min each) on H100.

### Updated canonical_archs.json

The JSON file is **not** edited in place — it stays as the historical
record of the original 49-arch plan. The dropped 11 are flagged
**post-hoc** in analysis code via:

```python
DROPPED_FROM_HEADLINE = {
    "phase57_partB_h8_bare_multidistance_t10",
    "phase57_partB_h8_bare_multidistance_t12",
    "phase57_partB_h8_bare_multidistance_t14",
    "phase57_partB_h8_bare_multidistance_t16",
    "phase57_partB_h8_bare_multidistance_t18",
    "phase57_partB_h8_bare_multidistance_t20",
    "phase57_partB_h8_bare_multidistance_t24",
    "phase57_partB_h8_bare_multidistance_t28",
    "phase57_partB_h8_bare_multidistance_t32",
    "phase5b_subseq_h8_T32_s5",
    "phase5b_subseq_h8_T64_s5",
}
```

Analyzer scripts filter these out of leaderboard / Pareto plots.
The dropped archs may still get their probing entries (Agent A
runs `--headline` on whatever ckpts it has), so post-filtering is
defensive in case any ended up with partial data.

### Cross-seed coverage status (as of 2026-04-27 12:00)

| | seed=42 | seed=1 |
|---|---|---|
| Done on HF | 36/49 | 29/49 |
| Will-be-done after retrim | 38/38 (anchor cells in flight now) | depends on Agent C |

After this trim Agent A's remaining work is just 2 anchor cells
(~2 hr total) + the probing pass.
