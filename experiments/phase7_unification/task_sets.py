"""Phase 7 task-set definitions — single source of truth for the FULL
SAEBench task universe and the PAPER headline subset.

The FULL set is the original Phase 7 standard (`probe_cache_S32/`
contains all 36 task dirs). The PAPER set is the finalised
paper-headline reduction — pre-registered selection by cross-arch SD
within balanced clusters (4 bias_in_bios, 3 europarl, 3 amazon,
2 ag_news, 2 github_code, 2 coreference). Rationale + cluster
proportions in
`docs/han/research_logs/phase7_unification/agent_x_paper/2026-04-29-paper-task-set.md`.

All paper plots / leaderboard tables that go into the headline
filter to PAPER. The FULL set is reported in supplementary.
"""
from __future__ import annotations

# All 36 SAEBench tasks at S=32 (Phase 7 current methodology, FLIP on
# winogrande / wsc). Computed by enumerating probe_cache_S32/.
FULL = frozenset({
    "ag_news_business", "ag_news_scitech", "ag_news_sports", "ag_news_world",
    "amazon_reviews_cat0", "amazon_reviews_cat1", "amazon_reviews_cat2",
    "amazon_reviews_cat3", "amazon_reviews_cat5",
    "amazon_reviews_sentiment_5star",
    "bias_in_bios_set1_prof11", "bias_in_bios_set1_prof18",
    "bias_in_bios_set1_prof19", "bias_in_bios_set1_prof2",
    "bias_in_bios_set1_prof21",
    "bias_in_bios_set2_prof13", "bias_in_bios_set2_prof22",
    "bias_in_bios_set2_prof25", "bias_in_bios_set2_prof26",
    "bias_in_bios_set2_prof6",
    "bias_in_bios_set3_prof1", "bias_in_bios_set3_prof12",
    "bias_in_bios_set3_prof14", "bias_in_bios_set3_prof20",
    "bias_in_bios_set3_prof9",
    "europarl_de", "europarl_es", "europarl_fr", "europarl_it", "europarl_nl",
    "github_code_go", "github_code_java", "github_code_javascript",
    "github_code_python",
    "winogrande_correct_completion", "wsc_coreference",
})

# Paper headline 16-task subset (`PAPER`).
#
# Rationale (full discussion in `2026-04-29-paper-task-set.md`):
# Pre-registered selection — picks decided BEFORE checking the
# resulting leaderboard, to avoid reverse-engineering the set to make
# any one arch look good.
#
# Per-cluster picks motivated by:
#   - bias_in_bios (4): largest cluster; top-4 by cross-arch SD.
#   - europarl (3): represents the per-token-saturation spectrum —
#     `fr` (T-hurts because per-token saturated at AUC 0.996),
#     `de` (intermediate),
#     `nl` (T-helps because per-token only 0.871, distributed Dutch signal).
#     This single cluster contains the cleanest "natural" example of
#     where TXC's window structure helps vs hurts.
#   - amazon (3): cat5 + cat3 (top-SD product categories) + sentiment_5star.
#   - ag_news (2): top-2 SD topics.
#   - github_code (2): code language ID with single-token shortcuts —
#     a "TXC's structure should NOT help here" benchmark.
#   - coreference (2): winogrande + wsc — multi-token-by-construction;
#     winogrande is the cleanest empirical demonstration of TXC's
#     structural advantage (T monotonically helps from AUC 0.6 → 0.9).
#
# Cluster proportions: 25% bias_in_bios, 19% europarl, 19% amazon,
# 12.5% each of ag_news / github_code / coreference. More balanced
# than FULL (42% bias_in_bios) without dropping any cluster entirely.
PAPER = frozenset({
    "bias_in_bios_set1_prof11", "bias_in_bios_set1_prof2",
    "bias_in_bios_set3_prof20", "bias_in_bios_set3_prof9",
    "europarl_fr", "europarl_de", "europarl_nl",
    "amazon_reviews_cat5", "amazon_reviews_cat3",
    "amazon_reviews_sentiment_5star",
    "ag_news_business", "ag_news_scitech",
    "github_code_java", "github_code_python",
    "winogrande_correct_completion", "wsc_coreference",
})

assert PAPER <= FULL, "PAPER must be a subset of FULL"
assert len(PAPER) == 16
assert len(FULL) == 36

# All paper headline tables / plots / leaderboards filter to HEADLINE.
# Per Han 2026-04-29: PAPER is the finalised paper task set; we iterate
# using PAPER and (if time permits) re-run on FULL at the end as a
# robustness check / supplementary.
HEADLINE = PAPER

# ──────────────────────────────────────────────────────────────────
# Backwards-compatibility aliases (will be removed in a follow-up).
# Earlier code/writeups used FULL_36, PAPER_16, BALANCED_15 names.
# These pre-rename names are still defined so any out-of-tree
# references don't break, but new code should use FULL / PAPER.
FULL_36 = FULL
PAPER_16 = PAPER
