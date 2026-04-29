"""Phase 7 task-set definitions — single source of truth for the 36 / 15
SAEBench task subsets.

The 36-task set is the original Phase 7 standard (`probe_cache_S32/`
contains all 36 task dirs). The 15-task balanced set is the
paper-headline reduction — preserves k=20 top-3 ranking and k=5 top-6
cluster identity at 2.4× speedup. Rationale + per-task SD analysis in
`docs/han/research_logs/phase7_unification/agent_x_paper/2026-04-29-task-importance.md`.

All paper plots / leaderboard tables that go into the headline use
`BALANCED_15`. The full 36-task set is reported in supplementary.
"""
from __future__ import annotations

# All 36 SAEBench tasks at S=32 (Phase 7 current methodology, FLIP on
# winogrande / wsc). Computed by enumerating probe_cache_S32/.
FULL_36 = frozenset({
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

# Balanced 15-task headline set:
#   5 bias_in_bios   (top-5 SD: prof11, prof2, prof22, prof20, prof9)
#   2 europarl       (top-2 SD: it, fr)
#   2 amazon_cat     (top-2 SD: cat5, cat3)
#   1 amazon_sentiment
#   1 ag_news        (top SD: business)
#   2 github_code    (top-2 SD: java, python)
#   2 coreference    (winogrande, wsc — multi-token-dependence story)
# Cluster proportions approximately match full-36 (bias_in_bios 33% vs full 42%).
BALANCED_15 = frozenset({
    "bias_in_bios_set1_prof11", "bias_in_bios_set1_prof2",
    "bias_in_bios_set2_prof22", "bias_in_bios_set3_prof20",
    "bias_in_bios_set3_prof9",
    "europarl_it", "europarl_fr",
    "amazon_reviews_cat5", "amazon_reviews_cat3",
    "amazon_reviews_sentiment_5star",
    "ag_news_business",
    "github_code_java", "github_code_python",
    "winogrande_correct_completion", "wsc_coreference",
})

assert BALANCED_15 <= FULL_36, "BALANCED_15 must be a subset of FULL_36"
assert len(BALANCED_15) == 15
assert len(FULL_36) == 36
