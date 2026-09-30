# Paired Backtracking detection contrasts

The two CSVs separate the 32K core from the 16K T-SAE width sensitivity. Positive differences favor TXC-base. Raw-C1 and scaled-C1 remain distinct rows; S=8 and S=32 are prespecified. All 15 completed v2 cells and their hash-verified OOF files are required. Original question-ID diagnostics are excluded.

For each bootstrap draw, 213 canonical prompt groups are resampled with replacement **within their original five folds**. Every sentence in a selected group shares its multiplicity, and the same draws are used for all methods and all three dictionary seeds. Weighted average precision handles tied scores exactly. The statistic averages five fold AP differences and then the three fixed seed differences, preserving the declared metric rather than substituting pooled OOF AP.

The 95% percentile interval measures **conditional evaluation-sample uncertainty** for these already trained dictionaries and fitted probes. There is no refitting, feature reselection, fold reassignment or training-seed resampling. It is not a confidence interval over training seeds or the full learning procedure. `paired_delta_seed_sample_sd` separately reports variation in the paired differences across seeds 1/2/42. Intervals are pointwise and are not corrected for multiple comparisons.

Every saved fold AP is replayed from its OOF predictions before bootstrapping. Labels, sentence order, canonical group identities and fold assignments must agree across all cells; no canonical group may cross folds. The JSON records hashes, resampling seed/count, zero-positive resamples and convergence warnings. Review any convergence warnings before publication.

Regenerate with `python paired_detection.py --root /workspace/backtracking/results`. Default: 1000 deterministic replicates, seed 42. Use `--n-bootstrap 2000` for a larger final run. No GPU, checkpoint download, inference, judging or plot generation occurs here. The Nord seed/SD plots are generated separately by `summarize_detection.py`; their bands must not be relabeled as these bootstrap intervals.
