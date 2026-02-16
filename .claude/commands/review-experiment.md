Perform an experimental methodology review.

## Instructions

Read `CLAUDE.md` from the project root for project context.

### Identify the Content

If the user provided a file path in `$ARGUMENTS`, use that. Otherwise:

- Diff the current branch against `main` to find changed files.
- Look for experiment-related content: documents tagged with `results` or `proposal`, Python files in experiment directories, config files (`.yml`, `.yaml`, `.json`, `.toml`) that define experimental parameters.
- If no experiment-related files are found, inform the user and suggest a different reviewer.
- If multiple candidates, list them and ask.

### Review Dimensions

Read all relevant files thoroughly. Classify findings as:

- **CRITICAL** -- methodological flaws that invalidate results
- **SUGGESTION** -- improvements to experimental rigor or clarity
- **STYLE** -- presentation preferences

#### 1. Methodology

- Controls: are appropriate baselines or control conditions defined?
- Ablations: are key components tested in isolation where relevant?
- Experimental design: is the setup appropriate for the claims being made?
- Are there confounding variables that are not controlled for?

#### 2. Reproducibility

- Random seeds: are they set and documented?
- Environment: are dependencies, versions, and hardware specified?
- Configs: are all hyperparameters and settings captured (not hardcoded in scripts)?
- Data: is the data source specified and accessible?
- Steps: could another researcher reproduce this from the description alone?

#### 3. Metrics

- Are the chosen metrics appropriate for the task?
- Are metrics clearly defined (not assumed to be obvious)?
- Statistical significance: are results tested for significance where applicable?
- Error bars, confidence intervals, or variance reported?
- Are metrics computed on the correct data split?

#### 4. Data Pipeline

- Train/validation/test splits: are they defined and non-overlapping?
- Data leakage: is there any risk of test data influencing training?
- Preprocessing: are transformations documented and applied consistently?
- Are data sizes and distributions described?

#### 5. Hyperparameters

- Search strategy: how were hyperparameters chosen (grid search, manual, etc.)?
- Sensitivity: are results robust to hyperparameter changes?
- Are default values justified or just inherited?
- Are the final chosen values reported?

#### 6. Claims vs. Evidence

- Do the results actually support the conclusions drawn?
- Are limitations acknowledged?
- Is overgeneralization avoided (claiming more than the data shows)?
- Are negative or unexpected results reported honestly?
- Are comparisons to prior work fair and accurate?

### Output Format

Present findings grouped by severity:

```text
## CRITICAL (must fix)
- [description] (file:line or section reference)

## SUGGESTIONS (recommended)
- [description] (file:line or section reference)

## STYLE (minor)
- [description] (file:line or section reference)

## Summary
[2-3 sentence assessment of experimental soundness]
```

If no critical issues, say so explicitly.

## User input

$ARGUMENTS
