Perform a data quality review.

## Instructions

Read `CLAUDE.md` from the project root for project context.

### Identify the Content

If the user provided a file path in `$ARGUMENTS`, use that. Otherwise:

- Diff the current branch against `main` to find changed files.
- Look for data-related content: config files (`.yml`, `.yaml`, `.json`, `.toml`), data schemas, CSV/TSV files, data loading code, pipeline scripts.
- If no data-related files are found, inform the user and suggest a different reviewer.
- If multiple candidates, list them and ask.

### Review Dimensions

Read all relevant files thoroughly. Classify findings as:

- **CRITICAL** -- errors that would cause data corruption, loss, or incorrect results
- **SUGGESTION** -- improvements to data handling, documentation, or robustness
- **STYLE** -- formatting and naming preferences

#### 1. Schema and Format

- Are data types consistent (no mixed types in columns, no implicit type coercion)?
- Are fields documented (column names, units, expected ranges)?
- Are file formats appropriate for the data size and use case?
- Are encodings specified (UTF-8, etc.)?

#### 2. Completeness

- Are there missing values? How are they handled (dropped, imputed, flagged)?
- Is data coverage sufficient for the intended analysis?
- Are there unexpected gaps in time series, indices, or categorical coverage?

#### 3. Configs

- Are YAML/JSON/TOML configs syntactically valid?
- Do referenced file paths exist?
- Are there hardcoded absolute paths that should be relative?
- Are sensitive values (API keys, passwords) excluded from version control?
- Are default values sensible?

#### 4. Pipeline Correctness

- Are data transformations applied in the correct order?
- Are joins/merges using the correct keys?
- Are aggregations computed correctly (group-by keys, aggregation functions)?
- Is data filtered before or after transformations as intended?
- Are intermediate results validated or spot-checked?

#### 5. Versioning and Provenance

- Is the data source documented (where did this data come from)?
- Are there checksums or hashes for verifying data integrity?
- Is there a clear versioning scheme if data changes over time?
- Are data generation scripts or download instructions provided?

#### 6. Documentation

- Do data files have accompanying README or header comments?
- Are column definitions, units, and value ranges documented?
- Are preprocessing steps documented?
- Are known data quality issues or caveats noted?

### Output Format

Present findings grouped by severity:

```text
## CRITICAL (must fix)
- [description] (file:line or file reference)

## SUGGESTIONS (recommended)
- [description] (file:line or file reference)

## STYLE (minor)
- [description] (file:line or file reference)

## Summary
[2-3 sentence assessment of data quality]
```

If no critical issues, say so explicitly.

## User input

$ARGUMENTS
