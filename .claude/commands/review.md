Run a targeted review by auto-detecting content type, or specify a review type.

## Instructions

Read `CLAUDE.md` from the project root for context.

### Parse Arguments

`$ARGUMENTS` may contain:

- A file path (review that file)
- A review type: `doc`, `math`, `code`, `experiment`, `data`, or `all`
- Both a type and a path (e.g., `math docs/Theory.md`)
- Nothing (auto-detect from branch diff)

### Auto-Detection Logic

If no type is specified, determine what to review:

1. Diff the current branch against `main` to find all changed files.
2. Categorize each changed file:
   - `.md` files in `docs/` (excluding templates, .obsidian) -> **doc**
   - `.md` files containing LaTeX (`$$`, `$...$`, `\begin{`, `\frac`, `\sum`, `\int`) -> **math** (in addition to doc)
   - `.py` files in `src/` or `tests/` -> **code**
   - Config files (`.yml`, `.yaml`, `.json`, `.toml`) or data files -> **data**
   - Files tagged `results` or `proposal`, or files with experiment configs -> **experiment**
3. Report what was detected:

```text
Detected changes:
- 2 documentation files (1 with mathematical content)
- 3 Python files
- 1 config file

Running: /review-doc, /review-math, /review-code, /review-data
```

### Dispatch

For each detected (or specified) type, perform the corresponding review inline. Read and follow the exact review dimensions and output format defined in the matching command file:

- **doc**: Read and follow `.claude/commands/review-doc.md`
- **math**: Read and follow `.claude/commands/review-math.md`
- **code**: Read and follow `.claude/commands/review-code.md`
- **experiment**: Read and follow `.claude/commands/review-experiment.md`
- **data**: Read and follow `.claude/commands/review-data.md`

### Combined Output

Present a unified report with findings from all reviewers:

```text
# Review Report

## Documentation Review
[findings from review-doc dimensions]

## Mathematical Review
[findings from review-math dimensions, if applicable]

## Code Review
[findings from review-code dimensions, if applicable]

## Experiment Review
[findings from review-experiment dimensions, if applicable]

## Data Review
[findings from review-data dimensions, if applicable]

## Overall Summary
[2-3 sentences covering all aspects reviewed]
[Total: X critical, Y suggestions, Z style findings]
```

If `$ARGUMENTS` specifies a single type (e.g., `math`), only run that reviewer and use its native output format instead of the combined format.

## User input

$ARGUMENTS
