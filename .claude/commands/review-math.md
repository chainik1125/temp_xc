Perform a mathematical correctness review of a document.

## Instructions

Read `CLAUDE.md` from the project root for project context.

### Identify the Document

If the user provided a file path in `$ARGUMENTS`, use that. Otherwise:

- Diff the current branch against `main` to find changed `.md` files in `docs/`.
- Filter for files that contain mathematical content (LaTeX blocks, equations, proofs, theorems, definitions).
- If no math-heavy files are found, inform the user and suggest using `/review-doc` instead.
- If multiple candidates, list them and ask.

### Review Dimensions

Read the document thoroughly with focus on mathematical rigor. Classify findings as:

- **CRITICAL** -- mathematical errors that invalidate results or mislead the reader
- **SUGGESTION** -- improvements to mathematical presentation or rigor
- **STYLE** -- notation preferences and formatting

#### 1. Notation Consistency

- Variables, operators, and symbols are used consistently throughout
- The same symbol does not represent different things in different sections
- Notation is introduced before first use
- Standard notation is preferred over ad-hoc notation where conventions exist

#### 2. Definitions and Assumptions

- All key terms are formally defined before use
- Assumptions are stated explicitly (not buried in prose)
- Definitions are precise and unambiguous
- Domain restrictions and edge cases are noted

#### 3. Proof Logic and Derivations

- Each step follows logically from the previous
- No gaps in reasoning (implicit steps are flagged)
- Boundary conditions and edge cases are handled
- If a result is cited, the reference is provided
- Implications vs. bi-conditionals are used correctly

#### 4. Equation Correctness

- Verify arithmetic and algebraic manipulations
- Check dimensional consistency (units, shapes, types)
- Verify matrix dimensions are compatible in operations
- Check index ranges in summations and products
- Verify limits, boundary conditions, and special cases

#### 5. Statistical and Probabilistic Claims

- Distributions are specified with parameters
- Independence assumptions are stated
- Confidence intervals or significance levels are reported where applicable
- Sample sizes and experimental conditions are noted
- Claims about convergence or optimality cite supporting theory

#### 6. Figures and Tables (mathematical content)

- Axis labels, units, and legends are present and correct
- Captions accurately describe the content
- Data in tables is consistent with claims in prose
- Mathematical notation in figures matches the text

### Output Format

Present findings grouped by severity:

```text
## CRITICAL (must fix)
- [description] (line X) -- [brief explanation of the error]

## SUGGESTIONS (recommended)
- [description] (line X)

## STYLE (minor)
- [description] (line X)

## Summary
[2-3 sentence assessment of mathematical soundness]
```

If no critical issues, say so explicitly. Note any areas where you cannot fully verify correctness (e.g., domain-specific results that require experimental validation).

## User input

$ARGUMENTS
