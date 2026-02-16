Perform a code quality review.

## Instructions

Read `CLAUDE.md` from the project root for project context.

### Identify the Code

If the user provided a file path in `$ARGUMENTS`, use that. Otherwise:

- Diff the current branch against `main` to find changed `.py` files in `src/` and `tests/`.
- If no Python files changed, inform the user and suggest using `/review-doc` instead.
- If multiple files changed, review all of them (grouped by file).

### Review Dimensions

Read each file thoroughly. Classify findings as:

- **CRITICAL** -- bugs, security issues, data corruption risks, incorrect logic
- **SUGGESTION** -- improvements to design, performance, maintainability
- **STYLE** -- naming, formatting, minor readability

#### 1. Correctness

- Logic errors, off-by-one errors, incorrect conditions
- Unhandled exceptions or error states
- Race conditions (if applicable)
- Incorrect use of libraries or APIs

#### 2. Test Coverage

- Are there corresponding tests in `tests/` for new code in `src/`?
- Do tests cover edge cases and error paths?
- Are test assertions meaningful (not just `assert True`)?
- If no tests exist, flag as SUGGESTION with guidance on what to test

#### 3. Design and Structure

- Functions and classes have clear, single responsibilities
- No unnecessary coupling between modules
- Appropriate use of abstractions
- No duplicated logic that should be factored out

#### 4. Error Handling

- Exceptions are caught at the right level
- Error messages are informative
- Resources are properly cleaned up (context managers, finally blocks)
- Failure modes are documented or obvious

#### 5. Performance

- No obviously quadratic algorithms where linear would work
- Large data operations use appropriate structures (numpy, pandas vs. raw loops)
- No unnecessary memory copies of large arrays/tensors
- I/O operations are buffered appropriately

#### 6. Documentation

- Public functions have docstrings
- Complex logic has inline comments explaining *why*
- Type hints are present on function signatures
- Module-level docstrings explain the module's purpose

### Output Format

Present findings grouped by severity:

```text
## CRITICAL (must fix)
- [file:line] [description]

## SUGGESTIONS (recommended)
- [file:line] [description]

## STYLE (minor)
- [file:line] [description]

## Summary
[2-3 sentence assessment]
```

If reviewing multiple files, group findings by file within each severity section.

## User input

$ARGUMENTS
