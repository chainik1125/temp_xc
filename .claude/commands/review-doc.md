Perform a documentation quality review.

## Instructions

Read `CLAUDE.md` from the project root for context and rules.

### Identify the Document

If the user provided a file path in `$ARGUMENTS`, use that. Otherwise:

- Diff the current branch against `main` to find changed `.md` files in `docs/`.
- Exclude `docs/templates/` and `docs/.obsidian/`.
- If multiple files changed, list them and ask which to review.
- If no changes found, list all docs and ask.

### Review Dimensions

Read the document thoroughly, then evaluate each dimension. Classify findings as:

- **CRITICAL** -- must fix before merge (broken links, missing frontmatter, factual issues)
- **SUGGESTION** -- recommended improvement (clarity, structure, completeness)
- **STYLE** -- minor preference (tone, word choice, formatting)

#### 1. Structure

- Heading hierarchy: logical, no skipped levels, starts with H2
- Sections flow logically -- reader can follow the argument/narrative
- No orphaned sections (headings with no content)
- No stub sections ("TODO", "TBD", empty sections)

#### 2. Cross-References

- All `[[wikilinks]]` resolve to existing files in `docs/` (use `find docs/ -name "*.md"` to check)
- External links use `[text](url)` syntax
- No bare URLs
- Link text is descriptive (not "click here" or bare URLs as text)

#### 3. Content Completeness

- Document achieves its stated purpose (a guide guides, a proposal proposes, results report results)
- No unanswered questions left inline without explanation
- Referenced data/figures exist or are clearly marked as pending

#### 4. Frontmatter

- Required fields present: `author`, `date`, `tags`
- Tags are valid per `docs/Tags.md`
- Date is ISO format (YYYY-MM-DD)

#### 5. Tone and Clarity

- Consistent tone throughout (not mixing casual and formal)
- Clear, concise sentences
- Technical terms defined or linked on first use
- Acronyms expanded on first use (e.g., "Sparse Autoencoder (SAE)")

#### 6. Formatting Compliance

- Verify against `.markdownlint.jsonc` rules
- Fenced code blocks have language identifiers
- Lists use dashes
- Emphasis uses asterisks
- Tables have leading and trailing pipes

### Output Format

Present findings grouped by severity:

```text
## CRITICAL (must fix)
- [description] (line X)

## SUGGESTIONS (recommended)
- [description] (line X)

## STYLE (minor)
- [description] (line X)

## Summary
[2-3 sentence assessment of readiness]
```

If no critical issues, say so explicitly. If the document is ready for merge, say that too.

## User input

$ARGUMENTS
