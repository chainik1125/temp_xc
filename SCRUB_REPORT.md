# temp-bench anonymization scrub report

Branch: `temp-bench` (off `origin/final` @ `70a54f5d`).
Worktree: `/workspace/aniket/temp_xc_tempbench`.
No `git push` was performed; commits are local-only on this worktree.

## Commits (newest first)

| Sha | Subject |
|---|---|
| `6a7fb338` | anonymize: scrub residual author tokens and citation paths |
| `6ad639ee` | ship paper-figure renderers + reproduction guide |
| `1398662f` | anonymize: replace placeholder email in add_agent_clone.sh |
| `1a258375` | anonymize: rename author-keyed branch refs and scrub author names |
| `7d2a6228` | anonymize: replace personal GitHub URL with anonymized placeholder |
| `e95b6354` | anonymize: rewrite hf_url field in checkpoints manifest |
| `b2f00dfa` | anonymize: replace personal HF org with `${TEMP_BENCH_HF_ORG}` env var |
| `bab8ae6d` | anonymize: slim decisions log |
| `b27a7678` | anonymize: slim PROTOCOL.md to operational contract |
| `dd709c2a` | anonymize: drop author-keyed wasteland docs and agent briefings |

Total: **391 files changed, 14,247 insertions, 64,908 deletions**.

## File deletions (counts by tree)

| Tree | Deleted |
|---|---:|
| `docs/han/` | 167 files (research logs, literature review, plans) |
| `docs/dmitry/`, `docs/aniket/`, `docs/andre/` | 13 files |
| `docs/shared/`, `docs/templates/`, `docs/Tags.md`, `docs/huggingface-artifacts.md`, `docs/.obsidian/` | 11 files |
| `papers/` | 9 files (external paraphrases) |
| `RUNPOD_INSTRUCTIONS.md` (root) | 1 file |
| `purified/URGENT_HF_SYNC.md` | 1 file |
| `purified/agents/agent_*/` (13 dirs) + agent README + template | ~50 files |
| **Total** | ~252 deletions in commit `dd709c2a` |

## Bulk modifications (counts of files touched)

- HF org rewrite: 18 files (Python, shell, markdown, yaml).
- HF org rewrite in `manifest.jsonl`: 740 of 5,713 rows had their
  `hf_url` field rewritten to use `${TEMP_BENCH_HF_ORG}` placeholder.
  Total row count preserved.
- GitHub URL placeholder: 6 files.
- Branch-name renames + author-name scrub: 92 files (sed +
  python-driven regex pass; 297 individual substitutions; 35
  capitalisation fix-ups across 14 files).
- Renderer port + supporting data: 16 new files committed (8 scripts,
  3 numpy / json files for UMAP, 4 RLHF top-features.json, plus
  `REPRODUCE_FIGURES.md`).
- Final residual scrub (citation paths, `TXC-han` → `TXC-3D`, header
  comments): 31 files.

## Verification grep counts (post-scrub)

| Search | Hits | Notes |
|---|---:|---|
| Word-bounded `\b(han\|aniket\|dmitry\|andre)\b` excl. generated artifacts + renamed branches + `legacy/` paths | 1 | Single PNG binary noise (`c2_setup_d_pB05_np10_eauc_vs_k.png`). Verified via `strings`: only fragment `\}HaN` in random pixel data. |
| `\bbill\b` excluding judge_outputs.jsonl + traces / prompts | 206 | All in `purified/results/runs/c6_*/judge_outputs.jsonl` (LLM-generated rollouts referencing public figures like "Bill Gates"). Acceptable model-output residue per user decision 11. |
| `chainik1125 \| han1823123123 \| hxuany0` | 0 | Clean. |
| Original author-keyed branch tokens (`andre-?(steering\|safety)`, `han-?phase7`, `dmitry-(synthetic\|c6-redteam\|rlhf\|phase8)`, `aniket-(runpod\|ward-stage-b\|phase7-y)`) | 0 | Clean. |
| `em-nanda` excluding `case-em-nanda` | 0 | Clean. |
| Email regex on `*.py *.md *.tex *.sh` excluding `noreply@anthropic` and `anonymous@example` | 1 | Single `${GH_TOKEN}@github.com/anonymous-temp-bench.git` URL (regex false positive — not an email). |
| `manifest.jsonl` row count | 5,713 | Preserved (audit cited 2,252 — file grew between audit and now; structure intact). |
| Renderer count `*_paper_renderer.py` | 7 | All 7 renderers present + `c7_tex_snippets.py` = 8 plotting scripts total. |

All Python files in `purified/` parse (`ast.parse` clean across
`src/`, `tests/`, `experiments/`, `scripts/`).

## Decisions where the user's default fit; deviations

- **Decision 2 (delete agent dirs)**: per the user's default, all 13
  agent dirs were deleted, plus `purified/agents/_briefing_template.md`
  and `purified/agents/README.md` (both contained heavy author-name
  + per-agent-coordination content; would not have survived a scrub).
- **`agent_paper/decisions.md` move (decision 8)**: moved to
  `purified/decisions.md` before the agent-dir delete; then slimmed
  in commit `bab8ae6d`. Kept the 14 numbered locked decisions; dropped
  conversational framing, dated per-agent attribution, and per-agent
  deployment narratives. The slimmed file is 386 lines (was 952).
- **Decision 4 (HF org rewrite + manifest)**: `${TEMP_BENCH_HF_ORG}`
  used everywhere. In Python files I wrapped the placeholder in
  `os.environ.get("TEMP_BENCH_HF_ORG")` calls that raise
  `RuntimeError` if unset (literal `${TEMP_BENCH_HF_ORG}` would have
  been treated as a literal string by Python). Shell scripts use the
  literal `${TEMP_BENCH_HF_ORG}` (interpolated at runtime).
- **Decision 6 (branch renames)**: handled via ordered sed pass.
  Order matters: I substituted `em-nanda → case-em-nanda` after the
  longer-prefix patterns to avoid double-prefixing already-scrubbed
  references. Two `case-case-em-nanda` artefacts in `decisions.md`
  (where I had already pre-written `case-em-nanda` during the slim)
  were caught and fixed. Final scan: 0 hits for old branch names.
- **Wasteland-doc citation paths**: the `docs/{han,dmitry,aniket,andre}/`
  path components leak names through URL fragments even after the
  target docs are deleted. Renamed to `docs/legacy/` across all
  citations (component docs, source-comment provenance, configs).
  This breaks the link target but the targets were already deleted —
  the slug is now neutral.
- **`TXC-han` parameter-layout convention name** in
  `src/temp_bench/training/bricken.py`, `src/temp_bench/case_studies/em.py`,
  and `tests/test_em.py`: renamed to `TXC-3D` (describes the actual
  3-D `W_enc[T, d_in, d_sae]` parameter layout). 7 source-line
  occurrences updated.
- **`prior_reference` rename**: the directory
  `purified/results/c7_backtracking/aniket_reference/` was renamed
  to `prior_reference/`; 19 in-tree references (markdown + Python)
  updated.

## What I did not do (out of scope or by design)

- **No `git push`**, per user instruction. Commits live on the
  `temp-bench` worktree only. The user reviews before push.
- **No actual HF re-upload**. The manifest's `hf_url` placeholders
  point at `${TEMP_BENCH_HF_ORG}/temp-bench-models/...` — reviewers
  setting `TEMP_BENCH_HF_ORG=<org>` and re-uploading the artefacts
  would make those URLs resolvable. Re-upload is the user's call
  (requires their HF credentials).
- **Did not modify `purified/results/leaderboard.jsonl`** (load-bearing
  generated artifact; cited as untouchable).
- **Did not modify `purified/results/runs/c6_*/judge_outputs.jsonl`**
  (model-generated rollouts; the `lzxiao@gmail.com` hit from the
  audit is fabricated text, not real). Numerous "Bill" / "Han"
  matches inside these jsonls are LLM rollout text referring to
  public figures and remain — flagged as expected residue.
- **Did not edit
  `purified/results/c7_backtracking/stage_a/{traces,prompts,sentence_labels}.json`**
  (generated MATH-500 traces; the "Bill" mentions are from the public
  benchmark).
- **Did not push** any branches or worktrees. Worktree boundaries
  respected: `git worktree list` confirms `/workspace/aniket/temp_xc`
  (300k-tfa) and `/workspace/aniket/temp_xc_paper` (final-aniket) were
  not touched, only read from.

## Items to surface to the user

- **Manifest row count grew**. The audit cited 2,252 rows; the live
  file is 5,713 rows. 740 of those had `hf_url` rewritten; the other
  4,973 had `hf_url=null` or did not contain `han1823123123`
  (newer rows whose checkpoints had not been pushed when the row was
  appended). All are now safe regardless.
- **`.claude/commands/*.md`**: word-bounded grep matches inside these
  prompt-template files (`changes`, `unhandled`, `handled`, `hand-typed`,
  etc.) — substring-only, not author identity. Not in the scrub
  scope; flagged for awareness.
- **One PNG binary fragment**: `c2_setup_d_pB05_np10_eauc_vs_k.png`
  contains the byte sequence `\}HaN` in random pixel data. Not a
  real "Han" string — it's binary noise that happens to match the
  regex. Acceptable residue.
- **`Bill Gates` mentions in `judge_outputs.jsonl`** (~200 rows): LLM
  rollout text referencing the public figure. Per user decision 11
  these stay as model-generated content.
- **`figs/` is not gitignored**: the renderers create
  `purified/figs/<component>/` by default. If you don't want
  generated figures committed, add `figs/` to `.gitignore` before
  the user (or a reviewer) runs the renderers.
- **Tests not actually run** (`pytest`): I ran `python -m ast.parse`
  for syntax-validation only. The repo's tests touch HF tokens,
  network, and CUDA, all unavailable here. Recommend the user run
  `pytest -q` once with `TEMP_BENCH_HF_ORG` set before public release.

## Final state

Branch `temp-bench` is at `6a7fb338`, ahead of `origin/final` by 10
commits, working tree clean. Ready for user review.
