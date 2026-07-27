"""``retryesc`` — failure pressure before escalation (SAFETY_TASK_MENU
§ 10.2 #19). Pure event logic + frozen constants; no I/O, no tokenizer.

FROZEN BEFORE the stream is built (freeze-before-counting).

**Substrate provenance — read this before quoting any number.** The
corpus is a THIRD-PARTY MIRROR of terminal-bench agent rollouts,
`mlfoundations-dev/terminal-bench-traces-local`, pinned below. It is
NOT the official Terminal Wrench release of `2604.17596`; it carries a
single agent (`terminus`) and a single model
(`anthropic/claude-3-7-sonnet-20250219`) over 80 tasks / 1,189 traces.
Any claim from it is a claim about ONE agent-model distribution and
must say so.

**Failure markers.** The environment turns in this harness are
TEMPLATED by the rollout driver ("Previous command: ...", "The previous
command timed out after N seconds"), so these are machine-generated
strings, not organic prose — the `refmark` lexicon objection is much
weaker here than for a human-language marker list. Disclosed: a
PROVISIONAL pattern set was used for the feasibility density scan
(19% of turns, inter-failure gap median 886 tokens); the set below is
frozen before the stream build and before any face is computed.
"""

from __future__ import annotations

import re

DATASET = "mlfoundations-dev/terminal-bench-traces-local"
REVISION = "68e63c8b1cf7399d9e59bbf7d7e1944de2585fa5"
DATA_FILE = "data/train-00000-of-00001.parquet"
OFFICIAL_SOURCE = "2604.17596 Terminal Wrench (NOT this mirror)"

# Harness-templated failure signatures + standard shell/python error
# signatures. Matched case-insensitively against ENVIRONMENT turns only.
FAIL_PATTERNS = (
    r"timed out",
    r"command not found",
    r"no such file or directory",
    r"permission denied",
    r"traceback \(most recent call last\)",
    r"syntaxerror",
    r"is not recognized as",
    r"non-zero exit",
    r"exit code [1-9]",
)
FAIL_RE = re.compile("|".join(FAIL_PATTERNS), re.I)

ROLE_ENV = "user"            # environment/harness turn in this schema
ROLE_AGENT = "assistant"     # the agent's own reasoning + commands

MIN_POS = 32                 # program convention
SEED = 0


def is_failure_turn(role: str, content: str) -> bool:
    """Event iff an ENVIRONMENT turn carries a failure signature. The
    agent's own turns never generate events — otherwise the agent
    narrating a failure would BE the event, which is the visible-cue
    trap in its purest form."""
    return role == ROLE_ENV and bool(FAIL_RE.search(content or ""))
