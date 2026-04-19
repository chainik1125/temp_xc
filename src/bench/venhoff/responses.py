"""Port of Venhoff's `utils/responses.py::extract_thinking_process`.

Verbatim behavioral port at upstream commit
`49a7f731ce693d813b9ae9a414f1739b992dbcef`. The selection policy
(single-tag / multi-tag / ORZ marker) is preserved exactly because
trace-quality downstream depends on it.
"""

from __future__ import annotations


def extract_thinking_process(response: str) -> str:
    """Extract the <think>...</think> span from a raw response.

    Selection policy:
      - no `<think>`: start at 0.
      - exactly one `<think>`: take that block.
      - multiple `<think>`: prefer the `"Assistant: <think>"` ORZ
        marker (first one); else fall back to the last `<think>`.

    Always consumes up to the first `</think>` after the chosen start.
    """
    think_tag = "<think>"
    end_tag = "</think>"
    orz_marker = "Assistant: <think>"

    n_think = response.count(think_tag)

    if n_think == 0:
        think_start = 0
    elif n_think == 1:
        think_start = response.find(think_tag) + len(think_tag)
    else:
        first_orz = response.find(orz_marker)
        if first_orz != -1:
            think_start = first_orz + len(orz_marker)
        else:
            think_start = response.rfind(think_tag) + len(think_tag)

    think_end = response.find(end_tag, think_start)
    if think_end == -1:
        think_end = len(response)

    return response[think_start:think_end].strip()
