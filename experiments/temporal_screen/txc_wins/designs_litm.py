"""Lost in the Middle: the canonical positional effect, with the foil shipped by the authors.

Liu et al., *Lost in the Middle: How Language Models Use Long Contexts* (TACL 2024),
`nelson-liu/lost-in-the-middle`. Ten retrieved documents, one of which answers the question,
and the same item is released with the gold document at position 0, 4 and 9.

WHY THIS IS THE BEST STRUCTURAL FIT WE HAVE RUN. Every other task needed a matched foil we
built and a segmentation rule we had to justify. Here both are given:

  * the foil SHIPS -- same question, same answers, and the document multiset is identical
    between files. **Verified across all 2655 items, not sampled**: 0 question mismatches,
    0 title-multiset mismatches, 0 gold-index violations.
  * the segmentation IS the document boundary. No splitting at spaces, no delimiter to
    preserve, no rule to defend.

Because the multiset match is exact and positional-only, a write that merely makes the gold
answer more likely shifts A and B equally and cancels. This is the P1 argument in its
cleanest available form.

TWO DEVIATIONS FROM THE OBVIOUS CONFIG, both forced, both stated.

  1. `k_seg = 11`, NOT 10. Liu's prompt is
     `instruction\\n\\n{documents}\\n\\nQuestion: {q}\\nAnswer:` and the harness builds text as
     `carrier + seg0 + " " + seg1 + ...` with nothing after the last segment -- so the
     question block has nowhere to live unless it is a segment. Folding it into segment 10
     would put the write on the question tokens while pretending it covers only document 10.
     Segment 11 is therefore the question block, and segments 1-10 are exactly one retrieved
     document each. This is strictly more informative: the write profile now says how much
     weight the optimal write puts on the question versus the documents.
  2. Each document line carries ONE TRAILING SPACE that Liu's does not, because the harness
     joins segments with `" "` and the newline is carried at the START of each segment. Line
     STARTS are byte-identical to Liu's; only the line ends differ, by one space. Recorded
     because it is a real difference from the published prompt, and it is the smallest one
     available given the harness's builder.

METRIC. Probe mode, both continuations identical between A and B:

    cont1 = the item's gold answer
    cont2 = a different item's gold answer (deterministic, index-offset -- a plausible
            short answer that is wrong for this question)
    score(doc) = logP(cont1 | doc) - logP(cont2 | doc)
    reported   = score(A) - score(B)

Using a wrong-answer contrast rather than a bare `logP(answer)` makes the metric
discriminative rather than a fluency measure, and the difference-of-differences means any
length or frequency bias in either continuation cancels between A and B.

RELEVANCE, stated honestly: this is a retrieval-reliability paper, not a safety paper. The
safety-adjacent reading is RAG poisoning -- an adversarial document's influence depending on
where it lands in the context -- and it should not be sold past that. What it is, is the
canonical positional effect in the literature, positional by construction, with an exactly
matched foil. If a window code cannot beat a constant write here, it is hard to see where
it would.
"""
import gzip
import json
import pathlib

_HERE = pathlib.Path(__file__).resolve().parent
_DATA = _HERE / "litm_data"

# Verbatim from lost_in_the_middle/prompts/qa.prompt and prompting.py:get_qa_prompt.
PROMPT_HEAD = ("Write a high-quality answer for the given question using only the provided "
               "search results (some of which might be irrelevant).\n\n")
DOC_FMT = "Document [{i}](Title: {title}) {text}"
QUESTION_BLOCK = "\n\nQuestion: {question}\nAnswer:"

_CACHE = {}


def _load(gold_at):
    if gold_at not in _CACHE:
        p = _DATA / f"gold_at_{gold_at}.jsonl.gz"
        with gzip.open(p, "rt") as fh:
            _CACHE[gold_at] = [json.loads(l) for l in fh]
    return _CACHE[gold_at]


def _segments(item):
    """One segment per retrieved document, plus the question block as the last segment.

    The newline lives at the START of each segment after the first, so the harness's
    `" ".join` puts line breaks exactly where Liu does; only line ends gain a space.
    """
    segs = []
    for i, c in enumerate(item["ctxs"]):
        d = DOC_FMT.format(i=i + 1, title=c["title"], text=c["text"])
        segs.append(d if i == 0 else "\n" + d)
    segs.append(QUESTION_BLOCK.format(question=item["question"]))
    return segs


def make_litm(k_seg=11, gold_a=0, gold_b=9, pool="all"):
    """A = gold document at `gold_a`, B = the same item with gold at `gold_b`."""
    A, B = _load(gold_a), _load(gold_b)
    n = min(len(A), len(B))
    idx = list(range(n))
    h = n // 2
    idx = {"train": idx[:h], "eval": idx[h:], "all": idx}[pool]

    def make_pair(rng):
        for _ in range(200):
            j = idx[rng.randrange(len(idx))]
            a, b = A[j], B[j]
            if a["question"] != b["question"] or not a["answers"]:
                continue
            sa, sb = _segments(a), _segments(b)
            if len(sa) != k_seg or len(sb) != k_seg:
                continue          # not a 10-document item
            # A wrong-but-plausible short answer: another item's gold answer.
            other = A[(j + 1) % n]
            if not other["answers"] or other["answers"][0] == a["answers"][0]:
                other = A[(j + 7) % n]
            return (sa, sb, PROMPT_HEAD,
                    " " + a["answers"][0], " " + other["answers"][0])
        raise RuntimeError("no usable item; check k_seg matches the document count")

    return make_pair


# gold_at_0 vs gold_at_9 is end-versus-end; gold_at_0 vs gold_at_4 straddles the trough the
# paper is named for and may show a LARGER effect. Both registered so the baseline decides.
DESIGNS = {
    "litm_0v9": lambda k: make_litm(k, gold_a=0, gold_b=9),
    "litm_0v4": lambda k: make_litm(k, gold_a=0, gold_b=4),
    "litm_4v9": lambda k: make_litm(k, gold_a=4, gold_b=9),
    "litm_0v9_tr": lambda k: make_litm(k, gold_a=0, gold_b=9, pool="train"),
    "litm_0v9_ev": lambda k: make_litm(k, gold_a=0, gold_b=9, pool="eval"),
}
