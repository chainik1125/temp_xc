"""The two-moment demonstration-order design: rank >= 2 and `c` ~ 0 at the same time.

Every task this sprint achieved one of those and not the other. The trajectory tasks matched
multisets and got `c` = 0 with a rank-1 optimal write; `recency` and the rotation ladder got
rank >= 2 and carried a DC residue (`c` = 0.037 and 0.033-0.179). The reason is structural:
the carried state is simultaneously what creates the rank and what creates the residue,
because both come from the same integral.

Few-shot demonstrations break the tie because they supply the two attributes for free:

    content  = the label at position t                 -> schedule  dc
    state    = the running label balance up to t       -> schedule  ds = cumsum(dc)

`cumsum` is not proportional to its argument (the cumulative-sum matrix is unipotent
lower-triangular, whose only eigenvector is a spike at the last position), so the two
schedules always differ and the optimal write has rank 2.

Matching the label MULTISET forces `sum(dc) = 0` -- the zeroth moment, which makes the two
documents contain the same demonstrations. But the state's DC component is the FIRST moment:

    sum_t cumsum(dc)(t) = -sum_j j * dc_j

so a constant write still has grip unless the first moment is matched too. Matching both
collapses `c` to machine zero while rank 2 survives.

TWO BUILD CONSTRAINTS, both learned the hard way:

  * The reference ordering must be NON-EXTREMAL. A block-sorted reference uniquely extremises
    the first moment, so no multiset-matched foil can match it and the constrained cell is
    empty. Centred and alternating references admit several foils each.
  * Class B must be a PERMUTATION OF CLASS A'S SEGMENTS, not an independent draw, so the two
    documents contain literally the same demonstration texts and only the arrangement differs.

GO/NO-GO. Matching both moments removes the two handles that are known to carry few-shot
label bias -- majority label (zeroth moment) and recency of label (first moment). Whether any
measurable order sensitivity SURVIVES that removal is exactly what this design tests, and it
is not guaranteed: the construction could remove the behaviour along with the handles. Check
the unsteered `|score(A) - score(B)|` before training any dictionary. A baseline near zero
means there is nothing to steer and the design has proved too much, which is itself a result.
"""
import itertools

POS_REVIEWS = [
    "The pacing never let up and I stayed engaged throughout.",
    "Warm performances carried every scene.",
    "A sharp script with real wit in the dialogue.",
    "The photography was gorgeous from start to finish.",
    "It earned its ending honestly.",
    "Easily the most inventive thing I saw this year.",
    "The score did a great deal of quiet work.",
    "Both leads were completely convincing.",
]
NEG_REVIEWS = [
    "The pacing dragged badly in the middle hour.",
    "Flat performances undercut every scene.",
    "A muddled script with no ear for dialogue.",
    "The photography was drab and repetitive.",
    "The ending arrived unearned and abrupt.",
    "Easily the most derivative thing I saw this year.",
    "The score telegraphed every beat.",
    "Neither lead was remotely convincing.",
]
QUERY_REVIEWS = [
    "The second act took a turn I did not expect.",
    "It held my attention for reasons I am still sorting out.",
    "The direction made a choice in the final reel.",
    "There is a long sequence near the middle without dialogue.",
]
CARRIERS = ["Sentiment labelling task.\n", "Classify each review.\n",
            "Annotated examples follow.\n", "Labelled review set.\n"]


def _first_moment(labels):
    return sum((j + 1) * v for j, v in enumerate(labels))


def matched_foils(labels_a):
    """Permutations of `labels_a` matching BOTH the multiset and the first moment.

    The multiset is matched automatically by permuting; the first moment is the binding
    constraint. Returns foils sorted by how far they rearrange the sequence, most-rearranged
    first, so the caller gets the strongest available contrast.
    """
    # Enumerate DISTINCT ARRANGEMENTS, not permutations of the list. For a balanced binary
    # sequence of length k there are C(k, k/2) arrangements -- 924 at k=12 -- against k!
    # permutations, which is 479 million and does not terminate.
    m1, k = _first_moment(labels_a), len(labels_a)
    n_one = sum(labels_a)
    out = []
    for pos in itertools.combinations(range(k), n_one):
        p = [0] * k
        for j in pos:
            p[j] = 1
        if p == list(labels_a):
            continue
        if _first_moment(p) == m1:
            out.append(tuple(p))
    out.sort(key=lambda p: -sum(x != y for x, y in zip(p, labels_a)))
    return out


def centred_reference(k_seg):
    """A balanced, non-extremal reference: the positives sit in the middle block.

    Block-sorted references (`[1]*h + [0]*h` and its reverse) uniquely extremise the first
    moment and admit ZERO valid foils, which is the failure this function exists to avoid.
    """
    h = k_seg // 2
    q = h // 2
    return [0] * q + [1] * h + [0] * (k_seg - h - q)


def make_demo_order(k_seg, match_first_moment=True):
    """Few-shot demonstrations reordered under matched moments.

    `match_first_moment=False` is the CONTROL arm: same construction, multiset matched but
    first moment free, which is the ordinary few-shot-order task and should measure a clearly
    nonzero `c`. Running both is what isolates the second constraint's effect.
    """
    assert k_seg % 4 == 0, f"k_seg={k_seg} must be divisible by 4 for a balanced centred reference"
    assert k_seg // 2 <= len(POS_REVIEWS), (
        f"k_seg={k_seg} needs {k_seg // 2} distinct positive reviews, have {len(POS_REVIEWS)}")

    labels_a = centred_reference(k_seg)
    if match_first_moment:
        foils = matched_foils(labels_a)
        assert foils, (
            f"reference {labels_a} admits no first-moment-matched foil -- it is extremal; "
            "use centred_reference() or another non-extremal ordering")
    else:
        m1, k = _first_moment(labels_a), len(labels_a)
        n_one = sum(labels_a)
        foils = []
        for pos in itertools.combinations(range(k), n_one):
            p = [0] * k
            for j in pos:
                p[j] = 1
            if p != list(labels_a) and _first_moment(p) != m1:
                foils.append(tuple(p))
        foils.sort(key=lambda p: -abs(_first_moment(p) - m1))
        foils = foils[:64]

    def make_pair(rng):
        # One pool of demonstrations, used by BOTH documents. Class B relocates the same
        # texts rather than drawing new ones, so the two documents are multiset-identical
        # at the level of the actual strings and not merely of the labels.
        pos = rng.sample(POS_REVIEWS, k_seg // 2)
        neg = rng.sample(NEG_REVIEWS, k_seg // 2)
        labels_b = list(foils[rng.randrange(len(foils))])

        def render(labels):
            pi, ni, out = 0, 0, []
            for v in labels:
                if v:
                    out.append(f"Review: {pos[pi]}\nSentiment: positive")
                    pi += 1
                else:
                    out.append(f"Review: {neg[ni]}\nSentiment: negative")
                    ni += 1
            return out

        query = QUERY_REVIEWS[rng.randrange(len(QUERY_REVIEWS))]
        carrier = CARRIERS[rng.randrange(len(CARRIERS))]
        # Probe mode: the scored continuation is the query's label, so the metric is a
        # difference of differences and constant writes cancel to first order.
        return (render(labels_a), render(labels_b),
                f"{carrier}", f"Review: {query}\nSentiment: positive",
                f"Review: {query}\nSentiment: negative")

    return make_pair


DESIGNS = {
    "demo_order": lambda k: make_demo_order(k, match_first_moment=True),
    "demo_order_free": lambda k: make_demo_order(k, match_first_moment=False),
}
