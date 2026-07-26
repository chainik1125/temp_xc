"""Two-moment few-shot demonstration-order task: the sprint's own best candidate.

The only construction found tonight that supplies rank >= 2 AND c ~ 0 together. Every
other task got one or the other -- the rotation ladder had rank but c(grad) = 0.16-0.18,
recency had rank 2 but c = 0.067, and the multiset-matched trajectory tasks had c = 0 with
rank 1.

WHY FEW-SHOT DEMONSTRATIONS SUPPLY THE PAIR FOR FREE
    content attribute : the LABEL at position t
    state attribute   : the RUNNING LABEL BALANCE up to t
The state's schedule is the running integral of the content's, and an integral is never
proportional to its integrand (cumsum(v) ~ v only for v = e_T, a single spike at the last
position). So A >= 2 automatically, with no second manipulated dimension to install.

WHY TWO MOMENTS AND NOT ONE
Matching the label multiset zeroes the ZEROTH moment, which kills the content DC. It does
NOT kill the state DC, because

    sum_t cumsum(dc)(t) = sum_j (T-j+1) dc_j = -sum_j j*dc_j

so the state's constant component is the FIRST moment. Multiset-matched-only foils measure
mean c = 0.19 (up to 0.32) on synthetic schedules; adding the first-moment constraint drops
that to ~1e-36 with rank 2 intact. A quadratic accumulator would need the second moment too,
so the pattern below matches moments 0, 1 AND 2.

THE PATTERN
    A = 0 1 0 1 1 1 0 0 0 1 0 1     positions of positives: {2,4,5,6,10,12}, sum 39
    B = 1 0 1 0 0 0 1 1 1 0 1 0     the exact complement
Balanced (6/6), so B is a permutation of A's twelve demonstrations. Matches moment 0 by
balance, moment 1 (39 = 78-39) and moment 2 (325 = 650-325). Hamming distance 12 -- EVERY
position flips, which maximises ||dc|| and so the measurable signal. Verified in
`docs/dmitry/sprints/2026-07-26_txcwins_10h/demo_order.py`.

NON-EXTREMAL REQUIREMENT, which costs an afternoon if missed. An extremal reference such as
[1,1,1,1,1,1,0,0,0,0,0,0] uniquely MINIMISES the first moment over balanced patterns, and a
minimum is attained uniquely, so it admits ZERO valid foils. The pattern above is interior.

WHAT IS AND IS NOT CONTROLLED. The two classes hold the same twelve demonstrations, so the
text multiset matches exactly and the text DC cancels with the label DC. Within a label the
review texts are drawn at random, so within-class text variation averages out of the mean
slab -- the object any fixed write is bounded by.
"""
from __future__ import annotations

# Label pattern pair. 1 = positive demonstration, 0 = negative.
PATTERN_A = (0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1)
PATTERN_B = (1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0)

CARRIER = "Classify the sentiment of each review.\n"

# Disjoint pools, matched for length (9-12 words) and for surface form, so that the label
# is carried by sentiment rather than by style, punctuation or length.
POSITIVE = [
    "The pacing never sagged and the final act earned its ending.",
    "Warm, funny writing that trusts the audience to keep up.",
    "Every performance lands, and the quiet scenes land hardest.",
    "A confident debut with real craft behind the camera.",
    "The score lifts the whole thing without ever overwhelming it.",
    "Sharp dialogue, generous characters, and an ending worth waiting for.",
    "It handles a difficult subject with unusual care and wit.",
    "The photography is gorgeous and the editing keeps it moving.",
    "Genuinely surprising, and it holds together on a second viewing.",
    "A small story told with enormous skill and obvious affection.",
    "The leads have chemistry and the script gives them room.",
    "Beautifully observed, and far more moving than I expected.",
    "It knows exactly what it is and executes it perfectly.",
    "Tense throughout, with a payoff that actually justifies the buildup.",
    "The humour is dry, precise, and never at anyone's expense.",
    "An elegant piece of work that rewards close attention throughout.",
    "The structure is clever without ever feeling pleased with itself.",
    "Rich, unhurried storytelling with a genuinely satisfying conclusion.",
    "Superb ensemble work, and the direction stays out of its way.",
    "It earns every emotional beat, which is rarer than it sounds.",
]
NEGATIVE = [
    "The pacing drags badly and the final act fumbles everything.",
    "Cold, lazy writing that assumes the audience won't notice.",
    "No performance lands, and the quiet scenes are the worst.",
    "A shapeless debut with no real craft behind the camera.",
    "The score smothers the whole thing and never lets up.",
    "Blunt dialogue, thin characters, and an ending that simply stops.",
    "It handles a difficult subject with startling carelessness and glibness.",
    "The photography is drab and the editing kills any momentum.",
    "Entirely predictable, and it falls apart on a second viewing.",
    "A small story told with enormous clumsiness and obvious indifference.",
    "The leads have no chemistry and the script strands them.",
    "Poorly observed, and considerably less moving than I expected.",
    "It has no idea what it is and executes nothing properly.",
    "Inert throughout, with a payoff that badly betrays the buildup.",
    "The humour is broad, clumsy, and usually at someone's expense.",
    "A graceless piece of work that punishes close attention throughout.",
    "The structure is muddled while remaining thoroughly pleased with itself.",
    "Thin, hurried storytelling with a completely unearned conclusion.",
    "Weak ensemble work, and the direction constantly gets in its way.",
    "It earns no emotional beat, which is worse than it sounds.",
]

LABELS = {1: "positive", 0: "negative"}


def _segment(text: str, label: int) -> str:
    return f"Review: {text} Sentiment: {LABELS[label]}."


def moments(pattern) -> tuple[int, int, int]:
    """(zeroth, first, second) moments of a label pattern, 1-indexed positions."""
    return (
        sum(pattern),
        sum((j + 1) * v for j, v in enumerate(pattern)),
        sum((j + 1) ** 2 * v for j, v in enumerate(pattern)),
    )


# A multiset-matched foil whose FIRST MOMENT differs (21 against A's 39). This is the control
# arm: identical generator, identical pattern A, identical pools -- the only thing that changes
# is the first-moment constraint, which is the design's actual novelty. Anything else would
# confound the constraint with the content.
PATTERN_B_FREE = (1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0)


def make_demo_order(k_seg: int = 12, pattern_a=PATTERN_A, pattern_b=PATTERN_B, pool="all",
                    allow_moment_mismatch=False):
    """Factory in the harness contract: make(k_seg) -> make_pair(rng).

    Class A is label 1. Both classes hold the SAME twelve demonstrations; only the
    positions of the positive ones differ, under a permutation matched on the first three
    moments of the label sequence.
    """
    if len(pattern_a) != k_seg or len(pattern_b) != k_seg:
        raise ValueError(
            f"patterns are length {len(pattern_a)}/{len(pattern_b)}, k_seg is {k_seg}; "
            "supply a moment-matched pair of the right length "
            "(see docs/dmitry/sprints/2026-07-26_txcwins_10h/demo_order.py)"
        )
    ma, mb = moments(pattern_a), moments(pattern_b)
    if ma[0] != mb[0]:
        raise ValueError(f"label multisets differ: {ma[0]} vs {mb[0]} positives")
    if ma[1] != mb[1] and not allow_moment_mismatch:
        raise ValueError(
            f"first moments differ ({ma[1]} vs {mb[1]}); the carried state will leave a "
            "constant component and c will not vanish"
        )
    # Held-out content. "disjoint pools" in the header means POSITIVE disjoint from NEGATIVE,
    # which is necessary but is NOT a train/eval split: without one, the dictionary is scored
    # on the same demonstrations it trained on and the claim is only "steers the ordering of
    # content it was trained on". Splitting the pools in half gives the stronger claim.
    assert pool in ("train", "eval", "all"), f"pool={pool!r}"
    _h_pos, _h_neg = len(POSITIVE) // 2, len(NEGATIVE) // 2
    _pos_pool = {"train": POSITIVE[:_h_pos], "eval": POSITIVE[_h_pos:], "all": POSITIVE}[pool]
    _neg_pool = {"train": NEGATIVE[:_h_neg], "eval": NEGATIVE[_h_neg:], "all": NEGATIVE}[pool]

    n_pos = ma[0]
    n_neg = k_seg - n_pos
    assert n_pos <= len(_pos_pool) and n_neg <= len(_neg_pool), (
        f"k_seg={k_seg} needs {n_pos}/{n_neg} demonstrations, pool {pool!r} has "
        f"{len(_pos_pool)}/{len(_neg_pool)}")

    def make_pair(rng):
        # One draw of twelve demonstrations, placed into both patterns. Class B is then a
        # permutation of class A rather than an independent sample -- an independent draw
        # would match the label counts only in expectation and can leave a lexical
        # imbalance pointing at the factor under test, which a CONSTANT write can exploit.
        pos = rng.sample(_pos_pool, n_pos)
        neg = rng.sample(_neg_pool, n_neg)

        def lay(pattern):
            pi, ni, out = 0, 0, []
            for v in pattern:
                if v:
                    out.append(_segment(pos[pi], 1)); pi += 1
                else:
                    out.append(_segment(neg[ni], 0)); ni += 1
            return out

        return lay(pattern_a), lay(pattern_b), CARRIER

    return make_pair


# Held-out ambiguous reviews for PROBE MODE. The query is shared between the two classes,
# so its own valence cancels in the difference of differences; what survives is the effect
# of demonstration ORDER on the query's predicted label, which is the documented in-context
# learning effect (majority-label and recency bias) rather than a statement about which
# demonstration ordering is more probable.
QUERY = [
    "It has moments that work and stretches that plainly do not.",
    "There is real craft here, though the story never quite arrives.",
    "Ambitious and uneven, sometimes in the very same scene.",
    "I admired more of it than I actually enjoyed watching.",
    "Competent throughout, but it rarely reaches for anything more.",
    "The parts are better assembled than the whole ever manages.",
]
QUERY_PREFIX = "Review: "
QUERY_SUFFIX = " Sentiment:"
CONT_POS = " positive"
CONT_NEG = " negative"


def make_demo_order_probe(k_seg: int = 12, pattern_a=PATTERN_A, pattern_b=PATTERN_B,
                          pool="all", allow_moment_mismatch=False):
    """Probe-mode factory: returns (sents_a, sents_b, carrier, cont1, cont2).

    Score is logP(" positive" | doc+query) - logP(" negative" | doc+query), and the reported
    quantity is that score for A minus for B. A write that simply pushes "positive" moves
    both classes equally and cancels exactly, so only a write treating positions differently
    can move it.
    """
    base = make_demo_order(k_seg, pattern_a, pattern_b, pool=pool,
                           allow_moment_mismatch=allow_moment_mismatch)

    def make_pair(rng):
        sa, sb, car = base(rng)
        q = QUERY[rng.randrange(len(QUERY))]
        # The query goes in BOTH continuations, not appended to a segment. Appending it to
        # the last segment would break the exact multiset match the whole design rests on,
        # since the last segment differs between the two classes. Here the shared query
        # prefix cancels exactly in logP(q+pos) - logP(q+neg), leaving the query's predicted
        # LABEL as the readout while sents_a and sents_b stay exact permutations.
        stem = f" {QUERY_PREFIX}{q}{QUERY_SUFFIX}"
        return sa, sb, car, stem + CONT_POS, stem + CONT_NEG

    return make_pair


# `demo_order` is ORDERING mode -- it scores logP(doc A) - logP(doc B), i.e. which arrangement
# of movie reviews is the more probable text. That is a fluency judgement with no connection to
# in-context learning, and running it by accident would answer a question nobody asked. It is
# kept only because its n=200 geometry screen is reported; the probe factory is the one to run.
DESIGNS = {"demo_order_ordering_ONLY_FOR_SCREEN": make_demo_order,
           "demo_order_probe": make_demo_order_probe,
           "demo_order_probe_tr": lambda k: make_demo_order_probe(k, pool="train"),
           "demo_order_probe_ev": lambda k: make_demo_order_probe(k, pool="eval"),
           "demo_order_probe_free": lambda k: make_demo_order_probe(
               k, pattern_b=PATTERN_B_FREE, allow_moment_mismatch=True)}


if __name__ == "__main__":
    import collections
    import random

    for nm, p in (("A", PATTERN_A), ("B", PATTERN_B)):
        print(f"pattern {nm} = {p}   moments (0,1,2) = {moments(p)}")
    assert moments(PATTERN_A) == moments(PATTERN_B), "moment match broken"
    print(f"Hamming distance = {sum(a != b for a, b in zip(PATTERN_A, PATTERN_B))} of "
          f"{len(PATTERN_A)}  (every position flips)")

    mp = make_demo_order(12)
    a, b, car = mp(random.Random(0))
    assert len(a) == len(b) == 12
    assert sorted(a) == sorted(b), "demonstration multiset differs between classes"
    print(f"\nmultiset identical: True    segments: {len(a)}")
    la = [1 if "positive" in s else 0 for s in a]
    lb = [1 if "positive" in s else 0 for s in b]
    print(f"label sequence A: {la}\nlabel sequence B: {lb}")
    assert tuple(la) == PATTERN_A and tuple(lb) == PATTERN_B
    print(f"label counts: A {dict(collections.Counter(la))}  B {dict(collections.Counter(lb))}")
    wa = [len(s.split()) for s in a]
    print(f"segment word counts: min {min(wa)}  max {max(wa)}  mean {sum(wa)/len(wa):.1f}")
    print(f"\ncarrier: {car!r}")
    print(f"A[0]: {a[0]}")
    print(f"B[0]: {b[0]}")
