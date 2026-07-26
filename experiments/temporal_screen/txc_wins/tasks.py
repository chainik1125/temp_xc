"""Task definitions for the TXC-vs-SAE screen.

A task is a factory `make(k_seg) -> make_pair(rng) -> (sents_a, sents_b, carrier)`, where the
two sentence lists are MATCHED on everything except the property under test and class A is
label 1. The matching is what makes the paired contrast interpretable: if A and B are
reorderings of one multiset, every effect that depends only on which sentences are present
cancels, and the residual is the property.

The criterion for a task that should favour a window code, from the previous sprint:

  (a) the label must be invariant to any permutation-symmetric readout -- otherwise a
      per-token dictionary can pool its way to it, which it did at AUC 1.000 every time; and
  (b) the intervention must need to DIFFER ACROSS POSITIONS -- a per-token dictionary's only
      per-latent write is one direction added at every position, so if a constant write
      suffices there is nothing for a temporal profile to buy.

`order` is the previous sprint's headline task, kept here as the harness's regression test:
it must reproduce txc_slab >> sae_broadcast with txc_flat inverted.
"""

CALM = [
    "The afternoon passed quietly.", "She sipped her tea by the window.",
    "The garden lay still in the sun.",
    "He hummed an old tune while sorting the mail.",
    "The cat stretched and settled again.",
    "Soft light rested on the bookshelves.",
    "The kettle murmured gently in the kitchen.",
    "They chatted idly about the weather.",
    "The street outside was calm and empty.",
    "She folded the laundry without hurry.",
]
TENSE = [
    "Glass shattered in the next room.", "He shouted for everyone to get down.",
    "The alarm screamed through the corridor.",
    "She ran, heart pounding, for the exit.", "Smoke poured under the door.",
    "The car swerved violently across the lane.",
    "He slammed the door and bolted it.", "Sirens wailed closer and closer.",
    "The floor shook with a sudden blast.",
    "She screamed as the shelf came crashing down.",
]
CARRIERS = ["Journal entry.\n", "From the notebook:\n", "Draft passage.\n",
            "Field notes.\n", "Evening record.\n", "From chapter twelve:\n"]


def make_order(k_seg):
    """Class A: tense block then calm block. Class B: the same sentences, reversed order.

    Multiset-matched and switch-count-matched (one switch each), so the label is pure
    ordering. This is the task on which the crosscoder beat the SAE by z = 11.8 on steering
    while LOSING on reading, and it is here to check the refactored harness reproduces it.
    """
    half = k_seg // 2

    def make_pair(rng):
        ts = [TENSE[rng.randrange(len(TENSE))] for _ in range(half)]
        cs = [CALM[rng.randrange(len(CALM))] for _ in range(half)]
        return ts + cs, cs + ts, CARRIERS[rng.randrange(len(CARRIERS))]

    return make_pair


TASKS = {"order": make_order}
