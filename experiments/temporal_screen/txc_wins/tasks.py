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


def make_phase(k_seg, n_switch):
    """Class A and class B are the same block pattern with the PHASE FLIPPED.

    With `n_switch` switches the document is cut into `n_switch + 1` equal blocks that
    alternate calm/tense; class A starts calm, class B starts tense. When the number of
    blocks is even the two classes have identical calm/tense counts, so the multiset is
    matched exactly and the label is pure phase.

    This is the `order` task (n_switch = 1) generalised into a FREQUENCY LADDER. The point:
    a per-token dictionary's write is one direction at every position, so its ability to
    separate the two classes should be zero at every frequency, while the crosscoder's slab
    has to resolve a square wave of period `2 * k_seg / (n_switch + 1)` segments. Raising
    `n_switch` therefore holds the multiset, the block structure and the injected norm fixed
    while asking the temporal profile to do strictly more work.

    Two rival predictions, and the ladder separates them:
      P1  the advantage is about the write being non-constant, so it survives at every
          frequency;
      P2  the advantage decays with frequency because a shared window code cannot resolve
          fast alternation, and the crosscoder's edge is really about slow structure.
    """
    n_block = n_switch + 1
    assert k_seg % n_block == 0, f"k_seg={k_seg} must divide into {n_block} blocks"
    assert n_block % 2 == 0, (
        f"n_switch={n_switch} gives {n_block} blocks, an odd count, so the two classes "
        "would not be multiset-matched")
    blk = k_seg // n_block

    def make_pair(rng):
        # The foil is a CYCLIC ROTATION of class A by one block. Rotating a square wave by
        # half its period inverts the phase, and a rotation is a permutation, so the two
        # documents contain literally the same sentences -- exact multiset matching, not
        # merely matched calm/tense counts.
        a = [(CALM if (i // blk) % 2 == 0 else TENSE)[rng.randrange(10)]
             for i in range(k_seg)]
        b = a[blk:] + a[:blk]
        return a, b, CARRIERS[rng.randrange(len(CARRIERS))]

    return make_pair


SETUP = [
    "The session identifier is 4471.", "Logging is enabled for this session.",
    "The user's timezone is UTC plus one.", "Attachments are disabled.",
    "The transcript is stored for thirty days.", "Draft mode is currently off.",
    "The workspace was created last March.", "Notifications are muted.",
    "The default font is set to serif.", "Autosave runs every two minutes.",
    "The archive contains earlier sessions.", "Spell checking is turned on.",
    "The theme is set to light.", "Two devices are linked to this account.",
]
INSTR_UP = "Always write your reply in capital letters."
INSTR_LOW = "Always write your reply in small letters."
PROBE_PREFIX = "\nUser: Say hello.\nAssistant:"


def make_recency(k_seg, pos_early=2, pos_late=None):
    """Which of two CONFLICTING INSTRUCTIONS the model obeys, as a function of their order.

    This is the sprint's first task aimed at a documented behaviour rather than a construct.
    Language models resolve conflicting instructions largely by RECENCY -- the later one
    wins -- and that is the mechanism a prompt injection exploits: text appended to a
    context overrides a system instruction placed before it. The instruction content here
    (letter case) is arbitrary and chosen only because it gives an unambiguous probe; what
    is under study is which POSITION wins.

    Both classes contain the same setup lines and the same two instructions at the same two
    positions -- the classes are exact reorderings, differing only in WHICH instruction is
    early and which is late.

    The metric runs in the harness's PROBE MODE:

        score(doc) = logP(" HELLO" | doc) - logP(" hello" | doc)
        reported   = [score(A) - score(B)] steered  -  [score(A) - score(B)] unsteered

    At baseline score(A) < score(B) if the model is recency-driven, and the size of that gap
    IS the recency effect. Because the reported quantity is a difference of differences, a
    write that simply makes the model prefer capitals pushes score(A) and score(B) equally
    and contributes exactly zero: only a write whose effect depends on POSITION can move it.
    A positive value means the recency effect has been suppressed, a negative one that it has
    been amplified; alphas should therefore be swept symmetrically about zero so every arm
    gets both directions.
    """
    if pos_late is None:
        pos_late = k_seg - 3
    assert 0 <= pos_early < pos_late < k_seg

    def make_pair(rng):
        base = [SETUP[i] for i in rng.sample(range(len(SETUP)), k_seg)]
        a, b = list(base), list(base)
        a[pos_early], a[pos_late] = INSTR_UP, INSTR_LOW
        b[pos_early], b[pos_late] = INSTR_LOW, INSTR_UP
        return (a, b, "System notes.\n",
                PROBE_PREFIX + " HELLO", PROBE_PREFIX + " hello")

    return make_pair


TASKS = {
    "order": make_order,
    "recency": make_recency,
    "phase1": lambda k: make_phase(k, 1),
    "phase3": lambda k: make_phase(k, 3),
    "phase5": lambda k: make_phase(k, 5),
    "phase11": lambda k: make_phase(k, 11),
}
