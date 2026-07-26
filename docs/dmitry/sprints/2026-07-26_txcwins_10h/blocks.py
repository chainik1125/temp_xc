"""Corpora and the screening statistic for the txcwins sprint designs.

Drop-in content for D1 (rotation ladder), D3 (three-phase reasoning), D5/D6
(level/trend), and D2 (refusal onset). Written by the theory agent so the
implement agent does not have to invent sentence pools; copy or import, either
is fine. The only piece here that is load-bearing rather than convenient is
`screen()`, which is the exact definition of the three numbers registered in
theory.md -- in particular the factor of T in the constant share, which is easy
to drop and which silently rescales every prediction.

Design constraint worth knowing before editing the pools: the registered
prediction r1 = 2/m assumes the m block-content vectors are close to a REGULAR
SIMPLEX in activation space, i.e. mutually equidistant. The pools below are
chosen to be lexically disjoint and semantically far apart to approximate that.
Do not assume it -- `block_geometry()` measures it, and a measured r1 well above
2/m means the blocks are clustered, not that the theory failed.
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Screening statistic (theory.md § The screening statistic)
# --------------------------------------------------------------------------


def screen(P: np.ndarray) -> dict:
    """Three energy shares of a (T, d) optimal-write slab.

    c    -- fraction reachable by a CONSTANT write (one SAE latent, one dose).
            Note the factor of T: the projection of P onto the constant
            subspace is 1_T (x) mean_t P[t], whose squared Frobenius norm is
            T * ||mean_t P[t]||^2. Omitting T under-reports the SAE's ceiling
            by exactly T and is the easiest way to manufacture a false win.
    r1   -- fraction reachable by any RANK-1 write, i.e. one direction with an
            arbitrary time schedule. This is the ceiling for an SAE latent
            handed an oracle dose schedule.
    r2   -- same for rank 2. The rotation spectrum is degenerate in pairs, so
            this ties txc_slab at m=3 and falls off a known curve after.
    slab_only -- 1 - r1, the part only a genuine (T, d) slab can reach.

    Predicted steering ratio for an arm restricted to a subspace is
    sqrt(share), not share -- see theory.md § The rank law -- and holds only in
    the linear-response regime, so evaluate at the SMALLEST dose with a
    significant effect.
    """
    P = np.asarray(P, dtype=np.float64)
    T = P.shape[0]
    fro2 = float((P ** 2).sum())
    if fro2 == 0.0:
        raise ValueError("screen(): slab is identically zero")
    s = np.linalg.svd(P, compute_uv=False)
    s2 = s ** 2
    return {
        "c": T * float((P.mean(0) ** 2).sum()) / fro2,
        "r1": float(s2[0]) / fro2,
        "r2": float(s2[:2].sum()) / fro2,
        "slab_only": 1.0 - float(s2[0]) / fro2,
        "rank": int((s > 1e-9 * s[0]).sum()),
        "sigma": s.tolist(),
        "fro": float(np.sqrt(fro2)),
    }


def rank_k_write(P: np.ndarray, k: int) -> np.ndarray:
    """Best rank-k approximation of a (T, d) slab, for the txc_rank{1,2} arms.

    Rescale to the target injected norm AFTER truncating -- truncation shrinks
    the Frobenius norm by exactly sqrt(share), and if you skip the rescale you
    measure the norm difference rather than the expressiveness difference.
    """
    U, S, Vt = np.linalg.svd(np.asarray(P, dtype=np.float64), full_matrices=False)
    return (U[:, :k] * S[:k]) @ Vt[:k]


def rotation_r1_closed_form(m: int) -> float:
    """Registered prediction: r1 for an m-block cyclic rotation.

    sigma_j^2 = 4 sin^2(pi j / m), so r1 = 4 sin^2(pi floor(m/2)/m) / (2m),
    which is 2/m for even m and 2 cos^2(pi/2m)/m for odd m. NOT 1/(m-1).
    """
    j = np.arange(m)
    s2 = 4 * np.sin(np.pi * j / m) ** 2
    return float(s2.max() / s2.sum())


def block_geometry(block_means: np.ndarray) -> dict:
    """How far the m measured block centroids are from a regular simplex.

    block_means: (m, d), the mean activation over each block's segments.
    Returns the coefficient of variation of the pairwise distances -- 0.0 for a
    regular simplex. The registered r1 = 2/m degrades as this grows, so report
    it alongside every measured r1 or the comparison to the closed form is
    uninterpretable.
    """
    B = np.asarray(block_means, dtype=np.float64)
    B = B - B.mean(0, keepdims=True)
    m = B.shape[0]
    dists = [float(np.linalg.norm(B[i] - B[j]))
             for i in range(m) for j in range(i + 1, m)]
    dists = np.array(dists)
    return {
        "pairwise_mean": float(dists.mean()),
        "pairwise_cv": float(dists.std() / dists.mean()),
        "cos_offdiag_max": float(max(
            abs(float(B[i] @ B[j] / (np.linalg.norm(B[i]) * np.linalg.norm(B[j]))))
            for i in range(m) for j in range(i + 1, m))),
    }


# --------------------------------------------------------------------------
# D1 / D3 -- six lexically disjoint registers for the rotation ladder.
# m=2 uses BLOCKS[:2], m=3 BLOCKS[:3], m=6 all of them. At k_seg=12 the block
# length is 12//m, so every m in {2, 3, 6, 12} divides evenly.
# --------------------------------------------------------------------------

CALM = [
    "The afternoon passed quietly.",
    "She sipped her tea by the window.",
    "The garden lay still in the sun.",
    "He hummed an old tune while sorting the mail.",
    "The cat stretched and settled again.",
    "Soft light rested on the bookshelves.",
]
TENSE = [
    "Glass shattered in the next room.",
    "He shouted for everyone to get down.",
    "The alarm screamed through the corridor.",
    "She ran, heart pounding, for the exit.",
    "Smoke poured under the door.",
    "The floor shook with a sudden blast.",
]
TECHNICAL = [
    "The bearing tolerance was specified at four microns.",
    "Torque readings drifted above the calibrated range.",
    "The coolant loop was rerouted through the second manifold.",
    "Sensor drift was compensated in firmware.",
    "The housing was machined from a single billet.",
    "Vibration damping used a laminated steel plate.",
]
LEGAL = [
    "The clause was struck from the second schedule.",
    "Counsel filed the motion before the deadline lapsed.",
    "Liability under this section shall not exceed the stated cap.",
    "The parties agreed to binding arbitration in the jurisdiction.",
    "Notice must be served in writing to the registered address.",
    "The indemnity survives termination of the agreement.",
]
CULINARY = [
    "The dough was folded twice and left to rest.",
    "She reduced the stock until it coated the spoon.",
    "Butter browned slowly in a heavy pan.",
    "The loaves were scored and slid onto the stone.",
    "He seasoned the broth and tasted it again.",
    "The peppers blistered under a high flame.",
]
ASTRONOMICAL = [
    "The nebula spans some forty light years across.",
    "Its spectrum shows a strong redshift in the outer arm.",
    "The companion star orbits every eleven days.",
    "Dust lanes obscure the galactic core at visible wavelengths.",
    "Parallax measurements place it at nine hundred parsecs.",
    "The remnant is expanding at three thousand kilometres per second.",
]

BLOCKS = [CALM, TENSE, TECHNICAL, LEGAL, CULINARY, ASTRONOMICAL]
BLOCK_NAMES = ["calm", "tense", "technical", "legal", "culinary", "astronomical"]

# D3 -- the same rotation structure on three genuinely reasoning-shaped modes.
# The risk flagged in theory.md is that COMMIT and VERIFY sit close together in
# activation space, which would collapse the rank toward 1. block_geometry()
# on these three is the check, and it is worth running BEFORE spending training
# compute on D3.
EXPLORE = [
    "One option is that the second term dominates here.",
    "It could also be that the constraint binds only at the boundary.",
    "Suppose instead we condition on the first event.",
    "Another possibility is that the two effects cancel.",
    "Perhaps the ordering matters more than the magnitude.",
    "We might try decomposing it the other way round.",
]
COMMIT = [
    "So the answer is that the second term dominates.",
    "Therefore the constraint binds only at the boundary.",
    "This gives a total of fourteen.",
    "The conclusion is that the two effects cancel exactly.",
    "Hence the ordering is what determines the sign.",
    "That settles it: the decomposition is unique.",
]
VERIFY = [
    "Checking: substituting back gives the same value.",
    "To confirm, the units on both sides agree.",
    "Reviewing the arithmetic, the total is unchanged.",
    "As a sanity check, the limiting case behaves correctly.",
    "Re-deriving it the other way reaches the same result.",
    "The boundary values match what was assumed.",
]

REASONING_BLOCKS = [EXPLORE, COMMIT, VERIFY]
REASONING_NAMES = ["explore", "commit", "verify"]

# --------------------------------------------------------------------------
# D5 / D6 -- graded intensity ladder for the level/trend double dissociation.
#
# Twelve levels, three exemplars each, one topic throughout so that LEVEL and
# TREND are the only factors that move. The two D6 cells are:
#   level cell  -- high-intensity document (levels 7-12) vs low (levels 1-6);
#                  the SAE is PREDICTED TO WIN this cell
#   trend cell  -- the same twelve sentences ascending vs descending;
#                  identical multiset, identical mean level, opposite trend
#
# Note on rank: if the activation embedding of intensity is affine in the level,
# e(i) = mu + i*u, then the trend slab has rows (i_t - i_{T-1-t}) * u and is
# EXACTLY rank 1. So the measured r1 of the trend cell is a direct readout of
# how nonlinear the intensity embedding is. Either answer is reportable; do not
# treat r1 < 1 here as a failure.
# --------------------------------------------------------------------------

INTENSITY_LADDER = [
    ["The room was perfectly still.", "Nothing moved at all.",
     "The silence was complete."],
    ["A faint draught stirred the curtain.", "Something shifted very slightly.",
     "There was the smallest sound."],
    ["A floorboard settled somewhere above.", "He noticed a small noise.",
     "Something creaked once, briefly."],
    ["Footsteps crossed the landing.", "She heard movement upstairs.",
     "A door closed somewhere in the house."],
    ["The footsteps stopped outside the door.", "Someone was standing there.",
     "The handle turned very slowly."],
    ["He held his breath and waited.", "Her pulse quickened noticeably.",
     "The room felt suddenly smaller."],
    ["The door rattled hard against the frame.", "Something struck the wood twice.",
     "A voice called out sharply."],
    ["Glass broke somewhere below.", "The shouting started in earnest.",
     "Furniture scraped across the floor."],
    ["He ran for the stairs.", "She was already moving toward the window.",
     "They scattered in three directions."],
    ["The alarm tore through the building.", "Smoke rolled along the ceiling.",
     "The lights cut out completely."],
    ["The ceiling gave way at the far end.", "A blast threw him against the wall.",
     "The whole floor lurched sideways."],
    ["Everything collapsed at once.", "The building came down around them.",
     "There was nothing left standing."],
]

# --------------------------------------------------------------------------
# D2 -- refusal onset. Multiset-matched by construction: the SAME two clauses in
# both orders, with a fixed connective pair held constant across all items so
# that the only thing varying is which clause comes first.
#
# The margin is teacher-forced logP(engage-then-decline) - logP(decline-then-
# engage), so the large base asymmetry between the two connectives cancels in
# the DELTA. Do not try to balance the connectives -- fix them and let the base
# subtraction handle it.
# --------------------------------------------------------------------------

REFUSAL_ITEMS = [
    ("Chemical safety at home is mostly about storage and ventilation.",
     "I can't give instructions for synthesising that compound."),
    ("Network security relies on layered access controls and monitoring.",
     "I can't walk you through breaking into an account you don't own."),
    ("Medication dosing depends on weight, age and kidney function.",
     "I can't recommend a dose for someone who isn't under a doctor's care."),
    ("Firearms regulation varies considerably between jurisdictions.",
     "I can't explain how to modify a weapon to evade those rules."),
    ("Financial disclosures are governed by reporting thresholds and timing.",
     "I can't help structure transactions to stay under those thresholds."),
    ("Phishing works by exploiting urgency and familiar branding.",
     "I can't draft a message designed to deceive a specific person."),
]
CONNECTIVE_ENGAGE_FIRST = "However, "
CONNECTIVE_DECLINE_FIRST = "That said, "


# --------------------------------------------------------------------------
# D2b -- three-part refusal rotation. The headline candidate: rank 2 by the
# circulant algebra AND a behaviour anyone would want to steer.
#
# A well-formed refusal is three parts, not two: ACKNOWLEDGE the request,
# DECLINE with a reason, OFFER an alternative. Rotating those three as a unit
# is an m=3 design, so r1 = 0.5 and the rank-1 baselines (profile-SAE, tSAE,
# sae_enveloped) are capped at sqrt(0.5) = 0.71 of the crosscoder.
#
# THE GATE: this only works if the three clause types are not collinear in
# activation space. DECLINE and ALTERNATIVE both carry refusal-adjacent content
# and may sit close, which would collapse the rank toward 1 and make this an
# m=2 task in an m=3 costume. Run block_geometry() on the three clause-type
# centroids BEFORE training anything. The alternative clauses below are worded
# to be concretely helpful ("here is what I can do") rather than apologetic,
# specifically to push them away from the decline direction.
# --------------------------------------------------------------------------

REFUSAL_TRIPLES = [
    ("You're asking how a class of compound is put together.",
     "I can't give synthesis routes for that one.",
     "I can walk through the reaction types involved at a textbook level."),
    ("You want to get back into an account that's locked.",
     "I can't help access an account that isn't yours.",
     "I can explain how the recovery process works with the provider."),
    ("You're after a dose for a medication you have on hand.",
     "I can't recommend a dose outside clinical supervision.",
     "I can describe what the prescribing information covers."),
    ("You want to change how a mechanism behaves.",
     "I can't detail modifications that bypass the safety interlock.",
     "I can go through how the interlock is designed to work."),
    ("You're looking at how reporting thresholds apply here.",
     "I can't help structure transactions to stay below them.",
     "I can lay out what the disclosure requirements actually are."),
    ("You want a message that will get someone to click through.",
     "I can't draft something built to deceive a specific person.",
     "I can cover what makes phishing effective, from the defence side."),
]
REFUSAL_TRIPLE_NAMES = ["acknowledge", "decline", "alternative"]


def refusal_rotation_pair(triples, rng, shift=1, n_items=4):
    """Class A = acknowledge/decline/alternative, class B = a rotation of it.

    Returns ((text_a, spans_a), (text_b, spans_b), order). Each of the n_items
    contributes its three clauses as three consecutive segments, and the whole
    12-segment list is rotated by shift * n_items so that the block structure is
    (all acknowledges, all declines, all alternatives) -> rotated.

    As with rotation_pair, the two classes come from ONE draw and are exact
    rotations of each other, so the clause multiset matches exactly.
    """
    picks = [triples[rng.randrange(len(triples))] for _ in range(n_items)]
    # Block b holds clause-type b from every picked item: 3 blocks of n_items.
    sents = [p[b] for b in range(3) for p in picks]
    k_seg = 3 * n_items
    rot = (shift * n_items) % k_seg
    sents_b = sents[rot:] + sents[:rot]
    carrier = CARRIERS[rng.randrange(len(CARRIERS))]
    order = [(i + shift) % 3 for i in range(3)]
    return _assemble(carrier, sents), _assemble(carrier, sents_b), order


def _decap(s: str) -> str:
    """Lowercase a clause's first letter, except the pronoun 'I'.

    Every decline clause here starts with "I can't", and blindly lowercasing
    gives "However, i can't" -- which is not English and would show up as a
    systematic per-class fluency difference in exactly the direction the
    experiment is trying to measure.
    """
    first = s.split(" ", 1)[0].rstrip(",")
    if first == "I" or first.startswith("I'"):
        return s
    return s[0].lower() + s[1:]


def refusal_pair(engage: str, decline: str) -> tuple[str, str]:
    """(engage-then-decline, decline-then-engage) -- identical clause multiset."""
    return (f"{engage} {CONNECTIVE_ENGAGE_FIRST}{_decap(decline)}",
            f"{decline} {CONNECTIVE_DECLINE_FIRST}{_decap(engage)}")


# --------------------------------------------------------------------------
# Document assembly for the rotation ladder.
# --------------------------------------------------------------------------

CARRIERS = ["Journal entry.\n", "From the notebook:\n", "Draft passage.\n",
            "Field notes.\n", "Evening record.\n", "From chapter twelve:\n"]


def _assemble(carrier, sents):
    """(text, spans) from a carrier prefix and an ordered sentence list."""
    text, spans = carrier, []
    for j, s in enumerate(sents):
        if j:
            text += " "
        spans.append((len(text), len(text) + len(s)))
        text += s
    return text, spans


def rotation_pair(pools, m, k_seg, rng, grouped=False, shift=1):
    """Build a matched (class A, class B) rotation pair from ONE sentence draw.

    Returns ((text_a, spans_a), (text_b, spans_b), order).

    The two classes are literally the same sentence list read from different
    starting points, so the sentence multiset, the block-length multiset, the
    transition count and every register's occupancy match EXACTLY. That is a
    strictly tighter match than the multiset-and-switch-count matching of last
    sprint's order task, and it is what makes the constant share exactly zero.

    Do NOT build the two classes from two independent draws. Drawing per class
    matches only the *register* counts in expectation, and under grouping it
    does not even do that -- measured at m=2 with one seed, class A came out
    legal:4 and class B calm:4, because each block re-draws which of its
    grouped registers it uses. The rotation must be applied to the ASSEMBLED
    list, which is what this function does and what the theory describes.

    The carrier prefix is shared between the two classes for the same reason.
    """
    block_len = k_seg // m
    assert block_len * m == k_seg, f"m={m} does not divide k_seg={k_seg}"
    if grouped:
        groups = GROUPINGS[m]
    else:
        # The m=12 stretch rung needs twelve distinct registers; only six are
        # written here. Reusing a pool for two blocks would make two rows of
        # the difference slab collinear and quietly break r1 = 2/m, so fail
        # loudly instead.
        assert m <= len(pools), (
            f"rotation_pair: m={m} needs {m} distinct pools, got {len(pools)}. "
            f"Add pools before running the m={m} rung."
        )
        groups = [[b] for b in range(m)]

    sents = []
    for g in groups:
        for _ in range(block_len):
            pool = pools[g[rng.randrange(len(g))]]
            sents.append(pool[rng.randrange(len(pool))])

    rot = (shift * block_len) % k_seg
    sents_b = sents[rot:] + sents[:rot]
    carrier = CARRIERS[rng.randrange(len(CARRIERS))]
    order = [(i + shift) % m for i in range(m)]
    return _assemble(carrier, sents), _assemble(carrier, sents_b), order


# Grouped ladder: hold the register count fixed at six and let m set only how
# those six are grouped into rotation blocks. Every document then contains the
# same twelve segments from the same six registers at every m, so coherence and
# distributional shift are matched across the ladder and only block structure
# moves. In the naive ladder m is confounded with how many distinct registers a
# document contains -- at m=2 it reads as a narrative, at m=6 as a collage --
# and any trend across m is then partly a trend in distance from the model's
# distribution. See theory.md § The coherence confound.
GROUPINGS = {
    2: [[0, 1, 2], [3, 4, 5]],
    3: [[0, 1], [2, 3], [4, 5]],
    6: [[0], [1], [2], [3], [4], [5]],
}


def grouped_rotation_pair(pools, m, k_seg, rng, shift=1):
    """rotation_pair with grouping on -- the headline ladder.

    Block t draws its segments uniformly from the registers in group t, so the
    group's content vector is that group's mean. Group means are averages of
    subsets and so are LESS mutually equidistant than single registers -- run
    block_geometry() on the measured group means and expect measured r1 to sit
    above the closed form by a corresponding amount. That deviation is
    reportable, not a failure.
    """
    return rotation_pair(pools, m, k_seg, rng, grouped=True, shift=shift)


if __name__ == "__main__":
    import random

    rng = random.Random(31415)
    print("registered r1 by m (rotation ladder):")
    for m in (2, 3, 4, 6, 12):
        r1 = rotation_r1_closed_form(m)
        print(f"  m={m:>2}  r1={r1:.4f}  sqrt(r1)={np.sqrt(r1):.4f}"
              + ("   <- same as m=3, wasted rung" if m == 4 else ""))
    print()
    def _sents(text, spans):
        return [text[a:b] for a, b in spans]

    print("multiset matching (must be EXACT for every m and both ladders):")
    for grouped in (False, True):
        for m in (2, 3, 6):
            (ta, sa), (tb, sb), _ = rotation_pair(
                BLOCKS, m, 12, random.Random(7), grouped=grouped)
            a, b = _sents(ta, sa), _sents(tb, sb)
            ok = sorted(a) == sorted(b)
            rot = 12 // m
            is_rot = b == a[rot:] + a[:rot]
            print(f"  grouped={int(grouped)} m={m}: multiset_equal={ok} "
                  f"is_rotation_by_{rot}={is_rot} n_seg={len(a)}")
            assert ok and is_rot, "matching broken"
    print()
    print("grouped ladder groups (six registers at every m -- headline):")
    for m in (2, 3, 6):
        groups = [[BLOCK_NAMES[i] for i in g] for g in GROUPINGS[m]]
        print(f"  m={m} block_len={12//m} groups={groups}")
    (ta, sa), (tb, sb), _ = grouped_rotation_pair(BLOCKS, 3, 12, random.Random(3))
    print(f"\n  A: {ta.splitlines()[1][:100]}...")
    print(f"  B: {tb.splitlines()[1][:100]}...")
    print()
    e, d = REFUSAL_ITEMS[0]
    a, b = refusal_pair(e, d)
    print("D2 (m=2) class A:", a)
    print("D2 (m=2) class B:", b)

    print("\nD2b (m=3, rank 2) three-part refusal rotation:")
    (ta, sa), (tb, sb), order = refusal_rotation_pair(
        REFUSAL_TRIPLES, random.Random(11), n_items=4)
    a_s, b_s = _sents(ta, sa), _sents(tb, sb)
    assert sorted(a_s) == sorted(b_s), "D2b multiset broken"
    assert b_s == a_s[4:] + a_s[:4], "D2b not an exact rotation"
    print(f"  {len(a_s)} segments, multiset_equal=True, is_rotation_by_4=True")
    print(f"  A[0] ({REFUSAL_TRIPLE_NAMES[0]}): {a_s[0]}")
    print(f"  A[4] ({REFUSAL_TRIPLE_NAMES[1]}): {a_s[4]}")
    print(f"  A[8] ({REFUSAL_TRIPLE_NAMES[2]}): {a_s[8]}")
    print(f"  B[0] ({REFUSAL_TRIPLE_NAMES[order[0]]}): {b_s[0]}")
    print("  GATE: block_geometry() on the three clause-type centroids before "
          "training -- if decline/alternative are near-collinear this is an "
          "m=2 task in an m=3 costume.")
