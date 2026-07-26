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


def rotation_doc(pools, m, k_seg, shift, rng):
    """Build one document and its segment spans.

    pools : list of m sentence pools
    shift : 0 for class A (canonical order), 1 for class B (rotated by one block)

    The two classes are the SAME m blocks read from different starting points,
    so the sentence multiset, the block-length multiset, the transition count
    and every block type's total occupancy match exactly. That is a strictly
    tighter match than the multiset-and-switch-count matching of last sprint's
    order task, and it is what makes the constant share exactly zero.
    """
    block_len = k_seg // m
    assert block_len * m == k_seg, f"m={m} does not divide k_seg={k_seg}"
    # The m=12 stretch rung needs twelve distinct registers; only six are
    # written here. Reusing a pool for two blocks would make two rows of the
    # difference slab collinear and quietly break the r1 = 2/m prediction, so
    # fail loudly instead.
    assert m <= len(pools), (
        f"rotation_doc: m={m} needs {m} distinct pools, got {len(pools)}. "
        f"Add pools before running the m={m} rung."
    )
    order = [(i + shift) % m for i in range(m)]
    sents = []
    for b in order:
        pool = pools[b]
        sents.extend(pool[rng.randrange(len(pool))] for _ in range(block_len))
    text, spans = CARRIERS[rng.randrange(len(CARRIERS))], []
    for j, s in enumerate(sents):
        if j:
            text += " "
        spans.append((len(text), len(text) + len(s)))
        text += s
    return text, spans, order


if __name__ == "__main__":
    import random

    rng = random.Random(31415)
    print("registered r1 by m (rotation ladder):")
    for m in (2, 3, 4, 6, 12):
        r1 = rotation_r1_closed_form(m)
        print(f"  m={m:>2}  r1={r1:.4f}  sqrt(r1)={np.sqrt(r1):.4f}"
              + ("   <- same as m=3, wasted rung" if m == 4 else ""))
    print()
    for m in (2, 3, 6):
        t, sp, order = rotation_doc(BLOCKS, m, 12, 0, rng)
        print(f"m={m} order={[BLOCK_NAMES[b] for b in order]}  "
              f"{len(sp)} segments")
    print()
    e, d = REFUSAL_ITEMS[0]
    a, b = refusal_pair(e, d)
    print("D2 class A:", a)
    print("D2 class B:", b)
