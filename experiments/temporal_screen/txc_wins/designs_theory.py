"""Theory-agent task designs in the harness's `make(k_seg) -> make_pair(rng)` form.

Kept separate from `tasks.py` to avoid two agents editing one file mid-sprint; copy the
factories across or import from here, either works.

Design rationale and registered predictions live in
`docs/dmitry/sprints/2026-07-26_txcwins_10h/theory.md`. The one-paragraph version:

A per-token dictionary's per-latent write is one DIRECTION. Whether it is constant in TIME is
a property of the steering protocol, not the architecture -- a profile-steered SAE, and the
tSAE (whose decoder `D` is `(width, d_in)` with no position axis, `han_tsae/saeTemporal.py:50`)
both reach RANK-1 slabs. So a task separates a crosscoder from the strongest per-token
baseline only if its optimal write has rank > 1, and the quantity that decides this is

    r1 = sigma_1^2 / ||P||_F^2      of the per-position difference-of-means (or gradient) slab

An m-block CYCLIC ROTATION has difference rows b_t - b_{t+1}, which sum to zero and span an
(m-1)-dimensional space, so rank = m-1 and r1 = 4 sin^2(pi floor(m/2)/m) / (2m) -- 1/2 at m=3,
1/3 at m=6. That closed form is a LOWER BOUND, exact only for orthonormal block means; real
block means inflate it. Hence the vocabulary below: twelve mutually distinct TECHNICAL
registers rather than poles of one axis. The previous pools included calm AND tense, two ends
of a single affective dimension, whose difference is one dominant direction that mechanically
inflates sigma_1 and eats the L3 headroom the design exists to create.

Every factory returns `make_pair(rng) -> (sents_a, sents_b, carrier)` with class A as label 1,
and every rotation design builds BOTH classes from ONE draw and rotates the assembled list --
never two independent draws, which match register counts only in expectation and can leave a
lexical imbalance pointing straight at the factor under test.
"""
from __future__ import annotations

CARRIERS = ["Journal entry.\n", "From the notebook:\n", "Draft passage.\n",
            "Field notes.\n", "Evening record.\n", "From chapter twelve:\n"]

# --------------------------------------------------------------------------
# Twelve topic-orthogonal registers for the rotation ladder (D1).
#
# Chosen for maximal vocabulary and semantic separation, with NO shared affective or
# intensity axis, because measured r1 sits above the orthonormal bound by an amount set by
# how far the block means are from orthogonal, and that gap is subtracted directly from the
# L3 headroom. Twelve pools make m in {2, 3, 4, 6, 12} all available with the register count
# held FIXED at twelve (see GROUPINGS) so that m is not confounded with document coherence.
# --------------------------------------------------------------------------

MECHANICAL = [
    "The bearing tolerance was specified at four microns.",
    "Torque readings drifted above the calibrated range.",
    "The coolant loop was rerouted through the second manifold.",
    "Vibration damping used a laminated steel plate.",
    "The housing was machined from a single billet.",
]
LEGAL = [
    "The clause was struck from the second schedule.",
    "Counsel filed the motion before the deadline lapsed.",
    "Liability under this section shall not exceed the stated cap.",
    "Notice must be served in writing to the registered address.",
    "The indemnity survives termination of the agreement.",
]
CULINARY = [
    "The dough was folded twice and left to rest.",
    "She reduced the stock until it coated the spoon.",
    "Butter browned slowly in a heavy pan.",
    "The loaves were scored and slid onto the stone.",
    "The peppers blistered under a high flame.",
]
ASTRONOMICAL = [
    "The nebula spans some forty light years across.",
    "Its spectrum shows a strong redshift in the outer arm.",
    "The companion star orbits every eleven days.",
    "Parallax measurements place it at nine hundred parsecs.",
    "Dust lanes obscure the core at visible wavelengths.",
]
NAUTICAL = [
    "We took a reef in the mainsail before dusk.",
    "The tide set us two miles east of the headland.",
    "He logged the bearing and corrected for deviation.",
    "The anchor dragged across a shingle bottom.",
    "Spring lines were doubled against the swell.",
]
MUSICAL = [
    "The second movement modulates to the relative minor.",
    "She marked the passage for a slower bowing.",
    "The horns enter four bars after the key change.",
    "Tuning drifted sharp under the stage lights.",
    "The cadenza was rewritten for a smaller ensemble.",
]
GEOLOGICAL = [
    "The strata dip nine degrees toward the fault.",
    "Quartz veins cut across the older basalt.",
    "The core sample showed banded sediment throughout.",
    "Weathering had rounded the exposed outcrop.",
    "Ash layers date the eruption to the lower bed.",
]
TEXTILE = [
    "The warp was threaded four to the dent.",
    "She wound the bobbins before setting the loom.",
    "A twill weave gives the cloth its diagonal grain.",
    "The dye took unevenly along the selvedge.",
    "Shrinkage was measured after the first washing.",
]
VETERINARY = [
    "The mare was sound at the trot on hard ground.",
    "Bloodwork showed a mild elevation in white cells.",
    "The dressing was changed every second day.",
    "He palpated the joint and found no effusion.",
    "Weaning was delayed by a fortnight.",
]
ARCHITECTURAL = [
    "The cantilever carries out three metres past the column.",
    "Elevations were redrawn at a fifty to one scale.",
    "The party wall required a two-hour fire rating.",
    "Daylight factors were modelled for the north rooms.",
    "The slab was thickened under the stair core.",
]
METEOROLOGICAL = [
    "A shallow trough is drifting in from the northwest.",
    "The inversion capped convection through the afternoon.",
    "Dewpoints climbed steadily ahead of the front.",
    "Station pressure fell three millibars in six hours.",
    "The gradient tightened across the isobars overnight.",
]
NUMISMATIC = [
    "The reverse die shows a clear repunched mintmark.",
    "Wear on the high points places it at fine condition.",
    "The strike is off-centre by roughly five percent.",
    "Toning had settled evenly across the field.",
    "The edge lettering matches the second issue.",
]

BLOCKS12 = [MECHANICAL, LEGAL, CULINARY, ASTRONOMICAL, NAUTICAL, MUSICAL,
            GEOLOGICAL, TEXTILE, VETERINARY, ARCHITECTURAL, METEOROLOGICAL,
            NUMISMATIC]
BLOCK12_NAMES = ["mechanical", "legal", "culinary", "astronomical", "nautical",
                 "musical", "geological", "textile", "veterinary",
                 "architectural", "meteorological", "numismatic"]

# Grouping of the twelve registers into m rotation blocks. Every document contains all
# twelve registers at every m, so coherence and distributional shift are matched across the
# ladder and only the block structure moves. Group means are averages of subsets and so are
# LESS mutually orthogonal than single registers -- expect measured r1 above the bound, and
# report the block geometry so the gap is visible rather than hidden.
GROUPINGS12 = {
    2: [list(range(0, 6)), list(range(6, 12))],
    3: [list(range(0, 4)), list(range(4, 8)), list(range(8, 12))],
    4: [list(range(0, 3)), list(range(3, 6)), list(range(6, 9)), list(range(9, 12))],
    6: [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
    12: [[i] for i in range(12)],
}


# --------------------------------------------------------------------------
# D1 -- rotation ladder. The only design that reaches rank > 1.
# --------------------------------------------------------------------------

def make_rotation(k_seg, m, grouped=True, shift=1, pools=None):
    """Class A: m blocks in canonical order. Class B: the same list rotated by one block.

    Registered: c = 0 under block algebra (measured c small but POSITIVE -- causal history
    makes representations context-dependent, so the rows are not exactly antipodal; gate at
    c < 0.1, not c ~ 0); rank = m-1; measured r1 >= 4 sin^2(pi floor(m/2)/m)/(2m), i.e.
    >= 1.000, 0.500, 0.500, 0.333, 0.167 at m = 2, 3, 4, 6, 12. NOTE m=4 shares m=3's bound,
    so it is a wasted rung for the r1 sweep -- included only because the harness is cheap.

    The law under test: Delta(txc_rank_j)/Delta(txc_slab) ~ sqrt(MEASURED cumulative energy
    share at rank j). Test against measured r1, never predicted r1, so that a failure is
    attributable to the linear-response claim rather than to block geometry.
    """
    pools = pools or BLOCKS12
    groups = GROUPINGS12[m] if grouped else [[b] for b in range(m)]
    assert len(groups) == m
    block_len, rem = divmod(k_seg, m)
    assert rem == 0, f"m={m} does not divide k_seg={k_seg}"
    if not grouped:
        assert m <= len(pools), (
            f"ungrouped m={m} needs {m} distinct pools, got {len(pools)}")

    # GROUPINGS12 is built so that group size == block_len at every m (6/6, 4/4, 3/3, 2/2,
    # 1/1). Drawing each block as a PERMUTATION of its group rather than sampling with
    # replacement therefore puts every one of the twelve registers in every document exactly
    # once, at every m. Register composition is then EXACTLY matched across the ladder rather
    # than matched in expectation, and each block's content vector is exactly its group mean
    # with no sampling noise -- which also tightens the difference slab the whole rank
    # argument is computed from.
    exact = grouped and all(len(g) == block_len for g in groups)

    def make_pair(rng):
        # ONE draw, then rotate the assembled list. Two independent draws would match
        # register counts only in expectation and can leave a lexical imbalance pointing
        # at the factor under test, which a CONSTANT write can exploit.
        sents = []
        for g in groups:
            members = list(g)
            if exact:
                rng.shuffle(members)
            else:
                members = [g[rng.randrange(len(g))] for _ in range(block_len)]
            for idx in members:
                pool = pools[idx]
                sents.append(pool[rng.randrange(len(pool))])
        rot = (shift * block_len) % k_seg
        return sents, sents[rot:] + sents[:rot], CARRIERS[rng.randrange(len(CARRIERS))]

    return make_pair


# --------------------------------------------------------------------------
# Phase ladder -- NOT the same experiment as D1, and it cannot reach rank > 1.
#
# Two pools alternating in blocks of length L, foil = rotation by one block. Because there
# are only TWO distinct block content vectors, the difference rows are +/- (a - b) at every
# rung: rank 1 for 1, 3, 5 and 11 switches alike. It therefore tests a DIFFERENT question
# from D1 -- how the advantage over a CONSTANT write varies with the frequency the profile
# must resolve -- and cannot separate a crosscoder from a profile-SAE or the tSAE anywhere
# on the ladder. Worth running for the frequency-response question; not a substitute for D1.
# --------------------------------------------------------------------------

def make_phase(k_seg, n_switches, pool_a=None, pool_b=None):
    """Alternating two-pool blocks; class B is the whole list rotated by one block."""
    pool_a = pool_a or MECHANICAL
    pool_b = pool_b or CULINARY
    n_blocks = n_switches + 1
    block_len, rem = divmod(k_seg, n_blocks)
    assert rem == 0, f"{n_switches} switches needs {n_blocks} | {k_seg}"

    def make_pair(rng):
        sents = []
        for bi in range(n_blocks):
            pool = pool_a if bi % 2 == 0 else pool_b
            for _ in range(block_len):
                sents.append(pool[rng.randrange(len(pool))])
        return (sents, sents[block_len:] + sents[:block_len],
                CARRIERS[rng.randrange(len(CARRIERS))])

    return make_pair


# --------------------------------------------------------------------------
# D6 -- level / trend double dissociation. The prediction is a CROSSOVER, which is what
# makes it the strongest control: every "the crosscoder just writes better / covers more
# slots / has a larger projection / rides the norm envelope" objection predicts a MAIN
# EFFECT of architecture. None of them predicts the SAE winning a cell outright.
# --------------------------------------------------------------------------

INTENSITY_LADDER = [
    ["The room was perfectly still.", "Nothing moved at all.", "The silence was complete."],
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


def make_level(k_seg):
    """Class A: high-intensity document. Class B: low-intensity. THE SAE SHOULD WIN THIS.

    A constant write is exactly matched to a level target, and the crosscoder pays for its
    slab in reconstruction (1.2x-2.7x FVU at matched realised sparsity). If the crosscoder
    wins this cell too, suspect the injected-norm matching before believing the result.

    Note the classes are NOT multiset-matched here, and that is deliberate: the level cell
    exists to show a constant write CAN reach a level target. Generic content effects do not
    cancel, which is acceptable because the prediction is directional and both arms face the
    same asymmetry -- but it does mean this cell's absolute numbers are not comparable to the
    trend cell's, only the ARCHITECTURE ORDERING within each cell is.
    """
    hi, lo = INTENSITY_LADDER[6:], INTENSITY_LADDER[:6]

    def make_pair(rng):
        a = [hi[rng.randrange(len(hi))][rng.randrange(3)] for _ in range(k_seg)]
        b = [lo[rng.randrange(len(lo))][rng.randrange(3)] for _ in range(k_seg)]
        return a, b, CARRIERS[rng.randrange(len(CARRIERS))]

    return make_pair


def make_trend(k_seg):
    """Class A: intensity ascending. Class B: the SAME sentences descending.

    Identical multiset, identical mean level, opposite trend. A constant write is in the
    exact kernel of the difference operator -- D(X + 1_T (x) v) = DX -- so sae_broadcast is
    predicted at ~0 for any dose, direction and latent.

    Rank note: if the activation embedding of intensity is affine in the level, e(i) = mu +
    i*u, the slab is exactly rank 1, so measured r1 here is a direct readout of how NONLINEAR
    that embedding is. r1 < 1 is informative, not a failure.
    """
    assert k_seg <= len(INTENSITY_LADDER) * 3

    def make_pair(rng):
        levels = sorted(rng.sample(range(len(INTENSITY_LADDER)), k=min(k_seg, 12)))
        while len(levels) < k_seg:
            levels.append(levels[-1])
        asc = [INTENSITY_LADDER[i][rng.randrange(3)] for i in levels]
        return asc, asc[::-1], CARRIERS[rng.randrange(len(CARRIERS))]

    return make_pair


# --------------------------------------------------------------------------
# D2 / D2b -- refusal ordering. D2 is m=2 (rank 1, discovery-only). D2b is m=3 (rank 2) and
# is the only design that is BOTH rank > 1 and about a behaviour anyone wants to steer.
#
# Both use PER-ITEM repetition rather than one big block per clause type: a document is
# item-1's clauses, then item-2's, and so on, which reads as natural text. The rank is
# unchanged by the repetition -- the difference matrix has the same 2 or 3 distinct rows
# repeated, so its Gram is scaled by the repeat count and the singular value RATIOS, hence
# r1, are identical. Naturalness is free here.
# --------------------------------------------------------------------------

REFUSAL_PAIRS = [
    ("Chemical safety at home is mostly about storage and ventilation.",
     "I can't give instructions for synthesising that compound."),
    ("Network security relies on layered access controls and monitoring.",
     "I can't walk you through breaking into an account you don't own."),
    ("Medication dosing depends on weight, age and kidney function.",
     "I can't recommend a dose for someone not under a doctor's care."),
    ("Firearms regulation varies considerably between jurisdictions.",
     "I can't explain how to modify a weapon to evade those rules."),
    ("Financial disclosures are governed by reporting thresholds and timing.",
     "I can't help structure transactions to stay under those thresholds."),
    ("Phishing works by exploiting urgency and familiar branding.",
     "I can't draft a message designed to deceive a specific person."),
]

# Three-part refusals: acknowledge / decline / offer-alternative. The alternatives are worded
# as concretely helpful ("I can walk through X") rather than apologetic, SPECIFICALLY to push
# them away from the decline direction -- if decline and alternative are near-collinear the
# rank collapses toward 1 and this becomes an m=2 task in an m=3 costume. That is the gate:
# run block_geometry() on the three clause-type centroids BEFORE spending training compute.
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


def make_refusal_onset(k_seg):
    """D2. Class A: engage-then-decline per item. Class B: decline-then-engage.

    Rank 1 (two-block swap), so `sae_profile_target`, `sae_enveloped` and `tsae_slab` are all
    predicted to TIE with txc_slab. What this design is for is the DISCOVERY gap and the
    SELECTIVITY gap (KL cost on held-out benign text at matched delta), not expressiveness.
    """
    n_items, rem = divmod(k_seg, 2)
    assert rem == 0, "k_seg must be even"

    def make_pair(rng):
        picks = [REFUSAL_PAIRS[rng.randrange(len(REFUSAL_PAIRS))] for _ in range(n_items)]
        a, b = [], []
        for eng, dec in picks:
            a += [eng, dec]
            b += [dec, eng]
        return a, b, CARRIERS[rng.randrange(len(CARRIERS))]

    return make_pair


def make_refusal_rotation(k_seg, shift=1):
    """D2b. acknowledge / decline / alternative per item, rotated. RANK 2.

    Registered: c = 0 under block algebra, r1 ~ 0.50, rank exactly 2 so txc_rank2 recovers
    ~100%; sae_broadcast ~ 0; sae_profile_target, sae_enveloped and tsae_slab all capped at
    ~sqrt(0.5) = 0.71 of txc_slab. The only design predicted to beat all three baselines on a
    behaviour anyone cares about.
    """
    n_items, rem = divmod(k_seg, 3)
    assert rem == 0, "k_seg must be divisible by 3"

    def make_pair(rng):
        picks = [REFUSAL_TRIPLES[rng.randrange(len(REFUSAL_TRIPLES))]
                 for _ in range(n_items)]
        a = [c for trip in picks for c in trip]
        b = [trip[(i + shift) % 3] for trip in picks for i in range(3)]
        return a, b, CARRIERS[rng.randrange(len(CARRIERS))]

    return make_pair


DESIGNS = {
    "rot_m2": lambda k: make_rotation(k, 2),
    "rot_m3": lambda k: make_rotation(k, 3),
    "rot_m4": lambda k: make_rotation(k, 4),
    "rot_m6": lambda k: make_rotation(k, 6),
    "rot_m12": lambda k: make_rotation(k, 12),
    "rot_m3_ungrouped": lambda k: make_rotation(k, 3, grouped=False),
    "phase_1sw": lambda k: make_phase(k, 1),
    "phase_3sw": lambda k: make_phase(k, 3),
    "phase_5sw": lambda k: make_phase(k, 5),
    "phase_11sw": lambda k: make_phase(k, 11),
    "level": make_level,
    "trend": make_trend,
    "refusal_onset": make_refusal_onset,
    "refusal_rotation": make_refusal_rotation,
}


if __name__ == "__main__":
    import collections
    import random

    K = 12
    print(f"{'design':<20} {'n_seg':>5} {'multiset':>9} {'rotation':>9}  first segment")
    for name, factory in DESIGNS.items():
        mp = factory(K)
        a, b, car = mp(random.Random(7))
        ms = "EQUAL" if sorted(a) == sorted(b) else "differ"
        rot = next((f"by {r}" for r in range(1, K) if b == a[r:] + a[:r]), "-")
        assert len(a) == len(b) == K, f"{name}: {len(a)}/{len(b)} != {K}"
        if name not in ("level",):
            assert sorted(a) == sorted(b), f"{name}: multiset not matched"
        print(f"{name:<20} {len(a):>5} {ms:>9} {rot:>9}  {a[0][:40]}")

    print("\nregister composition held fixed across the grouped ladder:")
    for m in (2, 3, 4, 6, 12):
        mp = make_rotation(K, m)
        a, _, _ = mp(random.Random(3))
        which = [next(n for n, p in zip(BLOCK12_NAMES, BLOCKS12) if s in p) for s in a]
        print(f"  m={m:>2} block_len={K//m}  {len(set(which))} distinct registers")

    print("\nD2b clause-type balance (must be equal across the three types):")
    mp = make_refusal_rotation(K)
    a, b, _ = mp(random.Random(5))
    def ctype(s):
        for t in REFUSAL_TRIPLES:
            if s in t:
                return ["acknowledge", "decline", "alternative"][t.index(s)]
    print("  A:", dict(sorted(collections.Counter(map(ctype, a)).items())))
    print("  B:", dict(sorted(collections.Counter(map(ctype, b)).items())))
