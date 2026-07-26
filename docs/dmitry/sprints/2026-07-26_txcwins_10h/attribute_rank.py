"""Does rank track the ATTRIBUTE COUNT or the SCHEDULE COMPLEXITY?

Model: at position t the activation carries a set of semantic attributes,
    x_t = sum_a s_a(t) u_a + (whatever is shared by both classes)
so the difference slab is
    P[t] = sum_a [s_a^A(t) - s_a^B(t)] u_a  =  sum_a ds_a(t) u_a
        => P = S U,   S is (T, A) with columns ds_a,   U is (A, d) with rows u_a

Therefore  rank(P) = rank(S U) <= min(rank S, rank U) <= A.

SCHEDULE COMPLEXITY LIVES ENTIRELY INSIDE S's COLUMNS AND CANNOT RAISE ITS RANK
ABOVE A. That is the theorem. Tests below: (1) crank schedule complexity at A=1,
(2) sweep A, (3) the two ways equality fails.
"""
import numpy as np

rng = np.random.default_rng(0)
d, T = 1536, 12


def rank_r1(P):
    s = np.linalg.svd(P, compute_uv=False)
    return int((s > 1e-9 * s[0]).sum()), s[0] ** 2 / (s ** 2).sum()


def build(S, U):
    return S @ U


print("(1) ONE attribute, schedule complexity cranked to the maximum.")
u = rng.standard_normal((1, d)) / np.sqrt(d)
schedules = {
    "one switch      [+++...---]": np.array([1] * 6 + [-1] * 6),
    "3 switches": np.array(([1] * 3 + [-1] * 3) * 2),
    "11 switches (alternating)": np.array([(-1) ** t for t in range(T)]),
    "random +/-1": rng.choice([-1.0, 1.0], T),
    "random real-valued": rng.standard_normal(T),
    "smooth ramp": np.linspace(-1, 1, T),
    "two spikes only": np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0]),
}
for nm, s in schedules.items():
    rk, r1 = rank_r1(build(s.reshape(T, 1).astype(float), u))
    print(f"    {nm:<28} rank={rk}  r1={r1:.4f}")
print("    -> schedule complexity NEVER raises rank above the attribute count.\n")

print("(2) Sweep the attribute count A, schedules generic and independent.")
for A in (1, 2, 3, 4, 6):
    U = rng.standard_normal((A, d)) / np.sqrt(d)
    S = rng.standard_normal((T, A))
    rk, r1 = rank_r1(build(S, U))
    print(f"    A={A}:  rank={rk}  r1={r1:.4f}   (rank == A: {rk == A})")

print("\n    m-block ROTATION is the A=m case with ONE dependency (schedules sum")
print("    to zero, since every position holds exactly one block in each class),")
print("    hence rank m-1 -- which is why the closed form gave m-1 and not m:")
for m in (2, 3, 4, 6):
    U = rng.standard_normal((m, d)) / np.sqrt(d)
    blk = T // m
    S = np.zeros((T, m))
    for t in range(T):
        S[t, (t // blk) % m] += 1.0            # class A
        S[t, ((t // blk) + 1) % m] -= 1.0      # class B = rotation
    rk, r1 = rank_r1(build(S, U))
    print(f"      m={m}: rank={rk} (= m-1: {rk == m-1})  r1={r1:.4f}")

print("\n(3) The TWO ways rank < A, i.e. where the hypothesis needs its caveats.")
U = rng.standard_normal((2, d)) / np.sqrt(d)
S = rng.standard_normal((T, 2))
rk, _ = rank_r1(build(S, U)); print(f"    generic A=2                     rank={rk}")
S_prop = np.stack([S[:, 0], 2.3 * S[:, 0]], axis=1)
rk, _ = rank_r1(build(S_prop, U)); print(f"    schedules PROPORTIONAL          rank={rk}  <- collapses")
U_col = np.stack([U[0], -1.7 * U[0]])
rk, _ = rank_r1(build(S, U_col)); print(f"    directions COLLINEAR            rank={rk}  <- collapses")

print("\n(4) CONTENT + CARRIED STATE: why recency gets rank 2 for free.")
print("    A carried state's schedule is the running INTEGRAL of the content's.")
print("    An integral is never proportional to its integrand (unless trivial),")
print("    so content and its own maintained state are automatically 2 attributes.")
for nm, content in (("two spikes (recency)", np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0])),
                    ("one switch", np.array([1] * 6 + [-1] * 6)),
                    ("alternating", np.array([(-1.0) ** t for t in range(T)]))):
    state = np.cumsum(content).astype(float)        # what the model carries forward
    S = np.stack([content.astype(float), state], axis=1)
    U = rng.standard_normal((2, d)) / np.sqrt(d)
    rk, r1 = rank_r1(build(S, U))
    corr = abs(np.corrcoef(content.astype(float), state)[0, 1])
    print(f"    {nm:<22} rank={rk}  r1={r1:.4f}  |corr(content,state)|={corr:.3f}")
