# Signed-motion — AC-only order-sensitive benchmark

**Verdict: NEGATIVE** (the cautionary case). An order-sensitive bench where the
hidden sign `S` lives only in the step `Q_{t+1}−Q_t = S·v` (provably invisible to
any per-token encoder by the data-processing inequality). It is *not* real-task
motivated — it predates the autoresearch loop — but it is the order-sensitive
(AC) benchmark whose **memorization confound** (`#distinct windows = 2F`, so a
probe with ≥`2F` features can memorize) turned an apparent `s_temp = 1.0` into
the true negative: in the scarce regime no architecture recovers the sign. The
lesson — memorization-free probes + provable floors — is baked into
[`../README.md`](../README.md) and
the backtracking bench's design.

**Doc:** [`bench.md`](bench.md) (single combined writeup; § 8 = reproduction).
**Scripts** (`-m experiments.explorations.synthetic.signed_motion.<x>` / direct): `minisweep.sh`
(sweep), `populate.py` (fills the AUTO-RESULTS tables in `bench.md`),
`render_figs.py` (frontier figure). Generator + evaluator live in the framework
(`src/temp_bench/data/synthetic.py:signed_motion`,
`src/temp_bench/evals/signed_motion_recovery.py`). **`figs/`**.
