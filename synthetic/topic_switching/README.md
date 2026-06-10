# Topic-switching — change-point / sticky-dwell measurement

**Autoresearch investigation #2 (measurement only). Verdict: ABORT** — a sound
negative. Topic-switching, operationalized as MiniLM sentence-embedding clusters
(K=20) on fineweb-edu, **fails the temporal-ness gate**: 82% of the same-topic
autocorrelation is per-*document* composition (it survives the order-destroying
N1 permutation), the genuine order effect is small and short-ranged, the dwell
is ~geometric, and the labeler is too noisy (silhouette ≈ 0.017). No mirror or
benchmark is built; the shared [`../changepoint/bench_spec.md`](../changepoint/bench_spec.md)
stays gated on a stronger labeler or the EM judge.

A methodological note in the record: the first-pass auto-verdict was confounded
(it compared to N2, which doesn't preserve composition) and was corrected to
score order as **real − N1** — the same false-positive trap as signed_motion.

**Reading order:** [`prereg.md`](prereg.md) (frozen) →
[`measurement.md`](measurement.md) (the abort record).
**Script:** `-m synthetic.topic_switching.measure`. **`figs/`**, **`results/`**.
