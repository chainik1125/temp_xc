# P1-RM ↔ btk-only weight-equivalence table (auto: rm_equivalence.py, protocol (a))

3/10 pairs IDENTICAL (torch.equal on every shared tensor).

## ALIAS EXCLUSION LIST (184ebd47a assignment — any future arm-diff must exclude these train_keys)

Untrained-twin keys (n_steps=0; legitimate rows) whose byte-equal clusters aliased an external trained-row join into the phantom 'T8-exact' pair. One physical untrained model per cluster:

- pre-T8-untrained: `27e5b452ad79957d`, `3b99316b93e9bea2`, `a19178296a960d32`
- post-T8-untrained: `4cdb346b79c0ecf6`, `73da804cf540cd56`, `84a423f9dd529c0f`

House rule enforced by this checker: joins filter n_steps>0 AND this list, and DUPLICATE slot keys are surfaced, never silently pooled. (Prose correction on the 22:36 entry, receipts authoritative: +8.75e-3 was T8's k5 delta; k20 = +1.02e-2; largest |Δ| overall is T6's −1.63e-2.)

| arch | seed | T | tensors | verdict | Δauc | extra keys |
|---|---|---|---|---|---|---|
| batchtopk_sae | 1 | None | 7 | **IDENTICAL** | +0.00e+00 | threshold_set |
| batchtopk_sae | 2 | None | 0 | **METRIC-IDENTICAL (weights remote)** | +0.00e+00 | — |
| batchtopk_sae | 42 | None | 7 | **IDENTICAL** | +0.00e+00 | threshold_set |
| txc_batchtopk_pre | 1 | 6 | 7 | **DIVERGES** | -1.02e-02 | threshold_set |
| txc_batchtopk_pre | 42 | 1 | 7 | **IDENTICAL** | +0.00e+00 | threshold_set |
| txc_batchtopk_pre | 42 | 2 | 0 | **METRIC-DIVERGES (weights remote)** | +4.56e-03 | — |
| txc_batchtopk_pre | 42 | 4 | 7 | **DIVERGES** | +1.54e-03 | threshold_set |
| txc_batchtopk_pre | 42 | 6 | 7 | **DIVERGES** | -1.63e-02 | threshold_set |
| txc_batchtopk_pre | 42 | 8 | 7 | **DIVERGES** | +8.75e-03 | threshold_set |
| txc_batchtopk_pre | 42 | 16 | 7 | **DIVERGES** | +2.46e-03 | threshold_set |
