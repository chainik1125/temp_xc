# Draft email to Surya Ganguli

**Status:** fact-checked against the July 29 meeting transcript and the
[completed exact correlation recheck](../../../results/temporal_screen/correlation_recheck_20260729/summary.md).
The task-rollout sentence is explicitly an internal first-pass report from
Dmitry; its scratch script and full caches were not preserved, so it is not
presented as an independently reproduced result.

**Subject:** Follow-up on target-conditioned temporal scaling in LM activations

Hi Surya,

It was great speaking with you after your Q Labs talk. Dmitry Manning-Coe and
I are trying to develop a dictionary-architecture-independent screen for
language-model tasks that genuinely require ordered temporal
representations. The practical puzzle is that our Temporal Crosscoder helps
substantially on backtracking, but not on several other tasks we initially
expected to be temporal.

In a first-pass internal experiment, Dmitry generated task rollouts,
estimated two-point activation correlations as a function of lag, and reduced
each curve to a scalar decay length. That scalar did not cleanly separate the
tasks. Separately, an earlier
GPT-2/WikiText experiment found that an aggregate, de-persisted Frobenius norm
of lagged covariance matrices preferred a power law to a pure exponential
over the fitted range. We have now reproduced that restricted result.
Broader curve families, centering choices, and persistent-subspace checks
support the narrower conclusion of slow, heterogeneous, finite-range
multiscale dependence with a persistent component, rather than one stable
pure law. One hypothesis, which we have not established, is that ensemble
averaging can obscure sequence-specific persistent directions, while
unconditional low-frequency power can reflect structure unrelated to the task
target.

We are considering a grouped, nuisance-residualized target cross-covariance
operator,

\[
K_B(\tau)
=
\mathbb E\!\left[
\left(Z_{t-\tau}-\mathbb E[Z_{t-\tau}\mid B_t]\right)
\left(U_t-\mathbb E[U_t\mid B_t]\right)^\top
\right].
\]

where \(U_t\) encodes a local target and \(B_t\) includes the anchor, unordered
history, best single offset, and deployment-available nuisance variables. We
would estimate a grouped sampling floor and decompose the lag operator into
shared and sequence-specific modes. A separate nonlinear eligibility test
would ask whether ordered history predicts the target beyond capacity-matched
unordered and best-offset observers.

Do you think the finite-data resolvable-horizon argument in the Cagnetta et
al. work could be extended from an unconditional lagged token-covariance norm
to the leading singular modes of this target-conditioned operator, especially
with dependent samples grouped by rollout? We would ultimately like to relate
its target-relevant horizon to the horizon over which an unsupervised temporal
dictionary can resolve the same activation mode. If this overlaps with work
your group is pursuing, we would be very interested in comparing notes or
sending a short technical proposal.

Best,

Aniket
