# Draft response to Reviewer EAxU

This is a paste-ready draft for Reviewer 3. It is intentionally narrower than
the paper's submitted headline: the new backtracking sweep supports a robust
longer-context effect and a smaller order-sensitivity effect, not universal
TXC dominance.

## Response

Thank you for the detailed feedback and for recognizing that temporal
crosscoders are a well-motivated direction. We agree that the submitted
manuscript placed too much architectural detail in the appendix and contained
several missing or ambiguous citations. We will make the main text
self-contained and correct each presentation issue you identified.

**Empirical scope.** We do not intend to claim that TXCs dominate every
dictionary architecture: MLC and T-SAE are stronger on some tasks, while the
clearest TXC result is backtracking. To test the robustness and temporal basis
of that result, we added a separate matched 20K-step, three-seed backtracking
window sweep and a fixed-probe order perturbation. Ordered TXC detection AP
increases from \(0.216\pm0.005\) at \(T=1\) to \(0.256\pm0.012\) at \(T=6\);
every seed improves with longer context. At \(T=6\), applying the same trained
probe after within-window shuffling reduces AP to \(0.241\pm0.013\), an
ordered-minus-shuffled gap of \(0.015\pm0.005\). We interpret the first result
as evidence that the representation benefits from local context and the second
as a more limited sensitivity test showing that some of this signal depends on
token order. We will temper broad performance language and state explicitly
where TXC, T-SAE, and MLC each perform best.

**TXC and TXC-Pro definitions.** We will define both methods at first use in
the main text and point directly to a consolidated architecture table. Given a
window \(X_t=(x_t,\ldots,x_{t+T-1})\), a TXC forms one shared sparse code
\[
z_t=\sigma\!\left(\sum_{\tau=0}^{T-1}
W_{\mathrm{enc}}^{(\tau)}x_{t+\tau}+b_{\mathrm{enc}}\right)
\]
and reconstructs each position with
\[
\hat{x}_{t+\tau}=W_{\mathrm{dec}}^{(\tau)}z_t+b_{\mathrm{dec}}^{(\tau)}.
\]
TXC-base uses a fixed \(T=5\) window and this full-window sparse
reconstruction objective. TXC-Pro adds three fixed choices: a ten-position
encoder that samples five positions per training step and uses all ten at
evaluation, eight nested Matryoshka reconstruction groups, and
inverse-distance-weighted contrastive losses at shifts
\(\Delta\in\{1,2\}\). We agree that requiring readers to recover these
definitions from Appendix A made the paper unnecessarily difficult to follow.

**Missing citations and title collision.** We will add the archival citation
for the Matryoshka objective used in TXC-Pro: Bussmann, Nabeshima, Karvonen,
and Nanda, *Learning Multi-Level Features with Matryoshka Sparse
Autoencoders* (ICML 2025). We will cite Turner et al., *Model Organisms for
Emergent Misalignment*, for the bad-medical-advice model organism, and correct
the venue metadata for Temporal SAEs, Cunningham et al., Gao et al., and
Kantamneni et al.

We will also cite Bayazit, Mueller, and Bosselut. Their work crosscodes
representations across model checkpoints to study emergence and consolidation
over **pretraining time**; ours crosscodes activations across token positions
within a fixed model to study **sequence time**. The methods and scientific
questions are distinct, but the shared title is confusing. We will explain
this distinction in Related Work and rename our paper **“Temporal
Crosscoders: Sparse Feature Discovery Across Sequence Positions.”**

**Remaining presentation fixes.** “SAE-arditi” is not a new architecture. It
is a per-token TopK SAE using the Arditi--Chen implementation/checkpoint
convention for the Qwen bad-medical setup
(\(d_{\mathrm{SAE}}=32{,}768,\ k=128\)). We will rename it “TopK SAE
(Arditi--Chen setup),” define and cite it at first use, and use the same name
in the C6 results. We will merge the duplicated F.1/F.13 configuration
material into one table and repair the main-text-to-appendix cross-references.
Finally, we will remove the live Google Colab URL and replace it with an
archival citation while keeping the synthetic construction self-contained in
the manuscript.

We appreciate these comments: they identify real exposition and attribution
problems, and the revised structure will make both the contribution and its
limits substantially easier to assess.

## Internal verification notes

- The backtracking numbers above come from the committed three-seed
  \(T=1,\ldots,6\) package in commit `32108679`. They are mean \(\pm\) sample
  SD across dictionary seeds and use a 32-feature, question-grouped sparse
  probe.
- The shuffle is a deterministic test-time perturbation under the fixed
  ordered-trained probe. It is a representation-sensitivity control, not a
  retrained shuffled model and not a causal estimate of all temporal
  information.
- The paper's backtracking T-SAE point already uses
  \(d_{\mathrm{SAE}}=32{,}768\); see `tsae-capacity-audit.md`. Do not describe
  the existing T-SAE as width-underpowered.
- The Matryoshka citation is Bussmann et al. (ICML 2025), despite the reviewer
  attributing the title to Bhalla et al.
- Before posting, confirm the team approves the proposed new title and that
  Han's locked TXC-Pro definition matches the paragraph above verbatim.
