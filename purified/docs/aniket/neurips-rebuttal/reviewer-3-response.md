# Draft response to Reviewer EAxU

This is a paste-ready response. It deliberately makes the narrower claim
supported by the new controls: TXC benefits from longer local context on
backtracking, and part of that benefit is sensitive to token order.

## Response

Thank you for the detailed feedback. We agree that the submission left key
architecture definitions in the appendix, overstated the generality of the
empirical result, and omitted several relevant citations. We will make the
method self-contained in the main text and correct the attribution and
presentation issues below.

**Empirical scope.** We do not claim that TXCs dominate every dictionary
architecture: MLC and T-SAE are stronger on some tasks, while backtracking is
the clearest submitted TXC result. We added a matched 20K-step, three-seed
TXC-base detection sweep over \(T\in\{1,2,4,6,10\}\) and a fixed-probe order
perturbation. Ordered AP rises from \(0.218\pm0.005\) at \(T=1\) to
\(0.255\pm0.008\) at \(T=10\); every seed's \(T=10\) endpoint exceeds its
\(T=1\) endpoint, with a paired gain of \(0.037\pm0.009\). At \(T=10\),
shuffling each window under the same ordered-trained dictionary and probe
gives \(0.231\pm0.009\), an ordered-minus-shuffled gap of
\(0.023\pm0.007\). The first result supports useful local history; the smaller
perturbation gap shows that part, but not all, of the gain depends on order.
Because this is a fixed-probe covariate-shift control rather than a retrained
order-invariant model, we treat it as representation sensitivity, not a
causal estimate of unique temporal information.

We also added a real-language temporal task using the public
[KLiCKe](https://doi.org/10.17239/jowr-2025.17.01.02) corpus of human
keystroke logs. From the final five layer-10 activations before a deletion
burst, a cross-fitted sparse probe predicts whether the writer will delete
\(2,3,4,5,\) or \(6+\) model tokens. On 6,224 events from 2,510 held-out
writers, the frozen submitted \(T=5\) TXC obtains 1.236 equal-writer log loss
at the pre-specified \(S=32\) feature budget, versus 1.261 for the strongest
matched SAE (paired improvement 0.0257, 95% CI [0.0100, 0.0413]). Holding the
probe fixed while shuffling or reversing TXC inputs raises loss to 1.646 and
1.657. This uses one frozen dictionary pair, so we present it as a controlled
additional example rather than a multi-seed architecture comparison.

**TXC and TXC-Pro.** We will define both at first use and consolidate their
settings into one architecture table. For a window
\(X_t=(x_t,\ldots,x_{t+T-1})\), TXC forms one shared sparse code
\[
z_t=\sigma\!\left(\sum_{\tau=0}^{T-1}
W_{\mathrm{enc}}^{(\tau)}x_{t+\tau}+b_{\mathrm{enc}}\right),
\qquad
\hat{x}_{t+\tau}=W_{\mathrm{dec}}^{(\tau)}z_t+b_{\mathrm{dec}}^{(\tau)}.
\]
TXC-base uses a fixed \(T=5\) window and full-window reconstruction. TXC-Pro
adds a ten-position encoder that samples five positions during training and
uses all ten at evaluation, eight nested Matryoshka reconstruction groups,
and inverse-distance-weighted contrastive losses at shifts
\(\Delta\in\{1,2\}\).

**Missing citations and title collision.** We will cite Bussmann, Nabeshima,
Karvonen, and Nanda, *Learning Multi-Level Features with Matryoshka Sparse
Autoencoders* (ICML 2025), for the Matryoshka objective; Bhalla et al.,
*Temporal Sparse Autoencoders* (ICLR 2026), for T-SAE; and Turner, Soligo,
Taylor, Rajamanoharan, and Nanda, *Model Organisms for Emergent Misalignment*
(2025), for the bad-medical-advice organism. We will also correct the venue
metadata for Cunningham et al. (ICLR 2024), Gao et al. (ICLR 2025), and
Kantamneni et al. (ICML 2025).

We will cite Bayazit, Mueller, and Bosselut, *Crosscoding Through Time*
(ACL 2026). Their crosscoders span training checkpoints to study pretraining
time; ours span token positions within one fixed model to study sequence time.
The questions are distinct, but the shared title is confusing. We will explain
the distinction in Related Work and rename our paper **“Temporal Crosscoders:
Sparse Feature Discovery Across Sequence Positions.”**

**Remaining presentation fixes.** “SAE-arditi” is a per-token TopK(ReLU) SAE
in the Arditi--Chen setup, not a new architecture
(\(d_{\mathrm{SAE}}=32{,}768,\ k=128\)). We will rename it “TopK SAE
(Arditi--Chen setup),” define and cite it at first use, and use that name
throughout C6. We will merge F.1/F.13 into one configuration table and repair
the appendix cross-references. Finally, we will replace the live Colab citation
with Chanin and Garriga-Alonso, *SynthSAEBench: Evaluating Sparse Autoencoders
on Scalable Realistic Synthetic Data* (arXiv:2602.14687), while keeping our
construction self-contained.

## Internal verification notes

- The backtracking values come from the committed 15-cell
  \(T\in\{1,2,4,6,10\}\), three-seed package in commit `d9c7fc7b`. Values are
  mean \(\pm\) sample SD across dictionary seeds and use a 32-feature,
  question-grouped sparse probe.
- The shuffle is a deterministic test-time perturbation under the fixed
  ordered-trained probe. Do not call it a retrained shuffled model or a causal
  estimate of all temporal information.
- The deletion result uses one frozen seed-42 dictionary pair, five
  writer-grouped folds, and a 2,000-draw equal-writer bootstrap. Do not call
  the folds or bootstrap draws dictionary seeds.
- The paper's backtracking T-SAE point already uses
  \(d_{\mathrm{SAE}}=32{,}768\); see `tsae-capacity-audit.md`. Do not describe
  the existing T-SAE as width-underpowered.
- Confirm team approval of the proposed title and Han's locked TXC-Pro
  definition before posting.
