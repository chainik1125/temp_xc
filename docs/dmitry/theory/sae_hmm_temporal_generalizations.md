# Analytic generalization to a single-feature temporal layer and a simplest two-layer temporal crosscoder

This note extends the previous regular-SAE-on-HMM analysis to two architectures:

1. the **single-feature temporal self-coupling layer** corresponding to Proposal 3A in your temporal SAE / temporal crosscoder note;
2. the **simplest temporal crosscoder** with a single shared latent which reads from **two layers** and decodes back to them.

I keep the same spirit as the modular-addition dynamics note and the earlier HMM-SAE note: write the batched forward pass, take gradients under MSE, and solve the cleanest monosemantic case as far as possible.

The punchline is:

- for a regular time-local SAE, persistence \(\rho\) disappeared from the population objective;
- for Proposal 3A and for a temporal XC, persistence enters immediately through the **full temporal moment matrix** \(\Gamma_{ts}=\mathbb E[s_t s_s]\), or after mean-removal the lag-covariance matrix;
- in the single-feature monosemantic regime, the temporal-operator dynamics are **exactly solvable mode-by-mode**;
- for a two-layer temporal XC, the temporal dynamics are the same up to a simple **decoder-energy factor**, while the only genuinely new issue is an **encoder-split degeneracy** across the two read-in layers.

---

## 1. Setup

I use the same reset/HMM synthetic feature process as before.

Let a single ground-truth atom be \(a\in\mathbb R^d\), \(\|a\|_2=1\), and let the observation be

\[
x_t = a\, s_t,
\qquad s_t\in\{0,1\}.
\]

The support process \((s_t)\) is a stationary two-state reset/HMM process with

\[
\pi = \Pr(s_t=1),
\qquad
\Gamma_\tau := \mathbb E[s_t s_{t+\tau}].
\]

For the reset process with deterministic emissions,

\[
\Gamma_0 = \pi,
\qquad
\Gamma_\tau = \pi^2 + \pi(1-\pi)\rho^{|\tau|}\quad (\tau\neq 0).
\]

If a decoder bias or a separate mean-only feature absorbs the stationary mean, then the centered variable

\[
\xi_t := s_t-\pi
\]

has covariance

\[
C_\tau := \mathbb E[\xi_t\xi_{t+\tau}] = \pi(1-\pi)\rho^{|\tau|},
\]

which is often the cleaner object for the temporal layer, because it removes the trivial \(\pi^2\) baseline.

Throughout, I focus first on the **healthy monosemantic basin** in which the feature direction has already aligned correctly, so the only remaining dynamics are the scalar feature gain(s) and the temporal operator.

---

## 2. Proposal 3A: one feature talking only to itself across time

### 2.1 Batch forward pass

Take a length-\(L\) window

\[
X = [x_1,\dots,x_L]\in\mathbb R^{d\times L}.
\]

For a single latent feature,

\[
\nu = e^\top X + b\mathbf 1_L^\top \in \mathbb R^{1\times L},
\qquad
h = \operatorname{ReLU}(\nu) \in \mathbb R^{1\times L}.
\]

Let the temporal self-coupling operator be

\[
M = I + K\in\mathbb R^{L\times L},
\qquad
\tilde h = h M^\top.
\]

Decode with one decoder direction \(d\in\mathbb R^d\):

\[
\hat X = d\,\tilde h.
\]

A natural loss is

\[
\mathcal L(X)
=
\frac{1}{2L}\|X-\hat X\|_F^2
+ \frac{\lambda_h}{L} h\mathbf 1_L
+ \frac{\lambda_K}{2L}\|K\|_F^2
+ \frac{\lambda_e}{2}\|e\|_2^2
+ \frac{\lambda_d}{2}\|d\|_2^2.
\]

### 2.2 Exact batch gradients

Write the reconstruction residual

\[
R := \hat X - X.
\]

Then

\[
\nabla_d \mathcal L = \frac{1}{L} R\tilde h^\top + \lambda_d d,
\]

\[
g_{\tilde h} := \nabla_{\tilde h}\mathcal L = \frac{1}{L} d^\top R \in \mathbb R^{1\times L},
\]

\[
\nabla_K \mathcal L = g_{\tilde h}^\top h + \frac{\lambda_K}{L}K,
\]

\[
g_h = g_{\tilde h} M,
\qquad
m := \mathbf 1[\nu>0],
\]

\[
\nabla_e \mathcal L = X\big((g_h\odot m)^\top\big) + \lambda_e e,
\qquad
\nabla_b \mathcal L = (g_h\odot m)\mathbf 1_L.
\]

So at the exact batch level, Proposal 3A is just a standard single-feature SAE plus an additional linear temporal layer.

---

## 3. Exact monosemantic reduction for Proposal 3A

Assume the feature direction has already aligned:

\[
d = a,
\qquad e = a.
\]

Parameterize the local scalar feature amplitude by \(u\in(0,1)\) so that

\[
h_t = u s_t.
\]

This is the same healthy scalar basin as in the previous note.

Let

\[
s = (s_1,\dots,s_L)^\top \in \mathbb R^L.
\]

Then the reconstructed scalar sequence is

\[
\hat s = u M s,
\]

and the full population loss becomes

\[
\boxed{
\mathcal L(u,M)
=
\frac{1}{2L}\,\mathbb E\|s-uMs\|_2^2
+ \lambda_h \pi u
+ \frac{\lambda_K}{2L}\|K\|_F^2.
}
\]

If

\[
\Gamma := \mathbb E[s s^\top]\in\mathbb R^{L\times L},
\]

then

\[
\mathcal L(u,M)
=
\frac{1}{2L}\operatorname{Tr}\!\big[(I-uM)\Gamma(I-uM)^\top\big]
+ \lambda_h\pi u
+ \frac{\lambda_K}{2L}\|K\|_F^2.
\]

This is the exact analogue of the regular-SAE scalar basin, except that the scalar moment \(\pi\) is replaced by the **entire temporal moment matrix** \(\Gamma\).

### Main difference from the regular SAE

For a regular time-local SAE, only \(\Gamma_0 = \mathbb E[s_t^2]=\pi\) mattered.

For Proposal 3A, the whole matrix \(\Gamma_{ts}=\mathbb E[s_t s_s]\) enters. So this is the first point where temporal persistence \(\rho\) becomes visible to the full-batch population objective.

---

## 4. Exactly solvable part I: temporal operator dynamics at fixed local gain

Take \(u\) as fixed for the moment. Differentiating the population loss gives

\[
\nabla_M \mathcal L
=
\frac{u}{L}(uM-I)\Gamma + \frac{\lambda_K}{L}(M-I).
\]

In terms of \(K=M-I\),

\[
\boxed{
\nabla_K \mathcal L
=
\frac{1}{L}\Big(u^2 K\Gamma - u(1-u)\Gamma + \lambda_K K\Big).
}
\]

So discrete-time GD is

\[
\boxed{
K_{n+1}
=
K_n - \frac{\eta}{L}\Big(K_n(u^2\Gamma+\lambda_K I)-u(1-u)\Gamma\Big).
}
\]

This is already linear in \(K\) once \(u\) is fixed.

### 4.1 Exact solution in the temporal eigenbasis

Diagonalize the temporal moment matrix:

\[
\Gamma = Q\Lambda Q^\top,
\qquad
\Lambda = \operatorname{diag}(\lambda_1,\dots,\lambda_L).
\]

Define

\[
\bar K := Q^\top K Q.
\]

Then the GD update becomes

\[
\bar K_{ij,n+1}
=
\Bigl(1-\frac{\eta}{L}(u^2\lambda_j+\lambda_K)\Bigr)\bar K_{ij,n}
+ \frac{\eta}{L}u(1-u)\lambda_j\,\delta_{ij}.
\]

So:

- every **off-diagonal temporal mode** decays to zero;
- every **diagonal temporal mode** obeys an exact one-dimensional affine recursion.

For the diagonal temporal eigenmode \(k_{j,n}:=(\bar K_n)_{jj}\),

\[
\boxed{
k_{j,n+1}
=
\Bigl(1-\frac{\eta}{L}(u^2\lambda_j+\lambda_K)\Bigr)k_{j,n}
+ \frac{\eta}{L}u(1-u)\lambda_j.
}
\]

Hence the exact fixed point is

\[
\boxed{
k_j^\star(u)
=
\frac{u(1-u)\lambda_j}{u^2\lambda_j+\lambda_K}.
}
\]

and the exact discrete-time solution is

\[
\boxed{
k_{j,n}
=
k_j^\star(u)
+
\bigl(k_{j,0}-k_j^\star(u)\bigr)
\Bigl(1-\frac{\eta}{L}(u^2\lambda_j+\lambda_K)\Bigr)^n.
}
\]

### Interpretation

This is the clean temporal analogue of the regular-SAE scalar feature solution.

- A temporal mode with larger variance \(\lambda_j\) learns faster.
- The operator learns only along temporal modes already present in \(\Gamma\).
- Persistence \(\rho\) enters only through the temporal spectrum of \(\Gamma\).

In other words:

> Proposal 3A learns a temporal filter by matching the temporal eigenspectrum of the support process.

---

## 5. Exactly solvable part II: local gain dynamics at fixed temporal operator

Now fix \(M\) and optimize \(u\). Define

\[
A_M := \frac{1}{L}\operatorname{Tr}(M\Gamma M^\top),
\qquad
B_M := \frac{1}{L}\operatorname{Tr}(M\Gamma).
\]

Then

\[
\frac{\partial \mathcal L}{\partial u} = -B_M + u A_M + \lambda_h\pi.
\]

So discrete GD is exactly

\[
\boxed{
u_{n+1} = (1-\eta A_M)u_n + \eta(B_M-\lambda_h\pi).
}
\]

Hence

\[
\boxed{
u^\star(M)=\frac{B_M-\lambda_h\pi}{A_M},
\qquad
u_n = u^\star(M) + (u_0-u^\star(M))(1-\eta A_M)^n.
}
\]

So Proposal 3A is exactly solvable in alternating blocks:

- fixed \(u\): solve all temporal modes exactly;
- fixed \(M\): solve the local gain exactly.

The only remaining nonlinearity is the coupling between \(u\) and \(M\).

---

## 6. Integrating out the temporal layer

Because the temporal-layer problem is quadratic, we can eliminate \(K\) analytically for fixed \(u\).

From the mode solution,

\[
1+k_j^\star(u)
=
\frac{u\lambda_j+\lambda_K}{u^2\lambda_j+\lambda_K}.
\]

Plugging back into the loss gives the exact effective one-dimensional objective

\[
\boxed{
\mathcal L_{\mathrm{eff}}(u)
=
\frac{1}{2L}\sum_{j=1}^L
\frac{\lambda_j\lambda_K(1-u)^2}{u^2\lambda_j+\lambda_K}
+ \lambda_h\pi u.
}
\]

So in the single-feature Proposal 3A case, the whole problem reduces to a **one-dimensional scalar objective** after integrating out the temporal operator.

This is already very close to “solved”, except that the final stationary point for \(u\) is given by a scalar nonlinear equation rather than a universal closed-form elementary expression.

---

## 7. Frequency-domain form for a stationary HMM/reset process

For a long stationary window, \(\Gamma\) is approximately Toeplitz. If \(M\) is constrained to be convolutional / translation-equivariant, the temporal eigenbasis is approximately Fourier.

For the centered process \(\xi_t=s_t-\pi\), the spectrum is

\[
S_\xi(\omega)
=
\sum_{\tau=-\infty}^{\infty} C_\tau e^{-i\omega\tau}
=
\pi(1-\pi)
\frac{1-\rho^2}{1+\rho^2-2\rho\cos\omega}.
\]

So the optimal convolutional kernel mode is

\[
\boxed{
K^\star(\omega)
=
\frac{u(1-u)S_\xi(\omega)}{u^2 S_\xi(\omega)+\lambda_K}.
}
\]

As \(\rho\to 1\), the spectrum concentrates at low frequency, so the learned temporal operator becomes an increasingly **low-pass smoothing filter**.

This is probably the cleanest conceptual answer for the temporal layer: in the single-feature case, it is just learning the Wiener filter of the support process, subject to the local-gain bottleneck and the kernel penalty.

---

## 8. One-lag special case

To get a literal closed-form toy model, restrict Proposal 3A to a single causal lag:

\[
\tilde h_t = h_t + \beta h_{t-1}.
\]

Then

\[
\mathcal L(u,\beta)
=
\frac12\Big(\Gamma_0 - 2u(\Gamma_0+\beta\Gamma_1) + u^2(\Gamma_0+2\beta\Gamma_1+\beta^2\Gamma_0)\Big)
+ \lambda_h\pi u + \frac{\lambda_\beta}{2}\beta^2.
\]

For fixed \(u\),

\[
\boxed{
\beta_{n+1}
=
\bigl(1-\eta(u^2\Gamma_0+\lambda_\beta)\bigr)\beta_n
+ \eta u(1-u)\Gamma_1,
}
\]

so

\[
\boxed{
\beta^\star(u)
=
\frac{u(1-u)\Gamma_1}{u^2\Gamma_0+\lambda_\beta}.
}
\]

For fixed \(\beta\),

\[
\boxed{
u_{n+1}
=
\bigl(1-\eta A(\beta)\bigr)u_n + \eta(B(\beta)-\lambda_h\pi),
}
\]

with

\[
A(\beta)=\Gamma_0+2\beta\Gamma_1+\beta^2\Gamma_0,
\qquad
B(\beta)=\Gamma_0+\beta\Gamma_1.
\]

### 8.1 Centered one-lag case with no kernel penalty

If the mean has been removed, then \(\Gamma_0\) should be replaced by \(C_0=\pi(1-\pi)\) and \(\Gamma_1\) by \(C_1=\rho C_0\). With \(\lambda_\beta=0\), the fixed-point equations collapse to

\[
\boxed{
\beta^\star = \rho\,\frac{1-u^\star}{u^\star}.
}
\]

So the copying strength is directly proportional to the persistence \(\rho\), up to the local-vs-temporal load-sharing factor \((1-u^\star)/u^\star\).

This is the cleanest scalar formula I found.

---

## 9. Where the tensor-network parameterization enters

Proposal 3A itself only says “feature \(k\) talks to itself across time”. The **tensor-network / MPO / TT** part is a restriction on the admissible family of temporal operators \(K\).

The cleanest mathematically solvable version is to assume a linear parameterization

\[
K(\theta) = \sum_{r=1}^R \theta_r B_r,
\]

where the basis matrices \(B_r\) are the temporal operators allowed by the low-rank / tensor-network architecture.

Then, for fixed \(u\), the population loss is exactly quadratic in \(\theta\):

\[
\mathcal L(u,\theta)
=
\text{const}
- u(1-u)\sum_r \theta_r\langle B_r,\Gamma\rangle
+ \frac12\sum_{r,s}\theta_r H_{rs}(u)\theta_s,
\]

with

\[
H_{rs}(u)
=
\frac{u^2}{L}\operatorname{Tr}(B_r\Gamma B_s^\top)
+
\frac{\lambda_K}{L}\operatorname{Tr}(B_r B_s^\top),
\]

and

\[
g_r(u) = \frac{u(1-u)}{L}\operatorname{Tr}(B_r\Gamma).
\]

So the parameter update is exactly affine:

\[
\boxed{
\theta_{n+1} = (I-\eta H(u))\theta_n + \eta g(u).
}
\]

Hence

\[
\boxed{
\theta^\star(u) = H(u)^{-1}g(u).
}
\]

This is, I think, the cleanest “single-feature tensor-network SAE” result:

> if the temporal operator enters **linearly** in its parameters, then the full-batch population dynamics are solved exactly at fixed local gain.

If instead the temporal operator is parameterized by multiplicative MPO cores, then the core dynamics are nonlinear; but the induced loss in the operator \(K\) is still the quadratic form above. So the remaining difficulty is entirely in the nonlinear map from cores to \(K\), not in the synthetic temporal objective itself.

---

## 10. Simplest temporal crosscoder with two read-in layers

Now take two observed layers with the same latent support:

\[
x_t^{(1)} = a_1 s_t,
\qquad
x_t^{(2)} = a_2 s_t,
\]

with \(\|a_1\|_2=\|a_2\|_2=1\). The simplest temporal crosscoder has one shared latent:

\[
\nu_t = e_1^\top x_t^{(1)} + e_2^\top x_t^{(2)} + b,
\qquad
h_t = \operatorname{ReLU}(\nu_t),
\qquad
\tilde h = h M^\top,
\]

and two decoder heads

\[
\hat x_t^{(1)} = d_1 \tilde h_t,
\qquad
\hat x_t^{(2)} = d_2 \tilde h_t.
\]

A natural loss is

\[
\mathcal L_{XC}
=
\frac{1}{2L}\sum_{\ell=1}^2 \|X^{(\ell)}-d_\ell\tilde h\|_F^2
+ \frac{\lambda_h}{L}h\mathbf 1_L
+ \frac{\lambda_K}{2L}\|K\|_F^2
+ \sum_{\ell=1}^2 \frac{\lambda_{e,\ell}}{2}\|e_\ell\|_2^2
+ \sum_{\ell=1}^2 \frac{\lambda_{d,\ell}}{2}\|d_\ell\|_2^2.
\]

### 10.1 Exact batch gradients

Let

\[
R_\ell := \hat X^{(\ell)} - X^{(\ell)}.
\]

Then

\[
\nabla_{d_\ell}\mathcal L_{XC} = \frac{1}{L}R_\ell\tilde h^\top + \lambda_{d,\ell}d_\ell,
\]

\[
g_{\tilde h} = \frac{1}{L}\sum_{\ell=1}^2 d_\ell^\top R_\ell,
\qquad
\nabla_K\mathcal L_{XC} = g_{\tilde h}^\top h + \frac{\lambda_K}{L}K,
\]

\[
g_h = g_{\tilde h}M,
\qquad
\nabla_{e_\ell}\mathcal L_{XC} = X^{(\ell)}\big((g_h\odot m)^\top\big)+\lambda_{e,\ell}e_\ell.
\]

So the temporal XC is the exact same temporal layer, but with the scalar backpropagated error now receiving contributions from **both decoder heads**.

---

## 11. Exact monosemantic reduction for the temporal XC

Assume alignment

\[
d_1=a_1,
\qquad d_2=a_2,
\qquad e_1=c_1 a_1,
\qquad e_2=c_2 a_2,
\]

with \(c_1,c_2\ge 0\) and \(b=0\). Then

\[
h_t = (c_1+c_2)s_t = u s_t,
\qquad u:=c_1+c_2.
\]

Define the total decoder energy

\[
E_D := \|d_1\|_2^2 + \|d_2\|_2^2.
\]

For unit-norm decoder heads, \(E_D=2\). If the XC reads from two layers but reconstructs only one target layer, all the same formulas below still hold after replacing \(E_D\) by the squared norm of the reconstructed target decoder.

The population loss becomes

\[
\boxed{
\mathcal L_{XC}(u,M,c_1,c_2)
=
\frac{E_D}{2L}\operatorname{Tr}\!\big[(I-uM)\Gamma(I-uM)^\top\big]
+ \lambda_h\pi u
+ \frac{\lambda_K}{2L}\|K\|_F^2
+ \frac{\lambda_{e,1}}{2}c_1^2
+ \frac{\lambda_{e,2}}{2}c_2^2.
}
\]

So relative to the Proposal-3A SAE, the only structural changes are:

1. the reconstruction term gets multiplied by \(E_D\);
2. the encoder is split across two read-in layers.

### 11.1 Exact read-in split

At fixed effective gain \(u=c_1+c_2\), the reconstruction term depends only on \(u\), not on the individual split. Therefore the best split minimizes the encoder penalties subject to \(c_1+c_2=u\):

\[
\boxed{
c_1^\star = \frac{\lambda_{e,2}}{\lambda_{e,1}+\lambda_{e,2}}u,
\qquad
c_2^\star = \frac{\lambda_{e,1}}{\lambda_{e,1}+\lambda_{e,2}}u.
}
\]

The minimized encoder penalty is

\[
\boxed{
\frac{\lambda_{\mathrm{eff}}}{2}u^2,
\qquad
\lambda_{\mathrm{eff}} := \frac{\lambda_{e,1}\lambda_{e,2}}{\lambda_{e,1}+\lambda_{e,2}}.
}
\]

If \(\lambda_{e,1}=\lambda_{e,2}=\lambda_e\), then

\[
c_1^\star = c_2^\star = \frac{u}{2},
\qquad
\lambda_{\mathrm{eff}} = \frac{\lambda_e}{2},
\]

so the two-layer read-in halves the effective encoder \(L_2\) cost relative to putting the whole gain into one layer.

This is the first important XC-specific result:

> in the clean one-feature setting, the only thing the two-layer read-in does is to create a degenerate split which is resolved by regularization or noise.

Without such asymmetry, the split itself is not identifiable.

---

## 12. Exact temporal-mode dynamics for the temporal XC

With the encoder split integrated out, the temporal XC objective is

\[
\mathcal L_{XC}(u,M)
=
\frac{E_D}{2L}\operatorname{Tr}\!\big[(I-uM)\Gamma(I-uM)^\top\big]
+ \lambda_h\pi u
+ \frac{\lambda_K}{2L}\|K\|_F^2
+ \frac{\lambda_{\mathrm{eff}}}{2}u^2.
\]

### 12.1 Fixed-\(u\) operator dynamics

The exact gradient is

\[
\nabla_K \mathcal L_{XC}
=
\frac{1}{L}\Big(E_D u^2 K\Gamma - E_D u(1-u)\Gamma + \lambda_K K\Big).
\]

So the temporal eigenmodes obey

\[
\boxed{
k_{j,n+1}
=
\Bigl(1-\frac{\eta}{L}(E_D u^2\lambda_j+\lambda_K)\Bigr)k_{j,n}
+
\frac{\eta}{L}E_D u(1-u)\lambda_j.
}
\]

Hence

\[
\boxed{
k_j^\star(u)
=
\frac{E_D u(1-u)\lambda_j}{E_D u^2\lambda_j+\lambda_K}.
}
\]

Relative to Proposal 3A, the temporal XC simply multiplies the reconstruction signal by \(E_D\).

### 12.2 Fixed-\(K\) local-gain dynamics

Define

\[
A_M := \frac{1}{L}\operatorname{Tr}(M\Gamma M^\top),
\qquad
B_M := \frac{1}{L}\operatorname{Tr}(M\Gamma).
\]

Then

\[
\boxed{
u_{n+1}
=
\Bigl(1-\eta(E_D A_M+\lambda_{\mathrm{eff}})\Bigr)u_n
+ \eta(E_D B_M-\lambda_h\pi).
}
\]

So the exact fixed point is

\[
\boxed{
u^\star(M)
=
\frac{E_D B_M - \lambda_h\pi}{E_D A_M + \lambda_{\mathrm{eff}}}.
}
\]

This is the second important XC-specific result:

> in the clean one-feature case, the temporal XC differs from Proposal 3A mostly by a simple speed / signal prefactor \(E_D\) and by a reduced effective encoder penalty.

So if both decoder heads are unit norm, the temporal XC learns the same temporal modes, but faster and with a less costly shared encoder gain.

---

## 13. One-lag two-layer temporal XC

In the one-lag model

\[
\tilde h_t = h_t + \beta h_{t-1},
\]

the temporal fixed-point equation is still

\[
\boxed{
\beta^\star(u)
=
\frac{u(1-u)\Gamma_1}{u^2\Gamma_0 + \lambda_\beta}
}
\]

for the Proposal-3A SAE, and

\[
\boxed{
\beta^\star_{XC}(u)
=
\frac{E_D u(1-u)\Gamma_1}{E_D u^2\Gamma_0 + \lambda_\beta}
}
\]

for the two-layer temporal XC.

If \(\lambda_\beta=0\), then both reduce to

\[
\boxed{
\beta^\star = \frac{\Gamma_1}{\Gamma_0}\frac{1-u^\star}{u^\star}.
}
\]

So the optimal copying ratio itself is unchanged by the two-layer read-in; the crosscoder only changes the value of \(u^\star\) through \(E_D\) and \(\lambda_{\mathrm{eff}}\).

For the centered one-lag model with \(C_1=\rho C_0\), this becomes

\[
\boxed{
\beta^\star = \rho\,\frac{1-u^\star}{u^\star}.
}
\]

With \(\lambda_\beta=0\), the \(u^\star\) equation also collapses to a closed form. For Proposal 3A it is

\[
\boxed{
u^\star_{3A} = 1 - \frac{\lambda_h \pi}{C_0(1-\rho^2)}.
}
\]

and for the two-layer temporal XC it is

\[
\boxed{
u^\star_{XC} = \frac{E_D C_0(1-\rho^2)-\lambda_h \pi}{E_D C_0(1-\rho^2)+\lambda_{\mathrm{eff}}}.
}
\]

So in the centered one-lag idealization, the whole single-feature problem is solved all the way down to closed-form fixed points.

---

## 14. What is genuinely solved here

I think the following parts are essentially solved analytically.

### Proposal 3A single-feature temporal layer

- exact batch forward/backward equations;
- exact monosemantic population objective;
- exact temporal-mode GD at fixed \(u\);
- exact local-gain GD at fixed temporal operator;
- exact scalar effective objective after integrating out \(K\);
- exact one-lag formulas.

### Simplest two-layer temporal XC

- exact batch forward/backward equations;
- exact monosemantic reduction;
- exact optimal read-in split across layers;
- exact temporal-mode GD at fixed \(u\);
- exact local-gain GD at fixed \(K\);
- exact one-lag formulas.

So the main remaining nontrivial problem is not these single-feature cases themselves, but the next level up.

---

## 15. What still looks open

The genuinely open pieces now seem to be:

1. **True tensor-core dynamics.**  
   If the temporal operator is parameterized by nonlinear MPO/MPS cores rather than a linear basis, the core dynamics are nonlinear and carry gauge redundancies.

2. **Assignment / competition.**  
   In the single-feature setting there is no component-allocation problem, only a tradeoff between local gain and temporal propagation. To study how components get assigned, the first nontrivial case is at least two competing features.

3. **Local ambiguity.**  
   The cleanest temporal win should appear when two features are hard to tell apart locally but have different temporal persistence or different multi-layer footprints.

4. **Layer asymmetry beyond regularization.**  
   In the simplest XC case the split across the two layers is unidentifiable unless there is a penalty asymmetry or a noise asymmetry. To get truly XC-specific feature assignment, one should add different noise levels or different decoder footprints across layers.

---

## 16. Immediate experiments suggested by the theory

1. **Single-feature Proposal 3A, fixed aligned atom.**  
   Sweep \(\rho\), \(\pi\), \(\lambda_h\), \(\lambda_K\). Measure the learned temporal eigenvalues \(k_j\) and compare to
   \[
   k_j^\star(u)=\frac{u(1-u)\lambda_j}{u^2\lambda_j+\lambda_K}.
   \]

2. **One-lag special case.**  
   Train with \(\tilde h_t = h_t + \beta h_{t-1}\). Verify the theory curve
   \[
   \beta^\star(u)=\frac{u(1-u)\Gamma_1}{u^2\Gamma_0+\lambda_\beta},
   \]
   and after mean removal the cleaner proportionality to \(\rho\).

3. **Two-layer temporal XC with equal penalties.**  
   Verify that \(c_1\approx c_2\approx u/2\), the temporal operator matches the same mode formula, and the learning speed is rescaled by \(E_D=2\).

4. **Two-layer temporal XC with unequal penalties or noise.**  
   Verify that the read-in split follows the predicted weighted allocation. This is the cleanest way to break the cross-layer degeneracy.

5. **First real assignment toy.**  
   Move from one feature to two locally confusable features with different \(\rho\), or different readout strengths across the two layers. That should be the first synthetic setting where the temporal XC can do something qualitatively different from a local SAE or a local XC.

---

## 17. Bottom line

For the single-feature Proposal-3A temporal layer and for the simplest two-layer temporal XC, the synthetic HMM problem is substantially more tractable than I expected.

- The temporal layer is just learning a quadratic filter against the temporal moment operator of the support process.
- In the monosemantic basin, its GD dynamics are exactly solvable mode-by-mode.
- The two-layer temporal XC adds a simple decoder-energy factor and a read-in-split degeneracy that is analytically solvable under \(L_2\) penalties.

So I think the next genuinely interesting step is **not** to push these single-feature derivations further, but to move to the first two-feature ambiguous setting. That is where “how does the model assign components?” should start to become a real question rather than just a gain-splitting question.