
# MERA-Style Temporal Sparse Autoencoder

This note writes down a concrete **MERA-style tensor-network autoencoder** for sequence activations.  
The goal is to keep the SAE bias toward sparse, interpretable features while adding an explicit **multiscale temporal prior**: short-range correlations should be disentangled locally, and long-range structure should survive coarse-graining.

It is written as a concrete extension of the tensor-network / temporal SAE design space in the proposal document.

## 1. Architectural principle and encoded prior

A MERA-style architecture encodes the prior that:

1. **Local high-frequency temporal correlations should be stripped off before compression.**
2. **Temporally extended structure should become sparse at the scale matching its span.**
3. **Long-range interactions should be mediated hierarchically, not through a single narrow chain state.**

That differs from:

- a **local SAE**, which assumes the basic object is a per-token feature;
- an **MPS/HMM-style model**, which assumes the main temporal object is a single hidden chain state;
- a **dense latent sandwich layer**, which allows temporal correction but does not strongly constrain *how* long-range structure should be represented.

The MERA prior instead says:

> A feature that spans 2 tokens, 4 tokens, 8 tokens, ... should be representable by a small number of coefficients at the corresponding scales, plus a few finer residuals near boundaries.

So the latent ontology is explicitly **multiscale**.

---

## 2. General multiscale encoder / decoder

Let the window length be \(L = 2^J\), and each token activation be \(x_t \in \mathbb{R}^d\).

### 2.1 Local lift

First map each token into a local feature space of width \(m\):

\[
h_t^{(0)} = \rho(E_{\mathrm{loc}} x_t + b_{\mathrm{loc}}),
\qquad
E_{\mathrm{loc}} \in \mathbb{R}^{m \times d}.
\]

You can take \(\rho\) to be identity, ReLU, softplus, or any standard SAE nonlinearity.

### 2.2 One MERA analysis layer

At level \(\ell\), the representation is \(h_j^{(\ell)} \in \mathbb{R}^{m_\ell}\).

For each neighboring pair, do:

#### (a) Disentangler

\[
\begin{bmatrix}
\bar h_{2j-1}^{(\ell)} \\
\bar h_{2j}^{(\ell)}
\end{bmatrix}
=
U_\ell
\begin{bmatrix}
h_{2j-1}^{(\ell)} \\
h_{2j}^{(\ell)}
\end{bmatrix},
\qquad
U_\ell \in \mathbb{R}^{2m_\ell \times 2m_\ell},
\qquad
U_\ell^\top U_\ell = I.
\]

This is a learned local orthogonal mixing that removes short-range “entanglement”.

#### (b) Isometry / coarse-graining

\[
\begin{bmatrix}
h_j^{(\ell+1)} \\
z_j^{(\ell)}
\end{bmatrix}
=
\begin{bmatrix}
C_\ell \\
S_\ell
\end{bmatrix}
\begin{bmatrix}
\bar h_{2j-1}^{(\ell)} \\
\bar h_{2j}^{(\ell)}
\end{bmatrix}.
\]

A clean Haar-style choice is

\[
C_\ell = \frac{1}{\sqrt 2}\begin{bmatrix} I & I \end{bmatrix},
\qquad
S_\ell = \frac{1}{\sqrt 2}\begin{bmatrix} I & -I \end{bmatrix},
\]

with \(I = I_{m_\ell}\). Then:

- \(h_j^{(\ell+1)}\) is the coarse latent for a span of length \(2^{\ell+1}\),
- \(z_j^{(\ell)}\) is the detail latent for that span.

### 2.3 Sparsification

Apply shrinkage / thresholding scale by scale:

\[
a_j^{(\ell)} = \operatorname{shrink}_{\lambda_\ell}\!\big(z_j^{(\ell)}\big),
\qquad
a_{\mathrm{top}} = \operatorname{shrink}_{\lambda_{\mathrm{top}}}\!\big(h_1^{(J)}\big).
\]

### 2.4 Decoder

Run the hierarchy backward.

Given a coarse latent \(\hat h_j^{(\ell+1)}\) and a detail latent \(a_j^{(\ell)}\):

\[
\begin{bmatrix}
\hat{\bar h}_{2j-1}^{(\ell)} \\
\hat{\bar h}_{2j}^{(\ell)}
\end{bmatrix}
=
\begin{bmatrix}
C_\ell^\top & S_\ell^\top
\end{bmatrix}
\begin{bmatrix}
\hat h_j^{(\ell+1)} \\
a_j^{(\ell)}
\end{bmatrix}.
\]

Then undo the disentangler:

\[
\begin{bmatrix}
\hat h_{2j-1}^{(\ell)} \\
\hat h_{2j}^{(\ell)}
\end{bmatrix}
=
U_\ell^\top
\begin{bmatrix}
\hat{\bar h}_{2j-1}^{(\ell)} \\
\hat{\bar h}_{2j}^{(\ell)}
\end{bmatrix}.
\]

Finally decode back to activation space:

\[
\hat x_t = D_{\mathrm{loc}} \hat h_t^{(0)} + c_{\mathrm{dec}},
\qquad
D_{\mathrm{loc}} \in \mathbb{R}^{d \times m}.
\]

### 2.5 Loss

A simple training loss is

\[
\mathcal{L}
=
\frac{1}{L}\sum_{t=1}^L \|x_t - \hat x_t\|_2^2
+
\sum_{\ell,j}\lambda_\ell \|a_j^{(\ell)}\|_1
+
\lambda_{\mathrm{top}} \|a_{\mathrm{top}}\|_1
+
\gamma \sum_\ell \|U_\ell^\top U_\ell - I\|_F^2.
\]

If you parameterize \(U_\ell\) to be exactly orthogonal (e.g. with Householder products or Cayley transforms), the last term can be dropped.

---

## 3. Explicit 4-site analysis/synthesis matrices

To make the architecture completely explicit, consider a scalar 4-token input

\[
x =
\begin{bmatrix}
x_1 \\ x_2 \\ x_3 \\ x_4
\end{bmatrix}
\in \mathbb{R}^4.
\]

This is the simplest nontrivial binary MERA.

### 3.1 Boundary disentangler

Use one local rotation on the middle pair \((2,3)\):

\[
U_0(\theta)=
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & \cos\theta & -\sin\theta & 0 \\
0 & \sin\theta & \cos\theta & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}.
\]

### 3.2 Pairwise Haar analysis

Apply the 2-point Haar transform separately to the left and right pairs:

\[
H_{\mathrm{pair}} =
\begin{bmatrix}
\frac{1}{\sqrt 2} & \frac{1}{\sqrt 2} & 0 & 0 \\
\frac{1}{\sqrt 2} & -\frac{1}{\sqrt 2} & 0 & 0 \\
0 & 0 & \frac{1}{\sqrt 2} & \frac{1}{\sqrt 2} \\
0 & 0 & \frac{1}{\sqrt 2} & -\frac{1}{\sqrt 2}
\end{bmatrix}.
\]

This maps the rotated sequence to

\[
\begin{bmatrix}
c_L \\ d_L \\ c_R \\ d_R
\end{bmatrix}
=
H_{\mathrm{pair}}\,U_0(\theta)\,x.
\]

Here:

- \(c_L, c_R\) are coarse averages on the left/right halves,
- \(d_L, d_R\) are within-pair details.

### 3.3 Top Haar analysis

Now coarse-grain the two coarse channels:

\[
T_{\mathrm{top}} =
\begin{bmatrix}
\frac{1}{\sqrt 2} & 0 & \frac{1}{\sqrt 2} & 0 \\
\frac{1}{\sqrt 2} & 0 & -\frac{1}{\sqrt 2} & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}.
\]

Define the latent ordering as

\[
z =
\begin{bmatrix}
g \\ d_{\mathrm{top}} \\ d_L \\ d_R
\end{bmatrix}
=
T_{\mathrm{top}}
\begin{bmatrix}
c_L \\ d_L \\ c_R \\ d_R
\end{bmatrix}.
\]

So the full **encoder / analysis matrix** is

\[
z = E_{\mathrm{MERA}}(\theta)\,x,
\qquad
E_{\mathrm{MERA}}(\theta)
=
T_{\mathrm{top}}\,H_{\mathrm{pair}}\,U_0(\theta).
\]

Multiplying that out gives

\[
E_{\mathrm{MERA}}(\theta)=
\begin{bmatrix}
\frac12 &
\frac{\cos\theta+\sin\theta}{2} &
\frac{\cos\theta-\sin\theta}{2} &
\frac12
\\[4pt]
\frac12 &
\frac{\cos\theta-\sin\theta}{2} &
-\frac{\sin\theta+\cos\theta}{2} &
-\frac12
\\[4pt]
\frac{1}{\sqrt2} &
-\frac{\cos\theta}{\sqrt2} &
\frac{\sin\theta}{\sqrt2} &
0
\\[4pt]
0 &
\frac{\sin\theta}{\sqrt2} &
\frac{\cos\theta}{\sqrt2} &
-\frac{1}{\sqrt2}
\end{bmatrix}.
\]

Because every piece is orthogonal, the **decoder / synthesis matrix** is

\[
x = D_{\mathrm{MERA}}(\theta)\,z,
\qquad
D_{\mathrm{MERA}}(\theta) = E_{\mathrm{MERA}}(\theta)^\top.
\]

Explicitly,

\[
D_{\mathrm{MERA}}(\theta)=
\begin{bmatrix}
\frac12 & \frac12 & \frac{1}{\sqrt2} & 0
\\[4pt]
\frac{\cos\theta+\sin\theta}{2} &
\frac{\cos\theta-\sin\theta}{2} &
-\frac{\cos\theta}{\sqrt2} &
\frac{\sin\theta}{\sqrt2}
\\[4pt]
\frac{\cos\theta-\sin\theta}{2} &
-\frac{\sin\theta+\cos\theta}{2} &
\frac{\sin\theta}{\sqrt2} &
\frac{\cos\theta}{\sqrt2}
\\[4pt]
\frac12 & -\frac12 & 0 & -\frac{1}{\sqrt2}
\end{bmatrix}.
\]

### 3.4 Haar special case: \(\theta = 0\)

The simplest useful starting point is no learned boundary rotation at all:

\[
U_0(0) = I.
\]

Then

\[
E_{\mathrm{Haar}}=
\begin{bmatrix}
\frac12 & \frac12 & \frac12 & \frac12 \\
\frac12 & \frac12 & -\frac12 & -\frac12 \\
\frac{1}{\sqrt2} & -\frac{1}{\sqrt2} & 0 & 0 \\
0 & 0 & \frac{1}{\sqrt2} & -\frac{1}{\sqrt2}
\end{bmatrix},
\qquad
D_{\mathrm{Haar}} = E_{\mathrm{Haar}}^\top.
\]

So the explicit analysis equations are

\[
g = \frac{x_1 + x_2 + x_3 + x_4}{2},
\]

\[
d_{\mathrm{top}} = \frac{x_1 + x_2 - x_3 - x_4}{2},
\]

\[
d_L = \frac{x_1 - x_2}{\sqrt2},
\qquad
d_R = \frac{x_3 - x_4}{\sqrt2}.
\]

And the synthesis equations are

\[
\hat x_1 = \frac12 g + \frac12 d_{\mathrm{top}} + \frac{1}{\sqrt2} d_L,
\]

\[
\hat x_2 = \frac12 g + \frac12 d_{\mathrm{top}} - \frac{1}{\sqrt2} d_L,
\]

\[
\hat x_3 = \frac12 g - \frac12 d_{\mathrm{top}} + \frac{1}{\sqrt2} d_R,
\]

\[
\hat x_4 = \frac12 g - \frac12 d_{\mathrm{top}} - \frac{1}{\sqrt2} d_R.
\]

---

## 4. Multichannel vector-valued version

For actual activations, each token is \(x_t \in \mathbb{R}^d\), and each lifted token feature is \(h_t \in \mathbb{R}^m\).

For a 4-token window, stack the local lifted vectors into

\[
h =
\begin{bmatrix}
h_1 \\ h_2 \\ h_3 \\ h_4
\end{bmatrix}
\in \mathbb{R}^{4m}.
\]

Then the multichannel MERA encoder is

\[
z =
\big(E_{\mathrm{MERA}}(\theta)\otimes I_m\big)\,h.
\]

If \(h\) itself comes from a learned local encoder,

\[
h =
\operatorname{blkdiag}(E_{\mathrm{loc}},E_{\mathrm{loc}},E_{\mathrm{loc}},E_{\mathrm{loc}})\,
\begin{bmatrix}
x_1 \\ x_2 \\ x_3 \\ x_4
\end{bmatrix}.
\]

So the full 4-token encoder is

\[
z =
\big(E_{\mathrm{MERA}}(\theta)\otimes I_m\big)
\operatorname{blkdiag}(E_{\mathrm{loc}},E_{\mathrm{loc}},E_{\mathrm{loc}},E_{\mathrm{loc}})
\,x.
\]

The full decoder is

\[
\hat x =
\operatorname{blkdiag}(D_{\mathrm{loc}},D_{\mathrm{loc}},D_{\mathrm{loc}},D_{\mathrm{loc}})
\big(E_{\mathrm{MERA}}(\theta)^\top \otimes I_m\big)\,a,
\]

where \(a\) is the sparsified multiscale latent code.

This makes the analysis/synthesis operators completely explicit even in the vector-valued case.

---

## 5. Worked toy example

Take the 4-token scalar sequence

\[
x =
\begin{bmatrix}
4 \\ 2 \\ 1 \\ 1
\end{bmatrix}.
\]

Use the Haar special case (\(\theta = 0\)).

### 5.1 Encode

Compute the latent coefficients:

\[
g = \frac{4 + 2 + 1 + 1}{2} = 4,
\]

\[
d_{\mathrm{top}} = \frac{4 + 2 - 1 - 1}{2} = 2,
\]

\[
d_L = \frac{4 - 2}{\sqrt2} = \sqrt2,
\]

\[
d_R = \frac{1 - 1}{\sqrt2} = 0.
\]

So the latent code is

\[
z =
\begin{bmatrix}
4 \\ 2 \\ \sqrt2 \\ 0
\end{bmatrix}.
\]

### 5.2 Interpretation

This decomposition is sparse and scale-separated:

- \(g = 4\): one **whole-window** latent shared by all four positions,
- \(d_{\mathrm{top}} = 2\): one **left-half vs right-half** span latent,
- \(d_L = \sqrt2\): one **local correction** inside the left pair,
- \(d_R = 0\): no fine detail needed on the right pair.

This is exactly the multiscale ontology we want: a persistent global mode, a 2-token span correction, and a 1-token-scale residual.

### 5.3 Decode

Use

\[
\hat x = D_{\mathrm{Haar}}\,z = E_{\mathrm{Haar}}^\top z.
\]

Term by term:

\[
4
\begin{bmatrix}
1/2 \\ 1/2 \\ 1/2 \\ 1/2
\end{bmatrix}
+
2
\begin{bmatrix}
1/2 \\ 1/2 \\ -1/2 \\ -1/2
\end{bmatrix}
+
\sqrt2
\begin{bmatrix}
1/\sqrt2 \\ -1/\sqrt2 \\ 0 \\ 0
\end{bmatrix}
=
\begin{bmatrix}
2 \\ 2 \\ 2 \\ 2
\end{bmatrix}
+
\begin{bmatrix}
1 \\ 1 \\ -1 \\ -1
\end{bmatrix}
+
\begin{bmatrix}
1 \\ -1 \\ 0 \\ 0
\end{bmatrix}
=
\begin{bmatrix}
4 \\ 2 \\ 1 \\ 1
\end{bmatrix}.
\]

So the reconstruction is exact.

### 5.4 Why this is a good toy

A plain local SAE tends to explain this sequence with separate tokenwise activations unless its dictionary already contains span-level atoms.

The MERA representation instead says:

1. there is a global 4-token component,
2. plus a left-half / right-half contrast,
3. plus a single within-pair correction.

That is a much cleaner representation of persistent and span-level temporal structure.

---

## 6. Implementation pseudocode

Below is straightforward PyTorch-like pseudocode for a **Haar-first MERA SAE**.  
This version is deliberately simple:

- it assumes \(L = 2^J\),
- it starts with pairwise Haar coarse-graining,
- it uses optional learned orthogonal disentanglers \(U_\ell\),
- it sparsifies detail coefficients at each scale.

You can start with \(U_\ell = I\) everywhere and only later learn them.

### 6.1 Helper operations

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SQRT2 = math.sqrt(2.0)

def soft_threshold(x, lam):
    # signed shrinkage; replace if you want nonnegative latents
    return torch.sign(x) * F.relu(torch.abs(x) - lam)

def interleave(left, right):
    # left, right: [B, L/2, M]
    B, half, M = left.shape
    out = torch.empty(B, half * 2, M, device=left.device, dtype=left.dtype)
    out[:, 0::2, :] = left
    out[:, 1::2, :] = right
    return out
```

### 6.2 Optional orthogonal disentangler

```python
class PairOrthogonal(nn.Module):
    """
    Applies the same learned 2M x 2M orthogonal map to every adjacent pair.
    For a first implementation, set enabled=False or initialize to identity.
    """
    def __init__(self, m, enabled=True):
        super().__init__()
        self.enabled = enabled
        if enabled:
            # simplest unconstrained parameter; in practice prefer an
            # exactly orthogonal parameterization (Householder / Cayley / QR)
            self.weight = nn.Parameter(torch.eye(2 * m))
        else:
            self.register_buffer("weight", torch.eye(2 * m))

    def forward(self, h):
        # h: [B, L, M], L must be even
        if not self.enabled:
            return h

        B, L, M = h.shape
        pairs = h.view(B, L // 2, 2, M).reshape(B, L // 2, 2 * M)
        mixed = pairs @ self.weight.T
        return mixed.view(B, L // 2, 2, M).reshape(B, L, M)

    def inverse(self, h):
        if not self.enabled:
            return h

        B, L, M = h.shape
        pairs = h.view(B, L // 2, 2, M).reshape(B, L // 2, 2 * M)
        mixed = pairs @ self.weight  # inverse if weight is orthogonal
        return mixed.view(B, L // 2, 2, M).reshape(B, L, M)
```

### 6.3 MERA SAE module

```python
class MeraSAE(nn.Module):
    def __init__(self, d_model, m_latent, seq_len,
                 lambdas, lambda_top,
                 learn_disentanglers=False):
        super().__init__()
        assert seq_len > 0 and (seq_len & (seq_len - 1)) == 0,             "seq_len must be a power of two"
        self.d_model = d_model
        self.m_latent = m_latent
        self.seq_len = seq_len
        self.num_levels = int(math.log2(seq_len))

        # local tokenwise encoder/decoder
        self.enc = nn.Linear(d_model, m_latent, bias=True)
        self.dec = nn.Linear(m_latent, d_model, bias=True)

        assert len(lambdas) == self.num_levels
        self.lambdas = lambdas
        self.lambda_top = lambda_top

        self.disentanglers = nn.ModuleList([
            PairOrthogonal(m_latent, enabled=learn_disentanglers)
            for _ in range(self.num_levels)
        ])

    def local_encode(self, x):
        # x: [B, L, D]
        return self.enc(x)

    def local_decode(self, h):
        # h: [B, L, M]
        return self.dec(h)

    def analysis_step(self, h, level):
        """
        One Haar analysis step.
        Input:  h      [B, L_curr, M]
        Output: coarse [B, L_curr/2, M]
                detail [B, L_curr/2, M]
        """
        h = self.disentanglers[level](h)

        left = h[:, 0::2, :]
        right = h[:, 1::2, :]

        coarse = (left + right) / SQRT2
        detail = (left - right) / SQRT2
        return coarse, detail

    def synthesis_step(self, coarse, detail, level):
        """
        Inverse Haar step plus inverse disentangler.
        Inputs: coarse [B, L_curr/2, M]
                detail [B, L_curr/2, M]
        Output: h      [B, L_curr, M]
        """
        left_bar = (coarse + detail) / SQRT2
        right_bar = (coarse - detail) / SQRT2
        h_bar = interleave(left_bar, right_bar)
        h = self.disentanglers[level].inverse(h_bar)
        return h

    def forward(self, x):
        """
        Returns:
            x_hat
            latents: dict with multiscale detail codes and top code
        """
        # 1) local lift
        h = self.local_encode(x)

        # 2) analysis: collect detail codes
        details = []
        h_curr = h
        for level in range(self.num_levels):
            coarse, detail = self.analysis_step(h_curr, level)
            a_detail = soft_threshold(detail, self.lambdas[level])
            details.append(a_detail)
            h_curr = coarse

        # 3) top code
        top = soft_threshold(h_curr, self.lambda_top)

        # 4) synthesis: reconstruct h^(0)
        recon = top
        for level in reversed(range(self.num_levels)):
            recon = self.synthesis_step(recon, details[level], level)

        # 5) local decode back to x
        x_hat = self.local_decode(recon)

        latents = {
            "details": details,   # list of [B, L/2^(l+1), M]
            "top": top            # [B, 1, M]
        }
        return x_hat, latents
```

### 6.4 Training step

```python
def mera_sae_loss(x, x_hat, latents, lambda_l1_top=1.0, lambda_l1_detail=1.0):
    recon_loss = F.mse_loss(x_hat, x)

    detail_penalty = 0.0
    for d in latents["details"]:
        detail_penalty = detail_penalty + d.abs().mean()

    top_penalty = latents["top"].abs().mean()

    loss = recon_loss          + lambda_l1_detail * detail_penalty          + lambda_l1_top * top_penalty
    return loss, {
        "recon": recon_loss.detach(),
        "detail_l1": detail_penalty.detach(),
        "top_l1": top_penalty.detach(),
    }
```

### 6.5 Minimal practical recipe

A good first implementation is:

1. set all disentanglers to identity;
2. use a linear local encoder and decoder;
3. use Haar coarse/detail splits exactly as above;
4. impose \(L_1\) sparsity on all detail scales and on the top code;
5. only after this works, turn on learned orthogonal disentanglers.

That gives a very gauge-fixed baseline whose multiscale latents are easy to inspect.

---

## 7. Optional exact 4-site matrix check in code

For debugging, it is useful to verify the 4-site matrix formulas directly.

```python
import math
import torch

def mera4_encoder_matrix(theta):
    c = math.cos(theta)
    s = math.sin(theta)
    return torch.tensor([
        [0.5,        0.5 * (c + s),  0.5 * (c - s),  0.5],
        [0.5,        0.5 * (c - s), -0.5 * (s + c), -0.5],
        [1/math.sqrt(2), -c/math.sqrt(2),  s/math.sqrt(2), 0.0],
        [0.0,         s/math.sqrt(2),  c/math.sqrt(2), -1/math.sqrt(2)],
    ], dtype=torch.float32)

theta = 0.0
E = mera4_encoder_matrix(theta)
D = E.T

x = torch.tensor([4.0, 2.0, 1.0, 1.0])
z = E @ x
x_hat = D @ z

print("z =", z)         # expected [4, 2, sqrt(2), 0]
print("x_hat =", x_hat) # expected [4, 2, 1, 1]
```

---

## 8. Summary

The cleanest MERA-style temporal SAE is:

- **local token encoder** to lift activations into a feature space;
- **hierarchical Haar/MERA analysis** to split each scale into coarse and detail coefficients;
- **sparsity at every scale** so persistent span-level structure becomes sparse where it belongs;
- **transpose synthesis** back to token space.

The 4-site explicit matrices above give a complete concrete example, and the pseudocode is directly implementable.

Conceptually, the model is useful when the true latent ontology is:

- partly local,
- partly span-level,
- and naturally **hierarchical across timescales**.

That is exactly the regime where a MERA prior is meaningfully different from both local SAEs and single-chain tensor-network models.
