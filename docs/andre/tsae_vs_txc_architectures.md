---
author: Andre Shportko
date: 2026-05-04
tags:
  - reference
  - design
---

## T-SAE vs TXC — Architectures, Schemas, Loss Functions

Side-by-side reference for the two temporal sparse-autoencoder
architectures used throughout the temporal-crosscoders project.
Every formula and shape is grounded in
`temporal_crosscoders/models.py` and `temporal_crosscoders/train.py`,
with explicit file:line references.

Notation throughout:

- $B$ = batch size
- $T$ = window length (number of consecutive token positions per window)
- $d$ = activation dimension (`d_in`)
- $h$ = dictionary width (`d_sae`)
- $k$ = per-position TopK sparsity budget

## 1. T-SAE (Stacked SAE) — `StackedSAE`

Source: `temporal_crosscoders/models.py:77-125`.

### 1.1 Architecture

A *Stacked SAE* is $T$ **independent** TopK SAEs, one per window
position. Position $t$ has its own parameters $(W_{\text{enc}}^{(t)}, W_{\text{dec}}^{(t)}, b_{\text{enc}}^{(t)}, b_{\text{dec}}^{(t)})$
with no weight sharing across positions
(`temporal_crosscoders/models.py:94`).

Per-position parameters (each is one `TopKSAE` instance,
`models.py:22-72`):

$$
W_{\text{enc}}^{(t)} \in \mathbb{R}^{h \times d}, \quad
W_{\text{dec}}^{(t)} \in \mathbb{R}^{d \times h}, \quad
b_{\text{enc}}^{(t)} \in \mathbb{R}^{h}, \quad
b_{\text{dec}}^{(t)} \in \mathbb{R}^{d}
$$

Total parameters: $T \cdot (2 d h + h + d)$.

### 1.2 Forward schema

Input is a window $x \in \mathbb{R}^{B \times T \times d}$
(`models.py:101-116`).
For each position $t \in \{0, \ldots, T-1\}$:

$$
\begin{aligned}
\tilde{x}^{(t)} &= x_{:,t,:} - b_{\text{dec}}^{(t)}                        & \text{(centring; } \texttt{models.py:52}) \\
p^{(t)} &= \tilde{x}^{(t)} (W_{\text{enc}}^{(t)})^\top + b_{\text{enc}}^{(t)} & \text{(pre-activation; } \texttt{models.py:53}) \\
u^{(t)} &= \mathrm{TopK}_k(\mathrm{ReLU}(p^{(t)}))                          & \text{(TopK sparsity; } \texttt{models.py:54-57}) \\
\hat{x}^{(t)} &= u^{(t)} (W_{\text{dec}}^{(t)})^\top + b_{\text{dec}}^{(t)} & \text{(decode; } \texttt{models.py:60})
\end{aligned}
$$

with $\mathrm{TopK}_k$ keeping the $k$ largest values per row and
zeroing the rest. Output is $\hat{x} \in \mathbb{R}^{B \times T \times d}$,
$u \in \mathbb{R}^{B \times T \times h}$.

### 1.3 Sparsity

- **Per-position L0**: exactly $k$ active latents per position by
  construction (`models.py:54-57`).
- **Window-level L0**: $k \cdot T$ across the whole window — one TopK
  decision per position, no cross-position coupling.

### 1.4 Loss

Per-position MSE summed over $d$, averaged over $B$, then averaged
over the $T$ positions
(`models.py:66`, `models.py:115`):

$$
\mathcal{L}_{\text{T-SAE}}(x) = \frac{1}{T} \sum_{t=0}^{T-1}
\underbrace{\mathbb{E}_{B}\left[\, \sum_{i=1}^{d} \big(\hat{x}^{(t)}_i - x^{(t)}_i\big)^2 \,\right]}_{\text{per-position MSE}}
$$

There is **no L1 penalty** — the TopK activation provides hard sparsity.

### 1.5 Decoder normalisation

After every optimiser step, each position's decoder columns are
re-normalised to unit $\ell_2$ norm
(`models.py:46-49`, `train.py:99`):

$$
W_{\text{dec}}^{(t)}_{:, j} \leftarrow \frac{W_{\text{dec}}^{(t)}_{:, j}}{\max(\|W_{\text{dec}}^{(t)}_{:, j}\|_2, 10^{-8})}
$$

## 2. TXC (Temporal Crosscoder) — `TemporalCrosscoder`

Source: `temporal_crosscoders/models.py:130-198`.

### 2.1 Architecture

A *Temporal Crosscoder* uses a **single shared latent vector**
$z \in \mathbb{R}^h$ for the whole length-$T$ window, with
position-specific encoder and decoder weights
(`models.py:154-161`):

$$
W_{\text{enc}} \in \mathbb{R}^{T \times d \times h}, \quad
W_{\text{dec}} \in \mathbb{R}^{h \times T \times d}, \quad
b_{\text{enc}} \in \mathbb{R}^{h}, \quad
b_{\text{dec}} \in \mathbb{R}^{T \times d}
$$

The encoder bias is **shared across positions** (`models.py:160`);
the decoder bias is per-position (`models.py:161`).

Total parameters: $2 T d h + h + T d$ — same order as T-SAE
($T \cdot 2 d h$ leading term), but tied through a single
encoder→latent fan-in and a single latent→decoder fan-out.

### 2.2 Forward schema

Input is the same shape as T-SAE: $x \in \mathbb{R}^{B \times T \times d}$.
The encoder *projects every position* into the same latent space and
*sums*; a single TopK selects the active latents
(`models.py:168-174`):

$$
\begin{aligned}
p &= \sum_{t=0}^{T-1} x_{:,t,:} \, W_{\text{enc}}^{(t)} + b_{\text{enc}}            & \text{(einsum "btd,tds→bs"; } \texttt{models.py:170}) \\
z &= \mathrm{TopK}_{kT}(\mathrm{ReLU}(p))                                            & \text{(window-level TopK; } \texttt{models.py:171-174}) \\
\hat{x}_{:, t, :} &= z \, W_{\text{dec}}^{(t)} + b_{\text{dec}}^{(t)} \quad \forall t & \text{(einsum "bs,std→btd"; } \texttt{models.py:178})
\end{aligned}
$$

The decoder reconstructs **every** position from the **same** latent
vector $z$ — that is the cross-position weight tying that defines TXC.

### 2.3 Sparsity

- **Window-level L0**: exactly $k T$ active latents per window
  (`models.py:152` sets `self.k = k * T`).
- This deliberately **matches** T-SAE's window-level L0 = $k T$ for
  apples-to-apples comparison.
- **Per-position L0** is undefined / shared — the same $k T$ latents
  contribute to every position via a different $W_{\text{dec}}^{(t)}$ slice.

### 2.4 Loss

Window-level MSE summed over $d$, averaged over $B$ and over $T$
(`models.py:184`):

$$
\mathcal{L}_{\text{TXC}}(x) = \mathbb{E}_{B}\left[\, \frac{1}{T} \sum_{t=0}^{T-1} \sum_{i=1}^{d} \big(\hat{x}_{:,t,i} - x_{:,t,i}\big)^2 \,\right]
$$

Algebraically identical reduction to T-SAE's loss, but $\hat{x}$ is
produced from the *shared* $z$ rather than $T$ independent $u^{(t)}$.
Like T-SAE, **no L1 penalty** — TopK is the sparsity mechanism.

### 2.5 Decoder normalisation

`W_dec` is normalised over the $(T, d)$ axes per latent
(`models.py:163-166`):

$$
W_{\text{dec}}_{j, :, :} \leftarrow \frac{W_{\text{dec}}_{j, :, :}}{\max\left(\sqrt{\sum_{t, i} (W_{\text{dec}}_{j, t, i})^2}, 10^{-8}\right)}
$$

A single TXC decoder atom (one $j$) therefore has unit norm across the
whole length-$T$ window — its mass can be distributed arbitrarily over
the $T$ positions, but the total is constrained.

## 3. Side-by-side comparison

| | T-SAE (`StackedSAE`) | TXC (`TemporalCrosscoder`) |
|---|---|---|
| File:lines | `models.py:77-125` | `models.py:130-198` |
| Latent space | $T$ independent vectors $u^{(t)} \in \mathbb{R}^h$ | one shared vector $z \in \mathbb{R}^h$ |
| `W_enc` shape | $T \times (h \times d)$ separate matrices | one tensor $(T, d, h)$ |
| `W_dec` shape | $T \times (d \times h)$ separate matrices | one tensor $(h, T, d)$ |
| Encoder bias | per-position $b_{\text{enc}}^{(t)} \in \mathbb{R}^h$ | shared $b_{\text{enc}} \in \mathbb{R}^h$ |
| Decoder bias | per-position $b_{\text{dec}}^{(t)} \in \mathbb{R}^d$ | per-position, packed as $(T, d)$ |
| Sparsity rule | TopK $k$ per position | TopK $kT$ across the window |
| Per-position L0 | exactly $k$ | shared (any of the $kT$ active latents may write to any position) |
| Window L0 | $k T$ | $k T$ |
| Decoder normalisation axis | per-position-per-feature ($d$-axis) | per-feature over the window ($T \cdot d$ axes) |
| Cross-position coupling | none | full (one TopK + shared latent) |
| Inductive bias | "feature × position" atoms | "feature distributed across the window" atoms |
| Loss | $\frac{1}{T}\sum_t \mathrm{MSE}(\hat{x}^{(t)}, x^{(t)})$ | window MSE on $\hat{x}(z)$ vs $x$ |
| L1 / KL penalty | none (TopK only) | none (TopK only) |
| Decoder atom interpretation | one $W_{\text{dec}}^{(t)}_{:, j}$ acts at position $t$ only | one $W_{\text{dec}}_{j, :, :}$ acts at all $T$ positions |

The two architectures are **deliberately matched** on:

1. window-level L0 budget ($k T$ active latents per window),
2. parameter count (leading-order $2 T d h$), and
3. loss reduction (per-position MSE averaged over $T$).

They differ *only* in the **encoder–decoder coupling structure**:
T-SAE keeps positions independent, TXC ties them through a shared
latent. This is the single architectural variable the synthetic and
real-LM benchmarks across the project isolate.

## 4. Training scheme (shared between the two)

Source: `temporal_crosscoders/train.py:62-205`, `temporal_crosscoders/config.py`.

Both architectures are trained with the *same* recipe — the only
moving piece between them is the `model = ...` line
(`train.py:85` for T-SAE, the analogous TXCDR train fn for TXC).

| component | choice | code |
|---|---|---|
| optimiser | Adam | `train.py:86`, `train.py:163` |
| learning rate | `LEARNING_RATE = 3e-4` | `config.py` |
| betas | `ADAM_BETAS = (0.9, 0.999)` | `config.py` |
| grad clip | `GRAD_CLIP = 1.0` | `train.py:97`, `train.py:175` |
| precision | fp16 autocast (NLP runs) | `temporal_crosscoders/NLP/train.py` |
| batch | $B$ windows of length $T$ | `data.py` |
| TopK | hard, with `ReLU(pre)` then `topk` | `models.py:54-57`, `models.py:171-174` |
| decoder normalisation | unit-norm reproject after every step | `train.py:99`, `train.py:177` |
| L1 / KL | **none** | — |
| eval cadence | every `LOG_INTERVAL` steps | `train.py:101`, `train.py:179` |

Specifically, no auxiliary sparsity loss is applied — the only signal
is reconstruction MSE and the TopK projection forces the sparsity. The
`_normalize_decoder` step after every optimiser step is what keeps the
TopK ranking interpretable (`models.py:46-49`, `models.py:163-166`).

## 5. Worked shape walk-through ($B = 2$, $T = 5$, $d = 256$, $h = 128$, $k = 5$)

Same input window for both:

```text
x : (2, 5, 256)
```

### T-SAE

```text
for t in [0,1,2,3,4]:
    x[:, t, :]                      shape (2, 256)
    pre_t = (x_t - b_dec_t) @ W_enc_t.T + b_enc_t   shape (2, 128)
    u_t   = topK(relu(pre_t), k=5)                  shape (2, 128) with 5 nonzeros / row
    x_hat_t = u_t @ W_dec_t.T + b_dec_t             shape (2, 256)
x_hat : (2, 5, 256), u : (2, 5, 128); window L0 = 5*5 = 25
```

### TXC

```text
pre = einsum("btd,tds->bs", x, W_enc) + b_enc   shape (2, 128)
z   = topK(relu(pre), k=k*T=25)                  shape (2, 128) with 25 nonzeros / row
x_hat = einsum("bs,std->btd", z, W_dec) + b_dec  shape (2, 5, 256)
window L0 = 25 (matches T-SAE)
```

The key shape-level difference: T-SAE produces **a different sparse
vector at each position** (`u : (B, T, h)` with $k$ nonzeros per
$T$-slice), while TXC produces **one sparse vector for the whole
window** (`z : (B, h)` with $k T$ nonzeros).

## 6. Pointers

- Code: [`temporal_crosscoders/models.py`](../../temporal_crosscoders/models.py),
  [`temporal_crosscoders/train.py`](../../temporal_crosscoders/train.py),
  [`temporal_crosscoders/config.py`](../../temporal_crosscoders/config.py).
- Architectural overview (Dmitry): [[temporal_xc_architectures]].
- Sweep results comparing the two on synthetic data: [[v2_tx_v_sae]].
- NLP-scale results on Gemma-2-2b activations: [[nlp_gemma2_summary]].
- Cross-branch benchmark compilation: [[tempbench_metareport]].
